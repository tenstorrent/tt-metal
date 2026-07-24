# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
#
# Trace capture/replay for HunyuanImage-3.0 AR recaption decode (KV + single token).
#
# Prefill runs eager once. Decode uses a captured CQ0 graph replayed per AR token;
# host updates token/mask/RoPE/write-pos buffers between replays. Stage forcing stays
# in the host ``generate_text`` loop (Whisper-style).

from __future__ import annotations

import time

import torch

import ttnn

from models.experimental.hunyuan_image_3_0.ref.attention.mask import (
    build_attention_mask,
    build_attention_mask_query_row,
    to_additive,
)
from models.experimental.hunyuan_image_3_0.tt.ar_dual_cq import (
    COMPUTE_CQ,
    ArDualCQCoordinator,
    logits_host_to_torch,
)
from models.experimental.hunyuan_image_3_0.tt.ar_prefill import (
    recaption_trace_prefill_enabled,
    run_kv_prefill,
)


from models.experimental.hunyuan_image_3_0.tt.trace_config import recaption_trace_enabled as _hy_recaption_trace_enabled


def recaption_trace_enabled(device, *, sp_factor: int = 1, use_kv_cache: bool = True) -> bool:
    return _hy_recaption_trace_enabled(sp_factor=sp_factor, use_kv_cache=use_kv_cache)


class RecaptionDecodeTracer:
    """KV prefill (chunked) + optional CQ0 decode-trace replay for AR tokens.

    When ``return_device_logits=True`` (HY_DEVICE_SAMPLING), returns the ttnn logits
    tensor for on-device ``ttnn.sampling`` instead of D2H. Decode trace capture is
    skipped in that mode (``enable_decode_trace=False``) — prefill stays chunked.
    """

    def __init__(
        self,
        device,
        model,
        lm_head,
        *,
        wte_tt,
        prefix_embeds: torch.Tensor | None = None,
        prefix_input_ids: torch.Tensor | None = None,
        image_infos,
        attn_slices,
        kv_cache,
        max_cache_len: int,
        prefix_len: int,
        replicate_to_mesh=None,
        dual_cq: ArDualCQCoordinator | None = None,
        return_device_logits: bool = False,
        enable_decode_trace: bool = True,
    ):
        if prefix_embeds is None and prefix_input_ids is None:
            raise ValueError("RecaptionDecodeTracer requires prefix_embeds or prefix_input_ids")
        if prefix_input_ids is not None and wte_tt is None:
            raise ValueError("prefix_input_ids requires wte_tt for on-device prefix embedding")
        self.device = device
        self.model = model
        self.lm_head = lm_head
        self.wte_tt = wte_tt
        self.prefix_embeds = prefix_embeds
        self.prefix_input_ids = prefix_input_ids
        self.prefix_embeds_tt = None
        self.image_infos = image_infos
        self.attn_slices = attn_slices
        self.kv_cache = kv_cache
        self.max_cache_len = max_cache_len
        self.prefix_len = prefix_len
        self.replicate_to_mesh = replicate_to_mesh
        self.dual_cq = dual_cq
        self.return_device_logits = return_device_logits
        self.enable_decode_trace = enable_decode_trace and not return_device_logits

        self.trace_id = None
        self.prefill_trace_id = None
        self.logits_tt = None
        self.prefill_done = False
        self.replay_steps = 0
        self._timing_prefill_ms = 0.0
        self._timing_first_decode_ms = 0.0
        self._timing_replay_total_ms = 0.0
        self._timing_replay_count = 0

        self._cos_full = None
        self._sin_full = None
        self._token_ids_tt = None
        self._mask_tt = None
        self._cos_tt = None
        self._sin_tt = None
        self._write_pos_tt = None
        self._write_pos_host = None
        self._query_pos = 0
        self._cos_host: torch.Tensor | None = None
        self._sin_host: torch.Tensor | None = None

    def ensure_prefix_embeds_tt(self) -> ttnn.Tensor:
        """Materialize prefix hidden states on device (TT embed or one-shot H2D)."""
        if self.prefix_embeds_tt is not None:
            return self.prefix_embeds_tt
        if self.prefix_input_ids is not None:
            ids = self.prefix_input_ids
            if ids.ndim == 1:
                ids = ids.unsqueeze(0)
            ids = ids[:, : self.prefix_len].contiguous()
            self.prefix_embeds_tt = self.wte_tt.embed(ids)
            print(
                f"[recaption] prefix embedding on-device via wte "
                f"(ids [{ids.shape[0]}, {ids.shape[1]}], no host F.embedding H2D)",
                flush=True,
            )
        else:
            self.prefix_embeds_tt = self._upload_hidden(self.prefix_embeds[:1, : self.prefix_len].float())
            print(
                f"[recaption] prefix embeds H2D once "
                f"(shape={[1, self.prefix_len, int(self.prefix_embeds.shape[-1])]})",
                flush=True,
            )
        return self.prefix_embeds_tt

    def prefix_hidden_slice(self, start: int, end: int) -> ttnn.Tensor:
        """Device slice of prefix embeds ``[1, end-start, H]`` (caller owns / must free)."""
        prefix_tt = self.ensure_prefix_embeds_tt()
        h = int(prefix_tt.shape[-1])
        return ttnn.slice(prefix_tt, [0, start, 0], [1, end, h])

    def _upload_hidden(self, hidden_host: torch.Tensor) -> ttnn.Tensor:
        if self.replicate_to_mesh is not None:
            return self.replicate_to_mesh(hidden_host)
        return ttnn.from_torch(
            hidden_host,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=self.device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

    def _upload_mask_full(self, S: int) -> ttnn.Tensor:
        mask_bool = build_attention_mask(S, self.attn_slices, bsz=1)
        mask_add = to_additive(mask_bool, dtype=torch.bfloat16).reshape(1, 1, S, S)
        if self.replicate_to_mesh is not None:
            return self.replicate_to_mesh(mask_add)
        return ttnn.from_torch(
            mask_add,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=self.device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

    def _mask_row_host(self, query_pos: int, total_len: int | None = None) -> torch.Tensor:
        # Trace replay pads KV to ``max_cache_len`` so the mask must match that fixed W.
        # Eager (concat) decode grows K to ``query_pos+1`` — SDPA requires mask W == K len.
        w = self.max_cache_len if total_len is None else int(total_len)
        mask_bool = build_attention_mask_query_row(w, query_pos, self.attn_slices, bsz=1)
        return to_additive(mask_bool, dtype=torch.bfloat16).reshape(1, 1, 1, w)

    def _to_torch_replicated(self, tensor: ttnn.Tensor) -> torch.Tensor:
        """Mesh-safe D2H for tensors replicated across the batch (dim 0) dimension."""
        if hasattr(self.device, "get_num_devices") and self.device.get_num_devices() > 1:
            out = ttnn.to_torch(tensor, mesh_composer=ttnn.ConcatMeshToTensor(self.device, dim=0))
            return out[:1]
        return ttnn.to_torch(tensor)

    def _rope_slice_host(self, position: int) -> tuple[torch.Tensor, torch.Tensor]:
        if self._cos_host is None:
            self._cos_host = self._to_torch_replicated(self._cos_full).to(torch.bfloat16)
            self._sin_host = self._to_torch_replicated(self._sin_full).to(torch.bfloat16)
        cos_h = self._cos_host[:, :, position : position + 1, :].contiguous()
        sin_h = self._sin_host[:, :, position : position + 1, :].contiguous()
        return cos_h, sin_h

    def _read_logits(self, batch_size: int) -> torch.Tensor:
        vocab_parallel = getattr(self.lm_head, "vocab_parallel", False)
        if self.dual_cq is not None:
            self.dual_cq.launch_logits_d2h(self.logits_tt)
            return self.dual_cq.consume_logits(batch_size)
        return logits_host_to_torch(
            ttnn.from_device(self.logits_tt), self.device, batch_size, vocab_parallel=vocab_parallel
        )

    def _prefill_forward(self) -> ttnn.Tensor:
        """KV prefix prefill (chunked and/or trace-captured) returning last-token logits."""
        return run_kv_prefill(self, use_trace_prefill=recaption_trace_prefill_enabled())

    def _upload_trace_buffer(self, torch_data: torch.Tensor, *, dtype, layout) -> ttnn.Tensor:
        if self.replicate_to_mesh is not None and dtype == ttnn.bfloat16 and layout == ttnn.TILE_LAYOUT:
            return self.replicate_to_mesh(torch_data.contiguous())
        kwargs = dict(
            dtype=dtype,
            layout=layout,
            device=self.device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        if hasattr(self.device, "get_num_devices") and self.device.get_num_devices() > 1:
            kwargs["mesh_mapper"] = ttnn.ReplicateTensorToMesh(self.device)
        return ttnn.from_torch(torch_data.contiguous(), **kwargs)

    def _copy_trace_buffer(self, torch_data: torch.Tensor, device_tt: ttnn.Tensor, *, dtype, layout) -> None:
        host_tt = ttnn.from_torch(torch_data.contiguous(), dtype=dtype, layout=layout)
        ttnn.copy_host_to_device_tensor(host_tt, device_tt)

    def _init_trace_buffers(self, token_id: int, query_pos: int) -> None:
        self._query_pos = query_pos
        tok = torch.tensor([[token_id]], dtype=torch.int32)
        self._token_ids_tt = self._upload_trace_buffer(tok, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT)
        mask_host = self._mask_row_host(query_pos)
        self._mask_tt = self._upload_trace_buffer(mask_host, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
        cos_h, sin_h = self._rope_slice_host(query_pos)
        self._cos_tt = self._upload_trace_buffer(cos_h, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
        self._sin_tt = self._upload_trace_buffer(sin_h, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
        self._write_pos_host = torch.tensor([query_pos], dtype=torch.int32)
        self._write_pos_tt = self._upload_trace_buffer(
            self._write_pos_host, dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT
        )
        self.kv_cache.set_write_pos_tensor(self._write_pos_tt)

    def _decode_step(self) -> ttnn.Tensor:
        hidden_tt = self.wte_tt.embed(self._token_ids_tt)
        hidden = self.model.forward(
            inputs_embeds=hidden_tt,
            seq_len=self._query_pos + 1,
            image_infos=self.image_infos,
            attention_mask=self._mask_tt,
            kv_cache=self.kv_cache,
            use_cache=True,
            decode_step=True,
            cos_sin=(self._cos_tt, self._sin_tt),
        )
        ttnn.deallocate(hidden_tt)
        logits_tt = self.lm_head(hidden, last_token_only=True)
        ttnn.deallocate(hidden)
        return logits_tt

    def _prepare_trace_kv(self) -> None:
        if not self.kv_cache.trace_fixed:
            self.kv_cache.promote_to_trace_buffers(self.device, self.max_cache_len)

    def _capture(self, token_id: int, query_pos: int) -> None:
        t0 = time.perf_counter()
        self._prepare_trace_kv()
        self._init_trace_buffers(token_id, query_pos)

        logits_w = self._decode_step()
        ttnn.deallocate(logits_w)
        ttnn.synchronize_device(self.device)

        if self.dual_cq is not None:
            self.dual_cq.fence_compute_before_forward()

        print(
            f"[recaption] trace decode token #1 (query_pos={query_pos}): " f"warmup done, using begin_trace_capture",
            flush=True,
        )
        self.trace_id = ttnn.begin_trace_capture(self.device, cq_id=COMPUTE_CQ)
        self.logits_tt = self._decode_step()
        print(
            f"[recaption] trace decode token #1 (query_pos={query_pos}): "
            f"using end_trace_capture (NOT execute_trace)",
            flush=True,
        )
        ttnn.end_trace_capture(self.device, self.trace_id, cq_id=COMPUTE_CQ)
        ttnn.synchronize_device(self.device)
        self.kv_cache.seq_len = query_pos + 1
        self.replay_steps = 1
        self._timing_first_decode_ms = (time.perf_counter() - t0) * 1000
        print(
            f"[recaption] decode trace captured trace_id={self.trace_id} "
            f"took {self._timing_first_decode_ms:.2f} ms",
            flush=True,
        )

    def _update_trace_inputs(self, token_id: int, query_pos: int) -> None:
        self._query_pos = query_pos
        tok = torch.tensor([[token_id]], dtype=torch.int32)
        self._copy_trace_buffer(tok, self._token_ids_tt, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT)
        self._copy_trace_buffer(
            self._mask_row_host(query_pos), self._mask_tt, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT
        )
        cos_h, sin_h = self._rope_slice_host(query_pos)
        self._copy_trace_buffer(cos_h, self._cos_tt, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
        self._copy_trace_buffer(sin_h, self._sin_tt, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
        self._write_pos_host[0] = query_pos
        self._copy_trace_buffer(
            self._write_pos_host, self._write_pos_tt, dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT
        )

    def _replay(self, *, token_idx: int, query_pos: int) -> float:
        t0 = time.perf_counter()
        if self.dual_cq is not None:
            self.dual_cq.fence_compute_before_forward()
        if token_idx == 2:
            print(
                f"[recaption] trace decode token #{token_idx} (query_pos={query_pos}): "
                f"using execute_trace (NOT begin/end capture)",
                flush=True,
            )
        ttnn.execute_trace(self.device, self.trace_id, cq_id=COMPUTE_CQ, blocking=True)
        return (time.perf_counter() - t0) * 1000

    def _emit_logits(self, batch_size: int):
        if self.return_device_logits:
            return self.logits_tt
        return self._read_logits(batch_size)

    def _eager_decode(self, token_id: int | ttnn.Tensor, query_pos: int) -> ttnn.Tensor:
        """Single-token decode without CQ0 trace capture (device-sampling path).

        ``token_id`` may be a host int (legacy) or an on-device ``[B, 1]`` id tensor
        from ``ttnn.sampling`` — the latter avoids H2D on the sampling→embed edge.

        Text-only (empty ``attn_slices``): mask + RoPE stay on device (zeros row +
        ``slice_cos_sin``). Image-span rows still use the host mask builder.
        """
        if self._cos_full is None or self._sin_full is None:
            self._cos_full, self._sin_full = self.model.layers[0].self_attn.rope.prepare_cos_sin(
                self.max_cache_len, image_infos=self.image_infos
            )
        # Growing (non-trace_fixed) KV: after concat, K length == query_pos + 1 (== S).
        total_len = query_pos + 1
        owns_token = False
        if isinstance(token_id, ttnn.Tensor):
            token_tt = token_id
        else:
            tok = torch.tensor([[int(token_id)]], dtype=torch.int32)
            token_tt = self._upload_trace_buffer(tok, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT)
            owns_token = True

        text_only = not self.attn_slices
        if text_only:
            # Pure causal grow: every prior key is visible → additive zeros.
            mask_kwargs = dict(
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=self.device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            mask_tt = ttnn.zeros((1, 1, 1, total_len), **mask_kwargs)
            rope = self.model.layers[0].self_attn.rope
            cos_tt, sin_tt = rope.slice_cos_sin(self._cos_full, self._sin_full, query_pos)
        else:
            mask_tt = self._upload_trace_buffer(
                self._mask_row_host(query_pos, total_len=total_len),
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
            )
            cos_h, sin_h = self._rope_slice_host(query_pos)
            cos_tt = self._upload_trace_buffer(cos_h, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
            sin_tt = self._upload_trace_buffer(sin_h, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)

        hidden_tt = self.wte_tt.embed(token_tt)
        hidden = self.model.forward(
            inputs_embeds=hidden_tt,
            seq_len=total_len,
            image_infos=self.image_infos,
            attention_mask=mask_tt,
            kv_cache=self.kv_cache,
            use_cache=True,
            decode_step=True,
            cos_sin=(cos_tt, sin_tt),
        )
        logits_tt = self.lm_head(hidden, last_token_only=True)
        free = [mask_tt, cos_tt, sin_tt, hidden_tt, hidden]
        if owns_token:
            free.append(token_tt)
        for t in free:
            ttnn.deallocate(t)
        self.kv_cache.seq_len = query_pos + 1
        return logits_tt

    def forward_device_token(self, token_tt: ttnn.Tensor, *, seq_len: int):
        """Decode one AR step from an on-device token id ``[B, 1]`` (no token H2D)."""
        if seq_len <= self.prefix_len:
            raise ValueError(f"forward_device_token requires seq_len > prefix_len ({self.prefix_len}), got {seq_len}")
        if not self.prefill_done:
            t0 = time.perf_counter()
            self.logits_tt = self._prefill_forward()
            self.prefill_done = True
            self._timing_prefill_ms = (time.perf_counter() - t0) * 1000
            print(
                f"[recaption] chunked prefill done prefix_len={self.prefix_len} "
                f"took {self._timing_prefill_ms:.2f} ms (device-sampling)",
                flush=True,
            )
        query_pos = seq_len - 1
        self.logits_tt = self._eager_decode(token_tt, query_pos)
        self.replay_steps += 1
        return self._emit_logits(int(token_tt.shape[0]))

    def forward(self, ids: torch.Tensor):
        B, S = ids.shape
        if S < self.prefix_len:
            raise ValueError(f"sequence length {S} < prefix length {self.prefix_len}")

        if S == self.prefix_len:
            if not self.prefill_done:
                t0 = time.perf_counter()
                self.logits_tt = self._prefill_forward()
                if self.enable_decode_trace:
                    self._prepare_trace_kv()
                self.prefill_done = True
                self._timing_prefill_ms = (time.perf_counter() - t0) * 1000
                mode = "chunked/trace" if recaption_trace_prefill_enabled() else "chunked/eager"
                print(
                    f"[recaption] trace prefill done prefix_len={self.prefix_len} "
                    f"took {self._timing_prefill_ms:.2f} ms ({mode})",
                    flush=True,
                )
            return self._emit_logits(B)

        token_id = int(ids[0, -1].item())
        query_pos = S - 1

        if not self.enable_decode_trace:
            if not self.prefill_done:
                t0 = time.perf_counter()
                self.logits_tt = self._prefill_forward()
                self.prefill_done = True
                self._timing_prefill_ms = (time.perf_counter() - t0) * 1000
                print(
                    f"[recaption] chunked prefill done prefix_len={self.prefix_len} "
                    f"took {self._timing_prefill_ms:.2f} ms (device-sampling)",
                    flush=True,
                )
            self.logits_tt = self._eager_decode(token_id, query_pos)
            self.replay_steps += 1
            return self._emit_logits(B)

        if self.trace_id is None:
            if not self.prefill_done:
                t0 = time.perf_counter()
                self.logits_tt = self._prefill_forward()
                self._prepare_trace_kv()
                self.prefill_done = True
                self._timing_prefill_ms = (time.perf_counter() - t0) * 1000
                mode = "chunked/trace" if recaption_trace_prefill_enabled() else "chunked/eager"
                print(
                    f"[recaption] trace prefill done prefix_len={self.prefix_len} "
                    f"took {self._timing_prefill_ms:.2f} ms ({mode})",
                    flush=True,
                )
            self._capture(token_id, query_pos)
            return self._emit_logits(B)

        token_idx = self.replay_steps + 1
        self._update_trace_inputs(token_id, query_pos)
        replay_ms = self._replay(token_idx=token_idx, query_pos=query_pos)
        self.kv_cache.seq_len = query_pos + 1
        self.replay_steps += 1
        self._timing_replay_total_ms += replay_ms
        self._timing_replay_count += 1
        if self._timing_replay_count == 1:
            print(
                f"[recaption] trace decode token #{token_idx} took {replay_ms:.2f} ms",
                flush=True,
            )
        return self._emit_logits(B)

    def release(self) -> None:
        if self._timing_first_decode_ms > 0 or self._timing_replay_count > 0:
            avg_replay_ms = (
                self._timing_replay_total_ms / self._timing_replay_count if self._timing_replay_count else 0.0
            )
            print(
                "[recaption] trace timing summary: "
                f"prefill={self._timing_prefill_ms:.2f} ms, "
                f"first decode (begin/end capture)={self._timing_first_decode_ms:.2f} ms, "
                f"remaining decode (execute_trace)={self._timing_replay_total_ms:.2f} ms "
                f"over {self._timing_replay_count} tokens "
                f"(avg {avg_replay_ms:.2f} ms/token)",
                flush=True,
            )
        if self.prefill_trace_id is not None:
            ttnn.release_trace(self.device, self.prefill_trace_id)
            self.prefill_trace_id = None
        if self.trace_id is not None:
            ttnn.release_trace(self.device, self.trace_id)
            self.trace_id = None
        if self.prefix_embeds_tt is not None:
            ttnn.deallocate(self.prefix_embeds_tt)
            self.prefix_embeds_tt = None
