# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Kimi-Linear-48B-A3B-Instruct on a 1xN Blackhole mesh: embeddings -> 27 decoder layers -> norm -> LM head.

State model (all device-resident, allocated once, updated in place so decode traces stay valid):
  * 7 MLA latent caches [num_blocks, 1, block, 576], replicated per chip, addressed through vLLM-style page tables.
  * 20 KDA decode states: recurrent [B, H/tp, 128, 128] fp32 + 3 conv-history rows [1, B, C/tp] bf16, one row per decode slot.
  * 20 KDA prefill scratch states (B=1) carried across the chunks of one request, then copied into that request's slot.
Prefill runs one request at a time on a 32-multiple padded chunk (valid_len masks the padding), decode runs all B slots.
"""

from __future__ import annotations

import time
from pathlib import Path

import torch
from loguru import logger

import ttnn
from models.autoports.moonshotai_kimi_linear_48b_a3b_instruct.reference.config import KimiLinearConfig
from models.autoports.moonshotai_kimi_linear_48b_a3b_instruct.reference.weights import KimiCheckpoint
from models.autoports.moonshotai_kimi_linear_48b_a3b_instruct.tt.ccl import KimiCCL
from models.autoports.moonshotai_kimi_linear_48b_a3b_instruct.tt.kda.layer import ceil32
from models.autoports.moonshotai_kimi_linear_48b_a3b_instruct.tt.layer import KimiDecoderLayer, PrecisionPolicy
from models.autoports.moonshotai_kimi_linear_48b_a3b_instruct.tt.rms_norm import RMSNorm
from models.autoports.moonshotai_kimi_linear_48b_a3b_instruct.tt.weights import as_device_tensor, linear_weight, tp_of

TILE = 32


class KimiLinearModel:
    def __init__(
        self,
        mesh_device,
        cfg: KimiLinearConfig,
        checkpoint: KimiCheckpoint | None,
        *,
        max_batch_size: int,
        cache_path: Path | None,
        precision: PrecisionPolicy | None = None,
        block_size: int = 64,
        prefill_chunk: int = 2048,
        layers: list[int] | None = None,
    ):
        self.mesh_device, self.cfg = mesh_device, cfg
        self.tp = tp_of(mesh_device)
        self.num_devices = mesh_device.get_num_devices()
        self.max_batch_size = max_batch_size
        self.block_size = block_size
        self.prefill_chunk = prefill_chunk
        self.precision = precision or PrecisionPolicy()
        self.ccl = KimiCCL(mesh_device)
        self.cache_path = cache_path
        self.layer_indices = list(range(cfg.num_hidden_layers)) if layers is None else list(layers)
        ck = checkpoint
        t0 = time.time()
        # Embedding table sharded along HIDDEN across the mesh (189 MB/chip row-major): a 755 MB replicated row-major write
        # hangs the dispatch cores on this QB2 (observed 2026-09-10, same class as the earlier 800 MB embedding hang). The
        # lookup output [1,1,S,H/tp] is all-gathered to the replicated residual.
        self.embd_weight = as_device_tensor(
            mesh_device,
            None if ck is None else ck.embedding(),
            name="embed_tokens_hs",
            dtype=ttnn.bfloat16,
            shard_dim=-1,
            cache_path=cache_path,
            layout=ttnn.ROW_MAJOR_LAYOUT,
        )
        self.layers: list[KimiDecoderLayer] = []
        for i in self.layer_indices:
            sd = None if ck is None else ck.layer_state_dict(i)
            self.layers.append(
                KimiDecoderLayer(
                    mesh_device, cfg, sd, layer_idx=i, ccl=self.ccl, cache_path=cache_path, precision=self.precision
                )
            )
            del sd
            logger.info(f"layer {i} ready ({time.time() - t0:.0f}s)")
        self.final_norm = RMSNorm(
            mesh_device,
            None if ck is None else ck.final_norm(),
            eps=cfg.rms_norm_eps,
            name="final_norm",
            cache_path=cache_path,
        )
        # LM head: vocab column-sharded [hidden, vocab/tp]
        self.padded_vocab = ceil32(cfg.vocab_size)
        lm = None if ck is None else linear_weight(ck.lm_head())
        self.lm_head = as_device_tensor(
            mesh_device, lm, name="lm_head", dtype=ttnn.bfloat8_b, shard_dim=-1, cache_path=cache_path
        )
        self.lm_compute = ttnn.init_device_compute_kernel_config(
            mesh_device.arch(), math_fidelity=ttnn.MathFidelity.HiFi2, fp32_dest_acc_en=True, packer_l1_acc=True
        )
        self.kda_layers = [l for l in self.layers if l.is_kda]
        self.mla_layers = [l for l in self.layers if not l.is_kda]
        self.caches: list[ttnn.Tensor] | None = None
        self.kda_decode_states = None
        self.kda_prefill_states = None
        logger.info(f"model built: {len(self.layers)} layers, tp={self.tp}, {time.time() - t0:.0f}s")

    # ---- state --------------------------------------------------------------------------------
    def allocate_state(self, num_blocks: int, kv_dtype=None):
        kv_dtype = kv_dtype or self.precision.kv_cache
        self.caches = [l.attn.allocate_cache(num_blocks, dtype=kv_dtype) for l in self.mla_layers]
        self.kda_decode_states = [l.attn.allocate_decode_state(self.max_batch_size) for l in self.kda_layers]
        self.kda_prefill_states = [l.attn.allocate_prefill_state() for l in self.kda_layers]
        self._zero_prefill = [l.attn.allocate_prefill_state() for l in self.kda_layers]
        return self.caches

    def reset_prefill_scratch(self):
        for st, z in zip(self.kda_prefill_states, self._zero_prefill):
            ttnn.copy(z.recurrent, st.recurrent)
            ttnn.copy(z.convolution, st.convolution)

    # ---- helpers ------------------------------------------------------------------------------
    def _replicate(self, t: torch.Tensor, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT):
        mapper = ttnn.ReplicateTensorToMesh(self.mesh_device) if self.num_devices > 1 else None
        return ttnn.from_torch(t, dtype=dtype, layout=layout, device=self.mesh_device, mesh_mapper=mapper)

    def embed(self, tokens: ttnn.Tensor) -> ttnn.Tensor:
        """tokens uint32 [1, S] (row-major, device) -> [1, 1, S, hidden] bf16 tiled."""
        x = ttnn.embedding(
            tokens,
            self.embd_weight,
            layout=ttnn.TILE_LAYOUT,
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        x = ttnn.reshape(x, (1, 1, x.shape[-2], x.shape[-1]))  # [1,1,S,H/tp]
        return self.ccl.all_gather(x, dim=3)

    def lm_head_logits(self, x: ttnn.Tensor, gather: bool = True) -> ttnn.Tensor:
        """x [1,1,S,hidden] -> logits [1,1,S,vocab] (gathered) or the vocab shard [1,1,S,vocab/tp]."""
        h = self.final_norm(x)
        logits = ttnn.linear(
            h,
            self.lm_head,
            compute_kernel_config=self.lm_compute,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            dtype=ttnn.bfloat16,
        )
        ttnn.deallocate(h)
        return self.ccl.all_gather(logits, dim=3) if gather else logits

    # ---- prefill (one request) ----------------------------------------------------------------
    def prefill(
        self,
        tokens: torch.Tensor,
        page_table_row: torch.Tensor,
        *,
        slot: int,
        return_hidden: bool = False,
        all_logits: bool = False,
    ):
        """tokens int [T]; page_table_row int32 [1, max_blocks] (blocks of this request). Fills the MLA caches and the
        KDA decode slot ``slot``. Returns host logits of the last token [vocab] (fp32) (and the last-layer hidden [T,hidden] if asked).
        """
        assert self.caches is not None, "allocate_state() first"
        T = int(tokens.numel())
        pt = self._replicate(page_table_row.to(torch.int32), dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT)
        self.reset_prefill_scratch()
        hidden_chunks = []
        logit_chunks = []
        last_logits = None
        for start in range(0, T, self.prefill_chunk):
            end = min(T, start + self.prefill_chunk)
            valid = end - start
            T_pad = ceil32(valid)
            ids = torch.zeros(1, T_pad, dtype=torch.int32)
            ids[0, :valid] = tokens[start:end]
            x = self.embed(self._replicate(ids, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT))
            ki = mi = 0
            for layer in self.layers:
                if layer.is_kda:
                    x, new_state = layer.forward_prefill(x, kda_state=self.kda_prefill_states[ki], valid_len=valid)
                    st = self.kda_prefill_states[ki]
                    ttnn.copy(new_state.recurrent, st.recurrent)
                    ttnn.copy(new_state.convolution, st.convolution)
                    ttnn.deallocate(new_state.recurrent)
                    ttnn.deallocate(new_state.convolution)
                    ki += 1
                else:
                    x, _ = layer.forward_prefill(
                        x, cache=self.caches[mi], page_table=pt, user_id=0, valid_len=valid, chunk_start=start
                    )
                    mi += 1
            if return_hidden:
                hidden_chunks.append(ttnn.to_torch(ttnn.get_device_tensors(x)[0]).float()[0, 0, :valid])
            if all_logits:
                logits = self.lm_head_logits(x)  # [1,1,T_pad,vocab]
                logit_chunks.append(
                    ttnn.to_torch(ttnn.get_device_tensors(logits)[0]).float()[0, 0, :valid, : self.cfg.vocab_size]
                )
                ttnn.deallocate(logits)
            elif end == T:
                last = ttnn.slice(x, (0, 0, valid - 1, 0), (1, 1, valid, x.shape[-1]))
                logits = self.lm_head_logits(last)
                last_logits = (
                    ttnn.to_torch(ttnn.get_device_tensors(logits)[0]).float().reshape(-1)[: self.cfg.vocab_size]
                )
                ttnn.deallocate(logits)
                ttnn.deallocate(last)
            ttnn.deallocate(x)
        for layer_state, layer in zip(self.kda_prefill_states, self.kda_layers):
            layer.attn.prefill_state_to_decode(
                layer_state, self.kda_decode_states[self.kda_layers.index(layer)], slot=slot
            )
        if all_logits:
            all_l = torch.cat(logit_chunks, 0)  # [T, vocab]
            return (all_l, torch.cat(hidden_chunks, 0)) if return_hidden else all_l
        return (last_logits, torch.cat(hidden_chunks, 0)) if return_hidden else last_logits

    def prefill_all_logits(self, tokens: torch.Tensor, page_table_row: torch.Tensor, *, slot: int) -> torch.Tensor:
        return self.prefill(tokens, page_table_row, slot=slot, all_logits=True)

    def reset_slot(self, slot: int) -> None:
        """Zero the KDA decode state of one slot (the MLA caches are rewritten by the next prefill)."""
        for ds, z in zip(self.kda_decode_states, self._zero_prefill):
            if ds.recurrent.shape[0] == 1:
                ttnn.copy(z.recurrent, ds.recurrent)
                zc = ttnn.to_layout(z.convolution, ttnn.TILE_LAYOUT)
                for j, h in enumerate(ds.conv_history):
                    row = ttnn.slice(zc, (0, j, 0), (1, j + 1, zc.shape[-1]))
                    ttnn.copy(row, h)
                    ttnn.deallocate(row)
                ttnn.deallocate(zc)
            else:
                zero = ttnn.zeros(
                    (1,) + tuple(ds.recurrent.shape)[1:],
                    dtype=ds.recurrent.dtype,
                    layout=ttnn.TILE_LAYOUT,
                    device=self.mesh_device,
                )
                ttnn.experimental.slice_write(
                    zero, ds.recurrent, [slot, 0, 0, 0], [slot + 1] + list(tuple(ds.recurrent.shape)[1:]), [1, 1, 1, 1]
                )
                ttnn.deallocate(zero)
                for h in ds.conv_history:
                    zero = ttnn.zeros(
                        (1, 1, h.shape[-1]), dtype=h.dtype, layout=ttnn.TILE_LAYOUT, device=self.mesh_device
                    )
                    ttnn.experimental.slice_write(zero, h, [0, slot, 0], [1, slot + 1, h.shape[-1]], [1, 1, 1])
                    ttnn.deallocate(zero)

    def remap_slots(self, remap) -> None:
        """slot i takes the KDA state previously at slot remap[i] (vLLM batch condense); identity is a no-op."""
        idx = [int(remap[i]) for i in range(self.max_batch_size)]
        if all(i == j for i, j in enumerate(idx)):
            return
        for ds in self.kda_decode_states:
            rows = [ttnn.slice(ds.recurrent, (i, 0, 0, 0), (i + 1,) + tuple(ds.recurrent.shape)[1:]) for i in idx]
            new = ttnn.concat(rows, dim=0)
            ttnn.copy(new, ds.recurrent)
            ttnn.deallocate(new)
            for r in rows:
                ttnn.deallocate(r)
            for h in ds.conv_history:
                rows = [ttnn.slice(h, (0, i, 0), (1, i + 1, h.shape[-1])) for i in idx]
                new = ttnn.concat(rows, dim=1)
                ttnn.copy(new, h)
                ttnn.deallocate(new)
                for r in rows:
                    ttnn.deallocate(r)

    # ---- host-side decode inputs (for traced decode: created on host, copied into persistent device tensors) ---------
    def _host_decode_inputs(self, tokens: torch.Tensor, cur_pos: torch.Tensor, page_table: torch.Tensor):
        B = self.max_batch_size
        mapper = ttnn.ReplicateTensorToMesh(self.mesh_device) if self.num_devices > 1 else None
        tok = ttnn.from_torch(
            tokens.to(torch.int32).reshape(1, B), dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT, mesh_mapper=mapper
        )
        pos = ttnn.from_torch(
            cur_pos.to(torch.int32).reshape(B), dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT, mesh_mapper=mapper
        )
        pt = ttnn.from_torch(
            page_table.to(torch.int32), dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT, mesh_mapper=mapper
        )
        return tok, pos, pt

    # ---- decode (all slots) -------------------------------------------------------------------
    def decode(self, tokens: torch.Tensor, cur_pos: torch.Tensor, page_table: torch.Tensor) -> torch.Tensor:
        """tokens int [B]; cur_pos int32 [B] (position of the new token); page_table int32 [B, max_blocks] -> host logits [B, vocab] fp32."""
        B = self.max_batch_size
        assert tokens.numel() == B and cur_pos.numel() == B and page_table.shape[0] == B
        tok = self._replicate(tokens.to(torch.int32).reshape(1, B), dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT)
        pos = self._replicate(cur_pos.to(torch.int32).reshape(B), dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT)
        pt = self._replicate(page_table.to(torch.int32), dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT)
        logits = self.decode_device(tok, pos, pt)
        out = ttnn.to_torch(ttnn.get_device_tensors(logits)[0]).float().reshape(B, -1)[:, : self.cfg.vocab_size]
        ttnn.deallocate(logits)
        return out

    def decode_device(self, tok: ttnn.Tensor, pos: ttnn.Tensor, pt: ttnn.Tensor, gather: bool = True) -> ttnn.Tensor:
        """Device-only decode step (trace-capturable): tok uint32 [1,B], pos int32 [B], pt int32 [B, blocks] -> logits [1,1,B,vocab]."""
        x = self.embed(tok)
        ki = mi = 0
        for layer in self.layers:
            if layer.is_kda:
                x = layer.forward_decode(x, kda_state=self.kda_decode_states[ki])
                ki += 1
            else:
                x = layer.forward_decode(x, cache=self.caches[mi], page_table=pt, cur_pos=pos)
                mi += 1
        logits = self.lm_head_logits(x, gather=gather)
        ttnn.deallocate(x)
        return logits
