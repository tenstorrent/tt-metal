"""S2SlowTransformer: tt_transformers' Transformer with Fish S2 Pro's frame embedding and a tap on the
post-final-norm hidden state (what the fast codebook decoder consumes).

Fish's slow-tower input for position t is
    x_t = E_text[tok_t]                                       if tok_t is not a semantic token
    x_t = (E_text[tok_t] + sum_i E_cb[code_{i,t} + i*4096]) / sqrt(11)   otherwise
Phase A computes this on the host (bf16) and ships it as the residual; Phase D moves it on device.

Integration points with tt_transformers (models/tt_transformers/tt/model.py, generator.py):
  * prefill: `prepare_inputs_prefill` normally embeds tokens via `self.embd`; we swap `self.embd` for a stub
    returning our host-computed residual for the duration of the parent call (hidden-sharded like Embedding).
  * decode: `prepare_decode_inputs_host` returns a 5th host tensor (the residual); the Generator tolerates
    extras, calls `bind_decode_trace_inputs(device_inputs)` when it records the decode trace, and refreshes
    every input each step (host sampling => reload_inputs=True). `_transform_decode_inputs_device` returns it.
  * hidden tap: `forward` applies `self.norm` then `self.lm_head`; wrapping lm_head records its input.
"""
from __future__ import annotations

import contextlib
import math
from typing import Optional

import torch

import ttnn
from models.autoports.fishaudio_s2_pro.config import S2Config
from models.tt_transformers.tt.common import Mode, copy_host_to_device
from models.tt_transformers.tt.model import Transformer


class _LMHeadTap:
    def __init__(self, inner, owner):
        object.__setattr__(self, "_inner", inner)
        object.__setattr__(self, "_owner", owner)

    def __call__(self, x, *a, **k):
        # Keep the post-norm hidden state alive for the fast decoder: LMHead may deallocate its input, so hand
        # it a clone and stash the original (device op only => trace-capture safe). Stashes are PER MODE: a
        # decode-trace replay runs no Python, so the decode stash must keep pointing at the traced tensor even
        # after later (untraced) prefills — a single shared stash would go stale on the second request.
        self._owner._hidden[self._owner._mode] = x
        return self._inner(ttnn.clone(x), *a, **k)

    def forward(self, x, *a, **k):
        return self(x, *a, **k)

    def __getattr__(self, name):
        return getattr(self._inner, name)


class S2SlowTransformer(Transformer):
    def __init__(
        self,
        args,
        dtype,
        mesh_device,
        state_dict,
        weight_cache_path,
        *,
        codebook_table: torch.Tensor,
        s2cfg: S2Config,
        **kwargs,
    ):
        super().__init__(args, dtype, mesh_device, state_dict, weight_cache_path, **kwargs)
        self.s2cfg = s2cfg
        prefix = args.get_state_dict_prefix("", None)
        self.tok_table = state_dict[prefix + "tok_embeddings.weight"].to(torch.bfloat16)
        self.codebook_table = codebook_table.to(torch.bfloat16)
        self.cb_offsets = (torch.arange(s2cfg.num_codebooks, dtype=torch.int64) * s2cfg.codebook_size).view(-1, 1)
        self.cb_scale = 1.0 / math.sqrt(s2cfg.num_codebooks + 1)
        self._pending_codes: Optional[torch.Tensor] = None
        self._pending_offset = 0
        self._decode_extra_device = None
        self._hidden = {"prefill": None, "decode": None}
        self._mode = "prefill"
        self.lm_head = _LMHeadTap(self.lm_head, self)

    # ------------------------------------------------------------------ frame codes / embedding
    def set_frame_codes(self, codes: torch.Tensor, offset: int = 0):
        """codes: (num_codebooks, T) for absolute positions offset..offset+T of the next prefill/decode call."""
        self._pending_codes = codes.to(torch.int64).reshape(self.s2cfg.num_codebooks, -1)
        self._pending_offset = int(offset)

    def _codes_for(self, start: int, length: int) -> torch.Tensor:
        out = torch.zeros(self.s2cfg.num_codebooks, length, dtype=torch.int64)
        if self._pending_codes is not None:
            s = start - self._pending_offset
            src = self._pending_codes[:, max(s, 0) : max(s + length, 0)]
            if src.shape[1] > 0:
                dst0 = max(-s, 0)
                out[:, dst0 : dst0 + src.shape[1]] = src
        return out

    def embed_frames_host(self, tokens: torch.Tensor, codes: torch.Tensor) -> torch.Tensor:
        """tokens (T,) int64, codes (num_codebooks, T) int64 -> (T, D) bf16."""
        tokens = tokens.to(torch.int64)
        x = self.tok_table[tokens]
        cb = self.codebook_table[(codes + self.cb_offsets).reshape(-1)].view(
            self.s2cfg.num_codebooks, tokens.shape[0], -1
        )
        cb = cb.float().sum(dim=0)
        mask = ((tokens >= self.s2cfg.semantic_begin_id) & (tokens <= self.s2cfg.semantic_end_id)).view(-1, 1)
        return torch.where(mask, ((x.float() + cb) * self.cb_scale).to(torch.bfloat16), x)

    # ------------------------------------------------------------------ prefill
    @contextlib.contextmanager
    def _embd_override(self, x_tt):
        real = self.embd

        class _Stub:
            def __call__(self, tokens, memory_config=None):
                return x_tt

        self.embd = _Stub()
        try:
            yield
        finally:
            self.embd = real

    def prepare_inputs_prefill(
        self,
        tokens,
        start_pos=0,
        page_table=None,
        chunk_page_table=None,
        chunk_start_idx=None,
        trace_enabled=False,
        last_token_idx=None,
        global_user_id=None,
        batch_size=1,
        user_id=0,
        **kwargs,
    ):
        assert not trace_enabled, "S2SlowTransformer: traced prefill is Phase D"
        assert batch_size == 1, "S2SlowTransformer serves one request at a time"
        self._mode = "prefill"
        flat = tokens.reshape(-1).to(torch.int64)
        S = flat.shape[0]
        chunk_start = int(chunk_start_idx) if chunk_start_idx is not None else int(start_pos or 0)
        x = self.embed_frames_host(flat, self._codes_for(chunk_start, S)).view(1, 1, S, -1)
        x_tt = ttnn.from_torch(
            x,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=self.mesh_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ShardTensor2dMesh(self.mesh_device, dims=(None, 3), mesh_shape=self.args.cluster_shape),
        )
        with self._embd_override(x_tt):
            return super().prepare_inputs_prefill(
                tokens,
                start_pos=start_pos,
                page_table=page_table,
                chunk_page_table=chunk_page_table,
                chunk_start_idx=chunk_start_idx,
                trace_enabled=trace_enabled,
                last_token_idx=last_token_idx,
                global_user_id=global_user_id,
                batch_size=batch_size,
                user_id=user_id,
                **kwargs,
            )

    # ------------------------------------------------------------------ decode
    def prepare_decode_inputs_host(self, tokens, current_pos, page_table=None):
        base = super().prepare_decode_inputs_host(tokens, current_pos, page_table)
        tok = tokens.reshape(-1)[:1].to(torch.int64)
        pos = int(current_pos.reshape(-1)[0])
        x = self.embed_frames_host(tok, self._codes_for(pos, 1))  # (1, D)
        x_tt = self.args.prepare_residual_tensor_decode(
            x.view(1, 1, -1), self.args.get_residual_mem_config(Mode.DECODE, self.prefetcher), on_host=True
        )
        return (*base, x_tt)

    def prepare_inputs_decode(self, *inputs):
        host_inputs = self.prepare_decode_inputs_host(*inputs)
        device_inputs = copy_host_to_device(host_inputs, mesh_device=self.mesh_device)
        self.bind_decode_trace_inputs(device_inputs)
        return device_inputs

    def bind_decode_trace_inputs(self, device_inputs):
        self._decode_extra_device = device_inputs[4] if len(device_inputs) > 4 else None

    def _transform_decode_inputs_device(self, tokens):
        assert self._decode_extra_device is not None, "decode residual not staged (prepare_inputs_decode not called)"
        self._mode = "decode"
        mem = self.args.get_residual_mem_config(Mode.DECODE, self.prefetcher)
        x = ttnn.unsqueeze_to_4D(self._decode_extra_device)
        return ttnn.to_memory_config(x, mem)

    # ------------------------------------------------------------------ hidden state
    def read_last_hidden(self, mode: str = None) -> torch.Tensor:
        """Post-final-norm hidden state seen by the LM head, as (rows, D) float32 on host.
        Decode: rows = 32 (padded batch), row 0 is the user. Prefill: the last 32-token tile.
        `mode` selects the prefill or decode stash (default: the mode of the last forward)."""
        x = self._hidden[mode or self._mode]
        assert x is not None, f"no {mode or self._mode} forward has run yet"
        per = [ttnn.to_torch(t).float() for t in ttnn.get_device_tensors(x)]
        t = per[0]
        if len(per) > 1 and t.shape[-1] * len(per) == self.args.dim:
            t = torch.cat(per, dim=-1)
        return t.reshape(-1, t.shape[-1])
