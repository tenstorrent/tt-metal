"""Music3Backbone: tt_transformers' Transformer (the HF Qwen3-8B path) with MiniMax Music 3's frame embedding as the
residual, TWO users (conditional / unconditional rows for classifier-free guidance) and a post-final-norm hidden tap.

Integration points with tt_transformers (models/tt_transformers/tt/model.py, generator.py), as in the S2 Pro autoport:
  * prefill: `prepare_inputs_prefill` normally embeds tokens via `self.embd`; we swap `self.embd` for a stub returning a
    host-computed residual (the FULL 200k embedding table lives on the host; the device table is the sliced one).
  * decode: `prepare_decode_inputs_host` returns a 5th host tensor (the residual for both rows); the Generator copies
    every input each step (host sampling => reload_inputs), `bind_decode_trace_inputs` binds it at trace capture,
    `_transform_decode_inputs_device` returns it as the layer-0 input.
  * hidden tap: `forward` applies `self.norm` then `self.lm_head`; wrapping lm_head records its input per mode/user.
The device LM head is the SLICED head (16 385 rows: semantic codes + <|audio_end|>), see tt/weights.py.
"""
from __future__ import annotations

import contextlib
from typing import Dict, Optional

import torch

import ttnn
from models.autoports.minimaxai_minimax_music3.config import SLICED_VOCAB
from models.tt_transformers.tt.common import Mode, copy_host_to_device
from models.tt_transformers.tt.model import Transformer


class _LMHeadTap:
    def __init__(self, inner, owner):
        object.__setattr__(self, "_inner", inner)
        object.__setattr__(self, "_owner", owner)

    def __call__(self, x, *a, **k):
        # LMHead may deallocate its input: hand it a clone, stash the original (device op only => trace-capture safe).
        # Stashes are per mode (and per user for prefill): a decode-trace replay runs no Python, so the decode stash keeps
        # pointing at the traced tensor.
        o = self._owner
        if o._mode == "prefill":
            o._hidden_prefill[o._cur_user] = x
        else:
            o._hidden_decode = x
        return self._inner(ttnn.clone(x), *a, **k)

    def forward(self, x, *a, **k):
        return self(x, *a, **k)

    def __getattr__(self, name):
        return getattr(self._inner, name)


class Music3Backbone(Transformer):
    def __init__(self, args, dtype, mesh_device, state_dict, weight_cache_path, *, full_embed: torch.Tensor, **kwargs):
        super().__init__(args, dtype, mesh_device, state_dict, weight_cache_path, **kwargs)
        assert args.vocab_size >= SLICED_VOCAB, args.vocab_size
        self.full_embed = full_embed.to(torch.bfloat16)  # [200000, D] host table for the text prompt
        self._hidden_prefill: Dict[int, ttnn.Tensor] = {}
        self._hidden_decode: Optional[ttnn.Tensor] = None
        self._mode = "prefill"
        self._cur_user = 0
        self._pending_residual: Optional[torch.Tensor] = None
        self._decode_extra_device = None
        self.lm_head = _LMHeadTap(self.lm_head, self)

    # ------------------------------------------------------------------ prefill (text prompt, one user at a time)
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
        assert not trace_enabled, "Music3Backbone: traced prefill is a later optimization"
        assert batch_size == 1, "Music3Backbone prefills one row at a time (two rows per request)"
        self._mode = "prefill"
        self._cur_user = int(user_id)
        flat = tokens.reshape(-1).to(torch.int64)
        S = flat.shape[0]
        x = self.full_embed[flat].view(1, 1, S, -1)
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

    # ------------------------------------------------------------------ decode (both rows, one frame embedding)
    def set_decode_residual(self, x: torch.Tensor):
        """x [B, D] (bf16): the frame embedding fed to every row at the next decode step."""
        self._pending_residual = x.to(torch.bfloat16)

    def prepare_decode_inputs_host(self, tokens, current_pos, page_table=None):
        base = super().prepare_decode_inputs_host(tokens, current_pos, page_table)
        B = tokens.shape[0]
        x = (
            self._pending_residual
            if self._pending_residual is not None
            else torch.zeros(B, self.args.dim, dtype=torch.bfloat16)
        )
        assert x.shape == (B, self.args.dim), (x.shape, B)
        x_tt = self.args.prepare_residual_tensor_decode(
            x.view(B, 1, -1), self.args.get_residual_mem_config(Mode.DECODE, self.prefetcher), on_host=True
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

    # ------------------------------------------------------------------ hidden state readback
    def _to_host_rows(self, x) -> torch.Tensor:
        per = [ttnn.to_torch(t).float() for t in ttnn.get_device_tensors(x)]
        t = per[0]
        if len(per) > 1 and t.shape[-1] * len(per) == self.args.dim:
            t = torch.cat(per, dim=-1)
        return t.reshape(-1, t.shape[-1])

    def read_prefill_hidden(self, user_id: int, last_token_idx: int) -> torch.Tensor:
        """Post-final-norm hidden state of the last prompt token for one user: [D] float32."""
        x = self._hidden_prefill[user_id]
        rows = self._to_host_rows(x)
        return rows[last_token_idx % rows.shape[0]]

    def read_decode_hidden(self, batch: int) -> torch.Tensor:
        """Post-final-norm hidden states of the last decode step: [batch, D] float32 (rows = users)."""
        assert self._hidden_decode is not None, "no decode forward has run yet"
        return self._to_host_rows(self._hidden_decode)[:batch]

    def free_prefill_hidden(self):
        for x in self._hidden_prefill.values():
            try:
                ttnn.deallocate(x)
            except Exception:
                pass
        self._hidden_prefill = {}
