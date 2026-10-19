# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Functional decoder contract for GLM-5.3-Flash (tt-model-bringup stage 1, single device).

One ``FunctionalDecoder`` is one decoder layer (``layer_idx``) of any of the three layer kinds
(KDA + dense MLP: layers 0-2; DSA + MoE: 3+4k; KDA + MoE: the rest) with its device-owned
autoregressive state:

* KDA layers: a ``KdaState`` per user (recurrent ``[1,H,K,V]`` fp32 + conv history
  ``[1,3,3*8192]`` bf16) written by prefill, stacked into a ``[B,...]`` state for decode;
* DSA layers: a paged latent cache ``[max_blocks, 1, 32, 512]`` bf16 shared by ``max_batch``
  users with a ``[max_batch, blocks_per_seq]`` page table.

Tensor contract (the HF residual is the mHC stream tensor ``[B, S, 4, 4096]``):
* ``prefill_forward(streams, user, seq_len)``: one user; ``streams`` is the packed fp32 device
  tensor ``[1, 1, T_pad, 4*4096]`` (stream i at columns [i*4096, (i+1)*4096), T_pad % 32 == 0,
  rows past ``seq_len`` are padding and come back unspecified). Fills that user's cache/state
  from position 0. Returns the packed output streams.
* ``decode_forward(streams, positions)``: ``streams`` ``[1, 1, B, 4*4096]`` fp32, one token for
  each of the B prefetched users (user b = row b), ``positions`` ``[B]`` int32 row-major device
  tensor holding each user's current context length. Returns ``[1, 1, B, 4*4096]``.
Use ``host_streams_to_device`` / ``device_streams_to_host`` at the test boundary only.
"""

from __future__ import annotations

from pathlib import Path

import torch

import ttnn
from models.common.lightweightmodule import LightweightModule
from models.demos.deepseek_v3_d_p.tt.kda.kda import KdaState
from models.demos.glm53_flash.tt.config import GLM53FlashConfig
from models.demos.glm53_flash.tt.decoder_layer import LayerState, TtDecoderLayer
from models.demos.glm53_flash.tt.dsa import TtLatentCache

TILE = 32


def host_streams_to_device(streams: torch.Tensor, device, pad_to: int | None = None) -> ttnn.Tensor:
    """HF ``[1, S, n, C]`` (any dtype) -> packed ``[1, 1, T_pad, n*C]`` fp32 on device."""
    _, S, n, C = streams.shape
    T_pad = pad_to if pad_to is not None else -(-S // TILE) * TILE
    packed = torch.zeros(1, 1, T_pad, n * C, dtype=torch.float32)
    packed[0, 0, :S] = streams.reshape(S, n * C).float()
    return ttnn.from_torch(packed, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)


def device_streams_to_host(packed: ttnn.Tensor, n: int, seq_len: int | None = None) -> torch.Tensor:
    """Packed ``[1, 1, T, n*C]`` -> HF ``[1, S, n, C]`` fp32 (S = seq_len rows)."""
    t = ttnn.to_torch(packed).float()[0, 0]
    T, nC = t.shape
    S = T if seq_len is None else seq_len
    return t[:S].reshape(1, S, n, nC // n)


class FunctionalDecoder(LightweightModule):
    def __init__(
        self,
        layer: TtDecoderLayer,
        *,
        max_batch: int,
        max_seq_len: int,
        block_size: int = TILE,
    ):
        self.layer = layer
        self.cfg = layer.cfg
        self.device = layer.device
        self.layer_idx = layer.layer_idx
        self.max_batch = max_batch
        self.max_seq_len = max_seq_len
        self.cache = None
        if not layer.is_kda:
            self.cache = TtLatentCache(
                self.device, self.cfg, max_batch=max_batch, max_seq_len=max_seq_len, block_size=block_size
            )
        # persistent batched KDA state (max_batch slots): prefill writes slot `user`, decode reads
        # and updates all slots in place, so captured decode traces always see the live state.
        self.kda_batched: KdaState | None = None
        if layer.is_kda:
            self.kda_batched = layer.attn.allocate_state(max_batch)

    @classmethod
    def from_state_dict(
        cls,
        state_dict: dict[str, torch.Tensor],
        *,
        hf_config,
        layer_idx: int,
        mesh_device,
        max_batch: int = 1,
        max_seq_len: int = 4096,
        weight_cache_path: Path | None = None,
        weight_dtype=ttnn.bfloat8_b,
        **kwargs,
    ) -> "FunctionalDecoder":
        """``state_dict``: this layer's raw checkpoint tensors (prefix stripped, fp8 dequantized).
        ``hf_config``: a ``GLM53FlashConfig``, or a path to the checkpoint ``config.json``."""
        cfg = hf_config if isinstance(hf_config, GLM53FlashConfig) else GLM53FlashConfig.from_hf_config_json(hf_config)
        layer = TtDecoderLayer(
            mesh_device, cfg, layer_idx, state_dict, weight_cache_path=weight_cache_path, weight_dtype=weight_dtype
        )
        return cls(layer, max_batch=max_batch, max_seq_len=max_seq_len, **kwargs)

    # ------------------------------------------------------------------ state
    def _state_for_prefill(self, user: int) -> LayerState:
        if self.layer.is_kda:
            return LayerState(kda=self.layer.attn.allocate_state(1))
        return LayerState(cache=self.cache)

    def _write_user_kda_state(self, user: int, new: KdaState):
        """Copy a freshly prefilled single-user state into slot ``user`` of the persistent batched state."""
        dst = self.kda_batched
        if self.max_batch == 1:
            ttnn.copy(new.recurrent, dst.recurrent)
            ttnn.copy(new.convolution, dst.convolution)
        else:
            for src_t, dst_t in ((new.recurrent, dst.recurrent), (new.convolution, dst.convolution)):
                rows = []
                for u in range(self.max_batch):
                    if u == user:
                        rows.append(src_t)
                    else:
                        end = list(dst_t.shape)
                        start = [0] * len(end)
                        start[0], end[0] = u, u + 1
                        rows.append(ttnn.slice(dst_t, start, end))
                stacked = ttnn.concat(rows, dim=0)
                ttnn.copy(stacked, dst_t)
                ttnn.deallocate(stacked)
        ttnn.deallocate(new.recurrent)
        ttnn.deallocate(new.convolution)

    def _state_for_decode(self, batch: int) -> LayerState:
        if not self.layer.is_kda:
            return LayerState(cache=self.cache)
        if batch != self.max_batch:
            raise ValueError(f"decode batch {batch} must equal max_batch {self.max_batch} (fixed user slots)")
        return LayerState(kda=self.kda_batched)

    # ---------------------------------------------------------------- forwards
    def prefill_forward(self, streams: ttnn.Tensor, *, user: int = 0, seq_len: int | None = None) -> ttnn.Tensor:
        T = streams.shape[-2]
        seq_len = T if seq_len is None else seq_len
        if T % TILE or seq_len > T or seq_len <= 0:
            raise ValueError(f"T_pad={T} must be a multiple of {TILE} and 0 < seq_len={seq_len} <= T_pad")
        if user >= self.max_batch:
            raise ValueError(f"user {user} >= max_batch {self.max_batch}")
        if seq_len > self.max_seq_len:
            raise ValueError(f"seq_len {seq_len} > max_seq_len {self.max_seq_len}")
        out, new_state = self.layer.prefill_forward(streams, self._state_for_prefill(user), user=user, seq_len=seq_len)
        if self.layer.is_kda:
            self._write_user_kda_state(user, new_state.kda)
        return out

    def decode_forward(self, streams: ttnn.Tensor, *, positions: ttnn.Tensor) -> ttnn.Tensor:
        B = streams.shape[-2]
        out, new_state = self.layer.decode_forward(streams, self._state_for_decode(B), positions=positions)
        if self.layer.is_kda:
            assert new_state.kda is self.kda_batched, "KDA decode must update the persistent state in place"
        return out

    # ------------------------------------------------------------------- trace
    def capture_decode_trace(self, streams: ttnn.Tensor, positions: ttnn.Tensor, *, cq_id: int = 0):
        """Warm up (compile) one decode step on the persistent inputs, then capture the next one.
        Returns (trace_id, output tensor). Replay with ``replay_decode``; the state tensors are
        updated in place by the captured graph. The warm-up step consumes one real token: call
        this with the first decode token's inputs and treat its output as that step's result."""
        warm = self.decode_forward(streams, positions=positions)
        ttnn.synchronize_device(self.device)
        trace_id = ttnn.begin_trace_capture(self.device, cq_id=cq_id)
        out = self.decode_forward(streams, positions=positions)
        ttnn.end_trace_capture(self.device, trace_id, cq_id=cq_id)
        return trace_id, out, warm

    def replay_decode(self, trace_id, *, cq_id: int = 0, blocking: bool = True):
        ttnn.execute_trace(self.device, trace_id, cq_id=cq_id, blocking=blocking)
