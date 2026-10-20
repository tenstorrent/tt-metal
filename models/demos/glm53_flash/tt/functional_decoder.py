# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Functional decoder contract for GLM-5.3-Flash (tt-model-bringup stage 1, single device).

One ``FunctionalDecoder`` is one decoder layer (``layer_idx``) of any of the three layer kinds
(KDA + dense MLP: layers 0-2; DSA + MoE: 3+4k; KDA + MoE: the rest) with its device-owned
autoregressive state:

* KDA layers: a persistent ``[max_batch, ...]`` ``KdaState`` (recurrent ``[B,64,128,128]`` fp32 +
  conv history ``[B,3,3*8192]`` bf16); prefill writes slot ``user``, decode updates all slots in place;
* DSA layers: a paged latent cache ``[max_blocks, 1, 32, 512]`` bf16 shared by ``max_batch`` users
  (``[max_batch, blocks_per_seq]`` page table, any unique block mapping) and the pooled-key index
  (``TtIndexCache``), both maintained on every prefill/decode call.

Tensor contract (the HF residual is the mHC stream tensor ``[B, S, 4, 4096]``):
* ``prefill_forward(streams, user, seq_len, start=0)``: one user, one chunk; ``streams`` is the
  packed fp32 device tensor ``[1, 1, T_pad, 4*4096]`` (stream i at columns [i*4096, (i+1)*4096),
  T_pad % 32 == 0, rows past ``seq_len`` are padding and come back unspecified). Row i is the
  token at absolute position ``start + i``. Returns the packed output streams.
  Chunked prefill: a prompt may be prefilled in consecutive chunks. ``start == 0`` begins a new
  prompt (the user's KDA state is reset / the DSA cache is refilled from 0); every later chunk
  must satisfy ``start % 32 == 0``, ``start == next_start[user]`` (the previous chunk's
  ``start + seq_len``, so chunks are contiguous and a non-32-aligned ``seq_len`` is only possible
  in the last chunk of a prompt) and ``start + seq_len <= max_seq_len``. The chunk continues from
  the device-owned state: KDA reads the user's recurrent + conv carries out of the batched state
  and writes the new ones back; DSA writes the chunk's latent into the user's pages from block
  ``start // 32``, maintains the pooled-key index from position ``start`` (the ring holds the
  previous chunk's tail rows) and attends every chunk row over the full prefix [0, start + T_pad):
  rows at absolute positions < 2048 with the dense causal SDPA (K/V from the cached latent of
  [0, start) plus the chunk's), rows from 2048 on with the indexer + sparse_sdpa, so the result
  equals the single-shot prefill of the whole prompt (same kernels, same row classes).
* ``decode_forward(streams, positions)``: ``streams`` ``[1, 1, B, 4*4096]`` fp32, one token for
  each of the B == max_batch user slots (user b = row b); ``positions`` is a ``DecodePositions``
  built by ``make_decode_positions(host_positions)`` and refreshed in place by
  ``update_decode_positions`` (persistent ``[B]`` int32 device tensors: the position, position // 4
  and position % 4, plus the host values). Returns ``[1, 1, B, 4*4096]``.
  DSA layers attend densely while every user's position is < index_topk (2048) and switch to the
  indexer + sparse_sdpa path (host chunk geometry, not traced) as soon as one user is beyond.
* ``capture_decode_trace`` / ``replay_decode``: the dense decode graph is capturable; the captured
  graph reads the persistent position tensors and updates the state in place. Capturing a DSA
  layer whose ``max_seq_len`` exceeds index_topk is refused unless the caller opts in, because
  replays past position 2047 would run the dense attention outside HF semantics.
Use ``host_streams_to_device`` / ``device_streams_to_host`` at the test boundary only.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch

import ttnn
from models.common.lightweightmodule import LightweightModule
from models.demos.deepseek_v3_d_p.tt.kda.kda import KdaState
from models.demos.glm53_flash.tt.config import GLM53FlashConfig
from models.demos.glm53_flash.tt.decoder_layer import LayerState, TtDecoderLayer
from models.demos.glm53_flash.tt.dsa import TtLatentCache, decode_position_tensors

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


@dataclass
class DecodePositions:
    """Persistent decode-step position inputs (one entry per user slot)."""

    positions: ttnn.Tensor  # [B] int32: position of the token being decoded (= context length)
    pool_ids: ttnn.Tensor  # [B] int32: position // index_kpool
    ring_slots: ttnn.Tensor  # [B] int32: position % index_kpool
    host: list[int]  # the same values on the host (chunk geometry for the sparse path)


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
        # chunked prefill bookkeeping: the position the next chunk of each user must start at
        self.next_start: list[int] = [0] * max_batch

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
    def _state_for_prefill(self, user: int, start: int = 0) -> tuple[LayerState, list[ttnn.Tensor]]:
        """The layer state a prefill chunk of ``user`` starts from, plus the temporaries to release
        after the chunk. KDA: a zero state for a new prompt (start == 0), otherwise the user's
        current carries read out of the persistent batched state (slot ``user``)."""
        if not self.layer.is_kda:
            return LayerState(cache=self.cache), []
        if start == 0:
            return LayerState(kda=self.layer.attn.allocate_state(1)), []
        src = self.kda_batched
        if self.max_batch == 1:  # ttKDA only reads its input state; the new carries are fresh tensors
            return LayerState(kda=src), []
        rec = ttnn.slice(src.recurrent, [user, 0, 0, 0], [user + 1] + list(src.recurrent.shape)[1:])
        conv = ttnn.slice(src.convolution, [user, 0, 0], [user + 1] + list(src.convolution.shape)[1:])
        return LayerState(kda=KdaState(recurrent=rec, convolution=conv)), [rec, conv]

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
        if batch != self.max_batch:
            raise ValueError(f"decode batch {batch} must equal max_batch {self.max_batch} (fixed user slots)")
        if not self.layer.is_kda:
            return LayerState(cache=self.cache)
        return LayerState(kda=self.kda_batched)

    # -------------------------------------------------------------- positions
    def make_decode_positions(self, host_positions: list[int]) -> DecodePositions:
        """Upload the three persistent [B] int32 tensors for ``host_positions`` (one per user slot)."""
        if len(host_positions) != self.max_batch:
            raise ValueError(f"{len(host_positions)} positions for max_batch {self.max_batch}")
        pos, pool_ids, ring_slots = decode_position_tensors(self.device, host_positions, self.cfg.index_kpool)
        return DecodePositions(pos, pool_ids, ring_slots, list(host_positions))

    def update_decode_positions(self, dp: DecodePositions, host_positions: list[int]):
        """Refresh the persistent position tensors in place (trace-replay safe)."""
        if len(host_positions) != self.max_batch:
            raise ValueError(f"{len(host_positions)} positions for max_batch {self.max_batch}")
        P = self.cfg.index_kpool
        for t, vals in (
            (dp.positions, host_positions),
            (dp.pool_ids, [p // P for p in host_positions]),
            (dp.ring_slots, [p % P for p in host_positions]),
        ):
            ttnn.copy_host_to_device_tensor(
                ttnn.from_torch(torch.tensor(vals, dtype=torch.int32), dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT),
                t,
            )
        dp.host = list(host_positions)

    # ---------------------------------------------------------------- forwards
    def prefill_forward(
        self, streams: ttnn.Tensor, *, user: int = 0, seq_len: int | None = None, start: int = 0
    ) -> ttnn.Tensor:
        """One chunk of one user's prompt: row i of ``streams`` is the token at absolute position
        ``start + i``. ``start == 0`` starts a new prompt; a later chunk must start where the previous
        one ended (``next_start[user]``), on a 32-row boundary, and end within ``max_seq_len`` (see
        the module docstring for the full chunked-prefill contract)."""
        T = streams.shape[-2]
        seq_len = T if seq_len is None else seq_len
        if T % TILE or seq_len > T or seq_len <= 0:
            raise ValueError(f"T_pad={T} must be a multiple of {TILE} and 0 < seq_len={seq_len} <= T_pad")
        if user >= self.max_batch:
            raise ValueError(f"user {user} >= max_batch {self.max_batch}")
        if start < 0 or start % TILE:
            raise ValueError(f"chunk start {start} must be a non-negative multiple of {TILE}")
        if start + seq_len > self.max_seq_len:
            raise ValueError(f"start {start} + seq_len {seq_len} > max_seq_len {self.max_seq_len}")
        if start and start != self.next_start[user]:
            raise ValueError(
                f"user {user}: chunk start {start} is not contiguous with the previous chunk "
                f"(expected {self.next_start[user]}; a chunk with a non-{TILE}-aligned seq_len must be the last)"
            )
        state, temporaries = self._state_for_prefill(user, start)
        out, new_state = self.layer.prefill_forward(streams, state, user=user, seq_len=seq_len, start=start)
        for t in temporaries:
            ttnn.deallocate(t)
        if self.layer.is_kda:
            self._write_user_kda_state(user, new_state.kda)
        self.next_start[user] = start + seq_len
        return out

    def decode_forward(self, streams: ttnn.Tensor, *, positions: DecodePositions) -> ttnn.Tensor:
        """One token per user slot. DSA layers pick the dense (traceable) or sparse path from the
        host positions carried by ``positions``."""
        B = streams.shape[-2]
        if any(p >= self.max_seq_len for p in positions.host):
            raise ValueError(f"positions {positions.host} exceed max_seq_len {self.max_seq_len}")
        out, new_state = self.layer.decode_forward(streams, self._state_for_decode(B), positions=positions)
        if self.layer.is_kda:
            assert new_state.kda is self.kda_batched, "KDA decode must update the persistent state in place"
        self.next_start = [p + 1 for p in positions.host]  # a prefill chunk may continue after decode
        return out

    # ------------------------------------------------------------------- trace
    def capture_decode_trace(
        self, streams: ttnn.Tensor, positions: DecodePositions, *, cq_id: int = 0, allow_sparse_range: bool = False
    ):
        """Warm up (compile) one decode step on the persistent inputs, then capture the next one.
        Returns ``(trace_id, output tensor, warm-up output)``. Replay with ``replay_decode``; the
        state tensors are updated in place by the captured graph. The warm-up and the capture both
        consume the token at ``positions`` (the state advances twice), so capture on a throw-away
        token or re-prefill afterwards. Refused for a DSA layer whose ``max_seq_len`` reaches the
        sparse threshold unless ``allow_sparse_range`` (replays past position 2047 would run the dense
        attention, which is not HF semantics there)."""
        if (
            not self.layer.is_kda
            and self.max_seq_len > self.layer.attn.dense_exact_max_pos + 1
            and not allow_sparse_range
        ):
            raise ValueError(
                f"traced decode is HF-exact only through position {self.layer.attn.dense_exact_max_pos}; this decoder "
                f"allows max_seq_len={self.max_seq_len}. Use eager decode_forward beyond, or pass allow_sparse_range=True"
            )
        warm = self.decode_forward(streams, positions=positions)
        ttnn.synchronize_device(self.device)
        trace_id = ttnn.begin_trace_capture(self.device, cq_id=cq_id)
        out = self.decode_forward(streams, positions=positions)
        ttnn.end_trace_capture(self.device, trace_id, cq_id=cq_id)
        return trace_id, out, warm

    def replay_decode(self, trace_id, *, cq_id: int = 0, blocking: bool = True):
        ttnn.execute_trace(self.device, trace_id, cq_id=cq_id, blocking=blocking)
