# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""TTNN port of the Oobleck Snake1d activation.

Reference (diffusers / torch_ref):

    alpha = exp(alpha_log)
    beta  = exp(beta_log)
    y = x + 1/(beta + 1e-9) * sin(alpha * x)**2

The torch parameters live in shape ``[1, C, 1]`` (Conv1d-friendly). For TTNN
we keep tensors in ``[B, T, C]`` layout, so we reshape both parameters to
``[1, 1, C]`` once at init and let TTNN broadcast.
"""

from __future__ import annotations

import numpy as np

from .._ttnn import get_ttnn
from ..math_perf_env import (
    ace_step_vae_activation_compute_dtype,
    ace_step_vae_activation_memory_config,
    ace_step_vae_activation_storage_dtype,
    ace_step_vae_eltwise_kwargs,
    ace_step_vae_ensure_interleaved,
    ace_step_vae_typecast_kwargs,
)


def _require_ttnn():
    ttnn = get_ttnn()
    if ttnn is None:
        raise RuntimeError("ttnn is required for ace_step_v1_5.ttnn_impl.vae")
    return ttnn


def _snake_param_to_btc(arr) -> np.ndarray:
    """Reshape a Snake1d parameter from ``[1, C, 1]`` (or ``[C]``) to ``[1, 1, 1, C]``."""
    a = np.asarray(arr).astype(np.float32)
    if a.ndim == 3 and a.shape[0] == 1 and a.shape[-1] == 1:
        a = a.reshape(1, 1, 1, -1)
    elif a.ndim == 1:
        a = a.reshape(1, 1, 1, -1)
    else:
        a = a.reshape(1, 1, 1, -1)
    return np.ascontiguousarray(a)


class TtSnake1d:
    """Channel-wise Snake activation operating on ``[B, T, C]`` row-major tensors."""

    def __init__(
        self,
        *,
        alpha_host,
        beta_host,
        device,
        dtype=None,
        memory_config=None,
        output_memory_config=None,
    ) -> None:
        ttnn = _require_ttnn()
        self.ttnn = ttnn
        self.device = device
        self._storage_dtype = ace_step_vae_activation_storage_dtype(ttnn)
        self._compute_dtype = ace_step_vae_activation_compute_dtype(ttnn)
        if dtype is not None:
            self._storage_dtype = dtype
            self._compute_dtype = dtype
        self.dtype = self._storage_dtype

        # L1 eltwise chain: all intermediate tensors (ax, sin, square, term, y4) live in L1
        # rather than DRAM, eliminating ~5 DRAM round-trips per snake call.
        # By default the result is staged back to DRAM before returning so callers that
        # immediately invoke a k>7 conv1d (static CB region ~180 KiB on Blackhole) are safe.
        # Pass output_memory_config=L1_MEMORY_CONFIG when the caller's next op is k=1 conv
        # or another L1-safe op — this eliminates the snake DRAM write and the downstream
        # conv _maybe_l1 copy in one shot.
        l1_mc = ace_step_vae_activation_memory_config(ttnn)
        dram_mc = getattr(ttnn, "DRAM_MEMORY_CONFIG", None)
        self._l1_mc = l1_mc
        self._dram_mc = dram_mc
        # output_memory_config=None → DRAM (safe default); L1_MEMORY_CONFIG when CB-safe.
        self._output_mc = output_memory_config if output_memory_config is not None else dram_mc
        # Params are tiny (≤256 B each per snake instance); keeping them in L1 makes
        # both multiply operands fully L1-local (no DRAM broadcast read for alpha/beta).
        self.memory_config = memory_config if memory_config is not None else l1_mc
        self._out_kw = ace_step_vae_eltwise_kwargs(ttnn, l1_mc=l1_mc)
        self._typecast_kw = ace_step_vae_typecast_kwargs(ttnn, l1_mc=l1_mc)

        # BF16/BFP8 compute in TILE L1 (see ``ace_step_vae_activation_compute_dtype``).
        self.compute_dtype = self._compute_dtype

        alpha = _snake_param_to_btc(alpha_host)
        beta = _snake_param_to_btc(beta_host)
        self.hidden_dim = int(alpha.shape[-1])

        alpha_exp = np.exp(alpha).astype(np.float32)
        beta_recip = (1.0 / (np.exp(beta).astype(np.float32) + 1e-9)).astype(np.float32)

        self._alpha_exp = ttnn.as_tensor(
            alpha_exp,
            device=device,
            dtype=self.compute_dtype,
            layout=ttnn.TILE_LAYOUT,
            memory_config=self.memory_config,  # L1
        )
        self._beta_recip = ttnn.as_tensor(
            beta_recip,
            device=device,
            dtype=self.compute_dtype,
            layout=ttnn.TILE_LAYOUT,
            memory_config=self.memory_config,  # L1
        )

    def _is_l1_tile(self, x) -> bool:
        """Return ``True`` if *x* is already ``TILE`` layout in L1 (interleaved or HEIGHT_SHARDED).

        Used to detect tensors returned by ``TtConv1d`` with ``return_sharded=True`` so
        snake can skip the expensive ``to_layout`` Tilize step.
        """
        try:
            if x.layout != self.ttnn.TILE_LAYOUT:
                return False
            mc = x.memory_config()
            dram = getattr(self.ttnn, "DRAM_MEMORY_CONFIG", None)
            return mc != dram
        except Exception:
            return False

    def __call__(self, x, *, return_tile: bool = False):
        """Apply Snake activation in-place-ish.

        Args:
            x: TTNN tensor of shape ``[B, T, C]`` (row-major, DRAM) **or** TILE layout
               (DRAM or L1) from ``TtConv1d`` with ``return_tile=True`` /
               ``return_sharded=True``, **or** ``[B, 1, T, C]`` rank-4 tile.
            return_tile: When ``True``, return **TILE** layout in ``output_memory_config``
                (typically L1 for residual ``snake2``) instead of untilizing to ROW_MAJOR.
                Used at the snake2→conv2(k=1) boundary so conv2 can consume TILE L1 in0
                without a Tilize on ROW_MAJOR activations.

        Returns:
            ``[B, T, C]`` row-major tensor by default — CB-safe for k>7 conv1d callers.
            With ``return_tile=True``: ``[B, T, C]`` **TILE** in ``output_memory_config``.
            All intermediate tensors (ax, sin, s2, term, y4) live in L1 and are
            deallocated before returning, so the net L1 footprint on exit is zero.
        """
        ttnn = self.ttnn
        out_kw = self._out_kw
        l1_mc = self._l1_mc
        dram_mc = self._dram_mc

        # Wide conv outputs can be HEIGHT_SHARDED ROW_MAJOR; unsqueeze/tilize require interleaved.
        x = ace_step_vae_ensure_interleaved(ttnn, x, memory_config=dram_mc)

        orig_shape = tuple(x.shape)

        # Fast path: input is already TILE in L1 (HEIGHT_SHARDED or interleaved) from
        # conv1(return_sharded=True).  Convert to L1 INTERLEAVED before unsqueeze so the
        # rank-4 param broadcast and downstream to_layout check work unchanged.
        # This S2I (or no-op) stays entirely on-chip — no DRAM traffic.
        if len(orig_shape) == 3 and self._is_l1_tile(x):
            if l1_mc is not None:
                x = ttnn.to_memory_config(x, l1_mc)  # HEIGHT_SHARDED → L1 INTERLEAVED (NoC only)

        if len(orig_shape) == 3:
            x4 = ttnn.unsqueeze(x, 1)
            squeeze_back = True
        elif len(orig_shape) == 4:
            x4 = x
            squeeze_back = False
        else:
            raise ValueError(f"TtSnake1d expects rank-3 [B,T,C] or rank-4 [B,1,T,C]; got {orig_shape}")

        # Tilize to L1, or (if already TILE) just ensure we are in L1.
        # After the HEIGHT_SHARDED → L1 INTERLEAVED conversion above, x4 is L1 TILE
        # and both branches below are no-ops, saving the expensive Tilize kernel.
        _tilize_kw = {"memory_config": l1_mc} if l1_mc is not None else {}
        if x4.layout != ttnn.TILE_LAYOUT:
            # ROW_MAJOR → Tilize to L1 (normal path or DRAM ROW_MAJOR after return_tile).
            x4 = ttnn.to_layout(x4, ttnn.TILE_LAYOUT, **_tilize_kw)
        elif l1_mc is not None and x4.memory_config() != l1_mc:
            # Already TILE but in DRAM (e.g. from return_tile without sharding):
            # DMA copy DRAM→L1 (replaces the 4568 µs Tilize with a ~150 µs DMA copy).
            x4 = ttnn.to_memory_config(x4, l1_mc)

        x4_compute = x4
        if self.compute_dtype is not None and x4.dtype != self.compute_dtype:
            x4_compute = ttnn.typecast(x4, self.compute_dtype, **self._typecast_kw)

        # Eltwise chain — all intermediate tensors in L1.
        # alpha_exp / beta_recip are also L1, so no DRAM broadcast reads.
        ax = ttnn.multiply(x4_compute, self._alpha_exp, **out_kw)  # L1
        s = ttnn.sin(ax)  # L1 (inherits from ax)
        ttnn.deallocate(ax)
        s2 = ttnn.square(s)  # L1 (inherits from s)
        ttnn.deallocate(s)
        term = ttnn.multiply(s2, self._beta_recip, **out_kw)  # L1
        ttnn.deallocate(s2)
        y4 = ttnn.add(x4_compute, term, **out_kw)  # L1
        ttnn.deallocate(term)
        if x4_compute is not x4:
            ttnn.deallocate(x4_compute)

        if y4.dtype != self._storage_dtype:
            y4 = ttnn.typecast(y4, self._storage_dtype, **self._typecast_kw)

        y = ttnn.squeeze(y4, 1) if squeeze_back else y4

        if return_tile:
            # TILE passthrough (residual snake2 → conv2 k=1): skip Untilize so conv2 linear /
            # conv1d paths avoid a second Tilize on ROW_MAJOR L1 activations.
            if self._output_mc is not None and y.memory_config() != self._output_mc:
                y = ttnn.to_memory_config(y, self._output_mc)
            return y

        # Untilize into the requested output memory (DRAM by default).
        # DRAM is required when the caller's next op is k>7 conv1d: program compilation
        # raises TT_FATAL if any L1 buffer is alive in the static CB region (~0–180 KiB).
        # When output_memory_config=L1_MEMORY_CONFIG the result stays in L1, saving the
        # DRAM write here and the downstream _maybe_l1 copy in k=1 conv.
        _rm_out_kw = {"memory_config": self._output_mc} if self._output_mc is not None else {}
        out = ttnn.to_layout(y, ttnn.ROW_MAJOR_LAYOUT, **_rm_out_kw)
        # Free L1 TILE staging before k>7 conv1d compile (untilize output is DRAM).
        try:
            ttnn.deallocate(y4)
        except Exception:
            pass
        if squeeze_back:
            try:
                ttnn.deallocate(x4)
            except Exception:
                pass
        return out
