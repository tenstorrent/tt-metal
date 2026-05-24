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
    ace_step_vae_activation_memory_config,
    ace_step_vae_eltwise_kwargs,
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
    ) -> None:
        ttnn = _require_ttnn()
        self.ttnn = ttnn
        self.device = device
        self.dtype = dtype or getattr(ttnn, "bfloat16", None) or getattr(ttnn, "float16", None)
        if self.dtype is None:
            raise RuntimeError("TTNN build missing a usable dtype (bfloat16/float16)")

        # L1 eltwise chain: all intermediate tensors (ax, sin, square, term, y4) live in L1
        # rather than DRAM, eliminating ~5 DRAM round-trips per snake call.
        # The final to_layout (Untilize) stages the result back to DRAM *inside* this class
        # before returning, so callers always receive a DRAM ROW_MAJOR tensor.
        # This preserves CB safety: k>7 conv1d static CB region extends to ~180 KiB on
        # Blackhole; any live L1 activation in that band fails program validation at compile
        # time.  Since snake returns DRAM, no L1 tensor is alive when conv1d is called.
        l1_mc = ace_step_vae_activation_memory_config(ttnn)
        dram_mc = getattr(ttnn, "DRAM_MEMORY_CONFIG", None)
        self._l1_mc = l1_mc
        self._dram_mc = dram_mc
        # Params are tiny (≤256 B each per snake instance); keeping them in L1 makes
        # both multiply operands fully L1-local (no DRAM broadcast read for alpha/beta).
        self.memory_config = memory_config if memory_config is not None else l1_mc
        self._out_kw = ace_step_vae_eltwise_kwargs(ttnn, l1_mc=l1_mc)
        self._typecast_kw = ace_step_vae_typecast_kwargs(ttnn, l1_mc=l1_mc)

        # BF16 compute (not FP32): avoids ~784 μs DRAM FP32 BinaryNg per call; see perf19.
        self.compute_dtype = self.dtype

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

    def __call__(self, x):
        """Apply Snake activation in-place-ish.

        Args:
            x: TTNN tensor of shape ``[B, T, C]`` (row-major, DRAM) or ``[B, 1, T, C]`` (tile).

        Returns:
            ``[B, T, C]`` row-major **DRAM** tensor — CB-safe for k>7 conv1d callers.
            All intermediate tensors (ax, sin, s2, term, y4) live in L1 and are
            deallocated before returning, so the net L1 footprint on exit is zero.
        """
        ttnn = self.ttnn
        out_kw = self._out_kw
        l1_mc = self._l1_mc
        dram_mc = self._dram_mc

        orig_shape = tuple(x.shape)
        if len(orig_shape) == 3:
            x4 = ttnn.unsqueeze(x, 1)
            squeeze_back = True
        elif len(orig_shape) == 4:
            x4 = x
            squeeze_back = False
        else:
            raise ValueError(f"TtSnake1d expects rank-3 [B,T,C] or rank-4 [B,1,T,C]; got {orig_shape}")

        # Tilize to L1: DRAM read once here; all subsequent ops read/write L1 only.
        _tilize_kw = {"memory_config": l1_mc} if l1_mc is not None else {}
        x4 = ttnn.to_layout(x4, ttnn.TILE_LAYOUT, **_tilize_kw)

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

        if y4.dtype != self.dtype:
            y4 = ttnn.typecast(y4, self.dtype, **self._typecast_kw)

        # Untilize + stage back to DRAM in one fused to_layout call.
        # DRAM output is required: k>7 conv1d program compilation (warmup) raises
        # TT_FATAL if any L1 buffer is alive in the static CB region (~0–180 KiB).
        # After this call y4's L1 buffer is released (Python refcount → 0 on rebind).
        _rm_dram_kw = {"memory_config": dram_mc} if dram_mc is not None else {}
        if squeeze_back:
            y = ttnn.squeeze(y4, 1)
            return ttnn.to_layout(y, ttnn.ROW_MAJOR_LAYOUT, **_rm_dram_kw)
        return ttnn.to_layout(y4, ttnn.ROW_MAJOR_LAYOUT, **_rm_dram_kw)
