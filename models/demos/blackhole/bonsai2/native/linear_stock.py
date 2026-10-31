# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""A1-stock: exact ternary matmul from stock ops only, no new kernels.

The problem: ``y = x @ (T * s).T`` where ``T`` is ternary and ``s`` is one fp16
scale per 128-weight group. A stock ``ttnn.linear(x, W)`` can only apply one
scale to the whole K sum, and there are ``G = K/128`` different scales.

The trick is to split the K sum along groups and let the matmul do each group's
partial sum in one shot::

    X_bd[g, :] = x * mask[g, :]        # row g keeps only group g's K columns
    P          = X_bd @ T.T            # [G, N], each row the group's partial
    y          = sum_g P[g, :] * s[g, :]   # one multiply, one reduction

``X_bd`` is ``x`` replicated to G rows with a constant 0/1 mask applied, so
building it is one broadcast multiply by a mask the caller owns. ``P`` is
``[G, N]`` and the final combine is a broadcast multiply by the fp16 scale
matrix ``S`` (also ``[G, N]``) followed by a reduction over the G axis.

Why this is exact where the bfp4 storage is: ``bfloat4_b`` holds -1, 0 and +1
exactly (magnitude 1 has a zero mantissa), and the codes are only ever those
three values, so ``W_codes`` round-trips bit-exactly. What bfp4 does *not* do is
carry the scale, which is why the scale has to be applied outside the matmul --
that is the whole point of the block mask.

Cost: the matmul runs over G rows instead of 1 (or over one chunk of G for
prefill), so MVMUL work is G times higher but DRAM traffic is unchanged -- the
weight is read once per pass regardless. It stays bandwidth-bound as long as
G x 32 rows of compute is cheaper than one weight read, which is why the plan
asks for the measurement rather than assuming it.
"""

from __future__ import annotations

import numpy as np
import torch

import ttnn

TILE = 32
#: Group size the GGUF uses for PQ2_0 / PTQ1_0.
GROUP = 128
#: Rows are padded to a multiple of this so G rows are tile-aligned.
ROW_ALIGN = 32


def padded_rows(num_groups: int) -> int:
    """Smallest tile-aligned row count >= ``num_groups``."""
    return ((num_groups + ROW_ALIGN - 1) // ROW_ALIGN) * ROW_ALIGN


def block_mask(k: int, group: int = GROUP) -> np.ndarray:
    """``[G_padded, K]`` 0/1 mask where row g covers group g's K columns."""
    if k % group:
        raise ValueError(f"K={k} is not a multiple of the group {group}")
    num_groups = k // group
    rows = padded_rows(num_groups)
    m = np.zeros((rows, k), dtype=np.float32)
    for g in range(num_groups):
        m[g, g * group : (g + 1) * group] = 1.0
    return m


def scale_matrix(scales: np.ndarray, group: int = GROUP) -> np.ndarray:
    """``[N, K/G]`` fp16 scales -> ``[G_padded, N]`` float32, zero rows padded.

    The layout the combine step needs: transposed so a row per group lines up with
    ``P``'s rows, and padded with zeros so the padded groups contribute nothing
    to the reduction.
    """
    n, ngroups = scales.shape
    rows = padded_rows(ngroups)
    out = np.zeros((rows, n), dtype=np.float32)
    out[:ngroups] = scales.astype(np.float32).T
    return out


class TernaryLinearStock:
    """One ternary projection built from ``ttnn.linear`` plus a masked scale fold.

    The weight is stored as ``bfloat4_b`` codes (exact for -1/0/+1) and the scales
    as a ``[G_padded, N]`` fp32 matrix. A call is: mask the activation into G rows,
    one linear, one broadcast multiply, one reduction.
    """

    def __init__(self, device, codes: np.ndarray, scales: np.ndarray, group: int = GROUP):
        codes = np.asarray(codes)
        n, k = codes.shape
        if scales.shape != (n, k // group):
            raise ValueError(f"scales {scales.shape} != {(n, k // group)}")
        if codes.min() < -1 or codes.max() > 1:
            raise ValueError("codes must be in {-1, 0, +1} for exact bfloat4_b storage")

        self.device = device
        self.n, self.k, self.group = n, k, group
        self.num_groups = k // group
        self.rows = padded_rows(self.num_groups)
        self._scales = np.asarray(scales)
        self._mask_cache: dict = {}

        # bfloat4_b holds these three values exactly, which is what makes the
        # whole scheme lossless on the code side.
        self.w = ttnn.from_torch(
            torch.from_numpy(codes.astype(np.float32).T.copy()),  # [in, out]
            dtype=ttnn.bfloat4_b,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        self.mask = ttnn.from_torch(
            torch.from_numpy(block_mask(k, group)),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        self.scale = ttnn.from_torch(
            torch.from_numpy(scale_matrix(scales, group)),
            dtype=ttnn.float32,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

    @property
    def weight_bytes(self) -> int:
        """BFP4 codes plus the fp32 scale table, the real streaming cost."""
        return self.n * self.k // 2 + (self.n * self.k // 16) + self.rows * self.n * 4

    def _tiled_scale(self, m: int):
        """``[1, 1, m * rows, N]`` scales, row ``t*rows + g`` = group g's scale for token t.

        Cached like the mask: it depends only on the token count.
        """
        key = ("s", m)
        if key in self._mask_cache:
            return self._mask_cache[key]
        sm = scale_matrix(self._scales, self.group)  # [rows, N]
        tiled = np.tile(sm, (m, 1))
        t = ttnn.from_torch(
            torch.from_numpy(tiled).reshape(1, 1, m * self.rows, self.n),
            dtype=ttnn.float32,
            layout=ttnn.TILE_LAYOUT,
            device=self.device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        self._mask_cache[key] = t
        return t

    def _tiled_mask(self, m: int):
        """``[m * rows, 1, 1, K]`` mask: row ``t*rows + g`` covers group g for token t.

        Built lazily and cached, because it depends only on the token count and is
        the one tensor the multiply cannot broadcast for itself.
        """
        if m in self._mask_cache:
            return self._mask_cache[m]
        base = block_mask(self.k, self.group)
        tiled = np.tile(base, (m, 1))
        t = ttnn.from_torch(
            torch.from_numpy(tiled).reshape(1, 1, m * self.rows, self.k),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=self.device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        self._mask_cache[m] = t
        return t


    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """``x`` is ``[M, K]`` (M is the token count); returns ``[M, N]``."""
        x = x.to(torch.bfloat16)
        m = x.shape[0]
        if x.shape[-1] != self.k:
            raise ValueError(f"x has K={x.shape[-1]}, weight has K={self.k}")

        t_x = ttnn.from_torch(
            x.reshape(1, 1, m, self.k), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
            device=self.device, memory_config=ttnn.L1_MEMORY_CONFIG,
        )
        # [M, K] -> [M*G, K] masked, token-major: row (t*G + g) is token t with
        # only group g's columns kept.
        #
        # Both operands are tiled to the full shape rather than broadcast.
        # ttnn's eltwise broadcast accepts exactly one axis of unit extent
        # (binary_ng_device_operation.cpp throws "Invalid subtile broadcast type"
        # when both H extents are non-unit and the broadcast is over the middle
        # axis), and this multiply needs the token axis expanded, which is the
        # middle axis here. Tiling costs one host-side numpy repeat of a [M, K]
        # array, which is negligible next to the weight read.
        x_tiled = np.repeat(x.numpy(), self.rows, axis=0)  # [m*rows, K]
        t_x = ttnn.from_torch(
            torch.from_numpy(x_tiled).reshape(1, 1, m * self.rows, self.k),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=self.device,
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )
        x_bd = ttnn.mul(t_x, self._tiled_mask(m), memory_config=ttnn.L1_MEMORY_CONFIG)
        ttnn.deallocate(t_x)

        p = ttnn.linear(x_bd, self.w, memory_config=ttnn.L1_MEMORY_CONFIG)
        ttnn.deallocate(x_bd)
        # [M*G, N] * [M*G, N] elementwise (scale tiled per token, same reason as the
        # mask), then sum the G rows of each token.
        scaled = ttnn.multiply(p, self._tiled_scale(m), memory_config=ttnn.L1_MEMORY_CONFIG)
        ttnn.deallocate(p)
        y = ttnn.sum(ttnn.reshape(scaled, (1, 1, m * self.rows, self.n)), dim=2)
        ttnn.deallocate(scaled)
        return ttnn.to_torch(y).reshape(m, self.n).float()
