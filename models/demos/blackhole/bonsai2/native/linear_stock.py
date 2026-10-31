# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""A1-stock: exact ternary matmul from stock ops only, no new kernels.

The problem: ``y = x @ (T * s).T`` where ``T`` is ternary and ``s`` is one fp16
scale per 128-weight group. A stock ``ttnn.linear(x, W)`` can only apply one
scale to the whole K sum, and there are ``G = K/128`` different scales.

The trick is to split the K sum along groups and let the matmul do each group's
partial sum in one shot::

    X_bd[t, g, :] = x[t, :] * mask[g, :]      # row g keeps only group g's K columns
    P[t, g, :]    = X_bd[t, g, :] @ T.T       # [M, G, N], each group's partial
    y[t, :]       = sum_g P[t, g, :] * s[g, :]

``X_bd`` is ``x`` with one extra G axis and a constant 0/1 mask applied, so
building it is one broadcast multiply by a mask the caller owns. ``P`` is
``[M, G, N]`` and the final combine is a broadcast multiply by the scale matrix
``S`` (also ``[G, N]``) followed by a reduction over the G axis.

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

Two layout notes that matter and cost time to find:

  * Both the mask and the scale are multiplied in with a *broadcast over the
    leading (token) axis*, against an operand of shape ``[M, 1, G, :]``. ttnn's
    eltwise broadcast accepts exactly one axis of unit extent -- an earlier
    version tiled both operands to the full ``[M*G, :]`` shape on the host,
    which is correct but materializes ``M*G*K`` floats of numpy per call (1.4 GB
    at prefill M=128, K=17408). Broadcasting the token axis instead is both
    simpler and cheap.
  * The group reduction needs the token axis to be a *separate* axis. Summing
    ``dim=2`` of a flat ``[1, 1, M*G, N]`` sums every token together; the
    tensor has to be ``[M, 1, G, N]`` first.
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
    ``P``'s group axis, and padded with zeros so the padded groups contribute
    nothing to the reduction.
    """
    n, ngroups = scales.shape
    rows = padded_rows(ngroups)
    out = np.zeros((rows, n), dtype=np.float32)
    out[:ngroups] = scales.astype(np.float32).T
    return out


def _to_dev(device, a: np.ndarray, dtype, mem):
    """Host float array -> ttnn tensor, going through float32 so bf16 is exact."""
    t = torch.from_numpy(np.ascontiguousarray(a))
    if dtype == ttnn.bfloat16:
        t = t.to(torch.bfloat16)
    return ttnn.from_torch(t, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device, memory_config=mem)


class TernaryLinearStock:
    """One ternary projection built from ``ttnn.linear`` plus a masked scale fold.

    The weight is stored as ``bfloat4_b`` codes (exact for -1/0/+1) and the scales
    as a ``[G_padded, N]`` fp32 matrix. A call is: mask the activation into G
    rows, one linear, one broadcast multiply, one reduction.
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

        # bfloat4_b holds these three values exactly, which is what makes the
        # whole scheme lossless on the code side.
        self.w = _to_dev(
            device, codes.astype(np.float32).T.copy(), ttnn.bfloat4_b, ttnn.DRAM_MEMORY_CONFIG
        )
        # Mask and scale are [1, 1, rows, :] so the multiply broadcasts over the
        # token axis, which is the leading one on the [m, 1, rows, :] operands.
        self.mask = ttnn.reshape(
            _to_dev(device, block_mask(k, group), ttnn.bfloat16, ttnn.DRAM_MEMORY_CONFIG),
            (1, 1, self.rows, k),
        )
        self.scale = ttnn.reshape(
            _to_dev(device, scale_matrix(scales, group), ttnn.float32, ttnn.DRAM_MEMORY_CONFIG),
            (1, 1, self.rows, n),
        )

    @property
    def weight_bytes(self) -> int:
        """BFP4 codes plus the fp32 scale table, the real streaming cost."""
        return self.n * self.k // 2 + (self.n * self.k // 16) + self.rows * self.n * 4

    def __call__(self, x: torch.Tensor, mem=None):
        """``x`` is ``[M, K]`` (M is the token count); returns a device tensor ``[M, 1, 1, N]``.

        Returns the ttnn tensor rather than a torch one on purpose: every call site
        inside the model feeds the next op on device, and a ``to_torch`` here would
        charge every projection a 200 us host readback it does not need. Read the
        result back explicitly with ``ttnn.to_torch``.
        """
        m = x.shape[0]
        if x.shape[-1] != self.k:
            raise ValueError(f"x has K={x.shape[-1]}, weight has K={self.k}")
        mem = mem or ttnn.DRAM_MEMORY_CONFIG

        t_x = ttnn.reshape(
            _to_dev(self.device, x.float().numpy(), ttnn.bfloat16, mem), (m, 1, 1, self.k)
        )
        # [m, K] -> [m, G, K], row g of token t keeps only group g's columns.
        x_bd = ttnn.multiply(t_x, self.mask, memory_config=mem)
        ttnn.deallocate(t_x)

        p = ttnn.linear(
            ttnn.reshape(x_bd, (1, 1, m * self.rows, self.k)), self.w, memory_config=mem
        )
        ttnn.deallocate(x_bd)
        # [m*G, N] -> [m, 1, G, N] so the reduction is per token, not over all
        # m*G rows at once (which would sum every token together).
        p = ttnn.reshape(p, (m, 1, self.rows, self.n))
        scaled = ttnn.multiply(p, self.scale, memory_config=mem)
        ttnn.deallocate(p)
        y = ttnn.sum(scaled, dim=2, keepdim=True)
        ttnn.deallocate(scaled)
        return y
