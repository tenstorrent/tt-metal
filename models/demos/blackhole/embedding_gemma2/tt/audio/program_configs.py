# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Program configs for the audio tower's batched block matmuls (E5c).

The blocked band attention and the lconv causal shift are batched matmuls whose both operands
carry the batch (one 32-row block per batch entry): QK^T [B, H*nb, 32, 128] x [B, H*nb, 64, 128]^T,
PV [B, H*nb, 32, 64] x [B, H*nb, 64, 128], Sel [B, nb, 128, 64] x [B, nb, 64, 1024]. With no
program config ttnn runs them on 2-4 cores (E3 report: 356 / 229 / 273 us per layer at 1 x 768).
`MatmulMultiCoreReuseProgramConfig` spreads the batch x M-block work over the whole grid; the
config needs per_core_N == N, so one work unit is one 32-row block (or `per_core_M` of them) times
the full output width. Same math, same fidelity: outputs are bit-identical to the default config
(E5c micro-bench, `e5c/scripts/mm_micro.py`).
"""

from __future__ import annotations

import ttnn


def _largest_divisor(n: int, limit: int) -> int:
    return max(d for d in range(1, limit + 1) if n % d == 0)


def block_bmm_config(device, M: int, K: int, N: int, per_core_M: int = 1) -> ttnn.MatmulMultiCoreReuseProgramConfig:
    """Config for a batched [.., M, K] x [.., K, N] matmul with batched operands (M, K, N in elements)."""
    Mt, Kt, Nt = M // ttnn.TILE_SIZE, K // ttnn.TILE_SIZE, N // ttnn.TILE_SIZE
    if Mt % per_core_M != 0:
        raise ValueError(f"per_core_M {per_core_M} must divide {Mt} M tiles")
    # fp32 destination: at most 4 tiles per output subblock
    sbw = _largest_divisor(Nt, 4)
    sbh = max(d for d in range(1, per_core_M + 1) if per_core_M % d == 0 and d * sbw <= 4)
    return ttnn.MatmulMultiCoreReuseProgramConfig(
        compute_with_storage_grid_size=device.compute_with_storage_grid_size(),
        in0_block_w=_largest_divisor(Kt, 4),
        out_subblock_h=sbh,
        out_subblock_w=sbw,
        per_core_M=per_core_M,
        per_core_N=Nt,
    )
