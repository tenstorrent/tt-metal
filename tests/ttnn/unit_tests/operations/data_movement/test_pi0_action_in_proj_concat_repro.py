# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Pi0 EmitPy first ``ttnn_concat_0`` repro (tt-xla #4633).

- **Bias graph** (``pi0_bracket_action_in_proj_bias_codegen``): rank-3 + rank-2
  on dim=1 — ``ttnn.concat`` throws; test **fails** until concat/lowering is fixed.
- **No-bias graph** (``pi0_bracket_no_bias_codegen``): two rank-2 ``(50, 1024)``
  on dim=1 — same op, **passes**; contrasts with biased lowering shape bug.

https://github.com/tenstorrent/tt-xla/issues/4633
"""

import torch

import ttnn

_DRAM = ttnn.MemoryConfig(
    ttnn.TensorMemoryLayout.INTERLEAVED,
    ttnn.BufferType.DRAM,
    None,
)


def test_pi0_emitpy_bias_ttnn_concat0_single_op_matches_graph(device):
    """Mimics biased graph ``ttnn_concat_0``; fails until rank-mismatch concat is fixed."""
    torch.manual_seed(0)
    lhs = ttnn.from_torch(
        torch.randn(1, 50, 1024, dtype=torch.float32),
        layout=ttnn.TILE_LAYOUT,
        device=device,
        dtype=ttnn.float32,
    )
    rhs = ttnn.from_torch(
        torch.randn(50, 1024, dtype=torch.float32),
        layout=ttnn.TILE_LAYOUT,
        device=device,
        dtype=ttnn.float32,
    )
    ttnn.concat([lhs, rhs], 1, memory_config=_DRAM)


def test_pi0_emitpy_no_bias_ttnn_concat0_matching_ranks_passes(device):
    """Mimics no-bias graph ``ttnn_concat_0``: two ``(50, 1024)``, dim=1 (passing sanity)."""
    torch.manual_seed(1)
    a_t = torch.randn(50, 1024, dtype=torch.float32)
    b_t = torch.randn(50, 1024, dtype=torch.float32)
    golden = torch.cat([a_t, b_t], dim=1)

    a = ttnn.from_torch(
        a_t,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        dtype=ttnn.float32,
    )
    b = ttnn.from_torch(
        b_t,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        dtype=ttnn.float32,
    )
    out = ttnn.concat([a, b], 1, memory_config=_DRAM)
    out_t = ttnn.to_torch(out)
    assert tuple(out_t.shape) == (50, 2048)
    torch.testing.assert_close(out_t, golden, rtol=1.3e-6, atol=1e-5)
