# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Refinement 6d ablation — allgather (Pattern B) vs the current Pattern A on the
1-D master-bottleneck topology (NOT the golden suite; a committed characterization).

R6d's named lever is the **allgather (Pattern B)**: instead of gather -> single-master
fold -> broadcast (Pattern A, what the rms_norm xcore transport ships today via
`reduce_root_mcast`-style all-unicast/segmented-mcast), have every core receive all K
partials and fold them redundantly, "removing the master bottleneck".

The rms_norm cross-core stat exchange for the dominant BLOCK 8x8 perf shape
(`(1,1,8192,1024)`, BLOCK_SHARDED (1024,128) on an 8x8 grid) is exactly one
**1-D 8-core line per grid row** (K=8), C=8 tile-rows batched per round, 8 groups
packed across the grid (grid-filling). This ablation measures the three cross-core
collective topologies at that EXACT topology on device via the in-tree
`tensix_all_reduce` example (the collective in isolation, no rms compute), which is the
authoritative "which topology wins the 1-D line" evidence:

  MEASURED (blackhole_p150b `bh-50-...-49684`, median of 5 trials x 10 in-kernel
  collectives, num_tiles=8 ~ C=8 stat tiles/round; `AR_*` env + the example perf test):

    Placement 1x8, 8 groups (grid-filling, 64 cores) — matches BLOCK 8x8:
      reduce_root_mcast   (Pattern A, current) : 3189.6 ns   1.00x
      mcast_all_gather    (Pattern B, R6d lever): 6134.5 ns   0.52x   (~2x SLOWER)
      two_phase_reduce_mcast                    : 2291.3 ns   1.39x   (WINNER)
      unicast_all_gather  (Pattern B all-to-all): 11140.3 ns  0.29x

    Placement 1x8, 1 group (isolated, no contention) — same verdict:
      reduce_root_mcast    3182.2 ns 1.00x | mcast_all_gather 6135.5 ns 0.52x |
      two_phase_reduce_mcast 2272.7 ns 1.40x

  FINDING: on a 1-D line the allgather is decisively inferior to the current Pattern A,
  isolated OR grid-filling. The penalty is intrinsic: the rotating mcast allgather is
  K serial mcast sub-rounds, and the all-to-all form multiplies the per-core receive
  traffic by K (every core receives all K partials instead of only the master). No
  complementary step fixes that on a 1-D group — the traffic multiplication IS the
  mechanism. Eliminating the broadcast leg (~14% of a round) does not pay for doubling
  the gather. => R6d's named lever cannot close the BLOCK residual; parked (not shipped).

  The measured winner is `two_phase_reduce_mcast` (tile-index workers gather+fold
  disjoint tile-rows -> root assembles + mcasts): reduced communication volume WITHOUT
  the allgather's traffic multiplication. It only engages when C>1 (BLOCK's batched
  rows); WIDTH n x 1 groups are single-round (C=1) so it degenerates to root. Filed as
  Refinement 6e.

Reproduce the ns table directly:
    AR_GROUP_SHAPE=1,8 AR_NUM_GROUPS=8 AR_NUM_TILES=8 AR_KERNEL_ITERS=10 AR_TRIALS=5 \
    AR_VARIANTS=reduce_root_mcast,mcast_all_gather,two_phase_reduce_mcast,unicast_all_gather \
    AR_REPORT=/tmp/ar.md scripts/run_safe_pytest.sh \
    tests/ttnn/unit_tests/operations/examples/test_tensix_all_reduce.py::test_tensix_all_reduce_device_perf

This file's runnable assertions confirm all three collective topologies are NUMERICALLY
correct at the BLOCK 8x8 per-group topology (so the R6d finding is purely a perf verdict,
not a correctness gap), and that the shipped rms_norm op (Pattern A) is still correct on
the BLOCK 8x8 perf shape (a regression guard for the target shape).
"""

from __future__ import annotations

import pytest
import torch

import ttnn

from eval.sharding import shard_config
from ttnn.operations.rms_norm import rms_norm

# The three cross-core collective topologies at the center of R6/R6a/R6b/R6c/R6d.
from tests.ttnn.unit_tests.operations.examples.test_tensix_all_reduce import _make_input, _run_checked

# 1-D line per group (BLOCK 8x8: one grid row of K=8 cores). num_tiles=8 ~ C=8 batched
# stat tiles/round. Grid-filling (8 groups) matches the real op; isolated (1 group)
# controls for NoC contention (the verdict is the same either way — see the docstring).
_GROUP_SHAPE = (1, 8)
_NUM_TILES = 8


def _pcc(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a.flatten().to(torch.float32)
    b = b.flatten().to(torch.float32)
    a = a - a.mean()
    b = b - b.mean()
    denom = (a.norm() * b.norm()).item()
    return 1.0 if denom == 0 else torch.dot(a, b).item() / denom


@pytest.mark.parametrize("num_groups", [1, 8], ids=["isolated", "grid_filling"])
@pytest.mark.parametrize(
    "variant",
    ["reduce_root_mcast", "mcast_all_gather", "two_phase_reduce_mcast"],
    ids=["patternA_root", "patternB_allgather", "two_phase"],
)
def test_collective_topologies_correct_on_1d_line(variant, num_groups, device):
    """All three cross-core topologies are correct on the BLOCK 8x8 1-D line.

    Confirms R6d is a PERF verdict (allgather is correct but ~2x slower per the
    docstring's on-device numbers), not a correctness gap. Perf ns are measured via the
    tensix_all_reduce example's device-perf harness (reproduce command in the docstring).
    """
    tt_input, expected = _make_input(device, _GROUP_SHAPE, num_groups, _NUM_TILES)
    _run_checked(tt_input, expected, variant, _GROUP_SHAPE, num_groups, _NUM_TILES, kernel_iters=1)


def test_block_8x8_pattern_a_regression_guard(device):
    """The shipped rms_norm op (Pattern A xcore transport) stays correct on BLOCK 8x8.

    R6d ships no kernel change (the named allgather lever is measured-inferior and parked),
    so this only guards that the target shape is unregressed at the exact perf config.
    """
    torch.manual_seed(0)
    shape = (1, 1, 8192, 1024)
    torch_input = torch.randn(shape, dtype=torch.bfloat16)
    torch_gamma = torch.randn(1024, dtype=torch.bfloat16)
    expected = torch_input.float() * torch.rsqrt(torch_input.float().pow(2).mean(-1, keepdim=True) + 1e-6)
    expected = expected * torch_gamma.float().reshape(-1)

    in_cfg = shard_config(
        [1024, 128],
        (8, 8),
        ttnn.TensorMemoryLayout.BLOCK_SHARDED,
        layout=ttnn.TILE_LAYOUT,
        dtype=ttnn.bfloat16,
        device=device,
    )
    ttnn_input = ttnn.from_torch(
        torch_input, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=in_cfg
    )
    ttnn_gamma = ttnn.from_torch(
        torch_gamma.reshape(1, 1, 1, 1024),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    cfg = ttnn.ComputeConfigDescriptor()
    cfg.math_fidelity = ttnn.MathFidelity.HiFi2
    cfg.fp32_dest_acc_en = False
    cfg.math_approx_mode = False
    out = rms_norm(ttnn_input, gamma=ttnn_gamma, epsilon=1e-6, compute_kernel_config=cfg, memory_config=in_cfg)

    pcc = _pcc(ttnn.to_torch(out), expected)
    print(f"\nR6D_GUARD BLOCK 8x8 PCC={pcc:.6f}")
    assert pcc >= 0.9995, f"soft PCC gate: {pcc} < 0.9995"
