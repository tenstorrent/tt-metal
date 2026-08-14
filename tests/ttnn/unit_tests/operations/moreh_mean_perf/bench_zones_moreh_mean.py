# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Zone-scoped bench for moreh_mean along H -- isolates the reduce region itself.

Whole-op DEVICE KERNEL DURATION cannot say whether the branch's win lands in the
reduce math, in the deleted mask phase, or in the reader. This bench pairs with
in-kernel DeviceZoneScopedN instrumentation (applied by patch to the kernel and to
reduce_helpers_compute.inl) to answer that.

MARKER BUDGET is the binding constraint and it fails SILENTLY:
  250 optional markers per RISC per dispatch; a zone costs 2 (start + end); the
  count is EXECUTIONS, not distinct names. The per-tile wait/math zone pair inside
  the reduce helper therefore burns 4 markers per input tile.

    tiles/core = (columns/core) * Ht = 7 * Ht
    markers    = 7*Ht*4 (helper) + 7*2 (per-column) ~= 28*Ht + 14

      Ht=1  ->  ~42 markers   ok
      Ht=4  ->  ~126 markers  ok
      Ht=16 ->  ~462 markers  TRUNCATES -- zones past the cap vanish with no warning
      Ht=32 ->  ~910 markers  TRUNCATES

So this file deliberately covers only Ht=1 and Ht=4. The Ht=16/32 shapes from
bench_moreh_mean_h.py must NOT be added here without hoisting the zones out of the
per-tile loop first.

    unset TT_METAL_DPRINT_CORES     # zones share SRAM with DPRINT/Watcher
    rm -rf ~/.cache/tt-metal-cache  # force JIT recompile so the patch takes effect
    scripts/run_safe_pytest.sh --profile --run-all \
        tests/ttnn/unit_tests/operations/moreh_mean_perf/bench_zones_moreh_mean.py
"""

import pytest
import torch

import ttnn

TILE = 32
NC = 7
W = 64 * TILE  # Wt=64 -> NC*Wt = 448 columns -> 7 columns/core on a 64-core grid

# (id, origin_H) -- kept to Ht in {1,4} by the marker budget above.
CASES = [
    ("zragged_ht1", 17),  # Ht=1: the only tile is the partial one
    ("zragged_ht4", 3 * TILE + 17),  # Ht=4, ragged tail -> mask phase on base
    ("zaligned_ht4", 4 * TILE),  # Ht=4, no mask phase on either commit
]


@pytest.mark.parametrize("case", CASES, ids=[c[0] for c in CASES])
def test_moreh_mean_zones(case, device):
    case_id, origin_H = case
    torch.manual_seed(2024)
    x = ttnn.from_torch(
        torch.rand([1, NC, origin_H, W], dtype=torch.float32),
        device=device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        pad_value=float("nan"),
    )
    ttnn.operations.moreh.mean(x, dim=[2], keepdim=True)
