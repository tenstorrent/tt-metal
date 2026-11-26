# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Refinement 7 device-time bench: bfloat8_b input vs its bf16 twin (not collected by default: leading underscore).

    scripts/run_safe_pytest.sh --profile --no-py-zones \
        tests/ttnn/unit_tests/operations/groupnorm_sc_N_1_HW_C_b/_op_bench_bf8b.py

Read DEVICE KERNEL DURATION [ns] per op row of the ops_perf_results CSV (row order = case order). DRAM interleaved
TILE, gamma + beta in the input's dtype (TILE). No correctness check (the golden suite owns that).
"""

import pytest
import torch
import ttnn

from ttnn.operations.groupnorm_sc_N_1_HW_C_b import groupnorm_sc_N_1_HW_C_b

SHAPES = [
    ("vae_dram_65536x512", (1, 1, 65536, 512), 32),
    ("vae_dram_262144x256", (1, 1, 262144, 256), 32),
    ("refiner_dram_16384x1152", (1, 1, 16384, 1152), 32),
    ("mid_16384x320", (1, 1, 16384, 320), 32),
    ("mochi_4x6360x768", (4, 1, 6360, 768), 32),
]
CASES = [(f"{n}_{d}", s, g, dt) for n, s, g in SHAPES for d, dt in (("bf16", ttnn.bfloat16), ("bf8b", ttnn.bfloat8_b))]


@pytest.mark.parametrize("name, shape, G, dtype", CASES, ids=[c[0] for c in CASES])
def test_bench(device, name, shape, G, dtype):
    torch.manual_seed(0)
    C = shape[-1]
    x = ttnn.from_torch(torch.randn(shape, dtype=torch.bfloat16), dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)
    g = ttnn.from_torch(
        torch.randn(1, 1, 1, C, dtype=torch.bfloat16), dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device
    )
    b = ttnn.from_torch(
        torch.randn(1, 1, 1, C, dtype=torch.bfloat16), dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device
    )
    out = groupnorm_sc_N_1_HW_C_b(x, G, gamma=g, beta=b, eps=1e-6)
    ttnn.synchronize_device(device)
    out.deallocate()
