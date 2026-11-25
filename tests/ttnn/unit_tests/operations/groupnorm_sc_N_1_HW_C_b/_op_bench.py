# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Device-time bench for groupnorm_sc_N_1_HW_C_b (not collected by default: leading underscore).

Run under the profiler, one op dispatch per case:
    scripts/run_safe_pytest.sh --profile --no-py-zones \
        tests/ttnn/unit_tests/operations/groupnorm_sc_N_1_HW_C_b/_op_bench.py
then read DEVICE KERNEL DURATION [ns] per op row of the ops_perf_results CSV (row order = case order).
The cases are the DRAM-interleaved PERF FOCUS loose cases (feature_spec._PERF_CASES) plus TILE/RM pairs
of mid-size shapes. No correctness check here (the golden suite owns that).
"""

import os

import pytest
import torch
import ttnn

from ttnn.operations.groupnorm_sc_N_1_HW_C_b import TUNING_OVERRIDES, groupnorm_sc_N_1_HW_C_b


@pytest.fixture(autouse=True)
def _tuning_from_env():
    """A/B knob: GN_BENCH_CHUNK=<tiles> overrides chunk_tiles_target for the whole bench run."""
    saved = dict(TUNING_OVERRIDES)
    if os.environ.get("GN_BENCH_CHUNK"):
        TUNING_OVERRIDES["chunk_tiles_target"] = int(os.environ["GN_BENCH_CHUNK"])
    yield
    TUNING_OVERRIDES.update(saved)


CASES = [
    # name, shape, G, layout, eps
    ("vae_dram_1048576x128", (1, 1, 1048576, 128), 32, ttnn.TILE_LAYOUT, 1e-6),
    ("vae_dram_65536x512", (1, 1, 65536, 512), 32, ttnn.TILE_LAYOUT, 1e-6),
    ("vae_dram_262144x256", (1, 1, 262144, 256), 32, ttnn.TILE_LAYOUT, 1e-6),
    ("vae_dram_262144x512", (1, 1, 262144, 512), 32, ttnn.TILE_LAYOUT, 1e-6),
    ("vae_dram_1048576x256", (1, 1, 1048576, 256), 32, ttnn.TILE_LAYOUT, 1e-6),
    ("refiner_dram_16384x1152", (1, 1, 16384, 1152), 32, ttnn.TILE_LAYOUT, 1e-5),
    ("flux1_dram_16384x256", (1, 1, 16384, 256), 16, ttnn.TILE_LAYOUT, 1e-6),
    ("flux1_dram_1048576x64", (1, 1, 1048576, 64), 16, ttnn.TILE_LAYOUT, 1e-6),
    ("mochi_dram_8x407040x128", (8, 1, 407040, 128), 32, ttnn.TILE_LAYOUT, 1e-6),
    ("mid_4096x640_tile", (1, 1, 4096, 640), 32, ttnn.TILE_LAYOUT, 1e-5),
    ("mid_4096x640_rm", (1, 1, 4096, 640), 32, ttnn.ROW_MAJOR_LAYOUT, 1e-5),
    ("mid_16384x320_tile", (1, 1, 16384, 320), 32, ttnn.TILE_LAYOUT, 1e-5),
    ("mid_16384x320_rm", (1, 1, 16384, 320), 32, ttnn.ROW_MAJOR_LAYOUT, 1e-5),
]

# L1 BLOCK_SHARDED, out of place (Refinement 1). The TILE rows are the three SDXL attention / VAE-mid
# perf cases; the RM rows are the U-Net resnet shapes run out of place (their model form is in place).
SHARDED_CASES = [
    # name, shape, G, grid, layout, eps
    ("unet_tile_1024x1280", (1, 1, 1024, 1280), 32, (8, 8), ttnn.TILE_LAYOUT, 1e-6),
    ("unet_tile_4096x640_4x8", (1, 1, 4096, 640), 32, (4, 8), ttnn.TILE_LAYOUT, 1e-6),
    ("vae_tile_16384x512", (1, 1, 16384, 512), 32, (8, 8), ttnn.TILE_LAYOUT, 1e-6),
    ("unet_rm_oop_1024x1280", (1, 1, 1024, 1280), 32, (8, 8), ttnn.ROW_MAJOR_LAYOUT, 1e-5),
    ("unet_rm_oop_16384x320", (1, 1, 16384, 320), 32, (8, 8), ttnn.ROW_MAJOR_LAYOUT, 1e-5),
    ("unet_rm_oop_4096x320", (1, 1, 4096, 320), 32, (8, 8), ttnn.ROW_MAJOR_LAYOUT, 1e-5),
    ("unet_rm_oop_4096x640", (1, 1, 4096, 640), 32, (8, 8), ttnn.ROW_MAJOR_LAYOUT, 1e-5),
    ("unet_rm_oop_1024x640", (1, 1, 1024, 640), 32, (8, 8), ttnn.ROW_MAJOR_LAYOUT, 1e-5),
    ("refiner_rm_oop_256x1536", (1, 1, 256, 1536), 32, (8, 8), ttnn.ROW_MAJOR_LAYOUT, 1e-5),
    ("refiner_rm_oop_16384x384", (1, 1, 16384, 384), 32, (8, 8), ttnn.ROW_MAJOR_LAYOUT, 1e-5),
]


@pytest.mark.parametrize("name, shape, G, layout, eps", CASES, ids=[c[0] for c in CASES])
def test_bench(device, name, shape, G, layout, eps):
    torch.manual_seed(0)
    C = shape[-1]
    x = ttnn.from_torch(
        torch.randn(shape, dtype=torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=layout,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    g = ttnn.from_torch(
        torch.randn((1, 1, 1, C), dtype=torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
    )
    b = ttnn.from_torch(
        torch.randn((1, 1, 1, C), dtype=torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
    )
    out = groupnorm_sc_N_1_HW_C_b(x, G, gamma=g, beta=b, eps=eps)
    ttnn.synchronize_device(device)
    out.deallocate()
    x.deallocate()


@pytest.mark.parametrize("name, shape, G, grid, layout, eps", SHARDED_CASES, ids=[c[0] for c in SHARDED_CASES])
def test_bench_sharded(device, name, shape, G, grid, layout, eps):
    from eval.golden_tests.groupnorm_sc_N_1_HW_C_b.helpers import block_shard_memory_config

    torch.manual_seed(0)
    C = shape[-1]
    mem = block_shard_memory_config(shape, grid, layout=layout, dtype=ttnn.bfloat16, device=device)
    x = ttnn.from_torch(
        torch.randn(shape, dtype=torch.bfloat16), dtype=ttnn.bfloat16, layout=layout, device=device, memory_config=mem
    )
    g = ttnn.from_torch(
        torch.randn((1, 1, 1, C), dtype=torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
    )
    b = ttnn.from_torch(
        torch.randn((1, 1, 1, C), dtype=torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
    )
    out = groupnorm_sc_N_1_HW_C_b(x, G, gamma=g, beta=b, eps=eps)
    ttnn.synchronize_device(device)
    out.deallocate()
    x.deallocate()
