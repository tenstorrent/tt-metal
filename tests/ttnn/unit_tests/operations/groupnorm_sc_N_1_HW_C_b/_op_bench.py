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

import pytest
import torch
import ttnn

from ttnn.operations.groupnorm_sc_N_1_HW_C_b import groupnorm_sc_N_1_HW_C_b


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


# L1 BLOCK_SHARDED, in place (Refinement 2): the nine RM PERF FOCUS cases (their model form) plus the three TILE
# perf shapes run in place.
SHARDED_INPLACE_CASES = [
    # name, shape, G, grid, layout, eps
    ("unet_rm_ip_1024x1280", (1, 1, 1024, 1280), 32, (8, 8), ttnn.ROW_MAJOR_LAYOUT, 1e-5),
    ("unet_rm_ip_16384x320", (1, 1, 16384, 320), 32, (8, 8), ttnn.ROW_MAJOR_LAYOUT, 1e-5),
    ("unet_rm_ip_4096x320", (1, 1, 4096, 320), 32, (8, 8), ttnn.ROW_MAJOR_LAYOUT, 1e-5),
    ("unet_rm_ip_4096x640", (1, 1, 4096, 640), 32, (8, 8), ttnn.ROW_MAJOR_LAYOUT, 1e-5),
    ("unet_rm_ip_1024x640", (1, 1, 1024, 640), 32, (8, 8), ttnn.ROW_MAJOR_LAYOUT, 1e-5),
    ("refiner_rm_ip_256x1536", (1, 1, 256, 1536), 32, (8, 8), ttnn.ROW_MAJOR_LAYOUT, 1e-5),
    ("refiner_rm_ip_16384x384", (1, 1, 16384, 384), 32, (8, 8), ttnn.ROW_MAJOR_LAYOUT, 1e-5),
    ("unet_rm_ip_16384x640", (1, 1, 16384, 640), 32, (8, 8), ttnn.ROW_MAJOR_LAYOUT, 1e-5),
    ("unet_rm_ip_16384x960", (1, 1, 16384, 960), 32, (8, 8), ttnn.ROW_MAJOR_LAYOUT, 1e-5),
    ("unet_tile_ip_1024x1280", (1, 1, 1024, 1280), 32, (8, 8), ttnn.TILE_LAYOUT, 1e-6),
    ("unet_tile_ip_4096x640_4x8", (1, 1, 4096, 640), 32, (4, 8), ttnn.TILE_LAYOUT, 1e-6),
    ("vae_tile_ip_16384x512", (1, 1, 16384, 512), 32, (8, 8), ttnn.TILE_LAYOUT, 1e-6),
]


# Refinement 4: Float32 input / Float32 or TILE gamma-beta on DRAM interleaved shapes.
# name, shape, G, layout, eps, dtype, affine dtype, affine layout
FORMAT_CASES = [
    (
        "f32_vae_dram_65536x512",
        (1, 1, 65536, 512),
        32,
        ttnn.TILE_LAYOUT,
        1e-6,
        ttnn.float32,
        ttnn.float32,
        ttnn.TILE_LAYOUT,
    ),
    (
        "f32_refiner_dram_16384x1152",
        (1, 1, 16384, 1152),
        32,
        ttnn.TILE_LAYOUT,
        1e-5,
        ttnn.float32,
        ttnn.float32,
        ttnn.ROW_MAJOR_LAYOUT,
    ),
    (
        "f32_flux1_dram_16384x256",
        (1, 1, 16384, 256),
        16,
        ttnn.TILE_LAYOUT,
        1e-6,
        ttnn.float32,
        ttnn.float32,
        ttnn.ROW_MAJOR_LAYOUT,
    ),
    (
        "f32_mid_4096x640_rm",
        (1, 1, 4096, 640),
        32,
        ttnn.ROW_MAJOR_LAYOUT,
        1e-5,
        ttnn.float32,
        ttnn.float32,
        ttnn.ROW_MAJOR_LAYOUT,
    ),
    (
        "bf16_gtile_vae_dram_65536x512",
        (1, 1, 65536, 512),
        32,
        ttnn.TILE_LAYOUT,
        1e-6,
        ttnn.bfloat16,
        ttnn.bfloat16,
        ttnn.TILE_LAYOUT,
    ),
    (
        "bf16_gf32_vae_dram_65536x512",
        (1, 1, 65536, 512),
        32,
        ttnn.TILE_LAYOUT,
        1e-6,
        ttnn.bfloat16,
        ttnn.float32,
        ttnn.ROW_MAJOR_LAYOUT,
    ),
]


# Refinement 5: HW % 32 != 0 (Mochi, HW % 32 = 24; the reader k-fills each image's last tile row) next to its aligned
# twin (HW rounded up to whole tile rows): the fill and the trimmed RM reads should cost ~nothing.
# name, shape, G, layout, eps
HW_NON_ALIGNED_CASES = [
    ("mochi_hwna_4x6360x768_tile", (4, 1, 6360, 768), 32, ttnn.TILE_LAYOUT, 1e-6),
    ("mochi_hwna_4x6360x768_rm", (4, 1, 6360, 768), 32, ttnn.ROW_MAJOR_LAYOUT, 1e-6),
    ("mochi_twin_4x6368x768_tile", (4, 1, 6368, 768), 32, ttnn.TILE_LAYOUT, 1e-6),
    ("mochi_twin_4x6368x768_rm", (4, 1, 6368, 768), 32, ttnn.ROW_MAJOR_LAYOUT, 1e-6),
]


def _bench(
    device, shape, G, layout, eps, dtype=ttnn.bfloat16, aff_dtype=ttnn.bfloat16, aff_layout=ttnn.ROW_MAJOR_LAYOUT
):
    torch.manual_seed(0)
    C = shape[-1]
    x = ttnn.from_torch(
        torch.randn(shape, dtype=torch.float32 if dtype == ttnn.float32 else torch.bfloat16),
        dtype=dtype,
        layout=layout,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    g = ttnn.from_torch(torch.randn((1, 1, 1, C)), dtype=aff_dtype, layout=aff_layout, device=device)
    b = ttnn.from_torch(torch.randn((1, 1, 1, C)), dtype=aff_dtype, layout=aff_layout, device=device)
    out = groupnorm_sc_N_1_HW_C_b(x, G, gamma=g, beta=b, eps=eps)
    ttnn.synchronize_device(device)
    out.deallocate()
    x.deallocate()


@pytest.mark.parametrize("name, shape, G, layout, eps", CASES, ids=[c[0] for c in CASES])
def test_bench(device, name, shape, G, layout, eps):
    _bench(device, shape, G, layout, eps)


@pytest.mark.parametrize(
    "name, shape, G, layout, eps, dtype, aff_dtype, aff_layout", FORMAT_CASES, ids=[c[0] for c in FORMAT_CASES]
)
def test_bench_formats(device, name, shape, G, layout, eps, dtype, aff_dtype, aff_layout):
    _bench(device, shape, G, layout, eps, dtype, aff_dtype, aff_layout)


@pytest.mark.parametrize("name, shape, G, layout, eps", HW_NON_ALIGNED_CASES, ids=[c[0] for c in HW_NON_ALIGNED_CASES])
def test_bench_hw_non_aligned(device, name, shape, G, layout, eps):
    _bench(device, shape, G, layout, eps)


@pytest.mark.parametrize("name, shape, G, grid, layout, eps", SHARDED_CASES, ids=[c[0] for c in SHARDED_CASES])
def test_bench_sharded(device, name, shape, G, grid, layout, eps):
    _bench_sharded(device, shape, G, grid, layout, eps, in_place=False)


@pytest.mark.parametrize(
    "name, shape, G, grid, layout, eps", SHARDED_INPLACE_CASES, ids=[c[0] for c in SHARDED_INPLACE_CASES]
)
def test_bench_sharded_inplace(device, name, shape, G, grid, layout, eps):
    _bench_sharded(device, shape, G, grid, layout, eps, in_place=True)


def _bench_sharded(device, shape, G, grid, layout, eps, *, in_place):
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
    out = groupnorm_sc_N_1_HW_C_b(x, G, gamma=g, beta=b, eps=eps, in_place=in_place)
    ttnn.synchronize_device(device)
    # In place the output aliases the input: free the shard once, by force (a plain deallocate is a no-op while
    # the other handle is alive), so the next case plans against a clean L1.
    ttnn.deallocate(out, force=True)
    if not in_place:
        x.deallocate()


# Guard set (Perf 1): one representative per kernel path x layout x placement for the affine paths the focus
# benches (gamma + beta) do not run, plus an odd channels-per-group case (the reader's halfword affine-map path).
GUARD_CASES = [
    # name, shape, G, grid (None = DRAM interleaved), layout, in_place, affine
    ("g_rm_ip_1024x1280_gonly", (1, 1, 1024, 1280), 32, (8, 8), ttnn.ROW_MAJOR_LAYOUT, True, "gamma_only"),
    ("g_rm_ip_1024x1280_none", (1, 1, 1024, 1280), 32, (8, 8), ttnn.ROW_MAJOR_LAYOUT, True, "no_affine"),
    ("g_tile_oop_1024x1280_gonly", (1, 1, 1024, 1280), 32, (8, 8), ttnn.TILE_LAYOUT, False, "gamma_only"),
    ("g_tile_oop_1024x1280_none", (1, 1, 1024, 1280), 32, (8, 8), ttnn.TILE_LAYOUT, False, "no_affine"),
    ("g_dram_tile_4096x640_gonly", (1, 1, 4096, 640), 32, None, ttnn.TILE_LAYOUT, False, "gamma_only"),
    ("g_dram_tile_4096x640_none", (1, 1, 4096, 640), 32, None, ttnn.TILE_LAYOUT, False, "no_affine"),
    ("g_dram_rm_4096x640_gonly", (1, 1, 4096, 640), 32, None, ttnn.ROW_MAJOR_LAYOUT, False, "gamma_only"),
    ("g_dram_rm_4096x640_none", (1, 1, 4096, 640), 32, None, ttnn.ROW_MAJOR_LAYOUT, False, "no_affine"),
    ("g_dram_tile_16384x96_cg3", (1, 1, 16384, 96), 32, None, ttnn.TILE_LAYOUT, False, "gamma_beta"),
]


@pytest.mark.parametrize("name, shape, G, grid, layout, in_place, affine", GUARD_CASES, ids=[c[0] for c in GUARD_CASES])
def test_bench_guard(device, name, shape, G, grid, layout, in_place, affine):
    torch.manual_seed(0)
    C = shape[-1]
    if grid is None:
        mem = ttnn.DRAM_MEMORY_CONFIG
    else:
        from eval.golden_tests.groupnorm_sc_N_1_HW_C_b.helpers import block_shard_memory_config

        mem = block_shard_memory_config(shape, grid, layout=layout, dtype=ttnn.bfloat16, device=device)
    x = ttnn.from_torch(
        torch.randn(shape, dtype=torch.bfloat16), dtype=ttnn.bfloat16, layout=layout, device=device, memory_config=mem
    )

    def affine_row():
        return ttnn.from_torch(
            torch.randn((1, 1, 1, C), dtype=torch.bfloat16),
            dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=device,
        )

    g = affine_row() if affine in ("gamma_beta", "gamma_only") else None
    b = affine_row() if affine == "gamma_beta" else None
    out = groupnorm_sc_N_1_HW_C_b(x, G, gamma=g, beta=b, eps=1e-5, in_place=in_place)
    ttnn.synchronize_device(device)
    ttnn.deallocate(out, force=True)
    if not in_place:
        x.deallocate()


# Refinement 6: C % 32 != 0 (the last column slice is partial; compute lane-masks its pad channels) next to an aligned
# twin with the same tile width and span structure (same Cg, so the same lcm span and slice count): the partial slice's
# trimmed reads/writes and the lane mask should cost ~nothing.
# name, shape, G, layout, eps
C_NON_ALIGNED_CASES = [
    ("cna_16384x300_g30_tile", (1, 1, 16384, 300), 30, ttnn.TILE_LAYOUT, 1e-5),
    ("cna_16384x300_g30_rm", (1, 1, 16384, 300), 30, ttnn.ROW_MAJOR_LAYOUT, 1e-5),
    ("cna_twin_16384x320_g32_tile", (1, 1, 16384, 320), 32, ttnn.TILE_LAYOUT, 1e-5),
    ("cna_twin_16384x320_g32_rm", (1, 1, 16384, 320), 32, ttnn.ROW_MAJOR_LAYOUT, 1e-5),
    ("cna_65536x100_g1_tile", (1, 1, 65536, 100), 1, ttnn.TILE_LAYOUT, 1e-5),
    ("cna_65536x100_g1_rm", (1, 1, 65536, 100), 1, ttnn.ROW_MAJOR_LAYOUT, 1e-5),
    ("cna_twin_65536x128_g1_tile", (1, 1, 65536, 128), 1, ttnn.TILE_LAYOUT, 1e-5),
    ("cna_twin_65536x128_g1_rm", (1, 1, 65536, 128), 1, ttnn.ROW_MAJOR_LAYOUT, 1e-5),
]


@pytest.mark.parametrize("name, shape, G, layout, eps", C_NON_ALIGNED_CASES, ids=[c[0] for c in C_NON_ALIGNED_CASES])
def test_bench_c_non_aligned(device, name, shape, G, layout, eps):
    _bench(device, shape, G, layout, eps)
