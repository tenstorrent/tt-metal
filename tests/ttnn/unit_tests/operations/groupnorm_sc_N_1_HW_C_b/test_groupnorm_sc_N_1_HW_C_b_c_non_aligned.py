# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Refinement 6: c_non_aligned (C % 32 != 0), DRAM interleaved, TILE + ROW_MAJOR, bf16 + Float32.

A row is ceil(C / 32) tiles and S = ceil(C / lcm(Cg, 32)) spans; column slices of S / gx spans start on tile and group
boundaries, so only the last slice is partial (vc_last valid channels in ceil(vc_last / 32) tiles). The reader reads
and the writers write only those (RM: the stick's valid bytes), and compute lane-masks the pad channels after pass 1's
column collapse (COL_MASK, a select: TILE padding is unspecified, possibly NaN). The cases pin:
  * the golden c_non_aligned shapes (Cg = C with G = 1; Cg % 32 != 0 with the pad inside a group's lane span);
  * gx > 1 with a last slice holding fewer tiles than W (1 of 5), and several units per core whose slices differ;
  * TILE inputs whose pad columns hold garbage (finite and NaN), and an offset-heavy input (mean 3);
  * K < B (pass-2 re-reads, RM re-tilized from the valid bytes only), HW and C both non-aligned, every affine form.
"""

import pytest
import torch
import ttnn

from eval.golden_tests.groupnorm_sc_N_1_HW_C_b.helpers import (
    TOLERANCES,
    check_output,
    pytorch_groupnorm_sc_N_1_HW_C,
    run_groupnorm_sc_N_1_HW_C,
)
from ttnn.operations.groupnorm_sc_N_1_HW_C_b import TUNING_OVERRIDES, groupnorm_sc_N_1_HW_C_b

F32, BF16 = ttnn.float32, ttnn.bfloat16
TILE, RM = ttnn.TILE_LAYOUT, ttnn.ROW_MAJOR_LAYOUT
TORCH_DTYPE = {BF16: torch.bfloat16, F32: torch.float32}

SHAPES = [
    # golden feature_spec c_non_aligned entries
    pytest.param(((1, 1, 64, 17), 1), id="64x17_g1"),
    pytest.param(((1, 1, 64, 50), 1), id="64x50_g1"),
    pytest.param(((1, 1, 128, 100), 1), id="128x100_g1"),
    pytest.param(((2, 1, 64, 47), 1), id="2x64x47_g1"),
    pytest.param(((1, 1, 64, 48), 2), id="64x48_g2"),
    pytest.param(((1, 1, 128, 48), 3), id="128x48_g3_gx2"),
    pytest.param(((1, 1, 64, 80), 4), id="64x80_g4"),
    pytest.param(((1, 1, 128, 144), 4), id="128x144_g4"),
    pytest.param(((1, 1, 64, 200), 8), id="64x200_g8"),
    # gx = 3 slices of 5 tiles (Cg = 10, lcm 160): the last slice is 10 channels in 1 tile; 33 groups overall
    pytest.param(((1, 1, 256, 330), 33), id="256x330_g33_gx3"),
    # C < 16 (the pad covers most of face 0 and all of face 1)
    pytest.param(((2, 1, 96, 6), 3), id="2x96x6_g3"),
]


@pytest.mark.parametrize("inputs", SHAPES)
@pytest.mark.parametrize("layout", [TILE, RM], ids=["tile", "rm"])
@pytest.mark.parametrize("dtype", [BF16, F32], ids=["bf16", "fp32"])
def test_c_non_aligned(device, inputs, layout, dtype):
    torch.manual_seed(0)
    run_groupnorm_sc_N_1_HW_C(
        inputs,
        device=device,
        dtype=dtype,
        layout=layout,
        affine="gamma_beta",
        affine_dtype=dtype,
        affine_layout=RM if dtype == BF16 else TILE,
    )


@pytest.mark.parametrize(
    "affine, affine_dtype, affine_layout",
    [("gamma_beta", BF16, TILE), ("gamma_only", F32, RM), ("no_affine", "none", "none")],
    ids=["gb_bf16_tile", "g_fp32_rm", "none"],
)
@pytest.mark.parametrize("layout", [TILE, RM], ids=["tile", "rm"])
def test_c_non_aligned_affine(device, layout, affine, affine_dtype, affine_layout):
    torch.manual_seed(0)
    run_groupnorm_sc_N_1_HW_C(
        ((1, 1, 256, 330), 33),
        device=device,
        dtype=BF16,
        layout=layout,
        affine=affine,
        affine_dtype=affine_dtype,
        affine_layout=affine_layout,
    )


@pytest.mark.parametrize("layout", [TILE, RM], ids=["tile", "rm"])
@pytest.mark.parametrize("dtype", [BF16, F32], ids=["bf16", "fp32"])
def test_hw_and_c_non_aligned(device, layout, dtype):
    """Both edges partial: the last tile row's pad rows are k-filled (Refinement 5), the pad columns masked."""
    torch.manual_seed(0)
    run_groupnorm_sc_N_1_HW_C(
        ((2, 1, 50, 100), 4),
        device=device,
        dtype=dtype,
        layout=layout,
        affine="gamma_beta",
        affine_dtype=dtype,
        affine_layout=RM,
    )


def _run_direct(device, x, num_groups, dtype, layout, *, pad_fill=None):
    """Run the op on x (N, 1, HW, C); pad_fill (TILE only) writes that value into the pad columns of the last tile
    column (a logical-shape view over a padded tensor), which the op must never let into the statistics."""
    N, _, HW, C = x.shape
    torch.manual_seed(1)
    gamma = torch.randn(1, 1, 1, C).to(TORCH_DTYPE[dtype])
    beta = torch.randn(1, 1, 1, C).to(TORCH_DTYPE[dtype])
    expected = pytorch_groupnorm_sc_N_1_HW_C(x, num_groups, gamma=gamma, beta=beta)
    if pad_fill is None:
        tt_x = ttnn.from_torch(x, dtype=dtype, layout=layout, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    else:
        assert layout == TILE
        Cp = -(-C // 32) * 32
        xp = torch.full((N, 1, HW, Cp), pad_fill, dtype=x.dtype)
        xp[..., :C] = x
        tt_x = ttnn.from_torch(xp, dtype=dtype, layout=TILE, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        tt_x = ttnn.reshape(tt_x, [N, 1, HW, C], [N, 1, HW, Cp])
        assert tuple(tt_x.shape) == (N, 1, HW, C)
    tt_g = ttnn.from_torch(gamma, dtype=dtype, layout=RM, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    tt_b = ttnn.from_torch(beta, dtype=dtype, layout=RM, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    out = groupnorm_sc_N_1_HW_C_b(tt_x, num_groups, gamma=tt_g, beta=tt_b)
    check_output(out, expected, shape=(N, 1, HW, C), dtype=dtype, expected_layout=TILE, tolerance=TOLERANCES[dtype])


@pytest.mark.parametrize("pad_fill", [1.0e4, float("nan")], ids=["pad_1e4", "pad_nan"])
@pytest.mark.parametrize("dtype", [BF16, F32], ids=["bf16", "fp32"])
@pytest.mark.parametrize(
    "shape, num_groups",
    [((2, 1, 64, 47), 1), ((1, 1, 128, 80), 4), ((1, 1, 64, 330), 33)],
    ids=["2x64x47_g1", "128x80_g4", "64x330_g33_gx3"],
)
def test_tile_padding_garbage(device, shape, num_groups, dtype, pad_fill):
    torch.manual_seed(0)
    x = torch.randn(shape).to(TORCH_DTYPE[dtype])
    _run_direct(device, x, num_groups, dtype, TILE, pad_fill=pad_fill)


@pytest.mark.parametrize("layout", [TILE, RM], ids=["tile", "rm"])
@pytest.mark.parametrize("dtype", [BF16, F32], ids=["bf16", "fp32"])
def test_offset_heavy(device, layout, dtype):
    """Mean 3, std 1: a pad channel of zeros (or of stale ring data) inside a group would shift its mean visibly."""
    torch.manual_seed(0)
    x = (torch.randn(2, 1, 64, 200) + 3.0).to(TORCH_DTYPE[dtype])
    _run_direct(device, x, 8, dtype, layout)


def _run_with_tuning(device, inputs, layout, dtype, **tuning):
    saved = dict(TUNING_OVERRIDES)
    TUNING_OVERRIDES.update(tuning)
    try:
        torch.manual_seed(0)
        run_groupnorm_sc_N_1_HW_C(
            inputs,
            device=device,
            dtype=dtype,
            layout=layout,
            affine="gamma_beta",
            affine_dtype=BF16,
            affine_layout=RM,
        )
    finally:
        TUNING_OVERRIDES.clear()
        TUNING_OVERRIDES.update(saved)


@pytest.mark.parametrize("dtype", [BF16, F32], ids=["bf16", "fp32"])
@pytest.mark.parametrize("layout", [TILE, RM], ids=["tile", "rm"])
@pytest.mark.parametrize("max_resident", [0, 1], ids=["stream_only", "partial"])
def test_residency(device, layout, dtype, max_resident):
    """K < B: several chunks per core, the partial slice's chunks re-read in pass 2 (RM: valid bytes, re-tilized)."""
    _run_with_tuning(
        device, ((2, 1, 1024, 330), 33), layout, dtype, max_resident_chunks=max_resident, chunk_tiles_target=10
    )


@pytest.mark.parametrize("layout", [TILE, RM], ids=["tile", "rm"])
def test_multi_unit_per_core(device, layout):
    """Several units (images x slices) per core: full and partial slices alternate on one core (per-block vc)."""
    _run_with_tuning(device, ((3, 1, 64, 330), 33), layout, BF16, max_cores=4)
