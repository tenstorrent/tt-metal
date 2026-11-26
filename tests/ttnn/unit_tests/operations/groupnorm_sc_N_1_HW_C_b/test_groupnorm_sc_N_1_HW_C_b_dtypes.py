# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Refinement 4: Float32 input (DRAM interleaved, TILE + ROW_MAJOR), Float32 gamma / beta, TILE-layout gamma / beta.

The format paths the reader branches on:
  * the input ring / RM sticks / output share in the input's format (Float32 pages: 4096 B tiles, 128 B stick
    segments), the pilot k a Float32 sample's bf16 truncation (TILE: row 0 of faces 0/1 per tile);
  * gamma / beta staged from ROW_MAJOR sticks (bf16 or Float32 elements) or from TILE row 0 (one 64 B row-0 read of
    faces 0/1 per channel tile), built into affine maps in gamma / beta's format (word copies; bf16 odd C/G:
    halfword stores);
  * Float32 at the residency regimes (fully resident, partial, stream-only) and with several units per core.
The output contract (TILE, DRAM interleaved, output dtype = input dtype, TOLERANCES[dtype]) is checked by the golden
helper run_groupnorm_sc_N_1_HW_C.
"""

import pytest
import torch
import ttnn

from eval.golden_tests.groupnorm_sc_N_1_HW_C_b.helpers import run_groupnorm_sc_N_1_HW_C
from ttnn.operations.groupnorm_sc_N_1_HW_C_b import TUNING_OVERRIDES

F32, BF16 = ttnn.float32, ttnn.bfloat16
TILE, RM = ttnn.TILE_LAYOUT, ttnn.ROW_MAJOR_LAYOUT

SHAPES = [
    pytest.param(((1, 1, 256, 256), 8), id="256x256_g8"),
    pytest.param(((2, 1, 512, 128), 4), id="2x512x128_g4"),
    pytest.param(((1, 1, 64, 1280), 32), id="64x1280_g32"),
    pytest.param(((1, 1, 1024, 32), 8), id="1024x32_cg4"),
]


@pytest.mark.parametrize("inputs", SHAPES)
@pytest.mark.parametrize("layout", [TILE, RM], ids=["tile", "rm"])
@pytest.mark.parametrize(
    "affine, affine_dtype, affine_layout",
    [
        ("gamma_beta", F32, RM),
        ("gamma_beta", F32, TILE),
        ("gamma_beta", BF16, TILE),
        ("gamma_only", F32, RM),
        ("no_affine", "none", "none"),
    ],
    ids=["gb_f32_rm", "gb_f32_tile", "gb_bf16_tile", "g_f32_rm", "none"],
)
def test_fp32_input(device, inputs, layout, affine, affine_dtype, affine_layout):
    torch.manual_seed(0)
    run_groupnorm_sc_N_1_HW_C(
        inputs,
        device=device,
        dtype=F32,
        layout=layout,
        affine=affine,
        affine_dtype=affine_dtype,
        affine_layout=affine_layout,
    )


@pytest.mark.parametrize(
    "inputs",
    SHAPES
    + [
        # odd C/G: the bf16 affine map is built by halfword stores
        pytest.param(((1, 1, 64, 96), 32), id="64x96_cg3"),
        pytest.param(((1, 1, 128, 448), 8), id="128x448_cg56"),
    ],
)
@pytest.mark.parametrize("layout", [TILE, RM], ids=["tile", "rm"])
@pytest.mark.parametrize(
    "affine, affine_dtype, affine_layout",
    [
        ("gamma_beta", BF16, TILE),
        ("gamma_beta", F32, RM),
        ("gamma_beta", F32, TILE),
        ("gamma_only", BF16, TILE),
    ],
    ids=["gb_bf16_tile", "gb_f32_rm", "gb_f32_tile", "g_bf16_tile"],
)
def test_bf16_input_affine_formats(device, inputs, layout, affine, affine_dtype, affine_layout):
    torch.manual_seed(0)
    run_groupnorm_sc_N_1_HW_C(
        inputs,
        device=device,
        dtype=BF16,
        layout=layout,
        affine=affine,
        affine_dtype=affine_dtype,
        affine_layout=affine_layout,
    )


@pytest.mark.parametrize("inputs", [pytest.param(((1, 1, 64, 96), 32), id="64x96_cg3")])
@pytest.mark.parametrize("layout", [TILE, RM], ids=["tile", "rm"])
def test_fp32_input_odd_cg(device, inputs, layout):
    torch.manual_seed(0)
    run_groupnorm_sc_N_1_HW_C(
        inputs, device=device, dtype=F32, layout=layout, affine="gamma_beta", affine_dtype=F32, affine_layout=TILE
    )


def _run_with_tuning(device, inputs, layout, **tuning):
    saved = dict(TUNING_OVERRIDES)
    TUNING_OVERRIDES.update(tuning)
    try:
        torch.manual_seed(0)
        run_groupnorm_sc_N_1_HW_C(
            inputs,
            device=device,
            dtype=F32,
            layout=layout,
            affine="gamma_beta",
            affine_dtype=F32,
            affine_layout=TILE,
        )
    finally:
        TUNING_OVERRIDES.clear()
        TUNING_OVERRIDES.update(saved)


@pytest.mark.parametrize("layout", [TILE, RM], ids=["tile", "rm"])
@pytest.mark.parametrize("max_resident", [0, 1], ids=["stream_only", "partial"])
def test_fp32_residency(device, layout, max_resident):
    """K < B: the non-resident head is re-read in Float32 for pass 2 (and re-tilized for RM)."""
    _run_with_tuning(device, ((1, 1, 4096, 320), 32), layout, max_resident_chunks=max_resident, chunk_tiles_target=10)


@pytest.mark.parametrize("layout", [TILE, RM], ids=["tile", "rm"])
def test_fp32_multi_unit_per_core(device, layout):
    """Several statistic units per core (max_cores small): the per-block pilot / affine / output share in Float32."""
    _run_with_tuning(device, ((2, 1, 1024, 640), 32), layout, max_cores=4)


@pytest.mark.parametrize(
    "inputs",
    [
        pytest.param(((1, 1, 64, 4096), 32), id="64x4096_wide"),
        pytest.param(((1, 1, 16384, 512), 32), id="16384x512"),
    ],
)
@pytest.mark.parametrize("layout", [TILE, RM], ids=["tile", "rm"])
def test_fp32_large(device, inputs, layout):
    """The Float32 L1 solve (4096 B input pages) on a wide slice and a deep one."""
    torch.manual_seed(0)
    run_groupnorm_sc_N_1_HW_C(
        inputs, device=device, dtype=F32, layout=layout, affine="gamma_beta", affine_dtype=F32, affine_layout=RM
    )
