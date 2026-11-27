# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Refinement 5: hw_non_aligned (HW % 32 != 0), DRAM interleaved, TILE + ROW_MAJOR, bf16 + Float32.

Each image is ceil(HW / 32) tile rows; its last tile row holds HR = HW % 32 valid rows. The reader overwrites the pad
rows of that row's pass-1 copy with the pilot k (x - k = 0, x * (x - k) = 0: no contribution to either shifted
moment) and RM reads only the HR valid sticks (the next sticks are the next image's). The cases pin:
  * the golden hw_non_aligned shapes plus HR edges (1, 16, 31, HW < 16), Cg % 32 != 0, N > 1;
  * TILE inputs whose pad rows hold garbage (finite and NaN): TILE padding is unspecified, it must never be read;
  * an offset-heavy input (mean 3), where any pad row entering the moments would move the mean visibly;
  * K < B (the last chunk re-read in pass 2, stream only / partial) and several units per core (one fill per block).
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
from ttnn.operations.groupnorm_sc_N_1_HW_C_b import groupnorm_sc_N_1_HW_C_b

F32, BF16 = ttnn.float32, ttnn.bfloat16
TILE, RM = ttnn.TILE_LAYOUT, ttnn.ROW_MAJOR_LAYOUT
TORCH_DTYPE = {BF16: torch.bfloat16, F32: torch.float32}

SHAPES = [
    # golden feature_spec hw_non_aligned entries
    pytest.param(((1, 1, 17, 64), 1), id="17x64_g1"),
    pytest.param(((1, 1, 50, 128), 1), id="50x128_g1"),
    pytest.param(((1, 1, 47, 256), 1), id="47x256_g1"),
    pytest.param(((2, 1, 100, 128), 1), id="2x100x128_g1"),
    pytest.param(((4, 1, 6360, 768), 32), id="mochi_4x6360x768_g32"),
    # HR edges, partial channels per group, N > 1
    pytest.param(((1, 1, 33, 64), 2), id="33x64_hr1"),
    pytest.param(((3, 1, 48, 320), 32), id="3x48x320_hr16_cg10"),
    pytest.param(((2, 1, 63, 64), 2), id="2x63x64_hr31"),
    pytest.param(((1, 1, 5, 640), 32), id="5x640_cg20"),
]


@pytest.mark.parametrize("inputs", SHAPES)
@pytest.mark.parametrize("layout", [TILE, RM], ids=["tile", "rm"])
@pytest.mark.parametrize("dtype", [BF16, F32], ids=["bf16", "fp32"])
def test_hw_non_aligned(device, inputs, layout, dtype):
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
    [("gamma_only", BF16, TILE), ("no_affine", "none", "none")],
    ids=["g_bf16_tile", "none"],
)
@pytest.mark.parametrize("layout", [TILE, RM], ids=["tile", "rm"])
def test_hw_non_aligned_affine(device, layout, affine, affine_dtype, affine_layout):
    torch.manual_seed(0)
    run_groupnorm_sc_N_1_HW_C(
        ((2, 1, 100, 640), 32),
        device=device,
        dtype=BF16,
        layout=layout,
        affine=affine,
        affine_dtype=affine_dtype,
        affine_layout=affine_layout,
    )


def _run_direct(device, x, num_groups, dtype, layout, *, pad_fill=None):
    """Run the op on x (N, 1, HW, C); pad_fill (TILE only) writes that value into the pad rows of every image's last
    tile row (a logical-shape view over a padded tensor), which the op must never let into the statistics."""
    N, _, HW, C = x.shape
    torch.manual_seed(1)
    gamma = torch.randn(1, 1, 1, C).to(TORCH_DTYPE[dtype])
    beta = torch.randn(1, 1, 1, C).to(TORCH_DTYPE[dtype])
    expected = pytorch_groupnorm_sc_N_1_HW_C(x, num_groups, gamma=gamma, beta=beta)
    if pad_fill is None:
        tt_x = ttnn.from_torch(x, dtype=dtype, layout=layout, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    else:
        assert layout == TILE
        Hp = -(-HW // 32) * 32
        xp = torch.full((N, 1, Hp, C), pad_fill, dtype=x.dtype)
        xp[:, :, :HW, :] = x
        tt_x = ttnn.from_torch(xp, dtype=dtype, layout=TILE, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        tt_x = ttnn.reshape(tt_x, [N, 1, HW, C], [N, 1, Hp, C])
        assert tuple(tt_x.shape) == (N, 1, HW, C)
    tt_g = ttnn.from_torch(gamma, dtype=dtype, layout=RM, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    tt_b = ttnn.from_torch(beta, dtype=dtype, layout=RM, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    out = groupnorm_sc_N_1_HW_C_b(tt_x, num_groups, gamma=tt_g, beta=tt_b)
    check_output(out, expected, shape=(N, 1, HW, C), dtype=dtype, expected_layout=TILE, tolerance=TOLERANCES[dtype])


@pytest.mark.parametrize("pad_fill", [1.0e4, float("nan")], ids=["pad_1e4", "pad_nan"])
@pytest.mark.parametrize("dtype", [BF16, F32], ids=["bf16", "fp32"])
@pytest.mark.parametrize(
    "shape, num_groups",
    [((2, 1, 100, 128), 4), ((1, 1, 47, 320), 32)],
    ids=["2x100x128_g4", "47x320_g32"],
)
def test_tile_padding_garbage(device, shape, num_groups, dtype, pad_fill):
    torch.manual_seed(0)
    x = torch.randn(shape).to(TORCH_DTYPE[dtype])
    _run_direct(device, x, num_groups, dtype, TILE, pad_fill=pad_fill)


@pytest.mark.parametrize("layout", [TILE, RM], ids=["tile", "rm"])
@pytest.mark.parametrize("dtype", [BF16, F32], ids=["bf16", "fp32"])
def test_offset_heavy(device, layout, dtype):
    """Mean 3, std 1: a pad row of zeros (or of the next image's rows) inside the moments would shift every group mean
    by ~3 * pad / n."""
    torch.manual_seed(0)
    x = (torch.randn(2, 1, 40, 256) + 3.0).to(TORCH_DTYPE[dtype])
    _run_direct(device, x, 8, dtype, layout)


def _run_in_regime(device, assert_inputs_regime, inputs, layout, dtype, regime):
    """Runs a shape after checking the default plan puts it in `regime` (shapes are sized for a
    110-core Blackhole grid)."""
    kw = dict(dtype=dtype, layout=layout, affine_dtype=BF16, affine_layout=RM)
    assert_inputs_regime(regime, inputs, **kw)
    torch.manual_seed(0)
    run_groupnorm_sc_N_1_HW_C(inputs, device=device, affine="gamma_beta", **kw)


@pytest.mark.parametrize(
    "dtype, layout, inputs, regime",
    [
        # bf16 TILE has no hw-non-aligned shape that streams with default tuning (it stays resident
        # up to the shape limits); the streaming re-read is covered by fp32 TILE and bf16 RM.
        pytest.param(BF16, RM, ((2, 1, 2024, 1568), 1), {"stream"}, id="bf16-rm-stream_only"),
        pytest.param(F32, TILE, ((2, 1, 2024, 2496), 2), {"stream"}, id="fp32-tile-stream_only"),
        pytest.param(F32, RM, ((2, 1, 1000, 2240), 2), {"stream"}, id="fp32-rm-stream_only"),
        pytest.param(BF16, TILE, ((2, 1, 65512, 512), 32), {"partial"}, id="bf16-tile-partial"),
        pytest.param(BF16, RM, ((2, 1, 65512, 512), 32), {"partial"}, id="bf16-rm-partial"),
        pytest.param(F32, TILE, ((2, 1, 32744, 512), 32), {"partial"}, id="fp32-tile-partial"),
        pytest.param(F32, RM, ((2, 1, 16360, 640), 32), {"partial"}, id="fp32-rm-partial"),
    ],
)
def test_residency(device, assert_inputs_regime, dtype, layout, inputs, regime):
    """K < B: several chunks per core, the last (k-filled) chunk resident or re-read in pass 2 (RM: valid sticks
    only, re-tilized)."""
    _run_in_regime(device, assert_inputs_regime, inputs, layout, dtype, regime)


@pytest.mark.parametrize("layout", [TILE, RM], ids=["tile", "rm"])
@pytest.mark.parametrize(
    "inputs, regime",
    [
        pytest.param(((112, 1, 100, 32), 32), {"full", "multi"}, id="resident"),
        pytest.param(((112, 1, 16360, 64), 1), {"partial", "multi"}, id="partial"),
    ],
)
def test_multi_unit_per_core(device, assert_inputs_regime, layout, inputs, regime):
    """Several units (images x slices) per core: every block has a partial last tile row and fills it."""
    _run_in_regime(device, assert_inputs_regime, inputs, layout, BF16, regime)
