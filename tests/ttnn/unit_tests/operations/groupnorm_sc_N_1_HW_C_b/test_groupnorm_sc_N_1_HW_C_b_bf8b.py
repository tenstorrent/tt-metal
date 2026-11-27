# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Refinement 7: bfloat8_b input (TILE, DRAM interleaved; output bfloat8_b) and bfloat8_b TILE gamma / beta.

A bfloat8_b tile is a 64 B exponent section (one shared exponent per 16-lane face row) + one sign / 7-bit magnitude
byte per element. The reader paths this pins:
  * the pilot k: decoded from the staged row-0 mantissas + exponent section of each tile (exact in bf16);
  * hw_non_aligned: the pad rows of each image's last tile row are k-filled in the shared-exponent form - every k is
    truncated to one common exponent first, since a pad face row's 16 lanes may span several groups (C/G < 16);
  * c_non_aligned: the partial last column slice (pad lanes masked by compute, never in the output's valid channels);
  * bfloat8_b gamma / beta: staged row 0 decoded to bf16 into the affine maps (word copies; odd C/G halfword stores);
  * bfloat8_b gamma / beta under bf16 / Float32 input (TILE + ROW_MAJOR);
  * the residency regimes (1088 B pages in the L1 solve), several units per core, a large shape.
The golden contract (TILE output, output dtype = input dtype, TOLERANCES[bfloat8_b] = (0.99, 0.10)) is checked by
run_groupnorm_sc_N_1_HW_C; test_op_error_quantized_input checks the op's own error at a tighter bound.
"""

import pytest
import torch
import ttnn

from eval.golden_tests.groupnorm_sc_N_1_HW_C_b.helpers import (
    check_output,
    pytorch_groupnorm_sc_N_1_HW_C,
    run_groupnorm_sc_N_1_HW_C,
)
from ttnn.operations.groupnorm_sc_N_1_HW_C_b import groupnorm_sc_N_1_HW_C_b

B8, BF16, F32 = ttnn.bfloat8_b, ttnn.bfloat16, ttnn.float32
TILE, RM = ttnn.TILE_LAYOUT, ttnn.ROW_MAJOR_LAYOUT

SHAPES = [
    pytest.param(((1, 1, 256, 256), 8), id="256x256_g8"),
    pytest.param(((2, 1, 512, 128), 4), id="2x512x128_g4"),
    pytest.param(((1, 1, 64, 1280), 32), id="64x1280_g32"),
    pytest.param(((1, 1, 1024, 32), 8), id="1024x32_cg4"),
    pytest.param(((1, 1, 64, 96), 32), id="64x96_cg3"),  # odd C/G: halfword map stores
    # hw_non_aligned (HR edges, C/G < 16: several groups per pad face row)
    pytest.param(((1, 1, 17, 64), 1), id="17x64_g1"),
    pytest.param(((2, 1, 100, 128), 4), id="2x100x128_g4"),
    pytest.param(((3, 1, 48, 320), 32), id="3x48x320_hr16_cg10"),
    pytest.param(((2, 1, 63, 64), 16), id="2x63x64_hr31_cg4"),
    # c_non_aligned (partial last slice)
    pytest.param(((1, 1, 64, 100), 2), id="64x100_g2"),
    pytest.param(((1, 1, 32, 48), 2), id="32x48_cg24"),
    pytest.param(((2, 1, 50, 300), 30), id="2x50x300_both"),
]


@pytest.mark.parametrize("inputs", SHAPES)
@pytest.mark.parametrize(
    "affine, affine_dtype, affine_layout",
    [
        ("gamma_beta", B8, TILE),
        ("gamma_only", B8, TILE),
        ("gamma_beta", BF16, RM),
        ("gamma_beta", F32, TILE),
        ("no_affine", "none", "none"),
    ],
    ids=["gb_b8_tile", "g_b8_tile", "gb_bf16_rm", "gb_f32_tile", "none"],
)
def test_bf8b_input(device, inputs, affine, affine_dtype, affine_layout):
    torch.manual_seed(0)
    run_groupnorm_sc_N_1_HW_C(
        inputs,
        device=device,
        dtype=B8,
        layout=TILE,
        affine=affine,
        affine_dtype=affine_dtype,
        affine_layout=affine_layout,
    )


@pytest.mark.parametrize(
    "inputs",
    [
        pytest.param(((1, 1, 256, 256), 8), id="256x256_g8"),
        pytest.param(((1, 1, 64, 96), 32), id="64x96_cg3"),
        pytest.param(((2, 1, 100, 128), 4), id="2x100x128_g4"),
        pytest.param(((1, 1, 64, 100), 2), id="64x100_g2"),
    ],
)
@pytest.mark.parametrize("layout", [TILE, RM], ids=["tile", "rm"])
@pytest.mark.parametrize("dtype", [BF16, F32], ids=["bf16", "fp32"])
@pytest.mark.parametrize("affine", ["gamma_beta", "gamma_only"])
def test_bf8b_affine(device, inputs, layout, dtype, affine):
    torch.manual_seed(0)
    run_groupnorm_sc_N_1_HW_C(
        inputs, device=device, dtype=dtype, layout=layout, affine=affine, affine_dtype=B8, affine_layout=TILE
    )


@pytest.mark.parametrize(
    "shape, num_groups",
    [((2, 1, 40, 256), 32), ((1, 1, 47, 320), 32), ((2, 1, 64, 256), 16), ((1, 1, 64, 100), 2)],
    ids=["2x40x256_cg8_hr8", "47x320_cg10_hr15", "2x64x256_aligned", "64x100_c_non_aligned"],
)
def test_op_error_quantized_input(device, shape, num_groups):
    """The op's own error: the reference is computed from the device's bfloat8_b input (round-tripped), so input
    quantization is out of the comparison. Group offsets from -12 to +12 give the pilot k's very different exponents:
    every k is truncated to the largest one for the k-fill, and any k-fill / pilot inconsistency (a pad row adding
    k_fill - k to the moments) would shift a whole group's mean visibly."""
    torch.manual_seed(0)
    N, _, HW, C = shape
    Cg = C // num_groups
    offsets = torch.linspace(-12.0, 12.0, num_groups).repeat_interleave(Cg)
    x = (torch.randn(shape) + offsets).to(torch.bfloat16)
    gamma = torch.randn(1, 1, 1, C).to(torch.bfloat16)
    beta = torch.randn(1, 1, 1, C).to(torch.bfloat16)
    tt_x = ttnn.from_torch(x, dtype=B8, layout=TILE, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    tt_g = ttnn.from_torch(gamma, dtype=B8, layout=TILE, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    tt_b = ttnn.from_torch(beta, dtype=B8, layout=TILE, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    xq = ttnn.to_torch(tt_x).float()
    gq = ttnn.to_torch(tt_g).float()
    bq = ttnn.to_torch(tt_b).float()
    out = groupnorm_sc_N_1_HW_C_b(tt_x, num_groups, gamma=tt_g, beta=tt_b)
    expected = pytorch_groupnorm_sc_N_1_HW_C(xq, num_groups, gamma=gq, beta=bq)
    # The output's own bfloat8_b rounding (7-bit mantissas under a shared exponent) dominates what remains.
    check_output(out, expected, shape=shape, dtype=B8, expected_layout=TILE, tolerance=(0.9995, 0.03))


def _run_in_regime(device, assert_inputs_regime, inputs, regime):
    """Runs a shape after checking the default plan puts it in `regime` (shapes are sized for a
    110-core Blackhole grid)."""
    kw = dict(dtype=B8, layout=TILE, affine_dtype=B8, affine_layout=TILE)
    assert_inputs_regime(regime, inputs, **kw)
    torch.manual_seed(0)
    run_groupnorm_sc_N_1_HW_C(inputs, device=device, affine="gamma_beta", **kw)


@pytest.mark.parametrize(
    "inputs, regime",
    [
        # No hw-non-aligned bfloat8_b shape streams with default tuning; the streaming re-read is
        # the same as the aligned case's.
        pytest.param(((2, 1, 4096, 4096), 8), {"stream"}, id="aligned-stream_only"),
        pytest.param(((2, 1, 32768, 1280), 32), {"partial"}, id="aligned-partial"),
        pytest.param(((2, 1, 32744, 1280), 32), {"partial"}, id="hw_non_aligned-partial"),
    ],
)
def test_bf8b_residency(device, assert_inputs_regime, inputs, regime):
    """K < B: the non-resident head is re-read in bfloat8_b for pass 2 (hw_non_aligned: the k-filled last chunk
    resident or re-read)."""
    _run_in_regime(device, assert_inputs_regime, inputs, regime)


@pytest.mark.parametrize(
    "inputs, regime",
    [
        pytest.param(((112, 1, 32, 32), 32), {"full", "multi"}, id="aligned-resident"),
        pytest.param(((112, 1, 16384, 96), 2), {"partial", "multi"}, id="aligned-partial"),
        pytest.param(((112, 1, 100, 32), 32), {"full", "multi"}, id="hw_non_aligned-resident"),
    ],
)
def test_bf8b_multi_unit_per_core(device, assert_inputs_regime, inputs, regime):
    """Several statistic units per core: the per-block pilot decode / k quantization / affine decode."""
    _run_in_regime(device, assert_inputs_regime, inputs, regime)


@pytest.mark.parametrize(
    "inputs",
    [
        pytest.param(((1, 1, 64, 4096), 32), id="64x4096_wide"),
        pytest.param(((1, 1, 16384, 512), 32), id="16384x512"),
        pytest.param(((4, 1, 6360, 768), 32), id="mochi_4x6360x768"),
    ],
)
def test_bf8b_large(device, inputs):
    """The L1 solve with 1088 B input pages on a wide slice, a deep one and the Mochi hw_non_aligned shape."""
    torch.manual_seed(0)
    run_groupnorm_sc_N_1_HW_C(
        inputs, device=device, dtype=B8, layout=TILE, affine="gamma_beta", affine_dtype=B8, affine_layout=TILE
    )


# More than 64 channel tiles on one core (only bfloat8_b input fits): pass 2 must seed each tile from its own b'
# tile. Checked per channel tile, because one wrong tile out of 65 stays inside the whole-tensor bfloat8_b tolerance.
@pytest.mark.parametrize(
    "shape, num_groups", [((2, 1, 1024, 2080), 1), ((2, 1, 1024, 2080), 4), ((2, 1, 4096, 2112), 1)]
)
@pytest.mark.parametrize("affine_dtype, affine_layout", [(B8, TILE), (BF16, RM)])
def test_more_than_64_channel_tiles(device, shape, num_groups, affine_dtype, affine_layout):
    torch.manual_seed(0)
    C = shape[-1]
    x = ttnn.from_torch(
        torch.randn(shape).bfloat16(), dtype=B8, layout=TILE, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    gamma = ttnn.from_torch(torch.randn(1, 1, 1, C).bfloat16(), dtype=affine_dtype, layout=affine_layout, device=device)
    beta = ttnn.from_torch(torch.randn(1, 1, 1, C).bfloat16(), dtype=affine_dtype, layout=affine_layout, device=device)
    out = ttnn.to_torch(groupnorm_sc_N_1_HW_C_b(x, num_groups, gamma=gamma, beta=beta)).float()
    # Reference on the device-quantized operands.
    xq = ttnn.to_torch(x).float()
    gq = ttnn.to_torch(gamma).float().reshape(1, 1, 1, C)
    bq = ttnn.to_torch(beta).float().reshape(1, 1, 1, C)
    ref = torch.nn.functional.group_norm(xq.permute(0, 3, 1, 2), num_groups, eps=1e-5).permute(0, 2, 3, 1) * gq + bq
    per_tile = (out - ref).abs().reshape(-1, C // 32, 32).mean(dim=(0, 2))
    bad = [t for t, e in enumerate(per_tile.tolist()) if e > 0.05]
    assert not bad, f"channel tiles with mean abs error > 0.05: {bad}"
