# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Golden coverage cut for groupnorm_sc_N_1_HW_C_b: every value of every SUPPORTED axis in at least one case (values
combined across cases, not their product), plus the smallest golden INPUTS entry and the largest per topology (DRAM
interleaved and L1 BLOCK_SHARDED). Every case is a golden INPUTS entry, no cell is INVALID or EXCLUDED, and each is
checked as the golden harness checks it: ``pytorch_groupnorm_sc_N_1_HW_C`` on the torch inputs, ``TOLERANCES[dtype]``
and the placement contract (``check_placement``).

The op is called through the acceptance suite's route fixture (``groupnorm_sc_N_1_HW_C_b_op``), so the native run
covers these cases too.

Axis coverage (case ids in brackets):
  dtype          bf16 [min, dram_max, refiner, bs_*], fp32 [fp32_rm, fp32_tile_hw], bf8b [bf8b_c]
  layout         TILE [min, ...], ROW_MAJOR [refiner, fp32_rm, bs_max_rm_in_place, bs_rm_straddle]
  alignment      tile_aligned [min], hw_non_aligned [dram_max, fp32_tile_hw], c_non_aligned [fp32_rm, bf8b_c]
  affine         gamma_beta [min], gamma_only [refiner, bs_rm_straddle], no_affine [fp32_rm, bs_tile_pad_col]
  affine_dtype   bf16 [min], fp32 [refiner, fp32_tile_hw], bf8b [bf8b_c], none [fp32_rm]
  affine_layout  RM [min], TILE [refiner, bf8b_c], none [fp32_rm]
  memory_layout  INTERLEAVED [min], BLOCK_SHARDED [bs_*]
  shard_fit      none [min], even [bs_tile_even, bs_max_rm_in_place], ragged [bs_rm_straddle, bs_tile_pad_col]
  in_place       False [min], True [bs_max_rm_in_place, bs_rm_straddle]
"""

import pytest
import torch
import ttnn

from eval.golden_tests.groupnorm_sc_N_1_HW_C_b.feature_spec import INPUTS
from eval.golden_tests.groupnorm_sc_N_1_HW_C_b.helpers import (
    _TORCH_DTYPE,
    TOLERANCES,
    block_shard_memory_config,
    check_output,
    check_placement,
    create_ttnn_input_tensor,
    pytorch_groupnorm_sc_N_1_HW_C,
)
from tests.ttnn.unit_tests.operations.groupnorm_sc_N_1_HW_C_b.pre_migration.test_acceptance import (  # noqa: F401
    groupnorm_sc_N_1_HW_C_b_op,
)

BF16, F32, BF8 = ttnn.bfloat16, ttnn.float32, ttnn.bfloat8_b
TILE, RM = ttnn.TILE_LAYOUT, ttnn.ROW_MAJOR_LAYOUT
NONE = "none"


def _numel(entry):
    n = 1
    for d in entry[0]:
        n *= d
    return n


_DRAM_INPUTS = [e for e in INPUTS if len(e) == 2]
_BS_INPUTS = [e for e in INPUTS if len(e) == 3]
MIN_INPUT = min(INPUTS, key=_numel)
MAX_DRAM_INPUT = max(_DRAM_INPUTS, key=_numel)
MAX_BS_INPUT = max(_BS_INPUTS, key=_numel)

# (inputs, dtype, layout, affine, affine_dtype, affine_layout, in_place)
CUT = [
    pytest.param(MIN_INPUT, BF16, TILE, "gamma_beta", BF16, RM, False, id="min_1x1x32x32"),
    pytest.param(MAX_DRAM_INPUT, BF16, TILE, "gamma_beta", BF16, RM, False, id="dram_max_4x6360x768"),
    pytest.param(((1, 1, 16384, 1152), 32), BF16, RM, "gamma_only", F32, TILE, False, id="refiner_16384x1152"),
    pytest.param(((1, 1, 64, 50), 1), F32, RM, "no_affine", NONE, NONE, False, id="fp32_rm"),
    pytest.param(((1, 1, 47, 256), 1), F32, TILE, "gamma_beta", F32, RM, False, id="fp32_tile_hw"),
    pytest.param(((1, 1, 64, 200), 8), BF8, TILE, "gamma_beta", BF8, TILE, False, id="bf8b_c"),
    pytest.param(((1, 1, 1024, 1280), 32, (8, 8)), BF16, TILE, "gamma_beta", BF16, RM, False, id="bs_tile_even"),
    pytest.param(MAX_BS_INPUT, BF16, RM, "gamma_beta", BF16, RM, True, id="bs_max_rm_in_place_16384x384"),
    pytest.param(((2, 1, 256, 1280), 32, (8, 3)), BF16, RM, "gamma_only", BF16, RM, True, id="bs_rm_straddle"),
    pytest.param(((1, 1, 4096, 960), 32, (8, 8)), BF16, TILE, "no_affine", NONE, NONE, False, id="bs_tile_pad_col"),
]


def test_cut_uses_the_golden_extremes():
    """The cut's min / max entries are what the golden INPUTS hold today (a changed INPUTS list shows up here)."""
    assert MIN_INPUT == ((1, 1, 32, 32), 1)
    assert MAX_DRAM_INPUT == ((4, 1, 6360, 768), 32)
    assert MAX_BS_INPUT == ((1, 1, 16384, 384), 32, (8, 8))
    for param in CUT:
        assert param.values[0] in INPUTS, f"{param.id}: not a golden INPUTS entry"


@pytest.mark.parametrize("inputs, dtype, layout, affine, affine_dtype, affine_layout, in_place", CUT)
def test_golden_cut(
    device, groupnorm_sc_N_1_HW_C_b_op, inputs, dtype, layout, affine, affine_dtype, affine_layout, in_place
):
    op = groupnorm_sc_N_1_HW_C_b_op
    shape, num_groups = inputs[0], inputs[1]
    grid = inputs[2] if len(inputs) > 2 else None
    C = shape[-1]
    eps = 1e-5
    torch.manual_seed(0)
    torch_input = torch.randn(shape, dtype=torch.float32).to(_TORCH_DTYPE[dtype])
    torch_gamma = torch_beta = None
    if affine in ("gamma_beta", "gamma_only"):
        torch_gamma = torch.randn(1, 1, 1, C, dtype=torch.float32).to(_TORCH_DTYPE[affine_dtype])
    if affine == "gamma_beta":
        torch_beta = torch.randn(1, 1, 1, C, dtype=torch.float32).to(_TORCH_DTYPE[affine_dtype])
    expected = pytorch_groupnorm_sc_N_1_HW_C(torch_input, num_groups, gamma=torch_gamma, beta=torch_beta, eps=eps)

    input_cfg = (
        block_shard_memory_config(shape, grid, layout=layout, dtype=dtype, device=device)
        if grid is not None
        else ttnn.DRAM_MEMORY_CONFIG
    )
    kwargs = {}
    if torch_gamma is not None:
        kwargs["gamma"] = create_ttnn_input_tensor(torch_gamma, device, dtype=affine_dtype, layout=affine_layout)
    if torch_beta is not None:
        kwargs["beta"] = create_ttnn_input_tensor(torch_beta, device, dtype=affine_dtype, layout=affine_layout)
    if in_place:
        kwargs["in_place"] = True
    tt_input = create_ttnn_input_tensor(torch_input, device, dtype=dtype, layout=layout, memory_config=input_cfg)
    input_address = tt_input.buffer_address()

    out = op(tt_input, num_groups, **kwargs)

    assert op.last_dispatches == (1 if op.mode == "source" else 0)
    tolerance = TOLERANCES[dtype]
    check_placement(input_cfg, input_address, out, sharded=grid is not None, in_place=in_place, tolerance=tolerance)
    if not in_place:
        assert out.buffer_address() != input_address
    check_output(
        out,
        expected,
        shape=list(shape),
        dtype=dtype,
        expected_layout=layout if grid is not None else TILE,
        tolerance=tolerance,
    )
