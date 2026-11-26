# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Acceptance test for g6_to_memory_config (immutable spec for the implementer).

Production cases only (BB-22), taken from the golden suite's feature_spec.CASES,
at least two per regime of op_design.md (PULL: L1-sharded target; PUSH: L1-sharded
source; STAGE: both ends interleaved), plus the shard-grid forms the spec names
(single-core ranges, idle cores, off-origin core, uneven last shard) and the
host validation the spec requires.

Gate: placement op, graded exactly (helpers.TOLERANCES = ("exact", None)):
torch.equal on the values read back, plus the result's memory configuration,
dtype, layout and shape.
"""

import pytest
import torch
import ttnn

from eval.golden_tests.g6_to_memory_config.feature_spec import CASES
from eval.golden_tests.g6_to_memory_config.helpers import build_memory_config, check_placement
from ttnn.operations.g6_to_memory_config import g6_to_memory_config


def pytorch_reference(x: torch.Tensor) -> torch.Tensor:
    """Identity on the logical values."""
    return x.clone()


def _scenario(case_id, model="Qwen"):
    return CASES[(model, case_id)][0]


def _skip_if_grid_outside(device, memory):
    if "grid" not in memory:
        return
    size = device.dram_grid_size() if memory["buffer"] == "dram" else device.compute_with_storage_grid_size()
    for _, _, ex, ey in memory["grid"]:
        if ex > size.x - 1 or ey > size.y - 1:
            pytest.skip(f"shard grid end ({ex},{ey}) outside device grid ({size.x},{size.y})")


def _make_input(scenario, device):
    _skip_if_grid_outside(device, scenario["in_memory"])
    _skip_if_grid_outside(device, scenario["out_memory"])
    torch.manual_seed(42)
    torch_input = torch.randn(scenario["shape"], dtype=torch.float32).to(torch.bfloat16)
    tt_input = ttnn.from_torch(
        torch_input,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=build_memory_config(scenario["in_memory"]),
    )
    return torch_input, tt_input


def _check(tt_output, torch_input, out_config, tt_input):
    assert tt_output is not tt_input, "the op must return a new tensor"
    assert list(tt_output.shape) == list(torch_input.shape)
    assert tt_output.dtype == ttnn.bfloat16
    assert tt_output.layout == ttnn.TILE_LAYOUT
    check_placement(tt_output, out_config)
    readback = ttnn.to_torch(tt_output)
    expected = pytorch_reference(torch_input)
    assert readback.shape == expected.shape
    assert torch.equal(readback.to(expected.dtype), expected), "values differ from the input"


# (case id, what it pins). Shapes cover single-tile (2154), multi-tile, non-square,
# multi-batch (2078), ranks 3 and 4, tile-aligned and H-non-aligned.
CASES_UNDER_TEST = [
    # PULL: target L1 sharded
    pytest.param(2058, id="pull-dram_int-to-height_16_single_core_ranges-H1"),
    pytest.param(2067, id="pull-width8-to-width40-coalesced_runs"),
    pytest.param(2154, id="pull-dram_int-to-height_one_core_off_origin-single_tile_row"),
    pytest.param(2157, id="pull-height-to-block_same_core"),
    pytest.param(2064, id="pull-l1_int-to-width8"),
    pytest.param(2168, id="pull-rank3-dram_int-to-width40"),
    # PUSH: source L1 sharded, target interleaved
    pytest.param(2061, id="push-height_2_ranges-to-dram_int"),
    pytest.param(2063, id="push-width_uneven_last_shard-to-l1_int"),
    pytest.param(2065, id="push-width_34_shards_on_36_cores-to-l1_int"),
    pytest.param(2071, id="push-width_uneven_52-to-dram_int"),
    pytest.param(2155, id="push-block_one_core-to-l1_int"),
    # STAGE: both ends interleaved
    pytest.param(2078, id="stage-dram_int-to-l1_int-16x128x5120"),
    pytest.param(2108, id="stage-dram_int-to-l1_int-H77_non_aligned"),
    pytest.param(2089, id="stage-dram_int-to-l1_int-rank3-small"),
]


@pytest.mark.parametrize("case_id", CASES_UNDER_TEST)
def test_g6_to_memory_config(device, case_id):
    scenario = _scenario(case_id)
    out_config = build_memory_config(scenario["out_memory"])
    torch_input, tt_input = _make_input(scenario, device)
    tt_output = g6_to_memory_config(tt_input, out_config)
    _check(tt_output, torch_input, out_config, tt_input)


def test_repeat_and_cache_key(device):
    """Same input, two target grids that differ only in the shard spec (2070, 2073),
    then 2070 again: each result carries its own placement and identical values (BB-06, BB-07)."""
    results = []
    for case_id in (2070, 2073, 2070):
        scenario = _scenario(case_id)
        out_config = build_memory_config(scenario["out_memory"])
        torch_input, tt_input = _make_input(scenario, device)
        tt_output = g6_to_memory_config(tt_input, out_config)
        _check(tt_output, torch_input, out_config, tt_input)
        results.append(ttnn.to_torch(tt_output))
    assert torch.equal(results[0], results[2])


# ---------------------------------------------------------------------------
# Host validation required by the spec (ValueError or RuntimeError)
# ---------------------------------------------------------------------------


def _with_shard(memory, **changes):
    memory = dict(memory)
    memory.update(changes)
    return memory


def test_rejects_col_major_target(device, expect_error):
    scenario = _scenario(2068)
    _, tt_input = _make_input(scenario, device)
    with expect_error((ValueError, RuntimeError), ""):
        g6_to_memory_config(tt_input, build_memory_config(_with_shard(scenario["out_memory"], orientation="col_major")))


def test_rejects_shard_shape_not_multiple_of_32(device, expect_error):
    scenario = _scenario(2068)
    _, tt_input = _make_input(scenario, device)
    with expect_error((ValueError, RuntimeError), ""):
        g6_to_memory_config(tt_input, build_memory_config(_with_shard(scenario["out_memory"], shard_shape=[32, 200])))


def test_rejects_grid_with_too_few_cores(device, expect_error):
    # 2068: 48 tile columns over shards of 6 need 8 cores; give 4.
    scenario = _scenario(2068)
    _, tt_input = _make_input(scenario, device)
    with expect_error((ValueError, RuntimeError), ""):
        g6_to_memory_config(tt_input, build_memory_config(_with_shard(scenario["out_memory"], grid=[[0, 0, 3, 0]])))


def test_rejects_row_major_layout_input(device, expect_error):
    torch.manual_seed(42)
    x = torch.randn((1, 1, 32, 256), dtype=torch.float32).to(torch.bfloat16)
    tt_input = ttnn.from_torch(
        x, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    with expect_error((ValueError, RuntimeError), ""):
        g6_to_memory_config(tt_input, ttnn.L1_MEMORY_CONFIG)


def test_rejects_host_tensor(device, expect_error):
    torch.manual_seed(42)
    x = torch.randn((1, 1, 32, 256), dtype=torch.float32).to(torch.bfloat16)
    tt_host = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
    with expect_error((ValueError, RuntimeError), ""):
        g6_to_memory_config(tt_host, ttnn.L1_MEMORY_CONFIG)


def test_rejects_rank_5(device, expect_error):
    torch.manual_seed(42)
    x = torch.randn((1, 1, 1, 32, 256), dtype=torch.float32).to(torch.bfloat16)
    tt_input = ttnn.from_torch(
        x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    with expect_error((ValueError, RuntimeError), ""):
        g6_to_memory_config(tt_input, ttnn.L1_MEMORY_CONFIG)
