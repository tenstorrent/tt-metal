# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

# Hand-written regressions for the scatter routing seams the generated sweep in
# test_scatter_codegen_routing.py does not reach: the demotion predicate's axis spelling, an output
# placement that differs from the input's, and empty operands.

import pytest
import torch

import ttnn
from tests.ttnn.utils_for_testing import assert_equal

_force_native = ttnn._ttnn.operations.data_movement.scatter_force_native
_force_codegen = ttnn._ttnn.operations.data_movement.scatter_force_codegen


def _make_case(
    shape, index_shape, dim, layout, device, input_memory_config=None, index_dtype=ttnn.int32, index_max=None
):
    axis = dim if dim >= 0 else len(shape) + dim
    x = torch.rand(shape, dtype=torch.bfloat16)
    # torch.randint needs a positive range even when the index tensor itself is empty.
    high = max(1, int(shape[axis]) if index_max is None else int(index_max))
    index = torch.randint(0, high, index_shape, dtype=torch.int32)
    src = torch.rand(index_shape, dtype=torch.bfloat16)
    xt = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=layout, device=device, memory_config=input_memory_config)
    it = ttnn.from_torch(index, dtype=index_dtype, layout=layout, device=device)
    st = ttnn.from_torch(src, dtype=ttnn.bfloat16, layout=layout, device=device)
    return xt, it, st


def _run_auto_against_native(device, xt, it, st, dim, **kwargs):
    """Returns (auto output, program-cache entries added by the auto route beyond what native cached)."""
    device.clear_program_cache()
    golden = ttnn.to_torch(_force_native(xt, dim, it, st, **kwargs))
    entries_before = device.num_program_cache_entries()
    out = ttnn.scatter(xt, dim, it, st, **kwargs)
    assert_equal(golden, ttnn.to_torch(out))
    return out, device.num_program_cache_entries() - entries_before


# The exact-match carve-outs in is_demoted() are measured on the pre-last axis of a rank-4 tensor,
# which a caller may spell as either 2 or -2.
_CARVE_OUTS = [
    ([1, 1, 32, 64], [1, 1, 16, 64], ttnn.ROW_MAJOR_LAYOUT),
    ([1, 1, 32, 64], [1, 1, 16, 64], ttnn.TILE_LAYOUT),
    ([1, 1, 64, 128], [1, 1, 32, 128], ttnn.ROW_MAJOR_LAYOUT),
]


@pytest.mark.parametrize("dim", [2, -2])
@pytest.mark.parametrize("shape,index_shape,layout", _CARVE_OUTS, ids=lambda v: str(v) if isinstance(v, list) else None)
def test_scatter_codegen_demotion_matches_either_axis_spelling(device, shape, index_shape, layout, dim):
    xt, it, st = _make_case(shape, index_shape, dim, layout, device)
    _, added = _run_auto_against_native(device, xt, it, st, dim)
    assert added == 0, f"dim={dim} routed a perf-demoted case to codegen (program cache grew); expected native"


def test_scatter_codegen_tile_dram_input_l1_output_runs_tile_factory(device):
    # Ht == 1 qualifies for the row-major detour on shape alone, but the detour's ROW_MAJOR factories
    # cannot write an output whose buffer type differs from the input's; the call must take the TILE
    # factory instead of being rerouted into a validate failure.
    xt, it, st = _make_case([1, 1, 32, 64], [1, 1, 32, 32], -1, ttnn.TILE_LAYOUT, device)
    assert xt.memory_config().buffer_type == ttnn.BufferType.DRAM
    out, added = _run_auto_against_native(device, xt, it, st, -1, memory_config=ttnn.L1_MEMORY_CONFIG)
    assert out.memory_config().buffer_type == ttnn.BufferType.L1
    assert out.layout == ttnn.TILE_LAYOUT
    assert added > 0, "auto fell back to native; expected the codegen TILE factory to serve this call"


@pytest.mark.parametrize(
    "output_memory_config,expect_codegen",
    [(None, True), (ttnn.L1_MEMORY_CONFIG, False)],
    ids=["same_placement_detour", "mismatched_placement_demoted"],
)
def test_scatter_codegen_unit_row_tile_demotion_follows_detour(device, output_memory_config, expect_codegen):
    # A unit TILE row is only kept on codegen when the row-major detour serves it. With the output in
    # the input's own placement the detour runs; with a different placement the detour is unavailable
    # and the case is demoted to native rather than paying the TILE factories' padded-row cost.
    xt, it, st = _make_case([1, 1, 1, 64], [1, 1, 1, 32], -1, ttnn.TILE_LAYOUT, device)
    kwargs = {} if output_memory_config is None else {"memory_config": output_memory_config}
    _, added = _run_auto_against_native(device, xt, it, st, -1, **kwargs)
    if expect_codegen:
        assert added > 0, "auto fell back to native; expected the codegen row-major detour to serve this call"
    else:
        assert added == 0, "auto routed a unit TILE row the detour cannot serve to codegen; expected native"


def _error_of(fn):
    # The first line is the TT_FATAL/TT_THROW text; the rest is a backtrace that differs per call.
    try:
        fn()
    except RuntimeError as e:
        return str(e).splitlines()[0]
    return None


@pytest.mark.parametrize("layout", [ttnn.ROW_MAJOR_LAYOUT, ttnn.TILE_LAYOUT], ids=["row_major", "tile"])
@pytest.mark.parametrize(
    "shape,index_shape", [([0, 32], [0, 16]), ([2, 0, 32], [2, 0, 16])], ids=["[0,32]", "[2,0,32]"]
)
def test_scatter_codegen_empty_operands_stay_native(device, shape, index_shape, layout, expect_error):
    # A zero extent is a divisor in the codegen host arithmetic (CB page size, page map), which dies
    # with SIGFPE rather than an exception, so empties must never reach the codegen factories: auto
    # must behave exactly as native does on them (today native raises on a zero leading extent; a
    # zero scatter axis, e.g. [0] or [32, 0], crashes native itself and is not probed here), and the
    # forced entry refuses.
    xt, it, st = _make_case(shape, index_shape, -1, layout, device)
    device.clear_program_cache()
    native_error = _error_of(lambda: _force_native(xt, -1, it, st))
    entries_before = device.num_program_cache_entries()
    auto_error = _error_of(lambda: ttnn.scatter(xt, -1, it, st))
    assert auto_error == native_error, "auto diverged from native on an empty operand"
    assert device.num_program_cache_entries() == entries_before, "auto routed an empty tensor to codegen"
    with expect_error(RuntimeError, "scatter_force_codegen invoked for a case the codegen path does not support"):
        _force_codegen(xt, -1, it, st)
