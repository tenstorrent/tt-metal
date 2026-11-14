# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

# Merge-gate coverage for the scatter codegen kernels, selected by `-k codegen` from
# tests/pipeline_reorg/ttnn_merge_gate_tests.yaml: one forced-codegen-vs-native case per program
# factory, the same cases replayed through the program cache with fresh buffers, and a core grid of
# two disjoint ranges. The routing matrix and the slower edge cases live under tests/ttnn/nightly.

import pytest
import torch

import ttnn
from tests.ttnn.utils_for_testing import assert_equal

_force_native = ttnn._ttnn.operations.data_movement.scatter_force_native
_force_codegen = ttnn._ttnn.operations.data_movement.scatter_force_codegen

_TWO_CORES = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(1, 0))])
_TWO_RANGES = ttnn.CoreRangeSet(
    [
        ttnn.CoreRange(ttnn.CoreCoord(1, 0), ttnn.CoreCoord(3, 6)),
        ttnn.CoreRange(ttnn.CoreCoord(5, 0), ttnn.CoreCoord(6, 6)),
    ]
)

# One case per program factory, all on the last axis. The TILE pair shares a shape whose 400 B stick
# is not 32 B aligned, which keeps it off the row-major detour; on the default grid more columns than
# the two tile rows selects the column-split (streaming) factory, while a two-core grid keeps the
# row resident (interleaved). A reduce draws positions from 8 columns so every hit column takes
# several updates; adding a few bf16 values is exact in FP32, so the duplicates have one answer
# whatever the visit order (multiply, where rounding order shows, is compared against native in
# nightly).
_FACTORY_CASES = [
    pytest.param(
        [1, 1, 64, 200], [1, 1, 64, 100], ttnn.TILE_LAYOUT, {"sub_core_grids": _TWO_CORES}, None, id="tile_interleaved"
    ),
    pytest.param([1, 1, 64, 200], [1, 1, 64, 100], ttnn.TILE_LAYOUT, {}, None, id="tile_streaming"),
    pytest.param([1, 1, 32, 64], [1, 1, 32, 32], ttnn.ROW_MAJOR_LAYOUT, {}, None, id="row_major"),
    pytest.param(
        [1, 1, 32, 64], [1, 1, 32, 32], ttnn.ROW_MAJOR_LAYOUT, {"reduce": "add"}, 8, id="row_major_bf16_reduce"
    ),
]


def _case(device, shape, index_shape, layout, index_max):
    axis_len = shape[-1]
    if index_max is None:
        # Unique per row: a duplicate's winner is whichever update lands last, which no route promises.
        index = torch.argsort(torch.rand(index_shape[:-1] + [axis_len]), dim=-1)[..., : index_shape[-1]]
    else:
        index = torch.randint(0, index_max, index_shape)
    xt = ttnn.from_torch(torch.rand(shape, dtype=torch.bfloat16), dtype=ttnn.bfloat16, layout=layout, device=device)
    it = ttnn.from_torch(index.to(torch.int32).contiguous(), dtype=ttnn.int32, layout=layout, device=device)
    st = ttnn.from_torch(
        torch.rand(index_shape, dtype=torch.bfloat16), dtype=ttnn.bfloat16, layout=layout, device=device
    )
    return xt, it, st


@pytest.mark.parametrize("shape, index_shape, layout, kwargs, index_max", _FACTORY_CASES)
def test_scatter_codegen(device, shape, index_shape, layout, kwargs, index_max):
    torch.manual_seed(42)
    xt, it, st = _case(device, shape, index_shape, layout, index_max)

    golden = _force_native(xt, -1, it, st, **kwargs)
    output = _force_codegen(xt, -1, it, st, **kwargs)

    assert output.shape == golden.shape, f"Output shape {output.shape} does not match native shape {golden.shape}"
    assert_equal(ttnn.to_torch(golden), ttnn.to_torch(output))


@pytest.mark.parametrize("shape, index_shape, layout, kwargs, index_max", _FACTORY_CASES)
def test_pc_scatter_codegen(device, shape, index_shape, layout, kwargs, index_max):
    torch.manual_seed(42)
    num_iters = 3
    # A distinct allocation per iteration: the cached program carries the buffer addresses as common
    # runtime args and has to rebind them rather than replay the first dispatch's.
    tensors = [_case(device, shape, index_shape, layout, index_max) for _ in range(num_iters)]
    goldens = [ttnn.to_torch(_force_native(xt, -1, it, st, **kwargs)) for xt, it, st in tensors]

    for i in range(num_iters):
        xt, it, st = tensors[i]
        with device.cache_entries_counter.measure():
            output = _force_codegen(xt, -1, it, st, **kwargs)

        assert_equal(goldens[i], ttnn.to_torch(output))
        if i == 0:
            base_count = device.cache_entries_counter.total
        else:
            assert device.cache_entries_counter.total == base_count, "program cache entries differ on same configs"


@pytest.mark.parametrize(
    "layout, kwargs, index_max",
    [
        pytest.param(ttnn.TILE_LAYOUT, {}, None, id="tile"),
        pytest.param(ttnn.ROW_MAJOR_LAYOUT, {}, None, id="row_major"),
        pytest.param(ttnn.ROW_MAJOR_LAYOUT, {"reduce": "add"}, 8, id="row_major_bf16_reduce"),
    ],
)
def test_scatter_codegen_two_range_grid(device, layout, kwargs, index_max):
    # The factories number cores by their ordinal in the grid and stride by the core count, which
    # only holds if the work split and the kernels walk a two-range grid in the same order.
    torch.manual_seed(42)
    xt, it, st = _case(device, [1, 1, 64, 200], [1, 1, 64, 100], layout, index_max)

    golden = _force_native(xt, -1, it, st, sub_core_grids=_TWO_RANGES, **kwargs)
    output = _force_codegen(xt, -1, it, st, sub_core_grids=_TWO_RANGES, **kwargs)

    assert_equal(ttnn.to_torch(golden), ttnn.to_torch(output))


@pytest.mark.parametrize("shape, index_shape, layout, kwargs, index_max", _FACTORY_CASES)
def test_scatter_codegen_skips_out_of_range_indices(device, shape, index_shape, layout, kwargs, index_max):
    # Every reader skips an index at or past the row's logical width, which a negative int32 also is
    # once read as unsigned; a skip that did not hold would write outside the resident output page.
    # Native skips the same positions, so the two forced legs must agree bit for bit.
    torch.manual_seed(42)
    xt, it, st = _case(device, shape, index_shape, layout, index_max)
    index = ttnn.to_torch(it)
    index[..., 0] = -1
    index[..., 1] = 0x7FFFFFFF
    index[..., 2] = shape[-1]
    it = ttnn.from_torch(index.to(torch.int32).contiguous(), dtype=ttnn.int32, layout=layout, device=device)

    golden = _force_native(xt, -1, it, st, **kwargs)
    output = _force_codegen(xt, -1, it, st, **kwargs)
    assert_equal(ttnn.to_torch(golden), ttnn.to_torch(output))
