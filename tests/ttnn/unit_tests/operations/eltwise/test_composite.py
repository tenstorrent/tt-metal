# SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import ttnn

from tests.ttnn.nightly.unit_tests.operations.eltwise.backward.utility_funcs import (
    compare_pcc,
    data_gen_with_range,
)
from tests.ttnn.utils_for_testing import assert_with_ulp

pytestmark = pytest.mark.use_module_device


@pytest.mark.parametrize(
    "input_shapes",
    (
        (torch.Size([1, 1, 32, 32])),
        (torch.Size([1, 1, 320, 384])),
        (torch.Size([1, 3, 320, 384])),
    ),
)
@pytest.mark.parametrize(
    "min_val, max_val",
    [
        (None, None),
        (-10, None),
        (None, 10),
        (-10, 10),
        (1, -1),
        (0, 0),
        (-1.0, None),
        (None, 1.0),
        (None, None),
        (-0.5, None),
        (None, -0.5),
        (1.0, 0.0),
        (0.0, 1.0),
        ("tensor", None),
        (None, "tensor"),
        ("tensor", "tensor"),
    ],
)
def test_unary_composite_clamp_ttnn(input_shapes, min_val, max_val, device, expect_error):
    in_data1, input_tensor1 = data_gen_with_range(input_shapes, -100, 100, device)

    if min_val == "tensor":
        min, min_tensor = data_gen_with_range(input_shapes, -10, 10, device)
    elif min_val is None:
        min, min_tensor = None, None
    else:
        min, min_tensor = min_val, min_val

    if max_val == "tensor":
        max, max_tensor = data_gen_with_range(input_shapes, -10, 10, device)
    elif max_val is None:
        max, max_tensor = None, None
    else:
        max, max_tensor = max_val, max_val

    if min is None and max is None:
        with expect_error(RuntimeError, "Only one of 'min' or 'max' can be None. Please provide one value"):
            ttnn.clamp(input_tensor1, min_tensor, max_tensor)
    else:
        output_tensor = ttnn.clamp(input_tensor1, min_tensor, max_tensor)
        golden_function = ttnn.get_golden_function(ttnn.clamp)
        golden_tensor = golden_function(in_data1, min, max)
        comp_pass = compare_pcc([output_tensor], [golden_tensor])
        assert comp_pass


@pytest.mark.parametrize(
    "min_val, max_val",
    [
        (-10, None),
        (None, 10),
        (-10, 10),
    ],
)
def test_clamp_tensor_bounds_output_tensor(min_val, max_val, device):
    # Regression test for issue #55334: the tensor-bounds overload of ttnn.clamp
    # ignored the output_tensor argument and silently left the caller-supplied
    # buffer unwritten.
    input_shape = torch.Size([1, 1, 32, 32])
    in_data, input_tensor = data_gen_with_range(input_shape, -100, 100, device)

    min_tensor = (
        ttnn.full(input_shape, min_val, device=device, layout=ttnn.TILE_LAYOUT) if min_val is not None else None
    )
    max_tensor = (
        ttnn.full(input_shape, max_val, device=device, layout=ttnn.TILE_LAYOUT) if max_val is not None else None
    )

    preallocated_output = ttnn.zeros(input_shape, device=device, layout=ttnn.TILE_LAYOUT)
    result = ttnn.clamp(input_tensor, min_tensor, max_tensor, output_tensor=preallocated_output)

    golden_function = ttnn.get_golden_function(ttnn.clamp)
    golden_tensor = golden_function(in_data, min_val, max_val)

    # The op's return value must match the golden result...
    assert compare_pcc([result], [golden_tensor])
    # ...and the preallocated output_tensor buffer must have actually been written.
    assert compare_pcc([preallocated_output], [golden_tensor])


@pytest.mark.parametrize(
    "input_shapes",
    (
        (torch.Size([1, 1, 32, 32])),
        (torch.Size([1, 1, 320, 384])),
        (torch.Size([1, 3, 320, 384])),
    ),
)
@pytest.mark.parametrize(
    "min_val, max_val",
    [
        (None, None),
        (-10, None),
        (None, 10),
        (-10, 10),
        (1, -1),
        (0, 0),
        (-1.0, None),
        (None, 1.0),
        (None, None),
        (-0.5, None),
        (None, -0.5),
        (1.0, 0.0),
        (0.0, 1.0),
        ("tensor", None),
        (None, "tensor"),
        ("tensor", "tensor"),
    ],
)
def test_unary_composite_clip_ttnn(input_shapes, min_val, max_val, device, expect_error):
    in_data1, input_tensor1 = data_gen_with_range(input_shapes, -100, 100, device)

    if min_val == "tensor":
        min, min_tensor = data_gen_with_range(input_shapes, -10, 10, device)
    elif min_val is None:
        min, min_tensor = None, None
    else:
        min, min_tensor = min_val, min_val

    if max_val == "tensor":
        max, max_tensor = data_gen_with_range(input_shapes, -10, 10, device)
    elif max_val is None:
        max, max_tensor = None, None
    else:
        max, max_tensor = max_val, max_val

    if min is None and max is None:
        with expect_error(RuntimeError, "Only one of 'min' or 'max' can be None. Please provide one value"):
            ttnn.clip(input_tensor1, min_tensor, max_tensor)
    else:
        output_tensor = ttnn.clip(input_tensor1, min_tensor, max_tensor)
        golden_function = ttnn.get_golden_function(ttnn.clip)
        golden_tensor = golden_function(in_data1, min, max)
        comp_pass = compare_pcc([output_tensor], [golden_tensor])
        assert comp_pass


@pytest.mark.parametrize(
    "input_shapes",
    (torch.Size([1, 1, 89600, 32]),),
)
def test_unary_composite_mish_sharded_ttnn(input_shapes, device):
    in_data = torch.Tensor(size=input_shapes).uniform_(-20, 100).to(torch.bfloat16)
    shard_grid = ttnn.CoreRangeSet(
        {
            ttnn.CoreRange(
                ttnn.CoreCoord(0, 0),
                ttnn.CoreCoord(7, 6),
            ),
        }
    )
    n_cores = 56
    N, C, H, W = in_data.shape
    shard_spec = ttnn.ShardSpec(shard_grid, [N * C * H // n_cores, W], ttnn.ShardOrientation.ROW_MAJOR)
    input_mem_config = ttnn.MemoryConfig(
        ttnn.types.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.types.BufferType.L1, shard_spec
    )
    input_tensor = ttnn.from_torch(
        in_data,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=input_mem_config,
    )

    output_tensor = ttnn.mish(input_tensor)
    golden_function = ttnn.get_golden_function(ttnn.mish)
    golden_tensor = golden_function(in_data)
    comp_pass = compare_pcc([output_tensor], [golden_tensor])
    assert comp_pass


# Locks in accuracy at the lower domain boundary (x ~ 0.5), where the exact-summation
# cutoff (NUM_TERMS) places the Euler-Maclaurin tail closest to its asymptotic limit.
@pytest.mark.parametrize(
    "input_shapes",
    (
        (torch.Size([1, 1, 32, 32])),
        (torch.Size([1, 1, 320, 384])),
    ),
)
@pytest.mark.parametrize("k", [1, 2, 5, 10])
def test_unary_polygamma_boundary_ttnn(input_shapes, k, device):
    torch.manual_seed(213919)
    # Sample the boundary region x in [0.5, 1.0].
    torch_input = torch.rand(input_shapes, dtype=torch.bfloat16) * 0.5 + 0.5
    golden_function = ttnn.get_golden_function(ttnn.polygamma)
    golden_tensor = golden_function(torch_input, k)

    input_tensor = ttnn.from_torch(torch_input, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    output_tensor = ttnn.polygamma(input_tensor, k)
    output_tensor = ttnn.to_torch(output_tensor)

    assert_with_ulp(expected_result=golden_tensor, actual_result=output_tensor, ulp_threshold=2)


@pytest.mark.parametrize(
    "input_shapes",
    (
        (torch.Size([1, 1, 32, 32])),
        (torch.Size([1, 1, 320, 384])),
        (torch.Size([1, 3, 320, 384])),
    ),
)
@pytest.mark.parametrize(
    "min_val, max_val",
    [
        (None, None),
        (-10, None),
        (None, 10),
        (-10, 10),
        (1, -1),
        (0, 0),
        (-1, None),
        (None, 1),
        (None, None),
        (0, None),
        (None, 0),
        (1, 0),
        (0, 1),
    ],
)
def test_unary_composite_clamp_int_ttnn(input_shapes, min_val, max_val, device, expect_error):
    in_data1 = torch.randint(-100, 100, input_shapes, dtype=torch.int32)
    input_tensor1 = ttnn.from_torch(in_data1, dtype=ttnn.int32, layout=ttnn.TILE_LAYOUT, device=device)
    if min_val is None:
        min, min_tensor = None, None
    else:
        min, min_tensor = min_val, min_val

    if max_val is None:
        max, max_tensor = None, None
    else:
        max, max_tensor = max_val, max_val

    if min is None and max is None:
        with expect_error(RuntimeError, "Only one of 'min' or 'max' can be None. Please provide one value"):
            ttnn.clamp(input_tensor1, min_tensor, max_tensor)
    else:
        output_tensor = ttnn.clamp(input_tensor1, min_tensor, max_tensor)
        golden_function = ttnn.get_golden_function(ttnn.clamp)
        golden_tensor = golden_function(in_data1, min, max)
        comp_pass = compare_pcc([output_tensor], [golden_tensor])
        assert comp_pass


@pytest.mark.parametrize(
    "input_shapes",
    (
        (torch.Size([1, 1, 32, 32])),
        (torch.Size([1, 1, 320, 384])),
        (torch.Size([1, 1, 30, 50])),
    ),
)
@pytest.mark.parametrize(
    "torch_dtype, ttnn_dtype",
    ((torch.float32, ttnn.float32), (torch.bfloat16, ttnn.bfloat16)),
    ids=("float32", "bfloat16"),
)
@pytest.mark.parametrize("layout", (ttnn.TILE_LAYOUT, ttnn.ROW_MAJOR_LAYOUT), ids=("tile", "row_major"))
@pytest.mark.parametrize("fill", (float("-inf"), float("inf")), ids=("neg_inf", "pos_inf"))
@pytest.mark.parametrize("diagonal", (0, 1, -1))
@pytest.mark.parametrize("op", ("tril", "triu"))
def test_unary_composite_trilu_non_finite_ttnn(
    input_shapes, torch_dtype, ttnn_dtype, layout, fill, diagonal, op, device
):
    # The masked-out triangle is defined as zero. For float32 the old multiply ran on the SFPU
    # with fp32 dest, where inf * 0 is NaN, so the masked triangle came back NaN. bfloat16 is
    # kept as a regression guard: its SFPU multiply already forces x * 0 = 0, so it passed
    # before the fix and must keep passing.
    #
    # The other tril/triu tests never feed a non-finite value: test_unary_category6_bfloat16.py
    # samples bf16 without special values, and test_tril_triu_integer_dtype.py draws its float
    # cases from [-4, 4) and its integer cases from arange. Both use tiled inputs; the
    # row-major case here also runs the tilize in front of the float32 select.
    in_data = torch.full(input_shapes, fill, dtype=torch_dtype)
    input_tensor = ttnn.from_torch(in_data, dtype=ttnn_dtype, layout=layout, device=device)

    output = getattr(ttnn, op)(input_tensor, diagonal=diagonal)
    assert output.dtype == ttnn_dtype, f"{op} on {ttnn_dtype} returned {output.dtype}"
    # multiply tilizes a row-major input and returns TILE, and the float32 select does the same.
    assert output.layout == ttnn.TILE_LAYOUT, f"{op} on a {layout} input returned {output.layout}"

    output_tensor = ttnn.to_torch(output)
    # torch directly, with diagonal positional: the generic golden wrapper drops keyword args.
    golden_tensor = getattr(torch, op)(in_data, diagonal)

    n_nan = int(torch.isnan(output_tensor).sum())
    assert n_nan == 0, f"{op}({fill}) [{torch_dtype}] returned {n_nan} NaN of {output_tensor.numel()}"
    assert torch.equal(output_tensor, golden_tensor)
