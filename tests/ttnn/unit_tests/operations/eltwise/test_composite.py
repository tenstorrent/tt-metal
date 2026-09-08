# SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import ttnn

from tests.ttnn.nightly.unit_tests.operations.eltwise.backward.utility_funcs import (
    compare_pcc,
    data_gen_with_range,
)
from tests.ttnn.utils_for_testing import assert_with_ulp, generate_all_bfloat16_bitpatterns

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
    (
        (torch.Size([1, 1, 32, 32])),
        (torch.Size([1, 1, 320, 384])),
        (torch.Size([1, 3, 320, 384])),
    ),
)
def test_unary_composite_mish_ttnn(input_shapes, device):
    in_data1 = torch.Tensor(size=input_shapes).uniform_(-20, 100).to(torch.bfloat16)
    input_tensor1 = ttnn.from_torch(in_data1, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    output_tensor = ttnn.mish(input_tensor1)
    golden_function = ttnn.get_golden_function(ttnn.mish)
    golden_tensor = golden_function(in_data1)

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


@pytest.mark.parametrize(
    "input_shapes",
    (
        (torch.Size([1, 1, 32, 32])),
        (torch.Size([1, 1, 320, 384])),
        (torch.Size([1, 3, 320, 384])),
    ),
)
@pytest.mark.parametrize("k", [1, 5])
def test_unary_polygamma_ttnn(input_shapes, k, device):
    torch.manual_seed(213919)
    torch_input = torch.rand(input_shapes, dtype=torch.bfloat16) * 9.0 + 1.0
    golden_function = ttnn.get_golden_function(ttnn.polygamma)
    golden_tensor = golden_function(torch_input, k)

    input_tensor = ttnn.from_torch(torch_input, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    output_tensor = ttnn.polygamma(input_tensor, k)
    output_tensor = ttnn.to_torch(output_tensor)

    assert_with_ulp(golden_tensor, output_tensor, ulp_threshold=1)


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

    assert_with_ulp(golden_tensor, output_tensor, ulp_threshold=2)


# Every bfloat16 in [0.5, bf16 max] for every supported order. Before the (-1)^(n+1) * n! scale
# was folded into the accumulated terms, the raw Hurwitz sum sat n! below the result and flushed
# to zero (Tensix has no subnormals) while psi^(n)(x) was still a normal fp32: every n >= 2 had a
# band of large x returning exactly 0 (n = 11: x in [2256, 11072]).
@pytest.mark.parametrize("k", list(range(1, 12)))
@pytest.mark.parametrize("dtype", [ttnn.bfloat16, ttnn.float32])
def test_unary_polygamma_large_x_all_bitpatterns(k, dtype, device):
    torch_dtype = torch.bfloat16 if dtype == ttnn.bfloat16 else torch.float32
    x = generate_all_bfloat16_bitpatterns()
    # Only x >= 0.5 is supported; park everything else on an in-domain value.
    x = torch.where((x >= 0.5) & torch.isfinite(x), x, torch.ones_like(x)).to(torch_dtype)
    golden64 = torch.special.polygamma(k, x.to(torch.float64))

    input_tensor = ttnn.from_torch(x, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)
    output_tensor = ttnn.to_torch(ttnn.polygamma(input_tensor, k)).to(torch_dtype)

    # The result is a normal fp32 (with margin for the last exact term): it must not flush to 0.
    normal = golden64.abs() >= 2.0**-125
    zeros = (output_tensor == 0) & normal
    assert not zeros.any(), f"{int(zeros.sum())} lanes return 0 for a normal result, first x={x[zeros][0]}"

    # ULP check above the flush region, where the six exact terms are all still representable.
    keep = golden64.abs() >= 2.0**-100
    golden = torch.where(keep, golden64.to(torch_dtype), torch.zeros_like(x))
    output_tensor = torch.where(keep, output_tensor, torch.zeros_like(x))
    assert_with_ulp(golden, output_tensor, ulp_threshold=2 if dtype == ttnn.bfloat16 else 32)


@pytest.mark.parametrize(
    "input_shapes",
    (
        (torch.Size([1, 1, 32, 32])),
        (torch.Size([1, 1, 320, 384])),
        (torch.Size([1, 3, 320, 384])),
    ),
)
def test_unary_composite_tril_ttnn(input_shapes, device):
    in_data1, input_tensor1 = data_gen_with_range(input_shapes, -100, 100, device)

    output_tensor = ttnn.tril(input_tensor1)
    golden_function = ttnn.get_golden_function(ttnn.tril)
    golden_tensor = golden_function(in_data1)

    comp_pass = compare_pcc([output_tensor], [golden_tensor])
    assert comp_pass


@pytest.mark.parametrize(
    "input_shapes",
    (
        (torch.Size([1, 1, 32, 32])),
        (torch.Size([1, 1, 320, 384])),
        (torch.Size([1, 3, 320, 384])),
    ),
)
def test_unary_composite_triu_ttnn(input_shapes, device):
    in_data1, input_tensor1 = data_gen_with_range(input_shapes, -100, 100, device)

    output_tensor = ttnn.triu(input_tensor1)
    golden_function = ttnn.get_golden_function(ttnn.triu)
    golden_tensor = golden_function(in_data1)

    comp_pass = compare_pcc([output_tensor], [golden_tensor])
    assert comp_pass


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
