# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import pytest
import ttnn
from tests.ttnn.unit_tests.operations.eltwise.backward.utility_funcs import (
    data_gen_with_range,
    data_gen_with_range_dtype,
    compare_pcc,
)
from tests.ttnn.utils_for_testing import assert_with_pcc


@pytest.mark.parametrize(
    "input_shapes",
    (
        (torch.Size([1, 1, 32, 32])),
        (torch.Size([1, 1, 320, 384])),
        (torch.Size([1, 3, 320, 384])),
    ),
)
def test_unary_composite_cbrt_ttnn(input_shapes, device):
    in_data1, input_tensor1 = data_gen_with_range(input_shapes, -100, 100, device)

    output_tensor = ttnn.cbrt(input_tensor1)
    golden_function = ttnn.get_golden_function(ttnn.cbrt)
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
def test_unary_composite_clamp_ttnn(input_shapes, min_val, max_val, device):
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
        with pytest.raises(RuntimeError, match="Only one of 'min' or 'max' can be None. Please provide one value"):
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
def test_unary_composite_clip_ttnn(input_shapes, min_val, max_val, device):
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
        with pytest.raises(RuntimeError, match="Only one of 'min' or 'max' can be None. Please provide one value"):
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
def test_unary_composite_cosh_ttnn(input_shapes, device):
    in_data1, input_tensor1 = data_gen_with_range(input_shapes, -9, 9, device)

    output_tensor = ttnn.cosh(input_tensor1)
    golden_function = ttnn.get_golden_function(ttnn.cosh)
    golden_tensor = golden_function(in_data1)

    comp_pass = compare_pcc([output_tensor], [golden_tensor])
    assert comp_pass


@pytest.mark.parametrize(
    "input_shapes",
    [
        [32, 32],
        [5, 96, 64],
        [1, 3, 320, 384],
        [1, 2, 4, 32, 32],
    ],
)
def test_unary_composite_hardsigmoid_bf8b(input_shapes, device):
    in_data1, input_tensor1 = data_gen_with_range_dtype(input_shapes, -100, 100, device, False, False, ttnn.bfloat8_b)

    output_tensor = ttnn.hardsigmoid(input_tensor1)
    golden_function = ttnn.get_golden_function(ttnn.hardsigmoid)
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
def test_unary_composite_digamma_ttnn(input_shapes, device):
    in_data1, input_tensor1 = data_gen_with_range(input_shapes, 1, 100, device)

    output_tensor = ttnn.digamma(input_tensor1)
    golden_function = ttnn.get_golden_function(ttnn.digamma)
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
def test_unary_composite_hardsigmoid_ttnn(input_shapes, device):
    in_data1, input_tensor1 = data_gen_with_range(input_shapes, -100, 100, device)

    output_tensor = ttnn.hardsigmoid(input_tensor1)
    golden_function = ttnn.get_golden_function(ttnn.hardsigmoid)
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
def test_unary_composite_hardswish_ttnn(input_shapes, device):
    in_data1, input_tensor1 = data_gen_with_range(input_shapes, -100, 100, device)

    output_tensor = ttnn.hardswish(input_tensor1)
    golden_function = ttnn.get_golden_function(ttnn.hardswish)
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
def test_unary_composite_hardtanh_ttnn(input_shapes, device):
    in_data1, input_tensor1 = data_gen_with_range(input_shapes, -100, 100, device)

    output_tensor = ttnn.hardtanh(input_tensor1)
    golden_function = ttnn.get_golden_function(ttnn.hardtanh)
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
def test_unary_composite_lgamma_ttnn(input_shapes, device):
    in_data1, input_tensor1 = data_gen_with_range(input_shapes, 0.1, 100, device)

    output_tensor = ttnn.lgamma(input_tensor1)
    golden_function = ttnn.get_golden_function(ttnn.lgamma)
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
    (
        torch.Size([1, 1, 89600, 32]),
        torch.Size([1, 1, 89600, 128]),
    ),
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
def test_unary_composite_multigammaln_ttnn(input_shapes, device):
    in_data1, input_tensor1 = data_gen_with_range(input_shapes, 1.6, 100, device)

    output_tensor = ttnn.multigammaln(input_tensor1)
    golden_function = ttnn.get_golden_function(ttnn.multigammaln)
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
def test_unary_composite_polygamma_ttnn(input_shapes, device):
    in_data1, input_tensor1 = data_gen_with_range(input_shapes, 1, 10, device)
    k = 5
    output_tensor = ttnn.polygamma(input_tensor1, k)
    golden_function = ttnn.get_golden_function(ttnn.polygamma)
    golden_tensor = golden_function(in_data1, k)

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
def test_unary_composite_selu_ttnn(input_shapes, device):
    in_data1, input_tensor1 = data_gen_with_range(input_shapes, 1, 2, device)

    output_tensor = ttnn.selu(input_tensor1)
    golden_function = ttnn.get_golden_function(ttnn.selu)
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
def test_unary_composite_sinh_ttnn(input_shapes, device):
    in_data1, input_tensor1 = data_gen_with_range(input_shapes, -9, 9, device)

    output_tensor = ttnn.sinh(input_tensor1)
    golden_function = ttnn.get_golden_function(ttnn.sinh)
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
def test_unary_composite_softsign_ttnn(input_shapes, device):
    in_data1, input_tensor1 = data_gen_with_range(input_shapes, -100, 100, device)

    output_tensor = ttnn.softsign(input_tensor1)
    golden_function = ttnn.get_golden_function(ttnn.softsign)
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
def test_unary_composite_swish_ttnn(input_shapes, device):
    in_data1, input_tensor1 = data_gen_with_range(input_shapes, -100, 100, device)

    output_tensor = ttnn.swish(input_tensor1)
    golden_function = ttnn.get_golden_function(ttnn.swish)
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
def test_unary_composite_threshold_ttnn(input_shapes, device):
    in_data1, input_tensor1 = data_gen_with_range(input_shapes, -100, 100, device)
    threshold = 1.0
    value = 10.0
    output_tensor = ttnn.threshold(input_tensor1, threshold, value)
    golden_function = ttnn.get_golden_function(ttnn.threshold)
    golden_tensor = golden_function(in_data1, threshold, value)

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
        (torch.Size([1, 1, 32, 64])),
        (torch.Size([1, 1, 320, 384])),
        (torch.Size([1, 3, 320, 384])),
    ),
)
@pytest.mark.parametrize(
    "dim",
    [-1, 3],
)
def test_unary_glu_ttnn(input_shapes, dim, device):
    in_data, input_tensor = data_gen_with_range(input_shapes, -5, 5, device)
    golden_fn = ttnn.get_golden_function(ttnn.glu)

    output_tensor = ttnn.glu(input_tensor, dim)
    golden_tensor = golden_fn(in_data, dim)

    comp_pass = compare_pcc([output_tensor], [golden_tensor])
    assert comp_pass


@pytest.mark.parametrize(
    "input_shapes",
    (
        (torch.Size([1, 1, 32, 64])),
        (torch.Size([1, 1, 320, 384])),
        (torch.Size([1, 3, 320, 384])),
    ),
)
@pytest.mark.parametrize(
    "dim",
    [-1, 3],
)
def test_unary_reglu_ttnn(input_shapes, dim, device):
    in_data, input_tensor = data_gen_with_range(input_shapes, -5, 5, device)
    golden_fn = ttnn.get_golden_function(ttnn.reglu)

    output_tensor = ttnn.reglu(input_tensor, dim)
    golden_tensor = golden_fn(in_data, dim)

    comp_pass = compare_pcc([output_tensor], [golden_tensor])
    assert comp_pass


@pytest.mark.parametrize(
    "input_shapes",
    (
        (torch.Size([1, 1, 32, 64])),
        (torch.Size([1, 1, 320, 384])),
        (torch.Size([1, 3, 320, 384])),
    ),
)
@pytest.mark.parametrize(
    "dim",
    [-1, 3],
)
def test_unary_geglu_ttnn(input_shapes, dim, device):
    in_data, input_tensor = data_gen_with_range(input_shapes, -5, 5, device)
    golden_fn = ttnn.get_golden_function(ttnn.geglu)

    output_tensor = ttnn.geglu(input_tensor, dim)
    golden_tensor = golden_fn(in_data, dim)

    comp_pass = compare_pcc([output_tensor], [golden_tensor])
    assert comp_pass


@pytest.mark.parametrize(
    "input_shapes",
    (
        (torch.Size([1, 1, 32, 64])),  # Single core
        (torch.Size([1, 3, 320, 32 * 8])),  # Multi core
    ),
)
@pytest.mark.parametrize(
    "dim",
    [-1, 3],
)
def test_unary_swiglu_ttnn(input_shapes, dim, device):
    in_data, input_tensor = data_gen_with_range(input_shapes, -5, 5, device)
    golden_fn = ttnn.get_golden_function(ttnn.swiglu)

    output_tensor = ttnn.swiglu(input_tensor, dim)
    golden_tensor = golden_fn(in_data, dim)

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
    "param",
    {0.45, 7.7, 36.89, 58.4, 89.9},
)
def test_unary_softshrink(input_shapes, param, device):
    in_data, input_tensor = data_gen_with_range(input_shapes, -100, 100, device)

    output_tensor = ttnn.softshrink(input_tensor, lambd=param)
    golden_function = ttnn.get_golden_function(ttnn.softshrink)
    golden_tensor = golden_function(in_data, lambd=param)

    comp_pass = compare_pcc([output_tensor], [golden_tensor])
    assert comp_pass


@pytest.mark.parametrize(
    "input_shapes",
    (
        (torch.Size([1, 1, 32, 32])),
        (torch.Size([1, 1, 320, 384])),
        (torch.Size([1, 3, 320, 384])),
        (torch.Size([7, 185, 20])),
        (torch.Size([6, 45, 233])),
    ),
)
@pytest.mark.parametrize(
    "param",
    {-1e4, -98.5, -43.7, -8.5, 0.0, 0.45, 1.0, 7.7, 58.4, 89.9, 1e5},
)
def test_unary_logit(input_shapes, param, device):
    in_data = torch.Tensor(size=input_shapes).uniform_(-100, 100).to(torch.bfloat16)
    input_tensor = ttnn.from_torch(in_data, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

    output_tensor = ttnn.logit(input_tensor, eps=param)
    golden_function = ttnn.get_golden_function(ttnn.logit)
    golden_tensor = golden_function(in_data, eps=param, device=device)

    out = ttnn.to_torch(output_tensor)
    assert_with_pcc(golden_tensor, out, 0.99)


@pytest.mark.parametrize(
    "input_shapes",
    (
        (torch.Size([1, 1, 32, 32])),
        (torch.Size([1, 1, 320, 384])),
        (torch.Size([1, 3, 320, 384])),
    ),
)
@pytest.mark.parametrize(
    "param",
    {-98.5, -43.7, -8.5, 0.45, 7.7, 58.4, 89.9},
)
def test_unary_celu(input_shapes, param, device):
    in_data, input_tensor = data_gen_with_range(input_shapes, -100, 100, device)
    output_tensor = ttnn.celu(input_tensor, alpha=param)
    golden_function = ttnn.get_golden_function(ttnn.celu)
    golden_tensor = golden_function(in_data, alpha=param)

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
    "param",
    {-98.5, -43.7, -8.5, 0.45, 7.7, 58.4, 89.9},
)
@pytest.mark.parametrize("round_mode", [None, "trunc", "floor"])
def test_unary_rdiv(input_shapes, param, round_mode, device):
    in_data, input_tensor = data_gen_with_range(input_shapes, -100, 100, device)

    output_tensor = ttnn.rdiv(input_tensor, param, round_mode=round_mode)
    golden_function = ttnn.get_golden_function(ttnn.rdiv)
    golden_tensor = golden_function(in_data, param, round_mode=round_mode)

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
def test_unary_composite_clamp_int_ttnn(input_shapes, min_val, max_val, device):
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
        with pytest.raises(RuntimeError, match="Only one of 'min' or 'max' can be None. Please provide one value"):
            ttnn.clamp(input_tensor1, min_tensor, max_tensor)
    else:
        output_tensor = ttnn.clamp(input_tensor1, min_tensor, max_tensor)
        golden_function = ttnn.get_golden_function(ttnn.clamp)
        golden_tensor = golden_function(in_data1, min, max)
        comp_pass = compare_pcc([output_tensor], [golden_tensor])
        assert comp_pass
