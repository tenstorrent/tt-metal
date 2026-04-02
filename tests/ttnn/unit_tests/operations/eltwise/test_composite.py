# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import pytest
import ttnn
from tests.ttnn.nightly.unit_tests.operations.eltwise.backward.utility_funcs import (
    data_gen_with_range,
    data_gen_with_range_dtype,
    compare_pcc,
    compare_equal,
)
from tests.ttnn.utils_for_testing import assert_with_pcc

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
@pytest.mark.parametrize("k", [1, 5])
def test_unary_composite_polygamma_ttnn(input_shapes, k, device):
    import struct
    import numpy as np
    from loguru import logger

    def _float_to_bf16_bits(f):
        f32_bits = struct.unpack(">I", struct.pack(">f", f))[0]
        return f32_bits >> 16

    def _bf16_daz_normalize(bits):
        exp = (bits >> 7) & 0xFF
        mantissa = bits & 0x7F
        if (exp == 0) and (mantissa != 0):
            return 0x0000
        if bits == 0x8000:
            return 0x0000
        return bits

    def _bf16_value_order_index_daz(bits):
        bits = _bf16_daz_normalize(bits)
        exp = (bits >> 7) & 0xFF
        mantissa = bits & 0x7F
        if exp == 0xFF and mantissa != 0:
            return -1
        if bits == 0x7F80:
            return 65281
        if bits == 0xFF80:
            return -1
        if bits == 0x0000:
            return 32640
        if bits & 0x8000:
            return 0x7F7F - (bits & 0x7FFF)
        return 32640 + bits - 0x007F

    def _ulp_distance_bf16_daz(a, b):
        a_bits = _bf16_daz_normalize(_float_to_bf16_bits(a))
        b_bits = _bf16_daz_normalize(_float_to_bf16_bits(b))
        a_exp = (a_bits >> 7) & 0xFF
        b_exp = (b_bits >> 7) & 0xFF
        if (a_exp == 0xFF and (a_bits & 0x7F) != 0) or (b_exp == 0xFF and (b_bits & 0x7F) != 0):
            return -1
        idx_a = _bf16_value_order_index_daz(a_bits)
        idx_b = _bf16_value_order_index_daz(b_bits)
        if idx_a < 0 or idx_b < 0:
            return -1
        return abs(idx_a - idx_b)

    in_data1, input_tensor1 = data_gen_with_range(input_shapes, 1, 10, device)
    output_tensor = ttnn.polygamma(input_tensor1, k)
    output_torch = ttnn.to_torch(output_tensor)

    # High-precision reference
    ref_f64 = torch.special.polygamma(k, in_data1.to(torch.float64))

    result_flat = output_torch.flatten()
    ref_flat = ref_f64.flatten()

    worst_ulp = 0
    ulp_errors = []
    for i in range(len(result_flat)):
        res_val = result_flat[i].item()
        ref_val = ref_flat[i].item()
        if not np.isfinite(res_val) or not np.isfinite(ref_val):
            continue
        ref_bf16 = torch.tensor(ref_val, dtype=torch.bfloat16).item()
        ulp = _ulp_distance_bf16_daz(res_val, ref_bf16)
        if ulp < 0:
            continue
        ulp_errors.append(ulp)
        if ulp > worst_ulp:
            worst_ulp = ulp

    ulp_arr = np.array(ulp_errors)
    logger.info(
        f"polygamma(n={k}, shape={input_shapes}) ULP — max: {worst_ulp}, "
        f"mean: {np.mean(ulp_arr):.2f}, p99: {np.percentile(ulp_arr, 99):.1f}"
    )

    max_ulp = 1
    assert (
        worst_ulp <= max_ulp
    ), f"polygamma(n={k}, shape={input_shapes}) max ULP {worst_ulp} exceeds threshold {max_ulp}"


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
