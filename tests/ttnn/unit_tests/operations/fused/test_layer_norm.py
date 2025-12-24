# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest

import torch

import ttnn

from tests.ttnn.utils_for_testing import assert_with_pcc, assert_allclose, assert_relative_frobenius
from dataclasses import dataclass

pytestmark = pytest.mark.use_module_device


@dataclass
class AllCloseThresholds:
    rtol: float
    atol: float


allclose_thresholds = {
    # bfloat16 can accumulate a lot of error for fused ops. Rounding
    # error after a single operation will be 0.5 ULP in the worst case,
    # which is 0.5*2^-7=0.00390625 (a little less than 0.5%). Since we're doing
    # potentially thousands of operations in many tests, we'll allow up to 5%.
    torch.bfloat16: AllCloseThresholds(rtol=5e-2, atol=5e-2),
    # Unused for now, see https://github.com/tenstorrent/tt-metal/issues/33621
    # torch.float32: AllCloseThresholds(rtol=1e-5, atol=1e-8)
}


def assert_output_accuracy(torch_output, ttnn_output):
    dtype = ttnn_output.dtype
    if dtype == torch.bfloat16:
        return assert_allclose(
            torch_output, ttnn_output, rtol=allclose_thresholds[dtype].rtol, atol=allclose_thresholds[dtype].atol
        )
    elif dtype == torch.float32:
        # torch.float32 data is not being robustly converted to tt tensors
        # (see https://github.com/tenstorrent/tt-metal/issues/33621).
        # So we'll use relative Frobenius norm of the error instead, which is
        # looser than allclose (since it's a global metric), but better than PCC.
        return assert_relative_frobenius(torch_output, ttnn_output, threshold=0.01)
    else:
        raise ValueError(f"Robust checks are not implemented for dtype: {dtype}")


@pytest.mark.parametrize("h", [32])
@pytest.mark.parametrize("w", [64])
@pytest.mark.parametrize("use_welford", [True, False])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_layer_norm(device, h, w, use_welford, dtype):
    torch.manual_seed(0)

    torch_input_tensor = torch.rand((h, w), dtype=dtype)
    torch_output_tensor = torch.nn.functional.layer_norm(torch_input_tensor, normalized_shape=[w])

    input_tensor = ttnn.from_torch(torch_input_tensor, layout=ttnn.TILE_LAYOUT, device=device)
    program_config = ttnn.LayerNormDefaultProgramConfig(use_welford=use_welford)
    output_tensor = ttnn.layer_norm(input_tensor, program_config=program_config)
    output_tensor = ttnn.to_torch(output_tensor)

    assert_output_accuracy(torch_output_tensor, output_tensor)


@pytest.mark.parametrize("h", [32])
@pytest.mark.parametrize("w", [64])
@pytest.mark.parametrize("use_welford", [True, False])
def test_layer_norm_with_weight_and_bias(device, h, w, use_welford):
    torch.manual_seed(0)
    dtype = torch.bfloat16
    torch_input_tensor = torch.rand((h, w), dtype=dtype)
    torch_weight = torch.rand((w,), dtype=dtype)
    torch_bias = torch.rand((w,), dtype=dtype)

    torch_output_tensor = torch.nn.functional.layer_norm(
        torch_input_tensor, normalized_shape=[w], weight=torch_weight, bias=torch_bias
    )

    input_tensor = ttnn.from_torch(torch_input_tensor, layout=ttnn.TILE_LAYOUT, device=device)
    weight = ttnn.from_torch(torch_weight, layout=ttnn.TILE_LAYOUT, device=device)
    bias = ttnn.from_torch(torch_bias, layout=ttnn.TILE_LAYOUT, device=device)

    program_config = ttnn.LayerNormDefaultProgramConfig(use_welford=use_welford)
    output_tensor = ttnn.layer_norm(input_tensor, weight=weight, bias=bias, program_config=program_config)
    output_tensor = ttnn.from_device(output_tensor)
    output_tensor = ttnn.to_torch(output_tensor)

    assert_output_accuracy(torch_output_tensor, output_tensor)


@pytest.mark.parametrize("h", [32])
@pytest.mark.parametrize("w", [32])
@pytest.mark.parametrize("use_welford", [True, False])
def test_layer_norm_with_weight_and_bias_row_major(device, h, w, use_welford):
    torch.manual_seed(0)
    dtype = torch.bfloat16

    torch_input_tensor = torch.rand((h, w), dtype=dtype)
    torch_weight = torch.rand((w,), dtype=dtype)
    torch_bias = torch.rand((w,), dtype=dtype)

    torch_output_tensor = torch.nn.functional.layer_norm(
        torch_input_tensor, normalized_shape=[w], weight=torch_weight, bias=torch_bias
    )

    input_tensor = ttnn.from_torch(torch_input_tensor, layout=ttnn.TILE_LAYOUT, device=device)
    weight = ttnn.from_torch(torch_weight, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
    bias = ttnn.from_torch(torch_bias, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)

    program_config = ttnn.LayerNormDefaultProgramConfig(use_welford=use_welford)
    output_tensor = ttnn.layer_norm(input_tensor, weight=weight, bias=bias, program_config=program_config)
    output_tensor = ttnn.from_device(output_tensor)
    output_tensor = ttnn.to_torch(output_tensor)

    assert_output_accuracy(torch_output_tensor, output_tensor)


@pytest.mark.parametrize("h", [32])
@pytest.mark.parametrize("w", [64, 127, 519])
@pytest.mark.parametrize("use_welford", [True, False])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_layer_norm_with_weight_bias_and_residual_input(device, h, w, use_welford, dtype):
    torch.manual_seed(0)

    torch_input_tensor = torch.rand((h, w), dtype=dtype)
    torch_residual_input_tensor = torch.rand((h, w), dtype=dtype)
    torch_weight = torch.rand((w,), dtype=dtype)
    torch_bias = torch.rand((w,), dtype=dtype)
    torch_output_tensor = torch.nn.functional.layer_norm(
        torch_input_tensor + torch_residual_input_tensor, normalized_shape=[w], weight=torch_weight, bias=torch_bias
    )

    input_tensor = ttnn.from_torch(torch_input_tensor, layout=ttnn.TILE_LAYOUT, device=device)
    residual_input_tensor = ttnn.from_torch(torch_residual_input_tensor, layout=ttnn.TILE_LAYOUT, device=device)
    weight = ttnn.from_torch(torch_weight, layout=ttnn.TILE_LAYOUT, device=device)
    bias = ttnn.from_torch(torch_bias, layout=ttnn.TILE_LAYOUT, device=device)

    program_config = ttnn.LayerNormDefaultProgramConfig(use_welford=use_welford)
    output_tensor = ttnn.layer_norm(
        input_tensor,
        residual_input_tensor=residual_input_tensor,
        weight=weight,
        bias=bias,
        program_config=program_config,
    )
    output_tensor = ttnn.from_device(output_tensor)
    output_tensor = ttnn.to_torch(output_tensor)

    assert_output_accuracy(torch_output_tensor, output_tensor)


@pytest.mark.parametrize("h", [2])
@pytest.mark.parametrize("w", [512])
def test_layer_norm_with_tile_layout(device, h, w):
    torch.manual_seed(0)
    dtype = torch.bfloat16

    torch_input_tensor = torch.rand((1, h, w), dtype=dtype)
    torch_weight = torch.ones(w, dtype=dtype)
    torch_bias = torch.zeros(w, dtype=dtype)
    torch_output_tensor = torch.nn.functional.layer_norm(
        torch_input_tensor,
        (w,),
        torch_weight,
        torch_bias,
    )

    input_tensor = ttnn.from_torch(torch_input_tensor)
    input_tensor = ttnn.to_layout(input_tensor, ttnn.TILE_LAYOUT)
    input_tensor = ttnn.to_device(input_tensor, device)

    weight = ttnn.from_torch(torch_weight)
    weight = ttnn.to_layout(weight, ttnn.TILE_LAYOUT)
    weight = ttnn.to_device(weight, device)

    bias = ttnn.from_torch(torch_bias)
    bias = ttnn.to_layout(bias, ttnn.TILE_LAYOUT)
    bias = ttnn.to_device(bias, device)

    output_tensor = ttnn.layer_norm(
        input_tensor,
        weight=weight,
        bias=bias,
    )

    output_tensor = ttnn.from_device(output_tensor)
    output_tensor = ttnn.to_torch(output_tensor)

    assert_output_accuracy(torch_output_tensor, output_tensor)


@pytest.mark.parametrize("h", [1024, 2080])
@pytest.mark.parametrize("w", [3200, 4128])
@pytest.mark.parametrize("use_welford", [True, False])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_large_layer_norm(device, h, w, use_welford, dtype):
    torch.manual_seed(15)

    torch_input_tensor = torch.rand((h, w), dtype=dtype)
    torch_output_tensor = torch.nn.functional.layer_norm(torch_input_tensor, normalized_shape=[w])

    input_tensor = ttnn.from_torch(torch_input_tensor, layout=ttnn.TILE_LAYOUT, device=device)
    program_config = ttnn.LayerNormDefaultProgramConfig(use_welford=use_welford)
    output_tensor = ttnn.layer_norm(input_tensor, program_config=program_config)
    output_tensor = ttnn.from_device(output_tensor)
    output_tensor = ttnn.to_torch(output_tensor)

    assert_output_accuracy(torch_output_tensor, output_tensor)


@pytest.mark.parametrize("h", [2048])
@pytest.mark.parametrize("w", [4096])
@pytest.mark.parametrize("use_welford", [True, False])
def test_large_layer_norm_with_weight_and_bias(device, h, w, use_welford):
    torch.manual_seed(0)
    dtype = torch.bfloat16

    torch_input_tensor = torch.rand((h, w), dtype=dtype)
    torch_weight = torch.rand((w,), dtype=dtype)
    torch_bias = torch.rand((w,), dtype=dtype)

    torch_output_tensor = torch.nn.functional.layer_norm(
        torch_input_tensor, normalized_shape=[w], weight=torch_weight, bias=torch_bias
    )

    input_tensor = ttnn.from_torch(torch_input_tensor, layout=ttnn.TILE_LAYOUT, device=device)
    weight = ttnn.from_torch(torch_weight, layout=ttnn.TILE_LAYOUT, device=device)
    bias = ttnn.from_torch(torch_bias, layout=ttnn.TILE_LAYOUT, device=device)

    program_config = ttnn.LayerNormDefaultProgramConfig(use_welford=use_welford)
    output_tensor = ttnn.layer_norm(input_tensor, weight=weight, bias=bias, program_config=program_config)
    output_tensor = ttnn.from_device(output_tensor)
    output_tensor = ttnn.to_torch(output_tensor)

    assert_output_accuracy(torch_output_tensor, output_tensor)


@pytest.mark.parametrize("h", [2048])
@pytest.mark.parametrize("w", [4096])
@pytest.mark.parametrize("use_welford", [True, False])
def test_large_layer_norm_with_weight(device, h, w, use_welford):
    torch.manual_seed(0)
    dtype = torch.bfloat16

    torch_input_tensor = torch.rand((h, w), dtype=dtype)
    torch_weight = torch.rand((w,), dtype=dtype)

    torch_output_tensor = torch.nn.functional.layer_norm(torch_input_tensor, normalized_shape=[w], weight=torch_weight)

    input_tensor = ttnn.from_torch(torch_input_tensor, layout=ttnn.TILE_LAYOUT, device=device)
    weight = ttnn.from_torch(torch_weight, layout=ttnn.TILE_LAYOUT, device=device)

    program_config = ttnn.LayerNormDefaultProgramConfig(use_welford=use_welford)
    output_tensor = ttnn.layer_norm(input_tensor, weight=weight, program_config=program_config)
    output_tensor = ttnn.from_device(output_tensor)
    output_tensor = ttnn.to_torch(output_tensor)

    assert_output_accuracy(torch_output_tensor, output_tensor)


@pytest.mark.parametrize("h", [2048])
@pytest.mark.parametrize("w", [4096])
@pytest.mark.parametrize("use_welford", [True, False])
def test_large_layer_norm_with_bias(device, h, w, use_welford):
    torch.manual_seed(0)
    dtype = torch.bfloat16

    torch_input_tensor = torch.rand((h, w), dtype=dtype)
    torch_bias = torch.rand((w,), dtype=dtype)

    torch_output_tensor = torch.nn.functional.layer_norm(torch_input_tensor, normalized_shape=[w], bias=torch_bias)

    input_tensor = ttnn.from_torch(torch_input_tensor, layout=ttnn.TILE_LAYOUT, device=device)
    bias = ttnn.from_torch(torch_bias, layout=ttnn.TILE_LAYOUT, device=device)

    program_config = ttnn.LayerNormDefaultProgramConfig(use_welford=use_welford)
    output_tensor = ttnn.layer_norm(input_tensor, bias=bias, program_config=program_config)
    output_tensor = ttnn.from_device(output_tensor)
    output_tensor = ttnn.to_torch(output_tensor)

    assert_output_accuracy(torch_output_tensor, output_tensor)


@pytest.mark.parametrize("h, w", [(2048, 2048)])
@pytest.mark.parametrize("legacy_reduction", [True, False])
@pytest.mark.parametrize("legacy_rsqrt", [True, False])
def test_large_layer_norm_with_legacy_reduction_and_rsqrt(device, h, w, legacy_reduction, legacy_rsqrt):
    torch.manual_seed(0)
    dtype = torch.bfloat16

    torch_input_tensor = torch.rand((h, w), dtype=dtype)
    torch_bias = torch.rand((w,), dtype=dtype)

    torch_output_tensor = torch.nn.functional.layer_norm(torch_input_tensor, normalized_shape=[w], bias=torch_bias)

    input_tensor = ttnn.from_torch(torch_input_tensor, layout=ttnn.TILE_LAYOUT, device=device)
    bias = ttnn.from_torch(torch_bias, layout=ttnn.TILE_LAYOUT, device=device)

    program_config = ttnn.LayerNormDefaultProgramConfig(
        legacy_reduction=legacy_reduction, legacy_rsqrt=legacy_rsqrt, use_welford=False
    )
    compute_kernel_config = ttnn.init_device_compute_kernel_config(
        device.arch(),
        math_fidelity=ttnn.MathFidelity.HiFi4,
        math_approx_mode=False,
        fp32_dest_acc_en=True,
        packer_l1_acc=True,
    )
    output_tensor = ttnn.layer_norm(
        input_tensor, bias=bias, compute_kernel_config=compute_kernel_config, program_config=program_config
    )
    output_tensor = ttnn.from_device(output_tensor)
    output_tensor = ttnn.to_torch(output_tensor)

    # Non-fp32 accumulation is inaccurate, so we'll just compare pcc
    # to make sure it captures the general trend
    assert_with_pcc(torch_output_tensor, output_tensor, 0.97)


@pytest.mark.parametrize("h, w", [(32, 2592), (32, 3232), (1024, 2880)])
@pytest.mark.parametrize("use_welford", [True, False])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_large_layer_norm_with_weight_bias_and_residual_input(device, h, w, use_welford, dtype):
    torch.manual_seed(3333)

    torch_input_tensor = torch.rand((h, w), dtype=dtype)
    torch_residual_input_tensor = torch.rand((h, w), dtype=dtype)
    torch_weight = torch.rand((w,), dtype=dtype)
    torch_bias = torch.rand((w,), dtype=dtype)
    torch_output_tensor = torch.nn.functional.layer_norm(
        torch_input_tensor + torch_residual_input_tensor, normalized_shape=[w], weight=torch_weight, bias=torch_bias
    )

    input_tensor = ttnn.from_torch(torch_input_tensor, layout=ttnn.TILE_LAYOUT, device=device)
    residual_input_tensor = ttnn.from_torch(torch_residual_input_tensor, layout=ttnn.TILE_LAYOUT, device=device)
    weight = ttnn.from_torch(torch_weight, layout=ttnn.TILE_LAYOUT, device=device)
    bias = ttnn.from_torch(torch_bias, layout=ttnn.TILE_LAYOUT, device=device)

    program_config = ttnn.LayerNormDefaultProgramConfig(use_welford=use_welford)
    output_tensor = ttnn.layer_norm(
        input_tensor,
        residual_input_tensor=residual_input_tensor,
        weight=weight,
        bias=bias,
        program_config=program_config,
    )
    output_tensor = ttnn.from_device(output_tensor)
    output_tensor = ttnn.to_torch(output_tensor)

    if dtype == torch.float32 and use_welford and w == 3232:
        # Fallback to PCC, see https://github.com/tenstorrent/tt-metal/issues/33694
        assert_with_pcc(torch_output_tensor, output_tensor, 0.999)
    else:
        assert_output_accuracy(torch_output_tensor, output_tensor)


@pytest.mark.parametrize("use_welford", [True, False])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_l1_interleaved(device, use_welford, dtype):
    torch.manual_seed(0)

    h, w = 32, 64
    torch_input_tensor = torch.rand((h, w), dtype=dtype)
    torch_output_tensor = torch.nn.functional.layer_norm(torch_input_tensor, normalized_shape=[w])

    # Create L1 interleaved memory config
    l1_interleaved_mem_config = ttnn.MemoryConfig(
        memory_layout=ttnn.TensorMemoryLayout.INTERLEAVED,
        buffer_type=ttnn.BufferType.L1,
    )

    input_tensor = ttnn.from_torch(
        torch_input_tensor, layout=ttnn.TILE_LAYOUT, device=device, memory_config=l1_interleaved_mem_config
    )
    program_config = ttnn.LayerNormDefaultProgramConfig(use_welford=use_welford)
    output_tensor = ttnn.layer_norm(input_tensor, program_config=program_config)
    output_tensor = ttnn.from_device(output_tensor)
    output_tensor = ttnn.to_torch(output_tensor)

    assert_output_accuracy(torch_output_tensor, output_tensor)


@pytest.mark.parametrize("dim_a", [2048, 3072, 4096])
@pytest.mark.parametrize("dim_b", [2048, 3072, 4096])
@pytest.mark.parametrize("dtype", [ttnn.bfloat8_b, ttnn.bfloat16])
def test_layer_norm_across_dtypes(*, device: ttnn.Device, dim_a: int, dim_b: int, dtype: ttnn.DataType) -> None:
    torch.manual_seed(0)

    epsilon = 1e-5
    input_shape = [1, 1, dim_a, dim_b]

    torch_input = torch.randn(input_shape)
    torch_output = torch.nn.functional.layer_norm(torch_input, (input_shape[-1],), eps=epsilon)

    tt_input = ttnn.from_torch(torch_input, device=device, layout=ttnn.TILE_LAYOUT, dtype=dtype)
    tt_output = ttnn.layer_norm(tt_input, epsilon=epsilon)

    tt_output_torch = ttnn.to_torch(tt_output)

    if dtype == ttnn.bfloat16:
        assert_output_accuracy(torch_output, tt_output_torch)
    elif dtype == ttnn.bfloat8_b:
        assert_with_pcc(torch_output, tt_output_torch, pcc=0.987)


@pytest.mark.parametrize("h", [32, 2999, 32 * 64 + 18])
@pytest.mark.parametrize("w", [31, 487, 3821])
@pytest.mark.parametrize("use_welford", [True, False])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_layer_norm_with_padding(device, h, w, use_welford, dtype):
    """
    Test layer norm on a tensor that is padded with zeros
    in the width dimension.
    Compare against analytic layer norm calculation: (x - mean) / sqrt(var + eps)
    """

    torch.manual_seed(191919)

    # Fill a random number of columns with ones
    non_zero_columns = torch.randint(1, w + 1, (1,)).item()
    torch_input_tensor = torch.zeros((h, w), dtype=dtype)
    torch_input_tensor[:, :non_zero_columns] = torch.ones((h, non_zero_columns), dtype=dtype)

    # Convert to TTNN tensor
    tt_input_tensor = ttnn.from_torch(
        torch_input_tensor,
        layout=ttnn.Layout.TILE,
        device=device,
    )

    # Run layer norm
    program_config = ttnn.LayerNormDefaultProgramConfig(use_welford=use_welford)
    output_ttnn = ttnn.layer_norm(
        tt_input_tensor,
        program_config=program_config,
    )
    output_ttnn = ttnn.to_torch(output_ttnn)

    # Compute golden layer normoutput
    golden = ttnn.get_golden_function(ttnn.layer_norm)
    golden_output = golden(torch_input_tensor, weight=None, bias=None, eps=1e-5)

    assert_output_accuracy(golden_output, output_ttnn)
