"""
Test compute kernel configurations for pool operations.
Only average pooling operations support configurable compute kernel configs.
"""
import pytest
import ttnn
import torch


@pytest.mark.parametrize("device_params", [{"l1_small_size": 8192}], indirect=True)
def test_avg_pool2d_with_different_compute_configs(device):
    """Test average pool2d with different compute kernel configurations."""
    batch_size = 1
    channels = 32
    input_h, input_w = 16, 16
    kernel_h, kernel_w = 2, 2

    # Create input tensor
    nchw_shape = (batch_size, channels, input_h, input_w)
    input_torch = torch.randn(nchw_shape, dtype=torch.bfloat16)
    input_perm = torch.permute(input_torch, (0, 2, 3, 1))
    input_shape = (1, 1, batch_size * input_h * input_w, channels)
    input_reshaped = input_perm.reshape(input_shape)

    input_tensor = ttnn.from_torch(input_reshaped, device=device)

    # Test with default compute config (None)
    output_default = ttnn.avg_pool2d(
        input_tensor=input_tensor,
        batch_size=batch_size,
        input_h=input_h,
        input_w=input_w,
        channels=channels,
        kernel_size=[kernel_h, kernel_w],
        stride=[1, 1],
        padding=[0, 0],
        ceil_mode=False,
        count_include_pad=True,
    )

    # Test with LoFi compute config
    compute_kernel_config = ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.LoFi, math_approx_mode=False, fp32_dest_acc_en=False
    )

    output_lofi = ttnn.avg_pool2d(
        input_tensor=input_tensor,
        batch_size=batch_size,
        input_h=input_h,
        input_w=input_w,
        channels=channels,
        kernel_size=[kernel_h, kernel_w],
        stride=[1, 1],
        padding=[0, 0],
        ceil_mode=False,
        count_include_pad=True,
        compute_kernel_config=compute_kernel_config,
    )

    # Test with HiFi4 compute config
    compute_kernel_config_hifi4 = ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False, fp32_dest_acc_en=False
    )

    output_hifi4 = ttnn.avg_pool2d(
        input_tensor=input_tensor,
        batch_size=batch_size,
        input_h=input_h,
        input_w=input_w,
        channels=channels,
        kernel_size=[kernel_h, kernel_w],
        stride=[1, 1],
        padding=[0, 0],
        ceil_mode=False,
        count_include_pad=True,
        compute_kernel_config=compute_kernel_config_hifi4,
    )

    # Verify outputs are tensors with expected shapes
    expected_h = (input_h - kernel_h) // 1 + 1  # stride = 1, no padding
    expected_w = (input_w - kernel_w) // 1 + 1
    expected_nhw = batch_size * expected_h * expected_w

    assert output_default.shape == (1, 1, expected_nhw, channels)
    assert output_lofi.shape == (1, 1, expected_nhw, channels)
    assert output_hifi4.shape == (1, 1, expected_nhw, channels)

    print("All average pool compute kernel config tests passed!")


@pytest.mark.parametrize("device_params", [{"l1_small_size": 8192}], indirect=True)
def test_adaptive_avg_pool2d_with_compute_configs(device):
    """Test adaptive average pool2d with compute kernel configurations."""
    batch_size = 1
    channels = 32
    input_h, input_w = 8, 8
    output_h, output_w = 4, 4

    # Create input tensor
    nchw_shape = (batch_size, channels, input_h, input_w)
    input_torch = torch.randn(nchw_shape, dtype=torch.bfloat16)
    input_perm = torch.permute(input_torch, (0, 2, 3, 1))
    input_shape = (1, 1, batch_size * input_h * input_w, channels)
    input_reshaped = input_perm.reshape(input_shape)

    input_tensor = ttnn.from_torch(input_reshaped, device=device)

    # Test with fp32 accumulation compute config
    compute_kernel_config = ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False, fp32_dest_acc_en=True
    )

    output = ttnn.adaptive_avg_pool2d(
        input_tensor=input_tensor,
        batch_size=batch_size,
        input_h=input_h,
        input_w=input_w,
        channels=channels,
        output_size=[output_h, output_w],
        compute_kernel_config=compute_kernel_config,
    )

    # Verify output shape
    expected_nhw = batch_size * output_h * output_w
    assert output.shape == (1, 1, expected_nhw, channels)

    print("Adaptive average pool compute kernel config test passed!")
