# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

# Dilation test cases for Conv3d - Bounty #25940

from loguru import logger
import pytest
import ttnn
from tests.ttnn.utils_for_testing import check_with_pcc

from tests.ttnn.unit_tests.operations.conv.test_conv3d import (
    setup_conv3d_test,
    create_conv3d_config,
    prepare_weights,
    reshape_output,
)


@pytest.mark.parametrize(
    "input_shape, out_channels, kernel_size, stride, padding, dilation, padding_mode",
    [
        # Basic dilation tests with (1,1,1) - should match non-dilated convolution
        [(1, 32, 8, 8, 8), 32, (3, 3, 3), (1, 1, 1), (0, 1, 1), (1, 1, 1), "zeros"],
        # Uniform dilation (2,2,2)
        [(1, 32, 10, 10, 10), 32, (3, 3, 3), (1, 1, 1), (0, 1, 1), (2, 2, 2), "zeros"],
        # Asymmetric dilation (1,2,3)
        [(1, 64, 12, 12, 12), 64, (3, 3, 3), (1, 1, 1), (0, 1, 1), (1, 2, 3), "zeros"],
        # Larger dilation
        [(1, 32, 16, 16, 16), 32, (3, 3, 3), (1, 1, 1), (0, 2, 2), (3, 3, 3), "zeros"],
        # Dilation with stride
        [(1, 64, 14, 14, 14), 64, (3, 3, 3), (2, 2, 2), (0, 1, 1), (2, 2, 2), "zeros"],
        # Dilation with replicate padding
        [(1, 32, 10, 10, 10), 32, (3, 3, 3), (1, 1, 1), (0, 1, 1), (2, 2, 2), "replicate"],
        # Small kernel with dilation
        [(1, 64, 12, 12, 12), 64, (2, 2, 2), (1, 1, 1), (0, 1, 1), (2, 2, 2), "zeros"],
    ],
    ids=[
        "dilation_111_baseline",
        "dilation_222_uniform",
        "dilation_123_asymmetric",
        "dilation_333_large",
        "dilation_222_with_stride",
        "dilation_222_replicate",
        "dilation_222_small_kernel",
    ],
)
def test_conv3d_dilation(device, input_shape, out_channels, kernel_size, stride, padding, dilation, padding_mode):
    """Test Conv3d with various dilation configurations.
    
    This test validates the dilation support implementation for Conv3d operations (Bounty #25940).
    Dilation allows the convolution kernel to have gaps between its elements, which is useful
    for expanding the receptive field without increasing computation.
    """
    grid_size = device.compute_with_storage_grid_size()
    
    tt_input, conv3d_module, gt_output, kernel_config, output_dims = setup_conv3d_test(
        input_shape, out_channels, kernel_size, stride, padding, padding_mode, device, dilation=dilation
    )
    N, D_out, H_out, W_out = output_dims
    C = input_shape[1]

    # Prepare weights and bias for TTNN
    tt_weight, tt_bias = prepare_weights(conv3d_module, C, out_channels, device, C_in_block=0)

    # Create config and run TTNN conv3d
    config = create_conv3d_config(compute_with_storage_grid_size=grid_size)

    logger.info(
        f"Testing Conv3d with dilation={dilation}, kernel_size={kernel_size}, "
        f"stride={stride}, padding={padding}"
    )

    tt_output = ttnn.experimental.conv3d(
        input_tensor=tt_input,
        weight_tensor=tt_weight,
        bias_tensor=tt_bias,
        dtype=ttnn.bfloat16,
        output_channels=out_channels,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        dilation=dilation,
        padding_mode=padding_mode,
        config=config,
        compute_kernel_config=kernel_config,
    )

    # Reshape output and verify results
    tt_output = reshape_output(tt_output, N, D_out, H_out, W_out, out_channels, device)

    logger.info(f"Expected output shape: {gt_output.shape}")
    logger.info(f"Actual output shape: {tt_output.shape}")
    assert tt_output.shape == gt_output.shape, f"Shape mismatch: expected {gt_output.shape}, got {tt_output.shape}"

    pcc_passed, pcc_message = check_with_pcc(gt_output, tt_output, pcc=0.99)
    logger.info(f"PCC comparison result: {pcc_message}")
    assert pcc_passed, f"PCC check failed for dilation={dilation}: {pcc_message}"


@pytest.mark.parametrize("dilation", [(1, 1, 1), (2, 2, 2), (3, 3, 3), (1, 2, 1)], ids=["d111", "d222", "d333", "d121"])
@pytest.mark.parametrize("C", [32, 64])
def test_conv3d_dilation_sweep(device, dilation, C):
    """Sweep test for different dilation values with various channel counts."""
    input_shape = (1, C, 12, 12, 12)
    out_channels = C
    kernel_size = (3, 3, 3)
    stride = (1, 1, 1)
    padding = (0, 1, 1)
    padding_mode = "zeros"
    grid_size = device.compute_with_storage_grid_size()

    tt_input, conv3d_module, gt_output, kernel_config, output_dims = setup_conv3d_test(
        input_shape, out_channels, kernel_size, stride, padding, padding_mode, device, dilation=dilation
    )
    N, D_out, H_out, W_out = output_dims

    # Prepare weights and bias
    tt_weight, tt_bias = prepare_weights(conv3d_module, C, out_channels, device)

    config = create_conv3d_config(compute_with_storage_grid_size=grid_size)

    tt_output = ttnn.experimental.conv3d(
        input_tensor=tt_input,
        weight_tensor=tt_weight,
        bias_tensor=tt_bias,
        dtype=ttnn.bfloat16,
        output_channels=out_channels,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        dilation=dilation,
        padding_mode=padding_mode,
        config=config,
        compute_kernel_config=kernel_config,
    )

    tt_output = reshape_output(tt_output, N, D_out, H_out, W_out, out_channels, device)

    pcc_passed, pcc_message = check_with_pcc(gt_output, tt_output, pcc=0.99)
    logger.info(f"Dilation={dilation}, C={C}: {pcc_message}")
    assert pcc_passed, pcc_message
