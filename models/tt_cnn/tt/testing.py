# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass
from typing import Dict, Literal, Optional

import torch
from loguru import logger

import ttnn
from models.perf.device_perf_utils import check_device_perf, prep_device_perf_report, run_device_perf
from tests.ttnn.utils_for_testing import assert_with_pcc


def pad_channels_up_to_target(input_tensor, target=16):
    assert len(input_tensor.shape) == 4, "Expected input tensor to rank 4"
    N, C, H, W = input_tensor.shape
    if C < target:
        return torch.nn.functional.pad(input_tensor, (0, 0, 0, 0, 0, target - C), mode="constant", value=0)
    else:
        return input_tensor


def create_random_input_tensor(
    batch: int,
    groups: int,
    input_channels: int = 4,
    input_height: int = 1056,
    input_width: int = 160,
    channel_order: Literal["first", "last"] = "last",
    fold: bool = True,
    pad: bool = True,
    device=None,
    memory_config=None,
    mesh_mapper=None,
):
    torch_input_tensor = torch.randn(batch, input_channels * groups, input_height, input_width)

    # We almost always (unless running full model) want to ensure we have least 16 because conv2d requires it
    ttnn_input_tensor = pad_channels_up_to_target(torch_input_tensor, 16) if pad else torch_input_tensor

    ttnn_input_tensor = ttnn_input_tensor if channel_order == "first" else ttnn_input_tensor.permute(0, 2, 3, 1)

    if fold:
        if channel_order == "first":
            ttnn_input_tensor = ttnn_input_tensor.reshape(batch, 1, input_channels * groups, input_height * input_width)
        else:
            ttnn_input_tensor = ttnn_input_tensor.reshape(batch, 1, input_height * input_width, -1)

    ttnn_input_tensor = ttnn.from_torch(
        ttnn_input_tensor, dtype=ttnn.bfloat16, device=device, memory_config=memory_config, mesh_mapper=mesh_mapper
    )

    return torch_input_tensor, ttnn_input_tensor


@dataclass
class DevicePerformanceTestConfiguration:
    """Configuration for device performance tests."""

    model_name: str
    test_command: str
    batch_size: int
    expected_throughput_fps: float
    subdir: Optional[str] = None
    num_iterations: int = 1
    margin: float = 0.02
    columns: Optional[list] = None
    inference_time_key: str = "AVG DEVICE KERNEL SAMPLES/S"
    comments: str = ""


def run_device_perf_test(config: DevicePerformanceTestConfiguration) -> Dict[str, float]:
    """Run a device performance test."""
    if config.columns is None:
        columns = ["DEVICE FW", "DEVICE KERNEL", "DEVICE BRISC KERNEL"]
    else:
        columns = config.columns
    subdir = config.subdir or config.model_name

    logger.info(f"Running device performance test for {config.model_name}...")
    logger.info(f"Command: {config.test_command}")
    logger.info(f"Expected throughput: {config.expected_throughput_fps} fps")

    # Run device performance measurement
    post_processed_results = run_device_perf(
        config.test_command,
        subdir=subdir,
        num_iterations=config.num_iterations,
        cols=columns,
        batch_size=config.batch_size,
    )

    # Check against expected performance
    expected_perf_cols = {config.inference_time_key: config.expected_throughput_fps}
    expected_results = check_device_perf(
        post_processed_results,
        margin=config.margin,
        expected_perf_cols=expected_perf_cols,
        assert_on_fail=True,
    )

    # Prepare performance report
    prep_device_perf_report(
        model_name=config.model_name,
        batch_size=config.batch_size,
        post_processed_results=post_processed_results,
        expected_results=expected_results,
        comments=config.comments,
    )

    logger.info(f"Device performance test passed for {config.model_name}")

    return {
        "post_processed_results": post_processed_results,
        "expected_results": expected_results,
    }


def verify_conv2d_from_config(
    configuration,
    device,
    pcc_threshold: float = 0.999,
) -> None:
    """Verify a Conv2d layer from a configuration.

    This creates inputs, runs both TT and PyTorch implementations,
    and verifies correctness with PCC.

    Args:
        configuration: Conv2dConfiguration to test
        device: ttnn.Device to run on
        pcc_threshold: PCC threshold for verification (default: 0.999)

    Raises:
        AssertionError: If PCC check fails or shapes don't match

    Example:
        config = Conv2dConfiguration.with_random_weights(
            input_height=224, input_width=224,
            in_channels=64, out_channels=128,
            batch_size=1, kernel_size=(3, 3), padding=(1, 1),
        )

        verify_conv2d_from_config(config, device)
    """
    from models.tt_cnn.tt.builder import TtConv2d

    logger.info(f"Running test for conv2d with configuration:")
    logger.info(
        f"Config: {configuration.in_channels}->{configuration.out_channels}, "
        f"kernel={configuration.kernel_size}, input={configuration.input_height}x{configuration.input_width}"
    )

    shape = (
        configuration.batch_size,
        configuration.in_channels,
        configuration.input_height,
        configuration.input_width,
    )
    torch_input = torch.randn(shape, dtype=torch.bfloat16).float()

    nhwc = torch.permute(torch_input, (0, 2, 3, 1))
    ttnn_input = ttnn.from_torch(nhwc, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT)

    tt_layer = TtConv2d(configuration, device)
    ttnn_output = tt_layer(ttnn_input)

    torch_output = torch.nn.functional.conv2d(
        torch_input,
        ttnn.to_torch(configuration.weight),
        ttnn.to_torch(configuration.bias).reshape(-1) if configuration.bias is not None else None,
        stride=configuration.stride,
        padding=configuration.padding,
        dilation=configuration.dilation,
        groups=configuration.groups,
    )

    output_height, output_width = torch_output.shape[-2:]
    ttnn_output_torch = (
        ttnn.to_torch(ttnn_output)
        .reshape(configuration.batch_size, output_height, output_width, configuration.out_channels)
        .permute(0, 3, 1, 2)
    )

    assert (
        ttnn_output_torch.shape == torch_output.shape
    ), f"Shape mismatch: ttnn={ttnn_output_torch.shape}, torch={torch_output.shape}"

    pcc_passed, pcc_message = assert_with_pcc(torch_output, ttnn_output_torch, pcc_threshold)
    logger.success(f"Test passed with PCC: {pcc_message:.5f} > {pcc_threshold:.5f}")


def verify_maxpool2d_from_config(
    configuration,
    device,
    pcc_threshold: float = 0.999,
) -> None:
    """Verify a MaxPool2d layer from a configuration.

    This creates inputs, runs both TT and PyTorch implementations,
    and verifies correctness with PCC.

    Args:
        configuration: MaxPool2dConfiguration to test
        device: ttnn.Device to run on
        pcc_threshold: PCC threshold for verification (default: 0.999)

    Example:
        config = MaxPool2dConfiguration(
            input_height=224, input_width=224, channels=64,
            batch_size=1, kernel_size=(2, 2), stride=(2, 2),
        )

        verify_maxpool2d_from_config(config, device)
    """
    from models.tt_cnn.tt.builder import TtMaxPool2d

    logger.info(f"Running test for maxpool2d with configuration:")
    logger.info(
        f"Config: channels={configuration.channels}, kernel={configuration.kernel_size}, "
        f"input={configuration.input_height}x{configuration.input_width}"
    )

    shape = (
        configuration.batch_size,
        configuration.channels,
        configuration.input_height,
        configuration.input_width,
    )
    torch_input = torch.randn(shape, dtype=torch.bfloat16).float()

    nhwc = torch.permute(torch_input, (0, 2, 3, 1)).reshape(
        1, 1, configuration.batch_size * configuration.input_height * configuration.input_width, configuration.channels
    )
    ttnn_input = ttnn.from_torch(nhwc, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)

    tt_layer = TtMaxPool2d(configuration, device)
    ttnn_output = tt_layer(ttnn_input)

    torch_output = torch.nn.functional.max_pool2d(
        torch_input,
        kernel_size=configuration.kernel_size,
        stride=configuration.stride,
        padding=configuration.padding,
        dilation=configuration.dilation,
        ceil_mode=configuration.ceil_mode,
    )

    output_height, output_width = torch_output.shape[-2:]
    ttnn_output_torch = (
        ttnn.to_torch(ttnn_output)
        .reshape(configuration.batch_size, output_height, output_width, configuration.channels)
        .permute(0, 3, 1, 2)
    )

    assert (
        ttnn_output_torch.shape == torch_output.shape
    ), f"Shape mismatch: ttnn={ttnn_output_torch.shape}, torch={torch_output.shape}"

    pcc_passed, pcc_message = assert_with_pcc(torch_output, ttnn_output_torch, pcc_threshold)
    logger.success(f"Test passed with PCC: {pcc_message:.5f} > {pcc_threshold:.5f}")
