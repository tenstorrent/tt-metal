# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import torch
import ttnn
from tests.ttnn.utils_for_testing import check_with_pcc
from loguru import logger
import math


def get_abs_and_relative_error(tensor_a, tensor_b):
    abs_error = torch.abs(tensor_a - tensor_b).mean().item()
    # relative_error = (abs_error / (torch.abs(tensor_a) + 1e-8)).mean().item()  # Avoid division by zero

    # Create relative error, using NaN where tensor_a is zero
    rel_err = torch.where(tensor_a != 0, torch.abs(tensor_a - tensor_b) / torch.abs(tensor_a), float("nan"))
    # Compute mean ignoring NaN values
    relative_error = torch.nanmean(rel_err).item() if rel_err.numel() > 0 else float("nan")
    return abs_error, relative_error


def check_ttnn_output(
    layer_name, pytorch_output, ttnn_output, to_channel_first=False, output_channels=None, exp_pcc=0.999
):
    """
    Compare PyTorch and TTNN outputs with format conversion and channel slicing.

    Both tensors start in NCHW format (ttnn.to_torch() converts TTNN NHWC -> NCHW).
    They are converted to the target format together, then channels are sliced if needed.
    """
    # Convert TTNN output to torch (converts NHWC -> NCHW, already creates new tensor)
    ttnn_output_torch = ttnn.to_torch(ttnn_output)

    if len(pytorch_output.shape) == 4 and len(ttnn_output_torch.shape) == 4:
        if not to_channel_first:
            # Convert both from NCHW to NHWC
            pytorch_output = pytorch_output.permute(0, 2, 3, 1)
            ttnn_output_torch = ttnn_output_torch.permute(0, 2, 3, 1)

    # Slice channels if needed (after format conversion, so channel dim is correct)
    if output_channels is not None:
        if to_channel_first:
            # NCHW: channels at dim 1
            ttnn_output_torch = ttnn_output_torch[:, :output_channels, :, :]
            pytorch_output = pytorch_output[:, :output_channels, :, :]
        else:
            # NHWC: channels at dim 3
            ttnn_output_torch = ttnn_output_torch[:, :, :, :output_channels]
            pytorch_output = pytorch_output[:, :, :, :output_channels]

    abs_err, rel_err = get_abs_and_relative_error(pytorch_output, ttnn_output_torch)
    passed, pcc = check_with_pcc(pytorch_output, ttnn_output_torch, exp_pcc)
    special_char = "✅" if passed else "❌"
    logger.warning(f"{special_char} Output {layer_name}: {passed=}, {pcc=}, {abs_err=:.3f}, {rel_err=:.3f}")
    if passed and float(pcc) - exp_pcc > 0.001:
        logger.warning(
            f"⚠️  Output {layer_name} PCC is better than expected by {float(pcc)-exp_pcc:.3f}. Please update expected PCC value to {math.floor(float(pcc) * 1000) / 1000:.3f}."
        )

    return passed
