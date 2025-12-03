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


def check_with_tolerance(abs_err, rel_err, exp_abs_err, exp_rel_err):
    return abs_err <= exp_abs_err and rel_err <= exp_rel_err


def check_ttnn_output(
    layer_name,
    pytorch_output,
    ttnn_output,
    to_channel_first=False,
    output_channels=None,
    output_shape=None,
    exp_pcc=0.999,
    exp_abs_err=1e-08,
    exp_rel_err=1e-05,
):
    ttnn_output_torch = ttnn.to_torch(ttnn_output)

    if to_channel_first:
        ttnn_output_torch = ttnn_output_torch.permute(0, 3, 1, 2)  # NHWC to NCHW

    if output_shape:
        ttnn_output_torch = torch.reshape(ttnn_output_torch, output_shape)

    if output_channels is not None:
        logger.debug(f"Slicing {layer_name} output from {ttnn_output_torch.shape[1]} to {output_channels} channels")
        ttnn_output_torch = ttnn_output_torch[:, :output_channels, :, :]

    abs_err, rel_err = get_abs_and_relative_error(pytorch_output, ttnn_output_torch)
    passed_pcc, pcc = check_with_pcc(pytorch_output, ttnn_output_torch, exp_pcc)
    passed_tolerance = check_with_tolerance(abs_err, rel_err, exp_abs_err, exp_rel_err)
    special_char = "✅" if passed_pcc and passed_tolerance else "❌"
    logger.warning(
        f"{special_char} Output {layer_name}: {passed_pcc=}, {pcc=}, {passed_tolerance=}, {abs_err=:.3f}, {rel_err=:.3f}"
    )
    if passed_pcc and float(pcc) - exp_pcc > 0.001:
        logger.warning(
            f"⚠️  Output {layer_name} PCC is better than expected by {float(pcc)-exp_pcc:.3f}. Please update expected PCC value to {math.floor(float(pcc) * 1000) / 1000:.3f}."
        )

    return passed_pcc and passed_tolerance
