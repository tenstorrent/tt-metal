# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

import math
import struct

import pytest
import torch

import ttnn


def bfloat16_truncated(value: float) -> float:
    bits = struct.unpack("<I", struct.pack("<f", value))[0] & 0xFFFF0000
    return struct.unpack("<f", struct.pack("<I", bits))[0]


def compute_kernel_config(device):
    return ttnn.init_device_compute_kernel_config(
        device.arch(),
        math_fidelity=ttnn.MathFidelity.HiFi4,
        math_approx_mode=False,
        fp32_dest_acc_en=True,
        packer_l1_acc=False,
    )


# The scale that scale_mask_softmax fuses into the softmax used to be truncated to bfloat16, whatever the
# input dtype was. With a float32 input the truncation shows up in the probabilities: for head_dim 80 the
# scale 1/sqrt(80) is off by 0.43%, which moves the result further from the float64 reference than the
# kernel's own rounding does.
@pytest.mark.parametrize("head_dim", [80, 100])
def test_scale_mask_softmax_float32_scale_is_not_truncated_to_bfloat16(device, head_dim):
    scale = 1.0 / math.sqrt(head_dim)
    torch.manual_seed(3)
    logits = torch.randn(1, 4, 64, 128) * 8
    mask = torch.zeros(1, 1, 1, 128)

    config = compute_kernel_config(device)
    input_tensor = ttnn.from_torch(logits, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
    mask_tensor = ttnn.from_torch(mask, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
    output = ttnn.to_torch(
        ttnn.scale_mask_softmax(input_tensor, scale, mask_tensor, compute_kernel_config=config)
    ).double()

    prescaled_input = ttnn.from_torch(logits * scale, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
    prescaled = ttnn.to_torch(ttnn.softmax(prescaled_input, dim=-1, compute_kernel_config=config)).double()

    reference = torch.softmax(logits.double() * scale, dim=-1)
    truncated_reference = torch.softmax(logits.double() * bfloat16_truncated(scale), dim=-1)

    error = (output - reference).abs().max().item()
    truncated_error = (output - truncated_reference).abs().max().item()
    control_error = (prescaled - reference).abs().max().item()

    assert error < truncated_error, (
        f"the fused scale is applied with bfloat16 precision: error {error} against the float32 scale, "
        f"{truncated_error} against the bfloat16 truncated scale"
    )
    assert (
        error <= 3 * control_error + 1e-5
    ), f"the fused scale loses more precision than the pre-scaled softmax: {error} against {control_error}"


# The bfloat16 path keeps its bfloat16 scale, so this only checks that the float32 change left it alone.
def test_scale_mask_softmax_bfloat16_scale_path_is_unchanged(device):
    scale = 1.0 / math.sqrt(80)
    torch.manual_seed(3)
    logits = (torch.randn(1, 4, 64, 128) * 8).bfloat16()
    mask = torch.zeros(1, 1, 1, 128).bfloat16()

    config = compute_kernel_config(device)
    input_tensor = ttnn.from_torch(logits, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    mask_tensor = ttnn.from_torch(mask, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    output = ttnn.to_torch(
        ttnn.scale_mask_softmax(input_tensor, scale, mask_tensor, compute_kernel_config=config)
    ).double()

    reference = torch.softmax(logits.double() * scale, dim=-1)
    error = (output - reference).abs().max().item()
    assert error < 0.02, f"the bfloat16 path moved: error {error}"
