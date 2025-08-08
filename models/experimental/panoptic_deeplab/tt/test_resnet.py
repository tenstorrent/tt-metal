# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import ttnn
from models.experimental.panoptic_deeplab.tt.resnet.tt_resnet import TtResNet
from models.experimental.panoptic_deeplab.tt.resnet.tt_stem import TtStem
from models.experimental.panoptic_deeplab.tt.resnet.tt_bottleneck import TtBottleneck


def create_random_state_dict_for_stem():
    """Create a random state dict for the stem module."""
    state_dict = {}

    # Conv1: 3 -> 64 channels
    state_dict["conv1.weight"] = torch.randn(64, 3, 3, 3, dtype=torch.bfloat16)
    state_dict["conv1.norm.weight"] = torch.randn(64, dtype=torch.bfloat16)
    state_dict["conv1.norm.bias"] = torch.randn(64, dtype=torch.bfloat16)
    state_dict["conv1.norm.running_mean"] = torch.randn(64, dtype=torch.bfloat16)
    state_dict["conv1.norm.running_var"] = torch.rand(64, dtype=torch.bfloat16) + 0.1  # Avoid zeros

    # Conv2: 64 -> 64 channels
    state_dict["conv2.weight"] = torch.randn(64, 64, 3, 3, dtype=torch.bfloat16)
    state_dict["conv2.norm.weight"] = torch.randn(64, dtype=torch.bfloat16)
    state_dict["conv2.norm.bias"] = torch.randn(64, dtype=torch.bfloat16)
    state_dict["conv2.norm.running_mean"] = torch.randn(64, dtype=torch.bfloat16)
    state_dict["conv2.norm.running_var"] = torch.rand(64, dtype=torch.bfloat16) + 0.1

    # Conv3: 64 -> 128 channels
    state_dict["conv3.weight"] = torch.randn(128, 64, 3, 3, dtype=torch.bfloat16)
    state_dict["conv3.norm.weight"] = torch.randn(128, dtype=torch.bfloat16)
    state_dict["conv3.norm.bias"] = torch.randn(128, dtype=torch.bfloat16)
    state_dict["conv3.norm.running_mean"] = torch.randn(128, dtype=torch.bfloat16)
    state_dict["conv3.norm.running_var"] = torch.rand(128, dtype=torch.bfloat16) + 0.1

    return state_dict


def create_random_state_dict_for_bottleneck(in_channels, mid_channels, out_channels, has_shortcut=False):
    """Create a random state dict for a bottleneck block."""
    state_dict = {}

    # Conv1: in_channels -> mid_channels (1x1)
    state_dict["conv1.weight"] = torch.randn(mid_channels, in_channels, 1, 1, dtype=torch.bfloat16)
    state_dict["conv1.norm.weight"] = torch.randn(mid_channels, dtype=torch.bfloat16)
    state_dict["conv1.norm.bias"] = torch.randn(mid_channels, dtype=torch.bfloat16)
    state_dict["conv1.norm.running_mean"] = torch.randn(mid_channels, dtype=torch.bfloat16)
    state_dict["conv1.norm.running_var"] = torch.rand(mid_channels, dtype=torch.bfloat16) + 0.1

    # Conv2: mid_channels -> mid_channels (3x3)
    state_dict["conv2.weight"] = torch.randn(mid_channels, mid_channels, 3, 3, dtype=torch.bfloat16)
    state_dict["conv2.norm.weight"] = torch.randn(mid_channels, dtype=torch.bfloat16)
    state_dict["conv2.norm.bias"] = torch.randn(mid_channels, dtype=torch.bfloat16)
    state_dict["conv2.norm.running_mean"] = torch.randn(mid_channels, dtype=torch.bfloat16)
    state_dict["conv2.norm.running_var"] = torch.rand(mid_channels, dtype=torch.bfloat16) + 0.1

    # Conv3: mid_channels -> out_channels (1x1)
    state_dict["conv3.weight"] = torch.randn(out_channels, mid_channels, 1, 1, dtype=torch.bfloat16)
    state_dict["conv3.norm.weight"] = torch.randn(out_channels, dtype=torch.bfloat16)
    state_dict["conv3.norm.bias"] = torch.randn(out_channels, dtype=torch.bfloat16)
    state_dict["conv3.norm.running_mean"] = torch.randn(out_channels, dtype=torch.bfloat16)
    state_dict["conv3.norm.running_var"] = torch.rand(out_channels, dtype=torch.bfloat16) + 0.1

    # Shortcut (if needed)
    if has_shortcut:
        state_dict["shortcut.weight"] = torch.randn(out_channels, in_channels, 1, 1, dtype=torch.bfloat16)
        state_dict["shortcut.norm.weight"] = torch.randn(out_channels, dtype=torch.bfloat16)
        state_dict["shortcut.norm.bias"] = torch.randn(out_channels, dtype=torch.bfloat16)
        state_dict["shortcut.norm.running_mean"] = torch.randn(out_channels, dtype=torch.bfloat16)
        state_dict["shortcut.norm.running_var"] = torch.rand(out_channels, dtype=torch.bfloat16) + 0.1

    return state_dict


def create_random_state_dict_for_resnet():
    """Create a complete random state dict for the ResNet model."""
    state_dict = {}

    # Stem
    stem_dict = create_random_state_dict_for_stem()
    for k, v in stem_dict.items():
        state_dict[f"stem.{k}"] = v

    # res2: 3 blocks (128 -> 256 channels)
    for i in range(3):
        has_shortcut = i == 0
        in_channels = 128 if i == 0 else 256
        block_dict = create_random_state_dict_for_bottleneck(in_channels, 64, 256, has_shortcut)
        for k, v in block_dict.items():
            state_dict[f"res2.{i}.{k}"] = v

    # res3: 4 blocks (256 -> 512 channels)
    for i in range(4):
        has_shortcut = i == 0
        in_channels = 256 if i == 0 else 512
        block_dict = create_random_state_dict_for_bottleneck(in_channels, 128, 512, has_shortcut)
        for k, v in block_dict.items():
            state_dict[f"res3.{i}.{k}"] = v

    # res4: 6 blocks (512 -> 1024 channels)
    for i in range(6):
        has_shortcut = i == 0
        in_channels = 512 if i == 0 else 1024
        block_dict = create_random_state_dict_for_bottleneck(in_channels, 256, 1024, has_shortcut)
        for k, v in block_dict.items():
            state_dict[f"res4.{i}.{k}"] = v

    # res5: 3 blocks (1024 -> 2048 channels)
    for i in range(3):
        has_shortcut = i == 0
        in_channels = 1024 if i == 0 else 2048
        block_dict = create_random_state_dict_for_bottleneck(in_channels, 512, 2048, has_shortcut)
        for k, v in block_dict.items():
            state_dict[f"res5.{i}.{k}"] = v

    return state_dict


@pytest.mark.parametrize("device_params", [{"l1_small_size": 32768}], indirect=True)
@pytest.mark.parametrize(
    "batch_size, input_height, input_width",
    [
        (1, 32, 32),
        (1, 64, 64),
        # (2, 32, 32),  # Uncomment for more extensive testing
    ],
)
def test_ttnn_stem(device, batch_size, input_height, input_width):
    """Test TtStem module."""
    torch.manual_seed(0)

    # Create input tensor (batch, height, width, channels)
    torch_input = torch.randn(batch_size, input_height, input_width, 3, dtype=torch.bfloat16)
    ttnn_input = ttnn.from_torch(torch_input, device=device, layout=ttnn.TILE_LAYOUT)

    # Create random state dict
    state_dict = create_random_state_dict_for_stem()

    # Create TtStem
    tt_stem = TtStem(device=device, state_dict=state_dict)

    # Test forward pass
    output = tt_stem.forward(ttnn_input)

    # Check output shape
    assert output.shape[0] == batch_size
    assert output.shape[3] == 128  # Output channels should be 128
    print(f"TtStem test passed - Output shape: {output.shape}")


@pytest.mark.parametrize("device_params", [{"l1_small_size": 32768}], indirect=True)
@pytest.mark.parametrize(
    "batch_size, input_height, input_width, in_channels, mid_channels, out_channels, has_shortcut",
    [
        (1, 32, 32, 128, 64, 256, True),  # res2 first block
        (1, 32, 32, 256, 64, 256, False),  # res2 subsequent blocks
        (1, 16, 16, 256, 128, 512, True),  # res3 first block
        (1, 16, 16, 512, 128, 512, False),  # res3 subsequent blocks
    ],
)
def test_ttnn_bottleneck(
    device, batch_size, input_height, input_width, in_channels, mid_channels, out_channels, has_shortcut
):
    """Test TtBottleneck module."""
    torch.manual_seed(0)

    # Create input tensor (batch, height, width, channels)
    torch_input = torch.randn(batch_size, input_height, input_width, in_channels, dtype=torch.bfloat16)
    ttnn_input = ttnn.from_torch(torch_input, device=device, layout=ttnn.TILE_LAYOUT)

    # Create random state dict
    state_dict = create_random_state_dict_for_bottleneck(in_channels, mid_channels, out_channels, has_shortcut)

    # Create TtBottleneck
    tt_bottleneck = TtBottleneck(device=device, state_dict=state_dict, has_shortcut=has_shortcut)

    # Test forward pass
    output = tt_bottleneck.forward(ttnn_input)

    # Check output shape
    assert output.shape[0] == batch_size
    assert output.shape[3] == out_channels
    print(f"TtBottleneck test passed - Output shape: {output.shape}, has_shortcut: {has_shortcut}")


@pytest.mark.parametrize("device_params", [{"l1_small_size": 32768}], indirect=True)
@pytest.mark.parametrize(
    "batch_size, input_height, input_width",
    [
        (1, 32, 32),
        (1, 64, 64),
    ],
)
def test_ttnn_resnet(device, batch_size, input_height, input_width):
    """Test complete TtResNet module."""
    torch.manual_seed(0)

    # Create input tensor (batch, height, width, channels=3)
    torch_input = torch.randn(batch_size, input_height, input_width, 3, dtype=torch.bfloat16)
    ttnn_input = ttnn.from_torch(torch_input, device=device, layout=ttnn.TILE_LAYOUT)

    # Create random state dict
    state_dict = create_random_state_dict_for_resnet()

    # Create TtResNet
    tt_resnet = TtResNet(device=device, state_dict=state_dict)

    # Test forward pass with multi-output
    outputs = tt_resnet.forward(ttnn_input)

    # Check that all expected outputs are present
    assert "res2" in outputs
    assert "res3" in outputs
    assert "res4" in outputs
    assert "res5" in outputs

    # Check output shapes
    assert outputs["res2"].shape[0] == batch_size
    assert outputs["res2"].shape[3] == 256

    assert outputs["res3"].shape[0] == batch_size
    assert outputs["res3"].shape[3] == 512

    assert outputs["res4"].shape[0] == batch_size
    assert outputs["res4"].shape[3] == 1024

    assert outputs["res5"].shape[0] == batch_size
    assert outputs["res5"].shape[3] == 2048

    print(f"TtResNet multi-output test passed")
    print(f"  res2 shape: {outputs['res2'].shape}")
    print(f"  res3 shape: {outputs['res3'].shape}")
    print(f"  res4 shape: {outputs['res4'].shape}")
    print(f"  res5 shape: {outputs['res5'].shape}")

    # Test single output forward pass
    single_output = tt_resnet.forward_single_output(ttnn_input)
    assert single_output.shape[0] == batch_size
    assert single_output.shape[3] == 2048

    print(f"TtResNet single-output test passed - Output shape: {single_output.shape}")
