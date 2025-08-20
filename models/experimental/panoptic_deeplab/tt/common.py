# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import torch


def create_random_stem_state_dict():
    """Create random state dict for DeepLabStem."""
    state_dict = {}

    # Conv1: 3 -> 64 channels, 3x3 kernel, stride=2
    state_dict["conv1.weight"] = torch.randn(64, 3, 3, 3, dtype=torch.bfloat16)
    state_dict["conv1.norm.weight"] = torch.randn(64, dtype=torch.bfloat16)
    state_dict["conv1.norm.bias"] = torch.randn(64, dtype=torch.bfloat16)
    state_dict["conv1.norm.running_mean"] = torch.randn(64, dtype=torch.bfloat16)
    state_dict["conv1.norm.running_var"] = torch.abs(torch.randn(64, dtype=torch.bfloat16)) + 1e-5

    # Conv2: 64 -> 64 channels, 3x3 kernel, stride=1
    state_dict["conv2.weight"] = torch.randn(64, 64, 3, 3, dtype=torch.bfloat16)
    state_dict["conv2.norm.weight"] = torch.randn(64, dtype=torch.bfloat16)
    state_dict["conv2.norm.bias"] = torch.randn(64, dtype=torch.bfloat16)
    state_dict["conv2.norm.running_mean"] = torch.randn(64, dtype=torch.bfloat16)
    state_dict["conv2.norm.running_var"] = torch.abs(torch.randn(64, dtype=torch.bfloat16)) + 1e-5

    # Conv3: 64 -> 128 channels, 3x3 kernel, stride=1
    state_dict["conv3.weight"] = torch.randn(128, 64, 3, 3, dtype=torch.bfloat16)
    state_dict["conv3.norm.weight"] = torch.randn(128, dtype=torch.bfloat16)
    state_dict["conv3.norm.bias"] = torch.randn(128, dtype=torch.bfloat16)
    state_dict["conv3.norm.running_mean"] = torch.randn(128, dtype=torch.bfloat16)
    state_dict["conv3.norm.running_var"] = torch.abs(torch.randn(128, dtype=torch.bfloat16)) + 1e-5

    return state_dict


def create_random_bottleneck_state_dict(in_channels, bottleneck_channels, out_channels, has_shortcut=False):
    """Create random state dict for BottleneckBlock."""
    state_dict = {}

    # Conv1 weights (1x1 conv)
    state_dict["conv1.weight"] = torch.randn(bottleneck_channels, in_channels, 1, 1, dtype=torch.bfloat16)
    state_dict["conv1.norm.weight"] = torch.randn(bottleneck_channels, dtype=torch.bfloat16)
    state_dict["conv1.norm.bias"] = torch.randn(bottleneck_channels, dtype=torch.bfloat16)
    state_dict["conv1.norm.running_mean"] = torch.randn(bottleneck_channels, dtype=torch.bfloat16)
    state_dict["conv1.norm.running_var"] = torch.abs(torch.randn(bottleneck_channels, dtype=torch.bfloat16)) + 1e-5

    # Conv2 weights (3x3 conv)
    state_dict["conv2.weight"] = torch.randn(bottleneck_channels, bottleneck_channels, 3, 3, dtype=torch.bfloat16)
    state_dict["conv2.norm.weight"] = torch.randn(bottleneck_channels, dtype=torch.bfloat16)
    state_dict["conv2.norm.bias"] = torch.randn(bottleneck_channels, dtype=torch.bfloat16)
    state_dict["conv2.norm.running_mean"] = torch.randn(bottleneck_channels, dtype=torch.bfloat16)
    state_dict["conv2.norm.running_var"] = torch.abs(torch.randn(bottleneck_channels, dtype=torch.bfloat16)) + 1e-5

    # Conv3 weights (1x1 conv)
    state_dict["conv3.weight"] = torch.randn(out_channels, bottleneck_channels, 1, 1, dtype=torch.bfloat16)
    state_dict["conv3.norm.weight"] = torch.randn(out_channels, dtype=torch.bfloat16)
    state_dict["conv3.norm.bias"] = torch.randn(out_channels, dtype=torch.bfloat16)
    state_dict["conv3.norm.running_mean"] = torch.randn(out_channels, dtype=torch.bfloat16)
    state_dict["conv3.norm.running_var"] = torch.abs(torch.randn(out_channels, dtype=torch.bfloat16)) + 1e-5

    # Shortcut weights (if needed)
    if has_shortcut:
        state_dict["shortcut.weight"] = torch.randn(out_channels, in_channels, 1, 1, dtype=torch.bfloat16)
        state_dict["shortcut.norm.weight"] = torch.randn(out_channels, dtype=torch.bfloat16)
        state_dict["shortcut.norm.bias"] = torch.randn(out_channels, dtype=torch.bfloat16)
        state_dict["shortcut.norm.running_mean"] = torch.randn(out_channels, dtype=torch.bfloat16)
        state_dict["shortcut.norm.running_var"] = torch.abs(torch.randn(out_channels, dtype=torch.bfloat16)) + 1e-5

    return state_dict


def create_full_resnet_state_dict():
    """Create complete random state dict for ResNet with all layers."""
    state_dict = {}

    # Stem layers
    stem_state_dict = create_random_stem_state_dict()
    for key, value in stem_state_dict.items():
        state_dict[f"stem.{key}"] = value

    # res2 (3 blocks): 128->256 channels
    for i, (in_channels, has_shortcut) in enumerate([(128, True), (256, False), (256, False)]):
        bottleneck_state_dict = create_random_bottleneck_state_dict(in_channels, 64, 256, has_shortcut)
        for key, value in bottleneck_state_dict.items():
            state_dict[f"res2.{i}.{key}"] = value

    # res3 (4 blocks): 256->512 channels
    for i, (in_channels, has_shortcut) in enumerate([(256, True), (512, False), (512, False), (512, False)]):
        bottleneck_state_dict = create_random_bottleneck_state_dict(in_channels, 128, 512, has_shortcut)
        for key, value in bottleneck_state_dict.items():
            state_dict[f"res3.{i}.{key}"] = value

    # res4 (6 blocks): 512->1024 channels
    for i, (in_channels, has_shortcut) in enumerate(
        [(512, True), (1024, False), (1024, False), (1024, False), (1024, False), (1024, False)]
    ):
        bottleneck_state_dict = create_random_bottleneck_state_dict(in_channels, 256, 1024, has_shortcut)
        for key, value in bottleneck_state_dict.items():
            state_dict[f"res4.{i}.{key}"] = value

    # res5 (3 blocks): 1024->2048 channels
    for i, (in_channels, has_shortcut) in enumerate([(1024, True), (2048, False), (2048, False)]):
        bottleneck_state_dict = create_random_bottleneck_state_dict(in_channels, 512, 2048, has_shortcut)
        for key, value in bottleneck_state_dict.items():
            state_dict[f"res5.{i}.{key}"] = value

    return state_dict
