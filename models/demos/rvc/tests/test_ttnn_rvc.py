// File: models/demos/rvc/tests/test_ttnn_rvc.py
# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: MIT

import pytest
import torch
import ttnn
import numpy as np
import json
from models.demos.rvc.ttnn_rvc import TtnnRVC
from tests.ttnn.utils_for_testing import assert_with_pcc


@pytest.fixture(scope="module")
def device():
    device = ttnn.open_device(device_id=0)
    yield device
    ttnn.close_device(device)


def test_ttnn_rvc_end_to_end(device):
    # Mock state dict
    state_dict = {}
    state_dict["encoder.pre_net.conv_0.weight"] = torch.randn(192, 513, 3)
    state_dict["encoder.pre_net.conv_0.bias"] = torch.randn(192)
    state_dict["encoder.resblocks.0.conv1.weight"] = torch.randn(192, 192, 3)
    state_dict["encoder.resblocks.0.cond_layer_norm.weight"] = torch.randn(192, 256)
    state_dict["encoder.resblocks.0.conv2.weight"] = torch.randn(192, 192, 3)
    state_dict["encoder.resblocks.1.conv1.weight"] = torch.randn(192, 192, 3)
    state_dict["encoder.resblocks.1.cond_layer_norm.weight"] = torch.randn(192, 256)
    state_dict["encoder.resblocks.1.conv2.weight"] = torch.randn(192, 192, 3)
    state_dict["encoder.post_net.conv_0.weight"] = torch.randn(192, 192, 3)

    # Vocoder weights
    state_dict["vocoder.conv_pre.weight"] = torch.randn(512, 192, 7)
    state_dict["vocoder.conv_pre.bias"] = torch.randn(512)
    for i in range(5):
        state_dict[f"vocoder.upsampler.{i}.weight"] = torch.randn(512, 512, 16)
        state_dict[f"vocoder.upsampler.{i}.bias"] = torch.randn(512)
    for i in range(15):
        state_dict[f"vocoder.resblocks.{i}.conv_r1.weight"] = torch.randn(512, 512, 3)
        state_dict[f"vocoder.resblocks.{i}.conv_r1.bias"] = torch.randn(512)
        state_dict[f"vocoder.resblocks.{i}.conv_r2.weight"] = torch.randn(512, 512, 9)
        state_dict[f"vocoder.resblocks.{i}.conv_r2.bias"] = torch.randn(512)
    state_dict["vocoder.conv_post.weight"] = torch.randn(1, 512, 7)
    state_dict["vocoder.conv_post.bias"] = torch.randn(1)

    # Create index file using JSON instead of pickle
    index_data = {"feature_bank": np.random.rand(1000, 256).astype(np.float32).tolist()}
    index_file = "test_feature_index.json"
    with open(index_file, "w") as f:
        json.dump(index_data, f)

    # Model config
    model_config = {
        "feature_bank_shape": (1000, 256),
        "upsample_rates": [8, 8, 4, 2, 2],
        "upsample_kernel_sizes": [16, 16, 8, 4, 4],
        "reader_patterns_cache": {},
        "conv_cache": {},
    }

    # Create model
    model = TtnnRVC(device, state_dict, index_file, model_config)

    # Test input
    test_audio = torch.randn(513, 100)

    # Run model
    output = model(test_audio, search_ratio=0.75)

    # Check output shape
    assert output.shape == (1, 102400)  # 100 * 8 * 8 * 4 * 2 * 2

    # Cleanup
    import os

    os.remove(index_file)