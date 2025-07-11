# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC.

# SPDX-License-Identifier: Apache-2.0
import os
import tempfile
from pathlib import Path

import pytest
from transformers import AutoConfig

import ttnn


@pytest.fixture
def mesh_row(mesh_device):
    """
    DeepSeek runs many modules on a single 8-device row of a Galaxy system.
    This can be emulated on a T3K or by selecting a single submesh of a TG.

    For Galaxy+ systems (32+ devices), creates a submesh with shape (1, 8)
    and returns the first row. Otherwise, returns the original mesh_device.
    """
    if ttnn.get_num_devices() >= 32:
        rows = mesh_device.create_submeshes(ttnn.MeshShape(1, 8))
        yield rows[0]
    else:
        yield mesh_device


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test outputs."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def hf_config():
    """Load DeepSeek config for testing."""
    path = os.getenv("HF_MODEL", "/proj_sw/user_dev/deepseek-ai/DeepSeek-R1-0528")
    config = AutoConfig.from_pretrained(path, trust_remote_code=True)
    return config


@pytest.fixture
def hf_config_single_layer(hf_config):
    """Load DeepSeek config with a single layerfor testing."""
    hf_config.num_hidden_layers = 1
    return hf_config
