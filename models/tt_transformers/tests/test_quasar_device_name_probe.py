# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

# Probe only: validates the Quasar device-name mapping on a real (simulated) device.

import pytest
from loguru import logger

import ttnn
from models.common.device_utils import get_device_name
from models.tt_transformers.tt.model_config import ModelArgs, determine_device_name


def test_device_name_on_device(mesh_device):
    dram = mesh_device.dram_grid_size()
    logger.info(
        f"arch={mesh_device.arch()} arch_name={ttnn.get_arch_name()} "
        f"num_devices={mesh_device.get_num_devices()} dram_grid=({dram.x},{dram.y})"
    )
    name = get_device_name(mesh_device)
    logger.info(f"get_device_name={name} determine_device_name={determine_device_name(mesh_device)}")
    assert name == determine_device_name(mesh_device)
    if mesh_device.arch() == ttnn.device.Arch.QUASAR:
        assert name == "P150"


@pytest.mark.parametrize("hf_model", ["meta-llama/Llama-3.2-1B-Instruct"])
def test_model_args_on_device(mesh_device, hf_model, monkeypatch):
    monkeypatch.setenv("HF_MODEL", hf_model)
    monkeypatch.delenv("TT_CACHE_PATH", raising=False)
    args = ModelArgs(mesh_device, dummy_weights=True, max_batch_size=1, max_seq_len=128)
    logger.info(f"ModelArgs device_name={args.device_name} arch_name={args.arch_name}")
    assert args.device_name == determine_device_name(mesh_device)
