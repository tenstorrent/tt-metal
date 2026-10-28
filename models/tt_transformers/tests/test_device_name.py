# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import pytest

import ttnn
from models.common.device_utils import get_device_name
from models.tt_transformers.tt.model_config import ModelArgs, determine_device_name


def test_device_name_on_device(mesh_device):
    name = determine_device_name(mesh_device)
    assert get_device_name(mesh_device) == name
    if mesh_device.arch() == ttnn.device.Arch.QUASAR:
        assert name == "P150"


@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
@pytest.mark.parametrize("hf_model", ["meta-llama/Llama-3.2-1B-Instruct"])
def test_model_args_take_the_device_name(mesh_device, hf_model, monkeypatch):
    monkeypatch.setenv("HF_MODEL", hf_model)
    monkeypatch.delenv("TT_CACHE_PATH", raising=False)
    args = ModelArgs(mesh_device, dummy_weights=True, max_batch_size=1, max_seq_len=128)
    assert args.device_name == determine_device_name(mesh_device)
