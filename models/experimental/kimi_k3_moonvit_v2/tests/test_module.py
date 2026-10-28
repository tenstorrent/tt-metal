# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""CPU: the TT module tree loads the reference state dict by name, with the device upload stubbed.

Catches renamed or missing parameters and wrong pre-load transforms without a device.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from models.experimental.kimi_k3_moonvit_v2.tests.common import SHALLOW_REFERENCE
from models.experimental.kimi_k3_moonvit_v2.tt import module, vision_model
from models.experimental.kimi_k3_moonvit_v2.tt.module import LoadingError, Module, Parameter, rename_substate


@pytest.fixture
def host_parameters(monkeypatch):
    """Parameters keep the torch tensor instead of uploading it."""

    def load_torch_tensor(self, value):
        if tuple(value.shape) != self.shape:
            raise LoadingError(f"expected shape {self.shape}, got {tuple(value.shape)}")
        self._data = value

    monkeypatch.setattr(Parameter, "load_torch_tensor", load_torch_tensor)
    monkeypatch.setattr(vision_model, "compute_kernel_config", lambda mesh_device: None)
    return SimpleNamespace(shape=(1, 1))


@SHALLOW_REFERENCE
def test_vision_model_loads_reference(host_parameters, config, shallow_reference):
    tt = vision_model.TT_MoonViT_Vision_Model(config, mesh_device=host_parameters)
    state = shallow_reference.state_dict()
    tt.load_torch_state_dict(state)
    assert tt.is_loaded()

    tower = tt.vision_tower
    block = tower.encoder.blocks[0]
    torch.testing.assert_close(block.attention.wqkv.weight.data, state["vision_tower.encoder.blocks.0.wqkv.weight"].T)
    torch.testing.assert_close(block.norm0.weight.data, state["vision_tower.encoder.blocks.0.norm0.weight"][None])
    torch.testing.assert_close(
        tower.patch_embed.proj.weight.data,
        state["vision_tower.patch_embed.proj.weight"].reshape(config.hidden_size, -1).T,
    )
    assert torch.equal(tower.patch_embed.pos_emb_weight, state["vision_tower.patch_embed.pos_emb.weight"])
    torch.testing.assert_close(tt.mm_projector.fc0.weight.data, state["mm_projector.proj.0.weight"].T)
    torch.testing.assert_close(tt.mm_projector.fc1.weight.data, state["mm_projector.proj.2.weight"].T)
    torch.testing.assert_close(tt.mm_projector.post_norm.weight.data, state["mm_projector.post_norm.weight"][None])


@SHALLOW_REFERENCE
def test_rejects_missing_and_unexpected_keys(host_parameters, config, shallow_reference, expect_error):
    state = shallow_reference.state_dict()
    state.pop("vision_tower.encoder.final_layernorm.weight")
    state["vision_tower.encoder.extra.weight"] = torch.zeros(1)
    tt = vision_model.TT_MoonViT_Vision_Model(config, mesh_device=host_parameters)
    with expect_error(LoadingError, "final_layernorm"):
        tt.load_torch_state_dict(state)


def test_rename_substate():
    state = {"wqkv.weight": 1, "wo.weight": 2, "norm0.weight": 3}
    rename_substate(state, "wqkv", "attention.wqkv")
    assert state == {"attention.wqkv.weight": 1, "wo.weight": 2, "norm0.weight": 3}


def test_load_cached(tmp_path, monkeypatch):
    class Leaf(Module):
        def __init__(self):
            super().__init__()
            self.weight = Parameter(shape=(2,), device=None)

    monkeypatch.setattr(Parameter, "load_torch_tensor", lambda self, v: setattr(self, "_data", v))
    monkeypatch.setattr(Parameter, "save", lambda self, path: torch.save(self.data, path))
    monkeypatch.setattr(Parameter, "load", lambda self, path: setattr(self, "_data", torch.load(path)))

    calls = []
    source = lambda: calls.append(1) or {"weight": torch.tensor([1.0, 2.0])}  # noqa: E731
    module.load_cached(Leaf(), tmp_path, source)
    cached = Leaf()
    module.load_cached(cached, tmp_path, source)
    assert len(calls) == 1
    assert cached.is_loaded() and torch.equal(cached.weight.data, torch.tensor([1.0, 2.0]))
