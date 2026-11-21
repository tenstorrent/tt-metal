# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""CPU-only check that the SmolVLA vision loader does not silently leave the tower at random init."""

import importlib
import sys
from unittest import mock

import pytest
import torch
import torch.nn as nn

_STUBBED = (
    "ttnn",
    "models.tt_transformers.tt.common",
    "models.tt_transformers.tt.generator",
    "models.tt_transformers.tt.model_config",
    "models.experimental.smolvla.tt.ttnn_optimized_vit_smolvla",
)


@pytest.fixture(scope="module")
def load_vision_weights_strict():
    """Import the loader; on hardware-free machines stub the device stack only while the module is in use."""
    try:
        import ttnn

        have_ttnn = hasattr(ttnn, "Tensor")
    except Exception:
        have_ttnn = False
    target = "models.experimental.smolvla.tt.smol_vla"
    with pytest.MonkeyPatch.context() as mp:
        if not have_ttnn:
            for name in _STUBBED:
                mp.setitem(sys.modules, name, mock.MagicMock())
            sys.modules.pop(target, None)
        try:
            module = importlib.import_module(target)
            yield module.load_vision_weights_strict
        finally:
            if not have_ttnn:
                # Imported against stubs: drop it so later tests never see a module bound to mocks.
                sys.modules.pop(target, None)


PREFIX = "model.vlm_with_expert.vlm.model.vision_model."


class _Tower(nn.Module):
    def __init__(self):
        super().__init__()
        self.vision_model = nn.Linear(4, 4)


def _state_dict(prefix):
    return {prefix + "weight": torch.ones(4, 4), prefix + "bias": torch.ones(4)}


def test_loads_matching_prefix(load_vision_weights_strict):
    tower = _Tower()
    load_vision_weights_strict(tower, _state_dict(PREFIX))
    assert torch.equal(tower.vision_model.weight, torch.ones(4, 4))


def test_wrong_prefix_raises(load_vision_weights_strict, expect_error):
    with expect_error(KeyError, "vision_model"):
        load_vision_weights_strict(_Tower(), _state_dict("vlm.vision_model."))


def test_missing_key_raises(load_vision_weights_strict, expect_error):
    sd = _state_dict(PREFIX)
    del sd[PREFIX + "bias"]
    with expect_error(RuntimeError, "missing from state dict"):
        load_vision_weights_strict(_Tower(), sd)


class _TowerWithBuffer(_Tower):
    def __init__(self):
        super().__init__()
        self.register_buffer("position_ids", torch.arange(4))


def test_missing_buffer_key_is_tolerated(load_vision_weights_strict):
    tower = _TowerWithBuffer()
    load_vision_weights_strict(tower, _state_dict(PREFIX))
    assert torch.equal(tower.vision_model.weight, torch.ones(4, 4))
    assert torch.equal(tower.position_ids, torch.arange(4))
