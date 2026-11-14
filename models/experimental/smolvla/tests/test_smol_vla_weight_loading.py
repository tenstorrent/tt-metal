# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""CPU-only check that the SmolVLA vision loader does not silently leave the tower at random init."""

import sys
from unittest import mock

import torch
import torch.nn as nn

try:
    import ttnn

    _have_ttnn = hasattr(ttnn, "Tensor")
except Exception:
    _have_ttnn = False
if not _have_ttnn:
    # Hardware-free environments: stub the device stack so the module imports on CPU.
    for _name in (
        "ttnn",
        "models.tt_transformers.tt.common",
        "models.tt_transformers.tt.generator",
        "models.tt_transformers.tt.model_config",
        "models.experimental.smolvla.tt.ttnn_optimized_vit_smolvla",
    ):
        sys.modules[_name] = mock.MagicMock()

from models.experimental.smolvla.tt.smol_vla import load_vision_weights_strict  # noqa: E402

PREFIX = "model.vlm_with_expert.vlm.model.vision_model."


class _Tower(nn.Module):
    def __init__(self):
        super().__init__()
        self.vision_model = nn.Linear(4, 4)


def _state_dict(prefix):
    return {prefix + "weight": torch.ones(4, 4), prefix + "bias": torch.ones(4)}


def test_loads_matching_prefix():
    tower = _Tower()
    load_vision_weights_strict(tower, _state_dict(PREFIX))
    assert torch.equal(tower.vision_model.weight, torch.ones(4, 4))


def test_wrong_prefix_raises(expect_error):
    with expect_error(KeyError, "vision_model"):
        load_vision_weights_strict(_Tower(), _state_dict("vlm.vision_model."))


def test_missing_key_raises(expect_error):
    sd = _state_dict(PREFIX)
    del sd[PREFIX + "bias"]
    with expect_error(RuntimeError, "missing from state dict"):
        load_vision_weights_strict(_Tower(), sd)
