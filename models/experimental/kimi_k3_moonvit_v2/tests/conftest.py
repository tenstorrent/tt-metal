# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""Fixtures. `reference` is the torch golden with random or checkpoint weights; the TT modules
load its `state_dict()`."""

from __future__ import annotations

import pytest
import torch

from models.experimental.kimi_k3_moonvit_v2.reference.checkpoint import (
    CHECKPOINT_ENV,
    checkpoint_path,
    load_vision_state_dict,
)
from models.experimental.kimi_k3_moonvit_v2.reference.kimi_k3_vision import KimiK3VisionModel, load_vision_config
from models.experimental.kimi_k3_moonvit_v2.tests.common import TEST_LAYERS
from models.experimental.kimi_k3_moonvit_v2.tt.model_config import MoonViTConfig


@pytest.fixture(scope="session")
def config() -> MoonViTConfig:
    path = checkpoint_path()
    base = MoonViTConfig.from_hf_config(path) if path is not None else MoonViTConfig()
    return base.with_layers(TEST_LAYERS)


def build_reference(weights: str, num_layers: int | None) -> KimiK3VisionModel:
    """bf16, eval mode, as the checkpoint runs (upstream RMSNorm eps follows the input dtype).

    `num_layers=None` keeps every layer of the checkpoint's (or the default) config.
    """
    path = checkpoint_path()
    if weights == "checkpoint" and path is None:
        pytest.skip(f"set ${CHECKPOINT_ENV} to a Kimi-K3 checkpoint folder")
    vision_config = load_vision_config(path, num_hidden_layers=num_layers)
    torch.manual_seed(0)
    model = KimiK3VisionModel(vision_config)
    if weights == "checkpoint":
        model.load_state_dict(load_vision_state_dict(path, vision_config.vt_num_hidden_layers))
    else:
        # RMSNorm initializes to ones, which would hide a norm weight that fails to load.
        with torch.no_grad():
            for name, param in model.named_parameters():
                if "norm" in name:
                    param.copy_(1.0 + 0.1 * torch.randn_like(param))
    return model.to(torch.bfloat16).eval()


@pytest.fixture(scope="session")
def reference(request) -> KimiK3VisionModel:
    """Every layer; tests pick blocks from it."""
    return build_reference(request.param, None)


@pytest.fixture(scope="session")
def shallow_reference(request) -> KimiK3VisionModel:
    """The first TEST_LAYERS layers, for the full-model tests that `config` describes."""
    return build_reference(request.param, TEST_LAYERS)
