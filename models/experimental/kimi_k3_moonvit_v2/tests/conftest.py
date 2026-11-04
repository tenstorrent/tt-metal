# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""Fixtures. `reference` is the torch golden with random or checkpoint weights; the TT modules
load its `state_dict()`. `mesh_device` replaces the repo-wide fixture for this folder (see there)."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

import ttnn
from models.experimental.kimi_k3_moonvit_v2.reference.checkpoint import (
    CHECKPOINT_ENV,
    checkpoint_path,
    load_vision_state_dict,
)
from models.experimental.kimi_k3_moonvit_v2.reference.kimi_k3_vision import KimiK3VisionModel, load_vision_config
from models.experimental.kimi_k3_moonvit_v2.tests.common import (
    LOUDBOX_SHAPE,
    ON_SIMULATOR,
    RING_SIZE,
    SIMULATOR_MARK,
    TEST_LAYERS,
)
from models.experimental.kimi_k3_moonvit_v2.tt.model_config import MoonViTConfig
from tests.scripts.common import get_updated_device_params


TESTS_DIR = Path(__file__).parent


def pytest_configure(config):
    config.addinivalue_line("markers", f"{SIMULATOR_MARK}: sized for ttsim (`SIMULATOR` in common.py)")


def pytest_collection_modifyitems(config, items):
    """On ttsim, skip this folder's device tests that are not marked `simulator`. CPU-only tests
    (no `mesh_device`) still run."""
    if not ON_SIMULATOR:
        return
    skip = pytest.mark.skip(reason=f"not a sim test (no `{SIMULATOR_MARK}` mark); too slow on ttsim")
    for item in items:
        in_folder = TESTS_DIR in Path(item.path).parents
        if in_folder and "mesh_device" in item.fixturenames and item.get_closest_marker(SIMULATOR_MARK) is None:
            item.add_marker(skip)


@pytest.fixture
def mesh_device(request, device_params):
    """`(1, 1)`: one chip, opened on its own. `(1, RING_SIZE)`: the first row of the LoudBox's 2x4 mesh.

    All 8 chips are opened for the ring because fabric setup on a chip waits for every one of its
    ethernet links, including those to the other row, and times out if those chips are not opened
    too. Replaces the repo-wide fixture rather than wrapping it: a wrapper would take the mesh
    parameter, and the repo fixture would then open its default mesh."""
    shape = tuple(request.param)
    open_shape = LOUDBOX_SHAPE if shape == (1, RING_SIZE) else shape
    num_chips = open_shape[0] * open_shape[1]
    if ttnn.get_num_devices() < num_chips:
        pytest.skip(f"needs {num_chips} chips, the machine has {ttnn.get_num_devices()}")

    params = get_updated_device_params(device_params)
    fabric_config = params.pop("fabric_config", None)
    if fabric_config:
        ttnn.set_fabric_config(fabric_config, ttnn.FabricReliabilityMode.STRICT_INIT)
    parent = ttnn.open_mesh_device(mesh_shape=ttnn.MeshShape(*open_shape), **params)
    mesh = parent
    if open_shape != shape:
        mesh = parent.create_submesh(ttnn.MeshShape(*shape), ttnn.MeshCoordinate(0, 0))

    yield mesh

    if mesh is not parent:
        ttnn.close_mesh_device(mesh)
    ttnn.close_mesh_device(parent)
    if fabric_config:
        ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)


@pytest.fixture(scope="session")
def config() -> MoonViTConfig:
    """Every layer, from the checkpoint's `config.json` when $KIMI_K3_HF_MODEL is set."""
    path = checkpoint_path()
    return MoonViTConfig.from_hf_config(path) if path is not None else MoonViTConfig()


@pytest.fixture(scope="session")
def shallow_config(config) -> MoonViTConfig:
    """The first TEST_LAYERS layers, matching `shallow_reference`."""
    return config.with_layers(TEST_LAYERS)


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
    """The first TEST_LAYERS layers, for the full-model tests that `shallow_config` describes."""
    return build_reference(request.param, TEST_LAYERS)
