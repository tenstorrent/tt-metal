# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""Fixtures. `reference` is the torch golden with random or checkpoint weights; the TT modules
load its `state_dict()`. `mesh_device` replaces the repo-wide fixture for this folder (see there)."""

from __future__ import annotations

from pathlib import Path

import pytest

import ttnn
from models.experimental.kimi_k3_moonvit_v2.reference.checkpoint import checkpoint_path
from models.experimental.kimi_k3_moonvit_v2.reference.kimi_k3_vision import KimiK3VisionModel
from models.experimental.kimi_k3_moonvit_v2.tests.common import (
    LOUDBOX_SHAPE,
    ON_SIMULATOR,
    REAL_MARK,
    RING_SIZE,
    SIMULATOR_MARK,
    SYNTHETIC_MARK,
    WEIGHTS,
    build_reference,
)
from models.experimental.kimi_k3_moonvit_v2.tt.model_config import MoonViTConfig
from tests.scripts.common import get_updated_device_params


TESTS_DIR = Path(__file__).parent


def pytest_configure(config):
    config.addinivalue_line("markers", f"{SIMULATOR_MARK}: sized for ttsim (`SIMULATOR` in common.py)")
    config.addinivalue_line("markers", f"{SYNTHETIC_MARK}: runs with random weights (`SYNTHETIC` in common.py)")
    config.addinivalue_line("markers", f"{REAL_MARK}: runs with the checkpoint's weights (`REAL` in common.py)")


def pytest_generate_tests(metafunc):
    """Runs a test that builds `reference` once per weights mark it carries."""
    if "weights" not in metafunc.fixturenames:
        return
    weights = [mark for mark in WEIGHTS if metafunc.definition.get_closest_marker(mark)]
    if not weights:
        pytest.fail(
            f"{metafunc.definition.nodeid} builds a reference but has neither SYNTHETIC nor REAL", pytrace=False
        )
    metafunc.parametrize("weights", weights, scope="session")


@pytest.hookimpl(tryfirst=True)
def pytest_collection_modifyitems(config, items):
    """Each case keeps only the weights mark it runs with, so `-m synthetic`, `-m real` and `-k` select
    by weights (the decorators put both marks on every case). Runs before pytest's `-m`/`-k` filter.

    On ttsim, skip this folder's device tests that are not marked `simulator`. CPU-only tests
    (no `mesh_device`) still run."""
    for item in items:
        weights = item.callspec.params.get("weights") if hasattr(item, "callspec") else None
        if weights is not None:
            item.own_markers = [m for m in item.own_markers if m.name not in WEIGHTS or m.name == weights]
    if not ON_SIMULATOR:
        return
    skip = pytest.mark.skip(reason=f"not a sim test (no `{SIMULATOR_MARK}` mark); too slow on ttsim")
    for item in items:
        in_folder = TESTS_DIR in Path(item.path).parents
        if in_folder and "mesh_device" in item.fixturenames and item.get_closest_marker(SIMULATOR_MARK) is None:
            item.add_marker(skip)


@pytest.fixture
def mesh_device(request, device_params):
    """`(1, 1)`: one chip, opened on its own. `(1, RING_SIZE)`: the first row of the system mesh, which
    must be RING_SIZE wide: the LoudBox's 2x4 mesh, or a Galaxy's 8x4.

    Every chip of the system mesh is opened for the ring because fabric setup on a chip waits for every
    one of its ethernet links, including those to the other rows, and times out if those chips are not
    opened too. Replaces the repo-wide fixture rather than wrapping it: a wrapper would take the mesh
    parameter, and the repo fixture would then open its default mesh."""
    shape = tuple(request.param)
    open_shape = shape
    if shape == (1, RING_SIZE):
        open_shape = tuple(ttnn._ttnn.multi_device.SystemMeshDescriptor().shape())
        if len(open_shape) != 2 or open_shape[1] != RING_SIZE or open_shape[0] < LOUDBOX_SHAPE[0]:
            pytest.skip(f"needs a system mesh {RING_SIZE} wide with at least {LOUDBOX_SHAPE[0]} rows, got {open_shape}")
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
    """From the checkpoint's `config.json` when $KIMI_K3_HF_MODEL is set, else the K3 defaults."""
    path = checkpoint_path()
    return MoonViTConfig.from_hf_config(path) if path is not None else MoonViTConfig()


@pytest.fixture(scope="session")
def reference(weights) -> KimiK3VisionModel:
    """Every layer; tests pick blocks from it."""
    return build_reference(weights)
