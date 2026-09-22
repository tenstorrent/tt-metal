# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Shared fixtures for the package's tests.

The mesh is opened **once per session** and shared: opening an 8x4 Galaxy costs far more than any
individual test, and the decoder suite is meant to be runnable in one go. Host-only tests
(``test_reference_*``, ``test_golden_cache``) never request ``galaxy``, so they keep running with
no hardware attached.

The suite is skipped rather than failed when fewer than 32 devices are visible — a reduced mesh
would still "pass" numerically while measuring something other than the target topology, and the
spec's acceptance is defined at 8x4.

Run:

    scripts/run_safe_pytest.sh models/demos/mistral_medium_3_5_128b/tests -s
"""

import pytest
from loguru import logger

import ttnn
from models.demos.mistral_medium_3_5_128b.reference.model_config import MistralMediumConfig, PrefillSpec
from models.demos.mistral_medium_3_5_128b.tests.device_utils import L1_SMALL_SIZE, MESH_SHAPE
from models.demos.mistral_medium_3_5_128b.tt.ccl import CCLManager
from models.demos.mistral_medium_3_5_128b.tt.config import MeshConfig
from models.demos.mistral_medium_3_5_128b.utils.fabric_env import (
    ccl_topology_from_env,
    fabric_config_from_env,
    topology_name,
)


@pytest.fixture(scope="session")
def spec():
    """The binding prefill spec (``PREFILL_SPEC`` if set, else the recorded defaults)."""
    return PrefillSpec.from_env()


@pytest.fixture(scope="session")
def cfg():
    """The full-width model config read from the vendored ``config.json``."""
    return MistralMediumConfig.from_json()


@pytest.fixture(scope="session")
def galaxy():
    """The target 8x4 Blackhole Galaxy mesh, opened once for the whole session."""
    rows, cols = MESH_SHAPE
    ndev = ttnn.get_num_devices()
    if ndev < rows * cols:
        pytest.skip(f"target mesh {rows}x{cols} needs {rows * cols} devices, found {ndev}")

    ttnn.set_fabric_config(fabric_config_from_env())
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(rows, cols), l1_small_size=L1_SMALL_SIZE)
    logger.info(f"[mesh] opened {tuple(mesh.shape)} fabric={ttnn.get_fabric_config()} topology={topology_name()}")
    try:
        yield mesh
    finally:
        ttnn.close_mesh_device(mesh)
        ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)


@pytest.fixture(scope="session")
def mesh_config():
    """``MeshConfig`` at the spec's parallelism: SP=8 on rows, TP=4 on cols."""
    return MeshConfig(MESH_SHAPE, tp=MESH_SHAPE[1])


@pytest.fixture(scope="session")
def ccl(galaxy):
    """One ``CCLManager`` for the session; its semaphores are reset between tests."""
    return CCLManager(galaxy, num_links=2, topology=ccl_topology_from_env())


@pytest.fixture(autouse=True)
def _reset_ccl(request):
    """Reset the CCL semaphores after any test that used them.

    Global semaphores carry state across ops; a test that fails mid-collective would otherwise
    leave the next one reading a stale count and failing for a reason that has nothing to do with
    the code under test.
    """
    yield
    if "ccl" in request.fixturenames:
        request.getfixturevalue("ccl").reset_global_semaphores()
