# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Pytest configuration for the Qwen3.5/Qwen3.6 demo test suite.

Adds two CLI options:

* ``--max-prefill`` — cap on the prefill length that routine runs exercise. Any
  parametrized case whose ``seq_len`` / ``actual_len`` / ``T`` exceeds the cap is
  auto-skipped, so the long-context tests (T=4096, 73728, …) don't run unless the
  cap is raised.
* ``--test-modules`` — comma-separated module selector (kept for future use).

These layer on top of the repo-root ``conftest.py`` (which provides ``device``,
``mesh_device``, ``device_params``, ``reset_seeds``, ``ensure_gc``).
"""

import pytest
from loguru import logger


def pytest_addoption(parser):
    parser.addoption(
        "--max-prefill",
        action="store",
        type=int,
        default=8192,
        help="Max prefill length to run; longer parametrized cases auto-skip (raise to run long-context tests).",
    )
    parser.addoption(
        "--test-modules",
        action="store",
        default="all",
        help="Comma-separated modules to run (attention,gdn,mlp,rms_norm,rope,layer,model). Default: all.",
    )


@pytest.fixture
def test_modules(request):
    return request.config.getoption("--test-modules")


@pytest.fixture(autouse=True)
def _enforce_max_prefill(request):
    """Skip parametrized cases whose sequence length exceeds --max-prefill.

    Decode (length 1) always runs. qwen tests express the prefill length under a
    few different param names, so all three are checked.
    """
    callspec = getattr(request.node, "callspec", None)
    if callspec is None:
        return
    cap = request.config.getoption("--max-prefill")
    for key in ("seq_len", "actual_len", "T"):
        val = callspec.params.get(key)
        if isinstance(val, int) and val > cap:
            pytest.skip(f"{key}={val} > --max-prefill={cap}")


@pytest.fixture(scope="function")
def mesh_device(request, device_params):
    """Galaxy-aware override of the repo-root ``mesh_device`` fixture.

    On a Blackhole Galaxy the fabric routers span every chip in the cluster, so fabric
    only initializes when the WHOLE mesh is opened. Opening a partial mesh directly --
    which is what the root fixture does for ``parametrize_mesh_tp``'s (1,4) / (1,8) --
    fails the ethernet handshake during device init:

        fabric_firmware_initializer.cpp:271 Fabric Router Sync: Timeout after 10000 ms
        on Device 1 ... Ethernet handshake likely failed

    Measured on this 32-chip BH Galaxy: (4,8) opens fine under both FABRIC_1D and
    FABRIC_2D, while (1,2), (1,4), (1,8) and (2,4) all time out. Opening without fabric
    succeeds at any shape, which is why only the collective-using TP tests hit this.

    So: open the full mesh and carve the requested shape out of it with ``create_submesh``.
    The parent stays open for the lifetime of the test (closing it would tear down the
    fabric the submesh is using). On non-Galaxy systems -- single P150, P150x4, P150x8 --
    the requested shape IS the system mesh, so this takes the direct path and behaves
    exactly like the root fixture.
    """
    import ttnn
    from conftest import bh_2d_mesh_device_context

    try:
        param = request.param
    except (ValueError, AttributeError):
        param = None

    system_shape = ttnn._ttnn.multi_device.SystemMeshDescriptor().shape()
    system_size = system_shape.mesh_size()

    if param is None:
        grid_dims = tuple(system_shape)
    elif isinstance(param, tuple):
        assert len(param) == 2, "Device mesh grid shape should have exactly two elements."
        grid_dims = param
    else:
        grid_dims = (1, param)

    requested = grid_dims[0] * grid_dims[1]
    if requested > system_size:
        pytest.skip(f"Requested {requested} devices but the system mesh has {system_size}.")

    with bh_2d_mesh_device_context(device_params) as parent:
        if requested == parent.get_num_devices():
            logger.debug(f"qwen36: using the full {tuple(parent.shape)} mesh directly")
            yield parent
            return

        # create_submeshes (plural) is the routeable split -- it partitions the parent into
        # every group at once, which is what the runtime's 4x8 row-oriented view expects
        # (see tt_transformers.generator._galaxy_data_parallel_submesh_shape). Carving a
        # single submesh with create_submesh(shape, MeshCoordinate(0, 0)) instead yields a
        # mesh whose collectives fail to route: reduce_scatter on it dies with
        # "Could not find any forwarding direction from src (M0, D0) to dst (M0, D28)".
        submeshes = parent.create_submeshes(ttnn.MeshShape(*grid_dims))
        logger.debug(
            f"qwen36: split the {tuple(parent.shape)} parent into {len(submeshes)} x {grid_dims} submeshes; using [0]"
        )
        # Not closed here: bh_2d_mesh_device_context closes every submesh before the parent.
        yield submeshes[0]
