# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import pytest

import ttnn


@pytest.fixture(scope="session")
def ttnn_mesh_device(request):
    """Create and yield a mesh device for a given mesh shape, cleanup on teardown."""
    if not hasattr(request, "param"):
        pytest.skip(f"{__file__}: mesh_device fixture called without parametrization")

    if ttnn.device.is_blackhole():
        pytest.skip(f"{__file__}: Blackhole device is not supported for this test yet")

    # request.param is either a Sequence of ints or a dict with fabric_config and etc.
    params = getattr(request, "param", tuple())
    if isinstance(params, tuple):
        mesh_shape = params
        updated_params = dict()
    else:
        try:
            updated_params = params.copy()
            mesh_shape = updated_params.pop("mesh_shape")
        except Exception as e:
            pytest.skip(f"{__file__}: mesh_shape is required: {e}")

    # Pre-check: if no devices at all, skip without invoking C++ open
    num_pcie = ttnn.get_num_pcie_devices()
    if isinstance(num_pcie, int) and num_pcie == 0:
        pytest.skip(f"{__file__}: No TT devices detected on this system")

    # Pre-check: skip shapes that cannot fit into the SystemMesh to avoid native exceptions
    sys_desc = ttnn._ttnn.multi_device.SystemMeshDescriptor()  # type: ignore[attr-defined]
    sys_shape = tuple(sys_desc.shape())
    req_shape = tuple(mesh_shape)
    allowed = _allowed_req_shapes_for_system(sys_shape)
    if req_shape not in allowed:
        pytest.skip(
            f"{__file__}: Requested mesh {req_shape} unsupported on system {sys_shape}. "
            f"Allowed for this system: {allowed}"
        )

    # config fabric config
    fabric_config = updated_params.pop("fabric_config", None)
    if req_shape == (1, 1):
        # single device does not need fabric config
        pass
    else:
        # provide default fabric config for multi-device if not specified by request
        # todo)) ttnn.FabricConfig.FABRIC_1D_RING is the default for Galaxy 6U
        fabric_config = ttnn.FabricConfig.FABRIC_1D if fabric_config is None else fabric_config
        # set all other input arguments to default values by top-level conftest.py
        ttnn.set_fabric_config(
            fabric_config, ttnn.FabricReliabilityMode.STRICT_INIT, None, ttnn.FabricTensixConfig.DISABLED
        )

    # config dispatch core to default values by conftest.py
    updated_params["dispatch_core_config"] = ttnn.DispatchCoreConfig(type=None, axis=None, fabric_tensix_config=None)

    try:
        device = ttnn.open_mesh_device(mesh_shape=ttnn.MeshShape(mesh_shape), **updated_params)
    except Exception as e:
        pytest.skip(f"{__file__}: Mesh device unavailable or unsupported for this configuration: {e}")

    try:
        yield device
    finally:
        ttnn.close_mesh_device(device)


def _allowed_req_shapes_for_system(sys_shape: tuple[int, int]) -> set[tuple[int, int]]:
    """Recursively derive allowed requested shapes by traversing the candidate graph.

    We start from both orientations of the system shape and walk the
    `_CANDIDATE_REQ_SHAPES` graph, collecting reachable shapes. Finally,
    we keep only shapes that physically fit within the system shape (allowing rotation).
    """

    _CANDIDATE_REQ_SHAPES = {
        (1, 1): ((1, 1),),
        (1, 2): ((1, 2), (1, 1)),
        (1, 8): ((1, 8), (2, 4), (1, 2), (1, 1)),
        (2, 4): ((2, 4), (1, 8), (1, 2), (1, 1)),
        # [INFO] add more system shapes here
    }

    allowed: set[tuple[int, int]] = set()

    if sys_shape in _CANDIDATE_REQ_SHAPES:
        for mesh_shape in _CANDIDATE_REQ_SHAPES[sys_shape]:
            allowed.add(mesh_shape)

    return allowed
