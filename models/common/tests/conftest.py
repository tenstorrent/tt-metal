# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0


import contextlib
import fcntl
import os
import time

import pytest

import ttnn

# ==============================================================================
# Device Lock - Coordinates exclusive access to TT devices across processes
# ==============================================================================

_TT_DEVICE_LOCK_PATH = os.environ.get("TT_DEVICE_LOCK_PATH", "/tmp/tt_device.lock")
_TT_DEVICE_LOCK_TIMEOUT = float(os.environ.get("TT_DEVICE_LOCK_TIMEOUT", "60"))  # 1 min default


class DeviceLockTimeout(Exception):
    """Raised when acquiring the device lock times out."""


@contextlib.contextmanager
def tt_device_lock(lock_path: str = _TT_DEVICE_LOCK_PATH, timeout: float = _TT_DEVICE_LOCK_TIMEOUT):
    """
    Context manager for exclusive access to TT devices.

    Uses flock for cross-process coordination. Blocks until lock is acquired
    or timeout is reached.

    Usage:
        with tt_device_lock():
            mesh = ttnn.open_mesh_device(...)
            # ... do work ...
            ttnn.close_mesh_device(mesh)

    Debug stuck locks with: lsof /tmp/tt_device.lock

    Environment variables:
        TT_DEVICE_LOCK_PATH: Override lock file path (default: /tmp/tt_device.lock)
        TT_DEVICE_LOCK_TIMEOUT: Override timeout in seconds (default: 300)
    """
    lock_dir = os.path.dirname(lock_path)
    if lock_dir and not os.path.exists(lock_dir):
        os.makedirs(lock_dir, exist_ok=True)

    lock_file = open(lock_path, "a+")  # open the file in append mode to avoid truncation race condition among processes
    start_time = time.monotonic()
    lock_acquired = False

    try:
        # Poll for lock with timeout
        logged_waiting = False
        while True:
            try:
                fcntl.flock(lock_file, fcntl.LOCK_EX | fcntl.LOCK_NB)
                lock_acquired = True
                break
            except BlockingIOError:
                pass  # Lock held by another process

            if not logged_waiting:
                print(f"[tt_device_lock] Waiting for device lock (held by another process)...")
                print(f"[tt_device_lock] Debug with: lsof {lock_path}")
                logged_waiting = True

            if time.monotonic() - start_time >= timeout:
                lock_file.close()
                raise DeviceLockTimeout(
                    f"Timed out after {timeout}s waiting for device lock. " f"Check: lsof {lock_path}"
                )

            time.sleep(1)  # sleep for 1 second to avoid busy-waiting

        if logged_waiting:
            print(f"[tt_device_lock] Lock acquired after {time.monotonic() - start_time:.1f}s")

        # Write PID for debugging
        lock_file.truncate(0)  # clear the file
        lock_file.write(f"{os.getpid()}\n")
        lock_file.flush()

        yield

    finally:
        if lock_acquired:
            fcntl.flock(lock_file, fcntl.LOCK_UN)
        lock_file.close()


def pytest_collection_modifyitems(config, items):
    """Deselect tests where ttnn_mesh_device fixture doesn't match mesh_shape param.

    This enables tests to use cross-product parametrization (all meshes × all cases)
    while only running the valid combinations, without noisy skip messages.
    """
    selected = []
    deselected = []

    for item in items:
        if not hasattr(item, "callspec"):
            selected.append(item)
            continue

        params = item.callspec.params
        fixture_mesh = params.get("ttnn_mesh_device")
        required_mesh = params.get("mesh_shape")

        # Keep test if no mesh_shape param or if meshes match
        if required_mesh is None or fixture_mesh == required_mesh:
            selected.append(item)
        else:
            deselected.append(item)

    items[:] = selected
    if deselected:
        config.hook.pytest_deselected(items=deselected)


@pytest.fixture(scope="module")
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

    # Pre-check: if no devices at all, skip without invoking C++ open.
    # Some environments can throw here (e.g. transient driver/UMD issues); treat as "device unavailable".
    try:
        num_pcie = ttnn.get_num_pcie_devices()
    except Exception as e:
        pytest.skip(f"{__file__}: Unable to query TT devices on this system: {e}")

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

    parent_shape = _pick_parent_shape_for_submesh(sys_shape, req_shape)

    # config fabric config
    fabric_config = updated_params.pop("fabric_config", None)
    if parent_shape == (1, 1):
        # Single device does not need fabric config.
        pass
    else:
        # Provide default fabric config for the mesh we actually open (full system mesh).
        num_devices = parent_shape[0] * parent_shape[1]
        if fabric_config is None:
            if num_devices >= 8:
                fabric_config = ttnn.FabricConfig.FABRIC_1D_RING
            else:
                fabric_config = ttnn.FabricConfig.FABRIC_1D
        # set all other input arguments to default values by top-level conftest.py
        ttnn.set_fabric_config(
            fabric_config, ttnn.FabricReliabilityMode.STRICT_INIT, None, ttnn.FabricTensixConfig.DISABLED
        )

    # config dispatch core to default values by conftest.py
    updated_params["dispatch_core_config"] = ttnn.DispatchCoreConfig(type=None, axis=None, fabric_tensix_config=None)

    # If a test requests a submesh of a larger system mesh (e.g. request 2x4 on a 8x4 system),
    # fabric cannot be initialized on only the subset of devices. In that case, open the full
    # system mesh first, then return the "first" submesh. We intentionally rely on the default
    # offset behavior here (i.e. no explicit offset selection).
    parent_device = None
    submesh_device = None

    # Acquire exclusive lock to prevent concurrent device access across processes
    with tt_device_lock():
        try:
            if req_shape != parent_shape:
                parent_device = ttnn.open_mesh_device(mesh_shape=ttnn.MeshShape(parent_shape), **updated_params)
                submesh_device = parent_device.create_submesh(ttnn.MeshShape(req_shape))
                yield submesh_device
            else:
                parent_device = ttnn.open_mesh_device(mesh_shape=ttnn.MeshShape(parent_shape), **updated_params)
                yield parent_device
        except Exception as e:
            pytest.skip(f"{__file__}: Mesh device unavailable or unsupported for this configuration: {e}")
        finally:
            if submesh_device is not None:
                ttnn.close_mesh_device(submesh_device)
            if parent_device is not None:
                ttnn.close_mesh_device(parent_device)
            if fabric_config:
                ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)
            del parent_device


def _allowed_req_shapes_for_system(sys_shape: tuple[int, int]) -> set[tuple[int, int]]:
    # todo)) Different cluster has potentially different physical interconnects (in terms of number of links, topology, etc.).
    #        Thus, a tuple of ints may not be enough to fingerprint the parent/system mesh device. We need to use a more sophisticated fingerprinting mechanism so we can base the allowed list of (sub)mesh shapes on the parent/system mesh device.
    # [INFO] The most robust way to identify the underlying system is to use ttnn.cluster.get_cluster_type(), which returns a ClusterType enum that precisely identifies your hardware configuration. cluster.cpp:16-37

    _CANDIDATE_REQ_SHAPES = {
        (1, 1): ((1, 1),),
        (1, 2): ((1, 2), (1, 1)),
        (2, 4): ((2, 4), (1, 8), (1, 4), (1, 2), (1, 1)),
        (8, 4): ((8, 4), (4, 8), (1, 8), (1, 4), (1, 2), (1, 1)),
        # [INFO] add more system shapes here
    }

    allowed: set[tuple[int, int]] = set()

    if sys_shape in _CANDIDATE_REQ_SHAPES:
        for mesh_shape in _CANDIDATE_REQ_SHAPES[sys_shape]:
            allowed.add(mesh_shape)

    return allowed


def _pick_parent_shape_for_submesh(system_shape: tuple[int, int], requested_shape: tuple[int, int]) -> tuple[int, int]:
    # For multi-device workloads we always open the full system mesh (fabric cannot be launched on a subset),
    # but we may choose the *orientation* of the full mesh such that the requested submesh fits with the
    # default offset (i.e. "first submesh").
    if requested_shape == (1, 1):
        return (1, 1)

    # If the request uses all devices, treat it as a "full-mesh view" shape and open the parent mesh in that view.
    # This enables shapes like (1,32) on a system whose SystemMeshDescriptor reports (8,4).
    system_num_devices = system_shape[0] * system_shape[1]
    requested_num_devices = requested_shape[0] * requested_shape[1]
    if requested_num_devices == system_num_devices:
        return requested_shape

    if requested_shape[0] <= system_shape[0] and requested_shape[1] <= system_shape[1]:
        return system_shape

    rotated = (system_shape[1], system_shape[0])
    if requested_shape[0] <= rotated[0] and requested_shape[1] <= rotated[1]:
        return rotated

    # No orientation can fit this request without an explicit offset / mapping.
    pytest.skip(
        f"{__file__}: Requested submesh {requested_shape} does not fit within system mesh {system_shape} "
        f"(or its rotated view {rotated}) with default offset."
    )
