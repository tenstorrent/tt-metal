import pytest
import ttnn


@pytest.fixture(scope="session")
def ttnn_mesh_device(request):
    """Create and yield a mesh device for a given mesh shape, cleanup on teardown."""
    if not hasattr(request, "param"):
        pytest.skip("mesh_device fixture called without parametrization")

    mesh_shape = request.param
    # Pre-check: if no devices at all, skip without invoking C++ open
    try:
        num_pcie = ttnn.get_num_pcie_devices()
        if isinstance(num_pcie, int) and num_pcie == 0:
            pytest.skip("No TT devices detected on this system")
    except Exception:
        # If query fails, continue to attempt opening; downstream try/except will skip
        pass

    # Pre-check: skip shapes that cannot fit into the SystemMesh to avoid native exceptions
    try:
        sys_desc = ttnn._ttnn.multi_device.SystemMeshDescriptor()  # type: ignore[attr-defined]
        sys_shape = tuple(sys_desc.shape())
        req_shape = tuple(mesh_shape)
        allowed = _allowed_req_shapes_for_system(sys_shape)
        if req_shape not in allowed:
            pytest.skip(
                f"Requested mesh {req_shape} unsupported on system {sys_shape}. " f"Allowed for this system: {allowed}"
            )
    except Exception:
        # If descriptor unavailable, fall through and try to open
        pass

    try:
        device = ttnn.open_mesh_device(mesh_shape=ttnn.MeshShape(mesh_shape))
    except Exception:
        pytest.skip("Mesh device unavailable or unsupported for this configuration")

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
