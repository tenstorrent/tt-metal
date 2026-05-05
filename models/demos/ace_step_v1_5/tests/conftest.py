import os

import pytest


def require_ttnn():
    return pytest.importorskip("ttnn")


@pytest.fixture(scope="function")
def mesh_device():
    ttnn = require_ttnn()
    if not os.environ.get("MESH_DEVICE"):
        pytest.skip("Requires TT device (set MESH_DEVICE)")
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 1))
    try:
        yield mesh
    finally:
        ttnn.close_mesh_device(mesh)
