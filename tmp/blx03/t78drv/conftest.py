# t78 blx03 only (untracked, removed with ~/fasth3/t78): run the single-device conv3d halo tests on a 2x4
# submesh of the full 4x8 mesh, since a bare single-chip or 2x4 open is not allowed on the galaxy.
import pytest

import ttnn


def pytest_generate_tests(metafunc):
    if metafunc.function.__name__.startswith("test_conv3d_halo"):
        metafunc.parametrize("mesh_device", [(4, 8)], indirect=True)
        metafunc.parametrize("device_params", [{"fabric_config": ttnn.FabricConfig.FABRIC_1D}], indirect=True)


@pytest.fixture
def device(mesh_device):
    yield mesh_device.create_submesh(ttnn.MeshShape(2, 4))
