import ttnn
def test_dist(mesh_device):
    mesh_device.reshape((ttnn.MeshShape(2, 16))

# same error with MeshShape(16, 2)