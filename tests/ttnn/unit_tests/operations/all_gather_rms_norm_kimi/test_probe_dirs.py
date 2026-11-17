import ttnn


def test_dirs(mesh_device):
    rows, cols = tuple(mesh_device.shape)
    n = lambda c: mesh_device.get_fabric_node_id(ttnn.MeshCoordinate(0, c))
    for a in range(cols):
        for b in range(cols):
            if a != b:
                print(
                    "DIR",
                    a,
                    b,
                    ttnn.get_eth_forwarding_direction(n(a), n(b)),
                    list(ttnn.get_forwarding_link_indices(n(a), n(b))),
                    n(a),
                )
