import ttnn
from eval.golden_tests.all_to_all_matmul.conftest import _FabricMeshCache

cache = _FabricMeshCache()
mesh = cache.get((1, 4), payload_bytes=7232)
n = lambda c: mesh.get_fabric_node_id(ttnn.MeshCoordinate(0, c))
for a in range(4):
    for b in range(4):
        if a != b:
            print("DIR", a, b, ttnn.get_eth_forwarding_direction(n(a), n(b)), n(a))
cache.close()
