import ttnn
from eval.golden_tests.all_to_all_matmul.conftest import _FabricMeshCache
from ttnn.operations.all_to_all_matmul import all_to_all_matmul_program_descriptor as pd

cache = _FabricMeshCache()
mesh = cache.get((1, 4), payload_bytes=7232)
chips = pd._topology_plan(mesh, 1, 2)
for c, ch in sorted(chips.items()):
    print("CHIP", c, "p", ch["p"], "route", ch["route"], "ports", ch["ports"], "links", ch["links"])
cache.close()
