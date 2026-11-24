import ttnn
from eval.golden_tests.all_to_all_matmul.conftest import _FabricMeshCache
from ttnn.operations.all_to_all_matmul import all_to_all_matmul_program_descriptor as pd

cache = _FabricMeshCache()
mesh = cache.get((1, 4))
try:
    plan = pd._topology_plan(mesh, 1, 2)
    for coord, ch in plan.items():
        for (d, l), xy in ch["ports"].items():
            peer = ch["next"] if d == 0 else ch["prev"]
            chan = pd._eth_channel(mesh, coord, peer, ch["links"][d][l])
            print(
                "CHK",
                coord,
                (d, l),
                "eth",
                pd._chip_maps(mesh, coord)[0][chan],
                "port",
                xy,
                mesh.worker_core_from_logical_core(ttnn.CoreCoord(*xy)),
            )
except Exception as e:
    import traceback

    traceback.print_exc()
    print("ERR", e)
cache.close()
