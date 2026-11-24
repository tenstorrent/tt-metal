import ttnn
from eval.golden_tests.all_to_all_matmul.conftest import _FabricMeshCache
from ttnn.operations.all_to_all_matmul import all_to_all_matmul_program_descriptor as pd
from ttnn.operations.examples.fabric_all_gather import program_descriptor_with_inline_kernels as ag

cache = _FabricMeshCache()
mesh = cache.get((1, 4))
try:
    eth = pd._soc(mesh)[0]
    chips = pd._routes(mesh, 1)
    conns = {}
    for coord, ch in chips.items():
        conns[coord] = []
        for d, peer in ((0, ch["next"]), (1, ch["prev"])):
            li = list(ttnn.get_forwarding_link_indices(pd._node(mesh, coord), pd._node(mesh, peer)))[:2]
            for l in li:
                conns[coord].append((peer, l))
                chan = pd._eth_channel(mesh, coord, peer, l)
                print("CHAN", coord, peer, l, chan, eth[chan])
    grid = mesh.compute_with_storage_grid_size()
    allowed = {(x, y) for x in range(grid.x) for y in range(grid.y)}
    res = ag.probe_ethernet_cores(mesh, conns, allowed)
    for k, v in res.items():
        print("PROBE", k, v, ag.eth_noc_coord(mesh, k[0], v))
    plan = pd._topology_plan(mesh, 1, 2)
    for coord, ch in plan.items():
        print("PLAN", coord, ch["ports"], {d: ch["route"]})
except Exception as e:
    import traceback

    traceback.print_exc()
    print("ERR", e)
cache.close()
