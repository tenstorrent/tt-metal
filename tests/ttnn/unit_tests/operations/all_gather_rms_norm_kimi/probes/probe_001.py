import ttnn
from eval.golden_tests.all_gather_rms_norm_kimi.conftest import _FabricMeshCache
from eval.golden_tests.all_gather_rms_norm_kimi.topology import focus_mesh_shape
from ttnn.operations.examples.fabric_all_gather import program_descriptor_with_inline_kernels as ex
from ttnn.operations.all_gather_rms_norm_kimi import all_gather_rms_norm_kimi_program_descriptor as pd

cache = _FabricMeshCache()
mesh = cache.get(focus_mesh_shape()[0])
geo = pd.build_geometry(mesh, (1, 1, 640, 384), 1, 2)
print("grid", geo.gx, geo.gy, "noc grid", ex._noc_grid(mesh))
G = geo.G
for p in range(G):
    me = mesh.get_fabric_node_id(ttnn.MeshCoordinate(0, p))
    t2p = ex._chip_maps(mesh, (0, p))[2]
    print("chip", p, "t2p", t2p, "eth list", ex._chip_maps(mesh, (0, p))[0])
    for d, peer in ((0, (p + 1) % G), (1, (p - 1) % G)):
        pn = mesh.get_fabric_node_id(ttnn.MeshCoordinate(0, peer))
        links = ttnn.get_forwarding_link_indices(me, pn)
        for l in range(2):
            prog = ttnn.ProgramDescriptor()
            args = ttnn.setup_fabric_connection(me, pn, links[l], prog, ttnn.CoreCoord(0, 0))
            chan = args[0]
            eth = ex._chip_maps(mesh, (0, p))[0][chan]
            near = ex._worker_nearest_noc1(mesh, (0, p), eth, set(), ex.allowed_cores(mesh))
            cur = geo.ports[(l, d)]
            cv = mesh.worker_core_from_logical_core(cur)
            nv = mesh.worker_core_from_logical_core(near)
            print(
                f"  d{d} l{l} links={list(links)} chan={chan} eth={eth} nearest logical={near} virt={nv} | cur logical={cur} virt={cv} phys_x={t2p.get(cv.x,cv.x)}"
            )
for y in range(geo.gy):
    print(y, mesh.worker_core_from_logical_core(ttnn.CoreCoord(0, y)))
cache.close()
