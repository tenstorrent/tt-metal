import ttnn
from tests.scripts.common import get_updated_device_params
from ttnn.operations.examples.fabric_all_gather import program_descriptor_with_inline_kernels as fag

fc = ttnn.FabricConfig.FABRIC_2D
ttnn.set_fabric_config(fc)
p = get_updated_device_params({"fabric_config": fc})
p.pop("fabric_config")
mesh = ttnn.open_mesh_device(mesh_shape=ttnn.MeshShape(2, 4), **p)
try:
    rows, cols = tuple(mesh.shape)
    g = mesh.compute_with_storage_grid_size()
    print("GRID", g.x, g.y)
    print(
        "VX",
        [int(mesh.worker_core_from_logical_core(ttnn.CoreCoord(x, 0)).x) for x in range(g.x)],
        "VY",
        [int(mesh.worker_core_from_logical_core(ttnn.CoreCoord(0, y)).y) for y in range(g.y)],
    )
    node = lambda c: mesh.get_fabric_node_id(ttnn.MeshCoordinate(*c))
    conns = {}
    chans = {}
    for r in range(rows):
        for c in range(cols):
            lst = []
            for pr, pc in ((r, c + 1), (r, c - 1), (r + 1, c), (r - 1, c)):
                if 0 <= pr < rows and 0 <= pc < cols:
                    for l in ttnn.get_forwarding_link_indices(node((r, c)), node((pr, pc))):
                        lst.append(((pr, pc), int(l)))
                        pd = ttnn.ProgramDescriptor()
                        a = ttnn.setup_fabric_connection(node((r, c)), node((pr, pc)), int(l), pd, ttnn.CoreCoord(0, 0))
                        chans[((r, c), (pr, pc), int(l))] = int(a[0])
            conns[(r, c)] = lst
    allowed = fag.allowed_cores(mesh)
    eth = fag.probe_ethernet_cores(mesh, conns, allowed)
    for k in sorted(eth):
        coord = k[0]
        print(
            "CONN",
            k,
            "chan",
            chans[k],
            "probe",
            eth[k],
            "phys",
            fag.eth_noc_coord(mesh, coord, eth[k]),
            "maps",
            fag._chip_maps(mesh, coord)[1:],
        )
    for r in range(rows):
        for c in range(cols):
            print("CHIP", (r, c), fag._chip_maps(mesh, (r, c))[0])
finally:
    ttnn.close_mesh_device(mesh)
