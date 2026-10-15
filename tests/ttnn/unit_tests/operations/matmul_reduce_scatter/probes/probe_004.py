import os, yaml, ttnn
from tests.ttnn.unit_tests.operations.matmul_reduce_scatter.test_matmul_reduce_scatter import _open_mesh, _close_mesh
from ttnn.operations.examples.fabric_all_gather import program_descriptor_with_inline_kernels as fag

mesh = _open_mesh((1, 4), ttnn.FabricConfig.FABRIC_2D_TORUS_XY)
try:
    soc = yaml.safe_load(open(os.path.join(os.getcwd(), "tt_metal/soc_descriptors/blackhole_140_arch.yaml")))
    arch_eth = [tuple(int(v) for v in e.split("-")) for e in soc["eth"]]
    node = lambda c: mesh.get_fabric_node_id(ttnn.MeshCoordinate(*c))
    conns = {}
    for c in range(4):
        coord = (0, c)
        conns[coord] = []
        for peer in ((0, (c + 1) % 4), (0, (c - 1) % 4)):
            for link in ttnn.get_forwarding_link_indices(node(coord), node(peer))[:2]:
                conns[coord].append((peer, link))
    probed = fag.probe_ethernet_cores(mesh, conns, fag.allowed_cores(mesh))
    for coord, lst in conns.items():
        for peer, link in lst:
            prog = ttnn.ProgramDescriptor()
            ch = list(ttnn.setup_fabric_connection(node(coord), node(peer), link, prog, ttnn.CoreCoord(0, 0)))[0]
            phys = fag.eth_noc_coord(mesh, coord, probed[(coord, peer, link)])
            print(
                "MAP",
                coord,
                peer,
                link,
                "chan",
                ch,
                "arch_eth[chan]",
                arch_eth[ch] if ch < len(arch_eth) else None,
                "probe phys",
                phys,
            )
except Exception as e:
    print("ERR", repr(e))
finally:
    _close_mesh(mesh)
