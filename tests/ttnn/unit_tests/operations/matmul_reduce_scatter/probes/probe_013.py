import ttnn
from tests.scripts.common import get_updated_device_params
from ttnn.operations.examples.fabric_all_gather.program_descriptor_with_inline_kernels import _chip_maps

ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_2D, ttnn.FabricReliabilityMode.STRICT_INIT)
params = get_updated_device_params({"fabric_config": ttnn.FabricConfig.FABRIC_2D})
params.pop("fabric_config")
mesh = ttnn.open_mesh_device(mesh_shape=ttnn.MeshShape(2, 4), **params)
try:
    node = lambda r, c: mesh.get_fabric_node_id(ttnn.MeshCoordinate(r, c))
    for a, b in [((0, 0), (0, 1)), ((0, 1), (0, 0)), ((0, 1), (0, 2)), ((0, 2), (0, 1)), ((0, 0), (1, 0))]:
        links = ttnn.get_forwarding_link_indices(node(*a), node(*b))
        for l in links:
            pd = ttnn.ProgramDescriptor()
            args = list(ttnn.setup_fabric_connection(node(*a), node(*b), l, pd, ttnn.CoreCoord(0, 0)))
            eth_list, translated, t2p = _chip_maps(mesh, a)
            print(
                a,
                "->",
                b,
                "link",
                l,
                "chan",
                args[0],
                "eth phys",
                eth_list[args[0]] if eth_list and args[0] < len(eth_list) else None,
                "nargs",
                len(args),
                "translated",
                translated,
            )
    eth_list, translated, t2p = _chip_maps(mesh, (0, 0))
    print("eth list", eth_list)
    print("t2p", t2p)
finally:
    ttnn.close_mesh_device(mesh)
    ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)
