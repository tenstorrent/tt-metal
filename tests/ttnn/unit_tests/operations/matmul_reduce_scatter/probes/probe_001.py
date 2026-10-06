import ttnn
from tests.scripts.common import get_updated_device_params

shape = ttnn._ttnn.multi_device.SystemMeshDescriptor().shape()
shape = tuple(shape[i] for i in range(shape.dims()))
print("mesh", shape)
ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_2D, ttnn.FabricReliabilityMode.STRICT_INIT)
params = get_updated_device_params({"fabric_config": ttnn.FabricConfig.FABRIC_2D})
params.pop("fabric_config")
mesh = ttnn.open_mesh_device(mesh_shape=ttnn.MeshShape(*shape), **params)
g = mesh.compute_with_storage_grid_size()
print("grid", g.x, g.y)
d = mesh.dram_grid_size()
print("dram", d.x, d.y)
print("payload", ttnn.get_tt_fabric_max_payload_size_bytes())
print(
    "l1 base",
    ttnn.device.GetAllocatorBaseAddress(mesh, ttnn.BufferType.L1)
    if hasattr(ttnn.device, "GetAllocatorBaseAddress")
    else None,
)
print(
    "max unreserved",
    ttnn.device.get_max_worker_l1_unreserved_size()
    if hasattr(ttnn.device, "get_max_worker_l1_unreserved_size")
    else None,
)
print("l1 size", mesh.l1_size_per_core() if hasattr(mesh, "l1_size_per_core") else None)
for y in range(g.y):
    print(
        [
            (
                mesh.worker_core_from_logical_core(ttnn.CoreCoord(x, y)).x,
                mesh.worker_core_from_logical_core(ttnn.CoreCoord(x, y)).y,
            )
            for x in range(g.x)
        ]
    )
node = lambda r, c: mesh.get_fabric_node_id(ttnn.MeshCoordinate(r, c))
print("links (0,0)->(0,1)", ttnn.get_forwarding_link_indices(node(0, 0), node(0, 1)))
print("links (0,0)->(1,0)", ttnn.get_forwarding_link_indices(node(0, 0), node(1, 0)))
print(node(0, 0).mesh_id, node(0, 0).chip_id)
ttnn.close_mesh_device(mesh)
ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)
