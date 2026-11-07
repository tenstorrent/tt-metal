import ttnn, torch, collections

cfg = ttnn._ttnn.fabric.FabricRouterConfig()
cfg.max_packet_payload_size_bytes = 14 * 1024 + 64
ttnn.set_fabric_config(
    ttnn.FabricConfig.FABRIC_2D_TORUS_XY,
    ttnn.FabricReliabilityMode.STRICT_INIT,
    None,
    ttnn.FabricTensixConfig.DISABLED,
    ttnn.FabricUDMMode.DISABLED,
    ttnn.FabricManagerMode.DEFAULT,
    cfg,
)
mesh = ttnn.open_mesh_device(mesh_shape=ttnn.MeshShape(1, 4))
from ttnn.operations.all_gather_rms_norm import all_gather_rms_norm

x = ttnn.from_torch(
    torch.randn(1, 1, 640, 1536),
    dtype=ttnn.bfloat16,
    layout=ttnn.TILE_LAYOUT,
    device=mesh,
    mesh_mapper=ttnn.ShardTensorToMesh(mesh, dim=3),
)
all_gather_rms_norm(x, cluster_axis=1)  # warm: global semaphore
for trial in range(2):
    mesh.disable_and_clear_program_cache()
    ttnn.graph._cpp_begin_graph_capture(ttnn.graph.RunMode.NORMAL)
    y = all_gather_rms_norm(x, cluster_axis=1)
    tr = ttnn.graph._cpp_end_graph_capture()
    c = collections.Counter(n.get("node_type") for n in tr)
    print("NODE", trial, dict(c))
    cbs = [n for n in tr if n.get("node_type") == "circular_buffer_allocate"]
    print("cbs", len(cbs), cbs[:2])
    u = ttnn.graph.extract_resource_usage_per_core(tr)
    print("PEAK", trial, u.peak_total)
    mesh.enable_program_cache()
ttnn.close_mesh_device(mesh)
ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)
