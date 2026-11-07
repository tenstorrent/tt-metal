import ttnn

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
mesh = ttnn.open_mesh_device(mesh_shape=ttnn.MeshShape(2, 2))
import ttnn.operations.all_gather_rms_norm.all_gather_rms_norm_program_descriptor as PD
from eval.sharding import auto_shard_config

g = mesh.compute_with_storage_grid_size()
print("grid", g.x, g.y)
for ax in (0, 1):
    for L in (1, 2):
        chips, G = PD.plan(mesh, ax, False, L)
        print("axis", ax, "L", L, {c: sorted(ch.ports.values()) for c, ch in chips.items()})
T = ttnn.TensorMemoryLayout
for sh in [
    (1, 1, 640, 384),
    (1, 1, 64, 512),
    (1, 1, 32, 8192),
    (1, 1, 2048, 1280),
    (1, 1, 17, 50),
    (4, 128, 47),
    (1, 1, 32, 64),
]:
    for ml in (T.HEIGHT_SHARDED, T.WIDTH_SHARDED):
        for s in (sh, (*sh[:-1], 2 * sh[-1])):
            mc = auto_shard_config(list(s), ml, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16, device=mesh)
            ss = mc.shard_spec
            print(s, str(ml).split(".")[-1], list(ss.shape), ss.grid.num_cores(), ss.grid)
ttnn.close_mesh_device(mesh)
