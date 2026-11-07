import ttnn, torch

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

T = ttnn.TensorMemoryLayout
orig = PD.native_shard_split


def spy(geo, a, b):
    r = orig(geo, a, b)
    print(
        "NATIVE",
        r.kind if r else None,
        (r.split.k, r.split.n_groups, r.split.w_pitch, r.rows_per_group, len(r.cores)) if r else "",
    )
    return r


PD.native_shard_split = spy
origplan = PD.plan


def spyplan(m, ax, ring, L, avoid=frozenset()):
    r = origplan(m, ax, ring, L, avoid)
    if avoid:
        print("SH ports", sorted({c for ch in r[0].values() for c in ch.ports.values()}))
    return r


PD.plan = spyplan
for sh in [
    (1, 1, 640, 384),
    (1, 1, 64, 512),
    (1, 1, 32, 8192),
    (1, 1, 2048, 1280),
    (1, 1, 17, 50),
    (4, 128, 47),
    (1, 1, 32, 64),
    (1, 1, 128, 4096),
]:
    for ml in (T.HEIGHT_SHARDED, T.WIDTH_SHARDED):
        try:
            imc = auto_shard_config(list(sh), ml, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16, device=mesh)
            omc = auto_shard_config(
                [*sh[:-1], 2 * sh[-1]], ml, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16, device=mesh
            )
            x = (
                ttnn.from_torch(
                    torch.randn(2 * sh[0], 2 * sh[1], *sh[2:]) if len(sh) == 4 else torch.randn(*sh),
                    dtype=ttnn.bfloat16,
                    layout=ttnn.TILE_LAYOUT,
                    device=mesh,
                    memory_config=imc,
                    mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
                )
                if False
                else None
            )
            print("SH", sh, str(ml).split(".")[-1])
            ti = ttnn.allocate_tensor_on_device(ttnn.Shape(list(sh)), ttnn.bfloat16, ttnn.TILE_LAYOUT, mesh, imc)
            to = ttnn.allocate_tensor_on_device(
                ttnn.Shape([*sh[:-1], 2 * sh[-1]]), ttnn.bfloat16, ttnn.TILE_LAYOUT, mesh, omc
            )
            PD.create_mesh_program_descriptor(
                ti,
                None,
                to,
                cluster_axis=1,
                ring=False,
                L=1,
                epsilon=1e-6,
                compute_kernel_config=ttnn.ComputeConfigDescriptor(),
            )
            ti.deallocate()
            to.deallocate()
        except Exception as e:
            print("ERR", sh, ml, e)
ttnn.close_mesh_device(mesh)
