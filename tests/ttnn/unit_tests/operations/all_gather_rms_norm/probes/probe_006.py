import torch, ttnn
from ttnn.operations.all_gather_rms_norm import all_gather_rms_norm_program_descriptor as pd

cfg = ttnn._ttnn.fabric.FabricRouterConfig()
cfg.max_packet_payload_size_bytes = 14 * 1024 + 64
ttnn.set_fabric_config(
    ttnn.FabricConfig.FABRIC_2D_TORUS_XY,
    ttnn.FabricReliabilityMode.STRICT_SYSTEM_HEALTH_SETUP_MODE,
    None,
    ttnn.FabricTensixConfig.DISABLED,
    ttnn.FabricUDMMode.DISABLED,
    ttnn.FabricManagerMode.DEFAULT,
    cfg,
)
mesh = ttnn.open_mesh_device(mesh_shape=ttnn.MeshShape(1, 4))
orig = pd._check_l1_interleaved_fit


def spy(m, cbs, out):
    base = int(ttnn.get_allocator_base_address(m, ttnn.BufferType.L1))
    end = base + sum(-(-int(c.total_size) // 64) * 64 for c in cbs)
    print(
        f"MARGIN base={base} cb_end={end} out={int(out.buffer_address())} free={int(out.buffer_address())-end} nb={pd._num_banks(m,out)}"
    )
    return orig(m, cbs, out)


pd._check_l1_interleaved_fit = spy
from ttnn.operations.all_gather_rms_norm import all_gather_rms_norm

for shape, dt in [((1, 1, 2048, 1280), ttnn.float32), ((1, 1, 640, 384), ttnn.bfloat16)]:
    x = ttnn.from_torch(
        torch.randn(shape), dtype=dt, layout=ttnn.TILE_LAYOUT, device=mesh, mesh_mapper=ttnn.ReplicateTensorToMesh(mesh)
    )
    o = all_gather_rms_norm(
        x,
        cluster_axis=1,
        norm_type="layer_norm",
        topology=ttnn.Topology.Linear,
        num_links=2,
        memory_config=ttnn.L1_MEMORY_CONFIG,
    )
    ttnn.synchronize_device(mesh)
ttnn.close_mesh_device(mesh)
ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)
