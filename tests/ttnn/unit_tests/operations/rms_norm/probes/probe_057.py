import torch, ttnn
import ttnn.operations.rms_norm.rms_norm_program_descriptor as desc
from ttnn.operations.rms_norm import rms_norm

desc._FORCE_REGIME = "B"
desc._FORCE_TRANSPORT = 2  # TRANSPORT_REDUCE_BCAST

device = ttnn.open_device(device_id=0)
try:
    # confirm routing: a Regime-B descriptor should still allocate 3 sems
    shape = (1, 1, 32, 8192)
    ti = ttnn.from_torch(
        torch.zeros(shape, dtype=torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    ot = ttnn.allocate_tensor_on_device(
        ttnn.Shape(list(shape)), ttnn.bfloat16, ttnn.TILE_LAYOUT, device, ttnn.DRAM_MEMORY_CONFIG
    )
    prog, _ = desc.create_program_descriptor(ti, ot, None, 1e-6, None)
    print("FORCE_TRANSPORT routes to:", desc._select_transport(16), "sems:", len(prog.semaphores))

    for shp in [(1, 1, 32, 8192), (1, 1, 64, 8192), (1, 1, 32, 16384)]:
        W = shp[-1]
        # all-ones -> exactly 1.0
        xo = torch.ones(*shp)
        to = ttnn.from_torch(xo.bfloat16(), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
        oo = ttnn.to_torch(rms_norm(to)).float()
        max_ones_err = (oo - 1.0).abs().max().item()
        # random -> PCC
        x = torch.randn(*shp)
        ti2 = ttnn.from_torch(x.bfloat16(), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
        out = ttnn.to_torch(rms_norm(ti2)).float()
        ref = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6)
        a = out.flatten()
        b = ref.flatten()
        a = a - a.mean()
        b = b - b.mean()
        pcc = ((a * b).sum() / (a.norm() * b.norm() + 1e-12)).item()
        print(f"shape={shp} all-ones max_err={max_ones_err:.4f} PCC={pcc:.5f}")
finally:
    ttnn.close_device(device)
