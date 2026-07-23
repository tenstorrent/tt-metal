import torch, ttnn
from eval.sharding import auto_shard_config, shard_config
from ttnn.operations.rms_norm import rms_norm

device = ttnn.open_device(device_id=0)


def run(tag, shape, ml, gamma_layout, shard_hw=None, gridxy=None):
    W = shape[-1]
    torch.manual_seed(0)
    ti = torch.randn(shape, dtype=torch.bfloat16)
    x = ti.to(torch.float32)
    ref = x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + 1e-6)
    tg = torch.randn(W, dtype=torch.bfloat16)
    ref = ref * tg.to(torch.float32).reshape(-1)
    mc = (
        shard_config(shard_hw, gridxy, ml, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16, device=device)
        if shard_hw
        else auto_shard_config(list(shape), ml, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16, device=device)
    )
    xt = ttnn.from_torch(ti, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=mc)
    # RM gamma: interleaved DRAM (1,1,1,W)
    gt = ttnn.from_torch(tg.reshape(1, 1, 1, W), dtype=ttnn.bfloat16, layout=gamma_layout, device=device)
    cfg = ttnn.ComputeConfigDescriptor()
    cfg.fp32_dest_acc_en = True
    out = rms_norm(xt, gamma=gt, epsilon=1e-6, compute_kernel_config=cfg, memory_config=mc)
    r = ttnn.to_torch(out).to(torch.float32)
    pcc = torch.corrcoef(torch.stack([r.flatten(), ref.flatten()]))[0, 1].item()
    print(f"RESULT {tag}: PCC={pcc:.6f} maxdiff={(r-ref).abs().max().item():.4f}")


try:
    W = ttnn.TensorMemoryLayout.WIDTH_SHARDED
    B = ttnn.TensorMemoryLayout.BLOCK_SHARDED
    RM = ttnn.ROW_MAJOR_LAYOUT
    T = ttnn.TILE_LAYOUT
    # Part 1: TILE input + RM gamma, sharded
    run("WIDTH-RMgamma-perw2", (1, 1, 32, 512), W, RM, [32, 64], (8, 1))
    run("WIDTH-RMgamma-perw1", (1, 1, 32, 256), W, RM, [32, 32], (8, 1))
    run("WIDTH-RMgamma-R2", (1, 1, 64, 512), W, RM, [64, 64], (8, 1))
    run("BLOCK-RMgamma-256x512", (1, 1, 256, 512), B, RM)
    run("WIDTH-RMgamma-auto2048", (1, 1, 32, 2048), W, RM)
    run("WIDTH-RMgamma-nonalign50", (1, 1, 32, 50), W, RM)
    # regression: TILE gamma still works
    run("WIDTH-TILEgamma-perw2", (1, 1, 32, 512), W, T, [32, 64], (8, 1))
finally:
    ttnn.close_device(device)
