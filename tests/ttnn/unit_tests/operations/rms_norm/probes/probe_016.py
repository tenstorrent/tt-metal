import torch, ttnn
from eval.sharding import shard_config, auto_shard_config
from ttnn.operations.rms_norm import rms_norm

device = ttnn.open_device(device_id=0)


def run(tag, shape, ml, shard_hw=None, gridxy=None, gamma=True):
    W = shape[-1]
    torch.manual_seed(0)
    ti = torch.randn(shape, dtype=torch.bfloat16)
    x = ti.to(torch.float32)
    ref = x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + 1e-6)
    gt = None
    if gamma:
        tg = torch.randn(W, dtype=torch.bfloat16)
        ref = ref * tg.to(torch.float32).reshape(-1)
    mc = (
        shard_config(shard_hw, gridxy, ml, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16, device=device)
        if shard_hw
        else auto_shard_config(list(shape), ml, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16, device=device)
    )
    xt = ttnn.from_torch(ti, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=mc)
    if gamma:
        gt = ttnn.from_torch(tg.reshape(1, 1, 1, W), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    cfg = ttnn.ComputeConfigDescriptor()
    cfg.fp32_dest_acc_en = True
    out = rms_norm(xt, gamma=gt, epsilon=1e-6, compute_kernel_config=cfg, memory_config=mc)
    r = ttnn.to_torch(out).to(torch.float32)
    pcc = torch.corrcoef(torch.stack([r.flatten(), ref.flatten()]))[0, 1].item()
    print(f"RESULT {tag}: PCC={pcc:.6f} maxdiff={(r-ref).abs().max().item():.4f}")


try:
    W = ttnn.TensorMemoryLayout.WIDTH_SHARDED
    B = ttnn.TensorMemoryLayout.BLOCK_SHARDED
    run("WIDTH-perw2", (1, 1, 32, 512), W, [32, 64], (8, 1))
    run("WIDTH-perw2-R2", (1, 1, 64, 512), W, [64, 64], (8, 1))
    run("BLOCK-256x512", (1, 1, 256, 512), B)
    run("BLOCK-nogamma", (1, 1, 256, 512), B, gamma=False)
    run("WIDTH-perw1", (1, 1, 32, 256), W, [32, 32], (8, 1))  # regression check
    run("auto-WIDTH-2048", (1, 1, 32, 2048), W)
finally:
    ttnn.close_device(device)
