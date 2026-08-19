import torch, ttnn
from ttnn.operations.rms_norm import rms_norm, default_compute_kernel_config


def stats(e, a):
    rms = (a - e).pow(2).mean().sqrt().item() / e.std().item()
    pcc = torch.corrcoef(torch.stack([e.flatten().double(), a.flatten().double()]))[0, 1].item()
    return pcc, rms


dev = ttnn.open_device(device_id=0)
try:
    torch.manual_seed(0)
    for W in [17, 50]:
        shape = (1, 1, 64, W)
        tx = torch.randn(shape)
        tg = torch.randn(W).reshape(1, 1, 1, W)
        x = ttnn.from_torch(tx, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=dev)
        x32 = ttnn.to_torch(x).float()
        base = x32 * torch.rsqrt(x32.pow(2).mean(-1, keepdim=True) + 1e-6)
        for gdt, gn in [(ttnn.bfloat8_b, "bf8b"), (ttnn.bfloat16, "bf16")]:
            g = ttnn.from_torch(tg.to(torch.bfloat16), dtype=gdt, layout=ttnn.TILE_LAYOUT, device=dev)
            g_true = ttnn.to_torch(g).float()[..., :W]
            e = base * g_true
            for via in [1, 0]:
                out = ttnn.to_torch(
                    rms_norm(
                        x,
                        gamma=g,
                        compute_kernel_config=default_compute_kernel_config(),
                        _levers=dict(reduce_via_add=via),
                    )
                ).float()
                pcc, rms = stats(e, out)
                # per-column error profile: is it the partial tile's columns?
                colerr = (out - e).abs()[0, 0].mean(0)
                worst = torch.topk(colerr, min(4, W))
                print(
                    f"RESULT W={W:>3} gamma={gn} via_add={via} PCC={pcc:.6f} rms={rms:.5f} "
                    f"worst_cols={worst.indices.tolist()} vals={[round(v,4) for v in worst.values.tolist()]}"
                )
finally:
    ttnn.close_device(dev)
