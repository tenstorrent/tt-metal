import torch, ttnn
from ttnn.operations.rms_norm.perf_experiments.fused_scale import scale_bench as sb

dev = ttnn.open_device(device_id=0)
cfg = ttnn.ComputeConfigDescriptor()
cfg.math_fidelity = ttnn.MathFidelity.HiFi2
cfg.fp32_dest_acc_en = False
cfg.math_approx_mode = False

wt, block_ht, dest_block = 8, 1, 8
x, gamma, rms, ref = sb.make_inputs(dev, wt=wt, block_ht=block_ht, dtype=ttnn.bfloat16)
xt = ttnn.to_torch(x).float(); gt = ttnn.to_torch(gamma).float(); rt = ttnn.to_torch(rms).float()
for arm in ("baseline", "fused_rmsfull"):
    out = sb.run_arm(x, gamma, rms, arm=arm, wt=wt, block_ht=block_ht, dest_block=dest_block,
                     kernel_iters=1, compute_kernel_config=cfg)
    got = ttnn.to_torch(out).float()
    d = (got - ref).abs()
    flat = d.flatten()
    top = torch.topk(flat, 6)
    print("==", arm, "pcc", torch.corrcoef(torch.stack([got.flatten(), ref.flatten()]))[0,1].item(),
          "max|d|", d.max().item(), "mean|d|", d.mean().item())
    for v, i in zip(top.values.tolist(), top.indices.tolist()):
        r, c = i // got.shape[1], i % got.shape[1]
        print(f"   r{r} c{c}  got {got[r,c]:.5f} ref {ref[r,c]:.5f}  x {xt[r,c]:.5f} rms {rt[r,0]:.5f} gamma {gt[0,c]:.5f}")
    ttnn.deallocate(out)
ttnn.close_device(dev)
