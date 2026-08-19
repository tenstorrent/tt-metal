import torch, ttnn
from ttnn.operations.rms_norm.perf_experiments.fused_scale import scale_bench as sb

dev = ttnn.open_device(device_id=0)
cfg = ttnn.ComputeConfigDescriptor()
cfg.math_fidelity = ttnn.MathFidelity.HiFi2
cfg.fp32_dest_acc_en = False
cfg.math_approx_mode = False

wt, block_ht, dest_block = 8, 1, 8
x, gamma, rms, ref = sb.make_inputs(dev, wt=wt, block_ht=block_ht, dtype=ttnn.bfloat16)
for arm in ("baseline", "fused_rmsfull", "fused_inchain"):
    out = sb.run_arm(x, gamma, rms, arm=arm, wt=wt, block_ht=block_ht, dest_block=dest_block,
                     kernel_iters=1, compute_kernel_config=cfg)
    got = ttnn.to_torch(out).float()
    ratio = (got / ref)
    print(arm, "pcc", torch.corrcoef(torch.stack([got.flatten(), ref.flatten()]))[0,1].item())
    print("  ratio row0 [0:12]:", [round(v,4) for v in ratio[0,:12].tolist()])
    print("  ratio col0 [0:12]:", [round(v,4) for v in ratio[:12,0].tolist()])
    print("  ratio mean/std per-col-0:", ratio[:,0].mean().item(), ratio[:,0].std().item())
    print("  got row0[:6]", [round(v,4) for v in got[0,:6].tolist()])
    print("  ref row0[:6]", [round(v,4) for v in ref[0,:6].tolist()])
    ttnn.deallocate(out)
ttnn.close_device(dev)
