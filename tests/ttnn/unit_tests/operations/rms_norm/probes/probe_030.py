import torch, ttnn
from ttnn.operations.rms_norm.perf_experiments.fused_scale import scale_bench as sb

dev = ttnn.open_device(device_id=0)
def mkcfg(fp32):
    c = ttnn.ComputeConfigDescriptor()
    c.math_fidelity = ttnn.MathFidelity.HiFi2
    c.fp32_dest_acc_en = fp32
    c.math_approx_mode = False
    return c

wt = 16
for fp32 in (False, True):
    cfg = mkcfg(fp32)
    x, gamma, rms, ref = sb.make_inputs(dev, wt=wt, block_ht=1, dtype=ttnn.bfloat16)
    for blk in (1,2,3,4,5,6,7,8):
        out = sb.run_arm(x, gamma, rms, arm="fused_rmsfull", wt=wt, block_ht=1, dest_block=blk,
                         kernel_iters=1, compute_kernel_config=cfg)
        got = ttnn.to_torch(out).float()
        d = (got - ref).abs()
        pcc = torch.corrcoef(torch.stack([got.flatten(), ref.flatten()]))[0,1].item()
        bad_tiles = [t for t in range(wt) if d[:, t*32:(t+1)*32].max().item() > 0.3]
        print(f"RES: fp32_dest={fp32} blk={blk} pcc={pcc:.5f} bad_tiles={bad_tiles}")
        ttnn.deallocate(out)
ttnn.close_device(dev)
