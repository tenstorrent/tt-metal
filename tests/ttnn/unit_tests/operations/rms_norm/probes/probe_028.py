import torch, ttnn
from ttnn.operations.rms_norm.perf_experiments.fused_scale import scale_bench as sb

dev = ttnn.open_device(device_id=0)
cfg = ttnn.ComputeConfigDescriptor()
cfg.math_fidelity = ttnn.MathFidelity.HiFi2
cfg.fp32_dest_acc_en = False
cfg.math_approx_mode = False

for wt, blk in ((8,8),(8,4),(8,2),(8,1),(16,8)):
    x, gamma, rms, ref = sb.make_inputs(dev, wt=wt, block_ht=1, dtype=ttnn.bfloat16)
    for arm in ("fused_rmsfull",):
        out = sb.run_arm(x, gamma, rms, arm=arm, wt=wt, block_ht=1, dest_block=blk,
                         kernel_iters=1, compute_kernel_config=cfg)
        got = ttnn.to_torch(out).float()
        d = (got - ref).abs()
        # per-tile max error
        per_tile = [round(d[:, t*32:(t+1)*32].max().item(),4) for t in range(wt)]
        pcc = torch.corrcoef(torch.stack([got.flatten(), ref.flatten()]))[0,1].item()
        print(f"== wt={wt} dest_block={blk} pcc={pcc:.5f} per_tile_max_err={per_tile}")
        # which rows in the bad tile
        bad = [t for t,v in enumerate(per_tile) if v > 0.2]
        for t in bad[:2]:
            rows = (d[:, t*32:(t+1)*32].max(dim=1).values > 0.2).nonzero().flatten().tolist()
            print(f"    tile {t} bad rows: {rows}")
        ttnn.deallocate(out)
ttnn.close_device(dev)
