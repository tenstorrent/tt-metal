import torch, ttnn
from ttnn.operations.rms_norm import rms_norm
from ttnn.operations.rms_norm.rms_norm_program_descriptor import blocking_plan


def cfg_of(acc):
    c = ttnn.ComputeConfigDescriptor()
    c.math_fidelity = ttnn.MathFidelity.HiFi2
    c.fp32_dest_acc_en = acc
    c.math_approx_mode = False
    return c


dev = ttnn.open_device(device_id=0)
try:
    torch.manual_seed(0)
    W = 7168
    tx = torch.randn((1, 1, 32, W)).to(torch.bfloat16)
    x = ttnn.from_torch(tx, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev)
    x32 = tx.float()
    normed = x32 * torch.rsqrt(x32.pow(2).mean(-1, keepdim=True) + 1e-6)
    g = ttnn.from_torch(
        torch.ones(1, 1, 1, W).to(torch.bfloat16), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev
    )
    for acc in [True, False]:
        p = blocking_plan(x, g, None, dev, cfg_of(acc), None)
        print(
            f"RESULT plan fp32acc={acc} regime={p.regime} BLOCK_HT={p.BLOCK_HT} wr={p.WT_REDUCE_BLOCK} "
            f"ws={p.WT_SCALE_BLOCK} dest={p.DEST_BLOCK} in_depth={p.IN_BUF_DEPTH} out={p.OUT_BUF_DEPTH} "
            f"acc_dt={p.acc_dtype} ws_bytes={p.working_set_bytes()}"
        )
        out = ttnn.to_torch(rms_norm(x, gamma=g, compute_kernel_config=cfg_of(acc))).float()
        err = (out - normed).abs()[0, 0]
        rel = err / normed[0, 0].abs().clamp(min=1e-3)
        bad = rel > 0.05
        nb = int(bad.sum())
        print(f"RESULT map fp32acc={acc} nbad(rel>5%)={nb}/{bad.numel()} maxerr={err.max():.4f}")
        if nb:
            rows, cols = torch.nonzero(bad, as_tuple=True)
            print(
                f"RESULT map fp32acc={acc} bad rows uniq={sorted(set(rows.tolist()))[:8]}... nrows={len(set(rows.tolist()))}"
            )
            tiles = sorted(set((cols // 32).tolist()))
            print(f"RESULT map fp32acc={acc} bad col-tiles={tiles[:20]} ... ntiles={len(tiles)} (Wt={W//32})")
            print(f"RESULT map fp32acc={acc} bad col%32 uniq={sorted(set((cols % 32).tolist()))[:20]}")
        # per-column-tile mean error profile
        prof = err.mean(0).reshape(W // 32, 32).mean(1)
        top = torch.topk(prof, 8)
        print(
            f"RESULT prof fp32acc={acc} worst tiles={top.indices.tolist()} vals={[round(v,4) for v in top.values.tolist()]}"
        )
        print(
            f"RESULT prof fp32acc={acc} first8={[round(v,5) for v in prof[:8].tolist()]} last8={[round(v,5) for v in prof[-8:].tolist()]}"
        )
finally:
    ttnn.close_device(dev)
