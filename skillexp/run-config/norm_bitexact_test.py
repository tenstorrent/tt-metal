#!/usr/bin/env python3
"""Step 1 of the router-perturbation isolation: is the sharded norm output BIT-IDENTICAL
to the interleaved one?

sweep2 showed equal PCC to 8 decimals, which does not imply equal tensors.  If the bf16
outputs are bit-identical then the router sees identical input, cannot flip a topk
selection, and the expert-routing hypothesis for the 5.06e-3 whole-layer drop is dead too.
"""
import sys, torch, ttnn

HIDDEN, TILE, EPS = 2816, 32, 9.99999997e-7
WT = HIDDEN // TILE
GRIDS = [(2,(2,1)),(4,(4,1)),(8,(8,1)),(11,(11,1)),(22,(11,2)),(44,(11,4)),(88,(11,8)),(4,(2,2)),(8,(4,2))]

def cfg(c, g):
    gx, gy = g
    bw = WT // c
    mc = ttnn.create_sharded_memory_config((TILE, HIDDEN//c), ttnn.CoreGrid(x=gx,y=gy),
        ttnn.ShardStrategy.WIDTH, ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True)
    pc = ttnn.LayerNormShardedMultiCoreProgramConfig(compute_with_storage_grid_size=[gx,gy],
        subblock_w=next(v for v in (4,2,1) if bw%v==0), block_h=1, block_w=bw, inplace=False)
    return mc, pc

def main():
    torch.manual_seed(20260811)
    dev = ttnn.open_device(device_id=0)
    cc = ttnn.init_device_compute_kernel_config(dev.arch(), math_fidelity=ttnn.MathFidelity.HiFi4,
        math_approx_mode=False, fp32_dest_acc_en=True, packer_l1_acc=False)
    try:
        for iname, x in (("unit", torch.randn(1,1,1,HIDDEN)),
                         ("spiked", None),
                         ("bigscale", torch.randn(1,1,1,HIDDEN)*120.0)):
            if x is None:
                x = torch.randn(1,1,1,HIDDEN); x[..., ::311] *= 3000.0
            for wname, w in (("no_weight", None), ("weighted", 1.0+0.1*torch.randn(1,1,1,HIDDEN))):
                xt = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev,
                                     memory_config=ttnn.DRAM_MEMORY_CONFIG)
                wt = None if w is None else ttnn.from_torch(w, dtype=ttnn.bfloat16,
                        layout=ttnn.TILE_LAYOUT, device=dev, memory_config=ttnn.DRAM_MEMORY_CONFIG)
                base = ttnn.to_torch(ttnn.rms_norm(xt, epsilon=EPS, weight=wt,
                    compute_kernel_config=cc, memory_config=ttnn.DRAM_MEMORY_CONFIG)).to(torch.float32)
                print(f"--- {iname}/{wname}: interleaved reference, |max|={base.abs().max():.6g}")
                for c, g in GRIDS:
                    mc, pc = cfg(c, g)
                    xs = ttnn.to_memory_config(xt, mc)
                    ws = None if wt is None else ttnn.to_memory_config(wt, mc)
                    o = ttnn.rms_norm(xs, epsilon=EPS, weight=ws, compute_kernel_config=cc,
                                      memory_config=mc, program_config=pc)
                    got = ttnn.to_torch(ttnn.to_memory_config(o, ttnn.DRAM_MEMORY_CONFIG)).to(torch.float32)
                    d = (got - base).abs()
                    ndiff = int((d > 0).sum())
                    rel = float((d / base.abs().clamp(min=1e-30)).max())
                    tag = "BIT-IDENTICAL" if ndiff == 0 else f"{ndiff:>4}/{HIDDEN} elems differ"
                    print(f"      {c:>3}c {g[0]:>2}x{g[1]:<2}  {tag:<22} maxabs={float(d.max()):.3e} "
                          f"maxrel={rel:.3e}")
    finally:
        ttnn.close_device(dev)
    return 0

sys.exit(main())
