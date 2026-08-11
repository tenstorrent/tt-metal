#!/usr/bin/env python3
"""Isolate where gemma-4-26B sliding's 0.99963 -> 0.99457 whole-layer PCC drop comes from.

TEST 1  weight placement.  v2's tree passes the rms_norm weight through UNCHANGED (DRAM
        interleaved) while sharding only the activation; v3's tree RESHARDS the weight into
        the width-sharded config.  Same knob, two codes.  Do they differ?
TEST 2  router topk flips.  The router norm is the weight=None rms_norm whose output feeds
        ttnn.topk(k=8) over 128 experts.  Using the REAL layer-0 router.proj.weight,
        router.scale and per_expert_scale, count how many of the 8 selected experts change
        between the interleaved and the sharded norm, as a function of activation scale.
TEST 3  price a flip.  From the real softmax weights, what whole-layer PCC drop does losing
        the n-th expert imply?  Compare against the measured 5.055e-3.
"""
import sys, numpy as np, torch, ttnn

HIDDEN, TILE, EPS, K, NEXP = 2816, 32, 9.99999997e-7, 8, 128
WT = HIDDEN // TILE
W = np.load("/tmp/gemma_router_layer0.npz")
proj = torch.from_numpy(W["router.proj.weight"].astype(np.float32))          # [128, 2816]
rscale = torch.from_numpy(W["router.scale"].astype(np.float32))              # [2816]
pescale = torch.from_numpy(W["router.per_expert_scale"].astype(np.float32))  # [128]

def cfg(c, gx, gy):
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
        wln = torch.from_numpy(W["pre_feedforward_layernorm_2.weight"].astype(np.float32)).reshape(1,1,1,HIDDEN)
        wt_il = ttnn.from_torch(wln, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev,
                                memory_config=ttnn.DRAM_MEMORY_CONFIG)

        print("=== TEST 1: v2 style (interleaved weight) vs v3 style (resharded weight), REAL norm weight")
        for scale in (1.0, 30.0, 120.0):
            x = torch.randn(1,1,1,HIDDEN) * scale
            xt = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev,
                                 memory_config=ttnn.DRAM_MEMORY_CONFIG)
            il = ttnn.to_torch(ttnn.rms_norm(xt, epsilon=EPS, weight=wt_il, compute_kernel_config=cc,
                     memory_config=ttnn.DRAM_MEMORY_CONFIG)).to(torch.float32)
            for c, gx, gy in ((11,11,1), (88,11,8)):
                mc, pc = cfg(c,gx,gy)
                xs = ttnn.to_memory_config(xt, mc)
                o2 = ttnn.to_torch(ttnn.to_memory_config(ttnn.rms_norm(xs, epsilon=EPS, weight=wt_il,
                        compute_kernel_config=cc, memory_config=mc, program_config=pc),
                        ttnn.DRAM_MEMORY_CONFIG)).to(torch.float32)      # v2: weight left interleaved
                ws = ttnn.to_memory_config(wt_il, mc)
                o3 = ttnn.to_torch(ttnn.to_memory_config(ttnn.rms_norm(xs, epsilon=EPS, weight=ws,
                        compute_kernel_config=cc, memory_config=mc, program_config=pc),
                        ttnn.DRAM_MEMORY_CONFIG)).to(torch.float32)      # v3: weight resharded
                print(f"  scale {scale:>5}  {c:>2}c: v2style-vs-v3style differ in "
                      f"{int(((o2-o3).abs()>0).sum()):>4}/{HIDDEN} elems (max {float((o2-o3).abs().max()):.3e});"
                      f"  v2style-vs-interleaved {int(((o2-il).abs()>0).sum()):>4};"
                      f"  v3style-vs-interleaved {int(((o3-il).abs()>0).sum()):>4}")

        print("\n=== TEST 2: router topk flips, real router.proj.weight, weight=None norm (the router norm)")
        print("    scale | norm elems differing | logit gap 8th-9th | experts flipped / 8 | trials with a flip")
        for scale in (1.0, 5.0, 15.0, 30.0, 60.0, 120.0, 240.0):
            trials, flips, anyflip, ndiff_t, gaps = 24, 0, 0, 0, []
            for t in range(trials):
                x = torch.randn(1,1,1,HIDDEN) * scale
                xt = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev,
                                     memory_config=ttnn.DRAM_MEMORY_CONFIG)
                il = ttnn.to_torch(ttnn.rms_norm(xt, epsilon=EPS, weight=None,
                        compute_kernel_config=cc, memory_config=ttnn.DRAM_MEMORY_CONFIG)).to(torch.float32)
                mc, pc = cfg(11,11,1)
                xs = ttnn.to_memory_config(xt, mc)
                sh = ttnn.to_torch(ttnn.to_memory_config(ttnn.rms_norm(xs, epsilon=EPS, weight=None,
                        compute_kernel_config=cc, memory_config=mc, program_config=pc),
                        ttnn.DRAM_MEMORY_CONFIG)).to(torch.float32)
                ndiff_t += int(((il-sh).abs()>0).sum())
                def route(v):
                    r = (v.reshape(-1) * rscale)
                    lg = (r @ proj.T).to(torch.bfloat16).to(torch.float32)
                    tv, ti = torch.topk(lg, K, dim=-1, sorted=True)
                    return lg, set(ti.tolist()), torch.softmax(tv, -1)
                lg_i, set_i, sm_i = route(il)
                lg_s, set_s, sm_s = route(sh)
                s = torch.sort(lg_i, descending=True).values
                gaps.append(float(s[K-1] - s[K]))
                d = len(set_i - set_s)
                flips += d
                anyflip += 1 if d else 0
            print(f"    {scale:>5} | {ndiff_t/trials:>20.1f} | {np.mean(gaps):>17.4f} | "
                  f"{flips/trials:>19.3f} | {anyflip:>2}/{trials}")

        print("\n=== TEST 3: what a flip costs, from the real softmax weights")
        x = torch.randn(1,1,1,HIDDEN) * 60.0
        r = (x.reshape(-1)/x.reshape(-1).pow(2).mean().sqrt()) * rscale
        lg = (r @ proj.T).to(torch.bfloat16).to(torch.float32)
        tv, ti = torch.topk(lg, K, dim=-1, sorted=True)
        sm = torch.softmax(tv, -1) * pescale[ti]
        w = (sm / sm.sum()).tolist()
        print("    top-8 normalised routing weights:", " ".join(f"{v:.4f}" for v in w))
        for n in (K-1, K-2):
            wn = w[n]
            print(f"    swapping slot {n+1} (weight {wn:.4f}) for an independent expert of equal "
                  f"magnitude => 1-PCC ~ {wn*wn:.3e}")
        print(f"    MEASURED whole-layer 1-PCC drop to explain: 5.055e-03  "
              f"=> implies a flipped weight of ~{5.055e-3**0.5:.3f}")
    finally:
        ttnn.close_device(dev)
    return 0

sys.exit(main())
