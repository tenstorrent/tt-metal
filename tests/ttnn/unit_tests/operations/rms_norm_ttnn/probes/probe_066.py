# Monte Carlo with the REAL marginal: rows resampled from the captured Gemma q/k/v values, 4..64 tiles;
# plus the real k rows with elements permuted (same exact scale, different tile arrangement).
import torch, ttnn, os, math, glob
import ttnn.operations.rms_norm_ttnn.rms_norm_ttnn_program_descriptor as desc
OUT=os.environ["OUT"]; SD=os.environ["SD"]
dev = ttnn.open_device(device_id=0, l1_small_size=8192)
try:
    R, SEEDS, eps = 2048, 4, 1e-6
    def ulp(x): return torch.pow(2.0, torch.floor(torch.log2(x.abs().clamp_min(1e-30)))-7)
    pools = {}
    for tag in ("q","k","v"):
        vals = torch.cat([torch.load(f)["x"].double().flatten() for f in sorted(glob.glob(f"{SD}/phndump/phn_00?_{tag}.pt"))])
        pools[tag] = vals
        print(f"MC pool {tag}: {len(vals)} values, median|x| {vals.abs().median().item():.2f}, std {vals.std().item():.2f}, max {vals.abs().max().item():.1f}, frac>8*median {(vals.abs()>8*vals.abs().median()).double().mean().item():.4f}")
    kreal = torch.load(f"{SD}/phndump/phn_001_k.pt")["x"].double().squeeze()   # 64 rows x 256
    def run(fn, X, order):
        desc.REDUCE_ORDER, desc.REDUCE_ORDER_MAX_WT = order, 10**9
        return ttnn.to_torch(fn(X, epsilon=eps, memory_config=ttnn.DRAM_MEMORY_CONFIG)).double().reshape(R, -1)[:, 0]
    variants = (("gen", ttnn.rms_norm, "shipped"), ("nat", ttnn._native_rms_norm, "shipped"))
    def report(tag, T, E):
        e = {k: torch.cat(v) for k, v in E.items()}; N = len(e["gen"])
        d = e["gen"].abs() - e["nat"].abs(); gc = (d<0).sum().item(); nc = (d>0).sum().item(); n = gc+nc
        z = (gc-n/2)/math.sqrt(n/4) if n else 0.0; p = math.erfc(abs(z)/math.sqrt(2))
        print(f"MC {tag:>10s} {T:5d} | {e['gen'].mean().item():+9.3f} {e['nat'].mean().item():+9.3f} | {e['gen'].abs().mean().item():11.3f} {e['nat'].abs().mean().item():6.3f} | "
              f"{(e['gen']>0).double().mean().item():10.2f} {(e['nat']>0).double().mean().item():5.2f} | {gc/N:10.3f} {nc/N:10.3f} {p:8.1e} | {d.mean().item():+8.3f} ± {1.96*d.std().item()/math.sqrt(N):.3f}", flush=True)
        return e
    store = {}
    print(f"MC {'family':>10s} {'tiles':>5s} | {'bias gen':>9s} {'bias nat':>9s} | {'mean|e| gen':>11s} {'nat':>6s} | {'P(e>0) gen':>10s} {'nat':>5s} | {'gen closer':>10s} {'nat closer':>10s} {'p(sign)':>8s} | {'d|e| gen-nat ±95%':>18s}")
    for tag in ("k", "q", "v"):
        for T in (4, 8, 16, 32, 64):
            W = 32*T; E = {"gen": [], "nat": []}
            for sd in range(SEEDS):
                g = torch.Generator().manual_seed(7000 + 100*T + sd)
                idx = torch.randint(len(pools[tag]), (R, W), generator=g)
                x = pools[tag][idx]; x[:, 0] = 1.0; x = x.to(torch.bfloat16); xd = x.double()
                s_ex = 1/torch.sqrt(xd.pow(2).mean(-1)+eps); u = ulp(s_ex)
                X = ttnn.from_torch(x.reshape(1,1,R,W), ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev, memory_config=ttnn.DRAM_MEMORY_CONFIG)
                for name, fn, order in variants: E[name].append((run(fn, X, order)-s_ex)/u)
                X.deallocate(True)
            store[(f"pool-{tag}", T)] = report(f"pool-{tag}", T, E)
    # real k rows, elements permuted: 32 permutations x 64 rows = 2048 rows per seed
    E = {"gen": [], "nat": []}
    for sd in range(SEEDS):
        g = torch.Generator().manual_seed(9000+sd)
        rows = []
        for rep in range(R//64):
            perm = torch.rand(64, 256, generator=g).argsort(-1)
            rows.append(torch.gather(kreal, 1, perm))
        x = torch.cat(rows); x[:, 0] = 1.0; x = x.to(torch.bfloat16); xd = x.double()
        s_ex = 1/torch.sqrt(xd.pow(2).mean(-1)+eps); u = ulp(s_ex)
        X = ttnn.from_torch(x.reshape(1,1,R,256), ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        for name, fn, order in variants: E[name].append((run(fn, X, order)-s_ex)/u)
        X.deallocate(True)
    store[("k-permuted", 8)] = report("k-permuted", 8, E)
    desc.REDUCE_ORDER, desc.REDUCE_ORDER_MAX_WT = "shipped", 0
    torch.save(store, f"{OUT}/mc_realmarginal.pt")
    print("MC PROBE_OK", flush=True)
finally:
    ttnn.close_device(dev)
