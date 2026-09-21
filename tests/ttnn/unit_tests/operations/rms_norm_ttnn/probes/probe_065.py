# Paired Monte Carlo: row-scale error of three summation orders on few-hot rows, 4..64 tiles.
# Element 0 of every row is a witness x=1.0 so the output there IS the row scale (exact bf16 readout).
import torch, ttnn, os, math
import ttnn.operations.rms_norm_ttnn.rms_norm_ttnn_program_descriptor as desc
OUT=os.environ["OUT"]
dev = ttnn.open_device(device_id=0, l1_small_size=8192)
try:
    R, SEEDS, eps = 2048, 4, 1e-6
    tiles = [4, 8, 16, 32, 64]
    def ulp(x): return torch.pow(2.0, torch.floor(torch.log2(x.abs().clamp_min(1e-30)))-7)
    MED = 0.6745  # median |N(0,1)|
    def make(fam, W, g):
        x = torch.randn(R, W, generator=g)
        if fam == "gaussian":
            pass
        elif fam in ("scattered", "concentrated"):
            lam = 3.5 * W / 256                       # Gemma per-head: ~3.5 elements/row above 8x median at W=256
            k = torch.poisson(torch.full((R,), lam), generator=g).long().clamp(max=W//4)
            mag = (8 + 8*torch.rand(R, W, generator=g)) * MED   # 8..16 x median
            if fam == "scattered":
                pos = torch.rand(R, W, generator=g).argsort(-1)
            else:                                    # hot positions drawn from a small fixed channel pool
                pool = torch.randperm(W, generator=g)[:max(4, W//24)]   # ~12 channels at W=256
                pos = pool[torch.randint(len(pool), (R, W), generator=g)]
            hot = torch.zeros(R, W, dtype=torch.bool)
            for r in range(R):
                hot[r, pos[r, :k[r]]] = True
            x = torch.where(hot, torch.sign(x)*mag, x)
        elif fam == "massive":                        # 2 fixed channels at 100x median, every row (Sun et al.)
            ch = torch.randperm(W, generator=g)[:2]
            x[:, ch] = torch.sign(x[:, ch]) * 100 * MED
        x[:, 0] = 1.0
        return x.to(torch.bfloat16)
    def run(fn, X, order):
        desc.REDUCE_ORDER, desc.REDUCE_ORDER_MAX_WT = order, 10**9
        o = ttnn.to_torch(fn(X, epsilon=eps, memory_config=ttnn.DRAM_MEMORY_CONFIG)).double().reshape(R, -1)
        return o[:, 0]
    variants = (("gen", ttnn.rms_norm, "shipped"), ("genNO", ttnn.rms_norm, "native"), ("nat", ttnn._native_rms_norm, "shipped"))
    store = {}
    for fam in ("scattered", "concentrated", "massive", "gaussian"):
        print(f"MC ===== {fam}: {R*SEEDS} rows per width; e = (device scale - exact)/bf16 ulp; paired per row")
        print(f"MC {'tiles':>5s} | {'bias gen':>9s} {'bias genNO':>10s} {'bias nat':>9s} | {'mean|e| gen':>11s} {'genNO':>6s} {'nat':>6s} | {'P(e>0) gen':>10s} {'nat':>5s} | {'gen closer':>10s} {'nat closer':>10s} {'p(sign)':>8s} | {'d|e| gen-nat ±95%':>18s}")
        for T in tiles:
            W = 32*T; E = {v[0]: [] for v in variants}
            for sd in range(SEEDS):
                g = torch.Generator().manual_seed(1000*T + sd)
                x = make(fam, W, g); xd = x.double()
                s_ex = 1/torch.sqrt(xd.pow(2).mean(-1) + eps); u = ulp(s_ex)
                X = ttnn.from_torch(x.reshape(1,1,R,W), ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev, memory_config=ttnn.DRAM_MEMORY_CONFIG)
                for name, fn, order in variants:
                    E[name].append((run(fn, X, order) - s_ex)/u)
                X.deallocate(True)
            e = {k: torch.cat(v) for k, v in E.items()}; store[(fam, T)] = e
            N = len(e["gen"]); ci = lambda a: 1.96*a.std().item()/math.sqrt(N)
            d = e["gen"].abs() - e["nat"].abs()
            gc = (d < 0).sum().item(); nc = (d > 0).sum().item(); n = gc + nc
            # two-sided sign test, normal approximation
            z = (gc - n/2)/math.sqrt(n/4) if n else 0.0; p = math.erfc(abs(z)/math.sqrt(2))
            print(f"MC {T:5d} | {e['gen'].mean().item():+9.3f} {e['genNO'].mean().item():+10.3f} {e['nat'].mean().item():+9.3f} | "
                  f"{e['gen'].abs().mean().item():11.3f} {e['genNO'].abs().mean().item():6.3f} {e['nat'].abs().mean().item():6.3f} | "
                  f"{(e['gen']>0).double().mean().item():10.2f} {(e['nat']>0).double().mean().item():5.2f} | "
                  f"{gc/N:10.3f} {nc/N:10.3f} {p:8.1e} | {d.mean().item():+8.3f} ± {ci(d):.3f}", flush=True)
    desc.REDUCE_ORDER, desc.REDUCE_ORDER_MAX_WT = "shipped", 0
    torch.save(store, f"{OUT}/mc_fewhot.pt")
    print("MC PROBE_OK", flush=True)
finally:
    ttnn.close_device(dev)
