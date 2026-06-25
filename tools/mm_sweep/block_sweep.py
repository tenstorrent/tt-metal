import os, sys, math, statistics, json, torch, ttnn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sp_heur_backtest import pick_S_Pk  # in-repo (tools/mm_sweep/)

RAW = os.path.join(os.environ["TT_METAL_HOME"], "generated/profiler/.logs/profile_log_device.csv")
WARMUP, REPS = 2, 4
CHUNK = 1 + WARMUP + REPS
PEAK = 2048 * 64 * 1e9
TILE = 2048  # bf16 tile bytes
GX = GY = 8
L1_TILE_BUDGET = 720  # ~1.47MB of CB tiles; configs above this are skipped (and OOM is caught anyway)

# shape from argv: M K N
M, K, N = (int(x) for x in sys.argv[1:4])
Mt, Nt, Kt = M // 32, N // 32, K // 32
S, Pk = pick_S_Pk(Mt, Nt, Kt)
transpose = M > N
x_axis, y_axis = S * GX, GY // (S * Pk)
in0_par = x_axis if transpose else y_axis
in1_par = y_axis if transpose else x_axis
pcM, pcN = math.ceil(Mt / in0_par), math.ceil(Nt / in1_par)
print(f"shape {M}x{K}x{N}  Mt{Mt}xNt{Nt}xKt{Kt}  heuristic (S,Pk)=({S},{Pk})  per-core M={pcM} N={pcN}", flush=True)

# ---- OOM-safe block-candidate generator (prune rules, validated; see block-sweep notes) ----
# Candidate lists; capped per-shape at the smallest value that covers per-core (no wastefully-large blocks).
M_CAND = [1, 2, 3, 4, 5, 6, 8, 12, 16, 24, 32, 48, 64, 96, 128]
N_CAND = [1, 2, 3, 4, 6, 8, 9, 12, 16, 18, 24, 32, 48, 64, 96, 128]
K_CAND = [4, 8, 16]
IN_T = OUT_T = 2048  # bf16 in0/in1/out tile bytes
INTERM_T = 4096  # fp32 intermediate tile bytes
L1_CB_BUDGET = 1310720  # matches the factory auto-block sizer (1.25 MiB; OOM guard / upper bound)
Ktpb = Kt // max(1, Pk)  # per-band K tiles (K not exceeded -> no K-padding)


def footprint(mb, kb, nb):  # exact factory formula: in0/in1/out double-buffered, fp32 interm single
    return mb * kb * 2 * IN_T + kb * nb * 2 * IN_T + mb * nb * 2 * OUT_T + mb * nb * INTERM_T


def cap(cands, pc):  # keep up to the SMALLEST candidate >= per-core (e.g. pcN=17 -> allow 18, not 24/32)
    ge = [c for c in cands if c >= pc]
    hi = min(ge) if ge else max(cands)
    return [c for c in cands if c <= hi]


def m_candidates(pc):
    # Symmetric with N: cap at the smallest candidate covering per-core; drop single-tile (unless pc<=1).
    # NO ">=2 M-blocks" cap -- it overpruned small/skewed shapes (excluded the optimal single full block,
    # e.g. LTX pcM=5). The L1 budget prunes the large-pcM end (single huge block OOMs -> forces multi-block);
    # small pcM keeps the full block. The sweep then picks multi-block on big shapes, single-block on small.
    cs = cap(M_CAND, pc)
    if pc <= 1:
        return [1]
    return [m for m in cs if m >= 2] or [1]  # drop single-tile M unless that's the only option


def n_candidates(pc):
    cs = cap(N_CAND, pc)
    if pc <= 1:
        return [1]
    return [n for n in cs if n >= 2] or [1]  # N may be a single full block (no >=2-block rule)


def k_candidates(ktpb):
    cs = cap(K_CAND, ktpb)
    return [k for k in cs if ktpb % k == 0] or [min(cs)]  # divisors of K -> no K-padding


def adsub(mb, nb):
    def lg(v, c):
        for d in (4, 2, 1):
            if d <= c and v % d == 0:
                return d
        return 1

    if nb >= mb:
        sbw = lg(nb, 4)
        sbh = lg(mb, 4 // sbw)
    else:
        sbh = lg(mb, 4)
        sbw = lg(nb, 4 // sbh)
    return sbh, sbw


M_BLOCKS, N_BLOCKS, K_BLOCKS = m_candidates(pcM), n_candidates(pcN), k_candidates(Ktpb)
print(f"M_blocks={M_BLOCKS}\nN_blocks={N_BLOCKS}\nK_blocks={K_BLOCKS}  (Ktpb={Ktpb})", flush=True)


def durs():
    L = open(RAW).read().splitlines()
    h = [x.strip() for x in L[1].split(",")]
    ix = {k: i for i, k in enumerate(h)}
    zi, ti, ri, ty = ix["zone name"], ix["time[cycles since reset]"], ix["run host ID"], ix["type"]
    st, en = {}, {}
    for ln in L[2:]:
        f = ln.split(",")
        if len(f) <= zi or not f[zi].strip().endswith("-FW"):
            continue
        rid, t = f[ri].strip(), f[ti].strip()
        if not rid or not t:
            continue
        t = int(t)
        if f[ty].strip() == "ZONE_START":
            st[rid] = min(st.get(rid, t), t)
        elif f[ty].strip() == "ZONE_END":
            en[rid] = max(en.get(rid, t), t)
    return [en[r] - st[r] for r in sorted(st, key=int) if r in en]


def util(us):
    return 100 * 2 * M * K * N / (PEAK * us * 1e-6)


if os.path.exists(RAW):
    os.remove(RAW)
d = ttnn.open_device(device_id=0)
d.enable_program_cache()
cc = ttnn.init_device_compute_kernel_config(
    d.arch(), math_fidelity=ttnn.MathFidelity.HiFi2, math_approx_mode=False, fp32_dest_acc_en=True, packer_l1_acc=True
)
ta = torch.randn(M, K, dtype=torch.bfloat16)
tb = torch.randn(K, N, dtype=torch.bfloat16)
ref = ta.float() @ tb.float()
a = ttnn.from_torch(ta, dtype=ttnn.bfloat16, device=d, layout=ttnn.TILE_LAYOUT)
b = ttnn.from_torch(tb, dtype=ttnn.bfloat16, device=d, layout=ttnn.TILE_LAYOUT)


def pcc(ot):
    o = ot.flatten().float()
    r = ref.flatten()
    o = o - o.mean()
    rr = r - r.mean()
    return (torch.dot(o, rr) / (o.norm() * rr.norm() + 1e-12)).item()


man = []  # (tag, ok, pcc)


def run(cfg, tag):
    d.clear_program_cache()
    ok = True
    p = 0.0
    try:
        out = None
        for _ in range(1 + WARMUP):
            out = (
                ttnn.experimental.minimal_matmul(a, b, compute_kernel_config=cc, config=cfg)
                if cfg
                else ttnn.experimental.minimal_matmul(a, b, compute_kernel_config=cc)
            )
            out.deallocate()
        ttnn.synchronize_device(d)
        for _ in range(REPS):
            out = (
                ttnn.experimental.minimal_matmul(a, b, compute_kernel_config=cc, config=cfg)
                if cfg
                else ttnn.experimental.minimal_matmul(a, b, compute_kernel_config=cc)
            )
            ot = ttnn.to_torch(out)
            out.deallocate()
        ttnn.synchronize_device(d)
        ttnn.ReadDeviceProfiler(d)
        p = pcc(ot)
        if p < 0.99:
            ok = False
    except Exception as e:
        ok = False
    man.append((tag, ok, p))


# 1) auto-block baseline (no config, full auto path -> heuristic S/Pk + auto block)
for k in ("TT_MM_NUM_SLICES", "TT_MM_K_SLICES", "TT_MM_K_FUSED", "TT_MM_NO_AUTO_KPAR"):
    os.environ.pop(k, None)
run(None, "AUTO")

# 2) block sweep at the heuristic (S,Pk), pinned via env
os.environ["TT_MM_NUM_SLICES"] = str(S)
if Pk > 1:
    os.environ["TT_MM_K_SLICES"] = str(Pk)
    os.environ["TT_MM_K_FUSED"] = "1"
ncfg = 0
for mb in M_BLOCKS:
    for nb in N_BLOCKS:
        sbh, sbw = adsub(mb, nb)
        if sbh * sbw != 4 and mb * nb >= 4:
            continue  # max-DST subblock only (except blocks too small to ever reach a 4-tile subblock)
        for kb in K_BLOCKS:
            if footprint(mb, kb, nb) > L1_CB_BUDGET:
                continue  # OOM guard (matches factory budget)
            cfg = ttnn.MinimalMatmulConfig(
                M_block_size=mb,
                K_block_size=kb,
                N_block_size=nb,
                subblock_h=sbh,
                subblock_w=sbw,
                compute_with_storage_grid_size=ttnn.CoreCoord(GX, GY),
            )
            run(cfg, (mb, kb, nb, sbh, sbw))
            ncfg += 1
ttnn.close_device(d)
print(f"ran {ncfg} block configs (+AUTO)", flush=True)

# parse durations in order; only OK configs consumed a CHUNK
ds = durs()
i = 0
res = []
for tag, ok, p in man:
    if ok and len(ds[i : i + CHUNK]) == CHUNK:
        us = statistics.median(ds[i : i + CHUNK][-REPS:]) / 1000
        res.append((tag, us, p))
        i += CHUNK
    elif ok:
        i += CHUNK  # ran but parse short; skip
auto = next((r for r in res if r[0] == "AUTO"), None)
blocks = [r for r in res if r[0] != "AUTO"]
blocks.sort(key=lambda r: r[1])
# dump ALL configs for offline prune analysis
dump = {
    "shape": [M, K, N],
    "pcM": pcM,
    "pcN": pcN,
    "S": S,
    "Pk": Pk,
    "auto_us": auto[1] if auto else None,
    "configs": [
        {"mb": t[0], "kb": t[1], "nb": t[2], "sbh": t[3], "sbw": t[4], "us": us, "util": util(us)}
        for (t, us, p) in blocks
    ],
}
json.dump(dump, open(os.path.join(os.environ.get("BSWEEP_OUT", "."), f"bsweep_{M}x{K}x{N}.json"), "w"), indent=0)
print("\n=== AUTO-block default ===", flush=True)
if auto:
    print(f"  AUTO  us={auto[1]:.1f}  util={util(auto[1]):.1f}%  PCC={auto[2]:.5f}", flush=True)
print(f"\n=== top 12 swept blocks (of {len(blocks)} valid PCC-pass) ===", flush=True)
print(
    f"  {'M_block':>7} {'K_block':>7} {'N_block':>7} {'sbh x sbw':>9} {'us':>8} {'util%':>6} {'PCC':>7} {'divides?':>9}",
    flush=True,
)
for (mb, kb, nb, sbh, sbw), us, p in blocks[:12]:
    div = (
        "M&N"
        if (pcM % mb == 0 and pcN % nb == 0)
        else ("M" if pcM % mb == 0 else ("N" if pcN % nb == 0 else "neither"))
    )
    print(f"  {mb:>7} {kb:>7} {nb:>7} {f'{sbh}x{sbw}':>9} {us:>8.1f} {util(us):>6.1f} {p:>7.4f} {div:>9}", flush=True)
if auto and blocks:
    best = blocks[0]
    print(
        f"\nbest-swept {best[1]:.1f}us ({util(best[1]):.1f}%) vs AUTO {auto[1]:.1f}us ({util(auto[1]):.1f}%) -> {auto[1]/best[1]:.3f}x",
        flush=True,
    )
print("DONE", flush=True)
