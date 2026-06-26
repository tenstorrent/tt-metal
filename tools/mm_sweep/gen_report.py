import os, json, types, collections, math, sys

# pure helpers (pick_S_Pk) from block_sweep_mp
src = open(os.path.join(os.path.dirname(__file__), "block_sweep_mp.py")).read().split("# ====")[0]
_m = types.ModuleType("_m")
_m.__dict__["os"] = os
_m.__dict__["math"] = math
exec(src, _m.__dict__)
PEAK = 2048 * 64 * 1e9
OUTD = "/home/cglagovich/mm_sweep_out"


def util(M, K, N, us):
    return 100 * 2 * M * K * N / (PEAK * us * 1e-6) if us else None


def best_of(rows):
    ok = [r for r in rows if r["pcc"] > 0.99 and r.get("us")]
    return min(ok, key=lambda r: r["us"]) if ok else None


# joint-sweep results (best over all swept S,Pk) — full 82-shape joint sweep across the galaxies
JOINT_SRC = os.environ.get("JOINT_SRC", f"{OUTD}/all_joint_out.json")
cand_rows = collections.defaultdict(list)
for r in json.load(open(JOINT_SRC)):
    cand_rows[(r["M"], r["K"], r["N"])].append(r)
cand_shapes = set(cand_rows)  # every shape was joint-swept

# main baseline (S=1,Pk=1, levers off) best-µs per shape, for the branch-vs-main comparison
BASE = {}
try:
    bb = collections.defaultdict(list)
    for r in json.load(open(f"{OUTD}/baseline_out.json")):
        bb[(r["M"], r["K"], r["N"])].append(r)
    BASE = {sh: best_of(rs) for sh, rs in bb.items() if best_of(rs)}
except Exception:
    pass


def section(name, shapes_json, out_json):
    order = [tuple(s) for s in json.load(open(shapes_json))]
    base = collections.defaultdict(list)
    for r in json.load(open(out_json)):
        base[(r["M"], r["K"], r["N"])].append(r)
    L = [f"## {name}", ""]
    L.append("| shape (M×K×N) | S | Pk | #cfg | best mb/kb/nb | sbh×sbw | µs | util% | main µs | vs main | note |")
    L.append("|---|---|---|---|---|---|---|---|---|---|---|")
    utils = []
    speeds = []
    for sh in order:
        M, K, N = sh
        Mt, Nt, Kt = M // 32, N // 32, K // 32
        hS, hPk = _m.pick_S_Pk(Mt, Nt, Kt)
        note = ""
        if sh in cand_shapes:  # joint-swept over (S,Pk): take the global best, flag heuristic misses
            rs = cand_rows[sh]
            b = best_of(rs)
            ncfg = len(rs)
            heur = best_of([r for r in rs if r["S"] == hS and r["Pk"] == hPk])
            bu = util(M, K, N, b["us"])
            hu = util(M, K, N, heur["us"]) if heur else None
            if (b["S"], b["Pk"]) != (hS, hPk) and hu and bu > hu * 1.02:
                note = f"**joint S,Pk sweep — beats heuristic** (was S={hS},Pk={hPk}: {hu:.1f}% @ {heur['us']:.0f}µs, +{bu-hu:.1f}%)"
            else:
                note = f"joint S,Pk sweep — heuristic (S={hS},Pk={hPk}) optimal"
        else:
            b = best_of(base.get(sh, []))
            ncfg = len(base.get(sh, []))
        if not b:
            L.append(f"| {M}×{K}×{N} | – | – | {ncfg} | (no PCC-pass) | – | – | – | – | – | {note} |")
            continue
        u = util(M, K, N, b["us"])
        utils.append(u)
        mb_ = BASE.get(sh)
        if mb_:
            sp = mb_["us"] / b["us"]
            speeds.append(sp)
            mcol, spcol = f"{mb_['us']:.1f}", f"**{sp:.2f}×**" if sp >= 1.02 else (
                f"{sp:.2f}×" if sp >= 0.98 else f"⚠{sp:.2f}×"
            )
        else:
            mcol, spcol = "–", "–"
        L.append(
            f"| {M}×{K}×{N} | {b['S']} | {b['Pk']} | {ncfg} | {b['mb']}/{b['kb']}/{b['nb']} | "
            f"{b['sbh']}×{b['sbw']} | {b['us']:.1f} | {u:.1f} | {mcol} | {spcol} | {note} |"
        )
    gm = math.exp(sum(math.log(x) for x in utils) / len(utils)) if utils else 0
    gsp = math.exp(sum(math.log(x) for x in speeds) / len(speeds)) if speeds else 0
    L += [
        "",
        f"_{len(utils)} shapes; geomean util **{gm:.1f}%**; geomean branch speedup vs main baseline "
        f"**{gsp:.2f}×**._",
        "",
    ]
    return "\n".join(L)


md = ["# minimal_matmul block sweep — best block per shape (WH Galaxy, 32 chips)", ""]
md.append(
    "Per-shape best (M/K/N block tiles + subblock) at the chosen **(S, Pk)** partition (S = N-slices, "
    "Pk = K-parallelism). Device time = median profiler FW-zone over REPS; util vs 2048×64 GFLOP/s peak "
    "(bf16 in/out, fp32 acc, HiFi2, 8×8 grid). Best = lowest µs among PCC≥0.99 configs."
)
md.append("")
md.append(
    "**Comparison:** *main µs* = the optimized **main baseline** (explicit blocks swept at S=1,Pk=1, large-N DRAM levers + auto prefetch/K-par disabled => main-equivalent dataflow); *vs main* = branch speedup (>1 = branch faster). "
)
md.append("")
md.append(
    "**S,Pk column:** most shapes ran at the auto-heuristic (S,Pk). Shapes with a **note** were "
    "*joint-swept* over (S,Pk)∈{1,2,4}×{1,2,4} (S·Pk≤8); their row shows the best-found (S,Pk), and the "
    "note flags where a non-heuristic partition beat the heuristic and by how much."
)
md.append("")
md.append(section("LTX shapes", f"{OUTD}/ltx.json", f"{OUTD}/ltx_out.json"))
md.append(section("FLUX shapes", f"{OUTD}/flux.json", f"{OUTD}/flux_out.json"))
out = sys.argv[1] if len(sys.argv) > 1 else "tools/mm_sweep/mm_sweep_results.md"
open(out, "w").write("\n".join(md))
print("wrote", out)
