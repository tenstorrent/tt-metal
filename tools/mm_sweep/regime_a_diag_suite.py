#!/usr/bin/env python3
# Driver for the regime_a_matmul test-only diagnostic ablations. Launches the ttnn gtest
# `unit_tests_ttnn --gtest_filter=RegimeADiagFixture.Run` once per (shape, config, mask) with the params in
# env vars; the test drives the INTERNAL ttnn::prim::regime_a_matmul_diag entry (mask in the hashed op
# param, never Python/nanobind). One config per process => the device-profiler CSV flushes on TearDown and
# is parsed here (per-RISC + wall). Ablations are non-additive critical-path counterfactuals.
#
# Modes:
#   smoke                 -> run every mask once on the 256x2048x1024 winner (no-deadlock check)
#   matrix                -> 4 shapes x {winner, best pure-K, best KxM, best KxN} x 8 ablations, 3x relaunch
#   mscale                -> full + key ablations over M={32,64,128,256} x {(2048,1024),(6144,4608)}
import json, os, statistics, subprocess, sys

sys.path.insert(0, os.path.dirname(__file__))
import oracle_ablate as oa  # reuse parse_csv (per-RISC + wall from the profiler CSV)
import regime_a_bench as rb  # reuse classify_timeout

HERE = os.path.dirname(__file__)
ROOT = oa.ROOT
BIN = f"{ROOT}/build_Release/test/ttnn/unit_tests_ttnn"
FREQ = 1.35e9
PEAK512 = 512e9

# (name, mask). LOCAL_FEED (16) is intentionally not implemented yet (see MT8_FINDINGS).
ABLATIONS = [
    ("full", 0),
    ("skipin1", 1),
    ("skipin0", 2),
    ("skipfwd", 4),
    ("noreduce", 8),
    ("skipin0+in1", 3),
    ("skipin0+in1+noreduce", 11),
]
PRIMARY = [(256, 2048, 1024), (256, 6144, 768), (256, 6144, 2304), (256, 6144, 4608)]


def cdiv(a, b):
    return (a + b - 1) // b


def ideal_us(M, K, N):
    Mt, Kt, Nt = cdiv(M, 32), cdiv(K, 32), cdiv(N, 32)
    return (Mt * Kt + Kt * Nt + Mt * Nt) * 2048 / PEAK512 * 1e6


def _reset():
    subprocess.run(["tt-smi", "-r"], capture_output=True)


def run_one(M, K, N, cfg, mask, iters=8, timeout=150):
    """Launch the gtest for one (shape,cfg,mask); parse per-RISC + wall from the profiler CSV. Hang-safe."""
    Ns, Pk, Sm, kb, nsb = cfg
    try:
        os.remove(oa.BIN_CSV)
    except OSError:
        pass
    env = dict(os.environ)
    env.update(
        TT_METAL_DEVICE_PROFILER="1",
        TT_METAL_HOME=ROOT,
        ARCH_NAME="blackhole",
        RA_M=str(M),
        RA_K=str(K),
        RA_N=str(N),
        RA_NS=str(Ns),
        RA_PK=str(Pk),
        RA_SM=str(Sm),
        RA_KB=str(kb),
        RA_NSB=str(nsb),
        RA_MASK=str(mask),
        RA_ITERS=str(iters),
    )
    cmd = ["timeout", "-s", "TERM", str(timeout), BIN, "--gtest_filter=RegimeADiagFixture.Run"]
    try:
        r = subprocess.run(cmd, env=env, cwd=ROOT, capture_output=True, text=True, timeout=timeout + 30)
    except subprocess.TimeoutExpired:
        subprocess.run(["pkill", "-9", "-x", "unit_tests_ttnn"], capture_output=True)
        _reset()
        return {"cfg": list(cfg), "mask": mask, "ok": False, "cls": "hang", "wall_us": None}
    done = "DIAGDONE" in r.stdout
    passed = "[  PASSED  ]" in r.stdout
    if rb.classify_timeout(r.returncode) or not done:
        _reset()
        return {
            "cfg": list(cfg),
            "mask": mask,
            "ok": False,
            "cls": "hang" if rb.classify_timeout(r.returncode) else "fail",
            "rc": r.returncode,
            "wall_us": None,
            "stderr": r.stderr[-300:],
        }
    by, wall, _ = oa.parse_csv()
    wall_us = statistics.median(wall) / FREQ * 1e6 if wall else None
    risc = {
        t: {
            "n": len(c),
            "max_us": max(c) / FREQ * 1e6,
            "med_us": statistics.median(c) / FREQ * 1e6,
            "min_us": min(c) / FREQ * 1e6,
        }
        for t, c in sorted(by.items())
    }
    maxrel = None
    for line in r.stdout.splitlines():
        if "DIAGPCC" in line:
            maxrel = float(line.split("max_rel_err=")[1])
    return {
        "cfg": list(cfg),
        "mask": mask,
        "ok": bool(wall_us) and (mask != 0 or passed),
        "cls": "ok",
        "wall_us": wall_us,
        "risc": risc,
        "max_rel_err": maxrel,
    }


def pick_configs(M, K, N):
    """winner + best pure-K / KxM / KxN from the practical sweep JSON (falls back to auto if absent)."""
    p = f"{HERE}/regime_a_sweep_{M}x{K}x{N}.json"
    picks = {}
    if os.path.exists(p):
        res = [r for r in json.load(open(p))["results"] if r.get("cls") == "ok"]
        res.sort(key=lambda r: r["us_med"])

        def best(pred):
            for r in res:
                if pred(tuple(r["cfg"])):
                    return tuple(r["cfg"])
            return None

        picks["winner"] = tuple(res[0]["cfg"]) if res else None
        picks["pureK"] = best(lambda c: c[0] == 1 and c[2] == 1)
        picks["KxM"] = best(lambda c: c[2] > 1 and c[0] == 1)
        picks["KxN"] = best(lambda c: c[0] > 1 and c[2] == 1)
    # de-dup while keeping labels
    out = {}
    for label, c in picks.items():
        if c is not None:
            out.setdefault(c, []).append(label)
    return out  # {cfg: [labels]}


def smoke():
    M, K, N, cfg = 256, 2048, 1024, (1, 4, 2, 2, 2)
    for name, mask in ABLATIONS:
        r = run_one(M, K, N, cfg, mask)
        w = r.get("wall_us")
        print(
            f"[smoke] {name:22} mask={mask:2} cls={r['cls']} wall={w if w is None else round(w,1)}us "
            f"pcc_err={r.get('max_rel_err')}",
            flush=True,
        )


def matrix():
    out = []
    for M, K, N in PRIMARY:
        cfgs = pick_configs(M, K, N)
        print(
            f"\n=== {M}x{K}x{N} ideal={ideal_us(M,K,N):.1f}us configs={ {','.join(v):list(k) for k,v in cfgs.items()} }",
            flush=True,
        )
        for cfg, labels in cfgs.items():
            lab = "/".join(labels)
            for name, mask in ABLATIONS:
                runs = [run_one(M, K, N, cfg, mask) for _ in range(3)]
                oks = [x for x in runs if x["cls"] == "ok" and x["wall_us"]]
                wall = min(x["wall_us"] for x in oks) if oks else None  # min over relaunches (steadiest)
                med = statistics.median([x["wall_us"] for x in oks]) if oks else None
                rec = {
                    "M": M,
                    "K": K,
                    "N": N,
                    "cfg": list(cfg),
                    "labels": labels,
                    "ablation": name,
                    "mask": mask,
                    "wall_us_min": wall,
                    "wall_us_med": med,
                    "n_ok": len(oks),
                    "ideal_us": ideal_us(M, K, N),
                    "risc": (oks[0]["risc"] if oks else None),
                }
                out.append(rec)
                print(
                    f"  [{lab:14}] {name:22} wall_med={med if med is None else round(med,1)}us " f"({len(oks)}/3 ok)",
                    flush=True,
                )
                json.dump(out, open(f"{HERE}/regime_a_diag_matrix.json", "w"), indent=2)
    print("MATRIX DONE", flush=True)


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "smoke"
    {"smoke": smoke, "matrix": matrix}[mode]()
