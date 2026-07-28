#!/usr/bin/env python3
"""60-shape Mt<=8 corpus A/B of the whole-op link-balanced in0 ring order (diag bit10) vs production.

Promotion gate for bit10: measured at config=None (the DEPLOYED picker config, i.e. what would ship), two
relaunches per shape with the mask order reversed on the second, resumable per-shape JSONL checkpoints, and a
relative-PCC correctness check per shape (bit10 is host-only and correctness-preserving).

The factory's adopt/keep decision is captured from its debug log when available (TT_METAL_LOGGER_LEVEL=Debug),
so shapes can be split into "reordered" vs "kept production" — a shape the gate declines must be exactly
neutral, and any drift there is a harness/noise signal rather than a model error.

usage: corpus_ab.py [--relaunches N] [--filter MxKxN] [--masks 0,1024]
"""
import json, os, re, statistics, subprocess, sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.environ.get("TT_METAL_HOME", os.path.abspath(f"{HERE}/../.."))
WORKER = f"{HERE}/corpus_ab_worker.py"
CORPUS = f"{HERE}/../regime_a_current_perf.json"
OUT = f"{HERE}/results_v2/corpus_ab.jsonl"
MASKS = "0,1024"
RELAUNCHES = 2
FILTER = None

args = sys.argv[1:]
while args:
    a = args.pop(0)
    if a == "--relaunches":
        RELAUNCHES = int(args.pop(0))
    elif a == "--filter":
        FILTER = args.pop(0)
    elif a == "--masks":
        MASKS = args.pop(0)

MASK_LIST = [int(x) for x in MASKS.split(",")]


def done_keys():
    keys = set()
    if os.path.exists(OUT):
        for line in open(OUT):
            try:
                r = json.loads(line)
                keys.add((r["M"], r["K"], r["N"], r["relaunch"]))
            except Exception:  # noqa: BLE001
                pass
    return keys


def run(M, K, N, masks, verify):
    env = dict(os.environ)
    env.update(
        TT_METAL_DEVICE_PROFILER="1", TT_METAL_HOME=ROOT, ARCH_NAME="blackhole", TT_METAL_LOGGER_LEVEL="Debug"
    )
    try:
        p = subprocess.run(
            [sys.executable, WORKER, str(M), str(K), str(N), ",".join(str(m) for m in masks), str(verify)],
            env=env,
            cwd=ROOT,
            capture_output=True,
            text=True,
            timeout=600,
        )
    except subprocess.TimeoutExpired:
        subprocess.run(["pkill", "-9", "-f", "corpus_ab_worker"], capture_output=True)
        return {"outcome": "hang", "err": "timeout", "M": M, "K": K, "N": N}
    rec = None
    for line in p.stdout.splitlines():
        if line.startswith('{"outcome"'):
            rec = json.loads(line)
    if rec is None:
        return {"outcome": "runtime", "err": (p.stderr or p.stdout)[-300:], "M": M, "K": K, "N": N}
    # capture the factory's ring-balance decision (debug log)
    dec = re.findall(
        r"ring balance: background peak (\d+) B, production peak (\d+) B, balanced peak (\d+) B -> (ADOPT|keep)",
        p.stdout + p.stderr)
    if dec:
        bg, pp, np_, verb = dec[-1]
        rec["balance"] = {"bg": int(bg), "prod_peak": int(pp), "bal_peak": int(np_), "adopt": verb == "ADOPT"}
    return rec


def main():
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    corpus = json.load(open(CORPUS))["mt8"]
    shapes = sorted({(r["M"], r["K"], r["N"]) for r in corpus}, key=lambda z: (z[1], z[0], z[2]))
    # The corpus is Mt<=8 (M<=256). The two shapes with the largest measured in0-ring exposure are M=512
    # (Mt=16), i.e. outside it, so add the four golden shapes explicitly — a ring-order change should matter
    # most exactly where the ring is most exposed.
    for extra in ((256, 2048, 2048), (256, 2048, 6144), (512, 6144, 2304), (512, 6144, 4608)):
        if extra not in shapes:
            shapes.append(extra)
    if FILTER:
        shapes = [s for s in shapes if f"{s[0]}x{s[1]}x{s[2]}" == FILTER]
    have = done_keys()
    total = len(shapes) * RELAUNCHES
    n = 0
    for rl in range(RELAUNCHES):
        masks = MASK_LIST if rl % 2 == 0 else MASK_LIST[::-1]
        for (M, K, N) in shapes:
            n += 1
            if (M, K, N, rl) in have:
                continue
            rec = run(M, K, N, masks, 1 if rl == 0 else 0)
            rec["relaunch"] = rl
            rec.setdefault("M", M)
            rec.setdefault("K", K)
            rec.setdefault("N", N)
            with open(OUT, "a") as f:
                f.write(json.dumps(rec) + "\n")
                f.flush()
                os.fsync(f.fileno())
            if rec.get("outcome") == "ok":
                md = rec["modes"]
                b = md.get(str(MASK_LIST[0]), {}).get("median_us")
                v = md.get(str(MASK_LIST[1]), {}).get("median_us")
                d = (100 * (b - v) / b) if (b and v) else None
                ad = rec.get("balance", {}).get("adopt")
                print(
                    f"[{n}/{total}] {M}x{K}x{N} rl{rl}: base={b} bit10={v} "
                    f"{('%+.2f%%' % d) if d is not None else 'n/a'} adopt={ad}",
                    flush=True,
                )
            else:
                print(f"[{n}/{total}] {M}x{K}x{N} rl{rl}: {rec.get('outcome')} {rec.get('err','')[:120]}", flush=True)
    summarize()


def summarize():
    if not os.path.exists(OUT):
        print("no results yet")
        return
    rows = {}
    for line in open(OUT):
        r = json.loads(line)
        if r.get("outcome") != "ok":
            continue
        key = (r["M"], r["K"], r["N"])
        rows.setdefault(key, []).append(r)
    print(f"\n{'shape':18s} {'base':>8s} {'bit10':>8s} {'delta%':>8s} {'adopt':>6s} {'relPCC':>8s}")
    deltas, wins, regs, adopted = [], [], [], 0
    for key in sorted(rows, key=lambda z: (z[1], z[0], z[2])):
        rs = rows[key]
        ds, bs, vs = [], [], []
        for r in rs:
            b = r["modes"].get("0", {}).get("median_us")
            v = r["modes"].get("1024", {}).get("median_us")
            if b and v:
                ds.append(100 * (b - v) / b)
                bs.append(b)
                vs.append(v)
        if not ds:
            continue
        d = statistics.median(ds)
        ad = any(r.get("balance", {}).get("adopt") for r in rs)
        adopted += 1 if ad else 0
        pc = min((r.get("rel_pcc", {}).get("1024", 1.0) for r in rs if "rel_pcc" in r), default=None)
        deltas.append(d)
        (wins if d >= 2 else regs if d <= -2 else []).append((key, d))
        print(
            f"{f'{key[0]}x{key[1]}x{key[2]}':18s} {statistics.median(bs):8.2f} {statistics.median(vs):8.2f} "
            f"{d:+8.2f} {str(ad):>6s} {('%.5f' % pc) if pc is not None else '   n/a':>8s}"
        )
    print(f"\nmeasured {len(deltas)} shapes; adopted (reordered) on {adopted}")
    if deltas:
        print(f"median {statistics.median(deltas):+.2f}%  mean {statistics.mean(deltas):+.2f}%  "
              f"best {max(deltas):+.2f}%  worst {min(deltas):+.2f}%")
        print(f"wins >=2%: {len(wins)}   regressions <=-2%: {len(regs)}   neutral: {len(deltas)-len(wins)-len(regs)}")
        for tag, lst in (("WINS", wins), ("REGRESSIONS", regs)):
            if lst:
                print(f"  {tag}: " + ", ".join(f"{k[0]}x{k[1]}x{k[2]} {d:+.2f}%" for k, d in
                                              sorted(lst, key=lambda z: -abs(z[1]))))


if __name__ == "__main__":
    if "--summary" in sys.argv:
        summarize()
    else:
        main()
