#!/usr/bin/env python3
"""One persistent-session relaunch for the in0-read ablation (baseline / skip-redundant / skip-all).

ONE device session measures all three modes as sequential blocks (mode order given by argv, reversed on
alternate relaunches by the supervisor). Per block: 2 warmup + 16 timed resident-input iterations. The
test-only diagnostic mask is switched via the TT_REGIME_A_DIAG_IN0 env var between blocks (invoke() reads
it into the reflection-hashed attribute, so each mode is a distinct cached program). Per-mode kernel wall
+ per-RISC spans are recovered from the device-profiler CSV (flushed on close) by RUN-HOST-ID demux:
every op invocation is labelled in issue order and zipped against the chronologically-sorted run-host-ids.

Baseline PCC + cached-program replay are verified once (only when --verify is passed, i.e. relaunch 0).

argv: M K N Ns Pk Sm kb nsb order(csv of baseline,skip_redundant,skip_all) verify(0/1)
Emits one JSON line: per-mode 16-sample lists, per-mode per-RISC median spans, baseline pcc/replay.
"""
import sys, os, csv, json, statistics
import torch
import ttnn

ROOT = os.environ.get("TT_METAL_HOME", os.getcwd())
CSV = f"{ROOT}/generated/profiler/.logs/profile_log_device.csv"
FREQ = 1.35e9
WARMUP, TIMED = 2, 16
MASK = {"baseline": "0", "skip_redundant": "1", "skip_all": "2"}

M, K, N, Ns, Pk, Sm, kb, nsb = (int(x) for x in sys.argv[1:9])
ORDER = sys.argv[9].split(",")
VERIFY = int(sys.argv[10])


def parse_runids():
    """runid -> {'wall': max-over-(core,risc) us, 'start': min cyc, 'risc': {BRISC/NCRISC/TRISC: max us}}."""
    if not os.path.exists(CSV):
        return {}
    raw = {}
    for row in csv.reader(open(CSV)):
        if len(row) < 12:
            continue
        z = row[10].strip()
        if not z.endswith("-KERNEL"):
            continue
        runid = row[7].strip()
        raw.setdefault(runid, {}).setdefault((row[1], row[2], row[3]), []).append((row[11].strip(), int(row[5])))
    out = {}
    for runid, cr in raw.items():
        durs, start = {}, None
        for (x, y, proc), lst in cr.items():
            st = None
            for t, c in lst:
                if t == "ZONE_START":
                    st = c
                    start = c if start is None else min(start, c)
                elif t == "ZONE_END" and st is not None:
                    durs[(x, y, proc)] = c - st
                    st = None
        if not durs:
            continue
        wall = max(durs.values()) / FREQ * 1e6

        def risc(prefix):
            vs = [d for k, d in durs.items() if k[2].startswith(prefix)]
            return max(vs) / FREQ * 1e6 if vs else None

        out[runid] = {
            "wall": wall,
            "start": start,
            "risc": {"BRISC": risc("BRISC"), "NCRISC": risc("NCRISC"), "TRISC": risc("TRISC")},
        }
    return out


def pcc(a, b):
    a = a.flatten().float()
    b = b.flatten().float()
    a = a - a.mean()
    b = b - b.mean()
    d = (a.norm() * b.norm()).item()
    return 1.0 if d == 0 else torch.dot(a, b).item() / d


def main():
    try:
        os.remove(CSV)
    except OSError:
        pass
    res = {"outcome": "runtime", "err": "", "order": ORDER}
    labels = []  # one entry per op invocation, in issue order
    dev = ttnn.open_device(device_id=0)
    try:
        t0 = torch.randn(1, 1, M, K, dtype=torch.bfloat16)
        t1 = torch.randn(1, 1, K, N, dtype=torch.bfloat16)
        a = ttnn.from_torch(t0, layout=ttnn.TILE_LAYOUT, device=dev, dtype=ttnn.bfloat16)
        wc = ttnn.create_regime_a_weight_memory_config(list(t1.shape), ttnn.bfloat16, dev)
        b = ttnn.from_torch(t1, layout=ttnn.TILE_LAYOUT, device=dev, dtype=ttnn.bfloat16, memory_config=wc)
        cfg = ttnn.RegimeAMatmulConfig(k_slices=Pk, n_slices=Ns, m_slices=Sm, k_block_tiles=kb, n_subblock_tiles=nsb)

        def call(label):
            out = ttnn.experimental.regime_a_matmul(a, b, config=cfg)
            labels.append(label)
            return out

        # correctness preamble (baseline mask 0): PCC vs golden + cached-program replay, once per shape.
        if VERIFY:
            os.environ["TT_REGIME_A_DIAG_IN0"] = "0"
            out1 = call("pre")
            ttnn.synchronize_device(dev)
            h1 = ttnn.to_torch(ttnn.from_device(out1))
            out2 = call("pre")
            ttnn.synchronize_device(dev)
            h2 = ttnn.to_torch(ttnn.from_device(out2))
            ref = t0.float() @ t1.float()
            res["baseline_pcc"] = round(pcc(ref, h1), 6)
            res["cached_replay_pcc"] = round(pcc(ref, h2), 6)
            res["cached_replay_matches"] = bool(torch.equal(h1, h2))

        # timed blocks in the requested order
        for mode in ORDER:
            os.environ["TT_REGIME_A_DIAG_IN0"] = MASK[mode]
            for _ in range(WARMUP):
                call(f"{mode}_w")
            for _ in range(TIMED):
                call(f"{mode}_t")
            ttnn.synchronize_device(dev)
        ran = True
    except Exception as e:  # noqa: BLE001
        res["err"] = str(e)[:400]
        ran = False
    finally:
        ttnn.close_device(dev)

    if not ran:
        print(json.dumps(res), flush=True)
        return
    rid = parse_runids()
    order_ids = sorted(rid, key=lambda r: rid[r]["start"])
    if len(order_ids) != len(labels):
        res["outcome"] = "runtime"
        res["err"] = f"demux misalign: {len(order_ids)} runids vs {len(labels)} calls"
        print(json.dumps(res), flush=True)
        return
    per_mode = {}
    for m in MASK:
        tids = [i for i, lab in zip(order_ids, labels) if lab == f"{m}_t"]
        if not tids:
            continue
        walls = [rid[i]["wall"] for i in tids]
        risc = {}
        for rname in ("BRISC", "NCRISC", "TRISC"):
            xs = [rid[i]["risc"][rname] for i in tids if rid[i]["risc"][rname] is not None]
            risc[rname] = round(statistics.median(xs), 3) if xs else None
        per_mode[m] = {
            "samples_us": [round(w, 3) for w in walls],
            "median_us": round(statistics.median(walls), 3),
            "risc_spans_us": risc,
        }
    res["outcome"] = "ok"
    res["modes"] = per_mode
    print(json.dumps(res), flush=True)


if __name__ == "__main__":
    main()
