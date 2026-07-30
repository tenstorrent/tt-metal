#!/usr/bin/env python3
"""One persistent-session sweep of the in1 CB depth (TT_REGIME_A_CB1_DEPTH) at the DEPLOYED config.

cb1 holds `depth * kb * N_sub` tiles, so `depth` sets how many in1 blocks a reader can keep in flight.
Production is 4. If the in1 read is bound by latency x concurrency rather than DRAM bandwidth, raising the
depth should shrink the wall; if it is bandwidth- or issue-rate-bound, it should not move.

cb1_depth is a hashed operation attribute, so each depth is its own cached program and several depths can be
measured in one device session. It is correctness-preserving (pure buffering) -> PCC is checked at EVERY
depth, and bit-exactness against the production depth is reported. A depth that overflows the L1 budget makes
the planner reject the config; that is reported as outcome=infeasible rather than being silently clamped.

argv: M K N depths(csv) verify(0/1)
"""
import sys, os, csv, json, statistics
import torch
import ttnn

ROOT = os.environ.get("TT_METAL_HOME", os.getcwd())
CSV = f"{ROOT}/generated/profiler/.logs/profile_log_device.csv"
FREQ = 1.35e9
WARMUP, TIMED = 2, 12

M, K, N = (int(x) for x in sys.argv[1:4])
DEPTHS = [int(x) for x in sys.argv[4].split(",")]
VERIFY = int(sys.argv[5])


def parse_runids():
    if not os.path.exists(CSV):
        return {}
    raw = {}
    for row in csv.reader(open(CSV)):
        if len(row) < 12 or not row[10].strip().endswith("-KERNEL"):
            continue
        raw.setdefault(row[7].strip(), {}).setdefault((row[1], row[2], row[3]), []).append(
            (row[11].strip(), int(row[5]))
        )
    out = {}
    for runid, cr in raw.items():
        durs, start = {}, None
        for k, lst in cr.items():
            st = None
            for t, c in lst:
                if t == "ZONE_START":
                    st = c
                    start = c if start is None else min(start, c)
                elif t == "ZONE_END" and st is not None:
                    durs[k] = c - st
                    st = None
        if not durs:
            continue

        def risc(pfx):
            vs = [d for kk, d in durs.items() if kk[2].startswith(pfx)]
            return max(vs) / FREQ * 1e6 if vs else None

        out[runid] = {
            "wall": max(durs.values()) / FREQ * 1e6,
            "start": start if start is not None else 0,
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
    res = {"outcome": "runtime", "err": "", "M": M, "K": K, "N": N, "depths": DEPTHS}
    labels, infeasible, pccs, exact = [], [], {}, {}
    dev = ttnn.open_device(device_id=0)
    ok_depths = []
    try:
        t0 = torch.randn(1, 1, M, K, dtype=torch.bfloat16)
        t1 = torch.randn(1, 1, K, N, dtype=torch.bfloat16)
        a = ttnn.from_torch(t0, layout=ttnn.TILE_LAYOUT, device=dev, dtype=ttnn.bfloat16)
        wc = ttnn.create_regime_a_weight_memory_config(list(t1.shape), ttnn.bfloat16, dev)
        b = ttnn.from_torch(t1, layout=ttnn.TILE_LAYOUT, device=dev, dtype=ttnn.bfloat16, memory_config=wc)
        ref = t0.float() @ t1.float() if VERIFY else None
        base_out = None

        for d in DEPTHS:
            os.environ["TT_REGIME_A_CB1_DEPTH"] = str(d)
            try:
                if VERIFY:
                    o = ttnn.experimental.regime_a_matmul(a, b)
                    labels.append(f"d{d}_v")
                    ttnn.synchronize_device(dev)
                    h = ttnn.to_torch(ttnn.from_device(o))
                    pccs[str(d)] = round(pcc(ref, h), 6)
                    if base_out is None:
                        base_out = h
                    exact[str(d)] = bool(torch.equal(base_out, h))
                for _ in range(WARMUP):
                    ttnn.experimental.regime_a_matmul(a, b)
                    labels.append(f"d{d}_w")
                for _ in range(TIMED):
                    ttnn.experimental.regime_a_matmul(a, b)
                    labels.append(f"d{d}_t")
                ttnn.synchronize_device(dev)
                ok_depths.append(d)
            except Exception as e:  # noqa: BLE001  (planner rejection at this depth)
                infeasible.append({"depth": d, "err": str(e)[:200]})
                # drop the labels this depth may have appended before failing
                labels = [x for x in labels if not x.startswith(f"d{d}_")]
        ran = True
    except Exception as e:  # noqa: BLE001
        res["err"] = str(e)[:400]
        ran = False
    finally:
        ttnn.close_device(dev)

    res["infeasible"] = infeasible
    res["pcc"] = pccs
    res["bit_exact_vs_first"] = exact
    if not ran:
        print(json.dumps(res), flush=True)
        return
    rid = parse_runids()
    order_ids = sorted(rid, key=lambda r: rid[r]["start"])
    if len(order_ids) != len(labels):
        res["err"] = f"demux misalign {len(order_ids)} vs {len(labels)}"
        print(json.dumps(res), flush=True)
        return
    modes = {}
    for d in ok_depths:
        tids = [i for i, lab in zip(order_ids, labels) if lab == f"d{d}_t"]
        if not tids:
            continue
        walls = [rid[i]["wall"] for i in tids]
        risc = {}
        for rn in ("BRISC", "NCRISC", "TRISC"):
            xs = [rid[i]["risc"][rn] for i in tids if rid[i]["risc"][rn] is not None]
            risc[rn] = round(statistics.median(xs), 3) if xs else None
        modes[str(d)] = {
            "samples_us": [round(w, 3) for w in walls],
            "median_us": round(statistics.median(walls), 3),
            "risc_spans_us": risc,
        }
    res["outcome"] = "ok"
    res["modes"] = modes
    print(json.dumps(res), flush=True)


if __name__ == "__main__":
    main()
