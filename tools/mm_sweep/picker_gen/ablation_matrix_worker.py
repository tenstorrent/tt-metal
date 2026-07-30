#!/usr/bin/env python3
"""One persistent-session relaunch for the 22-mode critical-path ablation matrix.

ONE device session measures an arbitrary ordered list of diagnostic masks as sequential blocks (order
given by argv; the supervisor uses forward on even relaunches, reverse on odd). Per block: 2 warmup + 12
timed resident-input iterations. The 6-bit mask is switched via TT_REGIME_A_DIAG_MASK between blocks
(invoke() reads it into the reflection-hashed attribute => distinct cached program per mask). Per-mode
kernel wall + per-RISC spans are recovered from the device-profiler CSV (flushed on close) by RUN-HOST-ID
demux: every op invocation is labelled in issue order and zipped against chronologically-sorted run-host-ids.

Baseline (mask 0) PCC + cached-program replay verified once (when --verify).

argv: M K N Ns Pk Sm kb nsb masks(csv of ints) verify(0/1)
Emits one JSON line: {mask: {samples_us[12], median_us, risc_spans_us}} + baseline pcc/replay.
"""
import sys, os, csv, json, statistics
import torch
import ttnn

ROOT = os.environ.get("TT_METAL_HOME", os.getcwd())
CSV = f"{ROOT}/generated/profiler/.logs/profile_log_device.csv"
FREQ = 1.35e9
WARMUP, TIMED = 2, 12

M, K, N, Ns, Pk, Sm, kb, nsb = (int(x) for x in sys.argv[1:9])
MASKS = [int(x) for x in sys.argv[9].split(",")]
VERIFY = int(sys.argv[10])


def parse_runids():
    if not os.path.exists(CSV):
        return {}
    raw = {}
    for row in csv.reader(open(CSV)):
        if len(row) < 12:
            continue
        if not row[10].strip().endswith("-KERNEL"):
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
    res = {"outcome": "runtime", "err": "", "masks": MASKS}
    labels = []
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

        if VERIFY:
            os.environ["TT_REGIME_A_DIAG_MASK"] = "0"
            o1 = call("pre")
            ttnn.synchronize_device(dev)
            h1 = ttnn.to_torch(ttnn.from_device(o1))
            o2 = call("pre")
            ttnn.synchronize_device(dev)
            h2 = ttnn.to_torch(ttnn.from_device(o2))
            ref = t0.float() @ t1.float()
            res["baseline_pcc"] = round(pcc(ref, h1), 6)
            res["cached_replay_pcc"] = round(pcc(ref, h2), 6)
            res["cached_replay_matches"] = bool(torch.equal(h1, h2))

        for mk in MASKS:
            os.environ["TT_REGIME_A_DIAG_MASK"] = str(mk)
            for _ in range(WARMUP):
                call(f"m{mk}_w")
            for _ in range(TIMED):
                call(f"m{mk}_t")
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
        res["err"] = f"demux misalign {len(order_ids)} vs {len(labels)}"
        print(json.dumps(res), flush=True)
        return
    modes = {}
    for mk in MASKS:
        tids = [i for i, lab in zip(order_ids, labels) if lab == f"m{mk}_t"]
        if not tids:
            continue
        walls = [rid[i]["wall"] for i in tids]
        risc = {}
        for rn in ("BRISC", "NCRISC", "TRISC"):
            xs = [rid[i]["risc"][rn] for i in tids if rid[i]["risc"][rn] is not None]
            risc[rn] = round(statistics.median(xs), 3) if xs else None
        modes[str(mk)] = {
            "samples_us": [round(w, 3) for w in walls],
            "median_us": round(statistics.median(walls), 3),
            "risc_spans_us": risc,
        }
    res["outcome"] = "ok"
    res["modes"] = modes
    print(json.dumps(res), flush=True)


if __name__ == "__main__":
    main()
