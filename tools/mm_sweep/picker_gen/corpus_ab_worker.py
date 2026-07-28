#!/usr/bin/env python3
"""One persistent-session A/B of diagnostic masks on the 60-shape Mt<=8 corpus, at the DEPLOYED config.

config=None => the production auto-picker chooses, so this measures what would actually ship (the golden
4-shape configs are hand-specified and are NOT picker picks). diag_mask is part of the reflection program
hash, so each mask gets its own cached program; the picked config is identical across masks (auto_select_config
does not see diag_mask).

Per mask: 2 warmup + 12 timed resident-input iterations; kernel wall via device-profiler run-host-id demux.
Correctness here is RELATIVE (PCC of each mask's output vs mask 0) — cheap and exactly the property a
correctness-preserving host-side change must have; absolute correctness is covered by the 111-test regression
run separately under the mask.

argv: M K N masks(csv) verify(0/1)
Emits one JSON line: {mask: {samples_us, median_us, risc_spans_us}}, rel_pcc per mask, plus any
"ring balance" decision lines the factory logged (adopt vs keep production).
"""
import sys, os, csv, json, statistics
import torch
import ttnn

ROOT = os.environ.get("TT_METAL_HOME", os.getcwd())
CSV = f"{ROOT}/generated/profiler/.logs/profile_log_device.csv"
FREQ = 1.35e9
WARMUP, TIMED = 2, 12

M, K, N = (int(x) for x in sys.argv[1:4])
MASKS = [int(x) for x in sys.argv[4].split(",")]
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
    res = {"outcome": "runtime", "err": "", "M": M, "K": K, "N": N, "masks": MASKS}
    labels = []
    dev = ttnn.open_device(device_id=0)
    ran = False
    try:
        t0 = torch.randn(1, 1, M, K, dtype=torch.bfloat16)
        t1 = torch.randn(1, 1, K, N, dtype=torch.bfloat16)
        a = ttnn.from_torch(t0, layout=ttnn.TILE_LAYOUT, device=dev, dtype=ttnn.bfloat16)
        wc = ttnn.create_regime_a_weight_memory_config(list(t1.shape), ttnn.bfloat16, dev)
        b = ttnn.from_torch(t1, layout=ttnn.TILE_LAYOUT, device=dev, dtype=ttnn.bfloat16, memory_config=wc)

        def call(label):
            out = ttnn.experimental.regime_a_matmul(a, b)  # config=None => deployed picker
            labels.append(label)
            return out

        if VERIFY:
            ref = {}
            for mk in MASKS:
                os.environ["TT_REGIME_A_DIAG_MASK"] = str(mk)
                o = call("pre")
                ttnn.synchronize_device(dev)
                ref[mk] = ttnn.to_torch(ttnn.from_device(o))
            base = ref[MASKS[0]] if 0 not in ref else ref[0]
            res["rel_pcc"] = {str(mk): round(pcc(base, ref[mk]), 6) for mk in MASKS}
            res["rel_exact"] = {str(mk): bool(torch.equal(base, ref[mk])) for mk in MASKS}

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
    finally:
        ttnn.close_device(dev)

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
