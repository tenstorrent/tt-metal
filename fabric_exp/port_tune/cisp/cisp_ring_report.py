#!/usr/bin/env python3
"""cisp_ring_report.py RESULTS_DIR PREFIX ARM CHIP:CORE [...]: per NX unicast_write iteration, the sender program's
launch slot and kernel config base (kcb), from the sender's BRISC PC samples (the launch slot whose image holds the PC,
in the core's mailbox snapshots), against the slow flag (sender row below 97% of the off arm's median). Prints one line
per iteration for the first core and kcb / launch-slot tallies per chip. (Ring sizes: Tensix kernel-config ring 69 KB
from MEM_MAP_END 0xa3a0, bh_hal_tensix.cpp; launch ring 8 slots.)"""
import bisect, collections, csv, json, os, statistics, struct, sys
sys.path.insert(0, "/home/aagarwal/tt-metal/fabric_exp/consolidation")
sys.path.insert(0, "/home/aagarwal/tt-metal/fabric_exp/port_tune/cisp")
import hang_workers as hw, cisp_pcs2_report as P
D, PRE, arm = sys.argv[1], sys.argv[2], sys.argv[3]
cores = sys.argv[4:]
pre = os.path.join(D, f"{PRE}_{arm}_samples")
meta = json.load(open(pre + ".json"))
snaps = collections.defaultdict(list)
raw = open(pre + ".snap.bin", "rb").read(); step = 10 + 1488
for o in range(0, len(raw) - step + 1, step):
    t, k = struct.unpack_from("<dH", raw, o)
    hd = hw.parse_head(raw[o + 10:o + 10 + 1488])
    if hd: snaps[k].append((t + meta["t_start_epoch"], hd["slots"]))
ci = {(c["chip"], c["core"]): k for k, c in enumerate(meta["snap_cores"])}
win = P.windows(os.path.join(D, f"{PRE}_{arm}.log")); starts = [w[2] for w in win]
rows = list(csv.DictReader(open(os.path.join(D, f"{PRE}_{arm}_bandwidth_results.csv"))))
offm = collections.defaultdict(list)
for r in csv.DictReader(open(os.path.join(D, f"{PRE}_off_bandwidth_results.csv"))): offm[r["test_name"]].append(float(r["bandwidth_GB_s"]))
want = {(int(c.split(":")[0]), c.split(":")[1]) for c in cores}
it = collections.defaultdict(collections.Counter)
for t, i, pc in struct.iter_unpack("<dHI", open(pre + ".bin", "rb").read()):
    tg = meta["targets"][i]
    if (tg["chip"], tg["chan"]) not in want or tg["erisc"] != "brisc": continue
    t += meta["t_start_epoch"]
    j = bisect.bisect_right(starts, t) - 1
    if j < 0 or t > win[j][3] or "unicast_write" not in win[j][0]: continue
    k = ci[(tg["chip"], tg["chan"])]; ts = [s[0] for s in snaps[k]]
    q = bisect.bisect_right(ts, t) - 1
    for qq in (q, q + 1, q - 1):
        if 0 <= qq < len(snaps[k]):
            h = next(((si, sl["kcb"]) for si, sl in enumerate(snaps[k][qq][1]) if sl["text_size"][0] and sl["kcb"] + sl["text_off"][0] <= pc < sl["kcb"] + sl["text_off"][0] + sl["text_size"][0]), None)
            if h: it[(j, tg["chip"])][h] += 1; break
tab = collections.Counter()
for (j, chip), c in sorted(it.items()):
    slot, kcb = c.most_common(1)[0][0]
    r = float(rows[j]["bandwidth_GB_s"]) / statistics.median(offm[rows[j]["test_name"]])
    slow = r < 0.97
    tab[(chip, kcb, slow)] += 1; tab[(chip, "slot", slot, slow)] += 1
    if chip == min(w[0] for w in want):
        print(f"{j:3d} {win[j][0][-48:]:48s} it{win[j][1]} slot {slot} kcb {kcb:#x} ratio {r:.3f} {'SLOW' if slow else ''}")
for chip in sorted({w[0] for w in want}):
    ks = sorted({key[1] for key in tab if key[0] == chip and len(key) == 3})
    print(f"chip {chip}: " + "; ".join(f"{k:#x} slow {tab[(chip,k,True)]}/normal {tab[(chip,k,False)]}" for k in ks))
for chip in sorted({w[0] for w in want}):
    print(f"chip {chip} launch slot: " + "; ".join(f"{s} slow {tab[(chip,'slot',s,True)]}/normal {tab[(chip,'slot',s,False)]}" for s in range(8)))
