#!/usr/bin/env python3
"""Pass/step split of the e2e SP-prefill (b) and TP-decode (c) Tracy ops CSVs.
(b): per-die kernel/comm/span time for the last prefill pass. (c): per-step decode timing.
"""
import re
import sys

import numpy as np
import pandas as pd

KERNEL_COL = "DEVICE KERNEL DURATION [ns]"
FW_START_COL = "DEVICE FW START CYCLE"
FW_END_COL = "DEVICE FW END CYCLE"
BH_CLOCK_GHZ = 1.35

DIE_OF_DEVICE = {3: 0, 2: 1, 1: 2, 0: 3}


def find_embedding_rows(df):
    mask = df["OP CODE"].str.contains("Embedding", case=False, na=False) & ~df["OP CODE"].str.contains(
        "Rotary", case=False, na=False
    )
    return df.index[mask].tolist()


def split_passes(device_df):
    """Split a single device's device-op rows into passes at each Embeddings row
    (a pass starts at an Embeddings row and runs to just before the next one, or to
    the end of the frame for the last pass)."""
    emb_idx = find_embedding_rows(device_df)
    if not emb_idx:
        return [device_df], emb_idx
    passes = []
    for k, e in enumerate(emb_idx):
        end = emb_idx[k + 1] - 1 if k + 1 < len(emb_idx) else device_df.index[-1]
        passes.append(device_df.loc[e:end])
    # rows before the first Embeddings row (if any) are dropped -- not part of any pass
    return passes, emb_idx


def is_comm(op_code):
    return bool(re.search(r"(Send|Recv|Socket)", op_code))


def span_ms(rows):
    if rows.empty:
        return 0.0
    return (rows[FW_END_COL].iloc[-1] - rows[FW_START_COL].iloc[0]) / BH_CLOCK_GHZ / 1e6


def kernel_ms(rows):
    return rows[KERNEL_COL].sum() / 1e6


def analyze_b(path):
    print(f"\n{'='*100}\n(b) CSV: {path}\n{'='*100}")
    df = pd.read_csv(path, low_memory=False)
    device_df_all = df.loc[df[KERNEL_COL].notna()].copy()
    dev_ids = sorted(int(d) for d in device_df_all["DEVICE ID"].dropna().unique())
    print(f"Devices found: {dev_ids}")
    print(f"Die mapping used: {DIE_OF_DEVICE}")

    rows_out = []
    tail_info = {}
    for dev in dev_ids:
        d = device_df_all[device_df_all["DEVICE ID"] == dev].reset_index(drop=True)
        passes, emb_idx = split_passes(d)
        n_passes = len(passes)
        die = DIE_OF_DEVICE.get(dev, dev)
        print(f"\n-- DEVICE ID {dev} (die {die}): {len(passes)} pass(es) found (embedding rows: {len(emb_idx)}) --")
        for i, p in enumerate(passes):
            print(f"   pass {i}: {len(p)} ops, kernel_sum={kernel_ms(p):.3f} ms, span={span_ms(p):.3f} ms")
        last = passes[-1]
        n_ops = len(last)
        k_ms = kernel_ms(last)
        s_ms = span_ms(last)
        comm_mask = last["OP CODE"].apply(is_comm)
        comm_ms = kernel_ms(last[comm_mask])
        rows_out.append(
            dict(die=die, dev=dev, ops=n_ops, kernel_ms=k_ms, comm_ms=comm_ms, span_ms=s_ms, n_passes=n_passes)
        )
        if die == 3:
            tail_info["last_pass"] = last
            tail_info["dev"] = dev

    rows_out.sort(key=lambda r: r["die"])
    print(f"\n{'-'*100}\nA.1 Table: die, ops, kernel ms, comm ms, span ms (LAST pass per device)\n{'-'*100}")
    print(f"{'die':>4} {'ops':>6} {'kernel ms':>12} {'comm ms':>10} {'span ms':>10}")
    for r in rows_out:
        print(f"{r['die']:>4} {r['ops']:>6} {r['kernel_ms']:>12.3f} {r['comm_ms']:>10.3f} {r['span_ms']:>10.3f}")

    # die 3 tail ops: after the last layer -- final norm, LM head matmul, argmax.
    # Heuristic: find last SDPA/ChunkGdn (mixer) marker row in the last pass, tail = everything after it.
    if "last_pass" in tail_info:
        last = tail_info["last_pass"].reset_index(drop=True)
        marker_mask = last["OP CODE"].str.contains("SDPAOperation|ChunkGdn", na=False)
        marker_idx = last.index[marker_mask].tolist()
        print(
            f"\n{'-'*100}\nDie 3 (DEVICE ID {tail_info['dev']}) tail ops (after last GDN/attention marker row)\n{'-'*100}"
        )
        if marker_idx:
            tail_start = marker_idx[-1] + 1
            tail = last.loc[tail_start:]
            print(f"Last marker row index (within last pass): {marker_idx[-1]}; tail rows: {len(tail)}")
            print(f"Tail total kernel ms: {kernel_ms(tail):.3f}")
            print(f"{'OP CODE':<40}{'kernel us':>12}")
            for _, r in tail.iterrows():
                print(f"{str(r['OP CODE'])[:40]:<40}{r[KERNEL_COL]/1e3:>12.2f}")
        else:
            print("(no SDPA/ChunkGdn marker row found in die 3's last pass -- cannot bound the tail)")


STEP_RE = re.compile(r"^decode step (\d+)$")


def analyze_c(path):
    print(f"\n{'='*100}\n(c) CSV: {path}\n{'='*100}")
    df = pd.read_csv(path, low_memory=False).reset_index(drop=True)
    has_signpost_col = "OP TYPE" in df.columns
    is_sp = (df["OP TYPE"] == "signpost") if has_signpost_col else pd.Series(False, index=df.index)
    has_signposts = bool(is_sp.any())
    print(f"Signpost rows present: {has_signposts} (count={int(is_sp.sum())})")

    device_df_all = df.loc[df[KERNEL_COL].notna()].copy()
    dev_ids = sorted(int(d) for d in device_df_all["DEVICE ID"].dropna().unique())
    print(f"Devices found: {dev_ids}")

    # kernel sums for all devices (whole capture, no split) -- quick sanity numbers
    print(f"\n{'-'*100}\nAll-device kernel sums (whole capture, unsplit)\n{'-'*100}")
    for dev in dev_ids:
        d = device_df_all[device_df_all["DEVICE ID"] == dev]
        print(f"  DEVICE ID {dev}: {len(d)} ops, kernel_sum={kernel_ms(d):.3f} ms")

    dev0 = device_df_all[device_df_all["DEVICE ID"] == dev_ids[0]].reset_index(drop=True)
    passes, emb_idx = split_passes(dev0)
    print(
        f"\nDevice {dev_ids[0]}: {len(passes)} pass(es) found via Embeddings split (embedding rows: {len(emb_idx)}; expected 9 = 1 warmup + 8 timed)"
    )

    print(f"\n{'-'*100}\nB.1 Per-pass table (device {dev_ids[0]})\n{'-'*100}")
    print(f"{'pass':>5} {'ops':>6} {'kernel ms':>12} {'span ms':>10}")
    per_pass_stats = []
    for i, p in enumerate(passes):
        k = kernel_ms(p)
        s = span_ms(p)
        per_pass_stats.append(dict(idx=i, ops=len(p), kernel_ms=k, span_ms=s))
        print(f"{i:>5} {len(p):>6} {k:>12.3f} {s:>10.3f}")

    if len(passes) >= 2:
        timed = per_pass_stats[1:]  # drop pass 0 = warmup
        mean_k = np.mean([t["kernel_ms"] for t in timed])
        mean_s = np.mean([t["span_ms"] for t in timed])
        print(
            f"\nMean over {len(timed)} timed passes (excluding pass 0 warmup): kernel_ms={mean_k:.3f}, span_ms={mean_s:.3f}"
        )
    else:
        print("\n(fewer than 2 passes found -- cannot compute a warmup-excluded mean)")

    # signpost-based inject start/end window (kernel ms), on device 0's full frame
    if has_signposts:
        d0_full = df[(df["DEVICE ID"] == dev_ids[0]) | (df["OP TYPE"] == "signpost")].reset_index(drop=True)
        occ = (d0_full["OP TYPE"] == "signpost").cumsum()
        d0_full["_occ"] = occ
        label_by_occ = {}
        for i in d0_full.index[d0_full["OP TYPE"] == "signpost"]:
            label_by_occ[int(occ.loc[i])] = d0_full.loc[i, "OP CODE"]
        inj_start_occ = [o for o, l in label_by_occ.items() if l == "inject start"]
        inj_end_occ = [o for o, l in label_by_occ.items() if l == "inject end"]
        print(f"\n{'-'*100}\nSignpost labels found: {sorted(set(label_by_occ.values()))}\n{'-'*100}")
        if inj_start_occ and inj_end_occ:
            lo, hi = inj_start_occ[0], inj_end_occ[0]
            seg = d0_full[(d0_full["_occ"] >= lo) & (d0_full["_occ"] <= hi) & d0_full[KERNEL_COL].notna()]
            print(f"inject start -> inject end: {len(seg)} device ops, kernel_ms={kernel_ms(seg):.3f}")
        else:
            print("(no 'inject start'/'inject end' signpost pair found)")
    else:
        print("\n(no signpost rows in this CSV)")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python analyze_e2e_4k.py <b_ops_perf_results.csv> <c_ops_perf_results.csv>")
        sys.exit(1)
    analyze_b(sys.argv[1])
    analyze_c(sys.argv[2])
