#!/usr/bin/env python3
"""Compact job-log lines from tt-llk perf CSVs (TILE_LOOP rows). usage: ut_dump.py T|C <tag> <side> <run dir>
T: T|tag|side|module|key|mean(L1_TO_L1)|mean(PACK_ISOLATE)   C: C|tag|side|module|key|<counters per loop, CNAMES order>"""
import glob, sys
import pandas as pd

CN = ["FPU.FPU_COUNTER.cycles", "INSTRN_THREAD.THREAD_INSTRUCTIONS_0", "INSTRN_THREAD.THREAD_INSTRUCTIONS_1",
      "INSTRN_THREAD.THREAD_INSTRUCTIONS_2", "INSTRN_THREAD.THREAD_STALLS_0", "INSTRN_THREAD.THREAD_STALLS_1",
      "INSTRN_THREAD.THREAD_STALLS_2", "INSTRN_THREAD.PACK_INSTRN_AVAILABLE_2", "INSTRN_THREAD.CFG_INSTRN_AVAILABLE_2",
      "INSTRN_THREAD.SYNC_INSTRN_AVAILABLE_2", "INSTRN_THREAD.THCON_INSTRN_AVAILABLE_2", "INSTRN_THREAD.MOVE_INSTRN_AVAILABLE_2",
      "INSTRN_THREAD.UNPACK_INSTRN_AVAILABLE_0", "INSTRN_THREAD.CFG_INSTRN_AVAILABLE_0", "INSTRN_THREAD.THCON_INSTRN_AVAILABLE_0",
      "INSTRN_THREAD.MATH_INSTRN_AVAILABLE_1", "INSTRN_THREAD.CFG_INSTRN_AVAILABLE_1",
      "INSTRN_THREAD.WAITING_FOR_NONZERO_SEM_2", "INSTRN_THREAD.WAITING_FOR_NONFULL_SEM_2", "INSTRN_THREAD.WAITING_FOR_PACK_IDLE_2",
      "INSTRN_THREAD.WAITING_FOR_THCON_IDLE_2", "INSTRN_THREAD.WAITING_FOR_CFG_IDLE_2", "INSTRN_THREAD.WAITING_FOR_MATH_IDLE_2",
      "INSTRN_THREAD.WAITING_FOR_NONFULL_SEM_1", "INSTRN_THREAD.WAITING_FOR_NONZERO_SEM_1", "INSTRN_THREAD.WAITING_FOR_MATH_IDLE_1",
      "INSTRN_THREAD.WAITING_FOR_SRCA_VALID", "INSTRN_THREAD.WAITING_FOR_SRCA_CLEAR", "INSTRN_THREAD.WAITING_FOR_NONFULL_SEM_0",
      "INSTRN_THREAD.WAITING_FOR_NONZERO_SEM_0", "INSTRN_THREAD.WAITING_FOR_THCON_IDLE_0", "INSTRN_THREAD.WAITING_FOR_UNPACK_IDLE_0",
      "INSTRN_THREAD.WAITING_FOR_CFG_IDLE_0", "INSTRN_THREAD.WAITING_FOR_PACK_IDLE_0", "INSTRN_THREAD.WAITING_FOR_THCON_IDLE_1",
      "TDMA_UNPACK.UNPACK0_BUSY_THREAD0", "TDMA_UNPACK.SRCA_WRITE_REQ", "TDMA_UNPACK.SRCA_WRITE_NOT_BLOCKED_OVR",
      "TDMA_UNPACK.MATH_INSTRN_AVAILABLE", "TDMA_UNPACK.MATH_INSTRN_STARTED", "TDMA_UNPACK.MATH_SRC_DATA_READY",
      "TDMA_PACK.PACKER0_DEST_READ_REQ", "TDMA_PACK.DEST_READ_GRANTED_0", "TDMA_PACK.PACKER_BUSY",
      "TDMA_PACK.MATH_NOT_STALLED_DEST_WR_PORT", "TDMA_PACK.MATH_NOT_SCOREBOARD_STALLED",
      "L1.L1_0_UNPACKER_0", "L1.L1_0_UNPACKER_0_GRANT", "L1.L1_0_TDMA_BUNDLE_0_RISC", "L1.L1_0_TDMA_BUNDLE_0_GRANT",
      "L1.L1_0_TDMA_BUNDLE_1_TRISC", "L1.L1_0_TDMA_BUNDLE_1_GRANT"]
SKIP = ("marker", "run_index")


def key_cols(d):
    return [c for c in d.columns if c not in SKIP and not c.startswith(("mean(", "TEXT_SIZE", "L1_TO_L1_", "PACK_ISOLATE_",
                                                                         "UNPACK_ISOLATE_", "MATH_ISOLATE_", "L1_CONGESTION_"))]


def main():
    kind, tag, side, rdir = sys.argv[1:5]
    pat = "/**/*.counters.csv" if kind == "C" else "/**/perf_*.csv"
    if kind == "C":
        print("CH|" + ",".join(CN))
    for f in sorted(glob.glob(rdir + pat, recursive=True)):
        if kind == "T" and (f.endswith(".counters.csv") or "post" in f):
            continue
        mod = f.split("/")[-1].split(".")[0]
        d = pd.read_csv(f, low_memory=False)
        if "marker" in d:
            d = d[d.marker == "TILE_LOOP"]
        kc = key_cols(d)
        for _, r in d.iterrows():
            k = ";".join(str(r[c]) for c in kc)
            if kind == "T":
                v = [r.get(f"mean({t})", float("nan")) for t in ("L1_TO_L1", "PACK_ISOLATE")]
                print(f"T|{tag}|{side}|{mod}|{k}|{v[0]:.3f}|{v[1]:.3f}")
            else:
                lf = float(r["loop_factor"])
                v = [r.get("L1_TO_L1_" + c, float("nan")) / lf for c in CN]
                print(f"C|{tag}|{side}|{mod}|{k}|" + ",".join(f"{x:.2f}" for x in v))
        if kind == "T" and len(d):
            print(f"KH|{mod}|" + ";".join(kc))


if __name__ == "__main__":
    main()
