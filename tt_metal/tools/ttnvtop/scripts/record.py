#!/usr/bin/env python3
# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
#
# ttnvtop offline recorder. Polls the same /dev/shm files the viewer reads
# (tt_device_<id>_util + tt_program_registry) and writes one CSV row per
# (frame, chip, active program). Default 4 Hz, runs until Ctrl-C.
#
# Usage:
#   python tt_metal/tools/ttnvtop/scripts/record.py --out run.csv
#   python tt_metal/tools/ttnvtop/scripts/record.py --hz 10 --out run.csv
#
# Then post-process:
#   import pandas as pd
#   df = pd.read_csv("run.csv")
#   df.groupby("name")["f_pct"].max().sort_values()  # peak F% per program
#   df.groupby("name")["t_us"].apply(lambda s: s.max()-s.min())  # duration
#
# Requires the collector to be running (writes the per-chip SHM files) and
# the workload to have been launched with TTNVTOP_REGISTER_PROGRAMS=1 (writes
# the registry).

import argparse
import csv
import glob
import os
import signal
import struct
import sys
import time

# Per-chip schema, mirrors common/shm_schema.hpp.
HEADER_FMT = "<4sHHQIIQQIIII4I"
HEADER_SIZE = struct.calcsize(HEADER_FMT)
PER_CORE_FMT = "<6B10H2x3I"
PER_CORE_SIZE = struct.calcsize(PER_CORE_FMT)

# Program registry, mirrors common/program_registry.hpp.
REG_HEADER_SIZE = 48  # magic[4]+version+entry_size+capacity+writer_pid+epoch_us+atomic<u32>+reserved[4]
REG_ENTRY_FMT = "<II Q 96s"
REG_ENTRY_SIZE = struct.calcsize(REG_ENTRY_FMT)
REG_CAPACITY = 16384


def _read_chip(path):
    """Return (chip_id, aiclk_mhz, [(noc_x, noc_y, kernel_id, f_p1000, s_p1000, d_p1000), ...])"""
    with open(path, "rb") as f:
        data = f.read()
    hdr = struct.unpack(HEADER_FMT, data[:HEADER_SIZE])
    asic_id = hdr[3]
    n_cores = hdr[8]
    aiclk_mhz = hdr[11]
    rows = []
    for i in range(n_cores):
        r = struct.unpack(
            PER_CORE_FMT,
            data[HEADER_SIZE + i * PER_CORE_SIZE : HEADER_SIZE + (i + 1) * PER_CORE_SIZE],
        )
        # r layout: 6 u8 (noc_x, noc_y, lx, ly, is_remote, dispatched), 10 u16
        # (sfpu, dispatch_busy, compute_busy, unpack, pack, stall, noc0_in,
        # noc0_out, noc1_in, noc1_out), 3 u32 (samples_seen, last_kernel_id, reserved_1).
        rows.append((r[0], r[1], r[17], r[8], r[6], r[7]))
    return asic_id, aiclk_mhz, rows


def _read_registry(path):
    """Return {runtime_id: name} from /dev/shm/tt_program_registry. Empty if absent."""
    try:
        with open(path, "rb") as f:
            data = f.read()
    except FileNotFoundError:
        return {}
    if len(data) < REG_HEADER_SIZE + REG_ENTRY_SIZE:
        return {}
    if data[0:4] != b"TPRG":
        return {}
    out = {}
    base = REG_HEADER_SIZE
    for i in range(REG_CAPACITY):
        e = data[base + i * REG_ENTRY_SIZE : base + (i + 1) * REG_ENTRY_SIZE]
        if len(e) < REG_ENTRY_SIZE:
            break
        rid, _pid, _epoch, name = struct.unpack(REG_ENTRY_FMT, e)
        if rid == 0 and name[0:1] == b"\0":
            continue
        n = name.split(b"\0", 1)[0].decode("ascii", errors="replace")
        out[rid] = n
    return out


def _resolve(prog_id, names):
    """Reproduce the viewer's lookup heuristic: try raw, then encoded forms."""
    if prog_id in names:
        return names[prog_id]
    for dev in range(8):
        if (prog_id << 10 | dev) in names:
            return names[(prog_id << 10) | dev]
    return ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, help="output CSV path")
    ap.add_argument("--hz", type=float, default=4.0, help="poll rate (Hz)")
    ap.add_argument("--shm-glob", default="/dev/shm/tt_device_*_util")
    ap.add_argument("--registry", default="/dev/shm/tt_program_registry")
    args = ap.parse_args()

    period = 1.0 / args.hz
    stop = {"flag": False}

    def _sigint(*_):
        stop["flag"] = True

    signal.signal(signal.SIGINT, _sigint)
    signal.signal(signal.SIGTERM, _sigint)

    shm_paths = sorted(glob.glob(args.shm_glob))
    if not shm_paths:
        print(f"no SHM files match {args.shm_glob} — is ttnvtop-collector running?", file=sys.stderr)
        sys.exit(1)
    out_dir = os.path.dirname(os.path.abspath(args.out))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    print(f"recording {len(shm_paths)} chip(s) to {args.out} at {args.hz} Hz", file=sys.stderr)

    with open(args.out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["t_us", "chip_idx", "asic_id", "prog_id", "name", "cores", "f_pct", "s_pct", "d_pct"])

        names_cache = {}
        names_mtime = 0
        next_t = time.monotonic()
        t0 = time.monotonic_ns() // 1000

        while not stop["flag"]:
            now_ns = time.monotonic_ns()
            t_us = now_ns // 1000 - t0

            # Refresh names registry only if it changed.
            try:
                m = os.path.getmtime(args.registry)
                if m != names_mtime:
                    names_cache = _read_registry(args.registry)
                    names_mtime = m
            except FileNotFoundError:
                names_cache = {}
                names_mtime = 0

            for chip_idx, path in enumerate(shm_paths):
                try:
                    asic_id, aiclk, rows = _read_chip(path)
                except FileNotFoundError:
                    continue

                # Aggregate by kernel_id, only counting cores with D > 0.
                buckets = {}  # prog_id -> [cores, sum_f, sum_s, sum_d]
                for noc_x, noc_y, kid, fp, sp, dp in rows:
                    if dp == 0 or kid == 0:
                        continue
                    prog_id = (kid >> 10) & 0x1FFFFF
                    b = buckets.setdefault(prog_id, [0, 0, 0, 0])
                    b[0] += 1
                    b[1] += fp
                    b[2] += sp
                    b[3] += dp

                for prog_id, (cores, sf, ss, sd) in buckets.items():
                    name = _resolve(prog_id, names_cache)
                    w.writerow(
                        [
                            t_us,
                            chip_idx,
                            asic_id,
                            prog_id,
                            name,
                            cores,
                            f"{sf / cores / 10:.1f}",
                            f"{ss / cores / 10:.1f}",
                            f"{sd / cores / 10:.1f}",
                        ]
                    )

            f.flush()
            next_t += period
            sleep = next_t - time.monotonic()
            if sleep > 0:
                time.sleep(sleep)
            else:
                next_t = time.monotonic()  # we fell behind; drop forward

    print(f"\nrecording stopped: {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
