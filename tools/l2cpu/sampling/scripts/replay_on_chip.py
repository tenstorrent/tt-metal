#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-FileCopyrightText: © 2026 Kansei Motoe (kinginu)
# SPDX-License-Identifier: Apache-2.0
"""On-chip sampling replay (no model): recorded logits rows -> region -> firmware request -> token, compared with the
host build of the sampling library on the same row (expected tokens generated at run time, nothing recorded).

One chip epoch (fresh reset; the sampling image is started here):
    source tools/l2cpu/scripts/l2cpu_env.sh
    tools/l2cpu/scripts/l2cpu_run.sh "replay" $PY tools/l2cpu/sampling/scripts/replay_on_chip.py ROWS.npy \
        --rows 1000 --b32 100 --restart-b1 500 --restart-b32 50
    # batch 32 split over 1, 2 and 4 L2CPU tiles (all four started by one release; tile 0 also runs the above)
    tools/l2cpu/scripts/l2cpu_run.sh "replay split" $PY tools/l2cpu/sampling/scripts/replay_on_chip.py ROWS.npy \
        --tiles 4 --rows 0 --b32 0 --split 50 --b32-modes mix,bench

ROWS.npy: uint16 bfloat16 bits [N, 151936] (Qwen3 vocabulary; record one with the qwen3 example's decode harness,
or make a synthetic one with tools/l2cpu/sampling/lib/replay.py --make-synthetic). Prints the per-request x280 time
split from the firmware's timing records and exits 0 only if every token is identical.
"""
import argparse
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
L2CPU_DIR = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(L2CPU_DIR, "host"))

import numpy as np  # noqa: E402

from l2cpu import layout as L  # noqa: E402
from l2cpu.sampling import boot, boot_tiles  # noqa: E402
from l2cpu.sampling.fw import DEFAULT_IMAGE  # noqa: E402
from l2cpu.sampling.replay import (
    format_split,
    host_library,
    replay_b1,
    replay_b32,
    replay_split,  # noqa: E402
    stream_after_hang,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("rows_npy")
    ap.add_argument("--rows", type=int, default=1000, help="batch-1 rows (x 4 settings)")
    ap.add_argument("--b32", type=int, default=100, help="batch-32 requests")
    ap.add_argument("--b32-modes", default="mix", help="comma list: mix, bench")
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--image", default=DEFAULT_IMAGE)
    ap.add_argument("--restart-b1", type=int, default=None, help="L1 WARM restart before this batch-1 row")
    ap.add_argument("--restart-b32", type=int, default=None, help="L2 (RNMI) WARM restart before this batch-32 request")
    ap.add_argument(
        "--stream-hang",
        action="store_true",
        help="first: streamed batch 32, hart 0 hung (WFIPARK), recovery by WARM restart (L1 attempt, then "
        "RNMI), the next streamed requests must be served within 50 ms",
    )
    ap.add_argument(
        "--tiles",
        type=int,
        default=1,
        choices=[1, 2, 4],
        help="L2CPU tiles to start (one release); the batch-1 / batch-32 replays run on tile 0",
    )
    ap.add_argument("--split", type=int, default=0, help="batch-32 requests split over 1, 2, ... --tiles tiles")
    ap.add_argument("--json", default=None)
    a = ap.parse_args()
    import ttnn

    rows = np.load(a.rows_npy, mmap_mode="r")
    if rows.dtype != np.uint16 or rows.shape[1] < 151936:
        sys.exit(f"{a.rows_npy}: expected uint16 bf16 rows [N, >= 151936], got {rows.dtype} {rows.shape}")
    n_rows = min(a.rows, rows.shape[0])
    lib = host_library()
    out = dict(rows_file=os.path.abspath(a.rows_npy), image=os.path.abspath(a.image), steps=[])
    dev = ttnn.open_device(device_id=0)
    try:
        t0 = time.time()
        if a.tiles == 1:
            fw, region, info = boot(dev, a.image, log=None)
            fws = [fw]
        else:
            fws, regions, infos = boot_tiles(dev, range(a.tiles), a.image, log=None)
            fw = fws[0]
            print(
                f"tiles {list(range(a.tiles))} READY, regions {[hex(f.base) for f in fws]}, L2CPU_RESET "
                f"0x{infos[0]['reset']:08x}",
                flush=True,
            )
        out["boot_s"] = time.time() - t0
        print(
            f"sampling image READY at region 0x{fw.base:x} in {out['boot_s']:.2f} s (app flags "
            f"0x{fw.build_flags():x})",
            flush=True,
        )
        en = [[fw.hw.pa_read32(0x0C00_2000 + 0x80 * ctx + 4 * k) for k in range(4)] for ctx in range(8)]
        out["plic_enables"] = [[f"0x{w:08x}" for w in c] for c in en]
        plic_ok = en[0] == [1 << 6, 0, 0, 0] and all(w == 0 for c in en[1:] for w in c)
        print(
            f"PLIC enable words after boot (8 contexts x 4): {'only source 6 in context 0' if plic_ok else en}",
            flush=True,
        )
        ok = True
        if a.stream_hang:
            sh = stream_after_hang(fw, lib, rows, rows.shape[0])
            out["stream_hang"] = sh
            print(
                f"STREAM AFTER HANG: {'PASS' if sh['ok'] else 'FAIL'} hung request served {sh['hung_request_served']}, "
                f"restart {sh['restart']}, steps {[(s['phase'], s['served'], s['identical']) for s in sh['steps']]}",
                flush=True,
            )
            if not sh["ok"]:
                print("records", sh.get("records"), "\nlog tail:", sh.get("log_tail"), flush=True)
            ok &= sh["ok"]
        b1 = replay_b1(fw, lib, rows, n_rows, seed=a.seed, restart_at=a.restart_b1)
        out["b1"] = b1
        print(
            f"BATCH 1 REPLAY: {b1['identical']}/{b1['total']} identical to the host library (per setting "
            f"{b1['mismatches']}), {b1['wall_s']:.1f} s, restart {b1['restart']}",
            flush=True,
        )
        for s in b1["split"]:
            print(format_split(s), flush=True)
        ok &= b1["ok"]
        for mode in [m for m in a.b32_modes.split(",") if m] if a.b32 else []:
            b32 = replay_b32(
                fw,
                lib,
                rows,
                a.b32,
                rows.shape[0],
                mode=mode,
                restart_at=a.restart_b32 if mode == a.b32_modes.split(",")[0] else None,
            )
            out[f"b32_{mode}"] = b32
            print(
                f"BATCH 32 REPLAY ({mode}): {b32['identical']}/{b32['total']} identical ({a.b32} requests x 32 users, "
                f"uncached zone, 4 harts), {b32['wall_s']:.1f} s, restart {b32['restart']}",
                flush=True,
            )
            for s in b32["split"]:
                print(format_split(s), "| workers", [round(x, 1) for x in s["worker_us"]], flush=True)
            ok &= b32["ok"]
        for mode in [m for m in a.b32_modes.split(",") if m] if a.split else []:
            nt = 1
            while nt <= a.tiles:
                sp = replay_split(fws, lib, rows, a.split, rows.shape[0], nt, mode=mode)
                out[f"split_{mode}_{nt}"] = sp
                print(
                    f"SPLIT REPLAY ({mode}, {nt} tile(s) x {32 // nt} users): {sp['identical']}/{sp['total']} identical "
                    f"(global user index), x280 step median {sp['step_us']:.1f} us (p90 {sp['step_p90_us']:.1f}), "
                    f"per tile {[round(x, 1) for x in sp['tile_us']]}",
                    flush=True,
                )
                ok &= sp["ok"]
                nt *= 2
        out["tile_errors"] = [f.error() for f in fws[1:]]
        ok &= not any(out["tile_errors"])
        out["error"] = fw.error()
        out["counters"] = [fw.counters(h) for h in range(4)]
        out["restart_count"] = fw.r32(L.L2CPU_OFF_RESTART_COUNT)
        print("error word:", out["error"], "| restart_count", out["restart_count"], flush=True)
        out["ok"] = bool(ok)
        print("REPLAY", "PASS" if ok else "FAIL", flush=True)
    finally:
        ttnn.close_device(dev)
    if a.json:
        with open(a.json, "w") as f:
            json.dump(out, f, indent=1, default=str)
    return 0 if out.get("ok") else 1


if __name__ == "__main__":
    sys.exit(main())
