#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""End-to-end check of TT_METAL_PROFILE_DRAM_COUNTERS with a program that moves a known number of bytes.

The workload is the eltwise_binary programming example: 64 tiles of 2 KiB per input buffer, two input buffers read
from DRAM (256 KiB) and one output buffer written (128 KiB), so a 64 B word gives 4096 read words and 2048 write
words. It runs in slow dispatch, once with the option off and once with it on, then compares.

  python dram_validate.py --example build/programming_examples/metal_example_eltwise_binary \
      --expect-read-bytes 262144 --expect-write-bytes 131072

Run it on an otherwise idle card, from the tt-metal root. The device is opened only by the example.
"""

import argparse
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, __file__.rsplit("/", 1)[0])
import _common as c

# tt_llk_perf is a sibling of this directory.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))
from tt_llk_perf import dram_counters  # noqa: E402


def run_example(example, env_extra, profiler_dir):
    env = dict(os.environ)
    env["TT_METAL_SLOW_DISPATCH_MODE"] = "1"
    env["TT_METAL_PROFILER_DIR"] = str(profiler_dir)
    env.pop("TT_METAL_PROFILE_DRAM_COUNTERS", None)
    env.update(env_extra)
    t0 = time.monotonic()
    proc = subprocess.run(
        [example], env=env, capture_output=True, text=True, timeout=900
    )
    return proc, time.monotonic() - t0


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--example", required=True)
    p.add_argument("--arch", default="blackhole")
    p.add_argument("--expect-read-bytes", type=int, default=262144)
    p.add_argument("--expect-write-bytes", type=int, default=131072)
    p.add_argument(
        "--tolerance",
        type=float,
        default=0.02,
        help="relative error accepted on the main program",
    )
    p.add_argument("--device", default=c.default_device_path())
    args = p.parse_args()
    if c.card_busy(args.device):
        c.fail(f"{args.device} is open by another process; not running")

    work = Path(tempfile.mkdtemp(prefix="dram_validate_"))
    off_dir, on_dir = work / "off", work / "on"
    off, off_s = run_example(args.example, {}, off_dir)
    ok = True

    def check(cond, text):
        nonlocal ok
        print(("PASS " if cond else "FAIL ") + text)
        ok = ok and cond

    check(
        off.returncode == 0 and "Test Passed" in off.stdout + off.stderr,
        f"option off: example passes ({off_s:.1f} s)",
    )
    check(
        not (off_dir / ".logs" / dram_counters.TRAFFIC_LOG).exists(),
        "option off: no dram_traffic.csv written",
    )

    env_on = {"TT_METAL_PROFILE_DRAM_COUNTERS": "1"}
    on, on_s = run_example(args.example, env_on, on_dir)
    check(
        on.returncode == 0 and "Test Passed" in on.stdout + on.stderr,
        f"option on: example passes ({on_s:.1f} s)",
    )
    logs = on_dir / ".logs"
    traffic = logs / dram_counters.TRAFFIC_LOG
    check(traffic.is_file(), "option on: dram_traffic.csv written")
    if not traffic.is_file():
        print(on.stdout[-2000:], on.stderr[-2000:])
        raise SystemExit(1)

    rows = dram_counters.read_traffic(traffic)
    idle_path = logs / dram_counters.IDLE_LOG
    idle = dram_counters.read_idle(idle_path) if idle_path.is_file() else None
    programs = dram_counters.per_program(rows, args.arch, idle)
    raw_programs = dram_counters.per_program(rows, args.arch, None)
    print(dram_counters.format_table(raw_programs))
    target = max(programs, key=lambda q: q.read_bytes + q.write_bytes)
    raw_target = next(
        q for q in raw_programs if q.seq == target.seq and q.device == target.device
    )
    for label, prog in (
        ("self+idle corrected", target),
        ("self corrected only", raw_target),
    ):
        rd = prog.read_bytes / args.expect_read_bytes - 1
        wr = prog.write_bytes / args.expect_write_bytes - 1
        print(
            f"{label}: program {prog.program} read {prog.read_bytes} B ({rd:+.2%} vs expected), "
            f"write {prog.write_bytes} B ({wr:+.2%}), flags {prog.flags}"
        )
    check(
        abs(target.write_bytes / args.expect_write_bytes - 1) <= args.tolerance,
        "write bytes within tolerance (self+idle corrected)",
    )
    check(
        abs(target.read_bytes / args.expect_read_bytes - 1) <= args.tolerance,
        "read bytes within tolerance (self+idle corrected)",
    )
    check(
        not target.overlapped and not target.unsettled,
        "main program not flagged overlapped or unsettled",
    )
    print(f"work directory kept: {work}")
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
