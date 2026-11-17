#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""End-to-end check of TT_METAL_PROFILE_ETH_COUNTERS with the ethernet unit test that sends known buffers over every
connected link, on any multi-chip system. Slow dispatch, run once with the option off and once on, then compare.

  python eth_validate.py --arch blackhole --binary build_Release/test/tt_metal/unit_tests_eth

The test is MeshDeviceFixture.ActiveEthKernelsSendInterleavedBufferAllConnectedChips: for every connected pair, link and
erisc, four chip-to-chip transfers of about one unreserved L1 (a page-sized and a whole-L1 transfer size, L1 to DRAM
and DRAM to L1). Expect: the test is not skipped, at least four programs with sender bytes, receiver bytes of the same
order, no resends or drops on a clean link, and run rows (eth_run.csv) whose per-device totals are at least the sum of
that device's program rows.

The two-chip fixtures of the other ethernet tests skip under slow dispatch (TwoDeviceBlackholeFixture and
N300MeshDeviceFixture skip when it is set, and their test bodies skip without it), so they cannot be used for the
per-program path. Per-run totals under fast dispatch are checked with a CCL run, see the docs.

Run it on an otherwise idle system, from the tt-metal root. The test opens the devices; this script does not.
"""

import argparse
import os
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _common as c  # noqa: E402,F401
from tt_llk_perf import eth_counters  # noqa: E402

TEST = "MeshDeviceFixture.ActiveEthKernelsSendInterleavedBufferAllConnectedChips"


def run_test(binary, test, env_extra, profiler_dir):
    env = dict(os.environ)
    env["TT_METAL_SLOW_DISPATCH_MODE"] = "1"
    env["TT_METAL_PROFILER_DIR"] = str(profiler_dir)
    env.pop("TT_METAL_PROFILE_ETH_COUNTERS", None)
    env.update(env_extra)
    return subprocess.run(
        [binary, f"--gtest_filter={test}"],
        env=env,
        capture_output=True,
        text=True,
        timeout=1800,
    )


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--arch", default="blackhole", choices=("blackhole", "wormhole"))
    p.add_argument("--binary", required=True)
    args = p.parse_args()
    arch, test = args.arch, TEST

    work = Path(tempfile.mkdtemp(prefix="eth_validate_"))
    ok = True

    def check(cond, text):
        nonlocal ok
        print(("PASS " if cond else "FAIL ") + text)
        ok = ok and cond

    def skipped(proc):
        return "SKIPPED" in proc.stdout + proc.stderr or "[  PASSED  ] 0 tests" in (
            proc.stdout
        )

    off = run_test(args.binary, test, {}, work / "off")
    check(off.returncode == 0 and not skipped(off), "option off: test runs and passes")
    check(
        not any((work / "off" / ".logs").glob("eth_*.csv")),
        "option off: no eth_*.csv written",
    )
    on = run_test(
        args.binary, test, {"TT_METAL_PROFILE_ETH_COUNTERS": "1"}, work / "on"
    )
    check(on.returncode == 0 and not skipped(on), "option on: test runs and passes")
    logs = work / "on" / ".logs"
    traffic, run, idle = (
        logs / eth_counters.TRAFFIC_LOG,
        logs / eth_counters.RUN_LOG,
        logs / eth_counters.IDLE_LOG,
    )
    check(traffic.is_file(), "option on: eth_traffic.csv written")
    check(run.is_file(), "option on: eth_run.csv written")
    if not traffic.is_file():
        print(on.stdout[-2000:], on.stderr[-2000:])
        raise SystemExit(1)

    rows = eth_counters.read_traffic(traffic)
    rate = eth_counters.read_idle(idle) if idle.is_file() else None
    programs = eth_counters.per_program(rows, arch, rate)
    print(eth_counters.format_table(programs))
    for prog in programs:
        for core, t in sorted(prog.cores.items()):
            print(
                f"program {prog.program} device {prog.device} core {core} ch {t.channel}: tx {t.tx_bytes} B "
                f"rx {t.rx_bytes} B resends {t.get('TXQ_RESEND_CNT')} drops {t.get('RXQ_PKT_DROP_CNT')} "
                f"pkt_end tx {t.get('TXQ_PKT_END_CNT')} rx {t.get('RXQ_PKT_END_CNT')} words {t.get('TXQ_WORD_CNT')}"
            )
    senders = [p for p in programs if p.tx_bytes > 0]
    check(len(senders) >= 4, f"{len(senders)} programs with sender bytes (>= 4)")
    check(
        sum(p.rx_bytes for p in programs) > 0, "receiver bytes seen in program windows"
    )
    check(
        all(p.resends == 0 and p.drops == 0 for p in programs),
        "no resends and no drops",
    )
    check(not any(p.reset for p in programs), "no counter reset flagged")

    if run.is_file():
        runs = eth_counters.per_run(eth_counters.read_run(run), arch)
        for r in runs:
            print(
                f"run {r.seq} device {r.device}: tx {r.tx_bytes} B rx {r.rx_bytes} B in {r.elapsed_ns * 1e-9:.1f} s"
            )
            program_tx = sum(p.tx_bytes for p in programs if p.device == r.device)
            check(
                r.tx_bytes >= program_tx,
                f"device {r.device}: run tx {r.tx_bytes} >= program tx {program_tx}",
            )
    print(f"work directory kept: {work}")
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
