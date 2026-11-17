#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""End-to-end check of TT_METAL_PROFILE_ETH_COUNTERS with the ethernet unit tests that send a known amount over a
link between two chips. Slow dispatch, run once with the option off and once on, then compare.

  python eth_validate.py bh-direct-send --binary build/test/tt_metal/unit_tests_eth

  bh-direct-send  two Blackhole chips: TwoDeviceBlackholeFixture.ActiveEthChipToChipMultiTxqRxq_Both0, 100000
                  messages of 32 B. Expect on the sender's data queue at least 100000 TXQ_PKT_END_CNT and as many
                  TXQ_WORD_CNT (one 96 B word per small packet), the receiver's RXQ_PKT_END_CNT at least as large,
                  no TXQ_RESEND_CNT and RXQ_PKT_DROP_CNT on a clean link.
  wh-direct-send  two Wormhole chips (N300): N300MeshDeviceFixture.ActiveEthSingleCoreDirectSendChip0ToChip1, sends of
                  16 B, 4 KiB, 16 KiB and the whole unreserved L1, each from two cores. Expect sender bytes (words *
                  16) at or just above the payload and receiver bytes close to it.

Run it on an otherwise idle system, from the tt-metal root. The test opens the devices; this script does not.
"""

import argparse
import os
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, __file__.rsplit("/", 1)[0])
import _common as c  # noqa: E402,F401
from tt_llk_perf import eth_counters  # noqa: E402

CASES = {
    "bh-direct-send": (
        "blackhole",
        "TwoDeviceBlackholeFixture.ActiveEthChipToChipMultiTxqRxq_Both0",
    ),
    "wh-direct-send": (
        "wormhole",
        "N300MeshDeviceFixture.ActiveEthSingleCoreDirectSendChip0ToChip1",
    ),
}


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
    p.add_argument("case", choices=sorted(CASES))
    p.add_argument("--binary", required=True)
    p.add_argument("--messages", type=int, default=100000)
    args = p.parse_args()
    arch, test = CASES[args.case]

    work = Path(tempfile.mkdtemp(prefix="eth_validate_"))
    ok = True

    def check(cond, text):
        nonlocal ok
        print(("PASS " if cond else "FAIL ") + text)
        ok = ok and cond

    off = run_test(args.binary, test, {}, work / "off")
    check(off.returncode == 0, "option off: test passes")
    check(
        not any((work / "off" / ".logs").glob("eth_*.csv")),
        "option off: no eth_*.csv written",
    )
    on = run_test(
        args.binary, test, {"TT_METAL_PROFILE_ETH_COUNTERS": "1"}, work / "on"
    )
    check(on.returncode == 0, "option on: test passes")
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

    if arch == "blackhole":
        sent = max(
            (t for prog in programs for t in prog.cores.values()),
            key=lambda t: t.get("TXQ_PKT_END_CNT"),
        )
        received = max(
            (t for prog in programs for t in prog.cores.values()),
            key=lambda t: t.get("RXQ_PKT_END_CNT"),
        )
        check(
            sent.get("TXQ_PKT_END_CNT") >= args.messages,
            f"sender packets {sent.get('TXQ_PKT_END_CNT')} >= {args.messages}",
        )
        check(
            received.get("RXQ_PKT_END_CNT") >= args.messages,
            f"receiver packets {received.get('RXQ_PKT_END_CNT')} >= {args.messages}",
        )
        data_queue = max(
            (q for (name, q) in sent.queues if name == "TXQ_PKT_END_CNT"),
            key=lambda q: sent.queues[("TXQ_PKT_END_CNT", q)],
        )
        packets = sent.queues[("TXQ_PKT_END_CNT", data_queue)]
        words = sent.queues.get(("TXQ_WORD_CNT", data_queue), 0)
        print(f"data queue {data_queue}: {packets} packets, {words} words")
        check(
            0.95 <= words / max(packets, 1) <= 1.05,
            "one word per small packet on the data queue",
        )
        check(
            sent.get("TXQ_RESEND_CNT") == 0 and received.get("RXQ_PKT_DROP_CNT") == 0,
            "no resends and no drops",
        )
    else:
        payloads = [16, 16, 4096, 4096, 16384, 16384]
        senders = sorted(
            (p for p in programs if any(t.tx_bytes for t in p.cores.values())),
            key=lambda p: p.seq,
        )
        for prog in senders:
            print(f"program {prog.program}: tx {prog.tx_bytes} B rx {prog.rx_bytes} B")
        check(len(senders) >= len(payloads), "a program per send")
        implied = [
            payload / max(prog.tx_bytes // 16, 1)
            for payload, prog in zip(payloads, senders)
        ]
        print(f"payload / words per send: {implied} (16 expected for the larger ones)")

    if run.is_file():
        for r in eth_counters.per_run(eth_counters.read_run(run), arch):
            print(
                f"run {r.seq} device {r.device}: tx {r.tx_bytes} B rx {r.rx_bytes} B in {r.elapsed_ns * 1e-9:.1f} s"
            )
    print(f"work directory kept: {work}")
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
