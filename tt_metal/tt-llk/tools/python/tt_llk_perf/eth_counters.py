# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Decoder of the metal profiler's ethernet counter logs (eth_traffic.csv, eth_run.csv and eth_idle.csv).
Counter names, word sizes and which counters are registers come from tools/include/perf_counters/eth_counters_*.h
through headers.py.
"""

import csv
from collections import defaultdict
from typing import Dict, Iterable, List, NamedTuple, Optional, Tuple

from . import headers

TRAFFIC_LOG = "eth_traffic.csv"
RUN_LOG = "eth_run.csv"
IDLE_LOG = "eth_idle.csv"
FLAG_OVERLAPPED = 1
FLAG_RESET = 2
NO_QUEUE = headers.NO_QUEUE


class Row(NamedTuple):
    device: int
    program: int  # -1 in a run row
    seq: int  # launch_seq, or run_id in a run row
    flags: int
    elapsed_ns: int
    core: Tuple[int, int]  # translated coordinates of the ethernet core
    channel: int
    counter: str
    queue: int  # NO_QUEUE when the counter does not belong to a queue
    delta: int


def _row(r, with_program: bool) -> Row:
    return Row(
        int(r["device_id"]),
        int(r["program_id"]) if with_program else -1,
        int(r["launch_seq"]) if with_program else int(r["run_id"]),
        int(r["flags"]),
        int(r["elapsed_ns"]),
        (int(r["eth_x"]), int(r["eth_y"])),
        int(r["channel"]),
        r["counter"],
        int(r["queue"]),
        int(r["delta"]),
    )


def read_traffic(path) -> List[Row]:
    """Rows of eth_traffic.csv; the NONE marker row of a program that moved nothing is kept as counter 'NONE'."""
    with open(path, newline="") as f:
        return [_row(r, True) for r in csv.DictReader(f)]


def read_run(path) -> List[Row]:
    """Rows of eth_run.csv: the change of every counter from device initialization to device close."""
    with open(path, newline="") as f:
        return [_row(r, False) for r in csv.DictReader(f)]


def read_idle(path) -> Dict[tuple, float]:
    """(device, core, counter, queue) -> counts per second that move with nothing running, from eth_idle.csv."""
    idle = {}
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            key = (
                int(r["device_id"]),
                (int(r["eth_x"]), int(r["eth_y"])),
                r["counter"],
                int(r["queue"]),
            )
            idle[key] = float(r["counts_per_second"])
    return idle


def register_counters(arch, include_dir=None) -> set:
    """Names of the counters that are live queue registers; the others are words the base firmware refreshes."""
    tx = headers.eth_tx_queue_table(arch, include_dir)
    rx = headers.eth_rx_queue_table(arch, include_dir)
    return {n for n, _ in tx + rx}


class CoreTraffic(NamedTuple):
    core: Tuple[int, int]
    channel: int
    counters: Dict[str, int]  # name -> sum over queues
    queues: Dict[Tuple[str, int], int]  # (name, queue) -> count
    tx_bytes: int
    rx_bytes: int

    def get(self, name: str) -> int:
        return self.counters.get(name, 0)


class ProgramTraffic(NamedTuple):
    device: int
    program: int
    seq: int
    flags: int
    elapsed_ns: int
    cores: Dict[Tuple[int, int], CoreTraffic]
    tx_bytes: int
    rx_bytes: int

    @property
    def overlapped(self) -> bool:
        return bool(self.flags & FLAG_OVERLAPPED)

    @property
    def reset(self) -> bool:
        return bool(self.flags & FLAG_RESET)

    @property
    def tx_gbps(self) -> float:
        return self.tx_bytes / self.elapsed_ns if self.elapsed_ns else 0.0

    @property
    def rx_gbps(self) -> float:
        return self.rx_bytes / self.elapsed_ns if self.elapsed_ns else 0.0

    @property
    def resends(self) -> int:
        return sum(c.get("TXQ_RESEND_CNT") for c in self.cores.values())

    @property
    def drops(self) -> int:
        return sum(c.get("RXQ_PKT_DROP_CNT") for c in self.cores.values())


def _summarize(
    rows: List[Row], arch, idle, registers, include_dir
) -> Dict[Tuple[int, int], CoreTraffic]:
    layout = headers.eth_layout(arch, include_dir)
    first = rows[0] if rows else None
    sums: Dict[Tuple[int, int], Dict[Tuple[str, int], int]] = defaultdict(
        lambda: defaultdict(int)
    )
    channels: Dict[Tuple[int, int], int] = {}
    for row in rows:
        if row.counter == "NONE":
            continue
        count = row.delta
        if idle is not None and row.counter in registers:
            rate = idle.get((row.device, row.core, row.counter, row.queue), 0.0)
            count = max(0, round(count - rate * first.elapsed_ns * 1e-9))
        sums[row.core][(row.counter, row.queue)] += count
        channels[row.core] = row.channel
    cores = {}
    for core, per in sums.items():
        total: Dict[str, int] = defaultdict(int)
        for (name, _), count in per.items():
            total[name] += count
        cores[core] = CoreTraffic(
            core,
            channels[core],
            dict(total),
            dict(per),
            total.get("TXQ_WORD_CNT", 0) * layout.txq_word_bytes,
            total.get("RXQ_WORD_CNT", 0) * layout.rxq_word_bytes,
        )
    return cores


def per_program(
    rows: Iterable[Row],
    arch,
    idle: Optional[Dict[tuple, float]] = None,
    include_dir=None,
) -> List[ProgramTraffic]:
    """One entry per program launch, in launch order, with the idle rate of the live queue registers taken off when
    given. Bytes are queue words times the arch word size, summed over queues. The values read from base firmware L1
    are refreshed about once a second: they are exact only over windows of several seconds.
    """
    layout = headers.eth_layout(arch, include_dir)
    if layout.txq_word_bytes == 0:
        raise ValueError(f"arch {arch!r} has no known ethernet counter map")
    registers = register_counters(arch, include_dir)
    launches: Dict[tuple, List[Row]] = defaultdict(list)
    for row in rows:
        launches[(row.device, row.seq, row.program)].append(row)
    result = []
    for (device, seq, program), group in sorted(launches.items()):
        first = group[0]
        cores = _summarize(group, arch, idle, registers, include_dir)
        result.append(
            ProgramTraffic(
                device,
                program,
                seq,
                first.flags,
                first.elapsed_ns,
                cores,
                sum(c.tx_bytes for c in cores.values()),
                sum(c.rx_bytes for c in cores.values()),
            )
        )
    return result


def per_run(rows: Iterable[Row], arch, include_dir=None) -> List[ProgramTraffic]:
    """One entry per device open to close in eth_run.csv, as a ProgramTraffic with program -1 and seq the run id."""
    layout = headers.eth_layout(arch, include_dir)
    if layout.txq_word_bytes == 0:
        raise ValueError(f"arch {arch!r} has no known ethernet counter map")
    registers = register_counters(arch, include_dir)
    by_run: Dict[tuple, List[Row]] = defaultdict(list)
    for row in rows:
        by_run[(row.device, row.seq)].append(row)
    out = []
    for (device, run), group in sorted(by_run.items()):
        first = group[0]
        cores = _summarize(group, arch, None, registers, include_dir)
        out.append(
            ProgramTraffic(
                device,
                -1,
                run,
                first.flags,
                first.elapsed_ns,
                cores,
                sum(c.tx_bytes for c in cores.values()),
                sum(c.rx_bytes for c in cores.values()),
            )
        )
    return out


def format_table(programs: List[ProgramTraffic]) -> str:
    lines = [
        "device program seq   tx_bytes   rx_bytes  elapsed_us  tx_GB/s  rx_GB/s  resends  drops  flags"
    ]
    for p in programs:
        lines.append(
            f"{p.device:6d} {p.program:7d} {p.seq:3d} {p.tx_bytes:10d} {p.rx_bytes:10d} "
            f"{p.elapsed_ns / 1e3:11.1f} {p.tx_gbps:8.2f} {p.rx_gbps:8.2f} {p.resends:8d} {p.drops:6d}  {p.flags}"
        )
    return "\n".join(lines)


def main(argv=None) -> int:
    import argparse
    from pathlib import Path

    parser = argparse.ArgumentParser(
        description="Summarize eth_traffic.csv per program and eth_run.csv per device"
    )
    parser.add_argument(
        "log_dir", help="profiler log directory with the eth_*.csv files"
    )
    parser.add_argument("--arch", default="blackhole")
    parser.add_argument(
        "--no-idle", action="store_true", help="do not subtract the idle rate"
    )
    args = parser.parse_args(argv)
    log_dir = Path(args.log_dir)
    traffic = log_dir / TRAFFIC_LOG
    if traffic.is_file():
        idle_path = log_dir / IDLE_LOG
        idle = (
            read_idle(idle_path) if idle_path.is_file() and not args.no_idle else None
        )
        print(format_table(per_program(read_traffic(traffic), args.arch, idle)))
    run = log_dir / RUN_LOG
    if run.is_file():
        print("device run  tx_bytes  rx_bytes  elapsed_s  resends  drops")
        for p in per_run(read_run(run), args.arch):
            print(
                f"{p.device:6d} {p.seq:3d} {p.tx_bytes:9d} {p.rx_bytes:9d} {p.elapsed_ns * 1e-9:10.1f} {p.resends:8d} {p.drops:6d}"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
