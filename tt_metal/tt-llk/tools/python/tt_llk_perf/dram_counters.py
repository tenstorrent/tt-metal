# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Decoder of the metal profiler's DRAM traffic logs (dram_traffic.csv, dram_idle.csv and dram_controller.csv).
Counter names and the word size come from tools/include/perf_counters/dram_counters_*.h through headers.py.
"""

import csv
from collections import defaultdict
from typing import Dict, Iterable, List, NamedTuple, Optional, Tuple

from . import headers

TRAFFIC_LOG = "dram_traffic.csv"
IDLE_LOG = "dram_idle.csv"
CONTROLLER_LOG = "dram_controller.csv"
FLAG_OVERLAPPED = 1
FLAG_UNSETTLED = 2
FLAG_INTERVAL_WRAPPED = 4

# A full controller command moves 64 B; a narrower host access is still one command, so this is an upper bound.
COMMAND_BYTES = 64
READ_COMMANDS = ("RD_CMD", "RD_AP_CMD")
WRITE_COMMANDS = ("WR_CMD", "WR_AP_CMD")

READ_WORDS = ("NIU_SLV_RD_DATA_WORD_SENT",)
WRITE_WORDS = (
    "NIU_SLV_NONPOSTED_WR_DATA_WORD_RECEIVED",
    "NIU_SLV_POSTED_WR_DATA_WORD_RECEIVED",
)


class Row(NamedTuple):
    device: int
    program: int
    seq: int
    flags: int
    elapsed_ns: int
    tile: Tuple[int, int]
    noc: int
    counter: str
    words: int  # the capture's own register reads are already taken off
    self_words: int  # words of the capture's own reads that were taken off


class Idle(NamedTuple):
    per_pair: float
    per_second: float


def read_traffic(path) -> List[Row]:
    """Rows of dram_traffic.csv; the NONE marker row of a program that moved nothing is kept as counter 'NONE'."""
    rows = []
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            rows.append(
                Row(
                    int(r["device_id"]),
                    int(r["program_id"]),
                    int(r["launch_seq"]),
                    int(r["flags"]),
                    int(r["elapsed_ns"]),
                    (int(r["dram_x"]), int(r["dram_y"])),
                    int(r["noc"]),
                    r["counter"],
                    int(r["delta_words"]),
                    int(r["self_words"]),
                )
            )
    return rows


def read_idle(path) -> Dict[tuple, Idle]:
    """(device, tile, noc, counter) -> idle drift model of dram_idle.csv."""
    idle = {}
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            key = (
                int(r["device_id"]),
                (int(r["dram_x"]), int(r["dram_y"])),
                int(r["noc"]),
                r["counter"],
            )
            idle[key] = Idle(
                float(r["words_per_snapshot_pair"]), float(r["words_per_second"])
            )
    return idle


def expected_idle_words(idle: Optional[Idle], elapsed_ns: int) -> float:
    """Words the idle drift model predicts over one snapshot pair that spans elapsed_ns."""
    if idle is None:
        return 0.0
    return idle.per_pair + idle.per_second * elapsed_ns * 1e-9


class TileTraffic(NamedTuple):
    read_words: int
    write_words: int
    read_bytes: int
    write_bytes: int


class ProgramTraffic(NamedTuple):
    device: int
    program: int
    seq: int
    flags: int
    elapsed_ns: int
    tiles: Dict[Tuple[int, int], TileTraffic]
    read_bytes: int
    write_bytes: int

    @property
    def overlapped(self) -> bool:
        return bool(self.flags & FLAG_OVERLAPPED)

    @property
    def unsettled(self) -> bool:
        return bool(self.flags & FLAG_UNSETTLED)

    @property
    def read_gbps(self) -> float:
        return self.read_bytes / self.elapsed_ns if self.elapsed_ns else 0.0

    @property
    def write_gbps(self) -> float:
        return self.write_bytes / self.elapsed_ns if self.elapsed_ns else 0.0


def per_program(
    rows: Iterable[Row],
    arch,
    idle: Optional[Dict[tuple, Idle]] = None,
    include_dir=None,
) -> List[ProgramTraffic]:
    """One entry per program launch, in launch order, with the idle drift taken off every register when given.
    Bytes are words times the arch word size. A tile's numbers add both NoCs. The capture's own register reads
    are already removed from the deltas by the runtime; the idle model covers the traffic of other agents.
    """
    word_bytes = headers.dram_layout(arch, include_dir).word_bytes
    if word_bytes == 0:
        raise ValueError(f"arch {arch!r} has no known DRAM tile word size")
    launches: Dict[tuple, List[Row]] = defaultdict(list)
    for row in rows:
        launches[(row.device, row.seq, row.program)].append(row)
    result = []
    for (device, seq, program), group in sorted(launches.items()):
        first = group[0]
        sums: Dict[Tuple[int, int], Dict[str, int]] = defaultdict(
            lambda: defaultdict(int)
        )
        for row in group:
            if row.counter == "NONE":
                continue
            words = row.words
            if idle is not None:
                key = (device, row.tile, row.noc, row.counter)
                words = max(
                    0,
                    round(words - expected_idle_words(idle.get(key), first.elapsed_ns)),
                )
            sums[row.tile][row.counter] += words
        tiles = {}
        for tile, counters in sums.items():
            rd = sum(counters.get(n, 0) for n in READ_WORDS)
            wr = sum(counters.get(n, 0) for n in WRITE_WORDS)
            tiles[tile] = TileTraffic(
                rd,
                wr,
                rd * word_bytes,
                wr * word_bytes,
            )
        result.append(
            ProgramTraffic(
                device,
                program,
                seq,
                first.flags,
                first.elapsed_ns,
                tiles,
                sum(t.read_bytes for t in tiles.values()),
                sum(t.write_bytes for t in tiles.values()),
            )
        )
    return result


class ControllerRow(NamedTuple):
    device: int
    program: int
    seq: int
    flags: int
    elapsed_ns: int
    channel: int
    tile: Tuple[int, int]
    counter: str
    value: int


def read_controller(path) -> List[ControllerRow]:
    """Rows of dram_controller.csv: nonzero activity monitor counts of one controller interval per launch."""
    with open(path, newline="") as f:
        return [
            ControllerRow(
                int(r["device_id"]),
                int(r["program_id"]),
                int(r["launch_seq"]),
                int(r["flags"]),
                int(r["elapsed_ns"]),
                int(r["channel"]),
                (int(r["dram_x"]), int(r["dram_y"])),
                r["counter"],
                int(r["value"]),
            )
            for r in csv.DictReader(f)
        ]


class ControllerTraffic(NamedTuple):
    device: int
    program: int
    seq: int
    flags: int
    elapsed_ns: int
    channels: Dict[int, Dict[str, int]]
    read_commands: int
    write_commands: int
    activates: int

    @property
    def wrapped(self) -> bool:
        return bool(self.flags & FLAG_INTERVAL_WRAPPED)

    @property
    def row_hit_rate(self) -> float:
        """Share of read and write commands that found their row open."""
        commands = self.read_commands + self.write_commands
        return max(0.0, 1.0 - self.activates / commands) if commands else 0.0

    @property
    def command_bytes(self) -> int:
        return (self.read_commands + self.write_commands) * COMMAND_BYTES


def per_program_controller(rows: Iterable[ControllerRow]) -> List[ControllerTraffic]:
    """One entry per launch in launch order: counters per channel and the command totals over all channels.
    The flags of a launch are the union over its channels, so a wrap on any channel shows.
    """
    launches: Dict[tuple, List[ControllerRow]] = defaultdict(list)
    for row in rows:
        launches[(row.device, row.seq, row.program)].append(row)
    result = []
    for (device, seq, program), group in sorted(launches.items()):
        channels: Dict[int, Dict[str, int]] = defaultdict(dict)
        flags = 0
        for row in group:
            channels[row.channel][row.counter] = row.value
            flags |= row.flags

        def total(names):
            return sum(c.get(n, 0) for c in channels.values() for n in names)

        result.append(
            ControllerTraffic(
                device,
                program,
                seq,
                flags,
                group[0].elapsed_ns,
                dict(channels),
                total(READ_COMMANDS),
                total(WRITE_COMMANDS),
                total(("ACT_CMD",)),
            )
        )
    return result


def format_controller_table(programs: List[ControllerTraffic]) -> str:
    lines = ["device program seq  rd_cmds  wr_cmds  activates  row_hit  flags"]
    for p in programs:
        lines.append(
            f"{p.device:6d} {p.program:7d} {p.seq:3d} {p.read_commands:8d} {p.write_commands:8d} "
            f"{p.activates:10d} {p.row_hit_rate:8.2f}  {p.flags}"
        )
    return "\n".join(lines)


def format_table(programs: List[ProgramTraffic]) -> str:
    lines = [
        "device program seq   read_bytes  write_bytes  elapsed_us  rd_GB/s  wr_GB/s  flags"
    ]
    for p in programs:
        lines.append(
            f"{p.device:6d} {p.program:7d} {p.seq:3d} {p.read_bytes:12d} {p.write_bytes:12d} "
            f"{p.elapsed_ns / 1e3:11.1f} {p.read_gbps:8.2f} {p.write_gbps:8.2f}  {p.flags}"
        )
    return "\n".join(lines)


def main(argv=None) -> int:
    import argparse

    parser = argparse.ArgumentParser(
        description="Summarize dram_traffic.csv per program"
    )
    parser.add_argument("log_dir", help="profiler log directory with dram_traffic.csv")
    parser.add_argument("--arch", default="blackhole")
    parser.add_argument(
        "--no-idle", action="store_true", help="do not subtract the idle drift model"
    )
    args = parser.parse_args(argv)
    from pathlib import Path

    log_dir = Path(args.log_dir)
    rows = read_traffic(log_dir / TRAFFIC_LOG)
    idle_path = log_dir / IDLE_LOG
    idle = read_idle(idle_path) if idle_path.is_file() and not args.no_idle else None
    print(format_table(per_program(rows, args.arch, idle)))
    controller_path = log_dir / CONTROLLER_LOG
    if controller_path.is_file():
        print()
        print(
            format_controller_table(
                per_program_controller(read_controller(controller_path))
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
