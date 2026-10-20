#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Checks that perf builds with counters off contain no counter code, and that builds with counters on still do.

Disassembles every TRISC ELF and the BRISC ELF of two --compile-producer outputs (RUNNER_TEMP dirs) and reports each
address of a counter register, the launch word or the counter L1 region (counters.h) the code builds in a register or
keeps in data, and every llk_perf symbol. Counters off must have none. Every counters on ELF must have some, which also
proves the scan sees them.
usage: no_counter_code.py <objdump> <wormhole|blackhole|quasar> <counters off RUNNER_TEMP> <counters on RUNNER_TEMP>
"""

import re
import struct
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ELFS = ("unpack", "math", "pack", "sfpu")
SHF_ALLOC, SHF_EXECINSTR, SHT_NOBITS, SHT_SYMTAB = 2, 4, 8, 2

# perf_counters/registers.h: ref_period, mode, control, out_l and out_h of every bank, then PERF_CNT_ALL and
# PERF_CNT_MUX_CTRL; not DBG_FEATURE_DISABLE, which both builds clear before the TRISCs run.
_TT1XX_OFFSETS = (
    (0x0, 0x4, 0x8, 0x100, 0x104),  # INSTRN_THREAD
    (0x18, 0x1C, 0x20, 0x120, 0x124),  # FPU
    (0xC, 0x10, 0x14, 0x108, 0x10C),  # TDMA_UNPACK
    (0x30, 0x34, 0x38, 0x118, 0x11C),  # L1
    (0xF0, 0xF4, 0xF8, 0x110, 0x114),  # TDMA_PACK
    (0x3C, 0x218),
)
_QUASAR_OFFSETS = (
    (0x0, 0x4, 0x8, 0x98, 0x9C),  # INSTRN_THREAD
    (0x18, 0x1C, 0x20, 0xB0, 0xB4),  # FPU
    (0xC, 0x10, 0x14, 0xA0, 0xA4),  # TDMA_UNPACK
    (0x8C, 0x90, 0x94, 0xA8, 0xAC),  # TDMA_PACK
    (0x24, 0xF4, 0xA0AC, 0xA0B0),  # and the l1_client ctrl and cnt
)
_QUASAR_WINDOWS = (0x00800000,) + tuple(0x01800000 + n * 0x10000 for n in range(4))
_TT1XX = {0xFFB12000 + o for bank in _TT1XX_OFFSETS for o in bank}
REGISTERS = {
    # counters.h COUNTED_ZONES_REG: STREAM_SCRATCH_1 of overlay stream 0
    "wormhole": _TT1XX | {0xFFB40000 + 249 * 4},
    "blackhole": _TT1XX | {0xFFB40000 + 37 * 4},
    "quasar": {
        w + o for w in _QUASAR_WINDOWS for bank in _QUASAR_OFFSETS for o in bank
    },
}
# perf.h PERF_COUNTERS_BASE_ADDR up to the profiler region: config, zone results, Quasar's launch word.
COUNTER_L1 = range(0x169000, 0x16AFF0)
# Not counter code: the debug feature scrub and the zone object (its rendezvous) both builds run.
ALLOWED_SYMBOLS = ("clear_debug_feature", "perf_counter_scoped")

_INSN = re.compile(r"^\s*([0-9a-f]+):\s+(\S+)\s*([^#<]*)")
_MEM = re.compile(r"^(-?\d+)\((\w+)\)$")
_NO_RD = {"j", "jr", "ret", "nop", "ecall", "ebreak", "wfi"}
_NO_RD_PREFIXES = ("b", "fence", "csrw", "csrs", "csrc")  # branches, fences, CSR writes
_STORES = {"sb", "sh", "sw", "fsw"}
_CALLS = {"jal", "jalr", "call", "tail"}
_JUMPS = {"j", "jr", "ret", "mret"}


def _imm(text):
    return int(text, 0) & 0xFFFFFFFF


def counter_address(arch, value):
    return value in REGISTERS[arch] or value in COUNTER_L1


def scan_code(arch, disassembly):
    """Counter addresses the code builds: lui/addi/li/mv constants per register, checked at each load, store and addi."""
    hits, regs = [], {}
    for line in disassembly.splitlines():
        if line.endswith(">:"):
            regs = {}  # a function starts
            continue
        m = _INSN.match(line)
        if not m:
            continue
        pc, op = m.group(1), m.group(2)
        args = [a.strip() for a in m.group(3).split(",") if a.strip()]
        try:
            if op == "lui":
                regs[args[0]] = _imm(args[1]) << 12 & 0xFFFFFFFF
            elif op == "li":
                regs[args[0]] = _imm(args[1])
            elif op == "mv":
                regs[args[0]] = regs.get(args[1])
            elif op == "addi":
                base = 0 if args[1] == "zero" else regs.get(args[1])
                value = None if base is None else (base + int(args[2], 0)) & 0xFFFFFFFF
                regs[args[0]] = value
                if value is not None and counter_address(arch, value):
                    hits.append(f"{pc}: {op} {','.join(args)} = {value:#x}")
            elif len(args) >= 2 and _MEM.match(args[-1]):
                off, base = _MEM.match(args[-1]).groups()
                if regs.get(base) is not None:
                    addr = (regs[base] + int(off)) & 0xFFFFFFFF
                    if counter_address(arch, addr):
                        hits.append(f"{pc}: {op} {','.join(args)} -> {addr:#x}")
                if op not in _STORES:
                    regs[args[0]] = None
            elif op in _CALLS or op in _JUMPS:
                regs = (
                    {}
                )  # a call clobbers them; after a jump the next block is reached from elsewhere
            elif args and op not in _NO_RD and not op.startswith(_NO_RD_PREFIXES):
                regs[args[0]] = None
        except (ValueError, IndexError):
            if args:
                regs[args[0]] = None
    return hits


def scan_elf(objdump, arch, path):
    """Code hits, counter register addresses in loaded data, and llk_perf symbols of one ELF."""
    data = Path(path).read_bytes()
    (shoff,) = struct.unpack_from("<I", data, 0x20)
    shentsize, shnum, shstrndx = struct.unpack_from("<HHH", data, 0x2E)
    heads = [
        struct.unpack_from("<IIIIIIIIII", data, shoff + i * shentsize)
        for i in range(shnum)
    ]

    def name(off, table):
        return data[table[4] + off : data.index(b"\0", table[4] + off)].decode()

    hits = []
    for h in heads:
        section = name(h[0], heads[shstrndx])
        if h[2] & SHF_ALLOC and not h[2] & SHF_EXECINSTR and h[1] != SHT_NOBITS:
            for i in range(0, h[5] - 3, 4):
                (word,) = struct.unpack_from("<I", data, h[4] + i)
                if word in REGISTERS[arch]:
                    hits.append(f"{section}+{i:#x}: {word:#x}")
        if h[1] == SHT_SYMTAB:
            for off in range(h[4], h[4] + h[5], 16):
                symbol = name(struct.unpack_from("<I", data, off)[0], heads[h[6]])
                if ("llk_perf" in symbol or "llk4perf" in symbol) and not any(
                    a in symbol for a in ALLOWED_SYMBOLS
                ):
                    hits.append(f"symbol {symbol}")
    disassembly = subprocess.run(
        [objdump, "-d", "--no-show-raw-insn", str(path)],
        capture_output=True,
        text=True,
        check=True,
    )
    return scan_code(arch, disassembly.stdout) + hits


def elfs(root):
    build = Path(root) / "tt-llk-build"
    found = sorted(p for p in (build / "sources").rglob("*.elf") if p.stem in ELFS)
    return found + sorted(build.glob("shared*/elf/brisc.elf"))


def main(objdump, arch, off, on):
    off_elfs, on_elfs = elfs(off), elfs(on)
    if not off_elfs or not on_elfs:
        sys.exit("no ELFs in one of the builds: are these --compile-producer outputs?")
    with ThreadPoolExecutor() as pool:
        off_hits = list(pool.map(lambda p: scan_elf(objdump, arch, p), off_elfs))
        on_hits = list(pool.map(lambda p: scan_elf(objdump, arch, p), on_elfs))
    bad = 0
    for path, hits in zip(off_elfs, off_hits):
        if hits:
            bad += 1
            print(
                f"counters off {path.relative_to(off)}: {len(hits)} counter accesses, first {'; '.join(hits[:3])}"
            )
    blind = [p for p, hits in zip(on_elfs, on_hits) if not hits]
    for path in blind:
        print(f"counters on {path.relative_to(on)}: no counter access found")
    print(
        f"counters off: {len(off_elfs)} ELFs, {bad} with counter code; "
        f"counters on: {len(on_elfs)} ELFs, {len(on_elfs) - len(blind)} with counter code "
        f"({sum(map(len, on_hits))} accesses)"
    )
    if len(off_elfs) != len(on_elfs):
        print(
            f"the builds differ in ELF count: {len(off_elfs)} counters off, {len(on_elfs)} counters on"
        )
    return 1 if bad or blind or len(off_elfs) != len(on_elfs) else 0


if __name__ == "__main__":
    if len(sys.argv) != 5 or sys.argv[2] not in REGISTERS:
        sys.exit(__doc__)
    sys.exit(main(*sys.argv[1:]))
