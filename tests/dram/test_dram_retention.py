# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

"""GDDR retention and read-disturb test: nearly all of DRAM is written once, then read back later.

The health check runs test_gddr_retention in its weekly and full tiers (PYTESTS["gddr_retention"] in
tools/scaleout/exabox/health_check_test_suite/diag/device_tests.py), and
tests/tt_metal/tt_metal/deployment/test_runner.sh runs it every iteration. By hand, on every chip unless
RETENTION_CHIPS lists some (RETENTION_CHIPS=0,5):

    pytest tests/dram/test_dram_retention.py

test_gddr_retention writes each bank once with a hashed pattern, then reads and compares all of it in passes that start
RETENTION_GAPS_S apart (default 2,2,2: reads 2, 4 and 6 s after the write). Every sweep visits a bank in the same
order, so each address waits the gap. Each failing word is reported in the JSON with its chip, bank, address, expected
and read values, flipped bits, first failing read and its page's classification, up to LOG_CAP (112) words a slice per
read; the rest are counted as unlogged_words. RETENTION_SECONDS keeps reading at the last gap until that long after the
write; RETENTION_REPS=n reads each chunk n times a pass and counts every failing read.

test_gddr_retention_detects_changes is the negative test. It reads on the same default schedule, whatever
RETENTION_GAPS_S says. Between the write and the reads a third program flips bits of single words (the first and last
tested word, a slice boundary, the DIIM-751 address, and one more between the first and second reads), puts a line back
to the previous pattern's data, overwrites a line with another address's data and copies the first line above the swept
range into a line. Exactly those words must be reported, with their values and first read, and the three lines
classified as stale and as aliases of the differing address bit.

Coverage is [0x40, lowest kernel binary) of every bank, 99.9999% on Blackhole. That includes the reserved bottom, so the
profiler, device print, watcher and DRAM-backed command queue must be off. Every program is launched once before the
coverage buffer is sized, so its binaries are placed; later launches change only runtime args, which reuses the cached
programs, so nothing else lands in DRAM during the sweep.

Each Tensix core works on the bank whose optimal worker core is nearest, at most ceil(cores / 8) cores a bank; its two
data-movement RISCs take a contiguous slice each.

The pattern (one hash per 64 B line, XORed per word, unique per bank, address and seed) and the miss classification
(stale: the previous pattern's data; alias of address bit k: the data at the address with bit k flipped; other) come
from Nikola Milicevic's DRAM sweep stress test. A slice classifies its first RETENTION_DECODE_MAX (4096) bad pages of a
read; later ones are undecoded. RETENTION_PATTERNS=2 adds the complement (seed 2m + 1 inverts seed 2m), so every cell
holds both a 1 and a 0 and a miss can be classified stale.
"""

import json
import os
import random
import socket
import time
from collections import defaultdict
from pathlib import Path

import torch
from loguru import logger

import ttnn

BANKS = 8
PAGE_BYTES = 2048
PAGE_WORDS = PAGE_BYTES // 4
BARRIER_BYTES = 64
# Blackhole's DRAM bank size, dram_bank_size in its soc descriptor: 4080 MiB.
DRAM_TOP = 0xFF000000
ROW_WORDS = 512
LOG_OFF = 64
LOG_CAP = (ROW_WORDS - LOG_OFF) // 4
MAGIC = 0xC0FFEE
NONE = 0xFFFFFFFF
MASK32 = 0xFFFFFFFF
CB_SCRATCH = (0, 1)
SELF_TEST_BANK = 3

MODE_CHECK, MODE_FILL, MODE_POKE = 0, 1, 2
POKE_FLIP, POKE_LINE, POKE_COPY = 1, 2, 3
POKE_FIELDS = 5  # kind, bank, address, value, seed
MAX_POKES = 8
# Pass indices the write and poke programs report, so their rows can't pass for a read's.
FILL_PASS, POKE_PASS = 0x2000, 0x1000
# How a miscompared page was classified, logged with each of its words. An alias of address bit k is logged as k.
VERDICT_OTHER, VERDICT_STALE, VERDICT_UNDECODED = 0, 1, 0xFF

# Report row: header words, then LOG_CAP entries of (address, read, expected, rep | verdict << 16).
H_SEED, H_PASS, H_BAD_PAGES, H_BAD_WORDS, H_STALE, H_OTHER, H_UNDECODED, H_LOGGED = range(8)
H_WORDS_LO, H_WORDS_HI, H_BITS_LOST, H_BITS_GAINED, H_BANK, H_LO, H_HI, H_MAGIC = range(8, 16)
H_LANES, H_OUTSIDE, H_ALIAS = 16, 20, 21

REPO = Path(__file__).resolve().parents[2]
UNSAFE_ENV = ("TT_METAL_DEVICE_PROFILER", "TT_METAL_DPRINT_CORES", "TT_METAL_WATCHER", "TT_METAL_DRAM_BACKED_CQ")

KERNEL_DIR = "ttnn/cpp/ttnn/operations/experimental/diagnostics/dram/kernels"
DM_KERNEL = f"{KERNEL_DIR}/retention_dm.cpp"
# The constants above that DM_KERNEL takes as RETENTION_<name> defines, so it and the report decoder agree.
KERNEL_CONSTANTS = (
    "PAGE_BYTES PAGE_WORDS DRAM_TOP ROW_WORDS LOG_OFF LOG_CAP MAGIC NONE MODE_FILL MODE_POKE POKE_FIELDS POKE_FLIP "
    "POKE_COPY VERDICT_OTHER VERDICT_STALE VERDICT_UNDECODED H_SEED H_PASS H_BAD_PAGES H_BAD_WORDS H_STALE H_OTHER "
    "H_UNDECODED H_LOGGED H_WORDS_LO H_WORDS_HI H_BITS_LOST H_BITS_GAINED H_BANK H_LO H_HI H_MAGIC H_LANES H_OUTSIDE "
    "H_ALIAS"
).split()
DM_DEFINES = [(f"RETENTION_{name}", str(globals()[name])) for name in KERNEL_CONSTANTS]


def mix(value):
    value &= MASK32
    value ^= value >> 16
    value = (value * 0x7FEB352D) & MASK32
    value ^= value >> 15
    value = (value * 0x846CA68B) & MASK32
    value ^= value >> 16
    return value


def pattern_word(addr, bank, seed):
    """The host's copy of the kernel's pattern, for the negative test's expectations."""
    key = mix(mix((addr >> 6) ^ (((seed >> 1) * 0x9E3779B9) & MASK32)) ^ ((bank * 0x85EBCA77) & MASK32))
    value = key ^ ((((addr >> 2) & 15) * 0x01234567) & MASK32)
    return (~value & MASK32) if seed & 1 else value


def verdict_name(verdict):
    names = {VERDICT_OTHER: "other", VERDICT_STALE: "stale", VERDICT_UNDECODED: "undecoded"}
    return names.get(verdict, f"alias_bit{verdict}")


def env_int(name, default):
    return int(os.environ.get(name, str(default)), 0)


def env_float(name, default):
    return float(os.environ.get(name, str(default)))


# Read schedule: each read pass starts this many seconds after the one before it (the first, after the write).
# RETENTION_GAPS_S overrides it for test_gddr_retention; the negative test always uses it.
DEFAULT_GAPS_S = [2, 2, 2]


def read_config():
    unsafe = [name for name in UNSAFE_ENV if os.environ.get(name, "0") not in ("", "0")]
    assert not unsafe, f"{unsafe} use the reserved bottom of DRAM, which this test overwrites; unset them"
    cfg = {
        "gaps_s": [
            float(gap) for gap in os.environ.get("RETENTION_GAPS_S", ",".join(map(str, DEFAULT_GAPS_S))).split(",")
        ],
        "seconds": env_float("RETENTION_SECONDS", 0),
        "reps": env_int("RETENTION_REPS", 1),
        "max_passes": env_int("RETENTION_MAX_PASSES", 1 << 30),
        "patterns": env_int("RETENTION_PATTERNS", 1),
        "lo": env_int("RETENTION_LO", BARRIER_BYTES),
        "chunk": env_int("RETENTION_CHUNK", 16),
        "decode_max": env_int("RETENTION_DECODE_MAX", 4096),
        "self_test": env_int("RETENTION_SELF_TEST", 1),
        "seed0": (env_int("RETENTION_SEED", int(time.time()) & 0xFFFF) * 8) & 0xFFFFFFF8,
    }
    assert cfg["lo"] >= BARRIER_BYTES and cfg["lo"] % 64 == 0, "RETENTION_LO must be 64 B aligned and above the barrier"
    assert cfg["gaps_s"] and min(cfg["gaps_s"]) >= 0, "RETENTION_GAPS_S must list one or more gaps of 0 s or more"
    # Any of these at 0 would pass without comparing anything.
    assert min(cfg["reps"], cfg["patterns"], cfg["max_passes"]) >= 1, "RETENTION_REPS, _PATTERNS, _MAX_PASSES >= 1"
    return cfg


def select_chips():
    chips_env = os.environ.get("RETENTION_CHIPS", "all")
    if chips_env == "all":
        return list(range(ttnn.GetNumAvailableDevices()))
    return [int(chip) for chip in chips_env.split(",")]


def assign_banks(cores, anchors):
    """Each core goes to the bank whose optimal worker is nearest, with at most ceil(cores / banks) cores a bank."""
    capacity = -(-len(cores) // len(anchors))
    pairs = sorted(
        (abs(core_x - anchor_x) + abs(core_y - anchor_y), bank, index)
        for index, (core_x, core_y) in enumerate(cores)
        for bank, (anchor_x, anchor_y) in enumerate(anchors)
    )
    owner, load = [-1] * len(cores), [0] * len(anchors)
    for _, bank, index in pairs:
        if owner[index] < 0 and load[bank] < capacity:
            owner[index] = bank
            load[bank] += 1
    return owner


def split_range(lo, hi, parts):
    pages = -(-(hi - lo) // PAGE_BYTES)
    bounds = [min(hi, lo + (part * pages // parts) * PAGE_BYTES) for part in range(parts + 1)]
    return list(zip(bounds[:-1], bounds[1:]))


def dram_buffers(device):
    return sorted(
        (info.address, info.max_size_per_bank)
        for info in ttnn._ttnn.reports.get_buffers(device)
        if info.buffer_type == ttnn.BufferType.DRAM
    )


def allocate_rows(device, rows):
    return ttnn.allocate_tensor_on_device(
        ttnn.Shape([1, 1, rows, PAGE_WORDS]), ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT, device, ttnn.DRAM_MEMORY_CONFIG
    )


def reserve_dram(device, base, buffers):
    """Allocate the largest buffer that starts at `base` and ends below every other DRAM allocation in `buffers`.

    Returns the buffer and the covered end: the next allocation above it, since the sub-page gap between the two is
    free and nothing is allocated while the sweep runs."""
    limit = min((addr for addr, size in buffers if addr + size > base), default=DRAM_TOP)
    # Each probe is freed before the next, so every size is tried against the same empty space.
    best, low, high = 0, 1, (limit - base) // PAGE_BYTES
    while low <= high:
        pages = (low + high) // 2
        try:
            ttnn.deallocate(allocate_rows(device, pages * BANKS))
        except Exception:
            high = pages - 1
            continue
        best, low = pages, pages + 1
    assert best, "could not reserve any DRAM"
    tensor = allocate_rows(device, best * BANKS)
    assert tensor.buffer_address() == base, f"reservation at {tensor.buffer_address():#x}, expected {base:#x}"
    assert (
        limit - (base + best * PAGE_BYTES) < PAGE_BYTES
    ), f"reservation ends {limit - base - best * PAGE_BYTES} B short"
    return tensor, limit & ~63


def read_out(tt_out, slots, seed, pass_index):
    out = ttnn.to_torch(tt_out).reshape(slots, ROW_WORDS).to(torch.int64) & MASK32
    assert (out[:, H_MAGIC] == MAGIC).all(), "a RISC did not report"
    assert (out[:, H_SEED] == seed).all() and (out[:, H_PASS] == pass_index).all(), "stale report"
    return out


class ChipRun:
    def __init__(self, chip, device, cfg, poke=False):
        self.chip, self.device, self.cfg = chip, device, cfg
        grid = device.compute_with_storage_grid_size()
        self.cores = [(x, y) for y in range(grid.y) for x in range(grid.x)]
        self.grid = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(grid.x - 1, grid.y - 1))])
        anchors = [
            (core.x, core.y) for core in ttnn.get_optimal_dram_bank_to_logical_worker_assignment(device, ttnn.NOC.NOC_0)
        ]
        assert len(anchors) == BANKS, f"chip {chip}: {len(anchors)} DRAM banks, expected {BANKS}"
        self.core_bank = assign_banks(self.cores, anchors)
        self.slots = 2 * len(self.cores)
        self.tt_out = ttnn.from_torch(
            torch.zeros(1, 1, self.slots, ROW_WORDS, dtype=torch.int32),
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=device,
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )
        scratch_bytes = ROW_WORDS * 4 + (2 * cfg["chunk"] + 1) * PAGE_BYTES
        self.cbs = [
            ttnn.CBDescriptor(
                total_size=scratch_bytes,
                core_ranges=self.grid,
                format_descriptors=[
                    ttnn.CBFormatDescriptor(buffer_index=index, data_format=ttnn.uint32, page_size=scratch_bytes)
                ],
            )
            for index in CB_SCRATCH
        ]
        self.out_args = ttnn.TensorAccessorArgs(self.tt_out).get_compile_time_args()

        # Place every program's binaries before sizing the reservation; later launches only change runtime args,
        # which reuses the cached programs and allocates nothing in DRAM.
        self.buffers_at_open = dram_buffers(device)
        probe = allocate_rows(device, BANKS)
        self.dram_base = probe.buffer_address()
        self.covered_lo = self.covered_hi = self.dram_base
        self.dram = probe
        empty = [(self.core_bank[index // 2], self.dram_base, self.dram_base) for index in range(self.slots)]
        programs = [(MODE_FILL, FILL_PASS), (MODE_CHECK, 0)] + ([(MODE_POKE, POKE_PASS)] if poke else [])
        for mode, pass_index in programs:
            self.launch(mode, empty, NONE, NONE, 1, pass_index)
            read_out(self.tt_out, self.slots, NONE, pass_index)
        ttnn.deallocate(probe)
        # Read once, before reserving: ttnn's buffer report walks every page of an interleaved buffer, so with the
        # reservation's 16.7 million pages allocated, each read takes 0.2 s.
        buffers = dram_buffers(device)
        self.dram, self.covered_hi = reserve_dram(device, self.dram_base, buffers)
        self.covered_lo = cfg["lo"]
        # The sweep starts at the bottom of the bank, so a buffer allocated before the probe would be overwritten.
        inside = [(hex(addr), size) for addr, size in buffers if addr < self.covered_hi]
        assert not inside, f"chip {chip}: DRAM buffers inside the swept range {inside}"
        self.above = [(addr, size) for addr, size in buffers if addr >= self.covered_hi]

        self.ranges = [None] * self.slots
        for bank in range(BANKS):
            slots = [2 * index + risc for index, owner in enumerate(self.core_bank) if owner == bank for risc in (0, 1)]
            for slot, (lo, hi) in zip(slots, split_range(self.covered_lo, self.covered_hi, len(slots))):
                self.ranges[slot] = (bank, lo, hi)
        self.cores_per_bank = [self.core_bank.count(bank) for bank in range(BANKS)]
        self.failures = {}
        self.totals = defaultdict(int)
        covered = self.covered_hi - self.covered_lo
        logger.info(
            f"chip {chip}: covering [{self.covered_lo:#x}, {self.covered_hi:#x}) of every bank, {covered / 2**20:.2f} "
            f"of {DRAM_TOP / 2**20:.0f} MiB ({100 * covered / DRAM_TOP:.5f}%, {DRAM_TOP - covered} B uncovered); "
            f"cores per bank {self.cores_per_bank}; "
            f"other DRAM allocations above the range {[(hex(addr), size) for addr, size in self.above]}"
        )

    def launch(self, mode, ranges, seed, prev_seed, reps, pass_index, inject=(NONE, 0, 0), pokes=None):
        cfg = self.cfg

        def args(slot):
            bank, lo, hi = ranges[slot]
            slot_args = [
                bank,
                lo,
                hi,
                self.tt_out.buffer_address(),
                slot,
                seed,
                prev_seed,
                reps,
                pass_index,
                *(inject if bank == SELF_TEST_BANK else (NONE, 0, 0)),
                cfg["decode_max"],
                self.covered_lo,
                self.covered_hi,
            ]
            if mode == MODE_POKE:
                entries = (pokes or {}).get(slot, [])
                assert len(entries) <= MAX_POKES, f"slot {slot}: {len(entries)} pokes, at most {MAX_POKES}"
                # A fixed argument count keeps every launch on the program cached at warm-up.
                flat = [value for entry in entries for value in entry]
                slot_args += [len(entries)] + flat + [0] * (POKE_FIELDS * MAX_POKES - len(flat))
            return slot_args

        kernels = []
        for risc, (processor, noc) in enumerate(
            ((ttnn.DataMovementProcessor.RISCV_1, ttnn.NOC.NOC_0), (ttnn.DataMovementProcessor.RISCV_0, ttnn.NOC.NOC_1))
        ):
            runtime_args = ttnn.RuntimeArgs()
            for index, (x, y) in enumerate(self.cores):
                runtime_args[x][y] = args(2 * index + risc)
            kernels.append(
                ttnn.KernelDescriptor(
                    kernel_source=DM_KERNEL,
                    source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
                    core_ranges=self.grid,
                    compile_time_args=[CB_SCRATCH[risc], cfg["chunk"], mode] + self.out_args,
                    defines=DM_DEFINES,
                    runtime_args=runtime_args,
                    config=ttnn.DataMovementConfigDescriptor(processor=processor, noc=noc),
                )
            )
        ttnn.generic_op([self.dram, self.tt_out], ttnn.ProgramDescriptor(kernels=kernels, semaphores=[], cbs=self.cbs))

    def launch_pokes(self, entries, seed):
        """Run the poke entries, each on the first slice of its own bank."""
        first_slot = {}
        for slot, (bank, _, _) in enumerate(self.ranges):
            first_slot.setdefault(bank, slot)
        by_slot = defaultdict(list)
        for entry in entries:
            by_slot[first_slot[entry[1]]].append(entry)
        self.launch(MODE_POKE, self.ranges, seed, NONE, 1, POKE_PASS, pokes=by_slot)

    def check_coverage(self, out, reps, fill):
        for slot in range(self.slots):
            bank, lo, hi = self.ranges[slot]
            row = out[slot]
            assert (int(row[H_BANK]), int(row[H_LO]), int(row[H_HI])) == (bank, lo, hi), f"slot {slot} misreported"
            words = int(row[H_WORDS_LO]) | (int(row[H_WORDS_HI]) << 32)
            want = (hi - lo) // 4 * (1 if fill else reps)
            assert words == want, f"chip {self.chip} slot {slot}: {words} words touched, expected {want}"

    def record(self, out, label, pass_index, pass_start_s):
        totals = self.totals
        totals[f"{label}_passes"] = max(totals[f"{label}_passes"], pass_index + 1)
        for key, column in (
            ("bad_words", H_BAD_WORDS),
            ("bad_pages", H_BAD_PAGES),
            ("stale", H_STALE),
            ("other", H_OTHER),
            ("undecoded", H_UNDECODED),
            ("alias_outside", H_OUTSIDE),
            ("bits_lost", H_BITS_LOST),
            ("bits_gained", H_BITS_GAINED),
        ):
            totals[f"{label}_{key}"] += int(out[:, column].sum())
        for slot in torch.nonzero(out[:, H_BAD_WORDS]).flatten().tolist():
            row = out[slot]
            bank = int(row[H_BANK])
            aliases = {bit: int(count) for bit, count in enumerate(row[H_ALIAS : H_ALIAS + 32].tolist()) if count}
            for alias_bit, count in aliases.items():
                totals[f"{label}_alias_bit{alias_bit}"] += count
            for entry in range(min(int(row[H_LOGGED]), LOG_CAP)):
                addr, got, exp, tag = (int(word) for word in row[LOG_OFF + 4 * entry : LOG_OFF + 4 * entry + 4])
                key = (label, bank, addr)
                hit = self.failures.get(key)
                if hit is None:
                    hit = self.failures[key] = {
                        "pattern": label,
                        "chip": self.chip,
                        "bank": bank,
                        "addr": hex(addr),
                        "byte_in_64B_burst": addr & 63,
                        "expected": hex(exp),
                        "reads": {},
                        "first_pass": pass_index,
                        "first_rep": tag & 0xFFFF,
                        # Seconds from the start of the write to the start of that read pass.
                        "first_pass_start_s": round(pass_start_s, 3),
                        "classified_as": verdict_name(tag >> 16),
                        "bits_lost": hex(exp & ~got & MASK32),
                        "bits_gained": hex(got & ~exp & MASK32),
                    }
                hit["reads"][hex(got)] = hit["reads"].get(hex(got), 0) + 1
            if int(row[H_LOGGED]) > LOG_CAP:
                totals[f"{label}_unlogged_words"] += int(row[H_LOGGED]) - LOG_CAP
            logger.error(
                f"chip {self.chip} {label} pass {pass_index}: bank {bank} slot {slot} "
                f"bad words {int(row[H_BAD_WORDS])} stale {int(row[H_STALE])} other {int(row[H_OTHER])} "
                f"lanes {[int(count) for count in row[H_LANES:H_LANES + 4]]} alias bits {aliases}"
            )

    def self_test_ranges(self):
        """One page of the self-test bank, 64 MiB into the covered range, whose bit-26 alias source is covered too."""
        target = self.covered_lo + -(-(1 << 26) // PAGE_BYTES) * PAGE_BYTES
        source = target ^ (1 << 26)
        assert target + PAGE_BYTES <= self.covered_hi, f"self-test page {target:#x} is not covered"
        assert self.covered_lo <= source and source + PAGE_BYTES <= self.covered_hi, f"self-test source {source:#x}"
        slot = next(index for index, (bank, _, _) in enumerate(self.ranges) if bank == SELF_TEST_BANK)
        ranges = [(bank, lo, lo) for bank, lo, _ in self.ranges]
        ranges[slot] = (SELF_TEST_BANK, target, target + PAGE_BYTES)
        return ranges, target

    def verify_self_test(self, out, ranges, target):
        bad = out[:, H_BAD_WORDS].nonzero().flatten().tolist()
        assert len(bad) == 1 and ranges[bad[0]][0] == SELF_TEST_BANK, f"chip {self.chip} self-test: bad slots {bad}"
        row = out[bad[0]]
        words = int(row[H_BAD_WORDS])
        assert int(row[H_BAD_PAGES]) == 1 and words % 16 == 0 and words >= PAGE_WORDS // 2, f"self-test words {words}"
        assert [int(count) for count in row[H_LANES : H_LANES + 4]] == [0, 0, 0, words], "self-test lanes"
        aliases = {bit: int(count) for bit, count in enumerate(row[H_ALIAS : H_ALIAS + 32].tolist()) if count}
        assert aliases == {26: 1} and int(row[H_STALE]) == int(row[H_OTHER]) == 0, f"self-test aliases {aliases}"
        first, verdict = int(row[LOG_OFF]), verdict_name(int(row[LOG_OFF + 3]) >> 16)
        assert target <= first < target + PAGE_BYTES, f"self-test logged {first:#x}, planted {target:#x}"
        assert verdict == "alias_bit26", f"self-test words logged as {verdict}"

    def result(self):
        return {
            "chip": self.chip,
            "covered": [hex(self.covered_lo), hex(self.covered_hi)],
            "covered_mib_per_bank": (self.covered_hi - self.covered_lo) / 2**20,
            "covered_fraction_of_bank": (self.covered_hi - self.covered_lo) / DRAM_TOP,
            "uncovered": {
                "barrier": [0, hex(self.covered_lo)],
                "above": [hex(self.covered_hi), hex(DRAM_TOP)],
                "allocations_above": [[hex(addr), size] for addr, size in self.above],
            },
            "dram_buffers_at_open": [[hex(addr), size] for addr, size in self.buffers_at_open],
            "cores_per_bank": self.cores_per_bank,
            "totals": dict(self.totals),
            "failures": list(self.failures.values()),
        }


def run_self_test(runs, seed):
    """Plant a lane-3 alias of address bit 26 in one bank-3 page per chip; the checker must report exactly that."""
    plans = [run.self_test_ranges() for run in runs]
    for run, (ranges, target) in zip(runs, plans):
        run.launch(MODE_FILL, ranges, seed, NONE, 1, FILL_PASS, (target, 26, 0xFF000000))
    for run in runs:
        read_out(run.tt_out, run.slots, seed, FILL_PASS)
    for run, (ranges, _) in zip(runs, plans):
        run.launch(MODE_CHECK, ranges, seed, NONE, 1, 0)
    for run, (ranges, target) in zip(runs, plans):
        run.verify_self_test(read_out(run.tt_out, run.slots, seed, 0), ranges, target)
    logger.info(
        f"self-test: planted bank {SELF_TEST_BANK} lane 3 alias of bit 26 reported exactly on {len(runs)} chips"
    )


def run_pattern(runs, label, seed, prev_seed, cfg, timings, pokes=None):
    """Write every covered address once, then read and compare it after each gap.

    pokes[k], when given, maps each chip to the entries its poke program changes before read k, once the sweep before
    it is done.
    """
    fill_start = time.monotonic()
    for run in runs:
        run.launch(MODE_FILL, run.ranges, seed, prev_seed, 1, FILL_PASS)
    for run in runs:
        run.check_coverage(read_out(run.tt_out, run.slots, seed, FILL_PASS), 1, True)
    fill_end = time.monotonic()
    timings[f"{label}_fill_s"] = fill_end - fill_start
    gaps = cfg["gaps_s"]
    prev_start, prev_end = fill_start, fill_end
    pass_index, pass_seconds, waits = 0, [], []
    while pass_index < len(gaps) or prev_end - fill_start < cfg["seconds"]:
        if pokes and pass_index < len(pokes) and pokes[pass_index]:
            for run in runs:
                run.launch_pokes(pokes[pass_index][run.chip], seed)
            for run in runs:
                read_out(run.tt_out, run.slots, seed, POKE_PASS)
        # Gaps run start to start: the sweeps go through each bank in the same order, so an address waits the gap.
        target = prev_start + gaps[min(pass_index, len(gaps) - 1)]
        now = time.monotonic()
        if target > now:
            time.sleep(target - now)
        start = time.monotonic()
        for run in runs:
            run.launch(MODE_CHECK, run.ranges, seed, prev_seed, cfg["reps"], pass_index)
        outs = [read_out(run.tt_out, run.slots, seed, pass_index) for run in runs]
        end = time.monotonic()
        pass_seconds.append(end - start)
        # The first address of a range waited start - prev_start, the last end - prev_end.
        waits.append((round(start - prev_start, 3), round(end - prev_end, 3)))
        for run, out in zip(runs, outs):
            run.check_coverage(out, cfg["reps"], False)
            run.record(out, label, pass_index, start - fill_start)
        prev_start, prev_end = start, end
        pass_index += 1
        if pass_index >= cfg["max_passes"]:
            break
    timings[f"{label}_passes"] = pass_index
    timings[f"{label}_waits_s"] = waits
    timings[f"{label}_pass_s_mean"] = sum(pass_seconds) / len(pass_seconds)
    timings[f"{label}_pass_s_max"] = max(pass_seconds)
    covered = runs[0].covered_hi - runs[0].covered_lo
    fill_s, pass_s = timings[f"{label}_fill_s"], timings[f"{label}_pass_s_mean"]
    logger.info(
        f"{label}: fill {fill_s:.2f}s ({BANKS * covered / fill_s / 1e9:.1f} GB/s per chip); {pass_index} reads of "
        f"every address after waits of {', '.join(f'{low:.2f}-{high:.2f}s' for low, high in waits)}; a read pass "
        f"takes {pass_s:.2f}s ({BANKS * covered * cfg['reps'] / pass_s / 1e9:.1f} GB/s read and compared per chip); "
        f"bad words {sum(run.totals[f'{label}_bad_words'] for run in runs)}"
    )


def write_results(cfg, timings, runs, name, started, completed):
    """Write the JSON report, also for a run an exception stopped (completed false), so the failures it saw are kept."""
    timings["total_s"] = time.monotonic() - started
    out_dir = Path(os.environ.get("RETENTION_OUT_DIR", REPO / "generated" / "dram_retention"))
    out_dir.mkdir(parents=True, exist_ok=True)
    results = [run.result() for run in runs]
    # The host name keeps the reports of a multi-host job apart on a shared file system.
    parts = [name, socket.gethostname(), os.environ.get("SLURM_JOB_ID"), time.strftime("%Y%m%d_%H%M%S")]
    out_path = out_dir / f"{'_'.join(part for part in parts if part)}.json"
    report = {"completed": completed, "config": cfg, "timings": timings, "results": results}
    out_path.write_text(json.dumps(report, indent=1))
    logger.info(f"total {timings['total_s']:.1f}s; results: {out_path}")
    return results


def test_gddr_retention():
    started = time.monotonic()
    cfg = read_config()
    chips = select_chips()
    logger.info(f"chips {chips}, config {cfg}")
    timings, runs, completed = {}, [], False
    devices = ttnn.CreateDevices(chips)
    try:
        for chip in chips:
            runs.append(ChipRun(chip, devices[chip], cfg))
        timings["setup_s"] = time.monotonic() - started
        seed = cfg["seed0"]
        if cfg["self_test"]:
            run_self_test(runs, seed)
            seed += 2
        # The self-test wrote one page, so what the first pattern overwrites is unknown: no stale classification.
        prev = NONE
        for _ in range(cfg["patterns"]):
            run_pattern(runs, f"seed{seed:#x}", seed, prev, cfg, timings)
            prev, seed = seed, seed + 1
        for run in runs:
            ttnn.deallocate(run.dram)
        completed = True
    finally:
        # The report goes first, so it is on disk even if closing the chips fails.
        try:
            results = write_results(cfg, timings, runs, "gddr_retention", started, completed)
        finally:
            ttnn.CloseDevices(devices)
    failed = [
        (result["chip"], failure["bank"], failure["addr"], failure["pattern"])
        for result in results
        for failure in result["failures"]
    ]
    assert not failed, f"{len(failed)} failing words, first {failed[:10]}"


# Negative test. A word flip XORs `mask` into one word. DIIM-751: bank 7 of a Galaxy chip lost bits 5 and 6 of the byte
# at 0x1400A2D0 (byte 0x10 of its 64 B burst), always 1 to 0, while it was being re-read.
DIIM_751_ADDR, DIIM_751_MASK = 0x1400A2D0, 0x00000060
ALIAS_TEST_BIT = 20
# The same schedule as test_gddr_retention's default. The plan below needs at least two reads.
NEGATIVE_GAPS_S = DEFAULT_GAPS_S


def negative_plan(run, seed, prev_seed):
    """The pokes before each read on one chip, and the failures they must produce.

    Per bank, before the first read: word flips at the first and last tested word, at the start of the bank's second
    slice and at the DIIM-751 address, a random line put back to prev_seed's data (stale), a random line overwritten
    with the data that belongs at its address with bit ALIAS_TEST_BIT flipped (an alias), and the line 2^k below the end
    of the swept range overwritten with the first line above it (a kernel binary), an alias whose source the sweep
    doesn't cover. Before the second read: a flip at a random word. prev_seed is the complement of seed and no flip mask
    has a 0xFF byte, so no flip reads as stale.

    Expected failures map (bank, address) to (expected, read, first read, reads that see it, classification or None);
    the read is None where the host can't know it.
    """
    lo, hi = run.covered_lo, run.covered_hi
    assert lo <= DIIM_751_ADDR and 1 << 31 < hi < DRAM_TOP, f"chip {run.chip}: swept range {lo:#x}-{hi:#x} too small"
    reads = len(NEGATIVE_GAPS_S)
    rng = random.Random(seed * 1000003 + run.chip)
    pokes, expect = [[] for _ in range(reads)], {}
    for bank in range(BANKS):
        starts = sorted(slice_lo for owner, slice_lo, _ in run.ranges if owner == bank)
        flips = [(0, lo, 0x00000001), (0, hi - 4, 0x80000000), (0, starts[1], 0x00005A00)]
        flips.append((0, DIIM_751_ADDR, DIIM_751_MASK))
        # One change a page, so the page's classification belongs to that change.
        used_pages = {(addr - lo) // PAGE_BYTES for _, addr, _ in flips}

        def pick_line(low, high, clear=0):
            while True:
                line = (low + rng.randrange((high - low) // 64) * 64) & ~clear
                if (line - lo) // PAGE_BYTES not in used_pages:
                    used_pages.add((line - lo) // PAGE_BYTES)
                    return line

        flips.append((1, pick_line(lo, hi) + 4 * rng.randrange(16), 0x00A50000))
        for read_index, addr, mask in flips:
            pokes[read_index].append((POKE_FLIP, bank, addr, mask, 0))
            expected = pattern_word(addr, bank, seed)
            expect[(bank, addr)] = (expected, expected ^ mask, read_index, reads - read_index, None)
        stale_line = pick_line(lo, 1 << 30)
        alias_line = pick_line(1 << 30, 1 << 31, clear=1 << ALIAS_TEST_BIT)
        pokes[0].append((POKE_LINE, bank, stale_line, 0, prev_seed))
        pokes[0].append((POKE_LINE, bank, alias_line, 1 << ALIAS_TEST_BIT, seed))
        for word in range(16):
            addr = stale_line + 4 * word
            expect[(bank, addr)] = (
                pattern_word(addr, bank, seed),
                pattern_word(addr, bank, prev_seed),
                0,
                reads,
                "stale",
            )
            addr = alias_line + 4 * word
            expected, read = pattern_word(addr, bank, seed), pattern_word(addr ^ (1 << ALIAS_TEST_BIT), bank, seed)
            if read != expected:
                expect[(bank, addr)] = (expected, read, 0, reads, f"alias_bit{ALIAS_TEST_BIT}")
        # The line 2^k below hi, for a bit k >= 11 set in hi: its bit-k alias is hi, the first line above the sweep.
        copy_bit = next(
            bit for bit in range(11, 32) if hi >> bit & 1 and (hi - (1 << bit) - lo) // PAGE_BYTES not in used_pages
        )
        copy_line = hi - (1 << copy_bit)
        used_pages.add((copy_line - lo) // PAGE_BYTES)
        pokes[0].append((POKE_COPY, bank, copy_line, 1 << copy_bit, 0))
        for word in range(16):
            addr = copy_line + 4 * word
            expect[(bank, addr)] = (pattern_word(addr, bank, seed), None, 0, reads, f"alias_bit{copy_bit}")
    return pokes, expect


def check_reported(run, expect):
    reported = {(failure["bank"], int(failure["addr"], 16)): failure for failure in run.failures.values()}
    missing = sorted(set(expect) - set(reported))
    extra = sorted(set(reported) - set(expect))
    assert not missing and not extra, (
        f"chip {run.chip}: missing {[(bank, hex(addr)) for bank, addr in missing[:8]]}, "
        f"unexpected {[(bank, hex(addr)) for bank, addr in extra[:8]]}"
    )
    for (bank, addr), (expected, read, first_pass, reads, verdict) in expect.items():
        failure = reported[(bank, addr)]
        where = f"chip {run.chip} bank {bank} {addr:#x}"
        # The kernel's expected value checks the host's copy of the hash as well as the device's report.
        assert failure["expected"] == hex(expected), f"{where}: expected {failure['expected']}"
        assert failure["first_pass"] == first_pass, f"{where}: first read {failure['first_pass']}"
        if read is None:
            # Copied from outside the sweep: one value, seen by every read.
            assert (
                len(failure["reads"]) == 1 and sum(failure["reads"].values()) == reads
            ), f"{where}: {failure['reads']}"
        else:
            assert failure["reads"] == {hex(read): reads}, f"{where}: reads {failure['reads']}"
            assert failure["bits_lost"] == hex(expected & ~read & MASK32), f"{where}: bits lost {failure['bits_lost']}"
            assert failure["bits_gained"] == hex(read & ~expected & MASK32), f"{where}: gained {failure['bits_gained']}"
        assert verdict in (None, failure["classified_as"]), f"{where}: classified as {failure['classified_as']}"


def test_gddr_retention_detects_changes():
    started = time.monotonic()
    cfg = read_config()
    chips = select_chips()
    logger.info(f"negative test on chips {chips}, seed {cfg['seed0']:#x}: the miscompares logged below are planted")
    timings, runs, completed = {}, [], False
    devices = ttnn.CreateDevices(chips)
    try:
        for chip in chips:
            runs.append(ChipRun(chip, devices[chip], cfg, poke=True))
        seed = cfg["seed0"]
        run_self_test(runs, seed)
        seed += 2
        # The complement as the previous seed runs the stale check the way the second of two patterns runs it.
        prev_seed, label = seed + 1, f"seed{seed:#x}"
        plans = {run.chip: negative_plan(run, seed, prev_seed) for run in runs}
        pokes = [
            {chip: plan[0][read_index] for chip, plan in plans.items()} for read_index in range(len(NEGATIVE_GAPS_S))
        ]
        negative_cfg = {**cfg, "gaps_s": NEGATIVE_GAPS_S, "seconds": 0, "reps": 1, "max_passes": len(NEGATIVE_GAPS_S)}
        run_pattern(runs, label, seed, prev_seed, negative_cfg, timings, pokes)
        for run in runs:
            ttnn.deallocate(run.dram)
        completed = True
    finally:
        try:
            write_results(cfg, timings, runs, "gddr_retention_negative", started, completed)
        finally:
            ttnn.CloseDevices(devices)
    for run in runs:
        expect = plans[run.chip][1]
        check_reported(run, expect)
        # No flip reads as stale, so the stale pages are the stale lines, once a read.
        stale = run.totals[f"{label}_stale"]
        assert stale == len(NEGATIVE_GAPS_S) * BANKS, f"chip {run.chip}: {stale} stale pages"
        logger.info(f"chip {run.chip}: every change was reported exactly, {len(expect)} words")
