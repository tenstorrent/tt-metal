# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

"""Targeted GDDR6 ACTIVATE-word stress: reads chosen per-bank addresses so that every read forces a row activation
whose command/address word has a chosen shape, and checks each read for silent wrong-address data.

The whole-DRAM sweep issues mostly row hits, so it sends few ACT words and none of a chosen shape. Here each RISC
reads a short list of (controller, per-bank address) entries in a loop. With a barrier after every read and two rows
of one GDDR bank in the list (ping-pong), the controller must close and reopen a row on each read, so the target row's
ACT word is sent at the read rate. Without the barrier (hammer, storm) many reads are in flight at once.

Per-bank address decode: col = addr[12:6], GDDR bank = addr[16:13], row = addr[31:17]. The ACT falling edge carries
CA9..CA0 = R13..R4, so row bit R(i+4) drives pin CA i. The rising edge carries CA9 = L (opcode), CA8 = R14,
CA7..CA4 = BA3..BA0, CA3..CA0 = R3..R0 (GDDR6 command truth table), assuming GDDR bank = BA. A read is checked word by word against a per-address hash; a
miscompared read is classified as stale (previous launch's data), alias of bit k (every wrong byte equals the data at
A ^ (1 << k), read back from DRAM), or other.
"""

import importlib.util
import json
import os
import random
import re
import socket
import subprocess
import time
from pathlib import Path

import pytest
import torch
from loguru import logger

import ttnn

_spec = importlib.util.spec_from_file_location("dram_sweep_stress", Path(__file__).with_name("test_dram_sweep_stress.py"))
sweep = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(sweep)

CTRLS = 8
HDR_WORDS = 96
LANE_OFF = 16
ALIAS_OFF = LANE_OFF + 4 * CTRLS
MAX_ENTRIES = 256
ROW_WORDS = HDR_WORDS + MAX_ENTRIES
ENTRY0 = 13
MAGIC = 0xACAC
NONE = 0xFFFFFFFF
FILL, READ, WRITE = 0, 1, 2
ROW_SHIFT, GBANK_SHIFT, COL_SHIFT = 17, 13, 6
L1_BUDGET = 64 * 1024

REPO = Path(__file__).resolve().parents[5]
OUT_DIR = Path(os.environ.get("STRESS_OUT_DIR", REPO / "repro_logs" / "act"))

DM_KERNEL = f"{sweep.KERNEL_DIR}/act_dm.cpp"
DM_DEFINES = sweep.MM_DEFINES + [
    ("ALIAS_OFF", str(ALIAS_OFF)),
    ("COL_SHIFT", str(COL_SHIFT)),
    ("ENTRY0", str(ENTRY0)),
    ("HDR_WORDS", str(HDR_WORDS)),
    ("LANE_OFF", str(LANE_OFF)),
    ("MODE_FILL", str(FILL)),
    ("MODE_WRITE", str(WRITE)),
    ("NO_ADDR", f"{NONE:#x}u"),
    ("OUT_MAGIC", str(MAGIC)),
    ("ROW_WORDS", str(ROW_WORDS)),
]


def addr_of(row, gbank, col):
    return (row << ROW_SHIFT) | (gbank << GBANK_SHIFT) | (col << COL_SHIFT)


def decode(a):
    return a >> ROW_SHIFT, (a >> GBANK_SHIFT) & 15, (a >> COL_SHIFT) & 127


def falling(row):
    """CA9..CA0 on the ACT falling edge, as a 10-bit word with bit i = pin CA i = R(i+4)."""
    return (row >> 4) & 0x3FF


def rising(row, gb):
    """CA9..CA0 on the ACT rising edge; CA9 is the opcode bit, always L."""
    return (row & 15) | (gb << 4) | ((row >> 14) & 1) << 8


def act_addr(rise, fall, col):
    """The address whose ACT drives word rise on the rising edge and word fall on the falling edge."""
    assert not rise >> 9, "CA9 is the ACT opcode bit on the rising edge, always L"
    return addr_of((rise & 15) | (fall << 4) | ((rise >> 8) & 1) << 14, (rise >> 4) & 15, col)


def ca_full_targets(t0, col0, space, ca_pins):
    """The original hit t0 (bank 1, BA1=0), the same row in bank 3 (BA1=1), whose rising CA5 matches the falling
    R9, then every ACT pin toggling in both directions against opposite-toggling neighbours (CA9 only 0 -> 1: its
    rising edge is the opcode). A target is None when it has no placement in the buffer."""
    targets = [("orig", t0), ("orig_bank3", t0 | 2 << GBANK_SHIFT)]
    for i in ca_pins:
        for up in (True, False) if i < 9 else (True,):
            targets.append((f"ca{i}_{'up' if up else 'dn'}", toggle_target(i, up, col0, space)[0]))
    return targets


# Many CA pins switching in one ACT, as (rising, falling) CA9..CA0 words. Every word keeps at least five ones, so
# CABI leaves it as is and the wire carries the same pattern with CABI on or off (alt_ev_up holds CA0 high for that).
# The vic_* and sw5 words copy the shape of the orig_bank3 hit: a victim pin holds its level while its neighbours
# switch to the opposite one. sw5 is that shape on CA5 alone; with CABI on both of its words invert, so on the wire
# the victim holds L and the neighbours rise, the mirror of the same shape.
MULTI_PIN = {
    "alt_ev_dn": (0x155, 0x2AA),  # every pin toggles: even pins 1 -> 0, odd pins 0 -> 1
    "alt_ev_up": (0x0AB, 0x155),  # even pins 0 -> 1, odd 1 -> 0, CA0 holds H, CA9 holds L
    "vic_ev_hi": (0x1FF, 0x155),  # even pins hold H, odd pins 1 -> 0 on both sides of each, CA9 holds L
    "vic_od_hi": (0x1FF, 0x2AA),  # CA1/3/5/7 hold H, even pins 1 -> 0 on both sides of each, CA9 0 -> 1
    "sw5": (0x070, 0x020),  # CA5 holds H, CA4 and CA6 1 -> 0, the rest hold L
}


def multi_pin_targets(col, space):
    targets = []
    for name, (rise, fall) in MULTI_PIN.items():
        a = act_addr(rise, fall, col)
        row, gb, _ = decode(a)
        assert rising(row, gb) == rise and falling(row) == fall
        targets.append((name, a if space.ok(a) else None))
    return targets


def toggle_target(i, up, col, space):
    """Pin CA i toggles between the ACT edges (up: 0 -> 1, else 1 -> 0) and its neighbours toggle the other way;
    every other pin is 0 on both edges. Rows below the buffer get CA8 high on the rising edge (R14), the pin
    farthest from CA0/CA1, the only targets that need it. With CABI both one/two-hot edges invert together, so the
    up/down pair still covers both directions."""
    pin = 1 << i
    nb = sum(1 << j for j in (i - 1, i + 1) if 0 <= j < 10)
    rise, fall = (nb, pin) if up else (pin, nb)
    rise &= 0x1FF
    for extra in (0, 1 << 8):
        if extra and (rise | fall | nb) & extra:
            break
        a = act_addr(rise | extra, fall, col)
        if space.ok(a):
            return a, rise | extra, fall
    return None, rise, fall


def isolated_pins(f):
    bits = [(f >> i) & 1 for i in range(10)]
    return [i for i in range(10) if all(bits[j] != bits[i] for j in (i - 1, i + 1) if 0 <= j < 10)]


def shape(row):
    f = falling(row)
    cabi = f ^ 0x3FF if 10 - bin(f).count("1") > 5 else f
    return f"CA9..0 {f:010b} (CABI {cabi:010b}) isolated {isolated_pins(f)}"


class Space:
    """The writable per-bank range: the sweep buffer, so stray writes never touch another allocation."""

    def __init__(self, lo, hi, rb):
        self.lo, self.hi, self.rb = lo, hi, rb

    def ok(self, a):
        return self.lo <= a and a + self.rb <= self.hi

    def fit(self, a):
        """Same falling-edge shape, moved into the buffer with R14 (rising edge) and R0..R3 if needed."""
        row, gb, col = decode(a)
        for hi_bits in (0, 1 << 14):
            for lo_bits in range(16):
                b = addr_of(row ^ hi_bits ^ lo_bits, gb, col)
                if self.ok(b):
                    return b
        return None

    def partner(self, t, hot, s):
        """A second row in t's GDDR bank, so each read of t needs its own ACT; s spreads RISCs over rows."""
        row, gb, col = decode(t)
        for flip in ((1 << hot) if hot is not None else 0, 0):
            p = addr_of(row ^ flip ^ (1 + s % 7), gb, col)
            if self.ok(p):
                return p
        raise RuntimeError(f"no partner row for {t:#x}")


def hot_pin(row):
    iso = isolated_pins(falling(row))
    return 4 + (5 if 5 in iso else iso[0]) if iso else 9


def pingpong(t, ctrls, space, slots, n=32):
    hot = hot_pin(decode(t)[0])
    specs = []
    for s in range(slots):
        c = ctrls[s % len(ctrls)]
        specs.append((READ, 1, [(t, c), (space.partner(t, hot, s), c)] * (n // 2)))
    return specs


def hammer(t, ctrls, slots, n=32):
    return [(READ, 0, [(t, ctrls[s % len(ctrls)])] * n) for s in range(slots)]


def storm(row, col, ctrls, space, slots):
    specs = []
    for s in range(slots):
        c = ctrls[s % len(ctrls)]
        ents = []
        for gb in range(16):
            t = space.fit(addr_of(row, gb, col))
            ents += [(t, c), (space.partner(t, 5 + 4, s), c)]
        specs.append((READ, 0, ents))
    return specs


def sdpa(t, ctrls, space, slots, ratio, all_banks=False, one_bank=False):
    """Shared K-block streaming at t, private ping-pong rows in t's GDDR bank, and writers on other rows of it.

    all_banks spreads the shared stream over t's row in all 16 GDDR banks (16 reads each), so one run
    compares banks under the same load. one_bank wraps the stream around t's 8 KB of its row instead of running
    on into the next banks, so every stream ACT drives t's own rising-edge word.
    """
    row0, gb, _ = decode(t)
    low = (space.lo >> ROW_SHIFT) + 1
    # Q and W rows (low .. low+223) must not include t's row, or its reads stop needing their own ACT.
    if low <= row0 < low + 224:
        low = row0 + 1
    groups = [g for g, r in zip("SQW", ratio) for _ in range(r)]
    specs = []
    for s in range(slots):
        c = ctrls[s % len(ctrls)]
        g = groups[s % len(groups)]
        if g == "S" and all_banks:
            per = min(16, (128 << COL_SHIFT) // space.rb)
            ents = [(addr_of(row0, b, 0) + space.rb * i, c) for b in range(16) for i in range(per)]
            specs.append((READ, 0, ents))
        elif g == "S" and one_bank:
            base, span = addr_of(row0, gb, 0), 128 << COL_SHIFT
            off0 = t - base
            ents = [(base + (off0 + space.rb * i) % span, c) for i in range(min(MAX_ENTRIES, 39168 // space.rb))]
            specs.append((READ, 0, ents))
        elif g == "S":
            ents = [(t + space.rb * i, c) for i in range(min(MAX_ENTRIES, 39168 // space.rb))]
            specs.append((READ, 0, ents))
        elif g == "Q":
            q = addr_of(low + 2 * (s % 64), gb, 0)
            specs.append((READ, 1, [(q, c), (addr_of(low + 2 * (s % 64) + 1, gb, 0), c)] * 16))
        else:
            w = addr_of(low + 160 + (s % 64), gb, 64)
            specs.append((WRITE, 1, [(w + space.rb * i, c) for i in range(16)]))
    for _, _, ents in specs:
        assert all(space.ok(a) for a, _ in ents), "sdpa entries outside the buffer"
    return specs


def with_writes(specs, every):
    """Every `every`-th slot rewrites its first 16 entries, one barrier each, instead of reading them. The data written
    is the fill data, so readers of the same rows stay correct, while the bus sees write ACTs and read/write
    turnarounds on exactly the rows the pattern reads."""
    if not every:
        return specs
    return [(WRITE, 1, ents[:16]) if s % every == every - 1 else (m, b, ents) for s, (m, b, ents) in enumerate(specs)]


FORMS = ("pingpong", "hammer", "storm", "sdpa", "sdpa1", "sdpa16")


def form_specs(form, t, ctrls, space, slots, ratio):
    """Specs for one load shape at target t. storm reads t's row in all 16 GDDR banks, so only t's falling-edge
    word is fixed there; the rising edge varies with the bank."""
    if form == "pingpong":
        return pingpong(t, ctrls, space, slots)
    if form == "hammer":
        return hammer(t, ctrls, slots)
    if form == "storm":
        row, _, col = decode(t)
        return storm(row, col, ctrls, space, slots)
    if form in ("sdpa", "sdpa1", "sdpa16"):
        return sdpa(t, ctrls, space, slots, ratio, all_banks=form == "sdpa16", one_bank=form == "sdpa1")
    raise ValueError(f"unknown form {form}, expected one of {FORMS}")


MISDIRECT_BIT = 26


def wmis_target(t, space):
    """t, or t moved up by R12 then R14, so that t and t ^ 64 MB both lie in the buffer. A write misdirected by the
    CA5 alias lands 64 MB below t, and that cell must be ours to write and to check. R12 adds falling-edge CA8 and
    R14 rising-edge CA8, both away from CA5, so CA5 stays isolated on the falling edge."""
    if t is None or not t >> MISDIRECT_BIT & 1:
        return None
    for extra in (0, 1 << 29, 1 << 31):
        m = t | extra
        if space.ok(m) and space.ok(m ^ 1 << MISDIRECT_BIT):
            return m
    return None


def wmis(t, ctrls, space, slots):
    """Write misdirect: two of three RISCs write t's gold, alternating with a partner row of t's bank so every write
    needs its own ACT. The rest read t and every single-bit partner t ^ (1 << k) inside the buffer. FILL gave each
    partner its own gold, so a write of t's data that lands on a partner reads back as alias bit k there, and stays
    wrong until the next launch refills it."""
    hot = hot_pin(decode(t)[0])
    partners = [t ^ 1 << k for k in range(COL_SHIFT, 32) if space.ok(t ^ 1 << k)]
    assert t ^ 1 << MISDIRECT_BIT in partners, f"{t:#x}: 64 MB partner outside the buffer"
    specs = []
    for s in range(slots):
        c = ctrls[s % len(ctrls)]
        if s % 3 == 2:
            specs.append((READ, 0, [(a, c) for a in [t] + partners]))
        else:
            specs.append((WRITE, 1, [(t, c), (space.partner(t, hot, s), c)] * 8))
    return specs


def random_specs(rng, ctrls, space, slots):
    """One random draw: each RISC gets a random shape (stream, ping-pong, scatter or write burst) at random rows,
    columns and controllers. Half the draws start every slot in one random GDDR bank (long streams run on into the
    next banks), the load that concentrates ACTs the way sdpa does; the other half spread over all 16."""
    lo_row, hi_row = (space.lo >> ROW_SHIFT) + 1, (space.hi >> ROW_SHIFT) - 1
    step = space.rb >> COL_SHIFT
    hot = rng.randrange(16) if rng.random() < 0.5 else None

    def rand_addr():
        gb = hot if hot is not None else rng.randrange(16)
        return addr_of(rng.randrange(lo_row, hi_row), gb, rng.randrange(0, 128, step))

    specs = []
    for s in range(slots):
        c = ctrls[rng.randrange(len(ctrls))]
        kind = rng.randrange(4)
        if kind == 0:
            base = rand_addr()
            spec = (READ, 0, [(base + space.rb * i, c) for i in range(rng.randrange(16, MAX_ENTRIES + 1))])
        elif kind == 1:
            p = rand_addr()
            q = addr_of(rng.randrange(lo_row, hi_row), decode(p)[1], decode(p)[2])
            spec = (READ, 1, [(p, c), (q, c)] * 16)
        elif kind == 2:
            spec = (READ, 0, [(rand_addr(), c) for _ in range(64)])
        else:
            base = rand_addr()
            spec = (WRITE, 1, [(base + space.rb * i, c) for i in range(16)])
        # A stream or burst that runs past the buffer end is drawn again as a scatter.
        if not all(space.ok(a) for a, _ in spec[2]):
            spec = (READ, 0, [(rand_addr(), c) for _ in range(64)])
        specs.append(spec)
    return specs


def ca_shape_targets(gb, col):
    rows = []
    for i in range(10):
        r = 4 + i
        rows.append((f"pin{i}_alone", 1 << r))
        rows.append((f"pin{i}_hole", (0x3FF << 4) & ~(1 << r)))
        if i < 9:
            rows.append((f"pin{i}{i + 1}_pair", (3 << r)))
    return [(name, addr_of(r, gb, col)) for name, r in rows]


# PCI bus high nibble -> UBB tray slot, as in UMD's ClusterDescriptor::get_tray_id.
UBB_TRAY_BUS_IDS = {"ubb_wormhole": (0xC0, 0x80, 0x00, 0x40), "ubb_blackhole": (0x00, 0x40, 0xC0, 0x80)}
UBB_TRAY_BUS_IDS["ubb_blackhole_cf"] = UBB_TRAY_BUS_IDS["ubb_blackhole"]


def parse_fru(text):
    """{FRU name: {field: value}} from `ipmitool fru print` output."""
    frus, cur = {}, None
    for line in text.splitlines():
        key, sep, val = line.partition(":")
        if not sep:
            continue
        key, val = key.strip(), val.strip()
        if key == "FRU Device Description":
            cur = frus.setdefault(val.split(" (")[0], {})
        elif cur is not None:
            cur[key] = val
    return frus


def read_fru():
    """(`ipmitool fru print` text, source). The UBB FRUs need root, so TT_FRU_FILE may name a saved copy."""
    path = os.environ.get("TT_FRU_FILE")
    if path and os.path.exists(path):
        return Path(path).read_text(), path
    try:
        r = subprocess.run(["sudo", "-n", "ipmitool", "fru", "print"], capture_output=True, text=True, timeout=90)
    except (OSError, subprocess.TimeoutExpired) as e:
        return "", f"none ({type(e).__name__})"
    # ipmitool exits non-zero when any one FRU is unreadable, so judge by the output.
    if "FRU Device Description" in r.stdout:
        return r.stdout, "sudo ipmitool fru print"
    return "", f"none ({(r.stderr.strip().splitlines() or ['no output'])[-1]})"


def chip_identities(chips):
    """({device id: physical ASIC}, host line). Never raises: identity is reporting only.

    Device ids are renumbered when links or chips change between boots. The ASIC ID is unique per die and the
    tray serial per UBB tray, so those two name the hardware for syseng.
    """
    import yaml

    host = socket.gethostname()
    try:
        # BaseLoader keeps every value a string; YAML 1.1 would read a BDF like 0000:01:00.0 as a number.
        desc = yaml.load(Path(ttnn.cluster.serialize_cluster_descriptor()).read_text(), Loader=yaml.BaseLoader)
        text, source = read_fru()
        frus = parse_fru(text)
        board_type = {}
        for board in desc.get("boards") or []:
            fields = {k: v for entry in board for k, v in entry.items()}
            for c in fields.get("chips") or []:
                board_type[str(c)] = str(fields.get("board_type", "")).lower()
        ids = {}
        for c in chips:
            k = str(c)
            uid = (desc.get("chip_unique_ids") or {}).get(k)
            bus = (desc.get("chip_to_bus_id") or {}).get(k)
            bus = int(bus, 16) if bus is not None else None
            trays = UBB_TRAY_BUS_IDS.get(board_type.get(k), UBB_TRAY_BUS_IDS["ubb_blackhole"])
            ubb = trays.index(bus & 0xF0) + 1 if bus is not None and (bus & 0xF0) in trays else None
            pos = (desc.get("asic_locations") or {}).get(k)
            ids[c] = {
                "ubb": ubb,
                "asic_pos": int(pos) if pos is not None else (bus & 0x0F if bus is not None else None),
                "tray_serial": frus.get(f"UBB{ubb - 1}_FRU", {}).get("Board Serial") if ubb else None,
                "bdf": (desc.get("chip_pci_bdfs") or {}).get(k),
                "asic_id": f"{int(uid, 0):016X}" if uid is not None else None,
            }
        builtin = frus.get("Builtin FRU Device", {})
        chassis = builtin.get("Chassis Serial") or builtin.get("Product Serial")
        host_line = f"{host} chassis {chassis or 'unknown'} (FRU from {source})"
    except Exception as e:
        logger.warning(f"ASIC identity unavailable, chips reported by device id only: {type(e).__name__}: {e}")
        return {c: {} for c in chips}, host
    return ids, host_line


def ident_str(ident):
    """One line for logs, e.g. UBB4 ASIC8 tray T38S7TBA... 0000:c6:00.0 ASIC ID 09D1..."""
    if not ident:
        return "asic unknown"
    parts = [f"UBB{ident['ubb']} ASIC{ident['asic_pos']}" if ident.get("ubb") else None]
    parts += [f"tray {ident['tray_serial']}" if ident.get("tray_serial") else None, ident.get("bdf")]
    parts += [f"ASIC ID {ident['asic_id']}" if ident.get("asic_id") else None]
    return " ".join(p for p in parts if p) or "asic unknown"


class ActRun:
    def __init__(self, chip, device, cfg):
        gx, gy = cfg["grid"]
        self.chip, self.cfg = chip, cfg
        self.ident = {}
        self.cores = [(x, y) for y in range(gy) for x in range(gx)]
        self.slots = 2 * len(self.cores)
        self.grid = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(gx - 1, gy - 1))])
        self.tt_out = ttnn.from_torch(
            torch.zeros(1, 1, self.slots, ROW_WORDS, dtype=torch.int32),
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        self.buf, rows, self.bank_mb = sweep.allocate_buffer(device, cfg["bank_mb"])
        base = self.buf.buffer_address()
        self.space = Space(base, base + rows // sweep.BANKS * sweep.PAGE_BYTES, cfg["rb"])
        self.blocks = cfg["blocks"]
        logger.info(f"chip {chip}: writable per-bank range {self.space.lo:#x}..{self.space.hi:#x}")

    def program(self, specs, mode_override, seed, iters, compute_blocks, inj):
        rb = self.cfg["rb"]
        nmax = max(len(e) for _, _, e in specs)
        assert nmax <= MAX_ENTRIES and nmax * rb <= L1_BUDGET, (nmax, rb)
        scratch_bytes = ROW_WORDS * 4 + (2 * nmax + 1) * rb
        cbs = [
            ttnn.CBDescriptor(
                total_size=scratch_bytes,
                core_ranges=self.grid,
                format_descriptors=[
                    ttnn.CBFormatDescriptor(buffer_index=i, data_format=ttnn.uint32, page_size=scratch_bytes)
                ],
            )
            for i in sweep.CB_SCRATCH
        ]
        for idx, tiles in ((sweep.CB_IN0, sweep.KT), (sweep.CB_IN1, sweep.KT), (sweep.CB_OUT, 1)):
            cbs.append(
                ttnn.CBDescriptor(
                    total_size=tiles * 2048,
                    core_ranges=self.grid,
                    format_descriptors=[
                        ttnn.CBFormatDescriptor(buffer_index=idx, data_format=ttnn.bfloat16, page_size=2048)
                    ],
                )
            )
        out_args = ttnn.TensorAccessorArgs(self.tt_out).get_compile_time_args()
        kernels = []
        for risc, (processor, noc) in enumerate(
            ((ttnn.DataMovementProcessor.RISCV_1, ttnn.NOC.NOC_0), (ttnn.DataMovementProcessor.RISCV_0, ttnn.NOC.NOC_1))
        ):
            rt = ttnn.RuntimeArgs()
            for i, (x, y) in enumerate(self.cores):
                s = 2 * i + risc
                mode, barrier_each, ents = specs[s]
                inj_e, inj_bit, inj_mask = inj
                # Equal arg counts per core for equal nmax: the program cache keys on CB sizes (nmax) but
                # not on per-core arg counts, and a cache hit copies new args into the cached arg slots.
                rt[x][y] = [
                    self.tt_out.buffer_address(),
                    s,
                    FILL if mode_override == FILL else mode,
                    seed,
                    iters,
                    rb,
                    len(ents),
                    int(compute_blocks > 0),
                    barrier_each,
                    self.cfg["decode_max"],
                    inj_e,
                    inj_bit,
                    inj_mask,
                ] + [a | c for a, c in ents] + [0] * (nmax - len(ents))
            kernels.append(
                ttnn.KernelDescriptor(
                    kernel_source=DM_KERNEL,
                    source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
                    defines=DM_DEFINES,
                    core_ranges=self.grid,
                    compile_time_args=[sweep.CB_SCRATCH[risc], int(risc == 0)] + out_args,
                    runtime_args=rt,
                    config=ttnn.DataMovementConfigDescriptor(processor=processor, noc=noc),
                )
            )
        if compute_blocks > 0:
            rt = ttnn.RuntimeArgs()
            for x, y in self.cores:
                rt[x][y] = [compute_blocks]
            kernels.append(
                ttnn.KernelDescriptor(
                    kernel_source=sweep.COMPUTE_KERNEL,
                    source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
                    defines=sweep.MM_DEFINES,
                    core_ranges=self.grid,
                    compile_time_args=[],
                    runtime_args=rt,
                    config=ttnn.ComputeConfigDescriptor(),
                )
            )
        return ttnn.ProgramDescriptor(kernels=kernels, semaphores=[], cbs=cbs)

    def run(self, specs, mode, seed, iters, compute_blocks=0, inj=(NONE, 0, 0)):
        ttnn.generic_op([self.buf, self.tt_out], self.program(specs, mode, seed, iters, compute_blocks, inj))

    def read(self, seed):
        out = ttnn.to_torch(self.tt_out).reshape(self.slots, ROW_WORDS).to(torch.int64) & 0xFFFFFFFF
        assert (out[:, 15] == MAGIC).all(), "a RISC did not report"
        assert (out[:, 0] == seed).all(), f"stale runtime args: expected seed {seed}, got {out[:, 0].unique().tolist()}"
        return out


def summarize(out, specs):
    per_addr = {}
    for s in range(out.shape[0]):
        _, _, ents = specs[s]
        for e, (a, c) in enumerate(ents):
            v = int(out[s, HDR_WORDS + e])
            if v:
                per_addr[f"{c}:{a:#x}"] = per_addr.get(f"{c}:{a:#x}", 0) + v
    return {
        "bad_reads": int(out[:, 2].sum()),
        "bad_words": int(out[:, 3].sum()),
        "stale": int(out[:, 4].sum()),
        "other": int(out[:, 5].sum()),
        "undecoded": int(out[:, 6].sum()),
        "lanes": {
            c: v for c, v in enumerate(out[:, LANE_OFF:ALIAS_OFF].sum(0).reshape(CTRLS, 4).tolist()) if any(v)
        },
        "alias_bits": {k: int(v) for k, v in enumerate(out[:, ALIAS_OFF : ALIAS_OFF + 32].sum(0).tolist()) if v},
        "bad_by_addr": per_addr,
    }


def first_bad(out, specs):
    bad = [s for s in range(out.shape[0]) if out[s, 2]]
    if not bad:
        return None
    r = out[bad[0]]
    a, c = specs[bad[0]][2][int(r[7])]
    return {
        "slot": bad[0],
        "ctrl": c,
        "addr": hex(a + 4 * int(r[8])),
        "got": hex(int(r[9])),
        "exp": hex(int(r[10])),
        "iter": int(r[11]),
        "slots_bad": len(bad),
    }


def merge(acc, s):
    for k in ("bad_reads", "bad_words", "stale", "other", "undecoded"):
        acc[k] = acc.get(k, 0) + s[k]
    for key in ("alias_bits", "bad_by_addr"):
        d = acc.setdefault(key, {})
        for k, v in s[key].items():
            d[k] = d.get(k, 0) + v
    lanes = acc.setdefault("lanes", {})
    for c, v in s["lanes"].items():
        lanes[c] = [x + y for x, y in zip(lanes.get(c, [0] * 4), v)]


class Driver:
    def __init__(self, runs, cfg):
        self.runs, self.cfg = runs, cfg
        self.seed = cfg["seed0"]
        self.iters = cfg["iters"]
        self.on_launch_s = cfg["launch_s"]
        self.ramp = "v1"
        self.write_every = 0
        self.results = []

    def launch(self, specs_by_chip, arm):
        self.seed += 1
        for r in self.runs:
            r.run(specs_by_chip[r.chip], FILL, self.seed, 1)
        for r in self.runs:
            r.read(self.seed)
        t0 = time.perf_counter()
        for r in self.runs:
            r.run(specs_by_chip[r.chip], READ, self.seed, self.iters, r.blocks if arm == "on" else 0)
        outs = [r.read(self.seed) for r in self.runs]
        dt = time.perf_counter() - t0
        target = self.cfg["launch_s"]
        if arm == "off":
            if dt < target / 2:
                self.iters = min(self.iters * 2, 1 << 30)
            elif dt > target * 2:
                self.iters = max(1, self.iters // 2)
        # On the matmul arm compute should outlast the reads slightly, so the matmul runs for the whole
        # launch. A slow-read pattern leaves done == iters with compute already long, so growth is also
        # bounded by wall time; unbounded growth trips the dispatch timeout. A long on_launch_s stretches
        # compute past the reads on purpose, since the readers keep looping until compute finishes.
        t_on = self.on_launch_s
        for r, out in zip(self.runs, outs):
            done = int(out[:, 1].min())
            if arm != "on":
                continue
            # Compute-bound launches scale linearly with blocks, so one proportional step lands near t_on instead
            # of doubling past it. A read-bound launch under-reports the step, which only slows the approach.
            if self.ramp == "proportional":
                r.blocks = min(max(1, int(r.blocks * min(8.0, max(0.25, t_on / dt)))), 1 << 30)
                continue
            # Shrink below the 300 s dispatch timeout, and never double a launch that already overran.
            if dt > t_on * 2:
                r.blocks = max(1, int(r.blocks * t_on / dt))
            elif done <= self.iters and dt < t_on:
                r.blocks = min(r.blocks * 2, 1 << 30)
            elif dt < t_on / 2:
                r.blocks = min(int(r.blocks * min(8, t_on / dt)), 1 << 30)
            elif done > 2 * self.iters and dt > t_on:
                r.blocks = max(1, r.blocks // 2)
        return outs, dt

    def target(self, pattern, name, make_specs, seconds):
        """Runs off and on arms for `seconds` each on every chip and records one result row per (chip, arm)."""
        specs_by_chip = {r.chip: with_writes(make_specs(r), self.write_every) for r in self.runs}
        for arm in self.cfg["arms"]:
            acc = {r.chip: {"launches": 0, "reads": 0} for r in self.runs}
            first = {}
            passes, t_read = [], 0.0
            t_start = time.monotonic()
            t_end = t_start + seconds
            while True:
                outs, dt = self.launch(specs_by_chip, arm)
                t_read += dt
                p = [int(out[s, 1]) for r, out in zip(self.runs, outs) for s in range(r.slots)]
                passes.append((min(p), max(p)))
                for r, out in zip(self.runs, outs):
                    specs = specs_by_chip[r.chip]
                    a = acc[r.chip]
                    a["launches"] += 1
                    a["reads"] += sum(
                        int(out[s, 1]) * len(specs[s][2]) for s in range(r.slots) if specs[s][0] == READ
                    )
                    s = summarize(out, specs)
                    merge(a, s)
                    if s["bad_reads"] and r.chip not in first:
                        first[r.chip] = first_bad(out, specs) | {"t_s": round(time.monotonic() - t_start, 1)}
                        logger.error(
                            f"chip {r.chip} ({ident_str(r.ident)}) {pattern}/{name} arm {arm}: "
                            f"{s} first {first[r.chip]}"
                        )
                # Under the proportional ramp, stop once less than half a launch remains: one more would overrun
                # t_end by more than it fills.
                left = t_end - time.monotonic()
                if left <= 0 or (self.ramp == "proportional" and left < dt / 2):
                    break
            for r in self.runs:
                a = acc[r.chip]
                row = {"pattern": pattern, "target": name, "arm": arm, "chip": r.chip, **r.ident}
                row.update(first=first.get(r.chip), **a)
                self.results.append(row)
            bad = {c: a["bad_reads"] for c, a in acc.items() if a["bad_reads"]}
            # A launch whose min equals the floor had readers outlast the matmul.
            logger.info(
                f"{pattern}/{name} arm {arm}: {acc[self.runs[0].chip]['launches']} launches, "
                f"{sum(a['reads'] for a in acc.values()):.3e} reads, last launch {dt:.2f}s, "
                f"read runs {t_read:.1f}s of {time.monotonic() - t_start:.1f}s, "
                f"passes per RISC (min-max) by launch {' '.join(f'{lo}-{hi}' for lo, hi in passes)} "
                f"(floor {self.iters}), bad reads by chip {bad or 'none'}"
            )

    def self_test(self):
        """Plants a lane-3 alias of bit 26 in one read per RISC and checks the decode names it exactly."""
        self.seed += 2
        for r in self.runs:
            c = self.cfg["ctrls"][0]
            tgts = [r.space.fit(0x8000000 + s * r.space.rb) for s in range(r.slots)]
            srcs = [a ^ 1 << 26 for a in tgts]
            assert all(r.space.ok(a) for a in srcs), f"chip {r.chip} self-test sources {srcs[0]:#x}.. outside buffer"
            # The decode names the lowest bit whose partner matches, and the planted bytes also sit at tgt ^ 64
            # (the next part of the same slot). A source with no unique pattern, e.g. constant DRAM, then decodes
            # as bit 6, so the sources get a pattern of their own first.
            r.run([(READ, 1, [(a, c)]) for a in srcs], FILL, self.seed, 1)
            r.read(self.seed)
            specs = [(READ, 1, [(a, c)]) for a in tgts]
            r.run(specs, FILL, self.seed, 1, 0, (0, 26, 0xFF000000))
            r.read(self.seed)
            r.run(specs, READ, self.seed, 1)
            s = summarize(r.read(self.seed), specs)
            assert s["bad_reads"] == r.slots and s["stale"] == s["other"] == 0, f"chip {r.chip} self-test: {s}"
            assert s["alias_bits"] == {26: r.slots}, f"chip {r.chip} self-test: {s}"
            assert list(s["lanes"]) == [c] and s["lanes"][c][:3] == [0, 0, 0], f"chip {r.chip} self-test: {s}"
        logger.info(f"self-test: planted controller lane 3 alias of bit 26 decoded exactly on {len(self.runs)} chips")


KNOWN = {
    "hit_0x4003880": 0x4003880,
    "hit_0x4103880": 0x4103880,
    "never_0xC003880": 0xC003880,
    "0x8003880": 0x8003880,
    "0x4600000": 0x4600000,
    "0x3d01c00": 0x3D01C00,
    "pred_clean_0x4203880": 0x4203880,
}


def run_patterns(drv, cfg):
    ctrls, slots, rb = cfg["ctrls"], drv.runs[0].slots, cfg["rb"]
    t0 = KNOWN["hit_0x4003880"]
    row0, gb0, col0 = decode(t0)
    long_s, tgt_s = cfg["seconds"], cfg["target_seconds"]

    def fitted(r, a):
        b = r.space.fit(a)
        if b is None:
            raise RuntimeError(f"{a:#x} cannot be placed in chip {r.chip}'s buffer")
        return b

    for label in cfg["patterns"]:
        # "<pattern>_long" runs the matmul arm in launches of about long_launch_s instead of launch_s.
        pattern = label.removesuffix("_long")
        # "<pattern>_w" turns a third of the RISCs into writers on the pattern's own addresses.
        drv.write_every = 3 if pattern.endswith("_w") else 0
        pattern = pattern.removesuffix("_w")
        drv.ramp = "proportional" if pattern == "v2" else "v1"
        t_on = cfg["long_launch_s"] if label.endswith("_long") else cfg["launch_s"]
        if pattern == "v2":
            t_on = cfg["v2_launch_s"]
        # Compute carried into a new pattern runs at that pattern's read contention: storm -> sdpa and
        # sdpa -> sdpa16 each double the launch (120 s -> 240 s), close to the dispatch timeout. Start every
        # pattern at a quarter of the carried compute, scaled down to a lower target; the bounded per-launch
        # rule regrows it. Scaling up by the target ratio overshoots whenever the last launch ran past its
        # target (4 s x 60 = 300 s).
        if pattern != "v2":
            for r in drv.runs:
                r.blocks = max(1, int(r.blocks * min(1.0, t_on / drv.on_launch_s) / 4))
        drv.on_launch_s = t_on
        if pattern == "known":
            for name, a in KNOWN.items():
                b = fitted(drv.runs[0], a)
                logger.info(f"known {name}: {b:#x} row {decode(b)[0]:#x} {shape(decode(b)[0])}")
                drv.target(label, name, lambda r, a=a: pingpong(fitted(r, a), ctrls, r.space, slots), tgt_s)
        elif pattern == "pingpong":
            drv.target(label, hex(t0), lambda r: pingpong(fitted(r, t0), ctrls, r.space, slots), long_s)
        elif pattern == "hammer":
            drv.target(label, hex(t0), lambda r: hammer(fitted(r, t0), ctrls, slots), long_s)
        elif pattern == "storm":
            drv.target(label, f"row{row0:#x}", lambda r: storm(row0, col0, ctrls, r.space, slots), long_s)
        elif pattern == "sdpa":
            drv.target(
                label, f"ratio{cfg['sdpa_ratio']}", lambda r: sdpa(t0, ctrls, r.space, slots, cfg["sdpa_ratio"]), long_s
            )
        elif pattern == "sdpa16":
            drv.target(
                label,
                f"row{row0:#x}",
                lambda r: sdpa(t0, ctrls, r.space, slots, cfg["sdpa_ratio"], all_banks=True),
                long_s,
            )
        elif pattern == "banks":
            for gb in range(16):
                a = addr_of(row0, gb, col0)
                drv.target(label, f"gbank{gb}", lambda r, a=a: pingpong(fitted(r, a), ctrls, r.space, slots), tgt_s)
        elif pattern == "sdpa16_ca":
            # One row per CA pin: R(i+4) alone drives CA i opposite to both neighbours on the ACT falling edge,
            # the shape that aliases R9 on CA5. The SDPA16 load is the only shape that caught that alias.
            for i in cfg["ca_pins"]:
                t = drv.runs[0].space.fit(addr_of(1 << (4 + i), gb0, col0))
                if t is None:
                    logger.warning(f"sdpa16_ca pin{i}: no placement in the buffer, skipped")
                    continue
                logger.info(f"sdpa16_ca pin{i}: {t:#x} row {decode(t)[0]:#x} {shape(decode(t)[0])}")
                drv.target(
                    label,
                    f"pin{i}_row{decode(t)[0]:#x}",
                    lambda r, t=t: sdpa(fitted(r, t), ctrls, r.space, slots, cfg["sdpa_ratio"], all_banks=True),
                    long_s,
                )
        elif pattern == "ca_full":
            # Each target under every form in ACT_CA_FORMS; the default is the two loads that caught the CA5 alias.
            for name, t in ca_full_targets(t0, col0, drv.runs[0].space, cfg["ca_pins"]):
                if t is None or not drv.runs[0].space.ok(t):
                    logger.warning(f"ca_full {name}: no placement in the buffer, skipped")
                    continue
                row, gb, _ = decode(t)
                logger.info(f"ca_full {name}: {t:#x} row {row:#x} gbank {gb} rising {rising(row, gb):010b} {shape(row)}")
                for form in cfg["ca_forms"]:
                    drv.target(
                        label,
                        f"{name}_row{row:#x}_gb{gb}_{form}",
                        lambda r, t=t, f=form: form_specs(f, fitted(r, t), ctrls, r.space, slots, cfg["sdpa_ratio"]),
                        long_s,
                    )
        elif pattern == "wmis":
            if drv.write_every:
                raise ValueError("wmis has its own writers; _w would turn its readers into writers")
            # Every ca_full target whose falling edge drives R9 (the aliasing pin), placed so its 64 MB partner is
            # in the buffer.
            for name, t in ca_full_targets(t0, col0, drv.runs[0].space, cfg["ca_pins"]):
                m = wmis_target(t, drv.runs[0].space)
                if m is None:
                    continue
                row, gb, _ = decode(m)
                logger.info(f"wmis {name}: {m:#x} row {row:#x} gbank {gb} rising {rising(row, gb):010b} {shape(row)}")
                drv.target(label, f"{name}_row{row:#x}_gb{gb}", lambda r, m=m: wmis(m, ctrls, r.space, slots), long_s)
        elif pattern == "v2":
            space = drv.runs[0].space
            targets = dict(ca_full_targets(t0, col0, space, range(10)) + multi_pin_targets(col0, space))
            for c, name, form in cfg["v2_plan"]:
                t = targets.get(name)
                # "r<row>b<gbank>" names any (row, GDDR bank) at the hit's column, to drive one bank's ACT word.
                if (m := re.fullmatch(r"r(0x[0-9a-f]+)b(\d+)", name)) and t is None:
                    t = addr_of(int(m.group(1), 16), int(m.group(2)), col0)
                if t is None or not space.ok(t):
                    raise ValueError(f"v2 target {name} is not a ca_full, MULTI_PIN or r<row>b<gbank> target in the buffer")
                row, gb, _ = decode(t)
                logger.info(f"v2 {name}: {t:#x} rising {rising(row, gb):010b} falling {falling(row):010b}")
                # Controller "all" spreads the form's slots over the 8 controllers, the same address on each.
                cs = list(range(CTRLS)) if c == "all" else [c]
                tag = f"{'all_ctrls' if c == 'all' else f'c{c}'}_{name}_row{row:#x}_gb{gb}_{form}"
                drv.target(
                    label,
                    tag,
                    lambda r, t=t, cs=cs, f=form: form_specs(f, fitted(r, t), cs, r.space, slots, cfg["sdpa_ratio"]),
                    long_s,
                )
                if cfg.get("v2_stop_on_hit") and any(x["target"] == tag and x.get("bad_reads") for x in drv.results):
                    logger.error(f"v2: {tag} hit, the rest of the plan is skipped")
                    break
        elif pattern == "random":
            # Each draw's seed is logged and its RNG built per chip, so every chip runs the same draw and a hit
            # can be replayed with ACT_RANDOM_SEED.
            for d in range(cfg["random_draws"]):
                seed = cfg["random_seed"] + d
                drv.target(
                    label,
                    f"draw{d}_seed{seed}",
                    lambda r, seed=seed: random_specs(random.Random(seed), ctrls, r.space, slots),
                    long_s,
                )
        elif pattern == "ca_sweep":
            for name, a in ca_shape_targets(gb0, col0):
                b = drv.runs[0].space.fit(a)
                if b is None:
                    logger.warning(f"ca_sweep {name}: {a:#x} has no placement in the buffer, skipped")
                    continue
                logger.info(f"ca_sweep {name}: {b:#x} {shape(decode(b)[0])}")
                drv.target(label, name, lambda r, a=a: pingpong(fitted(r, a), ctrls, r.space, slots), tgt_s)
        else:
            raise ValueError(f"unknown pattern {pattern}")


def act_config(env=os.environ):
    """Chips and config from the ACT_* / STRESS_* knobs in env."""

    def env_int(name, default):
        return int(env.get(name, str(default)), 0)

    chips_env = env.get("ACT_CHIPS", "all")
    chips = list(range(ttnn.GetNumAvailableDevices())) if chips_env == "all" else [int(c) for c in chips_env.split(",")]
    ctrls_env = env.get("ACT_CTRLS", "3")
    gx, gy = (int(v) for v in env.get("STRESS_GRID", "12x10").split("x"))
    cfg = {
        "grid": (gx, gy),
        "bank_mb": env_int("STRESS_BANK_MB", 4096),
        "ctrls": list(range(CTRLS)) if ctrls_env == "all" else [int(c) for c in ctrls_env.split(",")],
        "patterns": env.get("ACT_PATTERNS", "known,pingpong,hammer,storm,sdpa,banks,ca_sweep").split(","),
        "arms": env.get("STRESS_ARMS", "off,on").split(","),
        "rb": env_int("ACT_READ_BYTES", 256),
        "iters": env_int("ACT_ITERS", 2000),
        "launch_s": float(env.get("ACT_LAUNCH_S", "2")),
        "long_launch_s": float(env.get("ACT_LONG_LAUNCH_S", "120")),
        "seconds": float(env.get("ACT_SECONDS", "20")),
        "target_seconds": float(env.get("ACT_TARGET_SECONDS", "4")),
        "sdpa_ratio": [int(v) for v in env.get("ACT_SDPA_RATIO", "1,1,1").split(",")],
        "ca_pins": [int(v) for v in env.get("ACT_CA_PINS", "0,1,2,3,4,5,6,7,8,9").split(",")],
        "ca_forms": env.get("ACT_CA_FORMS", "sdpa,sdpa16").split(","),
        "random_draws": env_int("ACT_RANDOM_DRAWS", 15),
        "random_seed": env_int("ACT_RANDOM_SEED", int(time.time()) & 0xFFFFF),
        "blocks": env_int("STRESS_BLOCKS", 4000000),
        "decode_max": env_int("STRESS_DECODE_MAX", 64),
        "seed0": env_int("STRESS_SEED", int(time.time()) & 0xFFFF) * 1024,
        "self_test": env_int("STRESS_SELF_TEST", 1),
        "v2_plan": [
            (c if c == "all" else int(c), *spec.split("/"))
            for c, spec in (e.split(":") for e in env.get("ACT_V2", "3:orig_bank3/sdpa16,3:orig_bank3/sdpa").split(","))
        ],
        "v2_launch_s": float(env.get("ACT_V2_LAUNCH_S", "60")),
    }
    return chips, cfg


def run_act(chips, cfg):
    assert all(form in FORMS for _, _, form in cfg["v2_plan"]), cfg["v2_plan"]
    assert all(form in FORMS for form in cfg["ca_forms"]), cfg["ca_forms"]
    assert all(a in ("on", "off") for a in cfg["arms"]), cfg["arms"]
    assert cfg["rb"] % 64 == 0 and 64 <= cfg["rb"] <= 2048, cfg["rb"]
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    logger.info(f"chips {chips}, config {cfg}")
    devices = ttnn.CreateDevices(chips)
    try:
        idents, host_line = chip_identities(chips)
        logger.info(f"asic map: host {host_line}")
        for c in chips:
            logger.info(f"asic map: chip {c} = {ident_str(idents[c])}")
        runs = [ActRun(c, devices[c], cfg) for c in chips]
        for r in runs:
            r.ident = idents[r.chip]
        drv = Driver(runs, cfg)
        if cfg["self_test"]:
            drv.self_test()
        try:
            run_patterns(drv, cfg)
        finally:
            (OUT_DIR / f"act_{os.environ.get('SLURM_JOB_ID', 'local')}_{int(time.time())}.json").write_text(
                json.dumps(
                    {"config": cfg, "host": host_line, "asics": idents, "results": drv.results}, indent=1, default=str
                )
            )
        for r in runs:
            ttnn.deallocate(r.buf)
    finally:
        ttnn.CloseDevices(devices)
    bad = sorted(
        {
            (row["chip"], row.get("asic_id"), row["pattern"], row["target"], row["arm"])
            for row in drv.results
            if row["bad_reads"]
        }
    )
    assert not bad, f"miscompares on (chip, asic_id, pattern, target, arm): {bad}"


@pytest.mark.timeout(0)
def test_dram_act_patterns():
    run_act(*act_config())


# Fixed config for the health check: defaults with no env knobs read, so a shell's ACT_* / STRESS_* settings
# cannot change what it screens. The SDPA-mimic read stream (sdpa form) on controller 3, row 0x200, GDDR bank 3,
# for 300 s with the matmul arm on: the load that exposes the lane-3 64 MB (bit 26) alias. Then the same form for
# 300 s on all 8 controllers at once, to reach an alias outside controller 3; each controller gets only an eighth of
# the stream there, so it does not replace the controller 3 pass. The sdpa16 form never caught a boot this one
# missed. On a faulty die the alias shows on some boots and not others, and a weak bad boot took up to about
# 4 minutes to hit, so one run screens one boot. The first hit fails the test without running the rest.
SCREEN_CFG = {
    "ctrls": [3],
    "patterns": ["v2"],
    "arms": ["on"],
    "iters": 4000,
    "seconds": 300.0,
    "v2_plan": [(3, "orig_bank3", "sdpa"), ("all", "orig_bank3", "sdpa")],
    "v2_stop_on_hit": True,
    "v2_launch_s": 60.0,
    "self_test": 1,
}


@pytest.mark.timeout(1200)
def test_dram_act_c3_screen():
    chips, cfg = act_config(env={})
    cfg.update(SCREEN_CFG)
    run_act(chips, cfg)


# The same screen widened further: the sdpa form for 300 s on all 8 controllers at once, then alone on each
# controller faults have been seen on (3, 1, 7, 0, in that order), so 1, 7 and 0 also get the full stream.
SCREEN_ALL_ORDER = ("all", 3, 1, 7, 0)
SCREEN_ALL_CFG = {
    **SCREEN_CFG,
    "v2_plan": [(c, "orig_bank3", "sdpa") for c in SCREEN_ALL_ORDER],
}


@pytest.mark.timeout(3600)
def test_dram_act_all_ctrls_screen():
    chips, cfg = act_config(env={})
    cfg.update(SCREEN_ALL_CFG)
    run_act(chips, cfg)
