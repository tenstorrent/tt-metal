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

REPO = Path(__file__).resolve().parents[4]
OUT_DIR = Path(os.environ.get("STRESS_OUT_DIR", REPO / "repro_logs" / "act"))

DM_KERNEL = f"""
#include "api/dataflow/dataflow_api.h"

FORCE_INLINE uint32_t mix(uint32_t x) {{
    x ^= x >> 16;
    x *= 0x7FEB352Du;
    x ^= x >> 15;
    x *= 0x846CA68Bu;
    x ^= x >> 16;
    return x;
}}

FORCE_INLINE uint32_t pat(uint32_t a, uint32_t ctrl, uint32_t seed) {{
    const uint32_t v = mix(mix((a >> 2) ^ (seed * 0x9E3779B9u)) ^ (ctrl * 0x85EBCA77u));
    return (seed & 1) ? ~v : v;
}}

FORCE_INLINE uint32_t byte_mask(uint32_t x) {{
    uint32_t m = 0;
    for (uint32_t b = 0; b < 4; ++b) {{
        m |= (x & (0xFFu << (8 * b))) ? (0xFFu << (8 * b)) : 0;
    }}
    return m;
}}

// Entries are 64 B aligned per-bank addresses with the controller id in the low bits.
FORCE_INLINE uint64_t dram(uint32_t e) {{ return get_noc_addr_from_bank_id<true>(e & 7, e & ~63u); }}

void kernel_main() {{
    const uint32_t out_addr = get_arg_val<uint32_t>(0);
    const uint32_t slot = get_arg_val<uint32_t>(1);
    const uint32_t mode = get_arg_val<uint32_t>(2);
    const uint32_t seed = get_arg_val<uint32_t>(3);
    const uint32_t iters = get_arg_val<uint32_t>(4);
    const uint32_t rb = get_arg_val<uint32_t>(5);
    const uint32_t n = get_arg_val<uint32_t>(6);
    const uint32_t compute_on = get_arg_val<uint32_t>(7);
    const uint32_t barrier_each = get_arg_val<uint32_t>(8);
    const uint32_t decode_max = get_arg_val<uint32_t>(9);
    const uint32_t inj_e = get_arg_val<uint32_t>(10);
    const uint32_t inj_bit = get_arg_val<uint32_t>(11);
    const uint32_t inj_mask = get_arg_val<uint32_t>(12);
    constexpr uint32_t scratch = get_compile_time_arg_val(0);
    constexpr bool feeds_compute = get_compile_time_arg_val(1);
    constexpr auto out_args = TensorAccessorArgs<2>();
    const auto out = TensorAccessor(out_args, out_addr);

    const uint32_t nw = rb / 4;
    const uint32_t hdr_addr = get_write_ptr(scratch);
    volatile tt_l1_ptr uint32_t* hdr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(hdr_addr);
    const uint32_t gold_addr = hdr_addr + {ROW_WORDS * 4};
    volatile tt_l1_ptr uint32_t* gold = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(gold_addr);
    const uint32_t work_addr = gold_addr + n * rb;
    volatile tt_l1_ptr uint32_t* work = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(work_addr);
    const uint32_t raw_addr = work_addr + n * rb;
    volatile tt_l1_ptr uint32_t* raw = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(raw_addr);
    for (uint32_t i = 0; i < {ROW_WORDS}; ++i) {{
        hdr[i] = 0;
    }}
    hdr[7] = {NONE}u;

    if constexpr (feeds_compute) {{
        if (compute_on) {{
            for (uint32_t c = 0; c < 2; ++c) {{
                const uint32_t cb = c ? {sweep.CB_IN1} : {sweep.CB_IN0};
                cb_reserve_back(cb, {sweep.KT});
                volatile tt_l1_ptr uint32_t* t = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(cb));
                for (uint32_t i = 0; i < {sweep.KT} * 512; ++i) {{
                    t[i] = mix(i * 2 + cb + seed * 0x10001u) & 0xBFFFBFFFu;
                }}
                cb_push_back(cb, {sweep.KT});
            }}
        }}
    }}

    for (uint32_t e = 0; e < n; ++e) {{
        const uint32_t ent = get_arg_val<uint32_t>({ENTRY0} + e);
        for (uint32_t j = 0; j < nw; ++j) {{
            gold[e * nw + j] = pat((ent & ~63u) + 4 * j, ent & 7, seed);
        }}
    }}

    uint32_t bad_reads = 0, bad_words = 0, stale = 0, other = 0, undecoded = 0, decoded = 0;
    uint32_t it = 0;

    auto check = [&](uint32_t e) {{
        const uint32_t ent = get_arg_val<uint32_t>({ENTRY0} + e);
        const uint32_t a = ent & ~63u;
        const uint32_t ctrl = ent & 7;
        volatile tt_l1_ptr uint32_t* w = work + e * nw;
        volatile tt_l1_ptr uint32_t* g = gold + e * nw;
        uint32_t bad = 0;
        for (uint32_t j = 0; j < nw; ++j) {{
            const uint32_t x = w[j] ^ g[j];
            if (x == 0) {{
                continue;
            }}
            for (uint32_t b = 0; b < 4; ++b) {{
                hdr[{LANE_OFF} + 4 * ctrl + b] += (x >> (8 * b)) & 0xFFu ? 1 : 0;
            }}
            if (hdr[7] == {NONE}u) {{
                hdr[7] = e;
                hdr[8] = j;
                hdr[9] = w[j];
                hdr[10] = g[j];
                hdr[11] = it;
            }}
            ++bad;
        }}
        if (bad == 0) {{
            return;
        }}
        ++bad_reads;
        bad_words += bad;
        hdr[{HDR_WORDS} + e] += 1;
        if (decoded >= decode_max) {{
            ++undecoded;
            return;
        }}
        ++decoded;
        bool is_stale = true;
        for (uint32_t j = 0; j < nw && is_stale; ++j) {{
            const uint32_t x = w[j] ^ g[j];
            is_stale = !(x && ((w[j] ^ pat(a + 4 * j, ctrl, seed - 1)) & byte_mask(x)));
        }}
        if (is_stale) {{
            ++stale;
            return;
        }}
        // Sources are read back rather than predicted: most of them lie outside the filled entries.
        for (uint32_t k = {COL_SHIFT}; k < 32; ++k) {{
            noc_async_read(dram((a ^ (1u << k)) | ctrl), raw_addr, rb);
            noc_async_read_barrier();
            bool ok = true;
            for (uint32_t j = 0; j < nw && ok; ++j) {{
                const uint32_t x = w[j] ^ g[j];
                ok = !(x && ((w[j] ^ raw[j]) & byte_mask(x)));
            }}
            if (ok) {{
                hdr[{ALIAS_OFF} + k] += 1;
                return;
            }}
        }}
        ++other;
    }};

    if (mode == {FILL}) {{
        for (uint32_t e = 0; e < n; ++e) {{
            const uint32_t ent = get_arg_val<uint32_t>({ENTRY0} + e);
            if (e == inj_e) {{
                noc_async_read(dram(((ent & ~63u) ^ (1u << inj_bit)) | (ent & 7)), raw_addr, rb);
                noc_async_read_barrier();
                for (uint32_t j = 0; j < nw; ++j) {{
                    gold[e * nw + j] = (gold[e * nw + j] & ~inj_mask) | (raw[j] & inj_mask);
                }}
            }}
            noc_async_write(gold_addr + e * rb, dram(ent), rb);
        }}
        noc_async_write_barrier();
        it = 1;
    }} else {{
        while (it < iters || (compute_on && !cb_pages_available_at_front({sweep.CB_OUT}, 1))) {{
            if (mode == {WRITE}) {{
                for (uint32_t e = 0; e < n; ++e) {{
                    noc_async_write(gold_addr + e * rb, dram(get_arg_val<uint32_t>({ENTRY0} + e)), rb);
                    if (barrier_each) {{
                        noc_async_write_barrier();
                    }}
                }}
                noc_async_write_barrier();
            }} else if (barrier_each) {{
                for (uint32_t e = 0; e < n; ++e) {{
                    noc_async_read(dram(get_arg_val<uint32_t>({ENTRY0} + e)), work_addr + e * rb, rb);
                    noc_async_read_barrier();
                    check(e);
                }}
            }} else {{
                for (uint32_t e = 0; e < n; ++e) {{
                    noc_async_read(dram(get_arg_val<uint32_t>({ENTRY0} + e)), work_addr + e * rb, rb);
                }}
                noc_async_read_barrier();
                for (uint32_t e = 0; e < n; ++e) {{
                    check(e);
                }}
            }}
            ++it;
        }}
    }}
    hdr[0] = seed;
    hdr[1] = it;
    hdr[2] = bad_reads;
    hdr[3] = bad_words;
    hdr[4] = stale;
    hdr[5] = other;
    hdr[6] = undecoded;
    hdr[15] = {MAGIC};
    noc_async_write(hdr_addr, out.get_noc_addr(slot), {ROW_WORDS * 4});
    noc_async_write_barrier();
}}
"""


def env_int(name, default):
    return int(os.environ.get(name, str(default)), 0)


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


def sdpa(t, ctrls, space, slots, ratio, all_banks=False):
    """Shared K-block streaming at t, private ping-pong rows in t's GDDR bank, and writers on other rows of it.

    all_banks spreads the shared stream over t's row in all 16 GDDR banks (16 reads each), so one run
    compares banks under the same load.
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


class ActRun:
    def __init__(self, chip, device, cfg):
        gx, gy = cfg["grid"]
        self.chip, self.cfg = chip, cfg
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
                    source_type=ttnn.KernelDescriptor.SourceType.SOURCE_CODE,
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
                    source_type=ttnn.KernelDescriptor.SourceType.SOURCE_CODE,
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
        specs_by_chip = {r.chip: make_specs(r) for r in self.runs}
        for arm in self.cfg["arms"]:
            acc = {r.chip: {"launches": 0, "reads": 0} for r in self.runs}
            first = {}
            t_end = time.monotonic() + seconds
            while True:
                outs, dt = self.launch(specs_by_chip, arm)
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
                        first[r.chip] = first_bad(out, specs)
                        logger.error(f"chip {r.chip} {pattern}/{name} arm {arm}: {s} first {first[r.chip]}")
                if time.monotonic() >= t_end:
                    break
            for r in self.runs:
                a = acc[r.chip]
                row = {"pattern": pattern, "target": name, "arm": arm, "chip": r.chip, "first": first.get(r.chip), **a}
                self.results.append(row)
            bad = {c: a["bad_reads"] for c, a in acc.items() if a["bad_reads"]}
            logger.info(
                f"{pattern}/{name} arm {arm}: {acc[self.runs[0].chip]['launches']} launches, "
                f"{sum(a['reads'] for a in acc.values()):.3e} reads, iters {self.iters}, last launch {dt:.2f}s, "
                f"bad reads by chip {bad or 'none'}"
            )

    def self_test(self):
        """Plants a lane-3 alias of bit 26 in one read per RISC and checks the decode names it exactly."""
        self.seed += 2
        for r in self.runs:
            c = self.cfg["ctrls"][0]
            specs = [(READ, 1, [(r.space.fit(0x4000000 + s * r.space.rb), c)]) for s in range(r.slots)]
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
        t_on = cfg["long_launch_s"] if label.endswith("_long") else cfg["launch_s"]
        # Compute carried into a new pattern runs at that pattern's read contention: storm -> sdpa and
        # sdpa -> sdpa16 each double the launch (120 s -> 240 s), close to the dispatch timeout. Start every
        # pattern at a quarter of the carried compute, scaled down to a lower target; the bounded per-launch
        # rule regrows it. Scaling up by the target ratio overshoots whenever the last launch ran past its
        # target (4 s x 60 = 300 s).
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
            # Every ACT pin toggling in both directions against opposite-toggling neighbours (CA9 only 0 -> 1:
            # its rising edge is the opcode), plus the original hit 0x4003880 (bank 1, BA1=0) and the same row in
            # bank 3 (BA1=1), whose rising CA5 matches the falling R9. Each under both loads that caught the CA5
            # alias: sdpa (one GDDR bank) and sdpa16 (the row in all 16 banks).
            targets = [("orig", t0), ("orig_bank3", t0 | 2 << GBANK_SHIFT)]
            for i in cfg["ca_pins"]:
                for up in (True, False) if i < 9 else (True,):
                    t = toggle_target(i, up, col0, drv.runs[0].space)[0]
                    targets.append((f"ca{i}_{'up' if up else 'dn'}", t))
            for name, t in targets:
                if t is None or not drv.runs[0].space.ok(t):
                    logger.warning(f"ca_full {name}: no placement in the buffer, skipped")
                    continue
                row, gb, _ = decode(t)
                logger.info(f"ca_full {name}: {t:#x} row {row:#x} gbank {gb} rising {rising(row, gb):010b} {shape(row)}")
                for form, all_banks in (("sdpa", False), ("sdpa16", True)):
                    drv.target(
                        label,
                        f"{name}_row{row:#x}_gb{gb}_{form}",
                        lambda r, t=t, ab=all_banks: sdpa(fitted(r, t), ctrls, r.space, slots, cfg["sdpa_ratio"], all_banks=ab),
                        long_s,
                    )
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


@pytest.mark.timeout(0)
def test_dram_act_patterns():
    chips_env = os.environ.get("ACT_CHIPS", "all")
    chips = list(range(ttnn.GetNumAvailableDevices())) if chips_env == "all" else [int(c) for c in chips_env.split(",")]
    ctrls_env = os.environ.get("ACT_CTRLS", "3")
    gx, gy = (int(v) for v in os.environ.get("STRESS_GRID", "12x10").split("x"))
    cfg = {
        "grid": (gx, gy),
        "bank_mb": env_int("STRESS_BANK_MB", 4096),
        "ctrls": list(range(CTRLS)) if ctrls_env == "all" else [int(c) for c in ctrls_env.split(",")],
        "patterns": os.environ.get("ACT_PATTERNS", "known,pingpong,hammer,storm,sdpa,banks,ca_sweep").split(","),
        "arms": os.environ.get("STRESS_ARMS", "off,on").split(","),
        "rb": env_int("ACT_READ_BYTES", 256),
        "iters": env_int("ACT_ITERS", 2000),
        "launch_s": float(os.environ.get("ACT_LAUNCH_S", "2")),
        "long_launch_s": float(os.environ.get("ACT_LONG_LAUNCH_S", "120")),
        "seconds": float(os.environ.get("ACT_SECONDS", "20")),
        "target_seconds": float(os.environ.get("ACT_TARGET_SECONDS", "4")),
        "sdpa_ratio": [int(v) for v in os.environ.get("ACT_SDPA_RATIO", "1,1,1").split(",")],
        "ca_pins": [int(v) for v in os.environ.get("ACT_CA_PINS", "0,1,2,3,4,5,6,7,8,9").split(",")],
        "random_draws": env_int("ACT_RANDOM_DRAWS", 15),
        "random_seed": env_int("ACT_RANDOM_SEED", int(time.time()) & 0xFFFFF),
        "blocks": env_int("STRESS_BLOCKS", 4000000),
        "decode_max": env_int("STRESS_DECODE_MAX", 64),
        "seed0": env_int("STRESS_SEED", int(time.time()) & 0xFFFF) * 1024,
        "self_test": env_int("STRESS_SELF_TEST", 1),
    }
    assert all(a in ("on", "off") for a in cfg["arms"]), cfg["arms"]
    assert cfg["rb"] % 64 == 0 and 64 <= cfg["rb"] <= 2048, cfg["rb"]
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    logger.info(f"chips {chips}, config {cfg}")
    devices = ttnn.CreateDevices(chips)
    try:
        runs = [ActRun(c, devices[c], cfg) for c in chips]
        drv = Driver(runs, cfg)
        if cfg["self_test"]:
            drv.self_test()
        try:
            run_patterns(drv, cfg)
        finally:
            (OUT_DIR / f"act_{os.environ.get('SLURM_JOB_ID', 'local')}_{int(time.time())}.json").write_text(
                json.dumps({"config": cfg, "results": drv.results}, indent=1, default=str)
            )
        for r in runs:
            ttnn.deallocate(r.buf)
    finally:
        ttnn.CloseDevices(devices)
    bad = sorted({(row["chip"], row["pattern"], row["target"], row["arm"]) for row in drv.results if row["bad_reads"]})
    assert not bad, f"miscompares on (chip, pattern, target, arm): {bad}"
