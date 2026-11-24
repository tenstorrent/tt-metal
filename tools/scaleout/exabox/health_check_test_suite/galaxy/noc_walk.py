#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""noc_walk.py: the seeded per-chip NOC data check, run on a mesh that is already open (fabric_open.py --noc-wall-s).

Every pass, one raw data-movement kernel on every Tensix core of every chip: the core writes its own 1 KB slice of two
destinations, a core's L1 (the pass's partner: core c -> core (c + shift) % cores, so every destination has one
writer) and one DRAM bank, as 512 B packets, reads it back into a second L1 buffer and compares it on the core. Six
pattern families per destination (walking 1, walking 0, checkerboard and its inverse, address as data, PRBS); even
passes run on NOC0 (RISC 0), odd passes on NOC1 (RISC 1), and the first two passes always run, so every chip gets
both NOCs. The shift moves every pass, so over passes every source reaches every core. Not covered: Ethernet cores as
sources or destinations (the link stages exercise them) and NOC multicast.

noc_check() returns the result row; nocwalk_lines() (this module) makes its NUMERIC lines: one NUMERIC FAIL per chip
with a bad word or a destination that did not finish, named by BDF, tray, ASIC and chip id, with the source and
destination core (logical x,y) or DRAM bank, the NOC, the pattern, the byte offset and the word got and wanted; a
mismatch every chip shows is one NUMERIC WARN naming the chips (the test's or software's, never a chip FAIL); a check
that could not run is a NUMERIC WARN; then one summary line."""
import time

from chunk_sched import short
from noc_a2a import end_name, grid, mcfg
from numeric_pinpoint import asic_of

SLICE = 1024  # bytes each core writes into each destination
WORDS = SLICE // 4
PKT = 512  # NOC write size
PATTERNS = ("walking-1", "walking-0", "checkerboard", "checkerboard-inv", "address", "PRBS")
DESTS = 2  # per core and pass: a core's L1, a DRAM bank
RW = 6  # result words per destination: done, mismatches, first byte offset, wanted, got, pattern
KEEP = 3  # example failures named per chip

# runtime args: src, readback, results, destination L1 address, DRAM buffer address, this core's slot (core index * SLICE),
# seed (this pass's), partner NOC x, y, DRAM bank
KERNEL = r"""
#include <cstdint>
#include "api/dataflow/dataflow_api.h"

static uint32_t word(uint32_t p, uint32_t v, uint32_t i, uint32_t a, uint32_t& s) {
    switch (p) {
        case 0: return 1u << ((i + v) & 31);
        case 1: return ~(1u << ((i + v) & 31));
        case 2: return (i & 1) ? 0x55555555u : 0xAAAAAAAAu;
        case 3: return (i & 1) ? 0xAAAAAAAAu : 0x55555555u;
        case 4: return a + 4 * i + v;
        default: s ^= s << 13; s ^= s >> 17; s ^= s << 5; return s;
    }
}

void kernel_main() {
    uint32_t i = 0;
    const uint32_t src = get_arg_val<uint32_t>(i++);
    const uint32_t rb = get_arg_val<uint32_t>(i++);
    const uint32_t res = get_arg_val<uint32_t>(i++);
    const uint32_t dl1 = get_arg_val<uint32_t>(i++);
    const uint32_t dram = get_arg_val<uint32_t>(i++);
    const uint32_t slot = get_arg_val<uint32_t>(i++);
    const uint32_t seed = get_arg_val<uint32_t>(i++);
    const uint32_t px = get_arg_val<uint32_t>(i++);
    const uint32_t py = get_arg_val<uint32_t>(i++);
    const uint32_t bank = get_arg_val<uint32_t>(i++);
    volatile tt_l1_ptr uint32_t* s = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(src);
    volatile tt_l1_ptr uint32_t* b = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(rb);
    volatile tt_l1_ptr uint32_t* r = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(res);
    const uint64_t dest[2] = {get_noc_addr(px, py, dl1), get_noc_addr_from_bank_id<true>(bank, dram + slot)};
    const uint32_t addr[2] = {dl1, dram + slot};
    for (uint32_t d = 0; d < 2; d++) {
        for (uint32_t k = 0; k < 6; k++) r[d * 6 + k] = 0;
    }
    for (uint32_t d = 0; d < 2; d++) {
        volatile tt_l1_ptr uint32_t* o = r + d * 6;
        for (uint32_t p = 0; p < 6; p++) {
            uint32_t st = seed ^ (d << 16) ^ p;
            if (!st) st = 1;
            for (uint32_t w = 0; w < SLICE_WORDS; w++) {
                s[w] = word(p, seed, w, addr[d], st);
                b[w] = ~s[w];
            }
            for (uint32_t off = 0; off < SLICE_BYTES; off += PKT_BYTES) {
                noc_async_write(src + off, dest[d] + off, PKT_BYTES);
            }
            noc_async_write_barrier();
            noc_async_read(dest[d], rb, SLICE_BYTES);
            noc_async_read_barrier();
            invalidate_l1_cache();
            uint32_t bad = 0;
            for (uint32_t w = 0; w < SLICE_WORDS; w++) {
                if (s[w] != b[w]) {
                    if (o[1] == 0 && bad == 0) {
                        o[2] = w * 4;
                        o[3] = s[w];
                        o[4] = b[w];
                        o[5] = p;
                    }
                    bad++;
                }
            }
            o[1] += bad;
        }
        o[0] = 1;
    }
}
"""


def program(ttnn, mesh, crs, gx, gy, k, seed, addrs, nb):
    """Pass k's program: the kernel on every core, NOC0 on even passes, NOC1 on odd ones. addrs: src, rb, res, dst, dram."""
    n = gx * gy
    shift = 1 + (k // 2 * 7 + k) % (n - 1) if n > 1 else 0
    wc = {}
    for y in range(gy):
        for x in range(gx):
            c = mesh.worker_core_from_logical_core(ttnn.CoreCoord(x, y))
            wc[x, y] = (int(c.x), int(c.y))
    rt = ttnn.RuntimeArgs()
    for y in range(gy):
        for x in range(gx):
            c = y * gx + x
            q = (c + shift) % n
            rt[x][y] = [*addrs[:4], addrs[4], c * SLICE, seed + k, *wc[q % gx, q // gx], (c + k) % nb]
    dm = ttnn.DataMovementProcessor
    p = k % 2
    src = KERNEL.replace("SLICE_WORDS", str(WORDS)).replace("SLICE_BYTES", str(SLICE)).replace("PKT_BYTES", str(PKT))
    kd = ttnn.KernelDescriptor(
        kernel_source=src,
        source_type=ttnn.KernelDescriptor.SourceType.SOURCE_CODE,
        core_ranges=crs,
        compile_time_args=[],
        runtime_args=rt,
        config=ttnn.DataMovementConfigDescriptor(
            processor=dm.RISCV_1 if p else dm.RISCV_0, noc=ttnn.NOC.NOC_1 if p else ttnn.NOC.NOC_0
        ),
    )
    return ttnn.ProgramDescriptor(kernels=[kd], semaphores=[], cbs=[]), shift


def noc_check(ttnn, torch, mesh, chips, wall_s, nb, beat, seed=0x5EED):
    """The passes above until wall_s s (at least two: NOC0 and NOC1). chips: mesh index -> dict(pci, tray, dev).
    Returns dict(passes, transfers (per chip), seed, grid, chips={mesh index: dict(bad, unfinished, noc_bad {0, 1},
    examples)}, error (the check could not run))."""
    gx, gy = grid(mesh)
    n = gx * gy
    crs = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(gx - 1, gy - 1))})
    out = dict(passes=0, transfers=0, seed=seed, grid=(gx, gy), error=None)
    st = {i: dict(bad=0, unfinished=0, noc_bad={0: 0, 1: 0}, examples=[]) for i in chips}
    out["chips"] = st
    live = []

    def put(rows, cols, mc):
        t = ttnn.from_torch(
            torch.zeros(rows, cols, dtype=torch.int32),
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=mesh,
            memory_config=mc,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
        )
        live.append(t)
        return t

    try:
        s, b, d = (put(n, WORDS, mcfg(ttnn, "h", crs, n, WORDS, n)) for _ in range(3))
        o = put(n, DESTS * RW, mcfg(ttnn, "h", crs, n, DESTS * RW, n))
        dr = put(n * nb, WORDS, ttnn.DRAM_MEMORY_CONFIG)
        addrs = [t.buffer_address() for t in (s, b, o, d, dr)]
        t0, k, last = time.monotonic(), 0, 0.0
        while k < 2 or time.monotonic() - t0 + last < wall_s:
            p0 = time.monotonic()
            beat(f"launch {k + 1}")
            prog, shift = program(ttnn, mesh, crs, gx, gy, k, seed, addrs, nb)
            ttnn.generic_op([s, b, d, dr, o], prog)
            ttnn.synchronize_device(mesh)
            beat(f"done {k + 1}")
            for i, t in enumerate(ttnn.get_device_tensors(o.cpu())):
                r = ttnn.to_torch(t).contiguous().view(torch.int32).reshape(n, DESTS, RW)
                c = st[i]
                for core in range(n):
                    for j in range(DESTS):
                        w = [int(v) & 0xFFFFFFFF for v in r[core, j]]
                        if w[0] != 1 or w[1]:
                            c["bad" if w[1] else "unfinished"] += w[1] or 1
                            c["noc_bad"][k % 2] += 1
                            if len(c["examples"]) < KEEP:
                                q = (core + shift) % n
                                dst = q if j == 0 else -1 - (core + k) % nb
                                c["examples"].append(
                                    dict(
                                        src=core,
                                        dst=dst,
                                        noc=k % 2,
                                        done=w[0] == 1,
                                        bad=w[1],
                                        off=w[2],
                                        want=w[3],
                                        got=w[4],
                                        pat=PATTERNS[w[5] % 6],
                                    )
                                )
            out["passes"] += 1
            out["transfers"] += DESTS * len(PATTERNS)
            k += 1
            last = time.monotonic() - p0
    except Exception as e:  # noqa: BLE001
        out["error"] = f"{type(e).__name__}: {str(e).splitlines()[0][:300] if str(e) else ''}"
    finally:
        for t in live:
            try:
                ttnn.deallocate(t)
            except Exception:  # noqa: BLE001
                pass
    return out


WHY = "after the end reset's glx_reset"
NAME = "C9"


def nocwalk_lines(r, chips):
    """The NOC walk row's finding lines (see the module text): one NUMERIC FAIL per flagged chip, one common-fault WARN,
    a not-checked WARN, then the summary. r: noc_check()'s dict, chips: mesh index -> dict(pci, tray, dev)."""
    if "passes" not in r:  # the check did not start (fabric_open.open_check's error row)
        return [f"NUMERIC WARN NOC data check {WHY} not checked: {r['error']}"]
    gx = r["grid"][0]
    clk = lambda t: time.strftime("%H:%M:%S", time.localtime(t))  # noqa: E731
    out = []
    flagged = {i: c for i, c in r["chips"].items() if c["bad"] or c["unfinished"]}
    common = len(chips) > 3 and len(flagged) == len(chips)  # every chip: the kernel's or software's, not a chip's
    order = lambda i: (chips[i].get("tray") or 0, str(chips[i].get("pci")))  # noqa: E731

    for i in [] if common else sorted(flagged, key=order):
        c, m = flagged[i], chips[i]
        ex = "; ".join(
            f"{end_name(e['src'], gx)} -> {end_name(e['dst'], gx)} on NOC{e['noc']}, {e['pat']}, "
            + (f"byte {e['off']}, got {e['got']:#010x} want {e['want']:#010x}" if e["done"] else "did not finish")
            for e in c["examples"]
        )
        out.append(
            f"NUMERIC FAIL {short(m.get('pci'))} (tray {m.get('tray') or '?'}) chip "
            f"{m.get('dev')} [{NAME.lower()}]: NOC data check {WHY}: ASIC {asic_of(m.get('pci')) or '?'}, {c['bad']} bad words, "
            f"{c['unfinished']} destinations unfinished, NOC0 {c['noc_bad'][0]} / NOC1 {c['noc_bad'][1]} failing "
            f"core-destinations; first: {ex}"
        )
    if common:
        e = next(iter(flagged.values()))["examples"][0]
        who = ", ".join(
            f"{short(chips[i].get('pci'))} (tray {chips[i].get('tray') or '?'})" for i in sorted(flagged, key=order)[:4]
        )
        out.append(
            f"NUMERIC WARN NOC data check {WHY}: all {len(chips)} chips failed (test or software, not judged a chip's "
            f"fault; {who} and the rest): "
            f"first {end_name(e['src'], gx)} -> {end_name(e['dst'], gx)} on NOC{e['noc']}, {e['pat']}"
        )
    if r["error"]:
        out.append(f"NUMERIC WARN NOC data check {WHY} not checked: {r['error']}")
    out.append(
        f"NUMERIC NOCWALK {WHY}: {clk(r['start'])}-{clk(r['end'])}, {r['passes']} passes (NOC0 and NOC1, "
        f"seed {r['seed']:#x}), {r['transfers']} transfers per core on {len(chips)} chips; "
        + (
            "every chip mismatched"
            if common
            else f"{len(flagged)} chip(s) mismatched"
            if flagged
            else "no chip mismatched"
            if r["error"]
            else "every chip matched"
        )
    )
    return out
