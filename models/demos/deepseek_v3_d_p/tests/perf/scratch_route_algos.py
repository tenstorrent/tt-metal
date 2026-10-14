# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""Scratch: the sparse all-gather's routing algorithms in isolation on one Tensix data-movement RISC (one chip, no
fabric). Per algorithm: the top-k ids of all G x S rows are loaded into L1 (timed separately), the algorithm runs over
them, its 1-byte-per-row route codes are checked against a Python model of the same algorithm, and the core's wall
clock gives the cycles. Routing geometry as scratch_sparse_ag_loudbox.py: 8-chip ring, 2 links, 640 rows per chip,
256 experts, 32 per chip, top-8.

Route code per row: a (hops forward, 3 bits) | b (hops back) << 3 | forward link << 6 | backward link << 7.
Algorithms:
  mask       the 7-bit mask of the other chips needing the row (no routing): code = mask
  hash       mask, 128-entry table of minimal arcs, a stateless choice from a hash of the row id
  rr         per origin: tie = row % (number of minimal arcs), links alternate per direction
  greedy_o   per origin: every minimal arc and link pair, scored on this origin's own hop loads (busiest, then sum)
  greedy_g   the prototype's global greedy: the same score on the whole ring's loads, rows in (row, origin) order
  scan       after greedy_g: one port (position 3, forward, link 0) builds its row list (page, upstream wait)
mask / hash / rr / greedy_o run over all 8 origins; a chip only does its own, so per chip = total / 8.
"""

import os

import pytest
import torch
from loguru import logger

import ttnn
from models.demos.deepseek_v3_d_p.tests.op_unit_tests.test_dispatch_fabric2d import PRODUCTION_ROUTING, _draw_indices

# SRA_S rows per chip, SRA_E experts (32 per chip at 256): small values for a quick functional check
G, K, L = 8, 8, 2
S = int(os.environ.get("SRA_S", "640"))
E = int(os.environ.get("SRA_E", "256"))
EPC = E // G
ALGO_FILTER = os.environ.get("SRA_ALGOS")
INC_EVERY = 8
SCAN_PORT = (3, 0, 0)  # position, direction (0 fwd), link
CLOCK_MHZ = 1350.0  # AICLK the wall clock counts (as the profiler's cycle -> us conversion here)
ALGOS = ["mask", "hash", "rr", "greedy_o", "greedy_g", "scan", "probe"]


def _ids(routing):
    torch.manual_seed(42)
    if routing == "dense":  # one expert on every chip
        return (torch.arange(G) * EPC).view(1, 1, G).expand(G, S, K).clone()
    if routing == "uniform":
        return torch.stack([torch.stack([torch.randperm(E)[:K] for _ in range(S)]) for _ in range(G)])
    if routing == "hot":
        return _draw_indices(1, G, S, K, E, 1.0, PRODUCTION_ROUTING[1])[0]
    idx = torch.zeros(G, S, K, dtype=torch.int64)
    for o in range(G):
        for t in range(S):
            g = torch.randperm(8)[:4]
            cand = (g.view(-1, 1) * EPC + torch.arange(EPC)).reshape(-1)
            idx[o, t] = cand[torch.randperm(4 * EPC)[:K]]
    return idx


def _table():
    """mask -> (cost, bitmask of the minimal arcs' forward hop counts a; b = cost - a)."""
    tab = []
    for m in range(128):
        dist = [d for d in range(1, G) if (m >> (d - 1)) & 1]
        if not dist:
            tab.append((0, 1))
            continue
        opts = [(a + max([G - d for d in dist if d > a], default=0), a) for a in range(G)]
        c = min(x for x, _ in opts)
        tab.append((c, sum(1 << a for x, a in opts if x == c)))
    return tab


TABLE = _table()


def _code(a, b, lf, lb):
    return a | (b << 3) | ((lf if a else 0) << 6) | ((lb if b else 0) << 7)


def reference(ids, algo):
    """Route codes [G * S] (row o * S + t) as the kernel computes them."""
    chip_of = [e // EPC for e in range(E)]
    idl = ids.tolist()

    def mask(o, t):
        bits = 0
        for e in idl[o][t]:
            bits |= 1 << chip_of[e]
        rot = ((bits >> o) | (bits << (G - o))) & 0xFF
        return rot >> 1

    codes = [0] * (G * S)
    if algo == "mask":
        for o in range(G):
            for t in range(S):
                codes[o * S + t] = mask(o, t)
        return codes

    def cands(m):
        c, bits = TABLE[m]
        return [(a, c - a) for a in range(G) if (bits >> a) & 1]

    def greedy_pick(cs, fwd_load, bwd_load):
        best = None
        for a, b in cs:
            for lf in range(L) if a else (0,):
                for lb in range(L) if b else (0,):
                    used = [fwd_load(lf, i) for i in range(a)] + [bwd_load(lb, i) for i in range(b)]
                    score = (max(used, default=0) + 1, sum(used))
                    if best is None or score < best[0]:
                        best = (score, a, b, lf, lb)
        return best[1:]

    if algo == "greedy_g":
        load = [[[0] * G for _ in range(L)] for _ in range(2)]
        for t in range(S):
            for o in range(G):
                m = mask(o, t)
                if m == 0:
                    continue
                a, b, lf, lb = greedy_pick(
                    cands(m), lambda l, i: load[0][l][(o + i) % G], lambda l, i: load[1][l][(o - i) % G]
                )
                for i in range(a):
                    load[0][lf][(o + i) % G] += 1
                for i in range(b):
                    load[1][lb][(o - i) % G] += 1
                codes[o * S + t] = _code(a, b, lf, lb)
        return codes

    for o in range(G):
        own = [[[0] * G for _ in range(L)] for _ in range(2)]
        cnt = [0, 0]
        for t in range(S):
            m = mask(o, t)
            if m == 0:
                continue
            cs = cands(m)
            if algo == "hash":
                h = (((o * S + t) * 2654435761) ^ (o * 40503)) & 0xFFFFFFFF
                a, b = cs[(((h >> 16) * len(cs)) >> 16)]
                lf, lb = (h >> 8) & 1, (h >> 9) & 1
            elif algo == "rr":
                a, b = cs[t % len(cs)]
                lf, lb = cnt[0] & 1, cnt[1] & 1
            else:  # greedy_o
                a, b, lf, lb = greedy_pick(cs, lambda l, i: own[0][l][i], lambda l, i: own[1][l][i])
            if a:
                cnt[0] += 1
            if b:
                cnt[1] += 1
            for i in range(a):
                own[0][lf][i] += 1
            for i in range(b):
                own[1][lb][i] += 1
            codes[o * S + t] = _code(a, b, lf, lb)
    return codes


def scan_reference(codes):
    """SCAN_PORT's row list: (page, wait) per row it sends, own rows first then origins outward (as the prototype)."""
    p, d, l = SCAN_PORT
    pages, waits = [], []
    up_pos = 0  # rows the upstream port sends before the current one, in its own order
    for k in range(G):
        o = (p - k) % G
        for t in range(S):
            c = codes[o * S + t]
            hops = c & 7 if d == 0 else (c >> 3) & 7
            lnk = (c >> 6) & 1 if d == 0 else (c >> 7) & 1
            if lnk != l:
                continue
            up_has = k > 0 and hops > k - 1  # the upstream port sends this row (its segment k - 1)
            if hops > k:
                pages.append(o * S + t)
                waits.append(0 if k == 0 else up_pos // INC_EVERY + 1)
            if up_has:
                up_pos += 1
    return pages, waits


_KERNEL = r"""
#include <cstdint>
#include "api/dataflow/dataflow_api.h"

constexpr uint32_t G = __G__, S = __S__, K = __K__, L = __L__, INC_EVERY = __INC__;
constexpr uint32_t ALGO = get_compile_time_arg_val(0);
constexpr uint32_t PROBE_LOADS = get_compile_time_arg_val(1);
constexpr uint32_t IDS_BYTES = get_compile_time_arg_val(2), CODES_BYTES = get_compile_time_arg_val(3);
constexpr uint32_t OUT_BYTES = get_compile_time_arg_val(4);
constexpr auto ids_args = TensorAccessorArgs<5>();
constexpr auto codes_args = TensorAccessorArgs<ids_args.next_compile_time_args_offset()>();
constexpr auto out_args = TensorAccessorArgs<codes_args.next_compile_time_args_offset()>();  // 0 mask 1 hash 2 rr 3 greedy_o 4 greedy_g 5 scan
constexpr uint16_t TABLE[128] = {__TABLE__};  // cost | arc bitmask << 4
constexpr uint8_t CHIP_OF[__E__] = {__CHIP_OF__};
constexpr uint32_t SCAN_P = __SP__, SCAN_D = __SD__, SCAN_L = __SL__;

inline uint32_t clk() { return *reinterpret_cast<volatile uint32_t*>(RISCV_DEBUG_REG_WALL_CLOCK_L); }

FORCE_INLINE uint32_t row_mask(const volatile tt_l1_ptr uint16_t* ids, uint32_t o, uint32_t t) {
    const volatile tt_l1_ptr uint16_t* r = ids + (o * S + t) * K;
    uint32_t bits = 0;
    for (uint32_t k = 0; k < K; ++k) {
        bits |= 1u << CHIP_OF[r[k]];
    }
    const uint32_t rot = ((bits >> o) | (bits << (G - o))) & 0xFF;
    return rot >> 1;
}

FORCE_INLINE uint8_t code(uint32_t a, uint32_t b, uint32_t lf, uint32_t lb) {
    return a | (b << 3) | ((a ? lf : 0) << 6) | ((b ? lb : 0) << 7);
}

// Every minimal arc and link pair, scored (busiest hop + 1, then total) on `fl`(l, i) / `bl`(l, i): the hop i loads.
template <typename FL, typename BL>
FORCE_INLINE uint8_t greedy_pick(uint32_t m, FL fl, BL bl, uint32_t& a_out, uint32_t& b_out, uint32_t& lf_out, uint32_t& lb_out) {
    const uint32_t cost = TABLE[m] & 0xF, arcs = TABLE[m] >> 4;
    uint32_t best_max = ~0u, best_sum = ~0u;
    for (uint32_t a = 0; a < G; ++a) {
        if (!((arcs >> a) & 1)) continue;
        const uint32_t b = cost - a;
        for (uint32_t lf = 0; lf < (a ? L : 1); ++lf) {
            for (uint32_t lb = 0; lb < (b ? L : 1); ++lb) {
                uint32_t mx = 0, sum = 0;
                for (uint32_t i = 0; i < a; ++i) { const uint32_t v = fl(lf, i); mx = v > mx ? v : mx; sum += v; }
                for (uint32_t i = 0; i < b; ++i) { const uint32_t v = bl(lb, i); mx = v > mx ? v : mx; sum += v; }
                mx += 1;
                if (mx < best_max || (mx == best_max && sum < best_sum)) {
                    best_max = mx; best_sum = sum; a_out = a; b_out = b; lf_out = lf; lb_out = lb;
                }
            }
        }
    }
    return code(a_out, b_out, lf_out, lb_out);
}

void kernel_main() {
    size_t arg = 0;
    // DRAM tensors, one page each, staged through this core's CBs (reserved first, then their write pointers):
    // ids; route codes, 1 byte per row (the scan reads the host's); out = [cycles, -, scan n, -][pages][waits]
    const auto ids_acc = TensorAccessor(ids_args, get_arg_val<uint32_t>(arg++), IDS_BYTES);
    const auto codes_acc = TensorAccessor(codes_args, get_arg_val<uint32_t>(arg++), CODES_BYTES);
    const auto out_acc = TensorAccessor(out_args, get_arg_val<uint32_t>(arg++), OUT_BYTES);
    cb_reserve_back(0, 1);
    cb_reserve_back(1, 1);
    cb_reserve_back(2, 1);
    const uint32_t ids_l1 = get_write_ptr(0);
    const uint32_t codes_l1 = get_write_ptr(1);
    const uint32_t out_l1 = get_write_ptr(2);
    noc_async_read(ids_acc.get_noc_addr(0), ids_l1, IDS_BYTES);
    if constexpr (ALGO == 5) {
        noc_async_read(codes_acc.get_noc_addr(0), codes_l1, CODES_BYTES);
    }
    noc_async_read_barrier();
    volatile tt_l1_ptr uint32_t* out = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(out_l1);
    volatile tt_l1_ptr uint8_t* codes = reinterpret_cast<volatile tt_l1_ptr uint8_t*>(codes_l1);
    const volatile tt_l1_ptr uint16_t* ids = reinterpret_cast<const volatile tt_l1_ptr uint16_t*>(ids_l1);

    uint32_t t0 = clk();
    out[1] = 0;

    if constexpr (ALGO == 6) {  // probe: where the buffers are and what the core sees, no lookups
        out[2] = ids_l1;
        out[3] = codes_l1;
        out[20] = out_l1;
        out[21] = reinterpret_cast<uint32_t>(&CHIP_OF[0]);
        out[22] = reinterpret_cast<uint32_t>(&TABLE[0]);
        for (uint32_t i = 0; i < 16; ++i) out[4 + i] = ids[i];
        if (PROBE_LOADS) {
            for (uint32_t i = 0; i < 4; ++i) out[23 + i] = TABLE[i + 1];
        }
    }

    t0 = clk();
    if constexpr (ALGO == 0) {
        for (uint32_t o = 0; o < G; ++o)
            for (uint32_t t = 0; t < S; ++t) codes[o * S + t] = row_mask(ids, o, t);
    } else if constexpr (ALGO == 1 || ALGO == 2 || ALGO == 3) {
        for (uint32_t o = 0; o < G; ++o) {
            uint16_t own[2][L][G] = {};
            uint32_t cnt0 = 0, cnt1 = 0;
            uint32_t modc[G + 1] = {};  // t % n for n = 1 .. G, kept incrementally (no division)
            for (uint32_t t = 0; t < S; ++t) {
                const uint32_t m = row_mask(ids, o, t);
                uint8_t c = 0;
                if (m != 0) {
                    const uint32_t cost = TABLE[m] & 0xF, arcs = TABLE[m] >> 4;
                    uint32_t a = 0, b = 0, lf = 0, lb = 0;
                    if constexpr (ALGO == 3) {
                        greedy_pick(m, [&](uint32_t l, uint32_t i) { return (uint32_t)own[0][l][i]; },
                                    [&](uint32_t l, uint32_t i) { return (uint32_t)own[1][l][i]; }, a, b, lf, lb);
                    } else {
                        const uint32_t n = __builtin_popcount(arcs);
                        uint32_t k;
                        if constexpr (ALGO == 1) {
                            const uint32_t h = ((o * S + t) * 2654435761u) ^ (o * 40503u);
                            k = ((h >> 16) * n) >> 16;
                            lf = (h >> 8) & 1;
                            lb = (h >> 9) & 1;
                        } else {
                            k = modc[n];
                            lf = cnt0 & 1;
                            lb = cnt1 & 1;
                        }
                        for (a = 0; a < G; ++a) {  // the k-th minimal arc
                            if ((arcs >> a) & 1) { if (k == 0) break; --k; }
                        }
                        b = cost - a;
                    }
                    if (a) ++cnt0;
                    if (b) ++cnt1;
                    for (uint32_t i = 0; i < a; ++i) ++own[0][lf][i];
                    for (uint32_t i = 0; i < b; ++i) ++own[1][lb][i];
                    c = code(a, b, lf, lb);
                }
                codes[o * S + t] = c;
                for (uint32_t n = 1; n <= G; ++n) { if (++modc[n] == n) modc[n] = 0; }
            }
        }
    } else if constexpr (ALGO == 4) {
        uint16_t load[2][L][G] = {};
        for (uint32_t t = 0; t < S; ++t) {
            for (uint32_t o = 0; o < G; ++o) {
                const uint32_t m = row_mask(ids, o, t);
                uint8_t c = 0;
                if (m != 0) {
                    uint32_t a = 0, b = 0, lf = 0, lb = 0;
                    c = greedy_pick(m, [&](uint32_t l, uint32_t i) { return (uint32_t)load[0][l][(o + i) & (G - 1)]; },
                                    [&](uint32_t l, uint32_t i) { return (uint32_t)load[1][l][(o - i) & (G - 1)]; }, a, b, lf, lb);
                    for (uint32_t i = 0; i < a; ++i) ++load[0][lf][(o + i) & (G - 1)];
                    for (uint32_t i = 0; i < b; ++i) ++load[1][lb][(o - i) & (G - 1)];
                }
                codes[o * S + t] = c;
            }
        }
    } else if constexpr (ALGO == 5) {  // scan: SCAN port's list from the route codes
        volatile tt_l1_ptr uint32_t* pages = out + 4;
        volatile tt_l1_ptr uint32_t* waits = pages + G * S;
        uint32_t n = 0, up_pos = 0;
        for (uint32_t k = 0; k < G; ++k) {
            const uint32_t o = (SCAN_P - k) & (G - 1);
            for (uint32_t t = 0; t < S; ++t) {
                const uint32_t c = codes[o * S + t];
                const uint32_t hops = SCAN_D == 0 ? (c & 7) : ((c >> 3) & 7);
                const uint32_t lnk = SCAN_D == 0 ? ((c >> 6) & 1) : ((c >> 7) & 1);
                if (lnk != SCAN_L) continue;
                if (hops > k) {
                    pages[n] = o * S + t;
                    waits[n] = k == 0 ? 0 : up_pos / INC_EVERY + 1;
                    ++n;
                }
                if (k > 0 && hops > k - 1) ++up_pos;
            }
        }
        out[2] = n;
    }
    if constexpr (ALGO != 6) {
        out[0] = clk() - t0;  // algorithm
    }
    if constexpr (ALGO == 6) {  // probe: the kernel's view of the out address, through the codes buffer
        volatile tt_l1_ptr uint32_t* dbg = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(codes_l1);
        const uint64_t a = out_acc.get_noc_addr(0);
        dbg[0] = (uint32_t)a;
        dbg[1] = (uint32_t)(a >> 32);
        dbg[2] = get_arg_val<uint32_t>(2);
        dbg[3] = (uint32_t)codes_acc.get_noc_addr(0);
        dbg[4] = out_l1;
        dbg[5] = codes_l1;
        dbg[6] = ids_l1;
        dbg[7] = ids[0] | (ids[1] << 16);
        noc_async_write(codes_l1, codes_acc.get_noc_addr(0), CODES_BYTES);
        noc_async_write_barrier();
        return;
    }
    noc_async_write(codes_l1, codes_acc.get_noc_addr(0), CODES_BYTES);
    noc_async_write(out_l1, out_acc.get_noc_addr(0), OUT_BYTES);
    noc_async_write_barrier();
}
"""


def _source():
    tab = ", ".join(str(c | (bits << 4)) for c, bits in TABLE)
    chip = ", ".join(str(e // EPC) for e in range(E))
    rep = {
        "__G__": G, "__S__": S, "__K__": K, "__L__": L, "__INC__": INC_EVERY, "__E__": E,
        "__TABLE__": tab, "__CHIP_OF__": chip, "__SP__": SCAN_PORT[0], "__SD__": SCAN_PORT[1], "__SL__": SCAN_PORT[2],
    }  # fmt: skip
    s = _KERNEL
    for k, v in rep.items():
        s = s.replace(k, str(v))
    return s


def _words64(nbytes):
    """uint32 words of a 64-byte aligned buffer holding nbytes (a DRAM page must be 64-byte aligned)."""
    return -(-nbytes // 64) * 16


def _single_page(mesh_device, words, torch_t=None):
    """A one-page DRAM tensor of `words` uint32 (the kernel reads / writes it whole through a TensorAccessor)."""
    t = torch.zeros(1, 1, 1, words, dtype=torch.int32)
    if torch_t is not None:
        flat = torch_t.reshape(-1)
        t.view(-1)[: flat.numel()] = flat
    return ttnn.from_torch(
        t, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT, device=mesh_device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )


@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
@pytest.mark.parametrize("routing", ["uniform", "hot", "grouped", "dense"])
@pytest.mark.parametrize("risc", ["RISCV_0", "RISCV_1"], ids=["brisc", "ncrisc"])
def test_route_algos(mesh_device, routing, risc):
    ids = _ids(routing)
    ids16 = ids.to(torch.int32).reshape(-1, 2)
    packed = (ids16[:, 0] | (ids16[:, 1] << 16)).to(torch.int32)  # two uint16 ids per uint32
    ids_w = _words64(G * S * K * 2)
    ids_t = _single_page(mesh_device, ids_w, packed)
    core = ttnn.CoreCoord(0, 0)
    cores = ttnn.CoreRangeSet([ttnn.CoreRange(core, core)])
    codes_w, out_w = _words64(G * S), _words64((4 + 2 * G * S) * 4)
    ids_bytes, codes_bytes, out_bytes = ids_w * 4, codes_w * 4, out_w * 4
    host_codes = {}
    lines = []
    for ai, algo in enumerate(ALGOS):
        if ALGO_FILTER and algo not in ALGO_FILTER.split(","):
            continue
        want = None if algo == "probe" else reference(ids, "greedy_g" if algo == "scan" else algo)
        if algo == "scan":
            codes_in = torch.tensor(want, dtype=torch.int32).to(torch.uint8).view(torch.int32)
            codes_t = _single_page(mesh_device, codes_w, codes_in)
        else:
            codes_t = _single_page(mesh_device, codes_w)
        out_t = _single_page(mesh_device, out_w)
        program = ttnn.ProgramDescriptor()
        program.cbs = [
            ttnn.CBDescriptor(
                total_size=sz, core_ranges=cores, format_descriptors=[ttnn.CBFormatDescriptor(i, ttnn.uint32, sz)]
            )
            for i, sz in enumerate((ids_bytes, codes_bytes, out_bytes))
        ]
        acc = [list(ttnn.TensorAccessorArgs(t).get_compile_time_args()) for t in (ids_t, codes_t, out_t)]
        rt = ttnn.RuntimeArgs()  # filled before the descriptor is built: KernelDescriptor keeps its own copy
        rt[core.x][core.y] = [int(ids_t.buffer_address()), int(codes_t.buffer_address()), int(out_t.buffer_address())]
        program.kernels = [
            ttnn.KernelDescriptor(
                kernel_source=_source(),
                source_type=ttnn.KernelDescriptor.SourceType.SOURCE_CODE,
                core_ranges=cores,
                compile_time_args=[ai, int(os.environ.get("SRA_PROBE_LOADS", "0")), ids_bytes, codes_bytes, out_bytes]
                + acc[0]
                + acc[1]
                + acc[2],
                runtime_args=rt,
                config=ttnn.DataMovementConfigDescriptor(
                    processor=getattr(ttnn.DataMovementProcessor, risc),
                    noc=ttnn.NOC.RISCV_0_default if risc == "RISCV_0" else ttnn.NOC.RISCV_1_default,
                ),
            )
        ]
        logger.info(
            f"ROUTE_ALGO io: rt {[hex(x) for x in rt[core.x][core.y]]} acc {acc} sizes {(ids_bytes, codes_bytes, out_bytes)}"
        )
        mesh_desc = ttnn.MeshProgramDescriptor()
        mesh_desc[ttnn.MeshCoordinateRange(ttnn.MeshCoordinate(0, 0), ttnn.MeshCoordinate(0, 0))] = program
        logger.info(f"ROUTE_ALGO start {algo} ({routing}, {risc})")
        for _ in range(3):  # 1 warm-up; the last run is reported (all write the same codes)
            ttnn.generic_op([ids_t, codes_t, out_t], mesh_desc)
            ttnn.synchronize_device(mesh_device)
        res = ttnn.to_torch(out_t).reshape(-1).to(torch.int64) & 0xFFFFFFFF
        cyc, load = int(res[0]), int(res[1])
        if algo == "probe":
            dbg = (ttnn.to_torch(codes_t).reshape(-1).to(torch.int64) & 0xFFFFFFFF)[:8].tolist()
            logger.info(
                f"ROUTE_ALGO probe kernel view: out noc addr lo 0x{dbg[0]:x} hi 0x{dbg[1]:x}, rt arg 2 0x{dbg[2]:x}, "
                f"codes noc addr lo 0x{dbg[3]:x}, out_l1 0x{dbg[4]:x} codes_l1 0x{dbg[5]:x} ids_l1 0x{dbg[6]:x} "
                f"ids[0..1] 0x{dbg[7]:x} (expected {ids.reshape(-1)[:2].tolist()})"
            )
            continue
            logger.info(
                f"ROUTE_ALGO probe host: ids 0x{int(ids_t.buffer_address()):x} codes 0x{int(codes_t.buffer_address()):x} "
                f"out 0x{int(out_t.buffer_address()):x}"
            )
            logger.info(
                f"ROUTE_ALGO probe: ids_l1=0x{int(res[2]):x} codes_l1=0x{int(res[3]):x} out_l1=0x{int(res[20]):x} "
                f"first ids seen {res[4:20].tolist()} expected {ids.reshape(-1)[:16].tolist()} | CHIP_OF at 0x{int(res[21]):x}, "
                f"TABLE at 0x{int(res[22]):x}, TABLE[1..4] {res[23:27].tolist()} expected {[c | (b << 4) for c, b in TABLE[1:5]]}"
            )
            continue
        if algo == "scan":
            pages, waits = scan_reference(want)
            n = int(res[2])
            got_p = res[4 : 4 + n].tolist()
            got_w = res[4 + G * S : 4 + G * S + n].tolist()
            assert (n, got_p, got_w) == (len(pages), pages, waits), f"{algo}/{routing}: scan list differs"
            per_chip = cyc
            note = f"{n} rows in the port's list"
        else:
            got = ttnn.to_torch(codes_t).reshape(-1).view(torch.uint8)[: G * S].to(torch.int64).tolist()
            bad = sum(1 for x, y in zip(got, want) if x != y)
            assert bad == 0, f"{algo}/{routing}: {bad} route codes differ from the Python model"
            per_chip = cyc if algo == "greedy_g" else cyc / G
            note = "every chip runs it over all rows" if algo == "greedy_g" else "per chip = its own 640 rows"
        logger.info(f"ROUTE_ALGO done {algo} ({routing}, {risc})")
        lines.append(
            f"ROUTE_ALGO {risc} {routing:8s} {algo:9s} total {cyc:9d} cyc = {cyc / CLOCK_MHZ:8.1f} us | per chip "
            f"{per_chip / CLOCK_MHZ:7.1f} us ({note})"
        )
    logger.info("\n" + "\n".join(lines))
