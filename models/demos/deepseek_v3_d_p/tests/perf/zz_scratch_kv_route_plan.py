# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Scratch: dispatch_prep's route words for Iva's sparse KV push (GLM-5.3, 8x4 torus, 1M prefix), one owner chip.

Input: the 32 requesters' want bitmaps over this owner's rows (owner-major, one 4 KB page per requester: 1024 words =
32768 rows; at 1,044,480 rows over 32 owners an owner holds 32640). Each core takes a word range, transposes the 32
requester words of every 32-row word into per-row destination masks (scatter of the set bits), drops the owner's own
bit and, for every row with a destination, computes dispatch_prep's order (X- or Y-first by fewer tree links on the
8x4 torus, ties by a hash) and link (hash bit) - the same span tables (_COMMON). Output: a compacted list per core
(count, then (mask, row | order << 16 | link << 17) pairs). One chip: the work is local, the mesh shape is compile-time.

Run: TT_METAL_DEVICE_PROFILER=1 scripts/run_safe_pytest.sh --device auto <this file> -s
"""
import os

import numpy as np
import pytest
import torch
from loguru import logger
from ttnn.bringup.combine_rf.sparse_all_gather_2d import _COMMON, ring_table, span_subs

import ttnn

R, C, NQ, WO = 8, 4, 32, 1024  # mesh, requesters, words per requester page (32768 rows)
OWNED = 1044480 // 32  # rows an owner holds at a 1,044,480-token prefix
ME = 13  # owner chip (row 3, column 1)
_KEY = "DEVICE KERNEL DURATION [ns]"

_KERNEL = (
    _COMMON
    + r"""
constexpr uint32_t ME = __ME__, NQ = 32, WO = __WO__;
constexpr uint32_t PROBE = __PROBE__;  // 0 full, 1 reads only, 2 no route (masks + compaction only)
constexpr auto in_args = TensorAccessorArgs<0>();
constexpr auto out_args = TensorAccessorArgs<in_args.next_compile_time_args_offset()>();
FORCE_INLINE uint32_t links(uint32_t s) { return (s & 15) + (s >> 4); }
constexpr uint32_t OPT = __OPT__, BW = 16;
uint32_t mt[OPT ? BW * 32 : 1];  // (local data memory)
// bits 4 r -> bit r (r < 8)
FORCE_INLINE uint32_t compress8(uint32_t x) {
    x &= 0x11111111u;
    x = (x | (x >> 3)) & 0x03030303u;
    x = (x | (x >> 6)) & 0x000F000Fu;
    return (x | (x >> 12)) & 0xFFu;
}

void kernel_main() {
    const uint32_t in_addr = get_arg_val<uint32_t>(0), out_addr = get_arg_val<uint32_t>(1);
    const uint32_t w0 = get_arg_val<uint32_t>(2), w1 = get_arg_val<uint32_t>(3), idx = get_arg_val<uint32_t>(4);
    const uint32_t nw = w1 - w0;
    const uint32_t bm_l1 = (get_write_ptr(0) + 63) & ~63u, out_l1 = bm_l1 + ((NQ * nw * 4 + 63) & ~63u);
    const auto in = TensorAccessor(in_args, in_addr, WO * 4);
    const auto out = TensorAccessor(out_args, out_addr, 64 + nw * 32 * 8);
    for (uint32_t r = 0; r < NQ; ++r) {
        noc_async_read(in.get_noc_addr(r) + w0 * 4, bm_l1 + r * nw * 4, nw * 4);
    }
    noc_async_read_barrier();
    invalidate_l1_cache();
    volatile tt_l1_ptr uint32_t* o = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(out_l1);
    uint32_t n = 0;
    if constexpr (OPT && PROBE != 1) {
        // blocks of BW words: each requester's BW words are consecutive in L1 (L0 line hits), the set bits scattered
        // into per-row masks in local memory; the route by nibble arithmetic (C = 4, R = 8) over set bits only
        static_assert(C == 4 && R == 8);
        const uint32_t* bm = reinterpret_cast<const uint32_t*>(bm_l1);
        const uint32_t r0 = ME / C, c0 = ME % C;
        for (uint32_t k = 0; k < BW * 32; ++k) mt[k] = 0;
        for (uint32_t i0 = 0; i0 < nw; i0 += BW) {
            uint32_t u[BW];
            for (uint32_t j = 0; j < BW; ++j) u[j] = 0;
            for (uint32_t r = 0; r < NQ; ++r) {
                const uint32_t* p = bm + r * nw + i0;
                const uint32_t bit = 1u << r;
                for (uint32_t j = 0; j < BW; ++j) {
                    uint32_t x = p[j];
                    u[j] |= x;
                    for (; x; x &= x - 1) mt[j * 32 + __builtin_ctz(x)] |= bit;
                }
            }
            for (uint32_t j = 0; j < BW; ++j) {
                for (uint32_t uu = u[j]; uu; uu &= uu - 1) {
                    const uint32_t t = __builtin_ctz(uu);
                    const uint32_t m = mt[j * 32 + t] & ~(1u << ME);
                    mt[j * 32 + t] = 0;
                    if (m == 0) continue;
                    const uint32_t row = (w0 + i0 + j) * 32 + t;
                    uint32_t flags = 0;
                    if constexpr (PROBE == 0) {
                        uint32_t x = m | (m >> 16);
                        x |= x >> 8;
                        x |= x >> 4;
                        const uint32_t cm = x & 15u, rm = compress8(m | (m >> 1) | (m >> 2) | (m >> 3));
                        uint32_t hx = links(xspan(c0, cm & ~(1u << c0), row)), hy = links(yspan(r0, rm & ~(1u << r0), row));
                        for (uint32_t cc = cm | (1u << c0); cc; cc &= cc - 1) {
                            hx += links(yspan(r0, compress8(m >> __builtin_ctz(cc)) & ~(1u << r0), row));
                        }
                        for (uint32_t rr = rm | (1u << r0); rr; rr &= rr - 1) {
                            hy += links(xspan(c0, (m >> (4 * __builtin_ctz(rr))) & 15u & ~(1u << c0), row));
                        }
                        const uint32_t hsh = row * 2654435761u;
                        const uint32_t order = hx < hy ? 0 : (hy < hx ? 1 : (hsh >> 31) & 1);
                        flags = order | (((hsh >> 29) & 1) << 1);
                    }
                    o[16 + 2 * n] = m;
                    o[16 + 2 * n + 1] = row | (flags << 16);
                    ++n;
                }
            }
        }
    } else if constexpr (PROBE != 1) {
        const uint32_t* bm = reinterpret_cast<const uint32_t*>(bm_l1);
        const uint32_t r0 = ME / C, c0 = ME % C;
        uint32_t a[NQ], mt[32];
        for (uint32_t t = 0; t < 32; ++t) mt[t] = 0;
        for (uint32_t i = 0; i < nw; ++i) {
            uint32_t u = 0;
            for (uint32_t r = 0; r < NQ; ++r) {
                a[r] = bm[r * nw + i];
                u |= a[r];
            }
            if (u == 0) continue;
            for (uint32_t r = 0; r < NQ; ++r) {  // transpose: bit r of mt[t] = bit t of a[r]
                for (uint32_t x = a[r]; x; x &= x - 1) mt[__builtin_ctz(x)] |= 1u << r;
            }
            for (; u; u &= u - 1) {
                const uint32_t t = __builtin_ctz(u);
                const uint32_t m = mt[t] & ~(1u << ME);
                mt[t] = 0;
                if (m == 0) continue;
                const uint32_t row = (w0 + i) * 32 + t;
                uint32_t flags = 0;
                if constexpr (PROBE == 0) {
                    const uint32_t cm = cols_of(m), rm = rows_of(m);
                    uint32_t hx = links(xspan(c0, cm & ~(1u << c0), row)), hy = links(yspan(r0, rm & ~(1u << r0), row));
                    for (uint32_t c = 0; c < C; ++c) {
                        if (((cm | (1u << c0)) >> c) & 1) hx += links(yspan(r0, rows_in_col(m, c) & ~(1u << r0), row));
                    }
                    for (uint32_t r = 0; r < R; ++r) {
                        if (((rm | (1u << r0)) >> r) & 1) hy += links(xspan(c0, cols_in_row(m, r) & ~(1u << c0), row));
                    }
                    const uint32_t hsh = row * 2654435761u;
                    const uint32_t order = hx < hy ? 0 : (hy < hx ? 1 : (hsh >> 31) & 1);
                    flags = order | (((hsh >> 29) & 1) << 1);
                }
                o[16 + 2 * n] = m;
                o[16 + 2 * n + 1] = row | (flags << 16);
                ++n;
            }
        }
    }
    o[0] = n;
    noc_async_write(out_l1, out.get_noc_addr(idx), 64 + n * 8);
    noc_async_write_barrier();
}
"""
)


def _src(probe, opt=0):
    src = _KERNEL
    for k, v in span_subs(R, C, True, True).items():
        src = src.replace(k, v)
    subs = dict(
        __R__=R,
        __C__=C,
        __S__=WO * 32,
        __GMAX__=R + C - 3,
        __RW__=2,
        __ME__=ME,
        __WO__=WO,
        __PROBE__=probe,
        __OPT__=opt,
    )
    for k, v in subs.items():
        src = src.replace(k, str(v))
    return src


def _prepare(dev, n_cores, probe, inp, out, opt=0):
    grid = dev.compute_with_storage_grid_size()
    nw = WO // n_cores
    cores = [ttnn.CoreCoord(i % grid.x, i // grid.x) for i in range(n_cores)]
    cset = ttnn.CoreRangeSet([ttnn.CoreRange(c, c) for c in cores])
    l1 = NQ * nw * 4 + 64 + 64 + nw * 32 * 8 + 64
    rt = ttnn.RuntimeArgs()
    for i, c in enumerate(cores):
        rt[c.x][c.y] = [int(inp.buffer_address()), int(out.buffer_address()), i * nw, (i + 1) * nw, i]
    program = ttnn.ProgramDescriptor()
    program.cbs = [
        ttnn.CBDescriptor(
            total_size=l1, core_ranges=cset, format_descriptors=[ttnn.CBFormatDescriptor(0, ttnn.uint32, l1)]
        )
    ]
    program.kernels = [
        ttnn.KernelDescriptor(
            kernel_source=_src(probe, opt),
            source_type=ttnn.KernelDescriptor.SourceType.SOURCE_CODE,
            core_ranges=cset,
            compile_time_args=list(ttnn.TensorAccessorArgs(inp).get_compile_time_args())
            + list(ttnn.TensorAccessorArgs(out).get_compile_time_args()),
            runtime_args=rt,
            config=ttnn.DataMovementConfigDescriptor(
                processor=ttnn.DataMovementProcessor.RISCV_0, noc=ttnn.NOC.RISCV_0_default
            ),
        )
    ]
    return program


# ---- host reference (dispatch_prep's order rule, in Python)
_XR, _YR = ring_table(C, True), ring_table(R, True)


def _span(n, table, pos, m, t):
    m &= ~(1 << pos) & ((1 << n) - 1)
    if m == 0:
        return 0
    rot = ((m >> pos) | (m << (n - pos))) & ((1 << n) - 1)
    return table[rot * 2 + ((((t * 2654435761) & 0xFFFFFFFF) >> 30) & 1)]


def _links(s):
    return (s & 15) + (s >> 4)


def _route(m, row):
    r0, c0 = ME // C, ME % C
    cim = lambda r: (m >> (r * C)) & ((1 << C) - 1)  # noqa: E731
    cm = 0
    for r in range(R):
        cm |= cim(r)
    rm = sum(1 << r for r in range(R) if cim(r))
    ric = lambda c: sum(((m >> (r * C + c)) & 1) << r for r in range(R))  # noqa: E731
    hx = _links(_span(C, _XR, c0, cm & ~(1 << c0), row))
    hy = _links(_span(R, _YR, r0, rm & ~(1 << r0), row))
    for c in range(C):
        if ((cm | (1 << c0)) >> c) & 1:
            hx += _links(_span(R, _YR, r0, ric(c) & ~(1 << r0), row))
    for r in range(R):
        if ((rm | (1 << r0)) >> r) & 1:
            hy += _links(_span(C, _XR, c0, cim(r) & ~(1 << c0), row))
    hsh = (row * 2654435761) & 0xFFFFFFFF
    order = 0 if hx < hy else (1 if hy < hx else (hsh >> 31) & 1)
    return order | (((hsh >> 29) & 1) << 1)


# ---- synthetic want bitmaps (Iva's 1M statistics: an owner's rows wanted by anyone ~5.5k of 32640, ~5.5 requesters)
def _masks(case, rng):
    m = np.zeros(WO * 32, dtype=np.uint64)
    if case == "empty":
        return m
    if case == "all":
        m[:OWNED] = (1 << 32) - 1
        return m
    union, k_mean, run = dict(median=(5500, 5.5, 1), median_runs=(5500, 5.5, 16), heavy=(9000, 6.5, 16))[case]
    rows = []
    while len(rows) < union:  # runs of wanted rows (top-k clusters), geometric lengths of mean `run`
        s = int(rng.integers(0, OWNED))
        rows.extend(range(s, min(OWNED, s + int(rng.geometric(1 / run)))))
    rows = np.unique(np.array(rows[:union]))
    for r in rows:
        k = min(NQ, 1 + int(rng.poisson(k_mean - 1)))
        m[r] = int(
            np.bitwise_or.reduce(np.left_shift(np.uint64(1), rng.choice(NQ, k, replace=False).astype(np.uint64)))
        )
    return m


def _bitmaps(m):
    """[NQ, WO] uint32: bit t of word w of requester q = row 32 w + t wanted by q."""
    bits = ((m[None, :] >> np.arange(NQ, dtype=np.uint64)[:, None]) & 1).astype(np.uint64)  # [NQ, rows]
    words = (bits.reshape(NQ, WO, 32) << np.arange(32, dtype=np.uint64)).sum(axis=2)
    return words.astype(np.uint32)


def _times(dev):
    ttnn.ReadDeviceProfiler(dev)
    out = []
    for programs in (ttnn.get_latest_programs_perf_data() or {}).values():
        out += [
            float(p.program_analyses_results[_KEY].duration) / 1e3
            for p in programs
            if _KEY in (p.program_analyses_results or {})
        ]
    return out


@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
def test_kv_route_plan(mesh_device):
    rng = np.random.default_rng(0)
    common = dict(layout=ttnn.ROW_MAJOR_LAYOUT, device=mesh_device, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    cases = os.environ.get("KVR_CASES", "empty,median,median_runs,heavy,all").split(",")
    cores = [int(x) for x in os.environ.get("KVR_CORES", "1,4,8,16,32,64").split(",")]
    opt = int(os.environ.get("KVR_OPT", "0"))
    results = {}
    for case in cases:
        m = _masks(case, rng)
        wanted = int(np.count_nonzero(m & ~np.uint64(1 << ME)))
        setbits = int(sum(bin(int(x)).count("1") for x in m[m != 0])) if case != "all" else OWNED * 32
        inp = ttnn.from_torch(
            torch.from_numpy(_bitmaps(m).view(np.int32)).reshape(1, 1, NQ, WO), dtype=ttnn.uint32, **common
        )
        for n in cores:
            nw = WO // n
            out = ttnn.from_torch(torch.zeros(1, 1, n, 16 + nw * 64, dtype=torch.int32), dtype=ttnn.uint32, **common)
            row = []
            for probe in (0, 1, 2):
                prog = _prepare(mesh_device, n, probe, inp, out, opt)
                mesh = ttnn.MeshProgramDescriptor()
                mesh[ttnn.MeshCoordinateRange(ttnn.MeshCoordinate(0, 0), ttnn.MeshCoordinate(0, 0))] = prog
                _times(mesh_device)
                best = None
                for _ in range(5):
                    ttnn.generic_op([inp, out], mesh)
                    ttnn.synchronize_device(mesh_device)
                    t = _times(mesh_device)
                    best = t[-1] if best is None else min(best, t[-1])
                row.append(best)
                if probe == 0:  # exactness vs the host reference
                    o = ttnn.to_torch(out).numpy().view(np.uint32).reshape(n, 16 + nw * 64)
                    got = []
                    for i in range(n):
                        cnt = int(o[i, 0])
                        got += [
                            (int(o[i, 16 + 2 * j + 1]) & 0xFFFF, int(o[i, 16 + 2 * j]), int(o[i, 16 + 2 * j + 1]) >> 16)
                            for j in range(cnt)
                        ]
                    nz = np.nonzero(m & ~np.uint64(1 << ME))[0]
                    if case == "all":  # (spot check: the full reference is 32k Python routes)
                        nz = nz[:: max(1, len(nz) // 2000)]
                        got = [g for g in got if g[0] in set(nz.tolist())]
                    want = [(int(r), int(m[r]) & ~(1 << ME), _route(int(m[r]) & ~(1 << ME), int(r))) for r in nz]
                    assert sorted(got) == want, f"{case} n={n}: mismatch ({len(got)} vs {len(want)} rows)"
            results[(case, n)] = row
            logger.info(
                f"KVROUTE opt{opt} {case:12s} wanted {wanted:6d} set bits {setbits:7d} cores {n:3d}: "
                f"full {row[0]:7.1f} us  reads only {row[1]:7.1f}  no route {row[2]:7.1f}"
            )
    logger.info("KVROUTE summary (us: full / reads only / masks+compaction, no route)")
    for case in cases:
        logger.info(
            f"KVROUTE opt{opt} {case:12s} "
            + "  ".join(
                f"{n}c {results[(case, n)][0]:.1f}/{results[(case, n)][1]:.1f}/{results[(case, n)][2]:.1f}"
                for n in cores
            )
        )
