// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Stencil matrix generation (setup, one launch per level). Each core writes its own rows of the 27-slot matrix
// values V and window byte offsets OFF (has_b: and of the right-hand side b) straight into DRAM, in the storage
// order of reference/problem.py: the tables the host would otherwise build and upload.
//   value of slot j: 26 on the diagonal, -1 for a neighbour inside the grid, 0 for a missing neighbour (whose column
//   is the row itself); padding rows: 1 on the diagonal, 0 elsewhere.
//   offset of column col: l1 + (col - lo) * 4 of the core's window piece (lo, l1) that holds col.
//   b = 26 - (neighbours - 1) on real rows, 0 on padding rows.
//   zero vectors (up to two, e.g. the initial guess x = 0 and CG's p): 0 on every row.
// Both data-movement RISCs run this kernel; RISC h takes the page-sized chunks of the core's rows with index % 2 == h.
// Everything that depends on the grid comes in runtime arguments, so one compiled binary serves every level and grid
// size (setup then only loads it from the kernel cache).
// META row per core (uint32): [0] nx [1] nbands [2] nl [3] pages per slot [4] R window pieces [5] unused
//   [6 .. 22) own row range (a, e) per colour block  [22 .. 30) end of each block's real rows
//   [30 .. 36) word offsets of: line_band, band_y0, band_ny, band_start, piece lo, piece l1 (layout: Geo in
//   host/native_setup.cpp)
#include "api/dataflow/dataflow_api.h"
#include <cstdint>

namespace {
constexpr uint32_t F_MINUS1 = 0xBF800000u, F_26 = 0x41D00000u, F_1 = 0x3F800000u;
constexpr uint32_t SET = 55 * 4096;  // one staging set: 27 offset pages, 27 value pages, one b page
constexpr uint32_t XCAP = 2048;      // value exceptions recorded per staging set (CB_X0 / CB_X1: 2 x 8 KB)

// The value pages of a staging set hold the interior template (-1, and 26 in slot 13) between chunks; a chunk writes
// only its exceptions (0 for a missing neighbour; padding rows' 0 / 1) and records them as (slot, word, count), so
// they are put back to the template once the chunk's writes have left L1. Most rows have no exception, so this halves
// the stores per row. Past XCAP exceptions the whole template is rewritten instead.
struct Exc {
    uint32_t* e;
    uint32_t n;
    bool full;
    void add(uint32_t j, uint32_t w0, uint32_t cnt) {
        if (cnt == 0) {
            return;
        }
        if (n < XCAP) {
            e[n++] = (j << 22) | (w0 << 11) | cnt;
        } else {
            full = true;
        }
    }
};
inline uint32_t tmpl(uint32_t j) { return j == 13 ? F_26 : F_MINUS1; }
void fill_template(uint32_t* Vv) {
    for (uint32_t j = 0; j < 27; ++j) {
        const uint32_t v = tmpl(j);
        for (uint32_t w = 0; w < 1024; ++w) {
            Vv[j * 1024 + w] = v;
        }
    }
}
void restore(uint32_t* Vv, Exc& x) {  // the set's previous chunk has left L1: put its exceptions back
    if (x.full) {
        fill_template(Vv);
    } else {
        for (uint32_t i = 0; i < x.n; ++i) {
            const uint32_t j = x.e[i] >> 22, w0 = (x.e[i] >> 11) & 2047, cnt = x.e[i] & 2047, v = tmpl(j);
            for (uint32_t t = 0; t < cnt; ++t) {
                Vv[j * 1024 + w0 + t] = v;
            }
        }
    }
    x.n = 0;
    x.full = false;
}

uint32_t f32_bits(uint32_t n) {  // fp32 bits of a small non-negative integer (exact)
    if (n == 0) {
        return 0;
    }
    const uint32_t e = 31 - __builtin_clz(n);
    return ((127 + e) << 23) | ((n << (23 - e)) & 0x7FFFFFu);
}

struct Geo {
    const uint32_t* m;  // read once from DRAM before init (after invalidate_l1_cache), then constant
    uint32_t nx, nb, nl, R;
    const uint32_t *lb, *by0, *bny, *bst, *plo, *pl1;
    void init(uint32_t addr) {
        m = (const uint32_t*)addr;
        nx = m[0];
        nb = m[1];
        nl = m[2];
        R = m[4];
        lb = m + m[30];
        by0 = m + m[31];
        bny = m + m[32];
        bst = m + m[33];
        plo = m + m[34];
        pl1 = m + m[35];
    }
    uint32_t cx(uint32_t p) const { return (nx - p + 1) / 2; }
    // q of the first point of grid line (y, z) in colour block cs
    uint32_t line_q(uint32_t cs, uint32_t y, uint32_t z) const {
        const uint32_t k = cs * nb + lb[(y & 1) * nl + (y >> 1)];
        return bst[k] + ((z >> 1) * bny[k] + ((y >> 1) - by0[k])) * cx(cs & 1);
    }
    // l1 - 4 lo of the window piece holding q (the last piece with lo <= q; pieces are sorted)
    uint32_t piece_base(uint32_t q) const {
        uint32_t lo = 0, hi = R;
        while (hi - lo > 1) {
            const uint32_t mid = (lo + hi) / 2;
            if (plo[mid] <= q) {
                lo = mid;
            } else {
                hi = mid;
            }
        }
        return pl1[lo] - 4 * plo[lo];
    }
};

// rows [q, q + len) of colour block c: one run along x at line (iy, iz), x index i0 .. i0 + len - 1 in the block
void fill_run(
    const Geo& g,
    uint32_t c,
    uint32_t q,
    int32_t i0,
    int32_t len,
    int32_t iy,
    int32_t iz,
    uint32_t* O,
    uint32_t* Vv,
    uint32_t* Bw,
    Exc& x) {
    const int32_t px = c & 1, nx = (int32_t)g.nx;
    const uint32_t sb = g.piece_base(q);  // the core's own rows sit in one window piece
    uint32_t K[27];                       // offset of slot j at x index i: K[j] + 4 i
    int32_t ilo[27], ihi[27];             // x indices whose slot-j neighbour is inside the grid
    for (int32_t sz = -1; sz <= 1; ++sz) {
        for (int32_t sy = -1; sy <= 1; ++sy) {
            const int32_t jy = iy + sy, jz = iz + sz;
            const bool line_ok = jy >= 0 && jy < nx && jz >= 0 && jz < nx;
            for (int32_t sx = -1; sx <= 1; ++sx) {
                const int32_t j = (sz + 1) * 9 + (sy + 1) * 3 + (sx + 1);
                ilo[j] = 1;
                ihi[j] = 0;
                K[j] = 0;
                if (!line_ok) {
                    continue;
                }
                // neighbour x = 2 i + px + sx, in block cs (x parity flips with sx = +-1) at x index i + d
                const int32_t d = (px + sx) < 0 ? -1 : (px + sx) >> 1;
                const uint32_t cs = (uint32_t)(px ^ (sx != 0)) + 2 * (jy & 1) + 4 * (jz & 1);
                ilo[j] = (px + sx) < 0 ? 1 : 0;
                ihi[j] = (nx - 1 - px - sx) >> 1;
                const int32_t f = i0 > ilo[j] ? i0 : ilo[j];
                const int32_t l = (i0 + len - 1) < ihi[j] ? (i0 + len - 1) : ihi[j];
                if (f > l) {
                    continue;  // no row of the run has this neighbour
                }
                const uint32_t lq = g.line_q(cs, jy, jz) + d;  // column of x index i: lq + i
                K[j] = g.piece_base(lq + f) + 4 * lq;
            }
        }
    }
    // slot by slot over the whole run: straight stores, then the rows at the run's ends whose neighbour is missing
    const uint32_t w0 = q & 1023;
    const uint32_t self0 = sb + 4 * q;  // offset of the run's first row; row t: self0 + 4 t
    for (int32_t j = 0; j < 27; ++j) {
        uint32_t* o = O + j * 1024 + w0;
        uint32_t* v = Vv + j * 1024 + w0;
        int32_t a = ilo[j] - i0, e = ihi[j] - i0 + 1;  // rows [a, e) of the run have this neighbour
        a = a < 0 ? 0 : (a > len ? len : a);
        e = e < a ? a : (e > len ? len : e);
        uint32_t off = K[j] + 4 * (i0 + a), self = self0 + 4 * a;
        for (int32_t t = a; t < e; ++t, off += 4) {
            o[t] = off;  // the value is the template's
        }
        for (int32_t t = 0; t < a; ++t) {  // missing neighbour: a zero entry on the row itself
            o[t] = self0 + 4 * t;
            v[t] = 0;
        }
        self = self0 + 4 * e;
        for (int32_t t = e; t < len; ++t, self += 4) {
            o[t] = self;
            v[t] = 0;
        }
        x.add(j, w0, a);
        x.add(j, w0 + e, len - e);
    }
    // b = 26 - (neighbours - 1): lines inside the grid times x neighbours inside the grid
    int32_t lines = 0;
    for (int32_t j = 1; j < 27; j += 3) {
        lines += ilo[j] <= ihi[j];  // the sx = 0 slot of each line: valid exactly when the line is
    }
    for (int32_t t = 0; t < len; ++t) {
        const int32_t x = 2 * (i0 + t) + px;
        const int32_t nxn = 1 + (x > 0) + (x < nx - 1);
        Bw[w0 + t] = f32_bits(27 - (uint32_t)(lines * nxn));
    }
}

// rows [q0, q1) of colour block c (one DRAM page at most) into the staging set at base
void fill_chunk(const Geo& g, uint32_t c, uint32_t q0, uint32_t q1, uint32_t real_end, uint32_t base, Exc& x) {
    // plain (not volatile) stores so the compiler can schedule the fill loops; kernel_main puts a compiler memory
    // barrier between the fill and the NoC writes that read these pages
    uint32_t* O = (uint32_t*)base;
    uint32_t* Vv = O + 27 * 1024;
    uint32_t* Bw = O + 54 * 1024;
    const uint32_t py = (c >> 1) & 1, pz = c >> 2, cxn = g.cx(c & 1), k0 = c * g.nb;
    const uint32_t end = q1 < real_end ? q1 : real_end;
    uint32_t jb = 0;
    for (uint32_t q = q0; q < end;) {
        while (jb + 1 < g.nb && g.bst[k0 + jb + 1] <= q) {
            ++jb;  // the last band starting at or before q (bands without lines share the next band's start)
        }
        const uint32_t k = k0 + jb, l = q - g.bst[k], line = l / cxn, i0 = l % cxn;
        const uint32_t len = (cxn - i0) < (end - q) ? (cxn - i0) : (end - q);
        const uint32_t iy = 2 * (g.by0[k] + line % g.bny[k]) + py, iz = 2 * (line / g.bny[k]) + pz;
        fill_run(g, c, q, (int32_t)i0, (int32_t)len, (int32_t)iy, (int32_t)iz, O, Vv, Bw, x);
        q += len;
    }
    const uint32_t p0 = q0 > real_end ? q0 : real_end;
    if (p0 < q1) {  // padding rows: every column is the row itself, diagonal 1
        const uint32_t sb = g.piece_base(p0);
        for (uint32_t q = p0; q < q1; ++q) {
            const uint32_t w = q & 1023;
            for (uint32_t j = 0; j < 27; ++j) {
                O[j * 1024 + w] = sb + 4 * q;
                Vv[j * 1024 + w] = j == 13 ? F_1 : 0;
            }
            Bw[w] = 0;
        }
        for (uint32_t j = 0; j < 27; ++j) {
            x.add(j, p0 & 1023, q1 - p0);
        }
    }
}
}  // namespace

void kernel_main() {
    const uint32_t v_addr = get_arg_val<uint32_t>(0), off_addr = get_arg_val<uint32_t>(1);
    const uint32_t meta_addr = get_arg_val<uint32_t>(2), b_addr = get_arg_val<uint32_t>(3);
    const uint32_t row = get_arg_val<uint32_t>(4), half = get_arg_val<uint32_t>(5);
    const uint32_t meta_pages = get_arg_val<uint32_t>(6), has_b = get_arg_val<uint32_t>(7);
    const uint32_t nzero = get_arg_val<uint32_t>(8);  // vectors to zero over this core's rows (0, 1 or 2)
    constexpr auto av = TensorAccessorArgs<0>();
    constexpr auto ao = TensorAccessorArgs<av.next_compile_time_args_offset()>();
    constexpr auto am = TensorAccessorArgs<ao.next_compile_time_args_offset()>();
    constexpr auto ab = TensorAccessorArgs<am.next_compile_time_args_offset()>();
    const auto V = TensorAccessor(av, v_addr, 4096);
    const auto OFF = TensorAccessor(ao, off_addr, 4096);
    const auto META = TensorAccessor(am, meta_addr, 4096);  // this core's row: pages row * meta_pages onwards
    const auto Bv = TensorAccessor(ab, b_addr, 4096);
    constexpr auto az0 = TensorAccessorArgs<ab.next_compile_time_args_offset()>();
    constexpr auto az1 = TensorAccessorArgs<az0.next_compile_time_args_offset()>();
    const auto Z0 = TensorAccessor(az0, get_arg_val<uint32_t>(9), 4096);
    const auto Z1 = TensorAccessor(az1, get_arg_val<uint32_t>(10), 4096);
    const uint32_t zpage = get_write_ptr(CB_Z);  // 4 KB of zeros, the source of the zero vectors' writes
    if (nzero) {
        const uint64_t zsrc = get_noc_addr(MEM_ZEROS_BASE);
        for (uint32_t o = 0; o < 4096; o += MEM_ZEROS_SIZE) {
            noc_async_read(zsrc, zpage + o, MEM_ZEROS_SIZE);
        }
        noc_async_read_barrier();
    }
    const uint32_t meta = get_write_ptr(half ? CB_M1 : CB_M0);
    for (uint32_t p = 0; p < meta_pages; ++p) {
        noc_async_read(META.get_noc_addr(row * meta_pages + p), meta + p * 4096, 4096);
    }
    noc_async_read_barrier();
    invalidate_l1_cache();
    Geo g;
    g.init(meta);
    const uint32_t* m = g.m;
    const uint32_t ntlv = m[3];
    const uint32_t stage = get_write_ptr(half ? CB_S1 : CB_S0);
    const uint32_t xbuf = get_write_ptr(half ? CB_X1 : CB_X0);
    Exc xs[2] = {{(uint32_t*)xbuf, 0, false}, {(uint32_t*)xbuf + XCAP, 0, false}};
    fill_template((uint32_t*)(stage + 27 * 4096));
    fill_template((uint32_t*)(stage + SET + 27 * 4096));
    uint32_t set = 0, chunk = 0;
    for (uint32_t c = 0; c < 8; ++c) {
        const uint32_t a = m[6 + 2 * c], e = m[7 + 2 * c], real_end = m[22 + c];
        for (uint32_t q0 = a; q0 < e;) {
            const uint32_t q1 = (q0 / 1024 + 1) * 1024 < e ? (q0 / 1024 + 1) * 1024 : e;
            if ((chunk++ & 1) == half) {
                const uint32_t base = stage + set * SET;
                // this set's previous chunk was written out before the last barrier: restore its value template
                restore((uint32_t*)(base + 27 * 4096), xs[set]);
                fill_chunk(g, c, q0, q1, real_end, base, xs[set]);
                asm volatile("" ::: "memory");  // every staging store is issued before the NoC reads the pages
                noc_async_write_barrier();  // the other set's writes have left L1 before it is filled again
                const uint32_t t = q0 >> 10, wb = (q0 & 1023) * 4, nb = (q1 - q0) * 4;
                for (uint32_t j = 0; j < 27; ++j) {
                    noc_async_write(base + j * 4096 + wb, OFF.get_noc_addr(j * ntlv + t) + wb, nb);
                    noc_async_write(base + (27 + j) * 4096 + wb, V.get_noc_addr(j * ntlv + t) + wb, nb);
                }
                if (has_b) {
                    noc_async_write(base + 54 * 4096 + wb, Bv.get_noc_addr(t) + wb, nb);
                }
                if (nzero > 0) {
                    noc_async_write(zpage + wb, Z0.get_noc_addr(t) + wb, nb);
                }
                if (nzero > 1) {
                    noc_async_write(zpage + wb, Z1.get_noc_addr(t) + wb, nb);
                }
                set ^= 1;
            }
            q0 = q1;
        }
    }
    noc_async_write_barrier();
}
