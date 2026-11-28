// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Matrix generation (setup). One launch runs a list of jobs; each core writes its own rows of each job's tables
// straight into DRAM, in the storage order of reference/problem.py: the tables the host would otherwise build and
// upload.
// Stencil job (kind 0): the 27-slot matrix values V and window byte offsets OFF of one level (has_b: and the
// right-hand side b; nzero: and zero vectors):
//   value of slot j: 26 on the diagonal, -1 for a neighbour inside the grid, 0 for a missing neighbour (whose column
//   is the row itself); padding rows: 1 on the diagonal, 0 elsewhere.
//   offset of column col: l1 + (col - lo) * 4 of the core's window piece (lo, l1) that holds col.
//   b = 26 - (neighbours - 1) on real rows, 0 on padding rows; zero vectors: 0 on every row.
// Transfer job (kind 1 restriction, 2 prolongation; host/native_setup.cpp's for_each_transfer_row): the 1-slot
//   weights (1 on real rows, 0 on padding rows) and window byte offsets of a grid transfer launch. Restriction: the
//   column of coarse point (x, y, z) is fine point (2x, 2y, 2z); prolongation: the column of fine point
//   (2x, 2y, 2z) is coarse point (x, y, z); a padding row takes the column of its block's last real row.
// Both data-movement RISCs run this kernel; RISC h takes the page-sized chunks of the core's rows with index % 2 == h.
// Everything that depends on the grid comes in runtime arguments, so one compiled binary serves every job and grid
// size (setup then only loads it from the kernel cache).
// The offsets of most slots are written by the core's TRISCs (generate_trisc.cpp, fill.h): this RISC describes each
// run of a chunk, hands slots DMSLOTS .. 26 to them, writes slots 0 .. DMSLOTS - 1 and the rest of the chunk itself,
// and waits for them before its NoC writes read the pages.
// Common arguments: [0] grid width [1] h [2] META address [3] META shared pages [4] META pages per core [5] jobs
//   [6] the launch's nonce (fill.h) [7], [8] per core row y, a nibble: how many of every 8 chunks RISC 1 takes,
//   then per job (10 words): kind, cores in the job, word offset of the job's META segment, V, OFF,
//   b, has_b, zero vectors, zero 0, zero 1.
// META: shared pages (every level's storage-order tables, read by every core), then each core's row of segments; a
//   core reads both into one L1 buffer, and every word offset below counts from its start.
// A job's META segment per core (uint32): [0] nx [1] nbands [2] nl [3] pages per slot [4] R window pieces [5] unused
//   [6 .. 22) own row range (a, e) per out block  [22 .. 30) end of each block's real rows
//   [30 .. 36) word offsets of: line_band, band_y0, band_ny, band_start, piece lo, piece l1 (layout: Geo in
//   host/native_setup.cpp). Transfer jobs: [0 .. 3) and [30 .. 34) describe the fine level, then
//   [36 .. 39) the coarse level's nx, nbands, nl, [39 .. 43) its table offsets, [43 .. 51) each block's padding
//   column.
#include "api/dataflow/dataflow_api.h"
#include "fill.h"
#include <cstdint>

namespace {
constexpr uint32_t F_MINUS1 = 0xBF800000u, F_26 = 0x41D00000u, F_1 = 0x3F800000u;
// one staging set: 27 offset pages, 27 value pages, one b page (SET_PAGES and XCAP: tt/shared_constants.h)
constexpr uint32_t SET = SET_PAGES * 4096;

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
    void init(const uint32_t* base, uint32_t addr) {  // base: the L1 META buffer (offsets count from it)
        m = (const uint32_t*)addr;
        nx = m[0];
        nb = m[1];
        nl = m[2];
        R = m[4];
        lb = base + m[30];
        by0 = base + m[31];
        bny = base + m[32];
        bst = base + m[33];
        plo = base + m[34];
        pl1 = base + m[35];
    }
    void init_coarse(const uint32_t* base, uint32_t addr) {  // a transfer job's coarse level: [36 ..) of the segment
        m = (const uint32_t*)addr;
        nx = m[36];
        nb = m[37];
        nl = m[38];
        R = 0;
        lb = base + m[39];
        by0 = base + m[40];
        bny = base + m[41];
        bst = base + m[42];
        plo = pl1 = nullptr;
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
    Exc& x,
    uint32_t* desc) {  // the run's description for the TRISCs (fill.h), or nullptr: write every slot here
    const int32_t px = c & 1, nx = (int32_t)g.nx;
    const uint32_t sb = g.piece_base(q);  // the core's own rows sit in one window piece
    uint32_t K[27];                       // offset of slot j at x index i: K[j] + 4 i
    int32_t ilo[27], ihi[27];             // x indices whose slot-j neighbour is inside the grid
    for (int32_t sz = -1; sz <= 1; ++sz) {
        for (int32_t sy = -1; sy <= 1; ++sy) {
            const int32_t jy = iy + sy, jz = iz + sz;
            const bool line_ok = jy >= 0 && jy < nx && jz >= 0 && jz < nx;
            // the neighbour line in both x parities (sx = 0: the run's own parity; sx = +-1: the other one). A run's
            // columns in one of them are one contiguous range (planned as one window piece), so its window base is
            // looked up once per parity and shared by sx = -1 and +1
            const uint32_t c0 = (uint32_t)px + 2 * (jy & 1) + 4 * (jz & 1);
            uint32_t lqs[2] = {0, 0}, base[2] = {0, 0};
            bool have[2] = {false, false};
            if (line_ok) {
                lqs[0] = g.line_q(c0, jy, jz);
                lqs[1] = g.line_q(c0 ^ 1, jy, jz);
            }
            for (int32_t sx = -1; sx <= 1; ++sx) {
                const int32_t j = (sz + 1) * 9 + (sy + 1) * 3 + (sx + 1);
                ilo[j] = 1;
                ihi[j] = 0;
                K[j] = 0;
                if (!line_ok) {
                    continue;
                }
                // neighbour x = 2 i + px + sx, in the block of parity px ^ (sx != 0), at x index i + d
                const int32_t d = (px + sx) < 0 ? -1 : (px + sx) >> 1;
                ilo[j] = (px + sx) < 0 ? 1 : 0;
                ihi[j] = (nx - 1 - px - sx) >> 1;
                const int32_t f = i0 > ilo[j] ? i0 : ilo[j];
                const int32_t l = (i0 + len - 1) < ihi[j] ? (i0 + len - 1) : ihi[j];
                if (f > l) {
                    continue;  // no row of the run has this neighbour
                }
                const uint32_t pp = sx != 0, lq = lqs[pp] + d;  // column of x index i: lq + i
                if (!have[pp]) {
                    base[pp] = g.piece_base(lq + f);
                    have[pp] = true;
                }
                K[j] = base[pp] + 4 * lq;
            }
        }
    }
    // slot by slot over the whole run: straight stores, then the rows at the run's ends whose neighbour is missing
    const uint32_t w0 = q & 1023;
    const uint32_t self0 = sb + 4 * q;  // offset of the run's first row; row t: self0 + 4 t
    if (desc) {
        desc[0] = w0;
        desc[1] = (uint32_t)i0;
        desc[2] = (uint32_t)len;
        desc[3] = self0;
    }
    for (int32_t j = 0; j < 27; ++j) {
        uint32_t* v = Vv + j * 1024 + w0;
        int32_t a = ilo[j] - i0, e = ihi[j] - i0 + 1;  // rows [a, e) of the run have this neighbour
        a = a < 0 ? 0 : (a > len ? len : a);
        e = e < a ? a : (e > len ? len : e);
        if (desc) {
            desc[4 + 3 * j] = K[j];
            desc[5 + 3 * j] = (uint32_t)a;
            desc[6 + 3 * j] = (uint32_t)e;
        } else {
            slot_offsets(O + j * 1024 + w0, K[j], a, e, len, i0, self0);
        }
        // the values are the template's, except a zero entry where the neighbour is missing (the column is the row)
        for (int32_t t = 0; t < a; ++t) {
            v[t] = 0;
        }
        for (int32_t t = e; t < len; ++t) {
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

// rows [q0, q1) of colour block c (one DRAM page at most) into the staging set at base: everything but the offsets of
// the first MAX_RUNS runs, which it describes in runs[] instead (fill.h) for own_slots and the TRISCs. Returns the
// number of runs described.
uint32_t fill_chunk(
    const Geo& g, uint32_t c, uint32_t q0, uint32_t q1, uint32_t real_end, uint32_t base, Exc& x, uint32_t* runs) {
    // plain (not volatile) stores so the compiler can schedule the fill loops; kernel_main puts a compiler memory
    // barrier between the fill and the NoC writes that read these pages
    uint32_t* O = (uint32_t*)base;
    uint32_t* Vv = O + 27 * 1024;
    uint32_t* Bw = O + 54 * 1024;
    const uint32_t py = (c >> 1) & 1, pz = c >> 2, cxn = g.cx(c & 1), k0 = c * g.nb;
    const uint32_t end = q1 < real_end ? q1 : real_end;
    uint32_t jb = 0, nr = 0;
    for (uint32_t q = q0; q < end;) {
        while (jb + 1 < g.nb && g.bst[k0 + jb + 1] <= q) {
            ++jb;  // the last band starting at or before q (bands without lines share the next band's start)
        }
        const uint32_t k = k0 + jb, l = q - g.bst[k], line = l / cxn, i0 = l % cxn;
        const uint32_t len = (cxn - i0) < (end - q) ? (cxn - i0) : (end - q);
        const uint32_t iy = 2 * (g.by0[k] + line % g.bny[k]) + py, iz = 2 * (line / g.bny[k]) + pz;
        uint32_t* desc = nr < MAX_RUNS ? runs + nr++ * RUN_WORDS : nullptr;
        fill_run(g, c, q, (int32_t)i0, (int32_t)len, (int32_t)iy, (int32_t)iz, O, Vv, Bw, x, desc);
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
    return nr;
}

// this RISC's share of the described runs' offsets: slots 0 .. DMSLOTS - 1 (the TRISCs write the rest)
void own_slots(uint32_t* O, const uint32_t* runs, uint32_t nr) {
    for (uint32_t r = 0; r < nr; ++r) {
        const uint32_t* d = runs + r * RUN_WORDS;
        for (uint32_t j = 0; j < DMSLOTS; ++j) {
            slot_offsets(O + j * 1024 + d[0], d[4 + 3 * j], (int32_t)d[5 + 3 * j], (int32_t)d[6 + 3 * j], (int32_t)d[2],
                         (int32_t)d[1], d[3]);
        }
    }
}
// a transfer job's out rows [q0, q1) of out block b (one DRAM page at most) into slot 0 of the staging set at base.
// F holds the fine level and the core's window pieces, C the coarse level.
void fill_transfer(
    const Geo& F,
    const Geo& C,
    uint32_t kind,
    uint32_t b,
    uint32_t q0,
    uint32_t q1,
    uint32_t real_end,
    uint32_t pad,
    uint32_t base,
    Exc& x) {
    uint32_t* O = (uint32_t*)base;
    uint32_t* Vv = O + 27 * 1024;
    const Geo& G = kind == 1 ? C : F;  // the level whose rows are the out rows
    const uint32_t ob = kind == 1 ? b : 0;
    const uint32_t px = ob & 1, py = (ob >> 1) & 1, pz = ob >> 2, cxn = G.cx(px), k0 = ob * G.nb;
    const uint32_t end = q1 < real_end ? q1 : real_end;
    uint32_t jb = 0;
    for (uint32_t q = q0; q < end;) {
        while (jb + 1 < G.nb && G.bst[k0 + jb + 1] <= q) {
            ++jb;
        }
        const uint32_t k = k0 + jb, l = q - G.bst[k], line = l / cxn, i0 = l % cxn;
        const uint32_t len = (cxn - i0) < (end - q) ? (cxn - i0) : (end - q);
        const uint32_t y = 2 * (G.by0[k] + line % G.bny[k]) + py, z = 2 * (line / G.bny[k]) + pz;
        const uint32_t w = q & 1023;
        if (kind == 1) {  // coarse x = 2 (i0 + t) + px -> fine column F.line_q(0, 2y, 2z) + x: every other word
            const uint32_t col0 = F.line_q(0, 2 * y, 2 * z) + 2 * i0 + px;
            const uint32_t off0 = F.piece_base(col0) + 4 * col0;  // a run's columns sit in one window piece
            for (uint32_t t = 0; t < len; ++t) {
                O[w + t] = off0 + 8 * t;
                Vv[w + t] = F_1;
            }
        } else {  // fine block 0: x = 2 (i0 + t) -> coarse (i0 + t, y / 2, z / 2), x parity alternating
            const uint32_t cy = y >> 1, cz = z >> 1;
            uint32_t lq[2] = {0, 0}, pb[2] = {0, 0};
            bool have[2] = {false, false};
            for (uint32_t t = 0; t < len; ++t) {
                const uint32_t cxv = i0 + t, p = cxv & 1;
                if (!have[p]) {
                    lq[p] = C.line_q(p + 2 * (cy & 1) + 4 * (cz & 1), cy, cz);
                    pb[p] = F.piece_base(lq[p] + (cxv >> 1));
                    have[p] = true;
                }
                O[w + t] = pb[p] + 4 * (lq[p] + (cxv >> 1));
                Vv[w + t] = F_1;
            }
        }
        q += len;
    }
    const uint32_t p0 = q0 > real_end ? q0 : real_end;
    if (p0 < q1) {  // padding rows: the block's last real row's column, weight 0
        const uint32_t off = F.piece_base(pad) + 4 * pad;
        for (uint32_t q = p0; q < q1; ++q) {
            O[q & 1023] = off;
            Vv[q & 1023] = 0;
        }
    }
    x.add(0, q0 & 1023, q1 - q0);  // slot 0 of the value template, put back once written
}
}  // namespace

void kernel_main() {
    // common arguments; this core's row follows from its logical position (arg 0: grid width)
    const uint32_t row = get_absolute_logical_y() * get_common_arg_val<uint32_t>(0) + get_absolute_logical_x();
    const uint32_t half = get_common_arg_val<uint32_t>(1), meta_addr = get_common_arg_val<uint32_t>(2);
    const uint32_t shared_pages = get_common_arg_val<uint32_t>(3), meta_pages = get_common_arg_val<uint32_t>(4);
    const uint32_t njobs = get_common_arg_val<uint32_t>(5), launch = get_common_arg_val<uint32_t>(6);
    // chunks the writer RISC (NoC 1) takes out of every 8, for this core's row: a nibble per row y in args 7, 8
    const uint32_t y = get_absolute_logical_y();
    const uint32_t share1 = (get_common_arg_val<uint32_t>(7 + (y >> 3)) >> (4 * (y & 7))) & 15;
    volatile tt_l1_ptr uint32_t* mb = (volatile tt_l1_ptr uint32_t*)get_write_ptr(CB_W);  // fill.h's work mailbox
    uint32_t sent[2] = {0, 0};  // messages sent per staging set
    constexpr auto ad = TensorAccessorArgs<0>();  // every table and vector: interleaved DRAM, 4 KB pages
    constexpr auto am = TensorAccessorArgs<ad.next_compile_time_args_offset()>();
    const auto META = TensorAccessor(am, meta_addr, 4096);  // shared pages, then this core's row of meta_pages
    const uint32_t zpage = get_write_ptr(CB_Z);             // 4 KB of zeros, the source of the zero vectors' writes
    {
        const uint64_t zsrc = get_noc_addr(MEM_ZEROS_BASE);
        for (uint32_t o = 0; o < 4096; o += MEM_ZEROS_SIZE) {
            noc_async_read(zsrc, zpage + o, MEM_ZEROS_SIZE);
        }
    }
    const uint32_t meta = get_write_ptr(half ? CB_M1 : CB_M0);
    for (uint32_t p = 0; p < shared_pages; ++p) {
        noc_async_read(META.get_noc_addr(p), meta + p * 4096, 4096);
    }
    for (uint32_t p = 0; p < meta_pages; ++p) {
        noc_async_read(META.get_noc_addr(shared_pages + row * meta_pages + p), meta + (shared_pages + p) * 4096, 4096);
    }
    noc_async_read_barrier();
    invalidate_l1_cache();
    const uint32_t stage = get_write_ptr(half ? CB_S1 : CB_S0);
    const uint32_t xbuf = get_write_ptr(half ? CB_X1 : CB_X0);
    Exc xs[2] = {{(uint32_t*)xbuf, 0, false}, {(uint32_t*)xbuf + XCAP, 0, false}};
    fill_template((uint32_t*)(stage + 27 * 4096));
    fill_template((uint32_t*)(stage + SET + 27 * 4096));
    uint32_t set = 0, chunk = 0;
    for (uint32_t job = 0; job < njobs; ++job) {
        const uint32_t J = 9 + 10 * job;
        const uint32_t kind = get_common_arg_val<uint32_t>(J), ncj = get_common_arg_val<uint32_t>(J + 1);
        if (row >= ncj) {
            continue;  // this core has no rows in this job
        }
        const uint32_t seg = meta + get_common_arg_val<uint32_t>(J + 2) * 4;
        const auto V = TensorAccessor(ad, get_common_arg_val<uint32_t>(J + 3), 4096);
        const auto OFF = TensorAccessor(ad, get_common_arg_val<uint32_t>(J + 4), 4096);
        const auto Bv = TensorAccessor(ad, get_common_arg_val<uint32_t>(J + 5), 4096);
        const uint32_t has_b = get_common_arg_val<uint32_t>(J + 6), nzero = get_common_arg_val<uint32_t>(J + 7);
        const auto Z0 = TensorAccessor(ad, get_common_arg_val<uint32_t>(J + 8), 4096);
        const auto Z1 = TensorAccessor(ad, get_common_arg_val<uint32_t>(J + 9), 4096);
        Geo g, gc;
        g.init((const uint32_t*)meta, seg);
        if (kind) {
            gc.init_coarse((const uint32_t*)meta, seg);
        }
        const uint32_t* m = g.m;
        const uint32_t ntlv = m[3];
        for (uint32_t c = 0; c < 8; ++c) {
            const uint32_t a = m[6 + 2 * c], e = m[7 + 2 * c], real_end = m[22 + c];
            for (uint32_t q0 = a; q0 < e;) {
                const uint32_t q1 = (q0 / 1024 + 1) * 1024 < e ? (q0 / 1024 + 1) * 1024 : e;
                const uint32_t mine = ((chunk + 1) * share1 / 8 != chunk * share1 / 8) == (half == 1);
                ++chunk;
                if (mine) {
                    const uint32_t base = stage + set * SET;
                    // this set's previous chunk was written out before the last barrier: restore its value template
                    restore((uint32_t*)(base + 27 * 4096), xs[set]);
                    if (kind) {
                        fill_transfer(g, gc, kind, c, q0, q1, real_end, m[43 + c], base, xs[set]);
                    } else {
                        // describe the runs, hand the offsets of slots DMSLOTS .. 26 to the TRISCs, write the rest
                        // here meanwhile, then wait for the TRISCs before the NoC reads the pages
                        const uint32_t i = 2 * half + set;
                        uint32_t* runs = (uint32_t*)(get_write_ptr(CB_W) + 4 * (MB_RUNS + i * MAX_RUNS * RUN_WORDS));
                        const uint32_t nr = fill_chunk(g, c, q0, q1, real_end, base, xs[set], runs);
                        const uint32_t seq = ++sent[set];
                        mb[16 * i + 1] = nr;
                        mb[16 * i + 2] = base;
                        mb[16 * i + 3] = (uint32_t)runs;
                        mb[16 * i + 4] = launch;
                        asm volatile("fence" ::: "memory");  // the run list and fields are in L1 before the number
                        mb[16 * i] = seq;
                        own_slots((uint32_t*)base, runs, nr);
                        for (uint32_t t = 0; t < 3; ++t) {
                            const volatile uint32_t* done = mb + MB_DONE + 16 * (4 * t + i);
                            while (done[0] != seq || done[1] != launch) {
                                invalidate_l1_cache();
                            }
                        }
                    }
                    asm volatile("" ::: "memory");  // every staging store is issued before the NoC reads the pages
                    noc_async_write_barrier();       // the other set's writes have left L1 before it is filled again
                    const uint32_t t = q0 >> 10, wb = (q0 & 1023) * 4, nb = (q1 - q0) * 4;
                    for (uint32_t j = 0; j < (kind ? 1u : 27u); ++j) {
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
    }
    noc_async_write_barrier();
    mb[MB_END + 16 * half] = launch;  // no more messages from this RISC: the TRISCs stop once both have said so
}
