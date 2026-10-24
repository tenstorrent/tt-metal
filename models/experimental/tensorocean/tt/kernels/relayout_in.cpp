// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Rearranges the per-step inputs from their natural row-major layout in DRAM into v27's per-core layout, one pass:
//   cell [L, M, M]                         -> CELL [L][NBX][2 planes][CELL_LEN]   (plane p: cell rows p, p+2, ...;
//                                                                                  column by column, H per column)
//   f1, mask1 [L, N+1, 2N+1]; f2, mask2 [L, N, N+1] -> FMK [6 groups][NBX][NBLK][L][f|mask][128]
//                                                    (group items column by column, H per column)
// One job = (level l, half): half 0 = tracer values + family 2 (vertical edges), half 1 = family 1 (slanted).
// A job reads whole natural rows (one large read per row, every number read once; f, then mask), then builds and
// writes the blocks of all NBX strips one strip at a time. Jobs [j_first, j_end) with stride j_step; the reader and the
// writer RISC of every core run this kernel, each with its own scratch CB. A header (namespace R) is prepended.
#include "api/dataflow/dataflow_api.h"

template <typename A>
inline void paged_write(const A& d, uint32_t off, uint32_t l1, uint32_t nbytes) {  // off: bytes in a PAGE-paged tensor
    while (nbytes) {
        const uint32_t page = off / R::PAGE, in = off % R::PAGE;
        const uint32_t n = R::PAGE - in < nbytes ? R::PAGE - in : nbytes;
        noc_async_write(l1, d.get_noc_addr(page) + in, n);
        off += n;
        l1 += n;
        nbytes -= n;
    }
}

constexpr uint32_t C1 = 2 * R::N + 1, C2 = R::N + 1;  // row lengths of family 1 and 2 arrays
constexpr uint32_t ru64(uint32_t b) { return (b + 63) & ~63u; }
constexpr uint32_t PM = ru64(R::M * 4), P1 = ru64(C1 * 4), P2 = ru64(C2 * 4);  // L1 row pitches (64-byte aligned)

// read rows [row0, row0 + nrows) of level l of a natural [L, R, C] array, one page per row
template <typename A>
inline void rows_read(
    const A& s, uint32_t l, uint32_t rows_per_level, uint32_t nrows, uint32_t row_bytes, uint32_t pitch, uint32_t l1) {
    for (uint32_t r = 0; r < nrows; ++r) {
        noc_async_read(s.get_noc_addr(l * rows_per_level + r), l1 + r * pitch, row_bytes);
    }
}

// d[k * H] = s[k * ss] for k < n: 4 independent loads, then 4 stores, pointer steps instead of multiplies
inline void col_copy(uint32_t* d, const uint32_t* s, uint32_t n, uint32_t ss) {
    constexpr uint32_t H = R::H;
    uint32_t k = 0;
    for (; k + 4 <= n; k += 4) {
        const uint32_t a0 = s[0], a1 = s[ss], a2 = s[2 * ss], a3 = s[3 * ss];
        d[0] = a0;
        d[H] = a1;
        d[2 * H] = a2;
        d[3 * H] = a3;
        s += 4 * ss;
        d += 4 * H;
    }
    for (; k < n; ++k) {
        *d = *s;
        s += ss;
        d += H;
    }
}

// 4 source rows at once: d[c * H + k] = s_k[c * ss] for k < 4, c < n. The 4 stores of one column land next to
// each other (one 16-byte L1 word) instead of H words apart.
inline void quad_copy(
    uint32_t* d,
    const uint32_t* s0,
    const uint32_t* s1,
    const uint32_t* s2,
    const uint32_t* s3,
    uint32_t n,
    uint32_t ss) {
    for (uint32_t c = 0; c < n; ++c) {
        const uint32_t a0 = *s0, a1 = *s1, a2 = *s2, a3 = *s3;
        d[0] = a0;
        d[1] = a1;
        d[2] = a2;
        d[3] = a3;
        s0 += ss;
        s1 += ss;
        s2 += ss;
        s3 += ss;
        d += R::H;
    }
}

// 8 source rows at once: two 16-byte L1 words per column
inline void oct_copy(uint32_t* d, const uint32_t* const* sp, uint32_t n, uint32_t ss) {
    const uint32_t *s0 = sp[0], *s1 = sp[1], *s2 = sp[2], *s3 = sp[3], *s4 = sp[4], *s5 = sp[5], *s6 = sp[6],
                   *s7 = sp[7];
    for (uint32_t c = 0; c < n; ++c) {
        const uint32_t a0 = *s0, a1 = *s1, a2 = *s2, a3 = *s3, a4 = *s4, a5 = *s5, a6 = *s6, a7 = *s7;
        d[0] = a0;
        d[1] = a1;
        d[2] = a2;
        d[3] = a3;
        d[4] = a4;
        d[5] = a5;
        d[6] = a6;
        d[7] = a7;
        s0 += ss;
        s1 += ss;
        s2 += ss;
        s3 += ss;
        s4 += ss;
        s5 += ss;
        s6 += ss;
        s7 += ss;
        d += R::H;
    }
}

// rows [0, nrows) of a strip: d[c * H + r] = row(r)[c * ss], row(r) = base + (2 r + par) * pitch (+ col0 words)
inline void strip_copy(
    uint32_t* d, uint32_t base, uint32_t pitch, uint32_t par, uint32_t col0, uint32_t nrows, uint32_t n, uint32_t ss) {
    auto row = [&](uint32_t r) { return (const uint32_t*)(base + (2 * r + par) * pitch) + col0; };
    uint32_t r = 0;
    for (; r + 8 <= nrows; r += 8) {
        const uint32_t* sp[8] = {
            row(r), row(r + 1), row(r + 2), row(r + 3), row(r + 4), row(r + 5), row(r + 6), row(r + 7)};
        oct_copy(d + r, sp, n, ss);
    }
    for (; r + 4 <= nrows; r += 4) {
        quad_copy(d + r, row(r), row(r + 1), row(r + 2), row(r + 3), n, ss);
    }
    for (; r < nrows; ++r) {
        col_copy(d + r, row(r), n, ss);
    }
}

inline void zero_cols(uint32_t* d, uint32_t c_from, uint32_t c_to) {  // columns c_from .. c_to - 1, all H rows
    for (uint32_t i = c_from * R::H; i < c_to * R::H; ++i) {
        d[i] = 0;
    }
}

void kernel_main() {
    uint32_t a = 0;
    const uint32_t j_first = get_arg_val<uint32_t>(a++);
    const uint32_t j_end = get_arg_val<uint32_t>(a++);
    const uint32_t j_step = get_arg_val<uint32_t>(a++);
    const uint32_t cb = get_arg_val<uint32_t>(a++);
    const uint32_t ad_cell = get_arg_val<uint32_t>(a++), ad_f1 = get_arg_val<uint32_t>(a++),
                   ad_f2 = get_arg_val<uint32_t>(a++);
    const uint32_t ad_m1 = get_arg_val<uint32_t>(a++), ad_m2 = get_arg_val<uint32_t>(a++);
    const uint32_t ad_CELL = get_arg_val<uint32_t>(a++), ad_FMK = get_arg_val<uint32_t>(a++);
    constexpr auto t0 = TensorAccessorArgs<0>();
    constexpr auto t1 = TensorAccessorArgs<t0.next_compile_time_args_offset()>();
    constexpr auto t2 = TensorAccessorArgs<t1.next_compile_time_args_offset()>();
    constexpr auto t3 = TensorAccessorArgs<t2.next_compile_time_args_offset()>();
    constexpr auto t4 = TensorAccessorArgs<t3.next_compile_time_args_offset()>();
    constexpr auto t5 = TensorAccessorArgs<t4.next_compile_time_args_offset()>();
    constexpr auto t6 = TensorAccessorArgs<t5.next_compile_time_args_offset()>();
    const auto Cn = TensorAccessor(t0, ad_cell, R::M * 4);
    const auto F1 = TensorAccessor(t1, ad_f1, C1 * 4);
    const auto F2 = TensorAccessor(t2, ad_f2, C2 * 4);
    const auto M1 = TensorAccessor(t3, ad_m1, C1 * 4);
    const auto M2 = TensorAccessor(t4, ad_m2, C2 * 4);
    const auto CO = TensorAccessor(t5, ad_CELL, R::PAGE);
    const auto FO = TensorAccessor(t6, ad_FMK, R::PAGE);

    // scratch: whole rows of one level for one array (plus the tracer rows in half 0), then one strip's blocks
    const uint32_t base = (get_write_ptr(cb) + 63) & ~63u;
    const uint32_t rcell = base;              // M rows of the tracer values
    const uint32_t rfam = rcell + R::M * PM;  // rows of f or mask (family 1 or 2)
    const uint32_t rfam_bytes = ((R::N + 1) * P1 > R::N * P2) ? (R::N + 1) * P1 : R::N * P2;
    const uint32_t cellbuf = rfam + rfam_bytes;           // 2 planes x CELL_LEN
    const uint32_t fbuf = cellbuf + 2 * R::CELL_LEN * 4;  // 6 groups x {f, mask} x F_LEN
    {                                                     // positions that never receive data stay zero
        uint32_t* z = (uint32_t*)cellbuf;
        for (uint32_t i = 0; i < 2 * R::CELL_LEN + 12 * R::F_LEN; ++i) {
            z[i] = 0;
        }
    }
    auto fb = [&](uint32_t g, uint32_t arr) { return (uint32_t*)(fbuf + (g * 2 + arr) * R::F_LEN * 4); };
    uint32_t cell_hw = 0, hwa[2][2][2] = {};  // columns written so far, per buffer

    for (uint32_t j = j_first; j < j_end; j += j_step) {
        const uint32_t l = j / 2, half = j % 2;
        const uint32_t fam = half == 0 ? 2 : 1;   // half 0: tracer + vertical edges, 1: slanted
        for (uint32_t arr = 0; arr < 2; ++arr) {  // f, then mask
            noc_async_writes_flushed();
            if (half == 0 && arr == 0) {
                rows_read(Cn, l, R::M, R::M, R::M * 4, PM, rcell);
            }
            if (fam == 1) {
                rows_read(arr ? M1 : F1, l, R::N + 1, R::N + 1, C1 * 4, P1, rfam);
            } else {
                rows_read(arr ? M2 : F2, l, R::N, R::N, C2 * 4, P2, rfam);
            }
            noc_async_read_barrier();
            for (uint32_t x = 0; x < R::NBX; ++x) {
                const uint32_t oc0 = R::BAND0[x], w = R::BANDW[x];
                noc_async_writes_flushed();  // the previous strip's blocks have left L1
                if (half == 0 && arr == 0) {
                    // tracer planes: dst[c * H + r] = cell[2r + p][oc0 + c]
                    const uint32_t c1 = oc0 + w + 4 < R::M ? oc0 + w + 4 : R::M, w4 = c1 - oc0;
                    for (uint32_t p = 0; p < 2; ++p) {
                        uint32_t* d = (uint32_t*)(cellbuf + p * R::CELL_LEN * 4);
                        if (w4 < cell_hw) {
                            zero_cols(d, w4, cell_hw);
                        }
                        strip_copy(d, rcell, PM, p, oc0, R::MH, w4, 1);
                        paged_write(
                            CO,
                            ((l * R::NBX + x) * 2 + p) * R::CELL_LEN * 4,
                            cellbuf + p * R::CELL_LEN * 4,
                            R::CELL_LEN * 4);
                    }
                    cell_hw = w4;
                }
                for (uint32_t pr = 0; pr < 2; ++pr) {
                    const uint32_t g0 = fam == 1 ? 2 * pr : 4 + pr, ngr = fam == 1 ? 2 : 1;
                    const uint32_t rows = R::G_ROWS[g0], cols = R::G_COLS[g0];
                    const int32_t fc = (int32_t)(oc0 + w + 1 < cols ? oc0 + w + 1 : cols) - (int32_t)oc0;
                    const uint32_t fcols = fc > 0 ? (uint32_t)fc : 0;
                    uint32_t& hw = hwa[fam - 1][pr][arr];
                    uint32_t* d0 = fb(g0, arr);
                    uint32_t* d1 = fb(g0 + ngr - 1, arr);
                    if (fcols < hw) {
                        zero_cols(d0, fcols, hw);
                        if (ngr == 2) {
                            zero_cols(d1, fcols, hw);
                        }
                    }
                    if (fam == 1) {
                        strip_copy(d0, rfam, P1, pr, 2 * oc0, rows, fcols, 2);
                        strip_copy(d1, rfam, P1, pr, 2 * oc0 + 1, rows, fcols, 2);
                    } else {
                        strip_copy(d0, rfam, P2, pr, oc0, rows, fcols, 1);
                    }
                    hw = fcols;
                    for (uint32_t gg = g0; gg < g0 + ngr; ++gg) {
                        for (uint32_t blk = 0; blk < R::NBLK; ++blk) {
                            paged_write(
                                FO,
                                ((((gg * R::NBX + x) * R::NBLK + blk) * R::L + l) * 2 + arr) * 512,
                                (uint32_t)fb(gg, arr) + blk * 512,
                                512);
                        }
                    }
                }
            }
        }
    }
    noc_async_write_barrier();
}
