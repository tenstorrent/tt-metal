// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Host-side setup of the HPCG solver, compiled once and loaded with ctypes (host/native.py): the matrix of one
// multigrid level, and the per-core work plan of one sparse-row launch. Each function does in one pass what the
// numpy versions (reference/problem.py, tt/custom_ops/sparse_rows/plan.py) do with many array passes; the outputs
// are identical. Loops over grid planes and over cores run in parallel with OpenMP.

#include <algorithm>
#include <climits>
#include <cstdint>
#include <vector>

#include <omp.h>

namespace {

// Storage order of one level (reference/problem.py): colour blocks c = px + 2 py + 4 pz, y-bands inside a block,
// then z, y and x inside a band. line_band[py * nl + (y >> 1)]: band of colour-line y >> 1 for y parity py;
// band_y0 / band_ny / band_start[c * nbands + j]: first colour-line, number of lines and first q of band j of colour c.
struct Geo {
    int32_t nx, nbands, nl;
    const int32_t* line_band;
    const int32_t* band_y0;
    const int32_t* band_ny;
    const int64_t* band_start;
    int32_t cx(int p) const { return (nx - p + 1) / 2; }  // points of one parity along an axis
    int64_t count(int c) const { return (int64_t)cx(c & 1) * cx((c >> 1) & 1) * cx(c >> 2); }
    // q of the first point of grid line (y, z) in colour block cs (cs's y and z parities are y & 1 and z & 1)
    int64_t line_q(int cs, int32_t y, int32_t z) const {
        const int64_t k = (int64_t)cs * nbands + line_band[(y & 1) * nl + (y >> 1)];
        return band_start[k] + ((int64_t)(z >> 1) * band_ny[k] + ((y >> 1) - band_y0[k])) * cx(cs & 1);
    }
    int64_t q_of(int32_t x, int32_t y, int32_t z) const {
        return line_q((x & 1) + 2 * (y & 1) + 4 * (z & 1), y, z) + (x >> 1);
    }
    // f(q, i0, len, y, z): each run of real rows along x inside [a, e) of colour block c, i0 = the run's first x
    // index in the block (x = 2 i0 + px); rows past the block's real rows (padding) are skipped
    template <typename F>
    void for_each_run(int c, int64_t a, int64_t e, F f) const {
        const int py = (c >> 1) & 1, pz = c >> 2;
        const int64_t k0 = (int64_t)c * nbands;
        const int64_t end = std::min(e, band_start[k0] + count(c));
        const int32_t cxn = cx(c & 1);
        int jb = 0;
        for (int64_t q = a; q < end;) {
            while (jb + 1 < nbands && band_start[k0 + jb + 1] <= q) {
                ++jb;  // the last band starting at or before q (bands without lines share the next band's start)
            }
            const int64_t k = k0 + jb, l = q - band_start[k], line = l / cxn;
            const int32_t i0 = (int32_t)(l % cxn);
            const int32_t len = (int32_t)std::min<int64_t>(cxn - i0, end - q);
            f(q, i0, len, 2 * (band_y0[k] + (int32_t)(line % band_ny[k])) + py, 2 * (int32_t)(line / band_ny[k]) + pz);
            q += len;
        }
    }
    // f(q, x, y, z) for every real point, in storage order
    template <typename F>
    void for_each_point(F f) const {
        for (int c = 0; c < 8; ++c) {
            const int64_t a = band_start[(int64_t)c * nbands];
            for_each_run(c, a, a + count(c), [&](int64_t q, int32_t i0, int32_t len, int32_t y, int32_t z) {
                for (int32_t i = 0; i < len; ++i) {
                    f(q + i, 2 * (i0 + i) + (c & 1), y, z);
                }
            });
        }
    }
};

// The granules a core's window must hold, one bit each, and the words holding any (a core marks about 1% of the
// vector's granules, so merging visits only those words, in order, instead of scanning the vector).
struct Granules {
    std::vector<uint64_t> w;
    std::vector<int64_t> used;  // indices of the nonzero words, unsorted
    explicit Granules(int64_t ng) : w((ng + 63) / 64 + 1, 0) {}
    void put(int64_t i, uint64_t bits) {
        if (w[i] == 0) {
            used.push_back(i);
        }
        w[i] |= bits;
    }
    void set(int64_t g) { put(g >> 6, 1ull << (g & 63)); }
    void set_range(int64_t a, int64_t b) {  // granules a .. b, inclusive
        const int64_t wa = a >> 6, wb = b >> 6;
        const uint64_t lo = ~0ull << (a & 63), hi = ~0ull >> (63 - (b & 63));
        if (wa == wb) {
            put(wa, lo & hi);
            return;
        }
        put(wa, lo);
        for (int64_t i = wa + 1; i < wb; ++i) {
            put(i, ~0ull);
        }
        put(wb, hi);
    }
};

// Runs of marked granules in [gmin, gmax] whose gaps are at most wgap granules -> window pieces (q lo, words, L1
// byte offset), the first max_pieces of them stored; clears the marks. Returns the number of pieces.
int64_t merge_pieces(
    Granules& mask,
    int64_t gmin,
    int64_t gmax,
    int32_t gran,
    int32_t wgap,
    int64_t max_pieces,
    int64_t* plo,
    int64_t* pln,
    int64_t* pl1) {
    int64_t np = 0, l1 = 0, start = -1, last = -1;
    std::sort(mask.used.begin(), mask.used.end());
    for (const int64_t wi : mask.used) {
        uint64_t bits = mask.w[wi];
        mask.w[wi] = 0;
        while (bits) {  // marked granules in increasing order
            const int64_t g = wi * 64 + __builtin_ctzll(bits);
            bits &= bits - 1;
            if (start < 0) {
                start = g;
            } else if (g - last > wgap) {  // close the previous run
                if (np < max_pieces) {
                    plo[np] = start * gran;
                    pln[np] = (last + 1 - start) * gran;
                    pl1[np] = l1;
                }
                l1 += (last + 1 - start) * gran * 4;
                ++np;
                start = g;
            }
            last = g;
        }
    }
    mask.used.clear();
    if (start >= 0) {
        if (np < max_pieces) {
            plo[np] = start * gran;
            pln[np] = (last + 1 - start) * gran;
            pl1[np] = l1;
        }
        ++np;
    }
    return np;
}

// The grid transfer of HPCG's multigrid as a 1-slot matrix, from the storage-order formula of the fine level F and the
// coarse level C (the tables Level.link builds as R / Pm, without building them). kind 1, restriction: out rows are C's
// rows (blocks = C's colour blocks), the column of coarse point (x, y, z) is F's point (2x, 2y, 2z). kind 2,
// prolongation: out rows are F's colour block 0, the column of fine point (2x, 2y, 2z) is C's point (x, y, z).
// Real rows have weight 1; padding rows (the tail of a block) take the column of the block's last real row, weight 0.
// f(q, col, real) for every out row q in [a, e) of out block b.
template <typename Fn>
void for_each_transfer_row(int kind, const Geo& F, const Geo& C, int b, int64_t a, int64_t e, Fn f) {
    const Geo& O = kind == 1 ? C : F;  // the level whose rows are the out rows
    const int ob = kind == 1 ? b : 0;  // their colour block
    const int64_t real_end = O.band_start[(int64_t)ob * O.nbands] + O.count(ob);
    auto run = [&](int64_t q, int32_t i0, int32_t len, int32_t y, int32_t z) {
        for (int32_t i = 0; i < len; ++i) {
            const int32_t x = 2 * (i0 + i) + (ob & 1);
            f(q + i, kind == 1 ? F.q_of(2 * x, 2 * y, 2 * z) : C.q_of(x / 2, y / 2, z / 2), true);
        }
    };
    O.for_each_run(ob, a, e, run);
    if (std::max(a, real_end) < e) {  // padding rows: the column of the block's last real row
        int64_t pad = 0;
        O.for_each_run(ob, real_end - 1, real_end, [&](int64_t, int32_t i0, int32_t, int32_t y, int32_t z) {
            const int32_t x = 2 * i0 + (ob & 1);
            pad = kind == 1 ? F.q_of(2 * x, 2 * y, 2 * z) : C.q_of(x / 2, y / 2, z / 2);
        });
        for (int64_t q = std::max(a, real_end); q < e; ++q) {
            f(q, pad, false);
        }
    }
}

// The columns of for_each_transfer_row as ranges: f(slot, lo, hi) for each run of out rows in [a, e) of out block b
// (restriction: the run's columns lo, lo + 2, ..., hi, slot 0; prolongation: for each x parity p of the coarse
// point, the contiguous columns lo .. hi, slot p), then f(2, pad, pad) when [a, e) holds padding rows. Every granule
// a range spans holds one of its columns (they are at most 2 apart), so marking ranges marks the same granules.
template <typename Fn>
void for_each_transfer_range(int kind, const Geo& F, const Geo& C, int b, int64_t a, int64_t e, Fn f) {
    const Geo& O = kind == 1 ? C : F;
    const int ob = kind == 1 ? b : 0;
    const int64_t real_end = O.band_start[(int64_t)ob * O.nbands] + O.count(ob);
    O.for_each_run(ob, a, e, [&](int64_t, int32_t i0, int32_t len, int32_t y, int32_t z) {
        if (kind == 1) {  // coarse x = 2 (i0 + t) + px -> fine (2x, 2y, 2z): column F.line_q(0, 2y, 2z) + x
            const int64_t c0 = F.line_q(0, 2 * y, 2 * z) + 2 * i0 + (ob & 1);
            f(0, c0, c0 + 2 * (int64_t)(len - 1));
        } else {  // fine x = 2 (i0 + t) -> coarse (i0 + t, y / 2, z / 2): two x parities, each contiguous
            for (int p = 0; p < 2; ++p) {
                const int32_t t0 = ((i0 & 1) == p) ? 0 : 1;  // first t with (i0 + t) & 1 == p
                if (t0 >= len) {
                    continue;
                }
                const int32_t t1 = len - 1 - (((i0 + len - 1) & 1) == p ? 0 : 1);  // last such t
                const int64_t lq = C.line_q(p + 2 * ((y / 2) & 1) + 4 * ((z / 2) & 1), y / 2, z / 2);
                f(p, lq + ((i0 + t0) >> 1), lq + ((i0 + t1) >> 1));
            }
        }
    });
    if (std::max(a, real_end) < e) {
        int64_t pad = 0;
        for_each_transfer_row(kind, F, C, b, real_end - 1, real_end, [&](int64_t, int64_t col, bool) { pad = col; });
        f(2, pad, pad);
    }
}

// v[i] = v[the last mapped index before i] for every unmapped i (v[0] when there is none), in parallel: each thread
// finds its chunk's last mapped index, a short serial pass carries it into the next chunks, then each thread fills
// its chunk. Mapped entries are only read, so threads never race.
template <typename T, typename M>
void fill_from_previous(T* v, int64_t n, M mapped) {
    const int64_t nt = omp_get_max_threads(), chunk = (n + nt - 1) / nt;
    std::vector<int64_t> last(nt, -1), carry(nt, -1);
#pragma omp parallel for schedule(static, 1)
    for (int64_t c = 0; c < nt; ++c) {
        for (int64_t i = c * chunk; i < std::min(n, (c + 1) * chunk); ++i) {
            if (mapped(i)) {
                last[c] = i;
            }
        }
    }
    for (int64_t c = 1; c < nt; ++c) {
        carry[c] = last[c - 1] >= 0 ? last[c - 1] : carry[c - 1];
    }
#pragma omp parallel for schedule(static, 1)
    for (int64_t c = 0; c < nt; ++c) {
        int64_t run = carry[c];
        for (int64_t i = c * chunk; i < std::min(n, (c + 1) * chunk); ++i) {
            if (mapped(i)) {
                run = i;
            } else {
                v[i] = v[run < 0 ? 0 : run];
            }
        }
    }
}

}  // namespace

extern "C" {

// OpenMP threads of the calling thread's parallel regions (host/native.py sets a quarter of the logical CPUs)
void hpcg_set_threads(int32_t n) { omp_set_num_threads(n); }

// Row ownership of a launch (SparseRowPlan._ownership_numpy is the definition): own[(k * NB + b) * 2 + {0, 1}] =
// core k's rows [a, e) of out block b = [s[b], e[b]). With y-bands (cuts[cat[b] .. cat[b + 1]) holds block b's band
// starts and end), band j goes to cores j * pb .. (j + 1) * pb - 1 (pb = nc / bands), split at align-row cuts;
// without (cat[b] == cat[b + 1]), each core takes an equal contiguous share rounded up to align rows.
void hpcg_ownership(
    int32_t NB,
    const int64_t* s,
    const int64_t* e,
    const int64_t* cuts,
    const int64_t* cat,
    int32_t nc,
    int64_t align,
    int64_t* own) {
    auto up = [align](int64_t x) { return (x + align - 1) / align * align; };
    std::vector<int64_t> c;
    for (int32_t b = 0; b < NB; ++b) {
        const int64_t nb = cat[b + 1] - cat[b];
        if (nb > 0) {
            const int64_t B = nb - 1, pb = nc / B;
            for (int64_t j = 0; j < B; ++j) {
                const int64_t bs = j == 0 ? s[b] : cuts[cat[b] + j];
                const int64_t be = j + 1 == B ? e[b] : cuts[cat[b] + j + 1];
                c.assign(1, bs);
                for (int64_t i = 1; i < pb; ++i) {
                    c.push_back(std::min(be, up(bs + i * (be - bs) / pb)));
                }
                c.push_back(be);
                for (int64_t i = 0; i < pb; ++i) {
                    own[((j * pb + i) * NB + b) * 2] = c[i];
                    own[((j * pb + i) * NB + b) * 2 + 1] = std::max(c[i], c[i + 1]);
                }
            }
        } else {
            const int64_t per = up((e[b] - s[b] + nc - 1) / nc);  // the numpy form's ceil(n / (nc align)) align
            for (int32_t k = 0; k < nc; ++k) {
                const int64_t a = std::min(e[b], s[b] + k * per);
                own[(k * NB + b) * 2] = a;
                own[(k * NB + b) * 2 + 1] = std::min(e[b], a + per);
            }
        }
    }
}

// Matrix of one level, in the storage order of reference/problem.py (colour blocks, y-bands, natural order).
// line_band[py * nl + yl]: band of colour-line yl (= iy >> 1) for y parity py; band_y0 / band_ny [(c * nbands + j)]:
// first colour-line and number of lines of band j in colour c; band_start[c * nbands + j]: q of band j's first row.
// Outputs: pos[n] (q of each natural point), nat_at[P] (-1 on padding rows), V and COL ([27][stride]; every slot of
// rows [0, P) is written once, the row padding beyond P is left as allocated), B[P].
// Returns the number of true non-zeros.
int64_t hpcg_build_level(
    int32_t nx,
    int32_t nbands,
    int32_t nl,
    const int32_t* line_band,
    const int32_t* band_y0,
    const int32_t* band_ny,
    const int64_t* band_start,
    int64_t P,
    int64_t stride,
    int64_t* pos,
    int64_t* nat_at,
    float* V,
    int32_t* COL,
    float* B) {
    const Geo G{nx, nbands, nl, line_band, band_y0, band_ny, band_start};
    // Rows are visited in storage order (colour block, band, z line pair, y line, x), so each of the 27 value and
    // column rows is written front to back. Work items: (colour, band, z index), split over the threads.
    const int32_t nzc = (nx + 1) / 2;
    const int64_t items = 8LL * nbands * nzc;
    int64_t nnz = 0;
#pragma omp parallel for schedule(dynamic, 4) reduction(+ : nnz)
    for (int64_t it = 0; it < items; ++it) {
        const int c = (int)(it / ((int64_t)nbands * nzc));
        const int jb = (int)((it / nzc) % nbands);
        const int32_t iz2 = (int32_t)(it % nzc);
        const int px = c & 1, py = (c >> 1) & 1, pz = c >> 2;
        const int32_t iz = 2 * iz2 + pz;
        const int64_t k = (int64_t)c * nbands + jb;
        if (iz >= nx || band_ny[k] == 0) {
            continue;
        }
        const int32_t cxn = (nx - px + 1) / 2;
        for (int32_t yl = 0; yl < band_ny[k]; ++yl) {
            const int32_t iy = 2 * (band_y0[k] + yl) + py;
            int64_t q = band_start[k] + ((int64_t)iz2 * band_ny[k] + yl) * cxn;
            for (int32_t ix2 = 0; ix2 < cxn; ++ix2, ++q) {
                const int32_t ix = 2 * ix2 + px;
                const int64_t nat = ((int64_t)iz * nx + iy) * nx + ix;
                pos[nat] = q;
                nat_at[q] = nat;
                int cnt = 0;
                for (int sz = -1; sz <= 1; ++sz) {
                    for (int sy = -1; sy <= 1; ++sy) {
                        const int32_t jy = iy + sy, jz = iz + sz;
                        const bool line_ok = jy >= 0 && jy < nx && jz >= 0 && jz < nx;
                        for (int sx = -1; sx <= 1; ++sx) {
                            const int j = (sz + 1) * 9 + (sy + 1) * 3 + (sx + 1);
                            const int32_t jx = ix + sx;
                            if (!line_ok || jx < 0 || jx >= nx) {
                                COL[j * stride + q] = (int32_t)q;  // missing neighbour: a zero entry on the row itself
                                V[j * stride + q] = 0.0f;
                                continue;
                            }
                            V[j * stride + q] = (j == 13) ? 26.0f : -1.0f;
                            COL[j * stride + q] = (int32_t)G.q_of(jx, jy, jz);
                            ++cnt;
                        }
                    }
                }
                B[q] = 26.0f - (cnt - 1);
                nnz += cnt;
            }
        }
    }
    // padding rows (the tail of each colour block, past its last band): zero entries on the row itself, diagonal 1
    // so the Gauss-Seidel update keeps them at 0
    for (int c = 0; c < 8; ++c) {
        const int px = c & 1, py = (c >> 1) & 1, pz = c >> 2;
        const int64_t cnt = (int64_t)((nx - px + 1) / 2) * ((nx - py + 1) / 2) * ((nx - pz + 1) / 2);
        const int64_t a = band_start[(int64_t)c * nbands] + cnt;
        const int64_t e = c < 7 ? band_start[(int64_t)(c + 1) * nbands] : P;
#pragma omp parallel for schedule(static)
        for (int64_t q = a; q < e; ++q) {
            nat_at[q] = -1;
            B[q] = 0.0f;
            for (int j = 0; j < 27; ++j) {
                COL[j * stride + q] = (int32_t)q;
                V[j * stride + q] = 0.0f;
            }
            V[13 * stride + q] = 1.0f;
        }
    }
    return nnz;
}

// Grid transfer tables between fine level F and coarse level C, from the storage-order formula of both (no
// per-point tables), as Level.link in reference/problem.py. nb0: rows of F's colour block 0. R[qc]: q of coarse
// point qc's fine point (2x, 2y, 2z) (a padding row takes the previous real row's R); Rw[qc]: 1 on real rows.
// Pm[q] (q < nb0): coarse q of fine point q (a fine row without a coarse point takes the previous mapped row's Pm);
// Pw[q]: 1 on mapped rows. 32-bit outputs: the plans read the indices as int32 and the weights go up as fp32.
// Returns the largest R of a real row.
int64_t hpcg_link(
    int32_t nx,
    int32_t nbands,
    int32_t nl,
    const int32_t* line_band,
    const int32_t* band_y0,
    const int32_t* band_ny,
    const int64_t* band_start,
    int64_t nb0,
    int32_t cnx,
    int32_t cnbands,
    int32_t cnl,
    const int32_t* cline_band,
    const int32_t* cband_y0,
    const int32_t* cband_ny,
    const int64_t* cband_start,
    int64_t CP,
    int32_t* R,
    float* Rw,
    int32_t* Pm,
    float* Pw) {
    const Geo F{nx, nbands, nl, line_band, band_y0, band_ny, band_start};
    const Geo C{cnx, cnbands, cnl, cline_band, cband_y0, cband_ny, cband_start};
#pragma omp parallel for schedule(static)
    for (int64_t qc = 0; qc < CP; ++qc) {
        Rw[qc] = 0.0f;
    }
#pragma omp parallel for schedule(dynamic, 1)
    for (int c = 0; c < 8; ++c) {  // the coarse colour blocks in parallel
        const int64_t a = C.band_start[(int64_t)c * C.nbands];
        C.for_each_run(c, a, a + C.count(c), [&](int64_t q, int32_t i0, int32_t len, int32_t y, int32_t z) {
            for (int32_t i = 0; i < len; ++i) {
                R[q + i] = (int32_t)F.q_of(2 * (2 * (i0 + i) + (c & 1)), 2 * y, 2 * z);
                Rw[q + i] = 1.0f;
            }
        });
    }
    int64_t rmax = -1;
#pragma omp parallel for schedule(static) reduction(max : rmax)
    for (int64_t qc = 0; qc < CP; ++qc) {
        if (Rw[qc] > 0.0f) {
            rmax = std::max(rmax, (int64_t)R[qc]);
        }
    }
    if (CP > 0 && !(Rw[0] > 0.0f)) {
        R[0] = 0;  // no real row before: padding rows take R[0] = 0, as in numpy
    }
    fill_from_previous(R, CP, [&](int64_t qc) { return Rw[qc] > 0.0f; });
#pragma omp parallel for schedule(static)
    for (int64_t q = 0; q < nb0; ++q) {
        Pm[q] = 0;
        Pw[q] = 0.0f;
    }
#pragma omp parallel for schedule(static)
    for (int64_t qc = 0; qc < CP; ++qc) {  // distinct real rows have distinct fine points
        if (Rw[qc] > 0.0 && R[qc] < nb0) {
            Pm[R[qc]] = (int32_t)qc;
            Pw[R[qc]] = 1.0f;
        }
    }
    fill_from_previous(Pm, nb0, [&](int64_t q) { return Pw[q] > 0.0f; });  // Pm[0] = 0 when q = 0 is unmapped
    return rmax;
}

// Window pieces of every core of a Gauss-Seidel launch on the 27-point matrix of one level, from the storage-order
// formula instead of the column table: the same pieces hpcg_plan_windows computes from COL with exchange = 1 (the
// columns of a run of rows along x, for one neighbour line, are a contiguous range of q), and own_l1[k * NB + b]: the
// L1 byte offset of core k's own rows of block b in its window.
int64_t hpcg_plan_windows_stencil(
    int32_t nx,
    int32_t nbands,
    int32_t nl,
    const int32_t* line_band,
    const int32_t* band_y0,
    const int32_t* band_ny,
    const int64_t* band_start,
    int32_t nc,
    int32_t NB,
    const int64_t* own,
    int32_t gran,
    int32_t wgap,
    int64_t ng,
    int64_t empty_lo,
    int64_t max_pieces,
    int64_t* piece_lo,
    int64_t* piece_ln,
    int64_t* piece_l1,
    int64_t* npieces,
    int64_t* own_l1) {
    const Geo G{nx, nbands, nl, line_band, band_y0, band_ny, band_start};
    int64_t bad = 0;
#pragma omp parallel
    {
        Granules mask(ng);
#pragma omp for schedule(dynamic, 1) reduction(+ : bad)
        for (int32_t k = 0; k < nc; ++k) {
            int64_t gmin = ng, gmax = -1;
            auto mark = [&](int64_t lo, int64_t hi) {  // granules of q range [lo, hi]
                mask.set_range(lo / gran, hi / gran);
                gmin = std::min(gmin, lo / gran);
                gmax = std::max(gmax, hi / gran);
            };
            // one pending q range per neighbour slot (line offset sy, sz; same or other x parity): consecutive runs of
            // a band-plane have consecutive neighbour lines, whose ranges join up, so a range is marked once per
            // band-plane instead of once per grid line
            int64_t plo[18], phi[18];
            auto add = [&](int slot, int64_t lo, int64_t hi) {
                if (lo >= plo[slot] && lo <= phi[slot] + 1) {
                    phi[slot] = std::max(phi[slot], hi);
                    return;
                }
                if (phi[slot] >= plo[slot]) {
                    mark(plo[slot], phi[slot]);
                }
                plo[slot] = lo;
                phi[slot] = hi;
            };
            bool used = false;
            for (int32_t b = 0; b < NB; ++b) {
                const int64_t a = own[(k * NB + b) * 2], e = own[(k * NB + b) * 2 + 1];
                if (e <= a) {
                    continue;
                }
                used = true;
                mark(a, e - 1);  // own rows (padding rows' columns are themselves)
                for (int i = 0; i < 18; ++i) {
                    plo[i] = 0;
                    phi[i] = -2;
                }
                const int px = b & 1, py = (b >> 1) & 1, pz = b >> 2;
                // a partial grid line: rows i0 .. i0 + len - 1 of line (iy, iz)
                auto run = [&](int32_t i0, int32_t len, int32_t iy, int32_t iz) {
                    for (int sz = -1; sz <= 1; ++sz) {
                        for (int sy = -1; sy <= 1; ++sy) {
                            const int32_t jy = iy + sy, jz = iz + sz;
                            if (jy < 0 || jy >= nx || jz < 0 || jz >= nx) {
                                continue;  // missing neighbours: columns on the row itself
                            }
                            const int slot = 2 * ((sz + 1) * 3 + (sy + 1));
                            const int cs = px + 2 * (jy & 1) + 4 * (jz & 1), co = cs ^ 1;
                            const int64_t bs = G.line_q(cs, jy, jz), bo = G.line_q(co, jy, jz);
                            add(slot, bs + i0, bs + i0 + len - 1);  // sx = 0
                            // sx = -1 and +1 (the other x parity): x index ix2 - 1 .. ix2 (px = 0) or ix2 .. ix2 + 1
                            const int32_t cxo = G.cx(co & 1);
                            const int32_t lo = std::max(0, i0 - (1 - px)), hi = std::min(cxo - 1, i0 + len - 1 + px);
                            if (lo <= hi) {
                                add(slot + 1, bo + lo, bo + hi);
                            }
                        }
                    }
                };
                // whole grid lines iy, iy + 2, ..., iy + 2 (n - 1) of plane iz: every neighbour line is wholly a
                // column range of both x parities (x neighbours reach both ends), so each slot gets whole lines of
                // block B, one q range per band of B the lines fall in
                auto lines = [&](int slot, int B, int32_t jy0, int32_t jy1, int32_t jz) {
                    const int32_t cxb = G.cx(B & 1);
                    const int64_t pl = jz >> 1;
                    for (int32_t yl = jy0 >> 1; yl <= (jy1 >> 1);) {
                        const int64_t kk = (int64_t)B * G.nbands + G.line_band[(jy0 & 1) * G.nl + yl];
                        const int32_t y0 = G.band_y0[kk], ny = G.band_ny[kk], ye = std::min(jy1 >> 1, y0 + ny - 1);
                        const int64_t base = G.band_start[kk] + pl * ny * cxb;
                        add(slot, base + (int64_t)(yl - y0) * cxb, base + (int64_t)(ye - y0 + 1) * cxb - 1);
                        yl = ye + 1;
                    }
                };
                auto group = [&](int32_t iy, int32_t n, int32_t iz) {
                    for (int sz = -1; sz <= 1; ++sz) {
                        const int32_t jz = iz + sz;
                        if (jz < 0 || jz >= nx) {
                            continue;
                        }
                        for (int sy = -1; sy <= 1; ++sy) {
                            int32_t jy0 = iy + sy, jy1 = iy + 2 * (n - 1) + sy;
                            jy0 += jy0 < 0 ? 2 : 0;  // lines outside the grid: only the first or the last
                            jy1 -= jy1 >= nx ? 2 : 0;
                            if (jy0 > jy1) {
                                continue;
                            }
                            const int slot = 2 * ((sz + 1) * 3 + (sy + 1));
                            const int cs = px + 2 * (jy0 & 1) + 4 * (jz & 1);
                            lines(slot, cs, jy0, jy1, jz);
                            lines(slot + 1, cs ^ 1, jy0, jy1, jz);
                        }
                    }
                };
                const int64_t k0 = (int64_t)b * G.nbands, end = std::min(e, G.band_start[k0] + G.count(b));
                const int32_t cxn = G.cx(px);
                int jb = 0;
                for (int64_t q = a; q < end;) {  // G.for_each_run's walk, whole lines of a band-plane at once
                    while (jb + 1 < G.nbands && G.band_start[k0 + jb + 1] <= q) {
                        ++jb;
                    }
                    const int64_t kk = k0 + jb, l = q - G.band_start[kk], line = l / cxn;
                    const int32_t i0 = (int32_t)(l % cxn), ny = G.band_ny[kk];
                    const int32_t iy = 2 * (G.band_y0[kk] + (int32_t)(line % ny)) + py, iz = 2 * (int32_t)(line / ny) + pz;
                    if (i0 == 0 && end - q >= cxn) {
                        const int32_t n = (int32_t)std::min<int64_t>(ny - line % ny, (end - q) / cxn);
                        group(iy, n, iz);
                        q += (int64_t)n * cxn;
                    } else {
                        const int32_t len = (int32_t)std::min<int64_t>(cxn - i0, end - q);
                        run(i0, len, iy, iz);
                        q += len;
                    }
                }
                for (int i = 0; i < 18; ++i) {
                    if (phi[i] >= plo[i]) {
                        mark(plo[i], phi[i]);
                    }
                }
            }
            if (!used) {
                mark(empty_lo, empty_lo);
            }
            const int64_t np = merge_pieces(
                mask,
                gmin,
                gmax,
                gran,
                wgap,
                max_pieces,
                piece_lo + k * max_pieces,
                piece_ln + k * max_pieces,
                piece_l1 + k * max_pieces);
            npieces[k] = np;
            if (np > max_pieces) {
                bad += 1;
                continue;
            }
            // L1 byte offset of each block's own rows: inside the last piece starting at or before them
            const int64_t* lo = piece_lo + k * max_pieces;
            for (int32_t b = 0; b < NB; ++b) {
                const int64_t a = own[(k * NB + b) * 2], e = own[(k * NB + b) * 2 + 1];
                own_l1[k * NB + b] = 0;
                if (e <= a) {
                    continue;
                }
                const int64_t p = std::upper_bound(lo, lo + np, a) - lo - 1;
                if (p < 0 || e > lo[p] + piece_ln[k * max_pieces + p]) {
                    bad += 1;  // own rows split over window pieces (cannot happen: they are marked as one range)
                    continue;
                }
                own_l1[k * NB + b] = piece_l1[k * max_pieces + p] + (a - lo[p]) * 4;
            }
        }
    }
    return bad ? -1 : 0;
}

// to_device's host permutation (tt/layout.py): dst[t * 1024 + k] = src[t * 1024 + word_of_flat(k)], so that after
// TT-NN tilizes the [P / 32, 32] tensor, word w of DRAM page t holds src[t * 1024 + w].
void hpcg_tile_permute(const float* src, float* dst, int64_t P) {
#pragma omp parallel for schedule(static)
    for (int64_t t = 0; t < P / 1024; ++t) {
        const float* s = src + t * 1024;
        float* d = dst + t * 1024;
        for (int k = 0; k < 1024; ++k) {
            const int rr = k >> 5, c = k & 31;
            d[k] = s[((rr >> 4) * 2 + (c >> 4)) * 256 + (rr & 15) * 16 + (c & 15)];
        }
    }
}

// Window pieces of every core of one sparse-row launch: the granules (gran words) that its rows' columns touch
// (plus, with exchange, its own rows), merged into runs whose gaps are at most wgap granules. own[(k * NB + b) * 2]:
// row range of core k in out block b. empty_lo: placeholder column of a core without rows. Fills, per core k, up
// to max_pieces (lo, len, l1) entries at k * max_pieces, and npieces[k]. Returns -1 if a core needs more pieces.
int64_t hpcg_plan_windows(
    int32_t nslot,
    int64_t col_stride,
    const int32_t* COL,
    int32_t nc,
    int32_t NB,
    const int64_t* own,
    int32_t exchange,
    int32_t gran,
    int32_t wgap,
    int64_t ng,
    int64_t empty_lo,
    int64_t max_pieces,
    int64_t* piece_lo,
    int64_t* piece_ln,
    int64_t* piece_l1,
    int64_t* npieces) {
    int64_t bad = 0;
#pragma omp parallel
    {
        Granules mask(ng);
#pragma omp for schedule(dynamic, 1) reduction(+ : bad)
        for (int32_t k = 0; k < nc; ++k) {
            int64_t gmin = ng, gmax = -1;
            auto mark = [&](int64_t g) {
                mask.set(g);
                gmin = std::min(gmin, g);
                gmax = std::max(gmax, g);
            };
            bool used = false;
            for (int32_t b = 0; b < NB; ++b) {
                const int64_t a = own[(k * NB + b) * 2], e = own[(k * NB + b) * 2 + 1];
                if (e <= a) {
                    continue;
                }
                used = true;
                for (int32_t j = 0; j < nslot; ++j) {
                    const int32_t* col = COL + j * col_stride;
                    for (int64_t q = a; q < e; ++q) {
                        mark(col[q] / gran);
                    }
                }
                if (exchange) {
                    for (int64_t g = a / gran; g <= (e - 1) / gran; ++g) {
                        mark(g);
                    }
                }
            }
            if (!used) {
                mark(empty_lo / gran);
            }
            const int64_t np = merge_pieces(
                mask,
                gmin,
                gmax,
                gran,
                wgap,
                max_pieces,
                piece_lo + k * max_pieces,
                piece_ln + k * max_pieces,
                piece_l1 + k * max_pieces);
            npieces[k] = np;
            if (np > max_pieces) {
                bad += 1;
            }
        }
    }
    return bad ? -1 : 0;
}

// Window byte offset of every column of every core's rows: off[j * off_stride + q] = l1 of the piece holding the
// column + (column - piece lo) * 4; own_l1[k * NB + b] = window byte offset of core k's first row of block b
// (exchange launches). Returns the number of columns found outside their core's pieces (0 when the plan is valid).
int64_t hpcg_plan_offsets(
    int32_t nslot,
    int64_t col_stride,
    const int32_t* COL,
    int32_t nc,
    int32_t NB,
    const int64_t* own,
    int32_t exchange,
    int32_t gran,
    int64_t ng,
    int64_t max_pieces,
    const int64_t* piece_lo,
    const int64_t* piece_ln,
    const int64_t* piece_l1,
    const int64_t* npieces,
    int64_t off_stride,
    int32_t* off,
    int64_t* own_l1) {
    int64_t bad = 0;
#pragma omp parallel
    {
        std::vector<int32_t> piece_of(ng, -1);
#pragma omp for schedule(dynamic, 1) reduction(+ : bad)
        for (int32_t k = 0; k < nc; ++k) {
            const int64_t* lo = piece_lo + k * max_pieces;
            const int64_t* ln = piece_ln + k * max_pieces;
            const int64_t* l1 = piece_l1 + k * max_pieces;
            for (int64_t p = 0; p < npieces[k]; ++p) {
                for (int64_t g = lo[p] / gran; g < (lo[p] + ln[p]) / gran; ++g) {
                    piece_of[g] = (int32_t)p;
                }
            }
            for (int32_t b = 0; b < NB; ++b) {
                const int64_t a = own[(k * NB + b) * 2], e = own[(k * NB + b) * 2 + 1];
                if (exchange) {
                    own_l1[k * NB + b] = 0;
                }
                if (e <= a) {
                    continue;
                }
                for (int32_t j = 0; j < nslot; ++j) {
                    const int32_t* col = COL + j * col_stride;
                    int32_t* o = off + j * off_stride;
                    for (int64_t q = a; q < e; ++q) {
                        const int64_t c = col[q];
                        const int32_t p = piece_of[c / gran];
                        if (p < 0) {
                            ++bad;
                            continue;
                        }
                        o[q] = (int32_t)(l1[p] + (c - lo[p]) * 4);
                    }
                }
                if (exchange) {
                    const int32_t p = piece_of[a / gran];
                    if (p < 0 || e > lo[p] + ln[p]) {
                        ++bad;
                    } else {
                        own_l1[k * NB + b] = l1[p] + (a - lo[p]) * 4;
                    }
                }
            }
            for (int64_t p = 0; p < npieces[k]; ++p) {
                for (int64_t g = lo[p] / gran; g < (lo[p] + ln[p]) / gran; ++g) {
                    piece_of[g] = -1;
                }
            }
        }
    }
    return bad;
}

}  // extern "C"

namespace {

// Exchange entries of a Gauss-Seidel launch: for each block b, each core m that owns rows, and each piece of m's
// window, every other core k whose own rows of block b overlap the piece gets an entry (k, b, m, s, f, dst): rows
// [s, f) go to byte offset dst of m's window. Own ranges of a block are sorted by core. Returns the entries, 6 words
// each. The (b, m) pairs run in parallel, each into its own list (one pass), and the lists are joined in the serial
// order (b, then m, then piece, then k).
std::vector<int64_t> exchange_entries(
    int32_t nc,
    int32_t NB,
    const int64_t* own,
    int64_t max_pieces,
    const int64_t* piece_lo,
    const int64_t* piece_ln,
    const int64_t* piece_l1,
    const int64_t* npieces) {
    std::vector<uint8_t> has_rows(nc, 0);
    for (int32_t m = 0; m < nc; ++m) {
        for (int32_t b = 0; b < NB; ++b) {
            if (own[(m * NB + b) * 2 + 1] > own[(m * NB + b) * 2]) {
                has_rows[m] = 1;
            }
        }
    }
    std::vector<int64_t> E((size_t)NB * nc);  // E[b * nc + k]: end of core k's rows of block b
    for (int32_t b = 0; b < NB; ++b) {
        for (int32_t k = 0; k < nc; ++k) {
            E[(size_t)b * nc + k] = own[(k * NB + b) * 2 + 1];
        }
    }
    // the q span of block b's own rows: only window pieces that reach into it can overlap them
    std::vector<int64_t> span_lo(NB, INT64_MAX), span_hi(NB, INT64_MIN);
    for (int32_t b = 0; b < NB; ++b) {
        for (int32_t k = 0; k < nc; ++k) {
            const int64_t A = own[(k * NB + b) * 2], Ek = own[(k * NB + b) * 2 + 1];
            if (Ek > A) {
                span_lo[b] = std::min(span_lo[b], A);
                span_hi[b] = std::max(span_hi[b], Ek);
            }
        }
    }
    // entries of pair (b, m), appended to out
    auto pair = [&](int32_t b, int32_t m, std::vector<int64_t>& out) {
        if (!has_rows[m] || span_hi[b] <= span_lo[b]) {
            return;  // a core without rows reads no window: its placeholder piece gets nothing
        }
        const int64_t* Eb = E.data() + (size_t)b * nc;
        const int64_t* lo_m = piece_lo + m * max_pieces;
        // pieces are sorted by lo and disjoint: the first that can reach span_lo is the last starting at or before it
        int64_t p0 = std::upper_bound(lo_m, lo_m + npieces[m], span_lo[b]) - lo_m - 1;
        for (int64_t p = std::max<int64_t>(p0, 0); p < npieces[m] && lo_m[p] < span_hi[b]; ++p) {
            const int64_t plo = piece_lo[m * max_pieces + p], phi = plo + piece_ln[m * max_pieces + p];
            int32_t k = (int32_t)(std::upper_bound(Eb, Eb + nc, plo) - Eb);  // first core with E > plo
            for (; k < nc; ++k) {
                const int64_t A = own[(k * NB + b) * 2], Ek = Eb[k];
                if (A >= phi) {
                    break;
                }
                if (Ek <= A || k == m) {
                    continue;
                }
                const int64_t s = std::max(A, plo), f = std::min(Ek, phi);
                out.insert(out.end(), {k, b, m, s, f, piece_l1[m * max_pieces + p] + (s - plo) * 4});
            }
        }
    };
    const int64_t npair = (int64_t)NB * nc;
    std::vector<std::vector<int64_t>> lists(npair);
#pragma omp parallel for schedule(dynamic, 8)
    for (int64_t i = 0; i < npair; ++i) {
        pair((int32_t)(i / nc), (int32_t)(i % nc), lists[i]);
    }
    std::vector<int64_t> at(npair + 1, 0);
    for (int64_t i = 0; i < npair; ++i) {
        at[i + 1] = at[i] + (int64_t)lists[i].size();
    }
    std::vector<int64_t> ent(at[npair]);
#pragma omp parallel for schedule(dynamic, 8)
    for (int64_t i = 0; i < npair; ++i) {
        std::copy(lists[i].begin(), lists[i].end(), ent.begin() + at[i]);
    }
    return ent;
}

}  // namespace

extern "C" {

// exchange_entries' entries (k, b, m, s, f, dst) into ent; with ent == nullptr only counts. Returns the number of
// entries.
int64_t hpcg_plan_exchange(
    int32_t nc,
    int32_t NB,
    const int64_t* own,
    int64_t max_pieces,
    const int64_t* piece_lo,
    const int64_t* piece_ln,
    const int64_t* piece_l1,
    const int64_t* npieces,
    int64_t* ent) {
    const std::vector<int64_t> e = exchange_entries(nc, NB, own, max_pieces, piece_lo, piece_ln, piece_l1, npieces);
    if (ent) {
        std::copy(e.begin(), e.end(), ent);
    }
    return (int64_t)e.size() / 6;
}

// Per-core metadata rows of one sparse-row launch (layout in tt/custom_ops/sparse_rows/kernels/common.h), the
// same rows as SparseRowPlan.meta's Python version. Inputs per core k: window pieces plo/pln/pl1[k * max_pieces + i], i < npieces[k],
// neighbours nbr[nat[k] .. nat[k+1]) (sorted), own (a, e) and own_l1 per block, exchange rows (5 words each)
// xrows[xat[k * NB + b] .. xat[k * NB + b + 1]). With m == nullptr, returns the longest row; otherwise writes rows
// into m ([nc][w] 32-bit words, each row's padding zeroed here: the table as uploaded) and the segment counts (all phases,
// Gauss-Seidel phases, phase 8) per core.
int64_t hpcg_plan_meta(
    int32_t nc,
    int32_t NB,
    int32_t nph,
    const int64_t* phases,
    int64_t max_pieces,
    const int64_t* plo,
    const int64_t* pln,
    const int64_t* pl1,
    const int64_t* npieces,
    const int64_t* nat,
    const int64_t* nbr,
    const int64_t* noc_xy,
    const int64_t* own,
    const int64_t* own_l1,
    const int64_t* xat,
    const int64_t* xrows,
    int64_t tpw,
    int64_t w,
    int32_t* m,
    int64_t* nseg,
    int64_t* nseg_gs,
    int64_t* nseg_r) {
    int64_t longest = 0;
    for (int32_t k = 0; k < nc; ++k) {
        const int64_t R = npieces[k], nnb = nat[k + 1] - nat[k];
        std::vector<int64_t> segs_in(NB);
        int64_t nsegs = 0, nx = 0;
        for (int32_t b = 0; b < NB; ++b) {
            const int64_t a = own[(k * NB + b) * 2], e = own[(k * NB + b) * 2 + 1];
            segs_in[b] = e > a ? (e - 1) / tpw - a / tpw + 1 : 0;  // segments never cross an L1 page (tpw words)
            nsegs += segs_in[b];
            nx += xat[k * NB + b + 1] - xat[k * NB + b];
        }
        const int64_t pos = 5 + 3 * R + 7 * NB + 3 * nnb + nph + 2;  // body start
        const int64_t len = pos + 2 * nsegs + 5 * nx;
        longest = std::max(longest, len);
        if (m == nullptr) {
            continue;
        }
        int32_t* r = m + (int64_t)k * w;
        std::fill(r, r + w, 0);  // m may be reused memory: the reserved words and the padding read as 0
        r[0] = (int32_t)(R);
        r[1] = (int32_t)(nph);
        r[2] = (int32_t)(nnb + 1);
        r[3] = (int32_t)(nnb);
        r[4] = (int32_t)(NB);
        for (int64_t i = 0; i < R; ++i) {
            r[5 + 3 * i] = (int32_t)(plo[k * max_pieces + i]);
            r[6 + 3 * i] = (int32_t)(pln[k * max_pieces + i]);
            r[7 + 3 * i] = (int32_t)(pl1[k * max_pieces + i]);
        }
        int32_t* blk = r + 5 + 3 * R;
        int32_t* nbw = blk + 7 * NB;
        for (int64_t i = 0; i < nnb; ++i) {
            const int64_t mm = nbr[nat[k] + i];
            const int64_t* f = std::lower_bound(nbr + nat[mm], nbr + nat[mm + 1], (int64_t)k);
            nbw[3 * i] = (int32_t)(noc_xy[2 * mm]);
            nbw[3 * i + 1] = (int32_t)(noc_xy[2 * mm + 1]);
            nbw[3 * i + 2] = (int32_t)(1 + (f - (nbr + nat[mm])));  // my slot on m
        }
        int32_t* ph = nbw + 3 * nnb;
        for (int32_t p = 0; p < nph; ++p) {
            ph[p] = (int32_t)(phases[p]);
        }
        int64_t at = pos;
        for (int32_t b = 0; b < NB; ++b) {
            const int64_t a = own[(k * NB + b) * 2], e = own[(k * NB + b) * 2 + 1];
            blk[7 * b] = (int32_t)(at);
            blk[7 * b + 1] = (int32_t)(segs_in[b]);
            for (int64_t q = a; q < e;) {
                const int64_t q1 = std::min(e, (q / tpw + 1) * tpw);
                r[at++] = (int32_t)(q);
                r[at++] = (int32_t)(q1);
                q = q1;
            }
        }
        for (int32_t b = 0; b < NB; ++b) {
            const int64_t x0 = xat[k * NB + b], x1 = xat[k * NB + b + 1];
            blk[7 * b + 2] = (int32_t)(at);
            blk[7 * b + 3] = (int32_t)(x1 - x0);
            blk[7 * b + 4] = (int32_t)(own[(k * NB + b) * 2]);
            blk[7 * b + 5] = (int32_t)(own[(k * NB + b) * 2 + 1]);
            blk[7 * b + 6] = (int32_t)(own_l1[k * NB + b]);
            for (int64_t i = 5 * x0; i < 5 * x1; ++i) {
                r[at++] = (int32_t)(xrows[i]);
            }
        }
        int64_t s = 0, sg = 0, sr = 0;
        for (int32_t p = 0; p < nph; ++p) {
            s += segs_in[phases[p] & 7];
            sg += phases[p] < 8 ? segs_in[phases[p]] : 0;
            sr += phases[p] == 8 ? segs_in[0] : 0;
        }
        nseg[k] = s;
        nseg_gs[k] = sg;
        nseg_r[k] = sr;
    }
    return longest;
}

// Window pieces of every core of a grid transfer launch (kind 1 restriction, 2 prolongation; see
// for_each_transfer_row), without the column table: the same pieces hpcg_plan_windows computes from Level.link's
// R / Pm with exchange = 0. Columns of consecutive out rows are evenly spaced (at most 2 apart) per x parity, so a
// run of rows marks whole granule ranges.
int64_t hpcg_plan_windows_transfer(
    int32_t kind,
    int32_t nx,
    int32_t nbands,
    int32_t nl,
    const int32_t* line_band,
    const int32_t* band_y0,
    const int32_t* band_ny,
    const int64_t* band_start,
    int32_t cnx,
    int32_t cnbands,
    int32_t cnl,
    const int32_t* cline_band,
    const int32_t* cband_y0,
    const int32_t* cband_ny,
    const int64_t* cband_start,
    int32_t nc,
    int32_t NB,
    const int64_t* own,
    int32_t gran,
    int32_t wgap,
    int64_t ng,
    int64_t empty_lo,
    int64_t max_pieces,
    int64_t* piece_lo,
    int64_t* piece_ln,
    int64_t* piece_l1,
    int64_t* npieces) {
    const Geo F{nx, nbands, nl, line_band, band_y0, band_ny, band_start};
    const Geo C{cnx, cnbands, cnl, cline_band, cband_y0, cband_ny, cband_start};
    int64_t bad = 0;
#pragma omp parallel
    {
        Granules mask(ng);
#pragma omp for schedule(dynamic, 1) reduction(+ : bad)
        for (int32_t k = 0; k < nc; ++k) {
            int64_t gmin = ng, gmax = -1;
            bool used = false;
            for (int32_t b = 0; b < NB; ++b) {
                const int64_t a = own[(k * NB + b) * 2], e = own[(k * NB + b) * 2 + 1];
                if (e <= a) {
                    continue;
                }
                used = true;
                // pending range per slot: consecutive runs' ranges join up (marked once per band-plane, not per line)
                int64_t plo[3] = {0, 0, 0}, phi[3] = {-2, -2, -2};
                auto mark = [&](int64_t lo, int64_t hi) {
                    mask.set_range(lo / gran, hi / gran);
                    gmin = std::min(gmin, lo / gran);
                    gmax = std::max(gmax, hi / gran);
                };
                for_each_transfer_range(kind, F, C, b, a, e, [&](int slot, int64_t lo, int64_t hi) {
                    if (lo >= plo[slot] && lo <= phi[slot] + 2) {  // + 2: restriction columns step by 2
                        phi[slot] = std::max(phi[slot], hi);
                        return;
                    }
                    if (phi[slot] >= plo[slot]) {
                        mark(plo[slot], phi[slot]);
                    }
                    plo[slot] = lo;
                    phi[slot] = hi;
                });
                for (int i = 0; i < 3; ++i) {
                    if (phi[i] >= plo[i]) {
                        mark(plo[i], phi[i]);
                    }
                }
            }
            if (!used) {
                mask.set(empty_lo / gran);
                gmin = gmax = empty_lo / gran;
            }
            const int64_t np = merge_pieces(
                mask,
                gmin,
                gmax,
                gran,
                wgap,
                max_pieces,
                piece_lo + k * max_pieces,
                piece_ln + k * max_pieces,
                piece_l1 + k * max_pieces);
            npieces[k] = np;
            if (np > max_pieces) {
                bad += 1;
            }
        }
    }
    return bad ? -1 : 0;
}

// The 1-slot tables of a grid transfer launch: off[q] = window byte offset of row q's column in its core's window,
// w[q] = 1 on real rows, 0 on padding rows (Level.link's Rw / Pw). Returns the number of columns found outside their
// core's pieces (0 when the plan is valid).
int64_t hpcg_transfer_tables(
    int32_t kind,
    int32_t nx,
    int32_t nbands,
    int32_t nl,
    const int32_t* line_band,
    const int32_t* band_y0,
    const int32_t* band_ny,
    const int64_t* band_start,
    int32_t cnx,
    int32_t cnbands,
    int32_t cnl,
    const int32_t* cline_band,
    const int32_t* cband_y0,
    const int32_t* cband_ny,
    const int64_t* cband_start,
    int32_t nc,
    int32_t NB,
    const int64_t* own,
    int64_t max_pieces,
    const int64_t* piece_lo,
    const int64_t* piece_ln,
    const int64_t* piece_l1,
    const int64_t* npieces,
    int32_t* off,
    float* w) {
    const Geo F{nx, nbands, nl, line_band, band_y0, band_ny, band_start};
    const Geo C{cnx, cnbands, cnl, cline_band, cband_y0, cband_ny, cband_start};
    int64_t bad = 0;
#pragma omp parallel for schedule(dynamic, 1) reduction(+ : bad)
    for (int32_t k = 0; k < nc; ++k) {
        const int64_t* lo = piece_lo + k * max_pieces;
        const int64_t* ln = piece_ln + k * max_pieces;
        const int64_t* l1 = piece_l1 + k * max_pieces;
        const int64_t np = npieces[k];
        int64_t p = 0;
        for (int32_t b = 0; b < NB; ++b) {
            const int64_t a = own[(k * NB + b) * 2], e = own[(k * NB + b) * 2 + 1];
            if (e <= a) {
                continue;
            }
            for_each_transfer_row(kind, F, C, b, a, e, [&](int64_t q, int64_t col, bool real) {
                if (!(p < np && lo[p] <= col && col < lo[p] + ln[p])) {  // the piece holding col (pieces sorted)
                    p = std::upper_bound(lo, lo + np, col) - lo - 1;
                    if (p < 0 || col >= lo[p] + ln[p]) {
                        ++bad;
                        p = 0;
                        return;
                    }
                }
                off[q] = (int32_t)(l1[p] + (col - lo[p]) * 4);
                w[q] = real ? 1.0f : 0.0f;
            });
        }
    }
    return bad;
}

// Each out block's padding column of a grid transfer launch (for_each_transfer_row: the column of the block's last
// real row), for the device-generated tables (tt/custom_ops/stencil_matrix); -1 for a block without real rows.
void hpcg_transfer_pad(
    int32_t kind,
    int32_t nx,
    int32_t nbands,
    int32_t nl,
    const int32_t* line_band,
    const int32_t* band_y0,
    const int32_t* band_ny,
    const int64_t* band_start,
    int32_t cnx,
    int32_t cnbands,
    int32_t cnl,
    const int32_t* cline_band,
    const int32_t* cband_y0,
    const int32_t* cband_ny,
    const int64_t* cband_start,
    int32_t NB,
    int64_t* pad) {
    const Geo F{nx, nbands, nl, line_band, band_y0, band_ny, band_start};
    const Geo C{cnx, cnbands, cnl, cline_band, cband_y0, cband_ny, cband_start};
    const Geo& O = kind == 1 ? C : F;
    for (int32_t b = 0; b < NB; ++b) {
        const int ob = kind == 1 ? b : 0;
        const int64_t real_end = O.band_start[(int64_t)ob * O.nbands] + O.count(ob);
        pad[b] = -1;
        if (O.count(ob) > 0) {
            for_each_transfer_row(kind, F, C, b, real_end - 1, real_end, [&](int64_t, int64_t col, bool) {
                pad[b] = col;
            });
        }
    }
}

// hpcg_plan_meta in one call: sizes the rows (their width the longest row rounded up to align words), and when
// nc * width words fit in cap, writes them into m. Returns the longest row's length, or -(words needed) when they do
// not fit.
int64_t hpcg_plan_meta_into(
    int32_t nc,
    int32_t NB,
    int32_t nph,
    const int64_t* phases,
    int64_t max_pieces,
    const int64_t* plo,
    const int64_t* pln,
    const int64_t* pl1,
    const int64_t* npieces,
    const int64_t* nat,
    const int64_t* nbr,
    const int64_t* noc_xy,
    const int64_t* own,
    const int64_t* own_l1,
    const int64_t* xat,
    const int64_t* xrows,
    int64_t tpw,
    int64_t align,
    int64_t cap,
    int32_t* m,
    int64_t* nseg,
    int64_t* nseg_gs,
    int64_t* nseg_r) {
    const int64_t longest = hpcg_plan_meta(
        nc, NB, nph, phases, max_pieces, plo, pln, pl1, npieces, nat, nbr, noc_xy, own, own_l1, xat, xrows, tpw, 0,
        nullptr, nseg, nseg_gs, nseg_r);
    const int64_t w = (longest + align - 1) / align * align;
    if ((int64_t)nc * w > cap) {
        return -(int64_t)nc * w;
    }
    hpcg_plan_meta(
        nc, NB, nph, phases, max_pieces, plo, pln, pl1, npieces, nat, nbr, noc_xy, own, own_l1, xat, xrows, tpw, w, m,
        nseg, nseg_gs, nseg_r);
    return longest;
}

// exchange_entries' entries, already as SparseRowPlan uses them: grouped by (k, b) in their original order (a
// stable counting sort), each as (NoC x, NoC y of m, s, f, dst) in rows[at[k * NB + b] .. at[k * NB + b + 1]); and
// every core's neighbours (the cores it sends rows to or receives rows from), sorted, in nbr[nat[k] .. nat[k + 1]].
// noc_xy[2 m], noc_xy[2 m + 1]: NoC x, y of core m. Returns the number of entries, or -needed when it exceeds cap
// (rows holds cap entries; nbr holds nc * nc).
int64_t hpcg_plan_exchange_grouped(
    int32_t nc,
    int32_t NB,
    const int64_t* own,
    int64_t max_pieces,
    const int64_t* piece_lo,
    const int64_t* piece_ln,
    const int64_t* piece_l1,
    const int64_t* npieces,
    const int64_t* noc_xy,
    int64_t cap,
    int64_t* rows,
    int64_t* at,
    int64_t* nbr,
    int64_t* nat) {
    const std::vector<int64_t> ent = exchange_entries(nc, NB, own, max_pieces, piece_lo, piece_ln, piece_l1, npieces);
    const int64_t n = (int64_t)ent.size() / 6;
    if (n > cap) {
        return -n;
    }
    const int64_t nkey = (int64_t)nc * NB;
    for (int64_t i = 0; i <= nkey; ++i) {
        at[i] = 0;
    }
    for (int64_t i = 0; i < n; ++i) {
        ++at[ent[i * 6] * NB + ent[i * 6 + 1] + 1];
    }
    for (int64_t i = 0; i < nkey; ++i) {
        at[i + 1] += at[i];
    }
    std::vector<int64_t> fill(at, at + nkey);
    std::vector<uint8_t> adj((size_t)nc * nc, 0);
    for (int64_t i = 0; i < n; ++i) {
        const int64_t* e = &ent[i * 6];
        int64_t* r = rows + 5 * fill[e[0] * NB + e[1]]++;
        r[0] = noc_xy[2 * e[2]];
        r[1] = noc_xy[2 * e[2] + 1];
        r[2] = e[3];
        r[3] = e[4];
        r[4] = e[5];
        adj[e[0] * nc + e[2]] = adj[e[2] * nc + e[0]] = 1;
    }
    int64_t m = 0;
    for (int32_t k = 0; k < nc; ++k) {
        nat[k] = m;
        for (int32_t j = 0; j < nc; ++j) {
            if (adj[(size_t)k * nc + j]) {
                nbr[m++] = j;
            }
        }
    }
    nat[nc] = m;
    return n;
}

}  // extern "C"
