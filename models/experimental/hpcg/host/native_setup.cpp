// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Host-side setup of the HPCG solver, compiled once and loaded with ctypes (host/native.py): the matrix of one
// multigrid level, and the per-core work plan of one sparse-row launch. Each function does in one pass what the
// numpy versions (reference/problem.py, tt/custom_ops/sparse_rows/plan.py) do with many array passes; the outputs
// are identical. Loops over grid planes and over cores run in parallel with OpenMP.

#include <algorithm>
#include <cstdint>
#include <vector>

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

// Runs of marked granules in [gmin, gmax] whose gaps are at most wgap granules -> window pieces (q lo, words, L1
// byte offset), the first max_pieces of them stored; clears the mask. Returns the number of pieces.
int64_t merge_pieces(
    std::vector<uint8_t>& mask,
    int64_t gmin,
    int64_t gmax,
    int32_t gran,
    int32_t wgap,
    int64_t max_pieces,
    int64_t* plo,
    int64_t* pln,
    int64_t* pl1) {
    int64_t np = 0, l1 = 0, start = -1, last = -1;
    for (int64_t g = gmin; g <= gmax + 1; ++g) {
        const bool on = g <= gmax && mask[g];
        if (on) {
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
            mask[g] = 0;
        }
    }
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

}  // namespace

extern "C" {

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
// Pw[q]: 1 on mapped rows. Returns the largest R of a real row.
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
    int64_t* R,
    double* Rw,
    int64_t* Pm,
    double* Pw) {
    const Geo F{nx, nbands, nl, line_band, band_y0, band_ny, band_start};
    const Geo C{cnx, cnbands, cnl, cline_band, cband_y0, cband_ny, cband_start};
    for (int64_t qc = 0; qc < CP; ++qc) {
        Rw[qc] = 0.0;
    }
#pragma omp parallel for schedule(dynamic, 1)
    for (int c = 0; c < 8; ++c) {  // the coarse colour blocks in parallel
        const int64_t a = C.band_start[(int64_t)c * C.nbands];
        C.for_each_run(c, a, a + C.count(c), [&](int64_t q, int32_t i0, int32_t len, int32_t y, int32_t z) {
            for (int32_t i = 0; i < len; ++i) {
                R[q + i] = F.q_of(2 * (2 * (i0 + i) + (c & 1)), 2 * y, 2 * z);
                Rw[q + i] = 1.0;
            }
        });
    }
    int64_t rmax = -1, last = 0;
    for (int64_t qc = 0; qc < CP; ++qc) {
        if (Rw[qc] > 0.0) {
            rmax = std::max(rmax, R[qc]);
            last = qc;
        } else {
            // R[last] is final (last < qc); with no real row yet, last = 0 and R[0] = 0, as in numpy
            R[qc] = qc == 0 ? 0 : R[last];
        }
    }
    for (int64_t q = 0; q < nb0; ++q) {
        Pm[q] = 0;
        Pw[q] = 0.0;
    }
    for (int64_t qc = 0; qc < CP; ++qc) {
        if (Rw[qc] > 0.0 && R[qc] < nb0) {
            Pm[R[qc]] = qc;
            Pw[R[qc]] = 1.0;
        }
    }
    int64_t lastm = 0;
    for (int64_t q = 0; q < nb0; ++q) {
        if (Pw[q] > 0.0) {
            lastm = q;
        } else {
            Pm[q] = Pm[lastm];
        }
    }
    return rmax;
}

// Window pieces of every core of a Gauss-Seidel launch on the 27-point matrix of one level, from the storage-order
// formula instead of the column table: the same pieces hpcg_plan_windows computes from COL with exchange = 1 (the
// columns of a run of rows along x, for one neighbour line, are a contiguous range of q).
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
    int64_t* npieces) {
    const Geo G{nx, nbands, nl, line_band, band_y0, band_ny, band_start};
    int64_t bad = 0;
#pragma omp parallel
    {
        std::vector<uint8_t> mask(ng, 0);
#pragma omp for schedule(dynamic, 1) reduction(+ : bad)
        for (int32_t k = 0; k < nc; ++k) {
            int64_t gmin = ng, gmax = -1;
            auto mark = [&](int64_t lo, int64_t hi) {  // granules of q range [lo, hi]
                for (int64_t g = lo / gran; g <= hi / gran; ++g) {
                    mask[g] = 1;
                }
                gmin = std::min(gmin, lo / gran);
                gmax = std::max(gmax, hi / gran);
            };
            bool used = false;
            for (int32_t b = 0; b < NB; ++b) {
                const int64_t a = own[(k * NB + b) * 2], e = own[(k * NB + b) * 2 + 1];
                if (e <= a) {
                    continue;
                }
                used = true;
                mark(a, e - 1);  // own rows (padding rows' columns are themselves)
                G.for_each_run(b, a, e, [&](int64_t, int32_t i0, int32_t len, int32_t iy, int32_t iz) {
                    const int px = b & 1;
                    for (int sz = -1; sz <= 1; ++sz) {
                        for (int sy = -1; sy <= 1; ++sy) {
                            const int32_t jy = iy + sy, jz = iz + sz;
                            if (jy < 0 || jy >= nx || jz < 0 || jz >= nx) {
                                continue;  // missing neighbours: columns on the row itself
                            }
                            const int cs = px + 2 * (jy & 1) + 4 * (jz & 1), co = cs ^ 1;
                            const int64_t bs = G.line_q(cs, jy, jz), bo = G.line_q(co, jy, jz);
                            mark(bs + i0, bs + i0 + len - 1);  // sx = 0
                            // sx = -1 and +1 (the other x parity): x index ix2 - 1 .. ix2 (px = 0) or ix2 .. ix2 + 1
                            const int32_t cxo = G.cx(co & 1);
                            const int32_t lo = std::max(0, i0 - (1 - px)), hi = std::min(cxo - 1, i0 + len - 1 + px);
                            if (lo <= hi) {
                                mark(bo + lo, bo + hi);
                            }
                        }
                    }
                });
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
        std::vector<uint8_t> mask(ng, 0);
#pragma omp for schedule(dynamic, 1) reduction(+ : bad)
        for (int32_t k = 0; k < nc; ++k) {
            int64_t gmin = ng, gmax = -1;
            auto mark = [&](int64_t g) {
                mask[g] = 1;
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

// Exchange entries of a Gauss-Seidel launch: for each block b, each core m that owns rows, and each piece of m's
// window, every other core k whose own rows of block b overlap the piece gets an entry (k, b, m, s, f, dst): rows
// [s, f) go to byte offset dst of m's window. With ent == nullptr only counts. Own ranges of a block are sorted by
// core. Returns the number of entries.
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
    int64_t n = 0;
    std::vector<uint8_t> has_rows(nc, 0);
    for (int32_t m = 0; m < nc; ++m) {
        for (int32_t b = 0; b < NB; ++b) {
            if (own[(m * NB + b) * 2 + 1] > own[(m * NB + b) * 2]) {
                has_rows[m] = 1;
            }
        }
    }
    std::vector<int64_t> E(nc);
    for (int32_t b = 0; b < NB; ++b) {
        for (int32_t k = 0; k < nc; ++k) {
            E[k] = own[(k * NB + b) * 2 + 1];
        }
        for (int32_t m = 0; m < nc; ++m) {
            if (!has_rows[m]) {
                continue;  // a core without rows reads no window: its placeholder piece gets nothing
            }
            for (int64_t p = 0; p < npieces[m]; ++p) {
                const int64_t plo = piece_lo[m * max_pieces + p], phi = plo + piece_ln[m * max_pieces + p];
                int32_t k =
                    (int32_t)(std::upper_bound(E.begin(), E.end(), plo) - E.begin());  // first core with E > plo
                for (; k < nc; ++k) {
                    const int64_t A = own[(k * NB + b) * 2], Ek = E[k];
                    if (A >= phi) {
                        break;
                    }
                    if (Ek <= A || k == m) {
                        continue;
                    }
                    const int64_t s = std::max(A, plo), f = std::min(Ek, phi);
                    if (ent) {
                        int64_t* e = ent + n * 6;
                        e[0] = k;
                        e[1] = b;
                        e[2] = m;
                        e[3] = s;
                        e[4] = f;
                        e[5] = piece_l1[m * max_pieces + p] + (s - plo) * 4;
                    }
                    ++n;
                }
            }
        }
    }
    return n;
}

// Per-core metadata rows of one sparse-row launch (layout in tt/custom_ops/sparse_rows/kernels/common.h), the
// same rows as SparseRowPlan.meta's Python version. Inputs per core k: window pieces plo/pln/pl1[pat[k] .. pat[k+1]),
// neighbours nbr[nat[k] .. nat[k+1]) (sorted), own (a, e) and own_l1 per block, exchange rows (5 words each)
// xrows[xat[k * NB + b] .. xat[k * NB + b + 1]). With m == nullptr, returns the longest row; otherwise writes rows
// into m ([nc][w], zero-filled) and the segment counts (all phases, Gauss-Seidel phases, phase 8) per core.
int64_t hpcg_plan_meta(
    int32_t nc,
    int32_t NB,
    int32_t nph,
    const int64_t* phases,
    const int64_t* pat,
    const int64_t* plo,
    const int64_t* pln,
    const int64_t* pl1,
    const int64_t* nat,
    const int64_t* nbr,
    const int64_t* noc_xy,
    const int64_t* own,
    const int64_t* own_l1,
    const int64_t* xat,
    const int64_t* xrows,
    int64_t tpw,
    int64_t w,
    int64_t* m,
    int64_t* nseg,
    int64_t* nseg_gs,
    int64_t* nseg_r) {
    int64_t longest = 0;
    for (int32_t k = 0; k < nc; ++k) {
        const int64_t R = pat[k + 1] - pat[k], nnb = nat[k + 1] - nat[k];
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
        int64_t* r = m + (int64_t)k * w;
        r[0] = R;
        r[1] = nph;
        r[2] = nnb + 1;
        r[3] = nnb;
        r[4] = NB;
        for (int64_t i = 0; i < R; ++i) {
            r[5 + 3 * i] = plo[pat[k] + i];
            r[6 + 3 * i] = pln[pat[k] + i];
            r[7 + 3 * i] = pl1[pat[k] + i];
        }
        int64_t* blk = r + 5 + 3 * R;
        int64_t* nbw = blk + 7 * NB;
        for (int64_t i = 0; i < nnb; ++i) {
            const int64_t mm = nbr[nat[k] + i];
            const int64_t* f = std::lower_bound(nbr + nat[mm], nbr + nat[mm + 1], (int64_t)k);
            nbw[3 * i] = noc_xy[2 * mm];
            nbw[3 * i + 1] = noc_xy[2 * mm + 1];
            nbw[3 * i + 2] = 1 + (f - (nbr + nat[mm]));  // my slot on m
        }
        int64_t* ph = nbw + 3 * nnb;
        for (int32_t p = 0; p < nph; ++p) {
            ph[p] = phases[p];
        }
        int64_t at = pos;
        for (int32_t b = 0; b < NB; ++b) {
            const int64_t a = own[(k * NB + b) * 2], e = own[(k * NB + b) * 2 + 1];
            blk[7 * b] = at;
            blk[7 * b + 1] = segs_in[b];
            for (int64_t q = a; q < e;) {
                const int64_t q1 = std::min(e, (q / tpw + 1) * tpw);
                r[at++] = q;
                r[at++] = q1;
                q = q1;
            }
        }
        for (int32_t b = 0; b < NB; ++b) {
            const int64_t x0 = xat[k * NB + b], x1 = xat[k * NB + b + 1];
            blk[7 * b + 2] = at;
            blk[7 * b + 3] = x1 - x0;
            blk[7 * b + 4] = own[(k * NB + b) * 2];
            blk[7 * b + 5] = own[(k * NB + b) * 2 + 1];
            blk[7 * b + 6] = own_l1[k * NB + b];
            for (int64_t i = 5 * x0; i < 5 * x1; ++i) {
                r[at++] = xrows[i];
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

}  // extern "C"
