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
    double* B) {
    const int64_t n = (int64_t)nx * nx * nx;
#pragma omp parallel for schedule(static)
    for (int64_t q = 0; q < P; ++q) {
        nat_at[q] = -1;
    }
#pragma omp parallel for schedule(static)
    for (int32_t iz = 0; iz < nx; ++iz) {
        for (int32_t iy = 0; iy < nx; ++iy) {
            for (int32_t ix = 0; ix < nx; ++ix) {
                const int px = ix & 1, py = iy & 1, pz = iz & 1, c = px + 2 * py + 4 * pz;
                const int64_t cx = (nx - px + 1) / 2;
                const int j = line_band[py * nl + (iy >> 1)];
                const int64_t k = (int64_t)c * nbands + j;
                const int64_t q =
                    band_start[k] + ((int64_t)(iz >> 1) * band_ny[k] + ((iy >> 1) - band_y0[k])) * cx + (ix >> 1);
                const int64_t nat = ((int64_t)iz * nx + iy) * nx + ix;
                pos[nat] = q;
                nat_at[q] = nat;
            }
        }
    }
    int64_t nnz = 0;
#pragma omp parallel for schedule(static) reduction(+ : nnz)
    for (int32_t iz = 0; iz < nx; ++iz) {
        for (int32_t iy = 0; iy < nx; ++iy) {
            for (int32_t ix = 0; ix < nx; ++ix) {
                const int64_t nat = ((int64_t)iz * nx + iy) * nx + ix;
                const int64_t q = pos[nat];
                int cnt = 0;
                for (int sz = -1; sz <= 1; ++sz) {
                    for (int sy = -1; sy <= 1; ++sy) {
                        for (int sx = -1; sx <= 1; ++sx) {
                            const int j = (sz + 1) * 9 + (sy + 1) * 3 + (sx + 1);
                            const int32_t jx = ix + sx, jy = iy + sy, jz = iz + sz;
                            if (jx < 0 || jx >= nx || jy < 0 || jy >= nx || jz < 0 || jz >= nx) {
                                COL[j * stride + q] = (int32_t)q;  // missing neighbour: a zero entry on the row itself
                                V[j * stride + q] = 0.0f;
                                continue;
                            }
                            V[j * stride + q] = (j == 13) ? 26.0f : -1.0f;
                            COL[j * stride + q] = (int32_t)pos[((int64_t)jz * nx + jy) * nx + jx];
                            ++cnt;
                        }
                    }
                }
                B[q] = 26.0 - (cnt - 1);
                nnz += cnt;
            }
        }
    }
    // padding rows (at the end of each colour block): zero entries on the row itself, diagonal 1 so the Gauss-Seidel
    // update keeps them at 0
#pragma omp parallel for schedule(static)
    for (int64_t q = 0; q < P; ++q) {
        if (nat_at[q] >= 0) {
            continue;
        }
        B[q] = 0.0;
        for (int j = 0; j < 27; ++j) {
            COL[j * stride + q] = (int32_t)q;
            V[j * stride + q] = 0.0f;
        }
        V[13 * stride + q] = 1.0f;
    }
    (void)n;
    return nnz;
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
            int64_t np = 0, l1 = 0, start = -1, last = -1;
            for (int64_t g = gmin; g <= gmax + 1; ++g) {
                const bool on = g <= gmax && mask[g];
                if (on) {
                    if (start < 0) {
                        start = g;
                    } else if (g - last > wgap) {  // close the previous run
                        if (np < max_pieces) {
                            piece_lo[k * max_pieces + np] = start * gran;
                            piece_ln[k * max_pieces + np] = (last + 1 - start) * gran;
                            piece_l1[k * max_pieces + np] = l1;
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
                    piece_lo[k * max_pieces + np] = start * gran;
                    piece_ln[k * max_pieces + np] = (last + 1 - start) * gran;
                    piece_l1[k * max_pieces + np] = l1;
                }
                ++np;
            }
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

}  // extern "C"
