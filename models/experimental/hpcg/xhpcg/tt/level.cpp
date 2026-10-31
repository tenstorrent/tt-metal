// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "level.hpp"

#include <algorithm>

namespace hpcg_tt {

int32_t default_bands(int32_t nx_level) { return nx_level >= 160 ? 10 : nx_level >= 112 ? 5 : 1; }

int64_t slot_pages(int64_t n_tiles) { return n_tiles + (n_tiles % 2 == 0 ? 1 : 0); }

Level::Level(int32_t nx_, int32_t bands_) : nx(nx_), bands(bands_) {
    n = (int64_t)nx * nx * nx;
    const int32_t h = std::max(1, nx / 2);
    // lines per band for each y parity (the colours share these two): line i of a parity is in band i * bands / h.
    // An odd nx has one line more than h in its even parity, which can fall in one band more than asked for (13^3:
    // bands 0 and 1); the layout then has that band, as numpy's bincount (minlength=bands) gives it
    std::array<std::vector<int64_t>, 2> lines;
    for (int py = 0; py < 2; ++py) {
        lines[py].assign(bands, 0);
        const int32_t cy = (nx - py + 1) / 2;
        for (int32_t i = 0; i < cy; ++i) {
            const int64_t band = (int64_t)i * bands / h;
            if (band >= (int64_t)lines[py].size()) {
                lines[py].resize(band + 1, 0);
            }
            ++lines[py][band];
        }
    }
    int64_t p = 0;
    bstarts.assign(8, {});
    for (int c = 0; c < 8; ++c) {
        const int px = c & 1, py = (c >> 1) & 1, pz = c >> 2;
        const int64_t cx = (nx - px + 1) / 2, cy = (nx - py + 1) / 2, cz = (nx - pz + 1) / 2;
        starts[c] = p;
        bstarts[c].push_back(p);
        for (int64_t nl_band : lines[py]) {
            bstarts[c].push_back(bstarts[c].back() + nl_band * cx * cz);
        }
        p = (p + cx * cy * cz + TILE - 1) / TILE * TILE;
    }
    starts[8] = p;
    P = p;
    stride = slot_pages(P / TILE) * TILE;

    // the storage order as small tables (host/native_setup.cpp's Geo; reference/problem.py's _band_tables)
    nb = 0;
    for (const auto& b : bstarts) {
        nb = std::max(nb, (int32_t)b.size() - 1);
    }
    nl = (nx + 1) / 2;
    line_band.assign(2 * nl, 0);
    band_y0.assign(8 * nb, 0);
    band_ny.assign(8 * nb, 0);
    band_start.assign(8 * nb, 0);
    std::array<std::vector<int32_t>, 2> ny_py, y0_py;
    for (int py = 0; py < 2; ++py) {
        const int32_t cy = (nx - py + 1) / 2;
        ny_py[py].assign(nb, 0);
        y0_py[py].assign(nb, 0);
        for (int32_t i = 0; i < cy; ++i) {
            const int32_t lb = (int32_t)((int64_t)i * bands / h);  // nondecreasing: band j's lines are one run
            line_band[py * nl + i] = lb;
            if (ny_py[py][lb]++ == 0) {
                y0_py[py][lb] = i;
            }
        }
    }
    for (int c = 0; c < 8; ++c) {
        const int py = (c >> 1) & 1;
        const int nbc = (int)bstarts[c].size() - 1;
        for (int j = 0; j < nb; ++j) {
            band_start[c * nb + j] = bstarts[c].back();  // bands a colour lacks: empty, at the block's end
        }
        for (int j = 0; j < nbc; ++j) {
            band_y0[c * nb + j] = y0_py[py][j];
            band_ny[c * nb + j] = ny_py[py][j];
            band_start[c * nb + j] = bstarts[c][j];
        }
    }
}

int64_t Level::count(int c) const {
    const int64_t cx = (nx - (c & 1) + 1) / 2, cy = (nx - ((c >> 1) & 1) + 1) / 2, cz = (nx - (c >> 2) + 1) / 2;
    return cx * cy * cz;
}

int64_t Level::q_of(int32_t x, int32_t y, int32_t z) const {
    // host/native_setup.cpp's Geo::q_of: the line's first q in its colour block and band, then x
    const int cs = (x & 1) + 2 * (y & 1) + 4 * (z & 1);
    const int64_t k = (int64_t)cs * nb + line_band[(y & 1) * nl + (y >> 1)];
    const int64_t cx = (nx - (cs & 1) + 1) / 2;
    return band_start[k] + ((int64_t)(z >> 1) * band_ny[k] + ((y >> 1) - band_y0[k])) * cx + (x >> 1);
}

}  // namespace hpcg_tt
