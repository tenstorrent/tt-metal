// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// One multigrid level in the device storage order (reference/problem.py's Level with tables=False): 8 colour blocks
// c = (x & 1) + 2 (y & 1) + 4 (z & 1), each starting on a 1024-word boundary, rows band by band in y inside a block.
#pragma once
#include <array>
#include <cstdint>
#include <vector>

#include "../../tt/shared_constants.h"  // TILE and the other constants shared with the Python ops and the kernels

namespace hpcg_tt {

struct Level {
    int32_t nx = 0, bands = 1;
    int64_t n = 0;       // grid points, nx^3
    int64_t P = 0;       // vector length in storage order (colour blocks padded to 1024 words)
    int64_t stride = 0;  // row length of the 27-slot value / offset tables (an odd number of 1024-word pages)
    std::array<int64_t, 9> starts{};            // colour block c is [starts[c], starts[c + 1])
    std::vector<std::vector<int64_t>> bstarts;  // per colour block, its band starts and its end
    int32_t nb = 0, nl = 0;                     // bands per block, colour-lines per axis
    std::vector<int32_t> line_band, band_y0, band_ny;
    std::vector<int64_t> band_start;

    Level(int32_t nx, int32_t bands);
    int64_t count(int c) const;                           // real rows of colour block c
    int64_t q_of(int32_t x, int32_t y, int32_t z) const;  // storage position of grid point (x, y, z)
};

// y-bands per colour block for a level of nx^3 points (tt/hpcg.py's default_bands)
int32_t default_bands(int32_t nx_level);
// pages per slot of the 27-slot tables: odd, so a segment's 27 slot pieces sit in different DRAM banks
int64_t slot_pages(int64_t n_tiles);

}  // namespace hpcg_tt
