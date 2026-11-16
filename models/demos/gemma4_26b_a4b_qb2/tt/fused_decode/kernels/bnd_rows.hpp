// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// Row-0 <-> flat moves of the Gemma decode boundary (fused_decode/boundary.py), shared by bnd_reader.cpp and
// bnd_sender.cpp, which each take half of the tiles. Tile j of a [1, 1, 32, hidden] TILE tensor holds values
// 32 j .. 32 j + 31 of row 0 in face 0 row 0 (first 16) and face 1 row 0 (next 16); the flat vector holds value h at
// element h.
#pragma once

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"

// row 0 of BF16 tiles [j0, j1) of `src` -> flat BF16 at `dst` (L1 sources only: 32 B reads)
template <typename Acc>
FORCE_INLINE void gather_rows(const Acc& src, uint32_t dst, uint32_t j0, uint32_t j1) {
    for (uint32_t j = j0; j < j1; ++j) {
        noc_async_read(src.get_noc_addr(j, 0), dst + 64 * j, 32);
        noc_async_read(src.get_noc_addr(j, 512), dst + 64 * j + 32, 32);
    }
}

// flat `src` (BF16 or FP32) -> row 0 of tiles [j0, j1) of `dst`
template <typename Acc>
FORCE_INLINE void write_rows(const Acc& dst, uint32_t src, uint32_t j0, uint32_t j1, bool fp32) {
    if (fp32) {
        for (uint32_t j = j0; j < j1; ++j) {
            noc_async_write(src + 128 * j, dst.get_noc_addr(j, 0), 64);
            noc_async_write(src + 128 * j + 64, dst.get_noc_addr(j, 1024), 64);
        }
    } else {
        for (uint32_t j = j0; j < j1; ++j) {
            noc_async_write(src + 64 * j, dst.get_noc_addr(j, 0), 32);
            noc_async_write(src + 64 * j + 32, dst.get_noc_addr(j, 512), 32);
        }
    }
}
