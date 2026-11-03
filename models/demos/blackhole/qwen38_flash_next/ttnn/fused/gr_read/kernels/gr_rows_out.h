// SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// QWEN38_GR_ROWS_FAST=1 (GR gate v3, rows 2..8): the write of one 8x32 BF16 tile (two 8-row faces of 256 B in the CB)
// into the 32x32 TILE page of a DRAM tensor (face 0 at page byte 0, face 1 at 512, faces 2 and 3 at 1024).
//   GT_ZERO_CB defined (live form: the page must keep the 32-row contract, dead rows exact zeros): rows GT_ROWS..7 of
//   both faces
//     are zeroed in L1, both full faces are written, and rows 8..15 of both faces and faces 2, 3 are written from the
//     zero tile of GT_ZERO_CB (>= 512 B): four writes, one barrier for all six.
//   not defined: only rows 0..GT_ROWS-1 of both faces are written (the other rows of the page are not written).
// The caller reads GT_DST_FACE (= 512, the byte of face 1 in the page).

#pragma once

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/tensor/noc_traits.h"

template <typename Accessor>
FORCE_INLINE void gr_rows_write_tile(const Accessor& acc, uint32_t page, uint32_t tile_addr, uint32_t tile_bytes) {
    const uint32_t half = tile_bytes / 2;  // one 8-row face in the CB
#ifdef GT_ZERO_CB
    for (uint32_t f = 0; f < 2; ++f) {
        volatile tt_l1_ptr uint32_t* p =
            reinterpret_cast<volatile tt_l1_ptr uint32_t*>(tile_addr + f * half + GT_ROWS * 32);
        for (uint32_t w = 0; w < (half - GT_ROWS * 32) / 4; ++w) {
            p[w] = 0;
        }
    }
    noc_async_write(tile_addr, acc.get_noc_addr(page, 0), half);
    noc_async_write(tile_addr + half, acc.get_noc_addr(page, GT_DST_FACE), half);
    const uint32_t z = DataflowBuffer(GT_ZERO_CB).get_read_ptr();
    noc_async_write(z, acc.get_noc_addr(page, half), half);
    noc_async_write(z, acc.get_noc_addr(page, GT_DST_FACE + half), half);
    noc_async_write(z, acc.get_noc_addr(page, 2 * GT_DST_FACE), GT_DST_FACE);
    noc_async_write(z, acc.get_noc_addr(page, 3 * GT_DST_FACE), GT_DST_FACE);
#else
    noc_async_write(tile_addr, acc.get_noc_addr(page, 0), GT_ROWS * 32);
    noc_async_write(tile_addr + half, acc.get_noc_addr(page, GT_DST_FACE), GT_ROWS * 32);
#endif
}
