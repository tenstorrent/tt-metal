// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Stage 7b MLA megakernel: constants shared by the reader-core kernels (tt/mk_mla.py MlaReaders) and the latent core.
//
// A reader phase is NH sequential sub-matmuls y_j[T, NR] = x_j[T, KG] @ W_j[KG, NR] (j = 0 .. NH-1); a dense DS matmul
// is NH = 1, KG = K tiles. W streams from this reader's DRAM bank as NB = NH * KG / KB blocks of [KB, NR] tiles (the
// virtual DS layout of tt/mk_mla.py: rows j * KG .. of the reader's column slice belong to sub-matmul j), so the
// weight streamer (kernels/mk/reader_w.cpp) is the dense one, unchanged.
//   x_j  = tiles [(H0 + j) * KG, + KG) of the phase input   (SRC_TENSOR: L1 / DRAM interleaved, page = column tile)
//        = RMSNorm(x[0 : KX]) blocks                         (SRC_NORM: x pulled once, normalised by the compute)
//        = head (H0 + j)'s rows of a [1, T, H, KG * 32] tensor (SRC_ROWS: face-row gather, row t = token t)
//   y_j  -> output column tiles N0 + j * NG + c              (DST_TENSOR, c < NR)
//        -> head (N0 + j * NG + c) / NG's row of a [1, T, H, NG * 32] tensor (DST_ROWS: face-row scatter)
#pragma once

#include <cstdint>

namespace mla {
constexpr uint32_t SRC_TENSOR = 0, SRC_NORM = 1, SRC_ROWS = 2;
constexpr uint32_t DST_TENSOR = 0, DST_ROWS = 1;
constexpr uint32_t SKIP = 0xFFFFFFFFu;  // n0 of a reader that sits the phase out
constexpr uint32_t TILE_BYTES_BF16 = 2048;

// byte offset of row r's left 16-element face row inside a 32 x 32 bf16 tile (the right half is + 512)
inline uint32_t row_off(uint32_t r) { return (r >> 4) * 1024 + (r & 15) * 32; }
}  // namespace mla

#if !defined(COMPILE_FOR_TRISC)
#include "api/dataflow/dataflow_api.h"

namespace mla {
// Sub-tile (32-B face row) moves between cores. Measured on Blackhole (the W_uv gather): a NoC read whose
// source and destination differ mod 64 lands the source's 64-B-aligned-down bytes. Every face-row move therefore keeps
// the NoC transfer congruent mod 64 (a 64-B aligned staging slot, kernels/mla/ra.cpp gather_block / scatter_rows) and
// moves the 32 B within L1 with the RISC (invalidate_l1_cache = Blackhole's fence: orders the RISC's loads after the
// NoC data and its stores before a later NoC transfer / CB push).
// local L1 -> local L1, 32 B (RISC copy)
FORCE_INLINE void copy_row32(uint32_t src, uint32_t dst) {
    volatile tt_l1_ptr uint32_t* s = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(src);
    volatile tt_l1_ptr uint32_t* d = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(dst);
    for (uint32_t w = 0; w < 8; ++w) {
        d[w] = s[w];
    }
}
}  // namespace mla
#endif
