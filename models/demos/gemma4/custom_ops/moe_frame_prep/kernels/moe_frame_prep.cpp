// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
// MoeFramePrep: one data-movement kernel on 32 cores. Core u produces token u of this chip's slice of the decode MoE
// frame: frame row f = rowbase + u (rowbase = r * 32 on mesh row r). For f < R it untilizes hidden row f (two 32-byte
// face rows per column tile of the TILE-layout DRAM tensor) into the x row, copies the packed routing row's weights to
// the scores row and converts its bf16 expert ids to uint16; for f >= R (the frame's padding) it writes a zero x row,
// zero scores and the distinct ids 0..k-1.
//
// compile-time args: [cb_scratch, R, cols_t, K, packed_page, rowbase_page, x_page, ids_page, scores_page]
//                    + TensorAccessorArgs(hidden, packed, rowbase, x, ids, scores)
// runtime args: [hidden_addr, packed_addr, rowbase_addr, x_addr, ids_addr, scores_addr, my_token]
#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#include "api/tensor/noc_traits.h"

constexpr uint32_t CB_SCRATCH = get_compile_time_arg_val(0);
constexpr uint32_t R = get_compile_time_arg_val(1);
constexpr uint32_t COLS_T = get_compile_time_arg_val(2);
constexpr uint32_t K = get_compile_time_arg_val(3);
constexpr uint32_t PACKED_PAGE = get_compile_time_arg_val(4);
constexpr uint32_t ROWBASE_PAGE = get_compile_time_arg_val(5);
constexpr uint32_t X_PAGE = get_compile_time_arg_val(6);
constexpr uint32_t IDS_PAGE = get_compile_time_arg_val(7);
constexpr uint32_t SCORES_PAGE = get_compile_time_arg_val(8);
constexpr auto hidden_args = TensorAccessorArgs<9>();
constexpr auto packed_args = TensorAccessorArgs<hidden_args.next_compile_time_args_offset()>();
constexpr auto rowbase_args = TensorAccessorArgs<packed_args.next_compile_time_args_offset()>();
constexpr auto x_args = TensorAccessorArgs<rowbase_args.next_compile_time_args_offset()>();
constexpr auto ids_args = TensorAccessorArgs<x_args.next_compile_time_args_offset()>();
constexpr auto scores_args = TensorAccessorArgs<ids_args.next_compile_time_args_offset()>();
constexpr uint32_t TILE_BYTES = 2048;
constexpr uint32_t FACE_ROW_BYTES = 32;  // 16 bf16 of one face row
constexpr uint32_t ROW_BYTES = COLS_T * 64;
// scratch layout from a 64-byte aligned base: hidden row | face-row staging (one 64 B slot per face row, the row
// landing at the source's offset modulo 64: DRAM reads keep src and dst congruent modulo the DRAM alignment) |
// packed row | ids row | scores row (64-byte slots)
constexpr uint32_t OFF_STAGE = ROW_BYTES;
constexpr uint32_t OFF_PACKED = OFF_STAGE + 2 * COLS_T * 64;
constexpr uint32_t OFF_IDS = OFF_PACKED + 64;
constexpr uint32_t OFF_SCORES = OFF_PACKED + 128;

union FU32 {
    float f;
    uint32_t u;
};

FORCE_INLINE float bf16_to_f32(uint16_t b) {
    FU32 v;
    v.u = uint32_t(b) << 16;
    return v.f;
}

void kernel_main() {
    const uint32_t hidden_addr = get_arg_val<uint32_t>(0);
    const uint32_t packed_addr = get_arg_val<uint32_t>(1);
    const uint32_t rowbase_addr = get_arg_val<uint32_t>(2);
    const uint32_t x_addr = get_arg_val<uint32_t>(3);
    const uint32_t ids_addr = get_arg_val<uint32_t>(4);
    const uint32_t scores_addr = get_arg_val<uint32_t>(5);
    const uint32_t me = get_arg_val<uint32_t>(6);

    const uint32_t scratch = (get_write_ptr(CB_SCRATCH) + 63u) & ~63u;  // the CB carries 64 bytes of slack
    const auto s_hidden = TensorAccessor(hidden_args, hidden_addr, TILE_BYTES);
    const auto s_packed = TensorAccessor(packed_args, packed_addr, PACKED_PAGE);
    const auto s_rowbase = TensorAccessor(rowbase_args, rowbase_addr, ROWBASE_PAGE);
    const auto s_x = TensorAccessor(x_args, x_addr, X_PAGE);
    const auto s_ids = TensorAccessor(ids_args, ids_addr, IDS_PAGE);
    const auto s_scores = TensorAccessor(scores_args, scores_addr, SCORES_PAGE);

    // this chip's first frame row (the packed slot doubles as the landing buffer: 16 x u32 fit in 64 B)
    noc_async_read(s_rowbase.get_noc_addr(0), scratch + OFF_PACKED, ROWBASE_PAGE);
    noc_async_read_barrier();
    const uint32_t rowbase = *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(scratch + OFF_PACKED);
    const uint32_t f = rowbase + me;

    volatile tt_l1_ptr uint16_t* ids = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(scratch + OFF_IDS);
    volatile tt_l1_ptr uint16_t* scores = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(scratch + OFF_SCORES);
    if (f < R) {
        // hidden row f: row tile f / 32, row r = f % 32 inside it; element (r, j) of column tile c sits in face
        // (r / 16) * 2 + (j / 16) at byte (r % 16) * 32 + (j % 16) * 2: two contiguous 32-byte face rows per tile
        const uint32_t rt = f >> 5, r = f & 31;
        const uint32_t face_base = ((r >> 4) * 2) * 512 + (r & 15) * FACE_ROW_BYTES;
        const uint32_t lane = (r & 1) * FACE_ROW_BYTES;  // face_base modulo 64
        for (uint32_t c = 0; c < COLS_T; ++c) {
            const uint32_t page = rt * COLS_T + c;
            const uint32_t slot = scratch + OFF_STAGE + c * 128 + lane;
            noc_async_read(s_hidden.get_noc_addr(page, face_base), slot, FACE_ROW_BYTES);
            noc_async_read(s_hidden.get_noc_addr(page, face_base + 512), slot + 64, FACE_ROW_BYTES);
        }
        noc_async_read(s_packed.get_noc_addr(f), scratch + OFF_PACKED, PACKED_PAGE);
        noc_async_read_barrier();
        // compact the staged face rows into the contiguous row (8 words per face row)
        volatile tt_l1_ptr uint32_t* row = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(scratch);
        for (uint32_t i = 0; i < 2 * COLS_T; ++i) {
            volatile tt_l1_ptr uint32_t* src =
                reinterpret_cast<volatile tt_l1_ptr uint32_t*>(scratch + OFF_STAGE + i * 64 + lane);
            for (uint32_t w = 0; w < FACE_ROW_BYTES / 4; ++w) {
                row[i * (FACE_ROW_BYTES / 4) + w] = src[w];
            }
        }
        volatile tt_l1_ptr uint16_t* packed = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(scratch + OFF_PACKED);
        for (uint32_t s = 0; s < K; ++s) {
            scores[s] = packed[s];
            ids[s] = uint16_t(bf16_to_f32(packed[K + s]) + 0.5f);
        }
    } else {
        volatile tt_l1_ptr uint32_t* row = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(scratch);
        for (uint32_t i = 0; i < ROW_BYTES / 4; ++i) {
            row[i] = 0;
        }
        for (uint32_t s = 0; s < K; ++s) {
            scores[s] = 0;
            ids[s] = uint16_t(s);
        }
    }
    noc_async_write(scratch, s_x.get_noc_addr(me), ROW_BYTES);
    noc_async_write(scratch + OFF_IDS, s_ids.get_noc_addr(me), IDS_PAGE);
    noc_async_write(scratch + OFF_SCORES, s_scores.get_noc_addr(me), SCORES_PAGE);
    noc_async_write_barrier();
}
