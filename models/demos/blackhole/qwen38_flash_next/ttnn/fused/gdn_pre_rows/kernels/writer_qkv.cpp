// SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// Group A writer.  Per (column group, tile row) unit it takes the four finished bf16 tiles from CB_OUT, zeroes the
// rows at or past `rows` for q and k (v was masked by its own chain op in the compute kernel), and writes them:
//
//   * a v group (column group 8..19, value head g - 8) -> `v` [1, 1, T, 1536] token-major, page
//     c * 48 + 4 hv + d: the prep reader's flat-v address (c * Ct + rt) * HV * Vt + hv * Vt + ct at Ct = 1;
//   * a q or k group (key head g or g - 4) -> the three value heads 3 kh .. 3 kh + 2 of `q_c` / `k_c`
//     [12, NC, 32, 128], page (hv * NC + c) * 4 + d: the prep reader's head-major address hc * Ct * Kt + t.  The
//     GQA expand is this copy (the chain's exact 0/1 qk_expand matmul), so one unit writes 12 pages.
//
// Compile-time args: 0 chunks (NC); then TensorAccessorArgs of q_c, k_c, v.  Runtime args: 0 q_c, 1 k_c, 2 v
// addresses, 3 rows, 4 units, then (column group, tile row) pairs.

#include "api/dataflow/dataflow_api.h"
#include "api/tensor/noc_traits.h"
#include "../../kernels/zones.h"

namespace {
constexpr uint32_t CB_OUT = 14;
constexpr uint32_t HEAD_TILES = 4, QK_HEADS = 4, QK_GROUPS = 8, GQA = 3;
constexpr uint32_t VALUE_TILES = 48, TILE_ROWS = 32, FACE_ELEMS = 256, FACE_COLS = 16, BF16_TILE = 2048;

constexpr uint32_t face_element(uint32_t row, uint32_t col) {
    return ((row >> 4) * 2 + (col >> 4)) * FACE_ELEMS + (row & 15) * FACE_COLS + (col & 15);
}

// Zero rows `keep`..31 of a bf16 tile in L1: each row is 16 elements in its left face and 16 in its right one.
void zero_rows_bf16(uint32_t l1, uint32_t keep) {
    volatile tt_l1_ptr uint16_t* elements = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(l1);
    for (uint32_t row = keep; row < TILE_ROWS; ++row) {
        const uint32_t left = face_element(row, 0);
        for (uint32_t c = 0; c < FACE_COLS; ++c) {
            elements[left + c] = 0;
            elements[left + FACE_ELEMS + c] = 0;
        }
    }
}
}  // namespace

void kernel_main() {
    constexpr uint32_t CHUNKS = get_compile_time_arg_val(0);
    constexpr auto q_args = TensorAccessorArgs<1>();
    constexpr auto k_args = TensorAccessorArgs<q_args.next_compile_time_args_offset()>();
    constexpr auto v_args = TensorAccessorArgs<k_args.next_compile_time_args_offset()>();

    const uint32_t q_addr = get_arg_val<uint32_t>(0);
    const uint32_t k_addr = get_arg_val<uint32_t>(1);
    const uint32_t v_addr = get_arg_val<uint32_t>(2);
    const uint32_t rows = get_arg_val<uint32_t>(3);
    const uint32_t units = get_arg_val<uint32_t>(4);
    uint32_t arg = 5;

    const auto q_c = TensorAccessor(q_args, q_addr);
    const auto k_c = TensorAccessor(k_args, k_addr);
    const auto v = TensorAccessor(v_args, v_addr);

    const uint32_t last_chunk = (rows - 1) / TILE_ROWS;
    const uint32_t partial = rows % TILE_ROWS;

    for (uint32_t unit = 0; unit < units; ++unit) {
        const uint32_t group = get_arg_val<uint32_t>(arg++);
        const uint32_t chunk = get_arg_val<uint32_t>(arg++);

        cb_wait_front(CB_OUT, HEAD_TILES);
        const uint32_t l1 = get_read_ptr(CB_OUT);
        // q and k take their rows mask here: the chain multiplies the whole [1, T, 32, 128] tile of a masked token
        // by 0.0 (row_mask_bf16 broadcasts over the head and dim axes), and the bf16 multiply's clamp makes that
        // +0 in every element -- the same bits as zeroing the tile.  v is already masked: its own chain op, the
        // column-broadcast multiply by row_mask_bf16_col, ran in the compute kernel.
        if (group < QK_GROUPS) {
            if (chunk > last_chunk) {
                FUSED_ZONE("fz_gpr_wq_mask_all");
                for (uint32_t d = 0; d < HEAD_TILES; ++d) {
                    zero_rows_bf16(l1 + d * BF16_TILE, 0);
                }
            } else if (chunk == last_chunk && partial != 0) {
                FUSED_ZONE("fz_gpr_wq_mask_tail");
                for (uint32_t d = 0; d < HEAD_TILES; ++d) {
                    zero_rows_bf16(l1 + d * BF16_TILE, partial);
                }
            }
        }

        if (group >= QK_GROUPS) {
            FUSED_ZONE("fz_gpr_wq_v");
            const uint32_t head = group - QK_GROUPS;
            for (uint32_t d = 0; d < HEAD_TILES; ++d) {
                noc_async_write_page(chunk * VALUE_TILES + head * HEAD_TILES + d, v, l1 + d * BF16_TILE);
            }
        } else {
            FUSED_ZONE("fz_gpr_wq_qk");
            const uint32_t key_head = group < QK_HEADS ? group : group - QK_HEADS;
            for (uint32_t g = 0; g < GQA; ++g) {
                const uint32_t head = key_head * GQA + g;
                for (uint32_t d = 0; d < HEAD_TILES; ++d) {
                    const uint32_t page = (head * CHUNKS + chunk) * HEAD_TILES + d;
                    if (group < QK_HEADS) {
                        noc_async_write_page(page, q_c, l1 + d * BF16_TILE);
                    } else {
                        noc_async_write_page(page, k_c, l1 + d * BF16_TILE);
                    }
                }
            }
        }
        noc_async_write_barrier();
        cb_pop_front(CB_OUT, HEAD_TILES);
    }
}
