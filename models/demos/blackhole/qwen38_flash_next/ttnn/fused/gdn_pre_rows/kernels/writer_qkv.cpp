// SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// Group A writer. Per (column group, tile row) unit: takes cb_out's four bf16 tiles, zeroes rows at or past `rows`
// for q and k (v was masked by its chain op in the compute kernel), and writes them:
//   * a v group (8..19, value head g - 8) -> `v` [1, 1, T, 1536] token-major, page c * 48 + 4 hv + d;
//   * a q or k group (key head g or g - 4) -> value heads 3 kh .. 3 kh + 2 of `q_c` / `k_c` [12, NC, 32, 128], page
//     (hv * NC + c) * 4 + d. This copy is the GQA expand, so one unit writes 12 pages.
//
// Compile-time args: 0 chunks (NC); then TensorAccessorArgs of q_c, k_c, v.  Runtime args: 0 q_c, 1 k_c, 2 v
// addresses, 3 rows, 4 units, then (column group, tile row) pairs.

#include "api/dataflow/dataflow_api.h"
#include "api/tensor/noc_traits.h"

namespace {
constexpr uint32_t cb_out = 14;
constexpr uint32_t head_tiles = 4, qk_heads = 4, qk_groups = 8, value_heads_per_key_head = 3;
constexpr uint32_t value_tiles = 48, tile_rows = 32, face_elements = 256, face_columns = 16, bf16_tile_bytes = 2048;

constexpr uint32_t face_element(uint32_t row, uint32_t col) {
    return ((row >> 4) * 2 + (col >> 4)) * face_elements + (row & 15) * face_columns + (col & 15);
}

// Zero rows `keep`..31 of a bf16 tile in L1: each row is 16 elements in its left face and 16 in its right one.
void zero_rows_bf16(uint32_t l1, uint32_t keep) {
    volatile tt_l1_ptr uint16_t* elements = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(l1);
    for (uint32_t row = keep; row < tile_rows; ++row) {
        const uint32_t left = face_element(row, 0);
        for (uint32_t c = 0; c < face_columns; ++c) {
            elements[left + c] = 0;
            elements[left + face_elements + c] = 0;
        }
    }
}
}  // namespace

void kernel_main() {
    constexpr uint32_t chunks = get_compile_time_arg_val(0);
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

    const uint32_t last_chunk = (rows - 1) / tile_rows;
    const uint32_t partial = rows % tile_rows;

    for (uint32_t unit = 0; unit < units; ++unit) {
        const uint32_t group = get_arg_val<uint32_t>(arg++);
        const uint32_t chunk = get_arg_val<uint32_t>(arg++);

        cb_wait_front(cb_out, head_tiles);
        const uint32_t l1 = get_read_ptr(cb_out);
        // q and k take their rows mask here: the chain's whole-tile multiply by 0.0 gives +0 in every element under
        // the bf16 clamp, the same bits as zeroing. v was masked in the compute kernel.
        if (group < qk_groups) {
            if (chunk > last_chunk) {
                for (uint32_t d = 0; d < head_tiles; ++d) {
                    zero_rows_bf16(l1 + d * bf16_tile_bytes, 0);
                }
            } else if (chunk == last_chunk && partial != 0) {
                for (uint32_t d = 0; d < head_tiles; ++d) {
                    zero_rows_bf16(l1 + d * bf16_tile_bytes, partial);
                }
            }
        }

        if (group >= qk_groups) {
            const uint32_t head = group - qk_groups;
            for (uint32_t d = 0; d < head_tiles; ++d) {
                noc_async_write_page(chunk * value_tiles + head * head_tiles + d, v, l1 + d * bf16_tile_bytes);
            }
        } else {
            const uint32_t key_head = group < qk_heads ? group : group - qk_heads;
            for (uint32_t g = 0; g < value_heads_per_key_head; ++g) {
                const uint32_t head = key_head * value_heads_per_key_head + g;
                for (uint32_t d = 0; d < head_tiles; ++d) {
                    const uint32_t page = (head * chunks + chunk) * head_tiles + d;
                    if (group < qk_heads) {
                        noc_async_write_page(page, q_c, l1 + d * bf16_tile_bytes);
                    } else {
                        noc_async_write_page(page, k_c, l1 + d * bf16_tile_bytes);
                    }
                }
            }
        }
        noc_async_write_barrier();
        cb_pop_front(cb_out, head_tiles);
    }
}
