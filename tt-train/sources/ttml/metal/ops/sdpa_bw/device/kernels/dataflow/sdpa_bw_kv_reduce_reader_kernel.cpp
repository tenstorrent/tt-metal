// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// KV-backward head reduce, reader: for output tile p (grad_K tiles first, then grad_V) stages the G fp32
// per-head partial tiles ((p / row_tiles) * G + h) * row_tiles + p % row_tiles, h < G, as one block.

#include "api/dataflow/dataflow_api.h"

constexpr uint32_t cb_partials = tt::CBIndex::c_0;

constexpr uint32_t G = get_compile_time_arg_val(0);
constexpr uint32_t k_tiles = get_compile_time_arg_val(1);
constexpr uint32_t k_row_tiles = get_compile_time_arg_val(2);
constexpr uint32_t v_row_tiles = get_compile_time_arg_val(3);
constexpr auto gk_args = TensorAccessorArgs<4>();
constexpr auto gv_args = TensorAccessorArgs<gk_args.next_compile_time_args_offset()>();

template <typename Acc>
FORCE_INLINE void read_partials(const Acc& acc, uint32_t p, uint32_t row_tiles, uint32_t tile_bytes) {
    const uint32_t group_row = p / row_tiles;
    const uint32_t within = p % row_tiles;
    cb_reserve_back(cb_partials, G);
    uint32_t l1_addr = get_write_ptr(cb_partials);
    for (uint32_t h = 0; h < G; ++h) {
        noc_async_read_page((group_row * G + h) * row_tiles + within, acc, l1_addr);
        l1_addr += tile_bytes;
    }
    noc_async_read_barrier();
    cb_push_back(cb_partials, G);
}

void kernel_main() {
    uint32_t arg = 0;
    const uint32_t gk_addr = get_arg_val<uint32_t>(arg++);
    const uint32_t gv_addr = get_arg_val<uint32_t>(arg++);
    const uint32_t start = get_arg_val<uint32_t>(arg++);
    const uint32_t count = get_arg_val<uint32_t>(arg++);

    const uint32_t tile_bytes = get_tile_size(cb_partials);
    const auto gk_acc = TensorAccessor(gk_args, gk_addr, tile_bytes);
    const auto gv_acc = TensorAccessor(gv_args, gv_addr, tile_bytes);

    for (uint32_t i = 0; i < count; ++i) {
        const uint32_t p = start + i;
        if (p < k_tiles) {
            read_partials(gk_acc, p, k_row_tiles, tile_bytes);
        } else {
            read_partials(gv_acc, p - k_tiles, v_row_tiles, tile_bytes);
        }
    }
}
