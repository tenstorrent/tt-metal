// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "api/debug/dprint.h"
#include "internal/dataflow/dataflow_api_addrgen.h"

constexpr uint32_t N_t = get_compile_time_arg_val(0);
constexpr uint32_t block_size = get_compile_time_arg_val(1);
constexpr uint32_t num_experts_local = get_compile_time_arg_val(2);
constexpr auto out_ta = TensorAccessorArgs<3>();

constexpr auto cb_out_idx = tt::CBIndex::c_10;

void kernel_main() {
    size_t ra = 0;
    const uint32_t out_addr = get_arg_val<uint32_t>(ra++);

    uint32_t expert_n_rows[num_experts_local];
    uint32_t expert_start_row[num_experts_local];
    for (uint32_t e = 0; e < num_experts_local; e++) expert_n_rows[e] = get_arg_val<uint32_t>(ra++);
    for (uint32_t e = 0; e < num_experts_local; e++) expert_start_row[e] = get_arg_val<uint32_t>(ra++);

    const uint32_t tile_bytes = get_tile_size(cb_out_idx);
    const auto out_acc = TensorAccessor(out_ta, out_addr, tile_bytes);

    for (uint32_t e = 0; e < num_experts_local; e++) {
        const uint32_t n_rows = expert_n_rows[e];
        if (n_rows == 0)
            continue;

        const uint32_t y_base = expert_start_row[e] * N_t;

        for (uint32_t r = 0; r < n_rows; r++) {
            for (uint32_t c = 0; c < N_t; c += block_size) {
                cb_wait_front(cb_out_idx, block_size);
                uint32_t l1 = get_read_ptr(cb_out_idx);
                uint32_t tile_start = y_base + r * N_t + c;
                for (uint32_t i = 0; i < block_size && (c + i) < N_t; i++) {
                    noc_async_write(l1, out_acc.get_noc_addr(tile_start + i), tile_bytes);
                    l1 += tile_bytes;
                }
                noc_async_write_barrier();
                cb_pop_front(cb_out_idx, block_size);
            }
        }
    }
    DPRINT << "WRITER: done" << ENDL();
}
