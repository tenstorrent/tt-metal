// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// MoE Dispatch — Receiver Reader (streaming from DRAM)
//
// For each local expert, for each source device, for each tile-row:
//   1. Wait on tiles_ready_sem (sender wrote tile-row to dispatch_buf DRAM)
//   2. Read tile-row from dispatch_buf DRAM into cb_in
//   3. Push weight tiles to cb_w
//
// Expert-major, device-minor order matches sender go_sem ordering.
// Output row index increments sequentially: expert 0 dev 0 rows,
// expert 0 dev 1 rows, ..., expert 1 dev 0 rows, etc.

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "api/debug/dprint.h"
#include "internal/dataflow/dataflow_api_addrgen.h"

constexpr uint32_t K_t = get_compile_time_arg_val(0);
constexpr uint32_t N_t = get_compile_time_arg_val(1);
constexpr uint32_t block_size = get_compile_time_arg_val(2);
constexpr uint32_t E_local = get_compile_time_arg_val(3);
constexpr uint32_t num_devices = get_compile_time_arg_val(4);
constexpr uint32_t my_first_expert = get_compile_time_arg_val(5);
constexpr uint32_t tiles_per_batch = block_size * block_size;
constexpr uint32_t w_tiles_per_expert = K_t * N_t;

constexpr auto cb_in_idx = tt::CBIndex::c_0;
constexpr auto cb_w_idx = tt::CBIndex::c_1;

constexpr auto dispatched_ta = TensorAccessorArgs<6>();
constexpr auto w_ta = TensorAccessorArgs<dispatched_ta.next_compile_time_args_offset()>();

void kernel_main() {
    size_t ra = 0;
    uint32_t dispatched_addr = get_arg_val<uint32_t>(ra++);
    uint32_t w_addr = get_arg_val<uint32_t>(ra++);
    uint32_t tiles_ready_sem_addr = get_semaphore(get_arg_val<uint32_t>(ra++));

    // Per (device, local_expert) tile-row counts
    // Layout: expert_rows[d * E_local + e]
    uint32_t expert_rows[num_devices * E_local];
    for (uint32_t i = 0; i < num_devices * E_local; i++) {
        expert_rows[i] = get_arg_val<uint32_t>(ra++);
    }

    uint32_t tile_bytes = get_tile_size(cb_in_idx);
    const auto dispatched_acc = TensorAccessor(dispatched_ta, dispatched_addr, tile_bytes);
    const auto w_acc = TensorAccessor(w_ta, w_addr, tile_bytes);

    volatile tt_l1_ptr uint32_t* sem_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(tiles_ready_sem_addr);

    // dispatch_buf tile index — increments as we read rows sequentially
    uint32_t dispatch_tile_row = 0;
    uint32_t tiles_consumed = 0;

    for (uint32_t e = 0; e < E_local; e++) {
        uint32_t w_base = (my_first_expert + e) * w_tiles_per_expert;

        for (uint32_t d = 0; d < num_devices; d++) {
            uint32_t n_rows = expert_rows[d * E_local + e];

            for (uint32_t r = 0; r < n_rows; r++) {
                // Wait for sender to write this tile-row to dispatch_buf
                tiles_consumed++;
                noc_semaphore_wait_min(sem_ptr, tiles_consumed);

                // Read tile-row from dispatch_buf DRAM into cb_in
                uint32_t x_base = dispatch_tile_row * K_t;

                for (uint32_t p = 0; p < K_t; p += block_size) {
                    cb_reserve_back(cb_in_idx, block_size);
                    uint32_t l1 = get_write_ptr(cb_in_idx);
                    for (uint32_t i = 0; i < block_size && (p + i) < K_t; i++) {
                        noc_async_read(dispatched_acc.get_noc_addr(x_base + p + i), l1 + i * tile_bytes, tile_bytes);
                    }
                    noc_async_read_barrier();
                    cb_push_back(cb_in_idx, block_size);

                    // Weight tiles for each N output block
                    for (uint32_t n = 0; n < N_t; n += block_size) {
                        cb_reserve_back(cb_w_idx, tiles_per_batch);
                        uint32_t wl1 = get_write_ptr(cb_w_idx);
                        for (uint32_t kr = 0; kr < block_size; kr++) {
                            for (uint32_t nc = 0; nc < block_size; nc++) {
                                uint32_t tk = p + kr;
                                uint32_t tn = n + nc;
                                if (tk < K_t && tn < N_t) {
                                    noc_async_read(w_acc.get_noc_addr(w_base + tk * N_t + tn), wl1, tile_bytes);
                                }
                                wl1 += tile_bytes;
                            }
                        }
                        noc_async_read_barrier();
                        cb_push_back(cb_w_idx, tiles_per_batch);
                    }
                }

                dispatch_tile_row++;
            }
        }
    }
    DPRINT << "RECV: done, consumed=" << tiles_consumed << ENDL();
}
