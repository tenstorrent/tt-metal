// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Worker writer: drains cb_out tiled pages to grouped DRAM.
// - Skips DRAM writes for tile_rows >= offsets[E_local]/32 (grouped is pre-zeroed).
// - In the last chunk per tile-row, writes only last_chunk_tiles tiles
//   (the remaining are zero-pad from the reader).
// Always pops cb_out to keep the pipeline flowing.
//
// Compile-time args:
//   0: num_chunks
//   1: tiles_per_chunk
//   2: Wt
//   3: e_local
//   4: last_chunk_tiles  (tiles in final chunk; <= tiles_per_chunk)
//   5+: TensorAccessorArgs for grouped
//   next+: TensorAccessorArgs for offsets
//
// Runtime args:
//   0: grouped_addr
//   1: my_start     — first tile_row this core processes
//   2: stride       — tile_row step = num_workers (interleaved assignment)
//   3: my_count     — number of tile_rows
//   4: offsets_addr
//   5: plan_ready_sem_id — wait on this semaphore before reading offsets

#include "api/dataflow/dataflow_api.h"

constexpr uint32_t cb_out = tt::CBIndex::c_2;
constexpr uint32_t cb_offset = tt::CBIndex::c_5;
constexpr uint32_t num_chunks = get_compile_time_arg_val(0);
constexpr uint32_t tiles_per_chunk = get_compile_time_arg_val(1);
constexpr uint32_t Wt = get_compile_time_arg_val(2);
constexpr uint32_t e_local = get_compile_time_arg_val(3);
constexpr uint32_t last_chunk_tiles = get_compile_time_arg_val(4);
constexpr auto grouped_args = TensorAccessorArgs<5>();
constexpr auto offsets_args = TensorAccessorArgs<grouped_args.next_compile_time_args_offset()>();

constexpr uint32_t TILE_H = 32U;

void kernel_main() {
    const uint32_t grouped_addr = get_arg_val<uint32_t>(0);
    const uint32_t my_start = get_arg_val<uint32_t>(1);
    const uint32_t stride = get_arg_val<uint32_t>(2);
    const uint32_t my_count = get_arg_val<uint32_t>(3);
    const uint32_t offsets_addr = get_arg_val<uint32_t>(4);
    const uint32_t plan_ready_sem_id = get_arg_val<uint32_t>(5);

    // Wait for scan to finish writing plan/counts/offsets.
    volatile tt_l1_ptr uint32_t* plan_ready_sem =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(plan_ready_sem_id));
    noc_semaphore_wait(plan_ready_sem, 1U);

    const uint32_t tile_bytes = get_tile_size(cb_out);
    const auto grouped_addrgen = TensorAccessor(grouped_args, grouped_addr, tile_bytes);
    const auto offsets_addrgen = TensorAccessor(offsets_args, offsets_addr);

    cb_reserve_back(cb_offset, 1U);
    uint32_t scratch_addr = get_write_ptr(cb_offset);
    volatile tt_l1_ptr uint32_t* scratch_buf = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(scratch_addr);
    noc_async_read(get_noc_addr(0, offsets_addrgen), scratch_addr, (e_local + 1U) * sizeof(uint32_t));
    noc_async_read_barrier();
    uint32_t max_active_tiles = scratch_buf[e_local] / TILE_H;

    uint32_t tile_row = my_start;
    for (uint32_t step = 0; step < my_count; ++step, tile_row += stride) {
        bool do_write = tile_row < max_active_tiles;
        for (uint32_t chunk = 0; chunk < num_chunks; ++chunk) {
            cb_wait_front(cb_out, tiles_per_chunk);
            if (do_write) {
                bool is_last_chunk = (chunk == num_chunks - 1U);
                uint32_t tiles_to_write = is_last_chunk ? last_chunk_tiles : tiles_per_chunk;
                uint32_t src = get_read_ptr(cb_out);
                for (uint32_t t = 0; t < tiles_to_write; ++t) {
                    uint32_t c = chunk * tiles_per_chunk + t;
                    uint64_t dst_noc = get_noc_addr(tile_row * Wt + c, grouped_addrgen);
                    noc_async_write(src + t * tile_bytes, dst_noc, tile_bytes);
                }
                noc_async_write_barrier();
            }
            cb_pop_front(cb_out, tiles_per_chunk);
        }
    }
}
