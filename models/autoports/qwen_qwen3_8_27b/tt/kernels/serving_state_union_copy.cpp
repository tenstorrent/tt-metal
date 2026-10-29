// SPDX-License-Identifier: Apache-2.0
// Exact byte copies between a live request state row and the attention K/V pages
// of one scheduler block (union backing). The block's 16 attention layers' K then V
// tiles are treated as one byte stream; a state tensor occupies [region_base,
// region_base + pages_per_row * page_bytes) of the stream of one attention layer.
// Every page size, tile size and region base is a multiple of 64 bytes, so each
// segment is DRAM aligned. Row/block IDs are device inputs; changing them never
// changes the program.
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/core_local_mem.h"
#include "api/debug/assert.h"
#include "api/tensor/noc_traits.h"

void kernel_main() {
    constexpr uint32_t page_bytes = get_compile_time_arg_val(0);
    constexpr uint32_t pages_per_row = get_compile_time_arg_val(1);
    constexpr uint32_t num_cores = get_compile_time_arg_val(2);
    constexpr uint32_t live_rows = get_compile_time_arg_val(3);
    constexpr uint32_t blocks = get_compile_time_arg_val(4);
    constexpr uint32_t tiles_per_block = get_compile_time_arg_val(5);
    constexpr uint32_t tile_bytes = get_compile_time_arg_val(6);
    constexpr uint32_t region_base = get_compile_time_arg_val(7);
    constexpr uint32_t restore = get_compile_time_arg_val(8);
    constexpr uint32_t half_bytes = tiles_per_block * tile_bytes;
    static_assert(region_base + pages_per_row * page_bytes <= 2 * half_bytes);
    static_assert(page_bytes % 64 == 0 && tile_bytes % 64 == 0 && region_base % 64 == 0);
    constexpr auto live_args = TensorAccessorArgs<9>();
    constexpr auto key_args = TensorAccessorArgs<live_args.next_compile_time_args_offset()>();
    constexpr auto value_args = TensorAccessorArgs<key_args.next_compile_time_args_offset()>();
    constexpr auto ids_args = TensorAccessorArgs<value_args.next_compile_time_args_offset()>();
    const uint32_t live_address = get_arg_val<uint32_t>(0);
    const uint32_t key_address = get_arg_val<uint32_t>(1);
    const uint32_t value_address = get_arg_val<uint32_t>(2);
    const uint32_t ids_address = get_arg_val<uint32_t>(3);
    const uint32_t layer_row = get_arg_val<uint32_t>(4);
    const uint32_t first_page = get_arg_val<uint32_t>(5);
    const auto live = TensorAccessor(live_args, live_address, page_bytes);
    const auto key = TensorAccessor(key_args, key_address, tile_bytes);
    const auto value = TensorAccessor(value_args, value_address, tile_bytes);
    const auto ids = TensorAccessor(ids_args, ids_address, 32);
    CircularBuffer staging(0);
    CircularBuffer ids_cb(1);
    Noc noc;
    const CoreLocalMem<volatile uint32_t> row_ids(ids_cb.get_write_ptr());
    noc.async_read(ids, ids_cb, 32, {.page_id = layer_row}, {.offset_bytes = 0});
    noc.async_read_barrier();
    // Same [source, destination] convention as serving_state_copy.cpp.
    const uint32_t live_row = restore ? row_ids[1] : row_ids[0];
    const uint32_t block = restore ? row_ids[0] : row_ids[1];
    ASSERT(live_row < live_rows && block < blocks);
    for (uint32_t page = first_page; page < pages_per_row; page += num_cores) {
        staging.reserve_back(1);
        if constexpr (restore) {
            // Gather the page's stream segments into the staging write slot.
            uint32_t local = 0;
            while (local < page_bytes) {
                const uint32_t stream = region_base + page * page_bytes + local;
                const bool in_value = stream >= half_bytes;
                const uint32_t offset = in_value ? stream - half_bytes : stream;
                const uint32_t tile = block * tiles_per_block + offset / tile_bytes;
                const uint32_t within = offset % tile_bytes;
                const uint32_t bytes = (tile_bytes - within) < (page_bytes - local) ? (tile_bytes - within)
                                                                                    : (page_bytes - local);
                if (in_value) {
                    noc.async_read(value, staging, bytes, {.page_id = tile, .offset_bytes = within},
                                   {.offset_bytes = local});
                } else {
                    noc.async_read(key, staging, bytes, {.page_id = tile, .offset_bytes = within},
                                   {.offset_bytes = local});
                }
                local += bytes;
            }
        } else {
            noc.async_read(live, staging, page_bytes,
                           {.page_id = live_row * pages_per_row + page}, {.offset_bytes = 0});
        }
        noc.async_read_barrier();
        staging.push_back(1);
        staging.wait_front(1);
        auto staged = use<CircularBuffer::AddrSelector::READ_PTR>(staging);
        if constexpr (restore) {
            noc.async_write(staged, live, page_bytes, {.offset_bytes = 0},
                            {.page_id = live_row * pages_per_row + page});
        } else {
            uint32_t local = 0;
            while (local < page_bytes) {
                const uint32_t stream = region_base + page * page_bytes + local;
                const bool in_value = stream >= half_bytes;
                const uint32_t offset = in_value ? stream - half_bytes : stream;
                const uint32_t tile = block * tiles_per_block + offset / tile_bytes;
                const uint32_t within = offset % tile_bytes;
                const uint32_t bytes = (tile_bytes - within) < (page_bytes - local) ? (tile_bytes - within)
                                                                                    : (page_bytes - local);
                if (in_value) {
                    noc.async_write(staged, value, bytes, {.offset_bytes = local},
                                    {.page_id = tile, .offset_bytes = within});
                } else {
                    noc.async_write(staged, key, bytes, {.offset_bytes = local},
                                    {.page_id = tile, .offset_bytes = within});
                }
                local += bytes;
            }
        }
        noc.async_write_barrier();
        staging.pop_front(1);
    }
}
