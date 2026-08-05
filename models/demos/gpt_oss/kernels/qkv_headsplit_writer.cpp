// SPDX-License-Identifier: Apache-2.0
//
// Writer half of the multi-core gpt-oss decode QKV head split. See
// qkv_headsplit_reader.cpp for layout and work decomposition.
//
// The reader has already packed a full destination tile-row (up to 32 head rows,
// head_tiles tiles wide) into the matching CB. This writer just ships those whole
// tiles to the right pages of q / k / v, one batched barrier per work unit.

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/tensor/noc_traits.h"

constexpr uint32_t cb_q_id = get_compile_time_arg_val(0);
constexpr uint32_t cb_k_id = get_compile_time_arg_val(1);
constexpr uint32_t cb_v_id = get_compile_time_arg_val(2);
constexpr uint32_t num_q_heads = get_compile_time_arg_val(3);
constexpr uint32_t num_kv_heads = get_compile_time_arg_val(4);
constexpr uint32_t head_tiles = get_compile_time_arg_val(5);
constexpr uint32_t ct_q = 6;
constexpr uint32_t ct_k = TensorAccessorArgs<ct_q>::next_compile_time_args_offset();
constexpr uint32_t ct_v = TensorAccessorArgs<ct_k>::next_compile_time_args_offset();

constexpr uint32_t TILE_HEIGHT = 32;

void kernel_main() {
    const uint32_t q_addr = get_arg_val<uint32_t>(0);
    const uint32_t k_addr = get_arg_val<uint32_t>(1);
    const uint32_t v_addr = get_arg_val<uint32_t>(2);
    const uint32_t unit_start = get_arg_val<uint32_t>(3);
    const uint32_t unit_count = get_arg_val<uint32_t>(4);

    constexpr auto q_args = TensorAccessorArgs<ct_q>();
    constexpr auto k_args = TensorAccessorArgs<ct_k>();
    constexpr auto v_args = TensorAccessorArgs<ct_v>();
    const auto qt = TensorAccessor(q_args, q_addr);
    const auto kt = TensorAccessor(k_args, k_addr);
    const auto vt = TensorAccessor(v_args, v_addr);

    Noc noc;

    constexpr uint32_t q_tile_rows = (num_q_heads + TILE_HEIGHT - 1) / TILE_HEIGHT;
    constexpr uint32_t k_tile_rows = (num_kv_heads + TILE_HEIGHT - 1) / TILE_HEIGHT;

    for (uint32_t u = 0; u < unit_count; ++u) {
        const uint32_t unit = unit_start + u;

        uint32_t cb_id;
        uint32_t tile_row;
        uint32_t which;  // 0=q 1=k 2=v
        if (unit < q_tile_rows) {
            cb_id = cb_q_id;
            tile_row = unit;
            which = 0;
        } else if (unit < q_tile_rows + k_tile_rows) {
            cb_id = cb_k_id;
            tile_row = unit - q_tile_rows;
            which = 1;
        } else {
            cb_id = cb_v_id;
            tile_row = unit - q_tile_rows - k_tile_rows;
            which = 2;
        }

        CircularBuffer cb(cb_id);
        const uint32_t page_bytes = get_local_cb_interface(cb_id).fifo_page_size;

        cb.wait_front(head_tiles);
        const uint32_t base_page = tile_row * head_tiles;
        for (uint32_t t = 0; t < head_tiles; ++t) {
            const uint32_t off = t * page_bytes;
            if (which == 0) {
                noc.async_write(cb, qt, page_bytes, {.offset_bytes = off}, {.page_id = base_page + t});
            } else if (which == 1) {
                noc.async_write(cb, kt, page_bytes, {.offset_bytes = off}, {.page_id = base_page + t});
            } else {
                noc.async_write(cb, vt, page_bytes, {.offset_bytes = off}, {.page_id = base_page + t});
            }
        }
        noc.async_write_barrier();
        cb.pop_front(head_tiles);
    }
}
