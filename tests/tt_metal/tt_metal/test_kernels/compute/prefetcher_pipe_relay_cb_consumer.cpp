// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// TRISC consumer of a classic circular buffer laid over a PrefetcherPipe ring.
//
// One CB page is one delivered entry, so this consumer addresses the tiles inside a page the way
// the 1D matmul's compute kernel does: take the page's read pointer once, then step by the tile
// size. Firmware resets the CB pointers every launch while the pipe cursor is durable, so the
// unpacker re-aligns the CB to the pipe's checkpoint before its first wait.
//
// It records the first word of every tile it reads so the host can check that compute saw each
// delivered tile exactly once, in order.
//
// Compile-time args:
//   [0] cb_id
//   [1] prefetcher_pipe_id
//   [2] total_pages
//   [3] tiles_per_page
//   [4] tile_bytes
//
// Runtime args:
//   [0] result_l1_addr: [pages_consumed, first word of each tile ...]

#include "api/compute/compute_kernel_api.h"
#include "api/compute/common.h"
#include "api/dataflow/dataflow_buffer.h"

void kernel_main() {
#ifdef UCK_CHLKC_UNPACK
    constexpr uint32_t cb_id = get_compile_time_arg_val(0);
    constexpr uint32_t prefetcher_pipe_id = get_compile_time_arg_val(1);
    constexpr uint32_t total_pages = get_compile_time_arg_val(2);
    constexpr uint32_t tiles_per_page = get_compile_time_arg_val(3);
    constexpr uint32_t tile_bytes = get_compile_time_arg_val(4);

    experimental::align_local_dfb_to_prefetcher_pipe_slot(cb_id, prefetcher_pipe_id);
    DataflowBuffer relay_cb(cb_id);
    volatile tt_l1_ptr uint32_t* result = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_arg_val<uint32_t>(0));

    uint32_t pages_consumed = 0;
    for (uint32_t page = 0; page < total_pages; ++page) {
        relay_cb.wait_front(1);
        const uint32_t page_base = relay_cb.get_read_ptr() << cb_addr_shift;
        for (uint32_t tile = 0; tile < tiles_per_page; ++tile) {
            result[1 + page * tiles_per_page + tile] =
                *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(page_base + tile * tile_bytes);
        }
        relay_cb.pop_front(1);
        ++pages_consumed;
    }
    result[0] = pages_consumed;
#endif
}
