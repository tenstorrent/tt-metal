// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "dataflow_api.h"

void kernel_main() {
    uint32_t i = 0;
    uint32_t total_size_bytes = get_arg_val<uint32_t>(i); i+=1;
    uint32_t num_chunks = get_arg_val<uint32_t>(i); i+=1;
    uint32_t chunk_size_bytes = get_arg_val<uint32_t>(i); i+=1;
    uint32_t remainder_chunk_size_bytes = get_arg_val<uint32_t>(i); i+=1;
    constexpr uint32_t src_cb_id = get_compile_time_arg_val(0);
    constexpr uint32_t dst_cb_id = get_compile_time_arg_val(1);

    uint32_t src_cb_base_addr = get_write_ptr(src_cb_id); // TODO change to read
    uint32_t dst_cb_base_addr = get_write_ptr(dst_cb_id);

    // Copy from top of src cb to top of dst cb (backwards)
    uint32_t src_cb_addr = src_cb_base_addr + total_size_bytes;
    uint32_t dst_cb_addr = dst_cb_base_addr + total_size_bytes;
    for (uint32_t i = 0; i<num_chunks; i += 1) {
        src_cb_addr -= chunk_size_bytes;
        dst_cb_addr -= chunk_size_bytes;
        noc_async_read(get_noc_addr(src_cb_addr), dst_cb_addr, chunk_size_bytes);
        noc_async_read_barrier();
    }
    if(remainder_chunk_size_bytes > 0) {
        src_cb_addr -= remainder_chunk_size_bytes;
        dst_cb_addr -= remainder_chunk_size_bytes;
        noc_async_read(get_noc_addr(src_cb_addr), dst_cb_addr, remainder_chunk_size_bytes);
        noc_async_read_barrier();
    }
}
