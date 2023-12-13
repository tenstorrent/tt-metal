// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "dataflow_api.h"
#include "debug/dprint.h"

void kernel_main() {
    uint32_t i = 0;
    uint32_t total_size_bytes = get_arg_val<uint32_t>(i); i+=1;
    uint32_t num_chunks = get_arg_val<uint32_t>(i); i+=1;
    uint32_t chunk_size_bytes = get_arg_val<uint32_t>(i); i+=1;
    uint32_t remainder_chunk_size_bytes = get_arg_val<uint32_t>(i); i+=1;
    //DPRINT << "total_size_bytes=" << total_size_bytes << ENDL();
    //DPRINT << "num_chunks=" << num_chunks << ENDL();
    //DPRINT << "chunk_size_bytes=" << chunk_size_bytes << ENDL();
    //DPRINT << "remainder_chunk_size_bytes=" << remainder_chunk_size_bytes << ENDL();
    constexpr uint32_t src_cb_id = get_compile_time_arg_val(0);
    constexpr uint32_t dst_cb_id = get_compile_time_arg_val(1);
    //DPRINT << "src_cb_id=" << src_cb_id << ENDL();
    //DPRINT << "dst_cb_id=" << dst_cb_id << ENDL();
    uint32_t src_cb_base_addr = get_write_ptr(src_cb_id); // TODO change to read
    uint32_t dst_cb_base_addr = get_write_ptr(dst_cb_id);

    // Copy from top of src cb to top of dst cb (backwards)
    uint32_t src_cb_addr = src_cb_base_addr + total_size_bytes;
    uint32_t dst_cb_addr = dst_cb_base_addr + total_size_bytes;
    //DPRINT << "src_cb_addr=" << src_cb_addr << ENDL();
    //DPRINT << "dst_cb_addr=" << dst_cb_addr << ENDL();
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
