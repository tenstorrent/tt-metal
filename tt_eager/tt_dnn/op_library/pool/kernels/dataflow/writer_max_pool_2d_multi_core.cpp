// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "dataflow_api.h"

// #include "debug/dprint.h"
// SliceRange srt = SliceRange{ .h0 = 0, .h1 = 32, .hs = 8, .w0 = 0, .w1 = 32, .ws = 8 };
// SliceRange srr = SliceRange{ .h0 = 0, .h1 = 1, .hs = 8, .w0 = 0, .w1 = 32, .ws = 1 };
// SliceRange srr2 = SliceRange{ .h0 = 0, .h1 = 1, .hs = 8, .w0 = 0, .w1 = 64, .ws = 2 };

/**
 * Max-pool 2D.
 */
void kernel_main() {
    const uint32_t out_addr = get_arg_val<uint32_t>(1);
    const int32_t out_h = get_arg_val<int32_t>(10);
    const int32_t out_w = get_arg_val<int32_t>(11);
    const uint32_t out_nbytes_c = get_arg_val<uint32_t>(15);
    const uint32_t out_ntiles_c = get_arg_val<uint32_t>(21);
    const uint32_t out_cb_pagesize = get_arg_val<uint32_t>(23);
    const uint32_t out_w_loop_count = get_arg_val<uint32_t>(25);
    const uint32_t nbatch = get_arg_val<uint32_t>(27);
    const uint32_t out_hw = get_arg_val<uint32_t>(29);
    const uint32_t start_out_h_i = get_arg_val<uint32_t>(30);
    const uint32_t end_out_h_i = get_arg_val<uint32_t>(31);
    const uint32_t start_out_row_id = get_arg_val<uint32_t>(33);

    constexpr bool is_out_dram = get_compile_time_arg_val(1) == 1;
    constexpr uint32_t out_nelems = get_compile_time_arg_val(3);

    constexpr uint32_t out_cb_id = tt::CB::c_out0;

    // ROW_MAJOR output
    const InterleavedAddrGen<is_out_dram> s_out = {
        .bank_base_address = out_addr,
        .page_size = out_nbytes_c   // TODO: Ensure this is 32B aligned
    };

    uint32_t nsticks_per_core = get_arg_val<uint32_t>(40);
    uint32_t core_offset_out_row_id = get_arg_val<uint32_t>(41);
    uint32_t nsticks_per_core_by_nblocks = get_arg_val<uint32_t>(42);

    uint32_t out_row_id = 0;
    uint64_t out_noc_addr;
    #ifdef SHARDED_OUT
    constexpr uint32_t sharded_out_cb_id = tt::CB::c_out1;
    out_noc_addr = get_noc_addr(get_read_ptr(sharded_out_cb_id));
    cb_reserve_back(sharded_out_cb_id, out_nelems * nsticks_per_core_by_nblocks);
    #endif

    for (uint32_t stick = 0; stick < nsticks_per_core_by_nblocks; ++ stick) {
        cb_wait_front(out_cb_id, 1);
        uint32_t out_l1_read_addr = get_read_ptr(out_cb_id);
        for (uint32_t out_elem_i = 0; out_elem_i < out_nelems; ++ out_elem_i) {
            // Write as tiled tensor, need to handle each face.
            // NOTE: this assumes that the stick size is 64 (2 tiles width)

            #ifndef SHARDED_OUT
            out_noc_addr = get_noc_addr(core_offset_out_row_id + out_row_id, s_out);
            #endif

            noc_async_write(out_l1_read_addr, out_noc_addr, 128);
            out_noc_addr += 128;

            // // tile 0
            // // face 0
            // // write 16 elements from face0 // 32B
            // noc_async_write(out_l1_read_addr, out_noc_addr, 32);
            // out_noc_addr += 32;

            // // face 1
            // out_l1_read_addr += 512; // go to face 1
            // noc_async_write(out_l1_read_addr, out_noc_addr, 32);
            // out_noc_addr += 32;

            // // go to tile 1
            // out_l1_read_addr += 512 + 1024;

            // // face 0
            // // write 16 elements from face0 // 32B
            // noc_async_write(out_l1_read_addr, out_noc_addr, 32);
            // out_noc_addr += 32;

            // // face 1
            // out_l1_read_addr += 512; // go to face 1
            // noc_async_write(out_l1_read_addr, out_noc_addr, 32);
            // out_noc_addr += 32;

            // // go to next tile in the block (next channel)
            // out_l1_read_addr += 512 + 1024;

            ++ out_row_id;
        }
        noc_async_write_barrier();
        cb_pop_front(out_cb_id, 1);
    }
    #ifdef SHARDED_OUT
    cb_push_back(sharded_out_cb_id, out_nelems * nsticks_per_core_by_nblocks);
    cb_wait_front(sharded_out_cb_id, out_nelems * nsticks_per_core_by_nblocks);
    #endif
} // kernel_main()
