// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "dataflow_api.h"

// #include "debug/dprint.h"
// SliceRange srt = SliceRange{ .h0 = 0, .h1 = 32, .hs = 8, .w0 = 0, .w1 = 32, .ws = 8 };
// SliceRange srr = SliceRange{ .h0 = 0, .h1 = 1, .hs = 8, .w0 = 0, .w1 = 32, .ws = 1 };
// SliceRange srr2 = SliceRange{ .h0 = 0, .h1 = 1, .hs = 8, .w0 = 0, .w1 = 64, .ws = 2 };

// inline void print_pages(uint32_t l1_addr, uint32_t pagelen, uint32_t npages, uint32_t start = 0) {
//     volatile tt_l1_ptr uint16_t* ptr = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(l1_addr) + start * pagelen;
//     for (uint32_t page = 0; page < npages; ++ page) {
//         DPRINT << start + page << ": ";
//         for (uint32_t j = 0; j < pagelen; ++ j, ++ ptr) {
//             DPRINT << BF16(*ptr) << " ";
//         }
//         DPRINT << ENDL();
//     }
// }

/**
 * Max-pool 2D.
 */
void kernel_main() {
    const uint32_t out_addr = get_arg_val<uint32_t>(0);
    const uint32_t out_nbytes_c = get_arg_val<uint32_t>(1);
    const uint32_t out_ntiles_c = get_arg_val<uint32_t>(2);

    constexpr bool is_out_dram = get_compile_time_arg_val(1) == 1;
    constexpr uint32_t out_nelems = get_compile_time_arg_val(3);

    constexpr uint32_t out_cb_id = tt::CB::c_out0;

    uint32_t nsticks_per_core = get_arg_val<uint32_t>(3);
    uint32_t core_offset_out_row_id = get_arg_val<uint32_t>(4);
    uint32_t nsticks_per_core_by_nblocks = get_arg_val<uint32_t>(5);

    uint64_t out_noc_addr;
    constexpr uint32_t sharded_out_cb_id = tt::CB::c_out1;
    out_noc_addr = get_noc_addr(get_read_ptr(sharded_out_cb_id));
    cb_reserve_back(sharded_out_cb_id, out_nelems * nsticks_per_core_by_nblocks);

    // DPRINT << "W nsticks_per_core_by_nblocks = " << nsticks_per_core_by_nblocks << ENDL();

    for (uint32_t stick = 0; stick < nsticks_per_core_by_nblocks; ++ stick) {
        // DPRINT << "W ait: " << stick << ENDL();

        cb_wait_front(out_cb_id, out_nelems * out_ntiles_c);
        uint32_t out_l1_read_addr = get_read_ptr(out_cb_id);

        // print_pages(out_l1_read_addr, 32, 2);

        noc_async_write(out_l1_read_addr, out_noc_addr, 128);
        out_noc_addr += 128;

        // for (uint32_t out_elem_i = 0; out_elem_i < out_nelems; ++ out_elem_i) {
        //     // Write as tiled tensor, need to handle each face.
        //     // NOTE: this assumes that the stick size is 64 (2 tiles width)

        //     // tile 0
        //     // face 0
        //     // write 16 elements from face0 // 32B
        //     noc_async_write(out_l1_read_addr, out_noc_addr, 32);
        //     out_noc_addr += 32;

        //     // face 1
        //     out_l1_read_addr += 512; // go to face 1
        //     noc_async_write(out_l1_read_addr, out_noc_addr, 32);
        //     out_noc_addr += 32;

        //     // go to tile 1
        //     out_l1_read_addr += 512 + 1024;

        //     // face 0
        //     // write 16 elements from face0 // 32B
        //     noc_async_write(out_l1_read_addr, out_noc_addr, 32);
        //     out_noc_addr += 32;

        //     // face 1
        //     out_l1_read_addr += 512; // go to face 1
        //     noc_async_write(out_l1_read_addr, out_noc_addr, 32);
        //     out_noc_addr += 32;

        //     // go to next tile in the block (next channel)
        //     out_l1_read_addr += 512 + 1024;
        // }
        noc_async_write_barrier();

        // kernel_profiler::mark_time(14);
        cb_pop_front(out_cb_id, out_nelems * out_ntiles_c);

        // DPRINT << "W ritten: " << stick << ENDL();
    }
    cb_push_back(sharded_out_cb_id, out_nelems * nsticks_per_core_by_nblocks);
    cb_wait_front(sharded_out_cb_id, out_nelems * nsticks_per_core_by_nblocks);

    // DPRINT << "WRITER DONE!!!" << ENDL();
} // kernel_main()
