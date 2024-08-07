// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "debug/dprint.h"  // required in all kernels using DPRINT
#include "dataflow_api.h"

void kernel_main() {

        ////////// RUNTIME ARGS & VARS //////////
        uint32_t start_x = get_arg_val<uint32_t>(0);
        uint32_t start_y = get_arg_val<uint32_t>(1);
        uint32_t end_x = get_arg_val<uint32_t>(2);
        uint32_t end_y = get_arg_val<uint32_t>(3);
        uint32_t sender_addr = get_arg_val<uint32_t>(4);
        uint32_t receiver_addr = get_arg_val<uint32_t>(5);
        uint32_t src0_dram = get_arg_val<uint32_t>(6);
        uint32_t single_tile_size = get_arg_val<uint32_t>(7);
        uint32_t kernel_start_ac_target_val = 1;
        uint32_t num_dests = 3;

        ////////// BUFFER SETUP //////////
        uint64_t src0_dram_noc_addr = get_noc_addr(1, 0, src0_dram);
        constexpr uint32_t cb_id_in0 = tt::CB::c_in0; // index=0
        constexpr uint32_t cb_id_out0 = tt::CB::c_out0; // index=16
        uint32_t ublock_size_bytes = get_tile_size(cb_id_in0);
        uint32_t tile_l1_addr = get_write_ptr(cb_id_in0);

        ////////// READ TILE DRAM->L1 //////////
        noc_async_read(src0_dram_noc_addr, tile_l1_addr, ublock_size_bytes);
        noc_async_read_barrier();

        ////////// PRINT TILE SLICE //////////
        auto sr = SliceRange{.h0 = 0, .h1 = 32, .hs = 8, .w0 = 0, .w1 = 32, .ws = 8};
        DPRINT << TileSlice(cb_id_in0, 0, sr, true, false) << ENDL();

        cb_push_back(cb_id_in0, 1);

        ////////// SEMAPHORE SETUP //////////
        volatile tt_l1_ptr uint32_t* sender_addr_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(sender_addr);
        volatile tt_l1_ptr uint32_t* receiver_addr_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(receiver_addr);
        *(receiver_addr_ptr) = VALID; //

        ////////// WAIT FOR WORKER CORES' READINESS //////////
        noc_semaphore_wait(sender_addr_ptr, num_dests);

        ////////// RESET SEMAPHORE //////////
        noc_semaphore_set(sender_addr_ptr, 0); // 0 rather than INVALID because sender_add_ptr can take val >1

        ////////// MULTICAST TILE TO WORKERS //////////
        uint64_t identity_tile_global_multicast_addr =
                get_noc_multicast_addr(start_x, start_y, end_x, end_y, tile_l1_addr);
        noc_async_write_multicast(tile_l1_addr, identity_tile_global_multicast_addr, single_tile_size, num_dests);

        ////////// MULTICAST AC TARGET VAL TO WORKERS //////////
        uint64_t ac_global_multicast_addr =
                get_noc_multicast_addr(start_x, start_y, end_x, end_y, receiver_addr);
        noc_semaphore_set_multicast(receiver_addr, ac_global_multicast_addr, num_dests);

        noc_async_write_barrier();

        // coordinator kernel can safely do more work here

        DPRINT << "CORE " << (uint32_t) my_x[0] << "," << (uint32_t) my_y[0] << ": I am starting all inbound kernels in cores in given range" << ENDL() << ENDL();
}
