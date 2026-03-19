#include <cstdint>
#include "api/debug/dprint.h"
#include "api/dataflow/dataflow_api.h"
#include "hostdevcommon/common_values.hpp"

void kernel_main() {
    uint32_t dbg_tile = 999;

    uint32_t in_a_addr = get_arg_val<uint32_t>(0);
    uint32_t in_c_addr = get_arg_val<uint32_t>(1);
    uint32_t in0_mcast_sender_semaphore_addr = get_semaphore(get_arg_val<uint32_t>(2));
    uint32_t in0_mcast_receiver_semaphore_addr = get_semaphore(get_arg_val<uint32_t>(3));
    uint32_t in1_mcast_sender_semaphore_addr = get_semaphore(get_arg_val<uint32_t>(4));
    uint32_t in1_mcast_receiver_semaphore_addr = get_semaphore(get_arg_val<uint32_t>(5));
    uint32_t in0_block_w = get_arg_val<uint32_t>(6);
    uint32_t num_blocks = get_arg_val<uint32_t>(7);
    uint32_t per_core_M = get_arg_val<uint32_t>(8);
    uint32_t per_core_N = get_arg_val<uint32_t>(9);
    uint32_t in0_num_dests = get_arg_val<uint32_t>(10);
    uint32_t in1_num_dests = get_arg_val<uint32_t>(11);
    uint32_t Nt = get_arg_val<uint32_t>(12);
    uint32_t core_y = get_arg_val<uint32_t>(13);
    uint32_t out_subblock_h = get_arg_val<uint32_t>(14);
    uint32_t out_subblock_w = get_arg_val<uint32_t>(15);
    uint32_t start_core_x = get_arg_val<uint32_t>(16);
    uint32_t start_core_y = get_arg_val<uint32_t>(17);

    constexpr uint32_t cb_in_a = tt::CBIndex::c_0;
    constexpr uint32_t cb_in_b = tt::CBIndex::c_1;
    constexpr uint32_t cb_in_c = tt::CBIndex::c_2;

    const uint32_t tile_size_bytes = get_tile_size(cb_in_a);

    constexpr auto in_a_offset = TensorAccessorArgs<0>();
    constexpr auto in_c_offset = TensorAccessorArgs<in_a_offset.next_compile_time_args_offset()>();

    volatile tt_l1_ptr uint32_t* in0_mcast_receiver_semaphore_addr_ptr =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(in0_mcast_receiver_semaphore_addr);
    *(in0_mcast_receiver_semaphore_addr_ptr) = VALID;

    volatile tt_l1_ptr uint32_t* in0_mcast_sender_semaphore_addr_ptr =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(in0_mcast_sender_semaphore_addr);

    volatile tt_l1_ptr uint32_t* in1_mcast_receiver_semaphore_addr_ptr =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(in1_mcast_receiver_semaphore_addr);

    const auto in_a = TensorAccessor(in_a_offset, in_a_addr, tile_size_bytes);
    const auto in_c = TensorAccessor(in_c_offset, in_c_addr, tile_size_bytes);

    uint32_t l1_write_addr_in0;
    uint32_t l1_write_addr_in1;

    for (uint32_t block = 0; block < num_blocks; ++block) {
        // ---------------------------------------------------------------------------------------------------------------------
        // ---------------------------------------------------------------------------------------------------------------------
        // ---------------------------------------------------------------------------------------------------------------------
        // ---------------------------------------------------------------------------------------------------------------------
        // IN0
        cb_reserve_back(cb_in_a, in0_block_w * per_core_M);
        l1_write_addr_in0 = get_write_ptr(cb_in_a);

        uint32_t in0_start_address =
            l1_write_addr_in0;  // save the start address because we will move l1_write_addr_in0 below

        // reads one block
        for (uint32_t h = 0; h < per_core_M; ++h) {
            uint32_t in0_tensor_tile_id = num_blocks * in0_block_w * (h + core_y * per_core_M) +
                                          block * in0_block_w;  // num_blocks * in0_block_w is one row of tiles
            for (uint32_t w = 0; w < in0_block_w; ++w) {
                noc_async_read_tile(in0_tensor_tile_id, in_a, l1_write_addr_in0);
                l1_write_addr_in0 += tile_size_bytes;
                ++in0_tensor_tile_id;
            }
        }
        noc_async_read_barrier();

        noc_semaphore_wait(in0_mcast_sender_semaphore_addr_ptr, in0_num_dests);
        noc_semaphore_set(in0_mcast_sender_semaphore_addr_ptr, 0);

        // get data addresses and send data
        uint64_t in0_multicast_addr = get_noc_multicast_addr(
            start_core_x + 1,
            start_core_y + core_y,
            start_core_x + in0_num_dests,
            start_core_y + core_y,
            in0_start_address);
        noc_async_write_multicast(
            in0_start_address, in0_multicast_addr, tile_size_bytes * in0_block_w * per_core_M, in0_num_dests);

        // get semaphore addresses and send signal
        uint64_t in0_multicast_addr_semaphores = get_noc_multicast_addr(
            start_core_x + 1,
            start_core_y + core_y,
            start_core_x + in0_num_dests,
            start_core_y + core_y,
            in0_mcast_receiver_semaphore_addr);
        noc_semaphore_set_multicast(in0_mcast_receiver_semaphore_addr, in0_multicast_addr_semaphores, in0_num_dests);

        cb_push_back(cb_in_a, in0_block_w * per_core_M);

        // ---------------------------------------------------------------------------------------------------------------------
        // ---------------------------------------------------------------------------------------------------------------------
        // ---------------------------------------------------------------------------------------------------------------------
        // ---------------------------------------------------------------------------------------------------------------------
        // IN1

        cb_reserve_back(cb_in_b, in0_block_w * per_core_N);

        noc_semaphore_set(in1_mcast_receiver_semaphore_addr_ptr, INVALID);

        uint64_t in1_mcast_sender_semaphore_noc_addr =
            get_noc_addr(start_core_x + 0, start_core_y + 0, in1_mcast_sender_semaphore_addr);
        noc_semaphore_inc(in1_mcast_sender_semaphore_noc_addr, 1);

        noc_semaphore_wait(in1_mcast_receiver_semaphore_addr_ptr, VALID);

        cb_push_back(cb_in_b, in0_block_w * per_core_N);
    }

    // ---------------------------------------------------------------------------------------------------------------------
    // ---------------------------------------------------------------------------------------------------------------------
    // ---------------------------------------------------------------------------------------------------------------------
    // ---------------------------------------------------------------------------------------------------------------------
    // ---------------------------------------------------------------------------------------------------------------------
    // IN2
    uint32_t num_subblocks_h = per_core_M / out_subblock_h;
    uint32_t num_subblocks_w = per_core_N / out_subblock_w;

    for (uint32_t subblock_h = 0; subblock_h < num_subblocks_h; subblock_h++) {
        for (uint32_t subblock_w = 0; subblock_w < num_subblocks_w; subblock_w++) {
            cb_reserve_back(cb_in_c, out_subblock_h * out_subblock_w);
            uint32_t c_l1 = get_write_ptr(cb_in_c);
            for (uint32_t h = 0; h < out_subblock_h; ++h) {
                for (uint32_t w = 0; w < out_subblock_w; ++w) {
                    noc_async_read_tile(
                        Nt * (core_y * per_core_M + subblock_h * out_subblock_h + h) + subblock_w * out_subblock_w + w,
                        in_c,
                        c_l1);
                }
            }
            noc_async_read_barrier();
            cb_push_back(cb_in_c, out_subblock_h * out_subblock_w);
        }
    }
}
