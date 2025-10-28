// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "dataflow_api.h"
#include "debug/dprint.h"

void kernel_main() {
    constexpr std::uint32_t iteration = get_compile_time_arg_val(0);
    constexpr std::uint32_t page_size = get_compile_time_arg_val(1);

    std::uint32_t noc_x = get_arg_val<uint32_t>(0);
    std::uint32_t noc_y = get_arg_val<uint32_t>(1);
    bool mcast = get_arg_val<uint32_t>(2);
    std::uint32_t top_left_core_x = get_arg_val<uint32_t>(3);
    std::uint32_t top_left_core_y = get_arg_val<uint32_t>(4);
    std::uint32_t bottom_right_core_x = get_arg_val<uint32_t>(5);
    std::uint32_t bottom_right_core_y = get_arg_val<uint32_t>(6);
    std::uint32_t num_dests = get_arg_val<uint32_t>(7);

    constexpr uint32_t cb_id = 0;
    uint32_t l1_read_addr = get_read_ptr(cb_id);

    uint64_t mcast_addr_self_noc;
    if (noc_index == 0) {
        mcast_addr_self_noc = get_noc_multicast_addr(
            top_left_core_x, top_left_core_y, bottom_right_core_x, bottom_right_core_y, l1_read_addr, noc_index);
    } else {
        mcast_addr_self_noc = get_noc_multicast_addr(
            bottom_right_core_x, bottom_right_core_y, top_left_core_x, top_left_core_y, l1_read_addr, noc_index);
    }

    uint64_t addr_self_noc = get_noc_addr(noc_x, noc_y, l1_read_addr, noc_index);

    DPRINT << "Start" << ENDL();

    // Test stateful read API
    noc_async_read_set_state(addr_self_noc, noc_index);
    for (uint32_t i = 0; i < iteration; i++) {
        noc_async_read_with_state(l1_read_addr, l1_read_addr, page_size, noc_index);
    }

    // Test stateful read one packet API
    constexpr uint32_t vc = 0;
    noc_async_read_one_packet_set_state(addr_self_noc, page_size, vc, noc_index);
    for (uint32_t i = 0; i < iteration; i++) {
        noc_async_read_one_packet_with_state(l1_read_addr, l1_read_addr, vc, noc_index);
    }

    // Test stateful write one packet API
    noc_async_write_one_packet_set_state(addr_self_noc, page_size, noc_index);
    for (uint32_t i = 0; i < iteration; i++) {
        noc_async_write_one_packet_with_state(l1_read_addr, l1_read_addr, noc_index);
    }

    // Test gen_fast
    constexpr auto s_args = TensorAccessorArgs<2>();
    const auto s0 = TensorAccessor(s_args, l1_read_addr, page_size);

    for (uint32_t i = 0; i < iteration; i++) {
        uint32_t noc = noc_index;

        // uint32_t noc = noc_index;
        uint64_t noc_addr = get_noc_addr(noc_x, noc_y, l1_read_addr, noc);

        // Test read
        noc_async_read_one_packet(noc_addr, l1_read_addr, page_size, noc);
        noc_async_read(noc_addr, l1_read_addr, page_size, noc);
        // interleaved read
        noc_async_read_tile(i % 1024, s0, l1_read_addr, 0, noc);

        // Test semaphore
        noc_semaphore_inc(noc_addr, 1, noc);
        noc_semaphore_set_remote(l1_read_addr, noc_addr, noc);

        // Test write
        noc_async_write(l1_read_addr, noc_addr, page_size, noc);
        noc_async_write_one_packet(l1_read_addr, noc_addr, page_size, noc);
        // interleaved write
        noc_async_write_tile(i % 1024, s0, l1_read_addr, noc);

        // Test mcast
        if (mcast) {
            // write mcast
            noc_async_write_multicast_one_packet(
                l1_read_addr, mcast_addr_self_noc, page_size, num_dests - 1, false, noc);
            noc_async_write_multicast(l1_read_addr, mcast_addr_self_noc, page_size, num_dests - 1, false, noc);
            noc_async_write_multicast_loopback_src(l1_read_addr, mcast_addr_self_noc, page_size, num_dests, false, noc);
            // semaphore mcast
            noc_semaphore_set_multicast(l1_read_addr, mcast_addr_self_noc, num_dests - 1, false, noc);
            noc_semaphore_set_multicast_loopback_src(l1_read_addr, mcast_addr_self_noc, num_dests, false, noc);
        }

// dw_write skip BH since there's HW issue
#ifndef ARCH_BLACKHOLE
        noc_inline_dw_write(noc_addr, 1, 0xF, noc);
#endif
    }

    // barrier on all txns
    noc_async_full_barrier(noc_index);

    // Previous multicasts would have put trids into a non-zero state, so reset the barrier counter
    reset_noc_trid_barrier_counter(NOC_CLEAR_OUTSTANDING_REQ_MASK, noc_index);

    // DRAM sharded read API
    uint64_t src_addr = get_noc_addr_from_bank_id<true>(0, DRAM_ALIGNMENT);
    noc_async_read_one_packet_set_state<true>(src_addr, page_size, vc, noc_index);
    for (uint32_t i = 0; i < iteration; i++) {
        uint32_t trid = i % (NOC_MAX_TRANSACTION_ID + 1);
        noc_async_read_tile_dram_sharded_with_state_with_trid(src_addr, DRAM_ALIGNMENT, l1_read_addr, trid, noc_index);
    }

    for (uint32_t i = 0; i <= NOC_MAX_TRANSACTION_ID; i++) {
        noc_async_read_barrier_with_trid(i, noc_index);
    }

    // L1 sharded write API
    for (uint32_t i = 0; i < iteration; i++) {
        uint32_t trid = i % (NOC_MAX_TRANSACTION_ID + 1);
        noc_async_write_one_packet_with_trid(l1_read_addr, addr_self_noc, page_size, trid, write_cmd_buf, noc_index);
    }
    for (uint32_t i = 0; i <= NOC_MAX_TRANSACTION_ID; i++) {
        noc_async_write_barrier_with_trid(i, noc_index);
    }

    DPRINT << "END" << ENDL();
    DPRINT << "noc_mode " << (uint)noc_mode << ENDL();

    // Barrier test - test barrier itself working properly
    for (int noc = 0; noc < NUM_NOCS; noc++) {
        noc_async_read_barrier();
        noc_async_write_barrier();
        noc_async_writes_flushed();
        noc_async_posted_writes_flushed();
        noc_async_atomic_barrier();
        noc_async_full_barrier();
    }
}
