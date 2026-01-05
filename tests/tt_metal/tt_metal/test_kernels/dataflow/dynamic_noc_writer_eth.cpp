// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "internal/risc_attribs.h"

void kernel_main() {
    constexpr std::uint32_t num_iterations = get_compile_time_arg_val(0);
    constexpr std::uint32_t page_size = get_compile_time_arg_val(1);
    constexpr std::uint32_t num_riscs = get_compile_time_arg_val(2);

    std::uint32_t noc_x = get_arg_val<uint32_t>(0);
    std::uint32_t noc_y = get_arg_val<uint32_t>(1);
    std::uint32_t risc_index = get_arg_val<uint32_t>(2);

    // Multicast parameters
    bool mcast_enable = get_arg_val<uint32_t>(3);
    std::uint32_t top_left_core_x = get_arg_val<uint32_t>(4);
    std::uint32_t top_left_core_y = get_arg_val<uint32_t>(5);
    std::uint32_t bottom_right_core_x = get_arg_val<uint32_t>(6);
    std::uint32_t bottom_right_core_y = get_arg_val<uint32_t>(7);
    std::uint32_t num_dests = get_arg_val<uint32_t>(8);

    volatile tt_l1_ptr uint32_t* semaphore_ptrs[num_riscs];
    for (uint32_t i = 0; i < num_riscs; i++) {
        semaphore_ptrs[i] = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(get_arg_val<uint32_t>(9 + i)));
    }

    uint32_t l1_addr = get_arg_val<uint32_t>(9 + num_riscs);

    uint64_t remote_addr = get_noc_addr(noc_x, noc_y, l1_addr, noc_index);

    uint64_t mcast_addr = 0;
    if (mcast_enable) {
        // For NOC0, use forward order; for NOC1, use reverse order
        if (noc_index == 0) {
            mcast_addr = get_noc_multicast_addr(
                top_left_core_x, top_left_core_y, bottom_right_core_x, bottom_right_core_y, l1_addr, noc_index);
        } else {
            mcast_addr = get_noc_multicast_addr(
                bottom_right_core_x, bottom_right_core_y, top_left_core_x, top_left_core_y, l1_addr, noc_index);
        }
    }

    // Test stateful read API
    noc_async_read_set_state(remote_addr, noc_index);
    for (uint32_t i = 0; i < num_iterations; i++) {
        noc_async_read_with_state(l1_addr, l1_addr, page_size, noc_index);
    }

    // Test stateful read one packet API
    constexpr uint32_t vc = 0;
    noc_async_read_one_packet_set_state(remote_addr, page_size, vc, noc_index);
    for (uint32_t i = 0; i < num_iterations; i++) {
        noc_async_read_one_packet_with_state(l1_addr, l1_addr, vc, noc_index);
    }

    // Test stateful write one packet API
    noc_async_write_one_packet_set_state(remote_addr, page_size, noc_index);
    for (uint32_t i = 0; i < num_iterations; i++) {
        noc_async_write_one_packet_with_state(l1_addr, l1_addr, noc_index);
    }

    // Test various NOC operations using only our assigned NOC
    for (uint32_t i = 0; i < num_iterations; i++) {
        // Read operations
        noc_async_read(remote_addr, l1_addr, page_size, noc_index);
        noc_async_read_one_packet(remote_addr, l1_addr, page_size, noc_index);

        // Write operations
        noc_async_write(l1_addr, remote_addr, page_size, noc_index);
        noc_async_write_one_packet(l1_addr, remote_addr, page_size, noc_index);

        // Semaphore operations
        noc_semaphore_inc(remote_addr, 1, noc_index);
        noc_semaphore_set_remote(l1_addr, remote_addr, noc_index);

        // Multicast operations
        if (mcast_enable) {
            // Multicast write operations
            noc_async_write_multicast(l1_addr, mcast_addr, page_size, num_dests, false, noc_index);
            noc_async_write_multicast_one_packet(l1_addr, mcast_addr, page_size, num_dests, false, noc_index);

            // Multicast with loopback (includes source)
            noc_async_write_multicast_loopback_src(l1_addr, mcast_addr, page_size, num_dests, false, noc_index);

            // Multicast semaphore operations
            noc_semaphore_set_multicast(l1_addr, mcast_addr, num_dests, false, noc_index);
            noc_semaphore_set_multicast_loopback_src(l1_addr, mcast_addr, num_dests, false, noc_index);
        }
    }

    // Full barrier on our assigned NOC
    noc_async_full_barrier(noc_index);

    // Sync all RISCs before clearing TRID counters (multicast operations use TRIDs)
    *semaphore_ptrs[risc_index] = 1;
    for (uint32_t i = 0; i < num_riscs; i++) {
        noc_semaphore_wait(semaphore_ptrs[i], 1);
    }

    // Reset NOC transaction ID barrier counter after multicasts
    if (mcast_enable) {
        reset_noc_trid_barrier_counter(NOC_CLEAR_OUTSTANDING_REQ_MASK, noc_index);
    }

    // Sync all RISCs after clearing TRID counters
    *semaphore_ptrs[risc_index] = 2;
    for (uint32_t i = 0; i < num_riscs; i++) {
        noc_semaphore_wait(semaphore_ptrs[i], 2);
    }

    // Test DRAM read operations (if DRAM is available)
    uint64_t dram_addr = get_noc_addr_from_bank_id<true>(0, DRAM_ALIGNMENT);
    for (uint32_t i = 0; i < num_iterations; i++) {
        noc_async_read(dram_addr, l1_addr, page_size, noc_index);
        noc_async_write(l1_addr, dram_addr, page_size, noc_index);
    }
    noc_async_full_barrier(noc_index);

    // Final sync: Set local semaphore directly (both RISCs share same L1)
    *semaphore_ptrs[risc_index] = 3;

    // Wait for all RISCs to signal completion
    for (uint32_t i = 0; i < num_riscs; i++) {
        noc_semaphore_wait(semaphore_ptrs[i], 3);
    }

    // Test barrier APIs - comprehensive barrier verification
    noc_async_read_barrier();
    noc_async_write_barrier();
    noc_async_writes_flushed();
    noc_async_posted_writes_flushed();
    noc_async_atomic_barrier();
    noc_async_full_barrier();
}
