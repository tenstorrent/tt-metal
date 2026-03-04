// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// ETH DRAM streaming kernel: each ETH core reads from exactly ONE assigned DRAM
// bank into ETH L1, times the run, and writes start/end wall-clock timestamps
// to the first 16 bytes of the staging region.
//
// One bank per core means summing per-core bandwidths gives total DRAM bandwidth.
//
// Compile-time args (indices 0..2):
//   0: num_loops         – outer loop count
//   1: pages_per_bank    – pages read per iteration from the assigned bank
//   2: page_size_bytes   – bytes per page
//
// Runtime args (indices 0..2):
//   0: dram_src_addr       – base DRAM buffer address (interleaved across banks)
//   1: eth_l1_staging_addr – ETH L1 unreserved base; first 16 bytes hold timing output
//   2: bank_id             – which DRAM bank this core is assigned to

#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#include "internal/ethernet/tt_eth_api.h"

void kernel_main() {
    constexpr uint32_t num_loops = get_compile_time_arg_val(0);
    constexpr uint32_t pages_per_bank = get_compile_time_arg_val(1);
    constexpr uint32_t page_size_bytes = get_compile_time_arg_val(2);

    const uint32_t dram_src_addr = get_arg_val<uint32_t>(0);
    const uint32_t eth_l1_staging_addr = get_arg_val<uint32_t>(1);
    const uint32_t bank_id = get_arg_val<uint32_t>(2);

    // Timing output occupies the first 16 bytes; data staging starts after.
    volatile tt_l1_ptr uint32_t* timing_out = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(eth_l1_staging_addr);
    const uint32_t l1_data_addr = eth_l1_staging_addr + 16;

    // Compute the NOC address for this core's assigned DRAM bank once.
    // bank_to_dram_offset[bank_id] is added inside get_noc_addr_from_bank_id.
    const uint64_t bank_noc_base = get_noc_addr_from_bank_id<true>(bank_id, dram_src_addr);

    // Set stateful-packet read state once for the assigned bank and page size;
    // the state only needs to be re-set if the source NOC address changes.
    noc_async_read_one_packet_set_state(bank_noc_base, page_size_bytes);

    uint64_t t0 = eth_read_wall_clock();

    for (uint32_t iter = 0; iter < num_loops; iter++) {
        uint32_t dst = l1_data_addr;
        for (uint32_t p = 0; p < pages_per_bank; p++) {
            noc_async_read_one_packet_with_state(bank_noc_base + p * page_size_bytes, dst);
            dst += page_size_bytes;
        }
        // eth_noc_async_read_barrier calls run_routing() in its wait loop,
        // keeping the cooperative base firmware alive on active ETH cores.
        eth_noc_async_read_barrier();
    }

    uint64_t t1 = eth_read_wall_clock();

    // Write t0 and t1 as lo/hi uint32_t pairs to first 16 bytes of staging region.
    timing_out[0] = static_cast<uint32_t>(t0 & 0xFFFFFFFFu);
    timing_out[1] = static_cast<uint32_t>(t0 >> 32);
    timing_out[2] = static_cast<uint32_t>(t1 & 0xFFFFFFFFu);
    timing_out[3] = static_cast<uint32_t>(t1 >> 32);
}
