// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// ETH DRAM streaming kernel: each ETH core reads from exactly ONE assigned DRAM
// bank into ETH L1 for a fixed wall-clock duration, then writes its elapsed
// ticks and iteration count to the first 16 bytes of the assigned bank's DRAM
// region. Persisting the result in DRAM is necessary because active-ETH L1 is
// reclaimed by firmware after the kernel exits.
//
// The run is duration-bounded (not a fixed amount of work) so it ends together
// with the other generators even when it shares the ETH core with link traffic.
//
// One bank per core means summing per-core bandwidths gives total DRAM bandwidth.
//
// Compile-time args (indices 0..4):
//   0: duration_ticks_lo – run duration in wall-clock ticks (low 32 bits)
//   1: pages_per_bank    – pages read per iteration from the assigned bank
//   2: page_size_bytes   – bytes per page
//   3: noc_wait_cycles   – cycles to busy-wait between consecutive NOC page issues;
//                          0 = no throttle (maximum DRAM utilization)
//   4: duration_ticks_hi – run duration in wall-clock ticks (high 32 bits)
//
// Runtime args (indices 0..3):
//   0: dram_src_addr       – base DRAM buffer address (interleaved across banks)
//   1: eth_l1_staging_addr – ETH L1 unreserved base; first 64 bytes are reserved
//   2: bank_id             – which DRAM bank this core is assigned to
//   3: read_vc             – static NOC request VC for this stream
//
// Output (16 B): [elapsed_lo, elapsed_hi, iterations_lo, iterations_hi]

#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#include "internal/ethernet/tt_eth_api.h"

void kernel_main() {
    constexpr uint64_t duration_ticks =
        (static_cast<uint64_t>(get_compile_time_arg_val(4)) << 32) | get_compile_time_arg_val(0);
    constexpr uint32_t pages_per_bank = get_compile_time_arg_val(1);
    constexpr uint32_t page_size_bytes = get_compile_time_arg_val(2);
    constexpr uint32_t noc_wait_cycles = get_compile_time_arg_val(3);

    const uint32_t dram_src_addr = get_arg_val<uint32_t>(0);
    const uint32_t eth_l1_staging_addr = get_arg_val<uint32_t>(1);
    const uint32_t bank_id = get_arg_val<uint32_t>(2);
    const uint32_t read_vc = get_arg_val<uint32_t>(3);

    // The first 16 bytes hold timing output. Start data at the next 64-byte
    // NOC-word boundary; unaligned Blackhole read destinations truncate the
    // tail of each transaction.
    volatile tt_l1_ptr uint32_t* timing_out = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(eth_l1_staging_addr);
    constexpr uint32_t l1_staging_header_bytes = 64;
    const uint32_t l1_data_addr = eth_l1_staging_addr + l1_staging_header_bytes;

    // Compute the NOC address for this core's assigned DRAM bank once.
    // bank_to_dram_offset[bank_id] is added inside get_noc_addr_from_bank_id.
    const uint64_t bank_noc_base = get_noc_addr_from_bank_id<true>(bank_id, dram_src_addr);

    // Set stateful-packet read state once for the assigned bank and page size;
    // the state only needs to be re-set if the source NOC address changes.
    noc_async_read_one_packet_set_state<true>(bank_noc_base, page_size_bytes, read_vc);

    // Only the low clock word is read: both ERISCs may be running, and reading
    // the latched 64-bit clock here could disturb the link kernel's reads.
    uint32_t prev = eth_risc_reg_read(ETH_RISC_WALL_CLOCK_0);
    uint64_t elapsed = 0;
    uint64_t iterations = 0;

    while (elapsed < duration_ticks) {
        uint32_t dst = l1_data_addr;
        for (uint32_t p = 0; p < pages_per_bank; p++) {
            noc_async_read_one_packet_with_state<true, true>(bank_noc_base + p * page_size_bytes, dst, read_vc);
            dst += page_size_bytes;

            // Throttle DRAM utilization by busy-waiting between NOC issue commands.
            // noc_wait_cycles == 0 at compile time → this block is eliminated entirely.
            if constexpr (noc_wait_cycles) {
#pragma GCC unroll noc_wait_cycles
                for (uint32_t k = 0; k < noc_wait_cycles; k++) {
                    asm volatile("nop");
                }
            }
        }
        // Active-ETH kernels must service base-firmware routing while waiting.
        eth_noc_async_read_barrier();
        iterations++;
        const uint32_t now = eth_risc_reg_read(ETH_RISC_WALL_CLOCK_0);
        elapsed += now - prev;
        prev = now;
    }

    // Persist elapsed ticks and iteration count in the assigned DRAM bank.
    // The host cannot read these reliably from active-ETH L1 after completion.
    timing_out[0] = static_cast<uint32_t>(elapsed);
    timing_out[1] = static_cast<uint32_t>(elapsed >> 32);
    timing_out[2] = static_cast<uint32_t>(iterations);
    timing_out[3] = static_cast<uint32_t>(iterations >> 32);
    noc_async_write(eth_l1_staging_addr, bank_noc_base, 16);
    eth_noc_async_write_barrier();
}
