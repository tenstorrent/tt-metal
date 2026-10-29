// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// Pure DRAM read bandwidth probe: streams packets of one DRAM bank into a small L1 ring
// (overwritten, never consumed) with `depth` packets of `packet_bytes` in flight, tracked by
// transaction ids. Runs on NCRISC and/or BRISC; no compute, no CBs. With both_nocs the RISC issues
// alternate packets on NOC0 and NOC1.
//
// Compile-time args: 0 packet_bytes, 1 depth (<= 15, even with both_nocs), 2 iters, 3 packets per
// iteration, 4 both_nocs
// Runtime args: 0 bank_id, 1 dram_addr, 2 byte offset of this reader's first packet in the bank,
// 3 l1_addr, 4 vc, 5 byte stride between this reader's consecutive packets

#include "api/dataflow/dataflow_api.h"

constexpr uint32_t packet_bytes = get_compile_time_arg_val(0);
constexpr uint32_t depth = get_compile_time_arg_val(1);
constexpr uint32_t iters = get_compile_time_arg_val(2);
constexpr uint32_t packets = get_compile_time_arg_val(3);
constexpr bool both_nocs = get_compile_time_arg_val(4);

void kernel_main() {
    const uint32_t bank_id = get_arg_val<uint32_t>(0);
    const uint32_t dram_addr = get_arg_val<uint32_t>(1);
    const uint32_t offset = get_arg_val<uint32_t>(2);
    const uint32_t l1 = get_arg_val<uint32_t>(3);
    const uint32_t vc = get_arg_val<uint32_t>(4);
    const uint32_t stride = get_arg_val<uint32_t>(5);
    const uint8_t noc_a = noc_index;
    const uint8_t noc_b = both_nocs ? 1 - noc_index : noc_index;
    reset_noc_trid_barrier_counter(NOC_CLEAR_OUTSTANDING_REQ_MASK, noc_a);
    const uint64_t base_a = get_noc_addr_from_bank_id<true>(bank_id, dram_addr, noc_a);
    noc_async_read_one_packet_set_state<true>(base_a, packet_bytes, vc, noc_a);
    uint64_t base_b = base_a;
    if constexpr (both_nocs) {
        reset_noc_trid_barrier_counter(NOC_CLEAR_OUTSTANDING_REQ_MASK, noc_b);
        base_b = get_noc_addr_from_bank_id<true>(bank_id, dram_addr, noc_b);
        noc_async_read_one_packet_set_state<true>(base_b, packet_bytes, vc, noc_b);
    }
    uint32_t n = 0;
    for (uint32_t it = 0; it < iters; it++) {
        uint32_t src = offset;
        for (uint32_t p = 0; p < packets; p++, n++) {
            const uint32_t slot = n % depth;
            const uint32_t trid = slot + 1;
            const bool on_b = both_nocs && (slot & 1);
            const uint8_t noc = on_b ? noc_b : noc_a;
            if (n >= depth) {
                noc_async_read_barrier_with_trid(trid, noc);
            }
            noc_async_read_set_trid(trid, noc);
            noc_async_read_one_packet_with_state_with_trid(
                on_b ? base_b : base_a, src, l1 + slot * packet_bytes, trid, noc);
            src += stride;
        }
    }
    // per-trid barriers only: the other NOC's issued-read counter is not maintained for this RISC
    for (uint32_t slot = 0; slot < depth; slot++) {
        noc_async_read_barrier_with_trid(slot + 1, both_nocs && (slot & 1) ? noc_b : noc_a);
    }
}
