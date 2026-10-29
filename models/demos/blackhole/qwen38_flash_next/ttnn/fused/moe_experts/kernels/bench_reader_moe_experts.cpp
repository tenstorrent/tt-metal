// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Step-0 microbench of the routed-expert program: the DRAM rate of `readers` cores per bank streaming moe_compute's
// cached BF4 layout.  Each core reads `blocks` blocks (40 tiles = two 20-tile, 11,520 B transactions) of the bank
// piece it is told, `reps` times, into a ring of `slots` L1 slots with `inflight` blocks outstanding (one trid each).
// Runtime args: bank id, vc, byte offset of the first block in the bank, blocks, reps, expert stride in bytes (rep r
// reads expert (37 r) mod 64: scattered like the routing, not the same lines again).
#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/circular_buffer.h"

void kernel_main() {
    constexpr uint32_t slots = get_named_compile_time_arg_val("slots");
    constexpr uint32_t inflight = get_named_compile_time_arg_val("inflight");
    constexpr uint32_t cb_w = get_named_compile_time_arg_val("cb_w");
    constexpr uint32_t txn_bytes = get_named_compile_time_arg_val("txn_bytes");
    constexpr uint32_t txns_per_block = get_named_compile_time_arg_val("txns_per_block");
    const uint32_t bank = get_arg_val<uint32_t>(0);
    const uint32_t vc = get_arg_val<uint32_t>(1);
    const uint32_t base = get_arg_val<uint32_t>(2);  // buffer address + byte offset of this core's first block
    const uint32_t blocks = get_arg_val<uint32_t>(3);
    const uint32_t reps = get_arg_val<uint32_t>(4);
    const uint32_t expert_stride = get_arg_val<uint32_t>(5);
    constexpr uint32_t block_bytes = txn_bytes * txns_per_block;

    CircularBuffer cb(cb_w);
    const uint32_t l1 = cb.get_write_ptr();
    const uint64_t bank_base = get_noc_addr_from_bank_id<true>(bank, 0);
    noc_async_read_one_packet_set_state<true>(bank_base, txn_bytes, vc);
    uint32_t trid_issue = 1, trid_wait = 1, slot = 0, pending = 0;
    for (uint32_t rep = 0; rep < reps; ++rep) {
        const uint32_t expert_base = base + ((rep * 37) & 63) * expert_stride;
        for (uint32_t b = 0; b < blocks; ++b) {
            noc_async_read_set_trid(trid_issue);
            for (uint32_t t = 0; t < txns_per_block; ++t) {
                noc_async_read_one_packet_with_state_with_trid<false, false>(
                    bank_base,
                    expert_base + b * block_bytes + t * txn_bytes,
                    l1 + slot * block_bytes + t * txn_bytes,
                    trid_issue);
            }
            if (++slot == slots) {
                slot = 0;
            }
            if (++trid_issue > slots) {
                trid_issue = 1;
            }
            if (++pending == inflight) {
                noc_async_read_barrier_with_trid(trid_wait);
                if (++trid_wait > slots) {
                    trid_wait = 1;
                }
                --pending;
            }
        }
    }
    for (; pending > 0; --pending) {
        noc_async_read_barrier_with_trid(trid_wait);
        if (++trid_wait > slots) {
            trid_wait = 1;
        }
    }
}
