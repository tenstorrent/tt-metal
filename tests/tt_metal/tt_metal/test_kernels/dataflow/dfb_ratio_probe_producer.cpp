// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Test kernel: DFB producer whose ratio of NoC transactions to announced slots is selectable.
//
// Every mode fills the buffer correctly and every transfer targets the buffer itself. Only the ratio
// of NoC reads to push_back'd slots changes. This tests whether the buffer's credits track NoC
// transaction completions rather than push_back calls: if they do, a ratio other than one to one is
// enough to break it, with no second destination and nothing else unusual involved.
//
// Modes (arg 3):
//   0  one read per announced slot. The ordinary shape, and the control.
//   1  two half-entry reads per announced slot, both into the buffer. More transactions than
//      announcements, so a transaction-counted credit scheme over-grants.
//   2  one double-entry read per two announced slots. Fewer transactions than announcements, so a
//      transaction-counted scheme under-grants and the consumer should starve.
//   3  arg 4 reads per announced slot, each one that fraction of an entry. Mode 1 generalized, so the
//      size of the surplus can be pushed past the buffer's depth rather than staying just above one.
//
// Mode 2 requires an even entry count and a buffer at least two slots deep. Mode 3 requires the entry
// size to divide evenly by the read count.
//
// Runtime args:
//   arg 0: source DRAM address
//   arg 1: DRAM bank ID
//   arg 2: number of entries to announce in total
//   arg 3: ratio mode, as above
//   arg 4: reads per announced slot, for mode 3

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/dataflow/endpoints.h"
#include "api/dataflow/noc.h"

void kernel_main() {
    uint32_t src_addr = get_arg_val<uint32_t>(0);
    const uint32_t bank_id = get_arg_val<uint32_t>(1);
    const uint32_t num_entries = get_arg_val<uint32_t>(2);
    const uint32_t ratio_mode = get_arg_val<uint32_t>(3);
    const uint32_t sub_reads = get_arg_val<uint32_t>(4);

    Noc noc;
    AllocatorBank<AllocatorBankType::DRAM> dram_src;
    DataflowBuffer buf(dfb::my_local_dfb_name);
    const uint32_t entry_size = buf.get_entry_size();

    if (ratio_mode == 2) {
        for (uint32_t i = 0; i < num_entries; i += 2) {
            buf.reserve_back(2);
            noc.async_read(dram_src, buf, 2 * entry_size, {.bank_id = bank_id, .addr = src_addr}, {.offset_bytes = 0});
            noc.async_read_barrier();
            buf.push_back(2);
            src_addr += 2 * entry_size;
        }
        return;
    }

    if (ratio_mode == 3) {
        const uint32_t chunk = entry_size / sub_reads;
        for (uint32_t i = 0; i < num_entries; i++) {
            buf.reserve_back(1);
            for (uint32_t k = 0; k < sub_reads; k++) {
                const uint32_t offset = k * chunk;
                noc.async_read(
                    dram_src, buf, chunk, {.bank_id = bank_id, .addr = src_addr + offset}, {.offset_bytes = offset});
            }
            noc.async_read_barrier();
            buf.push_back(1);
            src_addr += entry_size;
        }
        return;
    }

    for (uint32_t i = 0; i < num_entries; i++) {
        buf.reserve_back(1);
        if (ratio_mode == 1) {
            const uint32_t half = entry_size / 2;
            noc.async_read(dram_src, buf, half, {.bank_id = bank_id, .addr = src_addr}, {.offset_bytes = 0});
            noc.async_read(dram_src, buf, half, {.bank_id = bank_id, .addr = src_addr + half}, {.offset_bytes = half});
        } else {
            noc.async_read(dram_src, buf, entry_size, {.bank_id = bank_id, .addr = src_addr}, {.offset_bytes = 0});
        }
        noc.async_read_barrier();
        buf.push_back(1);
        src_addr += entry_size;
    }
}
