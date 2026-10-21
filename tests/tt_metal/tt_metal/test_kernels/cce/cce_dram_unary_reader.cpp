// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

// dram_unary reader (MM-8 port): GDDR -> TL1 staging ring with SPA loads, handing each chunk to the writer hart on the
// same CCE through the filled semaphore. Staging data, semaphores and the status line go through the uncached alias.
// Before reusing a slot the reader waits for the writer's drained count, then `fence r, w`. After staging a chunk it
// issues `fence w, w` and the absolute filled store.

#include "api/dataflow/dataflow_api.h"
#include "cce_dram_unary_abi.h"
#include "cce_test_common.h"
#include "dev_mem_map.h"
#include "experimental/cce_gddr.h"
#include "risc_common.h"

namespace abi = cce_dram_unary_abi;
using cce_test::arg_u64;
using cce_test::on_one_line;
using cce_test::own_tl1_words;
using cce_test::poll_bounded;

static_assert(abi::kLineBytes == cce_test::kLineBytes);

void kernel_main() {
    const uint32_t status_offset = get_arg_val<uint32_t>(abi::kArgStatusOffset);
    const uint64_t budget_cycles = arg_u64(abi::kArgPollBudgetLo, abi::kArgPollBudgetHi);
    const uint32_t local_mimir = get_arg_val<uint32_t>(abi::kArgLocalMimir);
    const uint64_t source_offset = arg_u64(abi::kArgSourceOffsetLo, abi::kArgSourceOffsetHi);
    const uint32_t staging_offset = get_arg_val<uint32_t>(abi::kArgStagingOffset);
    const uint32_t chunk_bytes = get_arg_val<uint32_t>(abi::kArgChunkBytes);
    const uint32_t num_chunks = get_arg_val<uint32_t>(abi::kArgNumChunks);
    const uint32_t num_slots = get_arg_val<uint32_t>(abi::kArgNumSlots);
    const uint64_t filled_address =
        get_semaphore<ProgrammableCoreType::DRAM>(get_arg_val<uint32_t>(abi::kArgFilledSemaphoreId));
    const uint64_t drained_address =
        get_semaphore<ProgrammableCoreType::DRAM>(get_arg_val<uint32_t>(abi::kArgDrainedSemaphoreId));
    volatile uint32_t* filled = own_tl1_words(filled_address);
    volatile uint32_t* drained = own_tl1_words(drained_address);
    const uint32_t filled_at_start = *filled;
    const uint32_t drained_at_start = *drained;

    const bool args_valid = budget_cycles != 0 && chunk_bytes != 0 && chunk_bytes % abi::kLineBytes == 0 &&
                            chunk_bytes <= abi::kMaxChunkBytes && num_chunks != 0 && num_chunks <= abi::kMaxChunks &&
                            num_slots != 0 && num_slots <= abi::kMaxSlots && staging_offset % abi::kLineBytes == 0 &&
                            status_offset % abi::kLineBytes == 0 && !on_one_line(filled_address, drained_address);

    uint32_t code = abi::kCodeDone;
    uint32_t chunks_done = 0;
    uint32_t max_wait_cycles = 0;
    uint32_t timed_out_chunk = 0;
    uint32_t semaphore_observed = 0;
    if (!args_valid) {
        code = abi::kCodeBadArgs;
    } else {
        const uint32_t chunk_words = chunk_bytes / sizeof(uint32_t);
        for (uint32_t chunk = 0; chunk < num_chunks; ++chunk) {
            if (chunk >= num_slots) {
                const uint32_t slot_free_at = chunk - num_slots + 1;
                uint32_t wait_cycles = 0;
                const bool slot_free = poll_bounded(
                    [drained, slot_free_at, &semaphore_observed] {
                        semaphore_observed = *drained;
                        return semaphore_observed >= slot_free_at;
                    },
                    budget_cycles,
                    wait_cycles);
                max_wait_cycles = wait_cycles > max_wait_cycles ? wait_cycles : max_wait_cycles;
                if (!slot_free) {
                    code = abi::kCodeWaitTimeout;
                    timed_out_chunk = chunk;
                    break;
                }
                asm volatile("fence r, w" ::: "memory");
            }

            const volatile uint32_t* source = reinterpret_cast<const volatile uint32_t*>(experimental::cce_gddr_address(
                local_mimir, local_mimir, source_offset + static_cast<uint64_t>(chunk) * chunk_bytes));
            volatile uint32_t* slot =
                own_tl1_words(staging_offset + static_cast<uint64_t>(chunk % num_slots) * chunk_bytes);
            for (uint32_t word = 0; word < chunk_words; ++word) {
                slot[word] = source[word];
            }
            asm volatile("fence w, w" ::: "memory");
            *filled = chunk + 1;
            chunks_done = chunk + 1;
        }
    }

    volatile uint32_t* status = own_tl1_words(status_offset);
    status[abi::kStatusCode] = code;
    status[abi::kStatusChunksDone] = chunks_done;
    status[abi::kStatusMaxWaitCycles] = max_wait_cycles;
    status[abi::kStatusTimedOutChunk] = timed_out_chunk;
    status[abi::kStatusSemaphoreObserved] = semaphore_observed;
    status[abi::kStatusFilledSemaphoreAddress] = static_cast<uint32_t>(filled_address);
    status[abi::kStatusDrainedSemaphoreAddress] = static_cast<uint32_t>(drained_address);
    status[abi::kStatusFilledAtStart] = filled_at_start;
    status[abi::kStatusDrainedAtStart] = drained_at_start;
    asm volatile("fence w, w" ::: "memory");
    status[abi::kStatusMagic] = abi::kReaderDoneMagic;
}
