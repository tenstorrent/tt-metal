// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

// dram_unary reader (MM-8 port): GDDR -> TL1 staging ring with SPA loads, handing each chunk to the writer hart on the
// same CCE through the filled semaphore. Staging data, semaphores and the status line go through the uncached alias.
// Before reusing a slot the reader waits for the writer's drained count, then `fence r, w`. After staging a chunk it
// issues `fence w, w` and the absolute filled store.

#include "cce_dram_unary_abi.h"
#include "cce_dram_unary_common.h"
#include "cce_test_common.h"
#include "dev_mem_map.h"
#include "experimental/cce_gddr.h"
#include "risc_common.h"

namespace abi = cce_dram_unary_abi;
using cce_dram_unary::Args;
using cce_dram_unary::args_valid;
using cce_dram_unary::read_args;
using cce_dram_unary::read_semaphores;
using cce_dram_unary::Semaphores;
using cce_dram_unary::Status;
using cce_dram_unary::write_status;
using cce_test::arg_u64;
using cce_test::own_tl1_words;
using cce_test::poll_bounded;

void kernel_main() {
    const Args args = read_args();
    const uint64_t source_offset = arg_u64(abi::kArgSourceOffsetLo, abi::kArgSourceOffsetHi);
    const Semaphores semaphores = read_semaphores();

    Status status;
    if (!args_valid(args, semaphores)) {
        status.code = abi::kCodeBadArgs;
    } else {
        const uint32_t chunk_words = args.chunk_bytes / sizeof(uint32_t);
        for (uint32_t chunk = 0; chunk < args.num_chunks; ++chunk) {
            if (chunk >= args.num_slots) {
                const uint32_t slot_free_at = chunk - args.num_slots + 1;
                uint32_t wait_cycles = 0;
                const volatile uint32_t* drained = semaphores.drained;
                const bool slot_free = poll_bounded(
                    [drained, slot_free_at, &status] {
                        status.semaphore_observed = *drained;
                        return status.semaphore_observed >= slot_free_at;
                    },
                    args.budget_cycles,
                    wait_cycles);
                status.max_wait_cycles = wait_cycles > status.max_wait_cycles ? wait_cycles : status.max_wait_cycles;
                if (!slot_free) {
                    status.code = abi::kCodeWaitTimeout;
                    status.timed_out_chunk = chunk;
                    break;
                }
                asm volatile("fence r, w" ::: "memory");
            }

            const volatile uint32_t* source = reinterpret_cast<const volatile uint32_t*>(experimental::cce_gddr_address(
                args.local_mimir, source_offset + static_cast<uint64_t>(chunk) * args.chunk_bytes));
            volatile uint32_t* slot =
                own_tl1_words(args.staging_offset + static_cast<uint64_t>(chunk % args.num_slots) * args.chunk_bytes);
            for (uint32_t word = 0; word < chunk_words; ++word) {
                slot[word] = source[word];
            }
            asm volatile("fence w, w" ::: "memory");
            *semaphores.filled = chunk + 1;
            status.chunks_done = chunk + 1;
        }
    }

    write_status(args.status_offset, status, semaphores, abi::kReaderDoneMagic);
}
