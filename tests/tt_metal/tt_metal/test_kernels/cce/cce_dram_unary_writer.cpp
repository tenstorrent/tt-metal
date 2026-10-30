// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

// dram_unary writer (MM-8 port): TL1 staging ring -> GDDR with SPA stores, taking each chunk from the reader hart on
// the same CCE through the filled semaphore. Staging data, semaphores and the status line go through the uncached
// alias. The writer waits for the reader's filled count, then `fence r, r` before reading the slot. After writing a
// chunk it issues `fence r, w` and the absolute drained store, which frees the slot.

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
    const uint64_t destination_offset = arg_u64(abi::kArgDestinationOffsetLo, abi::kArgDestinationOffsetHi);
    const Semaphores semaphores = read_semaphores();

    Status status;
    if (!args_valid(args, semaphores)) {
        status.code = abi::kCodeBadArgs;
    } else {
        const uint32_t chunk_words = args.chunk_bytes / sizeof(uint32_t);
        for (uint32_t chunk = 0; chunk < args.num_chunks; ++chunk) {
            uint32_t wait_cycles = 0;
            const volatile uint32_t* filled = semaphores.filled;
            const bool chunk_staged = poll_bounded(
                [filled, chunk, &status] {
                    status.semaphore_observed = *filled;
                    return status.semaphore_observed > chunk;
                },
                args.budget_cycles,
                wait_cycles);
            status.max_wait_cycles = wait_cycles > status.max_wait_cycles ? wait_cycles : status.max_wait_cycles;
            if (!chunk_staged) {
                status.code = abi::kCodeWaitTimeout;
                status.timed_out_chunk = chunk;
                break;
            }
            asm volatile("fence r, r" ::: "memory");

            const volatile uint32_t* slot =
                own_tl1_words(args.staging_offset + static_cast<uint64_t>(chunk % args.num_slots) * args.chunk_bytes);
            volatile uint32_t* destination = reinterpret_cast<volatile uint32_t*>(experimental::cce_gddr_address(
                args.local_mimir, destination_offset + static_cast<uint64_t>(chunk) * args.chunk_bytes));
            for (uint32_t word = 0; word < chunk_words; ++word) {
                destination[word] = slot[word];
            }
            asm volatile("fence r, w" ::: "memory");
            *semaphores.drained = chunk + 1;
            status.chunks_done = chunk + 1;
        }
    }

    write_status(args.status_offset, status, semaphores, abi::kWriterDoneMagic);
}
