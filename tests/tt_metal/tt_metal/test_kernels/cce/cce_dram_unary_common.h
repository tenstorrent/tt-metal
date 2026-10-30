// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

// What the dram_unary reader and writer kernels share: the args other than the GDDR offset, the semaphores, the
// argument checks and the status line.

#pragma once

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#include "cce_dram_unary_abi.h"
#include "cce_test_common.h"

namespace cce_dram_unary {

namespace abi = cce_dram_unary_abi;

struct Args {
    uint32_t status_offset;
    uint64_t budget_cycles;
    uint32_t local_mimir;
    uint32_t staging_offset;
    uint32_t chunk_bytes;
    uint32_t num_chunks;
    uint32_t num_slots;
};

inline Args read_args() {
    Args args;
    args.status_offset = get_arg_val<uint32_t>(abi::kArgStatusOffset);
    args.budget_cycles = cce_test::arg_u64(abi::kArgPollBudgetLo, abi::kArgPollBudgetHi);
    args.local_mimir = get_arg_val<uint32_t>(abi::kArgLocalMimir);
    args.staging_offset = get_arg_val<uint32_t>(abi::kArgStagingOffset);
    args.chunk_bytes = get_arg_val<uint32_t>(abi::kArgChunkBytes);
    args.num_chunks = get_arg_val<uint32_t>(abi::kArgNumChunks);
    args.num_slots = get_arg_val<uint32_t>(abi::kArgNumSlots);
    return args;
}

// The filled and drained semaphores, and both counts as this hart read them before its first store.
struct Semaphores {
    uint64_t filled_address;
    uint64_t drained_address;
    volatile uint32_t* filled;
    volatile uint32_t* drained;
    uint32_t filled_at_start;
    uint32_t drained_at_start;
};

inline Semaphores read_semaphores() {
    Semaphores semaphores;
    semaphores.filled_address =
        get_semaphore<ProgrammableCoreType::DRAM>(get_arg_val<uint32_t>(abi::kArgFilledSemaphoreId));
    semaphores.drained_address =
        get_semaphore<ProgrammableCoreType::DRAM>(get_arg_val<uint32_t>(abi::kArgDrainedSemaphoreId));
    semaphores.filled = cce_test::own_tl1_words(semaphores.filled_address);
    semaphores.drained = cce_test::own_tl1_words(semaphores.drained_address);
    semaphores.filled_at_start = *semaphores.filled;
    semaphores.drained_at_start = *semaphores.drained;
    return semaphores;
}

inline bool args_valid(const Args& args, const Semaphores& semaphores) {
    return args.budget_cycles != 0 && args.chunk_bytes != 0 && args.chunk_bytes % abi::kLineBytes == 0 &&
           args.chunk_bytes <= abi::kMaxChunkBytes && args.num_chunks != 0 && args.num_chunks <= abi::kMaxChunks &&
           args.num_slots != 0 && args.num_slots <= abi::kMaxSlots && args.staging_offset % abi::kLineBytes == 0 &&
           args.status_offset % abi::kLineBytes == 0 &&
           !cce_test::on_one_line(semaphores.filled_address, semaphores.drained_address);
}

struct Status {
    uint32_t code = abi::kCodeDone;
    uint32_t chunks_done = 0;
    uint32_t max_wait_cycles = 0;
    uint32_t timed_out_chunk = 0;
    uint32_t semaphore_observed = 0;
};

inline void write_status(
    uint32_t status_offset, const Status& status, const Semaphores& semaphores, uint32_t done_magic) {
    volatile uint32_t* status_line = cce_test::own_tl1_words(status_offset);
    status_line[abi::kStatusCode] = status.code;
    status_line[abi::kStatusChunksDone] = status.chunks_done;
    status_line[abi::kStatusMaxWaitCycles] = status.max_wait_cycles;
    status_line[abi::kStatusTimedOutChunk] = status.timed_out_chunk;
    status_line[abi::kStatusSemaphoreObserved] = status.semaphore_observed;
    status_line[abi::kStatusFilledSemaphoreAddress] = static_cast<uint32_t>(semaphores.filled_address);
    status_line[abi::kStatusDrainedSemaphoreAddress] = static_cast<uint32_t>(semaphores.drained_address);
    status_line[abi::kStatusFilledAtStart] = semaphores.filled_at_start;
    status_line[abi::kStatusDrainedAtStart] = semaphores.drained_at_start;
    cce_test::publish_done_magic(status_line, done_magic);
}

}  // namespace cce_dram_unary
