// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <stdint.h>

#include "cce_test_status_abi.h"

// cce_dram_unary_{reader,writer} kernel ABI. Both kernels and the host (dram/test_mimir_dm_dram_unary.cpp) include it.
// Args, codes and status words are append-only: a later stage adds entries after the last one and never renumbers.
namespace cce_dram_unary_abi {

using cce_test_status_abi::kLineBytes;

// Runtime args, the same vector for both kernels except kArgStatusOffset. The reader copies chunk c of the GDDR source
// into staging slot c % num_slots in TL1 and the writer copies that slot to chunk c of the GDDR destination. GDDR
// operands are chiplet offsets on kArgLocalMimir, reached through cce_gddr_address. kArgStatusOffset and
// kArgStagingOffset are 64 B aligned offsets in the CCE's TL1. kArgChunkBytes is a whole number of lines. The two
// semaphores are DRAM-core semaphore ids: filled counts the chunks the reader has staged, drained the chunks the writer
// has written out; both start at 0 and hold absolute counts. kArgPollBudget* is rdcycle cycles per bounded wait and
// must be nonzero.
inline constexpr uint32_t kArgStatusOffset = 0, kArgPollBudgetLo = 1, kArgPollBudgetHi = 2, kArgLocalMimir = 3,
                          kArgSourceOffsetLo = 4, kArgSourceOffsetHi = 5, kArgDestinationOffsetLo = 6,
                          kArgDestinationOffsetHi = 7, kArgStagingOffset = 8, kArgChunkBytes = 9, kArgNumChunks = 10,
                          kArgNumSlots = 11, kArgFilledSemaphoreId = 12, kArgDrainedSemaphoreId = 13, kNumArgs = 14;

inline constexpr uint32_t kMaxChunkBytes = 4096, kMaxSlots = 4, kMaxChunks = 64;

// Status line, one per hart, written once at exit: every field, `fence w, w`, then the magic. kStatusChunksDone counts
// the chunks the hart finished (staged, or written out). kStatusMaxWaitCycles is the longest semaphore wait. On a wait
// timeout, kStatusTimedOutChunk is the chunk and kStatusSemaphoreObserved the last semaphore value read. The semaphore
// addresses are TL1 offsets, and the *AtStart words are both semaphores as the hart read them before its first store.
using cce_test_status_abi::kStatusCode;
using cce_test_status_abi::kStatusMagic;
inline constexpr uint32_t kStatusChunksDone = 2, kStatusMaxWaitCycles = 3, kStatusTimedOutChunk = 4,
                          kStatusSemaphoreObserved = 5, kStatusFilledSemaphoreAddress = 6,
                          kStatusDrainedSemaphoreAddress = 7, kStatusFilledAtStart = 8, kStatusDrainedAtStart = 9,
                          kStatusWords = 10;
static_assert(kStatusWords * sizeof(uint32_t) <= kLineBytes);
inline constexpr uint32_t kReaderDoneMagic = 0x2EAD0DEu, kWriterDoneMagic = 0x3B17E0DEu;
using cce_test_status_abi::kStatusPoison;

// kCodeDone: every chunk finished. kCodeWaitTimeout: a semaphore wait used its budget (the reader waits on drained to
// reuse a slot, the writer on filled). kCodeBadArgs also covers two semaphores on one 64 B line. A hart stops at its
// first timeout.
inline constexpr uint32_t kCodeDone = 1, kCodeWaitTimeout = 2, kCodeBadArgs = 3;

}  // namespace cce_dram_unary_abi
