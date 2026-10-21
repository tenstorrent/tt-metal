// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Scoped-lock cache ops, consumer hart (MM-8 port of the test_scoped_lock_cache CLM rows). It observes the producer
// hart's NEW stores on the same CCE: in kModeInvalidateOnAcquire through its own cache after acquiring the lock, in
// kModeFlushOnRelease through the uncached alias, which reads TL1 past every cache. It stores what it read to the
// result lines through the uncached alias before the status line.

#include "cce_scoped_lock_abi.h"
#include "cce_test_common.h"
#include "dev_mem_map.h"
#include "internal/scoped_lock_cache_ops.h"
#include "risc_common.h"

namespace abi = cce_scoped_lock_abi;
using cce_test::arg_u64;
using cce_test::own_tl1_words;
using cce_test::poll_bounded;

// One value per probe line, as read.
struct ProbeRead {
    uint32_t values[abi::kNumLines] = {};
};

static ProbeRead read_probe_words(const volatile uint32_t* probe_words) {
    ProbeRead read;
    for (uint32_t line = 0; line < abi::kNumLines; ++line) {
        read.values[line] = probe_words[line * abi::kWordsPerLine];
    }
    return read;
}

static void store_result_line(volatile uint32_t* result_words, uint32_t result_line, const ProbeRead& read) {
    for (uint32_t line = 0; line < abi::kNumLines; ++line) {
        result_words[result_line * abi::kWordsPerLine + line] = read.values[line];
    }
}

void kernel_main() {
    const uint32_t mode = get_arg_val<uint32_t>(abi::kArgMode);
    const uint32_t status_offset = get_arg_val<uint32_t>(abi::kArgStatusOffset);
    const uint64_t budget_cycles = arg_u64(abi::kArgPollBudgetLo, abi::kArgPollBudgetHi);
    const uint32_t region_offset = get_arg_val<uint32_t>(abi::kArgRegionOffset);
    const uint32_t lock_first_line = get_arg_val<uint32_t>(abi::kArgLockFirstLine);
    const uint32_t lock_num_lines = get_arg_val<uint32_t>(abi::kArgLockNumLines);
    const uint32_t warmed_flag_offset = get_arg_val<uint32_t>(abi::kArgWarmedFlagOffset);
    const uint32_t published_flag_offset = get_arg_val<uint32_t>(abi::kArgPublishedFlagOffset);
    const uint32_t result_offset = get_arg_val<uint32_t>(abi::kArgResultOffset);
    const bool args_valid = (mode == abi::kModeInvalidateOnAcquire || mode == abi::kModeFlushOnRelease) &&
                            budget_cycles != 0 && lock_num_lines != 0 &&
                            lock_first_line + lock_num_lines <= abi::kNumLines &&
                            region_offset % abi::kLineBytes == 0 && warmed_flag_offset % abi::kLineBytes == 0 &&
                            published_flag_offset % abi::kLineBytes == 0 && result_offset % abi::kLineBytes == 0 &&
                            status_offset % abi::kLineBytes == 0;

    uint32_t code = abi::kCodeDone;
    uint32_t flags = 0;
    uint32_t wait_cycles = 0;
    uint32_t last_poll = abi::kPollNone;
    bool last_poll_ready = false;
    uint32_t flag_observed = 0;
    if (!args_valid) {
        code = abi::kCodeBadArgs;
    } else {
        const volatile uint32_t* cached_probe =
            reinterpret_cast<const volatile uint32_t*>(static_cast<uintptr_t>(region_offset));
        const volatile uint32_t* uncached_probe = own_tl1_words(region_offset);
        volatile uint32_t* result_words = own_tl1_words(result_offset);
        const bool invalidate_mode = mode == abi::kModeInvalidateOnAcquire;

        ProbeRead warm;
        if (invalidate_mode) {
            warm = read_probe_words(cached_probe);
            asm volatile("fence r, w" ::: "memory");
            *own_tl1_words(warmed_flag_offset) = 1;
        }

        const volatile uint32_t* published = own_tl1_words(published_flag_offset);
        last_poll = abi::kPollPeer;
        last_poll_ready = poll_bounded(
            [published, &flag_observed] {
                flag_observed = *published;
                return flag_observed != 0;
            },
            budget_cycles,
            wait_cycles);
        if (!last_poll_ready) {
            code = abi::kCodePeerTimeout;
        } else {
            flags |= abi::kFlagPeerSignalled;
            asm volatile("fence r, r" ::: "memory");
            ProbeRead cached_after;
            if (invalidate_mode) {
                const uintptr_t lock_address = region_offset + lock_first_line * abi::kLineBytes;
                const uint32_t lock_bytes = lock_num_lines * abi::kLineBytes;
                scoped_lock_acquire_cache_ops(lock_address, lock_bytes);
                scoped_lock_release_cache_ops(lock_address, lock_bytes);
                cached_after = read_probe_words(cached_probe);
            }
            const ProbeRead uncached_after = read_probe_words(uncached_probe);

            if (invalidate_mode) {
                store_result_line(result_words, abi::kResultWarm, warm);
                store_result_line(result_words, abi::kResultCached, cached_after);
            }
            store_result_line(result_words, abi::kResultUncached, uncached_after);
        }
    }

    volatile uint32_t* status = own_tl1_words(status_offset);
    status[abi::kStatusCode] = code;
    status[abi::kStatusFlags] = flags;
    status[abi::kStatusWaitCycles] = wait_cycles;
    status[abi::kStatusPollCycles] = 0;
    status[abi::kStatusLastPoll] = last_poll;
    status[abi::kStatusLastPollReady] = last_poll_ready ? 1 : 0;
    status[abi::kStatusFlagObserved] = flag_observed;
    status[abi::kStatusMismatchedWords] = 0;
    status[abi::kStatusFirstMismatchWord] = 0;
    status[abi::kStatusObserved] = 0;
    status[abi::kStatusExpected] = 0;
    asm volatile("fence w, w" ::: "memory");
    status[abi::kStatusMagic] = abi::kConsumerDoneMagic;
}
