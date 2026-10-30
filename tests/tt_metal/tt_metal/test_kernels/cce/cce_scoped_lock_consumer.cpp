// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Scoped-lock cache ops, consumer hart (MM-8 port of the test_scoped_lock_cache CLM rows). It observes the producer
// hart's NEW stores on the same CCE: in kModeInvalidateOnAcquire through its own cache after acquiring the lock, in
// kModeFlushOnRelease through the uncached alias, which reads TL1 past every cache. It stores what it read to the
// result lines through the uncached alias before the status line. The MM41 modes (kModeMultiConsumer, kModeRingWrap,
// kModeNonSnoopingProducer) are described in cce_scoped_lock_abi.h.

#include "cce_scoped_lock_abi.h"
#include "cce_scoped_lock_common.h"
#include "cce_test_common.h"
#include "internal/scoped_lock_cache_ops.h"
#include "risc_common.h"

namespace abi = cce_scoped_lock_abi;
using cce_scoped_lock::lock_lines_valid;
using cce_scoped_lock::ring_wrap_valid;
using cce_scoped_lock::Status;
using cce_scoped_lock::write_status;
using cce_test::arg_u64;
using cce_test::DataCheck;
using cce_test::own_tl1_words;
using cce_test::poll_bounded;
using cce_test::status_line_valid;

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

// Whether every probe word of `probe_words` reads kNewBase + line.
static bool all_probe_words_new(const volatile uint32_t* probe_words, DataCheck& check) {
    check = DataCheck{};
    for (uint32_t line = 0; line < abi::kNumLines; ++line) {
        check.record(line * abi::kWordsPerLine, probe_words[line * abi::kWordsPerLine], abi::kNewBase + line);
    }
    return check.mismatched_words == 0;
}

void kernel_main() {
    const uint32_t mode = get_arg_val<uint32_t>(abi::kArgMode);
    const uint32_t status_offset = get_arg_val<uint32_t>(abi::kArgStatusOffset);
    if (!status_line_valid(status_offset)) {
        return;
    }
    const uint64_t budget_cycles = arg_u64(abi::kArgPollBudgetLo, abi::kArgPollBudgetHi);
    const uint32_t region_offset = get_arg_val<uint32_t>(abi::kArgRegionOffset);
    const uint32_t lock_first_line = get_arg_val<uint32_t>(abi::kArgLockFirstLine);
    const uint32_t lock_num_lines = get_arg_val<uint32_t>(abi::kArgLockNumLines);
    const uint32_t warmed_flag_offset = get_arg_val<uint32_t>(abi::kArgWarmedFlagOffset);
    const uint32_t published_flag_offset = get_arg_val<uint32_t>(abi::kArgPublishedFlagOffset);
    const uint32_t result_offset = get_arg_val<uint32_t>(abi::kArgResultOffset);
    const uint32_t consumer_count = get_arg_val<uint32_t>(abi::kArgConsumerCount);
    const uint32_t consumer_index = get_arg_val<uint32_t>(abi::kArgConsumerIndex);
    const uint32_t ring_slots = get_arg_val<uint32_t>(abi::kArgRingSlots);
    const uint32_t slot_lines = get_arg_val<uint32_t>(abi::kArgSlotLines);
    const uint32_t rounds = get_arg_val<uint32_t>(abi::kArgRounds);
    const bool offsets_valid = budget_cycles != 0 && region_offset % abi::kLineBytes == 0 &&
                               warmed_flag_offset % abi::kLineBytes == 0 &&
                               published_flag_offset % abi::kLineBytes == 0 && result_offset % abi::kLineBytes == 0 &&
                               status_offset % abi::kLineBytes == 0;
    const bool lock_valid = lock_lines_valid(lock_first_line, lock_num_lines);
    bool args_valid = false;
    switch (mode) {
        case abi::kModeInvalidateOnAcquire:
        case abi::kModeFlushOnRelease:
        case abi::kModeNonSnoopingProducer: args_valid = offsets_valid && lock_valid; break;
        case abi::kModeMultiConsumer:
            args_valid =
                offsets_valid && lock_valid && consumer_count <= abi::kMaxConsumers && consumer_index < consumer_count;
            break;
        case abi::kModeRingWrap: args_valid = offsets_valid && ring_wrap_valid(ring_slots, slot_lines, rounds); break;
        default: break;
    }

    Status status;
    if (!args_valid) {
        status.code = abi::kCodeBadArgs;
    } else if (mode == abi::kModeRingWrap) {
        volatile uint32_t* read_pointer = own_tl1_words(warmed_flag_offset);
        const volatile uint32_t* write_pointer = own_tl1_words(published_flag_offset);
        const uint32_t slot_bytes = slot_lines * abi::kLineBytes;
        for (uint32_t round = 0; round < rounds; ++round) {
            uint32_t round_wait_cycles = 0;
            status.last_poll = abi::kPollPeer;
            status.last_poll_ready = poll_bounded(
                [write_pointer, round, &status] {
                    status.flag_observed = *write_pointer;
                    return status.flag_observed > round;
                },
                budget_cycles,
                round_wait_cycles);
            status.wait_cycles += round_wait_cycles;
            if (!status.last_poll_ready) {
                status.code = abi::kCodePeerTimeout;
                break;
            }
            status.flags |= abi::kFlagPeerSignalled;
            asm volatile("fence r, r" ::: "memory");
            const uintptr_t slot_address = region_offset + (round % ring_slots) * slot_bytes;
            const volatile uint32_t* slot_words = reinterpret_cast<const volatile uint32_t*>(slot_address);
            scoped_lock_acquire_cache_ops(slot_address, slot_bytes);
            for (uint32_t word = 0; word < slot_bytes / sizeof(uint32_t); ++word) {
                status.check.record((round << 8) | word, slot_words[word], abi::ring_word(round, word));
            }
            scoped_lock_release_cache_ops(slot_address, slot_bytes);
            asm volatile("fence r, w" ::: "memory");
            *read_pointer = round + 1;
            status.rounds_done = round + 1;
        }
    } else {
        const volatile uint32_t* cached_probe =
            reinterpret_cast<const volatile uint32_t*>(static_cast<uintptr_t>(region_offset));
        const volatile uint32_t* uncached_probe = own_tl1_words(region_offset);
        volatile uint32_t* result_words = own_tl1_words(result_offset);
        const bool invalidate_mode = mode != abi::kModeFlushOnRelease;
        const uint32_t own_warmed_flag_offset =
            warmed_flag_offset + (mode == abi::kModeMultiConsumer ? consumer_index * abi::kLineBytes : 0);

        ProbeRead warm;
        if (invalidate_mode) {
            warm = read_probe_words(cached_probe);
            asm volatile("fence r, w" ::: "memory");
            *own_tl1_words(own_warmed_flag_offset) = 1;
        }

        const volatile uint32_t* published = own_tl1_words(published_flag_offset);
        status.last_poll = abi::kPollPeer;
        status.last_poll_ready = poll_bounded(
            [published, &status] {
                status.flag_observed = *published;
                return status.flag_observed != 0;
            },
            budget_cycles,
            status.wait_cycles);
        if (!status.last_poll_ready) {
            status.code = abi::kCodePeerTimeout;
        } else {
            status.flags |= abi::kFlagPeerSignalled;
        }
        if (status.code == abi::kCodeDone && mode == abi::kModeNonSnoopingProducer) {
            status.last_poll = abi::kPollData;
            status.last_poll_ready = poll_bounded(
                [uncached_probe, &status] { return all_probe_words_new(uncached_probe, status.check); },
                budget_cycles,
                status.poll_cycles);
            if (status.last_poll_ready) {
                status.flags |= abi::kFlagLanded;
            } else {
                status.code = abi::kCodeDataTimeout;
            }
        }
        if (status.code == abi::kCodeDone) {
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

    write_status(status_offset, status, abi::kConsumerDoneMagic);
}
