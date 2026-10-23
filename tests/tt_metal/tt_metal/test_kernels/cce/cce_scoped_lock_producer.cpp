// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Scoped-lock cache ops, producer hart (MM-8 port of the test_scoped_lock_cache CLM rows). It stores NEW to the probe
// words for the consumer hart on the same CCE: through the uncached alias in kModeInvalidateOnAcquire, through the
// cache under the lock in kModeFlushOnRelease. In kModeFlushOnRelease it then reads the locked lines once and polls
// them through the uncached alias, which shows the release flush reached TL1, and discards the probe lines from L1 D$
// and L2, so the unlocked lines' dirty NEW never reaches TL1. Then `fence w, w` and the published flag. Flags and the
// status line go through the uncached alias. The MM41 modes (kModeMultiConsumer, kModeRingWrap,
// kModeNonSnoopingProducer) are described in cce_scoped_lock_abi.h.

#include "cce_scoped_lock_abi.h"
#include "cce_test_common.h"
#include "dev_mem_map.h"
#include "experimental/cce_gddr.h"
#include "internal/scoped_lock_cache_ops.h"
#include "risc_common.h"

namespace abi = cce_scoped_lock_abi;
using cce_test::arg_u64;
using cce_test::DataCheck;
using cce_test::own_tl1_words;
using cce_test::poll_bounded;

// A status line the host can find: 64 B aligned, between the end of the harts' kernel text and the end of the
// hart-cacheable port. Otherwise the kernel writes nothing and the host deadline reports it.
static bool status_line_valid(uint32_t status_offset) {
    constexpr uint64_t kTestAreaBegin =
        MEM_CCE_KERNEL_BASE + static_cast<uint64_t>(MEM_CCE_KERNEL_SIZE) * MEM_CCE_LOCAL_HARTS;
    return status_offset % abi::kLineBytes == 0 && status_offset >= kTestAreaBegin &&
           status_offset + static_cast<uint64_t>(abi::kLineBytes) <= MEMORY_PORT_CACHEABLE_MEM_PORT_MEM_SIZE;
}

// Whether probe words [first_line, first_line + num_lines) of `probe_words` read kNewBase + line.
static bool new_values_visible(
    const volatile uint32_t* probe_words, uint32_t first_line, uint32_t num_lines, DataCheck& check) {
    check = DataCheck{};
    for (uint32_t line = first_line; line < first_line + num_lines; ++line) {
        check.record(line * abi::kWordsPerLine, probe_words[line * abi::kWordsPerLine], abi::kNewBase + line);
    }
    return check.mismatched_words == 0;
}

// Whether every word w of a ring slot reads ring_word(round, w) through `slot_words`.
static bool ring_slot_landed(
    const volatile uint32_t* slot_words, uint32_t slot_bytes, uint32_t round, DataCheck& check) {
    check = DataCheck{};
    for (uint32_t word = 0; word < slot_bytes / sizeof(uint32_t); ++word) {
        check.record((round << 8) | word, slot_words[word], abi::ring_word(round, word));
    }
    return check.mismatched_words == 0;
}

static volatile uint32_t* sram_words(uint32_t cce_index, uint32_t tl1_offset) {
    return reinterpret_cast<volatile uint32_t*>(experimental::cce_sram_address(cce_index, tl1_offset));
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
    const uint32_t consumer_count = get_arg_val<uint32_t>(abi::kArgConsumerCount);
    const uint32_t ring_slots = get_arg_val<uint32_t>(abi::kArgRingSlots);
    const uint32_t slot_lines = get_arg_val<uint32_t>(abi::kArgSlotLines);
    const uint32_t rounds = get_arg_val<uint32_t>(abi::kArgRounds);
    const uint32_t target_cce = get_arg_val<uint32_t>(abi::kArgTargetCce);
    const bool offsets_valid = budget_cycles != 0 && region_offset % abi::kLineBytes == 0 &&
                               warmed_flag_offset % abi::kLineBytes == 0 &&
                               published_flag_offset % abi::kLineBytes == 0 && status_offset % abi::kLineBytes == 0;
    const bool lock_valid = lock_num_lines != 0 && lock_first_line + lock_num_lines <= abi::kNumLines;
    bool args_valid = false;
    switch (mode) {
        case abi::kModeInvalidateOnAcquire:
        case abi::kModeFlushOnRelease: args_valid = offsets_valid && lock_valid; break;
        case abi::kModeMultiConsumer:
            args_valid = offsets_valid && lock_valid && consumer_count != 0 && consumer_count <= abi::kMaxConsumers;
            break;
        case abi::kModeRingWrap:
            args_valid = offsets_valid && ring_slots != 0 && slot_lines != 0 &&
                         ring_slots * slot_lines <= abi::kNumLines && rounds != 0 && rounds <= 0xFF;
            break;
        case abi::kModeNonSnoopingProducer: args_valid = offsets_valid; break;
        default: break;
    }

    uint32_t code = abi::kCodeDone;
    uint32_t flags = 0;
    uint32_t wait_cycles = 0;
    uint32_t poll_cycles = 0;
    uint32_t last_poll = abi::kPollNone;
    bool last_poll_ready = false;
    uint32_t flag_observed = 0;
    uint32_t rounds_done = 0;
    DataCheck check;
    if (!args_valid) {
        code = abi::kCodeBadArgs;
    } else if (mode == abi::kModeInvalidateOnAcquire) {
        volatile uint32_t* uncached_probe = own_tl1_words(region_offset);
        volatile uint32_t* warmed = own_tl1_words(warmed_flag_offset);
        last_poll = abi::kPollPeer;
        last_poll_ready = poll_bounded(
            [warmed, &flag_observed] {
                flag_observed = *warmed;
                return flag_observed != 0;
            },
            budget_cycles,
            wait_cycles);
        if (last_poll_ready) {
            flags |= abi::kFlagPeerSignalled;
            asm volatile("fence r, w" ::: "memory");
            for (uint32_t line = 0; line < abi::kNumLines; ++line) {
                uncached_probe[line * abi::kWordsPerLine] = abi::kNewBase + line;
            }
            asm volatile("fence w, w" ::: "memory");
            *own_tl1_words(published_flag_offset) = 1;
        } else {
            code = abi::kCodePeerTimeout;
        }
    } else if (mode == abi::kModeFlushOnRelease) {
        volatile uint32_t* cached_probe = reinterpret_cast<volatile uint32_t*>(static_cast<uintptr_t>(region_offset));
        const uintptr_t lock_address = region_offset + lock_first_line * abi::kLineBytes;
        const uint32_t lock_bytes = lock_num_lines * abi::kLineBytes;
        for (uint32_t line = 0; line < abi::kNumLines; ++line) {
            (void)cached_probe[line * abi::kWordsPerLine];
        }
        scoped_lock_acquire_cache_ops(lock_address, lock_bytes);
        for (uint32_t line = 0; line < abi::kNumLines; ++line) {
            cached_probe[line * abi::kWordsPerLine] = abi::kNewBase + line;
        }
        scoped_lock_release_cache_ops(lock_address, lock_bytes);
        if (new_values_visible(own_tl1_words(region_offset), lock_first_line, lock_num_lines, check)) {
            flags |= abi::kFlagLandedAtFirstRead;
        }

        last_poll = abi::kPollData;
        last_poll_ready = poll_bounded(
            [region_offset, lock_first_line, lock_num_lines, &check] {
                return new_values_visible(own_tl1_words(region_offset), lock_first_line, lock_num_lines, check);
            },
            budget_cycles,
            poll_cycles);
        for (uint32_t line = 0; line < abi::kNumLines; ++line) {
            invalidate_l1_dcache(region_offset + line * abi::kLineBytes);
        }
        invalidate_l2_cache_range(region_offset, abi::kRegionBytes);
        if (last_poll_ready) {
            flags |= abi::kFlagLanded;
            asm volatile("fence w, w" ::: "memory");
            *own_tl1_words(published_flag_offset) = 1;
        } else {
            code = abi::kCodeDataTimeout;
        }
    } else if (mode == abi::kModeMultiConsumer) {
        last_poll = abi::kPollPeer;
        last_poll_ready = true;
        for (uint32_t consumer = 0; consumer < consumer_count && last_poll_ready; ++consumer) {
            volatile uint32_t* warmed = own_tl1_words(warmed_flag_offset + consumer * abi::kLineBytes);
            uint32_t consumer_wait_cycles = 0;
            last_poll_ready = poll_bounded(
                [warmed, &flag_observed] {
                    flag_observed = *warmed;
                    return flag_observed != 0;
                },
                budget_cycles,
                consumer_wait_cycles);
            wait_cycles += consumer_wait_cycles;
        }
        if (!last_poll_ready) {
            code = abi::kCodePeerTimeout;
        } else {
            flags |= abi::kFlagPeerSignalled;
            asm volatile("fence r, rw" ::: "memory");
            volatile uint32_t* cached_probe =
                reinterpret_cast<volatile uint32_t*>(static_cast<uintptr_t>(region_offset));
            const uintptr_t lock_address = region_offset + lock_first_line * abi::kLineBytes;
            const uint32_t lock_bytes = lock_num_lines * abi::kLineBytes;
            scoped_lock_acquire_cache_ops(lock_address, lock_bytes);
            for (uint32_t line = lock_first_line; line < lock_first_line + lock_num_lines; ++line) {
                cached_probe[line * abi::kWordsPerLine] = abi::kNewBase + line;
            }
            scoped_lock_release_cache_ops(lock_address, lock_bytes);
            if (new_values_visible(own_tl1_words(region_offset), lock_first_line, lock_num_lines, check)) {
                flags |= abi::kFlagLandedAtFirstRead;
            }
            last_poll = abi::kPollData;
            last_poll_ready = poll_bounded(
                [region_offset, lock_first_line, lock_num_lines, &check] {
                    return new_values_visible(own_tl1_words(region_offset), lock_first_line, lock_num_lines, check);
                },
                budget_cycles,
                poll_cycles);
            if (last_poll_ready) {
                flags |= abi::kFlagLanded;
                asm volatile("fence w, w" ::: "memory");
                *own_tl1_words(published_flag_offset) = 1;
            } else {
                code = abi::kCodeDataTimeout;
            }
        }
    } else if (mode == abi::kModeRingWrap) {
        const volatile uint32_t* read_pointer = own_tl1_words(warmed_flag_offset);
        volatile uint32_t* write_pointer = own_tl1_words(published_flag_offset);
        const uint32_t slot_bytes = slot_lines * abi::kLineBytes;
        for (uint32_t round = 0; round < rounds; ++round) {
            if (round >= ring_slots) {
                const uint32_t needed = round + 1 - ring_slots;
                uint32_t round_wait_cycles = 0;
                last_poll = abi::kPollPeer;
                last_poll_ready = poll_bounded(
                    [read_pointer, needed, &flag_observed] {
                        flag_observed = *read_pointer;
                        return flag_observed >= needed;
                    },
                    budget_cycles,
                    round_wait_cycles);
                wait_cycles += round_wait_cycles;
                if (!last_poll_ready) {
                    code = abi::kCodePeerTimeout;
                    break;
                }
                flags |= abi::kFlagPeerSignalled;
                asm volatile("fence r, rw" ::: "memory");
            }
            const uintptr_t slot_address = region_offset + (round % ring_slots) * slot_bytes;
            volatile uint32_t* slot_words = reinterpret_cast<volatile uint32_t*>(slot_address);
            scoped_lock_acquire_cache_ops(slot_address, slot_bytes);
            for (uint32_t word = 0; word < slot_bytes / sizeof(uint32_t); ++word) {
                slot_words[word] = abi::ring_word(round, word);
            }
            scoped_lock_release_cache_ops(slot_address, slot_bytes);
            uint32_t round_poll_cycles = 0;
            last_poll = abi::kPollData;
            last_poll_ready = poll_bounded(
                [slot_address, slot_bytes, round, &check] {
                    return ring_slot_landed(own_tl1_words(slot_address), slot_bytes, round, check);
                },
                budget_cycles,
                round_poll_cycles);
            poll_cycles += round_poll_cycles;
            if (!last_poll_ready) {
                code = abi::kCodeDataTimeout;
                break;
            }
            flags |= abi::kFlagLanded;
            asm volatile("fence w, w" ::: "memory");
            *write_pointer = round + 1;
            rounds_done = round + 1;
        }
    } else {
        const volatile uint32_t* warmed = sram_words(target_cce, warmed_flag_offset);
        volatile uint32_t* probe_words = sram_words(target_cce, region_offset);
        last_poll = abi::kPollPeer;
        last_poll_ready = poll_bounded(
            [warmed, &flag_observed] {
                flag_observed = *warmed;
                return flag_observed != 0;
            },
            budget_cycles,
            wait_cycles);
        if (last_poll_ready) {
            flags |= abi::kFlagPeerSignalled;
            asm volatile("fence" ::: "memory");
            for (uint32_t line = 0; line < abi::kNumLines; ++line) {
                probe_words[line * abi::kWordsPerLine] = abi::kNewBase + line;
            }
            asm volatile("fence" ::: "memory");
            *sram_words(target_cce, published_flag_offset) = 1;
        } else {
            code = abi::kCodePeerTimeout;
        }
    }

    volatile uint32_t* status = own_tl1_words(status_offset);
    status[abi::kStatusCode] = code;
    status[abi::kStatusFlags] = flags;
    status[abi::kStatusWaitCycles] = wait_cycles;
    status[abi::kStatusPollCycles] = poll_cycles;
    status[abi::kStatusLastPoll] = last_poll;
    status[abi::kStatusLastPollReady] = last_poll_ready ? 1 : 0;
    status[abi::kStatusFlagObserved] = flag_observed;
    status[abi::kStatusMismatchedWords] = check.mismatched_words;
    status[abi::kStatusFirstMismatchWord] = check.first_mismatch_word;
    status[abi::kStatusObserved] = check.observed;
    status[abi::kStatusExpected] = check.expected;
    status[abi::kStatusRounds] = rounds_done;
    asm volatile("fence w, w" ::: "memory");
    status[abi::kStatusMagic] = abi::kProducerDoneMagic;
}
