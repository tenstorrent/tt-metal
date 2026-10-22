// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Cache operations on a hart's TL1 data area (quasar_cache and quasar_cache_perf port). Cached accesses use the
// hart-local TL1 address; the hart observes TL1 through the uncached alias, which bypasses L1 D$ and L2. The status
// line is written once at exit through the uncached alias.

#include "cce_cache_ops_abi.h"
#include "cce_test_common.h"
#include "dev_mem_map.h"
#include "risc_common.h"

namespace abi = cce_cache_ops_abi;
using cce_test::arg_u64;
using cce_test::read_cycle_counter;

template <typename T>
static volatile T* cached(uint32_t offset) {
    return reinterpret_cast<volatile T*>(static_cast<uint64_t>(offset));
}

template <typename T>
static volatile T* uncached(uint32_t offset) {
    return reinterpret_cast<volatile T*>(static_cast<uint64_t>(offset) + static_cast<uint64_t>(MEM_L1_UNCACHED_BASE));
}

struct Check {
    uint32_t matches = 0;
    uint32_t first_mismatch_word = 0;
    uint32_t observed = 0;
};

static Check check_words(const volatile uint32_t* words, uint32_t num_words, uint32_t value) {
    Check check;
    bool mismatch_seen = false;
    for (uint32_t word = 0; word < num_words; ++word) {
        const uint32_t observed = words[word];
        if (observed == value + word) {
            check.matches++;
        } else if (!mismatch_seen) {
            mismatch_seen = true;
            check.first_mismatch_word = word;
            check.observed = observed;
        }
    }
    return check;
}

static void store_words(volatile uint32_t* words, uint32_t num_words, uint32_t value) {
    for (uint32_t word = 0; word < num_words; ++word) {
        words[word] = value + word;
    }
}

template <typename LineOp>
static void for_each_line(uint32_t data_offset, uint32_t num_bytes, LineOp line_op) {
    for (uint32_t line = data_offset; line < data_offset + num_bytes; line += abi::kLineBytes) {
        line_op(line);
    }
}

static void invalidate_l1_lines(uint32_t data_offset, uint32_t num_bytes) {
    for_each_line(data_offset, num_bytes, [](uint32_t line) { invalidate_l1_dcache(line); });
}

static void invalidate_lines(uint32_t data_offset, uint32_t num_bytes) {
    for_each_line(data_offset, num_bytes, [](uint32_t line) {
        invalidate_l1_dcache(line);
        invalidate_l2_cache_line(line);
    });
}

static bool is_store_mode(uint32_t mode) {
    return mode <= abi::kModeL2InvalidateLine || mode == abi::kModeProducer || mode == abi::kModeL1InvalidateFull;
}

static bool is_fresh_read_mode(uint32_t mode) {
    return mode == abi::kModeInvalidateFreshRead || mode == abi::kModeL2InvalidateFreshRead;
}

static bool is_write_path_mode(uint32_t mode) {
    return mode >= abi::kModeWriteUncached1B && mode <= abi::kModeWriteCachedFastFlush;
}

// Inlined so that no call frame sits on the stack across the whole-L1 D$ discard of kModeL1InvalidateFull.
static inline __attribute__((always_inline)) void run_store_mode(
    uint32_t mode, uint32_t data_offset, uint32_t num_bytes) {
    switch (mode) {
        case abi::kModeL1FlushLine:
            for_each_line(data_offset, num_bytes, [](uint32_t line) { flush_l1_dcache(line); });
            invalidate_l1_lines(data_offset, num_bytes);
            flush_l2_cache_range(data_offset, num_bytes);
            break;
        case abi::kModeL1FlushFull:
            flush_l1_dcache(0);
            invalidate_l1_lines(data_offset, num_bytes);
            flush_l2_cache_range(data_offset, num_bytes);
            break;
        case abi::kModeL1InvalidateLine:
            invalidate_l1_lines(data_offset, num_bytes);
            flush_l2_cache_range(data_offset, num_bytes);
            break;
        case abi::kModeL1InvalidateFull:
            invalidate_l1_dcache(0);
            flush_l2_cache_range(data_offset, num_bytes);
            break;
        case abi::kModeL2FlushLine:
            for_each_line(data_offset, num_bytes, [](uint32_t line) { flush_l2_cache_line(line); });
            break;
        case abi::kModeL2FlushRange:
        case abi::kModeProducer: flush_l2_cache_range(data_offset, num_bytes); break;
        case abi::kModeL2FlushFull: flush_l2_cache_full(); break;
        case abi::kModeL2InvalidateLine:
            flush_l1_dcache(0);
            for_each_line(data_offset, num_bytes, [](uint32_t line) { invalidate_l2_cache_line(line); });
            flush_l2_cache_range(data_offset, num_bytes);
            break;
        default: break;
    }
}

// Word i of pass 1 is value + i; pass 0 writes its complement, so a lost pass-1 store shows.
static uint32_t pass_word(uint32_t pass, uint32_t value, uint32_t word) {
    return pass == 0 ? ~(value + word) : value + word;
}

static void run_write_path(uint32_t mode, uint32_t data_offset, uint32_t num_words, uint32_t value) {
    const uint32_t num_bytes = num_words * sizeof(uint32_t);
    for (uint32_t pass = 0; pass < 2; ++pass) {
        if (mode == abi::kModeWriteUncached1B) {
            volatile uint8_t* bytes = uncached<uint8_t>(data_offset);
            for (uint32_t byte = 0; byte < num_bytes; ++byte) {
                bytes[byte] = static_cast<uint8_t>(pass_word(pass, value, byte / 4) >> (8 * (byte % 4)));
            }
            asm volatile("fence" ::: "memory");
            continue;
        }
        volatile uint64_t* doublewords =
            mode == abi::kModeWriteUncached8B ? uncached<uint64_t>(data_offset) : cached<uint64_t>(data_offset);
        for (uint32_t index = 0; index < num_words / 2; ++index) {
            doublewords[index] = (static_cast<uint64_t>(pass_word(pass, value, 2 * index + 1)) << 32) |
                                 pass_word(pass, value, 2 * index);
        }
        if (mode == abi::kModeWriteUncached8B) {
            asm volatile("fence" ::: "memory");
        } else if (mode == abi::kModeWriteCachedRangeFlush) {
            flush_l2_cache_range(data_offset, num_bytes);
        } else {
            volatile uint64_t* flush_register = reinterpret_cast<volatile uint64_t*>(L2_FLUSH_ADDR);
            for (uint32_t line = data_offset & ~(abi::kLineBytes - 1); line < data_offset + num_bytes;
                 line += abi::kLineBytes) {
                *flush_register = line;
            }
        }
    }
    if (mode == abi::kModeWriteCachedFastFlush) {
        asm volatile("fence" ::: "memory");
    }
}

static bool wait_for_flag(uint32_t flag_offset, uint32_t flag_value, uint64_t budget_cycles) {
    uint32_t wait_cycles = 0;
    return cce_test::poll_bounded(
        [flag_offset, flag_value] { return *uncached<uint32_t>(flag_offset) == flag_value; },
        budget_cycles,
        wait_cycles);
}

static bool args_valid(uint32_t mode, uint32_t data_offset, uint32_t num_words, uint32_t flag_offset, uint64_t budget) {
    if (mode >= abi::kNumModes || num_words == 0 || num_words > abi::kMaxWords) {
        return false;
    }
    if (is_write_path_mode(mode)) {
        return data_offset % 8 == 0 && num_words % 2 == 0;
    }
    const bool whole_lines = data_offset % abi::kLineBytes == 0 && num_words % (abi::kLineBytes / 4) == 0;
    if (mode == abi::kModeProducer || mode == abi::kModeConsumer) {
        return whole_lines && flag_offset % abi::kLineBytes == 0 && budget != 0;
    }
    return whole_lines;
}

void kernel_main() {
    const uint32_t mode = get_arg_val<uint32_t>(abi::kArgMode);
    const uint32_t status_offset = get_arg_val<uint32_t>(abi::kArgStatusOffset);
    const uint32_t data_offset = get_arg_val<uint32_t>(abi::kArgDataOffset);
    const uint32_t num_words = get_arg_val<uint32_t>(abi::kArgNumWords);
    const uint32_t value = get_arg_val<uint32_t>(abi::kArgValue);
    const uint32_t old_value = get_arg_val<uint32_t>(abi::kArgOldValue);
    const uint32_t flag_offset = get_arg_val<uint32_t>(abi::kArgFlagOffset);
    const uint64_t budget_cycles = arg_u64(abi::kArgPollBudgetLo, abi::kArgPollBudgetHi);
    const uint32_t num_bytes = num_words * sizeof(uint32_t);

    uint32_t code = abi::kCodeDone;
    uint32_t cached_before = 0;
    uint32_t uncached_before = 0;
    uint32_t warm_old = 0;
    uint32_t stale_before = 0;
    uint32_t flag_ready = 0;
    uint32_t cycles = 0;
    Check check;
    if (!args_valid(mode, data_offset, num_words, flag_offset, budget_cycles)) {
        code = abi::kCodeBadArgs;
    } else {
        const uint64_t start = read_cycle_counter();
        if (mode == abi::kModeProducer) {
            flag_ready = wait_for_flag(flag_offset, abi::kFlagWarmed, budget_cycles) ? 1 : 0;
        }
        if (mode == abi::kModeProducer && flag_ready == 0) {
            code = abi::kCodeFlagTimeout;
        } else if (is_store_mode(mode)) {
            if (mode == abi::kModeL1InvalidateFull) {
                flush_l1_dcache(0);
            }
            store_words(cached<uint32_t>(data_offset), num_words, value);
            asm volatile("fence rw, rw" ::: "memory");
            cached_before = check_words(cached<uint32_t>(data_offset), num_words, value).matches;
            uncached_before = check_words(uncached<uint32_t>(data_offset), num_words, value).matches;
            run_store_mode(mode, data_offset, num_bytes);
            if (mode == abi::kModeProducer) {
                asm volatile("fence w, w" ::: "memory");
                *uncached<uint32_t>(flag_offset) = abi::kFlagReady;
            }
            check = check_words(uncached<uint32_t>(data_offset), num_words, value);
            if (mode == abi::kModeNoFlush) {
                invalidate_lines(data_offset, num_bytes);
            }
        } else if (is_fresh_read_mode(mode) || mode == abi::kModeConsumer) {
            warm_old = check_words(cached<uint32_t>(data_offset), num_words, old_value).matches;
            if (mode == abi::kModeConsumer) {
                asm volatile("fence r, w" ::: "memory");
                *uncached<uint32_t>(flag_offset) = abi::kFlagWarmed;
                flag_ready = wait_for_flag(flag_offset, abi::kFlagReady, budget_cycles) ? 1 : 0;
                asm volatile("fence r, rw" ::: "memory");
            } else {
                store_words(uncached<uint32_t>(data_offset), num_words, value);
                asm volatile("fence rw, rw" ::: "memory");
                flag_ready = 1;
            }
            if (flag_ready == 0) {
                code = abi::kCodeFlagTimeout;
            } else {
                stale_before = check_words(cached<uint32_t>(data_offset), num_words, old_value).matches;
                if (mode == abi::kModeL2InvalidateFreshRead) {
                    for_each_line(data_offset, num_bytes, [](uint32_t line) { invalidate_l2_cache_line(line); });
                } else {
                    invalidate_lines(data_offset, num_bytes);
                }
                check = check_words(cached<uint32_t>(data_offset), num_words, value);
            }
        } else {
            run_write_path(mode, data_offset, num_words, value);
            check = check_words(uncached<uint32_t>(data_offset), num_words, value);
        }
        cycles = static_cast<uint32_t>(read_cycle_counter() - start);
    }

    volatile uint32_t* status = uncached<uint32_t>(status_offset);
    status[abi::kStatusCode] = code;
    status[abi::kStatusUncachedBefore] = uncached_before;
    status[abi::kStatusStaleBefore] = stale_before;
    status[abi::kStatusMatches] = check.matches;
    status[abi::kStatusFirstMismatchWord] = check.first_mismatch_word;
    status[abi::kStatusObserved] = check.observed;
    status[abi::kStatusFlagReady] = flag_ready;
    status[abi::kStatusCycles] = cycles;
    status[abi::kStatusCachedBefore] = cached_before;
    status[abi::kStatusWarmOld] = warm_old;
    asm volatile("fence w, w" ::: "memory");
    status[abi::kStatusMagic] = abi::kDoneMagic;
}
