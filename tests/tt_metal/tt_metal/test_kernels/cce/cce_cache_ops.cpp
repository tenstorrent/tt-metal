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

template <typename T>
static volatile T* spa(uint32_t cce_index, uint32_t offset) {
    return reinterpret_cast<volatile T*>(experimental::cce_sram_address(cce_index, offset));
}

static bool wait_for_value(const volatile uint32_t* word, uint32_t value, uint64_t budget_cycles) {
    uint32_t wait_cycles = 0;
    return cce_test::poll_bounded([word, value] { return *word == value; }, budget_cycles, wait_cycles);
}

static bool all_new(const volatile uint32_t* words, uint32_t num_words, uint32_t value) {
    return check_words(words, num_words, value).matches == num_words;
}

// Every value of the status line. The MM-8 modes fill the fields they describe; the MM41 modes add kStatusPasses.
struct Report {
    uint32_t code = abi::kCodeDone;
    uint32_t cached_before = 0;
    uint32_t uncached_before = 0;
    uint32_t warm_old = 0;
    uint32_t stale_before = 0;
    uint32_t flag_ready = 0;
    uint32_t cycles = 0;
    uint32_t register_a = 0;
    uint32_t register_b = 0;
    uint32_t register_c = 0;
    uint32_t passes = 0;
    Check check;
};

struct Args {
    uint32_t mode = 0;
    uint32_t status_offset = 0;
    uint32_t data_offset = 0;
    uint32_t num_words = 0;
    uint32_t value = 0;
    uint32_t old_value = 0;
    uint32_t flag_offset = 0;
    uint64_t budget_cycles = 0;
    uint32_t stride_bytes = 0;
    uint32_t count = 0;
    uint32_t aux_offset = 0;
    uint32_t seed = 0;
    uint32_t pattern = 0;
    uint32_t passes = 0;
    uint32_t trace_offset = 0;
    uint32_t hart_slot = 0;
    uint32_t num_harts = 0;
    uint32_t other_offset = 0;
    uint32_t other_words = 0;
    uint32_t other_value = 0;
    uint32_t target_cce = 0;
    uint32_t steps = 0;
    uint32_t data_window_cycles = 0;
    uint32_t peer_status_offset = 0;
    uint32_t peer_magic = 0;

    uint32_t num_bytes() const { return num_words * sizeof(uint32_t); }
};

static Args read_args() {
    Args args;
    args.mode = get_arg_val<uint32_t>(abi::kArgMode);
    args.status_offset = get_arg_val<uint32_t>(abi::kArgStatusOffset);
    args.data_offset = get_arg_val<uint32_t>(abi::kArgDataOffset);
    args.num_words = get_arg_val<uint32_t>(abi::kArgNumWords);
    args.value = get_arg_val<uint32_t>(abi::kArgValue);
    args.old_value = get_arg_val<uint32_t>(abi::kArgOldValue);
    args.flag_offset = get_arg_val<uint32_t>(abi::kArgFlagOffset);
    args.budget_cycles = arg_u64(abi::kArgPollBudgetLo, abi::kArgPollBudgetHi);
    args.stride_bytes = get_arg_val<uint32_t>(abi::kArgStrideBytes);
    args.count = get_arg_val<uint32_t>(abi::kArgCount);
    args.aux_offset = get_arg_val<uint32_t>(abi::kArgAuxOffset);
    args.seed = get_arg_val<uint32_t>(abi::kArgSeed);
    args.pattern = get_arg_val<uint32_t>(abi::kArgPattern);
    args.passes = get_arg_val<uint32_t>(abi::kArgPasses);
    args.trace_offset = get_arg_val<uint32_t>(abi::kArgTraceOffset);
    args.hart_slot = get_arg_val<uint32_t>(abi::kArgHartSlot);
    args.num_harts = get_arg_val<uint32_t>(abi::kArgNumHarts);
    args.other_offset = get_arg_val<uint32_t>(abi::kArgOtherOffset);
    args.other_words = get_arg_val<uint32_t>(abi::kArgOtherWords);
    args.other_value = get_arg_val<uint32_t>(abi::kArgOtherValue);
    args.target_cce = get_arg_val<uint32_t>(abi::kArgTargetCce);
    args.steps = get_arg_val<uint32_t>(abi::kArgSteps);
    args.data_window_cycles = get_arg_val<uint32_t>(abi::kArgDataWindowCycles);
    args.peer_status_offset = get_arg_val<uint32_t>(abi::kArgPeerStatusOffset);
    args.peer_magic = get_arg_val<uint32_t>(abi::kArgPeerMagic);
    return args;
}

// The clean is cflush.d.l1 x0 then FLUSH64 of every cacheable TL1 line, as flush_l2_cache_full(). From the clean
// through the discard nothing is stored to the stack, so the discard drops only the data stores.
static uint32_t clean_store_and_discard(uint64_t data_address, uint32_t num_words, uint32_t value) {
    const uint64_t end_address = data_address + num_words * sizeof(uint32_t);
    uint64_t matches = 0;
    uint64_t pointer = 0;
    uint64_t expected = 0;
    uint64_t scratch = 0;
    asm volatile(
        "tt.cache.cflush.d.l1 x0\n\t"
        "fence\n\t"
        "li %[pointer], 0\n\t"
        "9:\n\t"
        "sd %[pointer], 0(%[flush_register])\n\t"
        "addi %[pointer], %[pointer], %[line_bytes]\n\t"
        "bltu %[pointer], %[cacheable_end], 9b\n\t"
        "fence\n\t"
        "lw %[scratch], 0(%[drain_address])\n\t"
        "fence\n\t"
        "mv %[pointer], %[data]\n\t"
        "addiw %[expected], %[value], 0\n\t"
        "1:\n\t"
        "sw %[expected], 0(%[pointer])\n\t"
        "addiw %[expected], %[expected], 1\n\t"
        "addi %[pointer], %[pointer], 4\n\t"
        "bltu %[pointer], %[end], 1b\n\t"
        "fence\n\t"
        "mv %[pointer], %[data]\n\t"
        "addiw %[expected], %[value], 0\n\t"
        "li %[matches], 0\n\t"
        "2:\n\t"
        "lw %[scratch], 0(%[pointer])\n\t"
        "bne %[scratch], %[expected], 3f\n\t"
        "addi %[matches], %[matches], 1\n\t"
        "3:\n\t"
        "addiw %[expected], %[expected], 1\n\t"
        "addi %[pointer], %[pointer], 4\n\t"
        "bltu %[pointer], %[end], 2b\n\t"
        "tt.cache.cdiscard.d.l1 x0\n\t"
        "fence\n\t"
        : [matches] "=&r"(matches), [pointer] "=&r"(pointer), [expected] "=&r"(expected), [scratch] "=&r"(scratch)
        : [data] "r"(data_address),
          [end] "r"(end_address),
          [value] "r"(value),
          [flush_register] "r"(L2_FLUSH_ADDR),
          [cacheable_end] "r"(static_cast<uint64_t>(MEMORY_PORT_CACHEABLE_MEM_PORT_MEM_SIZE)),
          [drain_address] "r"(MEM_PORT_NONCACHEABLE_BASE_ADDR),
          [line_bytes] "i"(L2_CACHE_LINE_SIZE)
        : "memory");
    return static_cast<uint32_t>(matches);
}

struct WipeResult {
    uint32_t warm_old = 0;
    uint32_t stale_before = 0;
    bool wiped = false;
};

// The clean is as clean_store_and_discard's. From the clean through the L1 D$ discard nothing is stored to the stack,
// so the wipe and the discard drop no live stack line. FULLINVALIDATE takes 0xFF: every hart's ready bit, written by
// this one hart.
static WipeResult clean_and_wipe(
    uint64_t data_address, uint32_t num_words, uint32_t value, uint32_t old_value, uint64_t budget_cycles) {
    const uint64_t end_address = data_address + num_words * sizeof(uint32_t);
    uint64_t warm_old = 0;
    uint64_t stale_before = 0;
    uint64_t wiped = 0;
    uint64_t pointer = 0;
    uint64_t expected = 0;
    uint64_t scratch = 0;
    uint64_t start = 0;
    asm volatile(
        "tt.cache.cflush.d.l1 x0\n\t"
        "fence\n\t"
        "li %[pointer], 0\n\t"
        "9:\n\t"
        "sd %[pointer], 0(%[flush_register])\n\t"
        "addi %[pointer], %[pointer], %[line_bytes]\n\t"
        "bltu %[pointer], %[cacheable_end], 9b\n\t"
        "fence\n\t"
        "lw %[scratch], 0(%[drain_address])\n\t"
        "fence\n\t"
        "mv %[pointer], %[data]\n\t"
        "addiw %[expected], %[old_value], 0\n\t"
        "li %[warm_old], 0\n\t"
        "1:\n\t"
        "lw %[scratch], 0(%[pointer])\n\t"
        "bne %[scratch], %[expected], 2f\n\t"
        "addi %[warm_old], %[warm_old], 1\n\t"
        "2:\n\t"
        "addiw %[expected], %[expected], 1\n\t"
        "addi %[pointer], %[pointer], 4\n\t"
        "bltu %[pointer], %[end], 1b\n\t"
        "mv %[pointer], %[data]\n\t"
        "addiw %[expected], %[value], 0\n\t"
        "3:\n\t"
        "add %[scratch], %[pointer], %[uncached_delta]\n\t"
        "sw %[expected], 0(%[scratch])\n\t"
        "addiw %[expected], %[expected], 1\n\t"
        "addi %[pointer], %[pointer], 4\n\t"
        "bltu %[pointer], %[end], 3b\n\t"
        "fence\n\t"
        "mv %[pointer], %[data]\n\t"
        "addiw %[expected], %[old_value], 0\n\t"
        "li %[stale_before], 0\n\t"
        "4:\n\t"
        "lw %[scratch], 0(%[pointer])\n\t"
        "bne %[scratch], %[expected], 5f\n\t"
        "addi %[stale_before], %[stale_before], 1\n\t"
        "5:\n\t"
        "addiw %[expected], %[expected], 1\n\t"
        "addi %[pointer], %[pointer], 4\n\t"
        "bltu %[pointer], %[end], 4b\n\t"
        "fence\n\t"
        "li %[scratch], %[all_harts_mask]\n\t"
        "sd %[scratch], 0(%[wipe_register])\n\t"
        "fence\n\t"
        "li %[wiped], 0\n\t"
        "rdcycle %[start]\n\t"
        "6:\n\t"
        "ld %[scratch], 0(%[wipe_register])\n\t"
        "beqz %[scratch], 7f\n\t"
        "rdcycle %[scratch]\n\t"
        "sub %[scratch], %[scratch], %[start]\n\t"
        "bltu %[scratch], %[budget], 6b\n\t"
        "j 8f\n\t"
        "7:\n\t"
        "li %[wiped], 1\n\t"
        "8:\n\t"
        "tt.cache.cdiscard.d.l1 x0\n\t"
        "fence\n\t"
        : [warm_old] "=&r"(warm_old),
          [stale_before] "=&r"(stale_before),
          [wiped] "=&r"(wiped),
          [pointer] "=&r"(pointer),
          [expected] "=&r"(expected),
          [scratch] "=&r"(scratch),
          [start] "=&r"(start)
        : [data] "r"(data_address),
          [end] "r"(end_address),
          [uncached_delta] "r"(static_cast<uint64_t>(MEM_L1_UNCACHED_BASE)),
          [value] "r"(value),
          [old_value] "r"(old_value),
          [flush_register] "r"(L2_FLUSH_ADDR),
          [cacheable_end] "r"(static_cast<uint64_t>(MEMORY_PORT_CACHEABLE_MEM_PORT_MEM_SIZE)),
          [drain_address] "r"(MEM_PORT_NONCACHEABLE_BASE_ADDR),
          [wipe_register] "r"(L2_FULL_INVALIDATE_ADDR),
          [budget] "r"(budget_cycles),
          [line_bytes] "i"(L2_CACHE_LINE_SIZE),
          [all_harts_mask] "i"((1 << MEM_CCE_LOCAL_HARTS) - 1)
        : "memory");
    return {static_cast<uint32_t>(warm_old), static_cast<uint32_t>(stale_before), wiped != 0};
}

// FLUSH64 has no completion register: an uncached read of the range's last line and a fence, as chippy drains it.
static void drain_flushed_range(uint32_t data_offset, uint32_t num_bytes) {
    (void)*uncached<uint32_t>(data_offset + num_bytes - sizeof(uint32_t));
    asm volatile("fence" ::: "memory");
}

static void run_discard_full(const Args& args, Report& report) {
    report.cached_before = clean_store_and_discard(args.data_offset, args.num_words, args.value);
    volatile uint64_t* flush_register = reinterpret_cast<volatile uint64_t*>(L2_FLUSH_ADDR);
    for_each_line(args.data_offset, args.num_bytes(), [flush_register](uint32_t line) { *flush_register = line; });
    asm volatile("fence" ::: "memory");
    report.check = check_words(uncached<uint32_t>(args.data_offset), args.num_words, args.value);
}

// Drops every line of [offset, offset + num_bytes) from L1 D$ and L2 without write-back, so the next cached access
// reads TL1, which the host has just seeded, and no line cached by an earlier leg or launch carries over.
static void drop_cached_lines(uint32_t offset, uint32_t num_bytes) {
    const uint32_t first_line = offset & ~(abi::kLineBytes - 1);
    invalidate_lines(first_line, offset + num_bytes - first_line);
}

// Words of [offset, offset + num_words) that a cached read sees as other than old_value + i before drop_cached_lines:
// lines an earlier leg or launch left cached (recorded only).
static uint32_t carried_over_words(uint32_t offset, uint32_t num_words, uint32_t old_value) {
    return num_words - check_words(cached<uint32_t>(offset), num_words, old_value).matches;
}

static void run_wipe_full(const Args& args, Report& report) {
    report.cached_before = carried_over_words(args.data_offset, args.num_words, args.old_value);
    drop_cached_lines(args.data_offset, args.num_bytes());
    const WipeResult wipe =
        clean_and_wipe(args.data_offset, args.num_words, args.value, args.old_value, args.budget_cycles);
    report.warm_old = wipe.warm_old;
    report.stale_before = wipe.stale_before;
    report.register_a = wipe.wiped ? 1 : 0;
    report.check = check_words(cached<uint32_t>(args.data_offset), args.num_words, args.value);
    if (!wipe.wiped) {
        report.code = abi::kCodeWipeTimeout;
    }
}

static void run_ordering_producer(const Args& args, Report& report) {
    const volatile uint32_t* ack = uncached<uint32_t>(args.flag_offset + abi::kLineBytes);
    volatile uint32_t* flag = uncached<uint32_t>(args.flag_offset);
    volatile uint32_t* cached_words = cached<uint32_t>(args.data_offset);
    volatile uint64_t* flush_register = reinterpret_cast<volatile uint64_t*>(L2_FLUSH_ADDR);
    uint64_t leg_cycles = 0;
    for (uint32_t pass = 0; pass < args.passes; ++pass) {
        if (!wait_for_value(ack, pass, args.budget_cycles)) {
            report.code = abi::kCodeFlagTimeout;
            break;
        }
        store_words(cached_words, args.num_words, args.value + (pass << 16));
        const uint64_t start = read_cycle_counter();
        switch (args.steps) {
            case abi::kOrderingLegFlushNoFence:
                for_each_line(
                    args.data_offset, args.num_bytes(), [flush_register](uint32_t line) { *flush_register = line; });
                break;
            case abi::kOrderingLegFenceNoFlush: asm volatile("fence rw, rw" ::: "memory"); break;
            case abi::kOrderingLegL1ThenL2:
                for_each_line(args.data_offset, args.num_bytes(), [](uint32_t line) { flush_l1_dcache(line); });
                invalidate_l1_lines(args.data_offset, args.num_bytes());
                flush_l2_cache_range(args.data_offset, args.num_bytes());
                drain_flushed_range(args.data_offset, args.num_bytes());
                break;
            case abi::kOrderingLegFullHierarchy:
                flush_l1_dcache(0);
                invalidate_l1_lines(args.data_offset, args.num_bytes());
                flush_l2_cache_full();
                drain_flushed_range(args.data_offset, args.num_bytes());
                break;
            default: break;
        }
        leg_cycles += read_cycle_counter() - start;
        if (args.steps != abi::kOrderingLegFlushNoFence) {
            asm volatile("fence w, w" ::: "memory");
        }
        *flag = pass + 1;
        report.passes = pass + 1;
    }
    flush_l2_cache_range(args.data_offset, args.num_bytes());
    report.register_a = static_cast<uint32_t>(leg_cycles);
}

static void run_ordering_consumer(const Args& args, Report& report) {
    volatile uint32_t* ack = uncached<uint32_t>(args.flag_offset + abi::kLineBytes);
    const volatile uint32_t* flag = uncached<uint32_t>(args.flag_offset);
    const volatile uint32_t* uncached_words = uncached<uint32_t>(args.data_offset);
    uint32_t complete_at_flag = 0;
    bool incomplete_seen = false;
    for (uint32_t pass = 0; pass < args.passes; ++pass) {
        *ack = pass;
        if (!wait_for_value(flag, pass + 1, args.budget_cycles)) {
            report.code = abi::kCodeFlagTimeout;
            break;
        }
        const uint32_t pass_value = args.value + (pass << 16);
        Check first_read = check_words(uncached_words, args.num_words, pass_value);
        if (first_read.matches == args.num_words) {
            complete_at_flag++;
        } else {
            if (!incomplete_seen) {
                incomplete_seen = true;
                report.check = first_read;
            }
            uint32_t lag_cycles = 0;
            if (cce_test::poll_bounded(
                    [&] { return all_new(uncached_words, args.num_words, pass_value); },
                    args.data_window_cycles,
                    lag_cycles)) {
                report.register_a++;
                report.register_c = lag_cycles > report.register_c ? lag_cycles : report.register_c;
            } else {
                report.register_b++;
            }
        }
        report.passes = pass + 1;
    }
    report.check.matches = complete_at_flag;
}

static void store_partial(volatile uint8_t* line, uint32_t line_value) {
    line[abi::kPartialByteOffset] = static_cast<uint8_t>(line_value);
    *reinterpret_cast<volatile uint16_t*>(line + abi::kPartialHalfOffset) = static_cast<uint16_t>(line_value >> 8);
    *reinterpret_cast<volatile uint32_t*>(line + abi::kPartialWordOffset) = ~line_value;
    *reinterpret_cast<volatile uint64_t*>(line + abi::kPartialDoubleOffset) =
        (static_cast<uint64_t>(line_value) << 32) | (line_value ^ 0x5A5A5A5Au);
}

// The bytes of word `word` that store_partial writes, as a byte mask.
static uint32_t partial_mask(uint32_t word) {
    constexpr uint32_t kRanges[4][2] = {
        {abi::kPartialByteOffset, 1},
        {abi::kPartialHalfOffset, 2},
        {abi::kPartialWordOffset, 4},
        {abi::kPartialDoubleOffset, 8}};
    uint32_t mask = 0;
    for (const auto& range : kRanges) {
        for (uint32_t byte = range[0]; byte < range[0] + range[1]; ++byte) {
            if (byte / 4 == word) {
                mask |= 0xFFu << (8 * (byte % 4));
            }
        }
    }
    return mask;
}

struct PartialCheck {
    uint32_t partial_landed_lines = 0;
    uint32_t exact_lines = 0;
    uint32_t untouched_changed_lines = 0;
    Check first_mismatch;
};

static PartialCheck check_partial_lines(const Args& args) {
    constexpr uint32_t kWordsPerLine = abi::kLineBytes / sizeof(uint32_t);
    PartialCheck result;
    bool mismatch_seen = false;
    const volatile uint32_t* observed_words = uncached<uint32_t>(args.data_offset);
    for (uint32_t line = 0; line < args.num_words / kWordsPerLine; ++line) {
        alignas(8) uint32_t expected[kWordsPerLine];
        for (uint32_t word = 0; word < kWordsPerLine; ++word) {
            expected[word] = args.old_value + line * kWordsPerLine + word;
        }
        store_partial(reinterpret_cast<volatile uint8_t*>(expected), args.value + line);
        bool partial_landed = true;
        bool exact = true;
        bool untouched_changed = false;
        for (uint32_t word = 0; word < kWordsPerLine; ++word) {
            const uint32_t observed = observed_words[line * kWordsPerLine + word];
            const uint32_t mask = partial_mask(word);
            partial_landed = partial_landed && ((observed ^ expected[word]) & mask) == 0;
            untouched_changed = untouched_changed || ((observed ^ expected[word]) & ~mask) != 0;
            if (observed != expected[word]) {
                exact = false;
                if (!mismatch_seen) {
                    mismatch_seen = true;
                    result.first_mismatch.first_mismatch_word = line * kWordsPerLine + word;
                    result.first_mismatch.observed = observed;
                }
            }
        }
        result.partial_landed_lines += partial_landed ? 1 : 0;
        result.exact_lines += exact ? 1 : 0;
        result.untouched_changed_lines += untouched_changed ? 1 : 0;
    }
    return result;
}

static void run_partial_line_evict(const Args& args, Report& report) {
    report.cached_before = carried_over_words(args.data_offset, args.num_words, args.old_value);
    drop_cached_lines(args.data_offset, args.num_bytes());
    for (uint32_t line = 0; line < args.num_bytes() / abi::kLineBytes; ++line) {
        store_partial(cached<uint8_t>(args.data_offset + line * abi::kLineBytes), args.value + line);
    }
    asm volatile("fence rw, rw" ::: "memory");
    for (uint32_t line = 0; line < args.count; ++line) {
        (void)*cached<uint32_t>(args.aux_offset + line * args.stride_bytes);
    }
    asm volatile("fence rw, rw" ::: "memory");
    const PartialCheck before_flush = check_partial_lines(args);
    report.register_a = before_flush.partial_landed_lines;
    report.register_b = before_flush.untouched_changed_lines;
    flush_l2_cache_range(args.data_offset, args.num_bytes());
    const PartialCheck after_flush = check_partial_lines(args);
    report.register_c = after_flush.untouched_changed_lines;
    report.check = after_flush.first_mismatch;
    report.check.matches = after_flush.exact_lines;
}

static void run_l1_flush_target(const Args& args, Report& report) {
    volatile uint32_t* cached_words = cached<uint32_t>(args.data_offset);
    volatile uint32_t* uncached_words = uncached<uint32_t>(args.data_offset);
    store_words(cached_words, args.num_words, args.value);
    asm volatile("fence rw, rw" ::: "memory");
    report.cached_before = check_words(cached_words, args.num_words, args.value).matches;
    report.uncached_before = check_words(uncached_words, args.num_words, args.value).matches;
    if (args.mode == abi::kModeL1FlushLineTarget) {
        for_each_line(args.data_offset, args.num_bytes(), [](uint32_t line) { flush_l1_dcache(line); });
    } else {
        flush_l1_dcache(0);
    }
    report.register_a = check_words(uncached_words, args.num_words, args.value).matches;
    invalidate_l1_lines(args.data_offset, args.num_bytes());
    flush_l2_cache_range(args.data_offset, args.num_bytes());
    report.check = check_words(uncached_words, args.num_words, args.value);
}

static void run_shared_writer(const Args& args, Report& report) {
    const uint32_t first_line = args.data_offset & ~(abi::kLineBytes - 1);
    const uint32_t line_span_bytes = args.num_bytes() + args.data_offset % abi::kLineBytes;
    invalidate_l1_lines(first_line, line_span_bytes);
    report.flag_ready = wait_for_flag(args.flag_offset, abi::kFlagWarmed, args.budget_cycles) ? 1 : 0;
    if (report.flag_ready == 0) {
        report.code = abi::kCodeFlagTimeout;
        return;
    }
    volatile uint32_t* cached_words = cached<uint32_t>(args.data_offset);
    store_words(cached_words, args.num_words, args.value);
    asm volatile("fence w, w" ::: "memory");
    *uncached<uint32_t>(args.flag_offset) = abi::kFlagReady;
    report.uncached_before = check_words(uncached<uint32_t>(args.data_offset), args.num_words, args.value).matches;
    report.cached_before = check_words(cached_words, args.num_words, args.value).matches;
    for_each_line(first_line, line_span_bytes, [](uint32_t line) { flush_l1_dcache(line); });
    const uint32_t shared_l2_flag_offset = args.flag_offset + 2 * abi::kLineBytes;
    *uncached<uint32_t>(shared_l2_flag_offset) = abi::kFlagReady;
    if (!wait_for_flag(shared_l2_flag_offset, abi::kFlagWarmed, args.budget_cycles)) {
        report.code = abi::kCodeFlagTimeout;
        return;
    }
    flush_l2_cache_range(args.data_offset, args.num_bytes());
    drain_flushed_range(args.data_offset, args.num_bytes());
    asm volatile("fence w, w" ::: "memory");
    *uncached<uint32_t>(args.flag_offset + abi::kLineBytes) = abi::kFlagReady;
}

// The reader's cached read of the region: the writer's words new, every other word old.
static Check check_shared_region(const Args& args) {
    Check check;
    const uint32_t other_end = args.other_offset + args.other_words * sizeof(uint32_t);
    bool mismatch_seen = false;
    for (uint32_t word = 0; word < args.count; ++word) {
        const uint32_t address = args.aux_offset + word * sizeof(uint32_t);
        const bool written = address >= args.other_offset && address < other_end;
        const uint32_t expected =
            written ? args.value + (address - args.other_offset) / sizeof(uint32_t) : args.old_value + word;
        const uint32_t observed = *cached<uint32_t>(address);
        if (observed == expected) {
            check.matches++;
        } else if (!mismatch_seen) {
            mismatch_seen = true;
            check.first_mismatch_word = word;
            check.observed = observed;
        }
    }
    return check;
}

static void run_shared_reader(const Args& args, Report& report) {
    report.cached_before = carried_over_words(args.aux_offset, args.count, args.old_value);
    drop_cached_lines(args.aux_offset, args.count * sizeof(uint32_t));
    report.warm_old = check_words(
                          cached<uint32_t>(args.data_offset),
                          args.num_words,
                          args.old_value + (args.data_offset - args.aux_offset) / sizeof(uint32_t))
                          .matches;
    asm volatile("fence r, w" ::: "memory");
    *uncached<uint32_t>(args.flag_offset) = abi::kFlagWarmed;
    report.flag_ready = wait_for_flag(args.flag_offset, abi::kFlagReady, args.budget_cycles) ? 1 : 0;
    if (report.flag_ready == 0) {
        report.code = abi::kCodeFlagTimeout;
        return;
    }
    asm volatile("fence r, rw" ::: "memory");
    report.register_b = check_shared_region(args).matches;
    report.register_a = check_words(uncached<uint32_t>(args.other_offset), args.other_words, args.value).matches;
    const uint32_t shared_l2_flag_offset = args.flag_offset + 2 * abi::kLineBytes;
    if (!wait_for_flag(shared_l2_flag_offset, abi::kFlagReady, args.budget_cycles)) {
        report.code = abi::kCodeFlagTimeout;
        return;
    }
    asm volatile("fence r, rw" ::: "memory");
    invalidate_l1_lines(args.aux_offset, args.count * sizeof(uint32_t));
    report.register_c = check_shared_region(args).matches;
    asm volatile("fence r, w" ::: "memory");
    *uncached<uint32_t>(shared_l2_flag_offset) = abi::kFlagWarmed;
    if (!wait_for_flag(args.flag_offset + abi::kLineBytes, abi::kFlagReady, args.budget_cycles)) {
        report.code = abi::kCodeFlagTimeout;
        return;
    }
    asm volatile("fence r, rw" ::: "memory");
    invalidate_lines(args.aux_offset, args.count * sizeof(uint32_t));
    report.check = check_shared_region(args);
}

static void run_shared_concurrent_writer(const Args& args, Report& report) {
    drop_cached_lines(args.data_offset + args.hart_slot * args.stride_bytes, args.num_bytes());
    *uncached<uint32_t>(args.flag_offset + args.hart_slot * abi::kLineBytes) = abi::kFlagReady;
    uint32_t wait_cycles = 0;
    const bool every_hart_ready = cce_test::poll_bounded(
        [&args] {
            for (uint32_t slot = 0; slot < args.num_harts; ++slot) {
                if (*uncached<uint32_t>(args.flag_offset + slot * abi::kLineBytes) != abi::kFlagReady) {
                    return false;
                }
            }
            return true;
        },
        args.budget_cycles,
        wait_cycles);
    report.flag_ready = every_hart_ready ? 1 : 0;
    if (report.flag_ready == 0) {
        report.code = abi::kCodeFlagTimeout;
        return;
    }
    const uint32_t own_offset = args.data_offset + args.hart_slot * args.stride_bytes;
    volatile uint32_t* own_words = cached<uint32_t>(own_offset);
    const uint32_t slot_value = args.value + (args.hart_slot << 16);
    for (uint32_t pass = 0; pass < args.passes; ++pass) {
        store_words(own_words, args.num_words, slot_value + (pass << 8));
        report.passes = pass + 1;
    }
    asm volatile("fence" ::: "memory");
    report.cached_before = check_words(own_words, args.num_words, slot_value + ((args.passes - 1) << 8)).matches;
    flush_l2_cache_range(own_offset, args.num_bytes());
}

static void run_external_reader(const Args& args, Report& report) {
    volatile uint32_t* cached_words = cached<uint32_t>(args.data_offset);
    const volatile uint32_t* uncached_words = uncached<uint32_t>(args.data_offset);
    report.cached_before = carried_over_words(args.data_offset, args.num_words, args.old_value);
    drop_cached_lines(args.data_offset, args.num_bytes());
    report.warm_old = check_words(cached_words, args.num_words, args.old_value).matches;
    if (args.flag_offset != 0) {
        asm volatile("fence r, w" ::: "memory");
        *uncached<uint32_t>(args.flag_offset) = abi::kFlagWarmed;
    }
    uint32_t poll_cycles = 0;
    const bool landed = cce_test::poll_bounded(
        [&] { return all_new(uncached_words, args.num_words, args.value); }, args.budget_cycles, poll_cycles);
    report.flag_ready = landed ? 1 : 0;
    report.uncached_before = check_words(uncached_words, args.num_words, args.value).matches;
    if (report.flag_ready == 0) {
        report.code = abi::kCodeDataTimeout;
        return;
    }
    if (args.peer_status_offset != 0 &&
        !wait_for_value(uncached<uint32_t>(args.peer_status_offset), args.peer_magic, args.budget_cycles)) {
        report.code = abi::kCodePeerTimeout;
        return;
    }
    asm volatile("fence r, rw" ::: "memory");
    if ((args.steps & abi::kStepDiscriminate) != 0) {
        report.stale_before = check_words(cached_words, args.num_words, args.old_value).matches;
        if (report.stale_before != 0) {
            invalidate_l1_lines(args.data_offset, args.num_bytes());
            report.register_a = check_words(cached_words, args.num_words, args.value).matches;
            report.register_b = 1;
        }
    }
    if ((args.steps & abi::kStepL2Invalidate) != 0) {
        for_each_line(args.data_offset, args.num_bytes(), [](uint32_t line) { invalidate_l2_cache_line(line); });
    }
    report.check = check_words(cached_words, args.num_words, args.value);
}

static void run_spa_writer(const Args& args, Report& report) {
    if (args.flag_offset != 0) {
        const bool warmed =
            wait_for_value(spa<uint32_t>(args.target_cce, args.flag_offset), abi::kFlagWarmed, args.budget_cycles);
        report.flag_ready = warmed ? 1 : 0;
        if (report.flag_ready == 0) {
            report.code = abi::kCodeFlagTimeout;
            return;
        }
    }
    volatile uint32_t* target_words = spa<uint32_t>(args.target_cce, args.data_offset);
    store_words(target_words, args.num_words, args.value);
    asm volatile("fence" ::: "memory");
    report.check = check_words(target_words, args.num_words, args.value);
}

static void run_spa_reader(const Args& args, Report& report) {
    volatile uint32_t* target_flag = spa<uint32_t>(args.target_cce, args.flag_offset);
    *target_flag = abi::kFlagWarmed;
    asm volatile("fence" ::: "memory");
    report.flag_ready = wait_for_value(target_flag, abi::kFlagReady, args.budget_cycles) ? 1 : 0;
    if (report.flag_ready == 0) {
        report.code = abi::kCodeFlagTimeout;
        return;
    }
    asm volatile("fence r, r" ::: "memory");
    report.check = check_words(spa<uint32_t>(args.target_cce, args.data_offset), args.num_words, args.value);
}

static void run_race_action(const Args& args, uint32_t line) {
    switch (args.steps & abi::kRaceActionMask) {
        case abi::kRaceActionEvict:
            for (uint32_t index = 0; index < args.count; ++index) {
                (void)*cached<uint32_t>(args.aux_offset + index * args.stride_bytes);
            }
            break;
        case abi::kRaceActionFlush: *reinterpret_cast<volatile uint64_t*>(L2_FLUSH_ADDR) = line; break;
        case abi::kRaceActionInvalidate: *reinterpret_cast<volatile uint64_t*>(L2_INVALIDATE_ADDR) = line; break;
        default: break;
    }
}

static void run_dirty_race(const Args& args, Report& report) {
    volatile uint32_t* hart_words = cached<uint32_t>(args.data_offset);
    const volatile uint32_t* external_words = uncached<uint32_t>(args.other_offset);
    const uint32_t line = args.data_offset & ~(abi::kLineBytes - 1);
    report.cached_before = carried_over_words(line, abi::kLineBytes / sizeof(uint32_t), args.old_value);
    drop_cached_lines(line, abi::kLineBytes);
    store_words(hart_words, args.num_words, args.value);
    asm volatile("fence w, w" ::: "memory");
    *uncached<uint32_t>(args.flag_offset) = abi::kFlagReady;
    const cce_test::WaitDeadline deadline(args.budget_cycles);
    bool external_seen = false;
    while (!external_seen && report.passes < args.passes && !deadline.expired()) {
        if ((args.steps & abi::kRaceRedirty) != 0) {
            store_words(hart_words, args.num_words, args.value);
        }
        run_race_action(args, line);
        report.passes++;
        external_seen = all_new(external_words, args.other_words, args.other_value);
    }
    asm volatile("fence rw, rw" ::: "memory");
    report.flag_ready = external_seen ? 1 : 0;
    if (!external_seen) {
        report.code = abi::kCodeRaceUnresolved;
    }
    report.register_a = check_words(hart_words, args.num_words, args.value).matches;
    report.register_b = check_words(cached<uint32_t>(args.other_offset), args.other_words, args.other_value).matches;
    flush_l2_cache_line(line);
}

// Visits every line of the data once, in the order of args.pattern.
template <typename Visit>
static void for_each_pattern_line(const Args& args, Visit visit) {
    const uint32_t num_lines = args.num_bytes() / abi::kLineBytes;
    if (args.pattern == abi::kThrashSequential) {
        for (uint32_t line = 0; line < num_lines; ++line) {
            visit(line);
        }
    } else if (args.pattern == abi::kThrashConflictStride) {
        const uint32_t stride_lines = args.stride_bytes / abi::kLineBytes;
        for (uint32_t first = 0; first < stride_lines && first < num_lines; ++first) {
            for (uint32_t line = first; line < num_lines; line += stride_lines) {
                visit(line);
            }
        }
    } else {
        uint32_t modulus = 1;
        while (modulus < num_lines && modulus < (1u << 31)) {
            modulus <<= 1;
        }
        // x -> 5x + c mod a power of two has full period for odd c (Hull-Dobell), so every line is visited once.
        const uint32_t increment = args.seed | 1;
        uint32_t line = (args.seed >> 8) & (modulus - 1);
        for (uint32_t step = 0; step < modulus; ++step) {
            line = (5 * line + increment) & (modulus - 1);
            if (line < num_lines) {
                visit(line);
            }
        }
    }
}

static void run_thrash(const Args& args, Report& report) {
    constexpr uint32_t kWordsPerLine = abi::kLineBytes / sizeof(uint32_t);
    volatile uint32_t* cached_words = cached<uint32_t>(args.data_offset);
    uint32_t pass_cycles[abi::kMaxPasses] = {};
    bool mismatch_seen = false;
    for (uint32_t pass = 0; pass < args.passes; ++pass) {
        const uint64_t start = read_cycle_counter();
        for_each_pattern_line(args, [&](uint32_t line) {
            for (uint32_t word = line * kWordsPerLine; word < (line + 1) * kWordsPerLine; ++word) {
                if (pass != 0) {
                    const uint32_t observed = cached_words[word];
                    if (observed != abi::thrash_word(args.value, args.seed, pass - 1, word)) {
                        report.register_a++;
                        if (!mismatch_seen) {
                            mismatch_seen = true;
                            report.check.first_mismatch_word = word;
                            report.check.observed = observed;
                        }
                    }
                }
                cached_words[word] = abi::thrash_word(args.value, args.seed, pass, word);
            }
        });
        pass_cycles[pass] = static_cast<uint32_t>(read_cycle_counter() - start);
        report.passes = pass + 1;
    }
    flush_l2_cache_range(args.data_offset, args.num_bytes());
    const volatile uint32_t* uncached_words = uncached<uint32_t>(args.data_offset);
    for (uint32_t word = 0; word < args.num_words; ++word) {
        const uint32_t observed = uncached_words[word];
        report.register_b ^= observed;
        if (observed == abi::thrash_word(args.value, args.seed, args.passes - 1, word)) {
            report.check.matches++;
        }
    }
    if (args.trace_offset != 0) {
        volatile uint32_t* trace = uncached<uint32_t>(args.trace_offset);
        for (uint32_t pass = 0; pass < args.passes; ++pass) {
            trace[pass] = pass_cycles[pass];
        }
    }
}

static void run_churn_consumer(const Args& args, Report& report) {
    volatile uint32_t* cached_words = cached<uint32_t>(args.data_offset);
    const volatile uint32_t* uncached_words = uncached<uint32_t>(args.data_offset);
    report.cached_before = carried_over_words(args.data_offset, args.num_words, args.old_value);
    drop_cached_lines(args.data_offset, args.num_bytes());
    report.warm_old = check_words(cached_words, args.num_words, args.old_value).matches;
    const uint64_t hit_start = read_cycle_counter();
    (void)check_words(cached_words, args.num_words, args.old_value);
    report.register_c = static_cast<uint32_t>(read_cycle_counter() - hit_start);
    bool incomplete_seen = false;
    for (uint32_t round = 0; round < args.passes; ++round) {
        const uint32_t round_value = abi::churn_word(args.value, round, 0);
        asm volatile("fence rw, w" ::: "memory");
        *uncached<uint32_t>(args.flag_offset) = round + 1;
        uint32_t poll_cycles = 0;
        if (!cce_test::poll_bounded(
                [&] { return all_new(uncached_words, args.num_words, round_value); },
                args.budget_cycles,
                poll_cycles)) {
            report.code = abi::kCodeDataTimeout;
            report.uncached_before = check_words(uncached_words, args.num_words, round_value).matches;
            break;
        }
        asm volatile("fence r, rw" ::: "memory");
        if ((args.steps & abi::kStepL2Invalidate) != 0) {
            for_each_line(args.data_offset, args.num_bytes(), [](uint32_t line) { invalidate_l2_cache_line(line); });
        }
        const uint64_t refill_start = read_cycle_counter();
        const Check refill = check_words(cached_words, args.num_words, round_value);
        report.register_a += static_cast<uint32_t>(read_cycle_counter() - refill_start);
        report.stale_before += args.num_words - refill.matches;
        const uint64_t reread_start = read_cycle_counter();
        const Check reread = check_words(cached_words, args.num_words, round_value);
        report.register_b += static_cast<uint32_t>(read_cycle_counter() - reread_start);
        if (reread.matches == args.num_words) {
            report.check.matches++;
        } else if (!incomplete_seen) {
            incomplete_seen = true;
            report.check.first_mismatch_word = reread.first_mismatch_word;
            report.check.observed = reread.observed;
        }
        report.passes = round + 1;
    }
}

// The test TL1 area: from the end of the harts' kernel text (HAL DRAM UNRESERVED before its alignment) to the end of
// the hart-cacheable port. Every region a mode reads or writes lies inside it, in its own or the target CCE's TL1.
inline constexpr uint64_t kTestAreaBegin =
    MEM_CCE_KERNEL_BASE + static_cast<uint64_t>(MEM_CCE_KERNEL_SIZE) * MEM_CCE_LOCAL_HARTS;
inline constexpr uint64_t kTestAreaEnd = MEMORY_PORT_CACHEABLE_MEM_PORT_MEM_SIZE;

static bool in_test_area(uint64_t offset, uint64_t num_bytes) {
    return offset >= kTestAreaBegin && offset <= kTestAreaEnd && num_bytes <= kTestAreaEnd - offset;
}

// `num_lines` whole lines at a nonzero, 64 B aligned offset.
static bool lines_in_test_area(uint64_t offset, uint64_t num_lines) {
    return offset != 0 && offset % abi::kLineBytes == 0 && in_test_area(offset, num_lines * abi::kLineBytes);
}

// The lines at offset + i * stride_bytes for i < count.
static bool strided_lines_in_test_area(uint64_t offset, uint64_t count, uint64_t stride_bytes) {
    return count == 0 ||
           (offset % abi::kLineBytes == 0 && in_test_area(offset, (count - 1) * stride_bytes + abi::kLineBytes));
}

static bool status_line_valid(uint32_t status_offset) { return lines_in_test_area(status_offset, 1); }

static bool mm41_args_valid(const Args& args) {
    const uint64_t data_bytes = static_cast<uint64_t>(args.num_words) * sizeof(uint32_t);
    const bool words = args.data_offset % sizeof(uint32_t) == 0 && args.num_words != 0 &&
                       args.num_words <= abi::kMaxWords && in_test_area(args.data_offset, data_bytes);
    const bool whole_lines =
        words && args.data_offset % abi::kLineBytes == 0 && args.num_words % (abi::kLineBytes / 4) == 0;
    const bool budget = args.budget_cycles != 0;
    const bool optional_flag = args.flag_offset == 0 || lines_in_test_area(args.flag_offset, 1);
    switch (args.mode) {
        case abi::kModeDiscardFull:
        case abi::kModeWipeFull:
        case abi::kModeL1FlushLineTarget:
        case abi::kModeL1FlushFullTarget: return whole_lines && budget;
        case abi::kModeOrderingProducer:
        case abi::kModeOrderingConsumer:
            return whole_lines && budget && lines_in_test_area(args.flag_offset, 2) && args.passes != 0 &&
                   args.passes <= abi::kMaxPasses && args.steps <= abi::kOrderingLegFullHierarchy &&
                   args.data_window_cycles != 0;
        case abi::kModePartialLineEvict:
            return whole_lines && args.stride_bytes % abi::kLineBytes == 0 && args.stride_bytes != 0 &&
                   args.count != 0 && args.count <= abi::kMaxSweepLines &&
                   strided_lines_in_test_area(args.aux_offset, args.count, args.stride_bytes);
        case abi::kModeSharedWriter: return words && budget && lines_in_test_area(args.flag_offset, 3);
        case abi::kModeSharedReader: {
            const uint64_t region_end =
                static_cast<uint64_t>(args.aux_offset) + static_cast<uint64_t>(args.count) * sizeof(uint32_t);
            const uint64_t other_end =
                static_cast<uint64_t>(args.other_offset) + static_cast<uint64_t>(args.other_words) * sizeof(uint32_t);
            return words && budget && lines_in_test_area(args.flag_offset, 3) && args.count != 0 &&
                   args.count % (abi::kLineBytes / 4) == 0 && args.count <= abi::kMaxWords &&
                   lines_in_test_area(args.aux_offset, args.count / (abi::kLineBytes / 4)) &&
                   args.data_offset >= args.aux_offset && args.data_offset + data_bytes <= region_end &&
                   args.other_offset % sizeof(uint32_t) == 0 && args.other_words != 0 &&
                   args.other_offset >= args.aux_offset && other_end <= region_end;
        }
        case abi::kModeSharedConcurrentWriter:
            return words && budget && args.num_harts != 0 && args.num_harts <= MEM_CCE_LOCAL_HARTS &&
                   args.hart_slot < args.num_harts && lines_in_test_area(args.flag_offset, args.num_harts) &&
                   args.passes != 0 && args.passes <= 256 && args.num_words <= 256 &&
                   args.stride_bytes % sizeof(uint32_t) == 0 &&
                   in_test_area(
                       args.data_offset, static_cast<uint64_t>(args.num_harts - 1) * args.stride_bytes + data_bytes);
        case abi::kModeExternalReader:
            return whole_lines && budget && optional_flag &&
                   (args.peer_status_offset == 0 || lines_in_test_area(args.peer_status_offset, 1));
        case abi::kModeChurnConsumer:
            return whole_lines && budget && lines_in_test_area(args.flag_offset, 1) && args.passes != 0 &&
                   args.passes <= abi::kMaxPasses;
        case abi::kModeSpaWriter: return words && budget && optional_flag;
        case abi::kModeSpaReader: return words && budget && lines_in_test_area(args.flag_offset, 1);
        case abi::kModeDirtyRace:
            return words && budget && lines_in_test_area(args.flag_offset, 1) && args.passes != 0 &&
                   args.passes <= abi::kMaxRaceActions && args.other_words != 0 &&
                   args.other_offset % sizeof(uint32_t) == 0 &&
                   cce_test::on_one_line(args.data_offset, args.data_offset + args.num_bytes() - 1) &&
                   cce_test::on_one_line(args.data_offset, args.other_offset) &&
                   cce_test::on_one_line(
                       args.data_offset, args.other_offset + args.other_words * sizeof(uint32_t) - 1) &&
                   (args.steps & abi::kRaceActionMask) <= abi::kRaceActionInvalidate &&
                   args.count <= abi::kMaxEvictCount && args.stride_bytes % abi::kLineBytes == 0 &&
                   strided_lines_in_test_area(args.aux_offset, args.count, args.stride_bytes);
        case abi::kModeThrash:
            return whole_lines && args.passes != 0 && args.passes <= abi::kMaxPasses &&
                   args.pattern <= abi::kThrashRandom &&
                   (args.trace_offset == 0 || lines_in_test_area(args.trace_offset, 1)) &&
                   (args.pattern != abi::kThrashConflictStride ||
                    (args.stride_bytes % abi::kLineBytes == 0 && args.stride_bytes != 0));
        default: return false;
    }
}

static void run_mm41_mode(const Args& args, Report& report) {
    switch (args.mode) {
        case abi::kModeDiscardFull: run_discard_full(args, report); break;
        case abi::kModeWipeFull: run_wipe_full(args, report); break;
        case abi::kModeOrderingProducer: run_ordering_producer(args, report); break;
        case abi::kModeOrderingConsumer: run_ordering_consumer(args, report); break;
        case abi::kModePartialLineEvict: run_partial_line_evict(args, report); break;
        case abi::kModeL1FlushLineTarget:
        case abi::kModeL1FlushFullTarget: run_l1_flush_target(args, report); break;
        case abi::kModeSharedWriter: run_shared_writer(args, report); break;
        case abi::kModeSharedReader: run_shared_reader(args, report); break;
        case abi::kModeSharedConcurrentWriter: run_shared_concurrent_writer(args, report); break;
        case abi::kModeExternalReader: run_external_reader(args, report); break;
        case abi::kModeSpaWriter: run_spa_writer(args, report); break;
        case abi::kModeSpaReader: run_spa_reader(args, report); break;
        case abi::kModeDirtyRace: run_dirty_race(args, report); break;
        case abi::kModeThrash: run_thrash(args, report); break;
        case abi::kModeChurnConsumer: run_churn_consumer(args, report); break;
        default: report.code = abi::kCodeBadArgs; break;
    }
}

// The MM41 modes (kModeDiscardFull and later): reads every arg, runs the mode and writes the status line.
// Out of line so that kernel_main's frame, and the stack traffic of the MM-8 store modes between their stores and
// their cache operation, stay as they were before the MM41 modes: extra stack lines in the 4 KiB L1 D$ can evict the
// dirty data lines that kModeL1InvalidateLine expects the discard to drop.
static __attribute__((noinline)) void run_mm41_kernel() {
    const Args args = read_args();
    if (!status_line_valid(args.status_offset)) {
        return;
    }
    Report report;
    if (!mm41_args_valid(args)) {
        report.code = abi::kCodeBadArgs;
    } else {
        const uint64_t start = read_cycle_counter();
        run_mm41_mode(args, report);
        report.cycles = static_cast<uint32_t>(read_cycle_counter() - start);
    }

    volatile uint32_t* status = uncached<uint32_t>(args.status_offset);
    status[abi::kStatusCode] = report.code;
    status[abi::kStatusUncachedBefore] = report.uncached_before;
    status[abi::kStatusStaleBefore] = report.stale_before;
    status[abi::kStatusMatches] = report.check.matches;
    status[abi::kStatusFirstMismatchWord] = report.check.first_mismatch_word;
    status[abi::kStatusObserved] = report.check.observed;
    status[abi::kStatusFlagReady] = report.flag_ready;
    status[abi::kStatusCycles] = report.cycles;
    status[abi::kStatusCachedBefore] = report.cached_before;
    status[abi::kStatusWarmOld] = report.warm_old;
    status[abi::kStatusRegisterA] = report.register_a;
    status[abi::kStatusRegisterB] = report.register_b;
    status[abi::kStatusRegisterC] = report.register_c;
    status[abi::kStatusPasses] = report.passes;
    asm volatile("fence w, w" ::: "memory");
    status[abi::kStatusMagic] = abi::kDoneMagic;
}

static bool args_valid(uint32_t mode, uint32_t data_offset, uint32_t num_words, uint32_t flag_offset, uint64_t budget) {
    if (mode == abi::kModeNull || mode == abi::kModeReadConfig) {
        return true;
    }
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
    if (mode >= abi::kModeDiscardFull) {
        run_mm41_kernel();
        return;
    }
    const uint32_t status_offset = get_arg_val<uint32_t>(abi::kArgStatusOffset);
    if (!status_line_valid(status_offset)) {
        return;
    }
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
    uint32_t register_a = 0;
    uint32_t register_b = 0;
    uint32_t register_c = 0;
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
        } else if (mode == abi::kModeNull) {
        } else if (mode == abi::kModeReadConfig) {
            register_a = *reinterpret_cast<volatile uint32_t*>(TT_CLUSTER_CTRL_NOC_SNOOP_TL_MASTER_CFG_REG_ADDR);
            register_b = *reinterpret_cast<volatile uint32_t*>(TT_CACHE_CONTROLLER_CONFIGURATION_REG_ADDR);
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
    status[abi::kStatusRegisterA] = register_a;
    status[abi::kStatusRegisterB] = register_b;
    status[abi::kStatusRegisterC] = register_c;
    status[abi::kStatusPasses] = 0;
    asm volatile("fence w, w" ::: "memory");
    status[abi::kStatusMagic] = abi::kDoneMagic;
}
