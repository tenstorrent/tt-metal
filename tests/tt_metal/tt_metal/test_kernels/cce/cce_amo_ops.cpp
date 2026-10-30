// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

// RISC-V AMOs (Zaamo) and LR/SC (Zalrsc) on cached words of the hart's own TL1. Flags and the status line go through
// the uncached alias, and no iDMA or ITN write targets this TL1 while they are stored (MIM-560). Sub-word AMOs and
// amocas are not in the toolchain (no Zabha, no Zacas), so no mode issues them.

#include "cce_amo_ops_abi.h"
#include "cce_test_common.h"
#include "dev_mem_map.h"
#include "risc_common.h"

#if !defined(__riscv_zaamo) || !defined(__riscv_zalrsc)
#error "cce_amo_ops needs the Zaamo and Zalrsc extensions"
#endif

namespace abi = cce_amo_ops_abi;
using cce_test::arg_u64;
using cce_test::own_tl1_words;
using cce_test::poll_bounded;
using cce_test::read_cycle_counter;
using cce_test::WaitDeadline;

static volatile uint32_t* cached_word(uint32_t offset) {
    return reinterpret_cast<volatile uint32_t*>(static_cast<uintptr_t>(offset));
}

static inline __attribute__((always_inline)) uint32_t amoadd_w(volatile uint32_t* word, uint32_t value) {
    uint32_t previous;
    asm volatile("amoadd.w %0, %2, (%1)" : "=r"(previous) : "r"(word), "r"(value) : "memory");
    return previous;
}

// One LR/SC increment. Returns whether it stored within kMaxScRetries attempts; `attempts` counts them.
static inline __attribute__((always_inline)) bool lrsc_add_w(
    volatile uint32_t* word, uint32_t value, uint32_t& previous, uint32_t& attempts) {
    for (attempts = 1; attempts <= abi::kMaxScRetries; ++attempts) {
        uint32_t loaded;
        uint32_t failed;
        asm volatile(
            "lr.w %0, (%2)\n"
            "add %1, %0, %3\n"
            "sc.w %1, %1, (%2)\n"
            : "=&r"(loaded), "=&r"(failed)
            : "r"(word), "r"(value)
            : "memory");
        if (failed == 0) {
            previous = loaded;
            return true;
        }
    }
    attempts = abi::kMaxScRetries;
    return false;
}

static uint32_t flag_line(uint32_t flag_offset, uint32_t index) { return flag_offset + index * abi::kLineBytes; }

// Waits until the first `count` flag lines from `first_index` all read 1; `seen` counts those that did.
static bool wait_for_flags(
    uint32_t flag_offset, uint32_t first_index, uint32_t count, uint64_t budget_cycles, uint32_t& seen) {
    uint32_t wait_cycles = 0;
    return poll_bounded(
        [flag_offset, first_index, count, &seen] {
            seen = 0;
            for (uint32_t index = first_index; index < first_index + count; ++index) {
                seen += *own_tl1_words(flag_line(flag_offset, index)) == 1 ? 1 : 0;
            }
            return seen == count;
        },
        budget_cycles,
        wait_cycles);
}

static uint64_t low_bits(bool doubleword, uint64_t value) { return doubleword ? value : value & 0xFFFFFFFFull; }

#define AMO_CASE(op, mnemonic)                                                                           \
    case abi::op:                                                                                        \
        asm volatile(mnemonic " %0, %2, (%1)" : "=r"(previous) : "r"(address), "r"(operand) : "memory"); \
        break;

// One AMO of kOpAmo*; returns the old value (.w: its low 32 bits).
static uint64_t amo(uint32_t op, uintptr_t address, uint64_t operand) {
    uint64_t previous = 0;
    switch (op) {
        AMO_CASE(kOpAmoSwapW, "amoswap.w")
        AMO_CASE(kOpAmoAddW, "amoadd.w")
        AMO_CASE(kOpAmoXorW, "amoxor.w")
        AMO_CASE(kOpAmoAndW, "amoand.w")
        AMO_CASE(kOpAmoOrW, "amoor.w")
        AMO_CASE(kOpAmoMinW, "amomin.w")
        AMO_CASE(kOpAmoMaxW, "amomax.w")
        AMO_CASE(kOpAmoMinuW, "amominu.w")
        AMO_CASE(kOpAmoMaxuW, "amomaxu.w")
        AMO_CASE(kOpAmoSwapD, "amoswap.d")
        AMO_CASE(kOpAmoAddD, "amoadd.d")
        AMO_CASE(kOpAmoXorD, "amoxor.d")
        AMO_CASE(kOpAmoAndD, "amoand.d")
        AMO_CASE(kOpAmoOrD, "amoor.d")
        AMO_CASE(kOpAmoMinD, "amomin.d")
        AMO_CASE(kOpAmoMaxD, "amomax.d")
        AMO_CASE(kOpAmoMinuD, "amominu.d")
        AMO_CASE(kOpAmoMaxuD, "amomaxu.d")
        AMO_CASE(kOpAmoAddWAq, "amoadd.w.aq")
        AMO_CASE(kOpAmoAddWRl, "amoadd.w.rl")
        AMO_CASE(kOpAmoAddWAqRl, "amoadd.w.aqrl")
        AMO_CASE(kOpAmoAddDAqRl, "amoadd.d.aqrl")
        default: break;
    }
    return low_bits(abi::is_doubleword_op(op), previous);
}

#undef AMO_CASE

static bool is_amo_op(uint32_t op) { return op <= abi::kOpAmoMaxuD || (op >= abi::kOpAmoAddWAq && op < abi::kNumOps); }

// Outcome of one LR/SC or CAS operation. Done: it stored. CompareFailed: the CAS compare did not match (nothing
// stored). ScRetriesExhausted: sc failed kMaxScRetries times. BudgetExpired: the launch's budget ran out first.
enum class OpResult : uint32_t { Done, CompareFailed, ScRetriesExhausted, BudgetExpired };

static uint32_t failure_code(OpResult result) {
    return result == OpResult::BudgetExpired ? abi::kCodeBudgetExpired : abi::kCodeScRetriesExhausted;
}

// lr, compare with `expected`, sc of `desired`; any nonzero sc result is retried, up to kMaxScRetries attempts or the
// budget. `previous` is the value lr loaded last.
static OpResult lrsc_cas(
    bool doubleword,
    uintptr_t address,
    uint64_t expected,
    uint64_t desired,
    const WaitDeadline& deadline,
    uint64_t& previous,
    uint32_t& attempts) {
    // lr.w sign-extends, so the compare uses the sign-extended expected word.
    const uint64_t compare =
        doubleword ? expected : static_cast<uint64_t>(static_cast<int64_t>(static_cast<int32_t>(expected)));
    for (attempts = 1; attempts <= abi::kMaxScRetries; ++attempts) {
        if (deadline.expired()) {
            attempts -= 1;
            return OpResult::BudgetExpired;
        }
        uint64_t loaded;
        uint64_t sc_result;
        uint64_t mismatched;
        if (doubleword) {
            asm volatile(
                "li %1, 0\n"
                "li %2, 0\n"
                "lr.d %0, (%3)\n"
                "bne %0, %4, 1f\n"
                "sc.d %1, %5, (%3)\n"
                "j 2f\n"
                "1: li %2, 1\n"
                "2:\n"
                : "=&r"(loaded), "=&r"(sc_result), "=&r"(mismatched)
                : "r"(address), "r"(compare), "r"(desired)
                : "memory");
        } else {
            asm volatile(
                "li %1, 0\n"
                "li %2, 0\n"
                "lr.w %0, (%3)\n"
                "bne %0, %4, 1f\n"
                "sc.w %1, %5, (%3)\n"
                "j 2f\n"
                "1: li %2, 1\n"
                "2:\n"
                : "=&r"(loaded), "=&r"(sc_result), "=&r"(mismatched)
                : "r"(address), "r"(compare), "r"(desired)
                : "memory");
        }
        previous = low_bits(doubleword, loaded);
        if (mismatched != 0) {
            return OpResult::CompareFailed;
        }
        if (sc_result == 0) {
            return OpResult::Done;
        }
    }
    attempts = abi::kMaxScRetries;
    return OpResult::ScRetriesExhausted;
}

static OpResult lrsc_add(
    bool doubleword,
    uintptr_t address,
    uint64_t value,
    const WaitDeadline& deadline,
    uint64_t& previous,
    uint32_t& attempts) {
    for (attempts = 1; attempts <= abi::kMaxScRetries; ++attempts) {
        if (deadline.expired()) {
            attempts -= 1;
            return OpResult::BudgetExpired;
        }
        uint64_t loaded;
        uint64_t failed;
        if (doubleword) {
            asm volatile(
                "lr.d %0, (%2)\n"
                "add %1, %0, %3\n"
                "sc.d %1, %1, (%2)\n"
                : "=&r"(loaded), "=&r"(failed)
                : "r"(address), "r"(value)
                : "memory");
        } else {
            asm volatile(
                "lr.w %0, (%2)\n"
                "add %1, %0, %3\n"
                "sc.w %1, %1, (%2)\n"
                : "=&r"(loaded), "=&r"(failed)
                : "r"(address), "r"(value)
                : "memory");
        }
        if (failed == 0) {
            previous = low_bits(doubleword, loaded);
            return OpResult::Done;
        }
    }
    attempts = abi::kMaxScRetries;
    return OpResult::ScRetriesExhausted;
}

static uint64_t plain_load(bool doubleword, uintptr_t address) {
    return doubleword ? *reinterpret_cast<volatile uint64_t*>(address) : *reinterpret_cast<volatile uint32_t*>(address);
}

// Increment by CAS: start from a plain cached load and retry with the value the failed compare saw, until the budget
// runs out or kMaxCasTries. `failures` counts failed compares and failed sc; `attempts` counts every LR/SC attempt and
// `tries` the CAS tries.
static OpResult cas_add(
    bool doubleword,
    uintptr_t address,
    uint64_t value,
    const WaitDeadline& deadline,
    uint64_t& previous,
    uint32_t& attempts,
    uint32_t& failures,
    uint32_t& tries) {
    uint64_t expected = plain_load(doubleword, address);
    attempts = 0;
    for (tries = 1; tries <= abi::kMaxCasTries; ++tries) {
        uint32_t cas_attempts = 0;
        const OpResult result = lrsc_cas(
            doubleword, address, expected, low_bits(doubleword, expected + value), deadline, previous, cas_attempts);
        attempts += cas_attempts;
        failures += cas_attempts - (result == OpResult::Done ? 1 : 0);
        if (result != OpResult::CompareFailed) {
            return result;
        }
        expected = previous;
    }
    tries = abi::kMaxCasTries;
    return OpResult::ScRetriesExhausted;
}

// One increment of kOp*Add*. `failures` counts failed sc and compares; `cas_tries` is the CAS tries (0 unless CAS).
static OpResult increment(
    uint32_t op,
    uintptr_t address,
    uint64_t value,
    const WaitDeadline& deadline,
    uint64_t& previous,
    uint32_t& attempts,
    uint32_t& failures,
    uint32_t& cas_tries) {
    const bool doubleword = abi::is_doubleword_op(op);
    cas_tries = 0;
    if (op == abi::kOpAmoAddW || op == abi::kOpAmoAddD) {
        previous = amo(op, address, value);
        attempts = 1;
        return OpResult::Done;
    }
    if (op == abi::kOpLrScAddW || op == abi::kOpLrScAddD) {
        const OpResult result = lrsc_add(doubleword, address, value, deadline, previous, attempts);
        failures += result == OpResult::Done ? attempts - 1 : attempts;
        return result;
    }
    return cas_add(doubleword, address, value, deadline, previous, attempts, failures, cas_tries);
}

static bool flags_reach(uint32_t flag_offset, uint32_t first_index, uint32_t count, uint32_t value, uint32_t& seen) {
    seen = 0;
    for (uint32_t index = first_index; index < first_index + count; ++index) {
        seen += *own_tl1_words(flag_line(flag_offset, index)) >= value ? 1 : 0;
    }
    return seen == count;
}

static bool wait_until(
    uint32_t flag_offset,
    uint32_t first_index,
    uint32_t count,
    uint32_t value,
    const WaitDeadline& deadline,
    uint32_t& seen) {
    bool reached = flags_reach(flag_offset, first_index, count, value, seen);
    while (!reached && !deadline.expired()) {
        reached = flags_reach(flag_offset, first_index, count, value, seen);
    }
    return reached;
}

// Generation `generation` of the ready-flag barrier; returns whether every participant reached it.
static bool barrier(
    uint32_t flag_offset,
    uint32_t participant,
    uint32_t num_participants,
    uint32_t generation,
    const WaitDeadline& deadline,
    uint32_t& seen) {
    *own_tl1_words(flag_line(flag_offset, participant)) = generation;
    return wait_until(flag_offset, 0, num_participants, generation, deadline, seen);
}

static void discard_line(uint32_t line) {
    invalidate_l2_cache_line(line);
    invalidate_l1_dcache(line);
}

static void flush_line(uint32_t line) {
    flush_l1_dcache(line);
    flush_l2_cache_line(line);
}

static void discard_l1_range(uint32_t offset, uint32_t num_bytes) {
    for (uint32_t line = offset; line < offset + num_bytes; line += abi::kLineBytes) {
        invalidate_l1_dcache(line);
    }
}

static uint32_t read_line_cached(uint32_t line) {
    uint32_t sum = 0;
    for (uint32_t word = 0; word < abi::kLineBytes / sizeof(uint32_t); ++word) {
        sum += cached_word(line)[word];
    }
    return sum;
}

struct Args {
    uint32_t mode = 0;
    uint32_t status_offset = 0;
    uint64_t budget_cycles = 0;
    uint32_t target_offset = 0;
    uint32_t count = 0;
    uint32_t value = 0;
    uint32_t flag_offset = 0;
    uint32_t num_participants = 0;
    uint32_t participant = 0;
    uint32_t result_offset = 0;
    uint32_t table_offset = 0;
    uint32_t num_points = 0;
    uint32_t operation = 0;
    uint32_t stride_bytes = 0;
    uint32_t leg = 0;
    uint32_t seed = 0;
    uint32_t peer_cce = 0;
};

struct Status {
    uint32_t code = abi::kCodeDone;
    uint32_t words[abi::kStatusWords] = {};

    void fail(uint32_t failure) {
        if (code == abi::kCodeDone) {
            code = failure;
        }
    }
};

static bool aligned(uint32_t offset) { return offset % abi::kLineBytes == 0; }

// Start of the test TL1 area: the HAL's UNRESERVED base for the DRAM core before its alignment (qa_hal_dram.cpp:67-68).
inline constexpr uint64_t kTestAreaStart =
    MEM_CCE_KERNEL_BASE + static_cast<uint64_t>(MEM_CCE_KERNEL_SIZE) * MEM_CCE_LOCAL_HARTS;

// [offset, offset + num_bytes) is a nonempty, 64 B aligned region of the test area inside the cacheable port.
static bool region_valid(uint32_t offset, uint64_t num_bytes) {
    return offset != 0 && aligned(offset) && offset >= kTestAreaStart && num_bytes != 0 &&
           static_cast<uint64_t>(offset) + num_bytes <= MEMORY_PORT_CACHEABLE_MEM_PORT_MEM_SIZE;
}

static uint64_t lines(uint64_t num_lines) { return num_lines * abi::kLineBytes; }

static bool flags_valid(const Args& args) { return region_valid(args.flag_offset, lines(2 * abi::kMaxParticipants)); }

static bool participants_valid(const Args& args) {
    return args.num_participants != 0 && args.num_participants <= abi::kMaxParticipants &&
           args.participant < args.num_participants && flags_valid(args);
}

static bool args_valid(const Args& args) {
    if (args.mode >= abi::kNumModes || args.budget_cycles == 0) {
        return false;
    }
    const bool count_valid = args.count != 0 && args.count <= abi::kMaxCount;
    const uint64_t num_points = args.num_points;
    const uint64_t num_participants = args.num_participants;
    switch (args.mode) {
        case abi::kModeTwoHartIncrement:
            return count_valid && participants_valid(args) && region_valid(args.target_offset, lines(2));
        case abi::kModeOpMatrix:
            return args.num_participants == 1 && args.participant == 0 && args.num_points != 0 &&
                   args.num_points <= abi::kMaxOpEntries && region_valid(args.target_offset, lines(num_points)) &&
                   region_valid(args.table_offset, num_points * abi::kOpEntryWords * sizeof(uint32_t));
        case abi::kModeSweep:
            return participants_valid(args) && args.num_points != 0 && args.num_points <= abi::kMaxSweepPoints &&
                   abi::is_increment_op(args.operation) &&
                   args.stride_bytes % (abi::is_doubleword_op(args.operation) ? 8 : 4) == 0 &&
                   static_cast<uint64_t>(args.stride_bytes) * num_participants <= abi::kSweepPointBytes &&
                   region_valid(args.target_offset, num_points * abi::kSweepPointBytes) &&
                   region_valid(args.table_offset, num_points * abi::kSweepEntryWords * sizeof(uint32_t)) &&
                   region_valid(args.result_offset, num_participants * num_points * abi::kSweepResultBytes);
        case abi::kModeLineState:
            return args.leg <= abi::kLegInvalidateAfterWrite && region_valid(args.target_offset, lines(2));
        case abi::kModeOrdering:
            return count_valid && participants_valid(args) && args.num_participants == 2 &&
                   args.leg <= abi::kLegReleaseOnly && region_valid(args.target_offset, lines(3));
        case abi::kModeNeighbourWrite:
            return count_valid && participants_valid(args) && region_valid(args.target_offset, lines(1));
        case abi::kModeStorm:
        case abi::kModeSoak:
            return count_valid && participants_valid(args) && args.leg <= abi::kLegDisjoint &&
                   region_valid(
                       args.target_offset,
                       (args.leg == abi::kLegDisjoint ? num_participants : 1) * abi::kStormSetBytes) &&
                   region_valid(args.result_offset, lines(num_participants));
        case abi::kModeItnWriter:
        case abi::kModeItnReceiver: return flags_valid(args) && region_valid(args.target_offset, lines(1));
        default: return false;
    }
}

static void run_two_hart_increment(const Args& args, Status& status) {
    volatile uint32_t* amo_word = cached_word(args.target_offset);
    volatile uint32_t* lrsc_word = cached_word(args.target_offset + abi::kLineBytes);
    // The host wrote the initial words to TL1; drop any copy an earlier launch left cached.
    invalidate_l1_dcache(args.target_offset);
    invalidate_l1_dcache(args.target_offset + abi::kLineBytes);
    if (args.participant == 0) {
        invalidate_l2_cache_range(args.target_offset, 2 * abi::kLineBytes);
    }
    *own_tl1_words(flag_line(args.flag_offset, args.participant)) = 1;
    if (!wait_for_flags(
            args.flag_offset, 0, args.num_participants, args.budget_cycles, status.words[abi::kStatusPeersSeen])) {
        status.fail(abi::kCodePeerTimeout);
        return;
    }
    const uint64_t start = read_cycle_counter();
    for (uint32_t iteration = 0; iteration < args.count; ++iteration) {
        status.words[abi::kStatusAmoReturnSum] += amoadd_w(amo_word, args.value);
    }
    for (uint32_t iteration = 0; iteration < args.count; ++iteration) {
        uint32_t previous = 0;
        uint32_t attempts = 0;
        const bool stored = lrsc_add_w(lrsc_word, args.value, previous, attempts);
        status.words[abi::kStatusScFailures] += stored ? attempts - 1 : attempts;
        if (attempts > status.words[abi::kStatusMaxRetries]) {
            status.words[abi::kStatusMaxRetries] = attempts;
        }
        if (!stored) {
            status.fail(abi::kCodeScRetriesExhausted);
            break;
        }
        status.words[abi::kStatusLrScReturnSum] += previous;
    }
    status.words[abi::kStatusCycles] = static_cast<uint32_t>(read_cycle_counter() - start);
    flush_l1_dcache(0);
    asm volatile("fence rw, w" ::: "memory");
    *own_tl1_words(flag_line(args.flag_offset, abi::kMaxParticipants + args.participant)) = 1;
    if (args.participant == 0) {
        if (!wait_for_flags(
                args.flag_offset,
                abi::kMaxParticipants,
                args.num_participants,
                args.budget_cycles,
                status.words[abi::kStatusPeersSeen])) {
            status.fail(abi::kCodePeerTimeout);
        }
        asm volatile("fence r, rw" ::: "memory");
        status.words[abi::kStatusAmoFinal] = amoadd_w(amo_word, 0);
        status.words[abi::kStatusLrScFinal] = amoadd_w(lrsc_word, 0);
        flush_l1_dcache(0);
        flush_l2_cache_range(args.target_offset, 2 * abi::kLineBytes);
    }
}

static void run_op_matrix(const Args& args, const WaitDeadline& deadline, Status& status) {
    const uint32_t num_bytes = args.num_points * abi::kLineBytes;
    discard_l1_range(args.target_offset, num_bytes);
    invalidate_l2_cache_range(args.target_offset, num_bytes);
    const uint64_t start = read_cycle_counter();
    for (uint32_t entry = 0; entry < args.num_points; ++entry) {
        if (deadline.expired()) {
            status.fail(abi::kCodeBudgetExpired);
            break;
        }
        volatile uint32_t* words = own_tl1_words(args.table_offset + entry * abi::kOpEntryWords * sizeof(uint32_t));
        const uint32_t op = words[abi::kOpEntryOperation];
        const uint64_t operand =
            (static_cast<uint64_t>(words[abi::kOpEntryOperandHi]) << 32) | words[abi::kOpEntryOperandLo];
        const uint64_t compare =
            (static_cast<uint64_t>(words[abi::kOpEntryCompareHi]) << 32) | words[abi::kOpEntryCompareLo];
        const uintptr_t address = args.target_offset + entry * abi::kLineBytes;
        const bool doubleword = abi::is_doubleword_op(op);
        uint64_t previous = 0;
        uint32_t attempts = 1;
        if (is_amo_op(op)) {
            previous = amo(op, address, operand);
        } else if (op == abi::kOpLrScCasW || op == abi::kOpLrScCasD) {
            const OpResult result = lrsc_cas(doubleword, address, compare, operand, deadline, previous, attempts);
            if (result == OpResult::ScRetriesExhausted || result == OpResult::BudgetExpired) {
                status.fail(failure_code(result));
            }
        } else if (abi::is_increment_op(op)) {
            uint32_t failures = 0;
            uint32_t cas_tries = 0;
            const OpResult result = increment(op, address, operand, deadline, previous, attempts, failures, cas_tries);
            if (result != OpResult::Done) {
                status.fail(failure_code(result));
            }
        } else {
            status.fail(abi::kCodeBadArgs);
            continue;
        }
        words[abi::kOpEntryReturnedLo] = static_cast<uint32_t>(previous);
        words[abi::kOpEntryReturnedHi] = static_cast<uint32_t>(previous >> 32);
        words[abi::kOpEntryAttempts] = attempts;
    }
    status.words[abi::kStatusCycles] = static_cast<uint32_t>(read_cycle_counter() - start);
    flush_l1_dcache(0);
    flush_l2_cache_range(args.target_offset, num_bytes);
}

// Sets this participant's done flag, then participant 0 waits for every done flag and flushes `num_bytes` from L2.
static void finish_participants(const Args& args, uint32_t num_bytes, const WaitDeadline& deadline, Status& status) {
    flush_l1_dcache(0);
    *own_tl1_words(flag_line(args.flag_offset, abi::kMaxParticipants + args.participant)) = 1;
    if (args.participant == 0) {
        if (!wait_until(
                args.flag_offset,
                abi::kMaxParticipants,
                args.num_participants,
                1,
                deadline,
                status.words[abi::kStatusPeersSeen])) {
            status.fail(abi::kCodePeerTimeout);
        }
        asm volatile("fence r, rw" ::: "memory");
        flush_l2_cache_range(args.target_offset, num_bytes);
    }
}

// Drops every copy of `num_bytes` from the target before the first barrier, so each participant reads the host's
// initial values.
static void discard_participant_range(const Args& args, uint32_t num_bytes) {
    discard_l1_range(args.target_offset, num_bytes);
    if (args.participant == 0) {
        invalidate_l2_cache_range(args.target_offset, num_bytes);
    }
}

static void run_sweep(const Args& args, const WaitDeadline& deadline, Status& status) {
    const uint32_t num_bytes = args.num_points * abi::kSweepPointBytes;
    discard_participant_range(args, num_bytes);
    bool expired = false;
    for (uint32_t point = 0; point < args.num_points && !expired; ++point) {
        volatile uint32_t* entry = own_tl1_words(args.table_offset + point * abi::kSweepEntryWords * sizeof(uint32_t));
        const uint32_t value = entry[abi::kSweepEntryValue];
        const uint32_t count = entry[abi::kSweepEntryCount];
        if (!barrier(
                args.flag_offset,
                args.participant,
                args.num_participants,
                point + 1,
                deadline,
                status.words[abi::kStatusPeersSeen])) {
            status.fail(abi::kCodePeerTimeout);
            return;
        }
        const uintptr_t address =
            args.target_offset + point * abi::kSweepPointBytes + args.participant * args.stride_bytes;
        uint64_t return_sum = 0;
        uint32_t failures = 0;
        uint32_t max_attempts = 0;
        uint32_t max_cas_tries = 0;
        uint32_t interleaved = 0;
        uint64_t last = 0;
        const bool doubleword = abi::is_doubleword_op(args.operation);
        const uint64_t start = read_cycle_counter();
        for (uint32_t iteration = 0; iteration < count && iteration < abi::kMaxCount; ++iteration) {
            if (deadline.expired()) {
                expired = true;
                status.fail(abi::kCodeBudgetExpired);
                break;
            }
            uint64_t previous = 0;
            uint32_t attempts = 0;
            uint32_t cas_tries = 0;
            const OpResult result =
                increment(args.operation, address, value, deadline, previous, attempts, failures, cas_tries);
            max_attempts = attempts > max_attempts ? attempts : max_attempts;
            max_cas_tries = cas_tries > max_cas_tries ? cas_tries : max_cas_tries;
            if (result != OpResult::Done) {
                expired = result == OpResult::BudgetExpired;
                status.fail(failure_code(result));
                break;
            }
            interleaved += iteration != 0 && previous != low_bits(doubleword, last + value) ? 1 : 0;
            last = previous;
            return_sum += previous;
        }
        const uint32_t cycles = static_cast<uint32_t>(read_cycle_counter() - start);
        volatile uint32_t* result =
            own_tl1_words(args.result_offset + (args.participant * args.num_points + point) * abi::kSweepResultBytes);
        result[abi::kSweepResultReturnSumLo] = static_cast<uint32_t>(return_sum);
        result[abi::kSweepResultReturnSumHi] = static_cast<uint32_t>(return_sum >> 32);
        result[abi::kSweepResultFailures] = failures;
        result[abi::kSweepResultMaxAttempts] = max_attempts;
        result[abi::kSweepResultCycles] = cycles;
        result[abi::kSweepResultInterleaved] = interleaved;
        result[abi::kSweepResultMaxCasTries] = max_cas_tries;
    }
    finish_participants(args, num_bytes, deadline, status);
}

static void run_line_state(const Args& args, Status& status) {
    const uint32_t line = args.target_offset;
    const uint32_t control_line = line + abi::kLineBytes;
    volatile uint32_t* words = cached_word(line);
    switch (args.leg) {
        case abi::kLegClean:
            discard_line(line);
            read_line_cached(line);
            break;
        case abi::kLegDirty:
        case abi::kLegFlushBeforeAtomic:
            discard_line(line);
            for (uint32_t word = 1; word < abi::kLineBytes / sizeof(uint32_t); ++word) {
                words[word] = args.seed + word;
            }
            if (args.leg == abi::kLegFlushBeforeAtomic) {
                flush_line(line);
            }
            break;
        case abi::kLegWarm:
            discard_line(line);
            discard_line(control_line);
            read_line_cached(line);
            read_line_cached(control_line);
            status.words[abi::kStatusObservedFirst] = words[0];
            status.words[abi::kStatusObservedSecond] = *cached_word(control_line);
            return;
        default: break;
    }
    status.words[abi::kStatusObservedFirst] = *own_tl1_words(line);
    if (args.leg == abi::kLegAfterWrite || args.leg == abi::kLegInvalidateAfterWrite) {
        status.words[abi::kStatusObservedSecond] = *cached_word(control_line);
    }
    if (args.leg == abi::kLegInvalidateAfterWrite) {
        discard_line(line);
    }
    status.words[abi::kStatusReturnedLo] = amoadd_w(words, args.value);
    flush_line(line);
}

static uint32_t ordering_word(uint32_t seed, uint32_t round, uint32_t word) {
    return (seed & 0xFFFF0000u) | (round << 4) | word;
}

static uint32_t acquire_read_w(uintptr_t address) {
    uint32_t value;
    asm volatile("amoadd.w.aq %0, zero, (%1)" : "=r"(value) : "r"(address) : "memory");
    return value;
}

static void release_swap_w(uintptr_t address, uint32_t value) {
    asm volatile("amoswap.w.rl zero, %1, (%0)" ::"r"(address), "r"(value) : "memory");
}

static bool wait_acquire(uintptr_t address, uint32_t value, const WaitDeadline& deadline) {
    bool reached = acquire_read_w(address) == value;
    while (!reached && !deadline.expired()) {
        reached = acquire_read_w(address) == value;
    }
    return reached;
}

static void run_ordering(const Args& args, const WaitDeadline& deadline, Status& status) {
    const uint32_t data_line = args.target_offset;
    const uintptr_t flag_word = args.target_offset + abi::kLineBytes;
    const uintptr_t ack_word = args.target_offset + 2 * abi::kLineBytes;
    discard_participant_range(args, 3 * abi::kLineBytes);
    if (!barrier(
            args.flag_offset,
            args.participant,
            args.num_participants,
            1,
            deadline,
            status.words[abi::kStatusPeersSeen])) {
        status.fail(abi::kCodePeerTimeout);
        return;
    }
    volatile uint32_t* data = cached_word(data_line);
    const uint32_t words_per_line = abi::kLineBytes / sizeof(uint32_t);
    const uint64_t start = read_cycle_counter();
    for (uint32_t round = 0; round < args.count; ++round) {
        if (args.participant == 0) {
            for (uint32_t word = 0; word < words_per_line; ++word) {
                data[word] = ordering_word(args.seed, round, word);
            }
            if (args.leg == abi::kLegFlushRelease) {
                flush_l1_dcache(data_line);
            }
            release_swap_w(flag_word, round + 1);
            if (!wait_acquire(ack_word, round + 1, deadline)) {
                status.fail(abi::kCodePeerTimeout);
                break;
            }
        } else {
            if (!wait_acquire(flag_word, round + 1, deadline)) {
                status.fail(abi::kCodePeerTimeout);
                break;
            }
            invalidate_l1_dcache(data_line);
            bool round_mismatched = false;
            for (uint32_t word = 0; word < words_per_line; ++word) {
                const uint32_t observed = data[word];
                const uint32_t expected = ordering_word(args.seed, round, word);
                if (observed != expected && !round_mismatched) {
                    round_mismatched = true;
                    if (status.words[abi::kStatusMismatches] == 0) {
                        status.words[abi::kStatusFirstMismatch] = round;
                        status.words[abi::kStatusObservedFirst] = observed;
                        status.words[abi::kStatusObservedSecond] = expected;
                    }
                    status.words[abi::kStatusMismatches]++;
                }
            }
            release_swap_w(ack_word, round + 1);
        }
    }
    status.words[abi::kStatusCycles] = static_cast<uint32_t>(read_cycle_counter() - start);
}

static void run_neighbour_write(const Args& args, const WaitDeadline& deadline, Status& status) {
    discard_participant_range(args, abi::kLineBytes);
    if (!barrier(
            args.flag_offset,
            args.participant,
            args.num_participants,
            1,
            deadline,
            status.words[abi::kStatusPeersSeen])) {
        status.fail(abi::kCodePeerTimeout);
        return;
    }
    volatile uint32_t* words = cached_word(args.target_offset);
    const uint64_t start = read_cycle_counter();
    for (uint32_t iteration = 0; iteration < args.count; ++iteration) {
        if (deadline.expired()) {
            status.fail(abi::kCodeBudgetExpired);
            break;
        }
        if (args.participant == 0) {
            status.words[abi::kStatusAmoReturnSum] += amoadd_w(words, args.value);
        } else {
            words[args.participant] = abi::neighbour_word(args.seed, args.participant, iteration);
        }
    }
    status.words[abi::kStatusCycles] = static_cast<uint32_t>(read_cycle_counter() - start);
    if (args.participant == 0) {
        status.words[abi::kStatusAmoFinal] = amoadd_w(words, 0);
    }
    finish_participants(args, abi::kLineBytes, deadline, status);
}

static uint32_t storm_history_index(const abi::StormDraw& draw) { return (draw.doubleword ? 4 : 0) + draw.slot; }

static void run_storm(const Args& args, const WaitDeadline& deadline, Status& status) {
    const bool soak = args.mode == abi::kModeSoak;
    const uint32_t num_sets = args.leg == abi::kLegDisjoint ? args.num_participants : 1;
    const uint32_t num_bytes = num_sets * abi::kStormSetBytes;
    discard_participant_range(args, num_bytes);
    uint32_t generation = 1;
    if (!barrier(
            args.flag_offset,
            args.participant,
            args.num_participants,
            generation,
            deadline,
            status.words[abi::kStatusPeersSeen])) {
        status.fail(abi::kCodePeerTimeout);
        return;
    }
    const uint32_t set_offset =
        args.target_offset + (args.leg == abi::kLegDisjoint ? args.participant * abi::kStormSetBytes : 0);
    volatile uint32_t* result = own_tl1_words(args.result_offset + args.participant * abi::kLineBytes);
    uint32_t state = abi::storm_state(args.seed, args.participant);
    uint32_t schedule = abi::soak_schedule_state(args.seed);
    uint64_t last_add[2 * abi::kStormSlots] = {};
    uint64_t last_cas[2 * abi::kStormSlots] = {};
    uint32_t add_seen = 0;
    uint32_t cas_seen = 0;
    uint32_t add_violations = 0;
    uint32_t cas_violations = 0;
    uint32_t interleaved = 0;
    uint32_t cas_failures = 0;
    uint32_t max_attempts = 0;
    uint32_t max_cas_tries = 0;
    uint64_t swap_xor = 0;
    uint64_t swap_sum = 0;
    uint32_t barriers = 0;
    uint32_t max_barrier_cycles = 0;
    uint32_t done = 0;
    const uint64_t start = read_cycle_counter();
    bool expired = false;
    while (done < args.count && !expired) {
        const uint32_t burst = soak ? abi::soak_burst(schedule) : args.count;
        for (uint32_t index = 0; index < burst && done < args.count; ++index, ++done) {
            if (deadline.expired()) {
                expired = true;
                status.fail(abi::kCodeBudgetExpired);
                break;
            }
            const abi::StormDraw draw = abi::storm_draw(state, soak);
            const uintptr_t address = set_offset + abi::storm_slot_offset(draw);
            const uint32_t history = storm_history_index(draw);
            const uint32_t width_offset = draw.doubleword ? abi::kOpAmoSwapD - abi::kOpAmoSwapW : 0;
            uint64_t previous = 0;
            switch (draw.op) {
                case abi::kStormAddDiscard:
                    if (draw.doubleword) {
                        asm volatile("amoadd.d zero, %1, (%0)" ::"r"(address), "r"(draw.operand) : "memory");
                    } else {
                        asm volatile("amoadd.w zero, %1, (%0)" ::"r"(address), "r"(draw.operand) : "memory");
                    }
                    // This participant's own update: the next value it gets back is at least one higher.
                    last_add[history] += 1;
                    break;
                case abi::kStormAddReturn:
                    previous = amo(abi::kOpAmoAddW + width_offset, address, draw.operand);
                    add_violations += (add_seen >> history & 1u) != 0 && previous <= last_add[history] ? 1 : 0;
                    interleaved += (add_seen >> history & 1u) != 0 && previous > last_add[history] + 1 ? 1 : 0;
                    last_add[history] = previous;
                    add_seen |= 1u << history;
                    break;
                case abi::kStormSwap:
                    previous = amo(abi::kOpAmoSwapW + width_offset, address, draw.operand);
                    swap_xor ^= previous;
                    swap_sum += previous;
                    break;
                case abi::kStormCas: {
                    uint32_t attempts = 0;
                    uint32_t tries = 0;
                    const OpResult result = cas_add(
                        draw.doubleword, address, draw.operand, deadline, previous, attempts, cas_failures, tries);
                    max_attempts = attempts > max_attempts ? attempts : max_attempts;
                    max_cas_tries = tries > max_cas_tries ? tries : max_cas_tries;
                    if (result != OpResult::Done) {
                        expired = result == OpResult::BudgetExpired;
                        status.fail(failure_code(result));
                        break;
                    }
                    cas_violations += (cas_seen >> history & 1u) != 0 && previous <= last_cas[history] ? 1 : 0;
                    interleaved += (cas_seen >> history & 1u) != 0 && previous > last_cas[history] + 1 ? 1 : 0;
                    last_cas[history] = previous;
                    cas_seen |= 1u << history;
                    break;
                }
                case abi::kStormXor: amo(abi::kOpAmoXorW + width_offset, address, draw.operand); break;
                case abi::kStormOr: amo(abi::kOpAmoOrW + width_offset, address, draw.operand); break;
                case abi::kStormAnd: amo(abi::kOpAmoAndW + width_offset, address, draw.operand); break;
                case abi::kStormMax: amo(abi::kOpAmoMaxW + width_offset, address, draw.operand); break;
                case abi::kStormMin: amo(abi::kOpAmoMinW + width_offset, address, draw.operand); break;
                case abi::kStormMaxu: amo(abi::kOpAmoMaxuW + width_offset, address, draw.operand); break;
                case abi::kStormMinu: amo(abi::kOpAmoMinuW + width_offset, address, draw.operand); break;
                default: break;
            }
            if (expired) {
                break;
            }
        }
        if (soak && !expired && done < args.count && abi::soak_barrier(schedule)) {
            result[abi::kStormResultProgress] = done;
            const uint64_t barrier_start = read_cycle_counter();
            if (!barrier(
                    args.flag_offset,
                    args.participant,
                    args.num_participants,
                    ++generation,
                    deadline,
                    status.words[abi::kStatusPeersSeen])) {
                status.fail(abi::kCodePeerTimeout);
                break;
            }
            const uint32_t barrier_cycles = static_cast<uint32_t>(read_cycle_counter() - barrier_start);
            max_barrier_cycles = barrier_cycles > max_barrier_cycles ? barrier_cycles : max_barrier_cycles;
            barriers++;
        }
    }
    result[abi::kStormResultProgress] = done;
    result[abi::kStormResultAddViolations] = add_violations;
    result[abi::kStormResultCasViolations] = cas_violations;
    result[abi::kStormResultCasFailures] = cas_failures;
    result[abi::kStormResultMaxAttempts] = max_attempts;
    result[abi::kStormResultSwapXorLo] = static_cast<uint32_t>(swap_xor);
    result[abi::kStormResultSwapXorHi] = static_cast<uint32_t>(swap_xor >> 32);
    result[abi::kStormResultSwapSumLo] = static_cast<uint32_t>(swap_sum);
    result[abi::kStormResultSwapSumHi] = static_cast<uint32_t>(swap_sum >> 32);
    result[abi::kStormResultBarriers] = barriers;
    result[abi::kStormResultMaxBarrierCycles] = max_barrier_cycles;
    result[abi::kStormResultCycles] = static_cast<uint32_t>(read_cycle_counter() - start);
    result[abi::kStormResultInterleaved] = interleaved;
    result[abi::kStormResultMaxCasTries] = max_cas_tries;
    finish_participants(args, num_bytes, deadline, status);
}

static void run_itn_writer(const Args& args, const WaitDeadline& deadline, Status& status) {
    volatile uint32_t* ready = reinterpret_cast<volatile uint32_t*>(
        experimental::cce_sram_address(args.peer_cce, flag_line(args.flag_offset, 0)));
    bool is_ready = *ready == 1;
    while (!is_ready && !deadline.expired()) {
        is_ready = *ready == 1;
    }
    if (!is_ready) {
        status.fail(abi::kCodePeerTimeout);
        return;
    }
    status.words[abi::kStatusPeersSeen] = 1;
    volatile uint32_t* data =
        reinterpret_cast<volatile uint32_t*>(experimental::cce_sram_address(args.peer_cce, args.target_offset));
    *data = args.value;
    // SPA accesses are outside the cacheable port and may be I/O, which `fence w, w` does not order.
    asm volatile("fence" ::: "memory");
    // The data write has landed before the flag lets the receiver run its AMO (MIM-560).
    bool is_visible = *data == args.value;
    while (!is_visible && !deadline.expired()) {
        is_visible = *data == args.value;
    }
    if (!is_visible) {
        status.fail(abi::kCodeReadBackTimeout);
        return;
    }
    asm volatile("fence" ::: "memory");
    *reinterpret_cast<volatile uint32_t*>(
        experimental::cce_sram_address(args.peer_cce, flag_line(args.flag_offset, 1))) = 1;
    asm volatile("fence" ::: "memory");
}

static void run_itn_receiver(const Args& args, const WaitDeadline& deadline, Status& status) {
    const uint32_t line = args.target_offset;
    volatile uint32_t* words = cached_word(line);
    discard_line(line);
    read_line_cached(line);
    status.words[abi::kStatusObservedFirst] = words[0];
    *own_tl1_words(flag_line(args.flag_offset, 0)) = 1;
    // Only uncached loads until the writer's flag shows its ITN writes have landed (MIM-560).
    volatile uint32_t* written = own_tl1_words(flag_line(args.flag_offset, 1));
    bool is_written = *written == 1;
    while (!is_written && !deadline.expired()) {
        is_written = *written == 1;
    }
    if (!is_written) {
        status.fail(abi::kCodePeerTimeout);
        return;
    }
    status.words[abi::kStatusPeersSeen] = 1;
    status.words[abi::kStatusObservedSecond] = words[0];
    invalidate_l2_cache_line(line);
    invalidate_l1_dcache(line);
    status.words[abi::kStatusReturnedLo] = amoadd_w(words, args.value);
    flush_line(line);
}

void kernel_main() {
    Args args;
    args.mode = get_arg_val<uint32_t>(abi::kArgMode);
    args.status_offset = get_arg_val<uint32_t>(abi::kArgStatusOffset);
    args.budget_cycles = arg_u64(abi::kArgPollBudgetLo, abi::kArgPollBudgetHi);
    args.target_offset = get_arg_val<uint32_t>(abi::kArgTargetOffset);
    args.count = get_arg_val<uint32_t>(abi::kArgCount);
    args.value = get_arg_val<uint32_t>(abi::kArgValue);
    args.flag_offset = get_arg_val<uint32_t>(abi::kArgFlagOffset);
    args.num_participants = get_arg_val<uint32_t>(abi::kArgNumParticipants);
    args.participant = get_arg_val<uint32_t>(abi::kArgParticipant);
    args.result_offset = get_arg_val<uint32_t>(abi::kArgResultOffset);
    args.table_offset = get_arg_val<uint32_t>(abi::kArgTableOffset);
    args.num_points = get_arg_val<uint32_t>(abi::kArgNumPoints);
    args.operation = get_arg_val<uint32_t>(abi::kArgOperation);
    args.stride_bytes = get_arg_val<uint32_t>(abi::kArgStrideBytes);
    args.leg = get_arg_val<uint32_t>(abi::kArgLeg);
    args.seed = get_arg_val<uint32_t>(abi::kArgSeed);
    args.peer_cce = get_arg_val<uint32_t>(abi::kArgPeerCce);
    const WaitDeadline deadline(args.budget_cycles);

    Status status;
    if (!args_valid(args)) {
        status.fail(abi::kCodeBadArgs);
    } else {
        switch (args.mode) {
            case abi::kModeTwoHartIncrement: run_two_hart_increment(args, status); break;
            case abi::kModeOpMatrix: run_op_matrix(args, deadline, status); break;
            case abi::kModeSweep: run_sweep(args, deadline, status); break;
            case abi::kModeLineState: run_line_state(args, status); break;
            case abi::kModeOrdering: run_ordering(args, deadline, status); break;
            case abi::kModeNeighbourWrite: run_neighbour_write(args, deadline, status); break;
            case abi::kModeStorm:
            case abi::kModeSoak: run_storm(args, deadline, status); break;
            case abi::kModeItnWriter: run_itn_writer(args, deadline, status); break;
            case abi::kModeItnReceiver: run_itn_receiver(args, deadline, status); break;
            default: status.fail(abi::kCodeBadArgs); break;
        }
    }

    // No status store outside the test area; the host's deadline reports the missing status.
    if (!region_valid(args.status_offset, abi::kLineBytes)) {
        return;
    }
    volatile uint32_t* status_line = own_tl1_words(args.status_offset);
    status_line[abi::kStatusCode] = status.code;
    for (uint32_t word = abi::kStatusCode + 1; word < abi::kStatusWords; ++word) {
        status_line[word] = status.words[word];
    }
    asm volatile("fence w, w" ::: "memory");
    status_line[abi::kStatusMagic] = abi::kDoneMagic;
}
