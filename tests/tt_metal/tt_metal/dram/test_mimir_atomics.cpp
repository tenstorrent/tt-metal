// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

// MM42: Mimir test plan section 4.2 (atomics) on CCE harts. Tier A is RISC-V AMO and LR/SC on cached words of the
// hart's own TL1 (AMOs on the uncached alias or above 4 MiB are illegal). No AMO runs while an iDMA or ITN write may
// target that TL1 (MIM-560). Tier M is iDMA L1 accumulate into the issuing CCE's own TL1 (cce_idma_copy
// kModeAccumulate), checked against a host model of the cce-dv reference arithmetic.

#include <gtest/gtest.h>

#include <fmt/format.h>
#include <tt-logger/tt-logger.hpp>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <optional>
#include <set>
#include <string>
#include <vector>

#include <tt-metalium/host_api.hpp>
#include <tt-metalium/tt_metal.hpp>
#include "tt_metal/tt_metal/dram/mimir_access.hpp"
#include "tt_metal/tt_metal/dram/mimir_dm_common.hpp"
#include "tt_metal/tt_metal/dram/mimir_hw_state.hpp"
#include "tt_metal/tt_metal/dram/mimir_patterns.hpp"
#include "tt_metal/tt_metal/dram/mimir_test_common.hpp"
#include "tt_metal/tt_metal/test_kernels/cce/cce_amo_ops_abi.h"
#include "tt_metal/tt_metal/test_kernels/cce/cce_idma_copy_abi.h"

namespace tt::tt_metal::mimir_test::atomics {

namespace abi = cce_amo_ops_abi;
namespace idma_abi = cce_idma_copy_abi;
static_assert(abi::kStatusWords == idma_abi::kStatusWords, "one Program reads both kernels' status lines");

inline constexpr const char* kAmoOpsKernel = "tests/tt_metal/tt_metal/test_kernels/cce/cce_amo_ops.cpp";
inline constexpr const char* kIdmaCopyKernel = "tests/tt_metal/tt_metal/test_kernels/cce/cce_idma_copy.cpp";
// One poll budget for the ready wait and one for the done wait, plus launch. The later cce_amo_ops modes and
// cce_idma_copy share one budget per launch; the long ones take kLongBudgetCycles, four poll budgets.
inline constexpr std::chrono::seconds kHostDeadline{300};
inline constexpr uint64_t kLongBudgetCycles = 4 * kPollBudgetCycles;
// Target and flag lines in hart 0's data area; the flags start one page after the target.
inline constexpr uint64_t kFlagOffsetFromTarget = 0x1000;
// The later cce_amo_ops modes, in hart 0's data area: target lines, then flag lines, results and the host table.
inline constexpr uint64_t kModeFlagsFromTarget = 0x4000, kResultsFromTarget = 0x5000, kTableFromTarget = 0x6000;
// Accumulate operands in a hart's data area: sources at its start, destinations 64 KiB in.
inline constexpr uint64_t kAccumulateDestinationFromData = 0x10000;
// Source and destination words of the functional accumulate rows stay below 2^20, so 512 accumulates of INT32 stay
// positive and below the saturation magnitude.
inline constexpr uint32_t kAccumulateValueMask = 0x000FFFFFu;
inline constexpr uint32_t kLeadingGuardTag = 0x900;

// Probe-gated values.
// MimirAtomicProbe.TwoHartAmo: cross-hart amoadd.w and LR/SC totals were correct. Gates every multi-hart AMO case.
inline constexpr bool kTwoHartAmoPassed = true;
// MimirCacheProbe.EightHartLaunch: harts 0-7 of one CCE ran a kernel and posted status. Gates every 8-hart case.
inline constexpr bool kEightHartLaunchPassed = true;

// L1 accumulate formats (Quasar noc_parameters.h:379-385, cce-dv software/lib/include/cce_dv.hpp:396-401) and the
// one operation the emulator reference runs (emu_l1_atomic.cpp:230,239). INT32 and INT8 are sign-magnitude.
inline constexpr uint32_t kFormatFp32 = 0, kFormatFp16A = 1, kFormatFp16B = 2, kFormatInt32 = 0b011,
                          kFormatInt32Complement = 0b100, kFormatUint32 = 0b101, kFormatInt8 = 0b110,
                          kFormatUint8 = 0b111;
inline constexpr uint32_t kAccumulateAdd = 0x9;

std::string code_name(uint32_t code) {
    switch (code) {
        case abi::kCodeDone: return "done";
        case abi::kCodeBadArgs: return "bad_args";
        case abi::kCodePeerTimeout: return "peer_timeout";
        case abi::kCodeScRetriesExhausted: return "sc_retries_exhausted";
        case abi::kCodeReadBackTimeout: return "read_back_timeout";
        case abi::kCodeBudgetExpired: return "budget_expired";
    }
    return fmt::format("unknown_code_0x{:x}", code);
}

std::string status_detail(const std::vector<uint32_t>& status) {
    return fmt::format(
        "kernel code {}, amo final 0x{:08x}, lr/sc final 0x{:08x}, amo return sum 0x{:08x}, lr/sc return sum "
        "0x{:08x}, sc failures {}, max attempts {}, peers seen {}, {} cycles, returned 0x{:08x}{:08x}, observed "
        "0x{:08x} / 0x{:08x}, mismatches {} (first {})",
        code_name(status[abi::kStatusCode]),
        status[abi::kStatusAmoFinal],
        status[abi::kStatusLrScFinal],
        status[abi::kStatusAmoReturnSum],
        status[abi::kStatusLrScReturnSum],
        status[abi::kStatusScFailures],
        status[abi::kStatusMaxRetries],
        status[abi::kStatusPeersSeen],
        status[abi::kStatusCycles],
        status[abi::kStatusReturnedHi],
        status[abi::kStatusReturnedLo],
        status[abi::kStatusObservedFirst],
        status[abi::kStatusObservedSecond],
        status[abi::kStatusMismatches],
        status[abi::kStatusFirstMismatch]);
}

std::string idma_code_name(uint32_t code) {
    switch (code) {
        case idma_abi::kCodeLanded: return "landed";
        case idma_abi::kCodeAcked: return "acked";
        case idma_abi::kCodeVcSpaceTimeout: return "vc_space_timeout";
        case idma_abi::kCodeAckTimeout: return "ack_timeout";
        case idma_abi::kCodeDataTimeout: return "data_timeout";
        case idma_abi::kCodeBadArgs: return "bad_args";
    }
    return fmt::format("unknown_code_0x{:x}", code);
}

std::string idma_status_detail(const std::vector<uint32_t>& status) {
    const uint32_t flags = status[idma_abi::kStatusFlags];
    return fmt::format(
        "kernel code {} flags [vc_space {} acked {} landed {}], {} commands issued, last poll {} {}, cycles vc_space "
        "{} ack {} data {}, pending high water {}, probe {} / {}, kernel mismatches {} (observed 0x{:08x} expected "
        "0x{:08x})",
        idma_code_name(status[idma_abi::kStatusCode]),
        (flags & idma_abi::kFlagVcSpace) != 0,
        (flags & idma_abi::kFlagAcked) != 0,
        (flags & idma_abi::kFlagLanded) != 0,
        status[idma_abi::kStatusElementsIssued],
        status[idma_abi::kStatusLastPoll],
        status[idma_abi::kStatusLastPollReady] != 0 ? "done" : "timed out",
        status[idma_abi::kStatusVcSpaceCycles],
        status[idma_abi::kStatusAckCycles],
        status[idma_abi::kStatusDataCycles],
        status[idma_abi::kStatusPendingHighWater],
        status[idma_abi::kStatusProbeFirst],
        status[idma_abi::kStatusProbeSecond],
        status[idma_abi::kStatusMismatchedWords],
        status[idma_abi::kStatusObserved],
        status[idma_abi::kStatusExpected]);
}

// A timeout code may leave a command in flight, so the next run needs a fresh session (MIM-601).
std::string idma_code_note(uint32_t code) {
    return code == idma_abi::kCodeVcSpaceTimeout || code == idma_abi::kCodeAckTimeout ||
                   code == idma_abi::kCodeDataTimeout
               ? kFreshSessionNote
               : "";
}

std::string cce_skip_reason(uint32_t cce_index) {
    if (const std::string watcher_reason = mimir_watcher_skip_reason(); !watcher_reason.empty()) {
        return watcher_reason;
    }
    const MimirTopology topology = topology_from(configured_soc_desc());
    if (cce_index >= topology.num_cces) {
        return fmt::format("Descriptor exposes {} CCEs; CCE {} is not present.", topology.num_cces, cce_index);
    }
    return "";
}

std::string multi_hart_skip_reason() {
    return kTwoHartAmoPassed ? "" : "Needs MimirAtomicProbe.TwoHartAmo to pass on this descriptor (kTwoHartAmoPassed).";
}

uint32_t multi_hart_count() { return kEightHartLaunchPassed ? abi::kMaxParticipants : 2; }

uint32_t leg_seed(uint32_t tag) { return mimir_test_seed() ^ (tag * 0x9E3779B9u); }

std::vector<uint32_t> seeded_words(size_t num_words, uint32_t seed, uint32_t mask = 0xFFFFFFFFu) {
    std::vector<uint32_t> words = make_pattern(MimirPattern::Prng, num_words, 0, seed);
    for (uint32_t& word : words) {
        word &= mask;
    }
    return words;
}

uint64_t width_mask(bool doubleword) { return doubleword ? ~0ull : 0xFFFFFFFFull; }

uint64_t get_slot(const std::vector<uint32_t>& words, uint64_t byte_offset, bool doubleword) {
    const size_t word = byte_offset / sizeof(uint32_t);
    return doubleword ? (static_cast<uint64_t>(words[word + 1]) << 32) | words[word] : words[word];
}

void set_slot(std::vector<uint32_t>& words, uint64_t byte_offset, bool doubleword, uint64_t value) {
    const size_t word = byte_offset / sizeof(uint32_t);
    words[word] = static_cast<uint32_t>(value);
    if (doubleword) {
        words[word + 1] = static_cast<uint32_t>(value >> 32);
    }
}

// Sum of the values a word starting at `initial` returns to `num_updates` increments of `value`, mod 2^32.
uint32_t expected_return_sum(uint32_t initial, uint32_t value, uint32_t num_updates) {
    uint32_t sum = 0;
    for (uint32_t update = 0; update < num_updates; ++update) {
        sum += initial + update * value;
    }
    return sum;
}

// The same for a word or doubleword, each returned value taken at its width and summed mod 2^64.
uint64_t expected_wide_return_sum(uint64_t initial, uint64_t value, uint32_t num_updates, bool doubleword) {
    uint64_t sum = 0;
    for (uint32_t update = 0; update < num_updates; ++update) {
        sum += (initial + update * value) & width_mask(doubleword);
    }
    return sum;
}

std::string op_name(uint32_t op) {
    static const char* const kNames[abi::kNumOps] = {
        "amoswap.w",     "amoadd.w",  "amoxor.w",  "amoand.w",    "amoor.w",       "amomin.w",      "amomax.w",
        "amominu.w",     "amomaxu.w", "amoswap.d", "amoadd.d",    "amoxor.d",      "amoand.d",      "amoor.d",
        "amomin.d",      "amomax.d",  "amominu.d", "amomaxu.d",   "lr.w/sc.w cas", "lr.d/sc.d cas", "lr.w/sc.w add",
        "lr.d/sc.d add", "cas add.w", "cas add.d", "amoadd.w.aq", "amoadd.w.rl",   "amoadd.w.aqrl", "amoadd.d.aqrl"};
    return op < abi::kNumOps ? kNames[op] : fmt::format("op_{}", op);
}

std::string op_order(uint32_t op) {
    switch (op) {
        case abi::kOpAmoAddWAq: return "aq";
        case abi::kOpAmoAddWRl: return "rl";
        case abi::kOpAmoAddWAqRl:
        case abi::kOpAmoAddDAqRl: return "aqrl";
    }
    return op >= abi::kOpLrScCasW && op <= abi::kOpCasAddD ? "lr/sc, no aq/rl" : "none";
}

// The value an operation leaves in the word (or doubleword) that held `old`.
uint64_t op_result(uint32_t op, uint64_t old, uint64_t operand, uint64_t compare) {
    const bool doubleword = abi::is_doubleword_op(op);
    const uint64_t mask = width_mask(doubleword);
    old &= mask;
    operand &= mask;
    const int64_t signed_old = doubleword ? static_cast<int64_t>(old) : static_cast<int32_t>(old);
    const int64_t signed_operand = doubleword ? static_cast<int64_t>(operand) : static_cast<int32_t>(operand);
    switch (op) {
        case abi::kOpAmoSwapW:
        case abi::kOpAmoSwapD: return operand;
        case abi::kOpAmoXorW:
        case abi::kOpAmoXorD: return old ^ operand;
        case abi::kOpAmoAndW:
        case abi::kOpAmoAndD: return old & operand;
        case abi::kOpAmoOrW:
        case abi::kOpAmoOrD: return old | operand;
        case abi::kOpAmoMinW:
        case abi::kOpAmoMinD: return signed_old < signed_operand ? old : operand;
        case abi::kOpAmoMaxW:
        case abi::kOpAmoMaxD: return signed_old > signed_operand ? old : operand;
        case abi::kOpAmoMinuW:
        case abi::kOpAmoMinuD: return std::min(old, operand);
        case abi::kOpAmoMaxuW:
        case abi::kOpAmoMaxuD: return std::max(old, operand);
        case abi::kOpLrScCasW:
        case abi::kOpLrScCasD: return old == (compare & mask) ? operand : old;
        default: return (old + operand) & mask;
    }
}

struct AmoLayout {
    uint64_t target = 0;
    uint64_t flags = 0;
    uint64_t results = 0;
    uint64_t table = 0;
};

AmoLayout amo_layout() {
    const uint64_t target = tl1_data(0);
    return {
        .target = target,
        .flags = target + kModeFlagsFromTarget,
        .results = target + kResultsFromTarget,
        .table = target + kTableFromTarget};
}

struct AmoLaunch {
    uint32_t cce_index = 0;
    uint32_t hart = 0;
    uint32_t mode = 0;
    uint64_t target_offset = 0;
    uint64_t flag_offset = 0;
    uint64_t result_offset = 0;
    uint64_t table_offset = 0;
    uint32_t count = 0;
    uint32_t value = 0;
    uint32_t num_participants = 1;
    uint32_t participant = 0;
    uint32_t num_points = 0;
    uint32_t operation = 0;
    uint32_t stride_bytes = 0;
    uint32_t leg = 0;
    uint32_t seed = 0;
    uint32_t peer_cce = 0;
    uint64_t budget_cycles = kPollBudgetCycles;
};

std::vector<uint32_t> amo_args(const AmoLaunch& launch) {
    std::vector<uint32_t> args(abi::kNumArgs, 0);
    args[abi::kArgMode] = launch.mode;
    args[abi::kArgStatusOffset] = static_cast<uint32_t>(tl1_done_block(launch.hart));
    args[abi::kArgPollBudgetLo] = static_cast<uint32_t>(launch.budget_cycles);
    args[abi::kArgPollBudgetHi] = static_cast<uint32_t>(launch.budget_cycles >> 32);
    args[abi::kArgTargetOffset] = static_cast<uint32_t>(launch.target_offset);
    args[abi::kArgCount] = launch.count;
    args[abi::kArgValue] = launch.value;
    args[abi::kArgFlagOffset] = static_cast<uint32_t>(launch.flag_offset);
    args[abi::kArgNumParticipants] = launch.num_participants;
    args[abi::kArgParticipant] = launch.participant;
    args[abi::kArgResultOffset] = static_cast<uint32_t>(launch.result_offset);
    args[abi::kArgTableOffset] = static_cast<uint32_t>(launch.table_offset);
    args[abi::kArgNumPoints] = launch.num_points;
    args[abi::kArgOperation] = launch.operation;
    args[abi::kArgStrideBytes] = launch.stride_bytes;
    args[abi::kArgLeg] = launch.leg;
    args[abi::kArgSeed] = launch.seed;
    args[abi::kArgPeerCce] = launch.peer_cce;
    return args;
}

// One launch per participant, participant p on hart p of `cce_index`, all at `layout`.
std::vector<AmoLaunch> participant_launches(
    uint32_t cce_index, uint32_t mode, uint32_t num_participants, const AmoLayout& layout) {
    std::vector<AmoLaunch> launches;
    for (uint32_t participant = 0; participant < num_participants; ++participant) {
        launches.push_back(AmoLaunch{
            .cce_index = cce_index,
            .hart = participant,
            .mode = mode,
            .target_offset = layout.target,
            .flag_offset = layout.flags,
            .result_offset = layout.results,
            .table_offset = layout.table,
            .num_participants = num_participants,
            .participant = participant});
    }
    return launches;
}

struct IdmaBatch {
    uint32_t cce_index = 0;
    uint32_t hart = 0;
    uint32_t mode = idma_abi::kModeAccumulate;
    uint32_t backend_vc = idma_abi::kBackendLocalSram;
    uint32_t source_kind = idma_abi::kKindOwnTl1;
    uint32_t source_selector = 0;
    uint64_t source_offset = 0;
    uint32_t destination_kind = idma_abi::kKindOwnTl1;
    uint32_t destination_selector = 0;
    uint64_t destination_offset = 0;
    uint32_t len_bytes = 0;
    uint32_t num_commands = 1;
    uint32_t source_stride_bytes = 0;
    uint32_t destination_stride_bytes = 0;
    uint32_t first_trid = 0;
    uint32_t trid_stride = 0;
    uint32_t snoop = 0;
    uint32_t host_verified = 0;
    uint32_t format = kFormatInt32;
    uint32_t operation = kAccumulateAdd;
    uint32_t disable_saturation = 0;
    uint32_t expected_lo = 0;
    uint32_t expected_hi = 0;
    uint32_t command_buffer = 0;
    uint64_t poll_region_offset = 0;
    uint32_t poll_region_bytes = 0;
    uint64_t budget_cycles = kPollBudgetCycles;
};

std::vector<uint32_t> idma_args(const IdmaBatch& batch) {
    std::vector<uint32_t> args(idma_abi::kNumArgs, 0);
    args[idma_abi::kArgMode] = batch.mode;
    args[idma_abi::kArgBackendVc] = batch.backend_vc;
    args[idma_abi::kArgStatusOffset] = static_cast<uint32_t>(tl1_done_block(batch.hart));
    args[idma_abi::kArgLenBytes] = batch.len_bytes;
    args[idma_abi::kArgPollBudgetLo] = static_cast<uint32_t>(batch.budget_cycles);
    args[idma_abi::kArgPollBudgetHi] = static_cast<uint32_t>(batch.budget_cycles >> 32);
    args[idma_abi::kArgSrcKind] = batch.source_kind;
    args[idma_abi::kArgSrcSelector] = batch.source_selector;
    args[idma_abi::kArgSrcOffsetLo] = static_cast<uint32_t>(batch.source_offset);
    args[idma_abi::kArgSrcOffsetHi] = static_cast<uint32_t>(batch.source_offset >> 32);
    args[idma_abi::kArgDstKind] = batch.destination_kind;
    args[idma_abi::kArgDstSelector] = batch.destination_selector;
    args[idma_abi::kArgDstOffsetLo] = static_cast<uint32_t>(batch.destination_offset);
    args[idma_abi::kArgDstOffsetHi] = static_cast<uint32_t>(batch.destination_offset >> 32);
    args[idma_abi::kArgNumElements] = batch.num_commands;
    args[idma_abi::kArgSourceStrideBytes] = batch.source_stride_bytes;
    args[idma_abi::kArgTrid] = batch.first_trid;
    args[idma_abi::kArgSnoop] = batch.snoop;
    args[idma_abi::kArgDestinationStrideBytes] = batch.destination_stride_bytes;
    args[idma_abi::kArgTridStride] = batch.trid_stride;
    args[idma_abi::kArgAccumulateFormat] = batch.format;
    args[idma_abi::kArgAccumulateOp] = batch.operation;
    args[idma_abi::kArgAccumulateDisableSaturation] = batch.disable_saturation;
    args[idma_abi::kArgExpectedLo] = batch.expected_lo;
    args[idma_abi::kArgExpectedHi] = batch.expected_hi;
    args[idma_abi::kArgHostVerified] = batch.host_verified;
    args[idma_abi::kArgCommandBuffer] = batch.command_buffer;
    args[idma_abi::kArgPollRegionOffset] = static_cast<uint32_t>(batch.poll_region_offset);
    args[idma_abi::kArgPollRegionBytes] = batch.poll_region_bytes;
    return args;
}

// An accumulate batch on `hart` of `cce_index` from the start of the hart's data area into its destination area.
IdmaBatch own_accumulate(uint32_t cce_index, uint32_t hart) {
    return IdmaBatch{
        .cce_index = cce_index,
        .hart = hart,
        .source_selector = cce_index,
        .source_offset = tl1_data(hart),
        .destination_selector = cce_index,
        .destination_offset = tl1_data(hart) + kAccumulateDestinationFromData};
}

// Every kernel of one Program and the done magic of its status line.
struct ProgramLaunches {
    std::vector<CceKernelLaunch> kernels;
    std::vector<uint32_t> done_magics;
};

void add_amo_kernel(IDevice* device, const AmoLaunch& launch, ProgramLaunches& launches) {
    write_cce_tl1(
        device,
        launch.cce_index,
        tl1_done_block(launch.hart),
        std::vector<uint32_t>(abi::kStatusWords, abi::kStatusPoison));
    launches.kernels.push_back(
        {.cce_index = launch.cce_index,
         .kernel_path = kAmoOpsKernel,
         .runtime_args = amo_args(launch),
         .hart = launch.hart});
    launches.done_magics.push_back(abi::kDoneMagic);
}

void add_idma_kernel(IDevice* device, const IdmaBatch& batch, ProgramLaunches& launches) {
    write_cce_tl1(
        device,
        batch.cce_index,
        tl1_done_block(batch.hart),
        std::vector<uint32_t>(idma_abi::kStatusWords, idma_abi::kStatusPoison));
    launches.kernels.push_back(
        {.cce_index = batch.cce_index,
         .kernel_path = kIdmaCopyKernel,
         .runtime_args = idma_args(batch),
         .hart = batch.hart});
    launches.done_magics.push_back(idma_abi::kDoneMagic);
}

// Runs one Program; each status line in launch order, or nullopt after an ADD_FAILURE on the host deadline.
std::optional<std::vector<std::vector<uint32_t>>> run_program(
    IDevice* device, const ProgramLaunches& launches, const std::string& label) {
    log_info(tt::LogTest, "MM42 {}", label);
    Program program = CreateProgram();
    std::optional<std::vector<std::vector<uint32_t>>> statuses = run_cce_kernels_with_deadline(
        device, program, launches.kernels, launches.done_magics, abi::kStatusWords, kHostDeadline);
    if (!statuses.has_value()) {
        ADD_FAILURE() << label << ": no status after " << kHostDeadline.count() << " s" << kFreshSessionNote;
    }
    return statuses;
}

// Zeroes the flag lines of every distinct (CCE, flag offset), then runs the launches as one Program.
std::optional<std::vector<std::vector<uint32_t>>> run_amo(
    IDevice* device, const std::vector<AmoLaunch>& launches, const std::string& label) {
    ProgramLaunches program_launches;
    std::set<std::pair<uint32_t, uint64_t>> zeroed;
    for (const AmoLaunch& launch : launches) {
        if (launch.flag_offset != 0 && zeroed.insert({launch.cce_index, launch.flag_offset}).second) {
            write_cce_tl1(
                device,
                launch.cce_index,
                launch.flag_offset,
                std::vector<uint32_t>(2 * abi::kMaxParticipants * abi::kLineBytes / sizeof(uint32_t), 0));
        }
        add_amo_kernel(device, launch, program_launches);
    }
    return run_program(device, program_launches, label);
}

void expect_codes_done(
    const std::vector<std::vector<uint32_t>>& statuses,
    const std::vector<AmoLaunch>& launches,
    const std::string& label) {
    for (size_t index = 0; index < statuses.size(); ++index) {
        log_info(tt::LogTest, "MM42 {}: hart {} {}", label, launches[index].hart, status_detail(statuses[index]));
        EXPECT_EQ(statuses[index][abi::kStatusCode], abi::kCodeDone)
            << label << ": CCE" << launches[index].cce_index << " hart " << launches[index].hart << " "
            << status_detail(statuses[index]);
    }
}

// L1 accumulate arithmetic, ported from the cce-dv reference the emulator test checks against
// (software/lib/src/cce_dv.cpp at 92fa786a5489: signed_mag_add_int8 :491-545, signed_mag_add_int32 :547-596,
// signed_mag_add_uint8 :598-614, signed_mag_add_uint32 :616-633, AtomicAddIt :635-684; source first, then the old
// destination, as emu_l1_atomic.cpp:131-136). The signed formats are sign-magnitude. With saturation enabled, an
// operand at the format's maximum (0x7F/0xFF, 0x7FFFFFFF/0xFFFFFFFF) is the result; the magnitude is clamped either
// way.
uint32_t sign_magnitude_add(uint32_t a, uint32_t b, bool disable_saturation, uint32_t bits) {
    const uint32_t sign_mask = 1u << (bits - 1);
    const uint32_t magnitude_mask = sign_mask - 1;
    const uint32_t all_ones = bits == 32 ? 0xFFFFFFFFu : (1u << bits) - 1;
    if (!disable_saturation) {
        for (const uint32_t operand : {a, b}) {
            if (operand == all_ones || operand == magnitude_mask) {
                return operand;
            }
        }
    }
    const uint32_t sign_a = a & sign_mask;
    const uint32_t sign_b = b & sign_mask;
    const uint32_t magnitude_a = a & magnitude_mask;
    const uint32_t magnitude_b = b & magnitude_mask;
    uint32_t magnitude = 0;
    uint32_t sign = 0;
    if (sign_a == sign_b) {
        const uint64_t sum = static_cast<uint64_t>(magnitude_a) + magnitude_b;
        magnitude = static_cast<uint32_t>(std::min<uint64_t>(sum, magnitude_mask));
        sign = sign_a;
    } else if (magnitude_a >= magnitude_b) {
        magnitude = magnitude_a - magnitude_b;
        sign = sign_a;
    } else {
        magnitude = magnitude_b - magnitude_a;
        sign = sign_b;
    }
    return magnitude == 0 ? 0 : sign | magnitude;
}

uint32_t unsigned_saturating_add(uint32_t a, uint32_t b, bool disable_saturation, uint32_t bits) {
    const uint32_t all_ones = bits == 32 ? 0xFFFFFFFFu : (1u << bits) - 1;
    if (!disable_saturation && (a == all_ones || b == all_ones)) {
        return all_ones;
    }
    return static_cast<uint32_t>(std::min<uint64_t>(static_cast<uint64_t>(a) + b, all_ones));
}

// One 8-byte unit; nullopt for a format the reference has no rule for.
std::optional<uint64_t> accumulate_unit(
    uint32_t format, uint64_t source, uint64_t destination, bool disable_saturation) {
    uint32_t lane_bits = 0;
    bool sign_magnitude = false;
    switch (format) {
        case kFormatInt8:
            lane_bits = 8;
            sign_magnitude = true;
            break;
        case kFormatUint8: lane_bits = 8; break;
        case kFormatInt32:
            lane_bits = 32;
            sign_magnitude = true;
            break;
        case kFormatUint32: lane_bits = 32; break;
        default: return std::nullopt;
    }
    const uint64_t lane_mask = lane_bits == 32 ? 0xFFFFFFFFull : 0xFFull;
    uint64_t result = 0;
    for (uint32_t shift = 0; shift < 64; shift += lane_bits) {
        const uint32_t a = static_cast<uint32_t>((source >> shift) & lane_mask);
        const uint32_t b = static_cast<uint32_t>((destination >> shift) & lane_mask);
        const uint32_t lane = sign_magnitude ? sign_magnitude_add(a, b, disable_saturation, lane_bits)
                                             : unsigned_saturating_add(a, b, disable_saturation, lane_bits);
        result |= (static_cast<uint64_t>(lane) & lane_mask) << shift;
    }
    return result;
}

uint32_t destination_span_bytes(const IdmaBatch& batch) {
    return (batch.num_commands - 1) * batch.destination_stride_bytes + batch.len_bytes;
}

uint32_t source_span_bytes(const IdmaBatch& batch) {
    return (batch.num_commands - 1) * batch.source_stride_bytes + batch.len_bytes;
}

// Applies every command of `batch` to `destination`, the words from batch.destination_offset, with `source` the words
// from batch.source_offset. False when the model has no rule for the format and operation.
bool apply_accumulate(const IdmaBatch& batch, const std::vector<uint32_t>& source, std::vector<uint32_t>& destination) {
    if (batch.operation != kAccumulateAdd || !accumulate_unit(batch.format, 0, 0, false).has_value()) {
        return false;
    }
    for (uint32_t command = 0; command < batch.num_commands; ++command) {
        for (uint32_t unit = 0; unit < batch.len_bytes; unit += 8) {
            const uint64_t source_byte = static_cast<uint64_t>(command) * batch.source_stride_bytes + unit;
            const uint64_t destination_byte = static_cast<uint64_t>(command) * batch.destination_stride_bytes + unit;
            set_slot(
                destination,
                destination_byte,
                true,
                *accumulate_unit(
                    batch.format,
                    get_slot(source, source_byte, true),
                    get_slot(destination, destination_byte, true),
                    batch.disable_saturation != 0));
        }
    }
    return true;
}

// kModeAccumulate's data poll: the XOR of the destination span and its first word.
void set_expected(IdmaBatch& batch, const std::vector<uint32_t>& final_destination) {
    uint32_t xor_of_span = 0;
    for (uint32_t word = 0; word < destination_span_bytes(batch) / sizeof(uint32_t); ++word) {
        xor_of_span ^= final_destination[word];
    }
    batch.expected_lo = xor_of_span;
    batch.expected_hi = final_destination.front();
}

// kModeAccumulate's data poll over a poll region: every hart polls `region` (the words from region_offset) whole.
void set_expected_region(IdmaBatch& batch, uint64_t region_offset, const std::vector<uint32_t>& region) {
    batch.poll_region_offset = region_offset;
    batch.poll_region_bytes = static_cast<uint32_t>(region.size() * sizeof(uint32_t));
    batch.expected_lo = 0;
    for (const uint32_t word : region) {
        batch.expected_lo ^= word;
    }
    batch.expected_hi = region.front();
}

// Writes the source, the destination and a guard line before and after the destination span.
void write_accumulate_operands(
    IDevice* device,
    const IdmaBatch& batch,
    const std::vector<uint32_t>& source,
    const std::vector<uint32_t>& destination,
    uint32_t tag) {
    write_cce_tl1(device, batch.cce_index, batch.source_offset, source);
    write_cce_tl1(device, batch.cce_index, batch.destination_offset - kGuardBytes, make_guard(kLeadingGuardTag + tag));
    write_cce_tl1(device, batch.cce_index, batch.destination_offset, destination);
    write_cce_tl1(
        device, batch.cce_index, batch.destination_offset + destination.size() * sizeof(uint32_t), make_guard(tag));
}

void expect_accumulate_destination(
    IDevice* device,
    const IdmaBatch& batch,
    const std::vector<uint32_t>& expected,
    uint32_t tag,
    const std::string& label) {
    const uint64_t span_bytes = expected.size() * sizeof(uint32_t);
    EXPECT_EQ(
        read_cce_tl1(device, batch.cce_index, batch.destination_offset - kGuardBytes, kGuardBytes),
        make_guard(kLeadingGuardTag + tag))
        << label << ": guard before the destination overwritten";
    EXPECT_TRUE(compare_words(
        label + " destination",
        expected,
        read_cce_tl1(device, batch.cce_index, batch.destination_offset, static_cast<uint32_t>(span_bytes))));
    EXPECT_EQ(
        read_cce_tl1(device, batch.cce_index, batch.destination_offset + span_bytes, kGuardBytes), make_guard(tag))
        << label << ": guard after the destination overwritten";
}

std::string batch_config(const IdmaBatch& batch, uint32_t seed) {
    return fmt::format(
        "CCE{} hart {} cmdbuf {}, {} x {} B accumulate fmt 0b{:03b} op 0x{:x} disable_sat {}, TL1 0x{:x} (stride {}) "
        "-> "
        "TL1 0x{:x} (stride {}), VC {}, trid {} stride {}, seed 0x{:08x}",
        batch.cce_index,
        batch.hart,
        batch.command_buffer,
        batch.num_commands,
        batch.len_bytes,
        batch.format,
        batch.operation,
        batch.disable_saturation,
        batch.source_offset,
        batch.source_stride_bytes,
        batch.destination_offset,
        batch.destination_stride_bytes,
        batch.backend_vc,
        batch.first_trid,
        batch.trid_stride,
        seed);
}

class MimirAtomicCceTest : public ::testing::TestWithParam<uint32_t> {
protected:
    void SetUp() override {
        skip_unless_programmable_dram_package();
        if (const std::string skip_reason = cce_skip_reason(GetParam()); !skip_reason.empty()) {
            GTEST_SKIP() << skip_reason;
        }
    }
};

class MimirAtomicProbe : public MimirAtomicCceTest {};

// Harts 0 and 1 each do kCount amoadd.w and kCount LR/SC increments on two cached words; the totals and the sums of
// the returned values must show no lost or duplicated update. Gates every multi-hart AMO case on this CCE.
TEST_P(MimirAtomicProbe, TwoHartAmo) {
    constexpr uint32_t kNumHarts = 2;
    constexpr uint32_t kCount = 64;
    constexpr uint32_t kValue = 1;
    const uint32_t cce_index = GetParam();
    ScopedDevice device_closer;
    IDevice* device = device_closer.device();
    check_tl1_layout();
    record_descriptor_properties(soc_desc_of(device));

    const uint64_t target_offset = tl1_data(0);
    const uint64_t flag_offset = target_offset + kFlagOffsetFromTarget;
    const uint32_t seed = mimir_test_seed();
    const uint32_t amo_initial = seed * 0x9E3779B9u;
    const uint32_t lrsc_initial = amo_initial ^ 0x5A5A5A5Au;
    std::vector<uint32_t> target_lines(2 * abi::kLineBytes / sizeof(uint32_t), 0);
    target_lines[0] = amo_initial;
    target_lines[abi::kLineBytes / sizeof(uint32_t)] = lrsc_initial;
    write_cce_tl1(device, cce_index, target_offset, target_lines);
    write_cce_tl1(
        device,
        cce_index,
        flag_offset,
        std::vector<uint32_t>(2 * abi::kMaxParticipants * abi::kLineBytes / sizeof(uint32_t), 0));
    RecordProperty("seed", fmt::format("0x{:08x}", seed));
    RecordProperty(
        "operation", fmt::format("amoadd.w and LR/SC add, value {}, {} per hart, {} harts", kValue, kCount, kNumHarts));
    RecordProperty("target", fmt::format("CCE{} TL1 0x{:x} (cached, 64 B aligned)", cce_index, target_offset));

    std::vector<CceKernelLaunch> launches;
    for (uint32_t hart = 0; hart < kNumHarts; ++hart) {
        std::vector<uint32_t> args(abi::kNumArgs, 0);
        args[abi::kArgMode] = abi::kModeTwoHartIncrement;
        args[abi::kArgStatusOffset] = static_cast<uint32_t>(tl1_done_block(hart));
        args[abi::kArgPollBudgetLo] = static_cast<uint32_t>(kPollBudgetCycles);
        args[abi::kArgPollBudgetHi] = static_cast<uint32_t>(kPollBudgetCycles >> 32);
        args[abi::kArgTargetOffset] = static_cast<uint32_t>(target_offset);
        args[abi::kArgCount] = kCount;
        args[abi::kArgValue] = kValue;
        args[abi::kArgFlagOffset] = static_cast<uint32_t>(flag_offset);
        args[abi::kArgNumParticipants] = kNumHarts;
        args[abi::kArgParticipant] = hart;
        write_cce_tl1(
            device, cce_index, tl1_done_block(hart), std::vector<uint32_t>(abi::kStatusWords, abi::kStatusPoison));
        launches.push_back({.cce_index = cce_index, .kernel_path = kAmoOpsKernel, .runtime_args = args, .hart = hart});
    }
    const std::string label = fmt::format(
        "TwoHartAmo on CCE{}: harts 0-{}, {} amoadd.w and {} LR/SC adds of {} each, seed 0x{:08x}",
        cce_index,
        kNumHarts - 1,
        kCount,
        kCount,
        kValue,
        seed);
    log_info(tt::LogTest, "MM42 {}", label);
    Program program = CreateProgram();
    const std::optional<std::vector<std::vector<uint32_t>>> statuses = run_cce_kernels_with_deadline(
        device,
        program,
        launches,
        std::vector<uint32_t>(launches.size(), abi::kDoneMagic),
        abi::kStatusWords,
        kHostDeadline);
    ASSERT_TRUE(statuses.has_value()) << label << ": no status after " << kHostDeadline.count() << " s"
                                      << kFreshSessionNote;

    uint32_t amo_return_sum = 0;
    uint32_t lrsc_return_sum = 0;
    for (uint32_t hart = 0; hart < kNumHarts; ++hart) {
        const std::vector<uint32_t>& status = (*statuses)[hart];
        log_info(tt::LogTest, "MM42 {}: hart {} {}", label, hart, status_detail(status));
        RecordProperty(fmt::format("hart{}_status", hart), status_detail(status));
        EXPECT_EQ(status[abi::kStatusCode], abi::kCodeDone)
            << label << ": hart " << hart << " " << status_detail(status);
        amo_return_sum += status[abi::kStatusAmoReturnSum];
        lrsc_return_sum += status[abi::kStatusLrScReturnSum];
    }
    const uint32_t num_updates = kNumHarts * kCount;
    const uint32_t amo_expected = amo_initial + num_updates * kValue;
    const uint32_t lrsc_expected = lrsc_initial + num_updates * kValue;
    const std::vector<uint32_t>& first = statuses->front();
    EXPECT_EQ(first[abi::kStatusAmoFinal], amo_expected) << label << ": amoadd.w total (lost or duplicated update)";
    EXPECT_EQ(first[abi::kStatusLrScFinal], lrsc_expected) << label << ": LR/SC total (lost or duplicated update)";
    EXPECT_EQ(amo_return_sum, expected_return_sum(amo_initial, kValue, num_updates))
        << label << ": amoadd.w returned values are not each prior total exactly once";
    EXPECT_EQ(lrsc_return_sum, expected_return_sum(lrsc_initial, kValue, num_updates))
        << label << ": LR/SC returned values are not each prior total exactly once";
    const std::vector<uint32_t> host_view = read_cce_tl1(device, cce_index, target_offset, 2 * abi::kLineBytes);
    EXPECT_EQ(host_view[0], amo_expected) << label << ": host view of the amoadd.w word after the flush";
    EXPECT_EQ(host_view[abi::kLineBytes / sizeof(uint32_t)], lrsc_expected)
        << label << ": host view of the LR/SC word after the flush";
}

INSTANTIATE_TEST_SUITE_P(MM42, MimirAtomicProbe, ::testing::ValuesIn(mimir_contract::cce_indices()), cce_param_name);

struct SweepPoint {
    uint32_t value = 0;
    uint32_t count = 0;
};

struct SweepLeg {
    std::string name;
    uint32_t operation = abi::kOpAmoAddW;
    uint32_t num_participants = 1;
    uint32_t stride_bytes = 0;
    std::vector<SweepPoint> points;
    // Start each word just below its width's maximum so the increments wrap halfway through.
    bool wrap = false;
};

// Runs one kModeSweep launch and checks every word's total, the sums of the returned values (each prior total exactly
// once) and that no other word of the swept region changed.
void run_sweep_leg(IDevice* device, uint32_t cce_index, const SweepLeg& leg, uint32_t tag) {
    const AmoLayout layout = amo_layout();
    const uint32_t seed = leg_seed(tag);
    const bool doubleword = abi::is_doubleword_op(leg.operation);
    const uint32_t num_points = static_cast<uint32_t>(leg.points.size());
    const uint32_t num_words = leg.stride_bytes == 0 ? 1 : leg.num_participants;
    const std::string label = fmt::format(
        "{} on CCE{}: {} x {}, {} participants, stride {} B, {} points{}, seed 0x{:08x}",
        leg.name,
        cce_index,
        op_name(leg.operation),
        num_words == 1 ? "one shared word" : "own words",
        leg.num_participants,
        leg.stride_bytes,
        num_points,
        leg.wrap ? ", wrapping" : "",
        seed);
    record_leg(leg.name, "config", label);

    std::vector<uint32_t> region = seeded_words(num_points * abi::kSweepPointBytes / sizeof(uint32_t), seed);
    std::vector<uint32_t> expected_region = region;
    std::vector<uint32_t> table;
    std::vector<std::vector<uint64_t>> initials(num_points);
    for (uint32_t point = 0; point < num_points; ++point) {
        const SweepPoint& sweep_point = leg.points[point];
        table.push_back(sweep_point.value);
        table.push_back(sweep_point.count);
        const uint32_t updates_per_word = sweep_point.count * (num_words == 1 ? leg.num_participants : 1);
        for (uint32_t word = 0; word < num_words; ++word) {
            const uint64_t byte_offset = static_cast<uint64_t>(point) * abi::kSweepPointBytes + word * leg.stride_bytes;
            const uint64_t span = static_cast<uint64_t>(updates_per_word) * sweep_point.value;
            const uint64_t initial = leg.wrap ? (width_mask(doubleword) - span / 2) & width_mask(doubleword)
                                              : get_slot(region, byte_offset, doubleword);
            set_slot(region, byte_offset, doubleword, initial);
            set_slot(expected_region, byte_offset, doubleword, (initial + span) & width_mask(doubleword));
            initials[point].push_back(initial);
        }
    }
    write_cce_tl1(device, cce_index, layout.target, region);
    write_cce_tl1(device, cce_index, layout.table, table);
    const uint32_t result_words = leg.num_participants * num_points * abi::kSweepResultWords;
    write_cce_tl1(device, cce_index, layout.results, std::vector<uint32_t>(result_words, abi::kStatusPoison));

    std::vector<AmoLaunch> launches = participant_launches(cce_index, abi::kModeSweep, leg.num_participants, layout);
    for (AmoLaunch& launch : launches) {
        launch.num_points = num_points;
        launch.operation = leg.operation;
        launch.stride_bytes = leg.stride_bytes;
        launch.budget_cycles = kLongBudgetCycles;
    }
    const std::optional<std::vector<std::vector<uint32_t>>> statuses = run_amo(device, launches, label);
    if (!statuses.has_value()) {
        return;
    }
    expect_codes_done(*statuses, launches, label);
    const std::vector<uint32_t> results =
        read_cce_tl1(device, cce_index, layout.results, result_words * sizeof(uint32_t));
    uint32_t failures = 0;
    uint32_t interleaved = 0;
    uint32_t max_cas_tries = 0;
    for (uint32_t point = 0; point < num_points; ++point) {
        const SweepPoint& sweep_point = leg.points[point];
        uint64_t shared_return_sum = 0;
        std::string cycles;
        uint32_t point_failures = 0;
        uint32_t point_interleaved = 0;
        for (uint32_t participant = 0; participant < leg.num_participants; ++participant) {
            const uint32_t* result = &results[(participant * num_points + point) * abi::kSweepResultWords];
            const uint64_t return_sum = (static_cast<uint64_t>(result[abi::kSweepResultReturnSumHi]) << 32) |
                                        result[abi::kSweepResultReturnSumLo];
            cycles += fmt::format(
                "{}p{}: {} cycles, {} failures, max {} attempts, max {} CAS tries, {} interleaved returns",
                participant == 0 ? "" : "; ",
                participant,
                result[abi::kSweepResultCycles],
                result[abi::kSweepResultFailures],
                result[abi::kSweepResultMaxAttempts],
                result[abi::kSweepResultMaxCasTries],
                result[abi::kSweepResultInterleaved]);
            max_cas_tries = std::max(max_cas_tries, result[abi::kSweepResultMaxCasTries]);
            point_failures += result[abi::kSweepResultFailures];
            point_interleaved += result[abi::kSweepResultInterleaved];
            if (num_words == 1) {
                shared_return_sum += return_sum;
            } else {
                EXPECT_EQ(
                    return_sum,
                    expected_wide_return_sum(
                        initials[point][participant], sweep_point.value, sweep_point.count, doubleword))
                    << label << ": point " << point << " participant " << participant
                    << " returned values are not each prior total exactly once";
            }
        }
        if (num_words == 1) {
            EXPECT_EQ(
                shared_return_sum,
                expected_wide_return_sum(
                    initials[point][0], sweep_point.value, sweep_point.count * leg.num_participants, doubleword))
                << label << ": point " << point << " (value " << sweep_point.value << ", count " << sweep_point.count
                << ") returned values are not each prior total exactly once";
        }
        record_leg(
            leg.name,
            fmt::format("point{}", point),
            fmt::format("value {} count {}: {}", sweep_point.value, sweep_point.count, cycles));
        failures += point_failures;
        interleaved += point_interleaved;
    }
    // Interleaved returns on a shared word show the participants overlapped; with none, contention was not exercised.
    // The most CAS tries one CAS increment needed is the starvation measure.
    const std::string contention = fmt::format(
        "{} SC or compare failures, max {} CAS tries for one increment, {} interleaved returns{}",
        failures,
        max_cas_tries,
        interleaved,
        leg.num_participants == 1 || num_words > 1 ? ""
        : interleaved == 0                         ? ": no overlap seen, contention not exercised"
                                                   : ": participants overlapped");
    record_leg(leg.name, "contention", contention);
    log_info(tt::LogTest, "MM42 {}: {}", label, contention);
    EXPECT_TRUE(compare_words(
        label + " totals (lost or duplicated update) and untouched words",
        expected_region,
        read_cce_tl1(device, cce_index, layout.target, static_cast<uint32_t>(region.size() * sizeof(uint32_t)))));
}

// Runs a leg when its participant count is allowed on this descriptor; otherwise records why it did not.
void run_gated_sweep_leg(IDevice* device, uint32_t cce_index, const SweepLeg& leg, uint32_t tag) {
    if (leg.num_participants > 1 && !kTwoHartAmoPassed) {
        record_leg(leg.name, "skipped", multi_hart_skip_reason());
        log_warning(tt::LogTest, "MM42 {} on CCE{} skipped: {}", leg.name, cce_index, multi_hart_skip_reason());
        return;
    }
    run_sweep_leg(device, cce_index, leg, tag);
}

class MimirAtomicLocal : public MimirAtomicCceTest {};

// Every Zaamo op at both widths, the aq/rl orderings of amoadd, and LR/SC CAS (matching and failing compare) and
// increments, each once on its own line from one hart: returned value, final value and the line's other words.
TEST_P(MimirAtomicLocal, OpMatrix) {
    const uint32_t cce_index = GetParam();
    ScopedDevice device_closer;
    IDevice* device = device_closer.device();
    check_tl1_layout();
    record_descriptor_properties(soc_desc_of(device));
    RecordProperty(
        "unsupported",
        "sub-word AMO (amo*.b/.h, Zabha) and amocas.w/.d/.q (Zacas): not in the toolchain (__riscv_zabha / "
        "__riscv_zacas undefined), not executed; partial write: the only CCE partial-write store is the uncached "
        "partial store, a MIM-560 / GRENSV-1699 class, not executed; AMO on the uncached alias or above 4 MiB: "
        "illegal, not executed; misaligned AMO: not executed");

    struct Entry {
        uint32_t op = 0;
        bool compare_matches = true;
    };
    std::vector<Entry> entries;
    for (uint32_t op = abi::kOpAmoSwapW; op <= abi::kOpAmoMaxuD; ++op) {
        entries.push_back({op});
    }
    for (const uint32_t op : {abi::kOpLrScCasW, abi::kOpLrScCasD}) {
        entries.push_back({op, true});
        entries.push_back({op, false});
    }
    for (const uint32_t op :
         {abi::kOpLrScAddW,
          abi::kOpLrScAddD,
          abi::kOpCasAddW,
          abi::kOpCasAddD,
          abi::kOpAmoAddWAq,
          abi::kOpAmoAddWRl,
          abi::kOpAmoAddWAqRl,
          abi::kOpAmoAddDAqRl}) {
        entries.push_back({op});
    }
    ASSERT_LE(entries.size(), abi::kMaxOpEntries);
    const uint32_t num_entries = static_cast<uint32_t>(entries.size());

    const AmoLayout layout = amo_layout();
    const uint32_t seed = leg_seed(1);
    RecordProperty("seed", fmt::format("0x{:08x}", seed));
    const std::vector<uint32_t> lines = seeded_words(num_entries * abi::kLineBytes / sizeof(uint32_t), seed);
    const std::vector<uint32_t> operands = seeded_words(2 * num_entries, seed ^ 0x0BE7A2D5u);
    std::vector<uint32_t> table(num_entries * abi::kOpEntryWords, abi::kStatusPoison);
    std::vector<uint32_t> expected_lines = lines;
    for (uint32_t index = 0; index < num_entries; ++index) {
        const Entry& entry = entries[index];
        const bool doubleword = abi::is_doubleword_op(entry.op);
        const uint64_t line_byte = static_cast<uint64_t>(index) * abi::kLineBytes;
        const uint64_t old = get_slot(lines, line_byte, doubleword);
        const uint64_t operand = (static_cast<uint64_t>(operands[2 * index + 1]) << 32) | operands[2 * index];
        const uint64_t compare = entry.compare_matches ? old : old ^ 1;
        uint32_t* words = &table[index * abi::kOpEntryWords];
        words[abi::kOpEntryOperation] = entry.op;
        words[abi::kOpEntryOperandLo] = static_cast<uint32_t>(operand);
        words[abi::kOpEntryOperandHi] = static_cast<uint32_t>(operand >> 32);
        words[abi::kOpEntryCompareLo] = static_cast<uint32_t>(compare);
        words[abi::kOpEntryCompareHi] = static_cast<uint32_t>(compare >> 32);
        set_slot(expected_lines, line_byte, doubleword, op_result(entry.op, old, operand, compare));
    }
    write_cce_tl1(device, cce_index, layout.target, lines);
    write_cce_tl1(device, cce_index, layout.table, table);

    AmoLaunch launch = participant_launches(cce_index, abi::kModeOpMatrix, 1, layout).front();
    launch.num_points = num_entries;
    const std::string label =
        fmt::format("OpMatrix on CCE{}: {} operations, seed 0x{:08x}", cce_index, num_entries, seed);
    const std::optional<std::vector<std::vector<uint32_t>>> statuses = run_amo(device, {launch}, label);
    ASSERT_TRUE(statuses.has_value());
    expect_codes_done(*statuses, {launch}, label);

    const std::vector<uint32_t> results =
        read_cce_tl1(device, cce_index, layout.table, static_cast<uint32_t>(table.size() * sizeof(uint32_t)));
    const std::vector<uint32_t> host_lines =
        read_cce_tl1(device, cce_index, layout.target, static_cast<uint32_t>(lines.size() * sizeof(uint32_t)));
    for (uint32_t index = 0; index < num_entries; ++index) {
        const Entry& entry = entries[index];
        const bool doubleword = abi::is_doubleword_op(entry.op);
        const uint64_t line_byte = static_cast<uint64_t>(index) * abi::kLineBytes;
        const uint32_t* words = &results[index * abi::kOpEntryWords];
        const uint64_t returned =
            (static_cast<uint64_t>(words[abi::kOpEntryReturnedHi]) << 32) | words[abi::kOpEntryReturnedLo];
        const uint64_t old = get_slot(lines, line_byte, doubleword);
        const uint64_t final_value = get_slot(host_lines, line_byte, doubleword);
        const std::string name =
            op_name(entry.op) + (entry.op == abi::kOpLrScCasW || entry.op == abi::kOpLrScCasD
                                     ? (entry.compare_matches ? " (compare matches)" : " (compare fails)")
                                     : "");
        RecordProperty(
            fmt::format("op{:02}", index),
            fmt::format(
                "{}: width {} B, order {}, alignment natural (line start), old 0x{:x}, returned 0x{:x}, final 0x{:x}, "
                "{} LR/SC attempts",
                name,
                doubleword ? 8 : 4,
                op_order(entry.op),
                old,
                returned,
                final_value,
                words[abi::kOpEntryAttempts]));
        EXPECT_EQ(returned, old) << label << ": " << name << " returned value";
        EXPECT_EQ(final_value, get_slot(expected_lines, line_byte, doubleword)) << label << ": " << name << " result";
    }
    EXPECT_TRUE(compare_words(label + " every line", expected_lines, host_lines));
}

// The atomic_semaphore_bandwidth sweep (increment 1-4, 16-64 increments, and the directed 512 x 1) as amoadd.w on one
// cached word, plus 1024 x 1 so the senders overlap: one sender, two harts and, after EightHartLaunch, all eight. Every
// point starts at a barrier. Cycles and interleaved returns (the contention evidence) are recorded only.
TEST_P(MimirAtomicLocal, Hotspot) {
    const uint32_t cce_index = GetParam();
    ScopedDevice device_closer;
    IDevice* device = device_closer.device();
    check_tl1_layout();
    record_descriptor_properties(soc_desc_of(device));
    std::vector<SweepPoint> points;
    for (uint32_t count = 16; count <= 64; count *= 2) {
        for (uint32_t value = 1; value <= 4; ++value) {
            points.push_back({value, count});
        }
    }
    points.push_back({1, 512});
    points.push_back({1, abi::kMaxCount});
    std::vector<uint32_t> sender_counts = {1, 2};
    if (kEightHartLaunchPassed) {
        sender_counts.push_back(abi::kMaxParticipants);
    }
    for (const uint32_t senders : sender_counts) {
        run_gated_sweep_leg(
            device,
            cce_index,
            SweepLeg{
                .name = fmt::format("senders{}", senders),
                .operation = abi::kOpAmoAddW,
                .num_participants = senders,
                .points = points},
            senders);
    }
}

// Every hart increments its own word: adjacent words of one line, separate doublewords of one line, separate lines.
TEST_P(MimirAtomicLocal, Distributed) {
    if (const std::string skip_reason = multi_hart_skip_reason(); !skip_reason.empty()) {
        GTEST_SKIP() << skip_reason;
    }
    const uint32_t cce_index = GetParam();
    ScopedDevice device_closer;
    IDevice* device = device_closer.device();
    check_tl1_layout();
    record_descriptor_properties(soc_desc_of(device));
    const std::vector<SweepPoint> points = {{1, 64}, {3, 32}};
    const uint32_t harts = multi_hart_count();
    run_sweep_leg(
        device,
        cce_index,
        SweepLeg{
            .name = "adjacent_words",
            .operation = abi::kOpAmoAddW,
            .num_participants = harts,
            .stride_bytes = 4,
            .points = points},
        1);
    run_sweep_leg(
        device,
        cce_index,
        SweepLeg{
            .name = "separate_doublewords",
            .operation = abi::kOpAmoAddD,
            .num_participants = harts,
            .stride_bytes = 8,
            .points = points},
        2);
    run_sweep_leg(
        device,
        cce_index,
        SweepLeg{
            .name = "separate_lines",
            .operation = abi::kOpAmoAddW,
            .num_participants = harts,
            .stride_bytes = abi::kLineBytes,
            .points = points},
        3);
}

// Increments built from LR/SC and from CAS (LR, compare, SC) on one shared word, at both widths; one hart, then every
// hart the probes allow. SC and compare failures and the most attempts one increment took are recorded.
TEST_P(MimirAtomicLocal, LrScCas) {
    const uint32_t cce_index = GetParam();
    ScopedDevice device_closer;
    IDevice* device = device_closer.device();
    check_tl1_layout();
    record_descriptor_properties(soc_desc_of(device));
    uint32_t tag = 1;
    for (const uint32_t harts : {1u, multi_hart_count()}) {
        for (const uint32_t op : {abi::kOpLrScAddW, abi::kOpLrScAddD, abi::kOpCasAddW, abi::kOpCasAddD}) {
            run_gated_sweep_leg(
                device,
                cce_index,
                SweepLeg{
                    .name = fmt::format(
                        "{}_harts{}",
                        op == abi::kOpLrScAddW   ? "lrsc_w"
                        : op == abi::kOpLrScAddD ? "lrsc_d"
                        : op == abi::kOpCasAddW  ? "cas_w"
                                                 : "cas_d",
                        harts),
                    .operation = op,
                    .num_participants = harts,
                    .points = {{1, 512}}},
                tag++);
        }
    }
}

// Counters that start just below 2^32 (word) or 2^64 (doubleword) and wrap halfway through the increments.
TEST_P(MimirAtomicLocal, Wrap) {
    const uint32_t cce_index = GetParam();
    ScopedDevice device_closer;
    IDevice* device = device_closer.device();
    check_tl1_layout();
    record_descriptor_properties(soc_desc_of(device));
    uint32_t tag = 1;
    for (const uint32_t harts : {1u, multi_hart_count()}) {
        for (const uint32_t op : {abi::kOpAmoAddW, abi::kOpAmoAddD, abi::kOpLrScAddW, abi::kOpCasAddD}) {
            run_gated_sweep_leg(
                device,
                cce_index,
                SweepLeg{
                    .name = fmt::format("op{}_harts{}", op, harts),
                    .operation = op,
                    .num_participants = harts,
                    .points = {{3, 64}},
                    .wrap = true},
                tag++);
        }
    }
}

// Message passing between two harts with an AMO release (amoswap.w.rl) and acquire (amoadd.w.aq of 0). The producer's
// data is cached, so the gating leg flushes it from L1 D$ before the release; the release-only leg records whether the
// consumer saw the data without that flush.
TEST_P(MimirAtomicLocal, Ordering) {
    if (const std::string skip_reason = multi_hart_skip_reason(); !skip_reason.empty()) {
        GTEST_SKIP() << skip_reason;
    }
    constexpr uint32_t kRounds = 64;
    const uint32_t cce_index = GetParam();
    ScopedDevice device_closer;
    IDevice* device = device_closer.device();
    check_tl1_layout();
    record_descriptor_properties(soc_desc_of(device));
    const AmoLayout layout = amo_layout();
    for (const uint32_t leg : {abi::kLegFlushRelease, abi::kLegReleaseOnly}) {
        const std::string leg_name = leg == abi::kLegFlushRelease ? "flush_release" : "release_only";
        const uint32_t seed = leg_seed(leg + 1);
        const std::string label =
            fmt::format("Ordering {} on CCE{}: {} rounds, seed 0x{:08x}", leg_name, cce_index, kRounds, seed);
        write_cce_tl1(
            device, cce_index, layout.target, std::vector<uint32_t>(3 * abi::kLineBytes / sizeof(uint32_t), 0));
        std::vector<AmoLaunch> launches = participant_launches(cce_index, abi::kModeOrdering, 2, layout);
        for (AmoLaunch& launch : launches) {
            launch.count = kRounds;
            launch.leg = leg;
            launch.seed = seed;
        }
        const std::optional<std::vector<std::vector<uint32_t>>> statuses = run_amo(device, launches, label);
        if (!statuses.has_value()) {
            return;
        }
        expect_codes_done(*statuses, launches, label);
        const std::vector<uint32_t>& consumer = (*statuses)[1];
        const std::string result = fmt::format(
            "{} of {} rounds saw stale data (first round {}: observed 0x{:08x} expected 0x{:08x})",
            consumer[abi::kStatusMismatches],
            kRounds,
            consumer[abi::kStatusFirstMismatch],
            consumer[abi::kStatusObservedFirst],
            consumer[abi::kStatusObservedSecond]);
        record_leg(leg_name, "result", result);
        log_info(tt::LogTest, "MM42 {}: {}", label, result);
        if (leg == abi::kLegFlushRelease) {
            EXPECT_EQ(consumer[abi::kStatusMismatches], 0u) << label << ": " << result;
        }
    }
}

// Runs one kModeLineState leg on hart 0; its status line, or empty after the host deadline.
std::vector<uint32_t> run_line_state(
    IDevice* device, uint32_t cce_index, uint32_t leg, uint32_t value, uint32_t seed, const std::string& label) {
    AmoLaunch launch = participant_launches(cce_index, abi::kModeLineState, 1, amo_layout()).front();
    launch.flag_offset = 0;
    launch.leg = leg;
    launch.value = value;
    launch.seed = seed;
    const std::optional<std::vector<std::vector<uint32_t>>> statuses = run_amo(device, {launch}, label);
    if (!statuses.has_value()) {
        return {};
    }
    expect_codes_done(*statuses, {launch}, label);
    return statuses->front();
}

// An AMO on a clean cached line, on a dirty one, and after flushing the dirty line; then on a line an external
// writer (the host, not snooped) changed in TL1 while the hart held it cached: with L2 INVALIDATE64 and cdiscard first
// (gating) and without (recorded). The control line, read cached by the same warm launch and also rewritten by the
// host, shows whether the hart's cache still held its lines at the AMO.
TEST_P(MimirAtomicLocal, CleanDirtyLine) {
    constexpr uint32_t kValue = 5;
    const uint32_t cce_index = GetParam();
    ScopedDevice device_closer;
    IDevice* device = device_closer.device();
    check_tl1_layout();
    record_descriptor_properties(soc_desc_of(device));
    const AmoLayout layout = amo_layout();
    const uint32_t line_words = abi::kLineBytes / sizeof(uint32_t);

    for (const uint32_t leg : {abi::kLegClean, abi::kLegDirty, abi::kLegFlushBeforeAtomic}) {
        const std::string leg_name = leg == abi::kLegClean   ? "clean"
                                     : leg == abi::kLegDirty ? "dirty"
                                                             : "flush_before_atomic";
        const uint32_t seed = leg_seed(leg + 1);
        const std::string label = fmt::format("CleanDirtyLine {} on CCE{}, seed 0x{:08x}", leg_name, cce_index, seed);
        const std::vector<uint32_t> old_line = seeded_words(2 * line_words, seed);
        write_cce_tl1(device, cce_index, layout.target, old_line);
        const std::vector<uint32_t> status = run_line_state(device, cce_index, leg, kValue, seed, label);
        if (status.empty()) {
            return;
        }
        std::vector<uint32_t> expected(old_line.begin(), old_line.begin() + line_words);
        expected[0] += kValue;
        if (leg != abi::kLegClean) {
            for (uint32_t word = 1; word < line_words; ++word) {
                expected[word] = seed + word;
            }
        }
        record_leg(leg_name, "status", status_detail(status));
        EXPECT_EQ(status[abi::kStatusReturnedLo], old_line[0]) << label << ": value the AMO returned";
        EXPECT_TRUE(compare_words(
            label + " line after the flush",
            expected,
            read_cce_tl1(device, cce_index, layout.target, abi::kLineBytes)));
    }

    for (const uint32_t leg : {abi::kLegInvalidateAfterWrite, abi::kLegAfterWrite}) {
        const bool invalidates = leg == abi::kLegInvalidateAfterWrite;
        const std::string leg_name = invalidates ? "host_write_invalidate" : "host_write_no_invalidate";
        const uint32_t seed = leg_seed(leg + 1);
        const std::string label = fmt::format("CleanDirtyLine {} on CCE{}, seed 0x{:08x}", leg_name, cce_index, seed);
        const std::vector<uint32_t> old_lines = seeded_words(2 * line_words, seed);
        const std::vector<uint32_t> new_lines = seeded_words(2 * line_words, seed ^ 0x00FF00FFu);
        write_cce_tl1(device, cce_index, layout.target, old_lines);
        const std::vector<uint32_t> warm = run_line_state(device, cce_index, abi::kLegWarm, 0, seed, label + " warm");
        if (warm.empty()) {
            return;
        }
        EXPECT_EQ(warm[abi::kStatusObservedFirst], old_lines[0]) << label << ": warm read of the target line";
        write_cce_tl1(device, cce_index, layout.target, new_lines);
        const std::vector<uint32_t> status = run_line_state(device, cce_index, leg, kValue, seed, label);
        if (status.empty()) {
            return;
        }
        const uint32_t returned = status[abi::kStatusReturnedLo];
        const uint32_t control = status[abi::kStatusObservedSecond];
        const std::string result = fmt::format(
            "TL1 before the AMO 0x{:08x} (host wrote 0x{:08x}); AMO returned 0x{:08x} ({}); control line read cached "
            "0x{:08x} ({})",
            status[abi::kStatusObservedFirst],
            new_lines[0],
            returned,
            returned == new_lines[0]   ? "fresh"
            : returned == old_lines[0] ? "stale cached copy"
                                       : "neither",
            control,
            control == old_lines[line_words] ? "still cached: the hart held its lines across launches"
                                             : "not the warm value: the cache did not hold its lines, inconclusive");
        record_leg(leg_name, "result", result);
        log_info(tt::LogTest, "MM42 {}: {}", label, result);
        EXPECT_EQ(status[abi::kStatusObservedFirst], new_lines[0]) << label << ": TL1 before the AMO";
        EXPECT_EQ(control, old_lines[line_words]) << label << ": the hart did not hold its lines across launches";
        if (invalidates) {
            EXPECT_EQ(returned, new_lines[0]) << label << ": " << result;
            std::vector<uint32_t> expected(new_lines.begin(), new_lines.begin() + line_words);
            expected[0] += kValue;
            EXPECT_TRUE(compare_words(
                label + " line after the flush",
                expected,
                read_cce_tl1(device, cce_index, layout.target, abi::kLineBytes)));
        } else {
            EXPECT_EQ(returned, old_lines[0])
                << label << ": host writes are not snooped, so the AMO sees the stale line";
            EXPECT_EQ(read_cce_tl1(device, cce_index, layout.target, 4).front(), old_lines[0] + kValue)
                << label << ": word 0 after the flush";
        }
    }
}

// CleanDirtyLine's snoop-enabled leg: the sibling CCE's iDMA (snoop bit set) writes the line while this hart holds it
// cached, between launches; the AMO then runs with no invalidate. Asserts the fresh value: the iDMA snoop tag applies
// to writes from a sibling CCE into this CCE's TL1 (GRENSV-3774, GRENSV-1577); own-CCE iDMA writes into own TL1 are
// never snooped. Own hang-prone row (iDMA snoop bit).
TEST_P(MimirAtomicLocal, CleanDirtyLineSnoopedWrite) {
    constexpr uint32_t kValue = 5;
    const uint32_t cce_index = GetParam();
    const uint32_t writer_cce = sibling_cce(cce_index);
    ScopedDevice device_closer;
    IDevice* device = device_closer.device();
    check_tl1_layout();
    record_descriptor_properties(soc_desc_of(device));
    const AmoLayout layout = amo_layout();
    const uint32_t line_words = abi::kLineBytes / sizeof(uint32_t);
    const uint32_t seed = leg_seed(1);
    const std::string label = fmt::format(
        "CleanDirtyLineSnoopedWrite on CCE{}: writer CCE{} iDMA with snoop, seed 0x{:08x}",
        cce_index,
        writer_cce,
        seed);
    const std::vector<uint32_t> old_lines = seeded_words(2 * line_words, seed);
    const std::vector<uint32_t> new_line = seeded_words(line_words, seed ^ 0x00FF00FFu);
    const std::vector<uint32_t> new_control = seeded_words(line_words, seed ^ 0xFF00FF00u);
    write_cce_tl1(device, cce_index, layout.target, old_lines);
    write_cce_tl1(device, writer_cce, tl1_data(0), new_line);
    const std::vector<uint32_t> warm = run_line_state(device, cce_index, abi::kLegWarm, 0, seed, label + " warm");
    ASSERT_FALSE(warm.empty());

    IdmaBatch writer{
        .cce_index = writer_cce,
        .mode = idma_abi::kModeTridBatch,
        .backend_vc = idma_abi::kBackendRemoteSram,
        .source_selector = writer_cce,
        .source_offset = tl1_data(0),
        .destination_kind = idma_abi::kKindSramSpa,
        .destination_selector = cce_index,
        .destination_offset = layout.target,
        .len_bytes = abi::kLineBytes,
        .snoop = 1,
        .host_verified = 1};
    ProgramLaunches writer_launch;
    add_idma_kernel(device, writer, writer_launch);
    const std::optional<std::vector<std::vector<uint32_t>>> writer_status =
        run_program(device, writer_launch, label + " iDMA write");
    ASSERT_TRUE(writer_status.has_value());
    const uint32_t writer_code = writer_status->front()[idma_abi::kStatusCode];
    record_leg("writer", "status", idma_status_detail(writer_status->front()));
    ASSERT_EQ(writer_code, idma_abi::kCodeAcked)
        << label << ": " << idma_status_detail(writer_status->front()) << idma_code_note(writer_code);
    write_cce_tl1(device, cce_index, layout.target + abi::kLineBytes, new_control);

    const std::vector<uint32_t> status = run_line_state(device, cce_index, abi::kLegAfterWrite, kValue, seed, label);
    ASSERT_FALSE(status.empty());
    const uint32_t returned = status[abi::kStatusReturnedLo];
    const bool control_held = status[abi::kStatusObservedSecond] == old_lines[line_words];
    const std::string result = fmt::format(
        "TL1 before the AMO 0x{:08x} (iDMA wrote 0x{:08x}); AMO returned 0x{:08x} ({}); control line {}",
        status[abi::kStatusObservedFirst],
        new_line[0],
        returned,
        returned == new_line[0]    ? "fresh"
        : returned == old_lines[0] ? "stale cached copy"
                                   : "neither",
        control_held ? "still cached, so a fresh value means the snoop refreshed the line" : "not held, inconclusive");
    RecordProperty("result", result);
    log_info(tt::LogTest, "MM42 {}: {}", label, result);
    EXPECT_EQ(status[abi::kStatusObservedFirst], new_line[0]) << label << ": the iDMA write did not land in TL1";
    EXPECT_EQ(returned, new_line[0])
        << label << ": iDMA snoop packet tag not honoured for a sibling-CCE write (GRENSV-3774, GRENSV-1577); "
        << "seen on the mimir_2x_package emulator 2026-10-22. " << result;
}

// The model of kModeStorm / kModeSoak: replays every participant's draws over the initial region. Swap slots keep
// their initial value here; the check takes their final value from the device after testing it.
struct StormModel {
    std::vector<uint32_t> region;
    std::vector<std::set<uint64_t>> swap_values;
    uint64_t swap_token_xor = 0;
    uint64_t swap_token_sum = 0;
};

StormModel model_storm(
    const std::vector<uint32_t>& initial,
    uint32_t seed,
    uint32_t num_participants,
    uint32_t count,
    bool disjoint,
    bool random_slot) {
    StormModel model{.region = initial, .swap_values = std::vector<std::set<uint64_t>>(initial.size())};
    for (uint32_t participant = 0; participant < num_participants; ++participant) {
        uint32_t state = abi::storm_state(seed, participant);
        for (uint32_t operation = 0; operation < count; ++operation) {
            const abi::StormDraw draw = abi::storm_draw(state, random_slot);
            const uint64_t offset = (disjoint ? static_cast<uint64_t>(participant) * abi::kStormSetBytes : 0) +
                                    abi::storm_slot_offset(draw);
            const uint64_t mask = width_mask(draw.doubleword);
            const uint64_t old = get_slot(model.region, offset, draw.doubleword);
            const uint64_t operand = draw.operand & mask;
            const int64_t signed_old = draw.doubleword ? static_cast<int64_t>(old) : static_cast<int32_t>(old);
            const int64_t signed_operand =
                draw.doubleword ? static_cast<int64_t>(operand) : static_cast<int32_t>(operand);
            uint64_t next = old;
            switch (draw.op) {
                case abi::kStormAddDiscard:
                case abi::kStormAddReturn:
                case abi::kStormCas: next = (old + operand) & mask; break;
                case abi::kStormSwap:
                    model.swap_values[offset / sizeof(uint32_t)].insert(operand);
                    model.swap_token_xor ^= operand;
                    model.swap_token_sum += operand;
                    break;
                case abi::kStormXor: next = old ^ operand; break;
                case abi::kStormOr: next = old | operand; break;
                case abi::kStormAnd: next = old & operand; break;
                case abi::kStormMax: next = signed_old > signed_operand ? old : operand; break;
                case abi::kStormMin: next = signed_old < signed_operand ? old : operand; break;
                case abi::kStormMaxu: next = std::max(old, operand); break;
                case abi::kStormMinu: next = std::min(old, operand); break;
                default: break;
            }
            set_slot(model.region, offset, draw.doubleword, next);
        }
    }
    return model;
}

// Runs one kModeStorm or kModeSoak launch and checks it against the replayed model: every commutative slot's final
// value, each swap slot's final value is its initial value or a token swapped into it, the XOR and sum of every
// swapped-in and swapped-out value balance, add and CAS returns strictly increase per participant and slot, and the
// untouched bytes of every line are intact.
void run_storm_leg(
    IDevice* device,
    uint32_t cce_index,
    uint32_t mode,
    uint32_t layout_leg,
    uint32_t num_participants,
    uint32_t count,
    uint32_t tag,
    const std::string& leg_name) {
    const AmoLayout layout = amo_layout();
    const uint32_t seed = leg_seed(tag);
    const bool disjoint = layout_leg == abi::kLegDisjoint;
    const bool soak = mode == abi::kModeSoak;
    const uint32_t num_sets = disjoint ? num_participants : 1;
    const std::string label = fmt::format(
        "{} {} on CCE{}: {} participants x {} operations, {} words, seed 0x{:08x}",
        soak ? "Soak" : "Storm",
        leg_name,
        cce_index,
        num_participants,
        count,
        disjoint ? "disjoint" : "shared",
        seed);
    record_leg(leg_name, "config", label);

    std::vector<uint32_t> initial = seeded_words(num_sets * abi::kStormSetBytes / sizeof(uint32_t), seed);
    // Add and CAS slots start low enough that no increment wraps, so the returned values strictly increase.
    for (uint32_t set = 0; set < num_sets; ++set) {
        for (const uint32_t storm_class :
             {abi::storm_class_of(abi::kStormAddReturn), abi::storm_class_of(abi::kStormCas)}) {
            const uint64_t line = static_cast<uint64_t>(set) * abi::kStormSetBytes + storm_class * abi::kLineBytes;
            for (uint32_t slot = 0; slot < abi::kStormSlots; ++slot) {
                const uint64_t word_offset = line + abi::kStormWordSlotBytes + slot * 4;
                const uint64_t doubleword_offset = line + abi::kStormDoublewordSlotBytes + slot * 8;
                set_slot(initial, word_offset, false, get_slot(initial, word_offset, false) & 0x0FFFFFFFull);
                set_slot(initial, doubleword_offset, true, get_slot(initial, doubleword_offset, true) >> 4);
            }
        }
    }
    StormModel model = model_storm(initial, seed, num_participants, count, disjoint, soak);
    write_cce_tl1(device, cce_index, layout.target, initial);
    write_cce_tl1(
        device,
        cce_index,
        layout.results,
        std::vector<uint32_t>(num_participants * abi::kLineBytes / sizeof(uint32_t), abi::kStatusPoison));

    std::vector<AmoLaunch> launches = participant_launches(cce_index, mode, num_participants, layout);
    for (AmoLaunch& launch : launches) {
        launch.count = count;
        launch.leg = layout_leg;
        launch.seed = seed;
        launch.budget_cycles = kLongBudgetCycles;
    }
    const std::optional<std::vector<std::vector<uint32_t>>> statuses = run_amo(device, launches, label);
    if (!statuses.has_value()) {
        return;
    }
    expect_codes_done(*statuses, launches, label);
    const std::vector<uint32_t> results =
        read_cce_tl1(device, cce_index, layout.results, num_participants * abi::kLineBytes);
    const std::vector<uint32_t> actual =
        read_cce_tl1(device, cce_index, layout.target, static_cast<uint32_t>(initial.size() * sizeof(uint32_t)));

    uint64_t returned_xor = 0;
    uint64_t returned_sum = 0;
    std::vector<uint32_t> barrier_counts;
    uint32_t cas_failures = 0;
    uint32_t interleaved = 0;
    uint32_t max_cas_tries = 0;
    for (uint32_t participant = 0; participant < num_participants; ++participant) {
        const uint32_t* result = &results[participant * abi::kLineBytes / sizeof(uint32_t)];
        returned_xor ^=
            (static_cast<uint64_t>(result[abi::kStormResultSwapXorHi]) << 32) | result[abi::kStormResultSwapXorLo];
        returned_sum +=
            (static_cast<uint64_t>(result[abi::kStormResultSwapSumHi]) << 32) | result[abi::kStormResultSwapSumLo];
        barrier_counts.push_back(result[abi::kStormResultBarriers]);
        cas_failures += result[abi::kStormResultCasFailures];
        interleaved += result[abi::kStormResultInterleaved];
        max_cas_tries = std::max(max_cas_tries, result[abi::kStormResultMaxCasTries]);
        const std::string detail = fmt::format(
            "progress {}, add violations {}, CAS violations {}, CAS failures {}, max attempts {}, max CAS tries {}, "
            "interleaved returns {}, barriers {} (max wait {} cycles), {} cycles",
            result[abi::kStormResultProgress],
            result[abi::kStormResultAddViolations],
            result[abi::kStormResultCasViolations],
            result[abi::kStormResultCasFailures],
            result[abi::kStormResultMaxAttempts],
            result[abi::kStormResultMaxCasTries],
            result[abi::kStormResultInterleaved],
            result[abi::kStormResultBarriers],
            result[abi::kStormResultMaxBarrierCycles],
            result[abi::kStormResultCycles]);
        record_leg(leg_name, fmt::format("participant{}", participant), detail);
        EXPECT_EQ(result[abi::kStormResultProgress], count)
            << label << ": participant " << participant << " " << detail;
        EXPECT_EQ(result[abi::kStormResultAddViolations], 0u)
            << label << ": participant " << participant << " add returns did not strictly increase (duplicate or lost "
            << "update) " << detail;
        EXPECT_EQ(result[abi::kStormResultCasViolations], 0u)
            << label << ": participant " << participant << " CAS returns did not strictly increase " << detail;
    }
    const std::string contention = fmt::format(
        "{} CAS failures, max {} CAS tries for one increment, {} interleaved add/CAS returns{}",
        cas_failures,
        max_cas_tries,
        interleaved,
        disjoint           ? ""
        : interleaved == 0 ? ": no overlap seen, contention not exercised"
                           : ": participants overlapped");
    record_leg(leg_name, "contention", contention);
    log_info(tt::LogTest, "MM42 {}: {}", label, contention);
    if (soak) {
        EXPECT_TRUE(std::all_of(
            barrier_counts.begin(),
            barrier_counts.end(),
            [&](uint32_t barriers) { return barriers == barrier_counts.front(); }))
            << label << ": participants passed different numbers of barriers";
    }

    uint64_t final_xor = 0;
    uint64_t final_sum = 0;
    uint64_t initial_xor = 0;
    uint64_t initial_sum = 0;
    std::vector<uint32_t> expected = model.region;
    const uint32_t swap_class = abi::storm_class_of(abi::kStormSwap);
    for (uint32_t set = 0; set < num_sets; ++set) {
        const uint64_t line = static_cast<uint64_t>(set) * abi::kStormSetBytes + swap_class * abi::kLineBytes;
        for (const bool doubleword : {false, true}) {
            for (uint32_t slot = 0; slot < abi::kStormSlots; ++slot) {
                const uint64_t offset = line + (doubleword ? abi::kStormDoublewordSlotBytes + slot * 8
                                                           : abi::kStormWordSlotBytes + slot * 4);
                const uint64_t start = get_slot(initial, offset, doubleword);
                const uint64_t end = get_slot(actual, offset, doubleword);
                const std::set<uint64_t>& tokens = model.swap_values[offset / sizeof(uint32_t)];
                EXPECT_TRUE(end == start || tokens.count(end) != 0)
                    << label << ": swap slot at +0x" << std::hex << offset << " holds 0x" << end
                    << ", neither its initial value nor a value swapped into it" << std::dec;
                initial_xor ^= start;
                initial_sum += start;
                final_xor ^= end;
                final_sum += end;
                set_slot(expected, offset, doubleword, end);
            }
        }
    }
    EXPECT_EQ(returned_xor ^ final_xor, initial_xor ^ model.swap_token_xor)
        << label << ": XOR of swapped-out and final values does not balance the initial and swapped-in values";
    EXPECT_EQ(returned_sum + final_sum, initial_sum + model.swap_token_sum)
        << label << ": sum of swapped-out and final values does not balance the initial and swapped-in values";
    EXPECT_TRUE(compare_words(label + " final words against the model", expected, actual));
}

// Mixed AMO storm: increment (result discarded), read-and-increment, swap, CAS increment, xor, or, and, min, max,
// minu and maxu at both widths, drawn per operation, on shared words and on each hart's own words.
TEST_P(MimirAtomicLocal, Storm) {
    if (const std::string skip_reason = multi_hart_skip_reason(); !skip_reason.empty()) {
        GTEST_SKIP() << skip_reason;
    }
    constexpr uint32_t kCount = abi::kMaxCount;
    const uint32_t cce_index = GetParam();
    ScopedDevice device_closer;
    IDevice* device = device_closer.device();
    check_tl1_layout();
    record_descriptor_properties(soc_desc_of(device));
    run_storm_leg(device, cce_index, abi::kModeStorm, abi::kLegShared, multi_hart_count(), kCount, 1, "shared");
    run_storm_leg(device, cce_index, abi::kModeStorm, abi::kLegDisjoint, multi_hart_count(), kCount, 2, "disjoint");
}

// One hart makes amoadd.w on word 0 of a line while the other harts make cached ordinary stores to their own words of
// the same line. The AMO total, each writer's last store and the untouched words are checked word by word.
TEST_P(MimirAtomicLocal, NeighbourWriteRace) {
    if (const std::string skip_reason = multi_hart_skip_reason(); !skip_reason.empty()) {
        GTEST_SKIP() << skip_reason;
    }
    constexpr uint32_t kCount = abi::kMaxCount;
    constexpr uint32_t kValue = 1;
    const uint32_t cce_index = GetParam();
    ScopedDevice device_closer;
    IDevice* device = device_closer.device();
    check_tl1_layout();
    record_descriptor_properties(soc_desc_of(device));
    const AmoLayout layout = amo_layout();
    const uint32_t harts = multi_hart_count();
    const uint32_t seed = leg_seed(1);
    const std::string label = fmt::format(
        "NeighbourWriteRace on CCE{}: hart 0 {} amoadd.w, harts 1-{} {} cached stores, seed 0x{:08x}",
        cce_index,
        kCount,
        harts - 1,
        kCount,
        seed);
    RecordProperty("config", label);
    const std::vector<uint32_t> line = seeded_words(abi::kLineBytes / sizeof(uint32_t), seed);
    write_cce_tl1(device, cce_index, layout.target, line);
    std::vector<AmoLaunch> launches = participant_launches(cce_index, abi::kModeNeighbourWrite, harts, layout);
    for (AmoLaunch& launch : launches) {
        launch.count = kCount;
        launch.value = kValue;
        launch.seed = seed;
    }
    const std::optional<std::vector<std::vector<uint32_t>>> statuses = run_amo(device, launches, label);
    ASSERT_TRUE(statuses.has_value());
    expect_codes_done(*statuses, launches, label);
    const std::vector<uint32_t>& amo_status = statuses->front();
    std::vector<uint32_t> expected = line;
    expected[0] = line[0] + kCount * kValue;
    for (uint32_t hart = 1; hart < harts; ++hart) {
        expected[hart] = abi::neighbour_word(seed, hart, kCount - 1);
    }
    EXPECT_EQ(amo_status[abi::kStatusAmoReturnSum], expected_return_sum(line[0], kValue, kCount))
        << label << ": amoadd.w returned values are not each prior total exactly once";
    EXPECT_EQ(amo_status[abi::kStatusAmoFinal], expected[0]) << label << ": hart 0's view of the AMO word";
    const std::vector<uint32_t> actual = read_cce_tl1(device, cce_index, layout.target, abi::kLineBytes);
    for (uint32_t word = 0; word < actual.size(); ++word) {
        const std::string role = word == 0      ? "AMO word"
                                 : word < harts ? fmt::format("hart {}'s store word", word)
                                                : "untouched word";
        EXPECT_EQ(actual[word], expected[word]) << label << ": word " << word << " (" << role << ")";
    }
}

// A seeded random soak over the storm engine: operation, slot, width, burst length and barrier placement are drawn
// from the seed, and so are the participant count (2, or 2-8 after EightHartLaunch) and the layout.
TEST_P(MimirAtomicLocal, Soak) {
    if (const std::string skip_reason = multi_hart_skip_reason(); !skip_reason.empty()) {
        GTEST_SKIP() << skip_reason;
    }
    const uint32_t cce_index = GetParam();
    ScopedDevice device_closer;
    IDevice* device = device_closer.device();
    check_tl1_layout();
    record_descriptor_properties(soc_desc_of(device));
    const uint32_t seed = leg_seed(1);
    const uint32_t harts = kEightHartLaunchPassed ? 2 + seed % (abi::kMaxParticipants - 1) : 2;
    const uint32_t layout_leg = (seed >> 8) % 2 == 0 ? abi::kLegShared : abi::kLegDisjoint;
    RecordProperty("seed", fmt::format("0x{:08x}", seed));
    run_storm_leg(device, cce_index, abi::kModeSoak, layout_leg, harts, abi::kMaxCount, 1, "soak");
}

class MimirAtomicItnControl : public MimirAtomicCceTest {};

// Tier A's Mimir0.CCE0 -> Mimir0.CCE1 control, on every CCE as the receiver with its sibling as the writer: the writer
// hart stores a word into the receiver's TL1 over the ITN (plain SPA store), then a flag. The receiver holds the line
// cached, waits for the flag with uncached loads only, then L2 INVALIDATE64 and cdiscard, then amoadd.w, so no AMO
// runs while the ITN write may be in flight (MIM-560). The AMO must return the written word.
TEST_P(MimirAtomicItnControl, Cce0ToCce1PlainWrite) {
    constexpr uint32_t kValue = 7;
    const uint32_t receiver_cce = GetParam();
    const uint32_t writer_cce = sibling_cce(receiver_cce);
    ScopedDevice device_closer;
    IDevice* device = device_closer.device();
    check_tl1_layout();
    record_descriptor_properties(soc_desc_of(device));
    const AmoLayout layout = amo_layout();
    const uint32_t seed = leg_seed(1);
    const uint32_t written = seed ^ 0xC0FFEE00u;
    const std::string label = fmt::format(
        "ItnControl: CCE{} writes 0x{:08x} into CCE{} TL1 0x{:x}, then CCE{} amoadd.w {}, seed 0x{:08x}",
        writer_cce,
        written,
        receiver_cce,
        layout.target,
        receiver_cce,
        kValue,
        seed);
    RecordProperty("config", label);
    const std::vector<uint32_t> old_line = seeded_words(abi::kLineBytes / sizeof(uint32_t), seed);
    write_cce_tl1(device, receiver_cce, layout.target, old_line);
    const std::vector<AmoLaunch> launches = {
        AmoLaunch{
            .cce_index = writer_cce,
            .mode = abi::kModeItnWriter,
            .target_offset = layout.target,
            .flag_offset = layout.flags,
            .value = written,
            .peer_cce = receiver_cce},
        AmoLaunch{
            .cce_index = receiver_cce,
            .mode = abi::kModeItnReceiver,
            .target_offset = layout.target,
            .flag_offset = layout.flags,
            .value = kValue}};
    const std::optional<std::vector<std::vector<uint32_t>>> statuses = run_amo(device, launches, label);
    ASSERT_TRUE(statuses.has_value());
    expect_codes_done(*statuses, launches, label);
    const std::vector<uint32_t>& receiver = (*statuses)[1];
    const uint32_t cached_after_flag = receiver[abi::kStatusObservedSecond];
    const std::string snoop = fmt::format(
        "cached read after the flag, before the invalidate: 0x{:08x} ({})",
        cached_after_flag,
        cached_after_flag == old_line[0] ? "stale: the ITN write was not snooped"
        : cached_after_flag == written   ? "fresh: snooped, or the line was evicted"
                                         : "neither");
    RecordProperty("snoop", snoop);
    log_info(tt::LogTest, "MM42 {}: {}", label, snoop);
    EXPECT_EQ(receiver[abi::kStatusObservedFirst], old_line[0]) << label << ": cached read before the write";
    EXPECT_EQ(receiver[abi::kStatusReturnedLo], written) << label << ": the AMO did not see the ITN write";
    std::vector<uint32_t> expected = old_line;
    expected[0] = written + kValue;
    EXPECT_TRUE(
        compare_words(label + " line", expected, read_cce_tl1(device, receiver_cce, layout.target, abi::kLineBytes)));
}

INSTANTIATE_TEST_SUITE_P(MM42, MimirAtomicLocal, ::testing::ValuesIn(mimir_contract::cce_indices()), cce_param_name);
INSTANTIATE_TEST_SUITE_P(
    MM42, MimirAtomicItnControl, ::testing::ValuesIn(mimir_contract::cce_indices()), cce_param_name);

std::string format_name(uint32_t format) {
    switch (format) {
        case kFormatFp32: return "FP32";
        case kFormatFp16A: return "FP16_A";
        case kFormatFp16B: return "FP16_B";
        case kFormatInt32: return "INT32";
        case kFormatInt32Complement: return "INT32_COMPL";
        case kFormatUint32: return "UINT32";
        case kFormatInt8: return "INT8";
        case kFormatUint8: return "UINT8";
    }
    return fmt::format("fmt{}", format);
}

// Names what an accumulate left in the destination, by the first rule that reproduces it.
std::string classify_accumulate(
    const IdmaBatch& batch,
    const std::vector<uint32_t>& source,
    const std::vector<uint32_t>& initial,
    const std::vector<uint32_t>& observed) {
    if (observed == initial) {
        return "unchanged";
    }
    if (observed == source) {
        return "copy of the source";
    }
    for (const uint32_t format : {kFormatInt32, kFormatUint32, kFormatInt8, kFormatUint8}) {
        for (const uint32_t disable_saturation : {0u, 1u}) {
            IdmaBatch model_batch = batch;
            model_batch.format = format;
            model_batch.operation = kAccumulateAdd;
            model_batch.disable_saturation = disable_saturation;
            std::vector<uint32_t> modeled = initial;
            apply_accumulate(model_batch, source, modeled);
            if (modeled == observed) {
                return fmt::format("{} add, saturation {}", format_name(format), disable_saturation ? "off" : "on");
            }
        }
    }
    std::vector<uint32_t> twos_complement = initial;
    for (size_t word = 0; word < twos_complement.size(); ++word) {
        twos_complement[word] += source[word];
    }
    if (twos_complement == observed) {
        return "two's-complement 32-bit add";
    }
    return fmt::format(
        "other: word 0 0x{:08x} (initial 0x{:08x}, source 0x{:08x})", observed[0], initial[0], source[0]);
}

class MimirAccumProbe : public MimirAtomicCceTest {};

// One 64 B accumulate per (format, operation) from hart 0, on full-range seeded data. Op 0x9 with INT32, UINT32, INT8
// and UINT8 (both saturation settings) is checked against the model (the emulator reference runs INT32, INT8 and UINT8;
// UINT32 is commented out there and the Quasar register docs call it unsupported); INT32_COMPL, FP32, FP16_A and
// FP16_B with op 0x9, and INT32 with every other operation, are recorded by what the destination held. The first point,
// INT32 op 0x9, also records whether an accumulate acks at all on BE5. Stops at the first point that did not ack
// (MIM-601). Records only, except that every modeled INT32, INT8 and UINT8 add point must land and match the model.
// Own hang-prone row: an unsupported (format, operation) may never ack.
TEST_P(MimirAccumProbe, FmtOpMatrix) {
    const uint32_t cce_index = GetParam();
    ScopedDevice device_closer;
    IDevice* device = device_closer.device();
    check_tl1_layout();
    record_descriptor_properties(soc_desc_of(device));
    struct MatrixPoint {
        uint32_t format = 0;
        uint32_t operation = 0;
        uint32_t disable_saturation = 0;
    };
    std::vector<MatrixPoint> points;
    for (const uint32_t format : {kFormatInt32, kFormatUint32, kFormatInt8, kFormatUint8}) {
        points.push_back({format, kAccumulateAdd, 0});
        points.push_back({format, kAccumulateAdd, 1});
    }
    for (const uint32_t format : {kFormatInt32Complement, kFormatFp32, kFormatFp16A, kFormatFp16B}) {
        points.push_back({format, kAccumulateAdd, 0});
    }
    for (uint32_t operation = 0; operation < 16; ++operation) {
        if (operation != kAccumulateAdd) {
            points.push_back({kFormatInt32, operation, 0});
        }
    }
    const uint32_t line_words = abi::kLineBytes / sizeof(uint32_t);
    for (size_t index = 0; index < points.size(); ++index) {
        const MatrixPoint& point = points[index];
        const uint32_t tag = static_cast<uint32_t>(index + 1);
        const uint32_t seed = leg_seed(tag);
        IdmaBatch batch = own_accumulate(cce_index, 0);
        batch.len_bytes = abi::kLineBytes;
        batch.format = point.format;
        batch.operation = point.operation;
        batch.disable_saturation = point.disable_saturation;
        batch.first_trid = 1;
        const std::vector<uint32_t> source = seeded_words(line_words, seed);
        const std::vector<uint32_t> initial = seeded_words(line_words, seed ^ 0x0D0D0D0Du);
        std::vector<uint32_t> expected = initial;
        const bool modeled = apply_accumulate(batch, source, expected);
        if (!modeled) {
            IdmaBatch poll_batch = batch;
            poll_batch.format = kFormatInt32;
            poll_batch.operation = kAccumulateAdd;
            apply_accumulate(poll_batch, source, expected);
            batch.budget_cycles = kPollBudgetCycles / 4;
        }
        set_expected(batch, expected);
        const std::string point_name = fmt::format(
            "{}_op{}_sat{}", format_name(point.format), point.operation, point.disable_saturation ? "off" : "on");
        const std::string label = fmt::format("FmtOpMatrix {}: {}", point_name, batch_config(batch, seed));
        write_accumulate_operands(device, batch, source, initial, tag);
        ProgramLaunches launches;
        add_idma_kernel(device, batch, launches);
        const std::optional<std::vector<std::vector<uint32_t>>> statuses = run_program(device, launches, label);
        ASSERT_TRUE(statuses.has_value());
        const std::vector<uint32_t>& status = statuses->front();
        const uint32_t code = status[idma_abi::kStatusCode];
        const std::vector<uint32_t> observed =
            read_cce_tl1(device, cce_index, batch.destination_offset, abi::kLineBytes);
        const std::string classification = modeled && observed == expected
                                               ? "matches the model"
                                               : classify_accumulate(batch, source, initial, observed);
        const std::string result =
            fmt::format("{}{}; {}", modeled ? "" : "no model: ", classification, idma_status_detail(status));
        RecordProperty(point_name, result);
        log_info(tt::LogTest, "MM42 {}: {}", label, result);
        if (index == 0) {
            RecordProperty(
                "accumulate_acks_on_be5", (status[idma_abi::kStatusFlags] & idma_abi::kFlagAcked) != 0 ? "yes" : "no");
            EXPECT_EQ(code, idma_abi::kCodeLanded) << label << ": " << result << idma_code_note(code);
        }
        const bool asserts_model = modeled && point.format != kFormatUint32;
        if (asserts_model) {
            if (index != 0) {
                EXPECT_EQ(code, idma_abi::kCodeLanded) << label << ": " << result << idma_code_note(code);
            }
            EXPECT_TRUE(compare_words(label, expected, observed));
        }
        if ((status[idma_abi::kStatusFlags] & idma_abi::kFlagAcked) == 0) {
            std::string not_run;
            for (size_t later = index + 1; later < points.size(); ++later) {
                not_run += fmt::format(
                    "{}{}_op{}_sat{}",
                    later == index + 1 ? "" : ", ",
                    format_name(points[later].format),
                    points[later].operation,
                    points[later].disable_saturation ? "off" : "on");
            }
            RecordProperty("not_run", not_run.empty() ? "none" : not_run);
            ADD_FAILURE() << label << ": did not ack, so the rest of the matrix was not run (" << not_run << ")"
                          << kFreshSessionNote;
            return;
        }
    }
}

INSTANTIATE_TEST_SUITE_P(MM42, MimirAccumProbe, ::testing::ValuesIn(mimir_contract::cce_indices()), cce_param_name);

class MimirAtomicIdmaAccum : public MimirAtomicCceTest {};

// Runs the batches as one Program and checks every kernel landed. The batches must have their expected values set.
bool run_accumulate(IDevice* device, const std::vector<IdmaBatch>& batches, const std::string& label) {
    ProgramLaunches launches;
    for (const IdmaBatch& batch : batches) {
        add_idma_kernel(device, batch, launches);
    }
    const std::optional<std::vector<std::vector<uint32_t>>> statuses = run_program(device, launches, label);
    if (!statuses.has_value()) {
        return false;
    }
    for (size_t index = 0; index < batches.size(); ++index) {
        const std::vector<uint32_t>& status = (*statuses)[index];
        const uint32_t code = status[idma_abi::kStatusCode];
        log_info(tt::LogTest, "MM42 {}: hart {} {}", label, batches[index].hart, idma_status_detail(status));
        record_leg(fmt::format("hart{}", batches[index].hart), "status", idma_status_detail(status));
        EXPECT_EQ(code, idma_abi::kCodeLanded)
            << label << ": hart " << batches[index].hart << " " << idma_status_detail(status) << idma_code_note(code);
    }
    return true;
}

// Four 256 B accumulates (INT32, op 0x9) from four source blocks onto one destination block, from hart 0.
TEST_P(MimirAtomicIdmaAccum, SingleHart) {
    const uint32_t cce_index = GetParam();
    ScopedDevice device_closer;
    IDevice* device = device_closer.device();
    check_tl1_layout();
    record_descriptor_properties(soc_desc_of(device));
    const uint32_t seed = leg_seed(1);
    IdmaBatch batch = own_accumulate(cce_index, 0);
    batch.len_bytes = 256;
    batch.num_commands = 4;
    batch.source_stride_bytes = 256;
    batch.first_trid = 1;
    batch.trid_stride = 1;
    const std::vector<uint32_t> source =
        seeded_words(source_span_bytes(batch) / sizeof(uint32_t), seed, kAccumulateValueMask);
    const std::vector<uint32_t> initial =
        seeded_words(destination_span_bytes(batch) / sizeof(uint32_t), seed ^ 0x0D0D0D0Du, kAccumulateValueMask);
    std::vector<uint32_t> expected = initial;
    apply_accumulate(batch, source, expected);
    set_expected(batch, expected);
    const std::string label = "SingleHart: " + batch_config(batch, seed);
    RecordProperty("config", label);
    write_accumulate_operands(device, batch, source, initial, 1);
    ASSERT_TRUE(run_accumulate(device, {batch}, label));
    expect_accumulate_destination(device, batch, expected, 1, label);
}

uint32_t accumulate_hart_count() { return kEightHartLaunchPassed ? abi::kMaxParticipants : 2; }

// Every hart accumulates four 256 B blocks onto one shared destination. Each hart's data poll waits for the sum of
// every hart's contribution, so no hart stores its status while another's accumulate may be in flight (MIM-560).
TEST_P(MimirAtomicIdmaAccum, AllHartsShared) {
    const uint32_t cce_index = GetParam();
    ScopedDevice device_closer;
    IDevice* device = device_closer.device();
    check_tl1_layout();
    record_descriptor_properties(soc_desc_of(device));
    const uint32_t harts = accumulate_hart_count();
    const uint32_t seed = leg_seed(1);
    std::vector<IdmaBatch> batches;
    std::vector<uint32_t> initial;
    std::vector<uint32_t> expected;
    for (uint32_t hart = 0; hart < harts; ++hart) {
        IdmaBatch batch = own_accumulate(cce_index, hart);
        batch.destination_offset = tl1_data(0) + kAccumulateDestinationFromData;
        batch.len_bytes = 256;
        batch.num_commands = 4;
        batch.source_stride_bytes = 256;
        batch.first_trid = hart;
        const std::vector<uint32_t> source =
            seeded_words(source_span_bytes(batch) / sizeof(uint32_t), seed ^ hart, kAccumulateValueMask);
        write_cce_tl1(device, cce_index, batch.source_offset, source);
        if (hart == 0) {
            initial = seeded_words(
                destination_span_bytes(batch) / sizeof(uint32_t), seed ^ 0x0D0D0D0Du, kAccumulateValueMask);
            expected = initial;
        }
        apply_accumulate(batch, source, expected);
        batches.push_back(batch);
    }
    for (IdmaBatch& batch : batches) {
        set_expected(batch, expected);
    }
    const std::string label = fmt::format("AllHartsShared: {} harts, {}", harts, batch_config(batches.front(), seed));
    RecordProperty("config", label);
    write_cce_tl1(
        device, cce_index, batches.front().destination_offset - kGuardBytes, make_guard(kLeadingGuardTag + 1));
    write_cce_tl1(device, cce_index, batches.front().destination_offset, initial);
    write_cce_tl1(
        device, cce_index, batches.front().destination_offset + initial.size() * sizeof(uint32_t), make_guard(1));
    ASSERT_TRUE(run_accumulate(device, batches, label));
    expect_accumulate_destination(device, batches.front(), expected, 1, label);
}

// Every hart accumulates four 256 B blocks onto its own destination block; the blocks sit back to back from hart 0's
// destination. Every hart polls the whole region for every hart's result (kArgPollRegion*) before its status store, so
// no status store overlaps another hart's accumulate (MIM-560).
TEST_P(MimirAtomicIdmaAccum, AllHartsDisjoint) {
    constexpr uint32_t kBlockBytes = 256;
    const uint32_t cce_index = GetParam();
    ScopedDevice device_closer;
    IDevice* device = device_closer.device();
    check_tl1_layout();
    record_descriptor_properties(soc_desc_of(device));
    const uint32_t harts = accumulate_hart_count();
    const uint32_t seed = leg_seed(1);
    const uint64_t region_offset = tl1_data(0) + kAccumulateDestinationFromData;
    const uint32_t block_words = kBlockBytes / sizeof(uint32_t);
    const std::vector<uint32_t> initial = seeded_words(harts * block_words, seed ^ 0x0D0D0D0Du, kAccumulateValueMask);
    std::vector<uint32_t> expected = initial;
    std::vector<IdmaBatch> batches;
    for (uint32_t hart = 0; hart < harts; ++hart) {
        IdmaBatch batch = own_accumulate(cce_index, hart);
        batch.destination_offset = region_offset + static_cast<uint64_t>(hart) * kBlockBytes;
        batch.len_bytes = kBlockBytes;
        batch.num_commands = 4;
        batch.source_stride_bytes = kBlockBytes;
        batch.first_trid = hart;
        const std::vector<uint32_t> source =
            seeded_words(source_span_bytes(batch) / sizeof(uint32_t), seed ^ hart, kAccumulateValueMask);
        write_cce_tl1(device, cce_index, batch.source_offset, source);
        std::vector<uint32_t> block(expected.begin() + hart * block_words, expected.begin() + (hart + 1) * block_words);
        apply_accumulate(batch, source, block);
        std::copy(block.begin(), block.end(), expected.begin() + hart * block_words);
        batches.push_back(batch);
    }
    for (IdmaBatch& batch : batches) {
        set_expected_region(batch, region_offset, expected);
    }
    const std::string label = fmt::format(
        "AllHartsDisjoint: {} harts, own {} B blocks from TL1 0x{:x}, poll region {} B, {}",
        harts,
        kBlockBytes,
        region_offset,
        harts * kBlockBytes,
        batch_config(batches.front(), seed));
    RecordProperty("config", label);
    IdmaBatch region = batches.front();
    region.destination_offset = region_offset;
    write_cce_tl1(device, cce_index, region_offset - kGuardBytes, make_guard(kLeadingGuardTag + 1));
    write_cce_tl1(device, cce_index, region_offset, initial);
    write_cce_tl1(device, cce_index, region_offset + initial.size() * sizeof(uint32_t), make_guard(1));
    ASSERT_TRUE(run_accumulate(device, batches, label));
    expect_accumulate_destination(device, region, expected, 1, label);
}

// Bursts of 1, 8 and 64 back-to-back 64 B accumulates onto one line, on command buffer 0 and 1. The VC stays BE5
// (MIM-407: own TL1 to own TL1).
TEST_P(MimirAtomicIdmaAccum, BackToBack) {
    const uint32_t cce_index = GetParam();
    ScopedDevice device_closer;
    IDevice* device = device_closer.device();
    check_tl1_layout();
    record_descriptor_properties(soc_desc_of(device));
    uint32_t tag = 1;
    for (const uint32_t command_buffer : {0u, 1u}) {
        for (const uint32_t burst : {1u, 8u, idma_abi::kMaxElements}) {
            const uint32_t seed = leg_seed(tag);
            IdmaBatch batch = own_accumulate(cce_index, 0);
            batch.len_bytes = abi::kLineBytes;
            batch.num_commands = burst;
            batch.source_stride_bytes = abi::kLineBytes;
            batch.command_buffer = command_buffer;
            batch.first_trid = 2;
            const std::vector<uint32_t> source =
                seeded_words(source_span_bytes(batch) / sizeof(uint32_t), seed, kAccumulateValueMask);
            const std::vector<uint32_t> initial =
                seeded_words(abi::kLineBytes / sizeof(uint32_t), seed ^ 0x0D0D0D0Du, kAccumulateValueMask);
            std::vector<uint32_t> expected = initial;
            apply_accumulate(batch, source, expected);
            set_expected(batch, expected);
            const std::string label =
                fmt::format("BackToBack burst {} cmdbuf {}: {}", burst, command_buffer, batch_config(batch, seed));
            record_leg(fmt::format("cmdbuf{}_burst{}", command_buffer, burst), "config", label);
            write_accumulate_operands(device, batch, source, initial, tag);
            if (!run_accumulate(device, {batch}, label)) {
                return;
            }
            expect_accumulate_destination(device, batch, expected, tag, label);
            tag++;
        }
    }
}

// 64 posted 64 B accumulates onto 64 lines, issued back to back over trids 0-15. Records the highest pending count
// seen. Own hang-prone row.
TEST_P(MimirAtomicIdmaAccum, MaxOutstanding) {
    const uint32_t cce_index = GetParam();
    ScopedDevice device_closer;
    IDevice* device = device_closer.device();
    check_tl1_layout();
    record_descriptor_properties(soc_desc_of(device));
    const uint32_t seed = leg_seed(1);
    IdmaBatch batch = own_accumulate(cce_index, 0);
    batch.len_bytes = abi::kLineBytes;
    batch.num_commands = idma_abi::kMaxElements;
    batch.source_stride_bytes = abi::kLineBytes;
    batch.destination_stride_bytes = abi::kLineBytes;
    batch.first_trid = 0;
    batch.trid_stride = 1;
    const std::vector<uint32_t> source =
        seeded_words(source_span_bytes(batch) / sizeof(uint32_t), seed, kAccumulateValueMask);
    const std::vector<uint32_t> initial =
        seeded_words(destination_span_bytes(batch) / sizeof(uint32_t), seed ^ 0x0D0D0D0Du, kAccumulateValueMask);
    std::vector<uint32_t> expected = initial;
    apply_accumulate(batch, source, expected);
    set_expected(batch, expected);
    const std::string label = "MaxOutstanding: " + batch_config(batch, seed);
    RecordProperty("config", label);
    write_accumulate_operands(device, batch, source, initial, 1);
    ProgramLaunches launches;
    add_idma_kernel(device, batch, launches);
    const std::optional<std::vector<std::vector<uint32_t>>> statuses = run_program(device, launches, label);
    ASSERT_TRUE(statuses.has_value());
    const std::vector<uint32_t>& status = statuses->front();
    const uint32_t code = status[idma_abi::kStatusCode];
    RecordProperty("pending_high_water", static_cast<int>(status[idma_abi::kStatusPendingHighWater]));
    RecordProperty("vc_space_cycles", static_cast<int>(status[idma_abi::kStatusVcSpaceCycles]));
    log_info(tt::LogTest, "MM42 {}: {}", label, idma_status_detail(status));
    EXPECT_EQ(code, idma_abi::kCodeLanded) << label << ": " << idma_status_detail(status) << idma_code_note(code);
    expect_accumulate_destination(device, batch, expected, 1, label);
}

// AMO, then accumulate, then AMO on the same word, each a separate launch so nothing overlaps: amoadd.w (then a flush),
// a 64 B INT32 accumulate onto the line, then L2 INVALIDATE64, cdiscard and amoadd.w. The combined result must hold
// every step. Values stay positive and small, so two's-complement and sign-magnitude adds agree.
TEST_P(MimirAtomicIdmaAccum, MixedPathControl) {
    constexpr uint32_t kFirstValue = 3;
    constexpr uint32_t kSecondValue = 11;
    const uint32_t cce_index = GetParam();
    ScopedDevice device_closer;
    IDevice* device = device_closer.device();
    check_tl1_layout();
    record_descriptor_properties(soc_desc_of(device));
    const AmoLayout layout = amo_layout();
    const uint32_t line_words = abi::kLineBytes / sizeof(uint32_t);
    const uint32_t seed = leg_seed(1);
    const std::vector<uint32_t> initial = seeded_words(2 * line_words, seed, kAccumulateValueMask);
    write_cce_tl1(device, cce_index, layout.target, initial);
    const std::string label = fmt::format("MixedPathControl on CCE{}, seed 0x{:08x}", cce_index, seed);

    const std::vector<uint32_t> first =
        run_line_state(device, cce_index, abi::kLegClean, kFirstValue, seed, label + " first AMO");
    ASSERT_FALSE(first.empty());
    EXPECT_EQ(first[abi::kStatusReturnedLo], initial[0]) << label << ": first AMO returned value";

    IdmaBatch batch = own_accumulate(cce_index, 0);
    batch.source_offset = layout.table;
    batch.destination_offset = layout.target;
    batch.len_bytes = abi::kLineBytes;
    batch.first_trid = 1;
    const std::vector<uint32_t> source = seeded_words(line_words, seed ^ 0x5A5A5A5Au, kAccumulateValueMask);
    std::vector<uint32_t> after_accumulate(initial.begin(), initial.begin() + line_words);
    after_accumulate[0] += kFirstValue;
    apply_accumulate(batch, source, after_accumulate);
    set_expected(batch, after_accumulate);
    write_cce_tl1(device, cce_index, batch.source_offset, source);
    ASSERT_TRUE(run_accumulate(device, {batch}, label + " accumulate: " + batch_config(batch, seed)));

    const std::vector<uint32_t> second =
        run_line_state(device, cce_index, abi::kLegInvalidateAfterWrite, kSecondValue, seed, label + " second AMO");
    ASSERT_FALSE(second.empty());
    EXPECT_EQ(second[abi::kStatusObservedFirst], after_accumulate[0]) << label << ": TL1 after the accumulate";
    EXPECT_EQ(second[abi::kStatusReturnedLo], after_accumulate[0])
        << label << ": the second AMO did not see the accumulate result";
    std::vector<uint32_t> expected = after_accumulate;
    expected[0] += kSecondValue;
    EXPECT_TRUE(compare_words(
        label + " combined result", expected, read_cce_tl1(device, cce_index, layout.target, abi::kLineBytes)));
}

INSTANTIATE_TEST_SUITE_P(
    MM42, MimirAtomicIdmaAccum, ::testing::ValuesIn(mimir_contract::cce_indices()), cce_param_name);

}  // namespace tt::tt_metal::mimir_test::atomics
