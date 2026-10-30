// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

// MM-8 port of the test_scoped_lock_cache CLM rows (ScopedLockCacheCLM{InvalidateOnAcquire,FlushOnRelease,
// LockedRangeOnly}): a producer and a consumer hart on one CCE, one Program, with the scoped-lock cache ops
// (scoped_lock_cache_ops.h) on a probe region in TL1. The observer of an invalidate is the acquiring consumer's cached
// read after the producer moved TL1 under its cache; the observer of a flush is the consumer's read through the
// uncached alias, which reads TL1 past every cache. The host seeds the region, waits for both status lines with a
// wall-clock deadline, and checks the consumer's result lines. Locked lines must read NEW, as upstream. Unlocked lines
// are negative controls and must read OLD: the consumer's cached reads in both invalidate cases, which stay cached
// because the acquire invalidates only the locked lines, and the consumer's uncached reads in FlushOnRelease, which see
// TL1 because the producer discards its unlocked dirty lines instead of writing them back. FlushOnRelease also needs
// the locked lines in TL1 at the producer's first read after the release, as upstream reads them at once.

#include <gtest/gtest.h>

#include <fmt/format.h>
#include <fmt/ranges.h>
#include <tt-logger/tt-logger.hpp>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <optional>
#include <string>
#include <tuple>
#include <vector>

#include <tt-metalium/host_api.hpp>
#include <tt-metalium/tt_metal.hpp>
#include "tt_metal/tt_metal/dram/mimir_access.hpp"
#include "tt_metal/tt_metal/dram/mimir_dm_common.hpp"
#include "tt_metal/tt_metal/dram/mimir_hw_state.hpp"
#include "tt_metal/tt_metal/dram/mimir_patterns.hpp"
#include "tt_metal/tt_metal/dram/mimir_test_common.hpp"
#include "tt_metal/tt_metal/test_kernels/cce/cce_scoped_lock_abi.h"

namespace tt::tt_metal::mimir_test::dm_scoped_lock {

namespace abi = cce_scoped_lock_abi;

inline constexpr const char* kProducerKernel = "tests/tt_metal/tt_metal/test_kernels/cce/cce_scoped_lock_producer.cpp";
inline constexpr const char* kConsumerKernel = "tests/tt_metal/tt_metal/test_kernels/cce/cce_scoped_lock_consumer.cpp";
inline constexpr uint32_t kLockCce = 0;
inline constexpr uint32_t kProducerHart = 0;
inline constexpr uint32_t kConsumerHart = 1;
// A hart stops at its first timeout and the two harts' waits overlap, so a failing run spends about two budgets.
inline constexpr std::chrono::seconds kHostDeadline{400};
// Offsets in the producer hart's data area (tl1_data): the probe region, its guard, then the two flag lines.
inline constexpr uint64_t kWarmedFlagOffset = 0x400;
inline constexpr uint64_t kPublishedFlagOffset = 0x440;
inline constexpr uint32_t kResultBytes = abi::kNumResultLines * abi::kLineBytes;

static_assert(abi::kRegionBytes + kGuardBytes <= kWarmedFlagOffset);

enum class ClmCase : uint32_t { InvalidateOnAcquire, FlushOnRelease, LockedRangeOnly };

struct ClmParams {
    uint32_t mode = abi::kModeInvalidateOnAcquire;
    uint32_t lock_first_line = 0;
    uint32_t lock_num_lines = 0;
};

// The upstream cases' mode and locked lines (test_scoped_lock_cache.cpp ScopedLockCacheCLM*).
ClmParams params_for(ClmCase clm_case) {
    switch (clm_case) {
        case ClmCase::InvalidateOnAcquire: return {abi::kModeInvalidateOnAcquire, 0, 4};
        case ClmCase::FlushOnRelease: return {abi::kModeFlushOnRelease, 0, 4};
        case ClmCase::LockedRangeOnly: return {abi::kModeInvalidateOnAcquire, 3, 1};
    }
    return {};
}

std::string clm_case_label(ClmCase clm_case) {
    switch (clm_case) {
        case ClmCase::InvalidateOnAcquire: return "InvalidateOnAcquire";
        case ClmCase::FlushOnRelease: return "FlushOnRelease";
        case ClmCase::LockedRangeOnly: return "LockedRangeOnly";
    }
    return "Unknown";
}

std::string clm_case_name(const ::testing::TestParamInfo<ClmCase>& info) { return clm_case_label(info.param); }

std::vector<ClmCase> all_clm_cases() {
    return {ClmCase::InvalidateOnAcquire, ClmCase::FlushOnRelease, ClmCase::LockedRangeOnly};
}

std::string clm_skip_reason() {
    if (const std::string watcher_reason = mimir_watcher_skip_reason(); !watcher_reason.empty()) {
        return watcher_reason;
    }
    return "";
}

struct ClmLayout {
    uint64_t region_offset = 0;
    uint64_t warmed_flag_offset = 0;
    uint64_t published_flag_offset = 0;
    uint64_t result_offset = 0;
};

ClmLayout clm_layout() {
    return ClmLayout{
        .region_offset = tl1_data(kProducerHart),
        .warmed_flag_offset = tl1_data(kProducerHart) + kWarmedFlagOffset,
        .published_flag_offset = tl1_data(kProducerHart) + kPublishedFlagOffset,
        .result_offset = tl1_data(kConsumerHart)};
}

std::vector<uint32_t> kernel_args(const ClmParams& params, const ClmLayout& layout, uint64_t status_offset) {
    std::vector<uint32_t> args(abi::kNumArgs, 0);
    args[abi::kArgMode] = params.mode;
    args[abi::kArgStatusOffset] = static_cast<uint32_t>(status_offset);
    args[abi::kArgPollBudgetLo] = static_cast<uint32_t>(kPollBudgetCycles);
    args[abi::kArgPollBudgetHi] = static_cast<uint32_t>(kPollBudgetCycles >> 32);
    args[abi::kArgRegionOffset] = static_cast<uint32_t>(layout.region_offset);
    args[abi::kArgLockFirstLine] = params.lock_first_line;
    args[abi::kArgLockNumLines] = params.lock_num_lines;
    args[abi::kArgWarmedFlagOffset] = static_cast<uint32_t>(layout.warmed_flag_offset);
    args[abi::kArgPublishedFlagOffset] = static_cast<uint32_t>(layout.published_flag_offset);
    args[abi::kArgResultOffset] = static_cast<uint32_t>(layout.result_offset);
    return args;
}

// Line l's words all hold kOldBase + l, so the probe word starts OLD.
std::vector<uint32_t> old_region() {
    std::vector<uint32_t> words(abi::kRegionBytes / sizeof(uint32_t));
    for (uint32_t line = 0; line < abi::kNumLines; ++line) {
        std::fill_n(words.begin() + line * abi::kWordsPerLine, abi::kWordsPerLine, abi::kOldBase + line);
    }
    return words;
}

std::vector<uint32_t> sentinel_values(uint32_t base) {
    std::vector<uint32_t> values(abi::kNumLines);
    for (uint32_t line = 0; line < abi::kNumLines; ++line) {
        values[line] = base + line;
    }
    return values;
}

// Word l of result line `result_line`, for every probe line l.
std::vector<uint32_t> result_values(const std::vector<uint32_t>& result_words, uint32_t result_line) {
    const auto first = result_words.begin() + result_line * abi::kWordsPerLine;
    return std::vector<uint32_t>(first, first + abi::kNumLines);
}

std::vector<uint32_t> slice(const std::vector<uint32_t>& values, uint32_t first, uint32_t count) {
    return std::vector<uint32_t>(values.begin() + first, values.begin() + first + count);
}

// The values outside [first, first + count).
std::vector<uint32_t> outside_slice(const std::vector<uint32_t>& values, uint32_t first, uint32_t count) {
    std::vector<uint32_t> outside(values.begin(), values.begin() + first);
    outside.insert(outside.end(), values.begin() + first + count, values.end());
    return outside;
}

std::string mode_name(uint32_t mode) {
    switch (mode) {
        case abi::kModeInvalidateOnAcquire: return "InvalidateOnAcquire";
        case abi::kModeFlushOnRelease: return "FlushOnRelease";
    }
    return fmt::format("unknown_mode_{}", mode);
}

std::string code_name(uint32_t code) {
    switch (code) {
        case abi::kCodeDone: return "done";
        case abi::kCodePeerTimeout: return "peer_timeout";
        case abi::kCodeDataTimeout: return "data_timeout";
        case abi::kCodeBadArgs: return "bad_args";
    }
    return fmt::format("unknown_code_0x{:x}", code);
}

std::string poll_name(uint32_t poll) {
    switch (poll) {
        case abi::kPollNone: return "none";
        case abi::kPollPeer: return "peer";
        case abi::kPollData: return "data";
    }
    return fmt::format("unknown_poll_{}", poll);
}

std::string status_detail(const std::vector<uint32_t>& status) {
    const uint32_t flags = status[abi::kStatusFlags];
    return fmt::format(
        "kernel code {} flags [peer {} landed {} at first read {}], last poll {} {}, cycles wait {} poll {}, flag "
        "observed {}, kernel mismatches {} (first word {} observed 0x{:08x} expected 0x{:08x})",
        code_name(status[abi::kStatusCode]),
        (flags & abi::kFlagPeerSignalled) != 0,
        (flags & abi::kFlagLanded) != 0,
        (flags & abi::kFlagLandedAtFirstRead) != 0,
        poll_name(status[abi::kStatusLastPoll]),
        status[abi::kStatusLastPollReady] != 0 ? "done" : "timed out",
        status[abi::kStatusWaitCycles],
        status[abi::kStatusPollCycles],
        status[abi::kStatusFlagObserved],
        status[abi::kStatusMismatchedWords],
        status[abi::kStatusFirstMismatchWord],
        status[abi::kStatusObserved],
        status[abi::kStatusExpected]);
}

std::string hex_words(const std::vector<uint32_t>& words) { return fmt::format("{:#06x}", fmt::join(words, " ")); }

class MimirDmScopedLockClm : public ::testing::TestWithParam<ClmCase> {};

TEST_P(MimirDmScopedLockClm, CacheOps) {
    skip_unless_programmable_dram_package();
    const ClmCase clm_case = GetParam();
    if (const std::string skip_reason = clm_skip_reason(); !skip_reason.empty()) {
        GTEST_SKIP() << skip_reason;
    }
    ScopedDevice device_closer;
    IDevice* device = device_closer.device();
    check_tl1_layout();

    const metal_SocDescriptor& soc = soc_desc_of(device);
    record_descriptor_properties(soc);

    const ClmParams params = params_for(clm_case);
    const ClmLayout layout = clm_layout();
    const uint32_t tag = static_cast<uint32_t>(clm_case);
    const std::vector<uint32_t> region_guard = make_guard(2 * tag);
    const std::vector<uint32_t> result_guard = make_guard(2 * tag + 1);
    std::vector<uint32_t> region_fill = old_region();
    region_fill.insert(region_fill.end(), region_guard.begin(), region_guard.end());
    std::vector<uint32_t> result_fill = make_seed_pattern(tag, kResultBytes / sizeof(uint32_t));
    result_fill.insert(result_fill.end(), result_guard.begin(), result_guard.end());
    const std::vector<uint32_t> zero_line(abi::kWordsPerLine, 0);
    write_cce_tl1(device, kLockCce, layout.region_offset, region_fill);
    write_cce_tl1(device, kLockCce, layout.result_offset, result_fill);
    write_cce_tl1(device, kLockCce, layout.warmed_flag_offset, zero_line);
    write_cce_tl1(device, kLockCce, layout.published_flag_offset, zero_line);
    for (const uint32_t hart : {kProducerHart, kConsumerHart}) {
        write_cce_tl1(
            device, kLockCce, tl1_done_block(hart), std::vector<uint32_t>(abi::kStatusWords, abi::kStatusPoison));
    }

    const bool invalidate_mode = params.mode == abi::kModeInvalidateOnAcquire;
    const std::string region_label = fmt::format(
        "CCE{} TL1 probe region 0x{:x}, {} lines, locked lines {}..{}",
        kLockCce,
        layout.region_offset,
        abi::kNumLines,
        params.lock_first_line,
        params.lock_first_line + params.lock_num_lines - 1);
    const std::string result_label = fmt::format("CCE{} TL1 result lines 0x{:x}", kLockCce, layout.result_offset);
    const std::string label = fmt::format(
        "{} ({}): producer hart {}, consumer hart {}, {}",
        clm_case_label(clm_case),
        mode_name(params.mode),
        kProducerHart,
        kConsumerHart,
        region_label);
    const CoreCoord translated_core = translated_dram_core(soc, kLockCce);
    const std::string producer_destination =
        invalidate_mode ? region_label + " through the uncached alias" : region_label + " through the cache";
    const std::string consumer_source = invalidate_mode ? region_label +
                                                              " through the cache after acquire, and "
                                                              "through the uncached alias"
                                                        : region_label + " through the uncached alias";
    for (const auto& [leg, hart, leg_source, leg_destination, kernel] :
         {std::tuple{"producer", kProducerHart, std::string("NEW sentinels"), producer_destination, kProducerKernel},
          std::tuple{"consumer", kConsumerHart, consumer_source, result_label, kConsumerKernel}}) {
        record_leg(leg, "source", leg_source);
        record_leg(leg, "destination", leg_destination);
        record_leg(
            leg,
            "core",
            fmt::format("CCE{} hart {} translated ({}, {})", kLockCce, hart, translated_core.x, translated_core.y));
        record_leg(leg, "length", fmt::format("{} B ({} lines)", abi::kRegionBytes, abi::kNumLines));
        record_leg(
            leg,
            "alignment",
            fmt::format(
                "region {} result {}", layout.region_offset % abi::kLineBytes, layout.result_offset % abi::kLineBytes));
        record_leg(
            leg,
            "mode",
            fmt::format(
                "{} lock lines {}+{} (L2 INVALIDATE64 on acquire, FLUSH64 on release)",
                mode_name(params.mode),
                params.lock_first_line,
                params.lock_num_lines));
        record_leg(leg, "seed", fmt::format("OLD 0x{:x}+line, NEW 0x{:x}+line", abi::kOldBase, abi::kNewBase));
        record_leg(leg, "kernel", kernel);
    }

    Program program = CreateProgram();
    const std::optional<std::vector<std::vector<uint32_t>>> statuses = run_cce_kernels_with_deadline(
        device,
        program,
        {CceKernelLaunch{
             .cce_index = kLockCce,
             .kernel_path = kProducerKernel,
             .runtime_args = kernel_args(params, layout, tl1_done_block(kProducerHart)),
             .hart = kProducerHart},
         CceKernelLaunch{
             .cce_index = kLockCce,
             .kernel_path = kConsumerKernel,
             .runtime_args = kernel_args(params, layout, tl1_done_block(kConsumerHart)),
             .hart = kConsumerHart}},
        {abi::kProducerDoneMagic, abi::kConsumerDoneMagic},
        abi::kStatusWords,
        kHostDeadline);
    if (!statuses.has_value()) {
        const std::string timeout = fmt::format(
            "no status after {} s (host deadline): producer magic 0x{:08x}, consumer magic 0x{:08x}",
            kHostDeadline.count(),
            read_status_magic(device, kLockCce, kProducerHart),
            read_status_magic(device, kLockCce, kConsumerHart));
        record_leg("producer", "status", timeout);
        record_leg("consumer", "status", timeout);
        ADD_FAILURE() << label << ": " << timeout << kFreshSessionNote;
        return;
    }

    const std::vector<uint32_t>& producer_status = statuses->at(0);
    const std::vector<uint32_t>& consumer_status = statuses->at(1);
    const std::string producer_detail = status_detail(producer_status);
    const std::string consumer_detail = status_detail(consumer_status);
    record_leg("producer", "status", producer_detail);
    record_leg("consumer", "status", consumer_detail);

    const std::vector<uint32_t> result_words =
        read_cce_tl1(device, kLockCce, layout.result_offset, kResultBytes + kGuardBytes);
    const std::vector<uint32_t> warm = result_values(result_words, abi::kResultWarm);
    const std::vector<uint32_t> cached_after = result_values(result_words, abi::kResultCached);
    const std::vector<uint32_t> uncached_after = result_values(result_words, abi::kResultUncached);
    const std::vector<uint32_t> region_words =
        read_cce_tl1(device, kLockCce, layout.region_offset, abi::kRegionBytes + kGuardBytes);
    std::vector<uint32_t> host_probe(abi::kNumLines);
    for (uint32_t line = 0; line < abi::kNumLines; ++line) {
        host_probe[line] = region_words[line * abi::kWordsPerLine];
    }
    const std::string observed = fmt::format(
        "consumer warm [{}] cached after [{}] uncached after [{}], host probe words [{}]",
        hex_words(warm),
        hex_words(cached_after),
        hex_words(uncached_after),
        hex_words(host_probe));
    record_leg("consumer", "observed", observed);
    log_info(
        tt::LogTest,
        "MM8 scoped lock {}: producer {}; consumer {}; {}",
        label,
        producer_detail,
        consumer_detail,
        observed);

    ASSERT_EQ(producer_status[abi::kStatusCode], abi::kCodeDone) << label << ": producer " << producer_detail;
    ASSERT_EQ(consumer_status[abi::kStatusCode], abi::kCodeDone) << label << ": consumer " << consumer_detail;

    const std::vector<uint32_t> old_values = sentinel_values(abi::kOldBase);
    const std::vector<uint32_t> new_values = sentinel_values(abi::kNewBase);
    const uint32_t first = params.lock_first_line;
    const uint32_t count = params.lock_num_lines;
    if (invalidate_mode) {
        EXPECT_EQ(warm, old_values) << label << ": the consumer's cached reads before the lock did not see the seed. "
                                    << observed;
        EXPECT_EQ(uncached_after, new_values)
            << label << ": the producer's uncached stores are not in TL1. " << observed;
        EXPECT_EQ(slice(cached_after, first, count), slice(new_values, first, count))
            << label << ": a locked line still read stale through the cache after acquire. " << observed;
        EXPECT_EQ(outside_slice(cached_after, first, count), outside_slice(old_values, first, count))
            << label << ": an unlocked line missed the cache after acquire. " << observed;
    } else {
        EXPECT_NE(producer_status[abi::kStatusFlags] & abi::kFlagLandedAtFirstRead, 0u)
            << label << ": the locked lines were not in TL1 at the producer's first read after release. "
            << producer_detail;
        EXPECT_EQ(slice(uncached_after, first, count), slice(new_values, first, count))
            << label << ": a locked line did not reach TL1 after release. " << observed;
        EXPECT_EQ(outside_slice(uncached_after, first, count), outside_slice(old_values, first, count))
            << label << ": an unlocked line reached TL1 after release. " << observed;
    }
    EXPECT_TRUE(std::equal(region_guard.begin(), region_guard.end(), region_words.begin() + abi::kRegionBytes / 4))
        << label << ": guard after the probe region was overwritten";
    EXPECT_TRUE(std::equal(result_guard.begin(), result_guard.end(), result_words.begin() + kResultBytes / 4))
        << label << ": guard after the result lines was overwritten";
}

INSTANTIATE_TEST_SUITE_P(MM8, MimirDmScopedLockClm, ::testing::ValuesIn(all_clm_cases()), clm_case_name);

// MM41 scoped-lock analogues (plan increments 2-3), each on every CCE of the descriptor.

// Set by MM41/MimirCacheWriterProbe.ExternalWriters (SiblingSpa, SnoopOffInvalidate): whether a sibling CCE's SPA
// stores reach this CCE's cached copy with no invalidate.
inline constexpr bool kSiblingSpaWritesSnooped = false;
// Offsets in the region hart's data area: consumer c's warmed flag line at kMultiWarmedFlagOffset + c * 64.
inline constexpr uint64_t kMultiWarmedFlagOffset = 0x400;
inline constexpr uint64_t kMultiPublishedFlagOffset = 0x600;
static_assert(kMultiWarmedFlagOffset + abi::kMaxConsumers * abi::kLineBytes <= kMultiPublishedFlagOffset);
inline constexpr uint32_t kRingSlots = 4;
inline constexpr uint32_t kRingSlotLines = 2;
// Three trips round the ring.
inline constexpr uint32_t kRingRounds = 3 * kRingSlots;
static_assert(kRingSlots * kRingSlotLines <= abi::kNumLines);

// One kernel of an MM41 scoped-lock Program.
struct LockKernel {
    uint32_t cce_index = 0;
    uint32_t hart = kCceKernelHart;
    const char* kernel_path = kProducerKernel;
    std::vector<uint32_t> args;
    uint32_t done_magic = abi::kProducerDoneMagic;
};

std::vector<uint32_t> mm41_lock_args(uint32_t mode, uint64_t region_offset, uint32_t hart) {
    std::vector<uint32_t> args(abi::kNumArgs, 0);
    args[abi::kArgMode] = mode;
    args[abi::kArgStatusOffset] = static_cast<uint32_t>(tl1_done_block(hart));
    args[abi::kArgPollBudgetLo] = static_cast<uint32_t>(kPollBudgetCycles);
    args[abi::kArgPollBudgetHi] = static_cast<uint32_t>(kPollBudgetCycles >> 32);
    args[abi::kArgRegionOffset] = static_cast<uint32_t>(region_offset);
    return args;
}

// Runs every kernel in one Program and returns each status line in launch order; nullopt on the host deadline.
std::optional<std::vector<std::vector<uint32_t>>> run_lock_kernels(
    IDevice* device, const std::vector<LockKernel>& kernels) {
    std::vector<CceKernelLaunch> launches;
    std::vector<uint32_t> done_magics;
    for (const LockKernel& kernel : kernels) {
        write_cce_tl1(
            device,
            kernel.cce_index,
            tl1_done_block(kernel.hart),
            std::vector<uint32_t>(abi::kStatusWords, abi::kStatusPoison));
        launches.push_back(
            {.cce_index = kernel.cce_index,
             .kernel_path = kernel.kernel_path,
             .runtime_args = kernel.args,
             .hart = kernel.hart});
        done_magics.push_back(kernel.done_magic);
    }
    Program program = CreateProgram();
    return run_cce_kernels_with_deadline(device, program, launches, done_magics, abi::kStatusWords, kHostDeadline);
}

std::string mm41_lock_detail(const std::vector<uint32_t>& status) {
    return fmt::format("{}, rounds {}", status_detail(status), status[abi::kStatusRounds]);
}

// The consumer's result lines at `result_offset`: warm reads all old; cached reads after the lock new on the locked
// lines and `unlocked_cached` on the others; uncached reads `uncached` on every line.
void expect_consumer_results(
    IDevice* device,
    uint32_t cce_index,
    uint64_t result_offset,
    uint32_t lock_first_line,
    uint32_t lock_num_lines,
    const std::vector<uint32_t>& unlocked_cached,
    const std::vector<uint32_t>& uncached,
    const std::string& label) {
    const std::vector<uint32_t> result_words = read_cce_tl1(device, cce_index, result_offset, kResultBytes);
    const std::vector<uint32_t> warm = result_values(result_words, abi::kResultWarm);
    const std::vector<uint32_t> cached_after = result_values(result_words, abi::kResultCached);
    const std::vector<uint32_t> uncached_after = result_values(result_words, abi::kResultUncached);
    const std::string observed = fmt::format(
        "warm [{}] cached after [{}] uncached after [{}]",
        hex_words(warm),
        hex_words(cached_after),
        hex_words(uncached_after));
    log_info(tt::LogTest, "MM41 scoped lock {}: {}", label, observed);
    record_leg(label, "observed", observed);
    const std::vector<uint32_t> new_values = sentinel_values(abi::kNewBase);
    EXPECT_EQ(warm, sentinel_values(abi::kOldBase)) << label << ": cached reads before the lock. " << observed;
    EXPECT_EQ(uncached_after, uncached) << label << ": uncached reads after published. " << observed;
    EXPECT_EQ(slice(cached_after, lock_first_line, lock_num_lines), slice(new_values, lock_first_line, lock_num_lines))
        << label << ": a locked line read stale through the cache after acquire. " << observed;
    EXPECT_EQ(
        outside_slice(cached_after, lock_first_line, lock_num_lines),
        outside_slice(unlocked_cached, lock_first_line, lock_num_lines))
        << label << ": unlocked lines through the cache after acquire. " << observed;
}

class MimirCacheScopedLock : public ::testing::TestWithParam<uint32_t> {
protected:
    void SetUp() override {
        skip_unless_programmable_dram_package();
        if (const std::string skip_reason = mimir_watcher_skip_reason(); !skip_reason.empty()) {
            GTEST_SKIP() << skip_reason;
        }
    }
};

// One producer and seven consumers on one CCE (the DFB multi-consumer analogue). Each consumer warms the probe lines
// cached; the producer stores NEW under the lock and releases it; each consumer acquires and must read the locked
// lines NEW through its cache and the unlocked lines OLD.
TEST_P(MimirCacheScopedLock, MultiConsumer) {
    const uint32_t cce_index = GetParam();
    ScopedDevice device_closer;
    IDevice* device = device_closer.device();
    check_tl1_layout();
    record_descriptor_properties(soc_desc_of(device));

    const ClmParams params = params_for(ClmCase::InvalidateOnAcquire);
    const uint64_t region_offset = tl1_data(kProducerHart);
    const uint32_t consumer_count = abi::kMaxConsumers;
    const uint32_t tag = 0x40 + cce_index;
    std::vector<uint32_t> region_fill = old_region();
    const std::vector<uint32_t> region_guard = make_guard(tag);
    region_fill.insert(region_fill.end(), region_guard.begin(), region_guard.end());
    write_cce_tl1(device, cce_index, region_offset, region_fill);
    write_cce_tl1(
        device,
        cce_index,
        region_offset + kMultiWarmedFlagOffset,
        std::vector<uint32_t>(
            (kMultiPublishedFlagOffset - kMultiWarmedFlagOffset + abi::kLineBytes) / sizeof(uint32_t), 0));
    std::vector<LockKernel> kernels;
    std::vector<uint32_t> producer_args = mm41_lock_args(abi::kModeMultiConsumer, region_offset, kProducerHart);
    producer_args[abi::kArgLockFirstLine] = params.lock_first_line;
    producer_args[abi::kArgLockNumLines] = params.lock_num_lines;
    producer_args[abi::kArgWarmedFlagOffset] = static_cast<uint32_t>(region_offset + kMultiWarmedFlagOffset);
    producer_args[abi::kArgPublishedFlagOffset] = static_cast<uint32_t>(region_offset + kMultiPublishedFlagOffset);
    producer_args[abi::kArgConsumerCount] = consumer_count;
    kernels.push_back({.cce_index = cce_index, .hart = kProducerHart, .args = producer_args});
    for (uint32_t consumer = 0; consumer < consumer_count; ++consumer) {
        const uint32_t hart = consumer + 1;
        std::vector<uint32_t> args = producer_args;
        args[abi::kArgStatusOffset] = static_cast<uint32_t>(tl1_done_block(hart));
        args[abi::kArgConsumerIndex] = consumer;
        args[abi::kArgResultOffset] = static_cast<uint32_t>(tl1_data(hart));
        write_cce_tl1(
            device, cce_index, tl1_data(hart), make_seed_pattern(tag + hart, kResultBytes / sizeof(uint32_t)));
        kernels.push_back(
            {.cce_index = cce_index,
             .hart = hart,
             .kernel_path = kConsumerKernel,
             .args = args,
             .done_magic = abi::kConsumerDoneMagic});
    }
    const std::string label = fmt::format(
        "MultiConsumer on CCE{}: producer hart {}, consumers harts 1-{}, region TL1 0x{:x}, locked lines {}..{}",
        cce_index,
        kProducerHart,
        consumer_count,
        region_offset,
        params.lock_first_line,
        params.lock_first_line + params.lock_num_lines - 1);
    log_info(tt::LogTest, "MM41 scoped lock {}", label);
    const std::optional<std::vector<std::vector<uint32_t>>> statuses = run_lock_kernels(device, kernels);
    ASSERT_TRUE(statuses.has_value()) << label << ": no status after " << kHostDeadline.count() << " s"
                                      << kFreshSessionNote;
    const std::vector<uint32_t>& producer = statuses->front();
    log_info(tt::LogTest, "MM41 scoped lock {}: producer {}", label, mm41_lock_detail(producer));
    RecordProperty(
        "landed_at_first_read", (producer[abi::kStatusFlags] & abi::kFlagLandedAtFirstRead) != 0 ? "yes" : "no");
    ASSERT_EQ(producer[abi::kStatusCode], abi::kCodeDone) << label << ": producer " << mm41_lock_detail(producer);
    const std::vector<uint32_t> old_values = sentinel_values(abi::kOldBase);
    const std::vector<uint32_t> new_values = sentinel_values(abi::kNewBase);
    std::vector<uint32_t> tl1_values = old_values;
    std::copy_n(
        new_values.begin() + params.lock_first_line,
        params.lock_num_lines,
        tl1_values.begin() + params.lock_first_line);
    for (uint32_t consumer = 0; consumer < consumer_count; ++consumer) {
        const uint32_t hart = consumer + 1;
        const std::vector<uint32_t>& status = (*statuses)[hart];
        const std::string consumer_label = fmt::format("{} consumer hart {}", label, hart);
        log_info(tt::LogTest, "MM41 scoped lock {}: {}", consumer_label, mm41_lock_detail(status));
        EXPECT_EQ(status[abi::kStatusCode], abi::kCodeDone) << consumer_label << ": " << mm41_lock_detail(status);
        if (status[abi::kStatusCode] == abi::kCodeDone) {
            expect_consumer_results(
                device,
                cce_index,
                tl1_data(hart),
                params.lock_first_line,
                params.lock_num_lines,
                old_values,
                tl1_values,
                fmt::format("consumer{}", hart));
        }
    }
    const std::vector<uint32_t> region_words =
        read_cce_tl1(device, cce_index, region_offset, abi::kRegionBytes + kGuardBytes);
    for (uint32_t line = 0; line < abi::kNumLines; ++line) {
        EXPECT_EQ(region_words[line * abi::kWordsPerLine], tl1_values[line])
            << label << ": host probe word of line " << line;
    }
    EXPECT_TRUE(std::equal(region_guard.begin(), region_guard.end(), region_words.begin() + abi::kRegionBytes / 4))
        << label << ": guard after the probe region was overwritten";
}

// A producer fills slot r % kRingSlots of a kRingSlots-slot ring under the lock for kRingRounds rounds, gated by the
// consumer's read pointer; the consumer acquires each slot before reading it cached. A slot the consumer cached on
// its previous trip round the ring must read the new round's values.
TEST_P(MimirCacheScopedLock, RingWrap) {
    const uint32_t cce_index = GetParam();
    ScopedDevice device_closer;
    IDevice* device = device_closer.device();
    check_tl1_layout();
    record_descriptor_properties(soc_desc_of(device));

    const uint64_t region_offset = tl1_data(kProducerHart);
    const uint32_t tag = 0x48 + cce_index;
    std::vector<uint32_t> region_fill = old_region();
    const std::vector<uint32_t> region_guard = make_guard(tag);
    region_fill.insert(region_fill.end(), region_guard.begin(), region_guard.end());
    write_cce_tl1(device, cce_index, region_offset, region_fill);
    write_cce_tl1(device, cce_index, region_offset + kWarmedFlagOffset, std::vector<uint32_t>(abi::kWordsPerLine, 0));
    write_cce_tl1(
        device, cce_index, region_offset + kPublishedFlagOffset, std::vector<uint32_t>(abi::kWordsPerLine, 0));
    std::vector<uint32_t> producer_args = mm41_lock_args(abi::kModeRingWrap, region_offset, kProducerHart);
    producer_args[abi::kArgWarmedFlagOffset] = static_cast<uint32_t>(region_offset + kWarmedFlagOffset);
    producer_args[abi::kArgPublishedFlagOffset] = static_cast<uint32_t>(region_offset + kPublishedFlagOffset);
    producer_args[abi::kArgResultOffset] = static_cast<uint32_t>(tl1_data(kConsumerHart));
    producer_args[abi::kArgRingSlots] = kRingSlots;
    producer_args[abi::kArgSlotLines] = kRingSlotLines;
    producer_args[abi::kArgRounds] = kRingRounds;
    std::vector<uint32_t> consumer_args = producer_args;
    consumer_args[abi::kArgStatusOffset] = static_cast<uint32_t>(tl1_done_block(kConsumerHart));
    const std::string label = fmt::format(
        "RingWrap on CCE{}: producer hart {}, consumer hart {}, {} slots of {} lines at TL1 0x{:x}, {} rounds",
        cce_index,
        kProducerHart,
        kConsumerHart,
        kRingSlots,
        kRingSlotLines,
        region_offset,
        kRingRounds);
    log_info(tt::LogTest, "MM41 scoped lock {}", label);
    const std::optional<std::vector<std::vector<uint32_t>>> statuses = run_lock_kernels(
        device,
        {{.cce_index = cce_index, .hart = kProducerHart, .args = producer_args},
         {.cce_index = cce_index,
          .hart = kConsumerHart,
          .kernel_path = kConsumerKernel,
          .args = consumer_args,
          .done_magic = abi::kConsumerDoneMagic}});
    ASSERT_TRUE(statuses.has_value()) << label << ": no status after " << kHostDeadline.count() << " s"
                                      << kFreshSessionNote;
    const std::vector<uint32_t>& producer = (*statuses)[0];
    const std::vector<uint32_t>& consumer = (*statuses)[1];
    log_info(
        tt::LogTest,
        "MM41 scoped lock {}: producer {}; consumer {}",
        label,
        mm41_lock_detail(producer),
        mm41_lock_detail(consumer));
    ASSERT_EQ(producer[abi::kStatusCode], abi::kCodeDone) << label << ": producer " << mm41_lock_detail(producer);
    ASSERT_EQ(consumer[abi::kStatusCode], abi::kCodeDone) << label << ": consumer " << mm41_lock_detail(consumer);
    EXPECT_EQ(producer[abi::kStatusRounds], kRingRounds) << label;
    EXPECT_EQ(consumer[abi::kStatusRounds], kRingRounds) << label;
    EXPECT_EQ(consumer[abi::kStatusMismatchedWords], 0u)
        << label << ": consumer read stale slot words; first at round "
        << (consumer[abi::kStatusFirstMismatchWord] >> 8) << " word "
        << (consumer[abi::kStatusFirstMismatchWord] & 0xFF) << ": " << mm41_lock_detail(consumer);
    const uint32_t slot_words = kRingSlotLines * abi::kWordsPerLine;
    const std::vector<uint32_t> region_words =
        read_cce_tl1(device, cce_index, region_offset, kRingSlots * slot_words * sizeof(uint32_t));
    for (uint32_t slot = 0; slot < kRingSlots; ++slot) {
        const uint32_t last_round = kRingRounds - kRingSlots + slot;
        for (uint32_t word = 0; word < slot_words; ++word) {
            EXPECT_EQ(region_words[slot * slot_words + word], abi::ring_word(last_round, word))
                << label << ": host view of slot " << slot << " word " << word;
        }
    }
}

// A producer hart on the sibling CCE stores NEW to every probe word and the published flag in this CCE's TL1 by SPA,
// which this CCE's caches do not snoop by default; the consumer on this CCE warms the lines, waits until the stores
// have landed in TL1, acquires the lock and reads cached. Locked lines must read NEW; unlocked lines stay OLD unless
// the ExternalWriters probe showed sibling SPA stores are snooped.
TEST_P(MimirCacheScopedLock, NonSnoopingProducer) {
    const uint32_t cce_index = GetParam();
    ScopedDevice device_closer;
    IDevice* device = device_closer.device();
    check_tl1_layout();
    record_descriptor_properties(soc_desc_of(device));
    const uint32_t producer_cce = sibling_cce(cce_index);

    const ClmParams params = params_for(ClmCase::InvalidateOnAcquire);
    const ClmLayout layout = clm_layout();
    const uint32_t tag = 0x50 + cce_index;
    std::vector<uint32_t> region_fill = old_region();
    const std::vector<uint32_t> region_guard = make_guard(tag);
    region_fill.insert(region_fill.end(), region_guard.begin(), region_guard.end());
    write_cce_tl1(device, cce_index, layout.region_offset, region_fill);
    write_cce_tl1(device, cce_index, layout.result_offset, make_seed_pattern(tag, kResultBytes / sizeof(uint32_t)));
    write_cce_tl1(device, cce_index, layout.warmed_flag_offset, std::vector<uint32_t>(abi::kWordsPerLine, 0));
    write_cce_tl1(device, cce_index, layout.published_flag_offset, std::vector<uint32_t>(abi::kWordsPerLine, 0));
    std::vector<uint32_t> consumer_args = kernel_args(params, layout, tl1_done_block(kConsumerHart));
    consumer_args[abi::kArgMode] = abi::kModeNonSnoopingProducer;
    std::vector<uint32_t> producer_args = consumer_args;
    producer_args[abi::kArgStatusOffset] = static_cast<uint32_t>(tl1_done_block(kProducerHart));
    producer_args[abi::kArgTargetCce] = cce_index;
    const std::string label = fmt::format(
        "NonSnoopingProducer: CCE{} hart {} stores by SPA into CCE{} TL1 0x{:x}, consumer CCE{} hart {}, locked lines "
        "{}..{}",
        producer_cce,
        kProducerHart,
        cce_index,
        layout.region_offset,
        cce_index,
        kConsumerHart,
        params.lock_first_line,
        params.lock_first_line + params.lock_num_lines - 1);
    log_info(tt::LogTest, "MM41 scoped lock {}", label);
    const std::optional<std::vector<std::vector<uint32_t>>> statuses = run_lock_kernels(
        device,
        {{.cce_index = producer_cce, .hart = kProducerHart, .args = producer_args},
         {.cce_index = cce_index,
          .hart = kConsumerHart,
          .kernel_path = kConsumerKernel,
          .args = consumer_args,
          .done_magic = abi::kConsumerDoneMagic}});
    ASSERT_TRUE(statuses.has_value()) << label << ": no status after " << kHostDeadline.count() << " s"
                                      << kFreshSessionNote;
    const std::vector<uint32_t>& producer = (*statuses)[0];
    const std::vector<uint32_t>& consumer = (*statuses)[1];
    log_info(
        tt::LogTest,
        "MM41 scoped lock {}: producer {}; consumer {}",
        label,
        mm41_lock_detail(producer),
        mm41_lock_detail(consumer));
    ASSERT_EQ(producer[abi::kStatusCode], abi::kCodeDone) << label << ": producer " << mm41_lock_detail(producer);
    ASSERT_EQ(consumer[abi::kStatusCode], abi::kCodeDone)
        << label << ": consumer " << mm41_lock_detail(consumer)
        << (consumer[abi::kStatusCode] == abi::kCodeDataTimeout ? kFreshSessionNote : "");
    const std::vector<uint32_t> new_values = sentinel_values(abi::kNewBase);
    expect_consumer_results(
        device,
        cce_index,
        layout.result_offset,
        params.lock_first_line,
        params.lock_num_lines,
        kSiblingSpaWritesSnooped ? new_values : sentinel_values(abi::kOldBase),
        new_values,
        "consumer");
}

INSTANTIATE_TEST_SUITE_P(
    MM41, MimirCacheScopedLock, ::testing::ValuesIn(mimir_contract::cce_indices()), cce_param_name);

}  // namespace tt::tt_metal::mimir_test::dm_scoped_lock
