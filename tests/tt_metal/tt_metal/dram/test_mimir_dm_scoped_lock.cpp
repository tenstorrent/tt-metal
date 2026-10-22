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
    MIMIR_SKIP_UNLESS_EMU();
    const ClmCase clm_case = GetParam();
    if (const std::string skip_reason = clm_skip_reason(); !skip_reason.empty()) {
        GTEST_SKIP() << skip_reason;
    }
    IDevice* device = CreateDevice(0);
    ASSERT_NE(device, nullptr);
    ScopedDevice device_closer(device);
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
        read_cce_tl1(device, kLockCce, layout.result_offset, kResultBytes + abi::kLineBytes);
    const std::vector<uint32_t> warm = result_values(result_words, abi::kResultWarm);
    const std::vector<uint32_t> cached_after = result_values(result_words, abi::kResultCached);
    const std::vector<uint32_t> uncached_after = result_values(result_words, abi::kResultUncached);
    const std::vector<uint32_t> region_words =
        read_cce_tl1(device, kLockCce, layout.region_offset, abi::kRegionBytes + abi::kLineBytes);
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

}  // namespace tt::tt_metal::mimir_test::dm_scoped_lock
