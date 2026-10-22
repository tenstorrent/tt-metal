// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

// MM-8 quasar_cache / quasar_cache_perf port on a CCE hart's own TL1 data area. The hart's uncached-alias read is the
// primary observer. Host readback is a second observer: the MM-8 probe showed the host does not see cached stores
// before a flush, so a case that expects the old values asserts the host still sees them. MimirDmCachePerf ports the
// cache_perf write paths as correctness checks over a wider size range, with no timing asserts. Every store case also
// checks that the hart's own cached read sees the stores, so an expect-old case cannot pass on stores that never ran.

#include <gtest/gtest.h>

#include <fmt/format.h>
#include <tt-logger/tt-logger.hpp>

#include <chrono>
#include <cstdint>
#include <optional>
#include <string>
#include <vector>

#include <tt-metalium/host_api.hpp>
#include <tt-metalium/tt_metal.hpp>
#include "tt_metal/tt_metal/dram/mimir_access.hpp"
#include "tt_metal/tt_metal/dram/mimir_dm_common.hpp"
#include "tt_metal/tt_metal/dram/mimir_hw_state.hpp"
#include "tt_metal/tt_metal/dram/mimir_patterns.hpp"
#include "tt_metal/tt_metal/dram/mimir_test_common.hpp"
#include "tt_metal/tt_metal/test_kernels/cce/cce_cache_ops_abi.h"

namespace tt::tt_metal::mimir_test::dm_cache {

namespace abi = cce_cache_ops_abi;

inline constexpr const char* kCacheOpsKernel = "tests/tt_metal/tt_metal/test_kernels/cce/cce_cache_ops.cpp";
inline constexpr uint32_t kCce = 0;
inline constexpr uint32_t kProducerHart = 0;
inline constexpr uint32_t kConsumerHart = 1;
// Four lines, as upstream FlushRange.
inline constexpr uint32_t kCacheWords = 64;
inline constexpr uint32_t kOldValue = 0x01D00000u;
// Covers launch and the slowest mode (the full L2 flush walks all 4 MiB of TL1 by line).
inline constexpr std::chrono::seconds kHostDeadline{500};

std::vector<uint32_t> pattern_words(uint32_t value, uint32_t num_words) {
    std::vector<uint32_t> words(num_words);
    for (uint32_t word = 0; word < num_words; ++word) {
        words[word] = value + word;
    }
    return words;
}

std::string code_name(uint32_t code) {
    switch (code) {
        case abi::kCodeDone: return "done";
        case abi::kCodeBadArgs: return "bad_args";
        case abi::kCodeFlagTimeout: return "flag_timeout";
    }
    return fmt::format("unknown_code_0x{:x}", code);
}

struct HartLaunch {
    uint32_t hart = 0;
    uint32_t mode = 0;
    uint64_t data_offset = 0;
    uint32_t num_words = 0;
    uint32_t value = 0;
    uint64_t flag_offset = 0;
};

std::vector<uint32_t> kernel_args(const HartLaunch& launch) {
    std::vector<uint32_t> args(abi::kNumArgs, 0);
    args[abi::kArgMode] = launch.mode;
    args[abi::kArgStatusOffset] = static_cast<uint32_t>(tl1_done_block(launch.hart));
    args[abi::kArgDataOffset] = static_cast<uint32_t>(launch.data_offset);
    args[abi::kArgNumWords] = launch.num_words;
    args[abi::kArgValue] = launch.value;
    args[abi::kArgOldValue] = kOldValue;
    args[abi::kArgFlagOffset] = static_cast<uint32_t>(launch.flag_offset);
    args[abi::kArgPollBudgetLo] = static_cast<uint32_t>(kPollBudgetCycles);
    args[abi::kArgPollBudgetHi] = static_cast<uint32_t>(kPollBudgetCycles >> 32);
    return args;
}

// Launches one Program with every hart and returns each status line in launch order. Empty on the host deadline.
std::vector<std::vector<uint32_t>> run_program(IDevice* device, const std::vector<HartLaunch>& launches) {
    std::vector<CceKernelLaunch> kernels;
    for (const HartLaunch& launch : launches) {
        write_cce_tl1(
            device, kCce, tl1_done_block(launch.hart), std::vector<uint32_t>(abi::kStatusWords, abi::kStatusPoison));
        kernels.push_back(
            {.cce_index = kCce,
             .kernel_path = kCacheOpsKernel,
             .runtime_args = kernel_args(launch),
             .hart = launch.hart});
    }
    Program program = CreateProgram();
    return run_cce_kernels_with_deadline(
               device,
               program,
               kernels,
               std::vector<uint32_t>(kernels.size(), abi::kDoneMagic),
               abi::kStatusWords,
               kHostDeadline)
        .value_or(std::vector<std::vector<uint32_t>>{});
}

std::string status_detail(const std::vector<uint32_t>& status) {
    return fmt::format(
        "kernel code {}, cached before {}, uncached before {}, warm old {}, stale before {}, matches {} (first other "
        "word {} observed 0x{:08x}), flag {}, {} cycles",
        code_name(status[abi::kStatusCode]),
        status[abi::kStatusCachedBefore],
        status[abi::kStatusUncachedBefore],
        status[abi::kStatusWarmOld],
        status[abi::kStatusStaleBefore],
        status[abi::kStatusMatches],
        status[abi::kStatusFirstMismatchWord],
        status[abi::kStatusObserved],
        status[abi::kStatusFlagReady],
        status[abi::kStatusCycles]);
}

// Writes the old pattern and a guard line after it.
void prefill(IDevice* device, uint64_t data_offset, uint32_t num_words, uint32_t tag) {
    std::vector<uint32_t> words = pattern_words(kOldValue, num_words);
    const std::vector<uint32_t> guard = make_guard(tag);
    words.insert(words.end(), guard.begin(), guard.end());
    write_cce_tl1(device, kCce, data_offset, words);
}

// Checks the host's view of the data and the guard after it.
void expect_host_data(
    IDevice* device,
    uint64_t data_offset,
    const std::vector<uint32_t>& expected,
    uint32_t tag,
    const std::string& label) {
    const uint32_t num_bytes = static_cast<uint32_t>(expected.size() * sizeof(uint32_t));
    EXPECT_TRUE(compare_words("host readback", expected, read_cce_tl1(device, kCce, data_offset, num_bytes))) << label;
    EXPECT_EQ(read_cce_tl1(device, kCce, data_offset + num_bytes, kGuardBytes), make_guard(tag))
        << label << ": guard overwritten";
}

enum class CacheCase : uint32_t {
    NoFlushNotVisible,
    L1FlushLine,
    L1FlushFull,
    L1InvalidateLine,
    L2FlushLine,
    L2FlushRange,
    L2FlushFull,
    L2InvalidateLine,
    InvalidateFreshRead,
    HartToHartCoherent,
    L1InvalidateFull,
    L2InvalidateFreshRead,
};

struct CacheCaseSpec {
    const char* name;
    uint32_t mode;
    bool expect_new;
};

CacheCaseSpec spec_of(CacheCase cache_case) {
    switch (cache_case) {
        case CacheCase::NoFlushNotVisible: return {"NoFlushNotVisible", abi::kModeNoFlush, false};
        case CacheCase::L1FlushLine: return {"L1FlushLine", abi::kModeL1FlushLine, true};
        case CacheCase::L1FlushFull: return {"L1FlushFull", abi::kModeL1FlushFull, true};
        case CacheCase::L1InvalidateLine: return {"L1InvalidateLine", abi::kModeL1InvalidateLine, false};
        case CacheCase::L2FlushLine: return {"L2FlushLine", abi::kModeL2FlushLine, true};
        case CacheCase::L2FlushRange: return {"L2FlushRange", abi::kModeL2FlushRange, true};
        case CacheCase::L2FlushFull: return {"L2FlushFull", abi::kModeL2FlushFull, true};
        case CacheCase::L2InvalidateLine: return {"L2InvalidateLine", abi::kModeL2InvalidateLine, false};
        case CacheCase::InvalidateFreshRead: return {"InvalidateFreshRead", abi::kModeInvalidateFreshRead, true};
        case CacheCase::HartToHartCoherent: return {"HartToHartCoherent", abi::kModeProducer, true};
        case CacheCase::L1InvalidateFull: return {"L1InvalidateFull", abi::kModeL1InvalidateFull, false};
        case CacheCase::L2InvalidateFreshRead: return {"L2InvalidateFreshRead", abi::kModeL2InvalidateFreshRead, true};
    }
    return {"Unknown", abi::kNumModes, false};
}

std::string cache_case_name(const ::testing::TestParamInfo<CacheCase>& info) { return spec_of(info.param).name; }

bool is_fresh_read(CacheCase cache_case) {
    return cache_case == CacheCase::InvalidateFreshRead || cache_case == CacheCase::L2InvalidateFreshRead;
}

void run_hart_to_hart(IDevice* device, uint32_t value) {
    const uint64_t data_offset = tl1_data(kProducerHart);
    const uint64_t flag_offset = data_offset + kCacheWords * sizeof(uint32_t) + kGuardBytes;
    const uint32_t tag = static_cast<uint32_t>(CacheCase::HartToHartCoherent) + 1;
    prefill(device, data_offset, kCacheWords, tag);
    write_cce_tl1(device, kCce, flag_offset, std::vector<uint32_t>(abi::kLineBytes / sizeof(uint32_t), 0));
    const std::vector<HartLaunch> launches = {
        {.hart = kProducerHart,
         .mode = abi::kModeProducer,
         .data_offset = data_offset,
         .num_words = kCacheWords,
         .value = value,
         .flag_offset = flag_offset},
        {.hart = kConsumerHart,
         .mode = abi::kModeConsumer,
         .data_offset = data_offset,
         .num_words = kCacheWords,
         .value = value,
         .flag_offset = flag_offset},
    };
    const std::string label = fmt::format(
        "HartToHartCoherent on CCE{}: hart {} stores and flushes, hart {} reads warm, then invalidates and reads, TL1 "
        "0x{:x}, {} words",
        kCce,
        kProducerHart,
        kConsumerHart,
        data_offset,
        kCacheWords);
    log_info(tt::LogTest, "MM8 cache {}", label);
    const std::vector<std::vector<uint32_t>> statuses = run_program(device, launches);
    ASSERT_FALSE(statuses.empty()) << label << ": no status after " << kHostDeadline.count() << " s"
                                   << kFreshSessionNote;
    const std::vector<uint32_t>& producer = statuses[0];
    const std::vector<uint32_t>& consumer = statuses[1];
    log_info(
        tt::LogTest, "MM8 cache {}: producer {}; consumer {}", label, status_detail(producer), status_detail(consumer));
    ASSERT_EQ(producer[abi::kStatusCode], abi::kCodeDone) << label;
    ASSERT_EQ(consumer[abi::kStatusCode], abi::kCodeDone) << label;
    EXPECT_EQ(producer[abi::kStatusCachedBefore], kCacheWords) << label << ": producer cached view after its stores";
    EXPECT_EQ(producer[abi::kStatusUncachedBefore], 0u) << label << ": cached stores visible before the flush";
    EXPECT_EQ(producer[abi::kStatusMatches], kCacheWords) << label << ": producer uncached view after the flush";
    EXPECT_EQ(consumer[abi::kStatusWarmOld], kCacheWords) << label << ": consumer warm read before the stores";
    EXPECT_EQ(consumer[abi::kStatusStaleBefore], 0u)
        << label
        << ": consumer cached read after the flush, before its invalidate, still old (risc_common.h says DM "
           "cores are coherent)";
    EXPECT_EQ(consumer[abi::kStatusMatches], kCacheWords) << label << ": consumer cached view after invalidate";
    expect_host_data(device, data_offset, pattern_words(value, kCacheWords), tag, label);
}

class MimirDmCache : public ::testing::TestWithParam<CacheCase> {};

TEST_P(MimirDmCache, Op) {
    MIMIR_SKIP_UNLESS_EMU();
    const CacheCase cache_case = GetParam();
    if (const std::string skip_reason = mimir_watcher_skip_reason(); !skip_reason.empty()) {
        GTEST_SKIP() << skip_reason;
    }
    IDevice* device = CreateDevice(0);
    ASSERT_NE(device, nullptr);
    ScopedDevice device_closer(device);
    check_tl1_layout();

    const CacheCaseSpec spec = spec_of(cache_case);
    const uint32_t value = 0xCA000000u | (static_cast<uint32_t>(cache_case) << 16);
    if (cache_case == CacheCase::HartToHartCoherent) {
        run_hart_to_hart(device, value);
        return;
    }

    const uint64_t data_offset = tl1_data(kProducerHart);
    const uint32_t tag = static_cast<uint32_t>(cache_case) + 1;
    prefill(device, data_offset, kCacheWords, tag);
    const std::string label = fmt::format(
        "{} on CCE{} hart {}, TL1 0x{:x}, {} words", spec.name, kCce, kProducerHart, data_offset, kCacheWords);
    log_info(tt::LogTest, "MM8 cache {}", label);
    const std::vector<std::vector<uint32_t>> statuses = run_program(
        device,
        {{.hart = kProducerHart,
          .mode = spec.mode,
          .data_offset = data_offset,
          .num_words = kCacheWords,
          .value = value}});
    ASSERT_FALSE(statuses.empty()) << label << ": no status after " << kHostDeadline.count() << " s"
                                   << kFreshSessionNote;
    const std::vector<uint32_t>& status = statuses[0];
    log_info(tt::LogTest, "MM8 cache {}: {}", label, status_detail(status));
    ASSERT_EQ(status[abi::kStatusCode], abi::kCodeDone) << label << ": " << status_detail(status);
    if (is_fresh_read(cache_case)) {
        EXPECT_EQ(status[abi::kStatusStaleBefore], kCacheWords) << label << ": cached re-read before the invalidate";
    } else {
        EXPECT_EQ(status[abi::kStatusCachedBefore], kCacheWords) << label << ": cached view after the stores";
        EXPECT_EQ(status[abi::kStatusUncachedBefore], 0u) << label << ": cached stores visible before any flush";
    }
    EXPECT_EQ(status[abi::kStatusMatches], spec.expect_new ? kCacheWords : 0u)
        << label << ": hart view after the operation; " << status_detail(status);
    expect_host_data(device, data_offset, pattern_words(spec.expect_new ? value : kOldValue, kCacheWords), tag, label);
}

INSTANTIATE_TEST_SUITE_P(
    MM8,
    MimirDmCache,
    ::testing::Values(
        CacheCase::NoFlushNotVisible,
        CacheCase::L1FlushLine,
        CacheCase::L1FlushFull,
        CacheCase::L1InvalidateLine,
        CacheCase::L2FlushLine,
        CacheCase::L2FlushRange,
        CacheCase::L2FlushFull,
        CacheCase::L2InvalidateLine,
        CacheCase::InvalidateFreshRead,
        CacheCase::HartToHartCoherent,
        CacheCase::L1InvalidateFull,
        CacheCase::L2InvalidateFreshRead),
    cache_case_name);

enum class WritePath : uint32_t { Uncached1B, Uncached8B, CachedRangeFlush, CachedFastFlush };

uint32_t write_path_mode(WritePath write_path) {
    switch (write_path) {
        case WritePath::Uncached1B: return abi::kModeWriteUncached1B;
        case WritePath::Uncached8B: return abi::kModeWriteUncached8B;
        case WritePath::CachedRangeFlush: return abi::kModeWriteCachedRangeFlush;
        case WritePath::CachedFastFlush: return abi::kModeWriteCachedFastFlush;
    }
    return abi::kNumModes;
}

std::string write_path_name(const ::testing::TestParamInfo<WritePath>& info) {
    switch (info.param) {
        case WritePath::Uncached1B: return "Uncached1B";
        case WritePath::Uncached8B: return "Uncached8B";
        case WritePath::CachedRangeFlush: return "CachedRangeFlush";
        case WritePath::CachedFastFlush: return "CachedFastFlush";
    }
    return "Unknown";
}

// Upstream sweeps 8 B..2 KiB; 32 KiB adds a range larger than the L1 D$.
inline constexpr uint32_t kPerfSizesBytes[] = {8, 64, 200, 2048, 32 * 1024};

class MimirDmCachePerf : public ::testing::TestWithParam<WritePath> {};

TEST_P(MimirDmCachePerf, WritePath) {
    MIMIR_SKIP_UNLESS_EMU();
    if (const std::string skip_reason = mimir_watcher_skip_reason(); !skip_reason.empty()) {
        GTEST_SKIP() << skip_reason;
    }
    IDevice* device = CreateDevice(0);
    ASSERT_NE(device, nullptr);
    ScopedDevice device_closer(device);
    check_tl1_layout();

    const WritePath write_path = GetParam();
    const uint64_t data_offset = tl1_data(kProducerHart);
    for (uint32_t size_index = 0; size_index < std::size(kPerfSizesBytes); ++size_index) {
        const uint32_t num_words = kPerfSizesBytes[size_index] / sizeof(uint32_t);
        const uint32_t tag = static_cast<uint32_t>(write_path) * 8 + size_index + 1;
        const uint32_t value = 0x9E000000u | (tag << 16);
        prefill(device, data_offset, num_words, tag);
        const std::string label = fmt::format(
            "{} {} B on CCE{} hart {}, TL1 0x{:x}",
            write_path_name({write_path, 0}),
            kPerfSizesBytes[size_index],
            kCce,
            kProducerHart,
            data_offset);
        const std::vector<std::vector<uint32_t>> statuses = run_program(
            device,
            {{.hart = kProducerHart,
              .mode = write_path_mode(write_path),
              .data_offset = data_offset,
              .num_words = num_words,
              .value = value}});
        ASSERT_FALSE(statuses.empty()) << label << ": no status after " << kHostDeadline.count() << " s"
                                       << kFreshSessionNote;
        const std::vector<uint32_t>& status = statuses[0];
        log_info(tt::LogTest, "MM8 cache perf {}: {}", label, status_detail(status));
        ASSERT_EQ(status[abi::kStatusCode], abi::kCodeDone) << label;
        EXPECT_EQ(status[abi::kStatusMatches], num_words) << label << ": " << status_detail(status);
        expect_host_data(device, data_offset, pattern_words(value, num_words), tag, label);
    }
}

INSTANTIATE_TEST_SUITE_P(
    MM8,
    MimirDmCachePerf,
    ::testing::Values(
        WritePath::Uncached1B, WritePath::Uncached8B, WritePath::CachedRangeFlush, WritePath::CachedFastFlush),
    write_path_name);

}  // namespace tt::tt_metal::mimir_test::dm_cache
