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
#include <fmt/ranges.h>
#include <tt-logger/tt-logger.hpp>

#include <chrono>
#include <cstdint>
#include <filesystem>
#include <cstring>
#include <functional>
#include <optional>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "impl/context/metal_context.hpp"
#include "impl/program/program_impl.hpp"
#include "impl/program/slow_dispatch.hpp"
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/tt_metal.hpp>
#include "tt_metal/tt_metal/dram/mimir_access.hpp"
#include "tt_metal/tt_metal/dram/mimir_dm_common.hpp"
#include "tt_metal/tt_metal/dram/mimir_hw_state.hpp"
#include "tt_metal/tt_metal/dram/mimir_idma_copy.hpp"
#include "tt_metal/tt_metal/dram/mimir_patterns.hpp"
#include "tt_metal/tt_metal/dram/mimir_snoop_ctl.hpp"
#include "tt_metal/tt_metal/dram/mimir_test_common.hpp"
#include "tt_metal/tt_metal/test_kernels/cce/cce_cache_ops_abi.h"
#include "tt_metal/tt_metal/test_kernels/cce/cce_idma_copy_abi.h"

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
        case abi::kCodeWipeTimeout: return "wipe_timeout";
        case abi::kCodeDataTimeout: return "data_timeout";
        case abi::kCodeRaceUnresolved: return "race_unresolved";
        case abi::kCodePeerTimeout: return "peer_timeout";
    }
    return fmt::format("unknown_code_0x{:x}", code);
}

struct HartLaunch {
    uint32_t hart = kCceKernelHart;
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

void prefill(IDevice* device, uint64_t data_offset, uint32_t num_words, uint32_t tag) {
    std::vector<uint32_t> words = pattern_words(kOldValue, num_words);
    const std::vector<uint32_t> guard = make_guard(tag);
    words.insert(words.end(), guard.begin(), guard.end());
    write_cce_tl1(device, kCce, data_offset, words);
}

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
    skip_unless_programmable_dram_package();
    const CacheCase cache_case = GetParam();
    if (const std::string skip_reason = mimir_watcher_skip_reason(); !skip_reason.empty()) {
        GTEST_SKIP() << skip_reason;
    }
    ScopedDevice device_closer;
    IDevice* device = device_closer.device();
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
    skip_unless_programmable_dram_package();
    if (const std::string skip_reason = mimir_watcher_skip_reason(); !skip_reason.empty()) {
        GTEST_SKIP() << skip_reason;
    }
    ScopedDevice device_closer;
    IDevice* device = device_closer.device();
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

// MM41 probes (plan increment 1). Each runs on every CCE of the descriptor.

// Null kernels on every hart: well under one poll budget.
inline constexpr std::chrono::seconds kProbeHostDeadline{300};
inline constexpr uint32_t kSnoopMasterDefault = 0x6;

// Runs one cce_cache_ops kernel per (hart, args) on `cce_index` in one Program and returns each status line in launch
// order; nullopt on the host deadline.
std::optional<std::vector<std::vector<uint32_t>>> run_on_cce(
    IDevice* device,
    uint32_t cce_index,
    const std::vector<std::pair<uint32_t, std::vector<uint32_t>>>& hart_args,
    std::chrono::seconds host_deadline) {
    std::vector<CceKernelLaunch> kernels;
    for (const auto& [hart, args] : hart_args) {
        write_cce_tl1(
            device, cce_index, tl1_done_block(hart), std::vector<uint32_t>(abi::kStatusWords, abi::kStatusPoison));
        kernels.push_back({.cce_index = cce_index, .kernel_path = kCacheOpsKernel, .runtime_args = args, .hart = hart});
    }
    Program program = CreateProgram();
    return run_cce_kernels_with_deadline(
        device,
        program,
        kernels,
        std::vector<uint32_t>(kernels.size(), abi::kDoneMagic),
        abi::kStatusWords,
        host_deadline);
}

std::vector<uint32_t> probe_args(uint32_t mode, uint32_t hart) {
    std::vector<uint32_t> args(abi::kNumArgs, 0);
    args[abi::kArgMode] = mode;
    args[abi::kArgStatusOffset] = static_cast<uint32_t>(tl1_done_block(hart));
    args[abi::kArgPollBudgetLo] = static_cast<uint32_t>(kPollBudgetCycles);
    args[abi::kArgPollBudgetHi] = static_cast<uint32_t>(kPollBudgetCycles >> 32);
    return args;
}

class MimirCacheProbe : public ::testing::TestWithParam<uint32_t> {
protected:
    void SetUp() override {
        skip_unless_programmable_dram_package();
        if (const std::string skip_reason = mimir_watcher_skip_reason(); !skip_reason.empty()) {
            GTEST_SKIP() << skip_reason;
        }
    }
};

// Every hart of one CCE runs a kernel and posts its status.
TEST_P(MimirCacheProbe, EightHartLaunch) {
    const uint32_t cce_index = GetParam();
    ScopedDevice device_closer;
    IDevice* device = device_closer.device();
    check_tl1_layout();
    record_descriptor_properties(soc_desc_of(device));

    std::vector<std::pair<uint32_t, std::vector<uint32_t>>> hart_args;
    for (uint32_t hart = 0; hart < cce_hart_count(); ++hart) {
        hart_args.emplace_back(hart, probe_args(abi::kModeNull, hart));
    }
    const std::string label =
        fmt::format("EightHartLaunch on CCE{}: null kernel on harts 0-{}", cce_index, cce_hart_count() - 1);
    log_info(tt::LogTest, "MM41 {}", label);
    const std::optional<std::vector<std::vector<uint32_t>>> statuses =
        run_on_cce(device, cce_index, hart_args, kProbeHostDeadline);
    if (!statuses.has_value()) {
        std::string missing;
        for (uint32_t hart = 0; hart < cce_hart_count(); ++hart) {
            if (read_status_magic(device, cce_index, hart) != abi::kDoneMagic) {
                missing += fmt::format(" {}", hart);
            }
        }
        RecordProperty("harts_without_status", missing);
        FAIL() << label << ": no status from harts" << missing << " after " << kProbeHostDeadline.count() << " s"
               << kFreshSessionNote;
    }
    for (uint32_t hart = 0; hart < cce_hart_count(); ++hart) {
        EXPECT_EQ((*statuses)[hart][abi::kStatusCode], abi::kCodeDone)
            << label << ": hart " << hart << " " << status_detail((*statuses)[hart]);
    }
    RecordProperty("harts_with_status", static_cast<int>(cce_hart_count()));
}

// The hart's read of the snoop-master config matches the host's read of the same register and the 0x6 reset default,
// and the L2 CONFIGURATION fields give 64 B lines and a 128 KiB L2. Both registers are recorded.
TEST_P(MimirCacheProbe, SnoopConfig) {
    const uint32_t cce_index = GetParam();
    ScopedDevice device_closer;
    IDevice* device = device_closer.device();
    check_tl1_layout();
    record_descriptor_properties(soc_desc_of(device));
    const Cluster& cluster = MetalContext::instance().get_cluster();

    const std::string label = fmt::format("SnoopConfig on CCE{} hart {}", cce_index, kProducerHart);
    const uint32_t host_snoop_master = read_snoop_master_config(cluster, device->id(), cce_index);
    const std::optional<std::vector<std::vector<uint32_t>>> statuses = run_on_cce(
        device, cce_index, {{kProducerHart, probe_args(abi::kModeReadConfig, kProducerHart)}}, kProbeHostDeadline);
    ASSERT_TRUE(statuses.has_value()) << label << ": no status after " << kProbeHostDeadline.count() << " s"
                                      << kFreshSessionNote;
    const std::vector<uint32_t>& status = statuses->front();
    ASSERT_EQ(status[abi::kStatusCode], abi::kCodeDone) << label << ": " << status_detail(status);
    const uint32_t snoop_master = status[abi::kStatusRegisterA];
    const uint32_t l2_configuration = status[abi::kStatusRegisterB];
    const uint32_t l2_banks = l2_configuration & 0xFF;
    const uint32_t l2_ways = (l2_configuration >> 8) & 0xFF;
    const uint32_t l2_sets = 1u << ((l2_configuration >> 16) & 0xFF);
    const uint32_t l2_line_bytes = 1u << ((l2_configuration >> 24) & 0xFF);
    const std::string snoop_detail = fmt::format(
        "0x{:08x}: disable_inline_writes {} invalidate_flush_dirty_en {} clean_putdata_en {}{}",
        snoop_master,
        snoop_master & 0x1,
        (snoop_master >> 1) & 0x1,
        (snoop_master >> 2) & 0x1,
        snoop_master == kSnoopMasterDefault ? " (default)" : " (not the default 0x6)");
    const std::string l2_detail = fmt::format(
        "0x{:08x}: banks {} ways {} sets {} line {} B", l2_configuration, l2_banks, l2_ways, l2_sets, l2_line_bytes);
    log_info(tt::LogTest, "MM41 {}: snoop master {}; L2 CONFIGURATION {}", label, snoop_detail, l2_detail);
    RecordProperty("snoop_master_cfg", snoop_detail);
    RecordProperty("l2_configuration", l2_detail);
    EXPECT_EQ(snoop_master, host_snoop_master) << label << ": hart and host reads of NOC_SNOOP_TL_MASTER_CFG differ";
    EXPECT_EQ(snoop_master, kSnoopMasterDefault) << label << ": snoop master " << snoop_detail;
    EXPECT_EQ(l2_line_bytes, abi::kLineBytes) << label << ": L2 CONFIGURATION " << l2_detail;
    EXPECT_EQ(static_cast<uint64_t>(l2_banks) * l2_ways * l2_sets * l2_line_bytes, 128 * kKiB)
        << label << ": L2 size from L2 CONFIGURATION " << l2_detail;
}

INSTANTIATE_TEST_SUITE_P(MM41, MimirCacheProbe, ::testing::ValuesIn(mimir_contract::cce_indices()), cce_param_name);

// MM41 cases (plan increments 2-4). Each runs on every CCE of the descriptor unless it says otherwise. Kernels make no
// uncached store while an iDMA or another CCE may be writing into their TL1 (MIM-560): a reader polls the data with
// loads until the writer's words have landed, and only then stores its status.

namespace idma_abi = cce_idma_copy_abi;

// Measured by the removed MM41/MimirCacheProbe.L2Geometry (2026-10-22, every CCE of mimir_2x_package): L2
// CONFIGURATION reads 4 banks x 8 ways x 64 sets of 64 B, lines 16 KiB apart share a set, and replacement is not LRU,
// so eviction of a given line is forced only by at least 24 same-set reads.
inline constexpr uint32_t kL2ConflictStrideBytes = 16 * 1024;
inline constexpr uint32_t kL2EvictingReads = 24;
// Set by MM41/MimirCacheWriterProbe.ExternalWriters (2026-10-22): writes without a snoop bit do not reach the reader's
// cached copy without an invalidate (SiblingSpa and RemoteCce in SnoopOffInvalidate).
inline constexpr bool kSiblingSpaWritesSnooped = false;
inline constexpr bool kRemoteSpaWritesSnooped = false;
// The iDMA snoop packet tag applies to writes from a sibling CCE into this CCE's TL1: the receiving CCE's snoop master
// refreshes its cached copy (GRENSV-3774, chippy cce_xcce_snoop_dma; GRENSV-1577). The hardware never snoops an iDMA
// write from a CCE into its own TL1.
inline constexpr const char* kIdmaSnoopNote =
    "iDMA snoop packet tag not honoured; it applies only to a sibling CCE's iDMA writes into this CCE's TL1 "
    "(GRENSV-3774, GRENSV-1577), and the hardware never snoops a CCE's iDMA writes into its own TL1";

// Up to three sequential polls plus launch, as MM-8 sizes 400 s for three.
inline constexpr std::chrono::seconds kMm41HostDeadline{400};
// Stress launches: up to 192 KiB per pass, an L2 range flush and an uncached check of every word.
inline constexpr std::chrono::seconds kStressHostDeadline{900};
// Status lines of both kernels fit one line; read every word of the larger.
inline constexpr uint32_t kMixedStatusWords = idma_abi::kStatusWords;
static_assert(abi::kStatusWords <= kMixedStatusWords);
// Flag lines in a hart's data area, after the largest data region and its guard.
inline constexpr uint64_t kFlagLinesFromData = 0x3C000;
static_assert(abi::kMaxWords * sizeof(uint32_t) + kGuardBytes <= kFlagLinesFromData);
// Per-pass records, after the status line in the hart's done block.
inline constexpr uint64_t kTraceFromDoneBlock = abi::kLineBytes;
// GDDR sources of the MM41 iDMA writers in a DRAM view above DRAM UNRESERVED, clear of MM-8 (0x10000) and MM43
// (0x100000).
inline constexpr uint64_t kGddrSourceOffset = 0x200000;
// The ordering consumer's late-data window.
inline constexpr uint32_t kDataWindowCycles = static_cast<uint32_t>(kPollBudgetCycles / 16);
inline constexpr uint32_t kReaderHart = 0;
inline constexpr uint32_t kWriterHart = 1;
inline constexpr uint32_t kWordsPerLine = abi::kLineBytes / sizeof(uint32_t);

struct HartKernel {
    uint32_t cce_index = 0;
    uint32_t hart = kCceKernelHart;
    const char* kernel_path = kCacheOpsKernel;
    std::vector<uint32_t> args;
    uint32_t done_magic = abi::kDoneMagic;
};

// A TL1 word the host waits for before the second stage of run_staged.
struct HostFlag {
    uint32_t cce_index = 0;
    uint64_t offset = 0;
    uint32_t value = 0;
};

struct StagedRun {
    // kMixedStatusWords words per kernel, the first stage then the second, in launch order. Empty on the host
    // deadline: a kernel may still be running.
    std::vector<std::vector<uint32_t>> statuses;
    bool flag_seen = false;
    bool second_launched = false;
};

std::vector<CceKernelLaunch> to_launches(const std::vector<HartKernel>& kernels) {
    std::vector<CceKernelLaunch> launches;
    for (const HartKernel& kernel : kernels) {
        launches.push_back(
            {.cce_index = kernel.cce_index,
             .kernel_path = kernel.kernel_path,
             .runtime_args = kernel.args,
             .hart = kernel.hart});
    }
    return launches;
}

bool wait_for_host_flag(IDevice* device, const HostFlag& flag, std::chrono::seconds host_wait) {
    const std::chrono::steady_clock::time_point deadline = std::chrono::steady_clock::now() + host_wait;
    while (std::chrono::steady_clock::now() < deadline) {
        if (read_cce_tl1(device, flag.cce_index, flag.offset, sizeof(uint32_t)).front() == flag.value) {
            return true;
        }
    }
    return false;
}

// Launches `first` as one Program. With `flag`, waits for it, then runs `host_action` and launches `second` as a
// second Program; a CCE runs one Program at a time, so `second` uses CCEs `first` does not. Then waits for every
// status magic within `host_deadline` and ends the Programs.
StagedRun run_staged(
    IDevice* device,
    const std::vector<HartKernel>& first,
    const std::optional<HostFlag>& flag,
    const std::function<void()>& host_action,
    const std::vector<HartKernel>& second,
    std::chrono::seconds host_deadline) {
    StagedRun run;
    for (const std::vector<HartKernel>* stage : {&first, &second}) {
        for (const HartKernel& kernel : *stage) {
            write_cce_tl1(
                device,
                kernel.cce_index,
                tl1_done_block(kernel.hart),
                std::vector<uint32_t>(kMixedStatusWords, abi::kStatusPoison));
        }
    }
    std::vector<CceKernelLaunch> waited = to_launches(first);
    std::vector<uint32_t> done_magics;
    for (const HartKernel& kernel : first) {
        done_magics.push_back(kernel.done_magic);
    }
    Program first_program = CreateProgram();
    add_cce_kernels(device, first_program, waited);
    launch_on_idle_cces(device, first_program, waited);

    Program second_program = CreateProgram();
    run.flag_seen = !flag.has_value() || wait_for_host_flag(device, *flag, kPollBudgetHostTime);
    if (flag.has_value() && run.flag_seen) {
        if (host_action) {
            host_action();
        }
        if (!second.empty()) {
            const std::vector<CceKernelLaunch> second_launches = to_launches(second);
            add_cce_kernels(device, second_program, second_launches);
            launch_on_idle_cces(device, second_program, second_launches);
            run.second_launched = true;
            waited.insert(waited.end(), second_launches.begin(), second_launches.end());
            for (const HartKernel& kernel : second) {
                done_magics.push_back(kernel.done_magic);
            }
        }
    }
    if (!wait_for_status_magics(device, waited, done_magics, host_deadline)) {
        return run;
    }
    slow_dispatch::WaitProgramDone(*device, first_program);
    if (run.second_launched) {
        slow_dispatch::WaitProgramDone(*device, second_program);
    }
    for (const CceKernelLaunch& launch : waited) {
        run.statuses.push_back(
            read_cce_tl1(device, launch.cce_index, tl1_done_block(launch.hart), kMixedStatusWords * sizeof(uint32_t)));
    }
    return run;
}

StagedRun run_together(IDevice* device, const std::vector<HartKernel>& kernels, std::chrono::seconds host_deadline) {
    return run_staged(device, kernels, std::nullopt, {}, {}, host_deadline);
}

std::vector<uint32_t> mm41_args(uint32_t mode, uint32_t hart) {
    std::vector<uint32_t> args = probe_args(mode, hart);
    args[abi::kArgOldValue] = kOldValue;
    args[abi::kArgDataWindowCycles] = kDataWindowCycles;
    return args;
}

HartKernel cache_kernel(uint32_t cce_index, uint32_t hart, std::vector<uint32_t> args) {
    return {.cce_index = cce_index, .hart = hart, .args = std::move(args)};
}

std::string mm41_detail(const std::vector<uint32_t>& status) {
    return fmt::format(
        "{}, registers A {} B {} C {}, passes {}",
        status_detail(status),
        status[abi::kStatusRegisterA],
        status[abi::kStatusRegisterB],
        status[abi::kStatusRegisterC],
        status[abi::kStatusPasses]);
}

// A data or peer timeout may leave a writer's traffic in flight (MIM-601).
std::string cache_code_note(uint32_t code) {
    return code == abi::kCodeDataTimeout || code == abi::kCodePeerTimeout ? kFreshSessionNote : "";
}

// Set when a case ends in a state only a fresh emulator session clears; later MM41 cases in this process skip.
bool& fresh_session_required() {
    static bool required = false;
    return required;
}

std::string fresh_session_skip_reason() {
    return fresh_session_required()
               ? "An earlier MM41 case in this process left the CCE needing a fresh emulator session (MIM-601)."
               : "";
}

std::string idma_code_name(uint32_t code) {
    switch (code) {
        case idma_abi::kCodeLanded: return "landed";
        case idma_abi::kCodeAcked: return "acked";
        case idma_abi::kCodeVcSpaceTimeout: return "vc_space_timeout";
        case idma_abi::kCodeAckTimeout: return "ack_timeout";
        case idma_abi::kCodeDataTimeout: return "data_timeout";
        case idma_abi::kCodeBadArgs: return "bad_args";
        case idma_abi::kCodePeerTimeout: return "peer_timeout";
    }
    return fmt::format("unknown_code_0x{:x}", code);
}

std::string idma_code_note(uint32_t code) {
    return code == idma_abi::kCodeVcSpaceTimeout || code == idma_abi::kCodeAckTimeout ||
                   code == idma_abi::kCodeDataTimeout
               ? kFreshSessionNote
               : "";
}

std::string idma_detail(const std::vector<uint32_t>& status) {
    return fmt::format(
        "iDMA code {} flags 0x{:x}, {} commands issued, last poll {} {}, cycles vc_space {} ack {} data {}",
        idma_code_name(status[idma_abi::kStatusCode]),
        status[idma_abi::kStatusFlags],
        status[idma_abi::kStatusElementsIssued],
        status[idma_abi::kStatusLastPoll],
        status[idma_abi::kStatusLastPollReady] != 0 ? "done" : "timed out",
        status[idma_abi::kStatusVcSpaceCycles],
        status[idma_abi::kStatusAckCycles],
        status[idma_abi::kStatusDataCycles]);
}

IdmaOperand own_tl1_operand(uint32_t cce_index, uint64_t offset) { return {idma_abi::kKindOwnTl1, cce_index, offset}; }

IdmaOperand sram_operand(uint32_t cce_index, uint64_t offset) { return {idma_abi::kKindSramSpa, cce_index, offset}; }

IdmaOperand gddr_operand(const metal_SocDescriptor& soc, uint32_t view, uint64_t view_offset) {
    return {idma_abi::kKindGddr, mimir_of_view(soc, view), chiplet_offset_of(soc, view, view_offset)};
}

uint64_t gddr_source_view_offset() { return dram_unreserved_base() + kGddrSourceOffset; }

// One cce_idma_copy kModeTridBatch launch: num_commands commands from one source block (source stride 0) to
// destination + i * len_bytes, on trids 1, 2, ... With a nonzero start_flag_offset (the writer's own TL1) the writer
// issues only once that word is nonzero.
struct IdmaWriter {
    uint32_t cce_index = 0;
    uint32_t hart = kCceKernelHart;
    IdmaOperand source;
    IdmaOperand destination;
    uint32_t backend_vc = 0;
    uint32_t len_bytes = 0;
    uint32_t num_commands = 1;
    bool snoop = false;
    bool host_verified = false;
    uint32_t expected_xor = 0;
    uint64_t start_flag_offset = 0;
};

HartKernel idma_writer_kernel(const IdmaWriter& writer) {
    return {
        .cce_index = writer.cce_index,
        .hart = writer.hart,
        .kernel_path = kIdmaCopyKernel,
        .args = idma_copy_args(
            {.mode = idma_abi::kModeTridBatch,
             .backend_vc = writer.backend_vc,
             .status_offset = tl1_done_block(writer.hart),
             .len_bytes = writer.len_bytes,
             .poll_budget_cycles = kPollBudgetCycles,
             .source = writer.source,
             .destination = writer.destination,
             .expected_xor = writer.expected_xor,
             .num_elements = writer.num_commands,
             .trid = 1,
             .snoop = writer.snoop,
             .destination_stride_bytes = writer.len_bytes,
             .trid_stride = 1,
             .peer_flag_offset = writer.start_flag_offset,
             .host_verified = writer.host_verified}),
        .done_magic = idma_abi::kDoneMagic};
}

uint32_t idma_success_code(const IdmaWriter& writer) {
    return writer.host_verified ? idma_abi::kCodeAcked : idma_abi::kCodeLanded;
}

void prefill_on_cce(
    IDevice* device, uint32_t cce_index, uint64_t offset, const std::vector<uint32_t>& words, uint32_t tag) {
    std::vector<uint32_t> filled = words;
    const std::vector<uint32_t> guard = make_guard(tag);
    filled.insert(filled.end(), guard.begin(), guard.end());
    write_cce_tl1(device, cce_index, offset, filled);
}

void expect_host_on_cce(
    IDevice* device,
    uint32_t cce_index,
    uint64_t offset,
    const std::vector<uint32_t>& expected,
    uint32_t tag,
    const std::string& label) {
    const uint32_t num_bytes = static_cast<uint32_t>(expected.size() * sizeof(uint32_t));
    EXPECT_TRUE(compare_words("host readback", expected, read_cce_tl1(device, cce_index, offset, num_bytes))) << label;
    EXPECT_EQ(read_cce_tl1(device, cce_index, offset + num_bytes, kGuardBytes), make_guard(tag))
        << label << ": guard overwritten";
}

std::string two_mimir_skip_reason() {
    const MimirTopology topology = topology_from(configured_soc_desc());
    return topology.num_mimirs < 2
               ? fmt::format("Descriptor exposes {} Mimir; the case needs two.", topology.num_mimirs)
               : "";
}

// MimirCacheProbe's skips, and none once an earlier case needs a fresh session.
class Mm41CceCase : public MimirCacheProbe {
protected:
    void SetUp() override {
        MimirCacheProbe::SetUp();
        if (IsSkipped()) {
            return;
        }
        if (const std::string skip_reason = fresh_session_skip_reason(); !skip_reason.empty()) {
            GTEST_SKIP() << skip_reason;
        }
    }
};

class MimirCacheOps : public Mm41CceCase {};

// Two whole-cache legs, each after a full clean in the same inline-asm block (see kModeDiscardFull, kModeWipeFull).
// Discard: cdiscard.d.l1 x0 drops the stores, so TL1 keeps the old values. Wipe: FULLINVALIDATE written with every
// hart's bit and polled, then cdiscard.d.l1 x0, makes a line that an uncached store left stale read fresh. If
// FULLINVALIDATE never reads 0, the wipe is unsupported or still pending: the case fails, needs a fresh session, and
// no later MM41 case runs in this process.
TEST_P(MimirCacheOps, FullInvalidate) {
    const uint32_t cce_index = GetParam();
    ScopedDevice device_closer;
    IDevice* device = device_closer.device();
    check_tl1_layout();
    record_descriptor_properties(soc_desc_of(device));

    const uint64_t data_offset = tl1_data(kProducerHart);
    const uint32_t discard_tag = 0x45 + cce_index;
    const uint32_t discard_value = 0xD1000000u | (cce_index << 16);
    prefill_on_cce(device, cce_index, data_offset, pattern_words(kOldValue, kCacheWords), discard_tag);
    std::vector<uint32_t> discard_args = mm41_args(abi::kModeDiscardFull, kProducerHart);
    discard_args[abi::kArgDataOffset] = static_cast<uint32_t>(data_offset);
    discard_args[abi::kArgNumWords] = kCacheWords;
    discard_args[abi::kArgValue] = discard_value;
    const std::string discard_label = fmt::format(
        "FullInvalidate discard leg on CCE{} hart {}, TL1 0x{:x}, {} words",
        cce_index,
        kProducerHart,
        data_offset,
        kCacheWords);
    log_info(tt::LogTest, "MM41 {}", discard_label);
    const StagedRun discard =
        run_together(device, {cache_kernel(cce_index, kProducerHart, discard_args)}, kMm41HostDeadline);
    ASSERT_FALSE(discard.statuses.empty())
        << discard_label << ": no status after " << kMm41HostDeadline.count() << " s" << kFreshSessionNote;
    const std::vector<uint32_t>& discard_status = discard.statuses.front();
    log_info(tt::LogTest, "MM41 {}: {}", discard_label, mm41_detail(discard_status));
    ASSERT_EQ(discard_status[abi::kStatusCode], abi::kCodeDone) << discard_label << ": " << mm41_detail(discard_status);
    EXPECT_EQ(discard_status[abi::kStatusCachedBefore], kCacheWords)
        << discard_label << ": cached view after the stores";
    EXPECT_EQ(discard_status[abi::kStatusMatches], 0u)
        << discard_label << ": discarded stores reached TL1; " << mm41_detail(discard_status);
    expect_host_on_cce(
        device, cce_index, data_offset, pattern_words(kOldValue, kCacheWords), discard_tag, discard_label);

    const uint32_t wipe_tag = 0x49 + cce_index;
    const uint32_t wipe_value = 0xD2000000u | (cce_index << 16);
    prefill_on_cce(device, cce_index, data_offset, pattern_words(kOldValue, kCacheWords), wipe_tag);
    std::vector<uint32_t> wipe_args = mm41_args(abi::kModeWipeFull, kProducerHart);
    wipe_args[abi::kArgDataOffset] = static_cast<uint32_t>(data_offset);
    wipe_args[abi::kArgNumWords] = kCacheWords;
    wipe_args[abi::kArgValue] = wipe_value;
    const std::string wipe_label = fmt::format(
        "FullInvalidate wipe leg on CCE{} hart {} (FULLINVALIDATE 0xFF), TL1 0x{:x}, {} words",
        cce_index,
        kProducerHart,
        data_offset,
        kCacheWords);
    log_info(tt::LogTest, "MM41 {}", wipe_label);
    const StagedRun wipe = run_together(device, {cache_kernel(cce_index, kProducerHart, wipe_args)}, kMm41HostDeadline);
    ASSERT_FALSE(wipe.statuses.empty()) << wipe_label << ": no status after " << kMm41HostDeadline.count() << " s"
                                        << kFreshSessionNote;
    const std::vector<uint32_t>& wipe_status = wipe.statuses.front();
    log_info(tt::LogTest, "MM41 {}: {}", wipe_label, mm41_detail(wipe_status));
    RecordProperty("wipe_stale_words_before_drop", static_cast<int>(wipe_status[abi::kStatusCachedBefore]));
    if (wipe_status[abi::kStatusCode] == abi::kCodeWipeTimeout) {
        fresh_session_required() = true;
        RecordProperty(
            "full_invalidate", "unsupported or did not complete: FULLINVALIDATE did not read 0 within the poll budget");
        FAIL() << wipe_label << ": FULLINVALIDATE did not read 0 within the poll budget, so the wipe is unsupported or "
               << "still pending; " << mm41_detail(wipe_status) << kFreshSessionNote;
    }
    RecordProperty("full_invalidate", "completed");
    ASSERT_EQ(wipe_status[abi::kStatusCode], abi::kCodeDone) << wipe_label << ": " << mm41_detail(wipe_status);
    EXPECT_EQ(wipe_status[abi::kStatusWarmOld], kCacheWords) << wipe_label << ": cached read of the old values";
    EXPECT_EQ(wipe_status[abi::kStatusStaleBefore], kCacheWords)
        << wipe_label << ": the uncached store was already visible to a cached read, so the wipe is not tested";
    EXPECT_EQ(wipe_status[abi::kStatusMatches], kCacheWords)
        << wipe_label << ": cached read after the wipe; " << mm41_detail(wipe_status);
    expect_host_on_cce(device, cce_index, data_offset, pattern_words(wipe_value, kCacheWords), wipe_tag, wipe_label);
}

std::string ordering_leg_name(uint32_t leg) {
    switch (leg) {
        case abi::kOrderingLegFlushNoFence: return "FlushNoFence";
        case abi::kOrderingLegFenceNoFlush: return "FenceNoFlush";
        case abi::kOrderingLegL1ThenL2: return "L1ThenL2";
        case abi::kOrderingLegFullHierarchy: return "FullHierarchy";
    }
    return fmt::format("unknown_leg_{}", leg);
}

inline constexpr uint32_t kOrderingPasses = 8;

// Three section 4.1 legs, each kOrderingPasses passes of a producer hart against a consumer hart that reads TL1
// through the uncached alias. Data movement: the pass's data reaches TL1 within the consumer's window. Ordering: the
// data is complete at the consumer's first read after the flag. The no-fence leg gates on data movement and records
// ordering; the L1-then-L2 and full-hierarchy legs gate on ordering. Those two discard the data lines from L1 D$
// after their cflush, so a cflush that leaves the data in L1 D$ loses it. Rdcycle stamps of the two harts are not
// shown to share a clock, so cycle counts are recorded only. The full-hierarchy leg flushes all of L1 D$ and L2 each
// pass (cflush.d.l1 x0 then flush_l2_cache_full()); the full invalidate is FullInvalidate's.
TEST_P(MimirCacheOps, FlushFenceOrdering) {
    const uint32_t cce_index = GetParam();
    ScopedDevice device_closer;
    IDevice* device = device_closer.device();
    check_tl1_layout();
    record_descriptor_properties(soc_desc_of(device));

    const uint64_t data_offset = tl1_data(kProducerHart);
    const uint64_t flag_offset = data_offset + kFlagLinesFromData;
    for (const uint32_t leg :
         {abi::kOrderingLegFlushNoFence, abi::kOrderingLegL1ThenL2, abi::kOrderingLegFullHierarchy}) {
        const uint32_t tag = 0x50 + leg;
        const uint32_t value = 0x0D000000u | (leg << 20);
        prefill_on_cce(device, cce_index, data_offset, pattern_words(kOldValue, kCacheWords), tag);
        write_cce_tl1(device, cce_index, flag_offset, std::vector<uint32_t>(2 * kWordsPerLine, 0));
        std::vector<uint32_t> producer_args = mm41_args(abi::kModeOrderingProducer, kProducerHart);
        producer_args[abi::kArgDataOffset] = static_cast<uint32_t>(data_offset);
        producer_args[abi::kArgNumWords] = kCacheWords;
        producer_args[abi::kArgValue] = value;
        producer_args[abi::kArgFlagOffset] = static_cast<uint32_t>(flag_offset);
        producer_args[abi::kArgPasses] = kOrderingPasses;
        producer_args[abi::kArgSteps] = leg;
        std::vector<uint32_t> consumer_args = producer_args;
        consumer_args[abi::kArgMode] = abi::kModeOrderingConsumer;
        consumer_args[abi::kArgStatusOffset] = static_cast<uint32_t>(tl1_done_block(kConsumerHart));
        const std::string label = fmt::format(
            "FlushFenceOrdering {} on CCE{}: producer hart {}, consumer hart {}, {} passes over {} words",
            ordering_leg_name(leg),
            cce_index,
            kProducerHart,
            kConsumerHart,
            kOrderingPasses,
            kCacheWords);
        log_info(tt::LogTest, "MM41 {}", label);
        const StagedRun run = run_together(
            device,
            {cache_kernel(cce_index, kProducerHart, producer_args),
             cache_kernel(cce_index, kConsumerHart, consumer_args)},
            kMm41HostDeadline);
        ASSERT_FALSE(run.statuses.empty())
            << label << ": no status after " << kMm41HostDeadline.count() << " s" << kFreshSessionNote;
        const std::vector<uint32_t>& producer = run.statuses[0];
        const std::vector<uint32_t>& consumer = run.statuses[1];
        const std::string result = fmt::format(
            "complete at the flag {}/{}, complete later {} (largest lag {} cycles), never complete in the window {}, "
            "producer leg cycles {}",
            consumer[abi::kStatusMatches],
            kOrderingPasses,
            consumer[abi::kStatusRegisterA],
            consumer[abi::kStatusRegisterC],
            consumer[abi::kStatusRegisterB],
            producer[abi::kStatusRegisterA]);
        log_info(
            tt::LogTest,
            "MM41 {}: {}; producer {}; consumer {}",
            label,
            result,
            mm41_detail(producer),
            mm41_detail(consumer));
        RecordProperty(ordering_leg_name(leg), result);
        ASSERT_EQ(producer[abi::kStatusCode], abi::kCodeDone) << label << ": producer " << mm41_detail(producer);
        ASSERT_EQ(consumer[abi::kStatusCode], abi::kCodeDone) << label << ": consumer " << mm41_detail(consumer);
        EXPECT_EQ(consumer[abi::kStatusPasses], kOrderingPasses) << label;
        if (leg == abi::kOrderingLegFlushNoFence) {
            EXPECT_EQ(consumer[abi::kStatusRegisterB], 0u)
                << label << ": data never complete within the consumer's window; " << result;
        }
        if (leg == abi::kOrderingLegL1ThenL2 || leg == abi::kOrderingLegFullHierarchy) {
            EXPECT_EQ(consumer[abi::kStatusMatches], kOrderingPasses)
                << label << ": data incomplete at the consumer's first read after the flag; " << result;
        }
        expect_host_on_cce(
            device,
            cce_index,
            data_offset,
            pattern_words(value + ((kOrderingPasses - 1) << 16), kCacheWords),
            tag,
            label + " after the producer's final flush");
    }
}

// The line image kModePartialLineEvict expects: old word i = kOldValue + i, then the kernel's store_partial.
std::vector<uint32_t> partial_line_image(uint32_t value, uint32_t num_lines) {
    std::vector<uint32_t> words = pattern_words(kOldValue, num_lines * kWordsPerLine);
    for (uint32_t line = 0; line < num_lines; ++line) {
        uint8_t* bytes = reinterpret_cast<uint8_t*>(words.data() + line * kWordsPerLine);
        const uint32_t line_value = value + line;
        const uint8_t byte_value = static_cast<uint8_t>(line_value);
        const uint16_t half_value = static_cast<uint16_t>(line_value >> 8);
        const uint32_t word_value = ~line_value;
        const uint64_t double_value = (static_cast<uint64_t>(line_value) << 32) | (line_value ^ 0x5A5A5A5Au);
        std::memcpy(bytes + abi::kPartialByteOffset, &byte_value, sizeof(byte_value));
        std::memcpy(bytes + abi::kPartialHalfOffset, &half_value, sizeof(half_value));
        std::memcpy(bytes + abi::kPartialWordOffset, &word_value, sizeof(word_value));
        std::memcpy(bytes + abi::kPartialDoubleOffset, &double_value, sizeof(double_value));
    }
    return words;
}

inline constexpr uint32_t kPartialLines = 64;
// Three times the 128 KiB L2, so each of its sets sees 24 reads (kL2EvictingReads), from hart 1's window onward (loads
// only).
inline constexpr uint32_t kPartialEvictLines = 6144;

// Byte, halfword, word and doubleword cached stores into every line of 4 KiB, evicted by cached reads of twice the L2
// size: the untouched bytes of every line stay intact before and after the final flush, and the flush leaves each line
// exactly old-plus-partial. Lines whose partial bytes reached TL1 before the flush are recorded (dirty eviction).
TEST_P(MimirCacheOps, DirtyEvictionPartialLine) {
    const uint32_t cce_index = GetParam();
    ScopedDevice device_closer;
    IDevice* device = device_closer.device();
    check_tl1_layout();
    record_descriptor_properties(soc_desc_of(device));

    const uint64_t data_offset = tl1_data(kProducerHart);
    const uint64_t sweep_offset = tl1_data(1);
    TT_FATAL(
        sweep_offset + kPartialEvictLines * abi::kLineBytes <= tl1_test_base() + kTl1RemoteLandingOffset,
        "DirtyEvictionPartialLine eviction reads run into the remote landing area");
    const uint32_t num_words = kPartialLines * kWordsPerLine;
    const uint32_t tag = 0x58 + cce_index;
    const uint32_t value = 0x9A000000u | (cce_index << 16);
    prefill_on_cce(device, cce_index, data_offset, pattern_words(kOldValue, num_words), tag);
    std::vector<uint32_t> args = mm41_args(abi::kModePartialLineEvict, kProducerHart);
    args[abi::kArgDataOffset] = static_cast<uint32_t>(data_offset);
    args[abi::kArgNumWords] = num_words;
    args[abi::kArgValue] = value;
    args[abi::kArgAuxOffset] = static_cast<uint32_t>(sweep_offset);
    args[abi::kArgStrideBytes] = abi::kLineBytes;
    args[abi::kArgCount] = kPartialEvictLines;
    const std::string label = fmt::format(
        "DirtyEvictionPartialLine on CCE{} hart {}, {} lines at TL1 0x{:x}, {} eviction reads from 0x{:x}",
        cce_index,
        kProducerHart,
        kPartialLines,
        data_offset,
        kPartialEvictLines,
        sweep_offset);
    log_info(tt::LogTest, "MM41 {}", label);
    const StagedRun run = run_together(device, {cache_kernel(cce_index, kProducerHart, args)}, kMm41HostDeadline);
    ASSERT_FALSE(run.statuses.empty()) << label << ": no status after " << kMm41HostDeadline.count() << " s"
                                       << kFreshSessionNote;
    const std::vector<uint32_t>& status = run.statuses.front();
    log_info(tt::LogTest, "MM41 {}: {}", label, mm41_detail(status));
    RecordProperty("lines_written_back_by_eviction", static_cast<int>(status[abi::kStatusRegisterA]));
    RecordProperty("stale_words_before_drop", static_cast<int>(status[abi::kStatusCachedBefore]));
    ASSERT_EQ(status[abi::kStatusCode], abi::kCodeDone) << label << ": " << mm41_detail(status);
    EXPECT_GT(status[abi::kStatusRegisterA], 0u) << label << ": no partial line reached TL1 before the flush, so no "
                                                 << "dirty eviction was tested";
    EXPECT_EQ(status[abi::kStatusRegisterB], 0u) << label << ": untouched bytes changed after the eviction reads";
    EXPECT_EQ(status[abi::kStatusRegisterC], 0u) << label << ": untouched bytes changed after the flush";
    EXPECT_EQ(status[abi::kStatusMatches], kPartialLines) << label << ": " << mm41_detail(status);
    expect_host_on_cce(device, cce_index, data_offset, partial_line_image(value, kPartialLines), tag, label);
}

// cflush.d.l1 per line (Line) or of the whole L1 D$ (Full), then an uncached read before any L2 operation records
// whether the L1 flush reached TL1 or stopped in L2 (plan "needs investigation"). Then the line discard and the L2
// range flush, after which TL1 must hold the stores, as MM-8 Op/L1FlushLine and Op/L1FlushFull on CCE0.
void run_l1_flush_target(uint32_t cce_index, uint32_t mode, const std::string& name) {
    ScopedDevice device_closer;
    IDevice* device = device_closer.device();
    check_tl1_layout();
    record_descriptor_properties(soc_desc_of(device));

    const uint64_t data_offset = tl1_data(kProducerHart);
    const uint32_t tag = 0x5C + cce_index;
    const uint32_t value = 0x1F000000u | (mode << 16) | (cce_index << 12);
    prefill_on_cce(device, cce_index, data_offset, pattern_words(kOldValue, kCacheWords), tag);
    std::vector<uint32_t> args = mm41_args(mode, kProducerHart);
    args[abi::kArgDataOffset] = static_cast<uint32_t>(data_offset);
    args[abi::kArgNumWords] = kCacheWords;
    args[abi::kArgValue] = value;
    const std::string label = fmt::format(
        "{} on CCE{} hart {}, TL1 0x{:x}, {} words", name, cce_index, kProducerHart, data_offset, kCacheWords);
    log_info(tt::LogTest, "MM41 {}", label);
    const StagedRun run = run_together(device, {cache_kernel(cce_index, kProducerHart, args)}, kMm41HostDeadline);
    ASSERT_FALSE(run.statuses.empty()) << label << ": no status after " << kMm41HostDeadline.count() << " s"
                                       << kFreshSessionNote;
    const std::vector<uint32_t>& status = run.statuses.front();
    log_info(tt::LogTest, "MM41 {}: {}", label, mm41_detail(status));
    const uint32_t after_l1_flush = status[abi::kStatusRegisterA];
    ::testing::Test::RecordProperty(
        "l1_flush_target",
        after_l1_flush == kCacheWords ? "TL1 (complete before the L2 flush)"
        : after_l1_flush == 0         ? "L2 (no word in TL1 before the L2 flush)"
                                      : fmt::format("partial: {} of {} words in TL1", after_l1_flush, kCacheWords));
    ASSERT_EQ(status[abi::kStatusCode], abi::kCodeDone) << label << ": " << mm41_detail(status);
    EXPECT_EQ(status[abi::kStatusCachedBefore], kCacheWords) << label << ": cached view after the stores";
    EXPECT_EQ(status[abi::kStatusUncachedBefore], 0u) << label << ": cached stores visible before any flush";
    EXPECT_EQ(status[abi::kStatusMatches], kCacheWords) << label << ": " << mm41_detail(status);
    expect_host_on_cce(device, cce_index, data_offset, pattern_words(value, kCacheWords), tag, label);
}

TEST_P(MimirCacheOps, L1DcacheFlushLine) {
    run_l1_flush_target(GetParam(), abi::kModeL1FlushLineTarget, "L1DcacheFlushLine");
}

TEST_P(MimirCacheOps, L1DcacheFlushFull) {
    run_l1_flush_target(GetParam(), abi::kModeL1FlushFullTarget, "L1DcacheFlushFull");
}

INSTANTIATE_TEST_SUITE_P(MM41, MimirCacheOps, ::testing::ValuesIn(mimir_contract::cce_indices()), cce_param_name);

enum class SharedForm : uint32_t { WriterReader, EightWriters };

using SharedParam = std::tuple<uint32_t, SharedForm>;

std::string shared_param_name(const ::testing::TestParamInfo<SharedParam>& info) {
    return fmt::format(
        "Cce{}_{}",
        std::get<0>(info.param),
        std::get<1>(info.param) == SharedForm::WriterReader ? "WriterReader" : "EightWriters");
}

// The section 4.1 shared-L2 layouts. WriterReader: the writer's words and the reader's warmed words in a two-line
// region. EightWriters: hart s stores kSharedWriterWords words at s * stride from the region start.
struct SharedLayout {
    const char* name;
    uint32_t writer_first_word;
    uint32_t writer_words;
    uint32_t reader_first_word;
    uint32_t eight_writer_words;
    uint32_t eight_writer_stride_bytes;
};

inline constexpr SharedLayout kSameLineSameOffset{"SameLineSameOffset", 0, 4, 0, 4, 0};
inline constexpr SharedLayout kSameLineDiffOffset{"SameLineDiffOffset", 8, 4, 0, 2, 8};
inline constexpr SharedLayout kAdjacentLine{"AdjacentLine", 16, 4, 0, 4, abi::kLineBytes};
inline constexpr uint32_t kSharedRegionWords = 8 * kWordsPerLine;
inline constexpr uint32_t kSharedPasses = 16;

class MimirCacheSharedL2 : public ::testing::TestWithParam<SharedParam> {
protected:
    void SetUp() override {
        skip_unless_programmable_dram_package();
        if (const std::string skip_reason = mimir_watcher_skip_reason(); !skip_reason.empty()) {
            GTEST_SKIP() << skip_reason;
        }
        if (const std::string skip_reason = fresh_session_skip_reason(); !skip_reason.empty()) {
            GTEST_SKIP() << skip_reason;
        }
    }

    // WriterReader: the reader warms its words, the writer stores its words cached, `fence w, w` and sets the flag; the
    // reader reads the region cached with no cache operation, which is recorded only (it needs L1-to-L1 coherence that
    // nothing has shown). The writer then runs cflush.d.l1 per line, with no L2 flush, and sets a third flag; the
    // reader discards the region from its L1 D$ only and must read the writer's words new through the shared L2. The
    // writer then range-flushes and drains its words and sets a second flag; the reader invalidates the region and
    // must read the writer's words new, every other word old. EightWriters: every hart stores its words for
    // kSharedPasses passes at once, then flushes them; TL1 must hold every hart's last values (SameLineSameOffset: each
    // word is some hart's last value).
    void run_layout(const SharedLayout& layout) {
        const uint32_t cce_index = std::get<0>(GetParam());
        const SharedForm form = std::get<1>(GetParam());
        ScopedDevice device_closer;
        IDevice* device = device_closer.device();
        check_tl1_layout();
        record_descriptor_properties(soc_desc_of(device));

        const uint64_t region_offset = tl1_data(kProducerHart);
        const uint64_t flag_offset = region_offset + kFlagLinesFromData;
        const uint32_t tag = 0x60 + cce_index;
        constexpr uint32_t value = 0x5A000000u;
        prefill_on_cce(device, cce_index, region_offset, pattern_words(kOldValue, kSharedRegionWords), tag);
        write_cce_tl1(device, cce_index, flag_offset, std::vector<uint32_t>(cce_hart_count() * kWordsPerLine, 0));
        std::vector<uint32_t> expected = pattern_words(kOldValue, kSharedRegionWords);
        if (form == SharedForm::WriterReader) {
            const uint64_t writer_offset = region_offset + layout.writer_first_word * sizeof(uint32_t);
            std::vector<uint32_t> writer_args = mm41_args(abi::kModeSharedWriter, kWriterHart);
            writer_args[abi::kArgDataOffset] = static_cast<uint32_t>(writer_offset);
            writer_args[abi::kArgNumWords] = layout.writer_words;
            writer_args[abi::kArgValue] = value;
            writer_args[abi::kArgFlagOffset] = static_cast<uint32_t>(flag_offset);
            std::vector<uint32_t> reader_args = mm41_args(abi::kModeSharedReader, kReaderHart);
            reader_args[abi::kArgDataOffset] =
                static_cast<uint32_t>(region_offset + layout.reader_first_word * sizeof(uint32_t));
            reader_args[abi::kArgNumWords] = layout.writer_words;
            reader_args[abi::kArgValue] = value;
            reader_args[abi::kArgFlagOffset] = static_cast<uint32_t>(flag_offset);
            reader_args[abi::kArgAuxOffset] = static_cast<uint32_t>(region_offset);
            reader_args[abi::kArgCount] = kSharedRegionWords;
            reader_args[abi::kArgOtherOffset] = static_cast<uint32_t>(writer_offset);
            reader_args[abi::kArgOtherWords] = layout.writer_words;
            const std::string label = fmt::format(
                "{} on CCE{}: writer hart {} words {}..{}, reader hart {} warms words {}..{}, region TL1 0x{:x}",
                layout.name,
                cce_index,
                kWriterHart,
                layout.writer_first_word,
                layout.writer_first_word + layout.writer_words - 1,
                kReaderHart,
                layout.reader_first_word,
                layout.reader_first_word + layout.writer_words - 1,
                region_offset);
            log_info(tt::LogTest, "MM41 {}", label);
            const StagedRun run = run_together(
                device,
                {cache_kernel(cce_index, kReaderHart, reader_args), cache_kernel(cce_index, kWriterHart, writer_args)},
                kMm41HostDeadline);
            ASSERT_FALSE(run.statuses.empty())
                << label << ": no status after " << kMm41HostDeadline.count() << " s" << kFreshSessionNote;
            const std::vector<uint32_t>& reader = run.statuses[0];
            const std::vector<uint32_t>& writer = run.statuses[1];
            log_info(tt::LogTest, "MM41 {}: reader {}; writer {}", label, mm41_detail(reader), mm41_detail(writer));
            RecordProperty("stale_words_before_drop", static_cast<int>(reader[abi::kStatusCachedBefore]));
            RecordProperty("writer_words_in_tl1_at_flag", static_cast<int>(writer[abi::kStatusUncachedBefore]));
            RecordProperty("writer_words_in_tl1_at_read", static_cast<int>(reader[abi::kStatusRegisterA]));
            RecordProperty(
                "no_flush_cached_read",
                fmt::format(
                    "{} of {} words as expected after the writer's fence alone (L1-to-L1 coherence)",
                    reader[abi::kStatusRegisterB],
                    kSharedRegionWords));
            ASSERT_EQ(writer[abi::kStatusCode], abi::kCodeDone) << label << ": writer " << mm41_detail(writer);
            ASSERT_EQ(reader[abi::kStatusCode], abi::kCodeDone) << label << ": reader " << mm41_detail(reader);
            EXPECT_EQ(reader[abi::kStatusWarmOld], layout.writer_words) << label << ": reader's warm read";
            EXPECT_EQ(writer[abi::kStatusCachedBefore], layout.writer_words) << label << ": writer's cached view";
            EXPECT_EQ(reader[abi::kStatusRegisterC], kSharedRegionWords)
                << label << ": reader's cached read after the writer's cflush.d.l1 and its own L1 D$ discard, with no "
                << "L2 flush; " << mm41_detail(reader);
            EXPECT_EQ(reader[abi::kStatusMatches], kSharedRegionWords)
                << label << ": reader's cached read after the writer's flush and its own invalidate; "
                << mm41_detail(reader);
            for (uint32_t word = 0; word < layout.writer_words; ++word) {
                expected[layout.writer_first_word + word] = value + word;
            }
            expect_host_on_cce(device, cce_index, region_offset, expected, tag, label);
            return;
        }

        std::vector<HartKernel> kernels;
        for (uint32_t hart = 0; hart < cce_hart_count(); ++hart) {
            std::vector<uint32_t> args = mm41_args(abi::kModeSharedConcurrentWriter, hart);
            args[abi::kArgDataOffset] = static_cast<uint32_t>(region_offset);
            args[abi::kArgNumWords] = layout.eight_writer_words;
            args[abi::kArgValue] = value;
            args[abi::kArgFlagOffset] = static_cast<uint32_t>(flag_offset);
            args[abi::kArgHartSlot] = hart;
            args[abi::kArgNumHarts] = cce_hart_count();
            args[abi::kArgPasses] = kSharedPasses;
            args[abi::kArgStrideBytes] = layout.eight_writer_stride_bytes;
            kernels.push_back(cache_kernel(cce_index, hart, args));
        }
        const std::string label = fmt::format(
            "{} on CCE{}: {} harts each store {} words at stride {} B, {} passes, region TL1 0x{:x}",
            layout.name,
            cce_index,
            cce_hart_count(),
            layout.eight_writer_words,
            layout.eight_writer_stride_bytes,
            kSharedPasses,
            region_offset);
        log_info(tt::LogTest, "MM41 {}", label);
        const StagedRun run = run_together(device, kernels, kMm41HostDeadline);
        ASSERT_FALSE(run.statuses.empty())
            << label << ": no status after " << kMm41HostDeadline.count() << " s" << kFreshSessionNote;
        for (uint32_t hart = 0; hart < cce_hart_count(); ++hart) {
            const std::vector<uint32_t>& status = run.statuses[hart];
            log_info(tt::LogTest, "MM41 {}: hart {} {}", label, hart, mm41_detail(status));
            EXPECT_EQ(status[abi::kStatusCode], abi::kCodeDone)
                << label << ": hart " << hart << " " << mm41_detail(status);
            EXPECT_EQ(status[abi::kStatusPasses], kSharedPasses) << label << ": hart " << hart;
            if (layout.eight_writer_stride_bytes != 0) {
                EXPECT_EQ(status[abi::kStatusCachedBefore], layout.eight_writer_words)
                    << label << ": hart " << hart << " cached view of its own last values";
            }
        }
        const auto last_value = [](uint32_t hart, uint32_t word) {
            return value + (hart << 16) + ((kSharedPasses - 1) << 8) + word;
        };
        const std::vector<uint32_t> observed =
            read_cce_tl1(device, cce_index, region_offset, kSharedRegionWords * sizeof(uint32_t));
        if (layout.eight_writer_stride_bytes == 0) {
            std::string winners;
            for (uint32_t word = 0; word < layout.eight_writer_words; ++word) {
                bool from_some_hart = false;
                for (uint32_t hart = 0; hart < cce_hart_count(); ++hart) {
                    if (observed[word] == last_value(hart, word)) {
                        from_some_hart = true;
                        winners += fmt::format("{}{}", winners.empty() ? "" : " ", hart);
                    }
                }
                EXPECT_TRUE(from_some_hart)
                    << label << ": word " << word << " 0x" << std::hex << observed[word] << " is no hart's last value";
                expected[word] = observed[word];
            }
            RecordProperty("last_writer_by_word", winners);
        } else {
            for (uint32_t hart = 0; hart < cce_hart_count(); ++hart) {
                for (uint32_t word = 0; word < layout.eight_writer_words; ++word) {
                    expected[hart * layout.eight_writer_stride_bytes / sizeof(uint32_t) + word] =
                        last_value(hart, word);
                }
            }
        }
        expect_host_on_cce(device, cce_index, region_offset, expected, tag, label);
    }
};

TEST_P(MimirCacheSharedL2, SameLineSameOffset) { run_layout(kSameLineSameOffset); }

TEST_P(MimirCacheSharedL2, SameLineDiffOffset) { run_layout(kSameLineDiffOffset); }

TEST_P(MimirCacheSharedL2, AdjacentLine) { run_layout(kAdjacentLine); }

INSTANTIATE_TEST_SUITE_P(
    MM41,
    MimirCacheSharedL2,
    ::testing::Combine(
        ::testing::ValuesIn(mimir_contract::cce_indices()),
        ::testing::Values(SharedForm::WriterReader, SharedForm::EightWriters)),
    shared_param_name);

// External writers into a reader hart's TL1 data (section 4.1 external-writer coherence).
enum class ExternalWriter : uint32_t { Host, LocalIdma, LocalGddrIdma, SiblingSpa, RemoteCce, RemoteGddrIdma };
// SnoopOn: the writer's writes carry the snoop bit and the reader does no invalidate. SnoopOffInvalidate: unsnooped
// writes, then INVALIDATE64 before the final read. InvalidateOnly: unsnooped writes and only INVALIDATE64, with no
// cached read between the write and the invalidate.
enum class SnoopMode : uint32_t { SnoopOn, SnoopOffInvalidate, InvalidateOnly };

using WriterParam = std::tuple<uint32_t, ExternalWriter, SnoopMode>;

std::string writer_name(ExternalWriter writer) {
    switch (writer) {
        case ExternalWriter::Host: return "Host";
        case ExternalWriter::LocalIdma: return "LocalIdma";
        case ExternalWriter::LocalGddrIdma: return "LocalGddrIdma";
        case ExternalWriter::SiblingSpa: return "SiblingSpa";
        case ExternalWriter::RemoteCce: return "RemoteCce";
        case ExternalWriter::RemoteGddrIdma: return "RemoteGddrIdma";
    }
    return "Unknown";
}

std::string snoop_mode_name(SnoopMode mode) {
    switch (mode) {
        case SnoopMode::SnoopOn: return "SnoopOn";
        case SnoopMode::SnoopOffInvalidate: return "SnoopOffInvalidate";
        case SnoopMode::InvalidateOnly: return "InvalidateOnly";
    }
    return "Unknown";
}

std::string writer_param_name(const ::testing::TestParamInfo<WriterParam>& info) {
    return fmt::format(
        "Cce{}_{}_{}",
        std::get<0>(info.param),
        writer_name(std::get<1>(info.param)),
        snoop_mode_name(std::get<2>(info.param)));
}

// The host and a remote CCE's SPA stores have no snoop-on mechanism (plan R2 reaches only same-Mimir writers). Every
// iDMA writer here is the reader's own CCE, and the hardware never snoops a CCE's iDMA writes into its own TL1
// (GRENSV-3774, GRENSV-1577), so SnoopOn runs only for SiblingSpa.
std::vector<WriterParam> writer_params() {
    std::vector<WriterParam> params;
    for (const uint32_t cce_index : mimir_contract::cce_indices()) {
        for (const ExternalWriter writer :
             {ExternalWriter::Host,
              ExternalWriter::LocalIdma,
              ExternalWriter::LocalGddrIdma,
              ExternalWriter::SiblingSpa,
              ExternalWriter::RemoteCce,
              ExternalWriter::RemoteGddrIdma}) {
            for (const SnoopMode mode :
                 {SnoopMode::SnoopOn, SnoopMode::SnoopOffInvalidate, SnoopMode::InvalidateOnly}) {
                if (!(mode == SnoopMode::SnoopOn && writer != ExternalWriter::SiblingSpa)) {
                    params.emplace_back(cce_index, writer, mode);
                }
            }
        }
    }
    return params;
}

uint32_t reader_steps(SnoopMode mode) {
    switch (mode) {
        case SnoopMode::SnoopOn: return abi::kStepDiscriminate;
        case SnoopMode::SnoopOffInvalidate: return abi::kStepDiscriminate | abi::kStepL2Invalidate;
        case SnoopMode::InvalidateOnly: return abi::kStepL2Invalidate;
    }
    return 0;
}

// What a kModeExternalReader status, run with `steps`, shows about the writer's effect on the reader's cached copy.
std::string snoop_verdict(const std::vector<uint32_t>& status, uint32_t num_words, uint32_t steps) {
    if (status[abi::kStatusWarmOld] != num_words) {
        return fmt::format(
            "inconclusive: the warm read saw {} of {} old words, so the writer landed first",
            status[abi::kStatusWarmOld],
            num_words);
    }
    if ((steps & abi::kStepDiscriminate) == 0) {
        return status[abi::kStatusMatches] == num_words
                   ? "fresh after INVALIDATE64 alone"
                   : fmt::format(
                         "{} of {} words fresh after INVALIDATE64 alone (L1 D$ kept the old line)",
                         status[abi::kStatusMatches],
                         num_words);
    }
    if (status[abi::kStatusRegisterB] == 0) {
        return fmt::format(
            "cached re-read with no invalidate: {} of {} words old (snooped)",
            status[abi::kStatusStaleBefore],
            num_words);
    }
    return status[abi::kStatusRegisterA] == num_words
               ? fmt::format(
                     "{} words stale; fresh after the L1 D$ discard alone (L2 snooped, L1 D$ held the old line)",
                     status[abi::kStatusStaleBefore])
               : fmt::format(
                     "{} words stale; {} fresh after the L1 D$ discard (not snooped)",
                     status[abi::kStatusStaleBefore],
                     status[abi::kStatusRegisterA]);
}

// kModeExternalReader on `cce_index` hart kReaderHart over `region_offset`, `flag_offset` 0 for a writer that does not
// wait for the warmed flag.
HartKernel external_reader_kernel(
    uint32_t cce_index,
    uint64_t region_offset,
    uint32_t num_words,
    uint32_t value,
    uint64_t flag_offset,
    uint32_t steps) {
    std::vector<uint32_t> args = mm41_args(abi::kModeExternalReader, kReaderHart);
    args[abi::kArgDataOffset] = static_cast<uint32_t>(region_offset);
    args[abi::kArgNumWords] = num_words;
    args[abi::kArgValue] = value;
    args[abi::kArgFlagOffset] = static_cast<uint32_t>(flag_offset);
    args[abi::kArgSteps] = steps;
    return cache_kernel(cce_index, kReaderHart, args);
}

HartKernel spa_writer_kernel(
    uint32_t writer_cce,
    uint32_t target_cce,
    uint64_t region_offset,
    uint32_t num_words,
    uint32_t value,
    uint64_t flag_offset) {
    std::vector<uint32_t> args = mm41_args(abi::kModeSpaWriter, kReaderHart);
    args[abi::kArgDataOffset] = static_cast<uint32_t>(region_offset);
    args[abi::kArgNumWords] = num_words;
    args[abi::kArgValue] = value;
    args[abi::kArgFlagOffset] = static_cast<uint32_t>(flag_offset);
    args[abi::kArgTargetCce] = target_cce;
    return cache_kernel(writer_cce, kReaderHart, args);
}

inline constexpr uint32_t kExternalWords = 4 * kWordsPerLine;
inline constexpr uint32_t kIdmaCommands = 7;
inline constexpr uint32_t kIdmaCommandBytes = idma_abi::kMaxBytesInPacket;
inline constexpr uint32_t kIdmaSourceWords = kIdmaCommandBytes / sizeof(uint32_t);
// The reader's region is the last kExternalWords of the last iDMA command's destination, so it lands after the other
// commands' 48 KiB; an odd command count keeps the destination XOR equal to the source XOR.
inline constexpr uint64_t kExternalRegionFromData =
    static_cast<uint64_t>(kIdmaCommands) * kIdmaCommandBytes - kExternalWords * sizeof(uint32_t);
static_assert(kIdmaCommands % 2 == 1);

// A source block whose last kExternalWords words are value + i.
std::vector<uint32_t> idma_source_words(uint32_t value) {
    return pattern_words(value - (kIdmaSourceWords - kExternalWords), kIdmaSourceWords);
}

class MimirCacheWriterProbe : public ::testing::TestWithParam<WriterParam> {
protected:
    void SetUp() override {
        skip_unless_programmable_dram_package();
        skip_unless_mimir_local_package();
        if (const std::string skip_reason = mimir_watcher_skip_reason(); !skip_reason.empty()) {
            GTEST_SKIP() << skip_reason;
        }
        if (const std::string skip_reason = fresh_session_skip_reason(); !skip_reason.empty()) {
            GTEST_SKIP() << skip_reason;
        }
        const ExternalWriter writer = std::get<1>(GetParam());
        if (writer == ExternalWriter::RemoteCce || writer == ExternalWriter::RemoteGddrIdma) {
            if (const std::string skip_reason = two_mimir_skip_reason(); !skip_reason.empty()) {
                GTEST_SKIP() << skip_reason;
            }
        }
    }
};

// One writer and one snoop mode against a reader hart that warmed the lines cached (plan PC4). Records whether the
// writer's TL1 writes reach the reader's cached copy, separating L2 from L1 D$ with a per-line discard. Gates on the
// warm read seeing the old values, on the invalidated read being fresh (SnoopOffInvalidate, InvalidateOnly), on a
// snooped writer refreshing the cached copy with no invalidate (SnoopOn, SiblingSpa only: R2's RegionAttrs cmd_snoop
// per GRENSV-2290), and on the data in TL1. Every writer waits for the reader's warmed flag; a same-CCE iDMA writer
// starts with the reader in one Program and polls the flag before it issues. The iDMA writer writes 56 KiB of which
// the reader's lines are the end, so the reader also waits for the writer's done magic, which the writer stores only
// after its whole destination has landed, before any store of its own (MIM-560).
TEST_P(MimirCacheWriterProbe, ExternalWriters) {
    const uint32_t cce_index = std::get<0>(GetParam());
    const ExternalWriter writer = std::get<1>(GetParam());
    const SnoopMode mode = std::get<2>(GetParam());
    ScopedDevice device_closer;
    IDevice* device = device_closer.device();
    check_tl1_layout();
    const metal_SocDescriptor& soc = soc_desc_of(device);
    record_descriptor_properties(soc);
    const Cluster& cluster = MetalContext::instance().get_cluster();

    const uint64_t data_offset = tl1_data(kReaderHart);
    const uint64_t region_offset = data_offset + kExternalRegionFromData;
    const uint64_t flag_offset = data_offset + kFlagLinesFromData;
    const uint32_t tag = 0x70 + static_cast<uint32_t>(writer) * 4 + static_cast<uint32_t>(mode);
    // The gtest seed is below 2^17, so it is spread over the high bits before the mask.
    const uint32_t value = 0x80000000u | ((mimir_test_seed() * 0x9E3779B9u) & 0x7FF00000u) |
                           (static_cast<uint32_t>(writer) << 16) | (static_cast<uint32_t>(mode) << 12);
    record_leg("writer", "seed", fmt::format("value 0x{:08x} from mimir_test_seed 0x{:08x}", value, mimir_test_seed()));
    prefill_on_cce(device, cce_index, region_offset, pattern_words(kOldValue, kExternalWords), tag);
    write_cce_tl1(device, cce_index, flag_offset, std::vector<uint32_t>(kWordsPerLine, 0));
    const bool snoop = mode == SnoopMode::SnoopOn;
    const uint32_t steps = reader_steps(mode);
    HartKernel reader = external_reader_kernel(cce_index, region_offset, kExternalWords, value, flag_offset, steps);

    std::string writer_detail;
    std::optional<IdmaWriter> idma_writer;
    std::optional<ScopedSramRemapSnoop> remap_snoop;
    StagedRun run;
    switch (writer) {
        case ExternalWriter::Host: {
            writer_detail = "host write_cce_tl1 after the warmed flag";
            run = run_staged(
                device,
                {reader},
                HostFlag{cce_index, flag_offset, abi::kFlagWarmed},
                [&] { write_cce_tl1(device, cce_index, region_offset, pattern_words(value, kExternalWords)); },
                {},
                kMm41HostDeadline);
            break;
        }
        case ExternalWriter::LocalIdma:
        case ExternalWriter::LocalGddrIdma:
        case ExternalWriter::RemoteGddrIdma: {
            const std::vector<uint32_t> source = idma_source_words(value);
            IdmaWriter batch{
                .cce_index = cce_index,
                .hart = kWriterHart,
                .destination = own_tl1_operand(cce_index, data_offset),
                .backend_vc =
                    writer == ExternalWriter::LocalIdma ? idma_abi::kBackendLocalSram : idma_abi::kBackendFromRemote,
                .len_bytes = kIdmaCommandBytes,
                .num_commands = kIdmaCommands,
                .snoop = snoop,
                .start_flag_offset = flag_offset};
            if (writer == ExternalWriter::LocalIdma) {
                write_cce_tl1(device, cce_index, tl1_data(kWriterHart), source);
                batch.source = own_tl1_operand(cce_index, tl1_data(kWriterHart));
            } else {
                const uint32_t view = writer == ExternalWriter::LocalGddrIdma
                                          ? local_dram_view_for_cce(soc, cce_index)
                                          : base_dram_view_for_mimir(soc, next_mimir(mimir_index_for_cce(cce_index)));
                mimir_contract::write_gddr_view(device_closer.mesh(), view, gddr_source_view_offset(), source);
                batch.source = gddr_operand(soc, view, gddr_source_view_offset());
                batch.host_verified = true;
                batch.expected_xor = xor_of(source);
            }
            writer_detail = fmt::format(
                "CCE{} hart {} cce_idma_copy TridBatch, {} x {} B, VC {}, snoop {}, source kind {}",
                cce_index,
                kWriterHart,
                kIdmaCommands,
                kIdmaCommandBytes,
                batch.backend_vc,
                snoop,
                batch.source.kind);
            idma_writer = batch;
            reader.args[abi::kArgPeerStatusOffset] = static_cast<uint32_t>(tl1_done_block(kWriterHart));
            reader.args[abi::kArgPeerMagic] = idma_abi::kDoneMagic;
            run = run_together(device, {reader, idma_writer_kernel(batch)}, kMm41HostDeadline);
            break;
        }
        case ExternalWriter::SiblingSpa:
        case ExternalWriter::RemoteCce: {
            const uint32_t writer_cce =
                writer == ExternalWriter::SiblingSpa ? sibling_cce(cce_index) : next_mimir_cce(cce_index);
            if (writer == ExternalWriter::SiblingSpa) {
                remap_snoop.emplace(cluster, device->id(), writer_cce, snoop);
                RecordProperty(
                    "writer_remap_attrs",
                    fmt::format("0x{:08x} (was 0x{:08x})", remap_snoop->applied(), remap_snoop->original()));
                ASSERT_EQ((remap_snoop->applied() & kRemapCmdSnoopBit) != 0, snoop)
                    << "CCE" << writer_cce << " remap entry 1 cmd_snoop did not take: attrs 0x" << std::hex
                    << remap_snoop->applied();
            }
            writer_detail = fmt::format(
                "CCE{} hart {} SPA stores after the warmed flag, remap cmd_snoop {}", writer_cce, kReaderHart, snoop);
            run = run_together(
                device,
                {reader, spa_writer_kernel(writer_cce, cce_index, region_offset, kExternalWords, value, flag_offset)},
                kMm41HostDeadline);
            break;
        }
    }
    const std::string label = fmt::format(
        "ExternalWriters {} {} on CCE{}: reader hart {} TL1 0x{:x} {} words; writer {}",
        writer_name(writer),
        snoop_mode_name(mode),
        cce_index,
        kReaderHart,
        region_offset,
        kExternalWords,
        writer_detail);
    log_info(tt::LogTest, "MM41 {}", label);
    record_leg("writer", "config", writer_detail);
    ASSERT_FALSE(run.statuses.empty()) << label << ": no status after " << kMm41HostDeadline.count() << " s"
                                       << kFreshSessionNote;
    ASSERT_TRUE(run.flag_seen) << label << ": the reader never set the warmed flag";
    const std::vector<uint32_t>& reader_status = run.statuses[0];
    const std::string verdict = snoop_verdict(reader_status, kExternalWords, steps);
    RecordProperty("stale_words_before_drop", static_cast<int>(reader_status[abi::kStatusCachedBefore]));
    log_info(tt::LogTest, "MM41 {}: reader {}; {}", label, mm41_detail(reader_status), verdict);
    record_leg("reader", "status", mm41_detail(reader_status));
    RecordProperty("snoop_verdict", verdict);
    if (run.statuses.size() > 1) {
        const std::vector<uint32_t>& writer_status = run.statuses[1];
        if (idma_writer.has_value()) {
            log_info(tt::LogTest, "MM41 {}: writer {}", label, idma_detail(writer_status));
            record_leg("writer", "status", idma_detail(writer_status));
            EXPECT_EQ(writer_status[idma_abi::kStatusCode], idma_success_code(*idma_writer))
                << label << ": writer " << idma_detail(writer_status)
                << idma_code_note(writer_status[idma_abi::kStatusCode]);
        } else {
            log_info(tt::LogTest, "MM41 {}: writer {}", label, mm41_detail(writer_status));
            record_leg("writer", "status", mm41_detail(writer_status));
            EXPECT_EQ(writer_status[abi::kStatusCode], abi::kCodeDone)
                << label << ": writer " << mm41_detail(writer_status);
        }
    }
    ASSERT_EQ(reader_status[abi::kStatusCode], abi::kCodeDone)
        << label << ": reader " << mm41_detail(reader_status) << cache_code_note(reader_status[abi::kStatusCode]);
    if (mode == SnoopMode::SnoopOffInvalidate || mode == SnoopMode::InvalidateOnly) {
        EXPECT_EQ(reader_status[abi::kStatusWarmOld], kExternalWords)
            << label << ": the writer landed before the warm read; " << verdict;
        EXPECT_EQ(reader_status[abi::kStatusMatches], kExternalWords)
            << label << ": cached read after INVALIDATE64; " << mm41_detail(reader_status);
    }
    if (mode == SnoopMode::SnoopOn) {
        const std::string note = "RegionAttrs cmd_snoop (R2) not honoured; it was on 2026-10-22";
        EXPECT_EQ(reader_status[abi::kStatusWarmOld], kExternalWords)
            << label << ": the writer landed before the warm read, so the leg cannot show the snoop; " << verdict;
        EXPECT_EQ(reader_status[abi::kStatusStaleBefore], 0u)
            << label << ": snooped write left the cached copy stale with no invalidate (" << verdict << "): " << note;
        EXPECT_EQ(reader_status[abi::kStatusMatches], kExternalWords)
            << label << ": cached read with no invalidate; " << mm41_detail(reader_status) << ": " << note;
    }
    expect_host_on_cce(device, cce_index, region_offset, pattern_words(value, kExternalWords), tag, label);
}

INSTANTIATE_TEST_SUITE_P(MM41, MimirCacheWriterProbe, ::testing::ValuesIn(writer_params()), writer_param_name);

class MimirCacheCoherence : public Mm41CceCase {};

// `owner_cce` hart kProducerHart stores cached in its own TL1, flushes and sets the flag (kModeProducer);
// `remote_reader_cce` sets the warmed flag and then reads the data by SPA load once it sees the ready flag.
void run_flush_remote_read(
    IDevice* device, uint32_t owner_cce, uint32_t remote_reader_cce, uint32_t tag, const std::string& name) {
    const uint64_t data_offset = tl1_data(kProducerHart);
    const uint64_t flag_offset = data_offset + kFlagLinesFromData;
    const uint32_t value = 0xC0000000u | (owner_cce << 20) | (remote_reader_cce << 16);
    prefill_on_cce(device, owner_cce, data_offset, pattern_words(kOldValue, kCacheWords), tag);
    write_cce_tl1(device, owner_cce, flag_offset, std::vector<uint32_t>(kWordsPerLine, 0));
    std::vector<uint32_t> producer_args = mm41_args(abi::kModeProducer, kProducerHart);
    producer_args[abi::kArgDataOffset] = static_cast<uint32_t>(data_offset);
    producer_args[abi::kArgNumWords] = kCacheWords;
    producer_args[abi::kArgValue] = value;
    producer_args[abi::kArgFlagOffset] = static_cast<uint32_t>(flag_offset);
    std::vector<uint32_t> reader_args = mm41_args(abi::kModeSpaReader, kReaderHart);
    reader_args[abi::kArgDataOffset] = static_cast<uint32_t>(data_offset);
    reader_args[abi::kArgNumWords] = kCacheWords;
    reader_args[abi::kArgValue] = value;
    reader_args[abi::kArgFlagOffset] = static_cast<uint32_t>(flag_offset);
    reader_args[abi::kArgTargetCce] = owner_cce;
    const std::string label = fmt::format(
        "{} flush then remote read: CCE{} hart {} stores and flushes TL1 0x{:x}, CCE{} hart {} reads it by SPA",
        name,
        owner_cce,
        kProducerHart,
        data_offset,
        remote_reader_cce,
        kReaderHart);
    log_info(tt::LogTest, "MM41 {}", label);
    const StagedRun run = run_together(
        device,
        {cache_kernel(owner_cce, kProducerHart, producer_args),
         cache_kernel(remote_reader_cce, kReaderHart, reader_args)},
        kMm41HostDeadline);
    ASSERT_FALSE(run.statuses.empty()) << label << ": no status after " << kMm41HostDeadline.count() << " s"
                                       << kFreshSessionNote;
    const std::vector<uint32_t>& producer = run.statuses[0];
    const std::vector<uint32_t>& reader = run.statuses[1];
    log_info(tt::LogTest, "MM41 {}: producer {}; remote reader {}", label, mm41_detail(producer), mm41_detail(reader));
    ASSERT_EQ(producer[abi::kStatusCode], abi::kCodeDone) << label << ": producer " << mm41_detail(producer);
    ASSERT_EQ(reader[abi::kStatusCode], abi::kCodeDone) << label << ": remote reader " << mm41_detail(reader);
    EXPECT_EQ(producer[abi::kStatusCachedBefore], kCacheWords) << label << ": producer's cached view";
    EXPECT_EQ(producer[abi::kStatusMatches], kCacheWords) << label << ": producer's uncached view after the flush";
    EXPECT_EQ(reader[abi::kStatusMatches], kCacheWords)
        << label << ": remote reader after the flag; " << mm41_detail(reader);
    expect_host_on_cce(device, owner_cce, data_offset, pattern_words(value, kCacheWords), tag, label);
}

// `writer_cce` stores into `reader_cce`'s TL1 by SPA store once the reader warmed the lines; the reader reads cached
// (stale unless the writes are snooped), discards the lines from L1 D$, invalidates them in L2 and must read fresh.
void run_spa_write_invalidate(
    IDevice* device,
    uint32_t reader_cce,
    uint32_t writer_cce,
    bool writes_snooped,
    uint32_t tag,
    const std::string& name) {
    const uint64_t region_offset = tl1_data(kReaderHart);
    const uint64_t flag_offset = region_offset + kFlagLinesFromData;
    const uint32_t value = 0xC8000000u | (reader_cce << 20) | (writer_cce << 16);
    prefill_on_cce(device, reader_cce, region_offset, pattern_words(kOldValue, kCacheWords), tag);
    write_cce_tl1(device, reader_cce, flag_offset, std::vector<uint32_t>(kWordsPerLine, 0));
    const std::string label = fmt::format(
        "{} remote write then invalidate: CCE{} hart {} stores into CCE{} TL1 0x{:x} by SPA, CCE{} hart {} invalidates",
        name,
        writer_cce,
        kReaderHart,
        reader_cce,
        region_offset,
        reader_cce,
        kReaderHart);
    log_info(tt::LogTest, "MM41 {}", label);
    const StagedRun run = run_together(
        device,
        {external_reader_kernel(
             reader_cce,
             region_offset,
             kCacheWords,
             value,
             flag_offset,
             abi::kStepDiscriminate | abi::kStepL2Invalidate),
         spa_writer_kernel(writer_cce, reader_cce, region_offset, kCacheWords, value, flag_offset)},
        kMm41HostDeadline);
    ASSERT_FALSE(run.statuses.empty()) << label << ": no status after " << kMm41HostDeadline.count() << " s"
                                       << kFreshSessionNote;
    const std::vector<uint32_t>& reader = run.statuses[0];
    const std::vector<uint32_t>& writer = run.statuses[1];
    const std::string verdict = snoop_verdict(reader, kCacheWords, abi::kStepDiscriminate | abi::kStepL2Invalidate);
    log_info(
        tt::LogTest, "MM41 {}: reader {}; writer {}; {}", label, mm41_detail(reader), mm41_detail(writer), verdict);
    ::testing::Test::RecordProperty(name + "_snoop_verdict", verdict);
    ::testing::Test::RecordProperty(
        name + "_stale_words_before_drop", static_cast<int>(reader[abi::kStatusCachedBefore]));
    ASSERT_EQ(writer[abi::kStatusCode], abi::kCodeDone) << label << ": writer " << mm41_detail(writer);
    ASSERT_EQ(reader[abi::kStatusCode], abi::kCodeDone)
        << label << ": reader " << mm41_detail(reader) << cache_code_note(reader[abi::kStatusCode]);
    if (reader[abi::kStatusWarmOld] == kCacheWords) {
        EXPECT_EQ(reader[abi::kStatusStaleBefore], writes_snooped ? 0u : kCacheWords)
            << label << ": cached re-read before any invalidate (expected " << (writes_snooped ? "" : "not ")
            << "snooped, per the ExternalWriters probe)";
    }
    EXPECT_EQ(reader[abi::kStatusMatches], kCacheWords) << label << ": cached read after the invalidate";
    expect_host_on_cce(device, reader_cce, region_offset, pattern_words(value, kCacheWords), tag, label);
}

// Same-Mimir CCE pair, both directions across the CCE parameter: this CCE's flushed stores read by its sibling by
// SPA, and the sibling's SPA stores read by this CCE after an invalidate.
TEST_P(MimirCacheCoherence, CrossCce) {
    const uint32_t cce_index = GetParam();
    ScopedDevice device_closer;
    IDevice* device = device_closer.device();
    check_tl1_layout();
    record_descriptor_properties(soc_desc_of(device));
    const uint32_t sibling = sibling_cce(cce_index);
    RecordProperty("route", fmt::format("CCE{} <-> CCE{}", cce_index, sibling));
    run_flush_remote_read(device, cce_index, sibling, 0x80 + cce_index, "CrossCce");
    if (HasFatalFailure()) {
        return;
    }
    run_spa_write_invalidate(device, cce_index, sibling, kSiblingSpaWritesSnooped, 0x84 + cce_index, "CrossCce");
}

// The sibling CCE's iDMA writes into this CCE's warmed lines with the snoop bit set (cce_idma_copy kArgSnoop), after
// the reader set its warmed flag. The cached copy must be refreshed with no invalidate (the snoop packet tag on a
// sibling CCE's iDMA writes, GRENSV-3774, GRENSV-1577), and the read after the invalidate must be fresh.
TEST_P(MimirCacheCoherence, CrossCceSnoopOn) {
    const uint32_t cce_index = GetParam();
    ScopedDevice device_closer;
    IDevice* device = device_closer.device();
    check_tl1_layout();
    record_descriptor_properties(soc_desc_of(device));
    const uint32_t sibling = sibling_cce(cce_index);

    const uint64_t region_offset = tl1_data(kReaderHart);
    const uint64_t flag_offset = region_offset + kFlagLinesFromData;
    const uint32_t tag = 0x88 + cce_index;
    const uint32_t value = 0xCC000000u | (cce_index << 20) | (sibling << 16);
    prefill_on_cce(device, cce_index, region_offset, pattern_words(kOldValue, kCacheWords), tag);
    write_cce_tl1(device, cce_index, flag_offset, std::vector<uint32_t>(kWordsPerLine, 0));
    write_cce_tl1(device, sibling, tl1_data(kReaderHart), pattern_words(value, kCacheWords));
    const IdmaWriter writer{
        .cce_index = sibling,
        .hart = kReaderHart,
        .source = own_tl1_operand(sibling, tl1_data(kReaderHart)),
        .destination = sram_operand(cce_index, region_offset),
        .backend_vc = idma_abi::kBackendRemoteSram,
        .len_bytes = kCacheWords * sizeof(uint32_t),
        .snoop = true,
        .host_verified = true};
    const std::string label = fmt::format(
        "CrossCceSnoopOn: CCE{} iDMA (VC {}, snoop) into CCE{} TL1 0x{:x}, {} words, after the reader's warm",
        sibling,
        writer.backend_vc,
        cce_index,
        region_offset,
        kCacheWords);
    log_info(tt::LogTest, "MM41 {}", label);
    const StagedRun run = run_staged(
        device,
        {external_reader_kernel(
            cce_index,
            region_offset,
            kCacheWords,
            value,
            flag_offset,
            abi::kStepDiscriminate | abi::kStepL2Invalidate)},
        HostFlag{cce_index, flag_offset, abi::kFlagWarmed},
        {},
        {idma_writer_kernel(writer)},
        kMm41HostDeadline);
    ASSERT_FALSE(run.statuses.empty()) << label << ": no status after " << kMm41HostDeadline.count() << " s"
                                       << kFreshSessionNote;
    ASSERT_TRUE(run.second_launched) << label << ": the reader never set the warmed flag";
    const std::vector<uint32_t>& reader = run.statuses[0];
    const std::vector<uint32_t>& writer_status = run.statuses[1];
    const std::string verdict = snoop_verdict(reader, kCacheWords, abi::kStepDiscriminate | abi::kStepL2Invalidate);
    log_info(
        tt::LogTest,
        "MM41 {}: reader {}; writer {}; {}",
        label,
        mm41_detail(reader),
        idma_detail(writer_status),
        verdict);
    RecordProperty("snoop_verdict", verdict);
    RecordProperty("stale_words_before_drop", static_cast<int>(reader[abi::kStatusCachedBefore]));
    ASSERT_EQ(writer_status[idma_abi::kStatusCode], idma_success_code(writer))
        << label << ": writer " << idma_detail(writer_status) << idma_code_note(writer_status[idma_abi::kStatusCode]);
    ASSERT_EQ(reader[abi::kStatusCode], abi::kCodeDone)
        << label << ": reader " << mm41_detail(reader) << cache_code_note(reader[abi::kStatusCode]);
    EXPECT_EQ(reader[abi::kStatusWarmOld], kCacheWords) << label << ": warm read; " << verdict;
    EXPECT_EQ(reader[abi::kStatusStaleBefore], 0u)
        << label << ": snooped write left the cached copy stale (" << verdict << "): " << kIdmaSnoopNote;
    EXPECT_EQ(reader[abi::kStatusMatches], kCacheWords) << label << ": cached read after the invalidate";
    expect_host_on_cce(device, cce_index, region_offset, pattern_words(value, kCacheWords), tag, label);
}

// As CrossCce over D2D, with the CCE at the same location on the other Mimir (mimir_2x_package only). Gated on the
// ExternalWriters RemoteCce legs through kRemoteSpaWritesSnooped.
TEST_P(MimirCacheCoherence, CrossD2d) {
    const uint32_t cce_index = GetParam();
    if (const std::string skip_reason = two_mimir_skip_reason(); !skip_reason.empty()) {
        GTEST_SKIP() << skip_reason;
    }
    ScopedDevice device_closer;
    IDevice* device = device_closer.device();
    check_tl1_layout();
    record_descriptor_properties(soc_desc_of(device));
    const uint32_t remote = next_mimir_cce(cce_index);
    RecordProperty("route", fmt::format("CCE{} <-> CCE{} over D2D", cce_index, remote));
    run_flush_remote_read(device, cce_index, remote, 0x8C + cce_index, "CrossD2d");
    if (HasFatalFailure()) {
        return;
    }
    run_spa_write_invalidate(device, cce_index, remote, kRemoteSpaWritesSnooped, 0x90 + cce_index, "CrossD2d");
}

enum class RaceTarget : uint32_t { Neighbour, Same };

std::string race_target_name(RaceTarget target) { return target == RaceTarget::Neighbour ? "Neighbour" : "Same"; }

std::string race_action_name(uint32_t action) {
    switch (action) {
        case abi::kRaceActionEvict: return "Evict";
        case abi::kRaceActionFlush: return "Flush64";
        case abi::kRaceActionInvalidate: return "Invalidate64";
    }
    return fmt::format("unknown_action_{}", action);
}

inline constexpr uint32_t kRaceHartWords = 8;
inline constexpr uint64_t kRaceLineFromData = 0x800;
inline constexpr uint32_t kRaceHartValue = 0x4A000000u;
inline constexpr uint32_t kRaceExternalValue = 0xE4000000u;
inline constexpr uint32_t kSnoopMasterConfigs = 8;

INSTANTIATE_TEST_SUITE_P(MM41, MimirCacheCoherence, ::testing::ValuesIn(mimir_contract::cce_indices()), cce_param_name);

// One CCE and one NOC_SNOOP_TL_MASTER_CFG value per case, so a stall under one configuration loses no other.
using RaceParam = std::tuple<uint32_t, uint32_t>;

std::string race_param_name(const ::testing::TestParamInfo<RaceParam>& info) {
    return fmt::format("Cce{}_Cfg{}", std::get<0>(info.param), std::get<1>(info.param));
}

// Every CCE, the default configuration first.
std::vector<RaceParam> race_params() {
    std::vector<RaceParam> params;
    for (const uint32_t cce_index : mimir_contract::cce_indices()) {
        params.emplace_back(cce_index, kSnoopMasterDefault);
        for (uint32_t config = 0; config < kSnoopMasterConfigs; ++config) {
            if (config != kSnoopMasterDefault) {
                params.emplace_back(cce_index, config);
            }
        }
    }
    return params;
}

class MimirCacheCoherenceRace : public ::testing::TestWithParam<RaceParam> {
protected:
    void SetUp() override {
        skip_unless_programmable_dram_package();
        skip_unless_mimir_local_package();
        if (const std::string skip_reason = mimir_watcher_skip_reason(); !skip_reason.empty()) {
            GTEST_SKIP() << skip_reason;
        }
        if (const std::string skip_reason = fresh_session_skip_reason(); !skip_reason.empty()) {
            GTEST_SKIP() << skip_reason;
        }
    }
};

// Under one NOC_SNOOP_TL_MASTER_CFG value, for each target (the external words beside the hart's dirty words on the
// line, or the same words) and each hart action (conflict-stride eviction reads, FLUSH64, INVALIDATE64): the hart
// dirties its words cached and sets a flag, the sibling CCE's snooped iDMA writes the external words while the hart
// repeats the action until it reads them in TL1, and the host checks the line after a final FLUSH64. The snoop
// packet tag applies to the sibling's writes (GRENSV-3774, GRENSV-1577). The external words must survive under every
// configuration. On the Neighbour legs the hart's words must survive too under the configurations with inline writes
// enabled (bit 0 clear) unless the action is INVALIDATE64 (which may drop them); under the others they are recorded.
// The configuration is restored when the case ends. A leg whose hart ran out of budget before it read the external
// words stops the case: the iDMA may still be in flight (MIM-601).
TEST_P(MimirCacheCoherenceRace, DirtyLineRace) {
    const uint32_t cce_index = std::get<0>(GetParam());
    const uint32_t config = std::get<1>(GetParam());
    ScopedDevice device_closer;
    IDevice* device = device_closer.device();
    check_tl1_layout();
    record_descriptor_properties(soc_desc_of(device));
    const Cluster& cluster = MetalContext::instance().get_cluster();
    const uint32_t sibling = sibling_cce(cce_index);

    const uint32_t host_config = read_snoop_master_config(cluster, device->id(), cce_index);
    const std::optional<std::vector<std::vector<uint32_t>>> config_status = run_on_cce(
        device, cce_index, {{kProducerHart, probe_args(abi::kModeReadConfig, kProducerHart)}}, kProbeHostDeadline);
    ASSERT_TRUE(config_status.has_value()) << "SnoopConfig read on CCE" << cce_index << ": no status after "
                                           << kProbeHostDeadline.count() << " s" << kFreshSessionNote;
    ASSERT_EQ(config_status->front()[abi::kStatusCode], abi::kCodeDone);
    const uint32_t hart_config = config_status->front()[abi::kStatusRegisterA];
    RecordProperty("snoop_master_cfg_before", fmt::format("host 0x{:x}, hart 0x{:x}", host_config, hart_config));
    ASSERT_EQ(host_config, hart_config) << "The SMC-side NOC_SNOOP_TL_MASTER_CFG address of CCE" << cce_index
                                        << " does not read the register the hart reads; not writing it";

    const uint64_t line_offset = tl1_data(kReaderHart) + kRaceLineFromData;
    const uint64_t flag_offset = tl1_data(kReaderHart) + kFlagLinesFromData;
    TT_FATAL(
        line_offset + static_cast<uint64_t>(kL2EvictingReads) * kL2ConflictStrideBytes + abi::kLineBytes <=
            tl1_test_base() + kTl1RemoteLandingOffset,
        "DirtyLineRace eviction reads run into the remote landing area");
    write_cce_tl1(device, sibling, tl1_data(kReaderHart), pattern_words(kRaceExternalValue, kRaceHartWords));
    record_leg(
        "race",
        "config",
        fmt::format(
            "CCE{} line TL1 0x{:x}, {} hart words 0x{:08x}+i, external CCE{} iDMA VC {} snoop, {} words 0x{:08x}+i, "
            "eviction {} reads at stride {} B",
            cce_index,
            line_offset,
            kRaceHartWords,
            kRaceHartValue,
            sibling,
            idma_abi::kBackendRemoteSram,
            kRaceHartWords,
            kRaceExternalValue,
            kL2EvictingReads,
            kL2ConflictStrideBytes));

    const ScopedSnoopMasterConfig config_guard(cluster, device->id(), cce_index, config);
    ASSERT_EQ(config_guard.applied(), config)
        << "CCE" << cce_index << " NOC_SNOOP_TL_MASTER_CFG did not take the written value";
    RecordProperty("snoop_master_cfg", fmt::format("0x{:x}", config));
    const bool inline_writes_enabled = (config & 0x1) == 0;
    for (const RaceTarget target : {RaceTarget::Neighbour, RaceTarget::Same}) {
        for (const uint32_t action : {abi::kRaceActionEvict, abi::kRaceActionFlush, abi::kRaceActionInvalidate}) {
            const uint32_t tag = 0xA0 + action * 2 + static_cast<uint32_t>(target);
            const uint64_t external_offset =
                line_offset + (target == RaceTarget::Neighbour ? kRaceHartWords * sizeof(uint32_t) : 0);
            prefill_on_cce(device, cce_index, line_offset, pattern_words(kOldValue, kWordsPerLine), tag);
            write_cce_tl1(device, cce_index, flag_offset, std::vector<uint32_t>(kWordsPerLine, 0));
            std::vector<uint32_t> args = mm41_args(abi::kModeDirtyRace, kReaderHart);
            args[abi::kArgDataOffset] = static_cast<uint32_t>(line_offset);
            args[abi::kArgNumWords] = kRaceHartWords;
            args[abi::kArgValue] = kRaceHartValue;
            args[abi::kArgFlagOffset] = static_cast<uint32_t>(flag_offset);
            args[abi::kArgOtherOffset] = static_cast<uint32_t>(external_offset);
            args[abi::kArgOtherWords] = kRaceHartWords;
            args[abi::kArgOtherValue] = kRaceExternalValue;
            args[abi::kArgSteps] = action | (target == RaceTarget::Neighbour ? abi::kRaceRedirty : 0);
            // Well above what the poll budget allows for a FLUSH64 or INVALIDATE64 action, so the budget is the bound.
            args[abi::kArgPasses] = abi::kMaxRaceActions;
            args[abi::kArgAuxOffset] = static_cast<uint32_t>(line_offset + kL2ConflictStrideBytes);
            args[abi::kArgStrideBytes] = kL2ConflictStrideBytes;
            args[abi::kArgCount] = kL2EvictingReads;
            const IdmaWriter writer{
                .cce_index = sibling,
                .hart = kReaderHart,
                .source = own_tl1_operand(sibling, tl1_data(kReaderHart)),
                .destination = sram_operand(cce_index, external_offset),
                .backend_vc = idma_abi::kBackendRemoteSram,
                .len_bytes = kRaceHartWords * sizeof(uint32_t),
                .snoop = true,
                .host_verified = true};
            const std::string leg =
                fmt::format("cfg{}_{}_{}", config, race_target_name(target), race_action_name(action));
            const std::string label = fmt::format("DirtyLineRace on CCE{} {}", cce_index, leg);
            const StagedRun run = run_staged(
                device,
                {cache_kernel(cce_index, kReaderHart, args)},
                HostFlag{cce_index, flag_offset, abi::kFlagReady},
                {},
                {idma_writer_kernel(writer)},
                kMm41HostDeadline);
            ASSERT_FALSE(run.statuses.empty())
                << label << ": no status after " << kMm41HostDeadline.count() << " s" << kFreshSessionNote;
            ASSERT_TRUE(run.second_launched) << label << ": the hart never set its flag";
            const std::vector<uint32_t>& hart_status = run.statuses[0];
            const std::vector<uint32_t>& writer_status = run.statuses[1];
            const std::vector<uint32_t> observed =
                read_cce_tl1(device, cce_index, line_offset, kWordsPerLine * sizeof(uint32_t));
            EXPECT_EQ(read_cce_tl1(device, cce_index, line_offset + abi::kLineBytes, kGuardBytes), make_guard(tag))
                << label << ": guard overwritten";
            const uint32_t external_first = static_cast<uint32_t>((external_offset - line_offset) / sizeof(uint32_t));
            uint32_t external_kept = 0;
            uint32_t hart_new = 0;
            uint32_t hart_old = 0;
            for (uint32_t word = 0; word < kRaceHartWords; ++word) {
                external_kept += observed[external_first + word] == kRaceExternalValue + word ? 1 : 0;
                if (target == RaceTarget::Neighbour) {
                    hart_new += observed[word] == kRaceHartValue + word ? 1 : 0;
                    hart_old += observed[word] == kOldValue + word ? 1 : 0;
                }
            }
            const std::string result = fmt::format(
                "kernel {} after {} actions; external words kept {}/{}; hart words new {} old {}",
                code_name(hart_status[abi::kStatusCode]),
                hart_status[abi::kStatusPasses],
                external_kept,
                kRaceHartWords,
                hart_new,
                hart_old);
            log_info(tt::LogTest, "MM41 {}: {}; hart {}", label, result, mm41_detail(hart_status));
            RecordProperty(leg, result);
            record_leg(leg, "stale_words_before_drop", std::to_string(hart_status[abi::kStatusCachedBefore]));
            if (hart_status[abi::kStatusCode] == abi::kCodeRaceUnresolved) {
                const bool writer_acked = writer_status[idma_abi::kStatusCode] == idma_success_code(writer);
                const std::string writer_state =
                    writer_acked ? "the writer's command was acked, so its words were accepted but are not "
                                   "visible in TL1"
                                 : "the writer was not acked, so its iDMA is stuck";
                record_leg(leg, "writer", idma_detail(writer_status));
                fresh_session_required() = true;
                FAIL() << label << ": the hart's budget ran out before it read the external words, so the iDMA "
                       << "may still be in flight; " << writer_state << " (" << idma_detail(writer_status) << "); "
                       << result << (config == kSnoopMasterDefault ? std::string("; ") + kIdmaSnoopNote : "")
                       << kFreshSessionNote;
            }
            ASSERT_EQ(writer_status[idma_abi::kStatusCode], idma_success_code(writer))
                << label << ": writer " << idma_detail(writer_status)
                << idma_code_note(writer_status[idma_abi::kStatusCode]);
            EXPECT_EQ(hart_status[abi::kStatusCode], abi::kCodeDone)
                << label << ": " << result << ": " << kIdmaSnoopNote;
            EXPECT_EQ(external_kept, kRaceHartWords)
                << label << ": external write lost; " << result << ": " << kIdmaSnoopNote;
            if (target == RaceTarget::Neighbour && inline_writes_enabled && action != abi::kRaceActionInvalidate) {
                EXPECT_EQ(hart_new, kRaceHartWords)
                    << label << ": hart's dirty words lost; " << result << ": " << kIdmaSnoopNote;
            }
        }
    }
}

INSTANTIATE_TEST_SUITE_P(MM41, MimirCacheCoherenceRace, ::testing::ValuesIn(race_params()), race_param_name);

class MimirCacheStress : public Mm41CceCase {};

std::string thrash_pattern_name(uint32_t pattern) {
    switch (pattern) {
        case abi::kThrashSequential: return "Sequential";
        case abi::kThrashConflictStride: return "ConflictStride";
        case abi::kThrashRandom: return "Random";
    }
    return fmt::format("unknown_pattern_{}", pattern);
}

// Below, at and above the 4 KiB L1 D$ and the 128 KiB L2.
inline constexpr uint32_t kThrashSizesKiB[] = {2, 4, 8, 64, 128, 192};
inline constexpr uint32_t kThrashPasses = 4;

// Sequential, conflict-stride and seeded random passes over each size, verified on the device every pass and after
// a final flush; the host checks the kernel's counts, the final XOR against its model, a sample and the guard.
// Rdcycle per pass is recorded as the hit/miss/refill proxy (no CCE cache counters exist).
TEST_P(MimirCacheStress, ReuseThrash) {
    const uint32_t cce_index = GetParam();
    ScopedDevice device_closer;
    IDevice* device = device_closer.device();
    check_tl1_layout();
    record_descriptor_properties(soc_desc_of(device));

    const uint64_t data_offset = tl1_data(kProducerHart);
    const uint64_t trace_offset = tl1_done_block(kProducerHart) + kTraceFromDoneBlock;
    for (uint32_t pattern = abi::kThrashSequential; pattern <= abi::kThrashRandom; ++pattern) {
        for (uint32_t size_index = 0; size_index < std::size(kThrashSizesKiB); ++size_index) {
            const uint32_t num_words = kThrashSizesKiB[size_index] * 1024 / sizeof(uint32_t);
            const uint32_t tag = 0xB0 + pattern * 8 + size_index;
            const uint32_t seed = mimir_test_seed() ^ (tag * 0x9E3779B9u);
            const uint32_t value = 0x7E000000u | (tag << 12);
            write_cce_tl1(device, cce_index, data_offset + num_words * sizeof(uint32_t), make_guard(tag));
            std::vector<uint32_t> args = mm41_args(abi::kModeThrash, kProducerHart);
            args[abi::kArgDataOffset] = static_cast<uint32_t>(data_offset);
            args[abi::kArgNumWords] = num_words;
            args[abi::kArgValue] = value;
            args[abi::kArgSeed] = seed;
            args[abi::kArgPattern] = pattern;
            args[abi::kArgStrideBytes] = kL2ConflictStrideBytes;
            args[abi::kArgPasses] = kThrashPasses;
            args[abi::kArgTraceOffset] = static_cast<uint32_t>(trace_offset);
            const std::string leg = fmt::format("{}_{}KiB", thrash_pattern_name(pattern), kThrashSizesKiB[size_index]);
            const std::string label = fmt::format(
                "ReuseThrash {} on CCE{} hart {}, TL1 0x{:x}, {} passes, seed 0x{:08x}",
                leg,
                cce_index,
                kProducerHart,
                data_offset,
                kThrashPasses,
                seed);
            log_info(tt::LogTest, "MM41 {}", label);
            record_leg(
                leg,
                "config",
                fmt::format(
                    "{} words, stride {} B, {} passes, seed 0x{:08x}",
                    num_words,
                    kL2ConflictStrideBytes,
                    kThrashPasses,
                    seed));
            const StagedRun run =
                run_together(device, {cache_kernel(cce_index, kProducerHart, args)}, kStressHostDeadline);
            ASSERT_FALSE(run.statuses.empty())
                << label << ": no status after " << kStressHostDeadline.count() << " s" << kFreshSessionNote;
            const std::vector<uint32_t>& status = run.statuses.front();
            const std::vector<uint32_t> pass_cycles =
                read_cce_tl1(device, cce_index, trace_offset, kThrashPasses * sizeof(uint32_t));
            record_leg(leg, "pass_cycles", fmt::format("{}", fmt::join(pass_cycles, " ")));
            log_info(
                tt::LogTest, "MM41 {}: {}; pass cycles {}", label, mm41_detail(status), fmt::join(pass_cycles, " "));
            ASSERT_EQ(status[abi::kStatusCode], abi::kCodeDone) << label << ": " << mm41_detail(status);
            EXPECT_EQ(status[abi::kStatusPasses], kThrashPasses) << label;
            EXPECT_EQ(status[abi::kStatusRegisterA], 0u)
                << label << ": words that lost the previous pass's value; " << mm41_detail(status);
            EXPECT_EQ(status[abi::kStatusMatches], num_words) << label << ": TL1 after the final flush";
            uint32_t expected_xor = 0;
            std::vector<uint32_t> expected_first(kWordsPerLine);
            std::vector<uint32_t> expected_last(kWordsPerLine);
            for (uint32_t word = 0; word < num_words; ++word) {
                const uint32_t expected = abi::thrash_word(value, seed, kThrashPasses - 1, word);
                expected_xor ^= expected;
                if (word < kWordsPerLine) {
                    expected_first[word] = expected;
                }
                if (word >= num_words - kWordsPerLine) {
                    expected_last[word - (num_words - kWordsPerLine)] = expected;
                }
            }
            EXPECT_EQ(status[abi::kStatusRegisterB], expected_xor) << label << ": XOR of TL1 after the final flush";
            EXPECT_EQ(read_cce_tl1(device, cce_index, data_offset, abi::kLineBytes), expected_first)
                << label << ": first line";
            EXPECT_EQ(
                read_cce_tl1(
                    device, cce_index, data_offset + (num_words - kWordsPerLine) * sizeof(uint32_t), abi::kLineBytes),
                expected_last)
                << label << ": last line";
            EXPECT_EQ(
                read_cce_tl1(device, cce_index, data_offset + num_words * sizeof(uint32_t), kGuardBytes),
                make_guard(tag))
                << label << ": guard overwritten";
        }
    }
}

// 8 KiB, larger than the L1 D$: the hot set, and each round's iDMA preload destination.
inline constexpr uint32_t kChurnWords = idma_abi::kMaxBytesInPacket / sizeof(uint32_t);
inline constexpr uint32_t kChurnRounds = 4;

// One PreloadChurn leg: the consumer holds the hot set cached; each round the host launches the sibling CCE's iDMA
// preload from local GDDR into it once the consumer's round flag is up, and the consumer waits for the data in TL1
// (producer completion), invalidates in L2 (snoop off), reads it cached twice and checks it.
void run_churn_leg(distributed::MeshDevice& unit_mesh, IDevice* device, uint32_t cce_index, bool snoop) {
    const metal_SocDescriptor& soc = soc_desc_of(device);
    const uint32_t sibling = sibling_cce(cce_index);
    const uint64_t data_offset = tl1_data(kReaderHart);
    const uint64_t flag_offset = data_offset + kFlagLinesFromData;
    const uint32_t view = local_dram_view_for_cce(soc, cce_index);
    const uint32_t tag = 0xC0 + cce_index * 2 + (snoop ? 1 : 0);
    const uint32_t value = 0x30000000u | (snoop ? 0x08000000u : 0) | (cce_index << 24);
    const std::string leg = snoop ? "snoop" : "invalidate";
    prefill_on_cce(device, cce_index, data_offset, pattern_words(kOldValue, kChurnWords), tag);
    write_cce_tl1(device, cce_index, flag_offset, std::vector<uint32_t>(kWordsPerLine, 0));
    for (uint32_t round = 0; round < kChurnRounds; ++round) {
        mimir_contract::write_gddr_view(
            unit_mesh,
            view,
            gddr_source_view_offset() + static_cast<uint64_t>(round) * idma_abi::kMaxBytesInPacket,
            pattern_words(abi::churn_word(value, round, 0), kChurnWords));
    }
    std::vector<uint32_t> args = mm41_args(abi::kModeChurnConsumer, kReaderHart);
    args[abi::kArgDataOffset] = static_cast<uint32_t>(data_offset);
    args[abi::kArgNumWords] = kChurnWords;
    args[abi::kArgValue] = value;
    args[abi::kArgFlagOffset] = static_cast<uint32_t>(flag_offset);
    args[abi::kArgPasses] = kChurnRounds;
    args[abi::kArgSteps] = snoop ? 0 : abi::kStepL2Invalidate;
    const std::string label = fmt::format(
        "PreloadChurn {} on CCE{}: hot set TL1 0x{:x} {} B, {} rounds of CCE{} iDMA from GDDR view {} (VC {}, snoop "
        "{})",
        leg,
        cce_index,
        data_offset,
        kChurnWords * sizeof(uint32_t),
        kChurnRounds,
        sibling,
        view,
        idma_abi::kBackendFromRemote,
        snoop);
    log_info(tt::LogTest, "MM41 {}", label);
    record_leg(leg, "config", label);

    const HartKernel consumer = cache_kernel(cce_index, kReaderHart, args);
    write_cce_tl1(
        device, cce_index, tl1_done_block(kReaderHart), std::vector<uint32_t>(kMixedStatusWords, abi::kStatusPoison));
    const std::vector<CceKernelLaunch> consumer_launch = to_launches({consumer});
    Program consumer_program = CreateProgram();
    add_cce_kernels(device, consumer_program, consumer_launch);
    launch_on_idle_cces(device, consumer_program, consumer_launch);
    for (uint32_t round = 0; round < kChurnRounds; ++round) {
        if (!wait_for_host_flag(device, HostFlag{cce_index, flag_offset, round + 1}, kPollBudgetHostTime)) {
            ADD_FAILURE() << label << ": the consumer never asked for round " << round;
            break;
        }
        const IdmaWriter writer{
            .cce_index = sibling,
            .hart = kReaderHart,
            .source = gddr_operand(
                soc, view, gddr_source_view_offset() + static_cast<uint64_t>(round) * idma_abi::kMaxBytesInPacket),
            .destination = sram_operand(cce_index, data_offset),
            .backend_vc = idma_abi::kBackendFromRemote,
            .len_bytes = idma_abi::kMaxBytesInPacket,
            .snoop = snoop,
            .host_verified = true};
        const std::vector<CceKernelLaunch> writer_launch = to_launches({idma_writer_kernel(writer)});
        write_cce_tl1(
            device, sibling, tl1_done_block(kReaderHart), std::vector<uint32_t>(kMixedStatusWords, abi::kStatusPoison));
        Program writer_program = CreateProgram();
        add_cce_kernels(device, writer_program, writer_launch);
        launch_on_idle_cces(device, writer_program, writer_launch);
        ASSERT_TRUE(wait_for_status_magics(device, writer_launch, {idma_abi::kDoneMagic}, kMm41HostDeadline))
            << label << ": round " << round << " writer has no status after " << kMm41HostDeadline.count() << " s"
            << kFreshSessionNote;
        slow_dispatch::WaitProgramDone(*device, writer_program);
        const std::vector<uint32_t> writer_status =
            read_cce_tl1(device, sibling, tl1_done_block(kReaderHart), kMixedStatusWords * sizeof(uint32_t));
        ASSERT_EQ(writer_status[idma_abi::kStatusCode], idma_success_code(writer))
            << label << ": round " << round << " writer " << idma_detail(writer_status)
            << idma_code_note(writer_status[idma_abi::kStatusCode]);
    }
    ASSERT_TRUE(wait_for_status_magics(device, consumer_launch, {abi::kDoneMagic}, kMm41HostDeadline))
        << label << ": consumer has no status after " << kMm41HostDeadline.count() << " s" << kFreshSessionNote;
    slow_dispatch::WaitProgramDone(*device, consumer_program);
    const std::vector<uint32_t> status =
        read_cce_tl1(device, cce_index, tl1_done_block(kReaderHart), kMixedStatusWords * sizeof(uint32_t));
    const std::string result = fmt::format(
        "stale words before the drop {}, rounds {}, complete second reads {}, stale words at the first read {}, refill "
        "cycles {}, re-read cycles {}, warm hit cycles {}",
        status[abi::kStatusCachedBefore],
        status[abi::kStatusPasses],
        status[abi::kStatusMatches],
        status[abi::kStatusStaleBefore],
        status[abi::kStatusRegisterA],
        status[abi::kStatusRegisterB],
        status[abi::kStatusRegisterC]);
    log_info(tt::LogTest, "MM41 {}: {}; {}", label, result, mm41_detail(status));
    record_leg(leg, "result", result);
    record_leg(leg, "stale_words_before_drop", std::to_string(status[abi::kStatusCachedBefore]));
    ASSERT_EQ(status[abi::kStatusCode], abi::kCodeDone)
        << label << ": " << mm41_detail(status) << cache_code_note(status[abi::kStatusCode]);
    EXPECT_EQ(status[abi::kStatusWarmOld], kChurnWords) << label << ": warm read of the hot set";
    EXPECT_EQ(status[abi::kStatusPasses], kChurnRounds) << label;
    const std::string note = snoop ? std::string(": ") + kIdmaSnoopNote : std::string();
    EXPECT_EQ(status[abi::kStatusStaleBefore], 0u) << label << ": stale words consumed; " << result << note;
    EXPECT_EQ(status[abi::kStatusMatches], kChurnRounds) << label << ": " << result << note;
    expect_host_on_cce(
        device,
        cce_index,
        data_offset,
        pattern_words(abi::churn_word(value, kChurnRounds - 1, 0), kChurnWords),
        tag,
        label);
}

// Snoop-off preload with producer completion and a targeted INVALIDATE64 (gates), then snoop-on preload with no
// invalidate (gates: the snoop packet tag on a sibling CCE's iDMA writes, GRENSV-3774, GRENSV-1577). Hot-set refill and
// hit cycles are recorded.
TEST_P(MimirCacheStress, PreloadChurn) {
    const uint32_t cce_index = GetParam();
    ScopedDevice device_closer;
    IDevice* device = device_closer.device();
    check_tl1_layout();
    record_descriptor_properties(soc_desc_of(device));
    run_churn_leg(device_closer.mesh(), device, cce_index, false);
    if (HasFatalFailure()) {
        return;
    }
    run_churn_leg(device_closer.mesh(), device, cce_index, true);
}

// MM-8 MimirDmCachePerf's write paths past 32 KiB; kStatusCycles is recorded, not gated.
inline constexpr uint32_t kPerfSweepSizesKiB[] = {64, 128, 192};

TEST_P(MimirCacheStress, PerfSweep) {
    const uint32_t cce_index = GetParam();
    ScopedDevice device_closer;
    IDevice* device = device_closer.device();
    check_tl1_layout();
    record_descriptor_properties(soc_desc_of(device));

    const uint64_t data_offset = tl1_data(kProducerHart);
    for (const WritePath write_path :
         {WritePath::Uncached1B, WritePath::Uncached8B, WritePath::CachedRangeFlush, WritePath::CachedFastFlush}) {
        for (uint32_t size_index = 0; size_index < std::size(kPerfSweepSizesKiB); ++size_index) {
            const uint32_t num_words = kPerfSweepSizesKiB[size_index] * 1024 / sizeof(uint32_t);
            const uint32_t tag = 0xD0 + static_cast<uint32_t>(write_path) * 4 + size_index;
            const uint32_t value = 0x9F000000u | (tag << 12);
            write_cce_tl1(device, cce_index, data_offset + num_words * sizeof(uint32_t), make_guard(tag));
            std::vector<uint32_t> args = mm41_args(write_path_mode(write_path), kProducerHart);
            args[abi::kArgDataOffset] = static_cast<uint32_t>(data_offset);
            args[abi::kArgNumWords] = num_words;
            args[abi::kArgValue] = value;
            const std::string leg =
                fmt::format("{}_{}KiB", write_path_name({write_path, 0}), kPerfSweepSizesKiB[size_index]);
            const std::string label =
                fmt::format("PerfSweep {} on CCE{} hart {}, TL1 0x{:x}", leg, cce_index, kProducerHart, data_offset);
            log_info(tt::LogTest, "MM41 {}", label);
            const StagedRun run =
                run_together(device, {cache_kernel(cce_index, kProducerHart, args)}, kStressHostDeadline);
            ASSERT_FALSE(run.statuses.empty())
                << label << ": no status after " << kStressHostDeadline.count() << " s" << kFreshSessionNote;
            const std::vector<uint32_t>& status = run.statuses.front();
            log_info(tt::LogTest, "MM41 {}: {}", label, mm41_detail(status));
            RecordProperty(leg + "_cycles", static_cast<int>(status[abi::kStatusCycles]));
            ASSERT_EQ(status[abi::kStatusCode], abi::kCodeDone) << label << ": " << mm41_detail(status);
            EXPECT_EQ(status[abi::kStatusMatches], num_words) << label << ": " << mm41_detail(status);
            EXPECT_EQ(
                read_cce_tl1(device, cce_index, data_offset, abi::kLineBytes), pattern_words(value, kWordsPerLine))
                << label << ": first line";
            EXPECT_EQ(
                read_cce_tl1(
                    device, cce_index, data_offset + (num_words - kWordsPerLine) * sizeof(uint32_t), abi::kLineBytes),
                pattern_words(value + num_words - kWordsPerLine, kWordsPerLine))
                << label << ": last line";
            EXPECT_EQ(
                read_cce_tl1(device, cce_index, data_offset + num_words * sizeof(uint32_t), kGuardBytes),
                make_guard(tag))
                << label << ": guard overwritten";
        }
    }
}

INSTANTIATE_TEST_SUITE_P(MM41, MimirCacheStress, ::testing::ValuesIn(mimir_contract::cce_indices()), cce_param_name);

// MM4x compile-only row: every MM4x kernel JIT-compiles for every hart of CCE 0, with nothing launched.
inline constexpr const char* kMm4xKernels[] = {
    "tests/tt_metal/tt_metal/test_kernels/cce/cce_cache_ops.cpp",
    "tests/tt_metal/tt_metal/test_kernels/cce/cce_idma_copy.cpp",
    "tests/tt_metal/tt_metal/test_kernels/cce/cce_amo_ops.cpp",
    "tests/tt_metal/tt_metal/test_kernels/cce/cce_scoped_lock_producer.cpp",
    "tests/tt_metal/tt_metal/test_kernels/cce/cce_scoped_lock_consumer.cpp",
};

std::string mm4x_kernel_name(const ::testing::TestParamInfo<uint32_t>& info) {
    return std::filesystem::path(kMm4xKernels[info.param]).stem().string();
}

class MimirMm4xCompile : public ::testing::TestWithParam<uint32_t> {};

TEST_P(MimirMm4xCompile, Kernel) {
    skip_unless_programmable_dram_package();
    if (const std::string skip_reason = mimir_watcher_skip_reason(); !skip_reason.empty()) {
        GTEST_SKIP() << skip_reason;
    }
    ScopedDevice device_closer;
    IDevice* device = device_closer.device();
    std::vector<CceKernelLaunch> launches;
    for (uint32_t hart = 0; hart < cce_hart_count(); ++hart) {
        launches.push_back({.cce_index = 0, .kernel_path = kMm4xKernels[GetParam()], .hart = hart});
    }
    Program program = CreateProgram();
    add_cce_kernels(device, program, launches);
    EXPECT_NO_THROW(program.impl().compile(device, /*force_slow_dispatch=*/true)) << kMm4xKernels[GetParam()];
}

INSTANTIATE_TEST_SUITE_P(
    MM41, MimirMm4xCompile, ::testing::Range(0u, static_cast<uint32_t>(std::size(kMm4xKernels))), mm4x_kernel_name);

}  // namespace tt::tt_metal::mimir_test::dm_cache
