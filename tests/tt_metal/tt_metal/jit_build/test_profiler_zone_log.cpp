// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <string>

#include "jit_build/jit_build_utils.hpp"

namespace tt::jit_build::utils {
namespace {

class ProfilerZoneLogTest : public ::testing::Test {
protected:
    std::filesystem::path dir;
    void SetUp() override {
        auto path = (std::filesystem::temp_directory_path() / "tt-profiler-zone-log-XXXXXX").string();
        const auto* created = mkdtemp(path.data());
        ASSERT_NE(created, nullptr);
        dir = created;
    }
    void TearDown() override { std::filesystem::remove_all(dir); }
};

TEST_F(ProfilerZoneLogTest, KeepsOnlyProfilerDiagnosticsAndPreservesTheirSourceLocations) {
    const std::string first = "kernel.cpp:12: note: '#pragma message: FIRST,kernel.cpp,12,KERNEL_PROFILER'\n";
    const std::string second = "kernel.cpp:25: note: '#pragma message: SECOND,kernel.cpp,25,KERNEL_PROFILER'\n";
    const auto path = dir / "kernel.o.log";
    {
        std::ofstream log(path);
        log << "unrelated compiler warning\n" << first << "   12 | DeviceZoneScopedN(\"FIRST\");\n" << second;
    }
    EXPECT_EQ(read_profiler_zone_log(path), first + second);
    // A server cache hit must return the same metadata without invoking the compiler again.
    EXPECT_EQ(read_profiler_zone_log(path), first + second);
}

TEST_F(ProfilerZoneLogTest, UninstrumentedOrMissingLogsHaveNoMetadata) {
    const auto path = dir / "kernel.o.log";
    EXPECT_TRUE(read_profiler_zone_log(path).empty());
    std::ofstream(path) << "ordinary diagnostic\n";
    EXPECT_TRUE(read_profiler_zone_log(path).empty());
}

TEST_F(ProfilerZoneLogTest, NormalizesMissingFinalNewlineBeforeConcatenatingTargets) {
    const std::string diagnostic = "kernel.cpp:7: note: '#pragma message: ZONE,kernel.cpp,7,KERNEL_PROFILER'";
    const auto path = dir / "kernel.o.log";
    std::ofstream(path) << diagnostic;
    EXPECT_EQ(read_profiler_zone_log(path), diagnostic + '\n');
}

}  // namespace
}  // namespace tt::jit_build::utils
