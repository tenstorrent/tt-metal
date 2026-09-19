// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// TT_METAL_FW_SRC_BRISC selects a BRISC firmware extension. Only the JIT build can use it, so a
// non-empty value also disables the precompiled firmware: the two switches cannot disagree, and a caller does not have
// to remember a second variable.

#include <gtest/gtest.h>

#include <algorithm>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <optional>
#include <string>
#include <vector>

#include "llrt/rtoptions.hpp"
#include "llrt/hal.hpp"

namespace {

// Set (or clear) one environment variable for the test's life and put the old value back.
class ScopedEnv {
public:
    ScopedEnv(const char* name, const std::optional<std::string>& value) : name_(name) {
        if (const char* old = std::getenv(name)) {
            old_ = old;
        }
        apply(value);
    }
    ~ScopedEnv() { apply(old_); }

private:
    void apply(const std::optional<std::string>& value) {
        if (value.has_value()) {
            setenv(name_, value->c_str(), 1);
        } else {
            unsetenv(name_);
        }
    }
    const char* name_;
    std::optional<std::string> old_;
};

class RunTimeOptionsFirmwareSource : public ::testing::Test {
protected:
    void SetUp() override {
        auto pattern = (std::filesystem::temp_directory_path() / "brisc_fw_header_XXXXXX").string();
        const auto* directory = mkdtemp(pattern.data());
        ASSERT_NE(directory, nullptr);
        directory_ = directory;
        std::ofstream header(test_header_path());
        header << "#pragma once\n";
        header.close();
        ASSERT_FALSE(header.fail());
    }

    void TearDown() override {
        if (!directory_.empty()) {
            std::error_code error;
            std::filesystem::remove_all(directory_, error);
        }
    }

    std::string test_header_path() const { return (directory_ / "runtime_reload.h").string(); }

private:
    std::filesystem::path directory_;
};

}  // namespace

TEST_F(RunTimeOptionsFirmwareSource, CPU_BlazeVariantImpliesThePrecompiledFirmwareBypass) {
    ScopedEnv src("TT_METAL_FW_SRC_BRISC", "blaze");
    ScopedEnv header("TT_METAL_FW_HEADER_BRISC", test_header_path());
    ScopedEnv bypass("TT_METAL_DISABLE_PRECOMPILED_FW", std::nullopt);
    tt::llrt::RunTimeOptions opts;
    EXPECT_EQ(opts.get_brisc_firmware_variant(), tt::llrt::BriscFirmwareVariant::Blaze);
    EXPECT_TRUE(opts.get_disable_precompiled_fw()) << "a source override without the bypass would run stock firmware";
}

TEST_F(RunTimeOptionsFirmwareSource, CPU_NoOverrideLeavesTheStockFirmwareAndPrecompiledFirmware) {
    ScopedEnv src("TT_METAL_FW_SRC_BRISC", std::nullopt);
    ScopedEnv header("TT_METAL_FW_HEADER_BRISC", std::nullopt);
    ScopedEnv bypass("TT_METAL_DISABLE_PRECOMPILED_FW", std::nullopt);
    tt::llrt::RunTimeOptions opts;
    EXPECT_EQ(opts.get_brisc_firmware_variant(), tt::llrt::BriscFirmwareVariant::Default);
    EXPECT_FALSE(opts.get_disable_precompiled_fw());
}

TEST_F(RunTimeOptionsFirmwareSource, CPU_AnEmptyOverrideIsNoOverride) {
    ScopedEnv src("TT_METAL_FW_SRC_BRISC", "");
    ScopedEnv header("TT_METAL_FW_HEADER_BRISC", std::nullopt);
    ScopedEnv bypass("TT_METAL_DISABLE_PRECOMPILED_FW", std::nullopt);
    tt::llrt::RunTimeOptions opts;
    EXPECT_EQ(opts.get_brisc_firmware_variant(), tt::llrt::BriscFirmwareVariant::Default);
    EXPECT_FALSE(opts.get_disable_precompiled_fw());
}

TEST_F(RunTimeOptionsFirmwareSource, CPU_BlazeVariantRequiresAHeader) {
    ScopedEnv src("TT_METAL_FW_SRC_BRISC", "blaze");
    ScopedEnv header("TT_METAL_FW_HEADER_BRISC", std::nullopt);
    EXPECT_ANY_THROW(tt::llrt::RunTimeOptions{});
}

TEST_F(RunTimeOptionsFirmwareSource, CPU_HeaderRequiresTheBlazeVariant) {
    ScopedEnv src("TT_METAL_FW_SRC_BRISC", std::nullopt);
    ScopedEnv header("TT_METAL_FW_HEADER_BRISC", test_header_path());
    EXPECT_ANY_THROW(tt::llrt::RunTimeOptions{});
}

TEST_F(RunTimeOptionsFirmwareSource, CPU_MissingHeaderIsRejected) {
    ScopedEnv src("TT_METAL_FW_SRC_BRISC", "blaze");
    ScopedEnv header("TT_METAL_FW_HEADER_BRISC", "/missing/runtime_reload.h");
    EXPECT_ANY_THROW(tt::llrt::RunTimeOptions{});
}

TEST_F(RunTimeOptionsFirmwareSource, CPU_ArbitrarySourcePathIsRejected) {
    ScopedEnv src("TT_METAL_FW_SRC_BRISC", "/somewhere/else/brisc.cc");
    ScopedEnv header("TT_METAL_FW_HEADER_BRISC", std::nullopt);
    EXPECT_ANY_THROW(tt::llrt::RunTimeOptions{});
}

TEST_F(RunTimeOptionsFirmwareSource, CPU_UnsupportedTtLangVariantIsRejected) {
    ScopedEnv src("TT_METAL_FW_SRC_BRISC", "ttlang");
    ScopedEnv header("TT_METAL_FW_HEADER_BRISC", std::nullopt);
    EXPECT_ANY_THROW(tt::llrt::RunTimeOptions{});
}

TEST_F(RunTimeOptionsFirmwareSource, CPU_BlazeVariantHasASeparateCompileHash) {
    ScopedEnv src("TT_METAL_FW_SRC_BRISC", std::nullopt);
    ScopedEnv header("TT_METAL_FW_HEADER_BRISC", std::nullopt);
    tt::llrt::RunTimeOptions stock;
    ScopedEnv blaze_src("TT_METAL_FW_SRC_BRISC", "blaze");
    ScopedEnv blaze_header("TT_METAL_FW_HEADER_BRISC", test_header_path());
    tt::llrt::RunTimeOptions blaze;
    EXPECT_NE(stock.get_compile_hash_string(), blaze.get_compile_hash_string());
}

TEST_F(RunTimeOptionsFirmwareSource, CPU_BlazeDefineOnlyAppliesToBriscFirmware) {
    using namespace tt::tt_metal;
    ScopedEnv src("TT_METAL_FW_SRC_BRISC", std::nullopt);
    ScopedEnv header("TT_METAL_FW_HEADER_BRISC", std::nullopt);
    tt::llrt::RunTimeOptions stock;
    ScopedEnv blaze_src("TT_METAL_FW_SRC_BRISC", "blaze");
    ScopedEnv blaze_header("TT_METAL_FW_HEADER_BRISC", test_header_path());
    tt::llrt::RunTimeOptions blaze;
    for (const auto arch : {tt::ARCH::WORMHOLE_B0, tt::ARCH::BLACKHOLE}) {
        const Hal hal(arch, false, false, 0, false);
        const auto& query = hal.get_jit_build_query();
        for (const auto* opts : {&stock, &blaze}) {
            for (bool is_fw : {false, true}) {
                for (uint32_t processor = 0; processor < 2; ++processor) {
                    const HalJitBuildQueryInterface::Params params{
                        is_fw, HalProgrammableCoreType::TENSIX, HalProcessorClassType::DM, processor, *opts};
                    const auto defines = query.defines(params);
                    const bool enabled =
                        std::find(defines.begin(), defines.end(), "BLAZE_RUNTIME_RELOAD") != defines.end();
                    EXPECT_EQ(enabled, opts == &blaze && is_fw && processor == 0);
                    if (is_fw && processor == 0) {
                        EXPECT_EQ(
                            query.srcs(params), std::vector<std::string>{"tt_metal/hw/firmware/src/tt-1xx/brisc.cc"});
                    }
                }
            }
        }
    }
}
