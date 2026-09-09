// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// TT_METAL_FW_SRC_BRISC names a BRISC firmware source to JIT-build instead of the in-tree one.
// Only the JIT build can use it, so a non-empty value also disables the precompiled firmware: the
// two switches cannot disagree, and a caller does not have to remember a second variable.

#include <gtest/gtest.h>

#include <cstdlib>
#include <optional>
#include <string>

#include "llrt/rtoptions.hpp"

namespace {

// Set (or clear) one environment variable for the test's life and put the old value back.
class ScopedEnv {
public:
    ScopedEnv(const char* name, std::optional<std::string> value) : name_(name) {
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

}  // namespace

TEST(RunTimeOptionsFirmwareSource, BriscSourceOverrideImpliesThePrecompiledFirmwareBypass) {
    ScopedEnv src("TT_METAL_FW_SRC_BRISC", "/somewhere/else/brisc.cc");
    ScopedEnv bypass("TT_METAL_DISABLE_PRECOMPILED_FW", std::nullopt);
    tt::llrt::RunTimeOptions opts;
    EXPECT_EQ(opts.get_fw_src_brisc(), "/somewhere/else/brisc.cc");
    EXPECT_TRUE(opts.get_disable_precompiled_fw()) << "a source override without the bypass would run stock firmware";
}

TEST(RunTimeOptionsFirmwareSource, NoOverrideLeavesTheInTreeSourceAndPrecompiledFirmware) {
    ScopedEnv src("TT_METAL_FW_SRC_BRISC", std::nullopt);
    ScopedEnv bypass("TT_METAL_DISABLE_PRECOMPILED_FW", std::nullopt);
    tt::llrt::RunTimeOptions opts;
    EXPECT_TRUE(opts.get_fw_src_brisc().empty());
    EXPECT_FALSE(opts.get_disable_precompiled_fw());
}

TEST(RunTimeOptionsFirmwareSource, AnEmptyOverrideIsNoOverride) {
    ScopedEnv src("TT_METAL_FW_SRC_BRISC", "");
    ScopedEnv bypass("TT_METAL_DISABLE_PRECOMPILED_FW", std::nullopt);
    tt::llrt::RunTimeOptions opts;
    EXPECT_TRUE(opts.get_fw_src_brisc().empty());
    EXPECT_FALSE(opts.get_disable_precompiled_fw());
}
