// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// RunTimeOptions::is_qsr_s1_simulator() recognises the qsr.s1 emulation model from the simulator
// directory name alone (TT_METAL_SIMULATOR). It gates the automatic grendel_qsr1 ATT map and the
// watcher's qsr.s1 exceptions, so the rule is pinned here: the last path component must start with
// emu-qsr-s1, a trailing separator does not matter, and the prefix elsewhere in the path does not count.

#include <gtest/gtest.h>

#include <cstdlib>
#include <optional>
#include <string>

#include "llrt/rtoptions.hpp"

namespace {

// Set (or clear) one environment variable for the test's life and put the old value back.
class ScopedSimulatorEnv {
public:
    ScopedSimulatorEnv(const char* name, const std::optional<std::string>& value) : name_(name) {
        if (const char* old = std::getenv(name)) {
            old_ = old;
        }
        apply(value);
    }
    ~ScopedSimulatorEnv() { apply(old_); }

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

bool is_qsr_s1_for(const std::optional<std::string>& simulator) {
    ScopedSimulatorEnv env("TT_METAL_SIMULATOR", simulator);
    tt::llrt::RunTimeOptions opts;
    return opts.is_qsr_s1_simulator();
}

}  // namespace

TEST(RunTimeOptionsQsrS1Simulator, CPU_TheQsrS1DirectoryNameIsRecognised) {
    EXPECT_TRUE(is_qsr_s1_for("/models/emu-qsr-s1-t6x1_DM"));
    EXPECT_TRUE(is_qsr_s1_for("/models/emu-qsr-s1-t6x1_DM/"));
    EXPECT_TRUE(is_qsr_s1_for("emu-qsr-s1"));
}

TEST(RunTimeOptionsQsrS1Simulator, CPU_OtherQuasarModelsAreNot) {
    EXPECT_FALSE(is_qsr_s1_for("/models/emu-quasar-2x3_DISPATCH"));
    EXPECT_FALSE(is_qsr_s1_for("/models/emu-quasar-1x3"));
}

TEST(RunTimeOptionsQsrS1Simulator, CPU_ThePrefixOnlyCountsInTheLastPathComponent) {
    EXPECT_FALSE(is_qsr_s1_for("/models/emu-qsr-s1-t6x1_DM/other-model"));
    EXPECT_FALSE(is_qsr_s1_for("/emu-qsr-s1/emu-quasar-2x3_DISPATCH"));
    EXPECT_FALSE(is_qsr_s1_for("/models/not-emu-qsr-s1"));
}

TEST(RunTimeOptionsQsrS1Simulator, CPU_NoSimulatorIsNot) { EXPECT_FALSE(is_qsr_s1_for(std::nullopt)); }
