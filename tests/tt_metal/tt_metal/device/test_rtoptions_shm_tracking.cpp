// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// SHM allocation tracking is off unless TT_METAL_SHM_TRACKING_DISABLED=0.

#include <gtest/gtest.h>

#include <cstdlib>
#include <optional>
#include <string>

#include "llrt/rtoptions.hpp"

namespace {

// Set (or clear) TT_METAL_SHM_TRACKING_DISABLED for the test's life and put the old value back.
class ScopedShmTrackingEnv {
public:
    explicit ScopedShmTrackingEnv(const std::optional<std::string>& value) {
        if (const char* old = std::getenv(name_)) {
            old_ = old;
        }
        apply(value);
    }
    ~ScopedShmTrackingEnv() { apply(old_); }

private:
    void apply(const std::optional<std::string>& value) {
        if (value.has_value()) {
            setenv(name_, value->c_str(), 1);
        } else {
            unsetenv(name_);
        }
    }
    static constexpr const char* name_ = "TT_METAL_SHM_TRACKING_DISABLED";
    std::optional<std::string> old_;
};

bool is_tracking_enabled(const std::optional<std::string>& disabled) {
    ScopedShmTrackingEnv disabled_env(disabled);
    tt::llrt::RunTimeOptions opts;
    return !opts.get_shm_tracking_disabled();
}

}  // namespace

TEST(RunTimeOptionsShmTracking, CPU_OffByDefault) { EXPECT_FALSE(is_tracking_enabled(std::nullopt)); }

TEST(RunTimeOptionsShmTracking, CPU_OneKeepsItOff) { EXPECT_FALSE(is_tracking_enabled("1")); }

TEST(RunTimeOptionsShmTracking, CPU_ZeroTurnsItOn) { EXPECT_TRUE(is_tracking_enabled("0")); }

TEST(RunTimeOptionsShmTracking, CPU_OtherValuesKeepItOff) { EXPECT_FALSE(is_tracking_enabled("true")); }
