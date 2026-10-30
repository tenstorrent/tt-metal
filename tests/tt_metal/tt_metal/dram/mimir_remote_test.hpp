// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Shared by the Mimir tests that reach a remote Mimir over D2D (test_mimir_tl1.cpp and test_mimir_remote_dram.cpp).

#pragma once

#include <gtest/gtest.h>

#include <optional>

#include "tt_metal/tt_metal/dram/mimir_access.hpp"
#include "tt_metal/tt_metal/dram/mimir_hw_state.hpp"

namespace tt::tt_metal::mimir_test {

// SetUp runs the package skips every remote test starts with and opens no device. A test body then makes its own skips,
// runs MIMIR_REQUIRE_REMOTE_PREFLIGHT() and calls open_remote_device(), which opens the device once and closes it when
// the test ends.
class MimirRemoteTest : public ::testing::Test {
protected:
    void SetUp() override {
        skip_unless_programmable_dram_package();
        skip_unless_mimir_local_package();
    }

    IDevice* open_remote_device() {
        device_guard_.emplace();
        EXPECT_TRUE(mimir_remote_remap_result().empty()) << mimir_remote_remap_result();
        return device_guard_->device();
    }

    distributed::MeshDevice& mesh() { return device_guard_->mesh(); }

private:
    std::optional<ScopedDevice> device_guard_;
};

template <typename ParamType>
class MimirRemoteParamTest : public MimirRemoteTest, public ::testing::WithParamInterface<ParamType> {};

// One suite across test_mimir_tl1.cpp and test_mimir_remote_dram.cpp, so both files use this one fixture class.
class MimirRemoteMixed : public MimirRemoteTest {};

}  // namespace tt::tt_metal::mimir_test
