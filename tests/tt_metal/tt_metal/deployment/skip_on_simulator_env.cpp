// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Deployment qualification assumes silicon: it sweeps full DRAM ranges, expects fully cabled Ethernet, and its host
// monitors sleep between samples and treat an unchanged device counter as a hang. A simulator only advances when the
// host clocks it, so those monitors abort before the device can make progress. Skip the whole binary there.

#include <gtest/gtest.h>

#include "impl/context/metal_context.hpp"

namespace {

class SkipOnSimulator : public ::testing::Environment {
public:
    void SetUp() override {
        if (tt::tt_metal::MetalContext::instance().rtoptions().get_simulator_enabled()) {
            GTEST_SKIP() << "Deployment qualification tests require silicon";
        }
    }
};

const ::testing::Environment* const kSkipOnSimulator = ::testing::AddGlobalTestEnvironment(new SkipOnSimulator);

}  // namespace
