// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <gtest/gtest.h>

#include "device_fixture.hpp"

namespace tt::tt_metal::experimental::test_helpers {

// Real-hardware fixture for the Metal 2.0 Host API integration tests.
// Targets Wormhole B0 and Blackhole; requires TT_METAL_SLOW_DISPATCH_MODE=1.
class ProgramSpecHWTest : public tt::tt_metal::MeshDeviceFixture {
protected:
    void SetUp() override {
        MeshDeviceFixture::SetUp();
        if (this->IsSkipped()) {
            return;
        }
        // These tests target Gen1 (WH/BH) only
        if (devices_.at(0)->arch() != tt::ARCH::WORMHOLE_B0 && devices_.at(0)->arch() != tt::ARCH::BLACKHOLE) {
            GTEST_SKIP() << "Skipping: test requires Wormhole B0 or Blackhole hardware";
        }
    }
};

}  // namespace tt::tt_metal::experimental::test_helpers
