// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <gtest/gtest.h>

#include "device_fixture.hpp"
#include "mesh_dispatch_fixture.hpp"
#include "tt_metal/test_utils/env_vars.hpp"

namespace tt::tt_metal {

// LLK test fixtures that are dispatch-agnostic: they run under both slow dispatch
// (TT_METAL_SLOW_DISPATCH_MODE=1) and fast dispatch. The PR gate exercises them under
// fast dispatch while nightly continues to exercise them under slow dispatch.

class LLKMeshDeviceFixture : public MeshDeviceFixture {
protected:
    void SetUp() override {
        this->DetectDispatchMode();
        this->arch_ = tt::get_arch_from_string(tt::test_utils::get_umd_arch_name());

        // Limit to 2 devices for CI throughput; same rationale as MeshDeviceFixture.
        this->num_devices_ = tt::tt_metal::GetNumAvailableDevices();
        if (num_devices_ > 2) {
            this->num_devices_ = 2;
        }

        std::vector<ChipId> ids;
        for (ChipId id : tt::tt_metal::MetalContext::instance().get_cluster().all_chip_ids()) {
            ids.push_back(id);
        }
        this->create_devices(ids);
        init_max_cbs();
    }
};

// Same as LLKMeshDeviceFixture but skips the test when running under fast dispatch.
// Use this for tests that currently only produce correct results under slow dispatch
// (e.g. due to test-local async write/read sequencing bugs under FD). These still get
// coverage via the nightly slow-dispatch run.
class LLKMeshDeviceFixtureSlowDispatchOnly : public LLKMeshDeviceFixture {
protected:
    void SetUp() override {
        LLKMeshDeviceFixture::SetUp();
        if (::testing::Test::IsSkipped()) {
            return;
        }
        if (!this->IsSlowDispatch()) {
            GTEST_SKIP() << "Skipping: test requires slow dispatch (TT_METAL_SLOW_DISPATCH_MODE=1)";
        }
    }
};

class LLKMeshDeviceSingleCardFixture : public MeshDeviceSingleCardFixture {
protected:
    bool validate_dispatch_mode() override {
        this->DetectDispatchMode();
        return true;
    }
};

class LLKBlackholeSingleCardFixture : public LLKMeshDeviceSingleCardFixture {
protected:
    void SetUp() override {
        if (!this->validate_dispatch_mode()) {
            GTEST_SKIP();
        }
        this->arch_ = tt::get_arch_from_string(tt::test_utils::get_umd_arch_name());
        if (this->arch_ != tt::ARCH::BLACKHOLE) {
            GTEST_SKIP();
        }
        this->create_devices();
        init_max_cbs();
    }
};

class LLKQuasarMeshDeviceSingleCardFixture : public LLKMeshDeviceSingleCardFixture {
protected:
    void SetUp() override {
        if (!this->validate_dispatch_mode()) {
            GTEST_SKIP();
        }
        this->arch_ = tt::get_arch_from_string(tt::test_utils::get_umd_arch_name());
        if (this->arch_ != tt::ARCH::QUASAR) {
            GTEST_SKIP() << "Not a Quasar device";
        }
        this->create_devices();
        init_max_cbs();
    }
};

}  // namespace tt::tt_metal
