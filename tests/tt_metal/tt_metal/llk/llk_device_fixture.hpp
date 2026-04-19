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
//
// Performance: the fixtures share their MeshDevice across all tests in a suite via
// SetUpTestSuite/TearDownTestSuite. Per-test SetUp() does not re-open the device;
// it just exposes the suite-shared handles to the test instance. This avoids ~1s
// of TopologyDiscovery + MeshDevice::create_unit_meshes per test that the original
// per-test SetUp() pattern paid.

namespace detail {

// Per-suite shared MeshDevice state. Each LLK fixture class instantiates its own
// instance of this struct (so different suites can hold different device configs)
// but, within a suite, all tests reuse the same MeshDevice handles.
struct LLKSharedDevices {
    std::vector<std::shared_ptr<distributed::MeshDevice>> devices;
    tt::ARCH arch{tt::ARCH::Invalid};
    bool slow_dispatch{};
    uint32_t max_cbs{};
    bool initialized{false};
};

inline bool detect_slow_dispatch() { return getenv("TT_METAL_SLOW_DISPATCH_MODE") != nullptr; }

inline void log_dispatch_mode(bool slow_dispatch) {
    if (slow_dispatch) {
        log_info(tt::LogTest, "Running test using Slow Dispatch");
    } else {
        log_info(tt::LogTest, "Running test using Fast Dispatch");
    }
}

}  // namespace detail

class LLKMeshDeviceFixture : public MeshDeviceFixture {
protected:
    static detail::LLKSharedDevices& shared_state() {
        static detail::LLKSharedDevices s;
        return s;
    }

    static void SetUpTestSuite() {
        auto& s = shared_state();
        if (s.initialized) {
            return;
        }
        s.slow_dispatch = detail::detect_slow_dispatch();
        s.arch = tt::get_arch_from_string(tt::test_utils::get_umd_arch_name());

        // Limit to 2 chips for CI throughput; same rationale as MeshDeviceFixture.
        size_t num_devices = tt::tt_metal::GetNumAvailableDevices();
        if (num_devices > 2) {
            num_devices = 2;
        }
        std::vector<ChipId> ids;
        for (ChipId id : tt::tt_metal::MetalContext::instance().get_cluster().all_chip_ids()) {
            ids.push_back(id);
            if (ids.size() >= num_devices) {
                break;
            }
        }
        const auto& dispatch_core_config =
            tt::tt_metal::MetalContext::instance().rtoptions().get_dispatch_core_config();
        auto id_to_device = distributed::MeshDevice::create_unit_meshes(
            ids, DEFAULT_L1_SMALL_SIZE, DEFAULT_TRACE_REGION_SIZE, 1, dispatch_core_config);
        s.devices.clear();
        for (const auto& [device_id, device] : id_to_device) {
            s.devices.push_back(device);
        }
        s.max_cbs = tt::tt_metal::MetalContext::instance().hal().get_arch_num_circular_buffers();
        s.initialized = true;
    }

    static void TearDownTestSuite() {
        // Match the baseline (MeshDeviceFixture::TearDown) which only resets shared_ptrs;
        // no explicit close() — the MeshDevice destructor handles teardown.
        auto& s = shared_state();
        s.devices.clear();
        s.initialized = false;
    }

    void SetUp() override {
        auto& s = shared_state();
        if (!s.initialized) {
            // Defensive — should never happen because SetUpTestSuite runs first.
            SetUpTestSuite();
        }
        detail::log_dispatch_mode(s.slow_dispatch);
        this->slow_dispatch_ = s.slow_dispatch;
        this->arch_ = s.arch;
        this->devices_ = s.devices;
        this->max_cbs_ = s.max_cbs;
        this->num_devices_ = s.devices.size();
    }

    void TearDown() override {
        // Devices are owned by the suite-shared static; just drop the per-test references.
        this->devices_.clear();
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
    static detail::LLKSharedDevices& shared_state() {
        static detail::LLKSharedDevices s;
        return s;
    }

    static void SetUpTestSuite() {
        auto& s = shared_state();
        if (s.initialized) {
            return;
        }
        s.slow_dispatch = detail::detect_slow_dispatch();
        s.arch = tt::get_arch_from_string(tt::test_utils::get_umd_arch_name());

        std::vector<ChipId> ids;
        for (ChipId id : tt::tt_metal::MetalContext::instance().get_cluster().mmio_chip_ids()) {
            ids.push_back(id);
        }
        const auto& dispatch_core_config =
            tt::tt_metal::MetalContext::instance().rtoptions().get_dispatch_core_config();
        auto id_to_device = distributed::MeshDevice::create_unit_meshes(
            ids, DEFAULT_L1_SMALL_SIZE, DEFAULT_TRACE_REGION_SIZE, 1, dispatch_core_config);
        s.devices.clear();
        for (const auto& [device_id, device] : id_to_device) {
            s.devices.push_back(device);
        }
        s.max_cbs = tt::tt_metal::MetalContext::instance().hal().get_arch_num_circular_buffers();
        s.initialized = true;
    }

    static void TearDownTestSuite() {
        auto& s = shared_state();
        s.devices.clear();
        s.initialized = false;
    }

    bool validate_dispatch_mode() override {
        this->DetectDispatchMode();
        return true;
    }

    void SetUp() override {
        auto& s = shared_state();
        if (!s.initialized) {
            SetUpTestSuite();
        }
        detail::log_dispatch_mode(s.slow_dispatch);
        this->slow_dispatch_ = s.slow_dispatch;
        this->arch_ = s.arch;
        this->devices_ = s.devices;
        this->max_cbs_ = s.max_cbs;
        this->num_devices_ = s.devices.size();
    }

    void TearDown() override {
        // Devices are owned by the suite-shared static; just drop the per-test references.
        this->devices_.clear();
    }
};

class LLKBlackholeSingleCardFixture : public LLKMeshDeviceSingleCardFixture {
protected:
    void SetUp() override {
        LLKMeshDeviceSingleCardFixture::SetUp();
        if (::testing::Test::IsSkipped()) {
            return;
        }
        if (this->arch_ != tt::ARCH::BLACKHOLE) {
            GTEST_SKIP();
        }
    }
};

class LLKQuasarMeshDeviceSingleCardFixture : public LLKMeshDeviceSingleCardFixture {
protected:
    void SetUp() override {
        LLKMeshDeviceSingleCardFixture::SetUp();
        if (::testing::Test::IsSkipped()) {
            return;
        }
        if (this->arch_ != tt::ARCH::QUASAR) {
            GTEST_SKIP() << "Not a Quasar device";
        }
    }
};

}  // namespace tt::tt_metal
