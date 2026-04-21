// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <gtest/gtest.h>

#include <memory>

#include "device_fixture.hpp"
#include "mesh_dispatch_fixture.hpp"
#include "tt_metal/test_utils/env_vars.hpp"

namespace tt::tt_metal {

// LLK test fixtures that are dispatch-agnostic: they run under both slow dispatch
// (TT_METAL_SLOW_DISPATCH_MODE=1) and fast dispatch. The merge gate exercises them
// under fast dispatch while nightly continues to exercise them under slow dispatch.
//
// Performance: the fixtures share their MeshDevice across all tests in a suite via
// SetUpTestSuite. Per-test SetUp() does not re-open the device; it just exposes
// the suite-shared handles to the test instance. This avoids ~1s of
// TopologyDiscovery + MeshDevice::create_unit_meshes per test that the original
// per-test SetUp() pattern paid.

namespace detail {

// Per-fixture-chain shared MeshDevice state. Two distinct instances exist across
// the file — one for LLKMeshDeviceFixture (all chips), one for
// LLKMeshDeviceSingleCardFixture (MMIO chips only). Derived variants
// (LLKMeshDeviceFixtureSlowDispatchOnly, LLKBlackholeSingleCardFixture,
// LLKQuasarMeshDeviceSingleCardFixture) inherit shared_state() without
// overriding it, so they reuse their base class's handles.
struct LLKSharedDevices {
    std::vector<std::shared_ptr<distributed::MeshDevice>> devices;
    tt::ARCH arch{tt::ARCH::Invalid};
    uint32_t max_cbs{};
    bool initialized{false};
};

// gtest Environment that owns the device handles for one fixture chain.
//
// Why an Environment: the device handles are shared_ptr<MeshDevice> kept alive
// for the whole life of the test binary. If they were dropped during static
// destruction at process exit, MetalContext (also a function-local static)
// could be destroyed first and the MeshDevice dtor would touch a dead context.
// gtest guarantees Environment::TearDown() runs *before* RUN_ALL_TESTS()
// returns, while MetalContext is still alive — so dropping the handles here is
// always safe.
//
// (See https://google.github.io/googletest/advanced.html#global-set-up-and-tear-down.)
class LLKDeviceEnvironment : public ::testing::Environment {
public:
    LLKSharedDevices state;

    void TearDown() override {
        state.devices.clear();
        state.initialized = false;
    }
};

// One Environment instance per Tag — ::testing::AddGlobalTestEnvironment takes
// ownership of the heap allocation. Lazy: registered on first call (from the
// fixture's SetUpTestSuite), which happens before RUN_ALL_TESTS reaches
// Environment::TearDown.
template <class Tag>
inline LLKDeviceEnvironment& acquire_environment() {
    static LLKDeviceEnvironment* env = [] {
        auto owned = std::make_unique<LLKDeviceEnvironment>();
        ::testing::AddGlobalTestEnvironment(owned.get());
        return owned.release();  // gtest now owns this pointer
    }();
    return *env;
}

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
        return detail::acquire_environment<LLKMeshDeviceFixture>().state;
    }

    static void SetUpTestSuite() {
        auto& s = shared_state();
        if (s.initialized) {
            return;
        }
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

    // Per-suite cleanup: drop the device handles between suites so the next
    // suite re-opens fresh ones (avoids cross-suite allocator/dispatch state
    // bleed). The registered LLKDeviceEnvironment::TearDown is the final
    // safety net — it runs at end of RUN_ALL_TESTS in any path that reaches
    // gtest's normal completion (including failures and skips), guaranteeing
    // shutdown order vs MetalContext.
    static void TearDownTestSuite() {
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
        const bool slow_dispatch = detail::detect_slow_dispatch();
        detail::log_dispatch_mode(slow_dispatch);
        this->slow_dispatch_ = slow_dispatch;
        this->arch_ = s.arch;
        this->devices_ = s.devices;
        this->max_cbs_ = s.max_cbs;
        this->num_devices_ = s.devices.size();
    }

    void TearDown() override {
        // Devices are owned by the suite-shared Environment / static state;
        // just drop the per-test references.
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
        return detail::acquire_environment<LLKMeshDeviceSingleCardFixture>().state;
    }

    static void SetUpTestSuite() {
        auto& s = shared_state();
        if (s.initialized) {
            return;
        }
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

    // Per-suite cleanup; LLKDeviceEnvironment::TearDown is the final safety net.
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
        const bool slow_dispatch = detail::detect_slow_dispatch();
        detail::log_dispatch_mode(slow_dispatch);
        this->slow_dispatch_ = slow_dispatch;
        this->arch_ = s.arch;
        this->devices_ = s.devices;
        this->max_cbs_ = s.max_cbs;
        this->num_devices_ = s.devices.size();
    }

    void TearDown() override {
        // Devices are owned by the suite-shared Environment; just drop the per-test references.
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
