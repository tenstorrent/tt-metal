// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <gtest/gtest.h>

#include <memory>
#include <utility>
#include <vector>

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

    void reset() {
        devices.clear();
        initialized = false;
    }
};

// Per-tag shared state storage. TearDownTestSuite() must reset it before the
// test binary reaches static destruction so MeshDevice handles are dropped while
// MetalContext is still alive.
template <class Tag>
inline LLKSharedDevices& shared_state_storage() {
    static LLKSharedDevices state;
    return state;
}

inline bool detect_slow_dispatch() { return getenv("TT_METAL_SLOW_DISPATCH_MODE") != nullptr; }

// Host architecture from UMD (works before SetUp() fills arch_ on the fixture).
inline tt::ARCH detect_arch() { return tt::get_arch_from_string(tt::test_utils::get_umd_arch_name()); }

inline void log_dispatch_mode(bool slow_dispatch) {
    if (slow_dispatch) {
        log_info(tt::LogTest, "Running test using Slow Dispatch");
    } else {
        log_info(tt::LogTest, "Running test using Fast Dispatch");
    }
}

inline void populate_shared_state(LLKSharedDevices& s, const std::vector<ChipId>& ids) {
    s.arch = detect_arch();
    const auto& dispatch_core_config = tt::tt_metal::MetalContext::instance().rtoptions().get_dispatch_core_config();
    auto id_to_device = distributed::MeshDevice::create_unit_meshes(
        ids, DEFAULT_L1_SMALL_SIZE, DEFAULT_TRACE_REGION_SIZE, 1, dispatch_core_config);
    s.devices.clear();
    s.devices.reserve(id_to_device.size());
    for (auto& [_, device] : id_to_device) {
        s.devices.push_back(std::move(device));
    }
    s.max_cbs = tt::tt_metal::MetalContext::instance().hal().get_arch_num_circular_buffers();
    s.initialized = true;
}

template <class Fixture>
void apply_shared_state(Fixture& f, const LLKSharedDevices& s) {
    const bool slow_dispatch = detect_slow_dispatch();
    log_dispatch_mode(slow_dispatch);
    f.slow_dispatch_ = slow_dispatch;
    f.arch_ = s.arch;
    f.devices_ = s.devices;
    f.max_cbs_ = s.max_cbs;
    f.num_devices_ = s.devices.size();
}

}  // namespace detail

class LLKMeshDeviceFixture : public MeshDeviceFixture {
protected:
    template <class F>
    friend void detail::apply_shared_state(F&, const detail::LLKSharedDevices&);

    static detail::LLKSharedDevices& shared_state() { return detail::shared_state_storage<LLKMeshDeviceFixture>(); }

    static void SetUpTestSuite() {
        auto& s = shared_state();
        if (s.initialized) {
            return;
        }

        // Limit to 2 chips for CI throughput; same rationale as MeshDeviceFixture.
        // Use MMIO (host) chips only — same id source as the single-card LLK fixture.
        size_t num_devices = tt::tt_metal::GetNumAvailableDevices();
        if (num_devices > 2) {
            num_devices = 2;
        }
        const auto& mmio = tt::tt_metal::MetalContext::instance().get_cluster().mmio_chip_ids();
        std::vector<ChipId> ids(mmio.begin(), mmio.end());
        if (ids.size() > num_devices) {
            ids.resize(num_devices);
        }
        detail::populate_shared_state(s, ids);
    }

    // Per-suite cleanup: drop the device handles between suites so the next
    // suite re-opens fresh ones (avoids cross-suite allocator/dispatch state
    // bleed) and so handles are gone before process shutdown.
    static void TearDownTestSuite() { shared_state().reset(); }

    void SetUp() override {
        auto& s = shared_state();
        if (!s.initialized) {
            // Defensive — should never happen because SetUpTestSuite runs first.
            SetUpTestSuite();
        }
        detail::apply_shared_state(*this, s);
    }

    void TearDown() override {
        // Devices are owned by the suite-shared state; just drop the per-test references.
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
        if (!detail::detect_slow_dispatch()) {
            GTEST_SKIP() << "Skipping: test requires slow dispatch (TT_METAL_SLOW_DISPATCH_MODE=1)";
        }
        LLKMeshDeviceFixture::SetUp();
    }
};

class LLKMeshDeviceSingleCardFixture : public MeshDeviceSingleCardFixture {
protected:
    template <class F>
    friend void detail::apply_shared_state(F&, const detail::LLKSharedDevices&);

    static detail::LLKSharedDevices& shared_state() {
        return detail::shared_state_storage<LLKMeshDeviceSingleCardFixture>();
    }

    static void SetUpTestSuite() {
        auto& s = shared_state();
        if (s.initialized) {
            return;
        }
        const auto& mmio = tt::tt_metal::MetalContext::instance().get_cluster().mmio_chip_ids();
        std::vector<ChipId> ids(mmio.begin(), mmio.end());
        detail::populate_shared_state(s, ids);
    }

    // Per-suite cleanup; keep the static shared state empty at process shutdown.
    static void TearDownTestSuite() { shared_state().reset(); }

    void SetUp() override {
        auto& s = shared_state();
        if (!s.initialized) {
            // Defensive — should never happen because SetUpTestSuite runs first.
            SetUpTestSuite();
        }
        detail::apply_shared_state(*this, s);
    }

    void TearDown() override {
        // Devices are owned by the suite-shared state; just drop the per-test references.
        this->devices_.clear();
    }
};

class LLKBlackholeSingleCardFixture : public LLKMeshDeviceSingleCardFixture {
protected:
    // Skip suite-level device init on non-Blackhole hosts; every test will be
    // skipped in SetUp() anyway, so there's no point opening devices.
    static void SetUpTestSuite() {
        const auto arch = detail::detect_arch();
        if (arch != tt::ARCH::BLACKHOLE) {
            return;
        }
        LLKMeshDeviceSingleCardFixture::SetUpTestSuite();
    }

    void SetUp() override {
        if (!shared_state().initialized) {
            GTEST_SKIP() << "This test can only be run on Blackhole cards";
        }
        LLKMeshDeviceSingleCardFixture::SetUp();
    }
};

class LLKQuasarMeshDeviceSingleCardFixture : public LLKMeshDeviceSingleCardFixture {
protected:
    static void SetUpTestSuite() {
        const auto arch = detail::detect_arch();
        if (arch != tt::ARCH::QUASAR) {
            return;
        }
        LLKMeshDeviceSingleCardFixture::SetUpTestSuite();
    }

    void SetUp() override {
        if (!shared_state().initialized) {
            GTEST_SKIP() << "Not a Quasar device";
        }
        LLKMeshDeviceSingleCardFixture::SetUp();
    }
};

}  // namespace tt::tt_metal
