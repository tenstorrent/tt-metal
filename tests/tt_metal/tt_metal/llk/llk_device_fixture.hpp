// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <gtest/gtest.h>

#include <cstdlib>
#include <mutex>
#include <unordered_set>

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

// Shared MeshDevice state. Two distinct statics exist across the whole file —
// one owned by LLKMeshDeviceFixture::shared_state(), the other by
// LLKMeshDeviceSingleCardFixture::shared_state() — because those two base
// fixtures open different device pools (all chips vs MMIO-only). Derived
// variants (LLKMeshDeviceFixtureSlowDispatchOnly, LLKBlackholeSingleCardFixture,
// LLKQuasarMeshDeviceSingleCardFixture) inherit shared_state() without
// overriding it, so they reuse their base class's handles. Per-test SetUp()
// just exposes those handles to the test instance.
struct LLKSharedDevices {
    std::vector<std::shared_ptr<distributed::MeshDevice>> devices;
    tt::ARCH arch{tt::ARCH::Invalid};
    bool slow_dispatch{};
    uint32_t max_cbs{};
    bool initialized{false};
    bool cleanup_registered{false};
};

// Shutdown-ordering safety net for LLKSharedDevices.
//
// The two LLKSharedDevices statics (one per base fixture class) hold
// shared_ptr<MeshDevice>. MetalContext is also a function-local static (in
// metal). C++ destroys function-local statics in reverse order of *first
// construction* — and because SetUpTestSuite calls shared_state() before it
// touches MetalContext::instance(), the LLKSharedDevices used by that suite is
// normally constructed first, which means MetalContext is destroyed first at
// program exit. The LLKSharedDevices destructor then runs MeshDevice
// destructors that reach into an already-destroyed MetalContext → crash on
// shutdown.
//
// TearDownTestSuite() drops the shared_ptrs in the happy path, but it doesn't
// run when a test calls std::exit, when teardown itself is skipped, or in any
// other path where gtest's per-suite teardown is bypassed.
//
// The fix: register a process-wide std::atexit hook that drops the
// shared_ptrs from every LLKSharedDevices registered with CleanupRegistry.
// Per [basic.start.term], atexit handlers and static destructors are
// interleaved in reverse order of registration / completion-of-construction.
// By forcing MetalContext::instance() and CleanupRegistry::instance() to be
// constructed *before* we register the atexit hook, the hook is guaranteed to
// run *before* either of them is destroyed — so dropping the device
// shared_ptrs from the hook is safe.
//
// Residual risk: this fixes graceful exit and std::exit(). It does NOT fire on
// std::abort, _exit(), fatal signals, or std::terminate from an uncaught
// exception, all of which bypass atexit. Tests that abort the process can
// still produce a confusing shutdown crash on top of their real failure.
class CleanupRegistry {
public:
    static CleanupRegistry& instance() {
        static CleanupRegistry inst;
        return inst;
    }
    void add(LLKSharedDevices* s) {
        std::lock_guard<std::mutex> lk(mu_);
        states_.insert(s);
    }
    void clear_all() {
        std::lock_guard<std::mutex> lk(mu_);
        for (auto* s : states_) {
            s->devices.clear();
        }
    }

private:
    std::mutex mu_;
    std::unordered_set<LLKSharedDevices*> states_;
};

// Idempotent: safe to call multiple times per shared_state. Must be called from
// SetUpTestSuite (before any device creation) so the atexit hook is in place
// even if SetUpTestSuite later throws or the suite teardown is skipped.
inline void ensure_safe_cleanup_for(LLKSharedDevices& s) {
    if (s.cleanup_registered) {
        return;
    }
    // (1) Force MetalContext's function-local static to be constructed *before*
    //     we register atexit, so the hook is guaranteed to run before its dtor.
    (void)tt::tt_metal::MetalContext::instance();
    // (2) Force CleanupRegistry's static to be constructed *before* we register
    //     atexit, so the hook can safely call into it during shutdown.
    auto& reg = CleanupRegistry::instance();
    // (3) Register one process-wide atexit hook, even if multiple fixtures call.
    static std::once_flag flag;
    std::call_once(flag, []() { std::atexit([]() { CleanupRegistry::instance().clear_all(); }); });
    // (4) Register this shared_state with the registry so the hook clears it.
    reg.add(&s);
    s.cleanup_registered = true;
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
        static detail::LLKSharedDevices s;
        return s;
    }

    static void SetUpTestSuite() {
        auto& s = shared_state();
        // Belt-and-suspenders shutdown safety; idempotent. See CleanupRegistry comment.
        detail::ensure_safe_cleanup_for(s);
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
        // Happy-path cleanup. The CleanupRegistry atexit hook is the fallback
        // for paths that bypass this (see ensure_safe_cleanup_for comment).
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
        // Belt-and-suspenders shutdown safety; idempotent. See CleanupRegistry comment.
        detail::ensure_safe_cleanup_for(s);
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
        // Happy-path cleanup. The CleanupRegistry atexit hook is the fallback
        // for paths that bypass this (see ensure_safe_cleanup_for comment).
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
