// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
//
// Load-and-execute regression for a reversible active image: one real firmware
// runs end to end on all four harts and returns to idle. X280ActiveHw covers
// the session lifecycle around it. The case restores every range it perturbs,
// so it needs no ASIC reset to undo.

#include "x280/device/x280_runtime.hpp"

#include "x280_runtime_impl.hpp"
#include "x280_tile_impl.hpp"

#include "x280.h"
#include "x280_test_images.hpp"

#include <gtest/gtest.h>
#include <mesh_device.hpp>

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <functional>
#include <memory>
#include <stdexcept>
#include <string>
#include <thread>
#include <type_traits>
#include <utility>
#include <vector>

namespace {

using namespace x280;
using namespace std::chrono_literals;
namespace distributed = tt::tt_metal::distributed;

constexpr auto kRunTimeout = 20s;
constexpr auto kCleanupTimeout = 10s;

constexpr uint64_t kAtomicBase = X280_LIM_BASE + 0x00160000ull;
constexpr uint64_t kAtomicStatus = kAtomicBase + 0x000;
constexpr uint64_t kAtomicConfigReady = kAtomicBase + 0x080;
constexpr uint64_t kAtomicIterations = kAtomicBase + 0x100;
constexpr uint64_t kAtomicOpMode = kAtomicBase + 0x140;
constexpr uint64_t kAtomicPhaseMask = kAtomicBase + 0x180;
constexpr uint64_t kAtomicCounterAddress = kAtomicBase + 0x1c0;
constexpr uint64_t kAtomicDefaultCounter = kAtomicBase + 0x2000;
constexpr uint32_t kAtomicConfigReadyValue = 0x000c0ffe;
constexpr uint32_t kAtomicStatusDone = 0x10000003;
constexpr uint32_t kAtomicAmoaddW = 0;

template <typename T>
std::vector<uint8_t> bytes_of(const T& value) {
    static_assert(std::is_trivially_copyable_v<T>);
    std::vector<uint8_t> bytes(sizeof(T));
    std::memcpy(bytes.data(), &value, sizeof(T));
    return bytes;
}

template <typename T>
T read_pod(const detail::TileTestAccess& access, uint64_t address) {
    static_assert(std::is_trivially_copyable_v<T>);
    const auto bytes = access.read_block(address, sizeof(T));
    T value{};
    std::memcpy(&value, bytes.data(), sizeof(T));
    return value;
}

template <typename T>
void write_pod(detail::TileTestAccess& access, uint64_t address, const T& value) {
    access.write_block(address, bytes_of(value));
}

void zero_region(detail::TileTestAccess& access, uint64_t address, size_t size) {
    access.write_block(address, std::vector<uint8_t>(size, 0));
}

bool poll_until(std::chrono::milliseconds timeout, const std::function<bool()>& predicate) {
    const auto deadline = std::chrono::steady_clock::now() + timeout;
    do {
        if (predicate()) {
            return true;
        }
        std::this_thread::sleep_for(5ms);
    } while (std::chrono::steady_clock::now() < deadline);
    return predicate();
}

class LimSnapshot {
public:
    LimSnapshot(detail::TileTestAccess& access, uint64_t address, size_t size) :
        access_(&access), address_(address), bytes_(access.read_block(address, size)) {}

    LimSnapshot(const LimSnapshot&) = delete;
    LimSnapshot& operator=(const LimSnapshot&) = delete;

    ~LimSnapshot() {
        if (!restored_) {
            try {
                restore();
            } catch (const std::exception& error) {
                ADD_FAILURE() << "could not restore LIM range at 0x" << std::hex << address_ << ": " << error.what();
            }
        }
    }

    void restore() {
        access_->write_block(address_, bytes_);
        restored_ = true;
    }

private:
    detail::TileTestAccess* access_;
    uint64_t address_;
    std::vector<uint8_t> bytes_;
    bool restored_ = false;
};

class SessionCleanup {
public:
    SessionCleanup(X280Runtime& runtime, SessionId id) : runtime_(&runtime), id_(id) {}

    SessionCleanup(const SessionCleanup&) = delete;
    SessionCleanup& operator=(const SessionCleanup&) = delete;

    ~SessionCleanup() {
        if (live_) {
            const auto result = runtime_->cleanup(id_, kCleanupTimeout);
            if (!result.cleaned_up()) {
                ADD_FAILURE() << "active-session cleanup failed: " << describe(result);
            }
        }
    }

    CleanupResult cleanup() {
        const auto result = runtime_->cleanup(id_, kCleanupTimeout);
        live_ = false;
        return result;
    }

private:
    X280Runtime* runtime_;
    SessionId id_;
    bool live_ = true;
};

class X280ActiveWorkloadHw : public ::testing::Test {
protected:
    void SetUp() override {
        mesh_ = distributed::MeshDevice::create_unit_mesh(0);
        auto opened =
            X280Runtime::open(mesh_, RuntimeOpenConfig{.priming_policy = LimEccPrimingPolicy::RequirePrepared});
        if (!opened.runtime) {
            GTEST_SKIP() << "prepared X280 runtime unavailable: " << x280::test::unprepared(opened);
        }
        runtime_ = std::make_unique<X280Runtime>(std::move(*opened.runtime));
        const auto ids = runtime_->tile_ids();
        if (ids.empty()) {
            GTEST_SKIP() << "no enabled X280 tiles";
        }
        x280::test::boot_resident_idle(*runtime_, ids.front());
        if (IsSkipped() || HasFatalFailure()) {
            return;
        }
        tile_ = &runtime_->tile(ids.front());
        access_ = std::make_unique<detail::TileTestAccess>(*tile_);

        const auto state = tile_->inspect();
        ASSERT_TRUE(state.observation_valid) << state.invalid_reason;
        if (!state.resident_idle_live || !state.pll_mhz.has_value()) {
            GTEST_SKIP() << "requires a live resident idle with PLL telemetry";
        }
    }

    void TearDown() override {
        access_.reset();
        tile_ = nullptr;
        runtime_.reset();
        mesh_.reset();
    }

    SessionId enqueue(const std::string& image_path, uint64_t run_value) {
        ActiveRunRequest request;
        request.image_path = image_path;
        request.run_value = run_value;
        request.cleanup_timeout = kCleanupTimeout;
        const auto result = runtime_->enqueue(*tile_, request);
        EXPECT_EQ(result.outcome, EnqueueOutcome::Enqueued) << describe(result);
        if (!result.session) {
            throw std::runtime_error("enqueue did not return an active session");
        }
        return result.session;
    }

    ::testing::AssertionResult terminal_success(SessionId id) {
        auto& session = runtime_->session(id);
        std::array<HartState, kHartsPerTile> snapshot{};
        const bool ready = poll_until(kRunTimeout, [&] {
            snapshot = session.inspect();
            return std::all_of(snapshot.begin(), snapshot.end(), [&](const HartState& hart) {
                return hart_result_ready(hart, session.run_value());
            });
        });
        if (!ready) {
            return ::testing::AssertionFailure() << "workload did not commit terminal success: " << describe(snapshot);
        }
        return ::testing::AssertionSuccess();
    }

    ::testing::AssertionResult cleaned_up(const CleanupResult& result, uint64_t run_value) {
        if (result.outcome != CleanupOutcome::ReturnedToIdle) {
            return ::testing::AssertionFailure() << describe(result);
        }
        if (!active_all_idle(result.hartstates, run_value) && !active_all_returned(result.hartstates, run_value)) {
            return ::testing::AssertionFailure()
                   << "cleanup did not preserve a correlated final record: " << describe(result);
        }
        return ::testing::AssertionSuccess();
    }

    std::shared_ptr<distributed::MeshDevice> mesh_;
    std::unique_ptr<X280Runtime> runtime_;
    X280Tile* tile_ = nullptr;
    std::unique_ptr<detail::TileTestAccess> access_;
};

}  // namespace

TEST_F(X280ActiveWorkloadHw, ActiveImageLoadsExecutesAndReturnsToIdle) {
    const std::string firmware = x280::test::test_image("X280_ATOMIC_BENCH_FW_IMAGE", "firmware-atomic-bench.bin");

    LimSnapshot lim(*access_, kAtomicBase, 0x2100);
    zero_region(*access_, kAtomicBase, 0x2100);
    constexpr uint64_t run = 0x701ce00000000001ull;
    const SessionId id = enqueue(firmware, run);
    SessionCleanup cleanup(*runtime_, id);

    write_pod(*access_, kAtomicIterations, uint64_t{64});
    write_pod(*access_, kAtomicOpMode, kAtomicAmoaddW);
    // Phase-mask bit p enables a phase of p+1 harts; 0x8 runs all four.
    write_pod(*access_, kAtomicPhaseMask, uint32_t{0x8});
    write_pod(*access_, kAtomicCounterAddress, kAtomicDefaultCounter);
    // Doorbell last: the firmware latches the configuration above on this write.
    write_pod(*access_, kAtomicConfigReady, kAtomicConfigReadyValue);

    ASSERT_TRUE(
        poll_until(kRunTimeout, [&] { return read_pod<uint32_t>(*access_, kAtomicStatus) == kAtomicStatusDone; }));
    ASSERT_TRUE(terminal_success(id));

    const auto cleaned = cleanup.cleanup();
    ASSERT_TRUE(cleaned_up(cleaned, run));
}
