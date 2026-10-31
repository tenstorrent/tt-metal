// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
//
// X280Runtime::revive on hardware. Every test strands harts with
// firmware-strand.bin in a way that only an RNMI recovers, revives them, and
// proves the tile serves the sentinel again, so the binary is reversible: it
// never writes L2CPU_RESET and leaves the tile in resident idle.

#include "x280/device/x280_runtime.hpp"

#include "x280_hw_addrs.hpp"
#include "x280_runtime_impl.hpp"
#include "x280_tile_impl.hpp"

#include "x280.h"
#include "x280_test_images.hpp"

#include <gtest/gtest.h>
#include <mesh_device.hpp>
#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/kernel_types.hpp>

#include <algorithm>
#include <array>
#include <chrono>
#include <cstring>
#include <iostream>
#include <memory>
#include <string>
#include <thread>
#include <vector>

namespace {

using namespace x280;
using namespace std::chrono_literals;
using Clock = std::chrono::steady_clock;

// Mirrors src/strand.c.
constexpr uint64_t kStrandMagic = 0x535452414E440001ull;
enum StrandMode : uint32_t {
    kReturn = 0,
    kWfiIrqOff = 1,
    kSpinIrqOff = 2,
    kExceptionLoop = 3,
    kNocBurst = 4,
    kTrap = 5,
};
constexpr uint64_t kStrandResultMode = 0x57A9D000ull;

struct StrandArgs {
    uint64_t magic = kStrandMagic;
    uint64_t run_value = 0;
    std::array<uint32_t, kHartsPerTile> mode{};
    uint32_t noc_x = 0;
    uint32_t noc_y = 0;
    uint64_t target_addr = 0;
    uint8_t reserved[16]{};
};
static_assert(sizeof(StrandArgs) == 64);

// The burst lands in primed LIM inside the active region, above the strand
// image and below every active stack: a NoC loopback into the tile's own LIM,
// 4 KiB per hart.
constexpr uint64_t kBurstTarget = X280_LIM_BASE + 0x10000;

// RNMI cause of an interrupt entry (measured, tools/l2cpu README "RNMI").
constexpr uint64_t kRnmiCause = 0x8000000000000002ull;

const char* mode_name(uint32_t mode) {
    switch (mode) {
        case kReturn: return "return";
        case kWfiIrqOff: return "wfi_irq_off";
        case kSpinIrqOff: return "spin_irq_off";
        case kExceptionLoop: return "exception_loop";
        case kNocBurst: return "noc_burst";
        case kTrap: return "trap";
        default: return "unknown";
    }
}

struct Stats {
    std::vector<double> us;
    void add(std::chrono::microseconds v) { us.push_back(static_cast<double>(v.count())); }
    void add(Clock::duration v) { us.push_back(std::chrono::duration<double, std::micro>(v).count()); }
    double median() {
        std::sort(us.begin(), us.end());
        return us.empty() ? 0 : us[us.size() / 2];
    }
    double max() const { return us.empty() ? 0 : *std::max_element(us.begin(), us.end()); }
};

class X280ReviveHw : public ::testing::Test {
protected:
    void SetUp() override {
        mesh_ = tt::tt_metal::distributed::MeshDevice::create_unit_mesh(0);
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
        id_ = ids.front();
        tile_ = &runtime_->tile(id_);
        access_ = std::make_unique<detail::TileTestAccess>(*tile_);

        const auto state = tile_->inspect();
        ASSERT_TRUE(state.observation_valid) << state.invalid_reason;
        if (state.reset_released && !state.resident_idle_live) {
            // Left by an earlier process (a test that failed, or a strand in
            // another binary). Revive is the recovery under test, so it is
            // used here too, and what it found is printed.
            const auto revived = runtime_->revive(*tile_);
            std::cout << "[ revive  ] tile found out of resident idle at start: " << describe(revived) << "\n";
            ASSERT_TRUE(revived.revived()) << describe(revived) << "\n" << describe(state);
        }
        x280::test::boot_resident_idle(*runtime_, id_);
    }

    void TearDown() override {
        access_.reset();
        tile_ = nullptr;
        runtime_.reset();
        mesh_.reset();
    }

    std::string strand_image() const { return x280::test::test_image("X280_STRAND_FW_IMAGE", "firmware-strand.bin"); }
    std::string sentinel_image() const { return x280::test::test_image("X280_ACTIVE_FW_IMAGE", "firmware.bin"); }

    // Writes the argument block and enqueues the strand image.
    SessionId enqueue_strand(
        const std::array<uint32_t, kHartsPerTile>& modes, uint64_t run_value, bool revive_on_cleanup_timeout = false) {
        const auto available = tile_->available_lim();
        StrandArgs args;
        args.run_value = run_value;
        args.mode = modes;
        const auto coord = runtime_->virtual_core_from_tile(id_, tt::tt_metal::NOC::NOC_0);
        args.noc_x = coord.x();
        args.noc_y = coord.y();
        args.target_addr = kBurstTarget;
        std::vector<uint8_t> bytes(sizeof(args));
        std::memcpy(bytes.data(), &args, sizeof(args));
        tile_->write_available_lim(available.begin, bytes);

        ActiveRunRequest request;
        request.image_path = strand_image();
        request.run_value = run_value;
        request.cleanup_timeout = 3s;
        request.args_addr = available.begin;
        request.args_bytes = sizeof(args);
        request.revive_on_cleanup_timeout = revive_on_cleanup_timeout;
        const auto enqueued = runtime_->enqueue(*tile_, request);
        EXPECT_EQ(enqueued.outcome, EnqueueOutcome::Enqueued) << describe(enqueued);
        return enqueued.session;
    }

    // Every hart has published RUNNING and reached its mode (the result word
    // names it), so the stranded ones are where the mode puts them.
    void wait_stranded(SessionId id, const std::array<uint32_t, kHartsPerTile>& modes) {
        // The result words decide, not the aggregate RUNNING wait: a
        // conforming hart may already be past RUNNING, and a trapped one is
        // never counted as running.
        auto& session = runtime_->session(id);
        const auto deadline = Clock::now() + 3s;
        for (;;) {
            const auto harts = session.inspect();
            bool reached = true;
            for (uint32_t hart = 0; hart < kHartsPerTile; ++hart) {
                const uint64_t result = harts[hart].state.result;
                // The burst counts its rounds above bit 32.
                reached = reached && (result & 0xFFFFFFFFull) == kStrandResultMode + modes[hart];
            }
            if (reached) {
                break;
            }
            ASSERT_LT(Clock::now(), deadline) << "harts did not reach their modes: " << describe(harts);
        }
        // Let every stranded hart settle into its loop.
        std::this_thread::sleep_for(5ms);
    }

    // The sentinel runs to completion and returns: the tile serves again.
    void expect_sentinel(uint64_t value) {
        ActiveRunRequest request;
        request.image_path = sentinel_image();
        request.run_value = value;
        request.cleanup_timeout = 3s;
        const auto enqueued = runtime_->enqueue(*tile_, request);
        ASSERT_EQ(enqueued.outcome, EnqueueOutcome::Enqueued) << describe(enqueued);
        auto& session = runtime_->session(enqueued.session);
        const auto result = session.wait_until_result_ready(3s);
        ASSERT_EQ(result.outcome, WaitOutcome::ResultReady) << describe(result);
        EXPECT_TRUE(active_result_matches(result.hartstates, value)) << describe(result);
        const auto cleaned = runtime_->cleanup(enqueued.session, 3s);
        ASSERT_EQ(cleaned.outcome, CleanupOutcome::ReturnedToIdle) << describe(cleaned);
    }

    // A revived hart never re-enters the image it was taken out of: its
    // command and entry were cleared before the trigger, so idle stays idle.
    void expect_stays_idle() {
        std::this_thread::sleep_for(20ms);
        const auto state = tile_->inspect();
        ASSERT_TRUE(state.resident_idle_live) << describe(state);
        for (const auto& hart : state.harts) {
            EXPECT_EQ(hart.state.phase, X280_HART_PHASE_IDLE);
            EXPECT_EQ(hart.state.stage, X280_HART_STAGE_REVIVED);
            EXPECT_EQ(hart.control.command, X280_HART_COMMAND_NONE);
        }
    }

    // A revived burst leaves its windows programmed; the image would have
    // cleared them on a cooperative exit.
    void clear_strand_windows() {
        for (uint32_t index = 0; index < kHartsPerTile; ++index) {
            access_->write_noc_window_words(index, NocWindowWords{});
        }
    }

    std::shared_ptr<tt::tt_metal::distributed::MeshDevice> mesh_;
    std::unique_ptr<X280Runtime> runtime_;
    TileId id_;
    X280Tile* tile_ = nullptr;
    std::unique_ptr<detail::TileTestAccess> access_;
};

class X280ReviveModes : public X280ReviveHw, public ::testing::WithParamInterface<std::tuple<uint32_t, bool>> {};

}  // namespace

TEST_F(X280ReviveHw, BootInstalledTheHandlersAndTheIdleCarriesTheMarker) {
    for (uint32_t hart = 0; hart < kHartsPerTile; ++hart) {
        const uint64_t base = kRnmiHandlerBase + hart * kRnmiHandlerStride;
        EXPECT_EQ(access_->read_reg32(base + 0), static_cast<uint32_t>(X280_IDLE_RNMI_ENTRY_ADDR)) << hart;
        EXPECT_EQ(access_->read_reg32(base + 4), 0u) << hart;
        EXPECT_EQ(access_->read_reg32(base + 8), static_cast<uint32_t>(X280_IDLE_RNMI_EXC_ADDR)) << hart;
        EXPECT_EQ(access_->read_reg32(base + 12), 0u) << hart;
    }
    EXPECT_EQ(access_->read_reg32(kRnmiTrigger), 0u);
    EXPECT_TRUE(idle_rnmi_marker_matches(access_->read_u64(X280_IDLE_RNMI_MAGIC_ADDR)));
}

TEST_F(X280ReviveHw, ReviveOfAnIdleTileIsHarmless) {
    const auto revived = runtime_->revive(*tile_);
    ASSERT_EQ(revived.outcome, ReviveOutcome::Revived) << describe(revived);
    for (const auto& hart : revived.harts) {
        EXPECT_TRUE(hart.acknowledged);
        EXPECT_EQ(hart.revive_cause, kRnmiCause);
        // Taken in resident idle's own poll loop.
        EXPECT_GE(hart.revive_epc, X280_IDLE_FW_LOAD_ADDR);
        EXPECT_LT(hart.revive_epc, X280_IDLE_FW_LOAD_ADDR + X280_IDLE_FW_REGION_SIZE);
    }
    EXPECT_TRUE(tile_->healthy());
    expect_stays_idle();
    expect_sentinel(0x5E471E0000000001ull);
}

TEST_F(X280ReviveHw, RefusesWithoutWritingWhenTheHandlersAreNotInstalled) {
    const uint64_t base = kRnmiHandlerBase + 2 * kRnmiHandlerStride;
    const uint32_t saved = access_->read_reg32(base);
    access_->write_reg32(base, 0);
    const auto refused = runtime_->revive(*tile_);
    access_->write_reg32(base, saved);
    EXPECT_EQ(refused.outcome, ReviveOutcome::NotInstalled) << describe(refused);
    EXPECT_EQ(access_->read_reg32(kRnmiTrigger), 0u);
    EXPECT_TRUE(tile_->healthy());
    EXPECT_TRUE(tile_->inspect().resident_idle_live);
    for (const auto& hart : tile_->inspect().harts) {
        EXPECT_NE(hart.state.stage, X280_HART_STAGE_REVIVED);
    }
}

TEST_F(X280ReviveHw, RefusesAPartialMaskWhileASessionIsLive) {
    const std::array<uint32_t, kHartsPerTile> modes = {kReturn, kWfiIrqOff, kReturn, kReturn};
    const auto id = enqueue_strand(modes, 0x5E47000000000010ull);
    ASSERT_TRUE(id);
    wait_stranded(id, modes);
    const auto refused = runtime_->revive(*tile_, ReviveRequest{.hart_mask = 0x2});
    EXPECT_EQ(refused.outcome, ReviveOutcome::InvalidRequest) << describe(refused);
    EXPECT_TRUE(runtime_->session_live(id));

    const auto revived = runtime_->revive(*tile_);
    ASSERT_EQ(revived.outcome, ReviveOutcome::Revived) << describe(revived);
    EXPECT_FALSE(runtime_->session_live(id));
    EXPECT_EQ(runtime_->cleanup(id, 100ms).outcome, CleanupOutcome::UnknownSession);
    expect_stays_idle();
    expect_sentinel(0x5E471E0000000010ull);
}

// Each way of leaving the contract, on hart 1 alone and on all four harts.
TEST_P(X280ReviveModes, RevivesAStrandedSessionAndServesTheNextImage) {
    const auto [mode, all_four] = GetParam();
    std::array<uint32_t, kHartsPerTile> modes{kReturn, mode, kReturn, kReturn};
    if (all_four) {
        modes.fill(mode);
    }
    const uint64_t run = 0x5E47100000000000ull | (uint64_t{mode} << 8) | (all_four ? 1 : 0);
    const auto id = enqueue_strand(modes, run);
    ASSERT_TRUE(id);
    wait_stranded(id, modes);

    // The proof it is stranded: a plain STOP would not end it.
    const auto before = runtime_->session(id).inspect();
    for (uint32_t hart = 0; hart < kHartsPerTile; ++hart) {
        if (modes[hart] != kReturn) {
            EXPECT_EQ(before[hart].state.stage, X280_HART_STAGE_MAIN) << describe(before);
        }
    }

    const auto revived = runtime_->revive(*tile_);
    std::cout << "[ revive  ] " << mode_name(mode) << (all_four ? " x4: " : " x1: ") << describe(revived) << "\n";
    ASSERT_EQ(revived.outcome, ReviveOutcome::Revived) << describe(revived);
    EXPECT_FALSE(runtime_->session_live(id));
    EXPECT_TRUE(tile_->healthy());
    for (uint32_t hart = 0; hart < kHartsPerTile; ++hart) {
        const auto& revived_hart = revived.harts[hart];
        EXPECT_TRUE(revived_hart.acknowledged) << hart;
        if (modes[hart] == kReturn) {
            continue;
        }
        // Where the hart was: inside the active image.
        EXPECT_GE(revived_hart.revive_epc, X280_ACTIVE_FW_LOAD_ADDR) << hart;
        EXPECT_LT(revived_hart.revive_epc, X280_ACTIVE_FW_REGION_END) << hart;
        EXPECT_EQ(revived_hart.revive_cause, kRnmiCause) << hart;
        if (mode == kTrap) {
            // The trap record the image left is the evidence; the handler
            // cleared it in LIM.
            EXPECT_EQ(revived_hart.before.trap.trap_valid, 1u) << hart;
            EXPECT_EQ(revived_hart.before.trap.mcause, 3u) << hart;
            EXPECT_EQ(revived_hart.after.trap.trap_valid, 0u) << hart;
        }
        if (mode == kNocBurst) {
            // The burst was under way: it counts completed 4 KiB rounds above
            // bit 32 of its result word.
            EXPECT_GE(revived_hart.before.state.result >> 32, 1u) << hart;
        }
    }
    expect_stays_idle();
    if (mode == kNocBurst) {
        clear_strand_windows();
    }
    expect_sentinel(run + 1);
}

INSTANTIATE_TEST_SUITE_P(
    Strand,
    X280ReviveModes,
    ::testing::Combine(::testing::Values(kWfiIrqOff, kSpinIrqOff, kExceptionLoop, kNocBurst, kTrap), ::testing::Bool()),
    [](const auto& info) {
        return std::string(mode_name(std::get<0>(info.param))) + (std::get<1>(info.param) ? "_all" : "_hart1");
    });

TEST_F(X280ReviveHw, CleanupEscalatesToReviveWhenTheRequestOptsIn) {
    const std::array<uint32_t, kHartsPerTile> modes = {kSpinIrqOff, kReturn, kWfiIrqOff, kReturn};
    const uint64_t run = 0x5E47200000000001ull;
    const auto id = enqueue_strand(modes, run, /*revive_on_cleanup_timeout=*/true);
    ASSERT_TRUE(id);
    wait_stranded(id, modes);

    const auto cleaned = runtime_->cleanup(id, 50ms);
    ASSERT_EQ(cleaned.outcome, CleanupOutcome::Revived) << describe(cleaned);
    EXPECT_TRUE(cleaned.cleaned_up());
    ASSERT_TRUE(cleaned.revive.has_value());
    EXPECT_TRUE(cleaned.revive->revived());
    EXPECT_TRUE(tile_->healthy());
    EXPECT_FALSE(runtime_->session_live(id));
    expect_stays_idle();
    expect_sentinel(run + 1);
}

TEST_F(X280ReviveHw, WithoutOptingInCleanupStillLatchesAndReviveClearsTheLatch) {
    const std::array<uint32_t, kHartsPerTile> modes = {kReturn, kReturn, kReturn, kWfiIrqOff};
    const uint64_t run = 0x5E47300000000001ull;
    const auto id = enqueue_strand(modes, run);
    ASSERT_TRUE(id);
    wait_stranded(id, modes);

    const auto cleaned = runtime_->cleanup(id, 50ms);
    ASSERT_EQ(cleaned.outcome, CleanupOutcome::StillRunningActiveFw) << describe(cleaned);
    EXPECT_FALSE(cleaned.revive.has_value());
    ASSERT_FALSE(tile_->healthy());
    EXPECT_EQ(
        runtime_->enqueue(*tile_, ActiveRunRequest{.image_path = sentinel_image(), .run_value = 1}).outcome,
        EnqueueOutcome::Unhealthy);

    // Harts 0-2 sit in RETURNED_TO_IDLE waiting for an acknowledgement that
    // will not come; hart 3 is in WFI. Revive takes all four.
    const auto revived = runtime_->revive(*tile_);
    ASSERT_EQ(revived.outcome, ReviveOutcome::Revived) << describe(revived);
    EXPECT_TRUE(tile_->healthy());
    expect_sentinel(run + 1);
}

// 100 strand / revive / next-image cycles: one stranded hart (the others wait
// for STOP), four stranded harts, and the mechanism alone on an idle hart 1.
TEST_F(X280ReviveHw, HundredCyclesAndTheirWallClock) {
    constexpr int kCycles = 100;
    struct Variant {
        const char* name;
        std::array<uint32_t, kHartsPerTile> modes;
        bool strand;
        uint32_t mask;
    };
    const Variant variants[] = {
        {"one stranded hart", {kReturn, kWfiIrqOff, kReturn, kReturn}, true, kAllHartsMask},
        {"four stranded harts", {kWfiIrqOff, kWfiIrqOff, kWfiIrqOff, kWfiIrqOff}, true, kAllHartsMask},
        {"idle hart 1, mask 0x2", {}, false, 0x2},
    };
    for (const auto& variant : variants) {
        Stats ack;
        Stats idle;
        Stats call;
        Stats to_enqueued;
        Stats to_result;
        for (int cycle = 0; cycle < kCycles; ++cycle) {
            const uint64_t run = 0x5E47400000000000ull + (uint64_t(cycle) << 4);
            if (variant.strand) {
                const auto id = enqueue_strand(variant.modes, run);
                ASSERT_TRUE(id);
                wait_stranded(id, variant.modes);
            }
            const auto t0 = Clock::now();
            const auto revived = runtime_->revive(*tile_, ReviveRequest{.hart_mask = variant.mask});
            const auto t1 = Clock::now();
            ASSERT_EQ(revived.outcome, ReviveOutcome::Revived)
                << variant.name << " cycle " << cycle << ": " << describe(revived);
            ack.add(revived.trigger_to_ack);
            idle.add(revived.trigger_to_idle);
            call.add(t1 - t0);

            // The next image: trigger -> Enqueued, and -> its result. The
            // trigger lands inside revive(); its offset from the call is
            // (call - trigger_to_idle) at most, so these add trigger_to_idle
            // to the time after the call returned.
            ActiveRunRequest request;
            request.image_path = sentinel_image();
            request.run_value = run + 1;
            request.cleanup_timeout = 3s;
            const auto enqueued = runtime_->enqueue(*tile_, request);
            const auto t2 = Clock::now();
            ASSERT_EQ(enqueued.outcome, EnqueueOutcome::Enqueued) << describe(enqueued);
            const auto ready = runtime_->session(enqueued.session).wait_until_result_ready(3s);
            const auto t3 = Clock::now();
            ASSERT_EQ(ready.outcome, WaitOutcome::ResultReady) << describe(ready);
            to_enqueued.add(revived.trigger_to_idle + std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1));
            to_result.add(revived.trigger_to_idle + std::chrono::duration_cast<std::chrono::microseconds>(t3 - t1));
            const auto cleaned = runtime_->cleanup(enqueued.session, 3s);
            ASSERT_EQ(cleaned.outcome, CleanupOutcome::ReturnedToIdle) << describe(cleaned);
        }
        std::cout << "[ cycles  ] " << variant.name << ", " << kCycles << " cycles (us, median / max):"
                  << " trigger->ack " << ack.median() << " / " << ack.max() << ", trigger->idle " << idle.median()
                  << " / " << idle.max() << ", revive() call " << call.median() << " / " << call.max()
                  << ", trigger->next Enqueued " << to_enqueued.median() << " / " << to_enqueued.max()
                  << ", trigger->next result ready " << to_result.median() << " / " << to_result.max() << "\n";
    }
}
