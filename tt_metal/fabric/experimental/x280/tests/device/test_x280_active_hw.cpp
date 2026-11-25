// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
//
// Reversible active-session definitions. The fixture boots the first tile into
// resident idle when it is not live; the tests never write L2CPU_RESET.

#include "x280/device/x280_runtime.hpp"

#include "x280_runtime_impl.hpp"
#include "x280_tile_impl.hpp"

#include "x280.h"
#include "x280_test_images.hpp"

#include <gtest/gtest.h>
#include <mesh_device.hpp>

#include <chrono>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using namespace x280;
using namespace std::chrono_literals;

constexpr uint64_t kRunA = 0xA11CE0000000000Aull;
constexpr uint64_t kRunB = 0xB11CE0000000000Bull;
constexpr uint64_t kRunContract = 0xC047AC7000000000ull;

// Colon-separated paths, empty fields dropped.
std::vector<std::string> split_image_list(const std::string& list) {
    std::vector<std::string> paths;
    for (size_t begin = 0; begin <= list.size();) {
        const size_t end = list.find(':', begin);
        std::string path = list.substr(begin, end == std::string::npos ? std::string::npos : end - begin);
        if (!path.empty()) {
            paths.push_back(std::move(path));
        }
        if (end == std::string::npos) {
            break;
        }
        begin = end + 1;
    }
    return paths;
}

// The images the contract suite was asked to run. X280_ACTIVE_FW_IMAGES names them; when it is
// unset, the single-image variable the sentinel tests use names them; when
// both are unset or empty, the build's contract set (the sentinel and atomic-bench) is requested.
// An X280_ACTIVE_FW_IMAGES that is set but names no path opts the suite out.
struct RequestedImages {
    std::vector<std::string> paths;
    bool build_defaults = false;
};

RequestedImages requested_images() {
    if (const char* many = std::getenv("X280_ACTIVE_FW_IMAGES"); many != nullptr) {
        return {split_image_list(many), false};
    }
    if (const char* one = std::getenv("X280_ACTIVE_FW_IMAGE"); one != nullptr && *one != '\0') {
        return {split_image_list(one), false};
    }
    return {
        {std::string(X280_IMAGE_DIR) + "/firmware.bin",
         x280::test::test_image("X280_ATOMIC_BENCH_FW_IMAGE", "firmware-atomic-bench.bin")},
        true};
}

// The requested images that exist. A requested path that does not is dropped rather than failed,
// since a list may name images a checkout has not built; EveryRequestedImageResolves reports each
// drop and fails when nothing resolves, or when a build default is missing.
std::vector<std::string> images_from_env() {
    std::vector<std::string> images;
    for (auto& path : requested_images().paths) {
        if (std::ifstream(path).good()) {
            images.push_back(std::move(path));
        }
    }
    return images;
}

// Mesh, runtime and tile setup shared by the sentinel tests and the
// parameterised contract suite. Image-specific helpers take an explicit path.
class X280ActiveHwBase : public ::testing::Test {
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
        x280::test::boot_resident_idle(*runtime_, ids.front());
        tile_ = &runtime_->tile(ids.front());
        access_ = std::make_unique<detail::TileTestAccess>(*tile_);
    }

    void TearDown() override {
        access_.reset();
        tile_ = nullptr;
        runtime_.reset();
        mesh_.reset();
    }

    // Tile liveness only. It consults no image variable: the contract suite
    // takes its image from the test parameter, and a gate that also required
    // X280_ACTIVE_FW_IMAGE would skip every parameterised case for a variable
    // they do not read.
    void require_live_tile() {
        const auto state = tile_->inspect();
        ASSERT_TRUE(state.observation_valid) << state.invalid_reason;
        if (!state.resident_idle_live || !state.pll_mhz.has_value()) {
            GTEST_SKIP() << "requires a live resident idle with PLL telemetry";
        }
    }

    ActiveRunRequest request(const std::string& image, uint64_t value) const {
        ActiveRunRequest request;
        request.image_path = image;
        request.run_value = value;
        request.cleanup_timeout = 3s;
        return request;
    }

    SessionId enqueue_image(const std::string& image, uint64_t value) {
        const auto enqueued = runtime_->enqueue(*tile_, request(image, value));
        EXPECT_EQ(enqueued.outcome, EnqueueOutcome::Enqueued) << describe(enqueued);
        if (!enqueued.session) {
            throw std::runtime_error("enqueue did not return an active session");
        }
        return enqueued.session;
    }

    // Every conforming active image, whatever its cross-hart scheme and
    // whether or not it has work to do without a host driver.
    //
    // wait_until_result_ready is deliberately not part of this: it means
    // "every hart completed on its own and is parked in RUNNING", which an
    // image gated on a config it never receives cannot satisfy, and being told
    // to stop legitimately advances a hart past RUNNING. That is
    // expect_self_completion.
    void expect_contract(SessionId id, uint64_t value) {
        auto& session = runtime_->session(id);
        const auto running = session.wait_until_running(3s);
        ASSERT_EQ(running.outcome, WaitOutcome::Running) << describe(running);

        const auto cleaned = runtime_->cleanup(id, 3s);
        ASSERT_EQ(cleaned.outcome, CleanupOutcome::ReturnedToIdle) << describe(cleaned);
        EXPECT_TRUE(active_all_idle(cleaned.hartstates, value)) << describe(cleaned);
        EXPECT_TRUE(tile_->healthy());

        // Terminal fields outlive restoration: resident idle rewrites only
        // version, run value, heartbeat and phase (see lim_idle.c).
        for (const auto& hart : cleaned.hartstates) {
            EXPECT_EQ(hart.state.terminal_status, X280_HART_STATUS_SUCCESS) << describe(cleaned);
        }
    }

    // Self-completion: every hart finished unprompted and is parked in
    // RUNNING. Only images that have work without a host driver satisfy this.
    void expect_self_completion(SessionId id) {
        const auto result = runtime_->session(id).wait_until_result_ready(3s);
        ASSERT_EQ(result.outcome, WaitOutcome::ResultReady) << describe(result);
    }

    // The sentinel's own contract: every hart echoed the run value. No
    // lifecycle predicate implies this, so it is asserted explicitly.
    void expect_echo(const std::array<HartState, kHartsPerTile>& hartstates, uint64_t value) {
        EXPECT_TRUE(active_result_matches(hartstates, value)) << describe(hartstates);
    }

    // The runtime owns the tile and every session, so a test that drops it
    // rebinds through this rather than reaching a stale reference.
    void reopen() {
        access_.reset();
        tile_ = nullptr;
        runtime_.reset();

        auto opened =
            X280Runtime::open(mesh_, RuntimeOpenConfig{.priming_policy = LimEccPrimingPolicy::RequirePrepared});
        ASSERT_TRUE(opened.runtime) << opened.diagnostic;
        runtime_ = std::make_unique<X280Runtime>(std::move(*opened.runtime));
        const auto ids = runtime_->tile_ids();
        ASSERT_FALSE(ids.empty());
        tile_ = &runtime_->tile(ids.front());
        access_ = std::make_unique<detail::TileTestAccess>(*tile_);
    }

    std::shared_ptr<tt::tt_metal::distributed::MeshDevice> mesh_;
    std::unique_ptr<X280Runtime> runtime_;
    X280Tile* tile_ = nullptr;
    std::unique_ptr<detail::TileTestAccess> access_;
};

// Runtime mechanics against the sentinel: busy, reaping, the destruction
// sweep, skew, a rejected image. These test the runtime, not image conformance.
class X280ActiveHw : public X280ActiveHwBase {
protected:
    std::string success_image() const { return x280::test::test_image("X280_ACTIVE_FW_IMAGE", "firmware.bin"); }

    SessionId enqueue(uint64_t value) { return enqueue_image(success_image(), value); }

    ActiveRunRequest request(uint64_t value) const { return X280ActiveHwBase::request(success_image(), value); }

    // The sentinel completes on its own and echoes the run value, so it can
    // be held to the whole contract plus both extras.
    void expect_run(SessionId id, uint64_t value) {
        auto& session = runtime_->session(id);
        const auto running = session.wait_until_running(3s);
        ASSERT_EQ(running.outcome, WaitOutcome::Running) << describe(running);

        const auto result = session.wait_until_result_ready(3s);
        ASSERT_EQ(result.outcome, WaitOutcome::ResultReady) << describe(result);
        for (uint32_t hart = 0; hart < kHartsPerTile; ++hart) {
            EXPECT_TRUE(hart_result_ready(result.hartstates[hart], value));
        }
        expect_echo(result.hartstates, value);

        const auto cleaned = runtime_->cleanup(id, 3s);
        ASSERT_EQ(cleaned.outcome, CleanupOutcome::ReturnedToIdle) << describe(cleaned);
        EXPECT_TRUE(active_all_idle(cleaned.hartstates, value));
    }
};

// active-session-fault is excluded — it is the deliberate non-conformer and
// has its own gated binary and test.
class X280ActiveContract : public X280ActiveHwBase, public ::testing::WithParamInterface<std::string> {};

}  // namespace

TEST_P(X280ActiveContract, ConformsToTheActiveImageContract) {
    require_live_tile();
    if (::testing::Test::IsSkipped()) {
        return;
    }
    expect_contract(enqueue_image(GetParam(), kRunContract), kRunContract);
}

INSTANTIATE_TEST_SUITE_P(Images, X280ActiveContract, ::testing::ValuesIn(images_from_env()));

// An empty suite is legitimate when X280_ACTIVE_FW_IMAGES names no path, or only
// images this checkout has not built. Without this, gtest fails the binary for
// having a parameterised suite with no cases, where every other test here skips.
// It follows that gtest cannot tell a deliberately empty suite from a broken
// invocation — EveryRequestedImageResolves is what catches the latter.
GTEST_ALLOW_UNINSTANTIATED_PARAMETERIZED_TEST(X280ActiveContract);

// A path list that resolves to nothing instantiates no cases and passes in
// seconds, which is indistinguishable from every image conforming.
//
// Deliberately not a fixture test. X280ActiveHwBase::SetUp skips without a
// prepared runtime, and a broken path list is detectable with no hardware at
// all, so inheriting it would skip exactly where this has to fail.
TEST(X280ActiveContractImages, EveryRequestedImageResolves) {
    const auto [requested, build_defaults] = requested_images();
    if (requested.empty()) {
        GTEST_SKIP() << "X280_ACTIVE_FW_IMAGES names no paths: the contract suite is opted out";
    }

    std::vector<std::string> dropped;
    for (const auto& path : requested) {
        if (!std::ifstream(path).good()) {
            dropped.push_back(path);
        }
    }

    for (const auto& path : dropped) {
        std::cout << "[  DROPPED ] not readable, no contract case: " << path << "\n";
    }

    std::string names;
    for (const auto& path : dropped) {
        names += "\n  " + path;
    }
    if (build_defaults && !dropped.empty()) {
        FAIL() << "the build's contract images are missing, so X280ActiveContract would silently skip them; "
                  "build x280_images or set X280_ACTIVE_FW_IMAGES:"
               << names;
    }
    if (dropped.size() == requested.size()) {
        FAIL() << "none of the " << requested.size()
               << " requested paths resolve, so X280ActiveContract "
                  "instantiates no case and the suite proves nothing:"
               << names;
    }

    std::cout << "[   IMAGES ] " << requested.size() - dropped.size() << " of " << requested.size()
              << " requested images resolved\n";
}

TEST_F(X280ActiveHw, ResidentIdleGatePerformsNoActiveImageWrite) {
    require_live_tile();
    if (::testing::Test::IsSkipped()) {
        return;
    }

    const auto image_before = access_->read_block(X280_ACTIVE_FW_LOAD_ADDR, 64);
    const uint64_t heartbeat_addr = X280_HART_PROTOCOL_BASE + offsetof(x280_hart_protocol_t, state) +
                                    2 * sizeof(x280_hart_state_t) + offsetof(x280_hart_state_t, heartbeat);
    const uint64_t heartbeat = access_->read_u64(heartbeat_addr);
    access_->write_u64(heartbeat_addr, 0);
    const auto enqueued = runtime_->enqueue(*tile_, request(kRunA));
    access_->write_u64(heartbeat_addr, heartbeat);

    EXPECT_EQ(enqueued.outcome, EnqueueOutcome::NotResidentIdle);
    EXPECT_FALSE(enqueued.session);
    EXPECT_EQ(access_->read_block(X280_ACTIVE_FW_LOAD_ADDR, 64), image_before);
}

TEST_F(X280ActiveHw, ARejectedImageReleasesTheSlotAndLeavesTheTileHealthy) {
    require_live_tile();
    if (::testing::Test::IsSkipped()) {
        return;
    }

    // The reservation is taken before the image is read, so a rejected image
    // releases it. Rejection precedes any write to the per-hart records, which
    // is what distinguishes this from the latching path.
    ActiveRunRequest bad = request(kRunA);
    bad.image_path = "/nonexistent/x280-active-image.bin";
    const auto rejected = runtime_->enqueue(*tile_, bad);
    EXPECT_EQ(rejected.outcome, EnqueueOutcome::ImageRejected) << describe(rejected);
    EXPECT_FALSE(rejected.session);

    EXPECT_TRUE(tile_->healthy());
    EXPECT_FALSE(tile_->last_failure().has_value());
    EXPECT_TRUE(runtime_->sessions().empty());
    EXPECT_TRUE(tile_->inspect().resident_idle_live);

    const SessionId id = enqueue(kRunA);
    expect_run(id, kRunA);
}

TEST_F(X280ActiveHw, OneSkewedHartPreventsEarlyAggregateSuccess) {
    require_live_tile();
    if (::testing::Test::IsSkipped()) {
        return;
    }
    const SessionId id = enqueue(kRunA);
    auto& session = runtime_->session(id);
    const auto running = session.wait_until_running(3s);
    ASSERT_EQ(running.outcome, WaitOutcome::Running) << describe(running);

    const uint64_t phase_addr = X280_HART_PROTOCOL_BASE + offsetof(x280_hart_protocol_t, state) +
                                3 * sizeof(x280_hart_state_t) + offsetof(x280_hart_state_t, phase);
    const uint64_t phase = access_->read_u64(phase_addr);
    access_->write_u64(phase_addr, 0);
    const auto skewed = session.wait_until_result_ready(25ms);
    EXPECT_NE(skewed.outcome, WaitOutcome::ResultReady) << describe(skewed);
    for (uint32_t hart = 0; hart < 3; ++hart) {
        EXPECT_TRUE(hart_result_ready(skewed.hartstates[hart], kRunA)) << describe(skewed);
    }
    EXPECT_FALSE(hart_result_ready(skewed.hartstates[3], kRunA)) << describe(skewed);
    access_->write_u64(phase_addr, phase);

    const auto result = session.wait_until_result_ready(3s);
    ASSERT_EQ(result.outcome, WaitOutcome::ResultReady) << describe(result);
    const auto cleaned = runtime_->cleanup(id, 3s);
    ASSERT_TRUE(cleaned.cleaned_up()) << describe(cleaned);
}

TEST_F(X280ActiveHw, SecondEnqueueIsBusyWhileSessionIsActive) {
    require_live_tile();
    if (::testing::Test::IsSkipped()) {
        return;
    }
    const SessionId id = enqueue(kRunA);

    const auto busy = runtime_->enqueue(*tile_, request(kRunB));
    EXPECT_EQ(busy.outcome, EnqueueOutcome::Busy);
    EXPECT_FALSE(busy.session);
    expect_run(id, kRunA);
}

TEST_F(X280ActiveHw, RuntimeOwnedTileRunsDistinctAAndBWithoutChangingReset) {
    require_live_tile();
    if (::testing::Test::IsSkipped()) {
        return;
    }
    const uint32_t reset_before = access_->read_l2cpu_reset_raw();

    const SessionId first = enqueue(kRunA);
    expect_run(first, kRunA);
    const SessionId second = enqueue(kRunB);
    expect_run(second, kRunB);

    EXPECT_NE(kRunA, kRunB);
    EXPECT_NE(first.value, second.value);
    EXPECT_EQ(access_->read_l2cpu_reset_raw(), reset_before);
}

TEST_F(X280ActiveHw, RuntimeDestructionRestoresResidentIdle) {
    require_live_tile();
    if (::testing::Test::IsSkipped()) {
        return;
    }
    const SessionId id = enqueue(kRunA);
    const auto result = runtime_->session(id).wait_until_result_ready(3s);
    ASSERT_EQ(result.outcome, WaitOutcome::ResultReady) << describe(result);
    ASSERT_TRUE(runtime_->session_live(id));

    reopen();
    if (::testing::Test::HasFatalFailure()) {
        return;
    }

    // Without the sweep the harts stay in RUNNING_ACTIVE_FW, which this gate
    // rejects. The reopened runtime's registry is empty either way, so it
    // proves nothing about the sweep and is not asserted.
    const auto state = tile_->inspect();
    EXPECT_TRUE(state.observation_valid) << state.invalid_reason;
    EXPECT_TRUE(state.resident_idle_live);
}

TEST_F(X280ActiveHw, CleanupReapsTheSessionAndRefusesAStaleId) {
    require_live_tile();
    if (::testing::Test::IsSkipped()) {
        return;
    }
    const SessionId id = enqueue(kRunA);
    const auto result = runtime_->session(id).wait_until_result_ready(3s);
    ASSERT_EQ(result.outcome, WaitOutcome::ResultReady) << describe(result);

    const auto cleaned = runtime_->cleanup(id, 3s);
    ASSERT_TRUE(cleaned.cleaned_up()) << describe(cleaned);

    EXPECT_FALSE(runtime_->session_live(id));
    EXPECT_ANY_THROW((void)runtime_->session(id));
    const auto stale = runtime_->cleanup(id, 3s);
    EXPECT_EQ(stale.outcome, CleanupOutcome::UnknownSession) << describe(stale);

    const SessionId fresh = enqueue(kRunB);
    EXPECT_NE(fresh.value, id.value);
    expect_run(fresh, kRunB);
}

TEST_F(X280ActiveHw, SessionsListsLiveExecutions) {
    require_live_tile();
    if (::testing::Test::IsSkipped()) {
        return;
    }
    EXPECT_TRUE(runtime_->sessions().empty());

    const SessionId id = enqueue(kRunA);
    const auto live = runtime_->sessions();
    ASSERT_EQ(live.size(), 1u);
    EXPECT_EQ(live.front().value, id.value);

    expect_run(id, kRunA);
    if (::testing::Test::HasFatalFailure()) {
        return;
    }
    EXPECT_TRUE(runtime_->sessions().empty());
}

// One session per tile, so the listing is a per-tile table. Ordering comes from
// the registry's tile key, not from issue order: the second tile enqueued first
// still sorts by tile.
TEST_F(X280ActiveHw, SessionsAreListedGroupedByTile) {
    const auto ids = runtime_->tile_ids();
    if (ids.size() < 2) {
        GTEST_SKIP() << "requires at least two enabled X280 tiles";
    }

    X280Tile& low = runtime_->tile(ids.front());
    X280Tile& high = runtime_->tile(ids.back());
    for (const X280Tile* tile : {&low, &high}) {
        const auto state = tile->inspect();
        ASSERT_TRUE(state.observation_valid) << state.invalid_reason;
        if (!state.resident_idle_live) {
            GTEST_SKIP() << "requires two already-live resident idles";
        }
    }
    ASSERT_TRUE(runtime_->sessions().empty());

    // Enqueued high-tile first, so issue order and tile order disagree.
    const auto high_enqueued = runtime_->enqueue(high, request(kRunB));
    ASSERT_EQ(high_enqueued.outcome, EnqueueOutcome::Enqueued) << describe(high_enqueued);
    const auto low_enqueued = runtime_->enqueue(low, request(kRunA));
    ASSERT_EQ(low_enqueued.outcome, EnqueueOutcome::Enqueued) << describe(low_enqueued);

    const auto live = runtime_->sessions();
    ASSERT_EQ(live.size(), 2u);
    EXPECT_EQ(live.front().tile, tile_key(low.id()));
    EXPECT_EQ(live.back().tile, tile_key(high.id()));
    EXPECT_GT(live.front().value, live.back().value);

    for (const auto& id : {low_enqueued.session, high_enqueued.session}) {
        const auto cleaned = runtime_->cleanup(id, 3s);
        EXPECT_EQ(cleaned.outcome, CleanupOutcome::ReturnedToIdle) << describe(cleaned);
    }
    EXPECT_TRUE(runtime_->sessions().empty());
}

TEST_F(X280ActiveHw, SessionIdNamesTheTileItWasEnqueuedOn) {
    require_live_tile();
    if (::testing::Test::IsSkipped()) {
        return;
    }
    const SessionId id = enqueue(kRunA);

    EXPECT_EQ(id.tile, tile_key(tile_->id()));
    const auto live = runtime_->sessions();
    ASSERT_EQ(live.size(), 1u);
    // The listing carries the mapping, so no per-session lookup is needed to
    // learn which tile an execution occupies.
    EXPECT_EQ(live.front().tile, tile_key(tile_->id()));

    expect_run(id, kRunA);
}

// The registry is keyed by tile, so a reaped id would find the tile's next
// session if the generation did not discriminate. Reaping that one on behalf of
// a dead handle is the failure this guards.
TEST_F(X280ActiveHw, AReapedIdDoesNotReachTheTilesNextSession) {
    require_live_tile();
    if (::testing::Test::IsSkipped()) {
        return;
    }
    const SessionId stale = enqueue(kRunA);
    expect_run(stale, kRunA);
    if (::testing::Test::HasFatalFailure()) {
        return;
    }
    EXPECT_FALSE(runtime_->session_live(stale));

    const SessionId live = enqueue(kRunB);
    ASSERT_EQ(live.tile, stale.tile);
    ASSERT_NE(live.value, stale.value);

    // Same tile, so the lookup finds the live entry; only the generation stops
    // it being claimed.
    const auto refused = runtime_->cleanup(stale, 3s);
    EXPECT_EQ(refused.outcome, CleanupOutcome::UnknownSession) << describe(refused);
    EXPECT_FALSE(runtime_->session_live(stale));
    EXPECT_ANY_THROW((void)runtime_->session(stale));

    // The live session is untouched: still listed, still its own run.
    EXPECT_TRUE(runtime_->session_live(live));
    const auto listed = runtime_->sessions();
    ASSERT_EQ(listed.size(), 1u);
    EXPECT_EQ(listed.front().value, live.value);
    EXPECT_EQ(runtime_->session(live).run_value(), kRunB);

    expect_run(live, kRunB);
}

TEST_F(X280ActiveHw, SessionCarriesItsRunValue) {
    require_live_tile();
    if (::testing::Test::IsSkipped()) {
        return;
    }
    const SessionId id = enqueue(kRunA);
    auto& session = runtime_->session(id);
    EXPECT_EQ(session.id().value, id.value);
    EXPECT_EQ(session.tile(), tile_->id());
    ASSERT_EQ(session.run_value(), kRunA);

    expect_self_completion(id);
    if (::testing::Test::HasFatalFailure()) {
        return;
    }

    // A snapshot answers the aggregate predicates only because the session
    // supplies the run value they correlate against. The echo is asserted
    // explicitly: hart_result_ready does not inspect the payload, so "no hart
    // retained A" has to be checked rather than inferred.
    const auto snapshot = session.inspect();
    EXPECT_TRUE(active_all_running(snapshot, session.run_value())) << describe(snapshot);
    expect_echo(snapshot, session.run_value());
    EXPECT_FALSE(active_result_matches(snapshot, kRunB));

    const auto cleaned = runtime_->cleanup(id, 3s);
    ASSERT_TRUE(cleaned.cleaned_up()) << describe(cleaned);
}
