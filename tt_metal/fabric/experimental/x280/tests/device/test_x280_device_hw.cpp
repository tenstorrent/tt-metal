// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
//
// Hardware tests for the X280 component.
//
// Once runtime initialization reports ready, every test body is reversible: it
// saves what it perturbs and restores it, then verifies the restore. Nothing
// here releases reset. Automatic priming and its reset-required result are
// covered by the separately gated x280_prime_test.
//
// TileTestAccess is a private fault-injection seam for reversible states that
// the semantic public API cannot create. It borrows the production TileImpl,
// runtime lifetime, chip lock, and transport rather than constructing a
// parallel path.
//
// A runtime that cannot become ready reports a skip. A later invalid observation
// is a real problem and fails.
//
//   TT_METAL_RUNTIME_ROOT=<tt-metal> TT_VISIBLE_DEVICES=N x280_device_tests

#include "x280/device/x280_runtime.hpp"
#include "x280/device/x280_tile.hpp"

#include "x280_metal_transport.hpp"
#include "x280_hw_addrs.hpp"
#include "x280_runtime_impl.hpp"
#include "x280_tile_impl.hpp"
#include "x280_test_images.hpp"

#include "x280.h"
#include "rt/x280_noc_map.h"

#include <gtest/gtest.h>
#include <mesh_device.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-umd/types/noc_id.hpp>

#include <algorithm>
#include <cstdint>
#include <exception>
#include <ios>
#include <numeric>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

using namespace x280;

// Restores a LIM u64 when it goes out of scope, so a failing assertion cannot
// leave the tile perturbed.
class ScopedLimWord {
public:
    ScopedLimWord(detail::TileTestAccess& access, uint64_t addr) :
        access_(access), addr_(addr), original_(access.read_u64(addr)) {}

    // A destructor is implicitly noexcept, so an escaping transport error would
    // terminate the process instead of reporting the tile it left perturbed.
    ~ScopedLimWord() {
        try {
            access_.write_u64(addr_, original_);
        } catch (const std::exception& error) {
            ADD_FAILURE() << "failed to restore LIM word at 0x" << std::hex << addr_ << ": " << error.what();
        }
    }

    uint64_t original() const { return original_; }
    void set(uint64_t value) { access_.write_u64(addr_, value); }

private:
    detail::TileTestAccess& access_;
    uint64_t addr_;
    uint64_t original_;
};

// The same, for a LIM byte range.
class ScopedLimRegion {
public:
    ScopedLimRegion(detail::TileTestAccess& access, uint64_t addr, size_t size) :
        access_(access), addr_(addr), original_(access.read_block(addr, size)) {}

    ~ScopedLimRegion() {
        try {
            access_.write_block(addr_, original_);
        } catch (const std::exception& error) {
            ADD_FAILURE() << "failed to restore LIM region at 0x" << std::hex << addr_ << ": " << error.what();
        }
    }

    const std::vector<uint8_t>& original() const { return original_; }

private:
    detail::TileTestAccess& access_;
    uint64_t addr_;
    std::vector<uint8_t> original_;
};

class ScopedAvailableLimBlock {
public:
    ScopedAvailableLimBlock(X280Tile& tile, uint64_t address, size_t size) :
        tile_(tile), address_(address), original_(tile.read_available_lim(address, size)) {}

    ~ScopedAvailableLimBlock() {
        if (restored_) {
            return;
        }
        try {
            tile_.write_available_lim(address_, original_);
        } catch (const std::exception& error) {
            ADD_FAILURE() << "failed to restore available LIM block at 0x" << std::hex << address_ << ": "
                          << error.what();
        }
    }

    const std::vector<uint8_t>& original() const { return original_; }

    void restore() {
        tile_.write_available_lim(address_, original_);
        restored_ = true;
    }

private:
    X280Tile& tile_;
    uint64_t address_;
    std::vector<uint8_t> original_;
    bool restored_ = false;
};

std::vector<int> visible_device_ids();

bool needs_operator_preparation(const RuntimeOpenResult& opened) {
    return opened.requires_preparation() || opened.requires_reset();
}

class X280DeviceHw : public ::testing::Test {
protected:
    void SetUp() override {
        // tt-metal opens the device; the component binds to it. This order is
        // the contract, and doing it the other way round makes create_unit_mesh
        // throw because a silicon MetalContext already exists.
        mesh_ = tt::tt_metal::distributed::MeshDevice::create_unit_mesh(0);
        auto opened =
            X280Runtime::open(mesh_, RuntimeOpenConfig{.priming_policy = LimEccPrimingPolicy::RequirePrepared});
        if (!opened.runtime) {
            if (needs_operator_preparation(opened)) {
                GTEST_SKIP() << "X280 hardware needs operator preparation: " << x280::test::unprepared(opened);
            }
            FAIL() << "X280 runtime initialization failed: " << opened.diagnostic;
        }
        runtime_ = std::make_unique<X280Runtime>(std::move(*opened.runtime));

        const auto tiles = runtime_->tile_ids();
        ASSERT_FALSE(tiles.empty()) << "no enabled, unharvested L2CPU tiles";
        tile_id_ = tiles.front();

        tile_handle_ = &runtime_->tile(tile_id_);
        tile_access_ = std::make_unique<detail::TileTestAccess>(*tile_handle_);
    }

    void TearDown() override {
        tile_access_.reset();
        tile_handle_ = nullptr;
        runtime_.reset();
        mesh_.reset();
    }

    // Skips on a state only an operator can change; fails on a broken one.
    void require_primed(const TileState& state) {
        ASSERT_TRUE(state.observation_valid) << "invalid observation: " << state.invalid_reason;
        if (state.ecc == LimEccState::Unprimed) {
            GTEST_SKIP() << "LIM ECC is unprimed after runtime initialization";
        }
        if (state.ecc == LimEccState::LimUnavailable) {
            GTEST_SKIP() << "WayEnable leaves no LIM until an ASIC reset (tt-smi -r <device>).";
        }
    }

    std::shared_ptr<tt::tt_metal::distributed::MeshDevice> mesh_;
    std::unique_ptr<X280Runtime> runtime_;
    X280Tile* tile_handle_ = nullptr;
    std::unique_ptr<detail::TileTestAccess> tile_access_;
    TileIdentity tile_id_;
};

class X280FullMeshHw : public ::testing::Test {
protected:
    void SetUp() override {
        expected_ids_ = visible_device_ids();
        ASSERT_FALSE(expected_ids_.empty()) << "no Tenstorrent devices are visible";

        using tt::tt_metal::distributed::MeshDevice;
        using tt::tt_metal::distributed::MeshDeviceConfig;
        mesh_ = MeshDevice::create(MeshDeviceConfig(std::nullopt));
        auto mesh_ids = mesh_->get_device_ids();
        std::sort(mesh_ids.begin(), mesh_ids.end());
        ASSERT_EQ(mesh_ids, expected_ids_) << "system MeshDevice did not include every visible device";

        auto opened =
            X280Runtime::open(mesh_, RuntimeOpenConfig{.priming_policy = LimEccPrimingPolicy::RequirePrepared});
        if (!opened.runtime) {
            if (needs_operator_preparation(opened)) {
                GTEST_SKIP() << "X280 hardware needs operator preparation: " << x280::test::unprepared(opened);
            }
            FAIL() << "X280 runtime initialization failed: " << opened.diagnostic;
        }
        runtime_ = std::make_unique<X280Runtime>(std::move(*opened.runtime));
    }

    void TearDown() override {
        runtime_.reset();
        mesh_.reset();
    }

    std::vector<int> expected_ids_;
    std::shared_ptr<tt::tt_metal::distributed::MeshDevice> mesh_;
    std::unique_ptr<X280Runtime> runtime_;
};

std::vector<int> visible_device_ids() {
    const auto count = tt::tt_metal::GetNumAvailableDevices();
    std::vector<int> ids(count);
    std::iota(ids.begin(), ids.end(), 0);
    return ids;
}

void expect_inventory_matches_devices(const X280Runtime& runtime, const std::vector<int>& expected_device_ids) {
    const auto ids = runtime.tile_ids();
    if (ids.empty()) {
        ADD_FAILURE() << "runtime inventory is empty";
        return;
    }

    EXPECT_TRUE(std::is_sorted(ids.begin(), ids.end()));
    for (const auto& id : ids) {
        EXPECT_TRUE(std::binary_search(expected_device_ids.begin(), expected_device_ids.end(), id.chip_id))
            << "runtime returned tile identity for chip " << id.chip_id << " outside the mesh";
    }
    for (int chip_id : expected_device_ids) {
        EXPECT_TRUE(std::any_of(ids.begin(), ids.end(), [&](const TileId& id) { return id.chip_id == chip_id; }))
            << "runtime inventory omitted chip " << chip_id;
    }
    for (size_t i = 1; i < ids.size(); i++) {
        EXPECT_NE(ids[i - 1], ids[i]) << "runtime returned a duplicate tile identity";
    }
}

}  // namespace

// ---------------------------------------------------------------------------
// Mesh binding
// ---------------------------------------------------------------------------

// The physical chip id must come from build_id(), not id(): the latter is a
// process-global mesh counter that happens to be 0 for the first mesh a process
// opens, so using id() goes unnoticed on a single-card system.
TEST_F(X280DeviceHw, RuntimeInventoryUsesTheMeshsPhysicalChipId) {
    const auto tiles = runtime_->tile_ids();
    ASSERT_FALSE(tiles.empty());
    for (const auto& tile : tiles) {
        EXPECT_EQ(tile.chip_id, static_cast<int>(mesh_->build_id()));
    }
}

// Runtime inventory is globally stable across chips rather than identifying a
// tile only by its chip-local hardware index.
TEST(X280DeviceMesh, RuntimeInventoryHasStableCrossChipIdentity) {
    if (tt::tt_metal::GetNumAvailableDevices() < 2) {
        GTEST_SKIP() << "a multi-device Blackhole system is required";
    }

    using tt::tt_metal::distributed::MeshDevice;
    using tt::tt_metal::distributed::MeshDeviceConfig;
    using tt::tt_metal::distributed::MeshShape;

    auto mesh = MeshDevice::create(MeshDeviceConfig(MeshShape(1, 2)));
    auto expected_ids = mesh->get_device_ids();
    std::sort(expected_ids.begin(), expected_ids.end());

    const auto access = X280MetalTransport::bind(*mesh);
    for (int chip_id : expected_ids) {
        if (!access->is_mmio_local(chip_id)) {
            GTEST_SKIP() << "chip " << chip_id << " is not MMIO-local";
        }
    }

    auto opened = X280Runtime::open(mesh, RuntimeOpenConfig{.priming_policy = LimEccPrimingPolicy::RequirePrepared});
    if (!opened.runtime && needs_operator_preparation(opened)) {
        GTEST_SKIP() << "X280 hardware needs operator preparation: " << x280::test::unprepared(opened);
    }
    ASSERT_TRUE(opened.runtime) << opened.diagnostic;
    auto runtime = std::move(*opened.runtime);
    expect_inventory_matches_devices(runtime, expected_ids);
}

// The default MeshDeviceConfig uses the connected system mesh shape, so this
// covers every device visible to the process rather than a fixed-size subset.
TEST_F(X280FullMeshHw, RuntimeInventoryCoversEveryVisibleDeviceInOneMesh) {
    expect_inventory_matches_devices(*runtime_, expected_ids_);
}

TEST_F(X280FullMeshHw, RuntimeReadOnlyGateCoversEveryVisibleTile) {
    const auto ids = runtime_->tile_ids();
    ASSERT_FALSE(ids.empty());

    for (const auto& id : ids) {
        SCOPED_TRACE("chip " + std::to_string(id.chip_id) + " hw L2CPU " + std::to_string(id.hw_index));
        EXPECT_GE(id.hw_index, 0);
        EXPECT_EQ(hw_l2cpu_from_noc0(id.noc_x, id.noc_y), id.hw_index);

        const auto state = runtime_->tile(id).inspect();
        ASSERT_TRUE(state.observation_valid) << state.invalid_reason;
        EXPECT_FALSE(is_all_ones(state.raw_wayenable));
        EXPECT_EQ(state.ecc, LimEccState::Primed);
        EXPECT_EQ(state.lim_backed_bytes, lim_backed_bytes(state.raw_wayenable));
        EXPECT_TRUE(state.pll_mhz.has_value());
    }
}

TEST_F(X280FullMeshHw, RuntimeAvailableLimScratchWriteRestoresEveryTile) {
    const auto ids = runtime_->tile_ids();
    ASSERT_FALSE(ids.empty());
    size_t covered = 0;

    for (const auto& id : ids) {
        SCOPED_TRACE("chip " + std::to_string(id.chip_id) + " hw L2CPU " + std::to_string(id.hw_index));
        auto& tile = runtime_->tile(id);
        const auto available = tile.available_lim();
        ASSERT_GE(available.size(), kLimCacheLineBytes) << "configured LIM shape leaves no whole available cache line";

        const uint64_t address = available.begin;
        ScopedAvailableLimBlock restore(tile, address, kLimCacheLineBytes);
        std::vector<uint8_t> pattern(kLimCacheLineBytes);
        for (size_t i = 0; i < pattern.size(); i++) {
            pattern[i] = static_cast<uint8_t>(0x5Au ^ id.chip_id ^ id.hw_index ^ i);
        }

        tile.write_available_lim(address, pattern);
        EXPECT_EQ(tile.read_available_lim(address, pattern.size()), pattern);

        restore.restore();
        EXPECT_EQ(tile.read_available_lim(address, restore.original().size()), restore.original());
        covered++;
    }

    EXPECT_EQ(covered, ids.size());
}

// create_unit_meshes is Metal's supported way to hold one live unit mesh per
// device. Bind the runtime to each handle in turn so every visible physical
// device crosses the same X280 initialization and inventory boundary.
TEST(X280DeviceMesh, EveryVisibleDeviceOpensThroughAUnitMesh) {
    using tt::tt_metal::distributed::MeshDevice;

    const auto expected_ids = visible_device_ids();
    ASSERT_FALSE(expected_ids.empty()) << "no Tenstorrent devices are visible";

    auto meshes = MeshDevice::create_unit_meshes(expected_ids);
    ASSERT_EQ(meshes.size(), expected_ids.size());

    for (int chip_id : expected_ids) {
        SCOPED_TRACE("chip " + std::to_string(chip_id));
        const auto& mesh = meshes.at(chip_id);
        ASSERT_EQ(mesh->get_device_ids(), std::vector<int>{chip_id});

        const auto access = X280MetalTransport::bind(*mesh);
        if (!access->is_mmio_local(chip_id)) {
            GTEST_SKIP() << "chip " << chip_id << " is not MMIO-local";
        }

        auto opened =
            X280Runtime::open(mesh, RuntimeOpenConfig{.priming_policy = LimEccPrimingPolicy::RequirePrepared});
        if (!opened.runtime && needs_operator_preparation(opened)) {
            GTEST_SKIP() << "X280 hardware needs operator preparation: " << x280::test::unprepared(opened);
        }
        ASSERT_TRUE(opened.runtime) << opened.diagnostic;
        auto runtime = std::move(*opened.runtime);
        expect_inventory_matches_devices(runtime, std::vector<int>{chip_id});
    }
}

// Multiple unit submeshes may coexist in Metal, but X280 has one runtime owner.
// A rejected second handle must not invalidate the runtime bound to the first.
TEST(X280DeviceMesh, DifferentLiveUnitMeshIsRejectedWithoutDisturbingOwner) {
    using tt::tt_metal::distributed::MeshDevice;

    if (tt::tt_metal::GetNumAvailableDevices() < 2) {
        GTEST_SKIP() << "at least two visible devices are required";
    }

    auto meshes = MeshDevice::create_unit_meshes({0, 1});
    const auto& owner_mesh = meshes.at(0);
    const auto& other_mesh = meshes.at(1);

    auto owner_opened =
        X280Runtime::open(owner_mesh, RuntimeOpenConfig{.priming_policy = LimEccPrimingPolicy::RequirePrepared});
    if (!owner_opened.runtime && needs_operator_preparation(owner_opened)) {
        GTEST_SKIP() << "X280 hardware needs operator preparation: " << x280::test::unprepared(owner_opened);
    }
    ASSERT_TRUE(owner_opened.runtime) << owner_opened.diagnostic;
    auto owner = std::move(*owner_opened.runtime);
    const auto owner_ids = owner.tile_ids();

    auto rejected = X280Runtime::open(other_mesh);
    EXPECT_EQ(rejected.outcome, RuntimeOpenOutcome::Failed);
    EXPECT_FALSE(rejected.runtime);
    EXPECT_NE(rejected.diagnostic.find("different live MeshDevice"), std::string::npos) << rejected.diagnostic;
    EXPECT_EQ(owner.tile_ids(), owner_ids);
}

TEST_F(X280DeviceHw, RepeatedOpenPreservesReadyInventory) {
    const auto expected = runtime_->tile_ids();
    auto* expected_tile = &runtime_->tile(tile_id_);
    auto reopened = X280Runtime::open(mesh_);
    ASSERT_TRUE(reopened.runtime) << reopened.diagnostic;
    EXPECT_EQ(reopened.runtime->lim_shape(), LimShape{});
    EXPECT_EQ(reopened.runtime->tile_ids(), expected);
    EXPECT_EQ(&reopened.runtime->tile(tile_id_), expected_tile);
}

TEST_F(X280DeviceHw, LiveRuntimeRejectsAConflictingLimShapeWithoutMutation) {
    const auto before = tile_handle_->inspect();
    auto reopened = X280Runtime::open(
        mesh_,
        RuntimeOpenConfig{
            .priming_policy = LimEccPrimingPolicy::RequirePrepared,
            .lim_shape = LimShape{.lim_ways = 14},
        });

    EXPECT_EQ(reopened.outcome, RuntimeOpenOutcome::Failed);
    EXPECT_FALSE(reopened.runtime);
    EXPECT_NE(reopened.diagnostic.find("different LIM shape"), std::string::npos) << reopened.diagnostic;

    const auto after = tile_handle_->inspect();
    EXPECT_EQ(after.raw_wayenable, before.raw_wayenable);
    EXPECT_EQ(after.raw_ecc_marker, before.raw_ecc_marker);
    EXPECT_EQ(runtime_->lim_shape(), LimShape{});
}

// Every inventory identity resolves to the runtime-owned tile object.
TEST_F(X280DeviceHw, InventoryIdentitiesResolveToStableTiles) {
    const auto ids = runtime_->tile_ids();
    for (const auto& id : ids) {
        auto& tile = runtime_->tile(id);
        EXPECT_EQ(tile.id(), id);
        EXPECT_EQ(&runtime_->tile(id), &tile);
        const auto state = tile.inspect();
        EXPECT_TRUE(state.observation_valid || !state.invalid_reason.empty()) << "an invalid observation must say why";
    }
}

TEST_F(X280DeviceHw, TileRefusesIoAfterItsMeshCloses) {
    auto& tile = runtime_->tile(tile_id_);
    tile_access_.reset();
    tile_handle_ = nullptr;
    ASSERT_TRUE(mesh_->close());

    EXPECT_THROW((void)tile.inspect(), std::runtime_error);
    mesh_.reset();
}

// ---------------------------------------------------------------------------
// Observation
// ---------------------------------------------------------------------------

// The ECC prime marker is host-only LIM SRAM, so an unprimed machine can be
// presented to the classifier without touching WayEnable. This covers the read
// path, the marker address, and the classification on real silicon; it does not
// re-prove ECC initialisation on genuinely uninitialised SRAM, which only a
// fresh machine can show.
TEST_F(X280DeviceHw, EccClassificationFollowsTheMarker) {
    const auto before = tile_handle_->inspect();
    require_primed(before);
    if (::testing::Test::IsSkipped()) {
        return;
    }
    ASSERT_EQ(before.raw_ecc_marker, X280_ECC_PRIME_MARKER_MAGIC);

    {
        ScopedLimWord marker(*tile_access_, X280_ECC_PRIME_MARKER_ADDR);
        marker.set(0xDEADBEEF12345678ull);

        const auto perturbed = tile_handle_->inspect();
        EXPECT_TRUE(perturbed.observation_valid);
        EXPECT_EQ(perturbed.ecc, LimEccState::Unprimed) << "a non-magic marker must classify as unprimed";
        EXPECT_EQ(perturbed.raw_ecc_marker, 0xDEADBEEF12345678ull) << "the raw value must be preserved as read";
        // WayEnable is untouched, so the machine is unprimed rather than
        // committed. Getting this backwards would send an operator to reset a
        // board that only needed priming.
        EXPECT_EQ(perturbed.raw_wayenable & 0xFu, 0u);
    }

    const auto after = tile_handle_->inspect();
    EXPECT_EQ(after.ecc, LimEccState::Primed) << "marker not restored";
    EXPECT_EQ(after.raw_ecc_marker, X280_ECC_PRIME_MARKER_MAGIC);
}

// The extent word sits beside the marker in the same host-only LIM SRAM, so the
// prime record can be perturbed without touching WayEnable either. This covers
// the extent read path and the two classifications a bare tile cannot show: an
// extent that is not a whole number of primed lines, and no extent at all.
TEST_F(X280DeviceHw, EccClassificationFollowsTheExtent) {
    const auto before = tile_handle_->inspect();
    require_primed(before);
    if (::testing::Test::IsSkipped()) {
        return;
    }
    ASSERT_EQ(before.raw_ecc_marker, X280_ECC_PRIME_MARKER_MAGIC);

    {
        ScopedLimWord extent(*tile_access_, X280_ECC_PRIME_EXTENT_ADDR);
        extent.set(kLimEccPrimeLineSize + 1);

        const auto partial_line = tile_handle_->inspect();
        EXPECT_TRUE(partial_line.observation_valid);
        EXPECT_EQ(partial_line.ecc, LimEccState::Unprimed) << "a part-line extent is not a prime record";
        EXPECT_EQ(partial_line.raw_ecc_extent, kLimEccPrimeLineSize + 1) << "the raw value must be preserved as read";
        EXPECT_FALSE(partial_line.ecc_primed_bytes.has_value()) << "a rejected record must carry no extent";

        extent.set(0);

        const auto unrecorded = tile_handle_->inspect();
        EXPECT_TRUE(unrecorded.observation_valid);
        EXPECT_EQ(unrecorded.ecc, LimEccState::Primed) << "a zero extent is a marker seeded without one, still a prime";
        EXPECT_EQ(unrecorded.raw_ecc_extent, 0u);
        EXPECT_FALSE(unrecorded.ecc_primed_bytes.has_value()) << "a zero extent records no walk";
    }

    const auto after = tile_handle_->inspect();
    EXPECT_EQ(after.raw_ecc_extent, before.raw_ecc_extent) << "extent not restored";
    EXPECT_EQ(after.ecc, before.ecc);
}

// Liveness needs the tile out of reset as well as matching mailboxes, because
// LIM keeps the mailboxes across a return to reset. Whichever state this machine
// is in, the two must agree.
TEST_F(X280DeviceHw, LivenessAgreesWithResetState) {
    const auto state = tile_handle_->inspect();
    ASSERT_TRUE(state.observation_valid) << state.invalid_reason;

    bool all_harts_idle = true;
    for (const auto& hart : state.harts) {
        all_harts_idle =
            all_harts_idle && hart.observation_valid && hart.state.protocol_version == X280_HART_PROTOCOL_VERSION &&
            hart.state.phase == X280_HART_PHASE_IDLE && hart.state.heartbeat == X280_HART_IDLE_HEARTBEAT(hart.hart_id);
    }
    EXPECT_EQ(state.resident_idle_live, state.reset_released && all_harts_idle);
    if (!state.reset_released) {
        EXPECT_FALSE(state.resident_idle_live) << "a tile in reset cannot be running anything";
    }
}

// Metal translates coordinates but does not manage UMD's thread-local physical
// NOC selector. Production transport scopes every wrapper delegate to NOC0.
TEST_F(X280DeviceHw, InspectionPinsNoc0WhenCallerSelectedNoc1) {
    const auto baseline = tile_handle_->inspect();
    ASSERT_TRUE(baseline.observation_valid) << baseline.invalid_reason;

    tt::umd::NocIdSwitcher caller_noc(tt::umd::NocId::NOC1);
    const auto observed = tile_handle_->inspect();
    ASSERT_TRUE(observed.observation_valid) << observed.invalid_reason;
    EXPECT_EQ(observed.id, baseline.id);
    EXPECT_EQ(observed.reset_released, baseline.reset_released);
    EXPECT_EQ(observed.raw_wayenable, baseline.raw_wayenable);
    EXPECT_EQ(observed.raw_ecc_marker, baseline.raw_ecc_marker);
    EXPECT_EQ(observed.ecc, baseline.ecc);
}

// ---------------------------------------------------------------------------
// Boot refusals and runtime PLL ownership
// ---------------------------------------------------------------------------

TEST_F(X280DeviceHw, RuntimeConfirmsTheCurrentChipGlobalPll) {
    const auto state = tile_handle_->inspect();
    ASSERT_TRUE(state.observation_valid) << state.invalid_reason;
    if (!state.pll_mhz.has_value()) {
        GTEST_SKIP() << "needs PLL telemetry to confirm the current clock without moving it";
    }

    EXPECT_NO_THROW(runtime_->set_pll(tile_id_.chip_id, *state.pll_mhz));
    const auto confirmed = tile_handle_->inspect();
    ASSERT_TRUE(confirmed.observation_valid) << confirmed.invalid_reason;
    EXPECT_EQ(confirmed.pll_mhz, state.pll_mhz);
}

// A boot that would have to release reset must refuse unless asked,
// because the release cannot be undone for the chip-reset lifetime.
TEST_F(X280DeviceHw, ColdBootRefusedWithoutOptIn) {
    const auto state = tile_handle_->inspect();
    require_primed(state);
    if (::testing::Test::IsSkipped()) {
        return;
    }
    if (state.reset_released) {
        GTEST_SKIP() << "this tile is already out of reset, so boot() answers before the opt-in gate";
    }

    auto& tile = runtime_->tile(tile_id_);
    BootRequest request;
    request.idle_image_path = "/nonexistent";  // never read: the refusal comes first
    request.allow_reset_release = false;

    const auto result = tile.boot(request);
    EXPECT_EQ(result.outcome, BootOutcome::Failed);
    EXPECT_NE(result.diagnostic.find("allow_reset_release"), std::string::npos) << result.diagnostic;

    // And it really did not touch reset.
    EXPECT_EQ(tile_handle_->inspect().reset_released, state.reset_released);
}

// The record sits in the L2CPU scratch, which an ASIC reset clears. This
// fixture opens at the reset shape, which is reachable only through a reset
// since WayEnable is increase-only, and a prepare needs WayEnable 1: so the
// record must read None, and the refusal must leave it as found.
TEST_F(X280DeviceHw, PinnedInputPrepareRefusesTheResetShapeWithoutWriting) {
    const auto before = tile_handle_->pinned_input_state();
    EXPECT_EQ(before.status, PinnedInputStatus::None) << describe(before);

    const auto result = tile_handle_->prepare_pinned_input(PinnedInputPrepareRequest{});
    EXPECT_EQ(result.outcome, PinnedInputPrepareOutcome::WrongShape) << describe(result);
    EXPECT_FALSE(result.enqueue.has_value());
    EXPECT_EQ(tile_handle_->pinned_input_state().words, before.words);
}

// A tile out of reset without valid resident idle cannot be recovered in
// software: harts leave reset only once per chip-reset lifetime. The component
// must say so and stop — above all it must not try to reassert reset.
//
// Reached by clearing one hart's heartbeat on a tile that is genuinely running.
// The idle firmware stamps that word once during init and never reads it again,
// so this is invisible to the hart in its poll loop, and writing the value back
// restores the truth rather than faking it.
TEST_F(X280DeviceHw, OutOfResetWithoutResidentIdleDemandsReset) {
    const auto before = tile_handle_->inspect();
    ASSERT_TRUE(before.observation_valid) << before.invalid_reason;
    if (!before.resident_idle_live) {
        GTEST_SKIP() << "needs a tile already in resident idle to perturb";
    }

    {
        constexpr uint32_t kPerturbedHart = 2;
        const uint64_t heartbeat_addr = X280_HART_PROTOCOL_BASE + offsetof(x280_hart_protocol_t, state) +
                                        kPerturbedHart * sizeof(x280_hart_state_t) +
                                        offsetof(x280_hart_state_t, heartbeat);
        ScopedLimWord heartbeat(*tile_access_, heartbeat_addr);
        heartbeat.set(0);

        const auto perturbed = tile_handle_->inspect();
        ASSERT_TRUE(perturbed.reset_released);
        EXPECT_EQ(perturbed.harts[kPerturbedHart].state.heartbeat, 0u);
        for (uint32_t hart = 0; hart < kHartsPerTile; ++hart) {
            if (hart != kPerturbedHart) {
                EXPECT_EQ(perturbed.harts[hart].state.heartbeat, X280_HART_IDLE_HEARTBEAT(hart));
            }
        }
        ASSERT_FALSE(perturbed.resident_idle_live);

        auto& tile = runtime_->tile(tile_id_);
        BootRequest request;
        request.allow_reset_release = true;
        request.idle_image_path = "/nonexistent";  // never opened

        const auto result = tile.boot(request);
        EXPECT_EQ(result.outcome, BootOutcome::ResetRequiredNoResidentIdle) << result.diagnostic;
        EXPECT_TRUE(reset_required(result.outcome));
        EXPECT_NE(result.diagnostic.find("heartbeat"), std::string::npos) << result.diagnostic;

        EXPECT_TRUE(tile_handle_->inspect().reset_released) << "reset must not be reasserted";
    }

    const auto after = tile_handle_->inspect();
    EXPECT_TRUE(after.resident_idle_live) << "heartbeat not restored";
}

// ---------------------------------------------------------------------------
// Image bounds
// ---------------------------------------------------------------------------

// An oversized image must be rejected before any of it reaches LIM.
TEST_F(X280DeviceHw, OversizedIdleImageIsRejectedWithoutWritingLim) {
    const auto state = tile_handle_->inspect();
    require_primed(state);
    if (::testing::Test::IsSkipped()) {
        return;
    }
    if (state.reset_released) {
        GTEST_SKIP() << "restoring the region is a LIM write, which would land under a running hart";
    }

    // The whole region, not just up to the bound: a write that overran lands on
    // the marker at +0x800 and the idle stack above it, which is precisely what a
    // short snapshot would miss.
    ScopedLimRegion region(*tile_access_, X280_IDLE_FW_LOAD_ADDR, X280_IDLE_FW_REGION_SIZE);

    const std::vector<uint8_t> oversized(X280_IDLE_FW_REGION_SIZE, 0xA5);
    EXPECT_THROW(tile_access_->load_idle_image(oversized), std::runtime_error);

    // Read back before the guard restores, or the comparison proves nothing.
    const auto after = tile_access_->read_block(X280_IDLE_FW_LOAD_ADDR, region.original().size());
    EXPECT_EQ(after, region.original()) << "LIM was modified while loading an image that should never be written";
}

// The active region's upper limit reserves all four per-hart stacks.
TEST_F(X280DeviceHw, OversizedActiveImageIsRejectedWithoutWritingLim) {
    const auto state = tile_handle_->inspect();
    require_primed(state);
    if (::testing::Test::IsSkipped()) {
        return;
    }
    if (state.reset_released) {
        GTEST_SKIP() << "restoring the region is a LIM write, which would land under a running hart";
    }

    constexpr size_t kProbeBytes = 0x1000;
    ScopedLimRegion region(*tile_access_, X280_ACTIVE_FW_LOAD_ADDR, kProbeBytes);

    const std::vector<uint8_t> oversized(X280_ACTIVE_FW_REGION_END - X280_ACTIVE_FW_LOAD_ADDR, 0x5A);
    EXPECT_THROW(tile_access_->load_active_image(oversized), std::runtime_error);

    const auto after = tile_access_->read_block(X280_ACTIVE_FW_LOAD_ADDR, kProbeBytes);
    EXPECT_EQ(after, region.original()) << "the active region was written by an image that should never be loaded";
}

// ---------------------------------------------------------------------------
// Outbound NOC windows
// ---------------------------------------------------------------------------

// The runtime exposes read-only NOC diagnostics but no public NOC
// mutator. Reading both ownership classes establishes the dynamic-TLB route
// without perturbing firmware state.
TEST_F(X280DeviceHw, ReadsOutboundNocWindowsWithoutMutation) {
    auto& tile = runtime_->tile(tile_id_);
    const auto host = tile.read_noc_window(X280_TLB_HOST_BASE);
    EXPECT_TRUE(host.ok) << host.diagnostic;
    const auto firmware = tile.read_noc_window(X280_TLB_BENCH_WRITE);
    EXPECT_TRUE(firmware.ok) << firmware.diagnostic;
}

// ---------------------------------------------------------------------------
// LIM map
// ---------------------------------------------------------------------------

// Socket callers place config buffers and FIFOs at addresses they choose, and
// nothing downstream detects a collision with a running image. The component has
// to publish what it owns.
TEST_F(X280DeviceHw, PublishesALimMapThatExcludesTheBootProtocol) {
    auto& tile = runtime_->tile(tile_id_);
    const auto reserved = tile.reserved_lim();

    EXPECT_LE(reserved.begin, X280_IDLE_FW_LOAD_ADDR);
    EXPECT_TRUE(reserved.contains(X280_ACTIVE_FW_LOAD_ADDR, 1)) << "the active region must be reserved";
    EXPECT_TRUE(reserved.contains(X280_HART_PROTOCOL_BASE, sizeof(x280_hart_protocol_t)))
        << "every hart's lifecycle record must be reserved";
    EXPECT_TRUE(reserved.contains(X280_ECC_PRIME_MARKER_ADDR, 16))
        << "the ECC marker and the extent word beside it must be reserved";

    const auto state = tile.inspect();
    if (!state.observation_valid || state.lim_backed_bytes == 0) {
        GTEST_SKIP() << "LIM is unbacked, so there is no available window to check";
    }
    const auto available = tile.available_lim();
    EXPECT_GE(available.begin, reserved.end) << "available LIM must start above everything the component owns";
    EXPECT_GT(available.size(), 0u);
}

TEST_F(X280DeviceHw, AvailableLimIoRejectsInvalidRangesWithoutMutation) {
    auto& tile = runtime_->tile(tile_id_);
    const auto available = tile.available_lim();
    ASSERT_GE(available.size(), kLimCacheLineBytes);
    const auto before = tile.read_available_lim(available.begin, kLimCacheLineBytes);
    const std::vector<uint8_t> pattern(kLimCacheLineBytes, 0xA5);

    EXPECT_THROW(tile.write_available_lim(tile.reserved_lim().begin, pattern), std::runtime_error);
    EXPECT_THROW(tile.write_available_lim(available.begin + 4, pattern), std::runtime_error);
    EXPECT_THROW(tile.read_available_lim(available.end, 4), std::runtime_error);

    EXPECT_EQ(tile.read_available_lim(available.begin, before.size()), before);
}
