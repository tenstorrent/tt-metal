// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include <algorithm>
#include <cstdint>
#include <iostream>
#include <map>
#include <memory>
#include <numeric>
#include <optional>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include <tt-metalium/buffer.hpp>
#include <tt-metalium/buffer_distribution_spec.hpp>
#include <tt-metalium/buffer_types.hpp>
#include <tt-metalium/shape.hpp>
#include <tt-metalium/dispatch_core_common.hpp>
#include <tt-metalium/experimental/per_core_allocation/buffer.hpp>
#include <tt-metalium/kernel_types.hpp>
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/experimental/dispatch_context.hpp>
#include <tt-metalium/mesh_buffer.hpp>
#include <tt-metalium/mesh_coord.hpp>
#include <tt-metalium/mesh_device.hpp>
#include <tt-metalium/mesh_workload.hpp>
#include <tt-metalium/system_mesh.hpp>
#include <tt-metalium/tt_metal.hpp>

#include "impl/allocator/allocator.hpp"
#include "impl/context/metal_context.hpp"
#include "llrt/tt_cluster.hpp"
#include "tests/tt_metal/distributed/utils.hpp"
#include <umd/device/types/arch.hpp>

namespace tt::tt_metal::distributed::test {

class DispatchContextFixture : public ::testing::Test {
protected:
    void TearDown() override { experimental::DispatchContext::get().reset(); }
};

namespace {

// Why a guard test cannot run here, or nullopt if it can: manual FD switching needs Slow Dispatch,
// real hardware, and a Galaxy or Blackhole cluster.
std::optional<std::string> fd_preflight_skip_reason() {
    const auto& context = MetalContext::instance();
    if (context.rtoptions().get_fast_dispatch()) {
        return "This test can only be run with Slow Dispatch mode.";
    }
    const auto& cluster = context.get_cluster();
    if (cluster.is_mock_or_emulated()) {
        return "This test requires real hardware.";
    }
    if (!cluster.is_ubb_galaxy() && cluster.arch() != tt::ARCH::BLACKHOLE) {
        return "Manual Fast Dispatch setup is supported only on Galaxy and Blackhole clusters.";
    }
    return std::nullopt;
}

// Confirms that the chip is Blackhole, the dispatch core axis is a column (DispatchCoreAxis::COL),
// each chip is MMIO-accessible, and the compute grid has at least 13 columns and 2 rows (grid.x > 12 && grid.y > 1)
bool has_expected_dispatch_column(const MeshDevice& mesh) {
    // DispatchCoreConfig's axis is resolved only when MeshDevice::create
    // initializes the MetalContext, so this validation must happen afterwards.
    const auto& context = MetalContext::instance();
    const auto& cluster = context.get_cluster();
    const auto& dispatch_config = context.get_dispatch_core_config();
    if (cluster.arch() != tt::ARCH::BLACKHOLE || dispatch_config.get_dispatch_core_type() != DispatchCoreType::WORKER ||
        dispatch_config.get_dispatch_core_axis() != DispatchCoreAxis::COL) {
        return false;
    }
    for (ChipId chip : cluster.all_chip_ids()) {
        if (cluster.get_associated_mmio_device(chip) != chip) {
            return false;
        }
    }
    const CoreCoord grid = mesh.compute_with_storage_grid_size();
    return grid.x > 12 && grid.y > 1;
}

// Runs the default preflight and returns the refusal text, or an empty string if the session was
// allowed (in which case it is torn down again).
std::string capture_default_fd_refusal(MeshDevice* mesh) {
    try {
        experimental::DispatchContext::get().initialize_fast_dispatch(mesh);
    } catch (const std::runtime_error& error) {
        return error.what();
    }

    // Keep the process usable if a regression makes the expected refusal
    // unexpectedly succeed.
    experimental::DispatchContext::get().terminate_fast_dispatch(mesh);
    return {};
}

// Creates a BufferShardingArgs spec specifically targeting the two dispatch cores: CoreRange({12, 0}, {12, 1})
BufferShardingArgs two_dispatch_core_sharding_args(uint32_t page_size) {
    CoreRangeSet shard_grid(CoreRange({12, 0}, {12, 1}));
    ShardSpecBuffer shard_spec(
        shard_grid,
        /*shard_shape=*/{page_size, 1},
        ShardOrientation::ROW_MAJOR,
        /*page_shape=*/{page_size, 1},
        /*tensor2d_shape_in_pages=*/{2, 1});
    return BufferShardingArgs(shard_spec, TensorMemoryLayout::HEIGHT_SHARDED);
}

// Same shape as above (one page per core, HEIGHT_SHARDED, lockstep) but on an arbitrary grid.
BufferShardingArgs one_page_per_core_sharding_args(const CoreRangeSet& shard_grid, uint32_t page_size) {
    ShardSpecBuffer shard_spec(
        shard_grid,
        /*shard_shape=*/{page_size, 1},
        ShardOrientation::ROW_MAJOR,
        /*page_shape=*/{page_size, 1},
        /*tensor2d_shape_in_pages=*/{shard_grid.num_cores(), 1});
    return BufferShardingArgs(shard_spec, TensorMemoryLayout::HEIGHT_SHARDED);
}

// The allocator base is wherever a bottom-up L1 allocation lands. Read it from a throwaway buffer rather
// than a HAL constant so the tests follow the allocator, not the documentation.
DeviceAddr allocator_l1_base(MeshDevice* mesh) {
    constexpr uint32_t page_size = 4096;
    DeviceLocalBufferConfig probe_cfg{.page_size = page_size, .buffer_type = BufferType::L1, .bottom_up = true};
    ReplicatedBufferConfig probe_global{.size = page_size};
    auto probe = MeshBuffer::create(probe_global, probe_cfg, mesh);
    return probe->address();
}

// One 4 KB page per core on `grid`, lockstep or per-core, filled with a counter pattern from `salt`.
struct PlantedResident {
    std::shared_ptr<MeshBuffer> buffer;
    std::vector<uint32_t> pattern;
};

PlantedResident plant_one_page_per_core(
    MeshDevice* mesh, const CoreRangeSet& grid, bool bottom_up, bool per_core, uint32_t salt) {
    constexpr uint32_t page_size = 4096;
    auto sharding_args = one_page_per_core_sharding_args(grid, page_size);
    if (per_core) {
        experimental::per_core_allocation::set_per_core_allocation(sharding_args, true);
    }
    DeviceLocalBufferConfig cfg{
        .page_size = page_size, .buffer_type = BufferType::L1, .sharding_args = sharding_args, .bottom_up = bottom_up};
    ReplicatedBufferConfig global{.size = grid.num_cores() * page_size};

    PlantedResident resident;
    resident.buffer = MeshBuffer::create(global, cfg, mesh);
    resident.pattern.resize(grid.num_cores() * page_size / sizeof(uint32_t));
    std::iota(resident.pattern.begin(), resident.pattern.end(), salt);
    EnqueueWriteMeshBuffer(mesh->mesh_command_queue(), resident.buffer, resident.pattern);
    Finish(mesh->mesh_command_queue());
    return resident;
}

// Ground truth for every "allowed" verdict: force the session past the guard, push 2 MB of DRAM traffic
// through the dispatch cores (the canary's blast radius grows under traffic), tear down.
void run_forced_session_with_traffic(MeshDevice* mesh) {
    constexpr uint32_t page_size = 4096;
    constexpr uint32_t traffic_pages = 512;
    DeviceLocalBufferConfig dram{.page_size = page_size, .buffer_type = BufferType::DRAM, .bottom_up = true};
    ReplicatedBufferConfig dram_global{.size = traffic_pages * page_size};
    std::vector<uint32_t> payload(traffic_pages * page_size / sizeof(uint32_t));
    std::iota(payload.begin(), payload.end(), 1);

    experimental::FastDispatchSetupOptions force{.allow_destructive = true};
    experimental::DispatchContext::get().initialize_fast_dispatch(mesh, force);
    auto traffic = MeshBuffer::create(dram_global, dram, mesh);
    EnqueueWriteMeshBuffer(mesh->mesh_command_queue(), traffic, payload);
    Finish(mesh->mesh_command_queue());
    experimental::DispatchContext::get().terminate_fast_dispatch(mesh);
}

void expect_resident_intact(MeshDevice* mesh, const PlantedResident& resident, const char* what) {
    for (const auto& coord : MeshCoordinateRange(mesh->shape())) {
        std::vector<uint32_t> dst;
        ReadShard(mesh->mesh_command_queue(), dst, resident.buffer, coord);
        EXPECT_EQ(dst, resident.pattern) << what << " changed at " << coord;
    }
}

// Which bytes of `expected`, written raw at `base` on `core`, changed. Reads back with no allocator involved.
struct ChangedRange {
    bool any = false;
    DeviceAddr lo = 0;
    DeviceAddr hi = 0;
    size_t words = 0;
};

ChangedRange raw_l1_diff(
    IDevice* device, const CoreCoord& core, DeviceAddr base, const std::vector<uint32_t>& expected) {
    std::vector<uint32_t> got;
    ::tt::tt_metal::detail::ReadFromDeviceL1(
        device, core, static_cast<uint32_t>(base), static_cast<uint32_t>(expected.size() * sizeof(uint32_t)), got);
    ChangedRange changed;
    for (size_t i = 0; i < expected.size() && i < got.size(); i++) {
        if (got[i] == expected[i]) {
            continue;
        }
        const DeviceAddr addr = base + i * sizeof(uint32_t);
        if (!changed.any) {
            changed.lo = addr;
            changed.any = true;
        }
        changed.hi = addr + sizeof(uint32_t);
        changed.words++;
    }
    return changed;
}

void print_changed_range(const char* core_name, const ChangedRange& changed) {
    std::cout << "[fd footprint] " << core_name << ": ";
    if (!changed.any) {
        std::cout << "unchanged";
    } else {
        std::cout << std::hex << "changed [0x" << changed.lo << ", 0x" << changed.hi << ") (" << std::dec
                  << changed.words << " words)";
    }
    std::cout << std::endl;
}

}  // namespace

// WHAT: a plain (interleaved) L1 buffer with one page on every bank, allocated bottom-up, so it has data
//       on both dispatch cores.
// WHY:  the basic case: something is on the claimed cores. The guard must say so before any firmware is
//       written, name every chip and both cores, report the kind as "interleaved", and leave slow
//       dispatch usable afterwards.
// EXPECT: refused; a DRAM write/read still works; freeing the buffer lets a later session succeed.
TEST_F(DispatchContextFixture, RefusesResidentL1OnDispatchCores) {
    if (auto reason = fd_preflight_skip_reason(); reason.has_value()) {
        GTEST_SKIP() << *reason;
    }

    const MeshShape system_shape = MetalContext::instance().get_system_mesh().shape();
    auto mesh = MeshDevice::create(MeshDeviceConfig(system_shape));
    if (!has_expected_dispatch_column(*mesh)) {
        GTEST_SKIP() << "This test expects Blackhole dispatch cores (12,0) and (12,1).";
    }

    constexpr uint32_t page_size = 4096;
    const uint32_t num_banks = mesh->allocator_impl()->get_num_banks(BufferType::L1);
    DeviceLocalBufferConfig low_l1{.page_size = page_size, .buffer_type = BufferType::L1, .bottom_up = true};
    ReplicatedBufferConfig low_l1_global{.size = num_banks * page_size};  // one page on every bank
    auto resident = MeshBuffer::create(low_l1_global, low_l1, mesh.get());

    const std::string error = capture_default_fd_refusal(mesh.get());
    ASSERT_FALSE(error.empty()) << "Expected resident L1 to block Fast Dispatch setup.";
    EXPECT_NE(error.find("tt-blaze #2019"), std::string::npos);
    EXPECT_NE(error.find("core (12,0)"), std::string::npos);
    EXPECT_NE(error.find("core (12,1)"), std::string::npos);
    EXPECT_NE(error.find("interleaved allocation"), std::string::npos) << error;
    for (IDevice* device : mesh->get_devices()) {
        EXPECT_NE(error.find("chip " + std::to_string(device->id()) + " "), std::string::npos)
            << "Missing conflict for chip " << device->id();
    }

    // A refusal must leave the original Slow Dispatch queue usable.
    DeviceLocalBufferConfig dram{.page_size = page_size, .buffer_type = BufferType::DRAM, .bottom_up = true};
    ReplicatedBufferConfig dram_global{.size = page_size};
    auto probe = MeshBuffer::create(dram_global, dram, mesh.get());
    std::vector<uint32_t> src(page_size / sizeof(uint32_t));
    std::iota(src.begin(), src.end(), 7);
    EnqueueWriteMeshBuffer(mesh->mesh_command_queue(), probe, src);
    Finish(mesh->mesh_command_queue());
    std::vector<uint32_t> dst;
    ReadShard(mesh->mesh_command_queue(), dst, probe, MeshCoordinate(0, 0));
    EXPECT_EQ(dst, src);

    // A later clean session must succeed.
    resident.reset();
    ASSERT_NO_THROW(experimental::DispatchContext::get().initialize_fast_dispatch(mesh.get()));
    ASSERT_NO_THROW(experimental::DispatchContext::get().terminate_fast_dispatch(mesh.get()));
}

// WHAT: a tensor on (12,0)/(12,1) at the bottom of L1, but the session is started with
//       allow_destructive = true.
// WHY:  the override must mean "go ahead and accept the damage". Reading the data back afterwards
//       proves the damage is real: this is the #2019 corruption reproduced on purpose.
// EXPECT: no exception, and the data read back differs from what was written.
TEST_F(DispatchContextFixture, AllowDestructiveProceedsAndOverwritesResidentL1) {
    if (auto reason = fd_preflight_skip_reason(); reason.has_value()) {
        GTEST_SKIP() << *reason;
    }

    const MeshShape system_shape = MetalContext::instance().get_system_mesh().shape();
    auto mesh = MeshDevice::create(MeshDeviceConfig(system_shape));
    if (!has_expected_dispatch_column(*mesh)) {
        GTEST_SKIP() << "This test expects Blackhole dispatch cores (12,0) and (12,1).";
    }

    constexpr uint32_t page_size = 4096;
    DeviceLocalBufferConfig low_l1{
        .page_size = page_size,
        .buffer_type = BufferType::L1,
        .sharding_args = two_dispatch_core_sharding_args(page_size),
        .bottom_up = true};
    ReplicatedBufferConfig low_l1_global{.size = 2 * page_size};
    auto resident = MeshBuffer::create(low_l1_global, low_l1, mesh.get());

    std::vector<uint32_t> src(2 * page_size / sizeof(uint32_t));
    std::iota(src.begin(), src.end(), 0x12340000);
    EnqueueWriteMeshBuffer(mesh->mesh_command_queue(), resident, src);
    Finish(mesh->mesh_command_queue());

    experimental::FastDispatchSetupOptions options{.allow_destructive = true};
    ASSERT_NO_THROW(experimental::DispatchContext::get().initialize_fast_dispatch(mesh.get(), options));
    ASSERT_NO_THROW(experimental::DispatchContext::get().terminate_fast_dispatch(mesh.get()));

    std::vector<uint32_t> dst;
    ReadShard(mesh->mesh_command_queue(), dst, resident, MeshCoordinate(0, 0));
    EXPECT_NE(dst, src) << "The destructive override did not expose the expected dispatch-core overwrite.";
}

// WHAT: a PER-CORE tensor (the router's shape) on (12,0)/(12,1) at the bottom of L1.
// WHY:  per-core allocations are recorded only in the chip's allocator, in one list per core. This is
//       the exact #2019 shape, so the guard must read that per-core list. Needs HYBRID=1.
// EXPECT: refused, with the chip ledger named and core (12,1) in the message.
TEST_F(DispatchContextFixture, RefusesPerCoreResidentL1OnDispatchCores) {
    if (auto reason = fd_preflight_skip_reason(); reason.has_value()) {
        GTEST_SKIP() << *reason;
    }
    if (!MetalContext::instance().rtoptions().get_allocator_mode_hybrid()) {
        GTEST_SKIP() << "Per-core L1 allocation requires TT_METAL_ALLOCATOR_MODE_HYBRID=1.";
    }

    const MeshShape system_shape = MetalContext::instance().get_system_mesh().shape();
    auto mesh = MeshDevice::create(MeshDeviceConfig(system_shape));
    if (!has_expected_dispatch_column(*mesh)) {
        GTEST_SKIP() << "This test expects Blackhole dispatch cores (12,0) and (12,1).";
    }

    constexpr uint32_t page_size = 4096;
    auto sharding_args = two_dispatch_core_sharding_args(page_size);
    experimental::per_core_allocation::set_per_core_allocation(sharding_args, true);
    DeviceLocalBufferConfig low_l1{
        .page_size = page_size, .buffer_type = BufferType::L1, .sharding_args = sharding_args, .bottom_up = true};
    ReplicatedBufferConfig low_l1_global{.size = 2 * page_size};
    auto resident = MeshBuffer::create(low_l1_global, low_l1, mesh.get());
    ASSERT_NE(resident, nullptr);

    const std::string error = capture_default_fd_refusal(mesh.get());
    ASSERT_FALSE(error.empty()) << "Expected per-core resident L1 to block Fast Dispatch setup.";
    EXPECT_NE(error.find("chip ledger"), std::string::npos);
    EXPECT_NE(error.find("core (12,1)"), std::string::npos);
}

// WHAT: a persistent-arena region on (12,1), booked in the MESH allocator's arena.
// WHY:  arena regions are not buffers and are not in the allocator's free lists; the guard must read
//       the arena separately or it misses them.
// EXPECT: refused, naming the mesh arena ledger and core (12,1).
TEST_F(DispatchContextFixture, RefusesPersistentArenaResidentL1OnDispatchCores) {
    if (auto reason = fd_preflight_skip_reason(); reason.has_value()) {
        GTEST_SKIP() << *reason;
    }

    const MeshShape system_shape = MetalContext::instance().get_system_mesh().shape();
    auto mesh = MeshDevice::create(MeshDeviceConfig(system_shape));
    if (!has_expected_dispatch_column(*mesh)) {
        GTEST_SKIP() << "This test expects Blackhole dispatch core (12,1).";
    }

    auto& arena = mesh->allocator_impl()->persistent_l1();
    const auto allocation = arena.allocate(CoreRangeSet(CoreRange({12, 1})), /*size=*/4096, /*alignment=*/64);

    const std::string error = capture_default_fd_refusal(mesh.get());
    arena.deallocate(allocation.id);
    ASSERT_FALSE(error.empty()) << "Expected persistent arena L1 to block Fast Dispatch setup.";
    EXPECT_NE(error.find("mesh arena ledger"), std::string::npos);
    EXPECT_NE(error.find("core (12,1)"), std::string::npos);
}

// ---------------------------------------------------------------------------------------------------
// HOW TO READ THE GUARD TESTS (this one and everything below it)
//
// Fast dispatch (FD) claims cores of the last grid column for its firmware: (12,0) and (12,1) with one
// command queue, plus (12,2) and (12,3) with two. It writes its queues into their L1 without telling
// the allocator, so anything a test puts there before the session may be overwritten. The guard's job
// is to raise BEFORE the session if an L1 allocation lives on a claimed core, and to stay quiet
// otherwise.
//
// The rule is one question per claimed core: does any allocated L1 buffer, or persistent-arena region,
// have data on this core? The address does not matter. The dispatch owner's invariant is that dispatch
// cores hold no tensors while FD is active, so a resident at the very top of L1 is refused exactly like
// one at the base. Which core a buffer has data on is decided from the buffer's shard grid (or
// distribution spec; interleaved means every bank), never from the shared free list, because a
// lockstep buffer reserves its address range on every bank while its bytes live only on its own grid
// (AllowsLockstepResidentWithNoDataOnDispatchCores is the test that tells those two apart).
//
// Many tests use the same two steps:
//   1. ground truth: force the session with allow_destructive=true, push traffic through it, read the
//      planted data back. This shows what the firmware REALLY did, independent of the guard.
//   2. verdict: run the session with default options and check whether the guard raised.
//
// Every test has exactly one fixed expectation. Nothing is reinterpreted per allocator mode.
// ---------------------------------------------------------------------------------------------------

// WHAT: put a tensor on cores (0,0)-(1,1), far from the dispatch cores, but at the very bottom of L1
//       (bottom_up = true). Lockstep allocation books that address range on EVERY core, including
//       (12,0)/(12,1), even though the data is only on the four corner cores.
// WHY:  this is the lockstep false positive. A free-list-only check sees "something in use at the
//       bottom of (12,1)" and raises: a false alarm, no data is there. The guard reads the shard
//       grid instead, finds no dispatch core in it, and lets the session through.
// EXPECT: not refused. Step 1 proves the tensor survives a forced session. Step 2 requires no raise.
//       Same result with HYBRID on or off.
TEST_F(DispatchContextFixture, AllowsLockstepResidentWithNoDataOnDispatchCores) {
    if (auto reason = fd_preflight_skip_reason(); reason.has_value()) {
        GTEST_SKIP() << *reason;
    }

    const MeshShape system_shape = MetalContext::instance().get_system_mesh().shape();
    auto mesh = MeshDevice::create(MeshDeviceConfig(system_shape));
    if (!has_expected_dispatch_column(*mesh)) {
        GTEST_SKIP() << "This test expects Blackhole dispatch cores (12,0) and (12,1).";
    }

    constexpr uint32_t page_size = 4096;
    const CoreRangeSet control_grid(CoreRange({0, 0}, {1, 1}));  // no dispatch core in here
    DeviceLocalBufferConfig low_l1{
        .page_size = page_size,
        .buffer_type = BufferType::L1,
        .sharding_args = one_page_per_core_sharding_args(control_grid, page_size),
        .bottom_up = true};
    ReplicatedBufferConfig low_l1_global{.size = control_grid.num_cores() * page_size};
    auto resident = MeshBuffer::create(low_l1_global, low_l1, mesh.get());
    ASSERT_NE(resident, nullptr);

    // The mechanism under test, stated as an assertion: bank (12,1) reports this buffer's address
    // as its lowest occupied address even though the buffer has no shard on (12,1). The free list
    // cannot tell reservation from data, which is why the guard consults the buffer's grid.
    {
        const auto& mesh_allocator = *mesh->allocator_impl();
        const uint32_t bank = mesh_allocator.get_bank_ids_from_logical_core(BufferType::L1, CoreCoord(12, 1)).at(0);
        const auto lowest = mesh_allocator.get_lowest_occupied_l1_address(bank);
        ASSERT_TRUE(lowest.has_value());
        ASSERT_EQ(*lowest, resident->address())
            << "expected the lockstep reservation to be visible on bank (12,1); the premise of this test is wrong";
    }

    std::vector<uint32_t> src(control_grid.num_cores() * page_size / sizeof(uint32_t));
    std::iota(src.begin(), src.end(), 0x20190000);
    EnqueueWriteMeshBuffer(mesh->mesh_command_queue(), resident, src);
    Finish(mesh->mesh_command_queue());

    // 1. Ground truth, independent of the guard: force the session, push DRAM
    //    traffic through the dispatch cores (the canary's blast radius grows under traffic), tear
    //    down, and read the resident back on every device. It must be untouched.
    constexpr uint32_t traffic_pages = 512;  // 2 MB per device, enough to cycle the command-data queue
    DeviceLocalBufferConfig dram{.page_size = page_size, .buffer_type = BufferType::DRAM, .bottom_up = true};
    ReplicatedBufferConfig dram_global{.size = traffic_pages * page_size};
    std::vector<uint32_t> payload(traffic_pages * page_size / sizeof(uint32_t));
    std::iota(payload.begin(), payload.end(), 1);

    experimental::FastDispatchSetupOptions force{.allow_destructive = true};
    ASSERT_NO_THROW(experimental::DispatchContext::get().initialize_fast_dispatch(mesh.get(), force));
    auto traffic = MeshBuffer::create(dram_global, dram, mesh.get());
    EnqueueWriteMeshBuffer(mesh->mesh_command_queue(), traffic, payload);
    Finish(mesh->mesh_command_queue());
    ASSERT_NO_THROW(experimental::DispatchContext::get().terminate_fast_dispatch(mesh.get()));

    for (const auto& coord : MeshCoordinateRange(mesh->shape())) {
        std::vector<uint32_t> dst;
        ReadShard(mesh->mesh_command_queue(), dst, resident, coord);
        ASSERT_EQ(dst, src) << "resident with no data on a dispatch core was corrupted at " << coord
                            << "; the premise of this test is wrong, re-check the footprint model";
    }

    // 2. The verdict. A resident the firmware cannot touch must not block the session.
    const std::string error = capture_default_fd_refusal(mesh.get());
    EXPECT_TRUE(error.empty()) << "false positive: the preflight refused a resident that has no data on any "
                                  "dispatch core (lockstep reservation mistaken for data):\n"
                               << error;
}

// ---------------------------------------------------------------------------------------------------
// The tests from here down cover the guard's ledger coverage, its mesh-tree walk, and its known
// limits. Reading guide: see the block above AllowsLockstepResidentWithNoDataOnDispatchCores.
// ---------------------------------------------------------------------------------------------------

// WHAT: put a per-core tensor on the REST of the dispatch column, (12,2) down to (12,9) including the
//       sender core, at the bottom of L1. Nothing on (12,0) or (12,1).
// WHY:  the question "if I put stuff on cores other than the two fast dispatch needs, does the guard
//       stay quiet?" Per-core allocation books L1 only on those cores, so the guard sees an empty
//       (12,0)/(12,1) and must allow.
// EXPECT: not refused, and the tensor is intact after a forced session with traffic. Needs HYBRID=1
//       because per-core allocation does not exist without it.
TEST_F(DispatchContextFixture, PerCoreResidentOnColumn12NonDispatchCoresIsAllowed) {
    if (auto reason = fd_preflight_skip_reason(); reason.has_value()) {
        GTEST_SKIP() << *reason;
    }
    if (!MetalContext::instance().rtoptions().get_allocator_mode_hybrid()) {
        GTEST_SKIP() << "Per-core L1 allocation requires TT_METAL_ALLOCATOR_MODE_HYBRID=1.";
    }

    const MeshShape system_shape = MetalContext::instance().get_system_mesh().shape();
    auto mesh = MeshDevice::create(MeshDeviceConfig(system_shape));
    if (!has_expected_dispatch_column(*mesh)) {
        GTEST_SKIP() << "This test expects Blackhole dispatch cores (12,0) and (12,1).";
    }

    const CoreRangeSet rest_of_column(CoreRange({12, 2}, {12, 9}));
    auto resident =
        plant_one_page_per_core(mesh.get(), rest_of_column, /*bottom_up=*/true, /*per_core=*/true, 0x20190200);

    run_forced_session_with_traffic(mesh.get());
    expect_resident_intact(mesh.get(), resident, "per-core resident on (12,2)-(12,9)");

    const std::string error = capture_default_fd_refusal(mesh.get());
    EXPECT_TRUE(error.empty()) << "refused a per-core resident that has no data on (12,0)/(12,1):\n" << error;
}

// WHAT: same cores, (12,2)-(12,9), but a normal lockstep tensor placed at the TOP of L1 (the default).
// WHY:  lockstep books the address range on every bank, but the grid has no claimed core in it, so the
//       guard has nothing to complain about.
// EXPECT: not refused, tensor intact, HYBRID on or off. (The same tensor at the BOTTOM of L1 is the
//       lockstep false-positive case, AllowsLockstepResidentWithNoDataOnDispatchCores.)
TEST_F(DispatchContextFixture, LockstepResidentOnColumn12NonDispatchCoresAtTopIsAllowed) {
    if (auto reason = fd_preflight_skip_reason(); reason.has_value()) {
        GTEST_SKIP() << *reason;
    }

    const MeshShape system_shape = MetalContext::instance().get_system_mesh().shape();
    auto mesh = MeshDevice::create(MeshDeviceConfig(system_shape));
    if (!has_expected_dispatch_column(*mesh)) {
        GTEST_SKIP() << "This test expects Blackhole dispatch cores (12,0) and (12,1).";
    }

    const CoreRangeSet rest_of_column(CoreRange({12, 2}, {12, 9}));
    auto resident =
        plant_one_page_per_core(mesh.get(), rest_of_column, /*bottom_up=*/false, /*per_core=*/false, 0x20190300);

    run_forced_session_with_traffic(mesh.get());
    expect_resident_intact(mesh.get(), resident, "lockstep resident on (12,2)-(12,9) at the top of L1");

    const std::string error = capture_default_fd_refusal(mesh.get());
    EXPECT_TRUE(error.empty()) << "refused a lockstep resident with no data on (12,0)/(12,1):\n" << error;
}

// WHAT: reserve a small "persistent arena" region (the allocator's per-core reservation mechanism, used
//       by PrefetcherPipe) on (12,2), (12,3) and (12,9). Arena regions are tracked per core, so this
//       really is only on those cores.
// WHY:  same question as above, for the third kind of allocation. Also the one-queue partner of
//       TwoCqSessionChecksSecondDispatchPair: with one command queue, (12,2)/(12,3) are ordinary cores.
// EXPECT: not refused.
TEST_F(DispatchContextFixture, ArenaResidentOnColumn12NonDispatchCoresIsAllowed) {
    if (auto reason = fd_preflight_skip_reason(); reason.has_value()) {
        GTEST_SKIP() << *reason;
    }

    const MeshShape system_shape = MetalContext::instance().get_system_mesh().shape();
    auto mesh = MeshDevice::create(MeshDeviceConfig(system_shape));
    if (!has_expected_dispatch_column(*mesh)) {
        GTEST_SKIP() << "This test expects Blackhole dispatch cores (12,0) and (12,1).";
    }

    auto& arena = mesh->allocator_impl()->persistent_l1();
    const auto allocation = arena.allocate(
        CoreRangeSet(std::vector<CoreRange>{CoreRange({12, 2}), CoreRange({12, 3}), CoreRange({12, 9})}),
        /*size=*/4096,
        /*alignment=*/64);
    const std::string error = capture_default_fd_refusal(mesh.get());
    arena.deallocate(allocation.id);
    EXPECT_TRUE(error.empty()) << "refused an arena resident that is not on (12,0)/(12,1):\n" << error;
}

// WHAT: allocate the tensor on the ROOT mesh, on (12,0)/(12,1) at the bottom of L1, then start the
//       session from a 1x1 SUBMESH of that root.
// WHY:  blaze does exactly this: tensors can live in the root's allocator while two_phase_upload is
//       called with a stage submesh. Each mesh object has its own allocator, so the guard must walk up
//       to the root and look there too. With HYBRID off, the root's allocator is the ONLY place this
//       tensor is recorded.
// EXPECT: refused, naming both dispatch cores and the submesh's chip.
TEST_F(DispatchContextFixture, ResidentOnRootRefusedWhenSessionEnteredFromSubmesh) {
    if (auto reason = fd_preflight_skip_reason(); reason.has_value()) {
        GTEST_SKIP() << *reason;
    }

    const MeshShape system_shape = MetalContext::instance().get_system_mesh().shape();
    auto mesh = MeshDevice::create(MeshDeviceConfig(system_shape));
    if (!has_expected_dispatch_column(*mesh)) {
        GTEST_SKIP() << "This test expects Blackhole dispatch cores (12,0) and (12,1).";
    }
    auto submesh = mesh->create_submesh(MeshShape(1, 1));

    constexpr uint32_t page_size = 4096;
    DeviceLocalBufferConfig low_l1{
        .page_size = page_size,
        .buffer_type = BufferType::L1,
        .sharding_args = two_dispatch_core_sharding_args(page_size),
        .bottom_up = true};
    ReplicatedBufferConfig low_l1_global{.size = 2 * page_size};
    auto resident = MeshBuffer::create(low_l1_global, low_l1, mesh.get());  // owned by the ROOT

    const std::string error = capture_default_fd_refusal(submesh.get());  // session from the SUBMESH
    ASSERT_FALSE(error.empty()) << "a root-owned resident on the dispatch cores was not seen from a submesh session";
    EXPECT_NE(error.find("core (12,0)"), std::string::npos);
    EXPECT_NE(error.find("core (12,1)"), std::string::npos);
    const ChipId chip = submesh->get_devices()[0]->id();
    EXPECT_NE(error.find("chip " + std::to_string(chip) + " "), std::string::npos) << error;
}

// WHAT: the reverse: tensor allocated on a 1x1 submesh, session started from the root.
// WHY:  the guard must walk DOWN into every submesh as well. Also checks the message is per chip: the
//       one chip holding the tensor is named, chips that hold nothing are not.
// EXPECT: refused; only the submesh's chip appears in the message.
TEST_F(DispatchContextFixture, ResidentOnSubmeshRefusedWhenSessionEnteredFromRoot) {
    if (auto reason = fd_preflight_skip_reason(); reason.has_value()) {
        GTEST_SKIP() << *reason;
    }

    const MeshShape system_shape = MetalContext::instance().get_system_mesh().shape();
    auto mesh = MeshDevice::create(MeshDeviceConfig(system_shape));
    if (!has_expected_dispatch_column(*mesh)) {
        GTEST_SKIP() << "This test expects Blackhole dispatch cores (12,0) and (12,1).";
    }
    auto submesh = mesh->create_submesh(MeshShape(1, 1));

    constexpr uint32_t page_size = 4096;
    DeviceLocalBufferConfig low_l1{
        .page_size = page_size,
        .buffer_type = BufferType::L1,
        .sharding_args = two_dispatch_core_sharding_args(page_size),
        .bottom_up = true};
    ReplicatedBufferConfig low_l1_global{.size = 2 * page_size};
    auto resident = MeshBuffer::create(low_l1_global, low_l1, submesh.get());  // owned by the SUBMESH

    const std::string error = capture_default_fd_refusal(mesh.get());  // session from the ROOT
    ASSERT_FALSE(error.empty()) << "a submesh-owned resident on the dispatch cores was not seen from a root session";
    EXPECT_NE(error.find("core (12,0)"), std::string::npos);
    EXPECT_NE(error.find("core (12,1)"), std::string::npos);

    // The report must be per chip: the chip holding the resident is named, no other chip is.
    const ChipId resident_chip = submesh->get_devices()[0]->id();
    EXPECT_NE(error.find("chip " + std::to_string(resident_chip) + " "), std::string::npos) << error;
    for (IDevice* device : mesh->get_devices()) {
        if (device->id() != resident_chip) {
            EXPECT_EQ(error.find("chip " + std::to_string(device->id()) + " "), std::string::npos)
                << "chip " << device->id() << " holds no resident but was reported:\n"
                << error;
        }
    }
}

// WHAT: two 1x1 submeshes on different chips. Tensor on submesh B, session started from submesh A.
// WHY:  the session flashes fast-dispatch firmware onto EVERY active chip, not just the caller's, so a
//       tensor on another stage's chip is just as much at risk. This is blaze's multi-stage layout
//       (pipeline_builder/submesh_partition.py).
// EXPECT: refused, naming B's chip and not A's.
TEST_F(DispatchContextFixture, ResidentOnSiblingSubmeshRefused) {
    if (auto reason = fd_preflight_skip_reason(); reason.has_value()) {
        GTEST_SKIP() << *reason;
    }

    const MeshShape system_shape = MetalContext::instance().get_system_mesh().shape();
    auto mesh = MeshDevice::create(MeshDeviceConfig(system_shape));
    if (!has_expected_dispatch_column(*mesh)) {
        GTEST_SKIP() << "This test expects Blackhole dispatch cores (12,0) and (12,1).";
    }
    if (mesh->num_devices() < 2) {
        GTEST_SKIP() << "This test needs at least two devices.";
    }
    auto siblings = mesh->create_submeshes(MeshShape(1, 1));
    ASSERT_GE(siblings.size(), 2u);
    auto& session_mesh = siblings[0];
    auto& resident_mesh = siblings[1];

    constexpr uint32_t page_size = 4096;
    DeviceLocalBufferConfig low_l1{
        .page_size = page_size,
        .buffer_type = BufferType::L1,
        .sharding_args = two_dispatch_core_sharding_args(page_size),
        .bottom_up = true};
    ReplicatedBufferConfig low_l1_global{.size = 2 * page_size};
    auto resident = MeshBuffer::create(low_l1_global, low_l1, resident_mesh.get());

    const std::string error = capture_default_fd_refusal(session_mesh.get());
    ASSERT_FALSE(error.empty()) << "a resident owned by a sibling submesh was not seen";
    const ChipId resident_chip = resident_mesh->get_devices()[0]->id();
    const ChipId session_chip = session_mesh->get_devices()[0]->id();
    EXPECT_NE(error.find("chip " + std::to_string(resident_chip) + " "), std::string::npos) << error;
    EXPECT_EQ(error.find("chip " + std::to_string(session_chip) + " "), std::string::npos)
        << "the session's own chip holds nothing but was reported:\n"
        << error;
}

// WHAT: a submesh of a submesh (root -> 2-chip middle -> 1-chip leaf). Tensor on the leaf, session from
//       the root.
// WHY:  the pipeline builder (pipeline_builder/fork.py) creates nested submeshes, so the guard's walk
//       must recurse all the way down.
// EXPECT: refused, naming the leaf's chip.
TEST_F(DispatchContextFixture, ResidentOnNestedSubmeshRefused) {
    if (auto reason = fd_preflight_skip_reason(); reason.has_value()) {
        GTEST_SKIP() << *reason;
    }

    const MeshShape system_shape = MetalContext::instance().get_system_mesh().shape();
    auto mesh = MeshDevice::create(MeshDeviceConfig(system_shape));
    if (!has_expected_dispatch_column(*mesh)) {
        GTEST_SKIP() << "This test expects Blackhole dispatch cores (12,0) and (12,1).";
    }
    if (mesh->num_devices() < 2) {
        GTEST_SKIP() << "This test needs at least two devices.";
    }
    const MeshShape middle_shape = mesh->num_cols() >= 2 ? MeshShape(1, 2) : MeshShape(2, 1);
    auto middle = mesh->create_submesh(middle_shape);
    auto leaf = middle->create_submesh(MeshShape(1, 1));

    constexpr uint32_t page_size = 4096;
    DeviceLocalBufferConfig low_l1{
        .page_size = page_size,
        .buffer_type = BufferType::L1,
        .sharding_args = two_dispatch_core_sharding_args(page_size),
        .bottom_up = true};
    ReplicatedBufferConfig low_l1_global{.size = 2 * page_size};
    auto resident = MeshBuffer::create(low_l1_global, low_l1, leaf.get());

    const std::string error = capture_default_fd_refusal(mesh.get());
    ASSERT_FALSE(error.empty()) << "a resident owned by a submesh-of-a-submesh was not seen from a root session";
    const ChipId leaf_chip = leaf->get_devices()[0]->id();
    EXPECT_NE(error.find("chip " + std::to_string(leaf_chip) + " "), std::string::npos) << error;
}

// WHAT: an arena region on the PREFETCH core (12,0), booked in the CHIP's allocator rather than the
//       mesh's. (The existing arena test covers the mesh allocator and the dispatcher core (12,1).)
// WHY:  arena regions are not Buffer objects, so a walk of get_allocated_buffers() alone would let this
//       through; the guard must read the arena as well. Checks the core and chip in the message, not
//       the ledger wording.
// EXPECT: refused with a line for chip 0 core (12,0) and no line for core (12,1).
TEST_F(DispatchContextFixture, ChipArenaResidentOnPrefetchCoreRefused) {
    if (auto reason = fd_preflight_skip_reason(); reason.has_value()) {
        GTEST_SKIP() << *reason;
    }

    const MeshShape system_shape = MetalContext::instance().get_system_mesh().shape();
    auto mesh = MeshDevice::create(MeshDeviceConfig(system_shape));
    if (!has_expected_dispatch_column(*mesh)) {
        GTEST_SKIP() << "This test expects Blackhole dispatch cores (12,0) and (12,1).";
    }

    IDevice* device = mesh->get_devices()[0];
    auto& arena = device->allocator_impl()->persistent_l1();
    const auto allocation = arena.allocate(CoreRangeSet(CoreRange({12, 0})), /*size=*/4096, /*alignment=*/64);
    const std::string error = capture_default_fd_refusal(mesh.get());
    arena.deallocate(allocation.id);
    ASSERT_FALSE(error.empty()) << "a chip-arena resident on the prefetch core was not refused";
    EXPECT_NE(error.find("chip " + std::to_string(device->id()) + " dispatch core (12,0)"), std::string::npos) << error;
    EXPECT_EQ(error.find("core (12,1)"), std::string::npos) << "nothing was planted on (12,1):\n" << error;
}

// WHAT: a tensor sharded the "ND" way (BufferDistributionSpec) with one page on (12,0) and one on
//       (12,1), at the bottom of L1.
// WHY:  ND-sharded buffers have no classic shard spec; their core list lives in
//       buffer_distribution_spec(). A walk that only looks at shard_spec() would miss this shape
//       (or crash on it), so the guard must attribute cores from the distribution spec too.
// EXPECT: refused. Then a forced session must actually overwrite the page on (12,1), proving the
//       refusal protected real data. Written and read back raw, so the ND read path is not involved.
TEST_F(DispatchContextFixture, NdShardedResidentWithDataOnDispatchCoresRefused) {
    if (auto reason = fd_preflight_skip_reason(); reason.has_value()) {
        GTEST_SKIP() << *reason;
    }

    const MeshShape system_shape = MetalContext::instance().get_system_mesh().shape();
    auto mesh = MeshDevice::create(MeshDeviceConfig(system_shape));
    if (!has_expected_dispatch_column(*mesh)) {
        GTEST_SKIP() << "This test expects Blackhole dispatch cores (12,0) and (12,1).";
    }

    constexpr uint32_t page_size = 4096;
    const CoreRangeSet dispatch_pair(CoreRange({12, 0}, {12, 1}));
    BufferDistributionSpec nd_spec(Shape({2, 1}), Shape({1, 1}), dispatch_pair, ShardOrientation::ROW_MAJOR);
    DeviceLocalBufferConfig low_l1{
        .page_size = page_size,
        .buffer_type = BufferType::L1,
        .sharding_args = BufferShardingArgs(nd_spec),
        .bottom_up = true};
    ReplicatedBufferConfig low_l1_global{.size = 2 * page_size};
    auto resident = MeshBuffer::create(low_l1_global, low_l1, mesh.get());
    ASSERT_NE(resident, nullptr);

    const std::string error = capture_default_fd_refusal(mesh.get());
    ASSERT_FALSE(error.empty()) << "an ND-sharded resident with shards on both dispatch cores was not refused";
    EXPECT_NE(error.find("core (12,0)"), std::string::npos);
    EXPECT_NE(error.find("core (12,1)"), std::string::npos);

    // Ground truth: the page at the resident's address on (12,1) of chip 0 really is inside the footprint.
    IDevice* device = mesh->get_devices()[0];
    std::vector<uint32_t> pattern(page_size / sizeof(uint32_t));
    std::iota(pattern.begin(), pattern.end(), 0x20190400);
    ::tt::tt_metal::detail::WriteToDeviceL1(
        device, CoreCoord(12, 1), static_cast<uint32_t>(resident->address()), pattern);
    run_forced_session_with_traffic(mesh.get());
    const ChangedRange changed = raw_l1_diff(device, CoreCoord(12, 1), resident->address(), pattern);
    EXPECT_TRUE(changed.any) << "the forced session left the ND-sharded page on (12,1) untouched; the refusal "
                                "above would be conservative rather than protective";
}

// WHAT: an interleaved L1 buffer with one page on every bank, allocated top-down (the default), so its
//       pages sit at the top of every bank, far above where the firmware writes.
// WHY:  the address does not matter. Dispatch cores may not hold L1 allocations while fast dispatch is
//       active, and an interleaved buffer has a page on each of them. This is the placement an
//       address-window rule would allow and the dispatch owner's rule refuses.
// EXPECT: refused, reported as an "interleaved allocation" on both dispatch cores.
TEST_F(DispatchContextFixture, InterleavedL1ResidentAtTopRefused) {
    if (auto reason = fd_preflight_skip_reason(); reason.has_value()) {
        GTEST_SKIP() << *reason;
    }

    const MeshShape system_shape = MetalContext::instance().get_system_mesh().shape();
    auto mesh = MeshDevice::create(MeshDeviceConfig(system_shape));
    if (!has_expected_dispatch_column(*mesh)) {
        GTEST_SKIP() << "This test expects Blackhole dispatch cores (12,0) and (12,1).";
    }

    constexpr uint32_t page_size = 4096;
    const uint32_t num_banks = mesh->allocator_impl()->get_num_banks(BufferType::L1);
    DeviceLocalBufferConfig top_l1{.page_size = page_size, .buffer_type = BufferType::L1, .bottom_up = false};
    ReplicatedBufferConfig top_l1_global{.size = num_banks * page_size};
    auto resident = MeshBuffer::create(top_l1_global, top_l1, mesh.get());
    ASSERT_NE(resident, nullptr);

    const std::string error = capture_default_fd_refusal(mesh.get());
    ASSERT_FALSE(error.empty()) << "an interleaved L1 resident at the top of L1 was not refused";
    EXPECT_NE(error.find("core (12,0)"), std::string::npos) << error;
    EXPECT_NE(error.find("core (12,1)"), std::string::npos) << error;
    EXPECT_NE(error.find("interleaved allocation"), std::string::npos) << error;
}

// WHAT: open the mesh with TWO command queues instead of one. The second queue gets its own prefetcher
//       and dispatcher, expected on (12,2)/(12,3). Reserve arena regions there.
// WHY:  the guard asks the dispatch core manager for every core it has assigned on the chip, so the
//       second queue's pair is covered without anyone listing roles or queues by hand.
//       ArenaResidentOnColumn12NonDispatchCoresIsAllowed shows the same cores are fine with one queue.
// EXPECT: refused, naming (12,2) and (12,3). Skips if a two-queue slow-dispatch mesh cannot be opened
//       here.
TEST_F(DispatchContextFixture, TwoCqSessionChecksSecondDispatchPair) {
    if (auto reason = fd_preflight_skip_reason(); reason.has_value()) {
        GTEST_SKIP() << *reason;
    }

    const MeshShape system_shape = MetalContext::instance().get_system_mesh().shape();
    std::shared_ptr<MeshDevice> mesh;
    try {
        mesh = MeshDevice::create(
            MeshDeviceConfig(system_shape), DEFAULT_L1_SMALL_SIZE, DEFAULT_TRACE_REGION_SIZE, /*num_command_queues=*/2);
    } catch (const std::exception& exception) {
        GTEST_SKIP() << "Could not open a two-CQ slow-dispatch mesh here: " << exception.what();
    }
    if (!has_expected_dispatch_column(*mesh)) {
        GTEST_SKIP() << "This test expects Blackhole dispatch cores (12,0) and (12,1).";
    }
    ASSERT_EQ(mesh->num_hw_cqs(), 2);

    auto& arena = mesh->allocator_impl()->persistent_l1();
    const auto allocation = arena.allocate(
        CoreRangeSet(std::vector<CoreRange>{CoreRange({12, 2}), CoreRange({12, 3})}), /*size=*/4096, /*alignment=*/64);
    const std::string error = capture_default_fd_refusal(mesh.get());
    arena.deallocate(allocation.id);
    ASSERT_FALSE(error.empty()) << "arena residents on the second command queue's dispatch cores were not refused";
    EXPECT_NE(error.find("core (12,2)"), std::string::npos) << error;
    EXPECT_NE(error.find("core (12,3)"), std::string::npos) << error;
}

// WHAT: two ways to put bytes in L1 WITHOUT telling the allocator: (a) a buffer created at a fixed
//       address, (b) a raw write (WriteToDeviceL1). Do both on the dispatch cores, plus the raw write on
//       (12,5) as a control. Then run the session with default options.
// WHY:  this is the honest limit of any allocator-based guard: it can only see what went through the
//       allocator. Blaze does neither of these today, but both exist in tt-metal.
//       Writing the limit down as a test means we notice if it ever changes.
// EXPECT: NOT refused (the guard is blind), bytes on (12,0) and (12,1) are destroyed, (12,5) is
//       untouched. The test prints the measured footprint per core, so what the firmware really writes
//       is recorded on real hardware every run.
TEST_F(DispatchContextFixture, LedgerGuardIsBlindToFixedAddressAndRawWrites) {
    if (auto reason = fd_preflight_skip_reason(); reason.has_value()) {
        GTEST_SKIP() << *reason;
    }

    const MeshShape system_shape = MetalContext::instance().get_system_mesh().shape();
    auto mesh = MeshDevice::create(MeshDeviceConfig(system_shape));
    if (!has_expected_dispatch_column(*mesh)) {
        GTEST_SKIP() << "This test expects Blackhole dispatch cores (12,0) and (12,1).";
    }

    IDevice* device = mesh->get_devices()[0];
    const DeviceAddr base = allocator_l1_base(mesh.get());
    const DeviceAddr l1_top = mesh->l1_size_per_core();

    // (a) A fixed-address MeshBuffer on (12,1) at the base appears in no ledger.
    constexpr uint32_t page_size = 4096;
    DeviceLocalBufferConfig fixed_cfg{
        .page_size = page_size,
        .buffer_type = BufferType::L1,
        .sharding_args = one_page_per_core_sharding_args(CoreRangeSet(CoreRange({12, 1})), page_size),
        .bottom_up = true};
    ReplicatedBufferConfig fixed_global{.size = page_size};
    auto fixed = MeshBuffer::create(fixed_global, fixed_cfg, mesh.get(), /*address=*/base);
    ASSERT_NE(fixed, nullptr);
    {
        const auto& mesh_allocator = *mesh->allocator_impl();
        const auto& chip_allocator = *device->allocator_impl();
        const uint32_t mesh_bank =
            mesh_allocator.get_bank_ids_from_logical_core(BufferType::L1, CoreCoord(12, 1)).at(0);
        const uint32_t chip_bank =
            chip_allocator.get_bank_ids_from_logical_core(BufferType::L1, CoreCoord(12, 1)).at(0);
        const auto mesh_lowest = mesh_allocator.get_lowest_occupied_l1_address(mesh_bank);
        const auto chip_lowest = chip_allocator.get_lowest_occupied_l1_address(chip_bank);
        EXPECT_TRUE(!mesh_lowest.has_value() || *mesh_lowest > base)
            << "the fixed-address buffer is visible in the mesh ledger; this blind spot closed, update the test";
        EXPECT_TRUE(!chip_lowest.has_value() || *chip_lowest > base)
            << "the fixed-address buffer is visible in the chip ledger; this blind spot closed, update the test";
    }

    // (b) Raw-write a counter pattern over the first MiB above the base on both dispatch cores and on a
    //     control core of the same column. That covers the footprint measured on every pin so far
    //     ((12,1) to about 0x9C000, (12,0) to about 0x5B000) with room to spare.
    const DeviceAddr span_end = std::min<DeviceAddr>(base + 0x100000, l1_top);
    std::vector<uint32_t> pattern(static_cast<size_t>((span_end - base) / sizeof(uint32_t)));
    std::iota(pattern.begin(), pattern.end(), 0x5A000000);
    const CoreCoord prefetch_core(12, 0);
    const CoreCoord dispatch_core(12, 1);
    const CoreCoord control_core(12, 5);
    for (const CoreCoord& core : {prefetch_core, dispatch_core, control_core}) {
        ::tt::tt_metal::detail::WriteToDeviceL1(device, core, static_cast<uint32_t>(base), pattern);
    }

    // (c) The default session is NOT refused: none of the above is in a ledger.
    ASSERT_NO_THROW(experimental::DispatchContext::get().initialize_fast_dispatch(mesh.get()));
    {
        constexpr uint32_t traffic_pages = 512;
        DeviceLocalBufferConfig dram{.page_size = page_size, .buffer_type = BufferType::DRAM, .bottom_up = true};
        ReplicatedBufferConfig dram_global{.size = traffic_pages * page_size};
        std::vector<uint32_t> payload(traffic_pages * page_size / sizeof(uint32_t));
        std::iota(payload.begin(), payload.end(), 1);
        auto traffic = MeshBuffer::create(dram_global, dram, mesh.get());
        EnqueueWriteMeshBuffer(mesh->mesh_command_queue(), traffic, payload);
        Finish(mesh->mesh_command_queue());
    }
    ASSERT_NO_THROW(experimental::DispatchContext::get().terminate_fast_dispatch(mesh.get()));

    // (d) Footprint map: what the firmware actually wrote on each core.
    const ChangedRange on_prefetch = raw_l1_diff(device, prefetch_core, base, pattern);
    const ChangedRange on_dispatch = raw_l1_diff(device, dispatch_core, base, pattern);
    const ChangedRange on_control = raw_l1_diff(device, control_core, base, pattern);
    print_changed_range("(12,0) prefetch", on_prefetch);
    print_changed_range("(12,1) dispatch", on_dispatch);
    print_changed_range("(12,5) control", on_control);

    EXPECT_FALSE(on_control.any) << "the session changed L1 on (12,5), a core it should never touch";
    EXPECT_TRUE(on_dispatch.any) << "(12,1) untouched: the dispatcher did not write its queues";
    EXPECT_TRUE(on_prefetch.any) << "(12,0) untouched under 2 MB of traffic: the prefetcher did not write its queues";
    EXPECT_LE(on_dispatch.hi, span_end) << "the firmware wrote past the end of the raw pattern on (12,1)";
    EXPECT_LE(on_prefetch.hi, span_end) << "the firmware wrote past the end of the raw pattern on (12,0)";
}

// WHAT: a lockstep tensor WITH data on (12,0)/(12,1), at the top of L1. Force the session and check it
//       survives.
// WHY:  characterisation, not a verdict. The guard refuses this resident regardless (dispatch owns the
//       cores; the verdict for this placement is RefusesPerCoreResidentL1OnDispatchCoresAtTop and
//       InterleavedL1ResidentAtTopRefused). This test records the physical fact that on this pin the
//       firmware footprint does not reach the top of L1, so a change in that footprint is noticed.
// EXPECT: intact after a forced session with traffic.
TEST_F(DispatchContextFixture, FirmwareLeavesTopOfL1OnDispatchCoresIntact) {
    if (auto reason = fd_preflight_skip_reason(); reason.has_value()) {
        GTEST_SKIP() << *reason;
    }

    const MeshShape system_shape = MetalContext::instance().get_system_mesh().shape();
    auto mesh = MeshDevice::create(MeshDeviceConfig(system_shape));
    if (!has_expected_dispatch_column(*mesh)) {
        GTEST_SKIP() << "This test expects Blackhole dispatch cores (12,0) and (12,1).";
    }

    auto resident = plant_one_page_per_core(
        mesh.get(), CoreRangeSet(CoreRange({12, 0}, {12, 1})), /*bottom_up=*/false, /*per_core=*/false, 0x20190500);

    run_forced_session_with_traffic(mesh.get());
    expect_resident_intact(mesh.get(), resident, "resident at the top of L1 on (12,0)/(12,1)");
}

// WHAT: a PER-CORE tensor on (12,0)/(12,1) at the top of L1, far above where the firmware writes.
// WHY:  this is the DSv3/K2.6 shape: per-core gate_mm weights whose grid covers (12,0)-(12,7), resident
//       while the provider opens a second session for the hot/cold experts. Dispatch cores may not hold
//       L1 allocations while fast dispatch is active, at any address, so the guard refuses and those
//       weights have to move off the claimed cores (tt-blaze follow-up). Needs HYBRID=1.
// EXPECT: refused, naming the chip ledger and both cores.
TEST_F(DispatchContextFixture, RefusesPerCoreResidentL1OnDispatchCoresAtTop) {
    if (auto reason = fd_preflight_skip_reason(); reason.has_value()) {
        GTEST_SKIP() << *reason;
    }
    if (!MetalContext::instance().rtoptions().get_allocator_mode_hybrid()) {
        GTEST_SKIP() << "Per-core L1 allocation requires TT_METAL_ALLOCATOR_MODE_HYBRID=1.";
    }

    const MeshShape system_shape = MetalContext::instance().get_system_mesh().shape();
    auto mesh = MeshDevice::create(MeshDeviceConfig(system_shape));
    if (!has_expected_dispatch_column(*mesh)) {
        GTEST_SKIP() << "This test expects Blackhole dispatch cores (12,0) and (12,1).";
    }

    const CoreRangeSet dispatch_cores(CoreRange({12, 0}, {12, 1}));
    auto resident =
        plant_one_page_per_core(mesh.get(), dispatch_cores, /*bottom_up=*/false, /*per_core=*/true, 0x20190700);
    ASSERT_NE(resident.buffer, nullptr);

    const std::string error = capture_default_fd_refusal(mesh.get());
    ASSERT_FALSE(error.empty()) << "a per-core resident at the top of L1 on (12,0)/(12,1) was not refused";
    EXPECT_NE(error.find("chip ledger"), std::string::npos) << error;
    EXPECT_NE(error.find("core (12,0)"), std::string::npos) << error;
    EXPECT_NE(error.find("core (12,1)"), std::string::npos) << error;
}

// WHAT: open a single chip as a "unit mesh" (what the Python CreateDevice path does) and start a session from it,
//       with nothing planted.
// WHY:  regression. An earlier version of the guard threw "SubDeviceManagerTracker is not initialized"
//       here and broke three upstream ServiceCore tests. The parent mesh that create_unit_meshes builds
//       is never initialized, so it has no allocator; the guard must skip it instead of asking it
//       questions, while still walking its initialized unit submeshes.
// EXPECT: no exception.
TEST_F(DispatchContextFixture, UnitMeshSessionDoesNotThrowTrackerError) {
    if (auto reason = fd_preflight_skip_reason(); reason.has_value()) {
        GTEST_SKIP() << *reason;
    }

    const auto chip_ids = MetalContext::instance().get_cluster().all_chip_ids();
    ASSERT_FALSE(chip_ids.empty());
    std::map<int, std::shared_ptr<MeshDevice>> unit_meshes;
    try {
        unit_meshes = MeshDevice::create_unit_meshes(std::vector<int>{static_cast<int>(*chip_ids.begin())});
    } catch (const std::exception& exception) {
        GTEST_SKIP() << "Could not open a unit mesh here: " << exception.what();
    }
    ASSERT_EQ(unit_meshes.size(), 1u);
    std::shared_ptr<MeshDevice> unit = unit_meshes.begin()->second;
    ASSERT_NE(unit->get_parent_mesh(), nullptr);
    // Documents the cause: the parent view exists and has local devices, but was never initialized.
    EXPECT_FALSE(unit->get_parent_mesh()->is_initialized());
    EXPECT_FALSE(unit->get_parent_mesh()->get_view().get_devices().empty());

    std::string error;
    try {
        experimental::DispatchContext::get().initialize_fast_dispatch(unit.get());
    } catch (const std::runtime_error& exception) {
        error = exception.what();
    }
    if (error.empty()) {
        experimental::DispatchContext::get().terminate_fast_dispatch(unit.get());
    }
    EXPECT_TRUE(error.empty()) << "fast-dispatch setup from a unit mesh threw:\n" << error;
}

TEST_F(DispatchContextFixture, TestWritesAndWorkloads) {
    // Test using DispatchContext to turn FD on and off during runtime.
    const auto& rt_options = MetalContext::instance().rtoptions();
    if (rt_options.get_fast_dispatch()) {
        GTEST_SKIP() << "This test can only be run with Slow Dispatch mode.";
    }
    if (MetalContext::instance().get_cluster().is_mock_or_emulated()) {
        GTEST_SKIP() << "Mock/emulated devices cannot validate real data movement; see "
                        "MockDeviceFdSdToggleIsNoOp for the mock-specific no-op behavior.";
    }
    const MeshShape system_shape = MetalContext::instance().get_system_mesh().shape();
    auto mesh_device_ = MeshDevice::create(MeshDeviceConfig(system_shape));

    // Terminating without initializing should throw
    EXPECT_THROW(experimental::DispatchContext::get().terminate_fast_dispatch(mesh_device_.get()), std::runtime_error);

    uint32_t single_tile_size = ::tt::tile_size(DataFormat::UInt32);

    DeviceLocalBufferConfig per_device_buffer_config{
        .page_size = single_tile_size, .buffer_type = BufferType::DRAM, .bottom_up = true};

    const uint32_t tiles_per_device = 512;
    const uint32_t bytes_per_device = tiles_per_device * single_tile_size;
    const uint32_t num_programs = 5;

    ReplicatedBufferConfig global_buffer_config{.size = bytes_per_device};
    auto mesh_buffer = MeshBuffer::create(global_buffer_config, per_device_buffer_config, mesh_device_.get());

    std::vector<uint32_t> src_vec(bytes_per_device / sizeof(uint32_t), 0);
    std::iota(src_vec.begin(), src_vec.end(), 0);

    // Turn on Fast Dispatch for issuing writes.
    experimental::DispatchContext::get().initialize_fast_dispatch(mesh_device_.get());

    for (std::size_t logical_x = 0; logical_x < mesh_buffer->device()->num_cols(); logical_x++) {
        for (std::size_t logical_y = 0; logical_y < mesh_buffer->device()->num_rows(); logical_y++) {
            WriteShard(mesh_device_->mesh_command_queue(), mesh_buffer, src_vec, MeshCoordinate(logical_y, logical_x));
        }
    }
    Finish(mesh_device_->mesh_command_queue());

    // Turn off FD for running a workload and issuing reads.
    experimental::DispatchContext::get().terminate_fast_dispatch(mesh_device_.get());

    auto seed = 0;
    auto programs = tt::tt_metal::distributed::test::utils::create_random_programs(
        num_programs, mesh_device_->compute_with_storage_grid_size(), seed);

    for (uint32_t i = 0; i < num_programs; i++) {
        auto random_workload = std::make_shared<MeshWorkload>();
        random_workload->add_program(
            MeshCoordinateRange(
                MeshCoordinate{0, 0},
                MeshCoordinate{mesh_buffer->device()->num_rows() - 1, mesh_buffer->device()->num_cols() - 1}),
            std::move(*programs[i]));
        EnqueueMeshWorkload(mesh_device_->mesh_command_queue(), *random_workload, true);
    }

    for (std::size_t logical_x = 0; logical_x < mesh_buffer->device()->num_cols(); logical_x++) {
        for (std::size_t logical_y = 0; logical_y < mesh_buffer->device()->num_rows(); logical_y++) {
            std::vector<uint32_t> dst_vec = {};
            ReadShard(mesh_device_->mesh_command_queue(), dst_vec, mesh_buffer, MeshCoordinate(logical_y, logical_x));
            EXPECT_EQ(dst_vec, src_vec);
        }
    }
}

// Regression test for https://github.com/tenstorrent/tt-metal/issues/50634:
// A SD->FD->SD toggle must be a safe no-op on mock/emulated devices. These targets never create
// hardware command queues, so the FD teardown previously dereferenced an empty command-queue vector
// and segfaulted. Verify the round-trip completes and the device remains usable.
TEST_F(DispatchContextFixture, MockDeviceFdSdToggleIsNoOp) {
    const auto& rt_options = MetalContext::instance().rtoptions();
    if (rt_options.get_fast_dispatch()) {
        GTEST_SKIP() << "This test can only be run with Slow Dispatch mode.";
    }
    if (!MetalContext::instance().get_cluster().is_mock_or_emulated()) {
        GTEST_SKIP() << "This test only applies to mock/emulated devices.";
    }
    const MeshShape system_shape = MetalContext::instance().get_system_mesh().shape();
    auto mesh_device_ = MeshDevice::create(MeshDeviceConfig(system_shape));

    // SD -> FD -> SD toggle should not crash and should leave the device usable.
    experimental::DispatchContext::get().initialize_fast_dispatch(mesh_device_.get());
    experimental::DispatchContext::get().terminate_fast_dispatch(mesh_device_.get());
    Finish(mesh_device_->mesh_command_queue());
}

TEST(DispatchContext, DoubleInitWithoutTerminateShouldThrow) {
    const auto& rt_options = MetalContext::instance().rtoptions();
    if (rt_options.get_fast_dispatch()) {
        GTEST_SKIP() << "This test can only be run with Slow Dispatch mode.";
    }
    const MeshShape system_shape = MetalContext::instance().get_system_mesh().shape();
    auto mesh_device_ = MeshDevice::create(MeshDeviceConfig(system_shape));

    const auto& cluster = MetalContext::instance().get_cluster();
    if (cluster.is_mock_or_emulated()) {
        GTEST_SKIP() << "FD/SD toggle is a no-op on mock/emulated devices; the throw invariants do not apply. See "
                        "MockDeviceFdSdToggleIsNoOp.";
    }
    if (!cluster.is_ubb_galaxy() && cluster.arch() != tt::ARCH::BLACKHOLE) {
        GTEST_SKIP()
            << "Manually setting up and tearing down Fast Dispatch is only supported on Galaxy and Blackhole clusters.";
    }

    // Initialize fast dispatch
    experimental::DispatchContext::get().initialize_fast_dispatch(mesh_device_.get());

    // Double init without terminate should throw
    EXPECT_THROW(experimental::DispatchContext::get().initialize_fast_dispatch(mesh_device_.get()), std::runtime_error);

    // Clean up
    experimental::DispatchContext::get().terminate_fast_dispatch(mesh_device_.get());
}

// Stress test repeated SD <-> FD round-trips: verify buffer I/O and workload dispatch remain correct across cycles
TEST_F(DispatchContextFixture, RepeatedFdSdTransitionStress) {
    const auto& rt_options = MetalContext::instance().rtoptions();
    if (rt_options.get_fast_dispatch()) {
        GTEST_SKIP() << "This test can only be run with Slow Dispatch mode.";
    }
    const MeshShape system_shape = MetalContext::instance().get_system_mesh().shape();
    auto mesh_device_ = MeshDevice::create(MeshDeviceConfig(system_shape));

    const auto& cluster = MetalContext::instance().get_cluster();
    if (cluster.is_mock_or_emulated()) {
        GTEST_SKIP() << "Mock/emulated devices cannot validate real data movement; see "
                        "MockDeviceFdSdToggleIsNoOp for the mock-specific no-op behavior.";
    }
    if (!cluster.is_ubb_galaxy() && cluster.arch() != tt::ARCH::BLACKHOLE) {
        GTEST_SKIP()
            << "Manually setting up and tearing down Fast Dispatch is only supported on Galaxy and Blackhole clusters.";
    }

    uint32_t single_tile_size = ::tt::tile_size(DataFormat::UInt32);
    const uint32_t num_tiles = 64;
    const uint32_t num_programs = 5;

    CoreRangeSet shard_grid(CoreRange({0, 0}, {1, 1}));
    const uint32_t num_cores = 4;
    const uint32_t tiles_per_shard = num_tiles / num_cores;
    std::array<uint32_t, 2> shard_shape = {tiles_per_shard * single_tile_size, 1};
    std::array<uint32_t, 2> page_shape = {single_tile_size, 1};
    std::array<uint32_t, 2> tensor2d_shape = {num_tiles, 1};
    ShardSpecBuffer shard_spec(shard_grid, shard_shape, ShardOrientation::ROW_MAJOR, page_shape, tensor2d_shape);

    DeviceLocalBufferConfig fd_l1_config{
        .page_size = single_tile_size,
        .buffer_type = BufferType::L1,
        .sharding_args = BufferShardingArgs(shard_spec, TensorMemoryLayout::HEIGHT_SHARDED),
        .bottom_up = false};
    ReplicatedBufferConfig fd_l1_global{.size = num_tiles * single_tile_size};

    DeviceLocalBufferConfig sd_l1_config{
        .page_size = single_tile_size, .buffer_type = BufferType::L1, .bottom_up = false};
    ReplicatedBufferConfig sd_l1_global{.size = num_tiles * single_tile_size};

    DeviceLocalBufferConfig fd_dram_config{
        .page_size = single_tile_size, .buffer_type = BufferType::DRAM, .bottom_up = true};
    ReplicatedBufferConfig fd_dram_global{.size = num_tiles * single_tile_size};

    constexpr uint32_t num_cycles = 5;
    for (uint32_t cycle = 0; cycle < num_cycles; cycle++) {
        const uint32_t base = cycle * 10000;

        // FD phase 1: sharded L1 buffer
        experimental::DispatchContext::get().initialize_fast_dispatch(mesh_device_.get());

        auto fd_buf = MeshBuffer::create(fd_l1_global, fd_l1_config, mesh_device_.get());
        std::vector<uint32_t> fd_src_vec(num_tiles * single_tile_size / sizeof(uint32_t));
        std::iota(fd_src_vec.begin(), fd_src_vec.end(), base + 100);
        EnqueueWriteMeshBuffer(mesh_device_->mesh_command_queue(), fd_buf, fd_src_vec);
        Finish(mesh_device_->mesh_command_queue());

        for (const auto& coord : MeshCoordinateRange(mesh_device_->shape())) {
            std::vector<uint32_t> dst;
            ReadShard(mesh_device_->mesh_command_queue(), dst, fd_buf, coord);
            EXPECT_EQ(dst, fd_src_vec) << "Cycle " << cycle << ": sharded L1 readback failed in FD mode at " << coord;
        }

        experimental::DispatchContext::get().terminate_fast_dispatch(mesh_device_.get());

        // SD phase
        // Verify FD-written sharded buffer is still readable after FD->SD transition.
        for (const auto& coord : MeshCoordinateRange(mesh_device_->shape())) {
            std::vector<uint32_t> dst;
            ReadShard(mesh_device_->mesh_command_queue(), dst, fd_buf, coord);
            EXPECT_EQ(dst, fd_src_vec) << "Cycle " << cycle << ": sharded L1 data mismatch after FD->SD transition at "
                                       << coord;
        }

        // Write and verify interleaved L1 buffer in SD mode. Validated shard-by-shard because
        // EnqueueReadMeshBuffer is only defined for sharded global layouts on multi-device meshes.
        auto sd_buf = MeshBuffer::create(sd_l1_global, sd_l1_config, mesh_device_.get());
        std::vector<uint32_t> sd_src_vec(num_tiles * single_tile_size / sizeof(uint32_t));
        std::iota(sd_src_vec.begin(), sd_src_vec.end(), base + 200);
        EnqueueWriteMeshBuffer(mesh_device_->mesh_command_queue(), sd_buf, sd_src_vec);
        Finish(mesh_device_->mesh_command_queue());

        for (const auto& coord : MeshCoordinateRange(mesh_device_->shape())) {
            std::vector<uint32_t> dst;
            ReadShard(mesh_device_->mesh_command_queue(), dst, sd_buf, coord);
            EXPECT_EQ(dst, sd_src_vec) << "Cycle " << cycle << ": SD interleaved L1 verification failed at " << coord;
        }

        // Run random workloads to stress the dispatch path and dirty compute state.
        auto programs = tt::tt_metal::distributed::test::utils::create_random_programs(
            num_programs, mesh_device_->compute_with_storage_grid_size(), 0);
        for (uint32_t i = 0; i < num_programs; i++) {
            auto random_workload = std::make_shared<MeshWorkload>();
            random_workload->add_program(
                MeshCoordinateRange(
                    MeshCoordinate{0, 0}, MeshCoordinate{mesh_device_->num_rows() - 1, mesh_device_->num_cols() - 1}),
                std::move(*programs[i]));
            EnqueueMeshWorkload(mesh_device_->mesh_command_queue(), *random_workload, true);
        }
        Finish(mesh_device_->mesh_command_queue());

        // Verify SD buffer is uncorrupted after running workloads.
        for (const auto& coord : MeshCoordinateRange(mesh_device_->shape())) {
            std::vector<uint32_t> dst;
            ReadShard(mesh_device_->mesh_command_queue(), dst, sd_buf, coord);
            EXPECT_EQ(dst, sd_src_vec) << "Cycle " << cycle << ": SD buffer corrupted after running workloads at "
                                       << coord;
        }

        // An interleaved L1 buffer has a page on every bank, including the cores fast dispatch claims,
        // so it may not be resident when fast dispatch is re-entered: the L1 preflight would refuse
        // the session. Release it first. fd_buf is sharded on (0,0)-(1,1) and may stay.
        sd_buf.reset();

        // FD phase 2: DRAM buffer
        experimental::DispatchContext::get().initialize_fast_dispatch(mesh_device_.get());

        auto fd2_buf = MeshBuffer::create(fd_dram_global, fd_dram_config, mesh_device_.get());
        std::vector<uint32_t> fd2_src_vec(num_tiles * single_tile_size / sizeof(uint32_t));
        std::iota(fd2_src_vec.begin(), fd2_src_vec.end(), base + 300);
        for (std::size_t y = 0; y < mesh_device_->num_rows(); y++) {
            for (std::size_t x = 0; x < mesh_device_->num_cols(); x++) {
                WriteShard(mesh_device_->mesh_command_queue(), fd2_buf, fd2_src_vec, MeshCoordinate(y, x));
            }
        }
        Finish(mesh_device_->mesh_command_queue());

        for (std::size_t y = 0; y < mesh_device_->num_rows(); y++) {
            for (std::size_t x = 0; x < mesh_device_->num_cols(); x++) {
                std::vector<uint32_t> dst;
                ReadShard(mesh_device_->mesh_command_queue(), dst, fd2_buf, MeshCoordinate(y, x));
                EXPECT_EQ(dst, fd2_src_vec)
                    << "Cycle " << cycle << ": DRAM readback failed in FD mode at (" << y << "," << x << ")";
            }
        }

        experimental::DispatchContext::get().terminate_fast_dispatch(mesh_device_.get());
    }
}

TEST_F(DispatchContextFixture, AsyncSdStatePreservedAcrossFdTransition) {
    const auto& rt_options = MetalContext::instance().rtoptions();
    if (rt_options.get_fast_dispatch()) {
        GTEST_SKIP() << "This test can only be run with Slow Dispatch mode.";
    }
    const MeshShape system_shape = MetalContext::instance().get_system_mesh().shape();
    auto mesh_device_ = MeshDevice::create(MeshDeviceConfig(system_shape));

    const auto& cluster = MetalContext::instance().get_cluster();
    if (!cluster.is_ubb_galaxy() && cluster.arch() != tt::ARCH::BLACKHOLE) {
        GTEST_SKIP()
            << "Manually setting up and tearing down Fast Dispatch is only supported on Galaxy and Blackhole clusters.";
    }

    // Enable async slow dispatch before FD transition
    experimental::DispatchContext::get().enable_asynchronous_slow_dispatch(mesh_device_.get());
    EXPECT_TRUE(experimental::DispatchContext::get().is_asynchronous_slow_dispatch_enabled(mesh_device_.get()));

    // SD -> FD -> SD round-trip
    experimental::DispatchContext::get().initialize_fast_dispatch(mesh_device_.get());
    experimental::DispatchContext::get().terminate_fast_dispatch(mesh_device_.get());

    // Verify async SD state survived the round-trip
    EXPECT_TRUE(experimental::DispatchContext::get().is_asynchronous_slow_dispatch_enabled(mesh_device_.get()));
}

}  // namespace tt::tt_metal::distributed::test
