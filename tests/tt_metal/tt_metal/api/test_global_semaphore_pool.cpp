// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Tests for GlobalSemaphores created from the device's pool, on the host and in kernels.
// The test kernels read the semaphore's address from common runtime arg 0.

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <array>
#include <cstdint>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <tuple>
#include <vector>

#include <tt-metalium/allocator.hpp>
#include <tt-metalium/circular_buffer_config.hpp>
#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/experimental/global_semaphore_pool.hpp>
#include <tt-metalium/global_semaphore.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/mesh_device.hpp>
#include <tt-metalium/program_cache.hpp>
#include <tt-metalium/sub_device.hpp>
#include <tt-metalium/sub_device_types.hpp>

#include "command_queue_fixture.hpp"
#include "mesh_dispatch_fixture.hpp"
#include "distributed/mesh_device_impl.hpp"
#include "impl/allocator/allocator.hpp"
#include "impl/buffers/global_semaphore_pool.hpp"
#include "impl/context/metal_context.hpp"
#include "tt_metal/impl/dispatch/slow_dispatch.hpp"

namespace tt::tt_metal {
namespace {

using ::testing::AllOf;
using ::testing::HasSubstr;
using ::testing::ThrowsMessage;

constexpr const char* kCounterKernel = "tests/tt_metal/tt_metal/test_kernels/dataflow/global_semaphore_counter.cpp";
constexpr const char* kMcastKernel = "tests/tt_metal/tt_metal/test_kernels/dataflow/global_semaphore_mcast.cpp";
constexpr const char* kStopFlagKernel = "tests/tt_metal/tt_metal/test_kernels/dataflow/global_semaphore_stop_flag.cpp";
constexpr const char* kLateWriteKernel =
    "tests/tt_metal/tt_metal/test_kernels/dataflow/global_semaphore_late_write.cpp";
constexpr const char* kSyncerKernel = "tests/tt_metal/tt_metal/test_kernels/misc/sub_device/syncer.cpp";
constexpr const char* kBlankKernel = "tests/tt_metal/tt_metal/test_kernels/dataflow/blank.cpp";

uint32_t ReadWord(distributed::MeshDevice& mesh_device, const CoreCoord& core, DeviceAddr addr) {
    std::vector<uint32_t> out;
    slow_dispatch::ReadFromL1(mesh_device, core, static_cast<uint32_t>(addr), sizeof(uint32_t), out);
    return out.empty() ? 0xDEADBEEF : out[0];
}

void ExpectValueOnCores(
    distributed::MeshDevice& mesh_device, const CoreRangeSet& cores, DeviceAddr addr, uint32_t expected) {
    for (IDevice* device : mesh_device.get_devices()) {
        for (const CoreCoord& core : corerange_to_cores(cores)) {
            const auto words = MetalContext::instance().get_cluster().read_core(
                device->id(), device->worker_core_from_logical_core(core), addr, sizeof(uint32_t));
            EXPECT_EQ(words.at(0), expected) << "device " << device->id() << " core " << core.str();
        }
    }
}

void WriteValueOnCores(
    distributed::MeshDevice& mesh_device, const CoreRangeSet& cores, DeviceAddr addr, uint32_t value) {
    const std::vector<uint32_t> words{value};
    for (IDevice* device : mesh_device.get_devices()) {
        for (const CoreCoord& core : corerange_to_cores(cores)) {
            MetalContext::instance().get_cluster().write_core(
                device->id(), device->worker_core_from_logical_core(core), words, addr);
        }
        MetalContext::instance().get_cluster().l1_barrier(device->id());
    }
}

distributed::MeshWorkload MakeWorkload(distributed::MeshDevice& mesh_device, Program&& program) {
    distributed::MeshWorkload workload;
    workload.add_program(distributed::MeshCoordinateRange(mesh_device.shape()), std::move(program));
    return workload;
}

Program& OnlyProgram(distributed::MeshWorkload& workload) { return workload.get_programs().begin()->second; }

DataMovementConfig Riscv0(std::map<std::string, std::string> defines = {}) {
    return DataMovementConfig{
        .processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default, .defines = std::move(defines)};
}

const GlobalSemaphorePool& PoolOf(const distributed::MeshDevice& mesh_device) {
    const auto& pool = mesh_device.impl().global_semaphore_pool();
    TT_FATAL(pool != nullptr, "mesh device has no GlobalSemaphore pool");
    return *pool;
}

DeviceAddr PoolEnd(const GlobalSemaphorePool& pool) {
    return pool.base_address() + GlobalSemaphorePool::reservation_size(pool.num_slots(), pool.slot_size());
}

bool HasPool(const distributed::MeshDevice& mesh_device) {
    return mesh_device.impl().global_semaphore_pool() != nullptr;
}

constexpr uint32_t kTestPoolSlots = 64;

// Fixture devices stay open for the whole suite, so only the first call reserves anything.
bool ReserveTestPool(const std::vector<std::shared_ptr<distributed::MeshDevice>>& devices) {
    for (const auto& mesh_device : devices) {
        experimental::ReserveGlobalSemaphorePool(*mesh_device, kTestPoolSlots);
        if (!HasPool(*mesh_device)) {
            return false;
        }
    }
    return true;
}

// ---------------------------------------------------------------------------------------------------------------
// Host: allocation and reset on the fixture's devices
// ---------------------------------------------------------------------------------------------------------------

class GlobalSemaphorePoolFixture : public MeshDispatchFixture {
protected:
    void SetUp() override {
        MeshDispatchFixture::SetUp();
        if (IsSkipped()) {
            return;
        }
        if (!ReserveTestPool(devices_)) {
            GTEST_SKIP() << "Skipping: TT_METAL_GLOBAL_SEMAPHORE_POOL_SLOTS=0";
        }
    }
};

TEST_F(GlobalSemaphorePoolFixture, StartsAtZeroAndResetReachesOnlyItsCores) {
    const CoreRangeSet sem_cores(std::vector<CoreRange>{CoreRange({0, 0}, {2, 1}), CoreRange({3, 3}, {3, 3})});
    const CoreRangeSet untouched(CoreRange({0, 2}, {0, 2}));
    for (const auto& mesh_device : devices_) {
        auto sem = experimental::CreateGlobalSemaphoreFromPool(*mesh_device, sem_cores);
        EXPECT_EQ(sem.attribute_values(), std::make_tuple(sem_cores, BufferType::L1));
        const auto& pool = PoolOf(*mesh_device);
        EXPECT_GE(sem.address(), pool.base_address());
        EXPECT_LT(sem.address(), PoolEnd(pool));
        WriteValueOnCores(*mesh_device, untouched, sem.address(), 0x5A5A);

        ExpectValueOnCores(*mesh_device, sem_cores, sem.address(), 0);
        sem.reset_semaphore_value(0xABCD);
        ExpectValueOnCores(*mesh_device, sem_cores, sem.address(), 0xABCD);
        ExpectValueOnCores(*mesh_device, untouched, sem.address(), 0x5A5A);
    }
}

TEST_F(GlobalSemaphorePoolFixture, CopiesShareASlot) {
    const CoreRangeSet cores(CoreRange({0, 0}, {1, 1}));
    for (const auto& mesh_device : devices_) {
        const auto& pool = PoolOf(*mesh_device);
        const uint32_t live_before = pool.num_live();
        {
            std::optional<GlobalSemaphore> first = experimental::CreateGlobalSemaphoreFromPool(*mesh_device, cores);
            const DeviceAddr first_address = first->address();
            GlobalSemaphore copy = *first;
            first.reset();
            EXPECT_EQ(copy.address(), first_address);
            EXPECT_EQ(pool.num_live(), live_before + 1) << "the copy should keep the slot";
            auto second = experimental::CreateGlobalSemaphoreFromPool(*mesh_device, cores);
            EXPECT_NE(second.address(), first_address);
            EXPECT_EQ(pool.num_live(), live_before + 2);
        }
        EXPECT_EQ(pool.num_live(), live_before);
    }
}

TEST_F(GlobalSemaphorePoolFixture, RejectsCoresOutsideTheWorkerGrid) {
    for (const auto& mesh_device : devices_) {
        const CoreCoord grid = mesh_device->compute_with_storage_grid_size();
        const CoreRangeSet outside(CoreRange({0, 0}, {grid.x, 0}));
        EXPECT_ANY_THROW(experimental::CreateGlobalSemaphoreFromPool(*mesh_device, outside));
    }
}

// CCL ops create their semaphores late, after other cached programs already use L1 on the same cores.
// That must still work, and must not move the cached program's L1.
TEST_F(GlobalSemaphorePoolFixture, CreatedAfterCachedProgramsSealedTheirCores) {
    const CoreRangeSet program_cores(CoreRange({0, 0}, {3, 1}));
    const std::vector<CoreRangeSet> sem_core_sets{
        CoreRangeSet(CoreRange({0, 0}, {3, 1})),
        CoreRangeSet(CoreRange({1, 1}, {2, 2})),
        CoreRangeSet(std::vector<CoreRange>{CoreRange({0, 0}, {0, 0}), CoreRange({4, 3}, {4, 3})}),
    };
    for (const auto& mesh_device : devices_) {
        auto& arena = mesh_device->allocator_impl()->persistent_l1();

        Program program = CreateProgram();
        CreateKernel(program, kBlankKernel, program_cores, Riscv0());
        constexpr uint32_t kPageSize = 2048;
        CreateCircularBuffer(
            program,
            program_cores,
            CircularBufferConfig(2 * kPageSize, {{tt::CBIndex::c_0, tt::DataFormat::Float16_b}})
                .set_page_size(tt::CBIndex::c_0, kPageSize));
        auto workload = MakeWorkload(*mesh_device, std::move(program));
        RunProgram(mesh_device, workload);

        const uint32_t alignment = MetalContext::instance().hal().get_alignment(HalMemType::L1);
        EXPECT_ANY_THROW(arena.allocate(program_cores, alignment, alignment))
            << "the cached program should seal its cores";

        const DeviceAddr high_water_mark = arena.high_water_mark(program_cores);
        EXPECT_GE(high_water_mark, PoolEnd(PoolOf(*mesh_device)));

        std::vector<GlobalSemaphore> sems;
        for (const auto& cores : sem_core_sets) {
            sems.push_back(experimental::CreateGlobalSemaphoreFromPool(*mesh_device, cores));
            ExpectValueOnCores(*mesh_device, cores, sems.back().address(), 0);
        }
        EXPECT_EQ(arena.high_water_mark(program_cores), high_water_mark);

        RunProgram(mesh_device, workload);
        EXPECT_EQ(arena.high_water_mark(program_cores), high_water_mark);
    }
}

// ---------------------------------------------------------------------------------------------------------------
// Host: tests that open their own device, to set the pool size or to close it
// ---------------------------------------------------------------------------------------------------------------

class PoolSizeOverride {
public:
    explicit PoolSizeOverride(std::optional<uint32_t> slots) :
        previous_(MetalContext::instance().rtoptions().get_global_semaphore_pool_slots()) {
        MetalContext::instance().rtoptions().set_global_semaphore_pool_slots(slots);
    }
    ~PoolSizeOverride() { MetalContext::instance().rtoptions().set_global_semaphore_pool_slots(previous_); }

    PoolSizeOverride(const PoolSizeOverride&) = delete;
    PoolSizeOverride& operator=(const PoolSizeOverride&) = delete;

private:
    std::optional<uint32_t> previous_;
};

std::shared_ptr<distributed::MeshDevice> OpenUnitMesh(size_t num_cqs = 1) {
    DispatchCoreConfig dispatch_core_config = MetalContext::instance().resolve_dispatch_core_config();
    const auto cluster_type = MetalContext::instance().get_cluster().get_cluster_type();
    if (num_cqs >= 2 && (cluster_type == ClusterType::T3K || cluster_type == ClusterType::N300)) {
        dispatch_core_config = DispatchCoreConfig{DispatchCoreType::ETH};
    }
    return distributed::MeshDevice::create_unit_mesh(
        0, DEFAULT_L1_SMALL_SIZE, DEFAULT_TRACE_REGION_SIZE, num_cqs, dispatch_core_config);
}

bool IsFastDispatch() { return MetalContext::instance().rtoptions().get_fast_dispatch(); }

class GlobalSemaphorePoolOwnDevice : public ::testing::Test {
protected:
    void TearDown() override {
        if (mesh_device_) {
            mesh_device_->close();
            mesh_device_.reset();
        }
    }

    // Opens the device and reserves a pool of `slots`, or no pool if 0.
    void Open(uint32_t slots, size_t num_cqs = 1) {
        mesh_device_ = OpenUnitMesh(num_cqs);
        if (slots == 0) {
            return;
        }
        experimental::ReserveGlobalSemaphorePool(*mesh_device_, slots);
        if (!HasPool(*mesh_device_)) {
            GTEST_SKIP() << "Skipping: TT_METAL_GLOBAL_SEMAPHORE_POOL_SLOTS=0";
        }
    }

    PersistentL1Arena& Arena() const { return mesh_device_->allocator_impl()->persistent_l1(); }
    CoreRangeSet Grid() const {
        const CoreCoord grid = mesh_device_->compute_with_storage_grid_size();
        return CoreRangeSet(CoreRange({0, 0}, {grid.x - 1, grid.y - 1}));
    }

    std::shared_ptr<distributed::MeshDevice> mesh_device_;
};

TEST_F(GlobalSemaphorePoolOwnDevice, FullPoolErrorNamesCapacityPeakAndOverride) {
    Open(2);
    if (IsSkipped()) {
        return;
    }
    const CoreRangeSet cores(CoreRange({0, 0}, {1, 0}));
    auto first = experimental::CreateGlobalSemaphoreFromPool(*mesh_device_, cores);
    auto second = experimental::CreateGlobalSemaphoreFromPool(*mesh_device_, cores);
    EXPECT_THAT(
        [&] { experimental::CreateGlobalSemaphoreFromPool(*mesh_device_, cores); },
        ThrowsMessage<std::runtime_error>(
            AllOf(HasSubstr("all 2 slots"), HasSubstr("peak 2"), HasSubstr("ReserveGlobalSemaphorePool"))));
    EXPECT_EQ(PoolOf(*mesh_device_).num_live(), 2u);
}

TEST_F(GlobalSemaphorePoolOwnDevice, ReleasedSlotIsReused) {
    Open(1);
    if (IsSkipped()) {
        return;
    }
    const CoreRangeSet cores(CoreRange({0, 0}, {1, 1}));
    DeviceAddr address = 0;
    {
        auto sem = experimental::CreateGlobalSemaphoreFromPool(*mesh_device_, cores);
        address = sem.address();
        sem.reset_semaphore_value(9);
    }
    auto again = experimental::CreateGlobalSemaphoreFromPool(*mesh_device_, cores);
    EXPECT_EQ(again.address(), address);
    ExpectValueOnCores(*mesh_device_, cores, again.address(), 0);
    EXPECT_EQ(PoolOf(*mesh_device_).peak_live(), 1u);
}

// Without a reservation no L1 is taken. Reserving puts the pool at the start of free L1.
TEST_F(GlobalSemaphorePoolOwnDevice, WithoutReserveThereIsNoPoolAndL1IsUnchanged) {
    Open(0);
    const DeviceAddr base = mesh_device_->allocator()->get_base_allocator_addr(HalMemType::L1);
    EXPECT_FALSE(HasPool(*mesh_device_));
    EXPECT_EQ(Arena().high_water_mark(Grid()), base);

    const CoreRangeSet cores(CoreRange({0, 0}, {1, 1}));
    EXPECT_THAT(
        [&] { experimental::CreateGlobalSemaphoreFromPool(*mesh_device_, cores); },
        ThrowsMessage<std::runtime_error>(HasSubstr("ReserveGlobalSemaphorePool")));
    {
        auto sem = CreateGlobalSemaphore(*mesh_device_, cores, 3);
        ExpectValueOnCores(*mesh_device_, cores, sem.address(), 3);
    }

    PoolSizeOverride no_override(std::nullopt);
    experimental::ReserveGlobalSemaphorePool(*mesh_device_, 4);
    const auto& pool = PoolOf(*mesh_device_);
    EXPECT_EQ(pool.base_address(), base);
    EXPECT_EQ(pool.num_slots(), 4u);
    EXPECT_EQ(Arena().high_water_mark(Grid()), PoolEnd(pool));
}

TEST_F(GlobalSemaphorePoolOwnDevice, ReserveAfterAProgramThrows) {
    Open(0);
    const CoreRangeSet cores(CoreRange({0, 0}, {1, 1}));
    Program program = CreateProgram();
    CreateKernel(program, kBlankKernel, cores, Riscv0());
    constexpr uint32_t kPageSize = 2048;
    CreateCircularBuffer(
        program,
        cores,
        CircularBufferConfig(kPageSize, {{tt::CBIndex::c_0, tt::DataFormat::Float16_b}})
            .set_page_size(tt::CBIndex::c_0, kPageSize));
    auto workload = MakeWorkload(*mesh_device_, std::move(program));
    distributed::EnqueueMeshWorkload(mesh_device_->mesh_command_queue(), workload, /*blocking=*/true);

    EXPECT_THAT(
        [&] { experimental::ReserveGlobalSemaphorePool(*mesh_device_, 4); },
        ThrowsMessage<std::runtime_error>(HasSubstr("right after opening the device")));
    EXPECT_FALSE(HasPool(*mesh_device_));
}

TEST_F(GlobalSemaphorePoolOwnDevice, ReserveAgainIsANoOpUnlessThePoolMustGrow) {
    PoolSizeOverride no_override(std::nullopt);
    Open(4);
    if (IsSkipped()) {
        return;
    }
    const auto* pool = mesh_device_->impl().global_semaphore_pool().get();
    EXPECT_NO_THROW(experimental::ReserveGlobalSemaphorePool(*mesh_device_, 4));
    EXPECT_NO_THROW(experimental::ReserveGlobalSemaphorePool(*mesh_device_, 2));
    EXPECT_EQ(mesh_device_->impl().global_semaphore_pool().get(), pool);
    EXPECT_EQ(pool->num_slots(), 4u);
    EXPECT_THAT(
        [&] { experimental::ReserveGlobalSemaphorePool(*mesh_device_, 8); },
        ThrowsMessage<std::runtime_error>(HasSubstr("cannot grow to 8")));
}

TEST_F(GlobalSemaphorePoolOwnDevice, EnvOverrideReplacesTheRequestedSize) {
    PoolSizeOverride pool_size(3);
    Open(kTestPoolSlots);
    if (IsSkipped()) {
        return;
    }
    EXPECT_EQ(PoolOf(*mesh_device_).num_slots(), 3u);
}

// A slot whose old semaphore a kernel may still write is not reused until that kernel finishes. The kernel
// here runs on CQ1 and writes late; the new semaphore must still read 0.
TEST_F(GlobalSemaphorePoolOwnDevice, ReleasedSlotWaitsForInFlightWork) {
    if (!IsFastDispatch()) {
        GTEST_SKIP() << "Skipping: needs fast dispatch";
    }
    Open(1, /*num_cqs=*/2);
    if (IsSkipped()) {
        return;
    }
    const CoreCoord core{0, 0};
    const CoreRangeSet cores(CoreRange(core, core));
    auto& cq1 = mesh_device_->mesh_command_queue(1);

    std::optional<GlobalSemaphore> old_sem = experimental::CreateGlobalSemaphoreFromPool(*mesh_device_, cores);
    const DeviceAddr address = old_sem->address();

    Program program = CreateProgram();
    const KernelHandle kernel = CreateKernel(program, kLateWriteKernel, core, Riscv0());
    constexpr uint32_t kDelayCycles = 500'000'000;
    SetRuntimeArgs(program, kernel, core, {static_cast<uint32_t>(address), 0xBAD, kDelayCycles});
    auto workload = MakeWorkload(*mesh_device_, std::move(program));
    distributed::EnqueueMeshWorkload(cq1, workload, /*blocking=*/false);

    old_sem.reset();
    auto new_sem = experimental::CreateGlobalSemaphoreFromPool(*mesh_device_, cores);
    ASSERT_EQ(new_sem.address(), address);
    distributed::Finish(cq1);
    ExpectValueOnCores(*mesh_device_, cores, address, 0);
}

// quiesce_devices() resets the event counters. Slots waiting on events from before the reset are freed by it,
// instead of waiting for an event that will never come.
TEST_F(GlobalSemaphorePoolOwnDevice, QuiesceFreesSlotsWaitingOnEvents) {
    if (!IsFastDispatch()) {
        GTEST_SKIP() << "Skipping: needs fast dispatch";
    }
    Open(3);
    if (IsSkipped()) {
        return;
    }
    const CoreCoord core{0, 0};
    const CoreRangeSet cores(CoreRange(core, core));
    const auto& pool = PoolOf(*mesh_device_);

    auto kept = experimental::CreateGlobalSemaphoreFromPool(*mesh_device_, cores);
    std::optional<GlobalSemaphore> first = experimental::CreateGlobalSemaphoreFromPool(*mesh_device_, cores);
    std::optional<GlobalSemaphore> second = experimental::CreateGlobalSemaphoreFromPool(*mesh_device_, cores);

    Program program = CreateProgram();
    const KernelHandle kernel = CreateKernel(program, kLateWriteKernel, core, Riscv0());
    SetRuntimeArgs(program, kernel, core, {static_cast<uint32_t>(kept.address()), 0, 200'000'000});
    auto workload = MakeWorkload(*mesh_device_, std::move(program));
    distributed::EnqueueMeshWorkload(mesh_device_->mesh_command_queue(), workload, /*blocking=*/false);

    first.reset();
    second.reset();
    // Both returned slots wait behind the kernel. This takes one of them; the other stays pending.
    auto reused = experimental::CreateGlobalSemaphoreFromPool(*mesh_device_, cores);
    ASSERT_EQ(pool.num_pending(), 1u);

    mesh_device_->quiesce_devices();
    EXPECT_EQ(pool.num_pending(), 0u);
    auto after_quiesce = experimental::CreateGlobalSemaphoreFromPool(*mesh_device_, cores);
    ExpectValueOnCores(*mesh_device_, cores, after_quiesce.address(), 0);
}

// The host sets a flag that a kernel on sub-device 1 is waiting for. The reset only waits for sub-device 0,
// otherwise it would wait for the kernel it is meant to release.
TEST_F(GlobalSemaphorePoolOwnDevice, ResetCanReleaseAKernelOnAnotherSubDevice) {
    if (!IsFastDispatch()) {
        GTEST_SKIP() << "Skipping: needs fast dispatch";
    }
    Open(kTestPoolSlots);
    if (IsSkipped()) {
        return;
    }
    const CoreCoord syncer_core{2, 2};
    SubDevice sub_device_0(std::array{CoreRangeSet(CoreRange({0, 0}, {1, 1}))});
    SubDevice sub_device_1(std::array{CoreRangeSet(CoreRange(syncer_core, syncer_core))});
    constexpr DeviceAddr kLocalL1Size = 3200;
    auto manager = mesh_device_->create_sub_device_manager({sub_device_0, sub_device_1}, kLocalL1Size);
    mesh_device_->load_sub_device_manager(manager);

    auto sem =
        experimental::CreateGlobalSemaphoreFromPool(*mesh_device_, CoreRangeSet(CoreRange(syncer_core, syncer_core)));
    Program program = CreateProgram();
    const KernelHandle kernel = CreateKernel(program, kSyncerKernel, syncer_core, Riscv0());
    SetRuntimeArgs(program, kernel, syncer_core, {static_cast<uint32_t>(sem.address())});
    auto workload = MakeWorkload(*mesh_device_, std::move(program));
    auto& cq = mesh_device_->mesh_command_queue();
    distributed::EnqueueMeshWorkload(cq, workload, /*blocking=*/false);

    const std::array<SubDeviceId, 1> order_after{SubDeviceId{0}};
    sem.reset_semaphore_value(1, order_after);
    distributed::Finish(cq);
    EXPECT_EQ(ReadWord(*mesh_device_, syncer_core, sem.address()), 0u) << "the syncer should have consumed the flag";

    mesh_device_->clear_loaded_sub_device_manager();
    mesh_device_->remove_sub_device_manager(manager);
}

TEST_F(GlobalSemaphorePoolOwnDevice, SubmeshSharesItsParentsPool) {
    if (!IsFastDispatch()) {
        GTEST_SKIP() << "Skipping: needs fast dispatch to create a submesh";
    }
    Open(kTestPoolSlots);
    if (IsSkipped()) {
        return;
    }
    auto submesh = mesh_device_->create_submesh(distributed::MeshShape(1, 1));
    const CoreRangeSet cores(CoreRange({0, 0}, {1, 1}));

    EXPECT_EQ(submesh->impl().global_semaphore_pool(), mesh_device_->impl().global_semaphore_pool());
    EXPECT_GE(submesh->allocator_impl()->persistent_l1().high_water_mark(cores), PoolEnd(PoolOf(*mesh_device_)));

    // Only one of the two meshes may have used a command queue when the submesh closes.
    auto on_submesh = experimental::CreateGlobalSemaphoreFromPool(*submesh, cores);
    mesh_device_->quiesce_devices();
    auto on_parent = experimental::CreateGlobalSemaphoreFromPool(*mesh_device_, cores);
    EXPECT_NE(on_parent.address(), on_submesh.address());
    EXPECT_EQ(PoolOf(*mesh_device_).num_live(), 2u);
}

// Reserving through an existing submesh reserves the pool for its parent, and for the submesh too.
TEST_F(GlobalSemaphorePoolOwnDevice, ReserveOnAnExistingSubmeshCoversItsParent) {
    if (!IsFastDispatch()) {
        GTEST_SKIP() << "Skipping: needs fast dispatch to create a submesh";
    }
    Open(0);
    auto submesh = mesh_device_->create_submesh(distributed::MeshShape(1, 1));
    experimental::ReserveGlobalSemaphorePool(*submesh, 4);
    if (!HasPool(*submesh)) {
        GTEST_SKIP() << "Skipping: TT_METAL_GLOBAL_SEMAPHORE_POOL_SLOTS=0";
    }
    EXPECT_EQ(submesh->impl().global_semaphore_pool(), mesh_device_->impl().global_semaphore_pool());
    EXPECT_EQ(submesh->allocator_impl()->persistent_l1().high_water_mark(Grid()), PoolEnd(PoolOf(*mesh_device_)));
}

TEST_F(GlobalSemaphorePoolOwnDevice, UseAfterCloseThrows) {
    Open(kTestPoolSlots);
    if (IsSkipped()) {
        return;
    }
    const CoreRangeSet cores(CoreRange({0, 0}, {1, 1}));
    std::optional<GlobalSemaphore> sem = experimental::CreateGlobalSemaphoreFromPool(*mesh_device_, cores);

    mesh_device_->close();
    EXPECT_ANY_THROW(sem->reset_semaphore_value(1));
    EXPECT_ANY_THROW(experimental::CreateGlobalSemaphoreFromPool(*mesh_device_, cores));

    const std::weak_ptr<distributed::MeshDevice> weak_device = mesh_device_;
    mesh_device_.reset();
    ASSERT_TRUE(weak_device.expired());
    EXPECT_ANY_THROW(sem->reset_semaphore_value(1));
    EXPECT_NO_THROW(sem.reset());
}

// CCL ops keep their semaphores in cached programs. close() keeps the program cache, so those semaphores are
// destroyed later, in ~MeshDevice, and that must not crash.
TEST_F(GlobalSemaphorePoolOwnDevice, SemaphoreInProgramCacheIsReleasedWithTheMeshDevice) {
    struct SharedVariables {
        GlobalSemaphore sem;
    };

    Open(kTestPoolSlots);
    if (IsSkipped()) {
        return;
    }
    const CoreRangeSet cores(CoreRange({0, 0}, {1, 1}));
    {
        mesh_device_->enable_program_cache();
        Program program = CreateProgram();
        CreateKernel(program, kBlankKernel, cores, Riscv0());
        program_cache::detail::CachedMeshWorkload<SharedVariables> cached(
            MakeWorkload(*mesh_device_, std::move(program)),
            SharedVariables{experimental::CreateGlobalSemaphoreFromPool(*mesh_device_, cores)});
        mesh_device_->get_program_cache().insert(
            program_cache::detail::ProgramCacheKey{.hash = 0x6E5E, .canonical = "global-semaphore-pool-lifetime"},
            program_cache::detail::CachedProgramFactory(std::move(cached), 0));
    }
    ASSERT_GT(mesh_device_->num_program_cache_entries(), 0u);

    mesh_device_->close();
    EXPECT_NO_THROW(mesh_device_.reset());
}

// ---------------------------------------------------------------------------------------------------------------
// Device: GlobalSemaphore kernels
// ---------------------------------------------------------------------------------------------------------------

const CoreCoord kHub{0, 0};
const CoreRange kSenders({1, 0}, {3, 0});
constexpr uint32_t kNumSenders = 3;
constexpr uint32_t kIncrements = 16;
constexpr uint32_t kPerRun = kNumSenders * kIncrements;
const CoreRangeSet kHubAndSenders(CoreRange({0, 0}, {3, 0}));

bool IsWormholeOrBlackhole(const distributed::MeshDevice& mesh_device) {
    return mesh_device.arch() == tt::ARCH::WORMHOLE_B0 || mesh_device.arch() == tt::ARCH::BLACKHOLE;
}

uint32_t ReportAddr(const distributed::MeshDevice& mesh_device) {
    return mesh_device.allocator()->get_base_allocator_addr(HalMemType::L1) + 32 * 1024;
}

enum class HubMode { NONE, WAIT_AND_SET, DOWN };

struct CounterProgram {
    Program program;
    KernelHandle senders{};
    std::optional<KernelHandle> hub;
};

// Each sender up()s the hub's copy kIncrements times. The hub kernel, if any, waits for the total and clears it.
CounterProgram MakeCounterProgram(distributed::MeshDevice& mesh_device, HubMode hub_mode, const GlobalSemaphore& sem) {
    CounterProgram counter{.program = CreateProgram()};
    const CoreCoord hub_virtual = mesh_device.worker_core_from_logical_core(kHub);
    counter.senders = CreateKernel(counter.program, kCounterKernel, kSenders, Riscv0({{"GLOBAL_SEM_SENDER", "1"}}));
    SetRuntimeArgs(
        counter.program,
        counter.senders,
        kSenders,
        {static_cast<uint32_t>(hub_virtual.x), static_cast<uint32_t>(hub_virtual.y), kIncrements});
    SetCommonRuntimeArgs(counter.program, counter.senders, {static_cast<uint32_t>(sem.address())});
    if (hub_mode != HubMode::NONE) {
        std::map<std::string, std::string> defines;
        if (hub_mode == HubMode::DOWN) {
            defines.emplace("GLOBAL_SEM_USE_DOWN", "1");
        }
        counter.hub = CreateKernel(counter.program, kCounterKernel, kHub, Riscv0(defines));
        SetRuntimeArgs(counter.program, *counter.hub, kHub, {ReportAddr(mesh_device), kPerRun});
        SetCommonRuntimeArgs(counter.program, *counter.hub, {static_cast<uint32_t>(sem.address())});
    }
    return counter;
}

class GlobalSemaphoreKernelTest : public UnitMeshCQSingleCardFixture {
protected:
    void SetUp() override {
        UnitMeshCQSingleCardFixture::SetUp();
        if (IsSkipped()) {
            return;
        }
        if (!IsWormholeOrBlackhole(*devices_.at(0))) {
            GTEST_SKIP() << "Skipping: test requires Wormhole B0 or Blackhole hardware";
        }
        if (!ReserveTestPool(devices_)) {
            GTEST_SKIP() << "Skipping: TT_METAL_GLOBAL_SEMAPHORE_POOL_SLOTS=0";
        }
    }
};

class GlobalSemaphoreTraceTest : public UnitMeshCQSingleCardTraceFixture {
protected:
    void SetUp() override {
        UnitMeshCQSingleCardTraceFixture::SetUp();
        if (IsSkipped()) {
            return;
        }
        if (!IsWormholeOrBlackhole(*devices_.at(0))) {
            GTEST_SKIP() << "Skipping: test requires Wormhole B0 or Blackhole hardware";
        }
        if (!ReserveTestPool(devices_)) {
            GTEST_SKIP() << "Skipping: TT_METAL_GLOBAL_SEMAPHORE_POOL_SLOTS=0";
        }
    }
};

void RunCounterHub(distributed::MeshDevice& mesh_device, HubMode hub_mode) {
    auto sem = experimental::CreateGlobalSemaphoreFromPool(mesh_device, kHubAndSenders);
    auto counter = MakeCounterProgram(mesh_device, hub_mode, sem);

    const uint32_t report_addr = ReportAddr(mesh_device);
    std::vector<uint32_t> sentinel{0xFFFFFFFF, 0xFFFFFFFF};
    slow_dispatch::WriteToL1(mesh_device, kHub, report_addr, sentinel);

    auto workload = MakeWorkload(mesh_device, std::move(counter.program));
    distributed::EnqueueMeshWorkload(mesh_device.mesh_command_queue(), workload, /*blocking=*/true);

    std::vector<uint32_t> report;
    slow_dispatch::ReadFromL1(mesh_device, kHub, report_addr, 2 * sizeof(uint32_t), report);
    ASSERT_EQ(report.size(), 2u);
    EXPECT_EQ(report[0], kPerRun) << "hub saw the wrong total";
    EXPECT_EQ(report[1], 0u) << "hub did not drain the semaphore";
    EXPECT_EQ(ReadWord(mesh_device, kHub, sem.address()), 0u);
}

TEST_F(GlobalSemaphoreKernelTest, ManyToOneUpThenWaitAndSet) { RunCounterHub(*devices_.at(0), HubMode::WAIT_AND_SET); }

TEST_F(GlobalSemaphoreKernelTest, ManyToOneUpThenDown) { RunCounterHub(*devices_.at(0), HubMode::DOWN); }

// The semaphore outlives the program: a second program keeps counting from where the first stopped.
TEST_F(GlobalSemaphoreKernelTest, ValuePersistsAcrossPrograms) {
    auto& mesh_device = *devices_.at(0);
    auto sem = experimental::CreateGlobalSemaphoreFromPool(mesh_device, kHubAndSenders);
    for (int run = 0; run < 2; run++) {
        auto workload = MakeWorkload(mesh_device, MakeCounterProgram(mesh_device, HubMode::NONE, sem).program);
        distributed::EnqueueMeshWorkload(mesh_device.mesh_command_queue(), workload, /*blocking=*/true);
    }
    EXPECT_EQ(ReadWord(mesh_device, kHub, sem.address()), 2 * kPerRun);
}

// The same compiled program counts into a different semaphore once its common runtime arg is updated.
TEST_F(GlobalSemaphoreKernelTest, UpdatingTheAddressArgSwitchesSemaphore) {
    auto& mesh_device = *devices_.at(0);
    auto& cq = mesh_device.mesh_command_queue();
    auto first = experimental::CreateGlobalSemaphoreFromPool(mesh_device, kHubAndSenders);
    auto second = experimental::CreateGlobalSemaphoreFromPool(mesh_device, kHubAndSenders);
    ASSERT_NE(first.address(), second.address());

    auto counter = MakeCounterProgram(mesh_device, HubMode::NONE, first);
    const KernelHandle senders = counter.senders;
    auto workload = MakeWorkload(mesh_device, std::move(counter.program));
    distributed::EnqueueMeshWorkload(cq, workload, /*blocking=*/true);

    GetCommonRuntimeArgs(OnlyProgram(workload), senders)[0] = static_cast<uint32_t>(second.address());
    distributed::EnqueueMeshWorkload(cq, workload, /*blocking=*/true);

    EXPECT_EQ(ReadWord(mesh_device, kHub, first.address()), kPerRun);
    EXPECT_EQ(ReadWord(mesh_device, kHub, second.address()), kPerRun);
}

// reset_semaphore_value() goes through the command queue, so it runs between the programs enqueued before and
// after it.
TEST_F(GlobalSemaphoreKernelTest, ResetIsOrderedWithPrograms) {
    auto& mesh_device = *devices_.at(0);
    auto& cq = mesh_device.mesh_command_queue();
    auto sem = experimental::CreateGlobalSemaphoreFromPool(mesh_device, kHubAndSenders);
    auto workload = MakeWorkload(mesh_device, MakeCounterProgram(mesh_device, HubMode::NONE, sem).program);

    distributed::EnqueueMeshWorkload(cq, workload, /*blocking=*/false);
    sem.reset_semaphore_value(1000);
    distributed::EnqueueMeshWorkload(cq, workload, /*blocking=*/false);
    distributed::Finish(cq);

    EXPECT_EQ(ReadWord(mesh_device, kHub, sem.address()), 1000 + kPerRun);
    for (const auto& core : corerange_to_cores(CoreRangeSet(kSenders))) {
        EXPECT_EQ(ReadWord(mesh_device, core, sem.address()), 1000u) << core.str();
    }
}

// One core multicasts to the rectangle (1,0)-(3,1); the sender is outside it.
void RunMcast(distributed::MeshDevice& mesh_device, bool increment, uint32_t initial, uint32_t value) {
    const CoreRange dests({1, 0}, {3, 1});
    const CoreRangeSet all(std::vector<CoreRange>{CoreRange(kHub, kHub), dests});
    auto sem = experimental::CreateGlobalSemaphoreFromPool(mesh_device, all);
    sem.reset_semaphore_value(initial);

    std::map<std::string, std::string> defines;
    if (increment) {
        defines.emplace("GLOBAL_SEM_INC", "1");
    }
    Program program = CreateProgram();
    const KernelHandle kernel = CreateKernel(program, kMcastKernel, kHub, Riscv0(defines));
    const CoreCoord start = mesh_device.worker_core_from_logical_core(dests.start_coord);
    const CoreCoord end = mesh_device.worker_core_from_logical_core(dests.end_coord);
    SetRuntimeArgs(
        program,
        kernel,
        kHub,
        {static_cast<uint32_t>(start.x),
         static_cast<uint32_t>(start.y),
         static_cast<uint32_t>(end.x),
         static_cast<uint32_t>(end.y),
         static_cast<uint32_t>(dests.size()),
         value});
    SetCommonRuntimeArgs(program, kernel, {static_cast<uint32_t>(sem.address())});

    auto workload = MakeWorkload(mesh_device, std::move(program));
    distributed::EnqueueMeshWorkload(mesh_device.mesh_command_queue(), workload, /*blocking=*/true);

    const uint32_t expected_dest = increment ? initial + 2 * value : value;
    const uint32_t expected_sender = increment ? initial : value;
    for (const auto& core : corerange_to_cores(CoreRangeSet(dests))) {
        EXPECT_EQ(ReadWord(mesh_device, core, sem.address()), expected_dest) << core.str();
    }
    EXPECT_EQ(ReadWord(mesh_device, kHub, sem.address()), expected_sender);
}

TEST_F(GlobalSemaphoreKernelTest, SetMulticastCopiesValue) { RunMcast(*devices_.at(0), /*increment=*/false, 0, 0xA5); }

TEST_F(GlobalSemaphoreKernelTest, IncMulticastAddsToEveryDest) { RunMcast(*devices_.at(0), /*increment=*/true, 5, 3); }

// The host flips the semaphore while the kernel is polling value().
TEST_F(GlobalSemaphoreKernelTest, StopFlagSeenThroughValue) {
    auto& mesh_device = *devices_.at(0);
    auto sem = experimental::CreateGlobalSemaphoreFromPool(mesh_device, CoreRangeSet(CoreRange(kHub)));

    Program program = CreateProgram();
    const KernelHandle kernel = CreateKernel(program, kStopFlagKernel, kHub, Riscv0());
    SetRuntimeArgs(program, kernel, kHub, {ReportAddr(mesh_device)});
    SetCommonRuntimeArgs(program, kernel, {static_cast<uint32_t>(sem.address())});

    auto& cq = mesh_device.mesh_command_queue();
    auto workload = MakeWorkload(mesh_device, std::move(program));
    distributed::EnqueueMeshWorkload(cq, workload, /*blocking=*/false);
    std::vector<uint32_t> stop_value{7};
    slow_dispatch::WriteToL1(mesh_device, kHub, static_cast<uint32_t>(sem.address()), stop_value);
    distributed::Finish(cq);

    std::vector<uint32_t> report;
    slow_dispatch::ReadFromL1(mesh_device, kHub, ReportAddr(mesh_device), 2 * sizeof(uint32_t), report);
    ASSERT_EQ(report.size(), 2u);
    EXPECT_EQ(report[0], 7u);
    EXPECT_EQ(report[1], 0u) << "kernel gave up before it saw the flag";
}

// A trace keeps the address it was captured with; a later update only affects non-traced launches.
TEST_F(GlobalSemaphoreTraceTest, TraceReplayUsesCaptureTimeAddress) {
    auto& mesh_device = *devices_.at(0);
    auto& cq = mesh_device.mesh_command_queue();
    auto first = experimental::CreateGlobalSemaphoreFromPool(mesh_device, kHubAndSenders);
    auto second = experimental::CreateGlobalSemaphoreFromPool(mesh_device, kHubAndSenders);

    auto counter = MakeCounterProgram(mesh_device, HubMode::NONE, first);
    const KernelHandle senders = counter.senders;
    auto workload = MakeWorkload(mesh_device, std::move(counter.program));
    distributed::EnqueueMeshWorkload(cq, workload, /*blocking=*/true);  // warmup: capture cannot load binaries

    const auto tid = distributed::BeginTraceCapture(&mesh_device, cq.id());
    distributed::EnqueueMeshWorkload(cq, workload, /*blocking=*/false);
    mesh_device.end_mesh_trace(cq.id(), tid);

    GetCommonRuntimeArgs(OnlyProgram(workload), senders)[0] = static_cast<uint32_t>(second.address());
    mesh_device.replay_mesh_trace(cq.id(), tid, /*blocking=*/true);

    EXPECT_EQ(ReadWord(mesh_device, kHub, first.address()), 2 * kPerRun);
    EXPECT_EQ(ReadWord(mesh_device, kHub, second.address()), 0u);

    distributed::EnqueueMeshWorkload(cq, workload, /*blocking=*/true);
    EXPECT_EQ(ReadWord(mesh_device, kHub, second.address()), kPerRun);
    mesh_device.release_mesh_trace(tid);
}

}  // namespace
}  // namespace tt::tt_metal
