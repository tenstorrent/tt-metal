// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// GlobalSemaphoreSpace / GlobalSemaphore: host allocation and reset (fast and slow dispatch), and
// device-side GlobalSemaphore kernels (fast dispatch, WH/BH). The kernels read the semaphore's address
// from common runtime arg 0.

#include <gtest/gtest.h>
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
#include <tt-metalium/experimental/global_semaphore_space.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/mesh_device.hpp>
#include <tt-metalium/program_cache.hpp>

#include "command_queue_fixture.hpp"
#include "mesh_dispatch_fixture.hpp"
#include "impl/allocator/allocator.hpp"
#include "impl/context/metal_context.hpp"
#include "tt_metal/impl/dispatch/slow_dispatch.hpp"

namespace tt::tt_metal {
namespace {

using experimental::CreateGlobalSemaphoreSpace;

constexpr const char* kCounterKernel = "tests/tt_metal/tt_metal/test_kernels/dataflow/global_semaphore_counter.cpp";
constexpr const char* kMcastKernel = "tests/tt_metal/tt_metal/test_kernels/dataflow/global_semaphore_mcast.cpp";
constexpr const char* kStopFlagKernel = "tests/tt_metal/tt_metal/test_kernels/dataflow/global_semaphore_stop_flag.cpp";
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

// ---------------------------------------------------------------------------------------------------------------
// Host: allocation and reset
// ---------------------------------------------------------------------------------------------------------------

TEST_F(MeshDispatchFixture, GlobalSemaphoreFromSpaceStartsAtZeroAndResetReachesItsCores) {
    const CoreRangeSet space_cores(CoreRange({0, 0}, {3, 3}));
    const CoreRangeSet sem_cores(std::vector<CoreRange>{CoreRange({0, 0}, {2, 1}), CoreRange({3, 3}, {3, 3})});
    const CoreRangeSet untouched(CoreRange({0, 2}, {0, 2}));
    for (const auto& mesh_device : devices_) {
        auto space = CreateGlobalSemaphoreSpace(*mesh_device, space_cores, 4);
        auto sem = space.create_semaphore(sem_cores);
        EXPECT_EQ(sem.attribute_values(), std::make_tuple(sem_cores, BufferType::L1));
        WriteValueOnCores(*mesh_device, untouched, sem.address(), 0x5A5A);

        ExpectValueOnCores(*mesh_device, sem_cores, sem.address(), 0);
        sem.reset_semaphore_value(0xABCD);
        ExpectValueOnCores(*mesh_device, sem_cores, sem.address(), 0xABCD);
        ExpectValueOnCores(*mesh_device, untouched, sem.address(), 0x5A5A);
    }
}

TEST_F(MeshDispatchFixture, GlobalSemaphoreFromSpaceCopiesShareASlotThatIsReusedAfterTheLastCopy) {
    const CoreRangeSet cores(CoreRange({0, 0}, {1, 1}));
    for (const auto& mesh_device : devices_) {
        auto space = CreateGlobalSemaphoreSpace(*mesh_device, cores, 2);
        DeviceAddr first_address = 0;
        {
            std::optional<GlobalSemaphore> first = space.create_semaphore(cores);
            first_address = first->address();
            GlobalSemaphore copy = *first;
            first.reset();
            EXPECT_EQ(copy.address(), first_address);
            EXPECT_EQ(space.num_live_semaphores(), 1u) << "the copy should keep the slot";
            auto second = space.create_semaphore(cores);
            EXPECT_NE(second.address(), first_address);
            EXPECT_EQ(space.num_live_semaphores(), 2u);
        }
        EXPECT_EQ(space.num_live_semaphores(), 0u);
        auto again = space.create_semaphore(cores);
        EXPECT_EQ(again.address(), first_address);
    }
}

TEST_F(MeshDispatchFixture, GlobalSemaphoreSpaceRejectsOverCapacityAndForeignCores) {
    const CoreRangeSet cores(CoreRange({0, 0}, {1, 0}));
    for (const auto& mesh_device : devices_) {
        auto space = CreateGlobalSemaphoreSpace(*mesh_device, cores, 1);
        EXPECT_ANY_THROW(space.create_semaphore(CoreRangeSet(CoreRange({0, 0}, {2, 0}))));
        auto only = space.create_semaphore(cores);
        EXPECT_ANY_THROW(space.create_semaphore(CoreRangeSet(CoreRange({0, 0}, {0, 0}))));
        EXPECT_EQ(space.num_live_semaphores(), 1u);
    }
}

// How CCL ops create semaphores: lazily, after other cached programs have placed L1 on the same cores and
// sealed them. The space was reserved before that, so creating semaphores still works and does not move the
// arena's high-water mark under the cached program.
TEST_F(MeshDispatchFixture, GlobalSemaphoreFromSpaceCreatedAfterCachedProgramSealedItsCores) {
    const CoreRangeSet cores(CoreRange({0, 0}, {3, 1}));
    for (const auto& mesh_device : devices_) {
        auto space = CreateGlobalSemaphoreSpace(*mesh_device, cores, 4);
        auto& arena = mesh_device->allocator_impl()->persistent_l1();

        Program program = CreateProgram();
        CreateKernel(program, kBlankKernel, cores, Riscv0());
        constexpr uint32_t kPageSize = 2048;
        CreateCircularBuffer(
            program,
            cores,
            CircularBufferConfig(2 * kPageSize, {{tt::CBIndex::c_0, tt::DataFormat::Float16_b}})
                .set_page_size(tt::CBIndex::c_0, kPageSize));
        auto workload = MakeWorkload(*mesh_device, std::move(program));
        RunProgram(mesh_device, workload);

        const uint32_t alignment = MetalContext::instance().hal().get_alignment(HalMemType::L1);
        EXPECT_ANY_THROW(arena.allocate(cores, alignment, alignment)) << "the cached program should seal its cores";

        const DeviceAddr high_water_mark = arena.high_water_mark(cores);
        std::vector<GlobalSemaphore> sems;
        for (uint32_t i = 0; i < space.capacity(); i++) {
            sems.push_back(space.create_semaphore(cores));
            ExpectValueOnCores(*mesh_device, cores, sems.back().address(), 0);
        }
        EXPECT_EQ(arena.high_water_mark(cores), high_water_mark);

        sems.pop_back();
        sems.push_back(space.create_semaphore(cores));
        RunProgram(mesh_device, workload);
        EXPECT_EQ(arena.high_water_mark(cores), high_water_mark);
    }
}

TEST_F(MeshDispatchFixture, GlobalSemaphoreSpaceMustPrecedeProgramLocalL1) {
    const CoreRangeSet cores(CoreRange({0, 0}, {1, 1}));
    for (const auto& mesh_device : devices_) {
        Program program = CreateProgram();
        CreateKernel(program, kBlankKernel, cores, Riscv0());
        CreateCircularBuffer(
            program,
            cores,
            CircularBufferConfig(2048, {{tt::CBIndex::c_0, tt::DataFormat::Float16_b}})
                .set_page_size(tt::CBIndex::c_0, 2048));
        auto workload = MakeWorkload(*mesh_device, std::move(program));
        RunProgram(mesh_device, workload);
        EXPECT_ANY_THROW(CreateGlobalSemaphoreSpace(*mesh_device, cores, 1));
    }
}

std::shared_ptr<distributed::MeshDevice> OpenUnitMesh() {
    return distributed::MeshDevice::create_unit_mesh(
        0,
        DEFAULT_L1_SMALL_SIZE,
        DEFAULT_TRACE_REGION_SIZE,
        1,
        MetalContext::instance().resolve_dispatch_core_config());
}

TEST(GlobalSemaphoreSpaceDeviceLifetime, SemaphoreAndSpaceCanOutliveTheMeshDevice) {
    auto mesh_device = OpenUnitMesh();
    const CoreRangeSet cores(CoreRange({0, 0}, {1, 1}));
    std::optional<experimental::GlobalSemaphoreSpace> space = CreateGlobalSemaphoreSpace(*mesh_device, cores, 2);
    std::optional<GlobalSemaphore> sem = space->create_semaphore(cores);

    const std::weak_ptr<distributed::MeshDevice> weak_device = mesh_device;
    mesh_device->close();
    mesh_device.reset();
    ASSERT_TRUE(weak_device.expired());

    EXPECT_NO_THROW(space.reset());
    EXPECT_NO_THROW(sem.reset());
}

// CCL ops keep their semaphores in the cached program, and MeshDevice::close() leaves the program cache in
// place, so those semaphores are released from inside ~MeshDevice.
TEST(GlobalSemaphoreSpaceDeviceLifetime, SemaphoreInProgramCacheIsReleasedWithTheMeshDevice) {
    struct SharedVariables {
        GlobalSemaphore sem;
    };

    auto mesh_device = OpenUnitMesh();
    const CoreRangeSet cores(CoreRange({0, 0}, {1, 1}));
    {
        auto space = CreateGlobalSemaphoreSpace(*mesh_device, cores, 1);
        mesh_device->enable_program_cache();
        Program program = CreateProgram();
        CreateKernel(program, kBlankKernel, cores, Riscv0());
        program_cache::detail::CachedMeshWorkload<SharedVariables> cached(
            MakeWorkload(*mesh_device, std::move(program)), SharedVariables{space.create_semaphore(cores)});
        mesh_device->get_program_cache().insert(
            program_cache::detail::ProgramCacheKey{.hash = 0x6E5E, .canonical = "global-semaphore-space-lifetime"},
            program_cache::detail::CachedProgramFactory(std::move(cached), 0));
    }
    ASSERT_GT(mesh_device->num_program_cache_entries(), 0u);

    mesh_device->close();
    EXPECT_NO_THROW(mesh_device.reset());
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

// Senders each up() the hub's copy kIncrements times; the hub kernel (if any) waits for the total and drains it.
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

class GlobalSemaphoreSpaceKernelTest : public UnitMeshCQSingleCardFixture {
protected:
    void SetUp() override {
        UnitMeshCQSingleCardFixture::SetUp();
        if (IsSkipped()) {
            return;
        }
        if (!IsWormholeOrBlackhole(*devices_.at(0))) {
            GTEST_SKIP() << "Skipping: test requires Wormhole B0 or Blackhole hardware";
        }
        space_.emplace(CreateGlobalSemaphoreSpace(*devices_.at(0), CoreRangeSet(CoreRange({0, 0}, {3, 1})), 4));
    }

    void TearDown() override {
        space_.reset();
        UnitMeshCQSingleCardFixture::TearDown();
    }

    std::optional<experimental::GlobalSemaphoreSpace> space_;
};

class GlobalSemaphoreSpaceTraceTest : public UnitMeshCQSingleCardTraceFixture {
protected:
    void SetUp() override {
        UnitMeshCQSingleCardTraceFixture::SetUp();
        if (IsSkipped()) {
            return;
        }
        if (!IsWormholeOrBlackhole(*devices_.at(0))) {
            GTEST_SKIP() << "Skipping: test requires Wormhole B0 or Blackhole hardware";
        }
        space_.emplace(CreateGlobalSemaphoreSpace(*devices_.at(0), kHubAndSenders, 2));
    }

    void TearDown() override {
        space_.reset();
        UnitMeshCQSingleCardTraceFixture::TearDown();
    }

    std::optional<experimental::GlobalSemaphoreSpace> space_;
};

void RunCounterHub(distributed::MeshDevice& mesh_device, experimental::GlobalSemaphoreSpace& space, HubMode hub_mode) {
    auto sem = space.create_semaphore(kHubAndSenders);
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

TEST_F(GlobalSemaphoreSpaceKernelTest, ManyToOneUpThenWaitAndSet) {
    RunCounterHub(*devices_.at(0), *space_, HubMode::WAIT_AND_SET);
}

TEST_F(GlobalSemaphoreSpaceKernelTest, ManyToOneUpThenDown) { RunCounterHub(*devices_.at(0), *space_, HubMode::DOWN); }

// The semaphore outlives the program: a second program keeps counting from where the first stopped.
TEST_F(GlobalSemaphoreSpaceKernelTest, ValuePersistsAcrossPrograms) {
    auto& mesh_device = *devices_.at(0);
    auto sem = space_->create_semaphore(kHubAndSenders);
    for (int run = 0; run < 2; run++) {
        auto workload = MakeWorkload(mesh_device, MakeCounterProgram(mesh_device, HubMode::NONE, sem).program);
        distributed::EnqueueMeshWorkload(mesh_device.mesh_command_queue(), workload, /*blocking=*/true);
    }
    EXPECT_EQ(ReadWord(mesh_device, kHub, sem.address()), 2 * kPerRun);
}

// The same compiled program counts into a different semaphore once its common runtime arg is updated.
TEST_F(GlobalSemaphoreSpaceKernelTest, UpdatingTheAddressArgSwitchesSemaphore) {
    auto& mesh_device = *devices_.at(0);
    auto& cq = mesh_device.mesh_command_queue();
    auto first = space_->create_semaphore(kHubAndSenders);
    auto second = space_->create_semaphore(kHubAndSenders);
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

// reset_semaphore_value() goes through the command queue, so it lands after a program enqueued before it and
// before one enqueued after it.
TEST_F(GlobalSemaphoreSpaceKernelTest, ResetIsOrderedWithPrograms) {
    auto& mesh_device = *devices_.at(0);
    auto& cq = mesh_device.mesh_command_queue();
    auto sem = space_->create_semaphore(kHubAndSenders);
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
void RunMcast(
    distributed::MeshDevice& mesh_device,
    experimental::GlobalSemaphoreSpace& space,
    bool increment,
    uint32_t initial,
    uint32_t value) {
    const CoreRange dests({1, 0}, {3, 1});
    const CoreRangeSet all(std::vector<CoreRange>{CoreRange(kHub, kHub), dests});
    auto sem = space.create_semaphore(all);
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

TEST_F(GlobalSemaphoreSpaceKernelTest, SetMulticastCopiesValue) {
    RunMcast(*devices_.at(0), *space_, /*increment=*/false, 0, 0xA5);
}

TEST_F(GlobalSemaphoreSpaceKernelTest, IncMulticastAddsToEveryDest) {
    RunMcast(*devices_.at(0), *space_, /*increment=*/true, 5, 3);
}

// The host flips the semaphore while the kernel is polling value().
TEST_F(GlobalSemaphoreSpaceKernelTest, StopFlagSeenThroughValue) {
    auto& mesh_device = *devices_.at(0);
    auto sem = space_->create_semaphore(CoreRangeSet(CoreRange(kHub)));

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
TEST_F(GlobalSemaphoreSpaceTraceTest, TraceReplayUsesCaptureTimeAddress) {
    auto& mesh_device = *devices_.at(0);
    auto& cq = mesh_device.mesh_command_queue();
    auto first = space_->create_semaphore(kHubAndSenders);
    auto second = space_->create_semaphore(kHubAndSenders);

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
