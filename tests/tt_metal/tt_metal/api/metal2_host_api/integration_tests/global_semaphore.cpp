// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Fast-dispatch real-hardware tests for Metal 2.0 GlobalSemaphore bindings (WH/BH).
//
// Each test creates its persistent GlobalSemaphores first (program-local L1 seals a core's arena),
// builds a ProgramSpec whose kernels bind them through a GlobalSemaphoreParameter, and checks the
// semaphore words in L1 afterwards.

#include <gtest/gtest.h>
#include <cstdint>
#include <map>
#include <string>
#include <vector>

#include <tt-metalium/allocator.hpp>
#include <tt-metalium/device.hpp>
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/experimental/global_semaphore.hpp>
#include <tt-metalium/experimental/metal2_host_api/program.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_run_args.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_spec.hpp>
#include <tt-metalium/global_semaphore.hpp>
#include <tt-metalium/host_api.hpp>

#include "command_queue_fixture.hpp"
#include "tt_metal/impl/dispatch/slow_dispatch.hpp"

namespace tt::tt_metal::global_semaphore_hw_test {
namespace {

namespace m2 = experimental;

constexpr const char* kCounterKernel = "tests/tt_metal/tt_metal/test_kernels/dataflow/global_semaphore_counter.cpp";
constexpr const char* kMcastKernel = "tests/tt_metal/tt_metal/test_kernels/dataflow/global_semaphore_mcast.cpp";
constexpr const char* kStopFlagKernel = "tests/tt_metal/tt_metal/test_kernels/dataflow/global_semaphore_stop_flag.cpp";

const m2::GlobalSemaphoreParamName kParam{"gsem"};
const m2::NodeCoord kHub{0, 0};
const m2::NodeRange kSenders{m2::NodeCoord{1, 0}, m2::NodeCoord{3, 0}};
constexpr uint32_t kNumSenders = 3;
constexpr uint32_t kIncrements = 16;
const CoreRangeSet kHubAndSenders(CoreRange({0, 0}, {3, 0}));

bool IsGen1(const distributed::MeshDevice& mesh_device) {
    return mesh_device.arch() == tt::ARCH::WORMHOLE_B0 || mesh_device.arch() == tt::ARCH::BLACKHOLE;
}

uint32_t ReportAddr(const distributed::MeshDevice& mesh_device) {
    return mesh_device.allocator()->get_base_allocator_addr(HalMemType::L1) + 32 * 1024;
}

m2::DataMovementHardwareConfig Riscv0() {
    return m2::DataMovementHardwareConfig{
        .config_1xx = m2::DataMovementHardwareConfig::DataMovement1XXConfig{
            .processor = DataMovementProcessor::RISCV_0, .noc = NOC::NOC_0}};
}

m2::KernelSpec::SemaphoreBinding BindParam(const char* accessor) {
    return m2::KernelSpec::SemaphoreBinding{.semaphore_spec_name = kParam, .accessor_name = accessor};
}

m2::KernelSpec::CompilerOptions::Defines Defines(std::map<std::string, std::string> defines) {
    return m2::KernelSpec::CompilerOptions::Defines(std::move(defines));
}

uint32_t ReadWord(distributed::MeshDevice& mesh_device, const CoreCoord& core, uint32_t addr) {
    std::vector<uint32_t> out;
    slow_dispatch::ReadFromL1(mesh_device, core, addr, sizeof(uint32_t), out);
    return out.empty() ? 0xDEADBEEF : out[0];
}

enum class HubMode { NONE, WAIT_AND_SET, DOWN };

// Senders on kSenders each up() the hub's copy kIncrements times; the hub kernel (if any) waits for
// the total and drains it.
m2::ProgramSpec MakeCounterSpec(HubMode hub_mode) {
    m2::ProgramSpec spec{.name = "global_semaphore_counter"};
    spec.kernels.push_back(m2::KernelSpec{
        .unique_id = m2::KernelSpecName{"senders"},
        .source = kCounterKernel,
        .num_threads = 1,
        .compiler_options = {.defines = Defines({{"GLOBAL_SEM_SENDER", "1"}})},
        .semaphore_bindings = {BindParam("counter")},
        .runtime_arg_schema = {.common_runtime_arg_names = {"hub_x", "hub_y", "increments"}},
        .hw_config = Riscv0(),
    });
    spec.work_units.push_back(
        m2::WorkUnitSpec{.name = "senders", .kernels = {m2::KernelSpecName{"senders"}}, .target_nodes = kSenders});
    if (hub_mode != HubMode::NONE) {
        std::map<std::string, std::string> defines;
        if (hub_mode == HubMode::DOWN) {
            defines.emplace("GLOBAL_SEM_USE_DOWN", "1");
        }
        spec.kernels.push_back(m2::KernelSpec{
            .unique_id = m2::KernelSpecName{"hub"},
            .source = kCounterKernel,
            .num_threads = 1,
            .compiler_options = {.defines = Defines(defines)},
            .semaphore_bindings = {BindParam("counter")},
            .runtime_arg_schema = {.runtime_arg_names = {"report_addr", "expected"}},
            .hw_config = Riscv0(),
        });
        spec.work_units.push_back(
            m2::WorkUnitSpec{.name = "hub", .kernels = {m2::KernelSpecName{"hub"}}, .target_nodes = kHub});
    }
    spec.global_semaphore_parameters = {m2::GlobalSemaphoreParameter{.unique_id = kParam}};
    return spec;
}

m2::ProgramRunArgs MakeCounterArgs(distributed::MeshDevice& mesh_device, HubMode hub_mode, const GlobalSemaphore& sem) {
    const CoreCoord hub_virtual = mesh_device.worker_core_from_logical_core(CoreCoord{0, 0});
    m2::ProgramRunArgs args;
    args.kernel_run_args.push_back(m2::ProgramRunArgs::KernelRunArgs{
        .kernel = m2::KernelSpecName{"senders"},
        .common_runtime_arg_values =
            {{"hub_x", static_cast<uint32_t>(hub_virtual.x)},
             {"hub_y", static_cast<uint32_t>(hub_virtual.y)},
             {"increments", kIncrements}},
    });
    if (hub_mode != HubMode::NONE) {
        args.kernel_run_args.push_back(m2::ProgramRunArgs::KernelRunArgs{
            .kernel = m2::KernelSpecName{"hub"},
            .runtime_arg_values = m2::MakeRuntimeArgsForSingleNode(
                kHub, {{"report_addr", ReportAddr(mesh_device)}, {"expected", kNumSenders * kIncrements}}),
        });
    }
    args.global_semaphore_args.insert({kParam, m2::GlobalSemaphoreArgument{sem}});
    return args;
}

distributed::MeshWorkload MakeWorkload(distributed::MeshDevice& mesh_device, Program&& program) {
    distributed::MeshWorkload workload;
    workload.add_program(distributed::MeshCoordinateRange(mesh_device.shape()), std::move(program));
    return workload;
}

Program& OnlyProgram(distributed::MeshWorkload& workload) { return workload.get_programs().begin()->second; }

void RunCounterHub(distributed::MeshDevice& mesh_device, HubMode hub_mode) {
    auto sem = m2::CreatePersistentGlobalSemaphore(mesh_device, kHubAndSenders, 0);
    Program program = m2::MakeProgramFromSpec(mesh_device, MakeCounterSpec(hub_mode));
    m2::SetProgramRunArgs(program, MakeCounterArgs(mesh_device, hub_mode, sem));

    const uint32_t report_addr = ReportAddr(mesh_device);
    std::vector<uint32_t> sentinel{0xFFFFFFFF, 0xFFFFFFFF};
    slow_dispatch::WriteToL1(mesh_device, CoreCoord{0, 0}, report_addr, sentinel);

    auto workload = MakeWorkload(mesh_device, std::move(program));
    distributed::EnqueueMeshWorkload(mesh_device.mesh_command_queue(), workload, /*blocking=*/true);

    std::vector<uint32_t> report;
    slow_dispatch::ReadFromL1(mesh_device, CoreCoord{0, 0}, report_addr, 2 * sizeof(uint32_t), report);
    ASSERT_EQ(report.size(), 2u);
    EXPECT_EQ(report[0], kNumSenders * kIncrements) << "hub saw the wrong total";
    EXPECT_EQ(report[1], 0u) << "hub did not drain the semaphore";
    EXPECT_EQ(ReadWord(mesh_device, CoreCoord{0, 0}, sem.address()), 0u);
}

class GlobalSemaphoreHWTest : public UnitMeshCQSingleCardFixture {
protected:
    void SetUp() override {
        UnitMeshCQSingleCardFixture::SetUp();
        if (IsSkipped()) {
            return;
        }
        if (!IsGen1(*devices_.at(0))) {
            GTEST_SKIP() << "Skipping: test requires Wormhole B0 or Blackhole hardware";
        }
    }
};

class GlobalSemaphoreTraceHWTest : public UnitMeshCQSingleCardTraceFixture {
protected:
    void SetUp() override {
        UnitMeshCQSingleCardTraceFixture::SetUp();
        if (IsSkipped()) {
            return;
        }
        if (!IsGen1(*devices_.at(0))) {
            GTEST_SKIP() << "Skipping: test requires Wormhole B0 or Blackhole hardware";
        }
    }
};

TEST_F(GlobalSemaphoreHWTest, ManyToOneUpThenWaitAndSet) { RunCounterHub(*devices_.at(0), HubMode::WAIT_AND_SET); }

TEST_F(GlobalSemaphoreHWTest, ManyToOneUpThenDown) { RunCounterHub(*devices_.at(0), HubMode::DOWN); }

// The semaphore outlives the program: a second program keeps counting from where the first stopped.
TEST_F(GlobalSemaphoreHWTest, ValuePersistsAcrossPrograms) {
    auto& mesh_device = *devices_.at(0);
    auto sem = m2::CreatePersistentGlobalSemaphore(mesh_device, kHubAndSenders, 0);
    for (int run = 0; run < 2; run++) {
        Program program = m2::MakeProgramFromSpec(mesh_device, MakeCounterSpec(HubMode::NONE));
        m2::SetProgramRunArgs(program, MakeCounterArgs(mesh_device, HubMode::NONE, sem));
        auto workload = MakeWorkload(mesh_device, std::move(program));
        distributed::EnqueueMeshWorkload(mesh_device.mesh_command_queue(), workload, /*blocking=*/true);
    }
    EXPECT_EQ(ReadWord(mesh_device, CoreCoord{0, 0}, sem.address()), 2 * kNumSenders * kIncrements);
}

// Program cache hit with a different GlobalSemaphore: only the address table is rewritten.
TEST_F(GlobalSemaphoreHWTest, CacheHitRebindsDifferentSemaphore) {
    auto& mesh_device = *devices_.at(0);
    auto& cq = mesh_device.mesh_command_queue();
    auto first = m2::CreatePersistentGlobalSemaphore(mesh_device, kHubAndSenders, 0);
    auto second = m2::CreatePersistentGlobalSemaphore(mesh_device, kHubAndSenders, 0);
    ASSERT_NE(first.address(), second.address());

    Program program = m2::MakeProgramFromSpec(mesh_device, MakeCounterSpec(HubMode::NONE));
    m2::SetProgramRunArgs(program, MakeCounterArgs(mesh_device, HubMode::NONE, first));
    auto workload = MakeWorkload(mesh_device, std::move(program));
    distributed::EnqueueMeshWorkload(cq, workload, /*blocking=*/true);

    m2::ProgramRunArgs rebind;
    rebind.global_semaphore_args.insert({kParam, m2::GlobalSemaphoreArgument{second}});
    m2::UpdateProgramRunArgs(OnlyProgram(workload), rebind);
    distributed::EnqueueMeshWorkload(cq, workload, /*blocking=*/true);

    constexpr uint32_t kPerRun = kNumSenders * kIncrements;
    EXPECT_EQ(ReadWord(mesh_device, CoreCoord{0, 0}, first.address()), kPerRun);
    EXPECT_EQ(ReadWord(mesh_device, CoreCoord{0, 0}, second.address()), kPerRun);

    rebind.global_semaphore_args.clear();
    rebind.global_semaphore_args.insert({kParam, m2::GlobalSemaphoreArgument{first}});
    m2::UpdateProgramRunArgs(OnlyProgram(workload), rebind);
    distributed::EnqueueMeshWorkload(cq, workload, /*blocking=*/true);

    EXPECT_EQ(ReadWord(mesh_device, CoreCoord{0, 0}, first.address()), 2 * kPerRun);
    EXPECT_EQ(ReadWord(mesh_device, CoreCoord{0, 0}, second.address()), kPerRun);
}

// reset_semaphore_value() goes through the command queue, so it lands after a program enqueued
// before it and before one enqueued after it.
TEST_F(GlobalSemaphoreHWTest, ResetIsOrderedWithPrograms) {
    auto& mesh_device = *devices_.at(0);
    auto& cq = mesh_device.mesh_command_queue();
    auto sem = m2::CreatePersistentGlobalSemaphore(mesh_device, kHubAndSenders, 0);

    Program program = m2::MakeProgramFromSpec(mesh_device, MakeCounterSpec(HubMode::NONE));
    m2::SetProgramRunArgs(program, MakeCounterArgs(mesh_device, HubMode::NONE, sem));
    auto workload = MakeWorkload(mesh_device, std::move(program));

    distributed::EnqueueMeshWorkload(cq, workload, /*blocking=*/false);
    sem.reset_semaphore_value(1000);
    distributed::EnqueueMeshWorkload(cq, workload, /*blocking=*/false);
    distributed::Finish(cq);

    EXPECT_EQ(ReadWord(mesh_device, CoreCoord{0, 0}, sem.address()), 1000 + kNumSenders * kIncrements);
    for (const auto& core : corerange_to_cores(CoreRangeSet(CoreRange({1, 0}, {3, 0})))) {
        EXPECT_EQ(ReadWord(mesh_device, core, sem.address()), 1000u) << core.str();
    }
}

// One core multicasts to the rectangle (1,0)-(3,1); the sender is outside it.
class GlobalSemaphoreMcastTest : public GlobalSemaphoreHWTest {
protected:
    void RunMcast(bool increment, uint32_t initial, uint32_t value) {
        auto& mesh_device = *devices_.at(0);
        const CoreRange dests({1, 0}, {3, 1});
        const CoreRangeSet all(std::vector<CoreRange>{CoreRange({0, 0}, {0, 0}), dests});
        auto sem = m2::CreatePersistentGlobalSemaphore(mesh_device, all, initial);

        std::map<std::string, std::string> defines;
        if (increment) {
            defines.emplace("GLOBAL_SEM_INC", "1");
        }
        m2::ProgramSpec spec{
            .name = "global_semaphore_mcast",
            .kernels = {m2::KernelSpec{
                .unique_id = m2::KernelSpecName{"mcast"},
                .source = kMcastKernel,
                .num_threads = 1,
                .compiler_options = {.defines = Defines(defines)},
                .semaphore_bindings = {BindParam("flag")},
                .runtime_arg_schema =
                    {.common_runtime_arg_names = {"start_x", "start_y", "end_x", "end_y", "num_dests", "value"}},
                .hw_config = Riscv0(),
            }},
            .global_semaphore_parameters = {m2::GlobalSemaphoreParameter{.unique_id = kParam}},
            .work_units = {m2::WorkUnitSpec{
                .name = "mcast", .kernels = {m2::KernelSpecName{"mcast"}}, .target_nodes = m2::NodeCoord{0, 0}}},
        };
        Program program = m2::MakeProgramFromSpec(mesh_device, spec);

        const CoreCoord start = mesh_device.worker_core_from_logical_core(dests.start_coord);
        const CoreCoord end = mesh_device.worker_core_from_logical_core(dests.end_coord);
        m2::ProgramRunArgs args;
        args.kernel_run_args.push_back(m2::ProgramRunArgs::KernelRunArgs{
            .kernel = m2::KernelSpecName{"mcast"},
            .common_runtime_arg_values =
                {{"start_x", static_cast<uint32_t>(start.x)},
                 {"start_y", static_cast<uint32_t>(start.y)},
                 {"end_x", static_cast<uint32_t>(end.x)},
                 {"end_y", static_cast<uint32_t>(end.y)},
                 {"num_dests", static_cast<uint32_t>(dests.size())},
                 {"value", value}},
        });
        args.global_semaphore_args.insert({kParam, m2::GlobalSemaphoreArgument{sem}});
        m2::SetProgramRunArgs(program, args);

        auto workload = MakeWorkload(mesh_device, std::move(program));
        distributed::EnqueueMeshWorkload(mesh_device.mesh_command_queue(), workload, /*blocking=*/true);

        const uint32_t expected_dest = increment ? initial + 2 * value : value;
        const uint32_t expected_sender = increment ? initial : value;
        for (const auto& core : corerange_to_cores(CoreRangeSet(dests))) {
            EXPECT_EQ(ReadWord(mesh_device, core, sem.address()), expected_dest) << core.str();
        }
        EXPECT_EQ(ReadWord(mesh_device, CoreCoord{0, 0}, sem.address()), expected_sender);
    }
};

TEST_F(GlobalSemaphoreMcastTest, SetMulticastCopiesValue) { RunMcast(/*increment=*/false, 0, 0xA5); }

TEST_F(GlobalSemaphoreMcastTest, IncMulticastAddsToEveryDest) { RunMcast(/*increment=*/true, 5, 3); }

// The host flips the semaphore while the kernel is polling value().
TEST_F(GlobalSemaphoreHWTest, StopFlagSeenThroughValue) {
    auto& mesh_device = *devices_.at(0);
    const CoreCoord core{0, 0};
    auto sem = m2::CreatePersistentGlobalSemaphore(mesh_device, CoreRangeSet(CoreRange(core)), 0);

    m2::ProgramSpec spec{
        .name = "global_semaphore_stop_flag",
        .kernels = {m2::KernelSpec{
            .unique_id = m2::KernelSpecName{"poller"},
            .source = kStopFlagKernel,
            .num_threads = 1,
            .semaphore_bindings = {BindParam("stop")},
            .runtime_arg_schema = {.runtime_arg_names = {"report_addr"}},
            .hw_config = Riscv0(),
        }},
        .global_semaphore_parameters = {m2::GlobalSemaphoreParameter{.unique_id = kParam}},
        .work_units = {m2::WorkUnitSpec{
            .name = "poller", .kernels = {m2::KernelSpecName{"poller"}}, .target_nodes = m2::NodeCoord{0, 0}}},
    };
    Program program = m2::MakeProgramFromSpec(mesh_device, spec);
    m2::ProgramRunArgs args;
    args.kernel_run_args.push_back(m2::ProgramRunArgs::KernelRunArgs{
        .kernel = m2::KernelSpecName{"poller"},
        .runtime_arg_values = m2::MakeRuntimeArgsForSingleNode(kHub, {{"report_addr", ReportAddr(mesh_device)}}),
    });
    args.global_semaphore_args.insert({kParam, m2::GlobalSemaphoreArgument{sem}});
    m2::SetProgramRunArgs(program, args);

    auto& cq = mesh_device.mesh_command_queue();
    auto workload = MakeWorkload(mesh_device, std::move(program));
    distributed::EnqueueMeshWorkload(cq, workload, /*blocking=*/false);
    std::vector<uint32_t> stop_value{7};
    slow_dispatch::WriteToL1(mesh_device, core, static_cast<uint32_t>(sem.address()), stop_value);
    distributed::Finish(cq);

    std::vector<uint32_t> report;
    slow_dispatch::ReadFromL1(mesh_device, core, ReportAddr(mesh_device), 2 * sizeof(uint32_t), report);
    ASSERT_EQ(report.size(), 2u);
    EXPECT_EQ(report[0], 7u);
    EXPECT_EQ(report[1], 0u) << "kernel gave up before it saw the flag";
}

// A trace keeps the table it was captured with; a later Update only affects non-traced launches.
TEST_F(GlobalSemaphoreTraceHWTest, TraceReplayUsesCaptureTimeSemaphore) {
    auto& mesh_device = *devices_.at(0);
    auto& cq = mesh_device.mesh_command_queue();
    auto first = m2::CreatePersistentGlobalSemaphore(mesh_device, kHubAndSenders, 0);
    auto second = m2::CreatePersistentGlobalSemaphore(mesh_device, kHubAndSenders, 0);

    Program program = m2::MakeProgramFromSpec(mesh_device, MakeCounterSpec(HubMode::NONE));
    m2::SetProgramRunArgs(program, MakeCounterArgs(mesh_device, HubMode::NONE, first));
    auto workload = MakeWorkload(mesh_device, std::move(program));
    distributed::EnqueueMeshWorkload(cq, workload, /*blocking=*/true);  // warmup: capture cannot load binaries

    const auto tid = distributed::BeginTraceCapture(&mesh_device, cq.id());
    distributed::EnqueueMeshWorkload(cq, workload, /*blocking=*/false);
    mesh_device.end_mesh_trace(cq.id(), tid);

    m2::ProgramRunArgs rebind;
    rebind.global_semaphore_args.insert({kParam, m2::GlobalSemaphoreArgument{second}});
    m2::UpdateProgramRunArgs(OnlyProgram(workload), rebind);

    mesh_device.replay_mesh_trace(cq.id(), tid, /*blocking=*/true);

    constexpr uint32_t kPerRun = kNumSenders * kIncrements;
    EXPECT_EQ(ReadWord(mesh_device, CoreCoord{0, 0}, first.address()), 2 * kPerRun);
    EXPECT_EQ(ReadWord(mesh_device, CoreCoord{0, 0}, second.address()), 0u);

    distributed::EnqueueMeshWorkload(cq, workload, /*blocking=*/true);
    EXPECT_EQ(ReadWord(mesh_device, CoreCoord{0, 0}, second.address()), kPerRun);
    mesh_device.release_mesh_trace(tid);
}

}  // namespace
}  // namespace tt::tt_metal::global_semaphore_hw_test
