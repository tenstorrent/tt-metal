// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// GlobalSemaphores bound to Metal 2.0 kernels: on WH/BH silicon under fast dispatch, and on the Quasar
// simulator under slow dispatch. Each kernel constructs
// GlobalSemaphore(global_sem::<accessor>) and the host supplies the semaphore in
// ProgramRunArgs::global_semaphore_args.

#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include <cstdint>
#include <functional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <tt-metalium/allocator.hpp>
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/global_semaphore.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/workload/specification/program_spec.hpp>
#include <tt-metalium/experimental/metal2_host_api/program.hpp>
#include <tt-metalium/workload/run_args/program_run_args.hpp>

#include "command_queue_fixture.hpp"
#include "device_fixture.hpp"
#include "tt_metal/impl/dispatch/slow_dispatch.hpp"

namespace tt::tt_metal {
namespace {

using ::testing::HasSubstr;
using ::testing::ThrowsMessage;

constexpr const char* kCounterKernel = "tests/tt_metal/tt_metal/test_kernels/dataflow/global_semaphore_counter.cpp";
constexpr const char* kMcastKernel = "tests/tt_metal/tt_metal/test_kernels/dataflow/global_semaphore_mcast.cpp";
constexpr const char* kStopFlagKernel = "tests/tt_metal/tt_metal/test_kernels/dataflow/global_semaphore_stop_flag.cpp";

const CoreCoord kHub{0, 0};
const CoreRange kSenders({1, 0}, {3, 0});
constexpr uint32_t kNumSenders = 3;
constexpr uint32_t kIncrements = 16;
constexpr uint32_t kPerRun = kNumSenders * kIncrements;
const CoreRangeSet kHubAndSenders(CoreRange({0, 0}, {3, 0}));

class GlobalSemaphoreMetal2Test : public UnitMeshCQSingleCardFixture {
protected:
    void SetUp() override {
        UnitMeshCQSingleCardFixture::SetUp();
        if (IsSkipped()) {
            return;
        }
        const auto arch = devices_.at(0)->arch();
        if (arch != tt::ARCH::WORMHOLE_B0 && arch != tt::ARCH::BLACKHOLE) {
            GTEST_SKIP() << "Skipping: test requires Wormhole B0 or Blackhole hardware";
        }
    }
};

class GlobalSemaphoreMetal2TraceTest : public UnitMeshCQSingleCardTraceFixture {
protected:
    void SetUp() override {
        UnitMeshCQSingleCardTraceFixture::SetUp();
        if (IsSkipped()) {
            return;
        }
        const auto arch = devices_.at(0)->arch();
        if (arch != tt::ARCH::WORMHOLE_B0 && arch != tt::ARCH::BLACKHOLE) {
            GTEST_SKIP() << "Skipping: test requires Wormhole B0 or Blackhole hardware";
        }
    }
};

uint32_t ReportAddr(const distributed::MeshDevice& mesh_device) {
    return mesh_device.allocator()->get_base_allocator_addr(HalMemType::L1) + 32 * 1024;
}

uint32_t ReadWord(distributed::MeshDevice& mesh_device, const CoreCoord& core, DeviceAddr addr) {
    std::vector<uint32_t> out;
    slow_dispatch::ReadFromL1(mesh_device, core, static_cast<uint32_t>(addr), sizeof(uint32_t), out);
    return out.empty() ? 0xDEADBEEF : out[0];
}

distributed::MeshWorkload MakeWorkload(distributed::MeshDevice& mesh_device, Program&& program) {
    distributed::MeshWorkload workload;
    workload.add_program(distributed::MeshCoordinateRange(mesh_device.shape()), std::move(program));
    return workload;
}

Program& OnlyProgram(distributed::MeshWorkload& workload) { return workload.get_programs().begin()->second; }

KernelSpec MakeKernel(const std::string& name, const char* source, const std::string& accessor) {
    KernelSpec kernel{
        .unique_id = KernelSpecName{name},
        .source = source,
        .num_threads = 1,
        .hw_config =
            DataMovementHardwareConfig{
                .config_1xx =
                    DataMovementHardwareConfig::DataMovement1XXConfig{
                        .processor = DataMovementProcessor::RISCV_0,
                        .noc = NOC::NOC_0,
                    },
            },
    };
    kernel.global_semaphore_bindings.push_back(GlobalSemaphoreBinding{
        .global_semaphore_parameter_name = GlobalSemaphoreParamName{accessor}, .accessor_name = accessor});
    return kernel;
}

Group<GlobalSemaphoreParameter> OneParameter(const std::string& name) {
    return {GlobalSemaphoreParameter{.unique_id = GlobalSemaphoreParamName{name}}};
}

enum class HubMode { NONE, WAIT_AND_SET, DOWN };

// Each sender up()s the hub's copy kIncrements times. The hub kernel, if any, waits for the total and clears it.
ProgramSpec MakeCounterSpec(HubMode hub_mode) {
    ProgramSpec spec;
    spec.name = "global_semaphore_counter";
    spec.global_semaphore_parameters = OneParameter("counter");

    KernelSpec senders = MakeKernel("senders", kCounterKernel, "counter");
    senders.compiler_options.defines = {{"GLOBAL_SEM_SENDER", "1"}};
    senders.runtime_arg_schema.common_runtime_arg_names = {"hub_x", "hub_y", "increments"};
    spec.kernels.push_back(senders);
    spec.work_units.push_back(
        WorkUnitSpec{.name = "senders", .kernels = {KernelSpecName{"senders"}}, .target_nodes = kSenders});

    if (hub_mode != HubMode::NONE) {
        KernelSpec hub = MakeKernel("hub", kCounterKernel, "counter");
        if (hub_mode == HubMode::DOWN) {
            hub.compiler_options.defines = {{"GLOBAL_SEM_USE_DOWN", "1"}};
        }
        hub.runtime_arg_schema.runtime_arg_names = {"report_addr", "expected"};
        spec.kernels.push_back(hub);
        spec.work_units.push_back(
            WorkUnitSpec{.name = "hub", .kernels = {KernelSpecName{"hub"}}, .target_nodes = kHub});
    }
    return spec;
}

ProgramRunArgs MakeCounterRunArgs(distributed::MeshDevice& mesh_device, HubMode hub_mode, const GlobalSemaphore& sem) {
    const CoreCoord hub_virtual = mesh_device.worker_core_from_logical_core(kHub);
    ProgramRunArgs args;
    args.kernel_run_args.push_back(ProgramRunArgs::KernelRunArgs{
        .kernel = KernelSpecName{"senders"},
        .common_runtime_arg_values =
            {{"hub_x", static_cast<uint32_t>(hub_virtual.x)},
             {"hub_y", static_cast<uint32_t>(hub_virtual.y)},
             {"increments", kIncrements}},
    });
    if (hub_mode != HubMode::NONE) {
        args.kernel_run_args.push_back(ProgramRunArgs::KernelRunArgs{
            .kernel = KernelSpecName{"hub"},
            .runtime_arg_values = MakeRuntimeArgsForSingleNode(
                kHub, {{"report_addr", ReportAddr(mesh_device)}, {"expected", kPerRun}}),
        });
    }
    args.global_semaphore_args = {{GlobalSemaphoreParamName{"counter"}, std::cref(sem)}};
    return args;
}

distributed::MeshWorkload MakeCounterWorkload(
    distributed::MeshDevice& mesh_device, HubMode hub_mode, const GlobalSemaphore& sem) {
    Program program = experimental::MakeProgramFromSpec(mesh_device, MakeCounterSpec(hub_mode));
    experimental::SetProgramRunArgs(program, MakeCounterRunArgs(mesh_device, hub_mode, sem));
    return MakeWorkload(mesh_device, std::move(program));
}

void RunCounterHub(distributed::MeshDevice& mesh_device, HubMode hub_mode) {
    auto sem = CreateGlobalSemaphore(mesh_device, kHubAndSenders, 0);
    auto workload = MakeCounterWorkload(mesh_device, hub_mode, sem);

    const uint32_t report_addr = ReportAddr(mesh_device);
    std::vector<uint32_t> sentinel{0xFFFFFFFF, 0xFFFFFFFF};
    slow_dispatch::WriteToL1(mesh_device, kHub, report_addr, sentinel);

    distributed::EnqueueMeshWorkload(mesh_device.mesh_command_queue(), workload, /*blocking=*/true);

    std::vector<uint32_t> report;
    slow_dispatch::ReadFromL1(mesh_device, kHub, report_addr, 2 * sizeof(uint32_t), report);
    ASSERT_EQ(report.size(), 2u);
    EXPECT_EQ(report[0], kPerRun) << "hub saw the wrong total";
    EXPECT_EQ(report[1], 0u) << "hub did not drain the semaphore";
    EXPECT_EQ(ReadWord(mesh_device, kHub, sem.address()), 0u);
}

TEST_F(GlobalSemaphoreMetal2Test, ManyToOneUpThenWaitAndSet) { RunCounterHub(*devices_.at(0), HubMode::WAIT_AND_SET); }

TEST_F(GlobalSemaphoreMetal2Test, ManyToOneUpThenDown) { RunCounterHub(*devices_.at(0), HubMode::DOWN); }

// The semaphore outlives the program: a second program keeps counting from where the first stopped.
TEST_F(GlobalSemaphoreMetal2Test, ValuePersistsAcrossPrograms) {
    auto& mesh_device = *devices_.at(0);
    auto sem = CreateGlobalSemaphore(mesh_device, kHubAndSenders, 0);
    for (int run = 0; run < 2; run++) {
        auto workload = MakeCounterWorkload(mesh_device, HubMode::NONE, sem);
        distributed::EnqueueMeshWorkload(mesh_device.mesh_command_queue(), workload, /*blocking=*/true);
    }
    EXPECT_EQ(ReadWord(mesh_device, kHub, sem.address()), 2 * kPerRun);
}

// The same compiled program counts into a different semaphore after UpdateProgramRunArgs supplies only the
// new semaphore.
TEST_F(GlobalSemaphoreMetal2Test, UpdateProgramRunArgsSwitchesSemaphore) {
    auto& mesh_device = *devices_.at(0);
    auto& cq = mesh_device.mesh_command_queue();
    auto first = CreateGlobalSemaphore(mesh_device, kHubAndSenders, 0);
    auto second = CreateGlobalSemaphore(mesh_device, kHubAndSenders, 0);
    ASSERT_NE(first.address(), second.address());

    auto workload = MakeCounterWorkload(mesh_device, HubMode::NONE, first);
    distributed::EnqueueMeshWorkload(cq, workload, /*blocking=*/true);

    ProgramRunArgs update;
    update.global_semaphore_args = {{GlobalSemaphoreParamName{"counter"}, std::cref(second)}};
    experimental::UpdateProgramRunArgs(OnlyProgram(workload), update);
    distributed::EnqueueMeshWorkload(cq, workload, /*blocking=*/true);

    EXPECT_EQ(ReadWord(mesh_device, kHub, first.address()), kPerRun);
    EXPECT_EQ(ReadWord(mesh_device, kHub, second.address()), kPerRun);
}

// A trace keeps the address it was captured with; a later update only affects non-traced launches.
TEST_F(GlobalSemaphoreMetal2TraceTest, TraceReplayUsesCaptureTimeAddress) {
    auto& mesh_device = *devices_.at(0);
    auto& cq = mesh_device.mesh_command_queue();
    auto first = CreateGlobalSemaphore(mesh_device, kHubAndSenders, 0);
    auto second = CreateGlobalSemaphore(mesh_device, kHubAndSenders, 0);

    auto workload = MakeCounterWorkload(mesh_device, HubMode::NONE, first);
    distributed::EnqueueMeshWorkload(cq, workload, /*blocking=*/true);  // warmup: capture cannot load binaries

    const auto tid = distributed::BeginTraceCapture(&mesh_device, cq.id());
    distributed::EnqueueMeshWorkload(cq, workload, /*blocking=*/false);
    mesh_device.end_mesh_trace(cq.id(), tid);

    ProgramRunArgs update;
    update.global_semaphore_args = {{GlobalSemaphoreParamName{"counter"}, std::cref(second)}};
    experimental::UpdateProgramRunArgs(OnlyProgram(workload), update);
    mesh_device.replay_mesh_trace(cq.id(), tid, /*blocking=*/true);

    EXPECT_EQ(ReadWord(mesh_device, kHub, first.address()), 2 * kPerRun);
    EXPECT_EQ(ReadWord(mesh_device, kHub, second.address()), 0u);

    distributed::EnqueueMeshWorkload(cq, workload, /*blocking=*/true);
    EXPECT_EQ(ReadWord(mesh_device, kHub, second.address()), kPerRun);
    mesh_device.release_mesh_trace(tid);
}

// One core multicasts to the rectangle (1,0)-(3,1); the sender is outside it.
void RunMcast(distributed::MeshDevice& mesh_device, bool increment, uint32_t initial, uint32_t value) {
    const CoreRange dests({1, 0}, {3, 1});
    const CoreRangeSet all(std::vector<CoreRange>{CoreRange(kHub, kHub), dests});
    auto sem = CreateGlobalSemaphore(mesh_device, all, initial);

    ProgramSpec spec;
    spec.name = "global_semaphore_mcast";
    spec.global_semaphore_parameters = OneParameter("flag");
    KernelSpec kernel = MakeKernel("mcast", kMcastKernel, "flag");
    if (increment) {
        kernel.compiler_options.defines = {{"GLOBAL_SEM_INC", "1"}};
    }
    kernel.runtime_arg_schema.runtime_arg_names = {"start_x", "start_y", "end_x", "end_y", "num_dests", "value"};
    spec.kernels.push_back(kernel);
    spec.work_units.push_back(
        WorkUnitSpec{.name = "mcast", .kernels = {KernelSpecName{"mcast"}}, .target_nodes = kHub});

    const CoreCoord start = mesh_device.worker_core_from_logical_core(dests.start_coord);
    const CoreCoord end = mesh_device.worker_core_from_logical_core(dests.end_coord);
    ProgramRunArgs args;
    args.kernel_run_args.push_back(ProgramRunArgs::KernelRunArgs{
        .kernel = KernelSpecName{"mcast"},
        .runtime_arg_values = MakeRuntimeArgsForSingleNode(
            kHub,
            {{"start_x", static_cast<uint32_t>(start.x)},
             {"start_y", static_cast<uint32_t>(start.y)},
             {"end_x", static_cast<uint32_t>(end.x)},
             {"end_y", static_cast<uint32_t>(end.y)},
             {"num_dests", static_cast<uint32_t>(dests.size())},
             {"value", value}}),
    });
    args.global_semaphore_args = {{GlobalSemaphoreParamName{"flag"}, std::cref(sem)}};

    Program program = experimental::MakeProgramFromSpec(mesh_device, spec);
    experimental::SetProgramRunArgs(program, args);
    auto workload = MakeWorkload(mesh_device, std::move(program));
    distributed::EnqueueMeshWorkload(mesh_device.mesh_command_queue(), workload, /*blocking=*/true);

    const uint32_t expected_dest = increment ? initial + 2 * value : value;
    const uint32_t expected_sender = increment ? initial : value;
    for (const auto& core : corerange_to_cores(CoreRangeSet(dests))) {
        EXPECT_EQ(ReadWord(mesh_device, core, sem.address()), expected_dest) << core.str();
    }
    EXPECT_EQ(ReadWord(mesh_device, kHub, sem.address()), expected_sender);
}

TEST_F(GlobalSemaphoreMetal2Test, SetMulticastCopiesValue) { RunMcast(*devices_.at(0), /*increment=*/false, 0, 0xA5); }

TEST_F(GlobalSemaphoreMetal2Test, IncMulticastAddsToEveryDest) { RunMcast(*devices_.at(0), /*increment=*/true, 5, 3); }

// The host flips the semaphore while the kernel is polling value().
TEST_F(GlobalSemaphoreMetal2Test, StopFlagSeenThroughValue) {
    auto& mesh_device = *devices_.at(0);
    auto sem = CreateGlobalSemaphore(mesh_device, CoreRangeSet(CoreRange(kHub)), 0);

    ProgramSpec spec;
    spec.name = "global_semaphore_stop_flag";
    spec.global_semaphore_parameters = OneParameter("stop");
    KernelSpec kernel = MakeKernel("stop_flag", kStopFlagKernel, "stop");
    kernel.runtime_arg_schema.runtime_arg_names = {"report_addr"};
    spec.kernels.push_back(kernel);
    spec.work_units.push_back(
        WorkUnitSpec{.name = "stop_flag", .kernels = {KernelSpecName{"stop_flag"}}, .target_nodes = kHub});

    ProgramRunArgs args;
    args.kernel_run_args.push_back(ProgramRunArgs::KernelRunArgs{
        .kernel = KernelSpecName{"stop_flag"},
        .runtime_arg_values =
            MakeRuntimeArgsForSingleNode(kHub, {{"report_addr", ReportAddr(mesh_device)}}),
    });
    args.global_semaphore_args = {{GlobalSemaphoreParamName{"stop"}, std::cref(sem)}};

    Program program = experimental::MakeProgramFromSpec(mesh_device, spec);
    experimental::SetProgramRunArgs(program, args);
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

// ---------------------------------------------------------------------------------------------------------------
// Quasar (simulator, slow dispatch)
// ---------------------------------------------------------------------------------------------------------------

constexpr uint32_t kQuasarSenderThreads = 4;
constexpr uint32_t kQuasarIncrements = 4;

// Up to two worker nodes; the first is the hub. The sender kernel runs on every node with several threads, so
// increments from different threads and nodes arrive at the hub concurrently.
std::vector<CoreCoord> QuasarNodes(const distributed::MeshDevice& mesh_device) {
    const CoreCoord grid = mesh_device.compute_with_storage_grid_size();
    std::vector<CoreCoord> nodes{CoreCoord{0, 0}};
    if (grid.x > 1) {
        nodes.push_back(CoreCoord{1, 0});
    } else if (grid.y > 1) {
        nodes.push_back(CoreCoord{0, 1});
    }
    return nodes;
}

CoreRangeSet ToCoreRangeSet(const std::vector<CoreCoord>& nodes) {
    std::vector<CoreRange> ranges;
    for (const auto& node : nodes) {
        ranges.emplace_back(node, node);
    }
    return CoreRangeSet(ranges);
}

KernelSpec MakeQuasarKernel(
    const std::string& name, const char* source, const std::string& accessor, uint32_t num_threads) {
    KernelSpec kernel{
        .unique_id = KernelSpecName{name},
        .source = source,
        .num_threads = num_threads,
        .hw_config = DataMovementHardwareConfig{},
    };
    kernel.global_semaphore_bindings.push_back(GlobalSemaphoreBinding{
        .global_semaphore_parameter_name = GlobalSemaphoreParamName{accessor}, .accessor_name = accessor});
    return kernel;
}

Program MakeQuasarCounterProgram(
    distributed::MeshDevice& mesh_device,
    const std::vector<CoreCoord>& nodes,
    bool with_hub,
    const GlobalSemaphore& sem) {
    ProgramSpec spec;
    spec.name = "global_semaphore_counter_quasar";
    spec.global_semaphore_parameters = OneParameter("counter");

    KernelSpec senders = MakeQuasarKernel("senders", kCounterKernel, "counter", kQuasarSenderThreads);
    senders.compiler_options.defines = {{"GLOBAL_SEM_SENDER", "1"}};
    senders.runtime_arg_schema.common_runtime_arg_names = {"hub_x", "hub_y", "increments"};
    spec.kernels.push_back(senders);
    if (with_hub) {
        KernelSpec hub = MakeQuasarKernel("hub", kCounterKernel, "counter", 1);
        hub.compiler_options.defines = {{"GLOBAL_SEM_USE_DOWN", "1"}};
        hub.runtime_arg_schema.runtime_arg_names = {"report_addr", "expected"};
        spec.kernels.push_back(hub);
    }
    for (size_t i = 0; i < nodes.size(); i++) {
        std::vector<KernelSpecName> kernels{KernelSpecName{"senders"}};
        if (with_hub && i == 0) {
            kernels.push_back(KernelSpecName{"hub"});
        }
        spec.work_units.push_back(
            WorkUnitSpec{.name = "node_" + std::to_string(i), .kernels = kernels, .target_nodes = nodes[i]});
    }

    const CoreCoord hub_virtual = mesh_device.worker_core_from_logical_core(nodes[0]);
    const uint32_t total = static_cast<uint32_t>(nodes.size()) * kQuasarSenderThreads * kQuasarIncrements;
    ProgramRunArgs args;
    args.kernel_run_args.push_back(ProgramRunArgs::KernelRunArgs{
        .kernel = KernelSpecName{"senders"},
        .common_runtime_arg_values =
            {{"hub_x", static_cast<uint32_t>(hub_virtual.x)},
             {"hub_y", static_cast<uint32_t>(hub_virtual.y)},
             {"increments", kQuasarIncrements}},
    });
    if (with_hub) {
        args.kernel_run_args.push_back(ProgramRunArgs::KernelRunArgs{
            .kernel = KernelSpecName{"hub"},
            .runtime_arg_values = MakeRuntimeArgsForSingleNode(
                nodes[0], {{"report_addr", ReportAddr(mesh_device)}, {"expected", total}}),
        });
    }
    args.global_semaphore_args = {{GlobalSemaphoreParamName{"counter"}, std::cref(sem)}};

    Program program = experimental::MakeProgramFromSpec(mesh_device, spec);
    experimental::SetProgramRunArgs(program, args);
    return program;
}

class GlobalSemaphoreQuasarTest : public QuasarMeshDeviceSingleCardFixture {};

// Several threads up() the hub concurrently; one hub thread down()s the total.
TEST_F(GlobalSemaphoreQuasarTest, ManyToOneUpThenDown) {
    auto& mesh_device = *devices_.at(0);
    const auto nodes = QuasarNodes(mesh_device);
    auto sem = CreateGlobalSemaphore(mesh_device, ToCoreRangeSet(nodes), 0);
    const uint32_t total = static_cast<uint32_t>(nodes.size()) * kQuasarSenderThreads * kQuasarIncrements;

    const uint32_t report_addr = ReportAddr(mesh_device);
    std::vector<uint32_t> sentinel{0xFFFFFFFF, 0xFFFFFFFF};
    slow_dispatch::WriteToL1(mesh_device, nodes[0], report_addr, sentinel);

    auto workload = MakeWorkload(mesh_device, MakeQuasarCounterProgram(mesh_device, nodes, /*with_hub=*/true, sem));
    distributed::EnqueueMeshWorkload(mesh_device.mesh_command_queue(), workload, /*blocking=*/true);

    std::vector<uint32_t> report;
    slow_dispatch::ReadFromL1(mesh_device, nodes[0], report_addr, 2 * sizeof(uint32_t), report);
    ASSERT_EQ(report.size(), 2u);
    EXPECT_EQ(report[0], total) << "hub saw the wrong total";
    EXPECT_EQ(report[1], 0u) << "hub did not drain the semaphore";
}

// The value persists across programs, and UpdateProgramRunArgs switches the program to another semaphore.
TEST_F(GlobalSemaphoreQuasarTest, ValuePersistsAndUpdateSwitchesSemaphore) {
    auto& mesh_device = *devices_.at(0);
    auto& cq = mesh_device.mesh_command_queue();
    const auto nodes = QuasarNodes(mesh_device);
    auto first = CreateGlobalSemaphore(mesh_device, ToCoreRangeSet(nodes), 0);
    auto second = CreateGlobalSemaphore(mesh_device, ToCoreRangeSet(nodes), 0);
    const uint32_t per_run = static_cast<uint32_t>(nodes.size()) * kQuasarSenderThreads * kQuasarIncrements;

    auto workload = MakeWorkload(mesh_device, MakeQuasarCounterProgram(mesh_device, nodes, /*with_hub=*/false, first));
    distributed::EnqueueMeshWorkload(cq, workload, /*blocking=*/true);
    distributed::EnqueueMeshWorkload(cq, workload, /*blocking=*/true);

    ProgramRunArgs update;
    update.global_semaphore_args = {{GlobalSemaphoreParamName{"counter"}, std::cref(second)}};
    experimental::UpdateProgramRunArgs(OnlyProgram(workload), update);
    distributed::EnqueueMeshWorkload(cq, workload, /*blocking=*/true);

    EXPECT_EQ(ReadWord(mesh_device, nodes[0], first.address()), 2 * per_run);
    EXPECT_EQ(ReadWord(mesh_device, nodes[0], second.address()), per_run);
}

// The hub sets its copy and multicasts it to the second node, then increments it there twice by multicast.
TEST_F(GlobalSemaphoreQuasarTest, MulticastSetAndInc) {
    auto& mesh_device = *devices_.at(0);
    const auto nodes = QuasarNodes(mesh_device);
    if (nodes.size() < 2) {
        GTEST_SKIP() << "Skipping: test needs two worker nodes";
    }
    const CoreCoord dest = mesh_device.worker_core_from_logical_core(nodes[1]);
    for (const bool increment : {false, true}) {
        auto sem = CreateGlobalSemaphore(mesh_device, ToCoreRangeSet(nodes), 5);
        ProgramSpec spec;
        spec.name = "global_semaphore_mcast_quasar";
        spec.global_semaphore_parameters = OneParameter("flag");
        KernelSpec kernel = MakeQuasarKernel("mcast", kMcastKernel, "flag", 1);
        if (increment) {
            kernel.compiler_options.defines = {{"GLOBAL_SEM_INC", "1"}};
        }
        kernel.runtime_arg_schema.runtime_arg_names = {"start_x", "start_y", "end_x", "end_y", "num_dests", "value"};
        spec.kernels.push_back(kernel);
        spec.work_units.push_back(
            WorkUnitSpec{.name = "mcast", .kernels = {KernelSpecName{"mcast"}}, .target_nodes = nodes[0]});

        ProgramRunArgs args;
        args.kernel_run_args.push_back(ProgramRunArgs::KernelRunArgs{
            .kernel = KernelSpecName{"mcast"},
            .runtime_arg_values = MakeRuntimeArgsForSingleNode(
                nodes[0],
                {{"start_x", static_cast<uint32_t>(dest.x)},
                 {"start_y", static_cast<uint32_t>(dest.y)},
                 {"end_x", static_cast<uint32_t>(dest.x)},
                 {"end_y", static_cast<uint32_t>(dest.y)},
                 {"num_dests", 1u},
                 {"value", 3u}}),
        });
        args.global_semaphore_args = {{GlobalSemaphoreParamName{"flag"}, std::cref(sem)}};
        Program program = experimental::MakeProgramFromSpec(mesh_device, spec);
        experimental::SetProgramRunArgs(program, args);
        auto workload = MakeWorkload(mesh_device, std::move(program));
        distributed::EnqueueMeshWorkload(mesh_device.mesh_command_queue(), workload, /*blocking=*/true);

        EXPECT_EQ(ReadWord(mesh_device, nodes[1], sem.address()), increment ? 5u + 2 * 3u : 3u)
            << (increment ? "inc_multicast" : "set_multicast");
        EXPECT_EQ(ReadWord(mesh_device, nodes[0], sem.address()), increment ? 5u : 3u);
    }
}

// ---------------------------------------------------------------------------------------------------------------
// ProgramRunArgs validation
// ---------------------------------------------------------------------------------------------------------------

TEST_F(GlobalSemaphoreMetal2Test, SetProgramRunArgsWithoutTheSemaphoreFails) {
    auto& mesh_device = *devices_.at(0);
    auto sem = CreateGlobalSemaphore(mesh_device, kHubAndSenders, 0);
    Program program = experimental::MakeProgramFromSpec(mesh_device, MakeCounterSpec(HubMode::NONE));
    ProgramRunArgs args = MakeCounterRunArgs(mesh_device, HubMode::NONE, sem);
    args.global_semaphore_args = {};
    EXPECT_THAT(
        [&] { experimental::SetProgramRunArgs(program, args); },
        ThrowsMessage<std::runtime_error>(HasSubstr("has no GlobalSemaphoreArgument entry")));
}

TEST_F(GlobalSemaphoreMetal2Test, UnknownGlobalSemaphoreArgFails) {
    auto& mesh_device = *devices_.at(0);
    auto sem = CreateGlobalSemaphore(mesh_device, kHubAndSenders, 0);
    Program program = experimental::MakeProgramFromSpec(mesh_device, MakeCounterSpec(HubMode::NONE));
    ProgramRunArgs args = MakeCounterRunArgs(mesh_device, HubMode::NONE, sem);
    args.global_semaphore_args.insert({GlobalSemaphoreParamName{"other"}, std::cref(sem)});
    EXPECT_THAT(
        [&] { experimental::SetProgramRunArgs(program, args); },
        ThrowsMessage<std::runtime_error>(HasSubstr("unknown GlobalSemaphoreParameter 'other'")));

    experimental::SetProgramRunArgs(program, MakeCounterRunArgs(mesh_device, HubMode::NONE, sem));
    ProgramRunArgs update;
    update.global_semaphore_args = {{GlobalSemaphoreParamName{"other"}, std::cref(sem)}};
    EXPECT_THAT(
        [&] { experimental::UpdateProgramRunArgs(program, update); },
        ThrowsMessage<std::runtime_error>(HasSubstr("unknown GlobalSemaphoreParameter 'other'")));
}

TEST_F(GlobalSemaphoreMetal2Test, SemaphoreFromAnotherMeshDeviceFails) {
    if (devices_.size() < 2) {
        GTEST_SKIP() << "Skipping: test needs two mesh devices";
    }
    auto& mesh_device = *devices_.at(0);
    auto other_sem = CreateGlobalSemaphore(*devices_.at(1), kHubAndSenders, 0);
    Program program = experimental::MakeProgramFromSpec(mesh_device, MakeCounterSpec(HubMode::NONE));
    EXPECT_THAT(
        [&] { experimental::SetProgramRunArgs(program, MakeCounterRunArgs(mesh_device, HubMode::NONE, other_sem)); },
        ThrowsMessage<std::runtime_error>(HasSubstr("created on a different MeshDevice")));
}

TEST_F(GlobalSemaphoreMetal2Test, MergeProgramRunArgsRejectsTheSameSemaphoreTwice) {
    auto& mesh_device = *devices_.at(0);
    auto sem = CreateGlobalSemaphore(mesh_device, kHubAndSenders, 0);
    ProgramRunArgs base;
    base.global_semaphore_args = {{GlobalSemaphoreParamName{"counter"}, std::cref(sem)}};
    const std::vector<ProgramRunArgs> rest{base};
    EXPECT_THAT(
        [&] { (void)MergeProgramRunArgs(base, rest); },
        ThrowsMessage<std::runtime_error>(HasSubstr("specified in more than one ProgramRunArgs")));
}

}  // namespace
}  // namespace tt::tt_metal
