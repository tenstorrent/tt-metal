// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// GlobalSemaphore up() and write_and_up() over fabric between two neighbouring chips, on a 1D and a 2D fabric.
// The receiver's kernel down()s the total that the sender's kernel sends. The receiver is built either with the
// original host API or from a Metal 2.0 ProgramSpec that binds the semaphore; the sender always uses the original
// API, which is the one that sets up the fabric connection.

#include <gtest/gtest.h>
#include <array>
#include <cstdint>
#include <functional>
#include <map>
#include <string>
#include <vector>

#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/experimental/fabric/control_plane.hpp>
#include <tt-metalium/experimental/fabric/fabric.hpp>
#include <tt-metalium/experimental/metal2_host_api/program.hpp>
#include <tt-metalium/global_semaphore.hpp>
#include <tt-metalium/hal.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/kernel_types.hpp>
#include <tt-metalium/workload/run_args/program_run_args.hpp>
#include <tt-metalium/workload/specification/program_spec.hpp>

#include "fabric_fixture.hpp"
#include "utils.hpp"
#include "impl/context/metal_context.hpp"

namespace tt::tt_fabric::fabric_router_tests {
namespace {

using tt::tt_metal::CoreCoord;
using tt::tt_metal::CoreRange;
using tt::tt_metal::CoreRangeSet;

constexpr const char* kSenderKernel =
    "tests/tt_metal/tt_metal/test_kernels/dataflow/global_semaphore_fabric_sender.cpp";
// Runtime args {report_addr, expected}; the semaphore's address is common runtime arg 0.
constexpr const char* kReceiverKernelSource = R"(
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/global_semaphore.h"

void kernel_main() {
    const uint32_t report_addr = get_arg_val<uint32_t>(0);
    const uint32_t expected = get_arg_val<uint32_t>(1);
    GlobalSemaphore sem(GlobalSemaphoreBindingToken{0});
    sem.down(expected);
    volatile tt_l1_ptr uint32_t* report = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(report_addr);
    report[0] = expected;
    report[1] = sem.value();
}
)";
// The same receiver as a Metal 2.0 kernel, with the semaphore bound as global_sem::remote.
constexpr const char* kMetal2ReceiverKernelSource = R"(
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/global_semaphore.h"
#include "experimental/kernel_args.h"

void kernel_main() {
    const uint32_t report_addr = get_arg(args::report_addr);
    const uint32_t expected = get_arg(args::expected);
    GlobalSemaphore sem(global_sem::remote);
    sem.down(expected);
    volatile tt_l1_ptr uint32_t* report = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(report_addr);
    report[0] = expected;
    report[1] = sem.value();
}
)";
constexpr uint32_t kIncrements = 32;
constexpr uint32_t kPayloadBytes = 256;

enum class ReceiverApi { ORIGINAL, METAL2 };
enum class SendOp { UP, WRITE_AND_UP };

uint32_t ReadWord(tt::tt_metal::distributed::MeshDevice& mesh_device, const CoreCoord& core, uint32_t address) {
    tt::tt_metal::IDevice* device = mesh_device.get_devices().at(0);
    const auto words = tt::tt_metal::MetalContext::instance().get_cluster().read_core(
        device->id(), device->worker_core_from_logical_core(core), address, sizeof(uint32_t));
    return words.at(0);
}

tt::tt_metal::Program MakeOriginalReceiver(
    const CoreCoord& core, uint32_t sem_address, uint32_t report_addr, uint32_t expected) {
    auto program = tt::tt_metal::CreateProgram();
    const auto receiver = tt::tt_metal::CreateKernelFromString(
        program,
        kReceiverKernelSource,
        core,
        tt::tt_metal::DataMovementConfig{
            .processor = tt::tt_metal::DataMovementProcessor::RISCV_0, .noc = tt::tt_metal::NOC::RISCV_0_default});
    tt::tt_metal::SetRuntimeArgs(program, receiver, core, {report_addr, expected});
    tt::tt_metal::SetCommonRuntimeArgs(program, receiver, {sem_address});
    return program;
}

tt::tt_metal::Program MakeMetal2Receiver(
    tt::tt_metal::distributed::MeshDevice& mesh_device,
    const CoreCoord& core,
    const tt::tt_metal::GlobalSemaphore& sem,
    uint32_t report_addr,
    uint32_t expected) {
    using namespace tt::tt_metal;
    KernelSpec receiver{
        .unique_id = KernelSpecName{"receiver"},
        .source = KernelSpec::SourceCode{kMetal2ReceiverKernelSource},
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
    receiver.runtime_arg_schema.runtime_arg_names = {"report_addr", "expected"};
    receiver.global_semaphore_bindings.push_back(GlobalSemaphoreBinding{
        .global_semaphore_parameter_name = GlobalSemaphoreParamName{"remote"}, .accessor_name = "remote"});

    ProgramSpec spec;
    spec.name = "global_semaphore_fabric_receiver";
    spec.global_semaphore_parameters = {GlobalSemaphoreParameter{.unique_id = GlobalSemaphoreParamName{"remote"}}};
    spec.kernels.push_back(receiver);
    spec.work_units.push_back(
        WorkUnitSpec{.name = "receiver", .kernels = {KernelSpecName{"receiver"}}, .target_nodes = core});

    ProgramRunArgs args;
    args.kernel_run_args.push_back(ProgramRunArgs::KernelRunArgs{
        .kernel = KernelSpecName{"receiver"},
        .runtime_arg_values =
            MakeRuntimeArgsForSingleNode(core, {{"report_addr", report_addr}, {"expected", expected}}),
    });
    args.global_semaphore_args = {{GlobalSemaphoreParamName{"remote"}, std::cref(sem)}};

    Program program = tt::tt_metal::experimental::MakeProgramFromSpec(mesh_device, spec);
    tt::tt_metal::experimental::SetProgramRunArgs(program, args);
    return program;
}

void RunGlobalSemaphoreOverFabric(BaseFabricFixture* fixture, bool is_2d, ReceiverApi receiver_api, SendOp op) {
    FabricNodeId src_node(MeshId{0}, 0);
    FabricNodeId dst_node(MeshId{0}, 0);
    ChipId src_chip = 0;
    ChipId dst_chip = 0;
    bool found = false;
    for (const auto direction : {RoutingDirection::E, RoutingDirection::W, RoutingDirection::N, RoutingDirection::S}) {
        if (find_device_with_neighbor_in_direction(fixture, src_node, dst_node, src_chip, dst_chip, direction)) {
            found = true;
            break;
        }
    }
    if (!found) {
        GTEST_SKIP() << "No pair of neighbouring chips found";
    }

    const auto& src_device = fixture->get_device(src_chip);
    const auto& dst_device = fixture->get_device(dst_chip);

    const CoreCoord core{0, 0};
    auto sem = tt::tt_metal::CreateGlobalSemaphore(*dst_device, CoreRangeSet(CoreRange(core, core)), 0);
    const uint32_t sem_address = static_cast<uint32_t>(sem.address());
    const uint32_t report_addr =
        dst_device->allocator()->get_base_allocator_addr(tt::tt_metal::HalMemType::L1) + 32 * 1024;
    const uint32_t payload_addr = report_addr + 1024;

    if (op == SendOp::WRITE_AND_UP) {
        tt::tt_metal::IDevice* dst = dst_device->get_devices().at(0);
        tt::tt_metal::MetalContext::instance().get_cluster().write_core(
            dst->id(),
            dst->worker_core_from_logical_core(core),
            std::vector<uint32_t>(kPayloadBytes / sizeof(uint32_t), 0),
            payload_addr);
    }

    auto receiver_program = receiver_api == ReceiverApi::METAL2
                                ? MakeMetal2Receiver(*dst_device, core, sem, report_addr, kIncrements)
                                : MakeOriginalReceiver(core, sem_address, report_addr, kIncrements);

    std::map<std::string, std::string> sender_defines;
    if (is_2d) {
        sender_defines.emplace("FABRIC_2D", "1");
    }
    if (op == SendOp::WRITE_AND_UP) {
        sender_defines.emplace("GLOBAL_SEM_WRITE_AND_UP", "1");
    }
    auto sender_program = tt::tt_metal::CreateProgram();
    const auto sender = tt::tt_metal::CreateKernel(
        sender_program,
        kSenderKernel,
        core,
        tt::tt_metal::DataMovementConfig{
            .processor = tt::tt_metal::DataMovementProcessor::RISCV_0,
            .noc = tt::tt_metal::NOC::RISCV_0_default,
            .defines = sender_defines});
    const CoreCoord dst_virtual = dst_device->worker_core_from_logical_core(core);
    std::vector<uint32_t> sender_args{
        static_cast<uint32_t>(dst_virtual.x), static_cast<uint32_t>(dst_virtual.y), kIncrements};
    if (op == SendOp::WRITE_AND_UP) {
        sender_args.insert(sender_args.end(), {payload_addr, payload_addr, kPayloadBytes});
    }
    if (is_2d) {
        sender_args.push_back(dst_node.chip_id);
        sender_args.push_back(*dst_node.mesh_id);
    } else {
        sender_args.push_back(1);  // num_hops
    }
    append_fabric_connection_rt_args(*src_device, src_node, dst_node, 0, sender_program, core, sender_args);
    tt::tt_metal::SetRuntimeArgs(sender_program, sender, core, sender_args);
    tt::tt_metal::SetCommonRuntimeArgs(sender_program, sender, {sem_address});

    fixture->RunProgramNonblocking(dst_device, std::move(receiver_program));
    fixture->RunProgramNonblocking(src_device, std::move(sender_program));
    fixture->WaitForSingleProgramDone(src_device);
    fixture->WaitForSingleProgramDone(dst_device);

    EXPECT_EQ(ReadWord(*dst_device, core, report_addr), kIncrements) << "receiver did not see every increment";
    EXPECT_EQ(ReadWord(*dst_device, core, report_addr + sizeof(uint32_t)), 0u) << "receiver did not drain it";
    EXPECT_EQ(ReadWord(*dst_device, core, sem_address), 0u);
    if (op == SendOp::WRITE_AND_UP) {
        for (uint32_t i = 0; i < kPayloadBytes / sizeof(uint32_t); i++) {
            EXPECT_EQ(ReadWord(*dst_device, core, payload_addr + (i * sizeof(uint32_t))), 0xA5000000u + i)
                << "payload word " << i;
        }
    }
}

TEST_F(Fabric1DFixture, GlobalSemaphoreUpOverFabric) {
    RunGlobalSemaphoreOverFabric(this, /*is_2d=*/false, ReceiverApi::ORIGINAL, SendOp::UP);
}

TEST_F(Fabric2DFixture, GlobalSemaphoreUpOverFabric) {
    RunGlobalSemaphoreOverFabric(this, /*is_2d=*/true, ReceiverApi::ORIGINAL, SendOp::UP);
}

TEST_F(Fabric1DFixture, GlobalSemaphoreMetal2ReceiverUpOverFabric) {
    RunGlobalSemaphoreOverFabric(this, /*is_2d=*/false, ReceiverApi::METAL2, SendOp::UP);
}

TEST_F(Fabric2DFixture, GlobalSemaphoreMetal2ReceiverUpOverFabric) {
    RunGlobalSemaphoreOverFabric(this, /*is_2d=*/true, ReceiverApi::METAL2, SendOp::UP);
}

TEST_F(Fabric1DFixture, GlobalSemaphoreMetal2ReceiverWriteAndUpOverFabric) {
    RunGlobalSemaphoreOverFabric(this, /*is_2d=*/false, ReceiverApi::METAL2, SendOp::WRITE_AND_UP);
}

TEST_F(Fabric2DFixture, GlobalSemaphoreMetal2ReceiverWriteAndUpOverFabric) {
    RunGlobalSemaphoreOverFabric(this, /*is_2d=*/true, ReceiverApi::METAL2, SendOp::WRITE_AND_UP);
}

}  // namespace
}  // namespace tt::tt_fabric::fabric_router_tests
