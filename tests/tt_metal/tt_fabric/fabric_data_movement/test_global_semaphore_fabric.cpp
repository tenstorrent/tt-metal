// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// GlobalSemaphore up() over fabric between two neighbouring chips, on a 1D and a 2D fabric. The receiver's
// kernel down()s the total that the sender's kernel sends.

#include <gtest/gtest.h>
#include <array>
#include <cstdint>
#include <map>
#include <string>
#include <vector>

#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/experimental/fabric/control_plane.hpp>
#include <tt-metalium/experimental/fabric/fabric.hpp>
#include <tt-metalium/global_semaphore.hpp>
#include <tt-metalium/hal.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/kernel_types.hpp>

#include "fabric_fixture.hpp"
#include "utils.hpp"
#include "distributed/mesh_device_impl.hpp"
#include "impl/context/metal_context.hpp"

namespace tt::tt_fabric::fabric_router_tests {
namespace {

using tt::tt_metal::CoreCoord;
using tt::tt_metal::CoreRange;
using tt::tt_metal::CoreRangeSet;

constexpr const char* kSenderKernel =
    "tests/tt_metal/tt_metal/test_kernels/dataflow/global_semaphore_fabric_sender.cpp";
constexpr const char* kCounterKernel = "tests/tt_metal/tt_metal/test_kernels/dataflow/global_semaphore_counter.cpp";
constexpr uint32_t kIncrements = 32;

uint32_t ReadWord(tt::tt_metal::distributed::MeshDevice& mesh_device, const CoreCoord& core, uint32_t address) {
    tt::tt_metal::IDevice* device = mesh_device.get_devices().at(0);
    const auto words = tt::tt_metal::MetalContext::instance().get_cluster().read_core(
        device->id(), device->worker_core_from_logical_core(core), address, sizeof(uint32_t));
    return words.at(0);
}

void RunGlobalSemaphoreUpOverFabric(BaseFabricFixture* fixture, bool is_2d) {
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
    tt::tt_metal::experimental::ReserveGlobalSemaphorePool(*dst_device, 16);
    if (dst_device->impl().global_semaphore_pool() == nullptr) {
        GTEST_SKIP() << "Skipping: TT_METAL_GLOBAL_SEMAPHORE_POOL_SLOTS=0";
    }

    const CoreCoord core{0, 0};
    auto sem = tt::tt_metal::CreateGlobalSemaphore(*dst_device, CoreRangeSet(CoreRange(core, core)));
    const uint32_t sem_address = static_cast<uint32_t>(sem.address());
    const uint32_t report_addr =
        dst_device->allocator()->get_base_allocator_addr(tt::tt_metal::HalMemType::L1) + 32 * 1024;

    std::map<std::string, std::string> receiver_defines{{"GLOBAL_SEM_USE_DOWN", "1"}};
    auto receiver_program = tt::tt_metal::CreateProgram();
    const auto receiver = tt::tt_metal::CreateKernel(
        receiver_program,
        kCounterKernel,
        core,
        tt::tt_metal::DataMovementConfig{
            .processor = tt::tt_metal::DataMovementProcessor::RISCV_0,
            .noc = tt::tt_metal::NOC::RISCV_0_default,
            .defines = receiver_defines});
    tt::tt_metal::SetRuntimeArgs(receiver_program, receiver, core, {report_addr, kIncrements});
    tt::tt_metal::SetCommonRuntimeArgs(receiver_program, receiver, {sem_address});

    std::map<std::string, std::string> sender_defines;
    if (is_2d) {
        sender_defines.emplace("FABRIC_2D", "1");
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
    if (is_2d) {
        sender_args.push_back(dst_node.chip_id);
        sender_args.push_back(*dst_node.mesh_id);
    } else {
        sender_args.push_back(1);  // num_hops
    }
    append_fabric_connection_rt_args(src_node, dst_node, 0, sender_program, core, sender_args);
    tt::tt_metal::SetRuntimeArgs(sender_program, sender, core, sender_args);
    tt::tt_metal::SetCommonRuntimeArgs(sender_program, sender, {sem_address});

    fixture->RunProgramNonblocking(dst_device, std::move(receiver_program));
    fixture->RunProgramNonblocking(src_device, std::move(sender_program));
    fixture->WaitForSingleProgramDone(src_device);
    fixture->WaitForSingleProgramDone(dst_device);

    EXPECT_EQ(ReadWord(*dst_device, core, report_addr), kIncrements) << "receiver did not see every increment";
    EXPECT_EQ(ReadWord(*dst_device, core, report_addr + sizeof(uint32_t)), 0u) << "receiver did not drain it";
    EXPECT_EQ(ReadWord(*dst_device, core, sem_address), 0u);
}

TEST_F(Fabric1DFixture, GlobalSemaphoreUpOverFabric) { RunGlobalSemaphoreUpOverFabric(this, /*is_2d=*/false); }

TEST_F(Fabric2DFixture, GlobalSemaphoreUpOverFabric) { RunGlobalSemaphoreUpOverFabric(this, /*is_2d=*/true); }

}  // namespace
}  // namespace tt::tt_fabric::fabric_router_tests
