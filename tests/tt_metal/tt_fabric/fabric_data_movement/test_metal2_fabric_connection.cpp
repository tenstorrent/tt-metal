// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// End-to-end tests of Metal 2.0 fabric connections (ProgramSpec::fabric_connections): a Metal 2.0 sender kernel
// builds its fabric connection from a binding token and up()s a GlobalSemaphore on a neighbouring chip, where a
// Metal 2.0 receiver kernel down()s every increment.

#include <gtest/gtest.h>

#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/experimental/fabric/fabric.hpp>
#include <tt-metalium/global_semaphore.hpp>
#include <tt-metalium/hal.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/mesh_workload.hpp>
#include <tt-metalium/program.hpp>
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
using tt::tt_metal::Fabric1DTargets;
using tt::tt_metal::Fabric2DTargets;
using tt::tt_metal::FabricConnectionSpec;
using tt::tt_metal::FabricConnectionSpecName;
using tt::tt_metal::GlobalSemaphore;
using tt::tt_metal::ProgramSpec;
using tt::tt_metal::distributed::MeshDevice;
using tt::tt_metal::distributed::MeshWorkload;

constexpr uint32_t kIncrements = 32;
constexpr uint32_t kReportNotWritten = 0xFFFFFFFF;
const CoreCoord kCore{0, 0};

// Sends with a WorkerToFabricEdmSender built from the binding token fabric::to_receiver.
constexpr const char* kSenderFromBindingSource = R"(
#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/global_semaphore.h"
#include "experimental/kernel_args.h"
#include "tt_metal/fabric/hw/inc/edm_fabric/edm_fabric_worker_adapters.hpp"
#include "tt_metal/fabric/hw/inc/packet_header_pool.h"

void kernel_main() {
    auto sender = tt::tt_fabric::WorkerToFabricEdmSender::build_from_binding(fabric::to_receiver);
    sender.open();
    auto* hdr = reinterpret_cast<volatile tt_l1_ptr PACKET_HEADER_TYPE*>(PacketHeaderPool::allocate_header());
#if defined(FABRIC_2D)
    const GlobalSemaphore::ToChip chip{
        .dst_dev_id = static_cast<uint8_t>(get_arg(args::dst_dev_id)),
        .dst_mesh_id = static_cast<uint16_t>(get_arg(args::dst_mesh_id))};
#else
    const GlobalSemaphore::ToChip chip{.num_hops = 1};
#endif
    GlobalSemaphore sem(get_arg(args::sem_addr));
    Noc noc;
    for (uint32_t i = 0; i < get_arg(args::increments); i++) {
        sem.up(sender, hdr, chip, noc, get_arg(args::dst_noc_x), get_arg(args::dst_noc_y), 1);
    }
    sender.close();
}
)";

// Sends on the forward connection of a FabricConnectionManager built from the binding token fabric::to_receiver,
// whose targets are {forward, backward}.
constexpr const char* kSenderFromConnectionManagerSource = R"(
#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/global_semaphore.h"
#include "experimental/kernel_args.h"
#include "tt_metal/fabric/hw/inc/edm_fabric/fabric_connection_manager.hpp"
#include "tt_metal/fabric/hw/inc/packet_header_pool.h"

void kernel_main() {
    auto connections = FabricConnectionManager::build_from_binding<
        FabricConnectionManager::BuildFromArgsMode::BUILD_AND_OPEN_CONNECTION>(fabric::to_receiver);
    auto* hdr = reinterpret_cast<volatile tt_l1_ptr PACKET_HEADER_TYPE*>(PacketHeaderPool::allocate_header());
    const GlobalSemaphore::ToChip chip{.num_hops = 1};
    GlobalSemaphore sem(get_arg(args::sem_addr));
    Noc noc;
    for (uint32_t i = 0; i < get_arg(args::increments); i++) {
        sem.up(connections.get_forward_connection(), hdr, chip, noc, get_arg(args::dst_noc_x), get_arg(args::dst_noc_y), 1);
    }
    connections.close();
}
)";

// Sends on the only slot of a RoutingPlaneConnectionManager opened from the binding token fabric::to_receiver.
// On a 2D fabric, the slot's destination chip comes from the binding.
constexpr const char* kSenderFromRoutingPlaneSource = R"(
#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/global_semaphore.h"
#include "experimental/kernel_args.h"
#include "tt_metal/fabric/hw/inc/edm_fabric/routing_plane_connection_manager.hpp"
#include "tt_metal/fabric/hw/inc/api_common.h"

using namespace tt::tt_fabric::common::experimental;
#include "tt_metal/fabric/hw/inc/packet_header_pool.h"

void kernel_main() {
    tt::tt_fabric::RoutingPlaneConnectionManager connections;
    open_connections(connections, fabric::to_receiver);
    auto& slot = connections.get(0);
    auto* hdr = reinterpret_cast<volatile tt_l1_ptr PACKET_HEADER_TYPE*>(PacketHeaderPool::allocate_header());
#if defined(FABRIC_2D)
    const GlobalSemaphore::ToChip chip{
        .dst_dev_id = static_cast<uint8_t>(slot.dst_dev_id), .dst_mesh_id = slot.dst_mesh_id};
#else
    const GlobalSemaphore::ToChip chip{.num_hops = 1};
#endif
    GlobalSemaphore sem(get_arg(args::sem_addr));
    Noc noc;
    for (uint32_t i = 0; i < get_arg(args::increments); i++) {
        sem.up(slot.sender, hdr, chip, noc, get_arg(args::dst_noc_x), get_arg(args::dst_noc_y), 1);
    }
    close_connections(connections);
}
)";

// Down()s `expected` increments, then writes {expected, remaining value} to report_addr.
constexpr const char* kReceiverSource = R"(
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/global_semaphore.h"
#include "experimental/kernel_args.h"

void kernel_main() {
    const uint32_t expected = get_arg(args::expected);
    GlobalSemaphore sem(global_sem::remote);
    sem.down(expected);
    volatile tt_l1_ptr uint32_t* report = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_arg(args::report_addr));
    report[0] = expected;
    report[1] = sem.value();
}
)";

// Two neighbouring chips: the sender runs on src_device, the receiver on dst_device.
struct NeighbourPairForTest {
    std::shared_ptr<MeshDevice> src_device;
    std::shared_ptr<MeshDevice> dst_device;
    FabricNodeId dst_node;
    RoutingDirection direction;  // from the sender's chip to the receiver's chip
};

// Not a public API: the first pair of neighbouring chips that the fixture opened, if any.
std::optional<NeighbourPairForTest> FindNeighbourPairForTest(BaseFabricFixture* fixture) {
    for (const auto direction : {RoutingDirection::E, RoutingDirection::W, RoutingDirection::N, RoutingDirection::S}) {
        FabricNodeId src_node(MeshId{0}, 0);
        FabricNodeId dst_node(MeshId{0}, 0);
        ChipId src_chip = 0;
        ChipId dst_chip = 0;
        if (find_device_with_neighbor_in_direction(fixture, src_node, dst_node, src_chip, dst_chip, direction)) {
            return NeighbourPairForTest{
                fixture->get_device(src_chip), fixture->get_device(dst_chip), dst_node, direction};
        }
    }
    return std::nullopt;
}

RoutingDirection OppositeDirectionForTest(RoutingDirection direction) {
    switch (direction) {
        case RoutingDirection::E: return RoutingDirection::W;
        case RoutingDirection::W: return RoutingDirection::E;
        case RoutingDirection::N: return RoutingDirection::S;
        default: return RoutingDirection::N;
    }
}

// A ProgramSpec with one RISCV_0 kernel "sender" on kCore, built from `source`, that binds `connection`
// as fabric::to_receiver.
ProgramSpec MakeSenderSpecForTest(const char* source, FabricConnectionSpec connection) {
    using namespace tt::tt_metal;
    KernelSpec sender{
        .unique_id = KernelSpecName{"sender"},
        .source = KernelSpec::SourceCode{source},
        .num_threads = 1,
        .hw_config =
            DataMovementHardwareConfig{
                .config_1xx =
                    DataMovementHardwareConfig::DataMovement1XXConfig{
                        .processor = DataMovementProcessor::RISCV_0, .noc = NOC::NOC_0},
            },
    };
    sender.fabric_connection_bindings = {
        FabricConnectionBinding{.fabric_connection_spec_name = connection.unique_id, .accessor_name = "to_receiver"}};
    sender.runtime_arg_schema.common_runtime_arg_names = {
        "sem_addr", "dst_noc_x", "dst_noc_y", "increments", "dst_dev_id", "dst_mesh_id"};
    ProgramSpec spec;
    spec.name = "metal2_fabric_connection_sender";
    spec.kernels = {sender};
    spec.fabric_connections = {std::move(connection)};
    spec.work_units = {WorkUnitSpec{.name = "sender", .kernels = {KernelSpecName{"sender"}}, .target_nodes = kCore}};
    return spec;
}

// The sender's run args: up() `sem` kIncrements times on kCore of the receiver's chip.
tt::tt_metal::ProgramRunArgs MakeSenderRunArgsForTest(const NeighbourPairForTest& pair, const GlobalSemaphore& sem) {
    using namespace tt::tt_metal;
    const CoreCoord dst_virtual = pair.dst_device->worker_core_from_logical_core(kCore);
    ProgramRunArgs args;
    args.kernel_run_args.push_back(ProgramRunArgs::KernelRunArgs{
        .kernel = KernelSpecName{"sender"},
        .common_runtime_arg_values =
            {{"sem_addr", static_cast<uint32_t>(sem.address())},
             {"dst_noc_x", static_cast<uint32_t>(dst_virtual.x)},
             {"dst_noc_y", static_cast<uint32_t>(dst_virtual.y)},
             {"increments", kIncrements},
             {"dst_dev_id", pair.dst_node.chip_id},
             {"dst_mesh_id", *pair.dst_node.mesh_id}},
    });
    return args;
}

// Not a public API: an L1 address on kCore of `device` where the receiver writes its report.
uint32_t ReceiverReportAddressForTest(MeshDevice& device) {
    return device.allocator()->get_base_allocator_addr(tt::tt_metal::HalMemType::L1) + (32 * 1024);
}

// A workload whose Metal 2.0 kernel on kCore down()s kIncrements from `sem`, then reports how many it saw.
MeshWorkload MakeReceiverWorkloadForTest(MeshDevice& device, const GlobalSemaphore& sem) {
    using namespace tt::tt_metal;
    KernelSpec receiver{
        .unique_id = KernelSpecName{"receiver"},
        .source = KernelSpec::SourceCode{kReceiverSource},
        .num_threads = 1,
        .hw_config =
            DataMovementHardwareConfig{
                .config_1xx =
                    DataMovementHardwareConfig::DataMovement1XXConfig{
                        .processor = DataMovementProcessor::RISCV_0, .noc = NOC::NOC_0},
            },
    };
    receiver.global_semaphore_bindings = {GlobalSemaphoreBinding{
        .global_semaphore_parameter_name = GlobalSemaphoreParamName{"remote"}, .accessor_name = "remote"}};
    receiver.runtime_arg_schema.common_runtime_arg_names = {"report_addr", "expected"};
    ProgramSpec spec;
    spec.name = "metal2_fabric_connection_receiver";
    spec.kernels = {receiver};
    spec.global_semaphore_parameters = {GlobalSemaphoreParameter{.unique_id = GlobalSemaphoreParamName{"remote"}}};
    spec.work_units = {
        WorkUnitSpec{.name = "receiver", .kernels = {KernelSpecName{"receiver"}}, .target_nodes = kCore}};

    ProgramRunArgs args;
    args.kernel_run_args.push_back(ProgramRunArgs::KernelRunArgs{
        .kernel = KernelSpecName{"receiver"},
        .common_runtime_arg_values = {{"report_addr", ReceiverReportAddressForTest(device)}, {"expected", kIncrements}},
    });
    args.global_semaphore_args = {{GlobalSemaphoreParamName{"remote"}, std::cref(sem)}};
    MeshWorkload workload(device, spec);
    workload.get_programs().begin()->second.set_program_run_args(args);
    return workload;
}

// Not a public API: fills the receiver's report with kReportNotWritten, enqueues the receiver, then the sender, and
// waits for both.
void RunReceiverThenSenderForTest(const NeighbourPairForTest& pair, MeshWorkload& receiver, MeshWorkload& sender) {
    tt::tt_metal::IDevice* receiver_chip = pair.dst_device->get_devices().at(0);
    tt::tt_metal::MetalContext::instance().get_cluster().write_core(
        receiver_chip->id(),
        receiver_chip->worker_core_from_logical_core(kCore),
        std::vector<uint32_t>{kReportNotWritten, kReportNotWritten},
        ReceiverReportAddressForTest(*pair.dst_device));
    tt::tt_metal::distributed::EnqueueMeshWorkload(pair.dst_device->mesh_command_queue(), receiver, false);
    tt::tt_metal::distributed::EnqueueMeshWorkload(pair.src_device->mesh_command_queue(), sender, false);
    tt::tt_metal::distributed::Finish(pair.src_device->mesh_command_queue());
    tt::tt_metal::distributed::Finish(pair.dst_device->mesh_command_queue());
}

// Not a public API: the receiver's report: {increments it took, semaphore value left after it took them}.
std::vector<uint32_t> ReadReceiverReportForTest(MeshDevice& device) {
    tt::tt_metal::IDevice* chip = device.get_devices().at(0);
    return tt::tt_metal::MetalContext::instance().get_cluster().read_core(
        chip->id(),
        chip->worker_core_from_logical_core(kCore),
        ReceiverReportAddressForTest(device),
        2 * sizeof(uint32_t));
}

TEST_F(Fabric1DFixture, Metal2SenderBuiltFromBindingReachesNeighbour) {
    // A sender built from the binding of a direction target delivers every increment to the neighbouring chip.
    const std::optional<NeighbourPairForTest> pair = FindNeighbourPairForTest(this);
    if (!pair.has_value()) {
        GTEST_SKIP() << "No pair of neighbouring chips found";
    }
    GlobalSemaphore sem = tt::tt_metal::CreateGlobalSemaphore(*pair->dst_device, CoreRangeSet(CoreRange(kCore)), 0);
    FabricConnectionSpec connection{
        .unique_id = FabricConnectionSpecName{"to_receiver"},
        .targets = Fabric1DTargets{pair->direction},
        .links = {{kCore, 0}}};
    MeshWorkload sender(*pair->src_device, MakeSenderSpecForTest(kSenderFromBindingSource, connection));
    sender.get_programs().begin()->second.set_program_run_args(MakeSenderRunArgsForTest(*pair, sem));
    MeshWorkload receiver = MakeReceiverWorkloadForTest(*pair->dst_device, sem);
    RunReceiverThenSenderForTest(*pair, receiver, sender);
    EXPECT_EQ(ReadReceiverReportForTest(*pair->dst_device), (std::vector<uint32_t>{kIncrements, 0u}));
}

TEST_F(Fabric1DFixture, Metal2ConnectionManagerForwardSlotReachesNeighbour) {
    // A FabricConnectionManager built from a {forward, backward} binding sends on the forward slot's neighbour.
    const std::optional<NeighbourPairForTest> pair = FindNeighbourPairForTest(this);
    if (!pair.has_value()) {
        GTEST_SKIP() << "No pair of neighbouring chips found";
    }
    GlobalSemaphore sem = tt::tt_metal::CreateGlobalSemaphore(*pair->dst_device, CoreRangeSet(CoreRange(kCore)), 0);
    FabricConnectionSpec connection{
        .unique_id = FabricConnectionSpecName{"to_receiver"},
        .targets = Fabric1DTargets{pair->direction, OppositeDirectionForTest(pair->direction)},
        .links = {{kCore, 0}}};
    MeshWorkload sender(*pair->src_device, MakeSenderSpecForTest(kSenderFromConnectionManagerSource, connection));
    sender.get_programs().begin()->second.set_program_run_args(MakeSenderRunArgsForTest(*pair, sem));
    MeshWorkload receiver = MakeReceiverWorkloadForTest(*pair->dst_device, sem);
    RunReceiverThenSenderForTest(*pair, receiver, sender);
    EXPECT_EQ(ReadReceiverReportForTest(*pair->dst_device), (std::vector<uint32_t>{kIncrements, 0u}));
}

TEST_F(Fabric1DFixture, Metal2RoutingPlaneSlotReachesNeighbour) {
    // A RoutingPlaneConnectionManager opened from a binding sends on its slot to the neighbouring chip.
    const std::optional<NeighbourPairForTest> pair = FindNeighbourPairForTest(this);
    if (!pair.has_value()) {
        GTEST_SKIP() << "No pair of neighbouring chips found";
    }
    GlobalSemaphore sem = tt::tt_metal::CreateGlobalSemaphore(*pair->dst_device, CoreRangeSet(CoreRange(kCore)), 0);
    FabricConnectionSpec connection{
        .unique_id = FabricConnectionSpecName{"to_receiver"},
        .targets = Fabric1DTargets{pair->direction},
        .links = {{kCore, 0}}};
    MeshWorkload sender(*pair->src_device, MakeSenderSpecForTest(kSenderFromRoutingPlaneSource, connection));
    sender.get_programs().begin()->second.set_program_run_args(MakeSenderRunArgsForTest(*pair, sem));
    MeshWorkload receiver = MakeReceiverWorkloadForTest(*pair->dst_device, sem);
    RunReceiverThenSenderForTest(*pair, receiver, sender);
    EXPECT_EQ(ReadReceiverReportForTest(*pair->dst_device), (std::vector<uint32_t>{kIncrements, 0u}));
}

TEST_F(Fabric2DFixture, Metal2SenderBuiltFromBindingReachesPeer) {
    // A sender built from the binding of a peer target delivers every increment to the peer chip.
    const std::optional<NeighbourPairForTest> pair = FindNeighbourPairForTest(this);
    if (!pair.has_value()) {
        GTEST_SKIP() << "No pair of neighbouring chips found";
    }
    GlobalSemaphore sem = tt::tt_metal::CreateGlobalSemaphore(*pair->dst_device, CoreRangeSet(CoreRange(kCore)), 0);
    FabricConnectionSpec connection{
        .unique_id = FabricConnectionSpecName{"to_receiver"},
        .targets = Fabric2DTargets{pair->dst_node},
        .links = {{kCore, 0}}};
    MeshWorkload sender(*pair->src_device, MakeSenderSpecForTest(kSenderFromBindingSource, connection));
    sender.get_programs().begin()->second.set_program_run_args(MakeSenderRunArgsForTest(*pair, sem));
    MeshWorkload receiver = MakeReceiverWorkloadForTest(*pair->dst_device, sem);
    RunReceiverThenSenderForTest(*pair, receiver, sender);
    EXPECT_EQ(ReadReceiverReportForTest(*pair->dst_device), (std::vector<uint32_t>{kIncrements, 0u}));
}

TEST_F(Fabric2DFixture, Metal2RoutingPlaneSlotReachesPeer) {
    // A RoutingPlaneConnectionManager opened from a peer binding routes its slot to the peer chip.
    const std::optional<NeighbourPairForTest> pair = FindNeighbourPairForTest(this);
    if (!pair.has_value()) {
        GTEST_SKIP() << "No pair of neighbouring chips found";
    }
    GlobalSemaphore sem = tt::tt_metal::CreateGlobalSemaphore(*pair->dst_device, CoreRangeSet(CoreRange(kCore)), 0);
    FabricConnectionSpec connection{
        .unique_id = FabricConnectionSpecName{"to_receiver"},
        .targets = Fabric2DTargets{pair->dst_node},
        .links = {{kCore, 0}}};
    MeshWorkload sender(*pair->src_device, MakeSenderSpecForTest(kSenderFromRoutingPlaneSource, connection));
    sender.get_programs().begin()->second.set_program_run_args(MakeSenderRunArgsForTest(*pair, sem));
    MeshWorkload receiver = MakeReceiverWorkloadForTest(*pair->dst_device, sem);
    RunReceiverThenSenderForTest(*pair, receiver, sender);
    EXPECT_EQ(ReadReceiverReportForTest(*pair->dst_device), (std::vector<uint32_t>{kIncrements, 0u}));
}

}  // namespace
}  // namespace tt::tt_fabric::fabric_router_tests
