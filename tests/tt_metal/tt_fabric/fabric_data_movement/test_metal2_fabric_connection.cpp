// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// End-to-end tests of Metal 2.0 fabric connections (ProgramAdvancedOptions::fabric_connections): a Metal 2.0 sender
// kernel builds its fabric connection from a binding token and up()s a GlobalSemaphore on a neighbouring chip, where a
// Metal 2.0 receiver kernel down()s every increment.

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <functional>
#include <iterator>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include <tt-metalium/vocabulary/core_coord.hpp>
#include <tt-metalium/execution/distributed.hpp>
#include <tt-metalium/experimental/fabric/fabric.hpp>
#include <tt-metalium/memory/cross_program/global_semaphore.hpp>
#include <tt-metalium/discovery/hal.hpp>
#include <tt-metalium/workload/legacy/host_api.hpp>
#include <tt-metalium/workload/mesh_workload.hpp>
#include <tt-metalium/workload/program.hpp>
#include <tt-metalium/workload/run_args/program_run_args.hpp>
#include <tt-metalium/workload/specification/program_spec.hpp>

#include "fabric_fixture.hpp"
#include "tests/tt_metal/tt_metal/common/multi_device_fixture.hpp"
#include "utils.hpp"
#include "distributed/mesh_device_impl.hpp"
#include "impl/context/metal_context.hpp"

namespace tt::tt_fabric::fabric_router_tests {
namespace {

using tt::tt_metal::CoreCoord;
using tt::tt_metal::CoreRange;
using tt::tt_metal::CoreRangeSet;
using tt::tt_metal::GlobalSemaphore;
using tt::tt_metal::ProgramSpec;
using tt::tt_metal::distributed::MeshCoordinate;
using tt::tt_metal::distributed::MeshCoordinateRange;
using tt::tt_metal::distributed::MeshDevice;
using tt::tt_metal::distributed::MeshWorkload;
using tt::tt_metal::experimental::Fabric1DTargets;
using tt::tt_metal::experimental::Fabric2DTargets;
using tt::tt_metal::experimental::FabricConnectionBinding;
using tt::tt_metal::experimental::FabricConnectionsSpec;
using tt::tt_metal::experimental::FabricConnectionsSpecName;

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

// Writes {has forward connection, has backward connection} of a FabricConnectionManager built from the optional
// bindings fabric::fwd and fabric::bwd to report_addr, then sends on its only connection.
constexpr const char* kSenderFromOptionalBindingsSource = R"(
#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/global_semaphore.h"
#include "experimental/kernel_args.h"
#include "tt_metal/fabric/hw/inc/edm_fabric/fabric_connection_manager.hpp"
#include "tt_metal/fabric/hw/inc/packet_header_pool.h"

void kernel_main() {
    auto connections = FabricConnectionManager::build_from_optional_bindings<
        FabricConnectionManager::BuildFromArgsMode::BUILD_AND_OPEN_CONNECTION>(
        fabric::get_token_if_present<"fwd">(), fabric::get_token_if_present<"bwd">());
    volatile tt_l1_ptr uint32_t* report = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_arg(args::report_addr));
    report[0] = connections.has_forward_connection();
    report[1] = connections.has_backward_connection();
    auto& sender = connections.has_forward_connection() ? connections.get_forward_connection()
                                                         : connections.get_backward_connection();
    auto* hdr = reinterpret_cast<volatile tt_l1_ptr PACKET_HEADER_TYPE*>(PacketHeaderPool::allocate_header());
    const GlobalSemaphore::ToChip chip{.num_hops = 1};
    GlobalSemaphore sem(get_arg(args::sem_addr));
    Noc noc;
    for (uint32_t i = 0; i < get_arg(args::increments); i++) {
        sem.up(sender, hdr, chip, noc, get_arg(args::dst_noc_x), get_arg(args::dst_noc_y), 1);
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

// Up()s `increments` on the GlobalSemaphore of the neighbour of each target of fabric::to_neighbours, then down()s one
// batch of increments from each of them, and writes {increments taken, remaining value} to report_addr.
constexpr const char* kExchangeSource = R"(
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
    open_connections(connections, fabric::to_neighbours);
    auto* hdr = reinterpret_cast<volatile tt_l1_ptr PACKET_HEADER_TYPE*>(PacketHeaderPool::allocate_header());
    const GlobalSemaphore::ToChip chip{.num_hops = 1};
    GlobalSemaphore sem(global_sem::exchange);
    Noc noc;
    const uint32_t increments = get_arg(args::increments);
    for (uint32_t slot = 0; slot < connections.active_count(); slot++) {
        for (uint32_t i = 0; i < increments; i++) {
            sem.up(connections.get(slot).sender, hdr, chip, noc, get_arg(args::noc_x), get_arg(args::noc_y), 1);
        }
    }
    const uint32_t expected = increments * connections.active_count();
    close_connections(connections);
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

// Not a public API: whether the chip of the unit mesh `device` has a neighbour on its own mesh in `direction`.
bool HasNeighbourInDirectionForTest(MeshDevice& device, RoutingDirection direction) {
    const FabricNodeId node = device.get_fabric_node_id(MeshCoordinate(0, 0));
    return !tt::tt_metal::MetalContext::instance()
                .get_control_plane()
                .get_intra_chip_neighbors(node, direction)
                .empty();
}

RoutingDirection OppositeDirectionForTest(RoutingDirection direction) {
    switch (direction) {
        case RoutingDirection::E: return RoutingDirection::W;
        case RoutingDirection::W: return RoutingDirection::E;
        case RoutingDirection::N: return RoutingDirection::S;
        default: return RoutingDirection::N;
    }
}

// Not a public API: the first pair of neighbouring chips that the fixture opened whose sender's chip also has a
// neighbour in the opposite direction, if any.
std::optional<NeighbourPairForTest> FindNeighbourPairWithNeighbourBehindSenderForTest(BaseFabricFixture* fixture) {
    for (const auto& device : fixture->get_devices()) {
        const FabricNodeId src = device->get_fabric_node_id(MeshCoordinate(0, 0));
        for (const auto direction :
             {RoutingDirection::E, RoutingDirection::W, RoutingDirection::N, RoutingDirection::S}) {
            const auto& control_plane = tt::tt_metal::MetalContext::instance().get_control_plane();
            const auto ahead = control_plane.get_intra_chip_neighbors(src, direction);
            if (!ahead.empty() && HasNeighbourInDirectionForTest(*device, OppositeDirectionForTest(direction))) {
                const FabricNodeId dst(src.mesh_id, ahead.front());
                return NeighbourPairForTest{
                    device,
                    fixture->get_device(control_plane.get_physical_chip_id_from_fabric_node_id(dst)),
                    dst,
                    direction};
            }
        }
    }
    return std::nullopt;
}

// A ProgramSpec with one RISCV_0 kernel "sender" on kCore, built from `source`, that binds `connection`
// as fabric::<accessor_name>.
ProgramSpec MakeSenderSpecForTest(
    const char* source, FabricConnectionsSpec connection, const std::string& accessor_name = "to_receiver") {
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
    sender.advanced_options.fabric_connection_bindings = {
        FabricConnectionBinding{.fabric_connections_spec_name = connection.unique_id, .accessor_name = accessor_name}};
    sender.runtime_arg_schema.common_runtime_arg_names = {
        "sem_addr", "dst_noc_x", "dst_noc_y", "increments", "dst_dev_id", "dst_mesh_id", "report_addr"};
    ProgramSpec spec;
    spec.name = "metal2_fabric_connection_sender";
    spec.kernels = {sender};
    spec.advanced_options.fabric_connections = {std::move(connection)};
    spec.work_units = {WorkUnitSpec{.name = "sender", .kernels = {KernelSpecName{"sender"}}, .target_nodes = kCore}};
    return spec;
}

// Not a public API: an L1 address on kCore of `device` where the sender writes its report.
uint32_t SenderReportAddressForTest(MeshDevice& device) {
    return device.allocator()->get_base_allocator_addr(tt::tt_metal::HalMemType::L1) + (32 * 1024);
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
             {"dst_mesh_id", *pair.dst_node.mesh_id},
             {"report_addr", SenderReportAddressForTest(*pair.src_device)}},
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

// Not a public API: the report of a sender built from optional bindings: {has forward, has backward}.
std::vector<uint32_t> ReadSenderReportForTest(MeshDevice& device) {
    tt::tt_metal::IDevice* chip = device.get_devices().at(0);
    return tt::tt_metal::MetalContext::instance().get_cluster().read_core(
        chip->id(),
        chip->worker_core_from_logical_core(kCore),
        SenderReportAddressForTest(device),
        2 * sizeof(uint32_t));
}

TEST_F(Fabric1DFixture, Metal2SenderBuiltFromBindingReachesNeighbour) {
    // A sender built from the binding of a direction target delivers every increment to the neighbouring chip.
    const std::optional<NeighbourPairForTest> pair = FindNeighbourPairForTest(this);
    if (!pair.has_value()) {
        GTEST_SKIP() << "No pair of neighbouring chips found";
    }
    GlobalSemaphore sem = tt::tt_metal::CreateGlobalSemaphore(*pair->dst_device, CoreRangeSet(CoreRange(kCore)), 0);
    FabricConnectionsSpec connection{
        .unique_id = FabricConnectionsSpecName{"to_receiver"},
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
    const std::optional<NeighbourPairForTest> pair = FindNeighbourPairWithNeighbourBehindSenderForTest(this);
    if (!pair.has_value()) {
        GTEST_SKIP() << "No chip with neighbours in two opposite directions found";
    }
    GlobalSemaphore sem = tt::tt_metal::CreateGlobalSemaphore(*pair->dst_device, CoreRangeSet(CoreRange(kCore)), 0);
    FabricConnectionsSpec connection{
        .unique_id = FabricConnectionsSpecName{"to_receiver"},
        .targets = Fabric1DTargets{pair->direction, OppositeDirectionForTest(pair->direction)},
        .links = {{kCore, 0}}};
    MeshWorkload sender(*pair->src_device, MakeSenderSpecForTest(kSenderFromConnectionManagerSource, connection));
    sender.get_programs().begin()->second.set_program_run_args(MakeSenderRunArgsForTest(*pair, sem));
    MeshWorkload receiver = MakeReceiverWorkloadForTest(*pair->dst_device, sem);
    RunReceiverThenSenderForTest(*pair, receiver, sender);
    EXPECT_EQ(ReadReceiverReportForTest(*pair->dst_device), (std::vector<uint32_t>{kIncrements, 0u}));
}

TEST_F(Fabric1DFixture, Metal2ConnectionManagerFromOnlyForwardBindingSendsForward) {
    // A manager built from only a forward binding has only a forward connection, which reaches the neighbour.
    const std::optional<NeighbourPairForTest> pair = FindNeighbourPairForTest(this);
    if (!pair.has_value()) {
        GTEST_SKIP() << "No pair of neighbouring chips found";
    }
    GlobalSemaphore sem = tt::tt_metal::CreateGlobalSemaphore(*pair->dst_device, CoreRangeSet(CoreRange(kCore)), 0);
    FabricConnectionsSpec connection{
        .unique_id = FabricConnectionsSpecName{"to_receiver"},
        .targets = Fabric1DTargets{pair->direction},
        .links = {{kCore, 0}}};
    MeshWorkload sender(*pair->src_device, MakeSenderSpecForTest(kSenderFromOptionalBindingsSource, connection, "fwd"));
    sender.get_programs().begin()->second.set_program_run_args(MakeSenderRunArgsForTest(*pair, sem));
    MeshWorkload receiver = MakeReceiverWorkloadForTest(*pair->dst_device, sem);
    RunReceiverThenSenderForTest(*pair, receiver, sender);
    EXPECT_EQ(ReadSenderReportForTest(*pair->src_device), (std::vector<uint32_t>{1u, 0u}));
    EXPECT_EQ(ReadReceiverReportForTest(*pair->dst_device), (std::vector<uint32_t>{kIncrements, 0u}));
}

TEST_F(Fabric1DFixture, Metal2ConnectionManagerFromOnlyBackwardBindingSendsBackward) {
    // A manager built from only a backward binding has only a backward connection, which reaches the neighbour.
    const std::optional<NeighbourPairForTest> pair = FindNeighbourPairForTest(this);
    if (!pair.has_value()) {
        GTEST_SKIP() << "No pair of neighbouring chips found";
    }
    GlobalSemaphore sem = tt::tt_metal::CreateGlobalSemaphore(*pair->dst_device, CoreRangeSet(CoreRange(kCore)), 0);
    FabricConnectionsSpec connection{
        .unique_id = FabricConnectionsSpecName{"to_receiver"},
        .targets = Fabric1DTargets{pair->direction},
        .links = {{kCore, 0}}};
    MeshWorkload sender(*pair->src_device, MakeSenderSpecForTest(kSenderFromOptionalBindingsSource, connection, "bwd"));
    sender.get_programs().begin()->second.set_program_run_args(MakeSenderRunArgsForTest(*pair, sem));
    MeshWorkload receiver = MakeReceiverWorkloadForTest(*pair->dst_device, sem);
    RunReceiverThenSenderForTest(*pair, receiver, sender);
    EXPECT_EQ(ReadSenderReportForTest(*pair->src_device), (std::vector<uint32_t>{0u, 1u}));
    EXPECT_EQ(ReadReceiverReportForTest(*pair->dst_device), (std::vector<uint32_t>{kIncrements, 0u}));
}

TEST_F(Fabric1DFixture, Metal2RoutingPlaneSlotReachesNeighbour) {
    // A RoutingPlaneConnectionManager opened from a binding sends on its slot to the neighbouring chip.
    const std::optional<NeighbourPairForTest> pair = FindNeighbourPairForTest(this);
    if (!pair.has_value()) {
        GTEST_SKIP() << "No pair of neighbouring chips found";
    }
    GlobalSemaphore sem = tt::tt_metal::CreateGlobalSemaphore(*pair->dst_device, CoreRangeSet(CoreRange(kCore)), 0);
    FabricConnectionsSpec connection{
        .unique_id = FabricConnectionsSpecName{"to_receiver"},
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
    FabricConnectionsSpec connection{
        .unique_id = FabricConnectionsSpecName{"to_receiver"},
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
    FabricConnectionsSpec connection{
        .unique_id = FabricConnectionsSpecName{"to_receiver"},
        .targets = Fabric2DTargets{pair->dst_node},
        .links = {{kCore, 0}}};
    MeshWorkload sender(*pair->src_device, MakeSenderSpecForTest(kSenderFromRoutingPlaneSource, connection));
    sender.get_programs().begin()->second.set_program_run_args(MakeSenderRunArgsForTest(*pair, sem));
    MeshWorkload receiver = MakeReceiverWorkloadForTest(*pair->dst_device, sem);
    RunReceiverThenSenderForTest(*pair, receiver, sender);
    EXPECT_EQ(ReadReceiverReportForTest(*pair->dst_device), (std::vector<uint32_t>{kIncrements, 0u}));
}

// Every device of the system mesh, with the fabric in 1D mode.
class Metal2Fabric1DMeshFixture : public tt::tt_metal::MeshDeviceFixtureBase {
protected:
    Metal2Fabric1DMeshFixture() :
        MeshDeviceFixtureBase(Config{.num_cqs = 1, .fabric_config = tt::tt_fabric::FabricConfig::FABRIC_1D}) {}

    // The mesh that the Program under test runs on.
    MeshDevice& fixture_mesh_device() { return *mesh_device_; }
};

// A ProgramSpec with one RISCV_0 kernel "exchanger" on kCore, built from kExchangeSource. It binds a fabric
// connection with the targets `directions` as fabric::to_neighbours, and the GlobalSemaphore "exchange".
ProgramSpec MakeExchangeSpecForTest(const Fabric1DTargets& directions) {
    using namespace tt::tt_metal;
    KernelSpec exchanger{
        .unique_id = KernelSpecName{"exchanger"},
        .source = KernelSpec::SourceCode{kExchangeSource},
        .num_threads = 1,
        .hw_config =
            DataMovementHardwareConfig{
                .config_1xx =
                    DataMovementHardwareConfig::DataMovement1XXConfig{
                        .processor = DataMovementProcessor::RISCV_0, .noc = NOC::NOC_0},
            },
    };
    exchanger.advanced_options.fabric_connection_bindings = {FabricConnectionBinding{
        .fabric_connections_spec_name = FabricConnectionsSpecName{"to_neighbours"}, .accessor_name = "to_neighbours"}};
    exchanger.global_semaphore_bindings = {GlobalSemaphoreBinding{
        .global_semaphore_parameter_name = GlobalSemaphoreParamName{"exchange"}, .accessor_name = "exchange"}};
    exchanger.runtime_arg_schema.common_runtime_arg_names = {"noc_x", "noc_y", "increments", "report_addr"};
    ProgramSpec spec;
    spec.name = "metal2_fabric_connection_exchange";
    spec.kernels = {exchanger};
    spec.advanced_options.fabric_connections = {FabricConnectionsSpec{
        .unique_id = FabricConnectionsSpecName{"to_neighbours"}, .targets = directions, .links = {{kCore, 0}}}};
    spec.global_semaphore_parameters = {GlobalSemaphoreParameter{.unique_id = GlobalSemaphoreParamName{"exchange"}}};
    spec.work_units = {
        WorkUnitSpec{.name = "exchanger", .kernels = {KernelSpecName{"exchanger"}}, .target_nodes = kCore}};
    return spec;
}

// The exchanger's run args: up() `sem` kIncrements times on kCore of each neighbour.
tt::tt_metal::ProgramRunArgs MakeExchangeRunArgsForTest(MeshDevice& mesh_device, const GlobalSemaphore& sem) {
    using namespace tt::tt_metal;
    const CoreCoord virtual_core = mesh_device.worker_core_from_logical_core(kCore);
    ProgramRunArgs args;
    args.kernel_run_args.push_back(ProgramRunArgs::KernelRunArgs{
        .kernel = KernelSpecName{"exchanger"},
        .common_runtime_arg_values =
            {{"noc_x", static_cast<uint32_t>(virtual_core.x)},
             {"noc_y", static_cast<uint32_t>(virtual_core.y)},
             {"increments", kIncrements},
             {"report_addr", ReceiverReportAddressForTest(mesh_device)}},
    });
    args.global_semaphore_args = {{GlobalSemaphoreParamName{"exchange"}, std::cref(sem)}};
    return args;
}

// Not a public API: the directions in which the device at `coord` of `mesh_device` has a neighbour on its mesh.
Fabric1DTargets NeighbourDirectionsForTest(MeshDevice& mesh_device, const MeshCoordinate& coord) {
    const auto& control_plane = tt::tt_metal::MetalContext::instance().get_control_plane();
    const FabricNodeId node = mesh_device.get_fabric_node_id(coord);
    Fabric1DTargets directions;
    std::ranges::copy_if(
        std::array{RoutingDirection::E, RoutingDirection::W, RoutingDirection::N, RoutingDirection::S},
        std::back_inserter(directions),
        [&](RoutingDirection direction) { return !control_plane.get_intra_chip_neighbors(node, direction).empty(); });
    return directions;
}

// Not a public API: one exchange ProgramSpec for each device range of `mesh_device` whose devices have neighbours in
// the same directions, with those directions as its targets.
std::unordered_map<MeshCoordinateRange, ProgramSpec> MakeExchangeSpecsByRangeForTest(MeshDevice& mesh_device) {
    std::map<Fabric1DTargets, std::vector<MeshCoordinate>> devices_by_directions;
    for (const auto& coord : MeshCoordinateRange(mesh_device.shape())) {
        devices_by_directions[NeighbourDirectionsForTest(mesh_device, coord)].push_back(coord);
    }
    std::unordered_map<MeshCoordinateRange, ProgramSpec> specs;
    for (const auto& [directions, devices] : devices_by_directions) {
        const MeshCoordinateRange range(devices.front(), devices.back());
        TT_FATAL(
            range.shape().mesh_size() == devices.size(),
            "The devices with the same neighbour directions are not a range");
        specs.emplace(range, MakeExchangeSpecForTest(directions));
    }
    return specs;
}

// Not a public API: writes kReportNotWritten to the exchanger's report on each device of `mesh_device`.
void ClearExchangeReportsForTest(MeshDevice& mesh_device) {
    for (const auto& coord : MeshCoordinateRange(mesh_device.shape())) {
        tt::tt_metal::IDevice* chip = mesh_device.impl().get_device(coord);
        tt::tt_metal::MetalContext::instance().get_cluster().write_core(
            chip->id(),
            chip->worker_core_from_logical_core(kCore),
            std::vector<uint32_t>{kReportNotWritten, kReportNotWritten},
            ReceiverReportAddressForTest(mesh_device));
    }
}

// Not a public API: the exchanger's report on each device of `mesh_device`, in range order.
std::vector<std::vector<uint32_t>> ReadExchangeReportsForTest(MeshDevice& mesh_device) {
    std::vector<std::vector<uint32_t>> reports;
    for (const auto& coord : MeshCoordinateRange(mesh_device.shape())) {
        tt::tt_metal::IDevice* chip = mesh_device.impl().get_device(coord);
        reports.push_back(tt::tt_metal::MetalContext::instance().get_cluster().read_core(
            chip->id(),
            chip->worker_core_from_logical_core(kCore),
            ReceiverReportAddressForTest(mesh_device),
            2 * sizeof(uint32_t)));
    }
    return reports;
}

// Not a public API: for each device of `mesh_device` in range order, the report of an exchange that worked:
// {kIncrements for each neighbour on its mesh, 0}.
std::vector<std::vector<uint32_t>> ExpectedExchangeReportsForTest(MeshDevice& mesh_device) {
    std::vector<std::vector<uint32_t>> reports;
    for (const auto& coord : MeshCoordinateRange(mesh_device.shape())) {
        const auto neighbours = static_cast<uint32_t>(NeighbourDirectionsForTest(mesh_device, coord).size());
        reports.push_back({kIncrements * neighbours, 0u});
    }
    return reports;
}

TEST_F(Metal2Fabric1DMeshFixture, Metal2ProgramPerDeviceRangeConnectsEachDeviceToItsNeighbours) {
    // A MeshWorkload with one ProgramSpec for each range of devices that have the same neighbour directions connects
    // each device to its own neighbours: each device up()s the GlobalSemaphore of each neighbour, and takes one batch
    // of increments from each of them.
    GlobalSemaphore sem = tt::tt_metal::CreateGlobalSemaphore(fixture_mesh_device(), CoreRangeSet(CoreRange(kCore)), 0);
    MeshWorkload exchange(fixture_mesh_device(), MakeExchangeSpecsByRangeForTest(fixture_mesh_device()));
    for (auto& [range, program] : exchange.get_programs()) {
        program.set_program_run_args(MakeExchangeRunArgsForTest(fixture_mesh_device(), sem));
    }
    ClearExchangeReportsForTest(fixture_mesh_device());
    tt::tt_metal::distributed::EnqueueMeshWorkload(fixture_mesh_device().mesh_command_queue(), exchange, false);
    tt::tt_metal::distributed::Finish(fixture_mesh_device().mesh_command_queue());
    EXPECT_EQ(ReadExchangeReportsForTest(fixture_mesh_device()), ExpectedExchangeReportsForTest(fixture_mesh_device()));
}

}  // namespace
}  // namespace tt::tt_fabric::fabric_router_tests
