#include <gtest/gtest.h>

#include <tt-metalium/tt_metal.hpp>
#include <tt-metalium/control_plane.hpp>
#include <tt-metalium/fabric.hpp>

#include "tests/tt_metal/tt_fabric/common/fabric_fixture.hpp"  // Fabric2DFixture, BaseFabricFixture
#include "tests/tt_metal/tt_fabric/common/utils.hpp"           // find_device_with_neighbor_in_direction
#include <tt-metalium/allocator.hpp>                           // Allocator concrete type
#include <tt-metalium/hal.hpp>                                 // Hal::get_constants()
#include "tt_metal/fabric/fabric_context.hpp"                  // ControlPlane::get_fabric_context() definition
#include "tt_metal/fabric/fabric_host_utils.hpp"      // get_forwarding_link_indices / append_fabric_connection_rt_args
#include "tt_metal/fabric/hw/inc/tt_fabric_status.h"  // TT_FABRIC_STATUS_* and WORD_CNT indices

// Bring types/helpers into scope
using tt::tt_fabric::fabric_router_tests::Fabric2DFixture;
using BaseFabricFixture = tt::tt_fabric::fabric_router_tests::BaseFabricFixture;
using tt::tt_fabric::FabricNodeId;
using tt::tt_fabric::MeshId;
using tt::tt_fabric::RoutingDirection;
using tt::tt_fabric::fabric_router_tests::find_device_with_neighbor_in_direction;
using chip_id_t = tt::umd::chip_id_t;

namespace tt::tt_fabric {
namespace fabric_router_tests {

struct WorkerMemMap_local {
    uint32_t source_l1_buffer_address;
    uint32_t packet_payload_size_bytes;
    uint32_t test_results_address;
    uint32_t target_address;
    uint32_t notification_mailbox_address;
    uint32_t test_results_size_bytes;
};

// Utility function reused across tests to get address params
WorkerMemMap_local generate_worker_mem_map_local(tt_metal::IDevice* device, Topology topology) {
    constexpr uint32_t PACKET_HEADER_RESERVED_BYTES = 45056;
    constexpr uint32_t DATA_SPACE_RESERVED_BYTES = 851968;
    constexpr uint32_t TEST_RESULTS_SIZE_BYTES = 128;
    uint32_t NOTIFICATION_MAILBOX_ADDR_SIZE_BYTES = tt::tt_metal::hal::get_l1_alignment();

    uint32_t base_addr = device->allocator()->get_base_allocator_addr(tt_metal::HalMemType::L1);
    uint32_t source_l1_buffer_address = base_addr + PACKET_HEADER_RESERVED_BYTES;
    uint32_t test_results_address = source_l1_buffer_address + DATA_SPACE_RESERVED_BYTES;
    uint32_t target_address = source_l1_buffer_address;
    uint32_t notification_mailbox_address = test_results_address + TEST_RESULTS_SIZE_BYTES;

    uint32_t packet_payload_size_bytes = get_tt_fabric_max_payload_size_bytes();

    return {
        source_l1_buffer_address,
        packet_payload_size_bytes,
        test_results_address,
        target_address,
        notification_mailbox_address,
        TEST_RESULTS_SIZE_BYTES};
}
}  // namespace fabric_router_tests
}  // namespace tt::tt_fabric

// ---------- FORWARD DECLARATION IN THE CORRECT NAMESPACE ----------
namespace tt::tt_fabric::fabric_router_tests {
void run_unicast_test_bw_chips_local(
    BaseFabricFixture* fixture,
    chip_id_t src_physical_device_id,
    chip_id_t dst_physical_device_id,
    uint32_t num_hops,
    bool use_dram_dst = false) {
    CoreCoord sender_logical_core = {0, 0};
    CoreCoord receiver_logical_core = {1, 0};

    const auto& control_plane = tt::tt_metal::MetalContext::instance().get_control_plane();
    auto src_fabric_node_id = control_plane.get_fabric_node_id_from_physical_chip_id(src_physical_device_id);
    auto dst_fabric_node_id = control_plane.get_fabric_node_id_from_physical_chip_id(dst_physical_device_id);

    auto* sender_device = DevicePool::instance().get_active_device(src_physical_device_id);
    auto* receiver_device = DevicePool::instance().get_active_device(dst_physical_device_id);
    CoreCoord sender_virtual_core = sender_device->worker_core_from_logical_core(sender_logical_core);
    CoreCoord receiver_virtual_core = receiver_device->worker_core_from_logical_core(receiver_logical_core);

    const auto fabric_config = tt::tt_metal::MetalContext::instance().get_fabric_config();
    const auto topology = control_plane.get_fabric_context().get_fabric_topology();
    uint32_t is_2d_fabric = topology == Topology::Mesh;

    // test parameters
    auto worker_mem_map = generate_worker_mem_map_local(sender_device, topology);
    uint32_t num_packets = 10;
    uint32_t time_seed = std::chrono::system_clock::now().time_since_epoch().count();

    // common compile time args for sender and receiver
    std::vector<uint32_t> compile_time_args = {
        worker_mem_map.test_results_address,
        worker_mem_map.test_results_size_bytes,
        worker_mem_map.target_address,
        use_dram_dst,
        topology == Topology::Mesh,
        fabric_config == tt_fabric::FabricConfig::FABRIC_2D_DYNAMIC,
        0 /* is_chip_multicast */,
        0 /* additional_dir */};

    std::map<std::string, std::string> defines = {};
    if (is_2d_fabric) {
        defines["FABRIC_2D"] = "";
    }

    // Create the sender program
    auto sender_program = tt_metal::CreateProgram();
    auto sender_kernel = tt_metal::CreateKernel(
        sender_program,
        "tests/tt_metal/tt_metal/perf_microbenchmark/routing/kernels/tt_fabric_1d_tx.cpp",
        {sender_logical_core},
        tt_metal::DataMovementConfig{
            .processor = tt_metal::DataMovementProcessor::RISCV_0,
            .noc = tt_metal::NOC::RISCV_0_default,
            .compile_args = compile_time_args,
            .defines = defines});

    auto mesh_shape = control_plane.get_physical_mesh_shape(src_fabric_node_id.mesh_id);
    log_info(tt::LogTest, "mesh dimensions {:x}", mesh_shape.dims());
    log_info(tt::LogTest, "mesh dimension 0 {:x}", mesh_shape[0]);
    log_info(tt::LogTest, "mesh dimension 1 {:x}", mesh_shape[1]);

    // Set up destination address/coordinates. One bank should be enough for testing
    uint32_t dest_bank_id = 0;
    uint32_t dest_dram_addr =
        use_dram_dst ? receiver_device->allocator()->get_base_allocator_addr(tt_metal::HalMemType::DRAM) : 0;

    std::vector<uint32_t> sender_runtime_args = {
        worker_mem_map.source_l1_buffer_address,
        worker_mem_map.packet_payload_size_bytes,
        num_packets,
        time_seed,
        receiver_virtual_core.x,
        receiver_virtual_core.y,
        mesh_shape[1],
        src_fabric_node_id.chip_id,
        num_hops,
        1 /* fwd_range */,
        dst_fabric_node_id.chip_id,
        *dst_fabric_node_id.mesh_id};

    // Only add DRAM args if use_dram_dst is true
    if (use_dram_dst) {
        sender_runtime_args.insert(
            sender_runtime_args.end(), {dest_bank_id, dest_dram_addr, worker_mem_map.notification_mailbox_address});
    }

    // append the EDM connection rt args
    const auto& available_links = get_forwarding_link_indices(src_fabric_node_id, dst_fabric_node_id);
    EXPECT_EQ(available_links.size() > 0, true);

    uint32_t link_idx = available_links[0];
    append_fabric_connection_rt_args(
        src_fabric_node_id, dst_fabric_node_id, link_idx, sender_program, {sender_logical_core}, sender_runtime_args);

    tt_metal::SetRuntimeArgs(sender_program, sender_kernel, sender_logical_core, sender_runtime_args);

    // If using DRAM destination, zero out the mailbox
    // Simple notification mailbox with flushing atomic increment is used instead of 2-way handshake for simple testing
    if (use_dram_dst) {
        std::vector<uint32_t> zeros(tt::tt_metal::hal::get_l1_alignment() / sizeof(uint32_t), 0);  // zero out mailbox
        tt_metal::detail::WriteToDeviceL1(
            receiver_device,
            receiver_logical_core,
            worker_mem_map.notification_mailbox_address,
            zeros,
            CoreType::WORKER);
        tt::tt_metal::MetalContext::instance().get_cluster().l1_barrier(receiver_device->id());
    }

    std::vector<uint32_t> receiver_runtime_args = {worker_mem_map.packet_payload_size_bytes, num_packets, time_seed};

    // Only add DRAM args if use_dram_dst is true
    if (use_dram_dst) {
        receiver_runtime_args.insert(
            receiver_runtime_args.end(),
            {dest_bank_id, dest_dram_addr, worker_mem_map.notification_mailbox_address, 1 /* notification value */});
    }

    // Create the receiver program for validation
    auto receiver_program = tt_metal::CreateProgram();
    auto receiver_kernel = tt_metal::CreateKernel(
        receiver_program,
        "tests/tt_metal/tt_metal/perf_microbenchmark/routing/kernels/tt_fabric_1d_rx.cpp",
        {receiver_logical_core},
        tt_metal::DataMovementConfig{
            .processor = tt_metal::DataMovementProcessor::RISCV_0,
            .noc = tt_metal::NOC::RISCV_0_default,
            .compile_args = compile_time_args});

    tt_metal::SetRuntimeArgs(receiver_program, receiver_kernel, receiver_logical_core, receiver_runtime_args);

    // Launch sender and receiver programs and wait for them to finish
    fixture->RunProgramNonblocking(receiver_device, receiver_program);
    fixture->RunProgramNonblocking(sender_device, sender_program);
    fixture->WaitForSingleProgramDone(sender_device, sender_program);
    fixture->WaitForSingleProgramDone(receiver_device, receiver_program);

    // Validate the status and packets processed by sender and receiver
    std::vector<uint32_t> sender_status;
    std::vector<uint32_t> receiver_status;

    tt_metal::detail::ReadFromDeviceL1(
        sender_device,
        sender_logical_core,
        worker_mem_map.test_results_address,
        worker_mem_map.test_results_size_bytes,
        sender_status,
        CoreType::WORKER);

    tt_metal::detail::ReadFromDeviceL1(
        receiver_device,
        receiver_logical_core,
        worker_mem_map.test_results_address,
        worker_mem_map.test_results_size_bytes,
        receiver_status,
        CoreType::WORKER);

    EXPECT_EQ(sender_status[TT_FABRIC_STATUS_INDEX], TT_FABRIC_STATUS_PASS);
    EXPECT_EQ(receiver_status[TT_FABRIC_STATUS_INDEX], TT_FABRIC_STATUS_PASS);

    uint64_t sender_bytes =
        ((uint64_t)sender_status[TT_FABRIC_WORD_CNT_INDEX + 1] << 32) | sender_status[TT_FABRIC_WORD_CNT_INDEX];
    uint64_t receiver_bytes =
        ((uint64_t)receiver_status[TT_FABRIC_WORD_CNT_INDEX + 1] << 32) | receiver_status[TT_FABRIC_WORD_CNT_INDEX];
    EXPECT_EQ(sender_bytes, receiver_bytes);
}
}  // namespace tt::tt_fabric::fabric_router_tests

// Try a single direction; return true if a path was found and test executed.
static bool RunTestUnicastConnAPI_TryDirection(
    BaseFabricFixture* fixture, uint32_t num_hops, RoutingDirection direction, bool use_dram_dst) {
    const auto& control_plane = tt::tt_metal::MetalContext::instance().get_control_plane();

    FabricNodeId src_fabric_node_id(MeshId{0}, 0);
    FabricNodeId dst_fabric_node_id(MeshId{0}, 0);
    chip_id_t not_used_1{};
    chip_id_t not_used_2{};

    if (!find_device_with_neighbor_in_direction(
            fixture, src_fabric_node_id, dst_fabric_node_id, not_used_1, not_used_2, direction)) {
        return false;
    }

    chip_id_t src_phys = control_plane.get_physical_chip_id_from_fabric_node_id(src_fabric_node_id);
    chip_id_t dst_phys = control_plane.get_physical_chip_id_from_fabric_node_id(dst_fabric_node_id);

    tt::tt_fabric::fabric_router_tests::run_unicast_test_bw_chips_local(
        fixture, src_phys, dst_phys, num_hops, use_dram_dst);
    return true;
}

// Convenience wrapper: try EAST, WEST, NORTH, SOUTH (in that order).
static void RunTestUnicastConnAPI_Local(BaseFabricFixture* fixture, uint32_t num_hops, bool use_dram_dst = false) {
    const RoutingDirection dirs[] = {
        static_cast<RoutingDirection>(tt::tt_fabric::EAST),
        static_cast<RoutingDirection>(tt::tt_fabric::WEST),
        static_cast<RoutingDirection>(tt::tt_fabric::NORTH),
        static_cast<RoutingDirection>(tt::tt_fabric::SOUTH),
    };

    for (auto d : dirs) {
        if (RunTestUnicastConnAPI_TryDirection(fixture, num_hops, d, use_dram_dst)) {
            return;  // success on first available neighbor
        }
    }
    GTEST_SKIP() << "No path found between sender and receiver in any direction (E/W/N/S)";
}

// ---------- TEST ----------
TEST_F(Fabric2DFixture, TestPerf) {
    // One hop to a neighbor in any available direction, DRAM dest off (L1)
    RunTestUnicastConnAPI_Local(this, /*num_hops=*/1);
}
