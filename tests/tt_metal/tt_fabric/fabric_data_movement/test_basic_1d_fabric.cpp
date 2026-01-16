// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <chrono>
#include <gtest/gtest.h>
#include <cstdint>
#include <tt-metalium/experimental/fabric/control_plane.hpp>
#include "hostdevcommon/fabric_common.h"
#include <tt_metal/fabric/erisc_datamover_builder.hpp>
#include <array>
#include <cstddef>
#include <map>
#include <optional>
#include <utility>
#include <variant>
#include <vector>
#include <random>

#include <tt-metalium/allocator.hpp>
#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/data_types.hpp>
#include <tt-metalium/device.hpp>
#include "fabric/fabric_edm_packet_header.hpp"
#include "fabric_fixture.hpp"
#include "utils.hpp"
#include <tt-metalium/hal.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/kernel_types.hpp>
#include <tt-metalium/mesh_coord.hpp>
#include <tt-metalium/experimental/fabric/mesh_graph.hpp>
#include <tt_stl/span.hpp>
#include <tt-metalium/tt_metal.hpp>
#include <tt-metalium/experimental/fabric/fabric.hpp>
#include "tt_metal/fabric/hw/inc/tt_fabric_status.h"
#include "tt_metal/fabric/fabric_host_utils.hpp"
#include "tt_metal/fabric/fabric_context.hpp"
#include "tt_metal/fabric/fabric_builder_context.hpp"
#include <umd/device/types/core_coordinates.hpp>

namespace tt::tt_fabric::fabric_router_tests {
std::random_device rd;  // Non-deterministic seed source
std::mt19937 global_rng(rd());

struct WorkerMemMap {
    uint32_t source_l1_buffer_address;
    uint32_t packet_payload_size_bytes;
    uint32_t test_results_address;
    uint32_t target_address;
    uint32_t notification_mailbox_address;
    uint32_t test_results_size_bytes;
};

// Utility function reused across tests to get address params
WorkerMemMap generate_worker_mem_map(
    const std::shared_ptr<tt_metal::distributed::MeshDevice>& device, Topology /*topology*/) {
    constexpr uint32_t PACKET_HEADER_RESERVED_BYTES = 45056;
    constexpr uint32_t DATA_SPACE_RESERVED_BYTES = 851968;
    constexpr uint32_t TEST_RESULTS_SIZE_BYTES = 128;

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

// Helper struct for dual RISC memory layout generation
// Provides memory map generation that supports both single and dual RISC modes
struct DualRiscMemMapHelper {
    static constexpr uint32_t packet_header_reserved_bytes = 45056;
    static constexpr uint32_t test_results_size_bytes = 128;
    static constexpr uint32_t notification_size_bytes = 4096;  // Support up to 32 devices (Galaxy) with 128B per slot
    static constexpr uint32_t data_space_reserved_bytes = 851968;

    uint32_t data_space_size;
    uint32_t region_size;
    uint32_t num_packets;

    DualRiscMemMapHelper(bool dual_risc, uint32_t num_pkts) :
        data_space_size(dual_risc ? data_space_reserved_bytes / 2 : data_space_reserved_bytes),
        region_size(data_space_size + test_results_size_bytes + notification_size_bytes),
        num_packets(num_pkts) {
        static_assert(packet_header_reserved_bytes % 16 == 0, "packet_header_reserved_bytes must be 16-byte aligned");
        static_assert(test_results_size_bytes % 16 == 0, "test_results_size_bytes must be 16-byte aligned");
        static_assert(notification_size_bytes % 16 == 0, "notification_size_bytes must be 16-byte aligned");
        static_assert(
            data_space_reserved_bytes % 32 == 0, "data_space_reserved_bytes must be 32-byte aligned for halving");

        TT_FATAL(region_size % 16 == 0, "region_size must be 16-byte aligned");
        TT_FATAL(
            data_space_size >= num_packets * get_tt_fabric_max_payload_size_bytes() * 2,
            "data_space_size must be large enough for num_packets");
    }

    WorkerMemMap gen_mem_map(
        const std::shared_ptr<tt_metal::distributed::MeshDevice>& device, uint32_t base_offset) const {
        TT_FATAL(base_offset % 16 == 0, "base_offset must be 16-byte aligned");
        uint32_t base_addr = device->allocator()->get_base_allocator_addr(tt_metal::HalMemType::L1);
        uint32_t source_l1_buffer_address = base_addr + packet_header_reserved_bytes + base_offset;
        return WorkerMemMap{
            source_l1_buffer_address,
            get_tt_fabric_max_payload_size_bytes(),
            source_l1_buffer_address + data_space_size,
            source_l1_buffer_address,
            source_l1_buffer_address + data_space_size + test_results_size_bytes,
            test_results_size_bytes};
    }
};

std::vector<uint32_t> get_random_numbers_from_range(uint32_t start, uint32_t end, uint32_t count) {
    std::vector<uint32_t> range(end - start + 1);

    // generate the range
    std::iota(range.begin(), range.end(), start);

    // shuffle the range
    std::shuffle(range.begin(), range.end(), global_rng);

    return std::vector<uint32_t>(range.begin(), range.begin() + count);
}

std::shared_ptr<tt_metal::Program> create_receiver_program(
    const std::vector<uint32_t>& compile_time_args,
    const std::vector<uint32_t>& runtime_args,
    const CoreCoord& logical_core) {
    auto recv_program = std::make_shared<tt_metal::Program>();
    auto recv_kernel = tt_metal::CreateKernel(
        *recv_program,
        "tests/tt_metal/tt_metal/perf_microbenchmark/routing/kernels/tt_fabric_1d_rx.cpp",
        {logical_core},
        tt_metal::DataMovementConfig{
            .processor = tt_metal::DataMovementProcessor::RISCV_0,
            .noc = tt_metal::NOC::RISCV_0_default,
            .compile_args = compile_time_args});
    tt_metal::SetRuntimeArgs(*recv_program, recv_kernel, logical_core, runtime_args);
    return recv_program;
}

void get_mcast_receivers(
    std::unordered_map<RoutingDirection, std::vector<FabricNodeId>>& mcast_ref,
    std::vector<ChipId>& mcast_receiver_physical_device_ids,
    RoutingDirection trunk_direction,
    RoutingDirection branch_direction) {
    auto& control_plane = tt::tt_metal::MetalContext::instance().get_control_plane();
    if (mcast_ref.contains(branch_direction)) {
        auto node_ids = mcast_ref[branch_direction];
        for (auto node : node_ids) {
            auto curr_fabric_node_id = node;
            for (uint32_t i = 0; i < mcast_ref[trunk_direction].size(); i++) {
                auto neighbors = control_plane.get_intra_chip_neighbors(curr_fabric_node_id, trunk_direction);
                if (!neighbors.empty()) {
                    FabricNodeId rx_node_id(MeshId{curr_fabric_node_id.mesh_id}, neighbors[0]);
                    mcast_receiver_physical_device_ids.push_back(
                        control_plane.get_physical_chip_id_from_fabric_node_id(rx_node_id));
                    curr_fabric_node_id = rx_node_id;
                    log_info(tt::LogTest, "Mcast Rx MeshId {} ChipId {}", rx_node_id.mesh_id, rx_node_id.chip_id);
                }
            }
        }
    }
}

void RunTestLineMcast(BaseFabricFixture* fixture, const std::vector<McastRoutingInfo>& mcast_routing_info) {
    auto& control_plane= tt::tt_metal::MetalContext::instance().get_control_plane();
    auto user_meshes = control_plane.get_user_physical_mesh_ids();
    bool system_accommodates_mcast = false;
    for (const auto& mesh : user_meshes) {
        auto mesh_shape = control_plane.get_physical_mesh_shape(mesh);
        // Need at least 8 chips for all mcast tests
        if (mesh_shape.mesh_size() >= 8) {
            system_accommodates_mcast = true;
            break;
        }
    }
    if (!system_accommodates_mcast) {
        GTEST_SKIP() << "No mesh found for line mcast test";
    }
    // Setup mcast path
    ChipId mcast_start_phys_id = 0;                             // Physical ID for chip starting mcast
    FabricNodeId mcast_start_id(MeshId{0}, 0);                  // Mesh ID for chip starting mcast
    ChipId sender_phys_id;
    FabricNodeId sender_id(MeshId{0}, 0);                       // Mesh/Chip ID of mcast sender
    std::unordered_map<RoutingDirection, uint32_t> mcast_hops;  // Specify mcast path from mcast src chip
    std::unordered_map<RoutingDirection, std::vector<FabricNodeId>>
        mcast_group;  // Mesh IDs for chips involved in mcast
    std::unordered_map<RoutingDirection, std::vector<ChipId>>
        mcast_group_phys_ids_per_dir;  // Physical IDs for chips involved in mcast
    bool spine_hops = false;
    bool branch_hops = false;
    for (const auto& routing_info : mcast_routing_info) {
        mcast_hops[routing_info.mcast_dir] = routing_info.num_mcast_hops;
        if (routing_info.mcast_dir == RoutingDirection::E or routing_info.mcast_dir == RoutingDirection::W) {
            branch_hops = true;
        }
        if (routing_info.mcast_dir == RoutingDirection::N or routing_info.mcast_dir == RoutingDirection::S) {
            spine_hops = true;
        }
    }

    bool mcast_group_found = find_device_with_neighbor_in_multi_direction(
        fixture,
        sender_id,
        mcast_group,
        sender_phys_id,
        mcast_group_phys_ids_per_dir,
        mcast_hops);

    if (!mcast_group_found) {
        GTEST_SKIP() << "Mcast group not found for line mcast test";
    }

    // Compute physical IDs for mcast group chips
    std::vector<ChipId> mcast_group_phys_ids = {};
    if (spine_hops) {
        if (mcast_group_phys_ids_per_dir.contains(RoutingDirection::N)) {
            mcast_start_phys_id = mcast_group_phys_ids_per_dir[RoutingDirection::N][0];
            mcast_start_id = mcast_group[RoutingDirection::N][0];
            mcast_group_phys_ids.insert(mcast_group_phys_ids.end(), mcast_group_phys_ids_per_dir[RoutingDirection::N].begin(), mcast_group_phys_ids_per_dir[RoutingDirection::N].end());
            get_mcast_receivers(mcast_group, mcast_group_phys_ids, RoutingDirection::N, RoutingDirection::E);
            get_mcast_receivers(mcast_group, mcast_group_phys_ids, RoutingDirection::N, RoutingDirection::W);
        } else {
            mcast_start_phys_id = mcast_group_phys_ids_per_dir[RoutingDirection::S][0];
            mcast_start_id = mcast_group[RoutingDirection::S][0];
            mcast_group_phys_ids.insert(mcast_group_phys_ids.end(), mcast_group_phys_ids_per_dir[RoutingDirection::S].begin(), mcast_group_phys_ids_per_dir[RoutingDirection::S].end());
            get_mcast_receivers(mcast_group, mcast_group_phys_ids, RoutingDirection::S, RoutingDirection::E);
            get_mcast_receivers(mcast_group, mcast_group_phys_ids, RoutingDirection::S, RoutingDirection::W);
        }
    } else if (branch_hops) {
        if (mcast_group_phys_ids_per_dir.contains(RoutingDirection::E)) {
            mcast_start_phys_id = mcast_group_phys_ids_per_dir[RoutingDirection::E][0];
            mcast_start_id = mcast_group[RoutingDirection::E][0];
            mcast_group_phys_ids.insert(mcast_group_phys_ids.end(), mcast_group_phys_ids_per_dir[RoutingDirection::E].begin(), mcast_group_phys_ids_per_dir[RoutingDirection::E].end());

        } else {
            mcast_start_phys_id = mcast_group_phys_ids_per_dir[RoutingDirection::W][0];
            mcast_start_id = mcast_group[RoutingDirection::W][0];
            mcast_group_phys_ids.insert(mcast_group_phys_ids.end(), mcast_group_phys_ids_per_dir[RoutingDirection::W].begin(), mcast_group_phys_ids_per_dir[RoutingDirection::W].end());
        }
    }

    CoreCoord sender_logical_core = {0, 0};    // This core on the sender (remote chip) will make the mcast request
    CoreCoord receiver_logical_core = {1, 0};  // Data will be forwarded to this core on al chips in the mcast group

    const auto& fabric_context = control_plane.get_fabric_context();
    const auto& edm_config = fabric_context.get_builder_context().get_fabric_router_config();
    uint32_t is_2d_fabric = edm_config.topology == Topology::Mesh;

    auto sender_device = fixture->get_device(sender_phys_id);
    auto mcast_start_device = fixture->get_device(mcast_start_phys_id);
    std::vector<std::shared_ptr<tt_metal::distributed::MeshDevice>> mcast_group_devices = {};
    mcast_group_devices.reserve(mcast_group_phys_ids.size());
    for (auto id : mcast_group_phys_ids) {
        mcast_group_devices.push_back(fixture->get_device(id));
    }

    CoreCoord receiver_virtual_core = mcast_start_device->worker_core_from_logical_core(receiver_logical_core);

    auto receiver_noc_encoding =
        tt::tt_metal::MetalContext::instance().hal().noc_xy_encoding(receiver_virtual_core.x, receiver_virtual_core.y);

    auto mesh_shape = control_plane.get_physical_mesh_shape(sender_id.mesh_id);

    auto worker_mem_map = generate_worker_mem_map(sender_device, edm_config.topology);
    uint32_t num_packets = 100;
    uint32_t time_seed = std::chrono::system_clock::now().time_since_epoch().count();

    // common compile time args for sender and receiver
    // Note: Fabric Mcast with NOC writes to DRAM provides redudant coverage,
    // so use_dram_dst is set to 0; see run_unicast_bw_chips() for DRAM coverage
    std::vector<uint32_t> compile_time_args = {
        worker_mem_map.test_results_address,
        worker_mem_map.test_results_size_bytes,
        worker_mem_map.target_address,
        0 /* use_dram_dst */};

    std::map<std::string, std::string> defines = {};
    if (is_2d_fabric) {
        defines["FABRIC_2D"] = "";
    }

    auto sender_program = tt_metal::CreateProgram();
    auto sender_kernel = tt_metal::CreateKernel(
        sender_program,
        "tests/tt_metal/tt_metal/perf_microbenchmark/routing/kernels/tt_fabric_line_mcast_tx.cpp",
        {sender_logical_core},
        tt_metal::DataMovementConfig{
            .processor = tt_metal::DataMovementProcessor::RISCV_0,
            .noc = tt_metal::NOC::RISCV_0_default,
            .compile_args = compile_time_args,
            .defines = defines});

    std::vector<uint32_t> sender_runtime_args = {
        worker_mem_map.source_l1_buffer_address,
        worker_mem_map.packet_payload_size_bytes,
        num_packets,
        receiver_noc_encoding,
        time_seed,
        mcast_start_id.chip_id,
        *mcast_start_id.mesh_id};

    std::vector<uint32_t> mcast_header_rtas(4, 0);
    for (const auto& routing_info : mcast_routing_info) {
        mcast_header_rtas[static_cast<uint32_t>(
            control_plane.routing_direction_to_eth_direction(routing_info.mcast_dir))] = routing_info.num_mcast_hops;
    }
    sender_runtime_args.insert(sender_runtime_args.end(), mcast_header_rtas.begin(), mcast_header_rtas.end());
    // append the EDM connection rt args
    append_fabric_connection_rt_args(
        sender_id, mcast_start_id, 0, sender_program, {sender_logical_core}, sender_runtime_args);

    tt_metal::SetRuntimeArgs(sender_program, sender_kernel, sender_logical_core, sender_runtime_args);

    // Create the receiver programs for validation on all devices involved in the Mcast
    std::vector<uint32_t> receiver_runtime_args = {worker_mem_map.packet_payload_size_bytes, num_packets, time_seed};
    std::unordered_map<std::shared_ptr<tt_metal::distributed::MeshDevice>, std::shared_ptr<tt_metal::Program>>
        recv_programs;

    for (const auto& dev : mcast_group_devices) {
        recv_programs[dev] = create_receiver_program(compile_time_args, receiver_runtime_args, receiver_logical_core);
    }

    // Launch sender and receiver programs and wait for them to finish
    // NOLINTNEXTLINE(bugprone-nondeterministic-pointer-iteration-order)
    for (auto& [dev, recv_program] : recv_programs) {
        log_info(tt::LogTest, "Run receiver on: {}", dev->get_devices()[0]->id());
        fixture->RunProgramNonblocking(dev, *recv_program);
    }
    log_info(tt::LogTest, "Run Sender on: {}", sender_device->get_devices()[0]->id());
    fixture->RunProgramNonblocking(sender_device, sender_program);

    // NOLINTNEXTLINE(bugprone-nondeterministic-pointer-iteration-order)
    for (auto& [dev, recv_program] : recv_programs) {
        fixture->WaitForSingleProgramDone(dev, *recv_program);
    }
    fixture->WaitForSingleProgramDone(sender_device, sender_program);

    std::vector<uint32_t> sender_status;
    tt_metal::detail::ReadFromDeviceL1(
        sender_device->get_devices()[0],
        sender_logical_core,
        worker_mem_map.test_results_address,
        worker_mem_map.test_results_size_bytes,
        sender_status,
        CoreType::WORKER);

    EXPECT_EQ(sender_status[TT_FABRIC_STATUS_INDEX], TT_FABRIC_STATUS_PASS);
    uint64_t sender_bytes =
        ((uint64_t)sender_status[TT_FABRIC_WORD_CNT_INDEX + 1] << 32) | sender_status[TT_FABRIC_WORD_CNT_INDEX];

    // NOLINTNEXTLINE(bugprone-nondeterministic-pointer-iteration-order)
    for (auto& [dev, _] : recv_programs) {
        std::vector<uint32_t> receiver_status;
        tt_metal::detail::ReadFromDeviceL1(
            dev->get_devices()[0],
            receiver_logical_core,
            worker_mem_map.test_results_address,
            worker_mem_map.test_results_size_bytes,
            receiver_status,
            CoreType::WORKER);

        EXPECT_EQ(receiver_status[TT_FABRIC_STATUS_INDEX], TT_FABRIC_STATUS_PASS);
        uint64_t receiver_bytes =
            ((uint64_t)receiver_status[TT_FABRIC_WORD_CNT_INDEX + 1] << 32) | receiver_status[TT_FABRIC_WORD_CNT_INDEX];

        EXPECT_EQ(sender_bytes, receiver_bytes);
    }
}

void RunTestUnicastRaw(BaseFabricFixture* fixture, uint32_t num_hops, RoutingDirection direction) {
    CoreCoord sender_logical_core = {0, 0};
    CoreCoord receiver_logical_core = {1, 0};

    auto& control_plane = tt::tt_metal::MetalContext::instance().get_control_plane();

    FabricNodeId src_fabric_node_id(MeshId{0}, 0);
    FabricNodeId dst_fabric_node_id(MeshId{0}, 0);
    // Find a device num_hops away in specified direction.
    std::unordered_map<RoutingDirection, uint32_t> fabric_hops;
    std::unordered_map<RoutingDirection, std::vector<FabricNodeId>> end_fabric_node_ids_by_dir;
    ChipId src_physical_device_id;
    ChipId dst_physical_device_id;
    std::unordered_map<RoutingDirection, std::vector<ChipId>> physical_end_device_ids_by_dir;
    fabric_hops[direction] = num_hops;

    tt::tt_metal::distributed::MeshShape mesh_shape;
    std::vector<chan_id_t> eth_chans;
    chan_id_t edm_port;

    const auto& fabric_context = control_plane.get_fabric_context();
    const auto topology = fabric_context.get_fabric_topology();
    uint32_t is_2d_fabric = topology == Topology::Mesh;

    if (!is_2d_fabric) {
        // Find a device with enough neighbours in the specified directions
        if (!find_device_with_neighbor_in_multi_direction(
                fixture,
                src_fabric_node_id,
                end_fabric_node_ids_by_dir,
                src_physical_device_id,
                physical_end_device_ids_by_dir,
                fabric_hops)) {
            GTEST_SKIP() << "No path found between sender and receivers";
        }
        mesh_shape = control_plane.get_physical_mesh_shape(src_fabric_node_id.mesh_id);
        dst_physical_device_id = physical_end_device_ids_by_dir[direction][num_hops - 1];
        dst_fabric_node_id = end_fabric_node_ids_by_dir[direction][num_hops - 1];

        // get a port to connect to
        eth_chans = control_plane.get_active_fabric_eth_channels_in_direction(src_fabric_node_id, direction);
        if (eth_chans.empty()) {
            GTEST_SKIP() << "No active eth chans to connect to";
        }
    } else {
        const auto& devices = fixture->get_devices();
        // create a list of available deive ids in a random order
        // In 2D routing the source and desitnation devices can be anywhere on the mesh.
        auto random_dev_list = get_random_numbers_from_range(0, devices.size() - 1, devices.size());

        // pick the first two in the list to be src and dst devices for the test.
        src_physical_device_id = devices[random_dev_list[0]]->get_devices()[0]->id();
        dst_physical_device_id = devices[random_dev_list[1]]->get_devices()[0]->id();
        src_fabric_node_id = control_plane.get_fabric_node_id_from_physical_chip_id(src_physical_device_id);
        dst_fabric_node_id = control_plane.get_fabric_node_id_from_physical_chip_id(dst_physical_device_id);
        mesh_shape = control_plane.get_physical_mesh_shape(src_fabric_node_id.mesh_id);

        eth_chans = control_plane.get_forwarding_eth_chans_to_chip(src_fabric_node_id, dst_fabric_node_id);
        if (eth_chans.empty()) {
            log_info(
                tt::LogTest,
                "No fabric routers between Src MeshId {} ChipId {} - Dst MeshId {} ChipId {}",
                *(src_fabric_node_id.mesh_id),
                src_fabric_node_id.chip_id,
                *(dst_fabric_node_id.mesh_id),
                dst_fabric_node_id.chip_id);

            GTEST_SKIP() << "Skipping Test";
        }
    }

    // Pick any port, for now pick the 1st one in the set
    edm_port = *eth_chans.begin();

    log_info(tt::LogTest, "mesh dimensions {:x}", mesh_shape.dims());
    log_info(tt::LogTest, "mesh size {:x}", mesh_shape.mesh_size());
    log_info(tt::LogTest, "mesh dimension 0 {:x}", mesh_shape[0]);
    log_info(tt::LogTest, "mesh dimension 1 {:x}", mesh_shape[1]);
    log_info(tt::LogTest, "Src MeshId {} ChipId {}", *(src_fabric_node_id.mesh_id), src_fabric_node_id.chip_id);
    log_info(tt::LogTest, "Dst MeshId {} ChipId {}", *(dst_fabric_node_id.mesh_id), dst_fabric_node_id.chip_id);

    auto edm_direction = control_plane.get_eth_chan_direction(src_fabric_node_id, edm_port);
    log_info(tt::LogTest, "Using edm port {} in direction {}", edm_port, edm_direction);

    auto sender_device = fixture->get_device(src_physical_device_id);
    auto receiver_device = fixture->get_device(dst_physical_device_id);
    CoreCoord receiver_virtual_core = receiver_device->worker_core_from_logical_core(receiver_logical_core);

    // test parameters
    auto worker_mem_map = generate_worker_mem_map(sender_device, topology);
    uint32_t num_packets = 10;
    uint32_t time_seed = std::chrono::system_clock::now().time_since_epoch().count();

    // common compile time args for sender and receiver
    // Note: see run_unicast_dw_chips() for DRAM coverage
    std::vector<uint32_t> compile_time_args = {
        worker_mem_map.test_results_address,
        worker_mem_map.test_results_size_bytes,
        worker_mem_map.target_address,
        0 /* use_dram_dst */,
        topology == Topology::Mesh,
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

    auto worker_teardown_semaphore_id = tt_metal::CreateSemaphore(sender_program, sender_logical_core, 0);
    auto worker_buffer_index_semaphore_id = tt_metal::CreateSemaphore(sender_program, sender_logical_core, 0);
    append_worker_to_fabric_edm_sender_rt_args(
        edm_port, worker_teardown_semaphore_id, worker_buffer_index_semaphore_id, sender_runtime_args);

    tt_metal::SetRuntimeArgs(sender_program, sender_kernel, sender_logical_core, sender_runtime_args);

    std::vector<uint32_t> receiver_runtime_args = {worker_mem_map.packet_payload_size_bytes, num_packets, time_seed};

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
        sender_device->get_devices()[0],
        sender_logical_core,
        worker_mem_map.test_results_address,
        worker_mem_map.test_results_size_bytes,
        sender_status,
        CoreType::WORKER);

    tt_metal::detail::ReadFromDeviceL1(
        receiver_device->get_devices()[0],
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

void run_unicast_test_bw_chips(
    BaseFabricFixture* fixture,
    ChipId src_physical_device_id,
    ChipId dst_physical_device_id,
    uint32_t num_hops,
    bool use_dram_dst = false) {
    CoreCoord sender_logical_core = {0, 0};
    CoreCoord receiver_logical_core = {1, 0};

    const auto& control_plane= tt::tt_metal::MetalContext::instance().get_control_plane();
    auto src_fabric_node_id = control_plane.get_fabric_node_id_from_physical_chip_id(src_physical_device_id);
    auto dst_fabric_node_id = control_plane.get_fabric_node_id_from_physical_chip_id(dst_physical_device_id);

    auto sender_device = fixture->get_device(src_physical_device_id);
    auto receiver_device = fixture->get_device(dst_physical_device_id);
    CoreCoord receiver_virtual_core = receiver_device->worker_core_from_logical_core(receiver_logical_core);

    const auto topology = control_plane.get_fabric_context().get_fabric_topology();
    uint32_t is_2d_fabric = topology == Topology::Mesh;

    // test parameters
    auto worker_mem_map = generate_worker_mem_map(sender_device, topology);
    uint32_t num_packets = 10;
    uint32_t time_seed = std::chrono::system_clock::now().time_since_epoch().count();

    // common compile time args for sender and receiver
    std::vector<uint32_t> compile_time_args = {
        worker_mem_map.test_results_address,
        worker_mem_map.test_results_size_bytes,
        worker_mem_map.target_address,
        use_dram_dst,
        topology == Topology::Mesh,
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
    EXPECT_EQ(!available_links.empty(), true);

    uint32_t link_idx = available_links[0];
    append_fabric_connection_rt_args(
        src_fabric_node_id, dst_fabric_node_id, link_idx, sender_program, {sender_logical_core}, sender_runtime_args);

    tt_metal::SetRuntimeArgs(sender_program, sender_kernel, sender_logical_core, sender_runtime_args);

    // If using DRAM destination, zero out the mailbox
    // Simple notification mailbox with flushing atomic increment is used instead of 2-way handshake for simple testing
    if (use_dram_dst) {
        std::vector<uint32_t> zeros(tt::tt_metal::hal::get_l1_alignment() / sizeof(uint32_t), 0);  // zero out mailbox
        tt_metal::detail::WriteToDeviceL1(
            receiver_device->get_devices()[0],
            receiver_logical_core,
            worker_mem_map.notification_mailbox_address,
            zeros,
            CoreType::WORKER);
        tt::tt_metal::MetalContext::instance().get_cluster().l1_barrier(receiver_device->get_devices()[0]->id());
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
        sender_device->get_devices()[0],
        sender_logical_core,
        worker_mem_map.test_results_address,
        worker_mem_map.test_results_size_bytes,
        sender_status,
        CoreType::WORKER);

    tt_metal::detail::ReadFromDeviceL1(
        receiver_device->get_devices()[0],
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

void RunTestUnicastConnAPI(BaseFabricFixture* fixture, uint32_t num_hops, RoutingDirection direction, bool use_dram_dst) {
    const auto& control_plane= tt::tt_metal::MetalContext::instance().get_control_plane();

    FabricNodeId src_fabric_node_id(MeshId{0}, 0);
    FabricNodeId dst_fabric_node_id(MeshId{0}, 0);
    ChipId not_used_1;
    ChipId not_used_2;
    // Find a device with a neighbour in the East direction
    bool connection_found = find_device_with_neighbor_in_direction(
        fixture, src_fabric_node_id, dst_fabric_node_id, not_used_1, not_used_2, direction);
    if (!connection_found) {
        GTEST_SKIP() << "No path found between sender and receivers";
    }

    log_info(tt::LogTest, "Src MeshId {} ChipId {}", src_fabric_node_id.mesh_id, src_fabric_node_id.chip_id);
    log_info(tt::LogTest, "Dst MeshId {} ChipId {}", dst_fabric_node_id.mesh_id, dst_fabric_node_id.chip_id);
    log_info(tt::LogTest, "Dst Device is {} hops in direction: {}", num_hops, direction);

    ChipId src_physical_device_id = control_plane.get_physical_chip_id_from_fabric_node_id(src_fabric_node_id);
    ChipId dst_physical_device_id = control_plane.get_physical_chip_id_from_fabric_node_id(dst_fabric_node_id);

    run_unicast_test_bw_chips(fixture, src_physical_device_id, dst_physical_device_id, num_hops, use_dram_dst);
}

void RunTestUnicastConnAPIRandom(BaseFabricFixture* fixture) {
    const auto topology = tt::tt_metal::MetalContext::instance()
                              .get_control_plane()
                              .get_fabric_context()
                              .get_fabric_topology();
    uint32_t is_2d_fabric = topology == Topology::Mesh;
    if (!is_2d_fabric) {
        GTEST_SKIP() << "This test is only supported for 2D fabric currently";
    }

    auto devices = fixture->get_devices();
    // create a list of available deive ids in a random order
    // In 2D routing the source and desitnation devices can be anywhere on the mesh.
    auto random_dev_list = get_random_numbers_from_range(0, devices.size() - 1, 2);

    const auto src_physical_device_id = devices[random_dev_list[0]]->get_devices()[0]->id();
    const auto dst_physical_device_id = devices[random_dev_list[1]]->get_devices()[0]->id();

    log_info(tt::LogTest, "Src Phys ChipId {}", src_physical_device_id);
    log_info(tt::LogTest, "Dst Phys ChipId {}", dst_physical_device_id);

    run_unicast_test_bw_chips(
        fixture, src_physical_device_id, dst_physical_device_id, 0 /* num_hops, not needed for 2d */);
}

void RunTestUnicastRaw2D(
    BaseFabricFixture* fixture, uint32_t ns_hops, RoutingDirection ns_dir, uint32_t ew_hops, RoutingDirection ew_dir) {
    auto& control_plane = tt::tt_metal::MetalContext::instance().get_control_plane();

    // use control plane to find a mesh with (ns_hops + 1) * (ew_hops + 1) devices
    auto user_meshes = control_plane.get_user_physical_mesh_ids();
    std::optional<MeshId> mesh_id;
    for (const auto& mesh : user_meshes) {
        auto mesh_shape = control_plane.get_physical_mesh_shape(mesh);
        if (mesh_shape.mesh_size() >= (ns_hops + 1) * (ew_hops + 1)) {
            mesh_id = mesh;
            break;
        }
    }
    if (!mesh_id.has_value()) {
        GTEST_SKIP() << "No appropriate mesh found for 2d unicast test";
    }

    // Find a device num_hops away in specified direction.
    FabricNodeId src_fabric_node_id(MeshId{0}, 0);
    std::unordered_map<RoutingDirection, uint32_t> fabric_hops;
    std::unordered_map<RoutingDirection, uint32_t> branch_hops;

    std::unordered_map<RoutingDirection, std::vector<FabricNodeId>> end_fabric_node_ids_by_dir;
    ChipId src_phys_chip_id;
    std::unordered_map<RoutingDirection, std::vector<ChipId>> physical_end_device_ids_by_dir;

    if (ns_hops != 0) {
        fabric_hops[ns_dir] = ns_hops;
    }
    if (ew_hops != 0) {
        fabric_hops[ew_dir] = ew_hops;
    }

    tt::tt_metal::distributed::MeshShape mesh_shape;
    const auto topology = control_plane.get_fabric_context().get_fabric_topology();
    uint32_t is_2d_fabric = topology == Topology::Mesh;

    if (!is_2d_fabric) {
        GTEST_SKIP() << "Need 2D Fabric for this test.";
    }

    // Get the mcast sender device and mcast receiver devices that satisfy the input number of trunk hops
    if (!find_device_with_neighbor_in_multi_direction(
            fixture,
            src_fabric_node_id,
            end_fabric_node_ids_by_dir,
            src_phys_chip_id,
            physical_end_device_ids_by_dir,
            fabric_hops)) {
        log_info(
            tt::LogTest,
            "No destinations found for {} hops in direction {} and {} hops in direction {}.",
            ns_hops,
            ns_dir,
            ew_hops,
            ew_dir);
        GTEST_SKIP() << "Skipping Test";
    }

    mesh_shape = control_plane.get_physical_mesh_shape(src_fabric_node_id.mesh_id);
    uint32_t ew_dim = mesh_shape[1];
    auto device_offset = ns_dir == RoutingDirection::N ? -1 : 1;
    auto dst_fabric_node_id = src_fabric_node_id;
    if (ew_hops != 0) {
        dst_fabric_node_id = end_fabric_node_ids_by_dir[ew_dir][ew_hops - 1];
    }
    dst_fabric_node_id.chip_id += device_offset * ew_dim * ns_hops;
    auto dst_physical_device_id = control_plane.get_physical_chip_id_from_fabric_node_id(dst_fabric_node_id);
    auto src_physical_device_id = control_plane.get_physical_chip_id_from_fabric_node_id(src_fabric_node_id);

    log_info(tt::LogTest, "Src Phys ChipId {}", src_physical_device_id);
    log_info(tt::LogTest, "Dst Phys ChipId {}", dst_physical_device_id);

    run_unicast_test_bw_chips(
        fixture, src_physical_device_id, dst_physical_device_id, 0 /* num_hops, not needed for 2d */);
}

void RunTestUnicastTGGateways(BaseFabricFixture* fixture) {
    // TODO: remove this restriction once tunneling is disabled
    if (!fixture->slow_dispatch_) {
        log_info(tt::LogTest, "This test can only be run with TT_METAL_SLOW_DISPATCH_MODE currently");
        GTEST_SKIP();
    }

    if (tt::tt_metal::MetalContext::instance().get_cluster().get_cluster_type() != tt::tt_metal::ClusterType::TG) {
        log_info(tt::LogTest, "This test is only for TG");
        GTEST_SKIP();
    }

    // run tests b/w all pairs of TG gateways <> remote chip connections
    // this only tests connections with the immediate remote chip in the tunnel since other connections
    // are 'normal' and covered in other tests
    const std::vector<ChipId> mmio_chip_ids = {0, 1, 2, 3};
    for (const auto& mmio_chip_id : mmio_chip_ids) {
        const auto& tunnels_from_mmio =
            tt::tt_metal::MetalContext::instance().get_cluster().get_tunnels_from_mmio_device(mmio_chip_id);
        for (const auto& tunnel : tunnels_from_mmio) {
            // idx 0 in the tunnel is the mmio chip itself
            const auto remote_chip_id = tunnel[1];
            log_info(tt::LogTest, "Running tests for chips: {} and {}", mmio_chip_id, remote_chip_id);
            run_unicast_test_bw_chips(fixture, mmio_chip_id, remote_chip_id, 1);
            run_unicast_test_bw_chips(fixture, remote_chip_id, mmio_chip_id, 1);
        }
    }
}

void RunTestMCastConnAPI(
    BaseFabricFixture* fixture,
    RoutingDirection fwd_dir,
    uint32_t fwd_hops,
    RoutingDirection bwd_dir,
    uint32_t bwd_hops) {
    CoreCoord sender_logical_core = {0, 0};
    CoreCoord receiver_logical_core = {1, 0};
    std::vector<tt_metal::Program> receiver_programs;

    auto& control_plane = tt::tt_metal::MetalContext::instance().get_control_plane();

    // use control plane to find a mesh with 3 devices
    auto user_meshes = control_plane.get_user_physical_mesh_ids();
    std::optional<MeshId> mesh_id;
    for (const auto& mesh : user_meshes) {
        auto mesh_shape = control_plane.get_physical_mesh_shape(mesh);
        if (mesh_shape.mesh_size() >= 3) {
            mesh_id = mesh;
            break;
        }
    }
    if (!mesh_id.has_value()) {
        GTEST_SKIP() << "No mesh found for 3 chip mcast test";
    }

    // Find a device num_hops away in specified direction.
    FabricNodeId src_fabric_node_id(MeshId{0}, 0);
    std::unordered_map<RoutingDirection, uint32_t> fabric_hops;
    std::unordered_map<RoutingDirection, std::vector<FabricNodeId>> end_fabric_node_ids_by_dir;
    ChipId src_phys_chip_id;
    std::unordered_map<RoutingDirection, std::vector<ChipId>> physical_end_device_ids_by_dir;
    fabric_hops[fwd_dir] = fwd_hops;
    fabric_hops[bwd_dir] = bwd_hops;

    tt::tt_metal::distributed::MeshShape mesh_shape;
    const auto topology = control_plane.get_fabric_context().get_fabric_topology();
    uint32_t is_2d_fabric = topology == Topology::Mesh;

    // Get the mcast sender device and mcast receiver devices that satisfy the input number of hops in forward and
    // backward directions.
    if (!find_device_with_neighbor_in_multi_direction(
            fixture,
            src_fabric_node_id,
            end_fabric_node_ids_by_dir,
            src_phys_chip_id,
            physical_end_device_ids_by_dir,
            fabric_hops)) {
        log_info(
            tt::LogTest,
            "No Mcast destinations found for {} hops in {} and {} hops in {}",
            fwd_hops,
            fwd_dir,
            bwd_hops,
            bwd_dir);
        GTEST_SKIP() << "Skipping Test";
    }

    mesh_shape = control_plane.get_physical_mesh_shape(src_fabric_node_id.mesh_id);
    auto left_recv_phys_chip_id = physical_end_device_ids_by_dir[fwd_dir][fwd_hops - 1];
    auto left_first_hop_phys_chip_id = physical_end_device_ids_by_dir[fwd_dir][0];
    auto right_recv_phys_chip_id = physical_end_device_ids_by_dir[bwd_dir][bwd_hops - 1];
    auto right_first_hop_phys_chip_id = physical_end_device_ids_by_dir[bwd_dir][0];

    auto sender_device = fixture->get_device(src_phys_chip_id);
    auto left_recv_device = fixture->get_device(left_recv_phys_chip_id);
    auto right_recv_device = fixture->get_device(right_recv_phys_chip_id);

    auto left_fabric_node_id = end_fabric_node_ids_by_dir[fwd_dir][fwd_hops - 1];
    auto right_fabric_node_id = end_fabric_node_ids_by_dir[bwd_dir][bwd_hops - 1];

    CoreCoord receiver_virtual_core = left_recv_device->worker_core_from_logical_core(receiver_logical_core);

    // test parameters
    auto worker_mem_map = generate_worker_mem_map(sender_device, topology);
    uint32_t num_packets = 100;
    uint32_t time_seed = std::chrono::system_clock::now().time_since_epoch().count();

    // common compile time args for sender and receiver
    // Note: Fabric Mcast with NOC writes to DRAM provides redudant coverage,
    // so use_dram_dst is set to 0; see run_unicast_dw_chips() for DRAM coverage
    std::vector<uint32_t> compile_time_args = {
        worker_mem_map.test_results_address,
        worker_mem_map.test_results_size_bytes,
        worker_mem_map.target_address,
        0 /* use_dram_dst */,
        topology == Topology::Mesh,
        1 /* is_chip_multicast */,
        1 /* additional_dir */};

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

    log_info(tt::LogTest, "mesh dimensions {:x}", mesh_shape.dims());
    log_info(tt::LogTest, "mesh dimension 0 {:x}", mesh_shape[0]);
    log_info(tt::LogTest, "mesh dimension 1 {:x}", mesh_shape[1]);
    log_info(tt::LogTest, "Mcast Src MeshId {} ChipId {}", *(src_fabric_node_id.mesh_id), src_fabric_node_id.chip_id);
    log_info(
        tt::LogTest, "Mcast Fwd Dst MeshId {} ChipId {}", *(left_fabric_node_id.mesh_id), left_fabric_node_id.chip_id);
    log_info(tt::LogTest, "Mcast Fwd Dst Device is {} hops in direction: {}", fwd_hops, fwd_dir);
    log_info(
        tt::LogTest,
        "Mcast Bwd Dst MeshId {} ChipId {}",
        *(right_fabric_node_id.mesh_id),
        right_fabric_node_id.chip_id);
    log_info(tt::LogTest, "Mcast Bwd Dst Device is {} hops in direction: {}", bwd_hops, bwd_dir);

    std::vector<uint32_t> sender_runtime_args = {
        worker_mem_map.source_l1_buffer_address,
        worker_mem_map.packet_payload_size_bytes,
        num_packets,
        time_seed,
        receiver_virtual_core.x,
        receiver_virtual_core.y,
        mesh_shape[1],
        src_fabric_node_id.chip_id,
        1 /* fwd_start_distance */,
        fwd_hops /* fwd_range */,
        left_fabric_node_id.chip_id,
        *left_fabric_node_id.mesh_id};

    // append the EDM connection rt args for fwd connection
    ChipId dst_chip_id;
    uint32_t link_idx;

    if (is_2d_fabric) {
        dst_chip_id = left_recv_phys_chip_id;
    } else {
        dst_chip_id = left_first_hop_phys_chip_id;
    }
    link_idx =
        get_forwarding_link_indices(src_fabric_node_id, get_fabric_node_id_from_physical_chip_id(dst_chip_id))[0];
    const auto left_dst_fabric_node_id = tt::tt_fabric::get_fabric_node_id_from_physical_chip_id(dst_chip_id);
    append_fabric_connection_rt_args(
        src_fabric_node_id,
        left_dst_fabric_node_id,
        link_idx,
        sender_program,
        {sender_logical_core},
        sender_runtime_args);
    sender_runtime_args.push_back(1); /* bwd_start_distance */
    sender_runtime_args.push_back(bwd_hops); /* bwd_range */
    sender_runtime_args.push_back(right_fabric_node_id.chip_id);
    sender_runtime_args.push_back(*right_fabric_node_id.mesh_id);

    if (is_2d_fabric) {
        dst_chip_id = right_recv_phys_chip_id;
    } else {
        dst_chip_id = right_first_hop_phys_chip_id;
    }
    link_idx =
        get_forwarding_link_indices(src_fabric_node_id, get_fabric_node_id_from_physical_chip_id(dst_chip_id))[0];
    const auto right_dst_fabric_node_id = tt::tt_fabric::get_fabric_node_id_from_physical_chip_id(dst_chip_id);
    append_fabric_connection_rt_args(
        src_fabric_node_id,
        right_dst_fabric_node_id,
        link_idx,
        sender_program,
        {sender_logical_core},
        sender_runtime_args);

    tt_metal::SetRuntimeArgs(sender_program, sender_kernel, sender_logical_core, sender_runtime_args);

    std::vector<uint32_t> receiver_runtime_args = {worker_mem_map.packet_payload_size_bytes, num_packets, time_seed};

    // Create and launch the receiver program for validation on all mcast receiver devices
    for (auto& [routing_direction, physical_end_device_ids] : physical_end_device_ids_by_dir) {
        for (auto physical_end_device_id : physical_end_device_ids) {
            auto receiver_device = fixture->get_device(physical_end_device_id);
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
            fixture->RunProgramNonblocking(receiver_device, receiver_program);
            receiver_programs.push_back(std::move(receiver_program));
            log_info(tt::LogTest, "{} Rx Launched on physical device {}", routing_direction, physical_end_device_id);
        }
    }

    // Launch sender program and wait for sender to finish
    fixture->RunProgramNonblocking(sender_device, sender_program);
    fixture->WaitForSingleProgramDone(sender_device, sender_program);

    // Wait for receivers to finish
    for (const auto& [routing_direction, physical_end_device_ids] : physical_end_device_ids_by_dir) {
        for (uint32_t i = 0; i < physical_end_device_ids.size(); i++) {
            auto receiver_device = fixture->get_device(physical_end_device_ids[i]);
            fixture->WaitForSingleProgramDone(receiver_device, receiver_programs[i]);
        }
    }
    log_info(tt::LogTest, "All Receivers Finished");

    // Validate the status and packets processed by sender and receiver
    std::vector<uint32_t> sender_status;
    std::vector<uint32_t> left_recv_status;
    std::vector<uint32_t> right_recv_status;

    tt_metal::detail::ReadFromDeviceL1(
        sender_device->get_devices()[0],
        sender_logical_core,
        worker_mem_map.test_results_address,
        worker_mem_map.test_results_size_bytes,
        sender_status,
        CoreType::WORKER);

    EXPECT_EQ(sender_status[TT_FABRIC_STATUS_INDEX], TT_FABRIC_STATUS_PASS);

    uint64_t sender_bytes =
        ((uint64_t)sender_status[TT_FABRIC_WORD_CNT_INDEX + 1] << 32) | sender_status[TT_FABRIC_WORD_CNT_INDEX];

    for (const auto& [routing_direction, physical_end_device_ids] : physical_end_device_ids_by_dir) {
        for (int device_id : physical_end_device_ids) {
            log_info(tt::LogTest, "Checking Status of {} Rx on physical device {}", routing_direction, device_id);

            const auto& receiver_device = fixture->get_device(device_id);
            std::vector<uint32_t> recv_status;

            tt_metal::detail::ReadFromDeviceL1(
                receiver_device->get_devices()[0],
                receiver_logical_core,
                worker_mem_map.test_results_address,
                worker_mem_map.test_results_size_bytes,
                recv_status,
                CoreType::WORKER);
            EXPECT_EQ(recv_status[TT_FABRIC_STATUS_INDEX], TT_FABRIC_STATUS_PASS);
            uint64_t recv_bytes =
                ((uint64_t)recv_status[TT_FABRIC_WORD_CNT_INDEX + 1] << 32) | recv_status[TT_FABRIC_WORD_CNT_INDEX];

            EXPECT_EQ(sender_bytes, recv_bytes);
        }
    }
}

void RunTest2DMCastConnAPI(
    BaseFabricFixture* fixture, uint32_t north_hops, uint32_t south_hops, uint32_t east_hops, uint32_t west_hops) {
    uint32_t north_branch_east_hops = east_hops;
    uint32_t north_branch_west_hops = west_hops;
    uint32_t south_branch_east_hops = east_hops;
    uint32_t south_branch_west_hops = west_hops;

    CoreCoord sender_logical_core = {0, 0};
    CoreCoord receiver_logical_core = {1, 0};
    std::vector<tt_metal::Program> receiver_programs;

    auto& control_plane = tt::tt_metal::MetalContext::instance().get_control_plane();

    // use control plane to find a mesh with 3 devices
    auto user_meshes = control_plane.get_user_physical_mesh_ids();
    std::optional<MeshId> mesh_id;
    for (const auto& mesh : user_meshes) {
        auto mesh_shape = control_plane.get_physical_mesh_shape(mesh);
        if (mesh_shape.mesh_size() >= 3) {
            mesh_id = mesh;
            break;
        }
    }
    if (!mesh_id.has_value()) {
        GTEST_SKIP() << "No mesh found for 3 chip mcast test";
    }

    // Find a device num_hops away in specified direction.
    FabricNodeId src_fabric_node_id(MeshId{0}, 0);
    std::unordered_map<RoutingDirection, uint32_t> fabric_hops;

    std::unordered_map<RoutingDirection, std::vector<FabricNodeId>> end_fabric_node_ids_by_dir;
    ChipId src_phys_chip_id;
    std::unordered_map<RoutingDirection, std::vector<ChipId>> physical_end_device_ids_by_dir;

    if (north_hops > 0) {
        fabric_hops[RoutingDirection::N] = north_hops;
    }
    if (south_hops > 0) {
        fabric_hops[RoutingDirection::S] = south_hops;
    }
    if (east_hops > 0) {
        fabric_hops[RoutingDirection::E] = std::max({north_branch_east_hops, south_branch_east_hops, east_hops});
    }
    if (west_hops > 0) {
        fabric_hops[RoutingDirection::W] = std::max({north_branch_west_hops, south_branch_west_hops, west_hops});
    }

    tt::tt_metal::distributed::MeshShape mesh_shape;
    const auto topology = control_plane.get_fabric_context().get_fabric_topology();
    uint32_t is_2d_fabric = topology == Topology::Mesh;

    if (!is_2d_fabric) {
        GTEST_SKIP() << "Need 2D Fabric for this test.";
    }

    // Get the mcast sender device and mcast receiver devices that satisfy the input number of trunk hops
    if (!find_device_with_neighbor_in_multi_direction(
            fixture,
            src_fabric_node_id,
            end_fabric_node_ids_by_dir,
            src_phys_chip_id,
            physical_end_device_ids_by_dir,
            fabric_hops)) {
        log_info(
            tt::LogTest,
            "No Mcast destinations found for {} North Hops with {} East and {} West branch hops, {} South Hops with {} "
            "East and {} West branch hops, and {} East and {} West direct hops.",
            north_hops,
            north_branch_east_hops,
            north_branch_west_hops,
            south_hops,
            south_branch_east_hops,
            south_branch_west_hops,
            east_hops,
            west_hops);
        GTEST_SKIP() << "Skipping Test";
    }

    mesh_shape = control_plane.get_physical_mesh_shape(src_fabric_node_id.mesh_id);
    uint32_t ew_dim = mesh_shape[1];
    auto north_offset = -1;
    auto south_offset = 1;
    auto north_fabric_node_id = src_fabric_node_id;
    auto south_fabric_node_id = src_fabric_node_id;
    auto north_east_fabric_node_id = src_fabric_node_id;
    auto north_west_fabric_node_id = src_fabric_node_id;
    auto north_recv_phys_chip_id = src_phys_chip_id;
    auto south_recv_phys_chip_id = src_phys_chip_id;
    auto right_recv_phys_chip_id = src_phys_chip_id;
    auto left_recv_phys_chip_id = src_phys_chip_id;
    auto south_east_fabric_node_id = src_fabric_node_id;
    auto south_west_fabric_node_id = src_fabric_node_id;
    auto right_fabric_node_id = src_fabric_node_id;
    auto left_fabric_node_id = src_fabric_node_id;

    if (south_hops > 0) {
        south_fabric_node_id = end_fabric_node_ids_by_dir[RoutingDirection::S][south_hops - 1];
        south_recv_phys_chip_id = control_plane.get_physical_chip_id_from_fabric_node_id(south_fabric_node_id);
        if (south_branch_east_hops > 0) {
            south_east_fabric_node_id = end_fabric_node_ids_by_dir[RoutingDirection::E][south_branch_east_hops - 1];
            south_east_fabric_node_id.chip_id += south_offset * ew_dim;
            right_recv_phys_chip_id = control_plane.get_physical_chip_id_from_fabric_node_id(south_east_fabric_node_id);
        }
        if (south_branch_west_hops > 0) {
            south_west_fabric_node_id = end_fabric_node_ids_by_dir[RoutingDirection::W][south_branch_west_hops - 1];
            south_west_fabric_node_id.chip_id += south_offset * ew_dim;
            left_recv_phys_chip_id = control_plane.get_physical_chip_id_from_fabric_node_id(south_west_fabric_node_id);
        }
    }
    if (north_hops > 0) {
        north_fabric_node_id = end_fabric_node_ids_by_dir[RoutingDirection::N].at(0);
        north_recv_phys_chip_id = control_plane.get_physical_chip_id_from_fabric_node_id(north_fabric_node_id);
        if (north_branch_east_hops > 0) {
            north_east_fabric_node_id = end_fabric_node_ids_by_dir[RoutingDirection::E][north_branch_east_hops - 1];
            north_east_fabric_node_id.chip_id += north_offset * ew_dim;
            right_recv_phys_chip_id = control_plane.get_physical_chip_id_from_fabric_node_id(north_east_fabric_node_id);
        }
        if (north_branch_west_hops > 0) {
            north_west_fabric_node_id = end_fabric_node_ids_by_dir[RoutingDirection::W][north_branch_west_hops - 1];
            north_west_fabric_node_id.chip_id += north_offset * ew_dim;
            left_recv_phys_chip_id = control_plane.get_physical_chip_id_from_fabric_node_id(north_west_fabric_node_id);
        }
    }

    if (east_hops > 0) {
        right_fabric_node_id = end_fabric_node_ids_by_dir[RoutingDirection::E][east_hops - 1];
        right_recv_phys_chip_id = control_plane.get_physical_chip_id_from_fabric_node_id(right_fabric_node_id);
    }
    if (west_hops > 0) {
        left_fabric_node_id = end_fabric_node_ids_by_dir[RoutingDirection::W][west_hops - 1];
        left_recv_phys_chip_id = control_plane.get_physical_chip_id_from_fabric_node_id(left_fabric_node_id);
    }

    std::vector<uint32_t> rx_physical_device_ids;
    for (size_t i = 0; i < end_fabric_node_ids_by_dir[RoutingDirection::E].size(); i++) {
        if (i < east_hops) {
            auto east_node = end_fabric_node_ids_by_dir[RoutingDirection::E][i];
            rx_physical_device_ids.push_back(control_plane.get_physical_chip_id_from_fabric_node_id(east_node));
        }
    }
    for (size_t i = 0; i < end_fabric_node_ids_by_dir[RoutingDirection::W].size(); i++) {
        if (i < west_hops) {
            auto west_node = end_fabric_node_ids_by_dir[RoutingDirection::W][i];
            rx_physical_device_ids.push_back(control_plane.get_physical_chip_id_from_fabric_node_id(west_node));
        }
    }
    // North branch hops
    uint32_t trunk_hop = 1;
    for (auto trunk_node : end_fabric_node_ids_by_dir[RoutingDirection::N]) {
        rx_physical_device_ids.push_back(control_plane.get_physical_chip_id_from_fabric_node_id(trunk_node));
        for (size_t i = 0; i < end_fabric_node_ids_by_dir[RoutingDirection::E].size(); i++) {
            auto east_node = end_fabric_node_ids_by_dir[RoutingDirection::E][i];
            if (i < north_branch_east_hops) {
                east_node.chip_id += north_offset * trunk_hop * ew_dim;
                rx_physical_device_ids.push_back(control_plane.get_physical_chip_id_from_fabric_node_id(east_node));
            }
        }
        for (size_t i = 0; i < end_fabric_node_ids_by_dir[RoutingDirection::W].size(); i++) {
            auto west_node = end_fabric_node_ids_by_dir[RoutingDirection::W][i];
            if (i < north_branch_west_hops) {
                west_node.chip_id += north_offset * trunk_hop * ew_dim;
                rx_physical_device_ids.push_back(control_plane.get_physical_chip_id_from_fabric_node_id(west_node));
            }
        }
        trunk_hop++;
    }
    // South branch hops
    trunk_hop = 1;
    for (auto trunk_node : end_fabric_node_ids_by_dir[RoutingDirection::S]) {
        rx_physical_device_ids.push_back(control_plane.get_physical_chip_id_from_fabric_node_id(trunk_node));
        for (size_t i = 0; i < end_fabric_node_ids_by_dir[RoutingDirection::E].size(); i++) {
            auto east_node = end_fabric_node_ids_by_dir[RoutingDirection::E][i];
            if (i < south_branch_east_hops) {
                east_node.chip_id += south_offset * trunk_hop * ew_dim;
                rx_physical_device_ids.push_back(control_plane.get_physical_chip_id_from_fabric_node_id(east_node));
            }
        }
        for (size_t i = 0; i < end_fabric_node_ids_by_dir[RoutingDirection::W].size(); i++) {
            auto west_node = end_fabric_node_ids_by_dir[RoutingDirection::W][i];
            if (i < south_branch_west_hops) {
                west_node.chip_id += south_offset * trunk_hop * ew_dim;
                rx_physical_device_ids.push_back(control_plane.get_physical_chip_id_from_fabric_node_id(west_node));
            }
        }
        trunk_hop++;
    }

    log_info(tt::LogTest, "mesh dimensions {:x}", mesh_shape.dims());
    log_info(tt::LogTest, "mesh dimension 0 {:x}", mesh_shape[0]);
    log_info(tt::LogTest, "mesh dimension 1 {:x}", ew_dim);
    log_info(tt::LogTest, "Mcast Src MeshId {} ChipId {}", *(src_fabric_node_id.mesh_id), src_fabric_node_id.chip_id);
    log_info(
        tt::LogTest,
        "Mcast North East Branch Dst MeshId {} ChipId {}",
        *(north_east_fabric_node_id.mesh_id),
        north_east_fabric_node_id.chip_id);
    log_info(
        tt::LogTest,
        "Mcast East Branch Dst Device is {} hops in direction: RoutingDirection::E",
        north_branch_east_hops);
    log_info(
        tt::LogTest,
        "Mcast West Branch Dst MeshId {} ChipId {}",
        *(north_west_fabric_node_id.mesh_id),
        north_west_fabric_node_id.chip_id);
    log_info(
        tt::LogTest,
        "Mcast West Branch Dst Device is {} hops in direction: RoutingDirection::W",
        north_branch_west_hops);
    log_info(
        tt::LogTest,
        "Mcast North East Branch Dst MeshId {} ChipId {}",
        *(south_east_fabric_node_id.mesh_id),
        south_east_fabric_node_id.chip_id);
    log_info(
        tt::LogTest,
        "Mcast East Branch Dst Device is {} hops in direction: RoutingDirection::E",
        south_branch_east_hops);
    log_info(
        tt::LogTest,
        "Mcast West Branch Dst MeshId {} ChipId {}",
        *(south_west_fabric_node_id.mesh_id),
        south_west_fabric_node_id.chip_id);
    log_info(
        tt::LogTest,
        "Mcast West Branch Dst Device is {} hops in direction: RoutingDirection::W",
        south_branch_west_hops);
    log_info(
        tt::LogTest,
        "Mcast Right Direct Dst MeshId {} ChipId {}",
        *(right_fabric_node_id.mesh_id),
        right_fabric_node_id.chip_id);
    log_info(tt::LogTest, "Mcast Right Direct Dst Device is {} hops in direction  : RoutingDirection::E", east_hops);
    log_info(
        tt::LogTest,
        "Mcast Left Direct Dst MeshId {} ChipId {}",
        *(left_fabric_node_id.mesh_id),
        left_fabric_node_id.chip_id);
    log_info(tt::LogTest, "Mcast Left Direct Dst Device is {} hops in direction : RoutingDirection::W", west_hops);

    auto dst_recv_phys_chip_id = left_recv_phys_chip_id;
    if (dst_recv_phys_chip_id == src_phys_chip_id) {
        dst_recv_phys_chip_id = right_recv_phys_chip_id;
    }
    if (dst_recv_phys_chip_id == src_phys_chip_id) {
        dst_recv_phys_chip_id = north_recv_phys_chip_id;
    }
    if (dst_recv_phys_chip_id == src_phys_chip_id) {
        dst_recv_phys_chip_id = south_recv_phys_chip_id;
    }
    if (dst_recv_phys_chip_id == src_phys_chip_id) {
        GTEST_SKIP() << "No dst chip id found";
    }

    auto sender_device = fixture->get_device(src_phys_chip_id);
    auto dst_recv_device = fixture->get_device(dst_recv_phys_chip_id);

    CoreCoord receiver_virtual_core = dst_recv_device->worker_core_from_logical_core(receiver_logical_core);

    auto receiver_noc_encoding =
        tt::tt_metal::MetalContext::instance().hal().noc_xy_encoding(receiver_virtual_core.x, receiver_virtual_core.y);
    // test parameters
    auto worker_mem_map = generate_worker_mem_map(sender_device, topology);
    uint32_t num_packets = 100;
    uint32_t time_seed = std::chrono::system_clock::now().time_since_epoch().count();

    uint32_t mcast_mode;
    auto arbitrary_fabric_node_id = src_fabric_node_id;
    if (north_hops > 0 && south_hops > 0) {
        mcast_mode = 3;
        if (north_branch_east_hops > 0) {
            arbitrary_fabric_node_id = north_east_fabric_node_id;
        } else if (north_branch_west_hops > 0) {
            arbitrary_fabric_node_id = north_west_fabric_node_id;
        } else {
            arbitrary_fabric_node_id = north_fabric_node_id;
        }
    } else if (south_hops > 0) {
        mcast_mode = 2;
        if (south_branch_west_hops > 0) {
            arbitrary_fabric_node_id = south_west_fabric_node_id;
        } else if (south_branch_east_hops > 0) {
            arbitrary_fabric_node_id = south_east_fabric_node_id;
        } else {
            arbitrary_fabric_node_id = south_fabric_node_id;
        }
    } else if (north_hops > 0) {
        mcast_mode = 1;
        if (north_branch_east_hops > 0) {
            arbitrary_fabric_node_id = north_east_fabric_node_id;
        } else if (north_branch_west_hops > 0) {
            arbitrary_fabric_node_id = north_west_fabric_node_id;
        } else {
            arbitrary_fabric_node_id = north_fabric_node_id;
        }
    } else {
        mcast_mode = 0;
        if (east_hops > 0) {
            arbitrary_fabric_node_id = right_fabric_node_id;
        } else {
            arbitrary_fabric_node_id = left_fabric_node_id;
        }
    }
    // common compile time args for sender and receiver
    std::vector<uint32_t> compile_time_args = {
        worker_mem_map.test_results_address,
        worker_mem_map.test_results_size_bytes,
        worker_mem_map.target_address,
        0 /* use_dram_dst */,
        mcast_mode,
        topology == Topology::Mesh,
        1 /* is_chip_multicast */,
        1 /* additional_dir */};

    std::map<std::string, std::string> defines = {};
    defines["FABRIC_2D"] = "";

    auto sender_program = tt_metal::CreateProgram();
    auto sender_kernel = tt_metal::CreateKernel(
        sender_program,
        "tests/tt_metal/tt_metal/perf_microbenchmark/routing/kernels/tt_fabric_2d_mcast_tx.cpp",
        {sender_logical_core},
        tt_metal::DataMovementConfig{
            .processor = tt_metal::DataMovementProcessor::RISCV_0,
            .noc = tt_metal::NOC::RISCV_0_default,
            .compile_args = compile_time_args,
            .defines = defines});

    std::vector<uint32_t> sender_runtime_args = {
        worker_mem_map.source_l1_buffer_address,
        worker_mem_map.packet_payload_size_bytes,
        num_packets,
        receiver_noc_encoding,
        time_seed,
        ew_dim,
        src_fabric_node_id.chip_id,
        *mesh_id.value(),
        north_hops,
        (north_branch_west_hops << 16) | north_branch_east_hops,
    };

    // append the EDM connection rt args for fwd connection
    uint32_t link_idx;
    if (north_hops > 0) {
        if (north_branch_east_hops > 0) {
            link_idx = get_forwarding_link_indices(src_fabric_node_id, north_east_fabric_node_id)[0];
            append_fabric_connection_rt_args(
                src_fabric_node_id,
                north_east_fabric_node_id,
                link_idx,
                sender_program,
                {sender_logical_core},
                sender_runtime_args);
        } else if (north_branch_west_hops) {
            link_idx = get_forwarding_link_indices(src_fabric_node_id, north_west_fabric_node_id)[0];
            append_fabric_connection_rt_args(
                src_fabric_node_id,
                north_west_fabric_node_id,
                link_idx,
                sender_program,
                {sender_logical_core},
                sender_runtime_args);
        } else {
            link_idx = get_forwarding_link_indices(src_fabric_node_id, north_fabric_node_id)[0];
            append_fabric_connection_rt_args(
                src_fabric_node_id,
                north_fabric_node_id,
                link_idx,
                sender_program,
                {sender_logical_core},
                sender_runtime_args);
        }
    } else {
        link_idx = 0;
        append_fabric_connection_rt_args(
            src_fabric_node_id,
            arbitrary_fabric_node_id,
            link_idx,
            sender_program,
            {sender_logical_core},
            sender_runtime_args);
    }
    sender_runtime_args.push_back(south_hops);
    sender_runtime_args.push_back((south_branch_west_hops << 16) | south_branch_east_hops);

    if (south_hops > 0) {
        if (south_branch_west_hops > 0) {
            link_idx = get_forwarding_link_indices(src_fabric_node_id, south_west_fabric_node_id)[0];
            append_fabric_connection_rt_args(
                src_fabric_node_id,
                south_west_fabric_node_id,
                link_idx,
                sender_program,
                {sender_logical_core},
                sender_runtime_args);
        } else if (south_branch_east_hops > 0) {
            link_idx = get_forwarding_link_indices(src_fabric_node_id, south_east_fabric_node_id)[0];
            append_fabric_connection_rt_args(
                src_fabric_node_id,
                south_east_fabric_node_id,
                link_idx,
                sender_program,
                {sender_logical_core},
                sender_runtime_args);
        } else {
            link_idx = get_forwarding_link_indices(src_fabric_node_id, south_fabric_node_id)[0];
            append_fabric_connection_rt_args(
                src_fabric_node_id,
                south_fabric_node_id,
                link_idx,
                sender_program,
                {sender_logical_core},
                sender_runtime_args);
        }
    } else {
        link_idx = 0;
        append_fabric_connection_rt_args(
            src_fabric_node_id,
            arbitrary_fabric_node_id,
            link_idx,
            sender_program,
            {sender_logical_core},
            sender_runtime_args);
    }
    sender_runtime_args.push_back(left_fabric_node_id.chip_id);
    sender_runtime_args.push_back(right_fabric_node_id.chip_id);
    sender_runtime_args.push_back((west_hops << 16) | east_hops);
    if (west_hops > 0) {
        link_idx = get_forwarding_link_indices(src_fabric_node_id, left_fabric_node_id)[0];
        append_fabric_connection_rt_args(
            src_fabric_node_id,
            left_fabric_node_id,
            link_idx,
            sender_program,
            {sender_logical_core},
            sender_runtime_args);
    } else {
        link_idx = 0;
        append_fabric_connection_rt_args(
            src_fabric_node_id,
            arbitrary_fabric_node_id,
            link_idx,
            sender_program,
            {sender_logical_core},
            sender_runtime_args);
    }
    if (east_hops > 0) {
        link_idx = get_forwarding_link_indices(src_fabric_node_id, right_fabric_node_id)[0];
        append_fabric_connection_rt_args(
            src_fabric_node_id,
            right_fabric_node_id,
            link_idx,
            sender_program,
            {sender_logical_core},
            sender_runtime_args);
    } else {
        link_idx = 0;
        append_fabric_connection_rt_args(
            src_fabric_node_id,
            arbitrary_fabric_node_id,
            link_idx,
            sender_program,
            {sender_logical_core},
            sender_runtime_args);
    }
    tt_metal::SetRuntimeArgs(sender_program, sender_kernel, sender_logical_core, sender_runtime_args);

    std::vector<uint32_t> receiver_runtime_args = {worker_mem_map.packet_payload_size_bytes, num_packets, time_seed};

    // Create and launch the receiver program for validation on all mcast receiver devices
    for (auto physical_end_device_id : rx_physical_device_ids) {
        auto receiver_device = fixture->get_device(physical_end_device_id);
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
        fixture->RunProgramNonblocking(receiver_device, receiver_program);
        receiver_programs.push_back(std::move(receiver_program));
        log_info(tt::LogTest, "Rx Launched on physical device {}", physical_end_device_id);
    }
    // Launch sender program and wait for sender to finish
    fixture->RunProgramNonblocking(sender_device, sender_program);
    fixture->WaitForSingleProgramDone(sender_device, sender_program);

    // Wait for receivers to finish
    for (uint32_t i = 0; i < rx_physical_device_ids.size(); i++) {
        auto receiver_device = fixture->get_device(rx_physical_device_ids[i]);
        fixture->WaitForSingleProgramDone(receiver_device, receiver_programs[i]);
    }
    log_info(tt::LogTest, "All Receivers Finished");

    // Validate the status and packets processed by sender and receiver
    std::vector<uint32_t> sender_status;
    std::vector<uint32_t> left_recv_status;
    std::vector<uint32_t> right_recv_status;

    tt_metal::detail::ReadFromDeviceL1(
        sender_device->get_devices()[0],
        sender_logical_core,
        worker_mem_map.test_results_address,
        worker_mem_map.test_results_size_bytes,
        sender_status,
        CoreType::WORKER);

    EXPECT_EQ(sender_status[TT_FABRIC_STATUS_INDEX], TT_FABRIC_STATUS_PASS);

    uint64_t sender_bytes =
        ((uint64_t)sender_status[TT_FABRIC_WORD_CNT_INDEX + 1] << 32) | sender_status[TT_FABRIC_WORD_CNT_INDEX];

    for (unsigned int rx_physical_device_id : rx_physical_device_ids) {
        log_info(tt::LogTest, "Checking Status of Rx on physical device {}", rx_physical_device_id);

        const auto& receiver_device = fixture->get_device(rx_physical_device_id);
        std::vector<uint32_t> recv_status;

        tt_metal::detail::ReadFromDeviceL1(
            receiver_device->get_devices()[0],
            receiver_logical_core,
            worker_mem_map.test_results_address,
            worker_mem_map.test_results_size_bytes,
            recv_status,
            CoreType::WORKER);
        EXPECT_EQ(recv_status[TT_FABRIC_STATUS_INDEX], TT_FABRIC_STATUS_PASS);
        uint64_t recv_bytes =
            ((uint64_t)recv_status[TT_FABRIC_WORD_CNT_INDEX + 1] << 32) | recv_status[TT_FABRIC_WORD_CNT_INDEX];

        EXPECT_EQ(sender_bytes, recv_bytes);
    }
}

void RunTestChipMCast1D(BaseFabricFixture* fixture, RoutingDirection dir, uint32_t start_distance, uint32_t range) {
    CoreCoord sender_logical_core = {0, 0};
    CoreCoord receiver_logical_core = {1, 0};
    std::vector<tt_metal::Program> receiver_programs;

    auto& control_plane = tt::tt_metal::MetalContext::instance().get_control_plane();

    // use control plane to find a mesh with 3 devices
    auto user_meshes = control_plane.get_user_physical_mesh_ids();
    std::optional<MeshId> mesh_id;
    for (const auto& mesh : user_meshes) {
        auto mesh_shape = control_plane.get_physical_mesh_shape(mesh);
        if (mesh_shape.mesh_size() >= 3) {
            mesh_id = mesh;
            break;
        }
    }
    if (!mesh_id.has_value()) {
        GTEST_SKIP() << "No mesh found for 3 chip mcast test";
    }

    // Check topology and fabric config
    const auto topology = control_plane.get_fabric_context().get_fabric_topology();
    const auto fabric_config = tt::tt_metal::MetalContext::instance().get_fabric_config();
    ASSERT_TRUE(
        (topology == Topology::Linear || topology == Topology::Ring) &&
        (fabric_config == tt_fabric::FabricConfig::FABRIC_1D ||
         fabric_config == tt_fabric::FabricConfig::FABRIC_1D_RING));

    // Find a device num_hops away in specified direction.
    FabricNodeId src_fabric_node_id(MeshId{0}, 0);
    std::unordered_map<RoutingDirection, uint32_t> fabric_hops;
    std::unordered_map<RoutingDirection, std::vector<FabricNodeId>> end_fabric_node_ids_by_dir;
    ChipId src_phys_chip_id;
    std::unordered_map<RoutingDirection, std::vector<ChipId>> physical_end_device_ids_by_dir;
    fabric_hops[dir] = start_distance + range - 1;

    // Get the mcast sender device and mcast receiver devices that satisfy the input number of hops in forward and
    // backward directions.
    if (!find_device_with_neighbor_in_multi_direction(
            fixture,
            src_fabric_node_id,
            end_fabric_node_ids_by_dir,
            src_phys_chip_id,
            physical_end_device_ids_by_dir,
            fabric_hops)) {
        log_info(tt::LogTest, "No Mcast destinations found for {} hops in {}", fabric_hops[dir], dir);
        GTEST_SKIP() << "Skipping Test";
    }

    // adjust physical_end_device_ids_by_dir and end_fabric_node_ids_by_dir to start from start_distance
    auto first_hop_phys_chip_id = physical_end_device_ids_by_dir[dir][0];  // needed to get link_idx
    physical_end_device_ids_by_dir[dir] = std::vector(
        physical_end_device_ids_by_dir[dir].begin() + start_distance - 1, physical_end_device_ids_by_dir[dir].end());
    end_fabric_node_ids_by_dir[dir] = std::vector(
        end_fabric_node_ids_by_dir[dir].begin() + start_distance - 1, end_fabric_node_ids_by_dir[dir].end());

    tt::tt_metal::distributed::MeshShape mesh_shape = control_plane.get_physical_mesh_shape(src_fabric_node_id.mesh_id);
    auto last_recv_phys_chip_id = physical_end_device_ids_by_dir[dir][range - 1];

    auto sender_device = fixture->get_device(src_phys_chip_id);
    auto last_recv_device = fixture->get_device(last_recv_phys_chip_id);

    auto first_recv_fabric_node_id = end_fabric_node_ids_by_dir[dir][0];
    auto last_recv_fabric_node_id = end_fabric_node_ids_by_dir[dir][range - 1];

    CoreCoord receiver_virtual_core =
        last_recv_device->worker_core_from_logical_core(receiver_logical_core);

    // test parameters
    auto worker_mem_map = generate_worker_mem_map(sender_device, topology);
    uint32_t num_packets = 100;
    uint32_t time_seed = std::chrono::system_clock::now().time_since_epoch().count();

    // common compile time args for sender and receiver
    // Note: Fabric Mcast with NOC writes to DRAM provides redudant coverage,
    // so use_dram_dst is set to 0; see run_unicast_dw_chips() for DRAM coverage
    std::vector<uint32_t> compile_time_args = {
        worker_mem_map.test_results_address,
        worker_mem_map.test_results_size_bytes,
        worker_mem_map.target_address,
        0 /* use_dram_dst */,
        0 /* is_2d_fabric */,
        1 /* is_chip_multicast */,
        0 /* additional_dir */};

    std::map<std::string, std::string> defines = {};

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

    log_info(tt::LogTest, "mesh dimensions: {:x}", mesh_shape.dims());
    log_info(tt::LogTest, "mesh dimension 0: {:x}", mesh_shape[0]);
    log_info(tt::LogTest, "mesh dimension 1: {:x}", mesh_shape[1]);
    log_info(
        tt::LogTest, "Mcast Src: MeshId {} ChipId {}", src_fabric_node_id.mesh_id.get(), src_fabric_node_id.chip_id);
    log_info(
        tt::LogTest,
        "Mcast Dst: from MeshId {} ChipId {} to MeshId {} ChipId {}",
        first_recv_fabric_node_id.mesh_id.get(),
        first_recv_fabric_node_id.chip_id,
        last_recv_fabric_node_id.mesh_id.get(),
        last_recv_fabric_node_id.chip_id);
    log_info(
        tt::LogTest,
        "Mcast Receiver Core (Logical): {},{}",
        receiver_logical_core.x,
        receiver_logical_core.y);

    std::vector<uint32_t> sender_runtime_args = {
        worker_mem_map.source_l1_buffer_address,
        worker_mem_map.packet_payload_size_bytes,
        num_packets,
        time_seed,
        receiver_virtual_core.x,
        receiver_virtual_core.y,
        mesh_shape[1],
        src_fabric_node_id.chip_id,
        start_distance,
        range,
        last_recv_fabric_node_id.chip_id,
        *last_recv_fabric_node_id.mesh_id};

    // append the EDM connection rt args for fwd connection
    ChipId dst_chip_id = first_hop_phys_chip_id;
    const auto dst_fabric_node_id = tt::tt_fabric::get_fabric_node_id_from_physical_chip_id(dst_chip_id);
    uint32_t link_idx = get_forwarding_link_indices(src_fabric_node_id, dst_fabric_node_id)[0];
    append_fabric_connection_rt_args(
        src_fabric_node_id, dst_fabric_node_id, link_idx, sender_program, {sender_logical_core}, sender_runtime_args);

    tt_metal::SetRuntimeArgs(sender_program, sender_kernel, sender_logical_core, sender_runtime_args);

    std::vector<uint32_t> receiver_runtime_args = {worker_mem_map.packet_payload_size_bytes, num_packets, time_seed};

    // Create and launch the receiver program for validation on all mcast receiver devices
    for (auto& [routing_direction, physical_end_device_ids] : physical_end_device_ids_by_dir) {
        for (auto physical_end_device_id : physical_end_device_ids) {
            auto receiver_device = fixture->get_device(physical_end_device_id);
            // Create the receiver program for validation
            auto receiver_program = tt_metal::CreateProgram();
            auto receiver_kernel = tt_metal::CreateKernel(
                receiver_program,
                "tests/tt_metal/tt_metal/perf_microbenchmark/routing/kernels/tt_fabric_1d_rx.cpp",
                receiver_logical_core,
                tt_metal::DataMovementConfig{
                    .processor = tt_metal::DataMovementProcessor::RISCV_0,
                    .noc = tt_metal::NOC::RISCV_0_default,
                    .compile_args = compile_time_args});

            tt_metal::SetRuntimeArgs(
                receiver_program, receiver_kernel, receiver_logical_core, receiver_runtime_args);
            fixture->RunProgramNonblocking(receiver_device, receiver_program);
            receiver_programs.push_back(std::move(receiver_program));
            log_info(tt::LogTest, "{} Rx Launched on physical device {}", routing_direction, physical_end_device_id);
        }
    }

    // Launch sender program and wait for sender to finish
    fixture->RunProgramNonblocking(sender_device, sender_program);
    log_info(tt::LogTest, "Sender Launched on physical device {}", src_phys_chip_id);

    fixture->WaitForSingleProgramDone(sender_device, sender_program);
    log_info(tt::LogTest, "Sender Finished");

    // Wait for receivers to finish
    for (const auto& [routing_direction, physical_end_device_ids] : physical_end_device_ids_by_dir) {
        for (uint32_t i = 0; i < physical_end_device_ids.size(); i++) {
            auto receiver_device = fixture->get_device(physical_end_device_ids[i]);
            fixture->WaitForSingleProgramDone(receiver_device, receiver_programs[i]);
        }
    }
    log_info(tt::LogTest, "All Receivers Finished");

    // Validate the status and packets processed by sender and receiver
    std::vector<uint32_t> sender_status;
    std::vector<uint32_t> left_recv_status;
    std::vector<uint32_t> right_recv_status;

    tt_metal::detail::ReadFromDeviceL1(
        sender_device->get_devices()[0],
        sender_logical_core,
        worker_mem_map.test_results_address,
        worker_mem_map.test_results_size_bytes,
        sender_status,
        CoreType::WORKER);

    EXPECT_EQ(sender_status[TT_FABRIC_STATUS_INDEX], TT_FABRIC_STATUS_PASS);

    uint64_t sender_bytes =
        ((uint64_t)sender_status[TT_FABRIC_WORD_CNT_INDEX + 1] << 32) | sender_status[TT_FABRIC_WORD_CNT_INDEX];

    for (const auto& [routing_direction, physical_end_device_ids] : physical_end_device_ids_by_dir) {
        for (int device_id : physical_end_device_ids) {
            const auto& receiver_device = fixture->get_device(device_id);

            log_info(
                tt::LogTest,
                "Checking Status of {} Rx on core ({}, {}) on physical device {}",
                routing_direction,
                receiver_logical_core.x,
                receiver_logical_core.y,
                device_id);

            std::vector<uint32_t> recv_status;

            tt_metal::detail::ReadFromDeviceL1(
                receiver_device->get_devices()[0],
                receiver_logical_core,
                worker_mem_map.test_results_address,
                worker_mem_map.test_results_size_bytes,
                recv_status,
                CoreType::WORKER);
            EXPECT_EQ(recv_status[TT_FABRIC_STATUS_INDEX], TT_FABRIC_STATUS_PASS);
            uint64_t recv_bytes =
                ((uint64_t)recv_status[TT_FABRIC_WORD_CNT_INDEX + 1] << 32) | recv_status[TT_FABRIC_WORD_CNT_INDEX];

            EXPECT_EQ(sender_bytes, recv_bytes);
        }
    }
}

TEST_F(Fabric1DFixture, TestUnicastRaw) { RunTestUnicastRaw(this, 1, RoutingDirection::E); }

TEST_F(Galaxy1x32Fabric1DFixture, TestUnicastRaw_AllHops) {
    const size_t num_devices = tt::tt_metal::GetNumAvailableDevices();
    ASSERT_EQ(num_devices, 32) << "1x32 mesh required for this test";
    for (uint32_t hops = 1; hops < num_devices; hops++) {
        RunTestUnicastRaw(this, hops, RoutingDirection::E);
    }

    for (uint32_t hops = 1; hops < num_devices; hops++) {
        RunTestUnicastRaw(this, hops, RoutingDirection::W);
    }
}

TEST_F(Fabric1DFixture, TestUnicastConnAPI) { RunTestUnicastConnAPI(this, 1); }
TEST_F(Fabric1DFixture, TestUnicastConnAPIDRAM) { RunTestUnicastConnAPI(this, 1, RoutingDirection::E, true); }
TEST_F(Fabric1DFixture, TestUnicastTGGateways) { RunTestUnicastTGGateways(this); }
TEST_F(Fabric1DFixture, TestMCastConnAPI) { RunTestMCastConnAPI(this); }

// only chip multicast (test start and range)
TEST_F(Fabric1DFixture, TestChipMCast1DWithTracing) { RunTestChipMCast1D(this, RoutingDirection::E, 1, 3); }
TEST_F(Fabric1DFixture, TestChipMCast1DWithTracing2) { RunTestChipMCast1D(this, RoutingDirection::E, 2, 2); }

void RunEDMConnectionStressTest(
    BaseFabricFixture* fixture,
    const std::vector<size_t>& stall_durations_cycles,
    const std::vector<size_t>& message_counts,
    const std::vector<size_t>& packet_sizes,
    size_t num_times_to_connect,
    size_t num_iterations,
    size_t num_workers,
    size_t test_rows) {
    log_info(tt::LogTest, "Starting EDM connection stress test");
    auto& control_plane= tt::tt_metal::MetalContext::instance().get_control_plane();
    log_debug(tt::LogTest, "Control plane found");

    // use control plane to find a mesh with 3 devices
    auto user_meshes = control_plane.get_user_physical_mesh_ids();
    std::optional<MeshId> mesh_id;
    for (const auto& mesh : user_meshes) {
        auto mesh_shape = control_plane.get_physical_mesh_shape(mesh);
        if (mesh_shape.mesh_size() > 1) {
            mesh_id = mesh;
            break;
        }
    }
    if (!mesh_id.has_value()) {
        GTEST_SKIP() << "No mesh found for 2 chip connection stress test";
    }

    log_debug(tt::LogTest, "Mesh ID: {}", mesh_id.value());
    auto src_physical_device_id =
        control_plane.get_physical_chip_id_from_fabric_node_id(FabricNodeId(mesh_id.value(), 0));
    auto dst_physical_device_id =
        control_plane.get_physical_chip_id_from_fabric_node_id(FabricNodeId(mesh_id.value(), 1));

    auto sender_device = fixture->get_device(src_physical_device_id);
    auto receiver_device = fixture->get_device(dst_physical_device_id);

    // Set the destination address for fabric writes (constant for all workers)
    uint32_t fabric_write_dest_bank_addr = 0x50000;

    // For each epoch, run with increasing number of workers
    log_debug(tt::LogTest, "Starting EDM connection stress test");
    auto compute_with_storage_grid_size = sender_device->compute_with_storage_grid_size();
    size_t num_cols = compute_with_storage_grid_size.x;
    for (size_t iter = 0; iter < num_iterations; iter++) {
        log_info(tt::LogTest, "iter {}", iter);
        log_debug(tt::LogTest, "num_workers {}", num_workers);
        for (size_t c = 0; c < num_cols - (num_workers - 1); c++) {
            log_debug(tt::LogTest, "r={}, c={}", test_rows, c);

            // Set up worker cores for token ring
            auto worker_logical_cores = CoreRangeSet(CoreRange({{c, test_rows}, {c + num_workers - 1, test_rows}}));
            auto worker_logical_cores_vec = corerange_to_cores(worker_logical_cores, std::nullopt, false);

            // Map logical to virtual cores
            std::vector<CoreCoord> worker_virtual_cores;
            worker_virtual_cores.reserve(worker_logical_cores_vec.size());
            for (const auto& logical_core : worker_logical_cores_vec) {
                worker_virtual_cores.push_back(sender_device->worker_core_from_logical_core(logical_core));
            }

            // Create program
            auto program = tt_metal::CreateProgram();

            // Create semaphores for token passing (one per worker)
            auto connection_token_semaphore_id =
                tt_metal::CreateSemaphore(program, CoreRangeSet(worker_logical_cores), 0);

            // Create source packet buffer (one per worker)
            static constexpr uint32_t source_l1_cb_index = tt::CB::c_in0;
            static constexpr tt::DataFormat cb_df = tt::DataFormat::Bfp8;
            auto max_payload_size = *std::max_element(packet_sizes.begin(), packet_sizes.end());
            auto source_l1_cb_config =
                tt_metal::CircularBufferConfig(max_payload_size * 2, {{source_l1_cb_index, cb_df}})
                    .set_page_size(source_l1_cb_index, max_payload_size);
            CreateCircularBuffer(program, worker_logical_cores, source_l1_cb_config);

            // Configure common compile time args for all workers
            std::vector<uint32_t> compile_time_args = {
                static_cast<uint32_t>(stall_durations_cycles.size()),
                static_cast<uint32_t>(packet_sizes.size()),
                static_cast<uint32_t>(message_counts.size()),
            };

            // Create a kernel for each worker
            std::vector<std::vector<uint32_t>> runtime_args_per_worker(num_workers);

            for (size_t i = 0; i < num_workers; i++) {
                // Compute destination NOC coordinates for this worker
                auto dest_virtual_core = worker_virtual_cores[i];

                // Compute next worker index in the token ring
                size_t next_worker_idx = (i + 1) % num_workers;
                auto next_worker_virtual_core = worker_virtual_cores[next_worker_idx];

                // Prepare runtime args for this worker
                std::vector<uint32_t>& worker_args = runtime_args_per_worker[i];

                // Basic configuration
                worker_args.push_back(fabric_write_dest_bank_addr);  // Fabric write destination bank address
                worker_args.push_back(dest_virtual_core.x);          // Fabric write destination NOC X
                worker_args.push_back(dest_virtual_core.y);          // Fabric write destination NOC Y

                // Token ring configuration
                worker_args.push_back(i == 0 ? 1 : 0);                 // Is starting worker (first worker starts)
                worker_args.push_back(num_times_to_connect);           // How many times to connect during turn
                worker_args.push_back(next_worker_virtual_core.x);     // Next worker NOC X
                worker_args.push_back(next_worker_virtual_core.y);     // Next worker NOC Y
                worker_args.push_back(connection_token_semaphore_id);  // Address of next worker's token

                // Traffic pattern arrays (rotate starting index by worker ID for variation)
                worker_args.push_back(stall_durations_cycles.size());  // Number of stall durations

                // Rotate starting point for each worker to prevent lock-step behavior
                size_t stall_offset = i % stall_durations_cycles.size();
                for (size_t j = 0; j < stall_durations_cycles.size(); j++) {
                    size_t idx = (stall_offset + j) % stall_durations_cycles.size();
                    worker_args.push_back(stall_durations_cycles[idx]);
                }

                worker_args.push_back(packet_sizes.size());  // Number of packet sizes
                size_t packet_size_offset = i % packet_sizes.size();
                for (size_t j = 0; j < packet_sizes.size(); j++) {
                    size_t idx = (packet_size_offset + j) % packet_sizes.size();
                    worker_args.push_back(packet_sizes[idx]);
                }

                worker_args.push_back(message_counts.size());  // Number of message counts
                size_t message_count_offset = i % message_counts.size();
                for (size_t j = 0; j < message_counts.size(); j++) {
                    size_t idx = (message_count_offset + j) % message_counts.size();
                    worker_args.push_back(message_counts[idx]);
                }

                // Circular buffer indices for source data and packet headers
                worker_args.push_back(source_l1_cb_index);  // Source L1 circular buffer index

                worker_args.push_back(i % stall_durations_cycles.size());
                worker_args.push_back(i % packet_sizes.size());
                worker_args.push_back(i % message_counts.size());

                const auto sender_fabric_node_id =
                    tt::tt_fabric::get_fabric_node_id_from_physical_chip_id(sender_device->get_devices()[0]->id());
                const auto receiver_fabric_node_id =
                    tt::tt_fabric::get_fabric_node_id_from_physical_chip_id(receiver_device->get_devices()[0]->id());
                append_fabric_connection_rt_args(
                    sender_fabric_node_id,
                    receiver_fabric_node_id,
                    0,
                    program,
                    {worker_logical_cores_vec[i]},
                    worker_args);

                auto kernel = tt_metal::CreateKernel(
                    program,
                    "tests/tt_metal/tt_metal/perf_microbenchmark/routing/kernels/"
                    "edm_fabric_connection_test_kernel.cpp",
                    worker_logical_cores_vec[i],
                    tt_metal::DataMovementConfig{
                        .processor = tt_metal::DataMovementProcessor::RISCV_0,
                        .noc = tt_metal::NOC::RISCV_0_default,
                        .compile_args = compile_time_args});
                tt_metal::SetRuntimeArgs(program, kernel, worker_logical_cores_vec[i], runtime_args_per_worker[i]);
            }

            // Launch program and wait for completion
            auto start_time = std::chrono::high_resolution_clock::now();
            log_debug(tt::LogTest, "Launching program");
            fixture->RunProgramNonblocking(sender_device, program);
            fixture->WaitForSingleProgramDone(sender_device, program);
            auto end_time = std::chrono::high_resolution_clock::now();
            auto duration_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();

            log_info(tt::LogTest, "Iter {} with {} workers completed in {} ms", iter, num_workers, duration_ms);
        }
    }
}

TEST_F(NightlyFabric1DFixture, TestEDMConnectionStressTestQuick) {
    std::vector<size_t> stall_durations_cycles = {0,    100,  200,  300,   400,   700,   1000,  2000,  3000,  4000,
                                                  5000, 7000, 8000, 10000, 20000, 30000, 40000, 50000, 60000, 100000};
    std::vector<size_t> message_counts = {8, 100};
    std::vector<size_t> packet_sizes = {16, 4 * 1088};
    size_t num_times_to_connect = 20000;
    size_t num_iterations = 10;
    std::vector<size_t> worker_counts = {1, 3};
    std::vector<size_t> test_rows = {0, 4, 5, 6};

    for (auto num_workers : worker_counts) {
        for (auto r : test_rows) {
            RunEDMConnectionStressTest(
                this,
                stall_durations_cycles,
                message_counts,
                packet_sizes,
                num_times_to_connect,
                num_iterations,
                num_workers,
                r);
        }
    }
}

void FabricUnicastCommon(
    BaseFabricFixture* fixture,
    NocSendType noc_send_type,
    const std::vector<std::tuple<RoutingDirection, uint32_t>>& pair_ordered_dirs,
    FabricApiType api_type,
    bool with_state) {
    CoreCoord sender_logical_core = {0, 0};
    CoreCoord receiver_logical_core = {1, 0};
    uint32_t num_packets = 10;
    uint32_t time_seed = std::chrono::system_clock::now().time_since_epoch().count();

    auto& control_plane = tt::tt_metal::MetalContext::instance().get_control_plane();
    const auto topology = control_plane.get_fabric_context().get_fabric_topology();

    std::vector<std::tuple<RoutingDirection, uint32_t>> dir_configs = pair_ordered_dirs;
    if (topology == Topology::Mesh) {
        const auto cluster_type = tt::tt_metal::MetalContext::instance().get_cluster().get_cluster_type();
        size_t max_dirs = (cluster_type == tt::tt_metal::ClusterType::T3K) ? 3 : 4;
        if (dir_configs.size() > max_dirs) {
            dir_configs.resize(max_dirs);
        }
        if (dir_configs.empty()) {
            dir_configs = {std::make_tuple(RoutingDirection::E, 1)};
        }
    } else {
        if (dir_configs.empty()) {
            dir_configs = {std::make_tuple(RoutingDirection::E, 1)};
        }
        if (dir_configs.size() > 2) {
            dir_configs.resize(2);
        }
    }

    // Find a source device and first-hop receivers for the selected directions
    FabricNodeId src_fabric_node_id(MeshId{0}, 0);
    std::unordered_map<RoutingDirection, uint32_t> fabric_hops;
    for (auto [dir, num_hops] : dir_configs) {
        fabric_hops[dir] = num_hops;
    }

    std::unordered_map<RoutingDirection, std::vector<FabricNodeId>> end_fabric_node_ids_by_dir;
    ChipId src_physical_device_id;
    std::unordered_map<RoutingDirection, std::vector<ChipId>> physical_end_device_ids_by_dir;
    if (!find_device_with_neighbor_in_multi_direction(
            fixture,
            src_fabric_node_id,
            end_fabric_node_ids_by_dir,
            src_physical_device_id,
            physical_end_device_ids_by_dir,
            fabric_hops)) {
        GTEST_SKIP() << "No path found for requested directions";
    }

    std::vector<std::shared_ptr<tt_metal::distributed::MeshDevice>> receiver_devices;
    std::vector<FabricNodeId> dest_fabric_node_ids;
    receiver_devices.reserve(dir_configs.size());
    for (auto [dir, num_hops] : dir_configs) {
        // pick destination device at the num_hops-th neighbor
        uint32_t dst_index = num_hops - 1;
        auto dst_physical_device_id = physical_end_device_ids_by_dir[dir][dst_index];
        receiver_devices.push_back(fixture->get_device(dst_physical_device_id));
        // connection is to first hop for each direction
        dest_fabric_node_ids.push_back(tt::tt_fabric::get_fabric_node_id_from_physical_chip_id(dst_physical_device_id));
    }
    auto sender_device = fixture->get_device(src_physical_device_id);
    CoreCoord receiver_virtual_core = receiver_devices.back()->worker_core_from_logical_core(receiver_logical_core);

    tt_metal::Program sender_program = tt_metal::CreateProgram();

    auto worker_mem_map = generate_worker_mem_map(sender_device, topology);

    std::vector<uint32_t> compile_time_args = {
        worker_mem_map.test_results_address,
        worker_mem_map.test_results_size_bytes,
        worker_mem_map.notification_mailbox_address,
        worker_mem_map.target_address,
        noc_send_type,
        static_cast<uint32_t>(dir_configs.size()),
        with_state,
        0  // is_chip_multicast = 0
    };

    if (noc_send_type == NOC_UNICAST_INLINE_WRITE) {
        worker_mem_map.packet_payload_size_bytes = 4;
    }

    auto sender_kernel = tt_metal::CreateKernel(
        sender_program,
        (noc_send_type == NOC_FUSED_UNICAST_ATOMIC_INC || noc_send_type == NOC_UNICAST_ATOMIC_INC)
            ? "tests/tt_metal/tt_fabric/fabric_data_movement/kernels/test_linear_api_atomic_inc_sender.cpp"
            : "tests/tt_metal/tt_fabric/fabric_data_movement/kernels/test_linear_api_unicast_write_sender.cpp",
        {sender_logical_core},
        tt_metal::DataMovementConfig{
            .processor = tt_metal::DataMovementProcessor::RISCV_0,
            .noc = tt_metal::NOC::RISCV_0_default,
            .compile_args = compile_time_args});

    std::vector<uint32_t> sender_runtime_args = {
        worker_mem_map.source_l1_buffer_address,
        worker_mem_map.packet_payload_size_bytes,
        num_packets,
        time_seed,
        receiver_virtual_core.x,
        receiver_virtual_core.y,
    };
    for (auto [dir, num_hops] : dir_configs) {
        sender_runtime_args.push_back(num_hops);
    }

    append_routing_plane_connection_manager_rt_args(
        src_fabric_node_id,
        dest_fabric_node_ids,
        {},
        sender_program,
        sender_kernel,
        {sender_logical_core},
        sender_runtime_args,
        api_type);

    tt_metal::SetRuntimeArgs(sender_program, sender_kernel, sender_logical_core, sender_runtime_args);
    std::unordered_map<std::shared_ptr<tt_metal::distributed::MeshDevice>, tt_metal::Program> receiver_programs;
    for (const auto& recv_dev : receiver_devices) {
        receiver_programs[recv_dev] = tt_metal::CreateProgram();
        auto receiver_kernel = tt_metal::CreateKernel(
            receiver_programs[recv_dev],
            (noc_send_type == NOC_FUSED_UNICAST_ATOMIC_INC || noc_send_type == NOC_UNICAST_ATOMIC_INC)
                ? "tests/tt_metal/tt_fabric/fabric_data_movement/kernels/test_linear_api_atomic_inc_receiver.cpp"
                : "tests/tt_metal/tt_fabric/fabric_data_movement/kernels/test_linear_api_receiver.cpp",
            {receiver_logical_core},
            tt_metal::DataMovementConfig{
                .processor = tt_metal::DataMovementProcessor::RISCV_0,
                .noc = tt_metal::NOC::RISCV_0_default,
                .compile_args = compile_time_args});

        std::vector<uint32_t> receiver_runtime_args = {
            worker_mem_map.packet_payload_size_bytes,
            num_packets,
            time_seed,
        };
        tt_metal::SetRuntimeArgs(
            receiver_programs[recv_dev], receiver_kernel, receiver_logical_core, receiver_runtime_args);
    }
    for (auto& [recv_dev, receiver_program] : receiver_programs) {
        fixture->RunProgramNonblocking(recv_dev, receiver_program);
    }
    fixture->RunProgramNonblocking(sender_device, sender_program);
    fixture->WaitForSingleProgramDone(sender_device, sender_program);
    for (auto& [recv_dev, receiver_program] : receiver_programs) {
        fixture->WaitForSingleProgramDone(recv_dev, receiver_program);
    }

    std::vector<uint32_t> sender_status;
    tt_metal::detail::ReadFromDeviceL1(
        sender_device->get_devices()[0],
        sender_logical_core,
        worker_mem_map.test_results_address,
        worker_mem_map.test_results_size_bytes,
        sender_status,
        CoreType::WORKER);
    EXPECT_EQ(sender_status[TT_FABRIC_STATUS_INDEX], TT_FABRIC_STATUS_PASS);

    std::vector<uint32_t> receiver_status;
    for (const auto& recv_dev : receiver_devices) {
        tt_metal::detail::ReadFromDeviceL1(
            recv_dev->get_devices()[0],
            receiver_logical_core,
            worker_mem_map.test_results_address,
            worker_mem_map.test_results_size_bytes,
            receiver_status,
            CoreType::WORKER);
        EXPECT_EQ(receiver_status[TT_FABRIC_STATUS_INDEX], TT_FABRIC_STATUS_PASS);
        uint64_t sender_words =
            ((uint64_t)sender_status[TT_FABRIC_WORD_CNT_INDEX + 1] << 32) | sender_status[TT_FABRIC_WORD_CNT_INDEX];
        uint64_t receiver_words =
            ((uint64_t)receiver_status[TT_FABRIC_WORD_CNT_INDEX + 1] << 32) | receiver_status[TT_FABRIC_WORD_CNT_INDEX];
        EXPECT_EQ(sender_words, receiver_words);
    }
}

void UDMFabricUnicastCommon(
    BaseFabricFixture* fixture,
    NocSendType noc_send_type,
    const std::variant<
        std::tuple<RoutingDirection, uint32_t /*num_hops*/>,
        std::tuple<uint32_t /*src_node*/, uint32_t /*dest_node*/>>& routing_info,
    std::optional<RoutingDirection> override_initial_direction,
    std::optional<std::vector<std::pair<CoreCoord, CoreCoord>>> worker_coords_list,
    bool dual_risc) {
    // Build list of worker coordinate pairs - default to single pair (0,0) -> (1,0)
    std::vector<std::pair<CoreCoord, CoreCoord>> worker_pairs;
    if (worker_coords_list.has_value()) {
        worker_pairs = worker_coords_list.value();
    } else {
        worker_pairs.push_back({CoreCoord{0, 0}, CoreCoord{1, 0}});
    }
    uint32_t num_packets = 10;
    uint32_t time_seed = std::chrono::system_clock::now().time_since_epoch().count();
    // When two data movement kernels both write to fabric, we need to ensure packet ordering is preserved.
    // Dynamic NOC mode forces both kernels to use the same NOC (NOC0) to maintain that ordering.
    auto noc_mode = dual_risc ? tt_metal::NOC_MODE::DM_DYNAMIC_NOC : tt_metal::NOC_MODE::DM_DEDICATED_NOC;

    auto& control_plane = tt::tt_metal::MetalContext::instance().get_control_plane();

    // Determine source and destination based on routing_info variant
    FabricNodeId src_fabric_node_id(MeshId{0}, 0);
    FabricNodeId dest_fabric_node_id(MeshId{0}, 0);
    ChipId src_physical_device_id;
    ChipId dst_physical_device_id;

    if (std::holds_alternative<std::tuple<RoutingDirection, uint32_t>>(routing_info)) {
        // Original behavior: use direction and hops
        auto [dir, num_hops] = std::get<std::tuple<RoutingDirection, uint32_t>>(routing_info);

        std::unordered_map<RoutingDirection, uint32_t> fabric_hops;
        fabric_hops[dir] = num_hops;

        std::unordered_map<RoutingDirection, std::vector<FabricNodeId>> end_fabric_node_ids_by_dir;
        std::unordered_map<RoutingDirection, std::vector<ChipId>> physical_end_device_ids_by_dir;
        if (!find_device_with_neighbor_in_multi_direction(
                fixture,
                src_fabric_node_id,
                end_fabric_node_ids_by_dir,
                src_physical_device_id,
                physical_end_device_ids_by_dir,
                fabric_hops)) {
            GTEST_SKIP() << "No path found for requested direction";
        }

        // Get destination device at the num_hops-th neighbor
        uint32_t dst_index = num_hops - 1;
        dst_physical_device_id = physical_end_device_ids_by_dir[dir][dst_index];
        dest_fabric_node_id = tt::tt_fabric::get_fabric_node_id_from_physical_chip_id(dst_physical_device_id);
    } else {
        // New behavior: use explicit src and dest node IDs
        auto [src_node, dest_node] = std::get<std::tuple<uint32_t, uint32_t>>(routing_info);

        src_fabric_node_id = FabricNodeId(MeshId{0}, src_node);
        dest_fabric_node_id = FabricNodeId(MeshId{0}, dest_node);

        src_physical_device_id = control_plane.get_physical_chip_id_from_fabric_node_id(src_fabric_node_id);
        dst_physical_device_id = control_plane.get_physical_chip_id_from_fabric_node_id(dest_fabric_node_id);

        // Verify devices exist in fixture
        if (!fixture->get_device(src_physical_device_id) || !fixture->get_device(dst_physical_device_id)) {
            GTEST_SKIP() << "Source or destination device not available in fixture";
        }
    }

    auto receiver_device = fixture->get_device(dst_physical_device_id);
    auto sender_device = fixture->get_device(src_physical_device_id);

    tt_metal::Program sender_program = tt_metal::CreateProgram();
    tt_metal::Program receiver_program = tt_metal::CreateProgram();

    DualRiscMemMapHelper mem_helper(dual_risc, num_packets);
    auto worker_mem_map = mem_helper.gen_mem_map(sender_device, 0);
    auto worker_mem_map_risc1 = mem_helper.gen_mem_map(sender_device, mem_helper.region_size);

    if (noc_send_type == NOC_UNICAST_INLINE_WRITE or noc_send_type == NOC_UNICAST_ATOMIC_INC) {
        worker_mem_map.packet_payload_size_bytes = 16;  // l1 aligned
        worker_mem_map_risc1.packet_payload_size_bytes = 16;
    } else {
        auto single_payload_size_bytes = worker_mem_map.packet_payload_size_bytes;
        worker_mem_map.packet_payload_size_bytes = single_payload_size_bytes * 2 - 16;
        worker_mem_map_risc1.packet_payload_size_bytes = single_payload_size_bytes * 2 - 16;
    }

    log_debug(
        tt::LogTest,
        "RISC0 mem_map: source_l1_buffer=0x{:x}, target=0x{:x}, test_results=0x{:x}, notification=0x{:x}, "
        "payload_size={}, test_results_size={}",
        worker_mem_map.source_l1_buffer_address,
        worker_mem_map.target_address,
        worker_mem_map.test_results_address,
        worker_mem_map.notification_mailbox_address,
        worker_mem_map.packet_payload_size_bytes,
        worker_mem_map.test_results_size_bytes);

    if (dual_risc) {
        log_debug(
            tt::LogTest,
            "RISC1 mem_map: source_l1_buffer=0x{:x}, target=0x{:x}, test_results=0x{:x}, notification=0x{:x}, "
            "payload_size={}, test_results_size={}",
            worker_mem_map_risc1.source_l1_buffer_address,
            worker_mem_map_risc1.target_address,
            worker_mem_map_risc1.test_results_address,
            worker_mem_map_risc1.notification_mailbox_address,
            worker_mem_map_risc1.packet_payload_size_bytes,
            worker_mem_map_risc1.test_results_size_bytes);
        log_debug(
            tt::LogTest,
            "DualRisc: data_space_size={}, region_size={}, num_packets={}",
            mem_helper.data_space_size,
            mem_helper.region_size,
            num_packets);
    }

    // Define req_notification_size_bytes for read operations
    constexpr uint32_t req_notification_size_bytes = 128;

    // Set up fabric connection destination for override_initial_direction
    FabricNodeId fabric_connection_dest_node_id = dest_fabric_node_id;
    if (override_initial_direction.has_value()) {
        auto neighbors = control_plane.get_intra_chip_neighbors(src_fabric_node_id, override_initial_direction.value());
        if (neighbors.empty()) {
            GTEST_SKIP() << "No neighbor found in the specified initial direction "
                         << static_cast<int>(override_initial_direction.value()) << " from node "
                         << src_fabric_node_id.chip_id;
        }
        fabric_connection_dest_node_id = FabricNodeId(src_fabric_node_id.mesh_id, neighbors[0]);
    }

    // Select kernel paths based on operation type
    const char* sender_kernel_path =
        (noc_send_type == NOC_UNICAST_READ)
            ? "tests/tt_metal/tt_fabric/fabric_data_movement/kernels/test_udm_read_sender.cpp"
            : "tests/tt_metal/tt_fabric/fabric_data_movement/kernels/test_udm_sender.cpp";
    const char* receiver_kernel_path =
        (noc_send_type == NOC_UNICAST_READ)
            ? "tests/tt_metal/tt_fabric/fabric_data_movement/kernels/test_udm_read_receiver.cpp"
            : "tests/tt_metal/tt_fabric/fabric_data_movement/kernels/test_udm_receiver.cpp";

    // Build CoreRangeSets for all sender and receiver cores
    std::vector<CoreCoord> sender_cores, receiver_cores;
    for (const auto& [sender_logical_core, receiver_logical_core] : worker_pairs) {
        sender_cores.push_back(sender_logical_core);
        receiver_cores.push_back(receiver_logical_core);
    }
    CoreRangeSet sender_core_range(sender_cores);
    CoreRangeSet receiver_core_range(receiver_cores);

    // Sender compile time args (per-core receiver coords moved to runtime args)
    std::vector<uint32_t> sender_compile_time_args = {
        worker_mem_map.test_results_address,
        worker_mem_map.test_results_size_bytes,
        worker_mem_map.notification_mailbox_address,
        worker_mem_map.target_address,
        noc_send_type,
        worker_mem_map.source_l1_buffer_address,
        worker_mem_map.packet_payload_size_bytes,
        num_packets,
        time_seed,
        dest_fabric_node_id.chip_id,
        dest_fabric_node_id.mesh_id.get(),
        req_notification_size_bytes};

    auto sender_kernel_risc0 = tt_metal::CreateKernel(
        sender_program,
        sender_kernel_path,
        sender_core_range,
        tt_metal::DataMovementConfig{
            .processor = tt_metal::DataMovementProcessor::RISCV_0,
            .noc = tt_metal::NOC::RISCV_0_default,
            .noc_mode = noc_mode,
            .compile_args = sender_compile_time_args});

    // Create RISCV_1 sender kernel if dual_risc mode is enabled
    tt_metal::KernelHandle sender_kernel_risc1 = 0;
    if (dual_risc) {
        // Use separate memory map for RISC1
        std::vector<uint32_t> sender_compile_time_args_risc1 = {
            worker_mem_map_risc1.test_results_address,
            worker_mem_map_risc1.test_results_size_bytes,
            worker_mem_map_risc1.notification_mailbox_address,
            worker_mem_map_risc1.target_address,
            noc_send_type,
            worker_mem_map_risc1.source_l1_buffer_address,
            worker_mem_map_risc1.packet_payload_size_bytes,
            num_packets,
            time_seed,
            dest_fabric_node_id.chip_id,
            dest_fabric_node_id.mesh_id.get(),
            req_notification_size_bytes};

        sender_kernel_risc1 = tt_metal::CreateKernel(
            sender_program,
            sender_kernel_path,
            sender_core_range,
            tt_metal::DataMovementConfig{
                .processor = tt_metal::DataMovementProcessor::RISCV_1,
                .noc = tt_metal::NOC::RISCV_1_default,
                .noc_mode = noc_mode,
                .compile_args = sender_compile_time_args_risc1});
    }

    // Receiver compile time args (per-core sender coords moved to runtime args)
    std::vector<uint32_t> receiver_compile_time_args = {
        worker_mem_map.test_results_address,
        worker_mem_map.test_results_size_bytes,
        worker_mem_map.notification_mailbox_address,
        worker_mem_map.target_address,
        noc_send_type,
        worker_mem_map.packet_payload_size_bytes,
        num_packets,
        time_seed,
        req_notification_size_bytes,
        src_fabric_node_id.chip_id,
        src_fabric_node_id.mesh_id.get()};

    auto receiver_kernel_risc0 = tt_metal::CreateKernel(
        receiver_program,
        receiver_kernel_path,
        receiver_core_range,
        tt_metal::DataMovementConfig{
            .processor = tt_metal::DataMovementProcessor::RISCV_0,
            .noc = tt_metal::NOC::RISCV_0_default,
            .noc_mode = noc_mode,
            .compile_args = receiver_compile_time_args});

    // Create RISCV_1 receiver kernel if dual_risc mode is enabled
    tt_metal::KernelHandle receiver_kernel_risc1 = 0;
    if (dual_risc) {
        // Use separate memory map for RISC1
        std::vector<uint32_t> receiver_compile_time_args_risc1 = {
            worker_mem_map_risc1.test_results_address,
            worker_mem_map_risc1.test_results_size_bytes,
            worker_mem_map_risc1.notification_mailbox_address,
            worker_mem_map_risc1.target_address,
            noc_send_type,
            worker_mem_map_risc1.packet_payload_size_bytes,
            num_packets,
            time_seed,
            req_notification_size_bytes,
            src_fabric_node_id.chip_id,
            src_fabric_node_id.mesh_id.get()};

        receiver_kernel_risc1 = tt_metal::CreateKernel(
            receiver_program,
            receiver_kernel_path,
            receiver_core_range,
            tt_metal::DataMovementConfig{
                .processor = tt_metal::DataMovementProcessor::RISCV_1,
                .noc = tt_metal::NOC::RISCV_1_default,
                .noc_mode = noc_mode,
                .compile_args = receiver_compile_time_args_risc1});
    }

    // Set per-core runtime args (receiver/sender coords + fabric connection)
    for (const auto& [sender_logical_core, receiver_logical_core] : worker_pairs) {
        CoreCoord sender_virtual_core = sender_device->worker_core_from_logical_core(sender_logical_core);
        CoreCoord receiver_virtual_core = receiver_device->worker_core_from_logical_core(receiver_logical_core);

        // Sender runtime args: receiver coords first, then fabric connection
        std::vector<uint32_t> sender_runtime_args = {receiver_virtual_core.x, receiver_virtual_core.y};
        tt_metal::SetRuntimeArgs(sender_program, sender_kernel_risc0, sender_logical_core, sender_runtime_args);
        if (dual_risc) {
            tt_metal::SetRuntimeArgs(sender_program, sender_kernel_risc1, sender_logical_core, sender_runtime_args);
        }

        // Receiver runtime args: sender coords first, then fabric connection
        std::vector<uint32_t> receiver_runtime_args = {sender_virtual_core.x, sender_virtual_core.y};
        tt_metal::SetRuntimeArgs(receiver_program, receiver_kernel_risc0, receiver_logical_core, receiver_runtime_args);
        if (dual_risc) {
            tt_metal::SetRuntimeArgs(
                receiver_program, receiver_kernel_risc1, receiver_logical_core, receiver_runtime_args);
        }

        // Clear target L1 memory for atomic increments
        if (noc_send_type == NOC_UNICAST_ATOMIC_INC) {
            uint32_t total_size_to_clear = num_packets * worker_mem_map.packet_payload_size_bytes;
            std::vector<uint32_t> zeros(total_size_to_clear / sizeof(uint32_t), 0);
            tt_metal::detail::WriteToDeviceL1(
                receiver_device->get_devices()[0],
                receiver_logical_core,
                worker_mem_map.target_address,
                zeros,
                CoreType::WORKER);
            // Clear RISC1 target memory as well
            if (dual_risc) {
                tt_metal::detail::WriteToDeviceL1(
                    receiver_device->get_devices()[0],
                    receiver_logical_core,
                    worker_mem_map_risc1.target_address,
                    zeros,
                    CoreType::WORKER);
            }
        }
    }

    // Run programs
    fixture->RunProgramNonblocking(receiver_device, receiver_program);
    fixture->RunProgramNonblocking(sender_device, sender_program);
    fixture->WaitForSingleProgramDone(sender_device, sender_program);
    fixture->WaitForSingleProgramDone(receiver_device, receiver_program);

    // Helper lambda to check test results for a given RISC
    auto check_risc_results = [&](const CoreCoord& sender_core,
                                  const CoreCoord& receiver_core,
                                  const WorkerMemMap& mem_map,
                                  const std::string& risc_name) {
        std::vector<uint32_t> sender_status;
        tt_metal::detail::ReadFromDeviceL1(
            sender_device->get_devices()[0],
            sender_core,
            mem_map.test_results_address,
            mem_map.test_results_size_bytes,
            sender_status,
            CoreType::WORKER);
        EXPECT_EQ(sender_status[TT_FABRIC_STATUS_INDEX], TT_FABRIC_STATUS_PASS)
            << "Sender " << risc_name << " failed at core (" << sender_core.x << ", " << sender_core.y << ")";

        std::vector<uint32_t> receiver_status;
        tt_metal::detail::ReadFromDeviceL1(
            receiver_device->get_devices()[0],
            receiver_core,
            mem_map.test_results_address,
            mem_map.test_results_size_bytes,
            receiver_status,
            CoreType::WORKER);
        EXPECT_EQ(receiver_status[TT_FABRIC_STATUS_INDEX], TT_FABRIC_STATUS_PASS)
            << "Receiver " << risc_name << " failed at core (" << receiver_core.x << ", " << receiver_core.y << ")";

        uint64_t sender_words =
            ((uint64_t)sender_status[TT_FABRIC_WORD_CNT_INDEX + 1] << 32) | sender_status[TT_FABRIC_WORD_CNT_INDEX];
        uint64_t receiver_words =
            ((uint64_t)receiver_status[TT_FABRIC_WORD_CNT_INDEX + 1] << 32) | receiver_status[TT_FABRIC_WORD_CNT_INDEX];
        EXPECT_EQ(sender_words, receiver_words)
            << risc_name << " word count mismatch at sender (" << sender_core.x << ", " << sender_core.y
            << ") -> receiver (" << receiver_core.x << ", " << receiver_core.y << ")";
    };

    // Check results for all worker pairs
    for (const auto& [sender_logical_core, receiver_logical_core] : worker_pairs) {
        check_risc_results(sender_logical_core, receiver_logical_core, worker_mem_map, "RISC0");
        if (dual_risc) {
            check_risc_results(sender_logical_core, receiver_logical_core, worker_mem_map_risc1, "RISC1");
        }
    }
}

void UDMFabricUnicastAllToAllCommon(BaseFabricFixture* fixture, NocSendType noc_send_type, bool dual_risc) {
    // All-to-all test: all devices send to all other devices simultaneously
    // Sender cores are in the top half of compute grid, receiver cores are in the bottom half
    // Each receiver core receives from N-1 senders (one from each other device)
    // Each sender writes to a different L1 address offset on the receiver based on sender_device_idx

    auto& control_plane = tt::tt_metal::MetalContext::instance().get_control_plane();
    const auto& devices = fixture->get_devices();
    const size_t NUM_DEVICES = devices.size();

    if (NUM_DEVICES < 2) {
        GTEST_SKIP() << "Test requires at least 2 devices";
    }

    uint32_t num_packets = 10;
    uint32_t time_seed = std::chrono::system_clock::now().time_since_epoch().count();
    // When two data movement kernels both write to fabric, we need to ensure packet ordering is preserved.
    // Dynamic NOC mode forces both kernels to use the same NOC (NOC0) to maintain that ordering.
    auto noc_mode = dual_risc ? tt_metal::NOC_MODE::DM_DYNAMIC_NOC : tt_metal::NOC_MODE::DM_DEDICATED_NOC;

    // Get device info and create programs
    std::vector<FabricNodeId> fabric_node_ids;
    std::vector<ChipId> physical_device_ids;
    std::vector<std::shared_ptr<tt::tt_metal::distributed::MeshDevice>> device_ptrs;

    for (size_t i = 0; i < NUM_DEVICES; i++) {
        FabricNodeId fabric_node_id(MeshId{0}, static_cast<uint32_t>(i));
        ChipId physical_device_id = control_plane.get_physical_chip_id_from_fabric_node_id(fabric_node_id);
        const auto& device_ptr = fixture->get_device(physical_device_id);
        if (!device_ptr) {
            continue;
        }
        fabric_node_ids.push_back(fabric_node_id);
        physical_device_ids.push_back(physical_device_id);
        device_ptrs.push_back(device_ptr);
    }

    const size_t num_active_devices = device_ptrs.size();
    if (num_active_devices < 2) {
        GTEST_SKIP() << "Not enough active devices for all-to-all test";
    }

    const uint32_t num_other_devices = static_cast<uint32_t>(num_active_devices - 1);

    // Calculate number of sender/receiver cores per device (all cores in top/bottom half)
    // Split grid into top half (senders) and bottom half (receivers)
    auto grid_size = devices[0]->get_devices()[0]->compute_with_storage_grid_size();
    uint32_t receiver_y_start = grid_size.y / 2;
    uint32_t receiver_y_end = grid_size.y;
    uint32_t sender_rows = receiver_y_start;                  // Number of rows in top half
    uint32_t receiver_rows = receiver_y_end - receiver_y_start;  // Number of rows in bottom half
    uint32_t num_sender_cores = sender_rows * grid_size.x;
    uint32_t num_receiver_cores = receiver_rows * grid_size.x;
    // Use the minimum as the number of sender-receiver pairs
    uint32_t num_core_pairs = std::min(num_sender_cores, num_receiver_cores);

    log_info(
        tt::LogTest,
        "All-to-all test: {} devices, {} sender-receiver pairs per device, sending to {} other devices, dual_risc={}",
        num_active_devices,
        num_core_pairs,
        num_other_devices,
        dual_risc);

    DualRiscMemMapHelper mem_helper(dual_risc, num_packets);
    auto worker_mem_map = mem_helper.gen_mem_map(device_ptrs[0], 0);
    auto worker_mem_map_risc1 = mem_helper.gen_mem_map(device_ptrs[0], mem_helper.region_size);

    if (noc_send_type == NOC_UNICAST_INLINE_WRITE or noc_send_type == NOC_UNICAST_ATOMIC_INC) {
        worker_mem_map.packet_payload_size_bytes = 16;  // l1 aligned
        worker_mem_map_risc1.packet_payload_size_bytes = 16;
    } else {
        auto single_payload_size_bytes = worker_mem_map.packet_payload_size_bytes;
        worker_mem_map.packet_payload_size_bytes = single_payload_size_bytes;
        worker_mem_map_risc1.packet_payload_size_bytes = single_payload_size_bytes;
    }

    log_info(
        tt::LogTest,
        "RISC0 mem_map: source_l1_buffer={}, target={}, test_results={}, notification={}, "
        "payload_size={}, test_results_size={}",
        worker_mem_map.source_l1_buffer_address,
        worker_mem_map.target_address,
        worker_mem_map.test_results_address,
        worker_mem_map.notification_mailbox_address,
        worker_mem_map.packet_payload_size_bytes,
        worker_mem_map.test_results_size_bytes);

    if (dual_risc) {
        log_info(
            tt::LogTest,
            "RISC1 mem_map: source_l1_buffer={}, target={}, test_results={}, notification={}, "
            "payload_size={}, test_results_size={}",
            worker_mem_map_risc1.source_l1_buffer_address,
            worker_mem_map_risc1.target_address,
            worker_mem_map_risc1.test_results_address,
            worker_mem_map_risc1.notification_mailbox_address,
            worker_mem_map_risc1.packet_payload_size_bytes,
            worker_mem_map_risc1.test_results_size_bytes);
        log_info(
            tt::LogTest,
            "DualRisc: data_space_size={}, region_size={}, num_packets={}",
            mem_helper.data_space_size,
            mem_helper.region_size,
            num_packets);
    }

    // Per-sender L1 region size on receiver
    uint32_t per_sender_l1_size = num_packets * worker_mem_map.packet_payload_size_bytes;

    // Check if receiver has enough data space for N device slots (simple indexing: slot i for device i)
    // Note: slot receiver_device_idx is unused but we allocate it for simplicity
    uint32_t total_receiver_l1_needed = static_cast<uint32_t>(num_active_devices) * per_sender_l1_size;

    // Check data space for RISC0 - data must fit between target_address and test_results_address
    if (total_receiver_l1_needed > mem_helper.data_space_size) {
        GTEST_SKIP() << "Not enough data space for RISC0. Need " << total_receiver_l1_needed
                     << " bytes, but data_space_size is " << mem_helper.data_space_size;
    }

    // RISC1 uses same data_space_size, so no separate check needed

    constexpr uint32_t req_notification_size_bytes = 128;

    // Check if sender has enough notification space for N notification slots (simple indexing: slot i for device i)
    // Each provider writes to slot = provider_device_idx, reader polls all slots except its own
    uint32_t total_notification_l1_needed = static_cast<uint32_t>(num_active_devices) * req_notification_size_bytes;
    if (total_notification_l1_needed > DualRiscMemMapHelper::notification_size_bytes) {
        GTEST_SKIP() << "Not enough notification space. Need " << total_notification_l1_needed
                     << " bytes, but notification_size is " << DualRiscMemMapHelper::notification_size_bytes;
    }

    // Determine kernel paths based on operation type
    const bool is_read = (noc_send_type == NOC_UNICAST_READ);
    const char* sender_kernel_path =
        is_read ? "tests/tt_metal/tt_fabric/fabric_data_movement/kernels/test_udm_read_sender_all_to_all.cpp"
                : "tests/tt_metal/tt_fabric/fabric_data_movement/kernels/test_udm_sender_all_to_all.cpp";
    const char* receiver_kernel_path =
        is_read ? "tests/tt_metal/tt_fabric/fabric_data_movement/kernels/test_udm_read_receiver_all_to_all.cpp"
                : "tests/tt_metal/tt_fabric/fabric_data_movement/kernels/test_udm_receiver_all_to_all.cpp";

    // Create programs for each device (each device has both sender and receiver programs)
    std::vector<tt_metal::Program> programs;
    programs.reserve(num_active_devices);
    for (size_t dev_idx = 0; dev_idx < num_active_devices; dev_idx++) {
        programs.push_back(tt_metal::CreateProgram());
    }

    // Create sender and receiver kernels for each device
    for (size_t dev_idx = 0; dev_idx < num_active_devices; dev_idx++) {
        const auto& device_ptr = device_ptrs[dev_idx];

        // This device's index - used directly as L1 slot index (simple indexing)
        uint32_t this_device_idx = static_cast<uint32_t>(dev_idx);

        // Sender compile time args - includes num_destinations and sender_device_idx
        std::vector<uint32_t> sender_compile_time_args = {
            worker_mem_map.test_results_address,
            worker_mem_map.test_results_size_bytes,
            worker_mem_map.notification_mailbox_address,
            worker_mem_map.target_address,  // target_address_base
            noc_send_type,
            worker_mem_map.source_l1_buffer_address,
            worker_mem_map.packet_payload_size_bytes,
            num_packets,
            time_seed,
            req_notification_size_bytes,
            per_sender_l1_size,  // per_sender_l1_size
            num_other_devices,   // num_destinations (N-1)
            this_device_idx};    // sender_device_idx (writes to slot this_device_idx)

        // Receiver compile time args - includes num_devices and receiver_device_idx
        std::vector<uint32_t> receiver_compile_time_args = {
            worker_mem_map.test_results_address,
            worker_mem_map.test_results_size_bytes,
            worker_mem_map.notification_mailbox_address,
            worker_mem_map.target_address,  // target_address_base
            noc_send_type,
            worker_mem_map.packet_payload_size_bytes,
            num_packets,
            time_seed,
            req_notification_size_bytes,
            static_cast<uint32_t>(num_active_devices),  // num_devices (total N)
            per_sender_l1_size,                         // per_sender_l1_size
            this_device_idx};                           // receiver_device_idx (skip this slot)

        // Collect all sender-receiver core pairs (top half senders, bottom half receivers)
        std::vector<CoreCoord> sender_logical_cores;
        std::vector<CoreCoord> receiver_logical_cores;
        for (uint32_t i = 0; i < num_core_pairs; i++) {
            uint32_t x = i % grid_size.x;
            uint32_t sender_y = i / grid_size.x;
            uint32_t receiver_y = receiver_y_start + sender_y;
            sender_logical_cores.push_back({x, sender_y});
            receiver_logical_cores.push_back({x, receiver_y});
        }

        // Create sender kernel for all sender cores on this device
        CoreRangeSet sender_core_range(sender_logical_cores);
        auto sender_kernel_risc0 = tt_metal::CreateKernel(
            programs[dev_idx],
            sender_kernel_path,
            sender_core_range,
            tt_metal::DataMovementConfig{
                .processor = tt_metal::DataMovementProcessor::RISCV_0,
                .noc = tt_metal::NOC::RISCV_0_default,
                .noc_mode = noc_mode,
                .compile_args = sender_compile_time_args});

        // Create RISC1 sender kernel if dual_risc mode is enabled
        tt_metal::KernelHandle sender_kernel_risc1 = 0;
        if (dual_risc) {
            std::vector<uint32_t> sender_compile_time_args_risc1 = {
                worker_mem_map_risc1.test_results_address,
                worker_mem_map_risc1.test_results_size_bytes,
                worker_mem_map_risc1.notification_mailbox_address,
                worker_mem_map_risc1.target_address,
                noc_send_type,
                worker_mem_map_risc1.source_l1_buffer_address,
                worker_mem_map_risc1.packet_payload_size_bytes,
                num_packets,
                time_seed,
                req_notification_size_bytes,
                per_sender_l1_size,
                num_other_devices,
                this_device_idx};

            sender_kernel_risc1 = tt_metal::CreateKernel(
                programs[dev_idx],
                sender_kernel_path,
                sender_core_range,
                tt_metal::DataMovementConfig{
                    .processor = tt_metal::DataMovementProcessor::RISCV_1,
                    .noc = tt_metal::NOC::RISCV_1_default,
                    .noc_mode = noc_mode,
                    .compile_args = sender_compile_time_args_risc1});
        }

        // Set runtime args per sender core
        // Each sender sends to the corresponding receiver on ALL other devices
        for (uint32_t core_idx = 0; core_idx < num_core_pairs; core_idx++) {
            CoreCoord sender_logical_core = sender_logical_cores[core_idx];
            CoreCoord receiver_logical_core = receiver_logical_cores[core_idx];

            // Runtime args: for each destination: (noc_x, noc_y, dst_dev_id, dst_mesh_id)
            std::vector<uint32_t> sender_runtime_args;
            for (size_t dest_idx = 0; dest_idx < num_active_devices; dest_idx++) {
                if (dest_idx == dev_idx) {
                    continue;  // Skip self
                }
                const auto& dest_fabric_node_id = fabric_node_ids[dest_idx];
                const auto& dest_device_ptr = device_ptrs[dest_idx];
                CoreCoord receiver_virtual_core = dest_device_ptr->worker_core_from_logical_core(receiver_logical_core);

                sender_runtime_args.push_back(receiver_virtual_core.x);
                sender_runtime_args.push_back(receiver_virtual_core.y);
                sender_runtime_args.push_back(dest_fabric_node_id.chip_id);
                sender_runtime_args.push_back(dest_fabric_node_id.mesh_id.get());
            }
            tt_metal::SetRuntimeArgs(programs[dev_idx], sender_kernel_risc0, sender_logical_core, sender_runtime_args);
            if (dual_risc) {
                tt_metal::SetRuntimeArgs(
                    programs[dev_idx], sender_kernel_risc1, sender_logical_core, sender_runtime_args);
            }
        }

        // Create receiver kernel for all receiver cores on this device
        CoreRangeSet receiver_core_range(receiver_logical_cores);
        auto receiver_kernel_risc0 = tt_metal::CreateKernel(
            programs[dev_idx],
            receiver_kernel_path,
            receiver_core_range,
            tt_metal::DataMovementConfig{
                .processor = tt_metal::DataMovementProcessor::RISCV_0,
                .noc = tt_metal::NOC::RISCV_0_default,
                .noc_mode = noc_mode,
                .compile_args = receiver_compile_time_args});

        // Create RISC1 receiver kernel if dual_risc mode is enabled
        tt_metal::KernelHandle receiver_kernel_risc1 = 0;
        if (dual_risc) {
            std::vector<uint32_t> receiver_compile_time_args_risc1 = {
                worker_mem_map_risc1.test_results_address,
                worker_mem_map_risc1.test_results_size_bytes,
                worker_mem_map_risc1.notification_mailbox_address,
                worker_mem_map_risc1.target_address,
                noc_send_type,
                worker_mem_map_risc1.packet_payload_size_bytes,
                num_packets,
                time_seed,
                req_notification_size_bytes,
                static_cast<uint32_t>(num_active_devices),
                per_sender_l1_size,
                this_device_idx};

            receiver_kernel_risc1 = tt_metal::CreateKernel(
                programs[dev_idx],
                receiver_kernel_path,
                receiver_core_range,
                tt_metal::DataMovementConfig{
                    .processor = tt_metal::DataMovementProcessor::RISCV_1,
                    .noc = tt_metal::NOC::RISCV_1_default,
                    .noc_mode = noc_mode,
                    .compile_args = receiver_compile_time_args_risc1});
        }

        // For reads: receiver (data provider) needs to notify all readers
        // Runtime args: (noc_x, noc_y, dst_dev_id, dst_mesh_id) for each reader
        if (is_read) {
            for (uint32_t core_idx = 0; core_idx < num_core_pairs; core_idx++) {
                CoreCoord sender_logical_core = sender_logical_cores[core_idx];
                CoreCoord receiver_logical_core = receiver_logical_cores[core_idx];

                std::vector<uint32_t> receiver_runtime_args;
                for (size_t reader_idx = 0; reader_idx < num_active_devices; reader_idx++) {
                    if (reader_idx == dev_idx) {
                        continue;  // Skip self
                    }
                    const auto& reader_fabric_node_id = fabric_node_ids[reader_idx];
                    const auto& reader_device_ptr = device_ptrs[reader_idx];
                    // The reader's sender core at the same position reads from this receiver
                    CoreCoord reader_sender_virtual_core =
                        reader_device_ptr->worker_core_from_logical_core(sender_logical_core);

                    receiver_runtime_args.push_back(reader_sender_virtual_core.x);
                    receiver_runtime_args.push_back(reader_sender_virtual_core.y);
                    receiver_runtime_args.push_back(reader_fabric_node_id.chip_id);
                    receiver_runtime_args.push_back(reader_fabric_node_id.mesh_id.get());
                }
                tt_metal::SetRuntimeArgs(
                    programs[dev_idx], receiver_kernel_risc0, receiver_logical_core, receiver_runtime_args);
                if (dual_risc) {
                    tt_metal::SetRuntimeArgs(
                        programs[dev_idx], receiver_kernel_risc1, receiver_logical_core, receiver_runtime_args);
                }
            }
        }

        // Clear target L1 memory for all N device slots (simple indexing uses N slots, slot receiver_device_idx unused)
        if (noc_send_type == NOC_UNICAST_ATOMIC_INC) {
            uint32_t total_l1_to_clear = static_cast<uint32_t>(num_active_devices) * per_sender_l1_size;
            for (uint32_t core_idx = 0; core_idx < num_core_pairs; core_idx++) {
                CoreCoord receiver_logical_core = receiver_logical_cores[core_idx];
                std::vector<uint32_t> zeros(total_l1_to_clear / sizeof(uint32_t), 0);
                tt_metal::detail::WriteToDeviceL1(
                    device_ptr->get_devices()[0],
                    receiver_logical_core,
                    worker_mem_map.target_address,
                    zeros,
                    CoreType::WORKER);
                if (dual_risc) {
                    tt_metal::detail::WriteToDeviceL1(
                        device_ptr->get_devices()[0],
                        receiver_logical_core,
                        worker_mem_map_risc1.target_address,
                        zeros,
                        CoreType::WORKER);
                }
            }
        }
    }

    log_info(tt::LogTest, "All-to-all test starting");
    for (size_t dev_idx = 0; dev_idx < num_active_devices; dev_idx++) {
        fixture->RunProgramNonblocking(device_ptrs[dev_idx], programs[dev_idx]);
    }
    log_info(tt::LogTest, "All-to-all test waiting for finish");
    // Wait for all devices to complete
    for (size_t dev_idx = 0; dev_idx < num_active_devices; dev_idx++) {
        fixture->WaitForSingleProgramDone(device_ptrs[dev_idx], programs[dev_idx]);
    }
    log_info(tt::LogTest, "All-to-all test done");

    // Lambda to check results for a given RISC
    auto check_risc_results = [&](const WorkerMemMap& mem_map, const std::string& risc_name) {
        for (size_t dev_idx = 0; dev_idx < num_active_devices; dev_idx++) {
            const auto& device_ptr = device_ptrs[dev_idx];
            const auto& fabric_node_id = fabric_node_ids[dev_idx];

            uint64_t total_sender_bytes = 0;
            uint64_t total_receiver_bytes = 0;

            for (uint32_t core_idx = 0; core_idx < num_core_pairs; core_idx++) {
                uint32_t x = core_idx % grid_size.x;
                uint32_t sender_y = core_idx / grid_size.x;
                uint32_t receiver_y = receiver_y_start + sender_y;
                CoreCoord sender_logical_core = {x, sender_y};
                CoreCoord receiver_logical_core = {x, receiver_y};

                // Check sender status
                std::vector<uint32_t> sender_status;
                tt_metal::detail::ReadFromDeviceL1(
                    device_ptr->get_devices()[0],
                    sender_logical_core,
                    mem_map.test_results_address,
                    mem_map.test_results_size_bytes,
                    sender_status,
                    CoreType::WORKER);
                EXPECT_EQ(sender_status[TT_FABRIC_STATUS_INDEX], TT_FABRIC_STATUS_PASS)
                    << risc_name << " Sender failed on device " << fabric_node_id.chip_id << " core (" << x << ","
                    << sender_y << ")";

                uint64_t sender_words = ((uint64_t)sender_status[TT_FABRIC_WORD_CNT_INDEX + 1] << 32) |
                                        sender_status[TT_FABRIC_WORD_CNT_INDEX];
                total_sender_bytes += sender_words;

                // Check receiver status
                std::vector<uint32_t> receiver_status;
                tt_metal::detail::ReadFromDeviceL1(
                    device_ptr->get_devices()[0],
                    receiver_logical_core,
                    mem_map.test_results_address,
                    mem_map.test_results_size_bytes,
                    receiver_status,
                    CoreType::WORKER);
                EXPECT_EQ(receiver_status[TT_FABRIC_STATUS_INDEX], TT_FABRIC_STATUS_PASS)
                    << risc_name << " Receiver failed on device " << fabric_node_id.chip_id << " core (" << x << ","
                    << receiver_y << ")";

                uint64_t receiver_words = ((uint64_t)receiver_status[TT_FABRIC_WORD_CNT_INDEX + 1] << 32) |
                                          receiver_status[TT_FABRIC_WORD_CNT_INDEX];
                total_receiver_bytes += receiver_words;
            }

            uint64_t expected_bytes_per_core = num_other_devices * num_packets * mem_map.packet_payload_size_bytes;
            uint64_t expected_total_bytes = num_core_pairs * expected_bytes_per_core;
            EXPECT_EQ(total_sender_bytes, expected_total_bytes)
                << risc_name << " Total sender byte count mismatch on device " << fabric_node_id.chip_id;
            EXPECT_EQ(total_receiver_bytes, expected_total_bytes)
                << risc_name << " Total receiver byte count mismatch on device " << fabric_node_id.chip_id;
        }
    };

    // Validate RISC0 results
    check_risc_results(worker_mem_map, "RISC0");

    // Validate RISC1 results if dual_risc mode is enabled
    if (dual_risc) {
        check_risc_results(worker_mem_map_risc1, "RISC1");
    }

    log_info(
        tt::LogTest,
        "All-to-all test passed: {} devices, {} core pairs, each sending to {} other devices, dual_risc={}",
        num_active_devices,
        num_core_pairs,
        num_other_devices,
        dual_risc);
}

void Fabric2DMulticastCommon(
    BaseFabricFixture* fixture,
    NocSendType noc_send_type,
    const std::vector<std::vector<std::tuple<RoutingDirection, uint32_t, uint32_t>>>& connection_configs,
    bool with_state) {
    CoreCoord sender_logical_core = {0, 0};
    CoreCoord receiver_logical_core = {1, 0};
    uint32_t num_packets = 10;
    uint32_t time_seed = std::chrono::system_clock::now().time_since_epoch().count();

    auto& control_plane = tt::tt_metal::MetalContext::instance().get_control_plane();
    const auto topology = control_plane.get_fabric_context().get_fabric_topology();

    TT_FATAL(!connection_configs.empty(), "Must have at least 1 connection configuration");

    // Each connection configuration is a separate multicast route
    uint32_t num_connections = connection_configs.size();

    FabricNodeId src_fabric_node_id(MeshId{0}, 0);

    // Calculate max hops needed in each direction across all connections
    std::unordered_map<RoutingDirection, uint32_t> fabric_hops;
    for (const auto& dir_configs : connection_configs) {
        for (auto [dir, start_distance, range] : dir_configs) {
            uint32_t max_hop = start_distance + range;
            if (!fabric_hops.contains(dir) || fabric_hops[dir] < max_hop) {
                fabric_hops[dir] = max_hop;
            }
        }
    }

    std::unordered_map<RoutingDirection, std::vector<FabricNodeId>> end_fabric_node_ids_by_dir;
    ChipId src_physical_device_id;
    std::unordered_map<RoutingDirection, std::vector<ChipId>> physical_end_device_ids_by_dir;
    if (!find_device_with_neighbor_in_multi_direction(
            fixture,
            src_fabric_node_id,
            end_fabric_node_ids_by_dir,
            src_physical_device_id,
            physical_end_device_ids_by_dir,
            fabric_hops)) {
        GTEST_SKIP() << "No multicast destinations found for requested directions";
    }

    auto sender_device = fixture->get_device(src_physical_device_id);
    auto worker_mem_map = generate_worker_mem_map(sender_device, topology);

    // Setup connections and collect receiver devices for each multicast route
    std::vector<FabricNodeId> dest_fabric_node_ids;
    std::unordered_set<ChipId> receiver_device_ids;

    // Storage for per-connection E/W/N/S ranges
    std::vector<std::array<uint32_t, 4>> connection_ranges;  // [e, w, n, s] per connection

    // Helper lambda to traverse neighbors in a given direction
    auto traverse_direction =
        [&](FabricNodeId start_node, RoutingDirection dir, uint32_t range, const std::string& context) {
            auto curr_fabric_node_id = start_node;
            for (uint32_t hop = 0; hop < range; hop++) {
                auto neighbors = control_plane.get_intra_chip_neighbors(curr_fabric_node_id, dir);
                if (!neighbors.empty()) {
                    auto neighbor_fabric_node_id = FabricNodeId(curr_fabric_node_id.mesh_id, neighbors[0]);
                    auto neighbor_physical_chip_id =
                        control_plane.get_physical_chip_id_from_fabric_node_id(neighbor_fabric_node_id);
                    receiver_device_ids.insert(neighbor_physical_chip_id);
                    curr_fabric_node_id = neighbor_fabric_node_id;
                } else {
                    log_warning(
                        tt::LogTest,
                        "Not enough {} neighbors at {}, expected {} hops but only got {}",
                        dir,
                        context,
                        range,
                        hop);
                    break;
                }
            }
        };

    for (const auto& dir_configs : connection_configs) {
        TT_FATAL(!dir_configs.empty(), "Each connection must have at least 1 direction");

        // Extract E/W/N/S ranges for this connection
        uint32_t e_range = 0, w_range = 0, n_range = 0, s_range = 0;
        for (auto [dir, start_distance, range] : dir_configs) {
            switch (dir) {
                case RoutingDirection::E: e_range = range; break;
                case RoutingDirection::W: w_range = range; break;
                case RoutingDirection::N: n_range = range; break;
                case RoutingDirection::S: s_range = range; break;
                default: GTEST_SKIP() << "Invalid direction in connection configuration: " << static_cast<int>(dir);
            }
        }
        connection_ranges.push_back({e_range, w_range, n_range, s_range});

        // Determine trunk direction (N or S if present, otherwise use E/W only mode)
        RoutingDirection trunk_dir;
        uint32_t trunk_start_distance = 0;
        uint32_t trunk_range = 1;
        bool has_trunk = false;

        for (auto [dir, start_distance, range] : dir_configs) {
            if (dir == RoutingDirection::N || dir == RoutingDirection::S) {
                trunk_dir = dir;
                trunk_start_distance = start_distance;
                trunk_range = range;
                has_trunk = true;
                break;
            }
        }

        // Setup connection to first hop
        if (has_trunk) {
            // Trunk mode: establish connection to the first hop in the trunk direction
            if (physical_end_device_ids_by_dir[trunk_dir].size() > trunk_start_distance) {
                auto dest_node = end_fabric_node_ids_by_dir[trunk_dir][trunk_start_distance];
                dest_fabric_node_ids.push_back(dest_node);
            } else {
                GTEST_SKIP() << "Not enough hops in trunk direction for start_distance";
            }
        } else {
            // E/W only mode: use first E or W direction as the connection
            for (auto [dir, start_distance, range] : dir_configs) {
                if (dir == RoutingDirection::E || dir == RoutingDirection::W) {
                    if (physical_end_device_ids_by_dir[dir].size() > start_distance) {
                        auto dest_node = end_fabric_node_ids_by_dir[dir][start_distance];
                        dest_fabric_node_ids.push_back(dest_node);
                    } else {
                        GTEST_SKIP() << "Not enough hops in E/W direction for start_distance";
                    }
                    break;
                }
            }
        }

        // Enumerate receiver devices for this connection
        if (has_trunk) {
            // Trunk + branch mode: traverse trunk, then branch E/W from each trunk position
            for (uint32_t trunk_hop = trunk_start_distance;
                 trunk_hop < trunk_start_distance + trunk_range &&
                 trunk_hop < physical_end_device_ids_by_dir[trunk_dir].size();
                 trunk_hop++) {
                auto trunk_fabric_node_id = end_fabric_node_ids_by_dir[trunk_dir][trunk_hop];
                auto trunk_physical_chip_id = physical_end_device_ids_by_dir[trunk_dir][trunk_hop];
                receiver_device_ids.insert(trunk_physical_chip_id);

                if (e_range > 0) {
                    traverse_direction(
                        trunk_fabric_node_id,
                        RoutingDirection::E,
                        e_range,
                        "trunk position " + std::to_string(trunk_hop));
                }
                if (w_range > 0) {
                    traverse_direction(
                        trunk_fabric_node_id,
                        RoutingDirection::W,
                        w_range,
                        "trunk position " + std::to_string(trunk_hop));
                }
            }
        } else {
            // E/W only mode: multicast from source chip in E/W directions
            if (e_range > 0) {
                traverse_direction(src_fabric_node_id, RoutingDirection::E, e_range, "source");
            }
            if (w_range > 0) {
                traverse_direction(src_fabric_node_id, RoutingDirection::W, w_range, "source");
            }
        }
    }

    // Choose a receiver device to compute RX core coords
    if (receiver_device_ids.empty()) {
        GTEST_SKIP() << "No multicast receivers found";
    }
    auto last_recv_phys_chip_id = *receiver_device_ids.begin();
    auto last_recv_device = fixture->get_device(last_recv_phys_chip_id);
    CoreCoord receiver_virtual_core = last_recv_device->worker_core_from_logical_core(receiver_logical_core);

    if (noc_send_type == NOC_UNICAST_INLINE_WRITE) {
        worker_mem_map.packet_payload_size_bytes = 4;
    }

    std::vector<uint32_t> compile_time_args = {
        worker_mem_map.test_results_address,
        worker_mem_map.test_results_size_bytes,
        worker_mem_map.notification_mailbox_address,
        worker_mem_map.target_address,
        noc_send_type,
        num_connections,  // Number of connections (multicast routes)
        with_state,
        1  // is_chip_multicast = 1
    };

    std::vector<uint32_t> sender_runtime_args = {
        worker_mem_map.source_l1_buffer_address,
        worker_mem_map.packet_payload_size_bytes,
        num_packets,
        time_seed,
        receiver_virtual_core.x,
        receiver_virtual_core.y,
    };

    // For mesh API: pass e_hops, w_hops, n_hops, s_hops for each connection
    for (const auto& ranges : connection_ranges) {
        sender_runtime_args.push_back(ranges[0]);  // e_range
        sender_runtime_args.push_back(ranges[1]);  // w_range
        sender_runtime_args.push_back(ranges[2]);  // n_range
        sender_runtime_args.push_back(ranges[3]);  // s_range
    }

    auto sender_program = tt_metal::CreateProgram();
    auto sender_kernel = tt_metal::CreateKernel(
        sender_program,
        (noc_send_type == NOC_FUSED_UNICAST_ATOMIC_INC || noc_send_type == NOC_UNICAST_ATOMIC_INC)
            ? "tests/tt_metal/tt_fabric/fabric_data_movement/kernels/test_linear_api_atomic_inc_sender.cpp"
            : "tests/tt_metal/tt_fabric/fabric_data_movement/kernels/test_linear_api_unicast_write_sender.cpp",
        {sender_logical_core},
        tt_metal::DataMovementConfig{
            .processor = tt_metal::DataMovementProcessor::RISCV_0,
            .noc = tt_metal::NOC::RISCV_0_default,
            .compile_args = compile_time_args});

    // Append connection manager args for all connections at once
    // Using default (std::nullopt) for auto-detection: if N/S connections exist, validates no N+S mixing
    append_routing_plane_connection_manager_rt_args(
        src_fabric_node_id,
        dest_fabric_node_ids,
        {},
        sender_program,
        sender_kernel,
        {sender_logical_core},
        sender_runtime_args,
        tt::tt_fabric::FabricApiType::Mesh);

    tt_metal::SetRuntimeArgs(sender_program, sender_kernel, sender_logical_core, sender_runtime_args);

    // Build and launch receiver programs for all destination devices in the rectangular multicast region
    std::vector<std::pair<std::shared_ptr<tt_metal::distributed::MeshDevice>, tt_metal::Program>> receiver_programs;
    std::vector<uint32_t> receiver_runtime_args = {worker_mem_map.packet_payload_size_bytes, num_packets, time_seed};
    for (auto physical_end_device_id : receiver_device_ids) {
        // auto recv_fabric_node_id = control_plane.get_fabric_node_id_from_physical_chip_id(physical_end_device_id);
        auto receiver_device = fixture->get_device(physical_end_device_id);
        auto receiver_program = tt_metal::CreateProgram();
        auto receiver_kernel = tt_metal::CreateKernel(
            receiver_program,
            (noc_send_type == NOC_FUSED_UNICAST_ATOMIC_INC || noc_send_type == NOC_UNICAST_ATOMIC_INC)
                ? "tests/tt_metal/tt_fabric/fabric_data_movement/kernels/test_linear_api_atomic_inc_receiver.cpp"
                : "tests/tt_metal/tt_fabric/fabric_data_movement/kernels/test_linear_api_receiver.cpp",
            {receiver_logical_core},
            tt_metal::DataMovementConfig{
                .processor = tt_metal::DataMovementProcessor::RISCV_0,
                .noc = tt_metal::NOC::RISCV_0_default,
                .compile_args = compile_time_args});
        tt_metal::SetRuntimeArgs(receiver_program, receiver_kernel, receiver_logical_core, receiver_runtime_args);
        fixture->RunProgramNonblocking(receiver_device, receiver_program);
        receiver_programs.emplace_back(receiver_device, std::move(receiver_program));
    }

    fixture->RunProgramNonblocking(sender_device, sender_program);
    fixture->WaitForSingleProgramDone(sender_device, sender_program);

    for (auto& [dev, prog] : receiver_programs) {
        fixture->WaitForSingleProgramDone(dev, prog);
    }

    std::vector<uint32_t> sender_status;
    tt_metal::detail::ReadFromDeviceL1(
        sender_device->get_devices()[0],
        sender_logical_core,
        worker_mem_map.test_results_address,
        worker_mem_map.test_results_size_bytes,
        sender_status,
        CoreType::WORKER);
    EXPECT_EQ(sender_status[TT_FABRIC_STATUS_INDEX], TT_FABRIC_STATUS_PASS);
    uint64_t sender_words =
        ((uint64_t)sender_status[TT_FABRIC_WORD_CNT_INDEX + 1] << 32) | sender_status[TT_FABRIC_WORD_CNT_INDEX];

    for (auto& [dev, _] : receiver_programs) {
        std::vector<uint32_t> recv_status;
        tt_metal::detail::ReadFromDeviceL1(
            dev->get_devices()[0],
            receiver_logical_core,
            worker_mem_map.test_results_address,
            worker_mem_map.test_results_size_bytes,
            recv_status,
            CoreType::WORKER);
        EXPECT_EQ(recv_status[TT_FABRIC_STATUS_INDEX], TT_FABRIC_STATUS_PASS);
        uint64_t receiver_words =
            ((uint64_t)recv_status[TT_FABRIC_WORD_CNT_INDEX + 1] << 32) | recv_status[TT_FABRIC_WORD_CNT_INDEX];
        EXPECT_EQ(sender_words, receiver_words);
    }
}

void FabricMulticastCommon(
    BaseFabricFixture* fixture,
    NocSendType noc_send_type,
    const std::vector<std::tuple<RoutingDirection, uint32_t, uint32_t>>& pair_ordered_dir_configs,
    bool with_state) {
    CoreCoord sender_logical_core = {0, 0};
    CoreCoord receiver_logical_core = {1, 0};
    uint32_t num_packets = 10;
    uint32_t time_seed = std::chrono::system_clock::now().time_since_epoch().count();

    auto& control_plane = tt::tt_metal::MetalContext::instance().get_control_plane();
    const auto topology = control_plane.get_fabric_context().get_fabric_topology();

    // Limit directions per cluster (T3K: up to 3, TG: up to 4) and force range=1
    const auto cluster_type = tt::tt_metal::MetalContext::instance().get_cluster().get_cluster_type();
    size_t max_dirs = (cluster_type == tt::tt_metal::ClusterType::T3K) ? 3 : 4;
    std::vector<std::tuple<RoutingDirection, uint32_t, uint32_t>> dir_configs = pair_ordered_dir_configs;
    if (dir_configs.size() > max_dirs) {
        dir_configs.resize(max_dirs);
    }

    FabricNodeId src_fabric_node_id(MeshId{0}, 0);
    std::unordered_map<RoutingDirection, uint32_t> fabric_hops;
    for (auto [dir, start_distance, range] : dir_configs) {
        fabric_hops[dir] = start_distance + range - 1;
    }

    std::unordered_map<RoutingDirection, std::vector<FabricNodeId>> end_fabric_node_ids_by_dir;
    ChipId src_physical_device_id;
    std::unordered_map<RoutingDirection, std::vector<ChipId>> physical_end_device_ids_by_dir;
    if (!find_device_with_neighbor_in_multi_direction(
            fixture,
            src_fabric_node_id,
            end_fabric_node_ids_by_dir,
            src_physical_device_id,
            physical_end_device_ids_by_dir,
            fabric_hops)) {
        GTEST_SKIP() << "No multicast destinations found for requested directions";
    }

    auto sender_device = fixture->get_device(src_physical_device_id);
    auto worker_mem_map = generate_worker_mem_map(sender_device, topology);

    // Adjust lists to start from start_distance for each direction
    std::vector<FabricNodeId> dest_fabric_node_ids;
    for (auto [dir, start_distance, range] : dir_configs) {
        physical_end_device_ids_by_dir[dir] = std::vector(
            physical_end_device_ids_by_dir[dir].begin() + (start_distance - 1),
            physical_end_device_ids_by_dir[dir].end());
        end_fabric_node_ids_by_dir[dir] = std::vector(
            end_fabric_node_ids_by_dir[dir].begin() + (start_distance - 1), end_fabric_node_ids_by_dir[dir].end());
        dest_fabric_node_ids.push_back(end_fabric_node_ids_by_dir[dir][0]);
    }

    // Choose a receiver device to compute RX core coords (use last device from first configured dir)
    auto first_dir = std::get<0>(dir_configs.front());
    if (physical_end_device_ids_by_dir[first_dir].empty()) {
        GTEST_SKIP() << "No multicast receivers after start_distance adjustment";
    }
    auto last_recv_phys_chip_id = physical_end_device_ids_by_dir[first_dir].back();
    auto last_recv_device = fixture->get_device(last_recv_phys_chip_id);
    CoreCoord receiver_virtual_core = last_recv_device->worker_core_from_logical_core(receiver_logical_core);

    if (noc_send_type == NOC_UNICAST_INLINE_WRITE) {
        worker_mem_map.packet_payload_size_bytes = 4;
    }

    std::vector<uint32_t> compile_time_args = {
        worker_mem_map.test_results_address,
        worker_mem_map.test_results_size_bytes,
        worker_mem_map.notification_mailbox_address,
        worker_mem_map.target_address,
        noc_send_type,
        static_cast<uint32_t>(dir_configs.size()),
        with_state,
        1  // is_chip_multicast = 1
    };

    std::vector<uint32_t> sender_runtime_args = {
        worker_mem_map.source_l1_buffer_address,
        worker_mem_map.packet_payload_size_bytes,
        num_packets,
        time_seed,
        receiver_virtual_core.x,
        receiver_virtual_core.y,
    };
    for (auto [dir, start_distance, range] : dir_configs) {
        sender_runtime_args.push_back(start_distance);
        sender_runtime_args.push_back(range);
    }

    auto sender_program = tt_metal::CreateProgram();
    auto sender_kernel = tt_metal::CreateKernel(
        sender_program,
        (noc_send_type == NOC_FUSED_UNICAST_ATOMIC_INC || noc_send_type == NOC_UNICAST_ATOMIC_INC)
            ? "tests/tt_metal/tt_fabric/fabric_data_movement/kernels/test_linear_api_atomic_inc_sender.cpp"
            : "tests/tt_metal/tt_fabric/fabric_data_movement/kernels/test_linear_api_unicast_write_sender.cpp",
        {sender_logical_core},
        tt_metal::DataMovementConfig{
            .processor = tt_metal::DataMovementProcessor::RISCV_0,
            .noc = tt_metal::NOC::RISCV_0_default,
            .compile_args = compile_time_args});

    append_routing_plane_connection_manager_rt_args(
        src_fabric_node_id,
        dest_fabric_node_ids,
        {},
        sender_program,
        sender_kernel,
        {sender_logical_core},
        sender_runtime_args);

    tt_metal::SetRuntimeArgs(sender_program, sender_kernel, sender_logical_core, sender_runtime_args);

    // Build and launch receiver programs for all destination devices in all configured directions
    std::vector<std::pair<std::shared_ptr<tt_metal::distributed::MeshDevice>, tt_metal::Program>> receiver_programs;
    std::vector<uint32_t> receiver_runtime_args = {worker_mem_map.packet_payload_size_bytes, num_packets, time_seed};
    for (auto& [dir, start_distance, range] : dir_configs) {
        for (auto physical_end_device_id : physical_end_device_ids_by_dir[dir]) {
            auto receiver_device = fixture->get_device(physical_end_device_id);
            auto receiver_program = tt_metal::CreateProgram();
            auto receiver_kernel = tt_metal::CreateKernel(
                receiver_program,
                (noc_send_type == NOC_FUSED_UNICAST_ATOMIC_INC || noc_send_type == NOC_UNICAST_ATOMIC_INC)
                    ? "tests/tt_metal/tt_fabric/fabric_data_movement/kernels/test_linear_api_atomic_inc_receiver.cpp"
                    : "tests/tt_metal/tt_fabric/fabric_data_movement/kernels/test_linear_api_receiver.cpp",
                {receiver_logical_core},
                tt_metal::DataMovementConfig{
                    .processor = tt_metal::DataMovementProcessor::RISCV_0,
                    .noc = tt_metal::NOC::RISCV_0_default,
                    .compile_args = compile_time_args});
            tt_metal::SetRuntimeArgs(receiver_program, receiver_kernel, receiver_logical_core, receiver_runtime_args);
            fixture->RunProgramNonblocking(receiver_device, receiver_program);
            receiver_programs.emplace_back(receiver_device, std::move(receiver_program));
        }
    }

    fixture->RunProgramNonblocking(sender_device, sender_program);
    fixture->WaitForSingleProgramDone(sender_device, sender_program);

    for (auto& [dev, prog] : receiver_programs) {
        fixture->WaitForSingleProgramDone(dev, prog);
    }

    std::vector<uint32_t> sender_status;
    tt_metal::detail::ReadFromDeviceL1(
        sender_device->get_devices()[0],
        sender_logical_core,
        worker_mem_map.test_results_address,
        worker_mem_map.test_results_size_bytes,
        sender_status,
        CoreType::WORKER);
    EXPECT_EQ(sender_status[TT_FABRIC_STATUS_INDEX], TT_FABRIC_STATUS_PASS);
    uint64_t sender_words =
        ((uint64_t)sender_status[TT_FABRIC_WORD_CNT_INDEX + 1] << 32) | sender_status[TT_FABRIC_WORD_CNT_INDEX];

    for (auto& [dev, _] : receiver_programs) {
        std::vector<uint32_t> recv_status;
        tt_metal::detail::ReadFromDeviceL1(
            dev->get_devices()[0],
            receiver_logical_core,
            worker_mem_map.test_results_address,
            worker_mem_map.test_results_size_bytes,
            recv_status,
            CoreType::WORKER);
        EXPECT_EQ(recv_status[TT_FABRIC_STATUS_INDEX], TT_FABRIC_STATUS_PASS);
        uint64_t receiver_words =
            ((uint64_t)recv_status[TT_FABRIC_WORD_CNT_INDEX + 1] << 32) | recv_status[TT_FABRIC_WORD_CNT_INDEX];
        EXPECT_EQ(sender_words, receiver_words);
    }
}

// 1D Linear Fabric API Tests
TEST_F(NightlyFabric1DFixture, TestLinearFabricUnicastNocUnicastWrite) {
    FabricUnicastCommon(this, NOC_UNICAST_WRITE, {std::make_tuple(RoutingDirection::E, 1)});
}
TEST_F(NightlyFabric1DFixture, TestLinearFabricUnicastNocUnicastWriteMultiDir) {
    FabricUnicastCommon(
        this, NOC_UNICAST_WRITE, {std::make_tuple(RoutingDirection::E, 1), std::make_tuple(RoutingDirection::W, 1)});
}
TEST_F(NightlyFabric1DFixture, TestLinearFabricUnicastNocUnicastWriteWithState) {
    FabricUnicastCommon(
        this,
        NOC_UNICAST_WRITE,
        {std::make_tuple(RoutingDirection::E, 1), std::make_tuple(RoutingDirection::W, 1)},
        FabricApiType::Linear,
        true);
}

TEST_F(NightlyFabric1DFixture, TestLinearFabricUnicastNocAtomicInc) {
    FabricUnicastCommon(this, NOC_UNICAST_ATOMIC_INC, {std::make_tuple(RoutingDirection::E, 1)});
}
TEST_F(NightlyFabric1DFixture, TestLinearFabricUnicastNocAtomicIncMultiDir) {
    FabricUnicastCommon(
        this,
        NOC_UNICAST_ATOMIC_INC,
        {std::make_tuple(RoutingDirection::E, 1), std::make_tuple(RoutingDirection::W, 1)});
}
TEST_F(NightlyFabric1DFixture, TestLinearFabricUnicastNocAtomicIncWithState) {
    FabricUnicastCommon(
        this,
        NOC_UNICAST_ATOMIC_INC,
        {std::make_tuple(RoutingDirection::E, 1), std::make_tuple(RoutingDirection::W, 1)},
        FabricApiType::Linear,
        true);
}

TEST_F(NightlyFabric1DFixture, TestLinearFabricUnicastNocScatterWrite) {
    FabricUnicastCommon(this, NOC_UNICAST_SCATTER_WRITE, {std::make_tuple(RoutingDirection::E, 1)});
}
TEST_F(NightlyFabric1DFixture, TestLinearFabricUnicastNocScatterWriteMultiDir) {
    FabricUnicastCommon(
        this,
        NOC_UNICAST_SCATTER_WRITE,
        {std::make_tuple(RoutingDirection::E, 1), std::make_tuple(RoutingDirection::W, 1)});
}
TEST_F(NightlyFabric1DFixture, TestLinearFabricUnicastNocScatterWriteWithState) {
    FabricUnicastCommon(
        this,
        NOC_UNICAST_SCATTER_WRITE,
        {std::make_tuple(RoutingDirection::E, 1), std::make_tuple(RoutingDirection::W, 1)},
        FabricApiType::Linear,
        true);
}

TEST_F(NightlyFabric1DFixture, TestLinearFabricUnicastNocInlineWrite) {
    FabricUnicastCommon(this, NOC_UNICAST_INLINE_WRITE, {std::make_tuple(RoutingDirection::E, 1)});
}
TEST_F(NightlyFabric1DFixture, TestLinearFabricUnicastNocInlineWriteMultiDir) {
    FabricUnicastCommon(
        this,
        NOC_UNICAST_INLINE_WRITE,
        {std::make_tuple(RoutingDirection::E, 1), std::make_tuple(RoutingDirection::W, 1)});
}
TEST_F(NightlyFabric1DFixture, TestLinearFabricUnicastNocInlineWriteWithState) {
    FabricUnicastCommon(
        this,
        NOC_UNICAST_INLINE_WRITE,
        {std::make_tuple(RoutingDirection::E, 1), std::make_tuple(RoutingDirection::W, 1)},
        FabricApiType::Linear,
        true);
}

TEST_F(NightlyFabric1DFixture, TestLinearFabricUnicastNocFusedAtomicInc) {
    FabricUnicastCommon(this, NOC_FUSED_UNICAST_ATOMIC_INC, {std::make_tuple(RoutingDirection::E, 1)});
}
TEST_F(NightlyFabric1DFixture, TestLinearFabricUnicastNocFusedAtomicIncMultiDir) {
    FabricUnicastCommon(
        this,
        NOC_FUSED_UNICAST_ATOMIC_INC,
        {std::make_tuple(RoutingDirection::E, 1), std::make_tuple(RoutingDirection::W, 1)});
}
TEST_F(NightlyFabric1DFixture, TestLinearFabricUnicastNocFusedAtomicIncWithState) {
    FabricUnicastCommon(
        this,
        NOC_FUSED_UNICAST_ATOMIC_INC,
        {std::make_tuple(RoutingDirection::E, 1), std::make_tuple(RoutingDirection::W, 1)},
        FabricApiType::Linear,
        true);
}

TEST_F(NightlyFabric1DFixture, TestLinearFabricMulticastNocUnicastWrite) {
    FabricMulticastCommon(this, NOC_UNICAST_WRITE, {std::make_tuple(RoutingDirection::E, 1, 2)});
}
TEST_F(NightlyFabric1DFixture, TestLinearFabricMulticastNocUnicastWriteMultiDir) {
    FabricMulticastCommon(
        this,
        NOC_UNICAST_WRITE,
        {std::make_tuple(RoutingDirection::E, 1, 2), std::make_tuple(RoutingDirection::W, 1, 1)});
}
TEST_F(NightlyFabric1DFixture, TestLinearFabricMulticastNocUnicastWriteWithState) {
    FabricMulticastCommon(
        this,
        NOC_UNICAST_WRITE,
        {std::make_tuple(RoutingDirection::E, 1, 2), std::make_tuple(RoutingDirection::W, 1, 1)},
        true);
}

TEST_F(NightlyFabric1DFixture, TestLinearFabricMulticastNocAtomicInc) {
    FabricMulticastCommon(this, NOC_UNICAST_ATOMIC_INC, {std::make_tuple(RoutingDirection::E, 1, 2)});
}
TEST_F(NightlyFabric1DFixture, TestLinearFabricMulticastNocAtomicIncMultiDir) {
    FabricMulticastCommon(
        this,
        NOC_UNICAST_ATOMIC_INC,
        {std::make_tuple(RoutingDirection::E, 1, 2), std::make_tuple(RoutingDirection::W, 1, 1)});
}
TEST_F(NightlyFabric1DFixture, TestLinearFabricMulticastNocAtomicIncWithState) {
    FabricMulticastCommon(
        this,
        NOC_UNICAST_ATOMIC_INC,
        {std::make_tuple(RoutingDirection::E, 1, 2), std::make_tuple(RoutingDirection::W, 1, 1)},
        true);
}

TEST_F(NightlyFabric1DFixture, TestLinearFabricMulticastNocScatterWrite) {
    FabricMulticastCommon(this, NOC_UNICAST_SCATTER_WRITE, {std::make_tuple(RoutingDirection::E, 1, 2)});
}
TEST_F(NightlyFabric1DFixture, TestLinearFabricMulticastNocScatterWriteMultiDir) {
    FabricMulticastCommon(
        this,
        NOC_UNICAST_SCATTER_WRITE,
        {std::make_tuple(RoutingDirection::E, 1, 2), std::make_tuple(RoutingDirection::W, 1, 1)});
}
TEST_F(NightlyFabric1DFixture, TestLinearFabricMulticastNocScatterWriteWithState) {
    FabricMulticastCommon(
        this,
        NOC_UNICAST_SCATTER_WRITE,
        {std::make_tuple(RoutingDirection::E, 1, 2), std::make_tuple(RoutingDirection::W, 1, 1)},
        true);
}

TEST_F(NightlyFabric1DFixture, TestLinearFabricMulticastNocInlineWrite) {
    FabricMulticastCommon(this, NOC_UNICAST_INLINE_WRITE, {std::make_tuple(RoutingDirection::E, 1, 2)});
}
TEST_F(NightlyFabric1DFixture, TestLinearFabricMulticastNocInlineWriteMultiDir) {
    FabricMulticastCommon(
        this,
        NOC_UNICAST_INLINE_WRITE,
        {std::make_tuple(RoutingDirection::E, 1, 2), std::make_tuple(RoutingDirection::W, 1, 1)});
}
TEST_F(NightlyFabric1DFixture, TestLinearFabricMulticastNocInlineWriteWithState) {
    FabricMulticastCommon(
        this,
        NOC_UNICAST_INLINE_WRITE,
        {std::make_tuple(RoutingDirection::E, 1, 2), std::make_tuple(RoutingDirection::W, 1, 1)},
        true);
}

TEST_F(NightlyFabric1DFixture, TestLinearFabricMulticastNocFusedAtomicInc) {
    FabricMulticastCommon(this, NOC_FUSED_UNICAST_ATOMIC_INC, {std::make_tuple(RoutingDirection::E, 1, 2)});
}
TEST_F(NightlyFabric1DFixture, TestLinearFabricMulticastNocFusedAtomicIncMultiDir) {
    FabricMulticastCommon(
        this,
        NOC_FUSED_UNICAST_ATOMIC_INC,
        {std::make_tuple(RoutingDirection::E, 1, 2), std::make_tuple(RoutingDirection::W, 1, 1)});
}
TEST_F(NightlyFabric1DFixture, TestLinearFabricMulticastNocFusedAtomicIncWithState) {
    FabricMulticastCommon(
        this,
        NOC_FUSED_UNICAST_ATOMIC_INC,
        {std::make_tuple(RoutingDirection::E, 1, 2), std::make_tuple(RoutingDirection::W, 1, 1)},
        true);
}

// Test cases using the new Fabric1DTensixFixture to test tensix config with mux
TEST_F(Fabric1DTensixFixture, TestLinearFabricUnicastNocUnicastWriteMux) {
    FabricUnicastCommon(this, NOC_UNICAST_WRITE, {std::make_tuple(RoutingDirection::E, 1)});
}

TEST_F(Fabric1DTensixFixture, TestLinearFabricUnicastNocAtomicIncMux) {
    FabricUnicastCommon(this, NOC_UNICAST_ATOMIC_INC, {std::make_tuple(RoutingDirection::E, 1)});
}

TEST_F(Fabric1DTensixFixture, TestLinearFabricMulticastNocUnicastWriteMux) {
    FabricMulticastCommon(this, NOC_UNICAST_WRITE, {std::make_tuple(RoutingDirection::E, 1, 2)});
}

TEST_F(Fabric1DTensixFixture, TestLinearFabricMulticastNocAtomicIncMux) {
    FabricMulticastCommon(this, NOC_UNICAST_ATOMIC_INC, {std::make_tuple(RoutingDirection::E, 1, 2)});
}

}  // namespace tt::tt_fabric::fabric_router_tests
