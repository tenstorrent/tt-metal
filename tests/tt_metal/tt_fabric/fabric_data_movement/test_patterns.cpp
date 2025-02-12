// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <memory>
#include "fabric_fixture.hpp"
#include "tt_fabric/control_plane.hpp"
#include "tt_fabric/mesh_graph.hpp"
#include "tt_fabric/routing_table_generator.hpp"

namespace tt::tt_fabric {

TEST_F(FabricFixture, TestAsyncWriteRoutingPlane) {
    CoreCoord sender_logical_core = {0, 0};
    CoreCoord receiver_logical_core = {1, 0};
    std::pair<mesh_id_t, chip_id_t> start_mesh_chip_id;
    chip_id_t physical_start_device_id;
    std::pair<mesh_id_t, chip_id_t> end_mesh_chip_id;
    chip_id_t physical_end_device_id;
    bool connection_found = false;
    for (auto* device : devices_) {
        start_mesh_chip_id = control_plane_->get_mesh_chip_id_from_physical_chip_id(device->id());
        auto neighbors = control_plane_->get_intra_chip_neighbors(
            start_mesh_chip_id.first, start_mesh_chip_id.second, RoutingDirection::E);
        if (neighbors.size() > 0) {
            physical_start_device_id = device->id();
            end_mesh_chip_id = {start_mesh_chip_id.first, neighbors[0]};
            physical_end_device_id = control_plane_->get_physical_chip_id_from_mesh_chip_id(end_mesh_chip_id);
            connection_found = true;
            break;
        }
    }
    if (!connection_found) {
        GTEST_SKIP() << "No path found between sender and receivers";
    }
    auto* sender_device = DevicePool::instance().get_active_device(physical_start_device_id);
    auto* receiver_device = DevicePool::instance().get_active_device(physical_end_device_id);
    CoreCoord sender_virtual_core = sender_device->worker_core_from_logical_core(sender_logical_core);
    CoreCoord receiver_virtual_core = receiver_device->worker_core_from_logical_core(receiver_logical_core);

    uint32_t l1_alignment = hal.get_alignment(HalMemType::L1);

    uint32_t worker_unreserved_base_addr =
        hal.get_dev_addr(HalProgrammableCoreType::TENSIX, HalL1MemAddrType::UNRESERVED);
    uint32_t client_interface_addr = worker_unreserved_base_addr;
    uint32_t packet_header_addr = tt::round_up(
        client_interface_addr + sizeof(fabric_client_interface_t) + 4 * sizeof(fabric_router_l1_config_t),
        l1_alignment);
    uint32_t buffer_data_addr = packet_header_addr + PACKET_HEADER_SIZE_BYTES;
    uint32_t buffer_data_size = tt::constants::TILE_HW * sizeof(uint32_t);
    std::vector<uint32_t> buffer_data(buffer_data_size / sizeof(uint32_t), 0);
    tt::llrt::write_hex_vec_to_core(physical_end_device_id, receiver_virtual_core, buffer_data, buffer_data_addr);

    std::iota(buffer_data.begin(), buffer_data.end(), 0);
    tt::llrt::write_hex_vec_to_core(physical_start_device_id, sender_virtual_core, buffer_data, buffer_data_addr);

    tt::Cluster::instance().l1_barrier(physical_end_device_id);
    tt::Cluster::instance().l1_barrier(physical_start_device_id);

    auto receiver_noc_encoding = tt::tt_metal::hal.noc_xy_encoding(receiver_virtual_core.x, receiver_virtual_core.y);

    auto sender_program = tt_metal::CreateProgram();
    auto sender_kernel = tt_metal::CreateKernel(
        sender_program,
        "tests/tt_metal/tt_fabric/fabric_data_movement/kernels/fabric_async_write_routing_plane_sender.cpp",
        {sender_logical_core},
        tt_metal::DataMovementConfig{
            .processor = tt_metal::DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default});

    auto [sender_gk_noc_offset, sender_gk_interface_addr] =
        this->GetFabricData().get_gatekeeper_noc_addr(physical_start_device_id);

    uint32_t routing_plane = 0;
    std::vector<uint32_t> sender_runtime_args = {
        client_interface_addr,
        sender_gk_interface_addr,
        sender_gk_noc_offset,
        packet_header_addr,
        receiver_noc_encoding,
        buffer_data_addr,
        buffer_data_size,
        end_mesh_chip_id.first,
        end_mesh_chip_id.second,
        routing_plane};
    tt_metal::SetRuntimeArgs(sender_program, sender_kernel, sender_logical_core, sender_runtime_args);

    auto receiver_program = tt_metal::CreateProgram();
    auto receiver_kernel = tt_metal::CreateKernel(
        receiver_program,
        "tests/tt_metal/tt_fabric/fabric_data_movement/kernels/fabric_receiver.cpp",
        {receiver_logical_core},
        tt_metal::DataMovementConfig{
            .processor = tt_metal::DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default});

    auto [receiver_gk_noc_offset, receiver_gk_interface_addr] =
        this->GetFabricData().get_gatekeeper_noc_addr(physical_end_device_id);
    std::vector<uint32_t> receiver_runtime_args = {
        buffer_data_addr,
        buffer_data_size,
    };
    tt_metal::SetRuntimeArgs(receiver_program, receiver_kernel, receiver_logical_core, receiver_runtime_args);

    tt_metal::detail::LaunchProgram(receiver_device, receiver_program, false);
    tt_metal::detail::LaunchProgram(sender_device, sender_program, false);
    tt_metal::detail::WaitProgramDone(sender_device, sender_program);
    tt_metal::detail::WaitProgramDone(receiver_device, receiver_program);

    std::vector<uint32_t> received_buffer_data = tt::llrt::read_hex_vec_from_core(
        physical_end_device_id, receiver_virtual_core, buffer_data_addr, buffer_data_size);
    EXPECT_EQ(buffer_data, received_buffer_data);
}

TEST_F(FabricFixture, TestAsyncWrite) {
    CoreCoord sender_logical_core = {0, 0};
    CoreCoord receiver_logical_core = {1, 0};
    std::pair<mesh_id_t, chip_id_t> start_mesh_chip_id;
    chip_id_t physical_start_device_id;
    std::pair<mesh_id_t, chip_id_t> end_mesh_chip_id;
    chip_id_t physical_end_device_id;
    bool connection_found = false;
    for (auto* device : devices_) {
        start_mesh_chip_id = control_plane_->get_mesh_chip_id_from_physical_chip_id(device->id());
        auto neighbors = control_plane_->get_intra_chip_neighbors(
            start_mesh_chip_id.first, start_mesh_chip_id.second, RoutingDirection::E);
        if (neighbors.size() > 0) {
            physical_start_device_id = device->id();
            end_mesh_chip_id = {start_mesh_chip_id.first, neighbors[0]};
            physical_end_device_id = control_plane_->get_physical_chip_id_from_mesh_chip_id(end_mesh_chip_id);
            connection_found = true;
            break;
        }
    }
    auto routers = control_plane_->get_routers_to_chip(
        start_mesh_chip_id.first, start_mesh_chip_id.second, end_mesh_chip_id.first, end_mesh_chip_id.second);

    if (routers.empty()) {
        GTEST_SKIP() << "No path found between sender and receivers";
    }
    auto* sender_device = DevicePool::instance().get_active_device(physical_start_device_id);
    auto* receiver_device = DevicePool::instance().get_active_device(physical_end_device_id);
    CoreCoord sender_virtual_core = sender_device->worker_core_from_logical_core(sender_logical_core);
    CoreCoord receiver_virtual_core = receiver_device->worker_core_from_logical_core(receiver_logical_core);

    uint32_t l1_alignment = hal.get_alignment(HalMemType::L1);

    uint32_t worker_unreserved_base_addr =
        hal.get_dev_addr(HalProgrammableCoreType::TENSIX, HalL1MemAddrType::UNRESERVED);
    uint32_t client_interface_addr = worker_unreserved_base_addr;
    uint32_t packet_header_addr = tt::round_up(client_interface_addr + sizeof(fabric_client_interface_t), l1_alignment);
    uint32_t buffer_data_addr = packet_header_addr + PACKET_HEADER_SIZE_BYTES;
    uint32_t buffer_data_size = tt::constants::TILE_HW * sizeof(uint32_t);
    std::vector<uint32_t> buffer_data(buffer_data_size / sizeof(uint32_t), 0);
    tt::llrt::write_hex_vec_to_core(physical_end_device_id, receiver_virtual_core, buffer_data, buffer_data_addr);

    std::iota(buffer_data.begin(), buffer_data.end(), 0);
    tt::llrt::write_hex_vec_to_core(physical_start_device_id, sender_virtual_core, buffer_data, buffer_data_addr);

    tt::Cluster::instance().l1_barrier(physical_end_device_id);
    tt::Cluster::instance().l1_barrier(physical_start_device_id);

    auto receiver_noc_encoding = tt::tt_metal::hal.noc_xy_encoding(receiver_virtual_core.x, receiver_virtual_core.y);

    auto sender_program = tt_metal::CreateProgram();
    auto sender_kernel = tt_metal::CreateKernel(
        sender_program,
        "tests/tt_metal/tt_fabric/fabric_data_movement/kernels/fabric_async_write_sender.cpp",
        {sender_logical_core},
        tt_metal::DataMovementConfig{
            .processor = tt_metal::DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default});

    auto [sender_gk_noc_offset, sender_gk_interface_addr] =
        this->GetFabricData().get_gatekeeper_noc_addr(physical_start_device_id);

    auto& sender_virtual_router_coord = routers[0].second;
    auto sender_router_noc_xy =
        tt_metal::hal.noc_xy_encoding(sender_virtual_router_coord.x, sender_virtual_router_coord.y);
    std::vector<uint32_t> sender_runtime_args = {
        client_interface_addr,
        sender_gk_interface_addr,
        sender_gk_noc_offset,
        packet_header_addr,
        receiver_noc_encoding,
        buffer_data_addr,
        buffer_data_size,
        end_mesh_chip_id.first,
        end_mesh_chip_id.second,
        sender_router_noc_xy};
    tt_metal::SetRuntimeArgs(sender_program, sender_kernel, sender_logical_core, sender_runtime_args);

    auto receiver_program = tt_metal::CreateProgram();
    auto receiver_kernel = tt_metal::CreateKernel(
        receiver_program,
        "tests/tt_metal/tt_fabric/fabric_data_movement/kernels/fabric_receiver.cpp",
        {receiver_logical_core},
        tt_metal::DataMovementConfig{
            .processor = tt_metal::DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default});

    auto [receiver_gk_noc_offset, receiver_gk_interface_addr] =
        this->GetFabricData().get_gatekeeper_noc_addr(physical_end_device_id);
    std::vector<uint32_t> receiver_runtime_args = {
        buffer_data_addr,
        buffer_data_size,
    };
    tt_metal::SetRuntimeArgs(receiver_program, receiver_kernel, receiver_logical_core, receiver_runtime_args);

    tt_metal::detail::LaunchProgram(receiver_device, receiver_program, false);
    tt_metal::detail::LaunchProgram(sender_device, sender_program, false);
    tt_metal::detail::WaitProgramDone(sender_device, sender_program);
    tt_metal::detail::WaitProgramDone(receiver_device, receiver_program);

    std::vector<uint32_t> received_buffer_data = tt::llrt::read_hex_vec_from_core(
        physical_end_device_id, receiver_virtual_core, buffer_data_addr, buffer_data_size);
    EXPECT_EQ(buffer_data, received_buffer_data);
}

TEST_F(FabricFixture, TestAtomicInc) {
    CoreCoord sender_logical_core = {0, 0};
    CoreCoord receiver_logical_core = {1, 0};
    std::pair<mesh_id_t, chip_id_t> start_mesh_chip_id;
    chip_id_t physical_start_device_id;
    std::pair<mesh_id_t, chip_id_t> end_mesh_chip_id;
    chip_id_t physical_end_device_id;
    bool connection_found = false;
    for (auto* device : devices_) {
        start_mesh_chip_id = control_plane_->get_mesh_chip_id_from_physical_chip_id(device->id());
        auto neighbors = control_plane_->get_intra_chip_neighbors(
            start_mesh_chip_id.first, start_mesh_chip_id.second, RoutingDirection::E);
        if (neighbors.size() > 0) {
            physical_start_device_id = device->id();
            end_mesh_chip_id = {start_mesh_chip_id.first, neighbors[0]};
            physical_end_device_id = control_plane_->get_physical_chip_id_from_mesh_chip_id(end_mesh_chip_id);
            connection_found = true;
            break;
        }
    }
    auto routers = control_plane_->get_routers_to_chip(
        start_mesh_chip_id.first, start_mesh_chip_id.second, end_mesh_chip_id.first, end_mesh_chip_id.second);

    if (routers.empty()) {
        GTEST_SKIP() << "No path found between sender and receivers";
    }
    auto* sender_device = DevicePool::instance().get_active_device(physical_start_device_id);
    auto* receiver_device = DevicePool::instance().get_active_device(physical_end_device_id);
    CoreCoord sender_virtual_core = sender_device->worker_core_from_logical_core(sender_logical_core);
    CoreCoord receiver_virtual_core = receiver_device->worker_core_from_logical_core(receiver_logical_core);

    uint32_t l1_alignment = hal.get_alignment(HalMemType::L1);

    uint32_t worker_unreserved_base_addr =
        hal.get_dev_addr(HalProgrammableCoreType::TENSIX, HalL1MemAddrType::UNRESERVED);
    uint32_t client_interface_addr = worker_unreserved_base_addr;
    uint32_t packet_header_addr = tt::round_up(client_interface_addr + sizeof(fabric_client_interface_t), l1_alignment);
    uint32_t atomic_inc_addr = packet_header_addr + PACKET_HEADER_SIZE_BYTES;
    uint32_t atomic_inc_size = sizeof(uint32_t);
    std::vector<uint32_t> atomic_inc_data(atomic_inc_size / sizeof(uint32_t), 0);
    tt::llrt::write_hex_vec_to_core(physical_end_device_id, receiver_virtual_core, atomic_inc_data, atomic_inc_addr);

    uint32_t atomic_inc = 5;
    uint32_t wrap_boundary = 31;
    tt::Cluster::instance().l1_barrier(physical_end_device_id);
    tt::Cluster::instance().l1_barrier(physical_start_device_id);

    auto receiver_noc_encoding = tt::tt_metal::hal.noc_xy_encoding(receiver_virtual_core.x, receiver_virtual_core.y);

    auto sender_program = tt_metal::CreateProgram();
    auto sender_kernel = tt_metal::CreateKernel(
        sender_program,
        "tests/tt_metal/tt_fabric/fabric_data_movement/kernels/fabric_atomic_inc_sender.cpp",
        {sender_logical_core},
        tt_metal::DataMovementConfig{
            .processor = tt_metal::DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default});

    auto [sender_gk_noc_offset, sender_gk_interface_addr] =
        this->GetFabricData().get_gatekeeper_noc_addr(physical_start_device_id);

    auto& sender_virtual_router_coord = routers[0].second;
    auto sender_router_noc_xy =
        tt_metal::hal.noc_xy_encoding(sender_virtual_router_coord.x, sender_virtual_router_coord.y);
    std::vector<uint32_t> sender_runtime_args = {
        client_interface_addr,
        sender_gk_interface_addr,
        sender_gk_noc_offset,
        packet_header_addr,
        receiver_noc_encoding,
        atomic_inc_addr,
        atomic_inc,
        wrap_boundary,
        end_mesh_chip_id.first,
        end_mesh_chip_id.second,
        sender_router_noc_xy};
    tt_metal::SetRuntimeArgs(sender_program, sender_kernel, sender_logical_core, sender_runtime_args);

    auto receiver_program = tt_metal::CreateProgram();
    auto receiver_kernel = tt_metal::CreateKernel(
        receiver_program,
        "tests/tt_metal/tt_fabric/fabric_data_movement/kernels/fabric_receiver.cpp",
        {receiver_logical_core},
        tt_metal::DataMovementConfig{
            .processor = tt_metal::DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default});

    auto [receiver_gk_noc_offset, receiver_gk_interface_addr] =
        this->GetFabricData().get_gatekeeper_noc_addr(physical_end_device_id);
    std::vector<uint32_t> receiver_runtime_args = {
        atomic_inc_addr,
        sizeof(uint32_t),
    };
    tt_metal::SetRuntimeArgs(receiver_program, receiver_kernel, receiver_logical_core, receiver_runtime_args);

    tt_metal::detail::LaunchProgram(receiver_device, receiver_program, false);
    tt_metal::detail::LaunchProgram(sender_device, sender_program, false);
    tt_metal::detail::WaitProgramDone(sender_device, sender_program);
    tt_metal::detail::WaitProgramDone(receiver_device, receiver_program);

    std::vector<uint32_t> received_buffer_data = tt::llrt::read_hex_vec_from_core(
        physical_end_device_id, receiver_virtual_core, atomic_inc_addr, atomic_inc_size);
    EXPECT_EQ(atomic_inc, received_buffer_data[0]);
}

TEST_F(FabricFixture, TestAyncWriteAtomicInc) {
    CoreCoord sender_logical_core = {0, 0};
    CoreCoord receiver_logical_core = {1, 0};
    std::pair<mesh_id_t, chip_id_t> start_mesh_chip_id;
    chip_id_t physical_start_device_id;
    std::pair<mesh_id_t, chip_id_t> end_mesh_chip_id;
    chip_id_t physical_end_device_id;
    bool connection_found = false;
    for (auto* device : devices_) {
        start_mesh_chip_id = control_plane_->get_mesh_chip_id_from_physical_chip_id(device->id());
        auto neighbors = control_plane_->get_intra_chip_neighbors(
            start_mesh_chip_id.first, start_mesh_chip_id.second, RoutingDirection::E);
        if (neighbors.size() > 0) {
            physical_start_device_id = device->id();
            end_mesh_chip_id = {start_mesh_chip_id.first, neighbors[0]};
            physical_end_device_id = control_plane_->get_physical_chip_id_from_mesh_chip_id(end_mesh_chip_id);
            connection_found = true;
            break;
        }
    }
    auto routers = control_plane_->get_routers_to_chip(
        start_mesh_chip_id.first, start_mesh_chip_id.second, end_mesh_chip_id.first, end_mesh_chip_id.second);

    if (routers.empty()) {
        GTEST_SKIP() << "No path found between sender and receivers";
    }
    auto* sender_device = DevicePool::instance().get_active_device(physical_start_device_id);
    auto* receiver_device = DevicePool::instance().get_active_device(physical_end_device_id);
    CoreCoord sender_virtual_core = sender_device->worker_core_from_logical_core(sender_logical_core);
    CoreCoord receiver_virtual_core = receiver_device->worker_core_from_logical_core(receiver_logical_core);

    uint32_t l1_alignment = hal.get_alignment(HalMemType::L1);

    uint32_t worker_unreserved_base_addr =
        hal.get_dev_addr(HalProgrammableCoreType::TENSIX, HalL1MemAddrType::UNRESERVED);
    uint32_t client_interface_addr = worker_unreserved_base_addr;
    uint32_t packet_header_addr = tt::round_up(client_interface_addr + sizeof(fabric_client_interface_t), l1_alignment);
    uint32_t buffer_data_addr = packet_header_addr + PACKET_HEADER_SIZE_BYTES;
    uint32_t buffer_data_size = constants::TILE_HW;
    uint32_t atomic_inc_addr = tt::round_up(buffer_data_addr + buffer_data_size, l1_alignment);
    uint32_t atomic_inc_size = sizeof(uint32_t);
    uint32_t atomic_inc = 5;
    std::vector<uint32_t> buffer_data(buffer_data_size / sizeof(uint32_t), 0);
    tt::llrt::write_hex_vec_to_core(physical_end_device_id, receiver_virtual_core, buffer_data, buffer_data_addr);
    std::vector<uint32_t> atomic_inc_data(atomic_inc_size / sizeof(uint32_t), 0);
    tt::llrt::write_hex_vec_to_core(physical_end_device_id, receiver_virtual_core, atomic_inc_data, atomic_inc_addr);

    uint32_t wrap_boundary = 31;
    std::iota(buffer_data.begin(), buffer_data.end(), 0);
    tt::llrt::write_hex_vec_to_core(physical_start_device_id, sender_virtual_core, buffer_data, buffer_data_addr);

    tt::Cluster::instance().l1_barrier(physical_end_device_id);
    tt::Cluster::instance().l1_barrier(physical_start_device_id);

    auto receiver_noc_encoding = tt::tt_metal::hal.noc_xy_encoding(receiver_virtual_core.x, receiver_virtual_core.y);

    auto sender_program = tt_metal::CreateProgram();
    auto sender_kernel = tt_metal::CreateKernel(
        sender_program,
        "tests/tt_metal/tt_fabric/fabric_data_movement/kernels/fabric_async_write_atomic_inc_sender.cpp",
        {sender_logical_core},
        tt_metal::DataMovementConfig{
            .processor = tt_metal::DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default});

    auto [sender_gk_noc_offset, sender_gk_interface_addr] =
        this->GetFabricData().get_gatekeeper_noc_addr(physical_start_device_id);

    auto& sender_virtual_router_coord = routers[0].second;
    auto sender_router_noc_xy =
        tt_metal::hal.noc_xy_encoding(sender_virtual_router_coord.x, sender_virtual_router_coord.y);
    std::vector<uint32_t> sender_runtime_args = {
        client_interface_addr,
        sender_gk_interface_addr,
        sender_gk_noc_offset,
        packet_header_addr,
        receiver_noc_encoding,
        buffer_data_addr,
        atomic_inc_addr,
        buffer_data_size,
        atomic_inc,
        end_mesh_chip_id.first,
        end_mesh_chip_id.second,
        sender_router_noc_xy};
    tt_metal::SetRuntimeArgs(sender_program, sender_kernel, sender_logical_core, sender_runtime_args);

    auto receiver_program = tt_metal::CreateProgram();
    auto receiver_kernel = tt_metal::CreateKernel(
        receiver_program,
        "tests/tt_metal/tt_fabric/fabric_data_movement/kernels/fabric_receiver.cpp",
        {receiver_logical_core},
        tt_metal::DataMovementConfig{
            .processor = tt_metal::DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default});

    auto [receiver_gk_noc_offset, receiver_gk_interface_addr] =
        this->GetFabricData().get_gatekeeper_noc_addr(physical_end_device_id);
    std::vector<uint32_t> receiver_runtime_args = {
        atomic_inc_addr,
        sizeof(uint32_t),
    };
    tt_metal::SetRuntimeArgs(receiver_program, receiver_kernel, receiver_logical_core, receiver_runtime_args);

    tt_metal::detail::LaunchProgram(receiver_device, receiver_program, false);
    tt_metal::detail::LaunchProgram(sender_device, sender_program, false);
    tt_metal::detail::WaitProgramDone(sender_device, sender_program);
    tt_metal::detail::WaitProgramDone(receiver_device, receiver_program);

    std::vector<uint32_t> received_buffer_data = tt::llrt::read_hex_vec_from_core(
        physical_end_device_id, receiver_virtual_core, buffer_data_addr, buffer_data_size);
    EXPECT_EQ(buffer_data, received_buffer_data);
    received_buffer_data.clear();
    received_buffer_data = tt::llrt::read_hex_vec_from_core(
        physical_end_device_id, receiver_virtual_core, atomic_inc_addr, atomic_inc_size);
    EXPECT_EQ(atomic_inc, received_buffer_data[0]);
}

TEST_F(FabricFixture, TestAsyncWriteMulticastMultidirectional) {
    CoreCoord sender_logical_core = {0, 0};
    CoreCoord receiver_logical_core = {1, 0};
    std::pair<mesh_id_t, chip_id_t> start_mesh_chip_id;
    chip_id_t physical_start_device_id;
    std::unordered_map<RoutingDirection, std::vector<std::pair<mesh_id_t, chip_id_t>>> end_mesh_chip_ids_by_dir;
    std::unordered_map<RoutingDirection, std::vector<chip_id_t>> physical_end_device_ids_by_dir;
    uint32_t num_dirs = 2;
    std::unordered_map<RoutingDirection, uint32_t> mcast_hops;
    mcast_hops[RoutingDirection::E] = 2;
    for (auto* device : devices_) {
        start_mesh_chip_id = control_plane_->get_mesh_chip_id_from_physical_chip_id(device->id());
        std::unordered_map<RoutingDirection, std::vector<std::pair<mesh_id_t, chip_id_t>>>
            temp_end_mesh_chip_ids_by_dir;
        std::unordered_map<RoutingDirection, std::vector<chip_id_t>> temp_physical_end_device_ids_by_dir;
        bool connection_found = true;
        for (auto [routing_direction, num_hops] : mcast_hops) {
            bool direction_found = true;
            auto& temp_end_mesh_chip_ids = temp_end_mesh_chip_ids_by_dir[routing_direction];
            auto& temp_physical_end_device_ids = temp_physical_end_device_ids_by_dir[routing_direction];
            uint32_t curr_mesh_id = start_mesh_chip_id.first;
            uint32_t curr_chip_id = start_mesh_chip_id.second;
            for (uint32_t i = 0; i < num_hops; i++) {
                auto neighbors =
                    control_plane_->get_intra_chip_neighbors(curr_mesh_id, curr_chip_id, routing_direction);
                if (neighbors.size() > 0) {
                    temp_end_mesh_chip_ids.emplace_back(curr_mesh_id, neighbors[0]);
                    temp_physical_end_device_ids.push_back(
                        control_plane_->get_physical_chip_id_from_mesh_chip_id(temp_end_mesh_chip_ids.back()));
                    curr_mesh_id = temp_end_mesh_chip_ids.back().first;
                    curr_chip_id = temp_end_mesh_chip_ids.back().second;
                } else {
                    direction_found = false;
                    break;
                }
            }
            if (!direction_found) {
                connection_found = false;
                break;
            }
        }
        if (connection_found) {
            physical_start_device_id = device->id();
            end_mesh_chip_ids_by_dir = std::move(temp_end_mesh_chip_ids_by_dir);
            physical_end_device_ids_by_dir = std::move(temp_physical_end_device_ids_by_dir);
            break;
        }
    }
    if (end_mesh_chip_ids_by_dir.empty()) {
        GTEST_SKIP() << "No path found between sender and receivers";
    }

    auto* sender_device = DevicePool::instance().get_active_device(physical_start_device_id);
    CoreCoord sender_virtual_core = sender_device->worker_core_from_logical_core(sender_logical_core);

    CoreCoord receiver_virtual_core = sender_device->worker_core_from_logical_core(receiver_logical_core);

    uint32_t l1_alignment = hal.get_alignment(HalMemType::L1);

    uint32_t worker_unreserved_base_addr =
        hal.get_dev_addr(HalProgrammableCoreType::TENSIX, HalL1MemAddrType::UNRESERVED);
    uint32_t client_interface_addr = worker_unreserved_base_addr;
    uint32_t packet_header_addr =
        tt::round_up(client_interface_addr + sizeof(fabric_client_interface_t) * num_dirs, l1_alignment);
    uint32_t buffer_data_addr = packet_header_addr + PACKET_HEADER_SIZE_BYTES;
    uint32_t buffer_data_size = tt::constants::TILE_HW * sizeof(uint32_t);
    std::vector<uint32_t> buffer_data(buffer_data_size / sizeof(uint32_t), 0);
    std::vector<tt_metal::Program> receiver_programs;
    for (auto& [routing_direction, physical_end_device_ids] : physical_end_device_ids_by_dir) {
        for (auto physical_end_device_id : physical_end_device_ids) {
            auto* receiver_device = DevicePool::instance().get_active_device(physical_end_device_id);
            tt::llrt::write_hex_vec_to_core(
                physical_end_device_id, receiver_virtual_core, buffer_data, buffer_data_addr);
            tt::Cluster::instance().l1_barrier(physical_end_device_id);
            auto receiver_program = tt_metal::CreateProgram();
            auto receiver_kernel = tt_metal::CreateKernel(
                receiver_program,
                "tests/tt_metal/tt_fabric/fabric_data_movement/kernels/fabric_receiver.cpp",
                {receiver_logical_core},
                tt_metal::DataMovementConfig{
                    .processor = tt_metal::DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default});

            auto [receiver_gk_noc_offset, receiver_gk_interface_addr] =
                this->GetFabricData().get_gatekeeper_noc_addr(physical_end_device_id);
            std::vector<uint32_t> receiver_runtime_args = {
                buffer_data_addr,
                buffer_data_size,
            };
            tt_metal::SetRuntimeArgs(receiver_program, receiver_kernel, receiver_logical_core, receiver_runtime_args);

            tt_metal::detail::LaunchProgram(receiver_device, receiver_program, false);
            receiver_programs.push_back(std::move(receiver_program));
        }
    }

    std::iota(buffer_data.begin(), buffer_data.end(), 0);
    tt::llrt::write_hex_vec_to_core(physical_start_device_id, sender_virtual_core, buffer_data, buffer_data_addr);

    tt::Cluster::instance().l1_barrier(physical_start_device_id);

    auto receiver_noc_encoding = tt::tt_metal::hal.noc_xy_encoding(receiver_virtual_core.x, receiver_virtual_core.y);

    auto sender_program = tt_metal::CreateProgram();
    auto sender_kernel = tt_metal::CreateKernel(
        sender_program,
        "tests/tt_metal/tt_fabric/fabric_data_movement/kernels/fabric_async_write_multicast_sender.cpp",
        {sender_logical_core},
        tt_metal::DataMovementConfig{
            .processor = tt_metal::DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default});

    auto [sender_gk_noc_offset, sender_gk_interface_addr] =
        this->GetFabricData().get_gatekeeper_noc_addr(physical_start_device_id);

    std::unordered_map<RoutingDirection, uint32_t> sender_router_noc_xys;
    for (auto& [routing_direction, end_mesh_chip_ids] : end_mesh_chip_ids_by_dir) {
        auto routers = control_plane_->get_routers_to_chip(
            start_mesh_chip_id.first,
            start_mesh_chip_id.second,
            end_mesh_chip_ids[0].first,
            end_mesh_chip_ids[0].second);
        auto& sender_virtual_router_coord = routers[0].second;
        sender_router_noc_xys.try_emplace(
            routing_direction,
            tt_metal::hal.noc_xy_encoding(sender_virtual_router_coord.x, sender_virtual_router_coord.y));
    }
    std::vector<uint32_t> sender_runtime_args = {
        client_interface_addr,
        sender_gk_interface_addr,
        sender_gk_noc_offset,
        packet_header_addr,
        receiver_noc_encoding,
        buffer_data_addr,
        buffer_data_size,
        end_mesh_chip_ids_by_dir[RoutingDirection::E][0].first,
        end_mesh_chip_ids_by_dir[RoutingDirection::E][0].second,
        mcast_hops[RoutingDirection::E],
        sender_router_noc_xys[RoutingDirection::E]};
    tt_metal::SetRuntimeArgs(sender_program, sender_kernel, sender_logical_core, sender_runtime_args);

    tt_metal::detail::LaunchProgram(sender_device, sender_program, false);
    tt_metal::detail::WaitProgramDone(sender_device, sender_program);
    for (auto [routing_direction, physical_end_device_ids] : physical_end_device_ids_by_dir) {
        for (uint32_t i = 0; i < physical_end_device_ids.size(); i++) {
            auto* receiver_device = DevicePool::instance().get_active_device(physical_end_device_ids[i]);
            tt_metal::detail::WaitProgramDone(receiver_device, receiver_programs[i]);
        }
    }

    for (auto [routing_direction, physical_end_device_ids] : physical_end_device_ids_by_dir) {
        for (auto physical_end_device_id : physical_end_device_ids) {
            std::vector<uint32_t> received_buffer_data = tt::llrt::read_hex_vec_from_core(
                physical_end_device_id, receiver_virtual_core, buffer_data_addr, buffer_data_size);
            EXPECT_EQ(buffer_data, received_buffer_data);
        }
    }
}

TEST_F(FabricFixture, TestAsyncWriteMulticast) {
    CoreCoord sender_logical_core = {0, 0};
    CoreCoord receiver_logical_core = {1, 0};
    std::pair<mesh_id_t, chip_id_t> start_mesh_chip_id;
    chip_id_t physical_start_device_id;
    std::unordered_map<RoutingDirection, std::vector<std::pair<mesh_id_t, chip_id_t>>> end_mesh_chip_ids_by_dir;
    std::unordered_map<RoutingDirection, std::vector<chip_id_t>> physical_end_device_ids_by_dir;
    uint32_t num_dirs = 2;
    std::unordered_map<RoutingDirection, uint32_t> mcast_hops;
    mcast_hops[RoutingDirection::E] = 2;
    mcast_hops[RoutingDirection::W] = 1;
    // mcast_hops[RoutingDirection::N] = 1;
    // mcast_hops[RoutingDirection::S] = 0;
    for (auto* device : devices_) {
        start_mesh_chip_id = control_plane_->get_mesh_chip_id_from_physical_chip_id(device->id());
        std::unordered_map<RoutingDirection, std::vector<std::pair<mesh_id_t, chip_id_t>>>
            temp_end_mesh_chip_ids_by_dir;
        std::unordered_map<RoutingDirection, std::vector<chip_id_t>> temp_physical_end_device_ids_by_dir;
        bool connection_found = true;
        for (auto [routing_direction, num_hops] : mcast_hops) {
            bool direction_found = true;
            auto& temp_end_mesh_chip_ids = temp_end_mesh_chip_ids_by_dir[routing_direction];
            auto& temp_physical_end_device_ids = temp_physical_end_device_ids_by_dir[routing_direction];
            uint32_t curr_mesh_id = start_mesh_chip_id.first;
            uint32_t curr_chip_id = start_mesh_chip_id.second;
            for (uint32_t i = 0; i < num_hops; i++) {
                auto neighbors =
                    control_plane_->get_intra_chip_neighbors(curr_mesh_id, curr_chip_id, routing_direction);
                if (neighbors.size() > 0) {
                    temp_end_mesh_chip_ids.emplace_back(curr_mesh_id, neighbors[0]);
                    temp_physical_end_device_ids.push_back(
                        control_plane_->get_physical_chip_id_from_mesh_chip_id(temp_end_mesh_chip_ids.back()));
                    curr_mesh_id = temp_end_mesh_chip_ids.back().first;
                    curr_chip_id = temp_end_mesh_chip_ids.back().second;
                } else {
                    direction_found = false;
                    break;
                }
            }
            if (!direction_found) {
                connection_found = false;
                break;
            }
        }
        if (connection_found) {
            physical_start_device_id = device->id();
            end_mesh_chip_ids_by_dir = std::move(temp_end_mesh_chip_ids_by_dir);
            physical_end_device_ids_by_dir = std::move(temp_physical_end_device_ids_by_dir);
            break;
        }
    }
    if (end_mesh_chip_ids_by_dir.empty()) {
        GTEST_SKIP() << "No path found between sender and receivers";
    }

    auto* sender_device = DevicePool::instance().get_active_device(physical_start_device_id);
    CoreCoord sender_virtual_core = sender_device->worker_core_from_logical_core(sender_logical_core);

    CoreCoord receiver_virtual_core = sender_device->worker_core_from_logical_core(receiver_logical_core);

    uint32_t l1_alignment = hal.get_alignment(HalMemType::L1);

    uint32_t worker_unreserved_base_addr =
        hal.get_dev_addr(HalProgrammableCoreType::TENSIX, HalL1MemAddrType::UNRESERVED);
    uint32_t client_interface_addr = worker_unreserved_base_addr;
    uint32_t packet_header_addr =
        tt::round_up(client_interface_addr + sizeof(fabric_client_interface_t) * num_dirs, l1_alignment);
    uint32_t buffer_data_addr = packet_header_addr + PACKET_HEADER_SIZE_BYTES;
    uint32_t buffer_data_size = tt::constants::TILE_HW * sizeof(uint32_t);
    std::vector<uint32_t> buffer_data(buffer_data_size / sizeof(uint32_t), 0);
    std::vector<tt_metal::Program> receiver_programs;
    for (auto& [routing_direction, physical_end_device_ids] : physical_end_device_ids_by_dir) {
        for (auto physical_end_device_id : physical_end_device_ids) {
            auto* receiver_device = DevicePool::instance().get_active_device(physical_end_device_id);
            tt::llrt::write_hex_vec_to_core(
                physical_end_device_id, receiver_virtual_core, buffer_data, buffer_data_addr);
            tt::Cluster::instance().l1_barrier(physical_end_device_id);
            auto receiver_program = tt_metal::CreateProgram();
            auto receiver_kernel = tt_metal::CreateKernel(
                receiver_program,
                "tests/tt_metal/tt_fabric/fabric_data_movement/kernels/fabric_receiver.cpp",
                {receiver_logical_core},
                tt_metal::DataMovementConfig{
                    .processor = tt_metal::DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default});

            auto [receiver_gk_noc_offset, receiver_gk_interface_addr] =
                this->GetFabricData().get_gatekeeper_noc_addr(physical_end_device_id);
            std::vector<uint32_t> receiver_runtime_args = {
                buffer_data_addr,
                buffer_data_size,
            };
            tt_metal::SetRuntimeArgs(receiver_program, receiver_kernel, receiver_logical_core, receiver_runtime_args);

            tt_metal::detail::LaunchProgram(receiver_device, receiver_program, false);
            receiver_programs.push_back(std::move(receiver_program));
        }
    }

    std::iota(buffer_data.begin(), buffer_data.end(), 0);
    tt::llrt::write_hex_vec_to_core(physical_start_device_id, sender_virtual_core, buffer_data, buffer_data_addr);

    tt::Cluster::instance().l1_barrier(physical_start_device_id);

    auto receiver_noc_encoding = tt::tt_metal::hal.noc_xy_encoding(receiver_virtual_core.x, receiver_virtual_core.y);

    auto sender_program = tt_metal::CreateProgram();
    auto sender_kernel = tt_metal::CreateKernel(
        sender_program,
        "tests/tt_metal/tt_fabric/fabric_data_movement/kernels/"
        "fabric_async_write_multicast_multidirectional_sender.cpp",
        {sender_logical_core},
        tt_metal::DataMovementConfig{
            .processor = tt_metal::DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default});

    auto [sender_gk_noc_offset, sender_gk_interface_addr] =
        this->GetFabricData().get_gatekeeper_noc_addr(physical_start_device_id);

    std::unordered_map<RoutingDirection, uint32_t> sender_router_noc_xys;
    for (auto& [routing_direction, end_mesh_chip_ids] : end_mesh_chip_ids_by_dir) {
        auto routers = control_plane_->get_routers_to_chip(
            start_mesh_chip_id.first,
            start_mesh_chip_id.second,
            end_mesh_chip_ids[0].first,
            end_mesh_chip_ids[0].second);
        auto& sender_virtual_router_coord = routers[0].second;
        sender_router_noc_xys.try_emplace(
            routing_direction,
            tt_metal::hal.noc_xy_encoding(sender_virtual_router_coord.x, sender_virtual_router_coord.y));
    }
    std::vector<uint32_t> sender_runtime_args = {
        client_interface_addr,
        sender_gk_interface_addr,
        sender_gk_noc_offset,
        packet_header_addr,
        receiver_noc_encoding,
        buffer_data_addr,
        buffer_data_size,
        end_mesh_chip_ids_by_dir[RoutingDirection::E][0].first,
        end_mesh_chip_ids_by_dir[RoutingDirection::E][0].second,
        mcast_hops[RoutingDirection::E],
        sender_router_noc_xys[RoutingDirection::E],
        end_mesh_chip_ids_by_dir[RoutingDirection::W][0].first,
        end_mesh_chip_ids_by_dir[RoutingDirection::W][0].second,
        mcast_hops[RoutingDirection::W],
        sender_router_noc_xys[RoutingDirection::W]
        // end_mesh_chip_ids_by_dir[RoutingDirection::N][0].first,
        // end_mesh_chip_ids_by_dir[RoutingDirection::N][0].second,
        // mcast_hops[RoutingDirection::N],
        // sender_router_noc_xys[RoutingDirection::N],
        // end_mesh_chip_ids_by_dir[RoutingDirection::S][0].first,
        // end_mesh_chip_ids_by_dir[RoutingDirection::S][0].second,
        // mcast_hops[RoutingDirection::S],
        // sender_router_noc_xys[RoutingDirection::S]
    };
    tt_metal::SetRuntimeArgs(sender_program, sender_kernel, sender_logical_core, sender_runtime_args);

    tt_metal::detail::LaunchProgram(sender_device, sender_program, false);
    tt_metal::detail::WaitProgramDone(sender_device, sender_program);
    for (auto [routing_direction, physical_end_device_ids] : physical_end_device_ids_by_dir) {
        for (uint32_t i = 0; i < physical_end_device_ids.size(); i++) {
            auto* receiver_device = DevicePool::instance().get_active_device(physical_end_device_ids[i]);
            tt_metal::detail::WaitProgramDone(receiver_device, receiver_programs[i]);
        }
    }

    for (auto [routing_direction, physical_end_device_ids] : physical_end_device_ids_by_dir) {
        for (auto physical_end_device_id : physical_end_device_ids) {
            std::vector<uint32_t> received_buffer_data = tt::llrt::read_hex_vec_from_core(
                physical_end_device_id, receiver_virtual_core, buffer_data_addr, buffer_data_size);
            EXPECT_EQ(buffer_data, received_buffer_data);
        }
    }
}

}  // namespace tt::tt_fabric
