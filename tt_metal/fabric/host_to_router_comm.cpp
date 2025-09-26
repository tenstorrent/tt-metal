// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <tt-metalium/control_plane.hpp>
#include "impl/context/metal_context.hpp"
#include "host_to_router_comm.hpp"
#include "host_to_router_comm_helpers.hpp"
#include <hostdevcommon/fabric_common.h>
#include "fabric_context.hpp"
#include <cstdint>
#include <cstddef>

namespace tt::tt_fabric {

namespace {

void read_core_helper(
    const FabricNodeId& node_id, chan_id_t eth_chan_id, uint32_t address, void* out_ptr, size_t size) {
    const auto& control_plane = tt::tt_metal::MetalContext::instance().get_control_plane();
    const auto physical_chip_id = control_plane.get_physical_chip_id_from_fabric_node_id(node_id);

    const auto& cluster = tt::tt_metal::MetalContext::instance().get_cluster();
    const auto virtual_eth_core = cluster.get_virtual_eth_core_from_channel(physical_chip_id, eth_chan_id);
    cluster.read_core(out_ptr, size, tt_cxy_pair(physical_chip_id, virtual_eth_core), address);
}

void write_core_helper(
    const FabricNodeId& node_id, chan_id_t eth_chan_id, uint32_t address, const void* data, size_t size) {
    const auto& control_plane = tt::tt_metal::MetalContext::instance().get_control_plane();
    const auto physical_chip_id = control_plane.get_physical_chip_id_from_fabric_node_id(node_id);

    const auto& cluster = tt::tt_metal::MetalContext::instance().get_cluster();
    const auto virtual_eth_core = cluster.get_virtual_eth_core_from_channel(physical_chip_id, eth_chan_id);
    cluster.write_core(data, size, tt_cxy_pair(physical_chip_id, virtual_eth_core), address);
}

}  // namespace

HostToRouterCommInterface::HostToRouterCommInterface() {
    const auto& fabric_context = tt::tt_metal::MetalContext::instance().get_control_plane().get_fabric_context();
    host_to_router_comm_config_ptr_ = fabric_context.get_host_to_router_comm_config();
}

bool HostToRouterCommInterface::write_packet_to_router(
    ControlPacketHeader& packet, FabricNodeId& node_id, chan_id_t eth_chan_id) {
    const auto& control_plane = tt::tt_metal::MetalContext::instance().get_control_plane();
    auto& router_comm_context = control_plane.get_fabric_context().get_router_comm_context(node_id, eth_chan_id);
    auto& local_write_counter = router_comm_context.get_local_write_counter();

    if (!has_space_for_packet(node_id, eth_chan_id, local_write_counter)) {
        return false;
    }

    const auto next_buffer_slot_address = this->get_next_buffer_address(local_write_counter);
    write_core_helper(node_id, eth_chan_id, next_buffer_slot_address, &packet, sizeof(ControlPacketHeader));

    local_write_counter.increment();

    // update router's remote write counter
    this->update_router_write_counter(node_id, eth_chan_id, local_write_counter.get_counter());

    return true;
}

CommonFSMLog HostToRouterCommInterface::get_common_fsm_log(FabricNodeId& node_id, chan_id_t eth_chan_id) const {
    CommonFSMLog common_fsm_log;
    read_core_helper(
        node_id,
        eth_chan_id,
        host_to_router_comm_config_ptr_->get_common_fsm_log_address(),
        &common_fsm_log,
        sizeof(CommonFSMLog));
    return common_fsm_log;
}

uint32_t HostToRouterCommInterface::get_router_read_counter(FabricNodeId& node_id, chan_id_t eth_chan_id) const {
    uint32_t read_counter;
    read_core_helper(
        node_id,
        eth_chan_id,
        host_to_router_comm_config_ptr_->get_router_read_counter_address(),
        &read_counter,
        sizeof(uint32_t));
    return read_counter;
}

bool HostToRouterCommInterface::has_space_for_packet(
    FabricNodeId& node_id, chan_id_t eth_chan_id, HostChannelCounter& local_write_counter) const {
    uint32_t remote_read_counter = this->get_router_read_counter(node_id, eth_chan_id);
    return local_write_counter.get_counter() - remote_read_counter <
           host_to_router_comm_config_ptr_->get_num_buffer_slots();
}

uint32_t HostToRouterCommInterface::get_next_buffer_address(HostChannelCounter& local_write_counter) const {
    return host_to_router_comm_config_ptr_->get_buffer_base_address() +
           (local_write_counter.get_buffer_index() * sizeof(ControlPacketHeader));
}

void HostToRouterCommInterface::update_router_write_counter(
    FabricNodeId& node_id, chan_id_t eth_chan_id, uint32_t local_write_counter) const {
    write_core_helper(
        node_id,
        eth_chan_id,
        host_to_router_comm_config_ptr_->get_router_write_counter_address(),
        &local_write_counter,
        sizeof(uint32_t));
}

}  // namespace tt::tt_fabric
