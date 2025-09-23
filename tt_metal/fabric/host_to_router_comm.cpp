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

std::vector<uint32_t> read_core_helper(const FabricNodeId& node_id, chan_id_t eth_chan_id, uint32_t address) {
    const auto& control_plane = tt::tt_metal::MetalContext::instance().get_control_plane();
    const auto physical_chip_id = control_plane.get_physical_chip_id_from_fabric_node_id(node_id);

    const auto& cluster = tt::tt_metal::MetalContext::instance().get_cluster();
    const auto virtual_eth_core = cluster.get_virtual_eth_core_from_channel(physical_chip_id, eth_chan_id);
    return cluster.read_core(physical_chip_id, virtual_eth_core, address, sizeof(uint32_t));
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
    num_buffer_slots_ = fabric_context.get_control_channel_num_buffer_slots();
    remote_buffer_address_ = fabric_context.get_control_channel_buffer_base_address();
    remote_read_counter_address_ = fabric_context.get_control_channel_remote_read_counter_address();
    remote_write_counter_address_ = fabric_context.get_control_channel_remote_write_counter_address();
}

bool HostToRouterCommInterface::write_packet_to_router(
    ControlPacketHeader& packet, FabricNodeId& node_id, chan_id_t eth_chan_id) {
    uint32_t remote_read_counter = this->get_remote_read_counter(node_id, eth_chan_id);

    const auto& control_plane = tt::tt_metal::MetalContext::instance().get_control_plane();
    auto& router_comm_context = control_plane.get_fabric_context().get_router_comm_context(node_id, eth_chan_id);
    auto& local_write_counter = router_comm_context.get_local_write_counter();

    if (!has_space_for_packet(local_write_counter, remote_read_counter)) {
        return false;
    }

    const auto next_buffer_slot_address = this->get_next_buffer_address(local_write_counter);
    write_core_helper(node_id, eth_chan_id, next_buffer_slot_address, &packet, sizeof(ControlPacketHeader));

    local_write_counter.increment();

    // update router's remote write counter
    this->update_remote_write_counter(node_id, eth_chan_id, local_write_counter.get_counter());

    return true;
}

uint32_t HostToRouterCommInterface::get_remote_read_counter(FabricNodeId& node_id, chan_id_t eth_chan_id) const {
    return read_core_helper(node_id, eth_chan_id, remote_read_counter_address_)[0];
}

bool HostToRouterCommInterface::has_space_for_packet(
    HostChannelCounter& local_write_counter, uint32_t remote_read_counter) const {
    return local_write_counter.get_counter() - remote_read_counter < num_buffer_slots_;
}

uint32_t HostToRouterCommInterface::get_next_buffer_address(HostChannelCounter& local_write_counter) const {
    return remote_buffer_address_ + (local_write_counter.get_buffer_index() * sizeof(ControlPacketHeader));
}

void HostToRouterCommInterface::update_remote_write_counter(
    FabricNodeId& node_id, chan_id_t eth_chan_id, uint32_t local_write_counter) const {
    write_core_helper(node_id, eth_chan_id, remote_write_counter_address_, &local_write_counter, sizeof(uint32_t));
}

}  // namespace tt::tt_fabric
