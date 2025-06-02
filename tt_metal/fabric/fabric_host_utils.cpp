// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "control_plane.hpp"
#include "fabric_host_utils.hpp"

#include <tt-metalium/fabric_edm_types.hpp>
#include <tt-metalium/fabric_types.hpp>
#include <tt-metalium/assert.hpp>
#include <magic_enum/magic_enum.hpp>
#include <umd/device/types/cluster_descriptor_types.h>  // chip_id_t
#include <tt-metalium/metal_soc_descriptor.h>
#include "impl/context/metal_context.hpp"
#include <tt-metalium/erisc_datamover_builder.hpp>
#include <set>
#include <vector>
#include <algorithm>
#include "tt_metal/fabric/fabric_host_utils.hpp"
#include "fabric/hw/inc/fabric_routing_mode.h"
#include "fabric_context.hpp"

namespace tt::tt_fabric {

bool is_tt_fabric_config(tt::tt_metal::FabricConfig fabric_config) {
    return fabric_config == tt::tt_metal::FabricConfig::FABRIC_1D ||
           fabric_config == tt::tt_metal::FabricConfig::FABRIC_1D_RING ||
           fabric_config == tt::tt_metal::FabricConfig::FABRIC_2D ||
           fabric_config == tt::tt_metal::FabricConfig::FABRIC_2D_TORUS ||
           fabric_config == tt::tt_metal::FabricConfig::FABRIC_2D_DYNAMIC;
}

bool is_2d_fabric_config(tt::tt_metal::FabricConfig fabric_config) {
    return fabric_config == tt::tt_metal::FabricConfig::FABRIC_2D ||
           fabric_config == tt::tt_metal::FabricConfig::FABRIC_2D_TORUS ||
           fabric_config == tt::tt_metal::FabricConfig::FABRIC_2D_DYNAMIC;
}

uint32_t get_sender_channel_count(tt::tt_fabric::Topology topology) {
    if (topology == Topology::Mesh) {
        return FabricEriscDatamoverConfig::num_sender_channels_2d;
    } else {
        return FabricEriscDatamoverConfig::num_sender_channels_1d;
    }
}

uint32_t get_downstream_edm_count(tt::tt_fabric::Topology topology) {
    if (topology == Topology::Mesh) {
        return FabricEriscDatamoverConfig::num_downstream_edms_2d;
    } else {
        return FabricEriscDatamoverConfig::num_downstream_edms;
    }
}

FabricType get_fabric_type(tt::tt_metal::FabricConfig fabric_config, tt::ClusterType cluster_type) {
    if (cluster_type == tt::ClusterType::GALAXY && fabric_config == tt::tt_metal::FabricConfig::FABRIC_1D_RING) {
        return FabricType::TORUS_2D;
    }
    return FabricType::MESH;
}

std::vector<uint32_t> get_forwarding_link_indices_in_direction(
    chip_id_t src_chip_id, chip_id_t dst_chip_id, RoutingDirection direction) {
    const auto& control_plane= tt::tt_metal::MetalContext::instance().get_control_plane();
    const auto src_fabric_node_id = control_plane.get_fabric_node_id_from_physical_chip_id(src_chip_id);
    const auto dst_fabric_node_id = control_plane.get_fabric_node_id_from_physical_chip_id(dst_chip_id);
    const bool is_2d_fabric = control_plane.get_fabric_context().get_fabric_topology() == Topology::Mesh;

    const std::vector<chan_id_t>& fabric_channels =
        control_plane.get_active_fabric_eth_channels_in_direction(src_fabric_node_id, direction);

    // the subset of routers that support forwarding b/w those chips
    std::vector<chan_id_t> forwarding_channels;
    if (is_2d_fabric) {
        forwarding_channels =
            control_plane.get_forwarding_eth_chans_to_chip(src_fabric_node_id, dst_fabric_node_id, direction);
    } else {
        // for 1D check if each port has an active connection to the dst_chip_id
        const auto& cluster = tt::tt_metal::MetalContext::instance().get_cluster();
        const auto& soc_desc = cluster.get_soc_desc(src_chip_id);
        for (const auto& channel : fabric_channels) {
            const auto eth_core = soc_desc.get_eth_core_for_channel(channel, CoordSystem::LOGICAL);
            auto [connected_chip_id, connected_eth_core] =
                cluster.get_connected_ethernet_core(std::make_tuple(src_chip_id, CoreCoord{eth_core.x, eth_core.y}));
            if (connected_chip_id == dst_chip_id) {
                forwarding_channels.push_back(channel);
            }
        }
    }

    std::vector<uint32_t> link_indices;
    for (uint32_t i = 0; i < fabric_channels.size(); i++) {
        if (std::find(forwarding_channels.begin(), forwarding_channels.end(), fabric_channels[i]) !=
            forwarding_channels.end()) {
            link_indices.push_back(i);
        }
    }

    return link_indices;
}

std::vector<uint32_t> get_forwarding_link_indices(chip_id_t src_chip_id, chip_id_t dst_chip_id) {
    const auto& control_plane= tt::tt_metal::MetalContext::instance().get_control_plane();
    const auto src_fabric_node_id = control_plane.get_fabric_node_id_from_physical_chip_id(src_chip_id);
    const auto dst_fabric_node_id = control_plane.get_fabric_node_id_from_physical_chip_id(dst_chip_id);

    // find the forwarding direction b/w src and dest chip
    const auto& forwarding_direction = control_plane.get_forwarding_direction(src_fabric_node_id, dst_fabric_node_id);
    if (!forwarding_direction.has_value()) {
        return {};
    }

    return get_forwarding_link_indices_in_direction(src_chip_id, dst_chip_id, forwarding_direction.value());
}

void set_routing_mode(uint16_t routing_mode) {
    // override for forced routing mode
    if (routing_mode == ROUTING_MODE_UNDEFINED) {
        return;
    }

    // Validate dimension flags are orthogonal (only one can be set)
    TT_FATAL(
        __builtin_popcount(routing_mode & (ROUTING_MODE_1D | ROUTING_MODE_2D | ROUTING_MODE_3D)) == 1,
        "Only one dimension mode (1D, 2D, 3D) can be active at once");

    // Validate topology flags are orthogonal
    TT_FATAL(
        __builtin_popcount(
            routing_mode & (ROUTING_MODE_RING | ROUTING_MODE_LINE | ROUTING_MODE_MESH | ROUTING_MODE_TORUS)) == 1,
        "Only one topology mode (RING, LINE, MESH, TORUS) can be active at once");

    // Validate push/pull flags are orthogonal
    TT_FATAL(
        __builtin_popcount(routing_mode & (ROUTING_MODE_PUSH | ROUTING_MODE_PULL)) <= 1,
        "PUSH and PULL routing modes cannot be used together");

    // Validate push/pull flags are only for 2D
    TT_FATAL(
        !(routing_mode & (ROUTING_MODE_PUSH | ROUTING_MODE_PULL)) || (routing_mode & ROUTING_MODE_2D),
        "PUSH and PULL routing modes can only be used with 2D topology");

    // Validate 1D can't be used with MESH or TORUS
    TT_FATAL(
        !(routing_mode & ROUTING_MODE_1D) || !(routing_mode & (ROUTING_MODE_MESH | ROUTING_MODE_TORUS)),
        "1D routing mode cannot be combined with MESH or TORUS topology");

    // Validate 2D can't be used with LINE or RING
    TT_FATAL(
        !(routing_mode & ROUTING_MODE_2D) || !(routing_mode & (ROUTING_MODE_LINE | ROUTING_MODE_RING)),
        "2D routing mode cannot be combined with LINE or RING topology");

    auto& control_plane = tt::tt_metal::MetalContext::instance().get_control_plane();
    control_plane.set_routing_mode(routing_mode);
}

void set_routing_mode(Topology topology, tt::tt_metal::FabricConfig fabric_config, uint32_t dimension /*, take more*/) {
    // TODO: take more parameters to set detail routing mode
    TT_FATAL(
        dimension == 1 || dimension == 2 || dimension == 3,
        "Invalid dimension {}. Supported dimensions are 1, 2, or 3",
        dimension);

    uint16_t mode = (dimension == 3 ? ROUTING_MODE_3D : 0);
    if (topology == Topology::Ring) {
        mode |= (ROUTING_MODE_1D | ROUTING_MODE_RING);
    } else if (topology == Topology::Linear) {
        mode |= (ROUTING_MODE_1D | ROUTING_MODE_LINE);
    } else if (topology == Topology::Mesh) {
        mode |= (ROUTING_MODE_2D | ROUTING_MODE_MESH);
    }
    if (fabric_config == tt::tt_metal::FabricConfig::FABRIC_2D_DYNAMIC) {
        mode |= ROUTING_MODE_DYNAMIC;
    } else {
        mode |= ROUTING_MODE_LOW_LATENCY;
    }
    set_routing_mode(mode);
}

void get_optimal_noc_for_edm(
    tt::tt_fabric::FabricEriscDatamoverBuilder& edm_builder1,
    tt::tt_fabric::FabricEriscDatamoverBuilder& edm_builder2,
    const uint32_t num_links,
    const tt_fabric::Topology topology) {
    constexpr uint32_t ring_noc_selection_link_threshold = 3;
    constexpr uint32_t line_noc_selection_link_threshold = 2;
    bool enable_noc_selection_opt = false;
    if (topology == tt_fabric::Topology::Ring) {
        enable_noc_selection_opt =
            (num_links > ring_noc_selection_link_threshold) && (edm_builder1.my_noc_y != edm_builder2.my_noc_y);
    } else {
        enable_noc_selection_opt =
            (num_links > line_noc_selection_link_threshold) && (edm_builder1.my_noc_y != edm_builder2.my_noc_y);
    }
    log_debug(
        tt::LogTest,
        "device {} edm_builder1 {} {} is connecting to edm_builder2 {} {} num links {}",
        edm_builder1.my_chip_id,
        edm_builder1.my_noc_x,
        edm_builder1.my_noc_y,
        edm_builder2.my_noc_x,
        edm_builder2.my_noc_y,
        num_links);

    if (enable_noc_selection_opt) {
        if (edm_builder1.my_noc_x < edm_builder2.my_noc_x) {
            for (uint32_t i = 0; i < edm_builder1.config.num_receiver_channels; i++) {
                edm_builder1.config.receiver_channel_forwarding_noc_ids[i] = 0;
                edm_builder2.config.receiver_channel_forwarding_noc_ids[i] = 1;
            }
            for (uint32_t i = 0; i < edm_builder1.config.num_receiver_channels; i++) {
                edm_builder1.config.receiver_channel_local_write_noc_ids[i] = 1;
                edm_builder2.config.receiver_channel_local_write_noc_ids[i] = 1;
            }
            for (uint32_t i = 0; i < edm_builder1.config.num_sender_channels; i++) {
                edm_builder1.config.sender_channel_ack_noc_ids[i] = 1;
                edm_builder2.config.sender_channel_ack_noc_ids[i] = 0;
            }
        } else if (edm_builder1.my_noc_x > edm_builder2.my_noc_x) {
            for (uint32_t i = 0; i < edm_builder1.config.num_receiver_channels; i++) {
                edm_builder1.config.receiver_channel_forwarding_noc_ids[i] = 1;
                edm_builder2.config.receiver_channel_forwarding_noc_ids[i] = 0;
            }
            for (uint32_t i = 0; i < edm_builder1.config.num_receiver_channels; i++) {
                edm_builder1.config.receiver_channel_local_write_noc_ids[i] = 1;
                edm_builder2.config.receiver_channel_local_write_noc_ids[i] = 1;
            }
            for (uint32_t i = 0; i < edm_builder1.config.num_sender_channels; i++) {
                edm_builder1.config.sender_channel_ack_noc_ids[i] = 0;
                edm_builder2.config.sender_channel_ack_noc_ids[i] = 1;
            }
        }
    }
}

}  // namespace tt::tt_fabric
