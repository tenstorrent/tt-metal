// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

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
#include "fabric/hw/inc/fabric_routing_mode.h"

namespace tt::tt_fabric {

bool is_1d_fabric_config(tt::tt_metal::FabricConfig fabric_config) {
    return fabric_config == tt::tt_metal::FabricConfig::FABRIC_1D ||
           fabric_config == tt::tt_metal::FabricConfig::FABRIC_1D_RING;
}

bool is_2d_fabric_config(tt::tt_metal::FabricConfig fabric_config) {
    return fabric_config == tt::tt_metal::FabricConfig::FABRIC_2D ||
           fabric_config == tt::tt_metal::FabricConfig::FABRIC_2D_PUSH;
}

Topology get_1d_topology(tt::tt_metal::FabricConfig fabric_config) {
    switch (fabric_config) {
        case tt::tt_metal::FabricConfig::FABRIC_1D: return tt::tt_fabric::Topology::Linear;
        case tt::tt_metal::FabricConfig::FABRIC_1D_RING: return tt::tt_fabric::Topology::Ring;
        case tt::tt_metal::FabricConfig::DISABLED:
        case tt::tt_metal::FabricConfig::FABRIC_2D:
        case tt::tt_metal::FabricConfig::FABRIC_2D_PUSH:
        case tt::tt_metal::FabricConfig::CUSTOM:
            TT_THROW("Unsupported fabric config for 1D: {}", magic_enum::enum_name(fabric_config));
    }
    return tt::tt_fabric::Topology::Linear;
}

FabricType get_fabric_type(tt::tt_metal::FabricConfig fabric_config, tt::ClusterType cluster_type) {
    if (cluster_type == tt::ClusterType::GALAXY && fabric_config == tt::tt_metal::FabricConfig::FABRIC_1D_RING) {
        return FabricType::TORUS_2D;
    }
    return FabricType::MESH;
}

std::vector<chan_id_t> get_ordered_fabric_eth_chans(chip_id_t chip_id, const std::set<chan_id_t>& eth_chans) {
    std::vector<std::pair<chan_id_t, CoreCoord>> ordered_eth_chans_cores;
    auto soc_desc = tt::tt_metal::MetalContext::instance().get_cluster().get_soc_desc(chip_id);
    for (const auto& chan : eth_chans) {
        ordered_eth_chans_cores.push_back(
            std::make_pair(chan, soc_desc.get_eth_core_for_channel(chan, CoordSystem::VIRTUAL)));
    }

    std::sort(ordered_eth_chans_cores.begin(), ordered_eth_chans_cores.end(), [](const auto& a, const auto& b) {
        return a.second.x < b.second.x;
    });

    std::vector<chan_id_t> ordered_eth_chans;
    for (const auto& [chan, _] : ordered_eth_chans_cores) {
        ordered_eth_chans.push_back(chan);
    }
    return ordered_eth_chans;
}

void set_routing_mode(uint16_t routing_mode) {
    // override for forced routing mode
    if (routing_mode != ROUTING_MODE_UNDEFINED) {
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

    auto control_plane = tt::tt_metal::MetalContext::instance().get_cluster().get_control_plane();
    control_plane->set_routing_mode(routing_mode);
}

void set_routing_mode(Topology topology, uint32_t dimension /*, take more*/) {
    // TODO: take more parameters to set detail routing mode
    TT_FATAL(
        dimension == 1 || dimension == 2 || dimension == 3,
        "Invalid dimension {}. Supported dimensions are 1, 2, or 3",
        dimension);

    uint16_t mode = (dimension == 1 ? ROUTING_MODE_1D : dimension == 2 ? ROUTING_MODE_2D : ROUTING_MODE_3D);
    if (topology == Topology::Ring) {
        mode |= ROUTING_MODE_RING;
    } else if (topology == Topology::Linear) {
        mode |= ROUTING_MODE_LINE;
    } else if (topology == Topology::Mesh) {
        mode |= ROUTING_MODE_MESH;
    }

    mode |= ROUTING_MODE_LOW_LATENCY;
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
