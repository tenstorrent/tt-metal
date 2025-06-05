// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <stdint.h>
#include <tt-metalium/fabric_edm_types.hpp>
#include <tt-metalium/fabric_types.hpp>
#include <tt-metalium/mesh_graph.hpp>                   // FabricType
#include <umd/device/types/cluster_descriptor_types.h>  // chip_id_t
#include <llrt/tt_cluster.hpp>
#include <tt-metalium/erisc_datamover_builder.hpp>
#include <set>
#include <vector>

namespace tt::tt_fabric {

bool is_tt_fabric_config(tt::tt_metal::FabricConfig fabric_config);
bool is_2d_fabric_config(tt::tt_metal::FabricConfig fabric_config);

uint32_t get_sender_channel_count(tt::tt_fabric::Topology topology);
uint32_t get_downstream_edm_count(tt::tt_fabric::Topology topology);

void set_routing_mode(uint16_t routing_mode);
void set_routing_mode(Topology topology, tt::tt_metal::FabricConfig fabric_config, uint32_t dimension = 1);

FabricType get_fabric_type(tt::tt_metal::FabricConfig fabric_config, tt::ClusterType cluster_type);

std::vector<uint32_t> get_forwarding_link_indices_in_direction(
    chip_id_t src_chip_id, chip_id_t dst_chip_id, RoutingDirection direction);

// returns which links on a given src chip are available for forwarding the data to a dst chip
// these link indices can then be used to establish connection with the fabric routers
std::vector<uint32_t> get_forwarding_link_indices(chip_id_t src_chip_id, chip_id_t dst_chip_id);

void get_optimal_noc_for_edm(
    FabricEriscDatamoverBuilder& edm_builder1,
    FabricEriscDatamoverBuilder& edm_builder2,
    const uint32_t num_links,
    const Topology topology);

}  // namespace tt::tt_fabric
