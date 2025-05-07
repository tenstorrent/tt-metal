// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <stdint.h>
#include <tt-metalium/fabric_edm_types.hpp>
#include <tt-metalium/fabric_types.hpp>
#include <umd/device/types/cluster_descriptor_types.h>  // chip_id_t
#include <tt-metalium/erisc_datamover_builder.hpp>
#include "tt_metal/impl/context/metal_context.hpp"
#include "tt_metal/fabric/hw/inc/fabric_routing_mode.h"
#include <set>
#include <vector>

namespace tt::tt_fabric {

bool is_1d_fabric_config(tt::tt_metal::FabricConfig fabric_config);
bool is_2d_fabric_config(tt::tt_metal::FabricConfig fabric_config);

Topology get_1d_topology(tt::tt_metal::FabricConfig fabric_config);
void set_routing_mode(uint16_t routing_mode);
void set_routing_mode(Topology topology, uint32_t dimension = 1);

FabricType get_fabric_type(tt::tt_metal::FabricConfig fabric_config, tt::ClusterType cluster_type);

std::vector<chan_id_t> get_ordered_fabric_eth_chans(chip_id_t chip_id, const std::set<chan_id_t>& eth_chans);

void get_optimal_noc_for_edm(
    FabricEriscDatamoverBuilder& edm_builder1,
    FabricEriscDatamoverBuilder& edm_builder2,
    const uint32_t num_links,
    const Topology topology);

}  // namespace tt::tt_fabric
