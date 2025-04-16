// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <stdint.h>
#include <tt-metalium/fabric_edm_types.hpp>
#include <tt-metalium/fabric_types.hpp>
#include <umd/device/types/cluster_descriptor_types.h>  // chip_id_t
#include <set>
#include <vector>

namespace tt::tt_fabric {

bool is_1d_fabric_config(const tt::tt_metal::FabricConfig& fabric_config);
bool is_2d_fabric_config(const tt::tt_metal::FabricConfig& fabric_config);

Topology get_1d_topology(const tt::tt_metal::FabricConfig& fabric_config);

std::vector<chan_id_t> get_ordered_fabric_eth_chans(chip_id_t chip_id, const std::set<chan_id_t>& eth_chans);

}  // namespace tt::tt_fabric
