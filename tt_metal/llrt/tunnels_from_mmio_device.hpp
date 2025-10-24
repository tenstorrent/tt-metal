// SPDX-FileCopyrightText: © 2023 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <map>
#include <vector>

#include <umd/device/cluster.hpp>

namespace tt::llrt {

const std::unordered_set<ChipId>& get_devices_controlled_by_mmio_device(
    const std::unique_ptr<tt::umd::Cluster>& cluster, ChipId mmio_device_id);
std::map<ChipId, std::vector<std::vector<ChipId>>> discover_tunnels_from_mmio_device(
    const std::unique_ptr<tt::umd::Cluster>& cluster);

}  // namespace tt::llrt
