// SPDX-FileCopyrightText: © 2023 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <map>
#include <vector>

#include <umd/device/cluster.hpp>

namespace tt::llrt {

const std::unordered_set<chip_id_t>& get_devices_controlled_by_mmio_device(
    const std::unique_ptr<tt::umd::Cluster>& cluster, chip_id_t mmio_device_id);
std::map<chip_id_t, std::vector<std::vector<chip_id_t>>> discover_tunnels_from_mmio_device(
    const std::unique_ptr<tt::umd::Cluster>& cluster);

}  // namespace tt::llrt
