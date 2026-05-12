// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <map>
#include <unordered_set>
#include <vector>

#include <umd/device/types/cluster_descriptor_types.hpp>
#include <umd/device/cluster_descriptor.hpp>

namespace tt::llrt {

const std::unordered_set<ChipId>& get_devices_controlled_by_mmio_device(
    tt::umd::ClusterDescriptor& cluster_descriptor, ChipId mmio_device_id);

std::map<ChipId, std::vector<std::vector<ChipId>>> discover_tunnels_from_mmio_device(
    tt::umd::ClusterDescriptor& cluster_desc);

}  // namespace tt::llrt
