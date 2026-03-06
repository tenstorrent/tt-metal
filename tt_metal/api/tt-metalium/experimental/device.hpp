// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <tt-metalium/core_coord.hpp>

namespace tt::tt_metal {

// Forward declaration
enum NOC : uint8_t;

class IDevice;
}  // namespace tt::tt_metal

namespace tt::tt_metal::experimental::Device {

// Returns the hop distance between two logical worker coordinates on a given NOC
// This API is experimental and may evolve into a stable Device API in the future
uint32_t get_worker_noc_hop_distance(
    IDevice* device, const CoreCoord& logical_src, const CoreCoord& logical_dst, NOC noc);

}  // namespace tt::tt_metal::experimental::Device
