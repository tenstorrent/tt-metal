// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <vector>

#include <tt-metalium/core_coord.hpp>

namespace tt::tt_metal {

// Forward declaration
enum NOC : uint8_t;

class IDevice;
}  // namespace tt::tt_metal

namespace tt::tt_metal::distributed {
class MeshDevice;
class MeshCoordinate;
}  // namespace tt::tt_metal::distributed

namespace tt::tt_metal::experimental::Device {

struct DramBankNoc0ReadEndpoint {
    CoreCoord placement_coordinate;
    CoreCoord address_coordinate;
};

// Returns the hop distance between two logical worker coordinates on a given NOC
// This API is experimental and may evolve into a stable Device API in the future
uint32_t get_worker_noc_hop_distance(
    IDevice* device, const CoreCoord& logical_src, const CoreCoord& logical_dst, NOC noc);

// Returns the hop distance between two logical worker coordinates on a given NOC
// NOC distances may vary depending on the target device due to harvesting
// This API is experimental and may evolve into a stable Device API in the future
uint32_t get_worker_noc_hop_distance(
    distributed::MeshDevice* mesh_device,
    const distributed::MeshCoordinate& mesh_coord,
    const CoreCoord& logical_src,
    const CoreCoord& logical_dst,
    NOC noc);

// Returns additional addressable endpoints for one DRAM bank in NOC0 coordinates, excluding the
// bank's default endpoint. The endpoint list is topology-derived; callers must not infer raw
// subchannel ids from its order. This direct-routing API is restricted to unit meshes while its
// concurrency contract is being generalized.
std::vector<DramBankNoc0ReadEndpoint> get_additional_dram_bank_noc0_read_endpoints(IDevice* device, uint32_t dram_bank);
}  // namespace tt::tt_metal::experimental::Device
