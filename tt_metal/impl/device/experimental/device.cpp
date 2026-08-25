// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <limits>

#include <tt-metalium/experimental/device.hpp>
#include <tt-metalium/device.hpp>
#include <tt-metalium/mesh_device.hpp>
#include <tt_stl/assert.hpp>
#include "tt_metal/impl/device/device_impl.hpp"

namespace tt::tt_metal::experimental::Device {

namespace {

tt::tt_metal::Device* concrete_device(IDevice* device) {
    TT_FATAL(device != nullptr, "Device pointer cannot be null");

    if (auto* mesh = dynamic_cast<distributed::MeshDevice*>(device)) {
        TT_FATAL(mesh->num_devices() == 1, "Experimental NOC geometry APIs are only supported on unit MeshDevice.");
        return concrete_device(mesh->get_devices().front());
    }

    auto* dev = dynamic_cast<tt::tt_metal::Device*>(device);
    TT_FATAL(dev != nullptr, "Device pointer must be a valid Device or MeshDevice");
    return dev;
}

uint32_t noc_hop_distance(const CoreCoord& src, const CoreCoord& dst, const CoreCoord& grid_size, NOC noc) {
    if (noc == NOC::NOC_0) {
        // NOC0: Preferred +x -> +y
        uint32_t dist_right = src.x <= dst.x ? dst.x - src.x : grid_size.x - src.x + dst.x;
        uint32_t dist_bottom = src.y <= dst.y ? dst.y - src.y : grid_size.y - src.y + dst.y;
        return dist_right + dist_bottom;
    }  // NOC1: Preferred -y -> -x
    uint32_t dist_left = src.x >= dst.x ? src.x - dst.x : grid_size.x - dst.x + src.x;
    uint32_t dist_top = src.y >= dst.y ? src.y - dst.y : grid_size.y - dst.y + src.y;
    return dist_left + dist_top;
}

}  // namespace

uint32_t get_worker_noc_hop_distance(
    IDevice* device, const CoreCoord& logical_src, const CoreCoord& logical_dst, NOC noc) {
    auto* dev = concrete_device(device);
    return noc_hop_distance(
        dev->physical_worker_core_from_logical_core(logical_src),
        dev->physical_worker_core_from_logical_core(logical_dst),
        device->grid_size(),
        noc);
}

uint32_t get_worker_noc_hop_distance(
    distributed::MeshDevice* mesh_device,
    const distributed::MeshCoordinate& mesh_coord,
    const CoreCoord& logical_src,
    const CoreCoord& logical_dst,
    NOC noc) {
    TT_FATAL(mesh_device != nullptr, "MeshDevice pointer cannot be null");
    const auto linear_index = mesh_coord.to_linear_index(mesh_device->shape());
    return get_worker_noc_hop_distance(mesh_device->get_devices().at(linear_index), logical_src, logical_dst, noc);
}

CoreCoord get_closest_worker_to_eth_core(
    IDevice* device, const CoreCoord& logical_eth_core, NOC noc, uint32_t& noc_hops) {
    auto* dev = concrete_device(device);
    const auto eth_core = dev->physical_eth_core_from_logical_core(logical_eth_core);
    const auto grid_size = device->grid_size();
    const auto worker_grid_size = device->compute_with_storage_grid_size();

    CoreCoord closest = CoreCoord{0, 0};
    uint32_t min_hops_so_far = std::numeric_limits<uint32_t>::max();
    for (uint32_t y = 0; y < worker_grid_size.y; y++) {
        for (uint32_t x = 0; x < worker_grid_size.x; x++) {
            uint32_t hops = noc_hop_distance(
                dev->physical_worker_core_from_logical_core(CoreCoord{x, y}), eth_core, grid_size, noc);
            if (hops < min_hops_so_far) {
                min_hops_so_far = hops;
                closest = CoreCoord{x, y};
            }
        }
    }

    noc_hops = min_hops_so_far;
    return closest;
}

}  // namespace tt::tt_metal::experimental::Device
