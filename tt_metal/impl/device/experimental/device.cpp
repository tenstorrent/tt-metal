// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <limits>

#include <tt-metalium/experimental/device.hpp>
#include <tt-metalium/device.hpp>
#include <tt-metalium/mesh_device.hpp>
#include <tt_stl/assert.hpp>
#include "tt_metal/impl/device/device_impl.hpp"
#include "tt_metal/distributed/mesh_device_impl.hpp"

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
        dev->grid_size(),
        noc);
}

uint32_t get_worker_noc_hop_distance(
    distributed::MeshDevice* mesh_device,
    const distributed::MeshCoordinate& mesh_coord,
    const CoreCoord& logical_src,
    const CoreCoord& logical_dst,
    NOC noc) {
    TT_FATAL(mesh_device != nullptr, "MeshDevice pointer cannot be null");
    // Resolve by coordinate rather than by linear index into get_devices(): that vector holds only
    // the devices this rank drives, so on a submesh co-owned by several ranks a linear index over
    // the full mesh shape walks off its end (coordinate (2, 1) of a 4x2 -> index 5 into 4 local
    // devices).
    IDevice* device = nullptr;
    if (mesh_device->impl().is_local(mesh_coord)) {
        device = mesh_device->impl().get_device(mesh_coord);
    } else {
        // The hop metric is a device-local physical property: logical->physical worker coordinates
        // come from that chip's SoC descriptor, so this is a best-effort approximation, exact only
        // when the mesh is homogeneously harvested. A co-owner composing a peer's coordinate -- to
        // keep mesh-level allocation sequences symmetric -- has nothing else to measure on.
        // Mirrors get_optimal_dram_bank_to_logical_worker_assignment(NOC, coord).
        const auto local_devices = mesh_device->get_devices();
        TT_FATAL(
            !local_devices.empty(),
            "get_worker_noc_hop_distance: MeshCoordinate {} maps to a remote device and this mesh has no local "
            "devices to fall back to.",
            mesh_coord);
        device = local_devices.front();
    }
    return get_worker_noc_hop_distance(device, logical_src, logical_dst, noc);
}

CoreCoord get_closest_worker_to_eth_core(
    IDevice* device, const CoreCoord& logical_eth_core, NOC noc, uint32_t& noc_hops) {
    auto* dev = concrete_device(device);
    const auto eth_core = dev->physical_eth_core_from_logical_core(logical_eth_core);
    const auto grid_size = dev->grid_size();
    const auto worker_grid_size = dev->compute_with_storage_grid_size();

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
