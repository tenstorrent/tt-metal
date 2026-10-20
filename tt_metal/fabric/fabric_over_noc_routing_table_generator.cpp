// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <tt-metalium/experimental/fabric/fabric_over_noc_routing_table_generator.hpp>

#include <vector>

#include <tt_stl/assert.hpp>
#include <tt-metalium/experimental/fabric/topology_mapper.hpp>
#include <tt-umd/types/core_coordinates.hpp>

#include "llrt/metal_soc_descriptor.hpp"
#include "llrt/tt_cluster.hpp"

namespace tt::tt_fabric {

namespace {

void verify_core_coordinates(const std::vector<tt::umd::CoreCoord>& cores, uint32_t x_max, uint32_t y_max) {
    for (const auto& core : cores) {
        TT_FATAL(
            core.x < x_max && core.y < y_max,
            "Tensix core ({}, {}) exceeds fabric-over-noc routing table bounds ({}, {}); "
            "soc descriptor and fabric constants are inconsistent",
            core.x,
            core.y,
            x_max,
            y_max);
    }
}

}  // namespace

FabricOverNocRoutingTableGenerator::FabricOverNocRoutingTableGenerator(
    const TopologyMapper& topology_mapper, const tt::Cluster& cluster) :
    topology_mapper_(topology_mapper), cluster_(cluster) {}

void FabricOverNocRoutingTableGenerator::generate_routing_table() {
    const auto mapping = this->topology_mapper_.get_local_logical_mesh_chip_id_to_physical_chip_id_mapping();

    // Horizon x2 supports exactly one physical mesh (AXI crossbar, not Ethernet links).
    // flat_addr is indexed by routing plane [0] — assert this assumption holds.
    TT_FATAL(
        mapping.size() <= MAX_HORIZON_DEVS_PER_MESH,
        "configure_routing_tables_for_fabric_flat_address: expected at most {} devices in one mesh, got {}",
        MAX_HORIZON_DEVS_PER_MESH,
        mapping.size());

    {
        const auto& first_phys = mapping.begin()->second;
        const auto ip_variant = this->cluster_.get_soc_desc(first_phys).get_arch_descriptor().get_ip_variant();
        uint32_t max_x = 0, max_y = 0;
        if (ip_variant == tt::IpVariant::HORIZON) {
            max_x = MAX_HORIZON_NOC_X;
            max_y = MAX_HORIZON_NOC_Y;
        } else {
            TT_FATAL(false, "Quasar IP Variant not found for core coordinate verification");
        }
        for (const auto& [node, phys] : mapping) {
            verify_core_coordinates(
                this->cluster_.get_soc_desc(phys).get_cores(CoreType::TENSIX, CoordSystem::NOC0), max_x, max_y);
        }
    }

    this->routing_table_.assign(
        MAX_HORIZON_MESHES,
        std::vector<std::vector<FlatAddrEntry>>(
            MAX_HORIZON_DEVS_PER_MESH, std::vector<FlatAddrEntry>(MAX_HORIZON_DEVS_PER_MESH)));

    for (const auto& [src_node, src_phys] : mapping) {
        for (const auto& [dst_node, dst_phys] : mapping) {
            // Index by physical chip ID — the kernel receives dst_dev_id as the physical chip ID.
            const uint32_t dst_chip = dst_phys;

            if (dst_phys == src_phys) {
                this->fill_local_entries_(this->routing_table_[0][src_phys][dst_chip]);
                continue;
            }
            TT_FATAL(
                dst_chip < MAX_HORIZON_DEVS_PER_MESH,
                "Fabric-over-NoC routing table: dst chip {} >= MAX_HORIZON_DEVS_PER_MESH ({})",
                dst_chip,
                MAX_HORIZON_DEVS_PER_MESH);

            auto& entry = this->routing_table_[0][src_phys][dst_chip];
            auto* dst_device = this->cluster_.get_driver()->get_tt_device(dst_phys);
            if (!dst_device || dst_device->get_access_points().empty()) {
                continue;
            }

            const size_t ap_idx = this->select_cross_chip_access_point_(src_phys, dst_phys, *dst_device);
            this->fill_cross_chip_entries_(entry, dst_phys, *dst_device, ap_idx);
        }
    }
}

void FabricOverNocRoutingTableGenerator::fill_local_entries_(FlatAddrEntry& entry) const {
    // Every core on this device gets the same table, so no core can be treated as "self": every core, the
    // writer's own included, is reached through the REMOTE window. A REMOTE write with a core's own selector
    // reaches that core too.
    for (uint32_t phys_x = 0; phys_x < MAX_HORIZON_NOC_X; phys_x++) {
        for (uint32_t phys_y = 0; phys_y < MAX_HORIZON_NOC_Y; phys_y++) {
            const uint32_t selector = phys_x + phys_y * HORIZON_ATT_WORKER_GRID_X;
            entry.flat_addr[phys_x][phys_y] =
                HORIZON_REMOTE_WINDOW_COMPARE | (uint64_t(selector) << HORIZON_REMOTE_ENDPOINT_SHIFT);
        }
    }
}

size_t FabricOverNocRoutingTableGenerator::select_cross_chip_access_point_(
    ChipId src_phys, ChipId dst_phys, tt::umd::TTDevice& dst_device) const {
    const auto access_points = dst_device.get_access_points();
    // Select the first Access Point whose flat address falls outside LOCAL and REMOTE windows.
    // APs inside those windows loop back to the local chip rather than crossing the crossbar.
    size_t selected_ap_idx = access_points.size();
    const auto probe_cores = this->cluster_.get_soc_desc(dst_phys).get_cores(CoreType::TENSIX, CoordSystem::NOC0);
    if (!probe_cores.empty()) {
        const auto& probe_core = probe_cores[0];
        for (size_t i = 0; i < access_points.size(); ++i) {
            const auto probe_addr = dst_device.get_flat_address(
                access_points[i], probe_core, tt::umd::WindowFlags::UnicastWrite, tt::umd::IoOrdering::Posted, 0);
            if (!probe_addr.has_value()) {
                continue;
            }
            const bool in_local = (*probe_addr >> HORIZON_LOCAL_WINDOW_MASK_BITS) ==
                                  (HORIZON_LOCAL_WINDOW_COMPARE >> HORIZON_LOCAL_WINDOW_MASK_BITS);
            const bool in_remote = (*probe_addr >> HORIZON_REMOTE_WINDOW_MASK_BITS) ==
                                   (HORIZON_REMOTE_WINDOW_COMPARE >> HORIZON_REMOTE_WINDOW_MASK_BITS);
            if (!in_local && !in_remote) {
                selected_ap_idx = i;
                break;
            }
        }
    }
    TT_FATAL(
        selected_ap_idx < access_points.size(),
        "Fabric-over-NoC routing table: no valid cross-chip AP for src={} dst={}: "
        "all {} APs fall within LOCAL or REMOTE window",
        src_phys,
        dst_phys,
        access_points.size());
    return selected_ap_idx;
}

void FabricOverNocRoutingTableGenerator::fill_cross_chip_entries_(
    FlatAddrEntry& entry, ChipId dst_phys, tt::umd::TTDevice& dst_device, size_t ap_idx) const {
    const auto access_points = dst_device.get_access_points();
    // The flat address to reach core (x, y) on a destination chip is the same regardless of which source chip is
    // computing it. The devices' NOC2AXI ATTs differ; it holds because a flat address is the destination NOC2AXI's
    // crossbar window, independent of the sender.
    for (const auto& core : this->cluster_.get_soc_desc(dst_phys).get_cores(CoreType::TENSIX, CoordSystem::NOC0)) {
        const auto flat = dst_device.get_flat_address(
            access_points[ap_idx],
            core,
            tt::umd::WindowFlags::UnicastWrite,
            tt::umd::IoOrdering::Posted,
            /*offset=*/0);
        if (flat.has_value()) {
            entry.flat_addr[core.x][core.y] = *flat;
        }
    }
}

}  // namespace tt::tt_fabric
