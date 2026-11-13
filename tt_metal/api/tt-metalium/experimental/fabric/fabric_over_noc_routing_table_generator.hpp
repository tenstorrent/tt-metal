// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <vector>

#include <hostdevcommon/fabric_common.h>
#include <tt-metalium/device_types.hpp>
#include <tt-metalium/experimental/fabric/routing_table_generator.hpp>

namespace tt {
class Cluster;
namespace umd {
class TTDevice;
}  // namespace umd
}  // namespace tt

namespace tt::tt_fabric {
class TopologyMapper;

// Flat ATT addresses used to reach each core of one destination chip. 0 = unfilled slot.
struct FlatAddrEntry {
    std::array<std::array<uint64_t, MAX_HORIZON_NOC_Y>, MAX_HORIZON_NOC_X> flat_addr{};
};

using FlatAddrRoutingTable =
    RoutingTableT<FlatAddrEntry>;  // [mesh_id][src_chip][dst_chip], indexed by physical chip id

class FabricOverNocRoutingTableGenerator {
public:
    FabricOverNocRoutingTableGenerator(const TopologyMapper& topology_mapper, const tt::Cluster& cluster);

    // Builds the table from the topology mapper's chips. Safe to call more than once.
    void generate_routing_table();

    const FlatAddrRoutingTable& get_routing_table() const { return this->routing_table_; }

    void print_routing_tables() const;

private:
    const TopologyMapper& topology_mapper_;
    const tt::Cluster& cluster_;

    FlatAddrRoutingTable routing_table_;

    void fill_local_entries_(FlatAddrEntry& entry, uint32_t my_noc_x, uint32_t my_noc_y) const;
    size_t select_cross_chip_access_point_(ChipId src_phys, ChipId dst_phys, tt::umd::TTDevice& dst_device) const;
    void fill_cross_chip_entries_(
        FlatAddrEntry& entry, ChipId dst_phys, tt::umd::TTDevice& dst_device, size_t ap_idx) const;
};

}  // namespace tt::tt_fabric
