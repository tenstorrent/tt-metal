// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <optional>

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

class FabricFlatAddressRoutingTableGenerator {
public:
    FabricFlatAddressRoutingTableGenerator(const TopologyMapper& topology_mapper, const tt::Cluster& cluster);

    // Builds the table from the topology mapper's chips. Safe to call more than once.
    void generate_routing_table();

    const FlatAddrRoutingTable& get_routing_table() const { return this->routing_table_; }

    // Added to a cross-chip entry, the address of the same L1 word's posted-atomic window; 0 when there is none.
    uint64_t get_atomic_window_offset() const { return this->atomic_window_offset_.value_or(0); }

private:
    const TopologyMapper& topology_mapper_;
    const tt::Cluster& cluster_;

    FlatAddrRoutingTable routing_table_;
    std::optional<uint64_t> atomic_window_offset_;

    void fill_local_entries_(FlatAddrEntry& entry) const;
    void fill_cross_chip_entries_(FlatAddrEntry& entry, ChipId dst_phys, tt::umd::TTDevice& dst_device, size_t ap_idx);
};

}  // namespace tt::tt_fabric
