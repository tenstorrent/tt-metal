// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <stdint.h>
#include <cstddef>
#include <map>
#include <vector>

#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/tt_backend_api_types.hpp>
#include <umd/device/cluster_descriptor.hpp>
#include <umd/device/types/core_coordinates.hpp>
#include <umd/device/soc_descriptor.hpp>
#include <umd/device/types/xy_pair.hpp>
#include <umd/device/types/cluster_descriptor_types.hpp>

//! SocDescriptor contains information regarding the SOC configuration targeted.
/*!
    Should only contain relevant configuration for SOC
*/
struct metal_SocDescriptor : public tt::umd::SocDescriptor {
public:
    std::vector<size_t> dram_view_channels;
    std::vector<std::vector<tt::tt_metal::CoreCoord>>
        dram_view_worker_cores;                               // per dram view preferred worker endpoints for each noc
    std::vector<std::vector<tt::tt_metal::CoreCoord>> dram_view_eth_cores;  // per dram view preferred eth endpoints for each noc
    std::vector<size_t> dram_view_address_offsets;            // starting address offset

    // Per bank, ordered endpoint translated coordinates. The index is the y of a logical DRAM
    // CoreCoord, and the order is by role so that y means the same thing on every device regardless
    // of DRAM harvesting: index 0 = worker endpoint for NOC 0, index 1 = worker endpoint for NOC 1
    // (when it is a different subchannel), then the bank's remaining subchannels ascending.
    std::vector<std::vector<tt::tt_metal::CoreCoord>> dram_bank_endpoint_coords;

    uint64_t dram_core_size{};
    uint64_t dram_view_size{};

    std::map<tt::tt_metal::CoreCoord, int> logical_eth_core_to_chan_map;

    metal_SocDescriptor(const SocDescriptor& other, const tt::BoardType& board_type);

    tt::tt_metal::CoreCoord get_preferred_worker_core_for_dram_view(int dram_view, uint8_t noc) const;
    tt::tt_metal::CoreCoord get_preferred_eth_core_for_dram_view(int dram_view, uint8_t noc) const;

    // The DRAM cores Metal may place kernels/firmware on, in the requested coordinate system. This is
    // the single source of truth for "usable DRAM cores": every DRAM loop in Metal (firmware init,
    // launch-message reset, watcher, inspector) should iterate this rather than get_cores(DRAM) so the
    // usable set is defined in one place. On Blackhole it excludes each DRAM view's NOC0 worker
    // endpoint, which is owned by the syseng firmware (CMFW DRAM telemetry, SYS-1419) and runs no
    // DRISC firmware. Hardware without that restriction returns all DRAM cores -- callers never need
    // to special-case it.
    // A LOGICAL request returns Metal logical DRAM coords ({dram_view, dram_bank_endpoint_coords
    // index}, the space get_physical_dram_core_from_logical and CreateKernel(DramConfig) use), not
    // UMD's {channel, raw subchannel}.
    std::vector<tt::tt_metal::CoreCoord> get_metal_dram_cores(tt::CoordSystem coord_system) const;
    tt::tt_metal::CoreCoord get_logical_core_for_dram_view(int dram_view) const;
    size_t get_address_offset(int dram_view) const;
    size_t get_channel_for_dram_view(int dram_view) const;
    size_t get_num_dram_views() const;

    int get_dram_channel_from_logical_core(const tt::tt_metal::CoreCoord& logical_coord) const;

    tt::tt_metal::CoreCoord get_physical_ethernet_core_from_logical(const tt::tt_metal::CoreCoord& logical_coord) const;
    tt::tt_metal::CoreCoord get_logical_ethernet_core_from_physical(const tt::tt_metal::CoreCoord& physical_coord) const;
    tt::tt_metal::CoreCoord get_physical_tensix_core_from_logical(const tt::tt_metal::CoreCoord& logical_coord) const;
    tt::tt_metal::CoreCoord get_physical_dram_core_from_logical(const tt::tt_metal::CoreCoord& logical_coord) const;
    // Map a DRAM view + hardware subchannel to the logical CoreCoord used by CreateKernel(DramConfig).
    // logical.y indexes dram_bank_endpoint_coords (ordered by endpoint role), not the raw subchannel id.
    tt::tt_metal::CoreCoord get_logical_dram_core_for_subchannel(int dram_view, int subchannel) const;
    // Same logical space as get_logical_dram_core_for_subchannel, keyed by a TRANSLATED coord. Inverse
    // of get_physical_dram_core_from_logical. A DRAM view is a dram_view_size window of one hardware
    // channel -- what Metal allocates against as a DRAM bank -- so a channel split into several views
    // (Wormhole: two 1 GB views per channel) has one set of NOC endpoints serving all of them. This
    // returns the lowest view reaching the coord; it round-trips, but is not the only answer.
    tt::tt_metal::CoreCoord get_logical_dram_core_from_translated(
        const tt::tt_metal::CoreCoord& translated_coord) const;
    tt::tt_metal::CoreCoord get_physical_dispatch_engine_core_from_logical(const tt::tt_metal::CoreCoord& logical_coord) const;
    uint32_t get_num_dispatch_engine_cores() const;

    tt::tt_metal::CoreCoord get_physical_core_from_logical_core(const tt::tt_metal::CoreCoord& logical_coord, const tt::CoreType& core_type) const;

    tt::tt_metal::CoreCoord get_dram_grid_size() const;
    tt::tt_metal::CoreCoord get_dram_compute_grid_size() const;

    // Number of cores per DRAM bank ceiled to nearest integer
    int profiler_ceiled_core_count_perf_dram_bank = 0;
    std::map<tt::tt_metal::CoreCoord, int32_t> physical_routing_to_profiler_flat_id;

private:
    // Physical DRAM channel (device-descriptor numbering, with harvested-channel gaps) for a dram
    // view. Internal building block for get_channel_for_dram_view, which compacts it to the logical
    // index get_dram_core_for_channel expects; callers want the logical one.
    size_t get_physical_channel_for_dram_view(int dram_view) const;

    // True if `translated_coord` is any DRAM view's NOC0 worker endpoint (the subchannel a NOC0 DRAM
    // access routes to) -- the syseng-owned endpoint excluded by get_metal_dram_cores on Blackhole.
    // Argument must be a TRANSLATED (UMD) coord; a metal-logical {view, subchannel} coord never matches.
    bool is_noc0_dram_endpoint(const tt::tt_metal::CoreCoord& translated_coord) const;

    void load_dram_metadata_from_device_descriptor();
    void generate_logical_eth_coords_mapping();
    void generate_physical_routing_to_profiler_flat_id();
};
