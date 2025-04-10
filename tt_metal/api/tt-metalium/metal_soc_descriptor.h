// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <stdint.h>
#include <cstddef>
#include <map>
#include <vector>

#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/tt_backend_api_types.hpp>
#include <umd/device/tt_cluster_descriptor.h>
#include <umd/device/tt_core_coordinates.h>
#include <umd/device/tt_soc_descriptor.h>
#include <umd/device/tt_xy_pair.h>
#include <umd/device/types/xy_pair.h>

enum BoardType : uint32_t;

//! tt_SocDescriptor contains information regarding the SOC configuration targetted.
/*!
    Should only contain relevant configuration for SOC
*/
struct metal_SocDescriptor : public tt_SocDescriptor {
public:
    std::vector<size_t> dram_view_channels;
    std::vector<CoreCoord> dram_view_worker_cores;  // per channel preferred worker endpoint
    std::vector<CoreCoord> dram_view_eth_cores;     // per dram view preferred eth endpoint
    std::vector<size_t> dram_view_address_offsets;  // starting address offset

    uint64_t dram_core_size;
    uint64_t dram_view_size;

    std::map<CoreCoord, int> logical_eth_core_to_chan_map;

    metal_SocDescriptor(const tt_SocDescriptor& other, const BoardType& board_type);
    metal_SocDescriptor() = default;

    CoreCoord get_preferred_worker_core_for_dram_view(int dram_view) const;
    CoreCoord get_preferred_eth_core_for_dram_view(int dram_view) const;
    CoreCoord get_logical_core_for_dram_view(int dram_view) const;
    size_t get_address_offset(int dram_view) const;
    size_t get_channel_for_dram_view(int dram_view) const;
    size_t get_num_dram_views() const;

    int get_dram_channel_from_logical_core(const CoreCoord& logical_coord) const;

    CoreCoord get_physical_ethernet_core_from_logical(const CoreCoord& logical_coord) const;
    CoreCoord get_logical_ethernet_core_from_physical(const CoreCoord& physical_coord) const;
    CoreCoord get_physical_tensix_core_from_logical(const CoreCoord& logical_coord) const;
    CoreCoord get_physical_dram_core_from_logical(const CoreCoord& logical_coord) const;
    CoreCoord get_physical_core_from_logical_core(const CoreCoord& logical_coord, const CoreType& core_type) const;

    CoreCoord get_dram_grid_size() const;

    tt_cxy_pair convert_to_umd_coordinates(const tt_cxy_pair& physical_cxy) const;

    // During the transition of the UMD's api to CoreCoords, this function is needed to make the transition smoother.
    // At the moment, different coordinate systems are expected for grayskull and other architectures.
    CoordSystem get_umd_coord_system() const;

    // Number of cores per DRAM bank ceiled to nearest integer
    int profiler_ceiled_core_count_perf_dram_bank = 0;
    std::map<CoreCoord, int32_t> physical_routing_to_profiler_flat_id;

private:
    void load_dram_metadata_from_device_descriptor();
    void generate_logical_eth_coords_mapping();
    void generate_physical_routing_to_profiler_flat_id();
};
