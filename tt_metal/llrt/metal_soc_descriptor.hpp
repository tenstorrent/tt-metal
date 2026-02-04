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
#include <umd/device/cluster_descriptor.hpp>
#include <umd/device/types/core_coordinates.hpp>
#include <umd/device/soc_descriptor.hpp>
#include <umd/device/types/xy_pair.hpp>
#include <umd/device/types/cluster_descriptor_types.hpp>

//! SocDescriptor contains information regarding the SOC configuration targetted.
/*!
    Should only contain relevant configuration for SOC
*/
struct metal_SocDescriptor : public tt::umd::SocDescriptor {
public:
    std::vector<size_t> dram_view_channels;
    std::vector<std::vector<CoreCoord>>
        dram_view_worker_cores;                               // per dram view preferred worker endpoints for each noc
    std::vector<std::vector<CoreCoord>> dram_view_eth_cores;  // per dram view preferred eth endpoints for each noc
    std::vector<size_t> dram_view_address_offsets;            // starting address offset

    uint64_t dram_core_size{};
    uint64_t dram_view_size{};

    std::map<CoreCoord, int> logical_eth_core_to_chan_map;

    metal_SocDescriptor(const SocDescriptor& other, const tt::BoardType& board_type);
    metal_SocDescriptor() = default;

    CoreCoord get_preferred_worker_core_for_dram_view(int dram_view, uint8_t noc) const;
    CoreCoord get_preferred_eth_core_for_dram_view(int dram_view, uint8_t noc) const;
    CoreCoord get_logical_core_for_dram_view(int dram_view) const;
    size_t get_address_offset(int dram_view) const;
    size_t get_channel_for_dram_view(int dram_view) const;
    size_t get_num_dram_views() const;

    int get_dram_channel_from_logical_core(const CoreCoord& logical_coord) const;

    CoreCoord get_physical_ethernet_core_from_logical(const CoreCoord& logical_coord) const;
    CoreCoord get_logical_ethernet_core_from_physical(const CoreCoord& physical_coord) const;
    CoreCoord get_physical_tensix_core_from_logical(const CoreCoord& logical_coord) const;
    CoreCoord get_physical_dram_core_from_logical(const CoreCoord& logical_coord) const;
    CoreCoord get_physical_core_from_logical_core(const CoreCoord& logical_coord, const tt::CoreType& core_type) const;

    CoreCoord get_dram_grid_size() const;

    // Number of cores per DRAM bank ceiled to nearest integer
    int profiler_ceiled_core_count_perf_dram_bank = 0;
    std::map<CoreCoord, int32_t> physical_routing_to_profiler_flat_id;

private:
    void load_dram_metadata_from_device_descriptor();
    void generate_logical_eth_coords_mapping();
    void generate_physical_routing_to_profiler_flat_id();
};
