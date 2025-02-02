// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "tt_backend_api_types.hpp"
#include "core_coord.hpp"
#include "umd/device/tt_soc_descriptor.h"
#include "umd/device/tt_cluster_descriptor.h"

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

    std::vector<CoreCoord> logical_ethernet_cores;
    uint64_t dram_core_size;
    uint64_t dram_view_size;

    // in tt_SocDescriptor worker_log_to_routing_x and worker_log_to_routing_y map logical coordinates to NOC virtual
    // coordinates UMD accepts NOC virtual coordinates but Metal needs NOC physical coordinates to ensure a harvested
    // core is not targetted
    std::unordered_map<tt_xy_pair, CoreDescriptor> physical_cores;
    std::vector<tt_xy_pair> physical_workers;
    std::vector<tt_xy_pair> physical_harvested_workers;
    std::vector<tt_xy_pair> physical_ethernet_cores;

    std::map<CoreCoord, int> logical_eth_core_to_chan_map;
    std::map<int, CoreCoord> chan_to_logical_eth_core_map;

    metal_SocDescriptor(const tt_SocDescriptor& other, uint32_t harvesting_mask, const BoardType& board_type);
    metal_SocDescriptor() = default;

    CoreCoord get_preferred_worker_core_for_dram_view(int dram_view) const;
    CoreCoord get_preferred_eth_core_for_dram_view(int dram_view) const;
    CoreCoord get_logical_core_for_dram_view(int dram_view) const;
    size_t get_address_offset(int dram_view) const;
    size_t get_channel_for_dram_view(int dram_view) const;
    size_t get_num_dram_views() const;

    bool is_harvested_core(const CoreCoord& core) const;
    const std::vector<CoreCoord>& get_pcie_cores() const;
    const std::vector<CoreCoord> get_dram_cores() const;
    const std::vector<CoreCoord>& get_logical_ethernet_cores() const;
    const std::vector<CoreCoord>& get_physical_ethernet_cores() const;

    int get_dram_channel_from_logical_core(const CoreCoord& logical_coord) const;

    CoreCoord get_physical_ethernet_core_from_logical(const CoreCoord& logical_coord) const;
    CoreCoord get_logical_ethernet_core_from_physical(const CoreCoord& physical_coord) const;
    CoreCoord get_physical_tensix_core_from_logical(const CoreCoord& logical_coord) const;
    CoreCoord get_physical_dram_core_from_logical(const CoreCoord& logical_coord) const;
    CoreCoord get_physical_core_from_logical_core(const CoreCoord& logical_coord, const CoreType& core_type) const;

    CoreCoord get_dram_grid_size() const;

    tt_cxy_pair convert_to_umd_coordinates(const tt_cxy_pair& physical_cxy) const;

    // Number of cores per DRAM bank ceiled to nearest integer
    int profiler_ceiled_core_count_perf_dram_bank = 0;
    std::map<CoreCoord, int32_t> physical_routing_to_profiler_flat_id;

private:
    void generate_physical_descriptors_from_virtual(uint32_t harvesting_mask);
    void load_dram_metadata_from_device_descriptor();
    void generate_logical_eth_coords_mapping();
    void generate_physical_routing_to_profiler_flat_id();
    // This is temporary until virtual coordinates are enabled because BH chips on
    //  different cards use different physical PCIe NoC endpoints
    void update_pcie_cores(const BoardType& board_type);
};
