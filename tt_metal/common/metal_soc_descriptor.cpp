// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "metal_soc_descriptor.h"

#include <fstream>
#include <iostream>
#include <string>

#include <assert.hpp>
#include "umd/device/cluster.h"
#include "yaml-cpp/yaml.h"

CoreCoord metal_SocDescriptor::get_preferred_worker_core_for_dram_channel(int dram_chan) const {
    TT_ASSERT(
        dram_chan < this->preferred_worker_dram_core.size(),
        "dram_chan={} must be within range of preferred_worker_dram_core.size={}",
        dram_chan,
        this->preferred_worker_dram_core.size());
    return this->preferred_worker_dram_core.at(dram_chan);
};

CoreCoord metal_SocDescriptor::get_preferred_eth_core_for_dram_channel(int dram_chan) const {
    TT_ASSERT(
        dram_chan < this->preferred_eth_dram_core.size(),
        "dram_chan={} must be within range of preferred_eth_dram_core.size={}",
        dram_chan,
        this->preferred_eth_dram_core.size());
    return this->preferred_eth_dram_core.at(dram_chan);
};

CoreCoord metal_SocDescriptor::get_logical_core_for_dram_channel(int dram_chan) const {
    const uint32_t num_dram_channels = this->get_num_dram_channels();
    TT_FATAL(
        dram_chan < num_dram_channels,
        "dram_chan={} must be within range of num_dram_channels={}",
        dram_chan,
        num_dram_channels);
    return CoreCoord(dram_chan, 0);
}

size_t metal_SocDescriptor::get_address_offset(int dram_chan) const {
    TT_ASSERT(
        dram_chan < this->dram_address_offsets.size(),
        "dram_chan={} must be within range of dram_address_offsets.size={}",
        dram_chan,
        this->dram_address_offsets.size());
    return this->dram_address_offsets.at(dram_chan);
}

bool metal_SocDescriptor::is_harvested_core(const CoreCoord& core) const {
    for (const auto& core_it : this->physical_harvested_workers) {
        if (core_it == core) {
            return true;
        }
    }
    return false;
}

const std::vector<CoreCoord>& metal_SocDescriptor::get_pcie_cores() const { return this->pcie_cores; }

const std::vector<CoreCoord> metal_SocDescriptor::get_dram_cores() const {
    std::vector<CoreCoord> cores;

    // This is inefficient, but is currently not used in a perf path
    for (const auto& channel_it : this->dram_cores) {
        for (const auto& core_it : channel_it) {
            cores.push_back(core_it);
        }
    }

    return cores;
}

const std::vector<CoreCoord>& metal_SocDescriptor::get_physical_ethernet_cores() const {
    return this->physical_ethernet_cores;
}

const std::vector<CoreCoord>& metal_SocDescriptor::get_logical_ethernet_cores() const {
    return this->logical_ethernet_cores;
}

int metal_SocDescriptor::get_dram_channel_from_logical_core(const CoreCoord& logical_coord) const {
    const uint32_t num_dram_channels = this->get_num_dram_channels();
    TT_FATAL(
        (logical_coord.x < num_dram_channels) and (logical_coord.y == 0),
        "Bounds-Error -- Logical_core={} is outside of logical_grid_size={}",
        logical_coord.str(),
        CoreCoord(num_dram_channels, 1));
    return logical_coord.x;
}

CoreCoord metal_SocDescriptor::get_physical_ethernet_core_from_logical(const CoreCoord& logical_coord) const {
    const auto& eth_chan_map = this->logical_eth_core_to_chan_map;
    TT_FATAL(
        (eth_chan_map.find(logical_coord) != eth_chan_map.end()),
        "Bounds-Error -- Logical_core={} is outside of ethernet logical grid",
        logical_coord.str());
    return this->physical_ethernet_cores.at(eth_chan_map.at(logical_coord));
}

CoreCoord metal_SocDescriptor::get_logical_ethernet_core_from_physical(const CoreCoord& physical_coord) const {
    const auto& phys_eth_map = this->physical_ethernet_cores;
    auto it = std::find(phys_eth_map.begin(), phys_eth_map.end(), physical_coord);

    TT_FATAL(
        (it != phys_eth_map.end()),
        "Bounds-Error -- Physical_core={} is outside of ethernet physical grid",
        physical_coord.str());

    int chan = it - phys_eth_map.begin();
    return this->chan_to_logical_eth_core_map.at(chan);
}

CoreCoord metal_SocDescriptor::get_physical_tensix_core_from_logical(const CoreCoord& logical_coord) const {
    tt::umd::CoreCoord physical_coord =
        translate_coord_to({logical_coord, CoreType::TENSIX, CoordSystem::LOGICAL}, CoordSystem::PHYSICAL);
    return {physical_coord.x, physical_coord.y};
}

CoreCoord metal_SocDescriptor::get_physical_dram_core_from_logical(const CoreCoord& logical_coord) const {
    return this->get_preferred_worker_core_for_dram_channel(this->get_dram_channel_from_logical_core(logical_coord));
}

CoreCoord metal_SocDescriptor::get_physical_core_from_logical_core(
    const CoreCoord& logical_coord, const CoreType& core_type) const {
    switch (core_type) {
        case CoreType::ETH: return this->get_physical_ethernet_core_from_logical(logical_coord);
        case CoreType::WORKER: return this->get_physical_tensix_core_from_logical(logical_coord);
        case CoreType::DRAM: return this->get_physical_dram_core_from_logical(logical_coord);
        default: TT_THROW("Undefined conversion for core type.");
    }
}

CoreCoord metal_SocDescriptor::get_dram_grid_size() const { return CoreCoord(this->get_num_dram_channels(), 1); }

void metal_SocDescriptor::load_dram_metadata_from_device_descriptor() {
    YAML::Node device_descriptor_yaml = YAML::LoadFile(this->device_descriptor_file_path);
    int num_dram_channels = this->get_num_dram_channels();
    this->preferred_eth_dram_core.clear();
    for (const auto& core_node : device_descriptor_yaml["dram_preferred_eth_endpoint"]) {
        if (core_node.IsScalar()) {
            this->preferred_eth_dram_core.push_back(format_node(core_node.as<std::string>()));
        } else {
            TT_THROW("Only NOC coords supported for dram_preferred_eth_endpoint cores");
        }
    }
    if (this->preferred_eth_dram_core.size() != num_dram_channels) {
        TT_THROW(
            "Expected to specify preferred DRAM endpoint for ethernet core for {} channels but yaml specifies {} "
            "channels",
            num_dram_channels,
            this->preferred_eth_dram_core.size());
    }

    this->preferred_worker_dram_core.clear();
    for (const auto& core_node : device_descriptor_yaml["dram_preferred_worker_endpoint"]) {
        if (core_node.IsScalar()) {
            this->preferred_worker_dram_core.push_back(format_node(core_node.as<std::string>()));
        } else {
            TT_THROW("Only NOC coords supported for dram_preferred_worker_endpoint");
        }
    }
    if (this->preferred_worker_dram_core.size() != num_dram_channels) {
        TT_THROW(
            "Expected to specify preferred DRAM endpoint for worker core for {} channels but yaml specifies {} "
            "channels",
            num_dram_channels,
            this->preferred_worker_dram_core.size());
    }

    this->dram_address_offsets = device_descriptor_yaml["dram_address_offsets"].as<std::vector<size_t>>();
    if (this->dram_address_offsets.size() != num_dram_channels) {
        TT_THROW(
            "Expected DRAM offsets for {} channels but yaml specified {} channels",
            num_dram_channels,
            this->dram_address_offsets.size());
    }

    int dram_banks_per_prev_core = 0;
    int dram_banks_per_core = 0;
    for (int dram_channel = 0; dram_channel < this->dram_address_offsets.size(); dram_channel++) {
        if (this->dram_address_offsets.at(dram_channel) == 0) {
            if (dram_banks_per_prev_core > 0 and dram_banks_per_prev_core != dram_banks_per_core) {
                TT_THROW("Expected {} DRAM banks per DRAM core", dram_banks_per_prev_core);
            } else if (dram_banks_per_core != 0) {
                dram_banks_per_prev_core = dram_banks_per_core;
            }
            dram_banks_per_core = 1;
        } else {
            dram_banks_per_core++;
        }
    }

    this->dram_core_size = dram_banks_per_core * this->dram_bank_size;
}

// UMD expects virtual NOC coordinates for worker cores
tt_cxy_pair metal_SocDescriptor::convert_to_umd_coordinates(const tt_cxy_pair& physical_cxy) const {
    CoordSystem target_system = (this->arch == tt::ARCH::GRAYSKULL) ? CoordSystem::PHYSICAL : CoordSystem::VIRTUAL;
    tt::umd::CoreCoord virtual_coord =
        translate_coord_to((tt_xy_pair)physical_cxy, CoordSystem::PHYSICAL, target_system);
    return tt_cxy_pair(physical_cxy.chip, virtual_coord.x, virtual_coord.y);
}

void metal_SocDescriptor::generate_physical_descriptors_from_virtual(uint32_t harvesting_mask) {
    // No need to remap virtual descriptors to physical because Grayskull does not have translation tables enabled,
    // meaning UMD removes the physical harvested rows rather than using virtual coordinates
    if (harvesting_mask == 0 or (this->arch == tt::ARCH::GRAYSKULL)) {
        this->physical_cores = this->cores;
        this->physical_workers = this->workers;
        this->physical_harvested_workers = this->harvested_workers;

        return;
    }

    std::set<int> row_coordinates_to_remove;
    int row_coordinate = 0;
    int tmp = harvesting_mask;
    while (tmp) {
        if (tmp & 1) {
            row_coordinates_to_remove.insert(row_coordinate);
        }
        tmp = tmp >> 1;
        row_coordinate++;
    }

    std::set<int> virtual_harvested_rows;
    for (const CoreCoord& virtual_harvested_core : this->harvested_workers) {
        virtual_harvested_rows.insert(virtual_harvested_core.y);
    }

    if (row_coordinates_to_remove.size() != virtual_harvested_rows.size()) {
        TT_THROW(
            "Expected number of harvested rows removed by UMD ({}) to match number of harvested rows set in harvesting "
            "mask ({})",
            virtual_harvested_rows.size(),
            row_coordinates_to_remove.size());
    }

    // Columns are not harvested so virtual x == physical x
    std::set<int> virtual_y_coords;
    for (const auto& [virtual_noc_core, core_desc] : this->cores) {
        if (core_desc.type == CoreType::WORKER or core_desc.type == CoreType::HARVESTED) {
            virtual_y_coords.insert(virtual_noc_core.y);
        }
    }

    std::unordered_map<int, int> virtual_routing_to_physical_routing_y;
    auto virtual_y_coord_it = virtual_y_coords.begin();
    // worker grid size does not include harvested rows
    for (int logical_y_coord = 0; logical_y_coord < this->worker_grid_size.y; logical_y_coord++) {
        while (row_coordinates_to_remove.find(*virtual_y_coord_it) != row_coordinates_to_remove.end()) {
            virtual_y_coord_it++;
        }
        int physical_y_coord = *virtual_y_coord_it;
        virtual_y_coord_it++;
        // This branch will never be executed for Grayskull, but for completeness keeping it in here.
        // This will go away in the next PR anyway.
        CoordSystem target_system = (this->arch == tt::ARCH::GRAYSKULL) ? CoordSystem::PHYSICAL : CoordSystem::VIRTUAL;
        tt::umd::CoreCoord virtual_coord =
            translate_coord_to({0, logical_y_coord, CoreType::TENSIX, CoordSystem::LOGICAL}, target_system);
        virtual_routing_to_physical_routing_y.insert({virtual_coord.y, physical_y_coord});
    }

    // map physical harvested rows to virtual harvested rows
    std::unordered_map<int, int> virtual_harvested_row_to_physical_harvested_row;
    for (auto v_it = virtual_harvested_rows.begin(), p_it = row_coordinates_to_remove.begin();
         v_it != virtual_harvested_rows.end() and p_it != row_coordinates_to_remove.end();
         ++v_it, ++p_it) {
        virtual_routing_to_physical_routing_y.insert({*v_it, *p_it});
    }

    for (const auto& [virtual_noc_core, core_desc] : this->cores) {
        CoreCoord physical_noc_core = virtual_noc_core;
        CoreDescriptor phys_core_desc = core_desc;
        if (core_desc.type == CoreType::WORKER or core_desc.type == CoreType::HARVESTED) {
            physical_noc_core.y = virtual_routing_to_physical_routing_y.at(virtual_noc_core.y);
            phys_core_desc.coord = physical_noc_core;
            if (row_coordinates_to_remove.find(physical_noc_core.y) != row_coordinates_to_remove.end()) {
                phys_core_desc.type = CoreType::HARVESTED;
                this->physical_harvested_workers.push_back(physical_noc_core);
            } else {
                phys_core_desc.type = CoreType::WORKER;
                this->physical_workers.push_back(physical_noc_core);
            }
        }
        this->physical_cores.insert({physical_noc_core, phys_core_desc});
    }
}

void metal_SocDescriptor::generate_logical_eth_coords_mapping() {
    this->physical_ethernet_cores = this->ethernet_cores;
    for (int i = 0; i < this->physical_ethernet_cores.size(); i++) {
        CoreCoord core = {0, static_cast<size_t>(i)};
        this->logical_eth_core_to_chan_map.insert({core, i});
        this->chan_to_logical_eth_core_map.insert({i, core});
        this->logical_ethernet_cores.emplace_back(core);
    }
}

void metal_SocDescriptor::generate_physical_routing_to_profiler_flat_id() {
#if defined(TRACY_ENABLE)
    for (auto& core : this->physical_workers) {
        this->physical_routing_to_profiler_flat_id.emplace((CoreCoord){core.x, core.y}, 0);
    }

    for (auto& core : this->physical_ethernet_cores) {
        this->physical_routing_to_profiler_flat_id.emplace((CoreCoord){core.x, core.y}, 0);
    }

    int flat_id = 0;
    for (auto& core : this->physical_routing_to_profiler_flat_id) {
        this->physical_routing_to_profiler_flat_id[core.first] = flat_id;
        flat_id++;
    }

    int coreCount = this->physical_routing_to_profiler_flat_id.size();
    this->profiler_ceiled_core_count_perf_dram_bank = coreCount / this->get_num_dram_channels();
    if ((coreCount % this->get_num_dram_channels()) > 0) {
        this->profiler_ceiled_core_count_perf_dram_bank++;
    }

#endif
}

// TODO: This should be deleted once we switch to virtual coordinates
void metal_SocDescriptor::update_pcie_cores(const BoardType& board_type) {
    if (this->arch != tt::ARCH::BLACKHOLE) {
        return;
    }
    switch (board_type) {
        case P100:
        case UNKNOWN: {  // Workaround for BHs running FW that does not return board type in the cluster yaml
            this->pcie_cores = {CoreCoord(11, 0)};
        } break;
        case P150A: {
            this->pcie_cores = {CoreCoord(2, 0)};
        } break;
        default: TT_THROW("Need to update PCIe core assignment for new Blackhole type, file issue to abhullar");
    }
}

// UMD initializes and owns tt_SocDescriptor
// For architectures with translation tables enabled, UMD will remove the last x rows from the descriptors in
// tt_SocDescriptor (workers list and worker_log_to_routing_x/y maps) This creates a virtual coordinate system, where
// translation tables are used to convert virtual core coordinates to the true harvesting state. For architectures
// without translation tables enabled (Grayskull), UMD updates tt_SocDescriptor to contain the true harvesting state by
// removing the harvested physical coordiniates Metal needs the true harvesting state so we generate physical
// descriptors from virtual coordinates We also initialize additional lookup tables to translate physical coordinates to
// virtual coordinates because UMD APIs expect virtual coordinates.
metal_SocDescriptor::metal_SocDescriptor(
    const tt_SocDescriptor& other, uint32_t harvesting_mask, const BoardType& board_type) :
    tt_SocDescriptor(other) {
    this->generate_physical_descriptors_from_virtual(harvesting_mask);
    this->load_dram_metadata_from_device_descriptor();
    this->generate_logical_eth_coords_mapping();
    this->generate_physical_routing_to_profiler_flat_id();
    this->update_pcie_cores(board_type);
}
