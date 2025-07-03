// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "metal_soc_descriptor.h"

#include <assert.hpp>
#include <yaml-cpp/yaml.h>
#include <string>

#include <umd/device/types/arch.h>

enum BoardType : uint32_t;

CoreCoord metal_SocDescriptor::get_preferred_worker_core_for_dram_view(int dram_view, uint8_t noc) const {
    TT_ASSERT(
        dram_view < this->dram_view_worker_cores.size(),
        "dram_view={} must be within range of dram_view_worker_cores.size={}",
        dram_view,
        this->dram_view_worker_cores.size());
    TT_ASSERT(noc < 2, "Only 2 NOCs supported, noc={} is out of range", noc);
    return this->dram_view_worker_cores.at(dram_view).at(noc);
};

CoreCoord metal_SocDescriptor::get_preferred_eth_core_for_dram_view(int dram_view, uint8_t noc) const {
    TT_ASSERT(
        dram_view < this->dram_view_eth_cores.size(),
        "dram_view={} must be within range of dram_view_eth_cores.size={}",
        dram_view,
        this->dram_view_eth_cores.size());
    TT_ASSERT(noc < 2, "Only 2 NOCs supported, noc={} is out of range", noc);
    return this->dram_view_eth_cores.at(dram_view).at(noc);
};

CoreCoord metal_SocDescriptor::get_logical_core_for_dram_view(int dram_view) const {
    const uint32_t num_dram_views = this->get_num_dram_views();
    TT_FATAL(
        dram_view < num_dram_views,
        "dram_view={} must be within range of num_dram_views={}",
        dram_view,
        num_dram_views);
    return CoreCoord(dram_view, 0);
}

size_t metal_SocDescriptor::get_address_offset(int dram_view) const {
    TT_ASSERT(
        dram_view < this->dram_view_address_offsets.size(),
        "dram_view={} must be within range of dram_view_address_offsets.size={}",
        dram_view,
        this->dram_view_address_offsets.size());
    return this->dram_view_address_offsets.at(dram_view);
}

size_t metal_SocDescriptor::get_channel_for_dram_view(int dram_view) const {
    TT_ASSERT(
        dram_view < this->dram_view_channels.size(),
        "dram_view={} must be within range of dram_view_channels.size={}",
        dram_view,
        this->dram_view_channels.size());
    return this->dram_view_channels.at(dram_view);
}

size_t metal_SocDescriptor::get_num_dram_views() const { return this->dram_view_eth_cores.size(); }

int metal_SocDescriptor::get_dram_channel_from_logical_core(const CoreCoord& logical_coord) const {
    const uint32_t num_dram_views = this->get_num_dram_views();
    TT_FATAL(
        (logical_coord.x < num_dram_views) and (logical_coord.y == 0),
        "Bounds-Error -- Logical_core={} is outside of logical_grid_size={}",
        logical_coord.str(),
        CoreCoord(num_dram_views, 1));
    return logical_coord.x;
}

CoreCoord metal_SocDescriptor::get_physical_ethernet_core_from_logical(const CoreCoord& logical_coord) const {
    tt::umd::CoreCoord physical_coord =
        translate_coord_to({logical_coord, CoreType::ETH, CoordSystem::LOGICAL}, CoordSystem::PHYSICAL);
    return {physical_coord.x, physical_coord.y};
}

CoreCoord metal_SocDescriptor::get_logical_ethernet_core_from_physical(const CoreCoord& physical_coord) const {
    tt::umd::CoreCoord logical_coord =
        translate_coord_to({physical_coord, CoreType::ETH, CoordSystem::PHYSICAL}, CoordSystem::LOGICAL);
    return {logical_coord.x, logical_coord.y};
}

CoreCoord metal_SocDescriptor::get_physical_tensix_core_from_logical(const CoreCoord& logical_coord) const {
    tt::umd::CoreCoord physical_coord =
        translate_coord_to({logical_coord, CoreType::TENSIX, CoordSystem::LOGICAL}, CoordSystem::PHYSICAL);
    return {physical_coord.x, physical_coord.y};
}

// Kernels are not exposed to physical DRAM coordinates. Use case of this function is for host to get access to a DRAM
// core, host will likely not be bombarding DRAM with different nocs so SYS-1419 isn't a concern here and its safe to
// use noc 0
CoreCoord metal_SocDescriptor::get_physical_dram_core_from_logical(const CoreCoord& logical_coord) const {
    return this->get_preferred_worker_core_for_dram_view(this->get_dram_channel_from_logical_core(logical_coord), 0);
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

CoreCoord metal_SocDescriptor::get_dram_grid_size() const { return CoreCoord(this->get_num_dram_views(), 1); }

void metal_SocDescriptor::load_dram_metadata_from_device_descriptor() {
    YAML::Node device_descriptor_yaml = YAML::LoadFile(this->device_descriptor_file_path);
    this->dram_view_size = device_descriptor_yaml["dram_view_size"].as<uint64_t>();
    this->dram_core_size = device_descriptor_yaml["dram_views"].size() * this->dram_view_size;
    this->dram_view_channels.clear();
    this->dram_view_eth_cores.clear();
    this->dram_view_worker_cores.clear();
    this->dram_view_address_offsets.clear();

    for (const auto& dram_view : device_descriptor_yaml["dram_views"]) {
        size_t channel = dram_view["channel"].as<size_t>();
        if (channel >= get_grid_size(CoreType::DRAM).x) {
            // DRAM can be harvested and we don't create unique soc desc for diff harvesting
            break;
        }
        size_t address_offset = dram_view["address_offset"].as<size_t>();

        std::vector<CoreCoord> eth_dram_cores;
        std::vector<size_t> eth_endpoints;
        for (int eth_endpoint : dram_view["eth_endpoint"].as<std::vector<int>>()) {
            if (eth_endpoint >= get_grid_size(CoreType::DRAM).y) {
                TT_THROW(
                    "DRAM subchannel {} does not exist in the device descriptor, but is specified in "
                    "dram_view.eth_endpoint",
                    eth_endpoint);
            }
            tt::umd::CoreCoord eth_dram_endpoint_coord =
                get_dram_core_for_channel(channel, eth_endpoint, CoordSystem::TRANSLATED);
            eth_dram_cores.push_back({eth_dram_endpoint_coord.x, eth_dram_endpoint_coord.y});
            eth_endpoints.push_back(eth_endpoint);
        }

        std::vector<CoreCoord> worker_dram_cores;
        std::vector<size_t> worker_endpoints;
        for (int worker_endpoint : dram_view["worker_endpoint"].as<std::vector<int>>()) {
            if (worker_endpoint >= get_grid_size(CoreType::DRAM).y) {
                TT_THROW(
                    "DRAM subchannel {} does not exist in the device descriptor, but is specified in "
                    "dram_view.worker_endpoint",
                    worker_endpoint);
            }
            tt::umd::CoreCoord worker_endpoint_coord =
                get_dram_core_for_channel(channel, worker_endpoint, CoordSystem::TRANSLATED);

            worker_dram_cores.push_back({worker_endpoint_coord.x, worker_endpoint_coord.y});
            worker_endpoints.push_back(worker_endpoint);
        }

        this->dram_view_channels.push_back(channel);
        this->dram_view_address_offsets.push_back(address_offset);
        this->dram_view_eth_cores.push_back(eth_dram_cores);
        this->dram_view_worker_cores.push_back(worker_dram_cores);
    }
}

// UMD expects virtual NOC coordinates for worker cores
tt_cxy_pair metal_SocDescriptor::convert_to_umd_coordinates(const tt_cxy_pair& physical_cxy) const {
    tt::umd::CoreCoord virtual_coord =
        translate_coord_to((tt_xy_pair)physical_cxy, CoordSystem::PHYSICAL, get_umd_coord_system());
    return tt_cxy_pair(physical_cxy.chip, virtual_coord.x, virtual_coord.y);
}

CoordSystem metal_SocDescriptor::get_umd_coord_system() const {
    return CoordSystem::VIRTUAL;
}

void metal_SocDescriptor::generate_logical_eth_coords_mapping() {
    for (const auto& logical_coord : this->get_cores(CoreType::ETH, CoordSystem::LOGICAL)) {
        this->logical_eth_core_to_chan_map.insert({{logical_coord.x, logical_coord.y}, logical_coord.y});
    }
}

void metal_SocDescriptor::generate_physical_routing_to_profiler_flat_id() {
#if defined(TRACY_ENABLE)
    for (auto& core : get_cores(CoreType::TENSIX, CoordSystem::PHYSICAL)) {
        this->physical_routing_to_profiler_flat_id.emplace((CoreCoord){core.x, core.y}, 0);
    }

    for (auto& core : this->get_cores(CoreType::ETH, CoordSystem::PHYSICAL)) {
        this->physical_routing_to_profiler_flat_id.emplace((CoreCoord){core.x, core.y}, 0);
    }

    int flat_id = 0;
    for (auto& core : this->physical_routing_to_profiler_flat_id) {
        this->physical_routing_to_profiler_flat_id[core.first] = flat_id;
        flat_id++;
    }

    int coreCount = this->physical_routing_to_profiler_flat_id.size();
    this->profiler_ceiled_core_count_perf_dram_bank = coreCount / this->get_num_dram_views();
    if ((coreCount % this->get_num_dram_views()) > 0) {
        this->profiler_ceiled_core_count_perf_dram_bank++;
    }

#endif
}

// UMD initializes and owns tt_SocDescriptor
// For architectures with translation tables enabled, UMD will remove the last x rows from the descriptors in
// tt_SocDescriptor (workers list and worker_log_to_routing_x/y maps) This creates a virtual coordinate system, where
// translation tables are used to convert virtual core coordinates to the true harvesting state. For architectures
// without translation tables enabled, UMD updates tt_SocDescriptor to contain the true harvesting state by
// removing the harvested physical coordiniates Metal needs the true harvesting state so we generate physical
// descriptors from virtual coordinates We also initialize additional lookup tables to translate physical coordinates to
// virtual coordinates because UMD APIs expect virtual coordinates.
metal_SocDescriptor::metal_SocDescriptor(const tt_SocDescriptor& other, const BoardType& /*board_type*/) :
    tt_SocDescriptor(other) {
    this->load_dram_metadata_from_device_descriptor();
    this->generate_logical_eth_coords_mapping();
    this->generate_physical_routing_to_profiler_flat_id();
}
