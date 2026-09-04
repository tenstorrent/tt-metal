// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "llrt/metal_soc_descriptor.hpp"

#include <tt_stl/assert.hpp>
#include <yaml-cpp/yaml.h>

#include <umd/device/types/arch.hpp>

namespace {
// True if physical DRAM `channel` is harvested per `dram_harvesting_mask`. Single home for the
// bit-masking convention used across the DRAM-view helpers below.
bool is_dram_channel_harvested(uint32_t dram_harvesting_mask, size_t channel) {
    return (dram_harvesting_mask & (1u << channel)) != 0;
}

// Number of harvested DRAM channels with index strictly below `channel`. Maps a physical DRAM
// channel to its compacted/logical index via (physical - harvested_before): UMD presents only
// non-harvested channels, compacted, so a physical channel with a gap must be shifted down.
size_t harvested_before(uint32_t dram_harvesting_mask, size_t channel) {
    size_t count = 0;
    for (size_t c = 0; c < channel; ++c) {
        if (is_dram_channel_harvested(dram_harvesting_mask, c)) {
            ++count;
        }
    }
    return count;
}
}  // namespace

tt::tt_metal::xy_pair metal_SocDescriptor::get_preferred_worker_core_for_dram_view(int dram_view, uint8_t noc) const {
    TT_ASSERT(
        dram_view < this->dram_view_worker_cores.size(),
        "dram_view={} must be within range of dram_view_worker_cores.size={}",
        dram_view,
        this->dram_view_worker_cores.size());
    TT_ASSERT(noc < 2, "Only 2 NOCs supported, noc={} is out of range", noc);
    return this->dram_view_worker_cores.at(dram_view).at(noc);
};

bool metal_SocDescriptor::is_noc0_dram_endpoint(const tt::tt_metal::xy_pair& translated_coord) const {
    // dram_view_worker_cores (and thus get_preferred_worker_core_for_dram_view) holds TRANSLATED
    // coords, so this compares like-for-like against a TRANSLATED argument. See the header note.
    for (size_t dram_view = 0; dram_view < this->dram_view_worker_cores.size(); ++dram_view) {
        if (get_preferred_worker_core_for_dram_view(static_cast<int>(dram_view), /*noc=*/0) == translated_coord) {
            return true;
        }
    }
    return false;
}

std::vector<tt::tt_metal::xy_pair> metal_SocDescriptor::get_metal_dram_cores(tt::CoordSystem coord_system) const {
    // Blackhole reserves each DRAM view's NOC0 worker endpoint for the syseng firmware; no other
    // architecture has that restriction (and future ones won't), so the exclusion is confined to this
    // one spot rather than every DRAM loop in Metal.
    const bool exclude_noc0_endpoints = (this->arch == tt::ARCH::BLACKHOLE);
    std::vector<tt::tt_metal::xy_pair> dram_cores;
    const auto& umd_dram_cores = get_cores(tt::CoreType::DRAM, coord_system);
    dram_cores.reserve(umd_dram_cores.size());
    for (const tt::umd::CoreCoord& core : umd_dram_cores) {
        const tt::umd::CoreCoord translated = translate_coord_to(core, tt::CoordSystem::TRANSLATED);
        if (exclude_noc0_endpoints && is_noc0_dram_endpoint({translated.x, translated.y})) {
            continue;
        }
        // UMD's LOGICAL DRAM coord is {channel, raw subchannel}, but Metal's logical DRAM space is
        // {dram_view, index into dram_bank_endpoint_coords}, which orders the NOC0 worker endpoint
        // first rather than by subchannel id. Handing back the UMD coord would make a caller that
        // resolves it through get_physical_dram_core_from_logical land on a different core -- and for
        // any view whose worker_endpoint[0] is not subchannel 0, that core is the syseng-owned NOC0
        // endpoint this loop just excluded, whose mailbox is never initialized.
        if (coord_system == tt::CoordSystem::LOGICAL) {
            dram_cores.push_back(get_logical_dram_core_from_translated({translated.x, translated.y}));
        } else {
            dram_cores.push_back({core.x, core.y});
        }
    }
    return dram_cores;
}

tt::tt_metal::xy_pair metal_SocDescriptor::get_logical_dram_core_from_translated(
    const tt::tt_metal::xy_pair& translated_coord) const {
    // A view can share its NOC endpoints with the other views carved out of the same channel, so this
    // returns the lowest view index that reaches translated_coord. Every such coord resolves back to
    // translated_coord through get_physical_dram_core_from_logical, which is what callers rely on; use
    // get_logical_dram_core_for_subchannel instead when a specific view is wanted.
    for (size_t dram_view = 0; dram_view < this->dram_bank_endpoint_coords.size(); ++dram_view) {
        const auto& endpoints = this->dram_bank_endpoint_coords[dram_view];
        for (size_t idx = 0; idx < endpoints.size(); ++idx) {
            if (endpoints[idx] == translated_coord) {
                return tt::tt_metal::xy_pair{static_cast<uint32_t>(dram_view), static_cast<uint32_t>(idx)};
            }
        }
    }
    TT_THROW(
        "Translated DRAM core ({}, {}) is not a NOC endpoint of any of the {} DRAM views, so it has no logical "
        "DRAM coordinate. Every DRAM core Metal can address is reachable through some view, so this coord is "
        "either not a DRAM core or belongs to a harvested channel",
        translated_coord.x,
        translated_coord.y,
        this->dram_bank_endpoint_coords.size());
}

tt::tt_metal::xy_pair metal_SocDescriptor::get_preferred_eth_core_for_dram_view(int dram_view, uint8_t noc) const {
    TT_ASSERT(
        dram_view < this->dram_view_eth_cores.size(),
        "dram_view={} must be within range of dram_view_eth_cores.size={}",
        dram_view,
        this->dram_view_eth_cores.size());
    TT_ASSERT(noc < 2, "Only 2 NOCs supported, noc={} is out of range", noc);
    return this->dram_view_eth_cores.at(dram_view).at(noc);
};

tt::tt_metal::xy_pair metal_SocDescriptor::get_logical_core_for_dram_view(int dram_view) const {
    const uint32_t num_dram_views = this->get_num_dram_views();
    TT_FATAL(
        dram_view < num_dram_views,
        "dram_view={} must be within range of num_dram_views={}",
        dram_view,
        num_dram_views);
    return tt::tt_metal::xy_pair(dram_view, 0);
}

size_t metal_SocDescriptor::get_address_offset(int dram_view) const {
    TT_ASSERT(
        dram_view < this->dram_view_address_offsets.size(),
        "dram_view={} must be within range of dram_view_address_offsets.size={}",
        dram_view,
        this->dram_view_address_offsets.size());
    return this->dram_view_address_offsets.at(dram_view);
}

size_t metal_SocDescriptor::get_physical_channel_for_dram_view(int dram_view) const {
    TT_ASSERT(
        dram_view < this->dram_view_channels.size(),
        "dram_view={} must be within range of dram_view_channels.size={}",
        dram_view,
        this->dram_view_channels.size());
    return this->dram_view_channels.at(dram_view);
}

size_t metal_SocDescriptor::get_channel_for_dram_view(int dram_view) const {
    const size_t physical_channel = get_physical_channel_for_dram_view(dram_view);
    const uint32_t dram_harvesting_mask = this->harvesting_masks.dram_harvesting_mask;
    TT_ASSERT(
        !is_dram_channel_harvested(dram_harvesting_mask, physical_channel),
        "dram_view={} refers to harvested physical DRAM channel {}",
        dram_view,
        physical_channel);
    return physical_channel - harvested_before(dram_harvesting_mask, physical_channel);
}

size_t metal_SocDescriptor::get_num_dram_views() const { return this->dram_view_eth_cores.size(); }

int metal_SocDescriptor::get_dram_channel_from_logical_core(const tt::tt_metal::xy_pair& logical_coord) const {
    const uint32_t num_dram_views = this->get_num_dram_views();
    TT_FATAL(
        logical_coord.x < num_dram_views &&
            (dram_bank_endpoint_coords.empty() || logical_coord.y < dram_bank_endpoint_coords[logical_coord.x].size()),
        "Bounds-Error -- Logical DRAM core {} is outside valid range (num_views={}, endpoints_per_bank={})",
        logical_coord.str(),
        num_dram_views,
        dram_bank_endpoint_coords.empty() ? 1 : dram_bank_endpoint_coords[0].size());
    return logical_coord.x;
}

tt::tt_metal::xy_pair metal_SocDescriptor::get_physical_ethernet_core_from_logical(
    const tt::tt_metal::xy_pair& logical_coord) const {
    tt::umd::CoreCoord physical_coord =
        translate_coord_to({logical_coord, tt::CoreType::ETH, tt::CoordSystem::LOGICAL}, tt::CoordSystem::NOC0);
    return {physical_coord.x, physical_coord.y};
}

tt::tt_metal::xy_pair metal_SocDescriptor::get_logical_ethernet_core_from_physical(
    const tt::tt_metal::xy_pair& physical_coord) const {
    tt::umd::CoreCoord logical_coord =
        translate_coord_to({physical_coord, tt::CoreType::ETH, tt::CoordSystem::NOC0}, tt::CoordSystem::LOGICAL);
    return {logical_coord.x, logical_coord.y};
}

tt::tt_metal::xy_pair metal_SocDescriptor::get_physical_tensix_core_from_logical(
    const tt::tt_metal::xy_pair& logical_coord) const {
    tt::umd::CoreCoord physical_coord =
        translate_coord_to({logical_coord, tt::CoreType::TENSIX, tt::CoordSystem::LOGICAL}, tt::CoordSystem::NOC0);
    return {physical_coord.x, physical_coord.y};
}

tt::tt_metal::xy_pair metal_SocDescriptor::get_physical_dram_core_from_logical(
    const tt::tt_metal::xy_pair& logical_coord) const {
    TT_FATAL(
        logical_coord.x < dram_bank_endpoint_coords.size() &&
            logical_coord.y < dram_bank_endpoint_coords[logical_coord.x].size(),
        "Bounds-Error -- Logical DRAM core {} is outside dram_bank_endpoint_coords grid ({}x{})",
        logical_coord.str(),
        dram_bank_endpoint_coords.size(),
        dram_bank_endpoint_coords.empty() ? 0 : dram_bank_endpoint_coords[0].size());
    return dram_bank_endpoint_coords[logical_coord.x][logical_coord.y];
}

tt::tt_metal::xy_pair metal_SocDescriptor::get_logical_dram_core_for_subchannel(int dram_view, int subchannel) const {
    const int channel = static_cast<int>(get_channel_for_dram_view(dram_view));
    const tt::umd::CoreCoord phys_umd = get_dram_core_for_channel(channel, subchannel, tt::CoordSystem::TRANSLATED);
    const tt::tt_metal::xy_pair phys{phys_umd.x, phys_umd.y};
    TT_FATAL(
        dram_view >= 0 && static_cast<size_t>(dram_view) < dram_bank_endpoint_coords.size(),
        "dram_view {} out of range (num_views={})",
        dram_view,
        dram_bank_endpoint_coords.size());
    const auto& endpoints = dram_bank_endpoint_coords[static_cast<size_t>(dram_view)];
    for (size_t idx = 0; idx < endpoints.size(); ++idx) {
        if (endpoints[idx] == phys) {
            return tt::tt_metal::xy_pair{static_cast<uint32_t>(dram_view), static_cast<uint32_t>(idx)};
        }
    }
    TT_THROW(
        "DRAM subchannel {} on view {} (physical {}, {}) not found in dram_bank_endpoint_coords",
        subchannel,
        dram_view,
        phys.x,
        phys.y);
}

tt::tt_metal::xy_pair metal_SocDescriptor::get_physical_dispatch_engine_core_from_logical(
    const tt::tt_metal::xy_pair& logical_coord) const {
    const auto dispatch_noc0_cores = get_cores(tt::CoreType::DISPATCH, tt::CoordSystem::NOC0);
    TT_FATAL(
        logical_coord.y == 0,
        "Dispatch-engine logical y coordinate must be 0 (got {})",
        logical_coord.str());
    TT_FATAL(
        logical_coord.x < dispatch_noc0_cores.size(),
        "Dispatch-engine logical index {} out of range ({} dispatch cores in soc descriptor)",
        logical_coord.x,
        dispatch_noc0_cores.size());
    const tt::umd::CoreCoord& noc0_core = dispatch_noc0_cores[logical_coord.x];
    return {noc0_core.x, noc0_core.y};
}

uint32_t metal_SocDescriptor::get_num_dispatch_engine_cores() const {
    return static_cast<uint32_t>(get_cores(tt::CoreType::DISPATCH, tt::CoordSystem::NOC0).size());
}

tt::tt_metal::xy_pair metal_SocDescriptor::get_physical_core_from_logical_core(
    const tt::tt_metal::xy_pair& logical_coord, const tt::CoreType& core_type) const {
    switch (core_type) {
        case tt::CoreType::ETH: return this->get_physical_ethernet_core_from_logical(logical_coord);
        case tt::CoreType::WORKER: return this->get_physical_tensix_core_from_logical(logical_coord);
        case tt::CoreType::DRAM: return this->get_physical_dram_core_from_logical(logical_coord);
        case tt::CoreType::DISPATCH: return this->get_physical_dispatch_engine_core_from_logical(logical_coord);
        default: TT_THROW("Undefined conversion for core type.");
    }
}

tt::tt_metal::xy_pair metal_SocDescriptor::get_dram_grid_size() const {
    return tt::tt_metal::xy_pair(this->get_num_dram_views(), 1);
}

tt::tt_metal::xy_pair metal_SocDescriptor::get_dram_compute_grid_size() const {
    return tt::tt_metal::xy_pair(this->get_num_dram_views(), get_grid_size(tt::CoreType::DRAM).y);
}

void metal_SocDescriptor::load_dram_metadata_from_device_descriptor() {
    YAML::Node device_descriptor_yaml = YAML::LoadFile(this->device_descriptor_file_path);
    this->dram_view_size = device_descriptor_yaml["dram_view_size"].as<uint64_t>();
    const size_t num_dram_views_in_descriptor = device_descriptor_yaml["dram_views"].size();
    this->dram_core_size = num_dram_views_in_descriptor * this->dram_view_size;
    this->dram_view_channels.clear();
    this->dram_view_eth_cores.clear();
    this->dram_view_worker_cores.clear();
    this->dram_view_address_offsets.clear();
    this->dram_bank_endpoint_coords.clear();
    this->dram_view_channels.reserve(num_dram_views_in_descriptor);
    this->dram_view_eth_cores.reserve(num_dram_views_in_descriptor);
    this->dram_view_worker_cores.reserve(num_dram_views_in_descriptor);
    this->dram_view_address_offsets.reserve(num_dram_views_in_descriptor);
    this->dram_bank_endpoint_coords.reserve(num_dram_views_in_descriptor);

    const uint32_t dram_harvesting_mask = this->harvesting_masks.dram_harvesting_mask;

    for (const auto& dram_view : device_descriptor_yaml["dram_views"]) {
        size_t channel = dram_view["channel"].as<size_t>();
        if (is_dram_channel_harvested(dram_harvesting_mask, channel)) {
            continue;
        }
        const size_t logical_channel = channel - harvested_before(dram_harvesting_mask, channel);
        if (logical_channel >= get_grid_size(tt::CoreType::DRAM).x) {
            break;
        }
        size_t address_offset = dram_view["address_offset"].as<size_t>();

        const auto eth_endpoint_ids = dram_view["eth_endpoint"].as<std::vector<int>>();
        std::vector<tt::tt_metal::xy_pair> eth_dram_cores;
        std::vector<size_t> eth_endpoints;
        eth_dram_cores.reserve(eth_endpoint_ids.size());
        eth_endpoints.reserve(eth_endpoint_ids.size());
        for (int eth_endpoint : eth_endpoint_ids) {
            if (eth_endpoint >= get_grid_size(tt::CoreType::DRAM).y) {
                TT_THROW(
                    "DRAM subchannel {} does not exist in the device descriptor, but is specified in "
                    "dram_view.eth_endpoint",
                    eth_endpoint);
            }
            tt::umd::CoreCoord eth_dram_endpoint_coord =
                get_dram_core_for_channel(logical_channel, eth_endpoint, tt::CoordSystem::TRANSLATED);
            eth_dram_cores.push_back({eth_dram_endpoint_coord.x, eth_dram_endpoint_coord.y});
            eth_endpoints.push_back(eth_endpoint);
        }

        const auto worker_endpoint_ids = dram_view["worker_endpoint"].as<std::vector<int>>();
        std::vector<tt::tt_metal::xy_pair> worker_dram_cores;
        std::vector<size_t> worker_endpoints;
        worker_dram_cores.reserve(worker_endpoint_ids.size());
        worker_endpoints.reserve(worker_endpoint_ids.size());
        for (int worker_endpoint : worker_endpoint_ids) {
            if (worker_endpoint >= get_grid_size(tt::CoreType::DRAM).y) {
                TT_THROW(
                    "DRAM subchannel {} does not exist in the device descriptor, but is specified in "
                    "dram_view.worker_endpoint",
                    worker_endpoint);
            }
            tt::umd::CoreCoord worker_endpoint_coord =
                get_dram_core_for_channel(logical_channel, worker_endpoint, tt::CoordSystem::TRANSLATED);

            worker_dram_cores.push_back({worker_endpoint_coord.x, worker_endpoint_coord.y});
            worker_endpoints.push_back(worker_endpoint);
        }

        this->dram_view_channels.push_back(channel);
        this->dram_view_address_offsets.push_back(address_offset);
        this->dram_view_eth_cores.push_back(std::move(eth_dram_cores));
        this->dram_view_worker_cores.push_back(std::move(worker_dram_cores));

        // Order a bank's endpoints by role (see dram_bank_endpoint_coords): the worker endpoints in
        // NOC order first -- NOC0 at y == 0, the endpoint CMFW also uses for DRAM telemetry (SYS-1419)
        // -- then whatever subchannels are left, ascending. Endpoints that repeat a subchannel
        // (Wormhole declares the same one for both NOCs) are placed once, which leaves those
        // descriptors ordered exactly as before.
        const size_t num_subchannels = get_grid_size(tt::CoreType::DRAM).y;
        TT_FATAL(
            !worker_endpoints.empty(),
            "DRAM view {} declares no worker_endpoint, so its logical y=0 would not name the NOC0 worker endpoint",
            this->dram_view_channels.size() - 1);
        std::vector<bool> placed(num_subchannels, false);
        std::vector<tt::tt_metal::xy_pair> bank_endpoints;
        bank_endpoints.reserve(num_subchannels);
        const auto push_subchannel = [&](size_t sub) {
            placed[sub] = true;
            const tt::umd::CoreCoord coord =
                get_dram_core_for_channel(logical_channel, sub, tt::CoordSystem::TRANSLATED);
            bank_endpoints.push_back({coord.x, coord.y});
        };
        // worker_endpoints entries were bounds-checked above; each subchannel is visited once by
        // the second loop, so only the first needs the placed[] guard.
        for (const size_t worker_endpoint : worker_endpoints) {
            if (!placed[worker_endpoint]) {
                push_subchannel(worker_endpoint);
            }
        }
        for (size_t sub = 0; sub < num_subchannels; sub++) {
            if (!placed[sub]) {
                push_subchannel(sub);
            }
        }
        this->dram_bank_endpoint_coords.push_back(std::move(bank_endpoints));
    }
}

void metal_SocDescriptor::generate_logical_eth_coords_mapping() {
    for (const auto& logical_coord : this->get_cores(tt::CoreType::ETH, tt::CoordSystem::LOGICAL)) {
        this->logical_eth_core_to_chan_map.insert(
            {{logical_coord.x, logical_coord.y}, static_cast<int>(logical_coord.y)});
    }
}

void metal_SocDescriptor::generate_physical_routing_to_profiler_flat_id() {
#if defined(TRACY_ENABLE)
    for (auto& core : get_cores(tt::CoreType::TENSIX, tt::CoordSystem::NOC0)) {
        this->physical_routing_to_profiler_flat_id.emplace((tt::tt_metal::xy_pair){core.x, core.y}, 0);
    }

    for (auto& core : this->get_cores(tt::CoreType::ETH, tt::CoordSystem::NOC0)) {
        this->physical_routing_to_profiler_flat_id.emplace((tt::tt_metal::xy_pair){core.x, core.y}, 0);
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

// UMD initializes and owns SocDescriptor
// For architectures with translation tables enabled, UMD will remove the last x rows from the descriptors in
// SocDescriptor (workers list and worker_log_to_routing_x/y maps) This creates a virtual coordinate system, where
// translation tables are used to convert virtual core coordinates to the true harvesting state. For architectures
// without translation tables enabled, UMD updates SocDescriptor to contain the true harvesting state by
// removing the harvested physical coordinates Metal needs the true harvesting state so we generate physical
// descriptors from virtual coordinates We also initialize additional lookup tables to translate physical coordinates to
// virtual coordinates because UMD APIs expect virtual coordinates.
metal_SocDescriptor::metal_SocDescriptor(const SocDescriptor& other, const tt::BoardType& /*board_type*/) :
    SocDescriptor(other) {
    this->load_dram_metadata_from_device_descriptor();
    this->generate_logical_eth_coords_mapping();
    this->generate_physical_routing_to_profiler_flat_id();
}
