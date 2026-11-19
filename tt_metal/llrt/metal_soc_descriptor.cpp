// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "llrt/metal_soc_descriptor.hpp"

#include <algorithm>
#include <map>
#include <optional>
#include <string>
#include <tuple>
#include <utility>
#include <vector>
#include <tt_stl/assert.hpp>
#include <yaml-cpp/yaml.h>

#include <tt-umd/types/arch.hpp>
#include <tt-umd/tt_device/tt_device.hpp>

namespace {
// NOCs a DRAM view names a preferred endpoint for. Matches the `noc < 2` contract both
// preferred-endpoint accessors assert on, rather than being re-derived from one table's row length.
constexpr uint8_t k_num_dram_endpoint_nocs = 2;

// GDDR_MRISC_NOC2AXI_PORT: nibble i is the noc2axi port MRISC runs on for GDDR instance i, or
// kMriscPortHarvested. Ports are not dram_view endpoint indices; see blackhole_140_arch.yaml.
constexpr uint8_t kMriscPortsPerWord = 8;
constexpr uint32_t kMriscPortNibbleMask = 0xF;
constexpr uint32_t kMriscPortHarvested = 0xF;

uint32_t mrisc_port_for_channel(uint32_t mrisc_noc2axi_ports, size_t channel) {
    if (channel >= kMriscPortsPerWord) {
        return kMriscPortHarvested;
    }
    return (mrisc_noc2axi_ports >> (4 * channel)) & kMriscPortNibbleMask;
}

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

tt::tt_metal::CoreCoord metal_SocDescriptor::get_preferred_worker_core_for_dram_view(int dram_view, uint8_t noc) const {
    TT_ASSERT(
        dram_view < this->dram_view_worker_cores.size(),
        "dram_view={} must be within range of dram_view_worker_cores.size={}",
        dram_view,
        this->dram_view_worker_cores.size());
    TT_ASSERT(noc < 2, "Only 2 NOCs supported, noc={} is out of range", noc);
    return this->dram_view_worker_cores.at(dram_view).at(noc);
};

uint8_t metal_SocDescriptor::get_dram_endpoint_noc_mask(const tt::tt_metal::CoreCoord& translated_coord) const {
    // Both endpoint tables matter: worker and eth DRAM reads can route through different subchannels,
    // and a NIU serving either one has to keep forwarding to AXI. Both tables hold TRANSLATED coords
    // (see the header note), so this compares like-for-like against a TRANSLATED argument.
    uint8_t mask = 0;
    for (int dram_view = 0; dram_view < static_cast<int>(this->dram_view_worker_cores.size()); ++dram_view) {
        for (uint8_t noc = 0; noc < k_num_dram_endpoint_nocs; ++noc) {
            const bool is_view_endpoint = get_preferred_worker_core_for_dram_view(dram_view, noc) == translated_coord ||
                                          get_preferred_eth_core_for_dram_view(dram_view, noc) == translated_coord;
            if (is_view_endpoint) {
                mask |= static_cast<uint8_t>(1u << noc);
            }
        }
    }
    return mask;
}

bool metal_SocDescriptor::is_noc0_dram_endpoint(const tt::tt_metal::CoreCoord& translated_coord) const {
    return (get_dram_endpoint_noc_mask(translated_coord) & 0b1) != 0;
}

std::vector<tt::tt_metal::CoreCoord> metal_SocDescriptor::get_metal_dram_cores(tt::CoordSystem coord_system) const {
    // Blackhole reserves each DRAM view's NOC0 worker endpoint for the syseng firmware; no other
    // architecture has that restriction (and future ones won't), so the exclusion is confined to this
    // one spot rather than every DRAM loop in Metal.
    const bool exclude_noc0_endpoints = (this->arch == tt::ARCH::BLACKHOLE);
    std::vector<tt::tt_metal::CoreCoord> dram_cores;
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

tt::tt_metal::CoreCoord metal_SocDescriptor::get_logical_dram_core_from_translated(
    const tt::tt_metal::CoreCoord& translated_coord) const {
    // A view can share its NOC endpoints with the other views carved out of the same channel, so this
    // returns the lowest view index that reaches translated_coord. Every such coord resolves back to
    // translated_coord through get_physical_dram_core_from_logical, which is what callers rely on; use
    // get_logical_dram_core_for_subchannel instead when a specific view is wanted.
    for (size_t dram_view = 0; dram_view < this->dram_bank_endpoint_coords.size(); ++dram_view) {
        const auto& endpoints = this->dram_bank_endpoint_coords[dram_view];
        for (size_t idx = 0; idx < endpoints.size(); ++idx) {
            if (endpoints[idx] == translated_coord) {
                return tt::tt_metal::CoreCoord{static_cast<uint32_t>(dram_view), static_cast<uint32_t>(idx)};
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

tt::tt_metal::CoreCoord metal_SocDescriptor::get_preferred_eth_core_for_dram_view(int dram_view, uint8_t noc) const {
    TT_ASSERT(
        dram_view < this->dram_view_eth_cores.size(),
        "dram_view={} must be within range of dram_view_eth_cores.size={}",
        dram_view,
        this->dram_view_eth_cores.size());
    TT_ASSERT(noc < 2, "Only 2 NOCs supported, noc={} is out of range", noc);
    return this->dram_view_eth_cores.at(dram_view).at(noc);
};

tt::tt_metal::CoreCoord metal_SocDescriptor::get_logical_core_for_dram_view(int dram_view) const {
    const uint32_t num_dram_views = this->get_num_dram_views();
    TT_FATAL(
        dram_view < num_dram_views,
        "dram_view={} must be within range of num_dram_views={}",
        dram_view,
        num_dram_views);
    return tt::tt_metal::CoreCoord(dram_view, 0);
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

int metal_SocDescriptor::get_dram_channel_from_logical_core(const tt::tt_metal::CoreCoord& logical_coord) const {
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

tt::tt_metal::CoreCoord metal_SocDescriptor::get_physical_ethernet_core_from_logical(const tt::tt_metal::CoreCoord& logical_coord) const {
    tt::umd::CoreCoord physical_coord =
        translate_coord_to({logical_coord, tt::CoreType::ETH, tt::CoordSystem::LOGICAL}, tt::CoordSystem::NOC0);
    return {physical_coord.x, physical_coord.y};
}

tt::tt_metal::CoreCoord metal_SocDescriptor::get_logical_ethernet_core_from_physical(const tt::tt_metal::CoreCoord& physical_coord) const {
    tt::umd::CoreCoord logical_coord =
        translate_coord_to({physical_coord, tt::CoreType::ETH, tt::CoordSystem::NOC0}, tt::CoordSystem::LOGICAL);
    return {logical_coord.x, logical_coord.y};
}

tt::tt_metal::CoreCoord metal_SocDescriptor::get_physical_tensix_core_from_logical(const tt::tt_metal::CoreCoord& logical_coord) const {
    tt::umd::CoreCoord physical_coord =
        translate_coord_to({logical_coord, tt::CoreType::TENSIX, tt::CoordSystem::LOGICAL}, tt::CoordSystem::NOC0);
    return {physical_coord.x, physical_coord.y};
}

tt::tt_metal::CoreCoord metal_SocDescriptor::get_physical_dram_core_from_logical(const tt::tt_metal::CoreCoord& logical_coord) const {
    TT_FATAL(
        logical_coord.x < dram_bank_endpoint_coords.size() &&
            logical_coord.y < dram_bank_endpoint_coords[logical_coord.x].size(),
        "Bounds-Error -- Logical DRAM core {} is outside dram_bank_endpoint_coords grid ({}x{})",
        logical_coord.str(),
        dram_bank_endpoint_coords.size(),
        dram_bank_endpoint_coords.empty() ? 0 : dram_bank_endpoint_coords[0].size());
    return dram_bank_endpoint_coords[logical_coord.x][logical_coord.y];
}

tt::tt_metal::CoreCoord metal_SocDescriptor::get_logical_dram_core_for_subchannel(int dram_view, int subchannel) const {
    const int channel = static_cast<int>(get_channel_for_dram_view(dram_view));
    const tt::umd::CoreCoord phys_umd = get_dram_core_for_channel(channel, subchannel, tt::CoordSystem::TRANSLATED);
    const tt::tt_metal::CoreCoord phys{phys_umd.x, phys_umd.y};
    TT_FATAL(
        dram_view >= 0 && static_cast<size_t>(dram_view) < dram_bank_endpoint_coords.size(),
        "dram_view {} out of range (num_views={})",
        dram_view,
        dram_bank_endpoint_coords.size());
    const auto& endpoints = dram_bank_endpoint_coords[static_cast<size_t>(dram_view)];
    for (size_t idx = 0; idx < endpoints.size(); ++idx) {
        if (endpoints[idx] == phys) {
            return tt::tt_metal::CoreCoord{static_cast<uint32_t>(dram_view), static_cast<uint32_t>(idx)};
        }
    }
    TT_THROW(
        "DRAM subchannel {} on view {} (physical {}, {}) not found in dram_bank_endpoint_coords",
        subchannel,
        dram_view,
        phys.x,
        phys.y);
}

tt::tt_metal::CoreCoord metal_SocDescriptor::get_physical_dispatch_engine_core_from_logical(const tt::tt_metal::CoreCoord& logical_coord) const {
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

tt::tt_metal::CoreCoord metal_SocDescriptor::get_physical_core_from_logical_core(
    const tt::tt_metal::CoreCoord& logical_coord, const tt::CoreType& core_type) const {
    switch (core_type) {
        case tt::CoreType::ETH: return this->get_physical_ethernet_core_from_logical(logical_coord);
        case tt::CoreType::WORKER: return this->get_physical_tensix_core_from_logical(logical_coord);
        case tt::CoreType::DRAM: return this->get_physical_dram_core_from_logical(logical_coord);
        case tt::CoreType::DISPATCH: return this->get_physical_dispatch_engine_core_from_logical(logical_coord);
        default: TT_THROW("Undefined conversion for core type.");
    }
}

tt::tt_metal::CoreCoord metal_SocDescriptor::get_dram_grid_size() const { return tt::tt_metal::CoreCoord(this->get_num_dram_views(), 1); }

tt::tt_metal::CoreCoord metal_SocDescriptor::get_dram_compute_grid_size() const {
    return tt::tt_metal::CoreCoord(this->get_num_dram_views(), get_grid_size(tt::CoreType::DRAM).y);
}

std::optional<uint32_t> read_mrisc_noc2axi_ports(tt::umd::TTDevice* tt_device) {
    // Not in UMD's TelemetryTag enum, so it goes in by number.
    constexpr uint8_t kGddrMriscNoc2AxiPortTag = 72;

    if (tt_device == nullptr) {
        return std::nullopt;
    }
    auto* telemetry = tt_device->get_firmware_telemetry_reader();
    if (telemetry == nullptr || !telemetry->is_entry_available(kGddrMriscNoc2AxiPortTag)) {
        return std::nullopt;
    }
    return telemetry->read_entry(kGddrMriscNoc2AxiPortTag);
}

namespace {
// One DRAM view to load: the channel it reads (device-descriptor numbering, with harvested-channel gaps)
// and the device-descriptor entry that holds its endpoints and address offset.
struct DramViewEntry {
    size_t channel = 0;
    YAML::Node entry;
};

// One endpoint table of the device descriptor: the keys its entries keep their endpoints under, and its
// views in Metal view order.
struct DramEndpointTable {
    const char* name = "";
    const char* eth_endpoint_key = "";
    const char* worker_endpoint_key = "";
    std::vector<DramViewEntry> views;
};

// dram_views entries of the unharvested channels, in channel order.
std::vector<DramViewEntry> dram_views_in_channel_order(
    const YAML::Node& device_descriptor_yaml, uint32_t dram_harvesting_mask, size_t num_dram_channels) {
    std::vector<DramViewEntry> views;
    for (const auto& dram_view : device_descriptor_yaml["dram_views"]) {
        const size_t channel = dram_view["channel"].as<size_t>();
        if (is_dram_channel_harvested(dram_harvesting_mask, channel)) {
            continue;
        }
        if (channel - harvested_before(dram_harvesting_mask, channel) >= num_dram_channels) {
            break;
        }
        views.push_back({channel, dram_view});
    }
    return views;
}

// harvested_dram_views entries, entry k placed on the k-th unharvested channel in translated order: by
// the translated coord of the channel's subchannel 0, x first, then y. Empty when no channel is
// harvested, the descriptor has no harvested_dram_views, NOC translation is off, or some channel's
// subchannels are not one translated column with subchannel s at s rows below subchannel 0. In those
// cases entry k would not reach the same cores whichever channel is harvested.
std::vector<DramViewEntry> harvested_dram_views_in_translated_order(
    const tt::umd::SocDescriptor& soc_desc,
    const YAML::Node& device_descriptor_yaml,
    const std::string& descriptor_path) {
    const uint32_t dram_harvesting_mask = soc_desc.harvesting_masks.dram_harvesting_mask;
    const YAML::Node harvested_dram_views = device_descriptor_yaml["harvested_dram_views"];
    if (dram_harvesting_mask == 0 || !harvested_dram_views || !soc_desc.noc_translation_enabled) {
        return {};
    }
    const auto dram_grid = soc_desc.get_grid_size(tt::CoreType::DRAM);
    // {translated coord of subchannel 0, channel}
    std::vector<std::pair<tt::tt_metal::CoreCoord, size_t>> channels_by_position;
    channels_by_position.reserve(dram_grid.x);
    for (size_t channel = 0, logical_channel = 0; logical_channel < dram_grid.x; ++channel) {
        if (is_dram_channel_harvested(dram_harvesting_mask, channel)) {
            continue;
        }
        const tt::umd::CoreCoord first =
            soc_desc.get_dram_core_for_channel(static_cast<int>(logical_channel), 0, tt::CoordSystem::TRANSLATED);
        for (size_t sub = 1; sub < dram_grid.y; ++sub) {
            const tt::umd::CoreCoord coord = soc_desc.get_dram_core_for_channel(
                static_cast<int>(logical_channel), static_cast<int>(sub), tt::CoordSystem::TRANSLATED);
            if (coord.x != first.x || coord.y != first.y + sub) {
                return {};
            }
        }
        channels_by_position.push_back({{first.x, first.y}, channel});
        ++logical_channel;
    }
    std::sort(channels_by_position.begin(), channels_by_position.end(), [](const auto& a, const auto& b) {
        return std::tie(a.first.x, a.first.y) < std::tie(b.first.x, b.first.y);
    });
    TT_FATAL(
        channels_by_position.size() == harvested_dram_views.size(),
        "{} has {} harvested_dram_views entries, but DRAM harvesting mask {:#x} leaves {} channels",
        descriptor_path,
        harvested_dram_views.size(),
        dram_harvesting_mask,
        channels_by_position.size());
    std::vector<DramViewEntry> views;
    views.reserve(channels_by_position.size());
    for (size_t view = 0; view < channels_by_position.size(); ++view) {
        views.push_back({channels_by_position[view].second, harvested_dram_views[view]});
    }
    return views;
}

// The noc2axi port of each subchannel, per channel, from the dram_views noc2axi_ports lists.
std::map<size_t, std::vector<uint32_t>> noc2axi_ports_by_channel(const YAML::Node& device_descriptor_yaml) {
    std::map<size_t, std::vector<uint32_t>> ports;
    for (const auto& dram_view : device_descriptor_yaml["dram_views"]) {
        if (dram_view["noc2axi_ports"]) {
            ports.emplace(dram_view["channel"].as<size_t>(), dram_view["noc2axi_ports"].as<std::vector<uint32_t>>());
        }
    }
    return ports;
}

// True if every view's noc0 worker endpoint is on the noc2axi port CMFW reports MRISC on for that view's
// channel. A channel reported as kMriscPortHarvested (a disabled GDDR instance, even one the descriptor
// still lists) runs no MRISC, so it rules no table out.
bool noc0_endpoints_on_mrisc_ports(
    const DramEndpointTable& table,
    const std::map<size_t, std::vector<uint32_t>>& noc2axi_ports,
    uint32_t mrisc_noc2axi_ports,
    const std::string& descriptor_path) {
    for (const DramViewEntry& view : table.views) {
        const uint32_t mrisc_port = mrisc_port_for_channel(mrisc_noc2axi_ports, view.channel);
        if (mrisc_port == kMriscPortHarvested) {
            continue;
        }
        const YAML::Node worker_endpoints = view.entry[table.worker_endpoint_key];
        TT_FATAL(
            worker_endpoints && worker_endpoints.size() > 0,
            "The {} DRAM endpoint table in {} has no '{}' for channel {}",
            table.name,
            descriptor_path,
            table.worker_endpoint_key,
            view.channel);
        const auto noc0_subchannel = worker_endpoints[0].as<size_t>();
        const auto channel_ports = noc2axi_ports.find(view.channel);
        TT_FATAL(
            channel_ports != noc2axi_ports.end() && noc0_subchannel < channel_ports->second.size(),
            "dram_views in {} gives no noc2axi_ports entry for subchannel {} of channel {}, so the {} table's "
            "noc0 endpoint cannot be checked against MRISC",
            descriptor_path,
            noc0_subchannel,
            view.channel,
            table.name);
        if (channel_ports->second[noc0_subchannel] != mrisc_port) {
            return false;
        }
    }
    return true;
}

// BH-76 makes the noc2axi port CMFW loads MRISC on the noc0 endpoint of its GDDR instance, and a noc on
// any other port MRISC owns hangs the chip (SYS-1419). So load the first table whose noc0 endpoints sit
// on the ports CMFW reports, and refuse to guess when none does.
size_t select_dram_endpoint_table(
    const std::vector<DramEndpointTable>& tables,
    const YAML::Node& device_descriptor_yaml,
    uint32_t mrisc_noc2axi_ports,
    const std::string& descriptor_path) {
    const auto noc2axi_ports = noc2axi_ports_by_channel(device_descriptor_yaml);
    std::string table_names;
    for (size_t i = 0; i < tables.size(); ++i) {
        if (noc0_endpoints_on_mrisc_ports(tables[i], noc2axi_ports, mrisc_noc2axi_ports, descriptor_path)) {
            return i;
        }
        table_names += (i == 0 ? "" : ", ");
        table_names += tables[i].name;
    }
    TT_THROW(
        "CMFW reports MRISC on noc2axi ports {:#010x} (nibble i is GDDR instance i), which match the noc0 "
        "endpoints of none of the DRAM endpoint tables in {} ({}). Refusing to guess: a noc on a port MRISC "
        "owns hangs the chip (SYS-1419).",
        mrisc_noc2axi_ports,
        descriptor_path,
        table_names);
}
}  // namespace

void metal_SocDescriptor::load_dram_metadata_from_device_descriptor(
    const std::optional<uint32_t>& mrisc_noc2axi_ports, std::optional<uint64_t> dram_view_size_limit) {
    YAML::Node device_descriptor_yaml = YAML::LoadFile(this->device_descriptor_file_path);
    this->dram_view_size = device_descriptor_yaml["dram_view_size"].as<uint64_t>();
    if (dram_view_size_limit.has_value()) {
        this->dram_view_size = std::min(this->dram_view_size, *dram_view_size_limit);
    }
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

    std::vector<DramEndpointTable> tables;
    tables.push_back(
        {"pre-relocation",
         "eth_endpoint",
         "worker_endpoint",
         dram_views_in_channel_order(
             device_descriptor_yaml,
             this->harvesting_masks.dram_harvesting_mask,
             get_grid_size(tt::CoreType::DRAM).x)});
    size_t selected_table = 0;
    // No telemetry means CMFW older than 19.12, which runs MRISC where the pre-relocation table expects it.
    if (this->arch == tt::ARCH::BLACKHOLE && mrisc_noc2axi_ports.has_value()) {
        tables.push_back({"relocated", "relocated_eth_endpoint", "relocated_worker_endpoint", tables.front().views});
        auto harvested_views =
            harvested_dram_views_in_translated_order(*this, device_descriptor_yaml, this->device_descriptor_file_path);
        if (!harvested_views.empty()) {
            tables.push_back({"harvested", "eth_endpoint", "worker_endpoint", std::move(harvested_views)});
        }
        selected_table = select_dram_endpoint_table(
            tables, device_descriptor_yaml, mrisc_noc2axi_ports.value(), this->device_descriptor_file_path);
    }

    const DramEndpointTable& table = tables[selected_table];
    for (const DramViewEntry& view : table.views) {
        add_dram_view(view.channel, view.entry, table.eth_endpoint_key, table.worker_endpoint_key);
    }
}

void metal_SocDescriptor::add_dram_view(
    size_t channel, const YAML::Node& dram_view, const char* eth_endpoint_key, const char* worker_endpoint_key) {
    const size_t logical_channel = channel - harvested_before(this->harvesting_masks.dram_harvesting_mask, channel);
    size_t address_offset = dram_view["address_offset"].as<size_t>();

    TT_FATAL(
        dram_view[eth_endpoint_key] && dram_view[worker_endpoint_key],
        "DRAM view for channel {} is missing '{}'/'{}' in {}",
        channel,
        eth_endpoint_key,
        worker_endpoint_key,
        this->device_descriptor_file_path);

    const auto eth_endpoint_ids = dram_view[eth_endpoint_key].as<std::vector<int>>();
    std::vector<tt::tt_metal::CoreCoord> eth_dram_cores;
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

    const auto worker_endpoint_ids = dram_view[worker_endpoint_key].as<std::vector<int>>();
    std::vector<tt::tt_metal::CoreCoord> worker_dram_cores;
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
    std::vector<tt::tt_metal::CoreCoord> bank_endpoints;
    bank_endpoints.reserve(num_subchannels);
    const auto push_subchannel = [&](size_t sub) {
        placed[sub] = true;
        const tt::umd::CoreCoord coord = get_dram_core_for_channel(logical_channel, sub, tt::CoordSystem::TRANSLATED);
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

void metal_SocDescriptor::generate_logical_eth_coords_mapping() {
    for (const auto& logical_coord : this->get_cores(tt::CoreType::ETH, tt::CoordSystem::LOGICAL)) {
        this->logical_eth_core_to_chan_map.insert(
            {{logical_coord.x, logical_coord.y}, static_cast<int>(logical_coord.y)});
    }
}

void metal_SocDescriptor::generate_physical_routing_to_profiler_flat_id() {
#if defined(TRACY_ENABLE)
    for (auto& core : get_cores(tt::CoreType::TENSIX, tt::CoordSystem::NOC0)) {
        this->physical_routing_to_profiler_flat_id.emplace((tt::tt_metal::CoreCoord){core.x, core.y}, 0);
    }

    for (auto& core : this->get_cores(tt::CoreType::ETH, tt::CoordSystem::NOC0)) {
        this->physical_routing_to_profiler_flat_id.emplace((tt::tt_metal::CoreCoord){core.x, core.y}, 0);
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
metal_SocDescriptor::metal_SocDescriptor(
    const SocDescriptor& other,
    const tt::BoardType& /*board_type*/,
    std::optional<uint32_t> mrisc_noc2axi_ports,
    std::optional<uint64_t> dram_view_size_limit) :
    SocDescriptor(other) {
    this->load_dram_metadata_from_device_descriptor(mrisc_noc2axi_ports, dram_view_size_limit);
    this->generate_logical_eth_coords_mapping();
    this->generate_physical_routing_to_profiler_flat_id();
}
