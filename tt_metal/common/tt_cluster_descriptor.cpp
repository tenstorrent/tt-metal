
#include "tt_cluster_descriptor.h"

#include <fstream>
#include <memory>
#include <sstream>

#include "common/assert.hpp"
#include "device/tt_device.h"
#include "yaml-cpp/yaml.h"

// NOTE: It might be worthwhile to precompute this for every pair of directly connected chips, depending on how extensively router needs to use it
std::vector<std::tuple<ethernet_channel_t, ethernet_channel_t>> tt_ClusterDescriptor::get_directly_connected_ethernet_channels_between_chips(const chip_id_t &first, const chip_id_t &second) const {
    std::vector<std::tuple<ethernet_channel_t, ethernet_channel_t>> directly_connected_channels = {};
    if (this->enabled_active_chips.find(first) == this->enabled_active_chips.end() || this->enabled_active_chips.find(second) == this->enabled_active_chips.end()) {
        return {};
    }

    for (const auto &[first_ethernet_channel, connected_chip_and_channel] : this->ethernet_connections.at(first)) {
        if (std::get<0>(connected_chip_and_channel) == second) {
            directly_connected_channels.push_back({first_ethernet_channel, std::get<1>(connected_chip_and_channel)});
        }
    }

    return directly_connected_channels;
}

// const CoreCoord tt_ClusterDescriptor::get_chip_xy(const chip_id_t &chip_id) const {
//     // For now we only support a 1D cluster, so the mapping is trivial (where the chip ID is the x value of the xy location)
//     return CoreCoord(chip_id, 0);
// }

// const chip_id_t tt_ClusterDescriptor::get_chip_id_at_location(const CoreCoord &chip_location) const {
//     // For now we only support a 1D cluster, so the mapping is trivial (where the chip ID is the x value of the xy location)
//     return chip_location.x;
// }

bool tt_ClusterDescriptor::is_chip_mmio_capable(const chip_id_t &chip_id) const {
    return this->chips_with_mmio.find(chip_id) != this->chips_with_mmio.end();
}

chip_id_t tt_ClusterDescriptor::get_closest_mmio_capable_chip(const chip_id_t &chip) const {
    // For now we will assume a logically connected grid (chips only connect to those with IDs +/- 1 in any given coordinate direction, although later)
    // we may wish to use the ethernet channel connectivity to support more generic cluster topologies.
    int min_distance = std::numeric_limits<int>::max();
    chip_id_t closest_chip = chip;
    for (const chip_id_t &c : this->chips_with_mmio) {
        int distance = std::abs(c - chip);
        if (distance < min_distance) {
            min_distance = distance;
            closest_chip = c;
        }
    }

    TT_ASSERT(is_chip_mmio_capable(closest_chip));

    return closest_chip;

}

std::unique_ptr<tt_ClusterDescriptor> tt_ClusterDescriptor::create_from_yaml(const std::string &cluster_descriptor_file_path) {
    std::unique_ptr<tt_ClusterDescriptor> desc = std::unique_ptr<tt_ClusterDescriptor>(new tt_ClusterDescriptor());

    std::ifstream fdesc(cluster_descriptor_file_path);
    if (fdesc.fail()) {
        throw std::runtime_error("Error: cluster connectivity descriptor file " + cluster_descriptor_file_path + " does not exist!");
    }
    fdesc.close();

    YAML::Node yaml = YAML::LoadFile(cluster_descriptor_file_path);
    tt_ClusterDescriptor::load_chips_from_connectivity_descriptor(yaml, *desc);
    tt_ClusterDescriptor::load_ethernet_connections_from_connectivity_descriptor(yaml, *desc);

    desc->enable_all_devices();

    return desc;
}

std::unique_ptr<tt_ClusterDescriptor> tt_ClusterDescriptor::create_for_grayskull_cluster(
    const std::set<chip_id_t> &target_device_ids) {
    std::unique_ptr<tt_ClusterDescriptor> desc = std::unique_ptr<tt_ClusterDescriptor>(new tt_ClusterDescriptor());

    for (const chip_id_t &chip : target_device_ids) {
        desc->chips_with_mmio.insert(chip);
        desc->all_chips.insert(chip);
    }

    desc->enable_all_devices();

    return desc;
}

std::set<chip_id_t> get_sequential_chip_id_set(int num_chips) {
    std::set<chip_id_t> chip_ids;
    for (int i = 0; i < num_chips; ++i) {
        chip_ids.insert(static_cast<chip_id_t>(i));
    }
    return chip_ids;
}

void tt_ClusterDescriptor::load_ethernet_connections_from_connectivity_descriptor(YAML::Node &yaml, tt_ClusterDescriptor &desc) {
    TT_ASSERT(yaml["ethernet_connections"].IsSequence());
    for (YAML::Node &connected_endpoints : yaml["ethernet_connections"].as<std::vector<YAML::Node>>()) {
        TT_ASSERT(connected_endpoints.IsSequence());

        std::vector<YAML::Node> endpoints = connected_endpoints.as<std::vector<YAML::Node>>();
        TT_ASSERT(endpoints.size() == 2, "Currently ethernet cores can only connect to one other ethernet endpoint");

        int chip_0 = endpoints.at(0)["chip"].as<int>();
        int channel_0 = endpoints.at(0)["chan"].as<int>();
        int chip_1 = endpoints.at(1)["chip"].as<int>();
        int channel_1 = endpoints.at(1)["chan"].as<int>();
        desc.ethernet_connections[chip_0][channel_0] = {chip_1, channel_1};
        desc.ethernet_connections[chip_1][channel_1] = {chip_0, channel_0};
    }

    tt::log_debug(tt::LogDevice, "Ethernet Connectivity Descriptor:");
    for (const auto &[chip, chan_to_chip_chan_map] : desc.ethernet_connections) {
        for (const auto &[chan, chip_and_chan] : chan_to_chip_chan_map) {
            tt::log_debug(tt::LogDevice, "\tchip: {}, chan: {}  <-->  chip: {}, chan: {}", chip, chan, std::get<0>(chip_and_chan), std::get<1>(chip_and_chan));
        }
    }
}

void tt_ClusterDescriptor::load_chips_from_connectivity_descriptor(YAML::Node &yaml, tt_ClusterDescriptor &desc) {
    for (YAML::const_iterator node = yaml["chips"].begin(); node != yaml["chips"].end(); ++node) {
        chip_id_t chip_id = node->first.as<int>();
        CoreCoord chip_location = format_node(node->second.as<std::string>());
        desc.chip_locations.insert({chip_id, chip_location});
        desc.all_chips.insert(chip_id);
    }
    for (int chip_id : yaml["chips_with_mmio"].as<std::vector<int>>()) {
        desc.chips_with_mmio.insert(chip_id);
    }
    tt::log_debug(tt::LogDevice, "Device IDs and Locations:");
    for (const auto &[chip_id, chip_location] : desc.get_chip_locations()) {
        tt::log_debug(tt::LogDevice, "\tchip: {}, location: {}", chip_id, chip_location.str());
    }
}


void tt_ClusterDescriptor::specify_enabled_devices(const std::vector<chip_id_t> &chip_ids) {
    this->enabled_active_chips.clear();
    for (auto chip_id : chip_ids) {
        this->enabled_active_chips.insert(chip_id);
    }
}

void tt_ClusterDescriptor::enable_all_devices() {
    this->enabled_active_chips = this->all_chips;
}

bool tt_ClusterDescriptor::chips_have_ethernet_connectivity() const {
    return ethernet_connections.size() > 0;
}


std::unordered_map<chip_id_t, std::unordered_map<ethernet_channel_t, std::tuple<chip_id_t, ethernet_channel_t> > > tt_ClusterDescriptor::get_ethernet_connections() const {
    auto eth_connections = std::unordered_map<chip_id_t, std::unordered_map<ethernet_channel_t, std::tuple<chip_id_t, ethernet_channel_t> > >();

    for (const auto &[chip, channel_mapping] : this->ethernet_connections) {
        if (this->enabled_active_chips.find(chip) != this->enabled_active_chips.end()) {
            eth_connections[chip] = {};
            for (const auto &[src_channel, chip_channel] : channel_mapping) {
                const auto &[dest_chip, dest_channel] = chip_channel;
                if (this->enabled_active_chips.find(dest_chip) != this->enabled_active_chips.end()) {
                    eth_connections[chip][src_channel] = chip_channel;

                }
            }
        }
    }
    return eth_connections;
}

std::unordered_map<chip_id_t, CoreCoord> tt_ClusterDescriptor::get_chip_locations() const {
    auto locations = std::unordered_map<chip_id_t, CoreCoord>();
    for (auto chip_id : this->enabled_active_chips) {
        locations[chip_id] = chip_locations.at(chip_id);
    }

    return locations;
}


std::unordered_set<chip_id_t> tt_ClusterDescriptor::get_chips_with_mmio() const {
    auto chips = std::unordered_set<chip_id_t>();
    for (auto chip_id : chips_with_mmio) {
        if (this->enabled_active_chips.find(chip_id) != this->enabled_active_chips.end()) {
            chips.insert(chip_id);
        }
    }

    return chips;
}


std::unordered_set<chip_id_t> tt_ClusterDescriptor::get_all_chips() const {
    return this->enabled_active_chips;
}

std::size_t tt_ClusterDescriptor::get_number_of_chips() const {
    return this->enabled_active_chips.size();
}
