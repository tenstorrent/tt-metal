// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Topology Visualizer: Terminal-based visualization of physical topology
// and ethernet link status for TT-Metal devices

#include <iostream>
#include <iomanip>
#include <vector>
#include <map>
#include <set>
#include <string>
#include <sstream>
#include <algorithm>
#include <memory>

// TT-UMD includes
#include "umd/device/cluster.hpp"
#include "umd/device/cluster_descriptor.hpp"
#include "umd/device/soc_descriptor.hpp"
#include "umd/device/types/cluster_descriptor_types.hpp"
#include "umd/device/types/core_coordinates.hpp"

// TT-Metal includes
#include <tt-metalium/host_api.hpp>

using namespace tt;
using namespace tt::umd;

// ANSI Color codes
namespace Color {
const char* RESET = "\033[0m";
const char* BOLD = "\033[1m";
const char* GREEN = "\033[32m";
const char* RED = "\033[31m";
const char* YELLOW = "\033[33m";
const char* CYAN = "\033[36m";
const char* BLUE = "\033[34m";
const char* MAGENTA = "\033[35m";
const char* WHITE = "\033[37m";
const char* DIM = "\033[2m";
}  // namespace Color

// Ethernet link status
enum class LinkStatus {
    CONNECTED,     // Link exists in topology
    NOT_CONNECTED  // No link defined
};

struct EthernetLink {
    ChipId src_chip;
    uint32_t src_channel;
    ChipId dst_chip;
    uint32_t dst_channel;
    LinkStatus status;
};

struct SpatialLocation {
    uint64_t asic_id;  // Unique ASIC ID
    std::string hostname;
    uint32_t tray_id;
    uint32_t asic_location;
    bool has_spatial_info;  // True if tray_id and asic_location are valid
};

struct ChipTopology {
    std::vector<ChipId> chips;
    std::map<ChipId, std::vector<EthernetLink>> outgoing_links;
    std::map<ChipId, std::string> chip_names;
    std::map<ChipId, tt::ARCH> chip_archs;
    std::map<ChipId, BoardType> chip_board_types;
    std::map<ChipId, SpatialLocation> chip_spatial_locations;
};

// Include SVG export after ChipTopology is defined
#include "topology_svg_export.hpp"
#include "topology_svg_detailed.hpp"
#include "topology_3d_export.hpp"

std::string get_arch_name(tt::ARCH arch) {
    switch (arch) {
        case tt::ARCH::GRAYSKULL: return "Grayskull";
        case tt::ARCH::WORMHOLE_B0: return "Wormhole B0";
        case tt::ARCH::BLACKHOLE: return "Blackhole";
        case tt::ARCH::Invalid: return "Invalid";
        default: return "Unknown";
    }
}

std::string get_board_type_name(BoardType board_type) {
    switch (board_type) {
        case BoardType::N150: return "N150";
        case BoardType::N300: return "N300";
        case BoardType::GALAXY: return "Galaxy";
        case BoardType::P100: return "P100";
        case BoardType::P150: return "P150";
        case BoardType::P300: return "P300";
        case BoardType::UBB: return "UBB";
        case BoardType::UNKNOWN: return "Unknown";
        default: return "Unknown";
    }
}

// Check ethernet link status based on cluster descriptor
// Note: This checks if a link is defined in the topology, not runtime firmware status
// For runtime status, devices need to be initialized first (see system_health tool)
LinkStatus check_ethernet_link_status(ClusterDescriptor* cluster_desc, ChipId chip_id, uint32_t channel) {
    // Check if this channel has an active ethernet link defined in topology
    if (cluster_desc->ethernet_core_has_active_ethernet_link(chip_id, channel)) {
        return LinkStatus::CONNECTED;
    }
    return LinkStatus::NOT_CONNECTED;
}

ChipTopology discover_topology(Cluster* cluster) {
    ChipTopology topology;

    ClusterDescriptor* cluster_desc = cluster->get_cluster_description();
    const auto& eth_connections = cluster_desc->get_ethernet_connections();

    // Get all chips
    auto chip_ids = cluster->get_target_device_ids();
    topology.chips = std::vector<ChipId>(chip_ids.begin(), chip_ids.end());
    std::sort(topology.chips.begin(), topology.chips.end());

    // Get unique ASIC IDs (spatial identifiers)
    const auto& unique_chip_ids = cluster_desc->get_chip_unique_ids();

    // Get chip information
    for (ChipId chip_id : topology.chips) {
        const SocDescriptor& soc_desc = cluster->get_soc_descriptor(chip_id);
        topology.chip_archs[chip_id] = soc_desc.arch;
        topology.chip_board_types[chip_id] = cluster_desc->get_board_type(chip_id);

        // Get spatial location information
        SpatialLocation spatial_loc;
        if (unique_chip_ids.find(chip_id) != unique_chip_ids.end()) {
            spatial_loc.asic_id = unique_chip_ids.at(chip_id);
            spatial_loc.hostname = "localhost";

            // Extract tray_id and asic_location from unique ASIC ID
            // UBB layout: chip_id maps to physical tray/location
            // For a 32-chip system with 4 trays of 8 chips each:
            // Tray 1 (0-7), Tray 2 (8-15), Tray 3 (16-23), Tray 4 (24-31)
            spatial_loc.tray_id = (chip_id / 8) + 1;        // Trays numbered 1-4
            spatial_loc.asic_location = (chip_id % 8) + 1;  // Locations numbered 1-8
            spatial_loc.has_spatial_info = true;
        } else {
            spatial_loc.asic_id = chip_id;
            spatial_loc.has_spatial_info = false;
            spatial_loc.hostname = "localhost";
            spatial_loc.tray_id = 0;
            spatial_loc.asic_location = chip_id;
        }
        topology.chip_spatial_locations[chip_id] = spatial_loc;

        std::stringstream name_ss;
        name_ss << "Chip " << chip_id << " (0x" << std::hex << spatial_loc.asic_id << std::dec << ")";
        topology.chip_names[chip_id] = name_ss.str();
    }

    // Discover ethernet links
    for (ChipId chip_id : topology.chips) {
        const SocDescriptor& soc_desc = cluster->get_soc_descriptor(chip_id);
        uint32_t num_channels = soc_desc.get_num_eth_channels();

        for (uint32_t channel = 0; channel < num_channels; channel++) {
            // Check if this channel has an active ethernet link
            if (cluster_desc->ethernet_core_has_active_ethernet_link(chip_id, channel)) {
                if (eth_connections.find(chip_id) != eth_connections.end() &&
                    eth_connections.at(chip_id).find(channel) != eth_connections.at(chip_id).end()) {
                    auto [dst_chip, dst_channel] =
                        cluster_desc->get_chip_and_channel_of_remote_ethernet_core(chip_id, channel);

                    // Check link status (based on topology, not runtime firmware)
                    LinkStatus status = check_ethernet_link_status(cluster_desc, chip_id, channel);

                    EthernetLink link;
                    link.src_chip = chip_id;
                    link.src_channel = channel;
                    link.dst_chip = dst_chip;
                    link.dst_channel = dst_channel;
                    link.status = status;

                    topology.outgoing_links[chip_id].push_back(link);
                }
            }
        }
    }

    return topology;
}

void print_chip_details(const ChipTopology& topology) {
    std::cout << Color::BOLD << Color::CYAN
              << "\n╔════════════════════════════════════════════════════════════════════════════════════════════╗\n";
    std::cout << "║                                    CHIP DETAILS                                            ║\n";
    std::cout << "╚════════════════════════════════════════════════════════════════════════════════════════════╝"
              << Color::RESET << "\n\n";

    std::cout << std::setw(8) << "Chip ID" << std::setw(20) << "Unique ASIC ID" << std::setw(15) << "Architecture"
              << std::setw(15) << "Board Type" << std::setw(8) << "Tray" << std::setw(10) << "Location" << std::setw(15)
              << "Eth Channels"
              << "\n";
    std::cout << std::string(91, '-') << "\n";

    for (ChipId chip_id : topology.chips) {
        const auto& spatial_loc = topology.chip_spatial_locations.at(chip_id);

        // Format ASIC ID as hex
        std::stringstream asic_id_str;
        asic_id_str << "0x" << std::hex << std::setfill('0') << std::setw(12) << spatial_loc.asic_id;

        std::cout << std::dec << std::setfill(' ') << std::setw(8) << chip_id << std::setw(20) << asic_id_str.str()
                  << std::setw(15) << get_arch_name(topology.chip_archs.at(chip_id)) << std::setw(15)
                  << get_board_type_name(topology.chip_board_types.at(chip_id)) << std::setw(8) << spatial_loc.tray_id
                  << std::setw(10) << spatial_loc.asic_location;

        size_t num_links = topology.outgoing_links.count(chip_id) ? topology.outgoing_links.at(chip_id).size() : 0;
        std::cout << std::setw(15) << num_links << "\n";
    }
    std::cout << "\n";
}

void print_topology_matrix(const ChipTopology& topology) {
    std::cout << Color::BOLD << Color::CYAN << "\n╔══════════════════════════════════════════════════════════════╗\n";
    std::cout << "║              ETHERNET LINK TOPOLOGY MATRIX                   ║\n";
    std::cout << "╚══════════════════════════════════════════════════════════════╝" << Color::RESET << "\n\n";

    // Build connectivity matrix
    std::map<std::pair<ChipId, ChipId>, std::vector<EthernetLink>> connections;

    for (const auto& [chip_id, links] : topology.outgoing_links) {
        for (const auto& link : links) {
            connections[{link.src_chip, link.dst_chip}].push_back(link);
        }
    }

    // Print header
    std::cout << "     ";
    for (ChipId chip_id : topology.chips) {
        std::cout << std::setw(6) << chip_id;
    }
    std::cout << "\n";
    std::cout << "    " << std::string(topology.chips.size() * 6 + 1, '-') << "\n";

    // Print matrix
    for (ChipId src_chip : topology.chips) {
        std::cout << std::setw(3) << src_chip << " |";

        for (ChipId dst_chip : topology.chips) {
            if (src_chip == dst_chip) {
                std::cout << Color::DIM << std::setw(6) << "-" << Color::RESET;
            } else {
                auto key = std::make_pair(src_chip, dst_chip);
                if (connections.count(key) > 0) {
                    size_t num_links = connections[key].size();
                    bool all_connected = true;
                    bool any_connected = false;

                    for (const auto& link : connections[key]) {
                        if (link.status == LinkStatus::CONNECTED) {
                            any_connected = true;
                        } else {
                            all_connected = false;
                        }
                    }

                    if (all_connected) {
                        std::cout << Color::GREEN << std::setw(6) << num_links << Color::RESET;
                    } else if (any_connected) {
                        std::cout << Color::YELLOW << std::setw(6) << num_links << Color::RESET;
                    } else {
                        std::cout << Color::RED << std::setw(6) << num_links << Color::RESET;
                    }
                } else {
                    std::cout << std::setw(6) << ".";
                }
            }
        }
        std::cout << "\n";
    }

    std::cout << "\n";
    std::cout << "Legend: " << Color::GREEN << "■" << Color::RESET << " All links defined  " << Color::YELLOW << "■"
              << Color::RESET << " Partial  " << Color::RED << "■" << Color::RESET << " Not defined  "
              << ". No connection\n";
    std::cout << "Numbers indicate count of ethernet links between chips\n";
    std::cout << Color::DIM
              << "Note: Shows topology from cluster descriptor. Use 'system_health' for runtime link status.\n"
              << Color::RESET << "\n";
}

void print_ascii_topology_graph(const ChipTopology& topology) {
    std::cout << Color::BOLD << Color::CYAN << "\n╔══════════════════════════════════════════════════════════════╗\n";
    std::cout << "║              ASCII TOPOLOGY GRAPH                            ║\n";
    std::cout << "╚══════════════════════════════════════════════════════════════╝" << Color::RESET << "\n\n";

    const int CHIPS_PER_ROW = 8;

    std::vector<ChipId> sorted_chips = topology.chips;
    std::sort(sorted_chips.begin(), sorted_chips.end());

    // Build connection map for quick lookup
    std::map<std::pair<ChipId, ChipId>, int> connection_count;
    for (const auto& [chip_id, links] : topology.outgoing_links) {
        for (const auto& link : links) {
            if (link.status == LinkStatus::CONNECTED) {
                auto key = std::make_pair(std::min(chip_id, link.dst_chip), std::max(chip_id, link.dst_chip));
                connection_count[key]++;
            }
        }
    }

    // Draw chips in rows
    for (size_t row_start = 0; row_start < sorted_chips.size(); row_start += CHIPS_PER_ROW) {
        size_t row_end = std::min(row_start + CHIPS_PER_ROW, sorted_chips.size());

        // Draw top of boxes
        for (size_t i = row_start; i < row_end; i++) {
            std::cout << "┌─────┐";
            if (i < row_end - 1) {
                std::cout << "  ";
            }
        }
        std::cout << "\n";

        // Draw chip IDs
        for (size_t i = row_start; i < row_end; i++) {
            std::cout << "│" << Color::CYAN << std::setw(3) << sorted_chips[i] << Color::RESET << "  │";
            if (i < row_end - 1) {
                // Check if there's a connection to next chip
                ChipId chip1 = sorted_chips[i];
                ChipId chip2 = sorted_chips[i + 1];
                auto key = std::make_pair(std::min(chip1, chip2), std::max(chip1, chip2));
                if (connection_count.count(key) > 0) {
                    std::cout << Color::GREEN << "──" << Color::RESET;
                } else {
                    std::cout << "  ";
                }
            }
        }
        std::cout << "\n";

        // Draw bottom of boxes
        for (size_t i = row_start; i < row_end; i++) {
            std::cout << "└─────┘";
            if (i < row_end - 1) {
                std::cout << "  ";
            }
        }
        std::cout << "\n";

        // Draw vertical connections to next row (if exists)
        if (row_end < sorted_chips.size()) {
            for (size_t i = row_start; i < row_end; i++) {
                ChipId current_chip = sorted_chips[i];
                bool has_connection = false;

                // Check connections to chips in next row
                for (size_t j = row_end; j < std::min(row_end + CHIPS_PER_ROW, sorted_chips.size()); j++) {
                    ChipId next_chip = sorted_chips[j];
                    auto key = std::make_pair(std::min(current_chip, next_chip), std::max(current_chip, next_chip));
                    if (connection_count.count(key) > 0) {
                        has_connection = true;
                        break;
                    }
                }

                if (has_connection) {
                    std::cout << "   " << Color::GREEN << "│" << Color::RESET << "   ";
                } else {
                    std::cout << "       ";
                }
                if (i < row_end - 1) {
                    std::cout << "  ";
                }
            }
            std::cout << "\n";
        }
    }

    std::cout << "\n";
    std::cout << Color::GREEN << "──" << Color::RESET << " / " << Color::GREEN << "│" << Color::RESET
              << " = Connected chips    ";
    std::cout << "Box shows chip ID\n";
    std::cout << Color::DIM << "Note: Only showing adjacent connections. See matrix for full topology.\n"
              << Color::RESET << "\n";
}

void print_detailed_links(const ChipTopology& topology) {
    std::cout << Color::BOLD << Color::CYAN << "\n╔══════════════════════════════════════════════════════════════╗\n";
    std::cout << "║              DETAILED ETHERNET LINK STATUS                   ║\n";
    std::cout << "╚══════════════════════════════════════════════════════════════╝" << Color::RESET << "\n\n";

    for (ChipId chip_id : topology.chips) {
        if (topology.outgoing_links.count(chip_id) == 0 || topology.outgoing_links.at(chip_id).empty()) {
            continue;
        }

        std::cout << Color::BOLD << "Chip " << chip_id << Color::RESET << "\n";

        const auto& links = topology.outgoing_links.at(chip_id);
        for (const auto& link : links) {
            std::cout << "  CH" << std::setw(2) << link.src_channel << " -> ";
            std::cout << "Chip " << std::setw(2) << link.dst_chip << " CH" << std::setw(2) << link.dst_channel;
            std::cout << "  [";

            if (link.status == LinkStatus::CONNECTED) {
                std::cout << Color::GREEN << "CONNECTED" << Color::RESET;
            } else {
                std::cout << Color::RED << "NOT_CONNECTED" << Color::RESET;
            }

            std::cout << "]\n";
        }
        std::cout << "\n";
    }
}

void print_summary(const ChipTopology& topology) {
    size_t total_links = 0;
    size_t connected_links = 0;
    size_t not_connected_links = 0;

    for (const auto& [chip_id, links] : topology.outgoing_links) {
        for (const auto& link : links) {
            total_links++;
            if (link.status == LinkStatus::CONNECTED) {
                connected_links++;
            } else if (link.status == LinkStatus::NOT_CONNECTED) {
                not_connected_links++;
            }
        }
    }

    std::cout << Color::BOLD << Color::CYAN << "\n╔══════════════════════════════════════════════════════════════╗\n";
    std::cout << "║              SUMMARY                                         ║\n";
    std::cout << "╚══════════════════════════════════════════════════════════════╝" << Color::RESET << "\n\n";

    std::cout << "Total Chips:              " << topology.chips.size() << "\n";
    std::cout << "Total Ethernet Links:     " << total_links << "\n";
    std::cout << Color::GREEN << "Links Defined:            " << connected_links << Color::RESET << "\n";
    std::cout << Color::RED << "Links Not Defined:        " << not_connected_links << Color::RESET << "\n";

    if (total_links > 0) {
        double connectivity_pct = (100.0 * connected_links) / total_links;
        std::cout << "Topology Connectivity:    " << std::fixed << std::setprecision(1) << connectivity_pct << "%\n";
    }

    std::cout << "\n";
    std::cout << Color::YELLOW << "Note: For runtime link status (UP/DOWN), use:" << Color::RESET << "\n";
    std::cout << "      ./build/tools/umd/system_health\n";
    std::cout << "\n";
}

int main(int argc, char* argv[]) {
    try {
        std::cout << Color::BOLD << Color::BLUE;
        std::cout << "╔═══════════════════════════════════════════════════════════╗\n";
        std::cout << "║  TT-Metal Physical Topology & Ethernet Link Visualizer  ║\n";
        std::cout << "╚═══════════════════════════════════════════════════════════╝\n";
        std::cout << Color::RESET << "\n";

        std::cout << "Initializing cluster...\n";
        std::unique_ptr<Cluster> cluster = std::make_unique<Cluster>();

        std::cout << "Discovering topology...\n";
        ChipTopology topology = discover_topology(cluster.get());

        if (topology.chips.empty()) {
            std::cerr << Color::RED << "Error: No chips found in cluster!\n" << Color::RESET;
            return 1;
        }

        // Print all visualizations
        print_chip_details(topology);
        print_ascii_topology_graph(topology);
        print_topology_matrix(topology);
        print_detailed_links(topology);
        print_summary(topology);

        // Export SVG visualizations
        std::cout << "\n";
        TopologySVGExporter::export_to_svg(topology, "topology_visualization.svg");
        std::cout << "\n";
        TopologyDetailedSVGExporter::export_detailed_svg(topology, "topology_detailed.svg");
        std::cout << "\n";
        Topology3DExporter::export_3d_html(topology, "topology_3d.html");

        std::cout << "\n" << Color::GREEN << "Topology visualization complete!\n" << Color::RESET;

        return 0;
    } catch (const std::exception& e) {
        std::cerr << Color::RED << "Error: " << e.what() << "\n" << Color::RESET;
        return 1;
    }
}
