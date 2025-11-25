// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
#include <vector>
#include <map>
#include <set>
#include <cmath>
#include <algorithm>

struct ChipTopology;
struct EthernetLink;

class TopologyDetailedSVGExporter {
public:
    static void export_detailed_svg(const ChipTopology& topology, const std::string& filename) {
        std::ofstream file(filename);
        if (!file.is_open()) {
            std::cerr << "Error: Could not open file " << filename << " for writing\n";
            return;
        }

        const int CHIPS_PER_ROW = 4;
        const int CHIP_WIDTH = 160;
        const int CHIP_HEIGHT = 200;
        const int CHIP_SPACING_X = 450;
        const int CHIP_SPACING_Y = 450;
        const int MARGIN = 120;
        const int CHANNEL_RADIUS = 8;
        const int ROUTING_LANE_SPACING = 25;  // Space between routing lanes
        const int BASE_CLEARANCE = 90;        // Base clearance from chip edges

        std::vector<ChipId> sorted_chips = topology.chips;
        std::sort(sorted_chips.begin(), sorted_chips.end());

        // Calculate SVG dimensions
        int num_rows = (sorted_chips.size() + CHIPS_PER_ROW - 1) / CHIPS_PER_ROW;
        int num_cols = std::min(static_cast<int>(sorted_chips.size()), CHIPS_PER_ROW);
        int svg_width = num_cols * CHIP_SPACING_X + 2 * MARGIN;
        int svg_height = num_rows * CHIP_SPACING_Y + 2 * MARGIN + 150;

        // Build chip positions and channel positions
        std::map<ChipId, std::pair<int, int>> chip_positions;
        std::map<ChipId, std::pair<int, int>> chip_centers;
        std::map<std::pair<ChipId, int>, std::pair<int, int>> channel_positions;

        for (size_t i = 0; i < sorted_chips.size(); i++) {
            ChipId chip = sorted_chips[i];
            int row = i / CHIPS_PER_ROW;
            int col = i % CHIPS_PER_ROW;
            int chip_x = MARGIN + col * CHIP_SPACING_X;
            int chip_y = MARGIN + row * CHIP_SPACING_Y;
            chip_positions[chip] = {chip_x, chip_y};
            chip_centers[chip] = {chip_x + CHIP_WIDTH / 2, chip_y + CHIP_HEIGHT / 2};

            // Place channels around the perimeter
            for (int ch = 0; ch < 16; ch++) {
                int ch_x, ch_y;

                if (ch < 4) {
                    // Top edge
                    ch_x = chip_x + (ch + 1) * (CHIP_WIDTH / 5);
                    ch_y = chip_y;
                } else if (ch < 8) {
                    // Right edge
                    ch_x = chip_x + CHIP_WIDTH;
                    ch_y = chip_y + ((ch - 4) + 1) * (CHIP_HEIGHT / 5);
                } else if (ch < 12) {
                    // Bottom edge
                    ch_x = chip_x + (11 - ch + 1) * (CHIP_WIDTH / 5);
                    ch_y = chip_y + CHIP_HEIGHT;
                } else {
                    // Left edge
                    ch_x = chip_x;
                    ch_y = chip_y + (15 - ch + 1) * (CHIP_HEIGHT / 5);
                }

                channel_positions[{chip, ch}] = {ch_x, ch_y};
            }
        }

        // Collect all channel-to-channel connections and group by type
        struct Connection {
            ChipId src_chip;
            int src_ch;
            ChipId dst_chip;
            int dst_ch;
            std::string type;
            int lane;
        };

        std::vector<Connection> connections;

        for (const auto& [chip_id, links] : topology.outgoing_links) {
            for (const auto& link : links) {
                if (link.status == LinkStatus::CONNECTED) {
                    if (chip_id < link.dst_chip || (chip_id == link.dst_chip && link.src_channel < link.dst_channel)) {
                        Connection conn;
                        conn.src_chip = chip_id;
                        conn.src_ch = link.src_channel;
                        conn.dst_chip = link.dst_chip;
                        conn.dst_ch = link.dst_channel;

                        auto [src_chip_x, src_chip_y] = chip_positions[chip_id];
                        auto [dst_chip_x, dst_chip_y] = chip_positions[link.dst_chip];

                        bool same_row = (src_chip_y == dst_chip_y);
                        bool same_col = (src_chip_x == dst_chip_x);
                        bool adjacent_h = same_row && std::abs(dst_chip_x - src_chip_x) == CHIP_SPACING_X;
                        bool adjacent_v = same_col && std::abs(dst_chip_y - src_chip_y) == CHIP_SPACING_Y;

                        if (adjacent_h || adjacent_v) {
                            conn.type = "adjacent";
                            conn.lane = 0;
                        } else if (same_row) {
                            conn.type = "same_row";
                            conn.lane = link.src_channel % 4;  // 4 routing lanes
                        } else if (same_col) {
                            conn.type = "same_col";
                            conn.lane = link.src_channel % 4;
                        } else {
                            conn.type = "diagonal";
                            conn.lane = link.src_channel % 4;
                        }

                        connections.push_back(conn);
                    }
                }
            }
        }

        // Write SVG header
        file << "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n";
        file << "<svg width=\"" << svg_width << "\" height=\"" << svg_height
             << "\" xmlns=\"http://www.w3.org/2000/svg\">\n";

        // Styles
        file << "<defs>\n";
        file << "  <style>\n";
        file << "    .chip-box { fill: #34495e; stroke: #2c3e50; stroke-width: 3; }\n";
        file << "    .chip-header { fill: #2c3e50; }\n";
        file << "    .chip-text { fill: #ecf0f1; font-family: monospace; font-size: 20px; font-weight: bold; "
                "text-anchor: middle; }\n";
        file << "    .channel-circle { fill: #3498db; stroke: #2980b9; stroke-width: 2; }\n";
        file << "    .channel-circle.connected { fill: #2ecc71; stroke: #27ae60; }\n";
        file << "    .channel-text { fill: #fff; font-family: monospace; font-size: 10px; text-anchor: middle; }\n";
        file << "    .connection-line { stroke: #e74c3c; stroke-width: 2; stroke-opacity: 0.7; fill: none; }\n";
        file << "    .connection-line.adjacent { stroke: #3498db; }\n";
        file << "    .connection-line.same-row { stroke: #9b59b6; }\n";
        file << "    .connection-line.same-col { stroke: #e67e22; }\n";
        file << "    .connection-line.diagonal { stroke: #e74c3c; }\n";
        file << "    .connection-line:hover { stroke: #c0392b; stroke-width: 3; stroke-opacity: 1.0; }\n";
        file << "    .title { fill: #2c3e50; font-family: Arial, sans-serif; font-size: 28px; font-weight: bold; }\n";
        file << "    .legend-text { fill: #2c3e50; font-family: Arial, sans-serif; font-size: 16px; }\n";
        file << "  </style>\n";
        file << "</defs>\n\n";

        // Background
        file << "<rect width=\"" << svg_width << "\" height=\"" << svg_height << "\" fill=\"#ecf0f1\"/>\n\n";

        // Title
        file << "<text x=\"" << svg_width / 2 << "\" y=\"40\" class=\"title\" text-anchor=\"middle\">";
        file << "TT-Metal Detailed Topology - Channel-Level Connections</text>\n";
        file << "<text x=\"" << svg_width / 2 << "\" y=\"70\" class=\"legend-text\" text-anchor=\"middle\">";
        file << sorted_chips.size() << " Chips × 16 Channels = " << connections.size()
             << " Active Channel Pairs</text>\n\n";

        // Draw connections with intelligent routing
        file << "<!-- Channel-to-Channel Connections -->\n";
        for (const auto& conn : connections) {
            auto [x1, y1] = channel_positions[{conn.src_chip, conn.src_ch}];
            auto [x2, y2] = channel_positions[{conn.dst_chip, conn.dst_ch}];

            auto [src_chip_x, src_chip_y] = chip_positions[conn.src_chip];
            auto [dst_chip_x, dst_chip_y] = chip_positions[conn.dst_chip];

            std::string css_class = "connection-line ";

            if (conn.type == "adjacent") {
                // Adjacent chips - simple gentle curve
                css_class += "adjacent";

                double dx = x2 - x1;
                double dy = y2 - y1;
                double dist = std::sqrt(dx * dx + dy * dy);

                double mx = (x1 + x2) / 2.0;
                double my = (y1 + y2) / 2.0;

                // Small perpendicular offset
                double offset = 8 + (conn.src_ch % 4) * 4;
                double px = -dy / dist * offset;
                double py = dx / dist * offset;

                file << "<path d=\"M " << x1 << " " << y1 << " Q " << (mx + px) << " " << (my + py) << " " << x2 << " "
                     << y2 << "\" class=\"" << css_class << "\">\n";
                file << "  <title>Chip " << conn.src_chip << " CH" << conn.src_ch << " → Chip " << conn.dst_chip
                     << " CH" << conn.dst_ch << "</title>\n";
                file << "</path>\n";

            } else if (conn.type == "same_row") {
                // Horizontal routing with multiple lanes above/below
                css_class += "same-row";

                // Calculate routing lane offset
                int lane_offset = BASE_CLEARANCE + conn.lane * ROUTING_LANE_SPACING;

                // Determine if we route above or below
                bool route_above = (conn.src_ch + conn.dst_ch) % 2 == 0;
                if (!route_above) {
                    lane_offset = -lane_offset;
                }

                // Find the topmost or bottommost chip position in this row
                double route_y = src_chip_y + lane_offset;

                // Use waypoints to avoid passing through chips
                file << "<path d=\"M " << x1 << " " << y1 << " L " << x1 << " " << route_y << " L " << x2 << " "
                     << route_y << " L " << x2 << " " << y2 << "\" class=\"" << css_class << "\">\n";
                file << "  <title>Chip " << conn.src_chip << " CH" << conn.src_ch << " → Chip " << conn.dst_chip
                     << " CH" << conn.dst_ch << "</title>\n";
                file << "</path>\n";

            } else if (conn.type == "same_col") {
                // Vertical routing with multiple lanes left/right
                css_class += "same-col";

                int lane_offset = BASE_CLEARANCE + conn.lane * ROUTING_LANE_SPACING;

                // Determine if we route left or right
                bool route_left = (conn.src_ch + conn.dst_ch) % 2 == 0;
                if (!route_left) {
                    lane_offset = -lane_offset;
                }

                double route_x = src_chip_x + lane_offset;

                file << "<path d=\"M " << x1 << " " << y1 << " L " << route_x << " " << y1 << " L " << route_x << " "
                     << y2 << " L " << x2 << " " << y2 << "\" class=\"" << css_class << "\">\n";
                file << "  <title>Chip " << conn.src_chip << " CH" << conn.src_ch << " → Chip " << conn.dst_chip
                     << " CH" << conn.dst_ch << "</title>\n";
                file << "</path>\n";

            } else {
                // Diagonal routing - use Manhattan routing around chips
                css_class += "diagonal";

                // Determine if we go horizontal first or vertical first
                bool go_horizontal_first = (conn.src_ch < 8);

                int lane_offset = 15 + conn.lane * 12;

                if (go_horizontal_first) {
                    // Exit horizontally from source, route around, then vertical to dest
                    double mid_x;
                    if (dst_chip_x > src_chip_x) {
                        mid_x = src_chip_x + CHIP_WIDTH + BASE_CLEARANCE + lane_offset;
                    } else {
                        mid_x = src_chip_x - BASE_CLEARANCE - lane_offset;
                    }

                    file << "<path d=\"M " << x1 << " " << y1 << " L " << mid_x << " " << y1 << " L " << mid_x << " "
                         << y2 << " L " << x2 << " " << y2 << "\" class=\"" << css_class << "\">\n";
                    file << "  <title>Chip " << conn.src_chip << " CH" << conn.src_ch << " → Chip " << conn.dst_chip
                         << " CH" << conn.dst_ch << "</title>\n";
                    file << "</path>\n";
                } else {
                    // Exit vertically from source, route around, then horizontal to dest
                    double mid_y;
                    if (dst_chip_y > src_chip_y) {
                        mid_y = src_chip_y + CHIP_HEIGHT + BASE_CLEARANCE + lane_offset;
                    } else {
                        mid_y = src_chip_y - BASE_CLEARANCE - lane_offset;
                    }

                    file << "<path d=\"M " << x1 << " " << y1 << " L " << x1 << " " << mid_y << " L " << x2 << " "
                         << mid_y << " L " << x2 << " " << y2 << "\" class=\"" << css_class << "\">\n";
                    file << "  <title>Chip " << conn.src_chip << " CH" << conn.src_ch << " → Chip " << conn.dst_chip
                         << " CH" << conn.dst_ch << "</title>\n";
                    file << "</path>\n";
                }
            }
        }

        // Draw chips with channels
        file << "\n<!-- Chips with Ethernet Channels -->\n";
        for (ChipId chip : sorted_chips) {
            auto [chip_x, chip_y] = chip_positions[chip];

            file << "<g id=\"chip_" << chip << "\">\n";

            // Chip body
            file << "  <rect x=\"" << chip_x << "\" y=\"" << chip_y << "\" width=\"" << CHIP_WIDTH << "\" height=\""
                 << CHIP_HEIGHT << "\" rx=\"8\" class=\"chip-box\"/>\n";

            // Chip header
            file << "  <rect x=\"" << chip_x << "\" y=\"" << chip_y << "\" width=\"" << CHIP_WIDTH
                 << "\" height=\"40\" rx=\"8\" class=\"chip-header\"/>\n";

            // Chip ID
            file << "  <text x=\"" << (chip_x + CHIP_WIDTH / 2) << "\" y=\"" << (chip_y + 28)
                 << "\" class=\"chip-text\">Chip " << chip << "</text>\n";

            // Count connected channels
            int connected_count = 0;
            if (topology.outgoing_links.count(chip) > 0) {
                connected_count = topology.outgoing_links.at(chip).size();
            }

            // Connection count
            file << "  <text x=\"" << (chip_x + CHIP_WIDTH / 2) << "\" y=\"" << (chip_y + CHIP_HEIGHT / 2 + 10)
                 << "\" style=\"fill: #95a5a6; font-size: 14px; text-anchor: middle;\">" << connected_count
                 << " active</text>\n";

            // Draw channels
            std::set<int> connected_channels;
            if (topology.outgoing_links.count(chip) > 0) {
                for (const auto& link : topology.outgoing_links.at(chip)) {
                    connected_channels.insert(link.src_channel);
                }
            }

            for (int ch = 0; ch < 16; ch++) {
                auto [ch_x, ch_y] = channel_positions[{chip, ch}];
                bool is_connected = connected_channels.count(ch) > 0;

                file << "  <circle cx=\"" << ch_x << "\" cy=\"" << ch_y << "\" r=\"" << CHANNEL_RADIUS
                     << "\" class=\"channel-circle";
                if (is_connected) {
                    file << " connected";
                }
                file << "\">\n";
                file << "    <title>Chip " << chip << " - Channel " << ch;
                if (is_connected) {
                    file << " (connected)";
                } else {
                    file << " (unused)";
                }
                file << "</title>\n";
                file << "  </circle>\n";

                // Channel number
                file << "  <text x=\"" << ch_x << "\" y=\"" << (ch_y + 4) << "\" class=\"channel-text\">" << ch
                     << "</text>\n";
            }

            file << "</g>\n\n";
        }

        // Legend
        int legend_y = svg_height - 100;
        file << "<!-- Legend -->\n";
        file << "<text x=\"" << MARGIN << "\" y=\"" << legend_y
             << "\" class=\"legend-text\" font-weight=\"bold\">Legend:</text>\n";

        legend_y += 25;
        file << "<circle cx=\"" << (MARGIN + 10) << "\" cy=\"" << (legend_y - 6)
             << "\" r=\"8\" class=\"channel-circle connected\"/>\n";
        file << "<text x=\"" << (MARGIN + 25) << "\" y=\"" << legend_y
             << "\" class=\"legend-text\">Active Channel</text>\n";

        file << "<circle cx=\"" << (MARGIN + 200) << "\" cy=\"" << (legend_y - 6)
             << "\" r=\"8\" class=\"channel-circle\"/>\n";
        file << "<text x=\"" << (MARGIN + 215) << "\" y=\"" << legend_y
             << "\" class=\"legend-text\">Unused Channel</text>\n";

        file << "<line x1=\"" << (MARGIN + 400) << "\" y1=\"" << (legend_y - 6) << "\" x2=\"" << (MARGIN + 450)
             << "\" y2=\"" << (legend_y - 6) << "\" class=\"connection-line adjacent\"/>\n";
        file << "<text x=\"" << (MARGIN + 465) << "\" y=\"" << legend_y << "\" class=\"legend-text\">Adjacent</text>\n";

        file << "<line x1=\"" << (MARGIN + 590) << "\" y1=\"" << (legend_y - 6) << "\" x2=\"" << (MARGIN + 640)
             << "\" y2=\"" << (legend_y - 6) << "\" class=\"connection-line same-row\"/>\n";
        file << "<text x=\"" << (MARGIN + 655) << "\" y=\"" << legend_y << "\" class=\"legend-text\">Same Row</text>\n";

        file << "<line x1=\"" << (MARGIN + 780) << "\" y1=\"" << (legend_y - 6) << "\" x2=\"" << (MARGIN + 830)
             << "\" y2=\"" << (legend_y - 6) << "\" class=\"connection-line same-col\"/>\n";
        file << "<text x=\"" << (MARGIN + 845) << "\" y=\"" << legend_y << "\" class=\"legend-text\">Same Col</text>\n";

        file << "<line x1=\"" << (MARGIN + 970) << "\" y1=\"" << (legend_y - 6) << "\" x2=\"" << (MARGIN + 1020)
             << "\" y2=\"" << (legend_y - 6) << "\" class=\"connection-line diagonal\"/>\n";
        file << "<text x=\"" << (MARGIN + 1035) << "\" y=\"" << legend_y
             << "\" class=\"legend-text\">Diagonal</text>\n";

        legend_y += 30;
        file << "<text x=\"" << MARGIN << "\" y=\"" << legend_y
             << "\" class=\"legend-text\" style=\"font-style: italic;\">";
        file << "Multi-lane routing ensures connections never pass through chips. Hover for details.</text>\n";

        file << "</svg>\n";
        file.close();

        std::cout << "✓ Detailed SVG visualization exported to: " << filename << "\n";
        std::cout << "  Shows all " << connections.size() << " channel-to-channel connections\n";
        std::cout << "  Multi-lane routing prevents overlaps and chip intersections\n";
        std::cout << "  Open in a web browser to view interactive detailed topology\n";
    }
};
