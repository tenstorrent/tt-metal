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
#include <cmath>

struct ChipTopology;

class TopologySVGExporter {
public:
    static void export_to_svg(const ChipTopology& topology, const std::string& filename) {
        std::ofstream file(filename);
        if (!file.is_open()) {
            std::cerr << "Error: Could not open file " << filename << " for writing\n";
            return;
        }

        const int CHIPS_PER_ROW = 8;
        const int CHIP_SIZE = 100;
        const int CHIP_SPACING_X = 220;
        const int CHIP_SPACING_Y = 200;
        const int MARGIN = 80;

        std::vector<ChipId> sorted_chips = topology.chips;
        std::sort(sorted_chips.begin(), sorted_chips.end());

        // Calculate SVG dimensions
        int num_rows = (sorted_chips.size() + CHIPS_PER_ROW - 1) / CHIPS_PER_ROW;
        int num_cols = std::min(static_cast<int>(sorted_chips.size()), CHIPS_PER_ROW);
        int svg_width = num_cols * CHIP_SPACING_X + 2 * MARGIN;
        int svg_height = num_rows * CHIP_SPACING_Y + 2 * MARGIN + 100;  // Extra for legend

        // Build chip positions
        std::map<ChipId, std::pair<int, int>> chip_positions;
        for (size_t i = 0; i < sorted_chips.size(); i++) {
            int row = i / CHIPS_PER_ROW;
            int col = i % CHIPS_PER_ROW;
            int x = MARGIN + col * CHIP_SPACING_X + CHIP_SIZE / 2;
            int y = MARGIN + row * CHIP_SPACING_Y + CHIP_SIZE / 2;
            chip_positions[sorted_chips[i]] = {x, y};
        }

        // Collect connections
        std::map<std::pair<ChipId, ChipId>, int> connections;
        for (const auto& [chip_id, links] : topology.outgoing_links) {
            for (const auto& link : links) {
                if (link.status == LinkStatus::CONNECTED) {
                    auto key = std::make_pair(std::min(chip_id, link.dst_chip), std::max(chip_id, link.dst_chip));
                    connections[key]++;
                }
            }
        }

        // Write SVG header
        file << "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n";
        file << "<svg width=\"" << svg_width << "\" height=\"" << svg_height
             << "\" xmlns=\"http://www.w3.org/2000/svg\">\n";

        // Add styles
        file << "<defs>\n";
        file << "  <style>\n";
        file << "    .chip-box { fill: #2c3e50; stroke: #3498db; stroke-width: 3; }\n";
        file << "    .chip-box:hover { fill: #34495e; stroke: #e74c3c; stroke-width: 4; }\n";
        file << "    .chip-text { fill: #ecf0f1; font-family: monospace; font-size: 22px; text-anchor: middle; "
                "font-weight: bold; }\n";
        file << "    .link-weak { stroke: #95a5a6; stroke-width: 2; stroke-opacity: 0.5; }\n";
        file << "    .link-medium { stroke: #3498db; stroke-width: 3; stroke-opacity: 0.6; }\n";
        file << "    .link-strong { stroke: #2ecc71; stroke-width: 4; stroke-opacity: 0.7; }\n";
        file << "    .link-very-strong { stroke: #e74c3c; stroke-width: 5; stroke-opacity: 0.8; }\n";
        file << "    .title { fill: #2c3e50; font-family: Arial, sans-serif; font-size: 28px; font-weight: bold; }\n";
        file << "    .legend-text { fill: #2c3e50; font-family: Arial, sans-serif; font-size: 16px; }\n";
        file << "  </style>\n";
        file << "</defs>\n\n";

        // Background
        file << "<rect width=\"" << svg_width << "\" height=\"" << svg_height << "\" fill=\"#ecf0f1\"/>\n\n";

        // Title
        file << "<text x=\"" << svg_width / 2 << "\" y=\"30\" class=\"title\" text-anchor=\"middle\">";
        file << "TT-Metal Chip Topology - " << sorted_chips.size() << " Chips, " << connections.size()
             << " Unique Connections</text>\n\n";

        // Draw connections first (so they appear behind chips)
        file << "<!-- Connections -->\n";
        for (const auto& [conn_pair, link_count] : connections) {
            auto [chip1, chip2] = conn_pair;
            auto [x1, y1] = chip_positions[chip1];
            auto [x2, y2] = chip_positions[chip2];

            // Determine connection strength class
            std::string conn_class;
            if (link_count <= 2) {
                conn_class = "link-weak";
            } else if (link_count <= 4) {
                conn_class = "link-medium";
            } else if (link_count <= 6) {
                conn_class = "link-strong";
            } else {
                conn_class = "link-very-strong";
            }

            // Calculate control points for curved lines
            double dx = x2 - x1;
            double dy = y2 - y1;
            double dist = std::sqrt(dx * dx + dy * dy);

            // For long connections, add curvature
            if (dist > CHIP_SPACING_X * 1.3) {
                double cx = (x1 + x2) / 2.0;
                double cy = (y1 + y2) / 2.0;
                double offset = dist * 0.2;

                // Perpendicular offset for curve
                double px = -dy / dist * offset;
                double py = dx / dist * offset;

                file << "<path d=\"M " << x1 << " " << y1 << " Q " << (cx + px) << " " << (cy + py) << " " << x2 << " "
                     << y2 << "\" class=\"" << conn_class << "\" fill=\"none\">\n";
                file << "  <title>" << chip1 << " ↔ " << chip2 << " (" << link_count << " links)</title>\n";
                file << "</path>\n";
            } else {
                // Straight line for adjacent chips
                file << "<line x1=\"" << x1 << "\" y1=\"" << y1 << "\" x2=\"" << x2 << "\" y2=\"" << y2 << "\" class=\""
                     << conn_class << "\">\n";
                file << "  <title>" << chip1 << " ↔ " << chip2 << " (" << link_count << " links)</title>\n";
                file << "</line>\n";
            }
        }

        // Draw chips
        file << "\n<!-- Chips -->\n";
        for (ChipId chip : sorted_chips) {
            auto [x, y] = chip_positions[chip];
            int box_x = x - CHIP_SIZE / 2;
            int box_y = y - CHIP_SIZE / 2;

            file << "<g>\n";
            file << "  <rect x=\"" << box_x << "\" y=\"" << box_y << "\" width=\"" << CHIP_SIZE << "\" height=\""
                 << CHIP_SIZE << "\" rx=\"5\" class=\"chip-box\"/>\n";
            file << "  <text x=\"" << x << "\" y=\"" << (y + 8) << "\" class=\"chip-text\">" << chip << "</text>\n";

            // Add connection count
            int conn_count = 0;
            if (topology.outgoing_links.count(chip) > 0) {
                conn_count = topology.outgoing_links.at(chip).size();
            }
            file << "  <text x=\"" << x << "\" y=\"" << (y + 30)
                 << "\" style=\"fill: #95a5a6; font-size: 12px; text-anchor: middle;\">" << conn_count
                 << " links</text>\n";

            file << "  <title>Chip " << chip << " (" << conn_count << " connections)</title>\n";
            file << "</g>\n";
        }

        // Legend
        int legend_y = svg_height - 80;
        file << "\n<!-- Legend -->\n";
        file << "<text x=\"" << MARGIN << "\" y=\"" << legend_y
             << "\" class=\"legend-text\" font-weight=\"bold\">Connection Strength:</text>\n";

        legend_y += 25;
        file << "<line x1=\"" << MARGIN << "\" y1=\"" << legend_y << "\" x2=\"" << (MARGIN + 50) << "\" y2=\""
             << legend_y << "\" class=\"link-weak\"/>\n";
        file << "<text x=\"" << (MARGIN + 60) << "\" y=\"" << (legend_y + 6)
             << "\" class=\"legend-text\">1-2 links</text>\n";

        file << "<line x1=\"" << (MARGIN + 180) << "\" y1=\"" << legend_y << "\" x2=\"" << (MARGIN + 230) << "\" y2=\""
             << legend_y << "\" class=\"link-medium\"/>\n";
        file << "<text x=\"" << (MARGIN + 240) << "\" y=\"" << (legend_y + 6)
             << "\" class=\"legend-text\">3-4 links</text>\n";

        file << "<line x1=\"" << (MARGIN + 360) << "\" y1=\"" << legend_y << "\" x2=\"" << (MARGIN + 410) << "\" y2=\""
             << legend_y << "\" class=\"link-strong\"/>\n";
        file << "<text x=\"" << (MARGIN + 420) << "\" y=\"" << (legend_y + 6)
             << "\" class=\"legend-text\">5-6 links</text>\n";

        file << "<line x1=\"" << (MARGIN + 540) << "\" y1=\"" << legend_y << "\" x2=\"" << (MARGIN + 590) << "\" y2=\""
             << legend_y << "\" class=\"link-very-strong\"/>\n";
        file << "<text x=\"" << (MARGIN + 600) << "\" y=\"" << (legend_y + 6)
             << "\" class=\"legend-text\">7+ links</text>\n";

        legend_y += 25;
        file << "<text x=\"" << MARGIN << "\" y=\"" << legend_y
             << "\" class=\"legend-text\" style=\"font-style: italic;\">";
        file << "Hover over chips and connections for details. Curved lines = long-range connections.</text>\n";

        file << "</svg>\n";
        file.close();

        std::cout << "✓ SVG visualization exported to: " << filename << "\n";
        std::cout << "  Open in a web browser to view interactive topology\n";
    }
};
