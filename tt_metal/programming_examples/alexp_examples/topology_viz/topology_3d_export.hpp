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

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

struct ChipTopology;
struct EthernetLink;

class Topology3DExporter {
public:
    static void export_3d_html(const ChipTopology& topology, const std::string& filename) {
        std::ofstream file(filename);
        if (!file.is_open()) {
            std::cerr << "Error: Could not open file " << filename << " for writing\n";
            return;
        }

        const double CHIP_SIZE = 3.0;
        const double CHIP_SPACING = 8.0;
        const double TRAY_HEIGHT = 5.0;  // Vertical spacing between trays (rack-mounted)

        // UBB physical layout: each tray has 8 chips arranged as 2 rows × 4 columns
        const int CHIPS_PER_TRAY_ROW = 4;

        // Wormhole NOC grid size
        const int NOC_SIZE_X = 10;
        const int NOC_SIZE_Y = 12;
        const double TILE_SIZE = 0.25;     // Size of each NOC tile
        const double TILE_SPACING = 0.03;  // Small gap between tiles

        // Wormhole ethernet core physical locations (NOC coordinates)
        // Mapping: channel index -> NOC (x, y)
        struct EthCorePos {
            int noc_x;
            int noc_y;
        };
        const EthCorePos eth_core_positions[16] = {
            {9, 0},  // Ch 0
            {1, 0},  // Ch 1
            {8, 0},  // Ch 2
            {2, 0},  // Ch 3
            {7, 0},  // Ch 4
            {3, 0},  // Ch 5
            {6, 0},  // Ch 6
            {4, 0},  // Ch 7
            {9, 6},  // Ch 8
            {1, 6},  // Ch 9
            {8, 6},  // Ch 10
            {2, 6},  // Ch 11
            {7, 6},  // Ch 12
            {3, 6},  // Ch 13
            {6, 6},  // Ch 14
            {4, 6}   // Ch 15
        };

        // DRAM core locations (flattened from 6 banks x 3 ports)
        const std::vector<std::pair<int, int>> dram_cores = {
            {0, 0},
            {0, 1},
            {0, 11},
            {0, 5},
            {0, 6},
            {0, 7},
            {5, 0},
            {5, 1},
            {5, 11},
            {5, 2},
            {5, 9},
            {5, 10},
            {5, 3},
            {5, 4},
            {5, 8},
            {5, 5},
            {5, 6},
            {5, 7}};

        // ARC core location
        const std::pair<int, int> arc_core = {0, 10};

        // PCIE core location
        const std::pair<int, int> pcie_core = {0, 3};

        // Router core locations
        const std::vector<std::pair<int, int>> router_cores = {{0, 2}, {0, 4}, {0, 8}, {0, 9}};

        // Tensix/Worker core grid (8x10 excluding harvested)
        const std::vector<int> worker_x_locations = {1, 2, 3, 4, 6, 7, 8, 9};
        const std::vector<int> worker_y_locations = {1, 2, 3, 4, 5, 7, 8, 9, 10, 11};

        std::vector<ChipId> sorted_chips = topology.chips;
        std::sort(sorted_chips.begin(), sorted_chips.end());

        // Build chip positions (in 3D space) - trays stacked vertically like a rack
        std::map<ChipId, std::tuple<double, double, double>> chip_positions;

        for (const auto& chip : sorted_chips) {
            // Get spatial location
            const auto& spatial_loc = topology.chip_spatial_locations.at(chip);

            // UBB Physical layout within each tray:
            // Locations 1-4 = Row 0 (front row), Columns 0-3
            // Locations 5-8 = Row 1 (back row), Columns 0-3
            int tray_idx = spatial_loc.tray_id - 1;       // 0-based (0-3 for trays 1-4)
            int loc_idx = spatial_loc.asic_location - 1;  // 0-based (0-7)

            int row_in_tray = loc_idx / CHIPS_PER_TRAY_ROW;  // 0 or 1 (front/back)
            int col_in_tray = loc_idx % CHIPS_PER_TRAY_ROW;  // 0-3 (left to right)

            // Position: X = column, Y = tray (stacked vertically), Z = row (front/back)
            double x = col_in_tray * CHIP_SPACING;
            double y = tray_idx * TRAY_HEIGHT;      // Trays stacked vertically
            double z = row_in_tray * CHIP_SPACING;  // Front/back within tray
            chip_positions[chip] = {x, y, z};
        }

        // Group connections by chip pairs and assign spatial routing
        struct Connection {
            ChipId src_chip;
            int src_ch;
            ChipId dst_chip;
            int dst_ch;
            int group_index;  // Index within chip pair group
            int group_size;   // Total connections in this chip pair
            double base_offset;
            std::string color;
        };

        // First pass: group connections by chip pairs
        std::map<std::pair<ChipId, ChipId>, std::vector<Connection>> chip_pair_groups;

        for (const auto& [chip_id, links] : topology.outgoing_links) {
            for (const auto& link : links) {
                if (link.status == LinkStatus::CONNECTED) {
                    if (chip_id < link.dst_chip || (chip_id == link.dst_chip && link.src_channel < link.dst_channel)) {
                        Connection conn;
                        conn.src_chip = chip_id;
                        conn.src_ch = link.src_channel;
                        conn.dst_chip = link.dst_chip;
                        conn.dst_ch = link.dst_channel;

                        auto key = std::make_pair(std::min(chip_id, link.dst_chip), std::max(chip_id, link.dst_chip));
                        chip_pair_groups[key].push_back(conn);
                    }
                }
            }
        }

        // Second pass: assign routing parameters based on group membership
        std::vector<Connection> connections;

        for (auto& [chip_pair, group_conns] : chip_pair_groups) {
            auto [src_x, src_y, src_z] = chip_positions[chip_pair.first];
            auto [dst_x, dst_y, dst_z] = chip_positions[chip_pair.second];

            double dx = dst_x - src_x;
            double dy = dst_y - src_y;
            double dz = dst_z - src_z;
            double xz_dist = std::sqrt(dx * dx + dz * dz);

            // Sort connections by channel to ensure consistent ordering
            std::sort(group_conns.begin(), group_conns.end(), [](const Connection& a, const Connection& b) {
                return a.src_ch < b.src_ch;
            });

            int group_size = group_conns.size();

            // Base offset depends on connection type
            double base_offset;
            std::string base_color;

            if (std::abs(dy) < 0.1) {
                // Same tray connections
                if (xz_dist < CHIP_SPACING * 1.5) {
                    base_offset = 1.5;
                    base_color = "0x3498db";  // Blue
                } else {
                    base_offset = 2.0;
                    base_color = "0x9b59b6";  // Purple
                }
            } else {
                // Cross-tray connections
                if (xz_dist < CHIP_SPACING * 1.5) {
                    base_offset = 2.5;
                    base_color = "0x2ecc71";  // Green
                } else {
                    base_offset = 3.5;
                    base_color = "0xe74c3c";  // Red
                }
            }

            // Assign spatial parameters to each connection in group
            for (int i = 0; i < group_size; i++) {
                group_conns[i].group_index = i;
                group_conns[i].group_size = group_size;
                group_conns[i].base_offset = base_offset;
                group_conns[i].color = base_color;
                connections.push_back(group_conns[i]);
            }
        }

        // Write HTML with Three.js
        file << R"HTML(<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>TT-Metal 3D Topology Visualization</title>
    <style>
        body {
            margin: 0;
            overflow: hidden;
            font-family: 'Arial', sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        }
        #container {
            width: 100vw;
            height: 100vh;
        }
        #panel {
            position: absolute;
            top: 20px;
            right: 20px;
            width: 400px;
            max-height: 90vh;
            background: rgba(0, 0, 0, 0.9);
            color: white;
            border-radius: 10px;
            backdrop-filter: blur(10px);
            overflow: hidden;
            display: flex;
            flex-direction: column;
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.5);
        }
        #panel.collapsed {
            width: 60px;
            height: 60px;
        }
        #panel-header {
            padding: 15px;
            background: rgba(52, 152, 219, 0.3);
            border-bottom: 2px solid #3498db;
            display: flex;
            justify-content: space-between;
            align-items: center;
            cursor: pointer;
        }
        #panel-header h2 {
            margin: 0;
            font-size: 16px;
        }
        #toggle-btn {
            background: none;
            border: none;
            color: white;
            font-size: 20px;
            cursor: pointer;
            padding: 5px;
        }
        #panel-content {
            flex: 1;
            overflow-y: auto;
            padding: 15px;
        }
        #panel.collapsed #panel-content {
            display: none;
        }
        .tab-buttons {
            display: flex;
            gap: 10px;
            margin-bottom: 15px;
        }
        .tab-btn {
            flex: 1;
            padding: 8px;
            background: rgba(52, 152, 219, 0.2);
            border: 1px solid #3498db;
            color: white;
            cursor: pointer;
            border-radius: 5px;
            transition: all 0.3s;
        }
        .tab-btn:hover {
            background: rgba(52, 152, 219, 0.4);
        }
        .tab-btn.active {
            background: #3498db;
        }
        .tab-content {
            display: none;
        }
        .tab-content.active {
            display: block;
        }
        .search-box {
            width: 100%;
            padding: 8px;
            margin-bottom: 10px;
            background: rgba(255, 255, 255, 0.1);
            border: 1px solid #555;
            border-radius: 5px;
            color: white;
            font-size: 12px;
        }
        .item-list {
            max-height: 400px;
            overflow-y: auto;
        }
        .item {
            padding: 10px;
            margin-bottom: 5px;
            background: rgba(255, 255, 255, 0.05);
            border-radius: 5px;
            border-left: 3px solid #3498db;
            cursor: pointer;
            transition: all 0.2s;
        }
        .item:hover {
            background: rgba(255, 255, 255, 0.1);
            transform: translateX(5px);
        }
        .item.hidden {
            opacity: 0.4;
            border-left-color: #e74c3c;
        }
        .item-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 5px;
        }
        .item-header > div {
            display: flex;
            align-items: center;
            gap: 5px;
        }
        .item-title {
            font-weight: bold;
            font-size: 13px;
        }
        .item-details {
            font-size: 11px;
            color: #bbb;
        }
        .toggle-visibility {
            background: none;
            border: none;
            color: white;
            cursor: pointer;
            font-size: 16px;
            padding: 5px;
        }
        .expand-btn {
            background: none;
            border: none;
            color: #3498db;
            cursor: pointer;
            font-size: 14px;
            padding: 5px;
            margin-left: 5px;
        }
        .connections-dropdown {
            display: none;
            margin-top: 8px;
            padding: 8px;
            background: rgba(0, 0, 0, 0.3);
            border-radius: 3px;
            border-left: 2px solid #2ecc71;
        }
        .connections-dropdown.expanded {
            display: block;
        }
        .connection-item {
            padding: 5px;
            margin: 3px 0;
            background: rgba(255, 255, 255, 0.03);
            border-radius: 3px;
            font-size: 10px;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        .connection-item:hover {
            background: rgba(255, 255, 255, 0.08);
        }
        .connection-item.disabled {
            opacity: 0.4;
        }
        .stats {
            background: rgba(46, 204, 113, 0.1);
            padding: 10px;
            border-radius: 5px;
            margin-bottom: 10px;
            font-size: 12px;
        }
        .stats-row {
            display: flex;
            justify-content: space-between;
            margin: 5px 0;
        }
        .highlight {
            background: rgba(241, 196, 15, 0.3) !important;
            border-left-color: #f39c12 !important;
        }
        ::-webkit-scrollbar {
            width: 8px;
        }
        ::-webkit-scrollbar-track {
            background: rgba(255, 255, 255, 0.05);
        }
        ::-webkit-scrollbar-thumb {
            background: rgba(52, 152, 219, 0.5);
            border-radius: 4px;
        }
        #info {
            position: absolute;
            top: 20px;
            left: 20px;
            color: white;
            background: rgba(0, 0, 0, 0.85);
            padding: 15px;
            border-radius: 10px;
            font-size: 12px;
            max-width: 250px;
            backdrop-filter: blur(10px);
            box-shadow: 0 4px 16px rgba(0, 0, 0, 0.5);
        }
        #info h1 {
            margin: 0 0 8px 0;
            font-size: 16px;
            color: #3498db;
        }
        #info p {
            margin: 3px 0;
        }
        .stat {
            color: #2ecc71;
            font-weight: bold;
        }
        #controls {
            position: absolute;
            top: 140px;
            left: 20px;
            color: white;
            background: rgba(0, 0, 0, 0.85);
            padding: 12px 15px;
            border-radius: 10px;
            font-size: 11px;
            backdrop-filter: blur(10px);
            max-width: 280px;
            box-shadow: 0 4px 16px rgba(0, 0, 0, 0.5);
            line-height: 1.4;
        }
        #legend {
            position: absolute;
            bottom: 20px;
            left: 20px;
            color: white;
            background: rgba(0, 0, 0, 0.7);
            padding: 15px;
            border-radius: 10px;
            font-size: 12px;
            backdrop-filter: blur(10px);
        }
        .legend-item {
            display: flex;
            align-items: center;
            margin: 5px 0;
        }
        .legend-color {
            width: 30px;
            height: 3px;
            margin-right: 10px;
            border-radius: 2px;
        }
    </style>
</head>
<body>
    <div id="container"></div>

    <div id="info">
        <h1>&#x1F537; TT-Metal Topology 3D</h1>
        <p><span class="stat">)HTML";

        file << sorted_chips.size() << "</span> Chips</p>\n";
        file << "        <p><span class=\"stat\">" << connections.size() << "</span> Connections</p>\n";
        file << R"HTML(        <p>3D routing prevents overlaps</p>
    </div>

    <div id="controls">
        <strong>&#x1F3AE; Mouse Controls:</strong><br>
        <span style="font-size: 11px;">
        • <strong>Left Drag:</strong> Rotate view<br>
        • <strong>Right Drag / Shift+Drag:</strong> Pan<br>
        • <strong>Scroll:</strong> Smooth zoom<br>
        • <strong>Click:</strong> Select object<br>
        • <strong>Click again (same spot):</strong> Cycle overlapping<br>
        </span>
        <strong style="margin-top: 8px; display: block;">&#x2328; Keyboard:</strong><br>
        <span style="font-size: 11px;">
        • <strong>WASD:</strong> Pan (left/right/forward/back)<br>
        • <strong>Q/E:</strong> Move up/down<br>
        • <strong>Arrows:</strong> Rotate camera<br>
        • <strong>+/-:</strong> Zoom in/out<br>
        • <strong>R:</strong> Reset view<br>
        </span>
        <span style="color: #2ecc71; font-size: 9px; margin-top: 5px; display: block;">&#x1F4A1; Very precise clicking - aim directly at lines</span>
        <span style="color: #1abc9c; font-size: 9px;">Teal = hover preview | Orange = selected</span>
        <span style="color: #95a5a6; font-size: 9px;">Thin lines won't block chips behind them</span>
    </div>

    <div id="panel">
        <div id="panel-header" onclick="togglePanel()">
            <h2>&#x1F4CA; Control Panel</h2>
            <button id="toggle-btn">&minus;</button>
        </div>
        <div id="panel-content">
            <div class="tab-buttons">
                <button class="tab-btn active" onclick="switchTab('chips')">Chips</button>
                <button class="tab-btn" onclick="switchTab('connections')">Connections</button>
            </div>

            <div id="chips-tab" class="tab-content active">
                <input type="text" class="search-box" id="chip-search" placeholder="&#x1F50D; Search chips..." onkeyup="filterChips()">
                <div class="stats">
                    <div class="stats-row">
                        <span>Total Chips:</span>
                        <span id="total-chips">0</span>
                    </div>
                    <div class="stats-row">
                        <span>Visible:</span>
                        <span id="visible-chips">0</span>
                    </div>
                </div>
                <div class="item-list" id="chip-list"></div>
            </div>

            <div id="connections-tab" class="tab-content">
                <input type="text" class="search-box" id="conn-search" placeholder="&#x1F50D; Search connections..." onkeyup="filterConnections()">
                <div class="stats">
                    <div class="stats-row">
                        <span>Total Links:</span>
                        <span id="total-connections">0</span>
                    </div>
                    <div class="stats-row">
                        <span>Visible:</span>
                        <span id="visible-connections">0</span>
                    </div>
                </div>
                <div class="item-list" id="connection-list"></div>
            </div>
        </div>
    </div>

    <div id="legend">
        <strong>NOC Tile Types:</strong><br>
        <div class="legend-item">
            <div class="legend-color" style="background: #3498db; width: 20px; height: 6px;"></div>
            <span>Worker/Tensix</span>
        </div>
        <div class="legend-item">
            <div class="legend-color" style="background: #ff8c00; width: 20px; height: 6px;"></div>
            <span>Ethernet</span>
        </div>
        <div class="legend-item">
            <div class="legend-color" style="background: #27ae60; width: 20px; height: 6px;"></div>
            <span>DRAM</span>
        </div>
        <div class="legend-item">
            <div class="legend-color" style="background: #9b59b6; width: 20px; height: 6px;"></div>
            <span>ARC</span>
        </div>
        <div class="legend-item">
            <div class="legend-color" style="background: #f39c12; width: 20px; height: 6px;"></div>
            <span>PCIE</span>
        </div>
        <div class="legend-item">
            <div class="legend-color" style="background: #16a085; width: 20px; height: 6px;"></div>
            <span>Router</span>
        </div>
        <hr style="border: 1px solid rgba(255,255,255,0.2); margin: 10px 0;">
        <strong>Connection Types:</strong><br>
        <div class="legend-item">
            <div class="legend-color" style="background: #3498db;"></div>
            <span>Same Tray - Adjacent</span>
        </div>
        <div class="legend-item">
            <div class="legend-color" style="background: #9b59b6;"></div>
            <span>Same Tray - Diagonal</span>
        </div>
        <div class="legend-item">
            <div class="legend-color" style="background: #2ecc71;"></div>
            <span>Cross-Tray - Aligned</span>
        </div>
        <div class="legend-item">
            <div class="legend-color" style="background: #e74c3c;"></div>
            <span>Cross-Tray - Diagonal</span>
        </div>
    </div>

    <script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
    <script>
        // Data structures for tracking objects
        const chipObjects = [];
        const connectionObjects = [];
        const expandedChips = new Set();

        // Ethernet channel to NOC coordinate mapping
        const ethChannelToNoc = [
            {x: 9, y: 0},   // Ch 0
            {x: 1, y: 0},   // Ch 1
            {x: 8, y: 0},   // Ch 2
            {x: 2, y: 0},   // Ch 3
            {x: 7, y: 0},   // Ch 4
            {x: 3, y: 0},   // Ch 5
            {x: 6, y: 0},   // Ch 6
            {x: 4, y: 0},   // Ch 7
            {x: 9, y: 6},   // Ch 8
            {x: 1, y: 6},   // Ch 9
            {x: 8, y: 6},   // Ch 10
            {x: 2, y: 6},   // Ch 11
            {x: 7, y: 6},   // Ch 12
            {x: 3, y: 6},   // Ch 13
            {x: 6, y: 6},   // Ch 14
            {x: 4, y: 6}    // Ch 15
        ];

        // Setup scene, camera, renderer
        const scene = new THREE.Scene();
        scene.background = new THREE.Color(0x1a1a2e);
        scene.fog = new THREE.Fog(0x1a1a2e, 50, 150);

        const camera = new THREE.PerspectiveCamera(60, window.innerWidth / window.innerHeight, 0.1, 1000);
        camera.position.set(20, 10, 30);
        camera.lookAt(12, 7.5, 4);  // Look at center of rack (4 trays at 5 units each)

        const renderer = new THREE.WebGLRenderer({ antialias: true });
        renderer.setSize(window.innerWidth, window.innerHeight);
        renderer.shadowMap.enabled = true;
        renderer.shadowMap.type = THREE.PCFSoftShadowMap;
        document.getElementById('container').appendChild(renderer.domElement);

        // Add lights
        const ambientLight = new THREE.AmbientLight(0xffffff, 0.6);
        scene.add(ambientLight);

        const directionalLight = new THREE.DirectionalLight(0xffffff, 0.8);
        directionalLight.position.set(20, 30, 20);
        directionalLight.castShadow = true;
        directionalLight.shadow.mapSize.width = 2048;
        directionalLight.shadow.mapSize.height = 2048;
        scene.add(directionalLight);

        const pointLight = new THREE.PointLight(0x3498db, 0.5);
        pointLight.position.set(-10, 15, -10);
        scene.add(pointLight);

        // Add grid helper on ground
        const gridHelper = new THREE.GridHelper(40, 15, 0x444444, 0x222222);
        gridHelper.position.y = -1;
        gridHelper.position.x = 12;
        gridHelper.position.z = 4;
        scene.add(gridHelper);

)HTML";

        // Add chips
        file << "        // Add chips\n";
        for (ChipId chip : sorted_chips) {
            auto [x, y, z] = chip_positions[chip];
            const auto& spatial_loc = topology.chip_spatial_locations.at(chip);

            int connected_count = 0;
            if (topology.outgoing_links.count(chip) > 0) {
                connected_count = topology.outgoing_links.at(chip).size();
            }

            // Helper to check if a coordinate is in a set
            auto is_in_set = [](int nx, int ny, const std::vector<std::pair<int, int>>& coords) {
                for (const auto& [cx, cy] : coords) {
                    if (cx == nx && cy == ny) {
                        return true;
                    }
                }
                return false;
            };

            auto is_eth_core = [&](int nx, int ny) {
                for (int ch = 0; ch < 16; ch++) {
                    if (eth_core_positions[ch].noc_x == nx && eth_core_positions[ch].noc_y == ny) {
                        return true;
                    }
                }
                return false;
            };

            auto is_worker_core = [&](int nx, int ny) {
                bool x_match = false, y_match = false;
                for (int wx : worker_x_locations) {
                    if (wx == nx) {
                        x_match = true;
                    }
                }
                for (int wy : worker_y_locations) {
                    if (wy == ny) {
                        y_match = true;
                    }
                }
                return x_match && y_match;
            };

            file << "        {\n";
            file << "            // Create chip group with NOC grid\n";
            file << "            const chipGroup = new THREE.Group();\n";
            file << "            chipGroup.position.set(" << x << ", " << y << ", " << z << ");\n";
            file << "            \n";

            // Render NOC grid (10x12 tiles)
            file << "            // Render NOC grid (" << NOC_SIZE_X << "x" << NOC_SIZE_Y << ")\n";
            file << "            const tileSize = " << TILE_SIZE << ";\n";
            file << "            const tileSpacing = " << TILE_SPACING << ";\n";
            file << "            const tileGeo = new THREE.BoxGeometry(tileSize, 0.15, tileSize);\n";
            file << "            \n";

            // Calculate tile grid offset to center it within CHIP_SIZE
            double grid_width = NOC_SIZE_X * (TILE_SIZE + TILE_SPACING) - TILE_SPACING;
            double grid_depth = NOC_SIZE_Y * (TILE_SIZE + TILE_SPACING) - TILE_SPACING;
            double offset_x = -grid_width / 2.0 + TILE_SIZE / 2.0;
            double offset_z = -grid_depth / 2.0 + TILE_SIZE / 2.0;

            for (int noc_x = 0; noc_x < NOC_SIZE_X; noc_x++) {
                for (int noc_y = 0; noc_y < NOC_SIZE_Y; noc_y++) {
                    // Determine tile type and color
                    std::string color_hex;

                    if (is_eth_core(noc_x, noc_y)) {
                        color_hex = "0xff8c00";  // Dark orange - Ethernet
                    } else if (is_in_set(noc_x, noc_y, dram_cores)) {
                        color_hex = "0x27ae60";  // Green - DRAM
                    } else if (noc_x == arc_core.first && noc_y == arc_core.second) {
                        color_hex = "0x9b59b6";  // Purple - ARC
                    } else if (noc_x == pcie_core.first && noc_y == pcie_core.second) {
                        color_hex = "0xf39c12";  // Yellow - PCIE
                    } else if (is_in_set(noc_x, noc_y, router_cores)) {
                        color_hex = "0x16a085";  // Teal - Router
                    } else if (is_worker_core(noc_x, noc_y)) {
                        color_hex = "0x3498db";  // Blue - Worker/Tensix
                    } else {
                        color_hex = "0x2c3e50";  // Dark gray - Other/Harvested
                    }

                    double tile_x = offset_x + noc_x * (TILE_SIZE + TILE_SPACING);
                    double tile_z = offset_z + noc_y * (TILE_SIZE + TILE_SPACING);

                    file << "            {\n";
                    file << "                const tileMat = new THREE.MeshStandardMaterial({\n";
                    file << "                    color: " << color_hex << ",\n";
                    file << "                    metalness: 0.3,\n";
                    file << "                    roughness: 0.6,\n";
                    file << "                    emissive: " << color_hex << ",\n";
                    file << "                    emissiveIntensity: 0.15\n";
                    file << "                });\n";
                    file << "                const tile = new THREE.Mesh(tileGeo, tileMat);\n";
                    file << "                tile.position.set(" << tile_x << ", 0.1, " << tile_z << ");\n";
                    file << "                tile.castShadow = true;\n";
                    file << "                tile.receiveShadow = true;\n";
                    file << "                tile.userData.nocX = " << noc_x << ";\n";
                    file << "                tile.userData.nocY = " << noc_y << ";\n";
                    file << "                tile.userData.originalColor = " << color_hex << ";\n";
                    file << "                chipGroup.add(tile);\n";
                    file << "            }\n";
                }
            }

            file << "            scene.add(chipGroup);\n";
            file << "            \n";
            file << "            // Add chip label with spatial info\n";
            file << "            const canvas = document.createElement('canvas');\n";
            file << "            const context = canvas.getContext('2d');\n";
            file << "            canvas.width = 256;\n";
            file << "            canvas.height = 160;\n";
            file << "            context.fillStyle = '#ecf0f1';\n";
            file << "            context.font = 'bold 40px Arial';\n";
            file << "            context.textAlign = 'center';\n";
            file << "            context.fillText('Chip " << chip << "', 128, 40);\n";
            file << "            context.font = 'bold 32px Arial';\n";
            file << "            context.fillStyle = '#3498db';\n";
            file << "            context.fillText('Tray " << spatial_loc.tray_id << " / Loc "
                 << spatial_loc.asic_location << "', 128, 80);\n";
            file << "            context.fillStyle = '#95a5a6';\n";
            file << "            context.font = '28px Arial';\n";
            file << "            context.fillText('" << connected_count << " links', 128, 115);\n";
            file << "            const texture = new THREE.CanvasTexture(canvas);\n";
            file << "            const labelGeo = new THREE.PlaneGeometry(2.2, 1.4);\n";
            file << "            const labelMat = new THREE.MeshBasicMaterial({ map: texture, transparent: true });\n";
            file << "            const label = new THREE.Mesh(labelGeo, labelMat);\n";
            file << "            label.position.set(" << x << ", " << y << " + 0.5, " << z << ");\n";
            file << "            label.rotation.x = -Math.PI / 2;\n";
            file << "            scene.add(label);\n";
            file << "            \n";
            file << "            chipObjects.push({\n";
            file << "                id: " << chip << ",\n";
            file << "                mesh: chipGroup,\n";
            file << "                label: label,\n";
            file << "                visible: true,\n";
            file << "                connections: " << connected_count << ",\n";
            file << "                tray: " << spatial_loc.tray_id << ",\n";
            file << "                location: " << spatial_loc.asic_location << ",\n";
            file << "                originalColor: 0x34495e\n";
            file << "            });\n";
            file << "        }\n";
        }

        // Add tray labels (vertically stacked)
        file << "\n        // Add tray labels\n";
        for (int tray = 1; tray <= 4; tray++) {
            int tray_idx = tray - 1;
            // Position at the center of each tray's 2×4 grid
            double x_pos = (CHIPS_PER_TRAY_ROW * CHIP_SPACING) / 2.0 - CHIP_SPACING / 2.0;  // Center X
            double y_pos = tray_idx * TRAY_HEIGHT;                                          // Tray height
            double z_pos = -CHIP_SPACING * 0.8;                                             // In front of the tray

            file << "        {\n";
            file << "            const canvas = document.createElement('canvas');\n";
            file << "            const context = canvas.getContext('2d');\n";
            file << "            canvas.width = 200;\n";
            file << "            canvas.height = 120;\n";
            file << "            context.fillStyle = '#2ecc71';\n";
            file << "            context.font = 'bold 48px Arial';\n";
            file << "            context.textAlign = 'center';\n";
            file << "            context.fillText('TRAY " << tray << "', 100, 50);\n";
            file << "            context.fillStyle = '#95a5a6';\n";
            file << "            context.font = '28px Arial';\n";
            file << "            context.fillText('(2x4 layout)', 100, 90);\n";
            file << "            const texture = new THREE.CanvasTexture(canvas);\n";
            file << "            const labelGeo = new THREE.PlaneGeometry(3, 1.8);\n";
            file << "            const labelMat = new THREE.MeshBasicMaterial({ map: texture, transparent: true, side: "
                    "THREE.DoubleSide });\n";
            file << "            const label = new THREE.Mesh(labelGeo, labelMat);\n";
            file << "            label.position.set(" << x_pos << ", " << y_pos << ", " << z_pos << ");\n";
            file << "            scene.add(label);\n";
            file << "        }\n";
        }

        // Add connections
        file << "\n        // Add connections\n";
        for (const auto& conn : connections) {
            auto [src_x, src_y, src_z] = chip_positions[conn.src_chip];
            auto [dst_x, dst_y, dst_z] = chip_positions[conn.dst_chip];

            // Map ethernet core NOC coordinates to tile positions in grid
            const EthCorePos& src_core = eth_core_positions[conn.src_ch];
            const EthCorePos& dst_core = eth_core_positions[conn.dst_ch];

            // Calculate tile grid offset (same as used for rendering chips)
            double grid_width = NOC_SIZE_X * (TILE_SIZE + TILE_SPACING) - TILE_SPACING;
            double grid_depth = NOC_SIZE_Y * (TILE_SIZE + TILE_SPACING) - TILE_SPACING;
            double offset_x = -grid_width / 2.0 + TILE_SIZE / 2.0;
            double offset_z = -grid_depth / 2.0 + TILE_SIZE / 2.0;

            // Calculate tile positions for source and destination ethernet cores
            double src_tile_x = offset_x + src_core.noc_x * (TILE_SIZE + TILE_SPACING);
            double src_tile_z = offset_z + src_core.noc_y * (TILE_SIZE + TILE_SPACING);
            double dst_tile_x = offset_x + dst_core.noc_x * (TILE_SIZE + TILE_SPACING);
            double dst_tile_z = offset_z + dst_core.noc_y * (TILE_SIZE + TILE_SPACING);

            // Final connection endpoints (at tile positions on chip, slightly above surface)
            double src_x_final = src_x + src_tile_x;
            double dst_x_final = dst_x + dst_tile_x;
            double src_z_final = src_z + src_tile_z;
            double dst_z_final = dst_z + dst_tile_z;
            double src_y_final = src_y + 0.2;  // Slightly above chip surface
            double dst_y_final = dst_y + 0.2;

            // Intelligent 3D routing with spatial spreading for multiple connections
            double dx = dst_x - src_x;
            double dy = dst_y - src_y;
            double dz = dst_z - src_z;

            double abs_dx = std::abs(dx);
            double abs_dy = std::abs(dy);
            double abs_dz = std::abs(dz);

            // Calculate spatial spread factor for this connection within its group
            // Center the spread around 0 (range: -0.5 to +0.5)
            double spread_factor = 0.0;
            if (conn.group_size > 1) {
                spread_factor = (static_cast<double>(conn.group_index) / (conn.group_size - 1)) - 0.5;
            }

            // Scale spread based on connection type and group size
            double spread_scale = CHIP_SPACING * 0.4 * std::min(2.0, 0.5 + conn.group_size * 0.2);
            double lateral_spread = spread_factor * spread_scale;

            // Vertical spread to separate connections in 3D space
            double vertical_spread = spread_factor * TRAY_HEIGHT * 0.15;

            // Calculate base routing parameters
            double base_offset_scaled = conn.base_offset * CHIP_SPACING * (1.0 + std::abs(spread_factor) * 0.3);
            double y_offset_base = conn.base_offset * TRAY_HEIGHT * 0.25;

            file << "        {\n";
            file << "            // Multi-point curve with spatial spreading\n";

            if (abs_dy > 0.1) {
                // Cross-tray: use 5-point curve for smooth routing
                double mid_x = (src_x + dst_x) / 2.0;
                double mid_y = (src_y + dst_y) / 2.0 + y_offset_base + vertical_spread;
                double mid_z = (src_z + dst_z) / 2.0;

                // Determine primary spreading direction based on connection geometry
                double outward_offset = CHIP_SIZE * 2.0 + base_offset_scaled;

                double cp1_x, cp1_y, cp1_z;  // First control point (near source)
                double cp2_x, cp2_y, cp2_z;  // Mid control point
                double cp3_x, cp3_y, cp3_z;  // Second control point (near dest)

                if (abs_dx < 0.1 && abs_dz < 0.1) {
                    // Vertical alignment - spread radially outward
                    double angle = spread_factor * M_PI;  // Spread in a fan pattern
                    cp1_x = src_x + outward_offset * 0.3 * std::cos(angle);
                    cp1_z = src_z + outward_offset * 0.3 * std::sin(angle);
                    cp2_x = mid_x + outward_offset * std::cos(angle);
                    cp2_z = mid_z + outward_offset * std::sin(angle);
                    cp3_x = dst_x + outward_offset * 0.3 * std::cos(angle);
                    cp3_z = dst_z + outward_offset * 0.3 * std::sin(angle);
                } else if (abs_dx < 0.1) {
                    // X-aligned - spread in X direction
                    cp1_x = src_x + lateral_spread;
                    cp1_z = src_z + (dz > 0 ? outward_offset * 0.2 : -outward_offset * 0.2);
                    cp2_x = mid_x + lateral_spread * 1.5;
                    cp2_z = mid_z + (dz > 0 ? outward_offset * 0.4 : -outward_offset * 0.4);
                    cp3_x = dst_x + lateral_spread;
                    cp3_z = dst_z - (dz > 0 ? outward_offset * 0.2 : -outward_offset * 0.2);
                } else if (abs_dz < 0.1) {
                    // Z-aligned - spread in Z direction
                    cp1_x = src_x + (dx > 0 ? outward_offset * 0.2 : -outward_offset * 0.2);
                    cp1_z = src_z + lateral_spread;
                    cp2_x = mid_x + (dx > 0 ? outward_offset * 0.4 : -outward_offset * 0.4);
                    cp2_z = mid_z + lateral_spread * 1.5;
                    cp3_x = dst_x - (dx > 0 ? outward_offset * 0.2 : -outward_offset * 0.2);
                    cp3_z = dst_z + lateral_spread;
                } else {
                    // Diagonal - spread perpendicular to connection line
                    double xz_dist_local = std::sqrt(dx * dx + dz * dz);
                    double perp_x = -dz / xz_dist_local;  // Perpendicular in XZ plane
                    double perp_z = dx / xz_dist_local;
                    cp1_x = src_x + dx * 0.2 + perp_x * lateral_spread;
                    cp1_z = src_z + dz * 0.2 + perp_z * lateral_spread;
                    cp2_x = mid_x + perp_x * lateral_spread * 1.5;
                    cp2_z = mid_z + perp_z * lateral_spread * 1.5;
                    cp3_x = dst_x - dx * 0.2 + perp_x * lateral_spread;
                    cp3_z = dst_z - dz * 0.2 + perp_z * lateral_spread;
                }

                cp1_y = src_y + y_offset_base * 0.3 + vertical_spread * 0.5;
                cp2_y = mid_y;
                cp3_y = dst_y + y_offset_base * 0.3 + vertical_spread * 0.5;

                file << "            const curve = new THREE.CatmullRomCurve3([\n";
                file << "                new THREE.Vector3(" << src_x_final << ", " << src_y_final << ", "
                     << src_z_final << "),\n";
                file << "                new THREE.Vector3(" << cp1_x << ", " << cp1_y << ", " << cp1_z << "),\n";
                file << "                new THREE.Vector3(" << cp2_x << ", " << cp2_y << ", " << cp2_z << "),\n";
                file << "                new THREE.Vector3(" << cp3_x << ", " << cp3_y << ", " << cp3_z << "),\n";
                file << "                new THREE.Vector3(" << dst_x_final << ", " << dst_y_final << ", "
                     << dst_z_final << ")\n";
                file << "            ]);\n";
            } else {
                // Same-tray: use 4-point curve with lateral spreading
                double mid_x = (src_x + dst_x) / 2.0;
                double mid_z = (src_z + dst_z) / 2.0;
                double route_y = src_y + y_offset_base + vertical_spread;

                double cp1_x, cp1_z, cp2_x, cp2_z;

                if (abs_dx > abs_dz) {
                    // Primarily horizontal - spread in Z
                    bool route_forward = mid_z < CHIP_SPACING;
                    double z_dir = route_forward ? 1.0 : -1.0;
                    double route_z = (route_forward ? std::max(src_z, dst_z) : std::min(src_z, dst_z)) +
                                     base_offset_scaled * z_dir + lateral_spread;

                    cp1_x = src_x + dx * 0.25;
                    cp1_z = route_z;
                    cp2_x = dst_x - dx * 0.25;
                    cp2_z = route_z;
                } else {
                    // Primarily depth - spread in X
                    bool route_right = mid_x < CHIP_SPACING * 1.5;
                    double x_dir = route_right ? 1.0 : -1.0;
                    double route_x = (route_right ? std::max(src_x, dst_x) : std::min(src_x, dst_x)) +
                                     base_offset_scaled * x_dir + lateral_spread;

                    cp1_x = route_x;
                    cp1_z = src_z + dz * 0.25;
                    cp2_x = route_x;
                    cp2_z = dst_z - dz * 0.25;
                }

                file << "            const curve = new THREE.CatmullRomCurve3([\n";
                file << "                new THREE.Vector3(" << src_x_final << ", " << src_y_final << ", "
                     << src_z_final << "),\n";
                file << "                new THREE.Vector3(" << cp1_x << ", " << route_y << ", " << cp1_z << "),\n";
                file << "                new THREE.Vector3(" << cp2_x << ", " << route_y << ", " << cp2_z << "),\n";
                file << "                new THREE.Vector3(" << dst_x_final << ", " << dst_y_final << ", "
                     << dst_z_final << ")\n";
                file << "            ]);\n";
            }
            file << "            const points = curve.getPoints(50);\n";
            file << "            const geometry = new THREE.BufferGeometry().setFromPoints(points);\n";
            file << "            const material = new THREE.LineBasicMaterial({ \n";
            file << "                color: " << conn.color << ",\n";
            file << "                linewidth: 2.5,\n";
            file << "                opacity: 0.7,\n";
            file << "                transparent: true\n";
            file << "            });\n";
            file << "            const line = new THREE.Line(geometry, material);\n";
            file << "            line.userData = {\n";
            file << "                src: " << conn.src_chip << ",\n";
            file << "                srcCh: " << conn.src_ch << ",\n";
            file << "                dst: " << conn.dst_chip << ",\n";
            file << "                dstCh: " << conn.dst_ch << "\n";
            file << "            };\n";
            file << "            scene.add(line);\n";
            file << "            \n";
            file << "            connectionObjects.push({\n";
            file << "                src: " << conn.src_chip << ",\n";
            file << "                srcCh: " << conn.src_ch << ",\n";
            file << "                dst: " << conn.dst_chip << ",\n";
            file << "                dstCh: " << conn.dst_ch << ",\n";
            file << "                line: line,\n";
            file << "                visible: true,\n";
            file << "                originalColor: " << conn.color << ",\n";
            file << "                groupIndex: " << conn.group_index << ",\n";
            file << "                groupSize: " << conn.group_size << "\n";
            file << "            });\n";
            file << "        }\n";
        }

        file << R"HTML(

        // Mouse controls
        let isDragging = false;
        let isPanning = false;
        let previousMousePosition = { x: 0, y: 0 };
        let cameraRotation = { x: 0, y: 0 };
        let cameraDistance = 40;
        let targetCameraDistance = 40;
        let cameraLookAt = new THREE.Vector3(12, 7.5, 4); // Center of the cluster
        const minZoom = 5;
        const maxZoom = 120;

        renderer.domElement.addEventListener('mousedown', (e) => {
            // Right-click or Shift+Left-click for panning
            if (e.button === 2 || (e.button === 0 && e.shiftKey)) {
                isPanning = true;
                e.preventDefault();
            } else if (e.button === 0) {
                isDragging = true;
            }
            mouseMoved = false;
            mouseDownPos = { x: e.clientX, y: e.clientY };
            previousMousePosition = { x: e.clientX, y: e.clientY };
        });

        renderer.domElement.addEventListener('mousemove', (e) => {
            if (isPanning) {
                const deltaX = e.clientX - previousMousePosition.x;
                const deltaY = e.clientY - previousMousePosition.y;

                // Pan camera (translate look-at point)
                const panSpeed = 0.02;
                const right = new THREE.Vector3();
                const up = new THREE.Vector3(0, 1, 0);
                camera.getWorldDirection(right);
                right.cross(up).normalize();

                cameraLookAt.add(right.multiplyScalar(-deltaX * panSpeed));
                cameraLookAt.y += deltaY * panSpeed;

                previousMousePosition = { x: e.clientX, y: e.clientY };
                mouseMoved = true;
                updateCameraPosition();
            } else if (isDragging) {
                const deltaX = e.clientX - previousMousePosition.x;
                const deltaY = e.clientY - previousMousePosition.y;

                // Check if mouse moved significantly
                const totalDeltaX = e.clientX - mouseDownPos.x;
                const totalDeltaY = e.clientY - mouseDownPos.y;
                const distanceMoved = Math.sqrt(totalDeltaX * totalDeltaX + totalDeltaY * totalDeltaY);

                if (distanceMoved > 5) {
                    mouseMoved = true;
                }

                cameraRotation.y += deltaX * 0.005;
                cameraRotation.x += deltaY * 0.005;
                cameraRotation.x = Math.max(-Math.PI / 2 + 0.1, Math.min(Math.PI / 2 - 0.1, cameraRotation.x));

                previousMousePosition = { x: e.clientX, y: e.clientY };

                updateCameraPosition();
            }
        });

        renderer.domElement.addEventListener('mouseup', () => {
            isDragging = false;
            isPanning = false;
        });

        // Disable context menu on right-click
        renderer.domElement.addEventListener('contextmenu', (e) => {
            e.preventDefault();
        });

        renderer.domElement.addEventListener('wheel', (e) => {
            e.preventDefault();
            // Smooth zoom with better speed
            const zoomSpeed = e.deltaY > 0 ? 1.1 : 0.9;
            targetCameraDistance *= zoomSpeed;
            targetCameraDistance = Math.max(minZoom, Math.min(maxZoom, targetCameraDistance));
        });

        function updateCameraPosition() {
            // Smooth interpolation for zoom
            cameraDistance += (targetCameraDistance - cameraDistance) * 0.15;

            const offsetX = cameraDistance * Math.cos(cameraRotation.x) * Math.sin(cameraRotation.y);
            const offsetY = cameraDistance * Math.sin(cameraRotation.x);
            const offsetZ = cameraDistance * Math.cos(cameraRotation.x) * Math.cos(cameraRotation.y);

            camera.position.set(
                cameraLookAt.x + offsetX,
                cameraLookAt.y + offsetY,
                cameraLookAt.z + offsetZ
            );
            camera.lookAt(cameraLookAt);
        }

        // Keyboard controls
        const keys = {};
        window.addEventListener('keydown', (e) => {
            keys[e.key] = true;
        });
        window.addEventListener('keyup', (e) => {
            keys[e.key] = false;
        });

        function handleKeyboardControls() {
            const moveSpeed = 0.5;
            const rotateSpeed = 0.02;
            const zoomSpeed = 0.5;

            // WASD for panning
            if (keys['w'] || keys['W']) {
                cameraLookAt.z -= moveSpeed;
            }
            if (keys['s'] || keys['S']) {
                cameraLookAt.z += moveSpeed;
            }
            if (keys['a'] || keys['A']) {
                cameraLookAt.x -= moveSpeed;
            }
            if (keys['d'] || keys['D']) {
                cameraLookAt.x += moveSpeed;
            }

            // Q/E for vertical movement
            if (keys['q'] || keys['Q']) {
                cameraLookAt.y += moveSpeed;
            }
            if (keys['e'] || keys['E']) {
                cameraLookAt.y -= moveSpeed;
            }

            // Arrow keys for rotation
            if (keys['ArrowLeft']) {
                cameraRotation.y -= rotateSpeed;
            }
            if (keys['ArrowRight']) {
                cameraRotation.y += rotateSpeed;
            }
            if (keys['ArrowUp']) {
                cameraRotation.x -= rotateSpeed;
                cameraRotation.x = Math.max(-Math.PI / 2 + 0.1, cameraRotation.x);
            }
            if (keys['ArrowDown']) {
                cameraRotation.x += rotateSpeed;
                cameraRotation.x = Math.min(Math.PI / 2 - 0.1, cameraRotation.x);
            }

            // +/- for zoom
            if (keys['+'] || keys['=']) {
                targetCameraDistance *= 0.95;
                targetCameraDistance = Math.max(minZoom, targetCameraDistance);
            }
            if (keys['-'] || keys['_']) {
                targetCameraDistance *= 1.05;
                targetCameraDistance = Math.min(maxZoom, targetCameraDistance);
            }

            // R to reset view
            if (keys['r'] || keys['R']) {
                targetCameraDistance = 40;
                cameraRotation.x = 0;
                cameraRotation.y = 0;
                cameraLookAt.set(12, 7.5, 4);
                keys['r'] = false;
                keys['R'] = false;
            }
        }

        // Animation loop
        function animate() {
            requestAnimationFrame(animate);
            handleKeyboardControls();
            updateCameraPosition();
            renderer.render(scene, camera);
        }

        // Handle window resize
        window.addEventListener('resize', () => {
            camera.aspect = window.innerWidth / window.innerHeight;
            camera.updateProjectionMatrix();
            renderer.setSize(window.innerWidth, window.innerHeight);
        });

        // Start animation
        animate();

        // Panel control functions
        function togglePanel() {
            const panel = document.getElementById('panel');
            const btn = document.getElementById('toggle-btn');
            panel.classList.toggle('collapsed');
            btn.textContent = panel.classList.contains('collapsed') ? '+' : '−';
        }

        function switchTab(tab) {
            document.querySelectorAll('.tab-btn').forEach(btn => btn.classList.remove('active'));
            document.querySelectorAll('.tab-content').forEach(content => content.classList.remove('active'));

            if (tab === 'chips') {
                document.querySelector('.tab-btn:first-child').classList.add('active');
                document.getElementById('chips-tab').classList.add('active');
            } else {
                document.querySelector('.tab-btn:last-child').classList.add('active');
                document.getElementById('connections-tab').classList.add('active');
            }
        }

        function populateChipList() {
            const list = document.getElementById('chip-list');
            list.innerHTML = '';

            chipObjects.forEach((chipObj, index) => {
                // Find all connections for this chip
                const chipConnections = connectionObjects.filter(conn =>
                    conn.src === chipObj.id || conn.dst === chipObj.id
                );

                const item = document.createElement('div');
                item.className = 'item';
                item.id = `chip-item-${chipObj.id}`;

                let connectionsHtml = '';
                if (chipConnections.length > 0) {
                    const isExpanded = expandedChips.has(chipObj.id);
                    connectionsHtml = '<div class="connections-dropdown ' + (isExpanded ? 'expanded' : '') + '" id="chip-connections-' + chipObj.id + '">';
                    chipConnections.forEach(conn => {
                        const connIndex = connectionObjects.indexOf(conn);
                        const otherChip = conn.src === chipObj.id ? conn.dst : conn.src;
                        const direction = conn.src === chipObj.id ? '&rarr;' : '&larr;';
                        const directionId = conn.src === chipObj.id ? `from-${chipObj.id}` : `to-${chipObj.id}`;
                        connectionsHtml += `
                            <div class="connection-item ${conn.visible ? '' : 'disabled'}" id="chip-conn-${directionId}-${connIndex}">
                                <span>${direction} Chip ${otherChip} (Ch ${conn.src === chipObj.id ? conn.srcCh : conn.dstCh})</span>
                                <button class="toggle-visibility" onclick="event.stopPropagation(); toggleConnectionVisibility(${connIndex})">
                                    ${conn.visible ? '&#x1F441;' : '&#x1F576;'}
                                </button>
                            </div>
                        `;
                    });
                    connectionsHtml += '</div>';
                }

                const isExpanded = expandedChips.has(chipObj.id);
                const expandBtn = chipConnections.length > 0 ?
                    '<button class="expand-btn" onclick="event.stopPropagation(); toggleChipExpand(' + chipObj.id + ')">' +
                    (isExpanded ? '▲' : '▼') + '</button>' : '';

                item.innerHTML = `
                    <div class="item-header">
                        <span class="item-title">Chip ${chipObj.id}</span>
                        <div>
                            ${expandBtn}
                            <button class="toggle-visibility" onclick="event.stopPropagation(); toggleChipVisibility(${index})">
                                ${chipObj.visible ? '&#x1F441;' : '&#x1F576;'}
                            </button>
                        </div>
                    </div>
                    <div class="item-details">
                        <strong>Tray ${chipObj.tray}, Location ${chipObj.location}</strong><br>
                        ${chipObj.connections} connections
                    </div>
                    ${connectionsHtml}
                `;
                item.onclick = (e) => {
                    if (e.target.tagName !== 'BUTTON') {
                        highlightChip(index);
                    }
                };
                list.appendChild(item);
            });

            updateChipStats();
        }

        function populateConnectionList() {
            const list = document.getElementById('connection-list');
            list.innerHTML = '';

            connectionObjects.forEach((conn, index) => {
                const item = document.createElement('div');
                item.className = 'item';
                item.id = `conn-item-${index}`;
                item.innerHTML = `
                    <div class="item-header">
                        <span class="item-title">Chip ${conn.src} &rarr; ${conn.dst}</span>
                        <button class="toggle-visibility" onclick="toggleConnectionVisibility(${index})">
                            ${conn.visible ? '&#x1F441;' : '&#x1F576;'}
                        </button>
                    </div>
                    <div class="item-details">
                        Ch ${conn.srcCh} &rarr; Ch ${conn.dstCh} | Lane ${conn.groupIndex + 1}/${conn.groupSize}
                    </div>
                `;
                item.onclick = (e) => {
                    if (e.target.tagName !== 'BUTTON') {
                        highlightConnection(index);
                    }
                };

                // Add hover effect to temporarily pop up ethernet cores
                item.onmouseenter = (e) => {
                    if (e.target.tagName !== 'BUTTON') {
                        highlightConnection(index);
                    }
                };

                // Reset when mouse leaves
                item.onmouseleave = () => {
                    // Reset to normal view (all chips visible normally)
                    chipObjects.forEach(obj => {
                        obj.mesh.traverse((child) => {
                            if (child.isMesh && child.material && child.userData) {
                                if (!child.userData.originalColor) {
                                    child.userData.originalColor = child.material.color.getHex();
                                }
                                child.material.color.setHex(child.userData.originalColor);
                                child.material.emissive.setHex(child.userData.originalColor);
                                child.material.emissiveIntensity = 0.15;
                                child.scale.set(1, 1, 1);
                                child.material.needsUpdate = true;
                            }
                        });
                    });

                    // Reset connections
                    connectionObjects.forEach(c => {
                        c.line.material.color.setHex(c.originalColor);
                        c.line.material.opacity = 0.7;
                        c.line.material.linewidth = 2.5;
                    });

                    // Remove highlight from list
                    document.querySelectorAll('.item').forEach(item => item.classList.remove('highlight'));
                };

                list.appendChild(item);
            });

            updateConnectionStats();
        }

        function toggleChipExpand(chipId) {
            const dropdown = document.getElementById(`chip-connections-${chipId}`);
            const btn = event.target;
            if (dropdown.classList.contains('expanded')) {
                dropdown.classList.remove('expanded');
                btn.textContent = '▼';
                expandedChips.delete(chipId);
            } else {
                dropdown.classList.add('expanded');
                btn.textContent = '▲';
                expandedChips.add(chipId);
            }
        }

        function toggleChipVisibility(index) {
            const chipObj = chipObjects[index];
            chipObj.visible = !chipObj.visible;
            chipObj.mesh.visible = chipObj.visible;
            chipObj.label.visible = chipObj.visible;

            const item = document.getElementById(`chip-item-${chipObj.id}`);
            if (chipObj.visible) {
                item.classList.remove('hidden');
            } else {
                item.classList.add('hidden');
            }

            // Toggle all connections related to this chip
            connectionObjects.forEach((conn, connIndex) => {
                if (conn.src === chipObj.id || conn.dst === chipObj.id) {
                    conn.visible = chipObj.visible;
                    conn.line.visible = chipObj.visible;

                    // Update connection item in connections tab
                    const connItem = document.getElementById(`conn-item-${connIndex}`);
                    if (connItem) {
                        if (conn.visible) {
                            connItem.classList.remove('hidden');
                        } else {
                            connItem.classList.add('hidden');
                        }
                    }

                    // Update connection item in chip dropdown
                    const chipConnItem = document.getElementById(`chip-conn-${connIndex}`);
                    if (chipConnItem) {
                        if (conn.visible) {
                            chipConnItem.classList.remove('disabled');
                        } else {
                            chipConnItem.classList.add('disabled');
                        }
                    }
                }
            });

            // Refresh the chip list to update button states
            populateChipList();

            updateChipStats();
            updateConnectionStats();
        }

        function toggleConnectionVisibility(index) {
            const conn = connectionObjects[index];
            conn.visible = !conn.visible;
            conn.line.visible = conn.visible;

            // Update connection item in connections tab
            const item = document.getElementById(`conn-item-${index}`);
            if (item) {
                if (conn.visible) {
                    item.classList.remove('hidden');
                } else {
                    item.classList.add('hidden');
                }
                // Update the eye icon
                const btn = item.querySelector('.toggle-visibility');
                if (btn) {
                    btn.innerHTML = conn.visible ? '&#x1F441;' : '&#x1F576;';
                }
            }

            // Update connection item in both chip dropdowns (from and to)
            // Update "from" chip's dropdown (-> direction)
            const chipConnItemFrom = document.getElementById(`chip-conn-from-${conn.src}-${index}`);
            if (chipConnItemFrom) {
                if (conn.visible) {
                    chipConnItemFrom.classList.remove('disabled');
                } else {
                    chipConnItemFrom.classList.add('disabled');
                }
                const btn = chipConnItemFrom.querySelector('.toggle-visibility');
                if (btn) {
                    btn.innerHTML = conn.visible ? '&#x1F441;' : '&#x1F576;';
                }
            }

            // Update "to" chip's dropdown (<- direction)
            const chipConnItemTo = document.getElementById(`chip-conn-to-${conn.dst}-${index}`);
            if (chipConnItemTo) {
                if (conn.visible) {
                    chipConnItemTo.classList.remove('disabled');
                } else {
                    chipConnItemTo.classList.add('disabled');
                }
                const btn = chipConnItemTo.querySelector('.toggle-visibility');
                if (btn) {
                    btn.innerHTML = conn.visible ? '&#x1F441;' : '&#x1F576;';
                }
            }

            updateConnectionStats();
        }

        function highlightChip(index) {
            // Remove previous highlights
            document.querySelectorAll('.item').forEach(item => item.classList.remove('highlight'));

            const chipObj = chipObjects[index];
            document.getElementById(`chip-item-${chipObj.id}`).classList.add('highlight');

            // Highlight chip in 3D
            chipObjects.forEach(obj => {
                obj.mesh.traverse((child) => {
                    if (child.isMesh && child.material) {
                        child.material.emissiveIntensity = 0.2;
                    }
                });
            });
            chipObj.mesh.traverse((child) => {
                if (child.isMesh && child.material) {
                    child.material.emissiveIntensity = 1.0;
                }
            });

            // Highlight connected connections
            connectionObjects.forEach(conn => {
                if (conn.src === chipObj.id || conn.dst === chipObj.id) {
                    conn.line.material.opacity = 1.0;
                    conn.line.material.linewidth = 3;
                } else {
                    conn.line.material.opacity = 0.2;
                    conn.line.material.linewidth = 1;
                }
            });
        }

        function highlightConnection(index) {
            // Remove previous highlights in UI
            document.querySelectorAll('.item').forEach(item => item.classList.remove('highlight'));

            const conn = connectionObjects[index];
            document.getElementById(`conn-item-${index}`).classList.add('highlight');

            // Get NOC coordinates for the ethernet cores involved
            const srcNoc = ethChannelToNoc[conn.srcCh];
            const dstNoc = ethChannelToNoc[conn.dstCh];

            // POP UP and highlight specific ethernet core tiles in BRIGHT ORANGE
            console.log(`=== Highlighting connection: Chip ${conn.src} Ch ${conn.srcCh} -> Chip ${conn.dst} Ch ${conn.dstCh} ===`);
            console.log(`Source NOC: (${srcNoc.x}, ${srcNoc.y}), Dest NOC: (${dstNoc.x}, ${dstNoc.y})`);

            let srcFound = false;
            let dstFound = false;

            // First pass: Reset all chips to normal state
            chipObjects.forEach(obj => {
                obj.mesh.traverse((child) => {
                    if (child.isMesh && child.material && child.userData) {
                        // Store original color if not stored
                        if (!child.userData.originalColor) {
                            child.userData.originalColor = child.material.color.getHex();
                        }
                        // Reset to original
                        child.material.color.setHex(child.userData.originalColor);
                        child.material.emissive.setHex(child.userData.originalColor);
                        child.material.emissiveIntensity = 0.15;
                        child.scale.set(1, 1, 1);
                        child.material.needsUpdate = true;
                    }
                });
            });

            // Reset all connections to their original color
            connectionObjects.forEach((c, i) => {
                c.line.material.color.setHex(c.originalColor);
                c.line.material.opacity = 0.7;
                c.line.material.linewidth = 2.5;
            });

            // Now highlight the selected connection
            conn.line.material.color.setHex(0xffff00);  // Bright yellow
            conn.line.material.opacity = 1.0;
            conn.line.material.linewidth = 4;

            // Second pass: Highlight the specific ethernet cores and dim everything else
            chipObjects.forEach(obj => {
                if (obj.id === conn.src) {
                    // Source chip - find and POP UP the ethernet core
                    console.log(`Searching source chip ${obj.id} for NOC(${srcNoc.x}, ${srcNoc.y})`);
                    obj.mesh.traverse((child) => {
                        if (child.isMesh && child.material && child.userData) {
                            if (child.userData.nocX !== undefined && child.userData.nocY !== undefined) {
                                if (child.userData.nocX === srcNoc.x && child.userData.nocY === srcNoc.y) {
                                    // FOUND IT! HIGHLIGHT & POP UP ethernet core
                                    console.log(`✓✓✓ FOUND AND POPPING UP source eth core at NOC(${child.userData.nocX}, ${child.userData.nocY}) ✓✓✓`);
                                    child.material.color.setHex(0xff0000);  // Bright red
                                    child.material.emissive.setHex(0xff3300);  // Glowing orange-red
                                    child.material.emissiveIntensity = 5.0;  // VERY bright
                                    child.scale.set(1.8, 4.0, 1.8);  // Much bigger and taller
                                    child.material.needsUpdate = true;
                                    child.updateMatrix();
                                    child.updateMatrixWorld(true);
                                    srcFound = true;
                                } else {
                                    // Dim other tiles on source chip
                                    child.material.emissiveIntensity = 0.05;
                                    child.material.needsUpdate = true;
                                }
                            }
                        }
                    });
                    if (!srcFound) console.error(`✗✗✗ DID NOT FIND source eth core at NOC(${srcNoc.x}, ${srcNoc.y}) on chip ${obj.id} ✗✗✗`);
                } else if (obj.id === conn.dst) {
                    // Destination chip - find and POP UP the ethernet core
                    console.log(`Searching dest chip ${obj.id} for NOC(${dstNoc.x}, ${dstNoc.y})`);
                    obj.mesh.traverse((child) => {
                        if (child.isMesh && child.material && child.userData) {
                            if (child.userData.nocX !== undefined && child.userData.nocY !== undefined) {
                                if (child.userData.nocX === dstNoc.x && child.userData.nocY === dstNoc.y) {
                                    // FOUND IT! HIGHLIGHT & POP UP ethernet core
                                    console.log(`✓✓✓ FOUND AND POPPING UP dest eth core at NOC(${child.userData.nocX}, ${child.userData.nocY}) ✓✓✓`);
                                    child.material.color.setHex(0xff0000);  // Bright red
                                    child.material.emissive.setHex(0xff3300);  // Glowing orange-red
                                    child.material.emissiveIntensity = 5.0;  // VERY bright
                                    child.scale.set(1.8, 4.0, 1.8);  // Much bigger and taller
                                    child.material.needsUpdate = true;
                                    child.updateMatrix();
                                    child.updateMatrixWorld(true);
                                    dstFound = true;
                                } else {
                                    // Dim other tiles on dest chip
                                    child.material.emissiveIntensity = 0.05;
                                    child.material.needsUpdate = true;
                                }
                            }
                        }
                    });
                    if (!dstFound) console.error(`✗✗✗ DID NOT FIND dest eth core at NOC(${dstNoc.x}, ${dstNoc.y}) on chip ${obj.id} ✗✗✗`);
                } else {
                    // Dim all other chips significantly
                    obj.mesh.traverse((child) => {
                        if (child.isMesh && child.material) {
                            child.material.emissiveIntensity = 0.03;
                            child.material.needsUpdate = true;
                        }
                    });
                }
            });

            if (srcFound && dstFound) {
                console.log('%c✓✓✓ Successfully popped up BOTH ethernet cores! ✓✓✓', 'color: green; font-weight: bold; font-size: 14px;');
            } else {
                console.error(`%c✗✗✗ Failed to find/popup cores: srcFound=${srcFound}, dstFound=${dstFound} ✗✗✗`, 'color: red; font-weight: bold; font-size: 14px;');
            }
        }

        function filterChips() {
            const search = document.getElementById('chip-search').value.toLowerCase();
            chipObjects.forEach((chipObj, index) => {
                const item = document.getElementById(`chip-item-${chipObj.id}`);
                const text = `chip ${chipObj.id} ${chipObj.connections}`.toLowerCase();
                if (text.includes(search)) {
                    item.style.display = 'block';
                } else {
                    item.style.display = 'none';
                }
            });
        }

        function filterConnections() {
            const search = document.getElementById('conn-search').value.toLowerCase();
            connectionObjects.forEach((conn, index) => {
                const item = document.getElementById(`conn-item-${index}`);
                const text = `chip ${conn.src} ${conn.dst} ch ${conn.srcCh} ${conn.dstCh}`.toLowerCase();
                if (text.includes(search)) {
                    item.style.display = 'block';
                } else {
                    item.style.display = 'none';
                }
            });
        }

        function updateChipStats() {
            document.getElementById('total-chips').textContent = chipObjects.length;
            document.getElementById('visible-chips').textContent = chipObjects.filter(c => c.visible).length;
        }

        function updateConnectionStats() {
            document.getElementById('total-connections').textContent = connectionObjects.length;
            document.getElementById('visible-connections').textContent = connectionObjects.filter(c => c.visible).length;
        }

        // 3D Object Selection with Raycasting
        const raycaster = new THREE.Raycaster();
        raycaster.params.Line.threshold = 0.1; // Very precise - must click directly on line
        const mouse = new THREE.Vector2();
        let selectedObject = null;
        let mouseDownPos = { x: 0, y: 0 };
        let mouseMoved = false;
        let lastClickPos = { x: 0, y: 0 };
        let lastClickTime = 0;
        let overlappingObjects = [];
        let currentOverlapIndex = 0;

        // Create info tooltip for selection
        const tooltip = document.createElement('div');
        tooltip.style.position = 'absolute';
        tooltip.style.display = 'none';
        tooltip.style.background = 'rgba(0, 0, 0, 0.9)';
        tooltip.style.color = 'white';
        tooltip.style.padding = '10px';
        tooltip.style.borderRadius = '5px';
        tooltip.style.fontSize = '12px';
        tooltip.style.pointerEvents = 'none';
        tooltip.style.zIndex = '1000';
        tooltip.style.maxWidth = '300px';
        tooltip.style.backdropFilter = 'blur(10px)';
        tooltip.style.border = '2px solid #3498db';
        document.body.appendChild(tooltip);

        function onMouseClick(event) {
            // Don't process if user was dragging (rotating camera)
            if (mouseMoved) {
                return;
            }

            // Don't process clicks on UI panel
            if (event.target.closest('#panel') || event.target.closest('#info') ||
                event.target.closest('#controls') || event.target.closest('#legend')) {
                return;
            }

            // Calculate mouse position in normalized device coordinates (-1 to +1)
            const rect = renderer.domElement.getBoundingClientRect();
            mouse.x = ((event.clientX - rect.left) / rect.width) * 2 - 1;
            mouse.y = -((event.clientY - rect.top) / rect.height) * 2 + 1;

            // Update raycaster
            raycaster.setFromCamera(mouse, camera);

            // First raycast: ONLY connections (with large threshold to make them easy to select)
            // This allows selecting connections even if chips are in front
            const connectionPickables = [];
            connectionObjects.forEach((conn, index) => {
                if (conn.visible) {
                    conn.line.userData = { type: 'connection', index: index };
                    connectionPickables.push(conn.line);
                }
            });

            const connectionIntersects = raycaster.intersectObjects(connectionPickables, false);

            // Second raycast: ONLY chips (with normal threshold, recursive for groups)
            const chipPickables = [];
            chipObjects.forEach((chipObj, index) => {
                if (chipObj.visible) {
                    // EXTEND userData on the group and all children (don't overwrite!)
                    chipObj.mesh.userData = { type: 'chip', index: index, ...chipObj.mesh.userData };
                    chipObj.mesh.traverse((child) => {
                        child.userData = { ...child.userData, type: 'chip', index: index, parent: chipObj.mesh };
                    });
                    chipPickables.push(chipObj.mesh);
                }
            });

            const chipIntersects = raycaster.intersectObjects(chipPickables, true); // Recursive for groups

            // Combine ALL results (no occlusion filtering - can select anything visible)
            const allIntersects = [];

            // Track unique chip hits (tiles within same group count as one chip)
            const chipHits = new Set();
            chipIntersects.forEach(intersect => {
                const index = intersect.object.userData.index;
                if (!chipHits.has(index)) {
                    chipHits.add(index);
                    allIntersects.push({
                        type: 'chip',
                        index: index,
                        distance: intersect.distance
                    });
                }
            });

            connectionIntersects.forEach(intersect => {
                allIntersects.push({
                    type: 'connection',
                    index: intersect.object.userData.index,
                    distance: intersect.distance
                });
            });

            // Sort by distance - closest objects first (click to cycle through all)
            allIntersects.sort((a, b) => a.distance - b.distance);

            if (allIntersects.length > 0) {
                // Check if clicking in same location within 1.5 seconds (cycle through overlapping objects)
                const currentTime = Date.now();
                const clickDistance = Math.sqrt(
                    Math.pow(event.clientX - lastClickPos.x, 2) +
                    Math.pow(event.clientY - lastClickPos.y, 2)
                );

                // If clicking in same spot (within 50px) and within 1.5 seconds, cycle through objects
                if (clickDistance < 50 && (currentTime - lastClickTime) < 1500 && overlappingObjects.length > 1) {
                    currentOverlapIndex = (currentOverlapIndex + 1) % overlappingObjects.length;
                } else {
                    // New click location - use all intersected objects sorted by distance
                    overlappingObjects = allIntersects;
                    currentOverlapIndex = 0;
                }

                lastClickPos = { x: event.clientX, y: event.clientY };
                lastClickTime = currentTime;

                // Select the current object in the cycle
                if (overlappingObjects.length > 0) {
                    const selectedObj = overlappingObjects[currentOverlapIndex];

                    if (selectedObj.type === 'chip') {
                        selectChip(selectedObj.index, event.clientX, event.clientY, overlappingObjects.length, currentOverlapIndex);
                    } else if (selectedObj.type === 'connection') {
                        selectConnection(selectedObj.index, event.clientX, event.clientY, overlappingObjects.length, currentOverlapIndex);
                    }
                }
            } else {
                deselectAll();
                overlappingObjects = [];
                currentOverlapIndex = 0;
            }
        }

        function selectChip(index, mouseX, mouseY, totalObjects = 1, currentIndex = 0) {
            deselectAll();

            const chipObj = chipObjects[index];
            selectedObject = { type: 'chip', object: chipObj };

            // Find all connections involving this chip
            const chipConnections = connectionObjects.filter(conn =>
                conn.src === chipObj.id || conn.dst === chipObj.id
            );

            // Show cycling info if multiple objects at location
            let cycleInfo = '';
            if (totalObjects > 1) {
                cycleInfo = `<div style="background: rgba(241, 196, 15, 0.2); padding: 5px; margin-top: 5px; border-radius: 3px; border-left: 3px solid #f39c12;">
                    <strong style="color: #f39c12;">&#x1F504; ${totalObjects} objects here</strong><br>
                    <span style="font-size: 10px;">Showing ${currentIndex + 1}/${totalObjects}</span><br>
                    <span style="font-size: 10px; color: #2ecc71;">Click again to cycle &#x27A1;</span>
                </div>`;
            }

            // Show tooltip
            tooltip.innerHTML = `
                <strong style="color: #f39c12;">&#x1F50D; CHIP ${chipObj.id}</strong><br>
                <span style="color: #3498db;">Tray:</span> ${chipObj.tray}<br>
                <span style="color: #3498db;">Location:</span> ${chipObj.location}<br>
                <span style="color: #2ecc71;">Connections:</span> ${chipObj.connections}<br>
                ${cycleInfo}
                <span style="color: #95a5a6; font-size: 10px; margin-top: 5px; display: block;">Click away to deselect</span>
            `;
            tooltip.style.display = 'block';
            tooltip.style.left = (mouseX + 15) + 'px';
            tooltip.style.top = (mouseY + 15) + 'px';

            // Use same highlight behavior as panel (just increase intensity, no color change)
            highlightChip(index);

            // Highlight connections and pop up the ethernet cores involved
            connectionObjects.forEach(conn => {
                if (conn.src === chipObj.id || conn.dst === chipObj.id) {
                    // Highlight this connection
                    conn.line.material.opacity = 0.9;
                    conn.line.material.linewidth = 3;
                } else {
                    // Dim other connections
                    conn.line.material.opacity = 0.1;
                }
            });

            // Pop up ethernet cores on this chip AND on connected chips
            chipConnections.forEach(conn => {
                // Get the NOC coordinates for both ends of this connection
                const srcNoc = ethChannelToNoc[conn.srcCh];
                const dstNoc = ethChannelToNoc[conn.dstCh];

                // Pop up the ethernet core on the source chip
                const srcChipObj = chipObjects.find(c => c.id === conn.src);
                if (srcChipObj) {
                    srcChipObj.mesh.traverse((child) => {
                        if (child.isMesh && child.material && child.userData) {
                            if (child.userData.nocX === srcNoc.x && child.userData.nocY === srcNoc.y) {
                                child.material.emissive.setHex(0xff6600);
                                child.material.emissiveIntensity = 1.5;
                                child.material.color.setHex(0xff6600);
                                child.scale.set(1.0, 2.0, 1.0);
                                child.material.needsUpdate = true;
                            }
                        }
                    });
                }

                // Pop up the ethernet core on the destination chip
                const dstChipObj = chipObjects.find(c => c.id === conn.dst);
                if (dstChipObj) {
                    dstChipObj.mesh.traverse((child) => {
                        if (child.isMesh && child.material && child.userData) {
                            if (child.userData.nocX === dstNoc.x && child.userData.nocY === dstNoc.y) {
                                child.material.emissive.setHex(0xff6600);
                                child.material.emissiveIntensity = 1.5;
                                child.material.color.setHex(0xff6600);
                                child.scale.set(1.0, 2.0, 1.0);
                                child.material.needsUpdate = true;
                            }
                        }
                    });
                }
            });
        }

        function selectConnection(index, mouseX, mouseY, totalObjects = 1, currentIndex = 0) {
            deselectAll();

            const conn = connectionObjects[index];
            selectedObject = { type: 'connection', object: conn };

            // Highlight connection with thicker line
            conn.line.material.color.setHex(0xf39c12);
            conn.line.material.opacity = 1.0;
            conn.line.material.linewidth = 4;

            // Get NOC coordinates for the ethernet cores involved
            const srcNoc = ethChannelToNoc[conn.srcCh];
            const dstNoc = ethChannelToNoc[conn.dstCh];

            // Show tooltip
            // Determine connection type from color
            const colorToType = {
                '0x3498db': 'Blue (Same-Tray Adjacent)',
                '0x9b59b6': 'Purple (Same-Tray Diagonal)',
                '0x2ecc71': 'Green (Cross-Tray Aligned)',
                '0xe74c3c': 'Red (Cross-Tray Diagonal)'
            };
            const connType = colorToType[conn.originalColor] || 'Unknown';

            // Show cycling info if multiple objects at location
            let cycleInfo = '';
            if (totalObjects > 1) {
                cycleInfo = `<div style="background: rgba(241, 196, 15, 0.2); padding: 5px; margin-top: 5px; border-radius: 3px; border-left: 3px solid #f39c12;">
                    <strong style="color: #f39c12;">&#x1F504; ${totalObjects} objects here</strong><br>
                    <span style="font-size: 10px;">Showing ${currentIndex + 1}/${totalObjects}</span><br>
                    <span style="font-size: 10px; color: #2ecc71;">Click again to cycle &#x27A1;</span>
                </div>`;
            }

            tooltip.innerHTML = `
                <strong style="color: #f39c12;">&#x1F50D; ETH CONNECTION</strong><br>
                <span style="color: #3498db;">From:</span> Chip ${conn.src} Ch ${conn.srcCh}<br>
                <span style="color: #95a5a6; font-size: 10px;">  └─ ETH Core @ NOC(${srcNoc.x}, ${srcNoc.y})</span><br>
                <span style="color: #3498db;">To:</span> Chip ${conn.dst} Ch ${conn.dstCh}<br>
                <span style="color: #95a5a6; font-size: 10px;">  └─ ETH Core @ NOC(${dstNoc.x}, ${dstNoc.y})</span><br>
                <span style="color: #2ecc71;">Type:</span> ${connType}<br>
                <span style="color: #9b59b6;">Routing:</span> Lane ${conn.groupIndex + 1} of ${conn.groupSize}<br>
                ${cycleInfo}
                <span style="color: #95a5a6; font-size: 10px; margin-top: 5px; display: block;">Click away to deselect</span>
            `;
            tooltip.style.display = 'block';
            tooltip.style.left = (mouseX + 15) + 'px';
            tooltip.style.top = (mouseY + 15) + 'px';

            // Highlight in panel
            document.getElementById(`conn-item-${index}`).classList.add('highlight');

            // Highlight connection line
            conn.line.material.color.setHex(0xf39c12);
            conn.line.material.opacity = 1.0;
            conn.line.material.linewidth = 4;

            // Dim all other connections
            connectionObjects.forEach(c => {
                if (c !== conn) {
                    c.line.material.opacity = 0.2;
                }
            });

            // HIGHLIGHT THE ETHERNET CORES - Pop them up in bright orange!
            console.log(`=== Highlighting ETH Connection: Chip ${conn.src} Ch ${conn.srcCh} -> Chip ${conn.dst} Ch ${conn.dstCh} ===`);
            console.log(`Source NOC: (${srcNoc.x}, ${srcNoc.y}), Type: ${typeof srcNoc.x}`);
            console.log(`Dest NOC: (${dstNoc.x}, ${dstNoc.y}), Type: ${typeof dstNoc.x}`);

            chipObjects.forEach(obj => {
                if (obj.id === conn.src) {
                    // Source chip - find and pop up the ethernet core
                    console.log(`>>> Checking source chip ${obj.id}`);
                    console.log(`>>> obj.mesh type: ${obj.mesh.type}, isGroup: ${obj.mesh.isGroup}, children count: ${obj.mesh.children ? obj.mesh.children.length : 'undefined'}`);
                    let tileCount = 0;
                    let ethTileCount = 0;
                    obj.mesh.traverse((child) => {
                        if (child.isMesh && child.material && child.userData) {
                            if (child.userData.nocX !== undefined && child.userData.nocY !== undefined) {
                                tileCount++;
                                // Log EVERY tile's coordinates for debugging
                                console.log(`  Tile #${tileCount}: NOC(${child.userData.nocX}, ${child.userData.nocY}), Type of nocX: ${typeof child.userData.nocX}`);
                                // Check if this is an ethernet tile (orange color)
                                if (child.userData.originalColor === 0xff8c00) {
                                    ethTileCount++;
                                    console.log(`  ^^ This is an ETH tile`);
                                }
                                if (child.userData.nocX === srcNoc.x && child.userData.nocY === srcNoc.y) {
                                    console.log(`>>> POPPING UP source eth core tile at NOC(${child.userData.nocX}, ${child.userData.nocY}) - Highlighting!`);
                                    // Make it visible but subtle
                                    child.material.emissive.setHex(0xff3300);  // Bright orange-red
                                    child.material.emissiveIntensity = 2.0;  // Bright
                                    child.material.color.setHex(0xff3300);
                                    child.scale.set(1.0, 2.5, 1.0);  // Same width, pop up 2.5x in height
                                    child.material.needsUpdate = true;
                                    child.updateMatrix();
                                } else {
                                    child.material.emissiveIntensity = 0.05;  // Very dim
                                    child.scale.set(1, 1, 1);
                                }
                            }
                        }
                    });
                    console.log(`Source chip ${obj.id}: Found ${tileCount} tiles total, ${ethTileCount} ethernet tiles`);
                } else if (obj.id === conn.dst) {
                    // Destination chip - find and pop up the ethernet core
                    let tileCount = 0;
                    let ethTileCount = 0;
                    obj.mesh.traverse((child) => {
                        if (child.isMesh && child.material && child.userData) {
                            if (child.userData.nocX !== undefined && child.userData.nocY !== undefined) {
                                tileCount++;
                                // Log EVERY tile's coordinates for debugging
                                console.log(`  Tile #${tileCount}: NOC(${child.userData.nocX}, ${child.userData.nocY}), Type of nocX: ${typeof child.userData.nocX}`);
                                // Check if this is an ethernet tile (orange color)
                                if (child.userData.originalColor === 0xff8c00) {
                                    ethTileCount++;
                                    console.log(`  ^^ This is an ETH tile`);
                                }
                                if (child.userData.nocX === dstNoc.x && child.userData.nocY === dstNoc.y) {
                                    console.log(`>>> POPPING UP dest eth core tile at NOC(${child.userData.nocX}, ${child.userData.nocY}) - Highlighting!`);
                                    // Make it visible but subtle
                                    child.material.emissive.setHex(0xff3300);  // Bright orange-red
                                    child.material.emissiveIntensity = 2.0;  // Bright
                                    child.material.color.setHex(0xff3300);
                                    child.scale.set(1.0, 2.5, 1.0);  // Same width, pop up 2.5x in height
                                    child.material.needsUpdate = true;
                                    child.updateMatrix();
                                } else {
                                    child.material.emissiveIntensity = 0.05;  // Very dim
                                    child.scale.set(1, 1, 1);
                                }
                            }
                        }
                    });
                    console.log(`Dest chip ${obj.id}: Found ${tileCount} tiles total, ${ethTileCount} ethernet tiles`);
                } else {
                    // Other chips - dim everything
                    obj.mesh.traverse((child) => {
                        if (child.isMesh && child.material) {
                            child.material.emissiveIntensity = 0.1;
                            child.scale.set(1, 1, 1);
                        }
                    });
                }
            });
        }

        function deselectAll() {
            // Reset all chip materials and scales
            chipObjects.forEach(chipObj => {
                // Traverse group and reset all tile materials and scales
                chipObj.mesh.traverse((child) => {
                    if (child.isMesh && child.material) {
                        // Store original color if not already stored
                        if (!child.userData.originalColor) {
                            child.userData.originalColor = child.material.color.getHex();
                        }
                        // Reset to original color
                        child.material.color.setHex(child.userData.originalColor);
                        // Reset emissive based on tile's original color
                        child.material.emissive.setHex(child.userData.originalColor);
                        child.material.emissiveIntensity = 0.15;
                        // Reset scale for individual tiles
                        child.scale.set(1, 1, 1);
                        child.material.needsUpdate = true;
                    }
                });
                chipObj.mesh.scale.set(1, 1, 1);
            });

            // Reset all connection materials
            connectionObjects.forEach(conn => {
                conn.line.material.color.setHex(conn.originalColor);
                conn.line.material.opacity = 0.7;
                conn.line.material.linewidth = 2.5;
            });

            // Hide tooltip
            tooltip.style.display = 'none';

            // Clear highlights in panel
            document.querySelectorAll('.item').forEach(item => item.classList.remove('highlight'));

            // Reset cursor
            renderer.domElement.style.cursor = 'default';

            // Clear selection state
            selectedObject = null;
            overlappingObjects = [];
            currentOverlapIndex = 0;
        }

        // Add click event listener
        renderer.domElement.addEventListener('click', onMouseClick, false);

        // Hover highlighting (for better UX - shows what's clickable)
        let hoveredObject = null;
        renderer.domElement.addEventListener('mousemove', (event) => {
            // Update tooltip position if visible
            if (tooltip.style.display === 'block') {
                tooltip.style.left = (event.clientX + 15) + 'px';
                tooltip.style.top = (event.clientY + 15) + 'px';
            }

            // Don't hover-highlight while dragging
            if (isDragging) {
                return;
            }

            // Calculate mouse position
            const rect = renderer.domElement.getBoundingClientRect();
            mouse.x = ((event.clientX - rect.left) / rect.width) * 2 - 1;
            mouse.y = -((event.clientY - rect.top) / rect.height) * 2 + 1;

            raycaster.setFromCamera(mouse, camera);

            // Separate raycasts for connections and chips (prioritize connections)
            const connectionPickables = [];
            connectionObjects.forEach((conn) => {
                if (conn.visible) {
                    connectionPickables.push(conn.line);
                }
            });
            const connectionIntersects = raycaster.intersectObjects(connectionPickables, false);

            const chipPickables = [];
            chipObjects.forEach((chipObj, index) => {
                if (chipObj.visible) {
                    chipObj.mesh.traverse((child) => {
                        child.userData = { ...child.userData, type: 'chip', index: index, parent: chipObj.mesh };
                    });
                    chipPickables.push(chipObj.mesh);
                }
            });
            const chipIntersects = raycaster.intersectObjects(chipPickables, true); // Recursive for groups

            // Track unique chip hits for hover (tiles within same group count as one chip)
            const uniqueChipHits = new Map();
            chipIntersects.forEach(intersect => {
                const index = intersect.object.userData.index;
                if (!uniqueChipHits.has(index)) {
                    uniqueChipHits.set(index, intersect);
                }
            });

            // Combine all intersects and sort by distance (closest first, no occlusion)
            const allHoverIntersects = [
                ...connectionIntersects.map(i => ({ ...i, objType: 'connection' })),
                ...Array.from(uniqueChipHits.values()).map(i => ({ ...i, objType: 'chip' }))
            ].sort((a, b) => a.distance - b.distance);

            // Clear previous hover
            if (hoveredObject) {
                // Check if hover object is the currently selected object
                const isHoverSelected = selectedObject &&
                    hoveredObject.type === selectedObject.type &&
                    hoveredObject.object === selectedObject.object;

                if (!isHoverSelected) {
                    // Clear hover highlight (not the selected object)
                    if (hoveredObject.type === 'chip') {
                        hoveredObject.object.mesh.scale.set(1, 1, 1);
                    } else if (hoveredObject.type === 'connection') {
                        // Reset connection to appropriate state based on selection context
                        const conn = hoveredObject.object;
                        const originalColorHex = conn.originalColor || 0x3498db;
                        conn.line.material.color.setHex(originalColorHex);
                        conn.line.material.linewidth = 2.5;

                        // If a chip is selected, restore faded state; otherwise restore normal state
                        if (selectedObject && selectedObject.type === 'chip') {
                            const selectedChipId = selectedObject.object.id;
                            // Check if this connection is related to selected chip
                            if (conn.src === selectedChipId || conn.dst === selectedChipId) {
                                conn.line.material.opacity = 1.0;
                                conn.line.material.linewidth = 3;
                            } else {
                                conn.line.material.opacity = 0.2;
                                conn.line.material.linewidth = 2.5;
                            }
                        } else if (selectedObject && selectedObject.type === 'connection') {
                            // Another connection is selected, dim this one
                            conn.line.material.opacity = 0.2;
                        } else {
                            // Nothing selected, restore normal opacity
                            conn.line.material.opacity = 0.7;
                        }
                    }
                }
                hoveredObject = null;
                renderer.domElement.style.cursor = 'default';
            }

            // Apply new hover to CLOSEST object (works even when something is selected)
            if (allHoverIntersects.length > 0) {
                const closestIntersect = allHoverIntersects[0];
                const userData = closestIntersect.object.userData;

                if (userData.type === 'chip') {
                    const chipObj = chipObjects[userData.index];

                    // Check if this is the currently selected chip
                    const isSelected = selectedObject &&
                                      selectedObject.type === 'chip' &&
                                      selectedObject.object === chipObj;

                    if (!isSelected) {
                        // Apply hover effect (outline or scale boost)
                        chipObj.mesh.scale.set(1.05, 1.05, 1.05);
                    }
                    // If it's selected, it already has the selection highlight

                    hoveredObject = { type: 'chip', object: chipObj };
                    renderer.domElement.style.cursor = 'pointer';
                } else if (userData.type === 'connection') {
                    const conn = connectionObjects[userData.index];

                    // Check if this is the currently selected connection
                    const isSelected = selectedObject &&
                                      selectedObject.type === 'connection' &&
                                      selectedObject.object === conn;

                    if (!isSelected) {
                        // Apply hover effect (cyan/teal color for preview)
                        conn.line.material.linewidth = 3;
                        conn.line.material.opacity = 1.0;
                        conn.line.material.color.setHex(0x1abc9c); // Teal for hover preview
                    }
                    // If it's selected, it already has the orange selection highlight

                    hoveredObject = { type: 'connection', object: conn };
                    renderer.domElement.style.cursor = 'pointer';
                }
            } else {
                // No intersects - make sure cursor is default
                renderer.domElement.style.cursor = 'default';
            }
        });

        // Initialize lists
        setTimeout(() => {
            populateChipList();
            populateConnectionList();
        }, 100);

        console.log('3D Topology Visualization loaded successfully!');
        console.log('Click on chips or connections to select and view details');
    </script>
</body>
</html>
)HTML";

        file.close();

        std::cout << "[OK] 3D HTML visualization exported to: " << filename << "\n";
        std::cout << "  Interactive 3D view with " << connections.size() << " connections\n";
        std::cout << "  Connections routed at different heights to avoid overlaps\n";
        std::cout << "  Open " << filename << " in a web browser to explore\n";
    }
};
