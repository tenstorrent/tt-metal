// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// This is a simplified clean routing version for topology_3d_export.hpp
// Replace lines 693-804 with this content:

// Simple reliable routing
double dx = dst_x - src_x;
double dy = dst_y - src_y;
double dz = dst_z - src_z;

double abs_dx = std::abs(dx);
double abs_dy = std::abs(dy);
double abs_dz = std::abs(dz);

double clearance = CHIP_SIZE + 1.0;
double lift_height = TRAY_HEIGHT * 0.4 + conn.z_level * 0.5;

// Always route forward in Z and right in X to avoid negative coordinates
double route_z = std::max(src_z, dst_z) + clearance + CHIP_SPACING * conn.z_level;
double route_x = std::max(src_x, dst_x) + clearance + CHIP_SPACING * conn.z_level;
double mid_x = (src_x + dst_x) / 2.0;
double mid_z = (src_z + dst_z) / 2.0;

file << "        {\n";
file << "            const curve = new THREE.CatmullRomCurve3([\n";
file << "                new THREE.Vector3(" << src_x_final << ", " << src_y_final << ", " << src_z_final << "),\n";

if (abs_dy > 0.1) {
    // Cross-tray: arc high
    int tray_span = std::abs(static_cast<int>(dst_y / TRAY_HEIGHT) - static_cast<int>(src_y / TRAY_HEIGHT));
    double peak_y = std::max(src_y, dst_y) + TRAY_HEIGHT * 0.5 + tray_span * TRAY_HEIGHT * 0.3;
    file << "                new THREE.Vector3(" << mid_x << ", " << peak_y << ", " << route_z << "),\n";
} else if (abs_dx > abs_dz) {
    // Horizontal same-tray
    file << "                new THREE.Vector3(" << src_x << ", " << (src_y + lift_height) << ", " << route_z << "),\n";
    file << "                new THREE.Vector3(" << mid_x << ", " << (src_y + lift_height) << ", " << route_z << "),\n";
    file << "                new THREE.Vector3(" << dst_x << ", " << (dst_y + lift_height) << ", " << route_z << "),\n";
} else {
    // Depth same-tray
    file << "                new THREE.Vector3(" << route_x << ", " << (src_y + lift_height) << ", " << src_z << "),\n";
    file << "                new THREE.Vector3(" << route_x << ", " << (src_y + lift_height) << ", " << mid_z << "),\n";
    file << "                new THREE.Vector3(" << route_x << ", " << (dst_y + lift_height) << ", " << dst_z << "),\n";
}

file << "                new THREE.Vector3(" << dst_x_final << ", " << dst_y_final << ", " << dst_z_final << ")\n";
file << "            ]);\n";
