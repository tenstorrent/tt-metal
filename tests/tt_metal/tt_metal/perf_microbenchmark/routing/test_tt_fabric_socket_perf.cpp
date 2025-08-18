// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <vector>
#include <unordered_map>
#include <map>
#include <algorithm>
#include <fstream>
#include <random>
#include <filesystem>
#include <optional>
#include <iomanip>
#include <sstream>
#include <memory>

#include <iostream>
#include <chrono>
#include <cstring>

// TODO: include real fabric API
// #include "tt_metal/fabric/hw/inc/tt_fabric_api.h"

int main() {
    // Config (hard-coded for now)
    int src_device = 0;
    int dst_device = 1;
    int src_x = 0, src_y = 0;
    int dst_x = 0, dst_y = 0;
    size_t size_bytes = 1024;
    int num_packets = 100;

    // --- Fabric init ---
    // tt_fabric_init();

    // --- Create socket on source ---
    // auto src_sock = tt_fabric_socket_create(src_device, src_x, src_y, /*...*/);
    // auto dst_sock = tt_fabric_socket_create(dst_device, dst_x, dst_y, /*...*/);

    // --- Connect ---
    // tt_fabric_socket_connect(src_sock, dst_device, dst_x, dst_y, /*...*/);

    // --- Prepare payload ---
    char* buf = new char[size_bytes];
    std::memset(buf, 0xAB, size_bytes);

    // --- Send loop ---
    auto t0 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < num_packets; ++i) {
        // tt_fabric_send(src_sock, buf, size_bytes, /*flags*/ 0);
    }
    auto t1 = std::chrono::high_resolution_clock::now();

    // --- Measure simple average latency ---
    double total_s = std::chrono::duration<double>(t1 - t0).count();
    double avg_us = (total_s / num_packets) * 1e6;
    double throughput_gbps = (size_bytes * num_packets * 8.0) / (total_s * 1e9);

    std::cout << "Avg latency (us): " << avg_us << "\n";
    std::cout << "Throughput (Gbps): " << throughput_gbps << "\n";

    delete[] buf;

    // --- Cleanup ---
    // tt_fabric_socket_close(src_sock);
    // tt_fabric_socket_close(dst_sock);
    // tt_fabric_shutdown();

    return 0;
}
