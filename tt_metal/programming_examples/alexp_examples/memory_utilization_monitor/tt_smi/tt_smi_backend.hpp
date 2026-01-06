// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <string>
#include <vector>
#include <cstdint>

namespace tt_smi {

// Device telemetry data
struct TelemetryData {
    float temperature = -1.0f;
    float power = -1.0f;      // TDP in Watts
    uint32_t voltage_mv = 0;  // Vcore in millivolts
    uint32_t current_ma = 0;  // TDC in milliamps
    uint32_t aiclk_mhz = 0;
    std::string status = "Unknown";
    bool available = false;
};

// Per-process memory info
struct ProcessMemory {
    int pid = 0;
    std::string name;
    uint64_t dram_allocated = 0;
    uint64_t l1_allocated = 0;
    uint64_t l1_small_allocated = 0;
    uint64_t trace_allocated = 0;
    uint64_t cb_allocated = 0;
    uint64_t kernel_allocated = 0;
};

// Device information
struct Device {
    // Identity
    uint64_t chip_id = 0;
    uint64_t asic_id = 0;
    std::string arch_name;
    bool is_remote = false;

    // Display identifiers
    uint32_t tray_id = 0;
    uint32_t chip_in_tray = 0;
    uint8_t asic_location = 0;
    std::string display_id;  // Formatted: "T1:N5" or "1834:0R"

    // Telemetry
    TelemetryData telemetry;

    // Memory stats - initialized to 0 to avoid garbage values when no SHM
    uint64_t total_dram = 0;
    uint64_t used_dram = 0;
    uint64_t total_l1 = 0;
    uint64_t used_l1 = 0;
    uint64_t used_l1_small = 0;
    uint64_t used_trace = 0;
    uint64_t used_cb = 0;
    uint64_t used_kernel = 0;

    // Per-process breakdown
    std::vector<ProcessMemory> processes;

    // SHM availability
    bool has_shm = false;
};

// API Functions
std::vector<Device> enumerate_devices(bool shm_only = false);
bool update_device_telemetry(Device& device);
bool update_device_memory(Device& device);
int cleanup_dead_processes();
std::string format_bytes(uint64_t bytes);

// Device reset (warm reset via UMD)
void reset_devices(const std::vector<int>& device_ids = {}, bool reset_m3 = false);

}  // namespace tt_smi
