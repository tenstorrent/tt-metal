// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// tt-smi-simple: Simplified telemetry tool for MMIO-based devices
// Shows only telemetry from UMD, no DRAM allocation tracking

#include <iostream>
#include <iomanip>
#include <vector>
#include <map>
#include <string>
#include <chrono>
#include <thread>
#include <memory>
#include <optional>

// TT-UMD includes for direct device access
#include "umd/device/tt_device/tt_device.hpp"
#include "umd/device/firmware/firmware_info_provider.hpp"
#include "umd/device/types/arch.hpp"
#include "umd/device/cluster.hpp"
#include "umd/device/cluster_descriptor.hpp"

// TT-Metal includes (for device enumeration fallback)
#include <tt-metalium/host_api.hpp>

#define VERSION "v1.0"

namespace fs = std::filesystem;

// ANSI colors
namespace Color {
const char* RESET = "\033[0m";
const char* BOLD = "\033[1m";
const char* GREEN = "\033[32m";
const char* YELLOW = "\033[33m";
const char* RED = "\033[31m";
const char* CYAN = "\033[36m";
const char* WHITE = "\033[37m";
const char* BLUE = "\033[34m";
}  // namespace Color

struct TelemetryData {
    double asic_temperature = -1.0;
    std::optional<double> board_temperature;
    std::optional<uint32_t> aiclk;      // MHz
    std::optional<uint32_t> axiclk;     // MHz
    std::optional<uint32_t> arcclk;     // MHz
    std::optional<uint32_t> fan_speed;  // RPM
    std::optional<uint32_t> tdp;        // Watts
    std::optional<uint32_t> tdc;        // Amps
    std::optional<uint32_t> vcore;      // mV
};

struct DeviceInfo {
    int device_id;
    std::string arch_name;
    TelemetryData telemetry;
    bool is_available;
    bool telemetry_busy;  // Indicates if telemetry is temporarily unavailable due to device usage
};

class TTSmiSimple {
private:
    // Cache of UMD devices for telemetry (device_id -> TTDevice)
    mutable std::map<int, std::unique_ptr<tt::umd::TTDevice>> umd_device_cache_;

    // Cluster descriptor for topology discovery
    mutable std::unique_ptr<tt::umd::ClusterDescriptor> cluster_descriptor_;

    // Ensure cluster descriptor is initialized
    void ensure_cluster_descriptor_initialized() const {
        if (!cluster_descriptor_) {
            try {
                cluster_descriptor_ = tt::umd::Cluster::create_cluster_descriptor();
            } catch (const std::exception& e) {
                // If topology discovery fails, we'll fall back to local-only mode
                cluster_descriptor_ = nullptr;
            }
        }
    }

    // Check if a device is remote (without initializing it)
    bool is_device_remote(int device_id) const {
        ensure_cluster_descriptor_initialized();
        if (cluster_descriptor_) {
            try {
                return cluster_descriptor_->is_chip_remote(device_id);
            } catch (...) {
                return false;
            }
        }
        return false;
    }

    // Get or create cached UMD device for telemetry (local devices only)
    tt::umd::TTDevice* get_umd_device(int device_id, bool show_status = false) const {
        // Check if device is already cached
        auto it = umd_device_cache_.find(device_id);
        if (it != umd_device_cache_.end()) {
            return it->second.get();
        }

        // Skip remote devices - they don't need UMD telemetry
        if (is_device_remote(device_id)) {
            if (show_status) {
                std::cout << Color::YELLOW << "⚙️  Skipping device " << device_id << " (remote - no telemetry)"
                          << Color::RESET << std::endl;
            }
            return nullptr;
        }

        // Create and cache new device (local PCIe only)
        try {
            if (show_status) {
                std::cout << Color::YELLOW << "⚙️  Initializing device " << device_id << " for telemetry..."
                          << Color::RESET << std::flush;
            }

            std::unique_ptr<tt::umd::TTDevice> tt_device = tt::umd::TTDevice::create(device_id);

            if (!tt_device) {
                if (show_status) {
                    std::cout << " " << Color::RED << "Failed" << Color::RESET << std::endl;
                }
                return nullptr;
            }

            // Initialize the device (lightweight init for telemetry)
            tt_device->init_tt_device();

            if (show_status) {
                std::cout << " " << Color::GREEN << "Done" << Color::RESET << std::endl;
            }

            // Store in cache and return pointer
            auto* device_ptr = tt_device.get();
            umd_device_cache_[device_id] = std::move(tt_device);
            return device_ptr;

        } catch (const std::exception& e) {
            if (show_status) {
                std::cout << " " << Color::RED << "Error: " << e.what() << Color::RESET << std::endl;
            }
            return nullptr;
        }
    }

    // Read telemetry from a cached UMD device with retry logic
    std::pair<TelemetryData, bool> read_telemetry_from_cached_device(tt::umd::TTDevice* tt_device) const {
        TelemetryData data;
        bool telemetry_busy = false;

        if (!tt_device) {
            return {data, telemetry_busy};
        }

        // Try to read telemetry with retry logic (up to 3 attempts)
        const int max_retries = 3;
        bool telemetry_available = false;

        for (int attempt = 0; attempt < max_retries && !telemetry_available; ++attempt) {
            try {
                // Get firmware info provider for telemetry
                auto firmware_info = tt_device->get_firmware_info_provider();
                if (!firmware_info) {
                    telemetry_busy = true;
                    continue;
                }

                // Read basic telemetry first (temperature and clocks are usually available)
                double temp = firmware_info->get_asic_temperature();
                if (temp >= -50.0 && temp <= 100.0) {
                    data.asic_temperature = temp;
                    telemetry_available = true;
                }

                auto aiclk = firmware_info->get_aiclk();
                if (aiclk.has_value() && aiclk.value() > 0 && aiclk.value() <= 1100) {  // Max 1100 MHz for TT chips
                    data.aiclk = aiclk;
                    telemetry_available = true;
                }

                // Try to read power-related telemetry (these might fail when device is busy)
                bool power_telemetry_failed = false;

                try {
                    auto tdp = firmware_info->get_tdp();
                    if (tdp.has_value() && tdp.value() > 0 && tdp.value() <= 300) {
                        data.tdp = tdp;
                    } else {
                        power_telemetry_failed = true;
                    }
                } catch (...) {
                    power_telemetry_failed = true;
                }

                try {
                    auto tdc = firmware_info->get_tdc();
                    if (tdc.has_value() && tdc.value() > 0 && tdc.value() <= 350) {
                        data.tdc = tdc;
                    } else {
                        power_telemetry_failed = true;
                    }
                } catch (...) {
                    power_telemetry_failed = true;
                }

                try {
                    auto vcore = firmware_info->get_vcore();
                    if (vcore.has_value() && vcore.value() > 0 && vcore.value() <= 950) {  // Max 950mV for TT chips
                        data.vcore = vcore;
                    } else {
                        power_telemetry_failed = true;
                    }
                } catch (...) {
                    power_telemetry_failed = true;
                }

                // If basic telemetry is available but power telemetry failed,
                // mark as busy (device might be under load)
                if (telemetry_available && power_telemetry_failed) {
                    telemetry_busy = true;
                }

                // Read other telemetry values
                auto board_temp = firmware_info->get_board_temperature();
                if (board_temp.has_value() && board_temp.value() >= -50.0 && board_temp.value() <= 100.0) {
                    data.board_temperature = board_temp;
                }

                auto axiclk = firmware_info->get_axiclk();
                if (axiclk.has_value() && axiclk.value() > 0 && axiclk.value() <= 1100) {  // Max 1100 MHz for TT chips
                    data.axiclk = axiclk;
                }

                auto arcclk = firmware_info->get_arcclk();
                if (arcclk.has_value() && arcclk.value() > 0 && arcclk.value() <= 1100) {  // Max 1100 MHz for TT chips
                    data.arcclk = arcclk;
                }

                auto fan_speed = firmware_info->get_fan_speed();
                if (fan_speed.has_value() && fan_speed.value() < 20000) {
                    data.fan_speed = fan_speed;
                }

                // If we got here and have basic telemetry, we're done
                if (telemetry_available) {
                    break;
                }

            } catch (const std::exception& e) {
                // If this is not the last attempt, mark as busy and retry
                if (attempt < max_retries - 1) {
                    telemetry_busy = true;
                    std::this_thread::sleep_for(std::chrono::milliseconds(10));  // Brief pause before retry
                }
            }
        }

        return {data, telemetry_busy};
    }

    // Read telemetry from device by ID (wrapper for convenience)
    std::pair<TelemetryData, bool> read_telemetry_from_device(int device_id) {
        auto* tt_device = get_umd_device(device_id, false);
        return read_telemetry_from_cached_device(tt_device);
    }

    // Get architecture name
    const char* get_arch_name(tt::ARCH arch) {
        switch (arch) {
            case tt::ARCH::GRAYSKULL: return "Grayskull";
            case tt::ARCH::WORMHOLE_B0: return "Wormhole_B0";
            case tt::ARCH::BLACKHOLE: return "Blackhole";
            case tt::ARCH::QUASAR: return "Quasar";
            default: return "Unknown";
        }
    }

    // Print device table header
    void print_device_header() {
        std::cout << Color::BOLD << Color::CYAN;
        std::cout << "┌────────────────────────────────────────────────────────────────────────────────────────────────"
                     "─────────────────┐"
                  << std::endl;
        std::cout << "│ ";
        std::cout << std::left << std::setw(30) << "tt-smi-simple " VERSION;
        std::cout << std::right << std::setw(71);

        // Get current time
        auto now = std::chrono::system_clock::now();
        auto time_t = std::chrono::system_clock::to_time_t(now);
        auto tm = *std::localtime(&time_t);

        std::ostringstream time_str;
        time_str << std::put_time(&tm, "%H:%M:%S");
        std::cout << time_str.str();
        std::cout << " │" << std::endl;

        std::cout << "├────────────────────────────────────────────────────────────────────────────────────────────────"
                     "─────────────────┤"
                  << std::endl;
        std::cout << "│ " << std::left << std::setw(4) << "Chip" << std::setw(14) << "Name" << std::setw(12) << "Temp"
                  << std::setw(10) << "Power" << std::setw(8) << "Curr" << std::setw(10) << "Volt" << std::setw(12)
                  << "AICLK" << std::setw(8) << "HB" << std::setw(12) << "Status" << " │" << std::endl;
        std::cout << "├────────────────────────────────────────────────────────────────────────────────────────────────"
                     "─────────────────┤"
                  << Color::RESET << std::endl;
    }

    void print_device_footer() {
        std::cout << Color::BOLD << Color::CYAN;
        std::cout << "└────────────────────────────────────────────────────────────────────────────────────────────────"
                     "─────────────────┘";
        std::cout << Color::RESET << std::endl;
    }

public:
    TTSmiSimple() {}

    ~TTSmiSimple() {
        // device_cache_ will be auto-cleaned by unique_ptr
    }

    void run(bool watch_mode = false, int refresh_ms = 1000) {
        static bool first_run = true;

        do {
            // Clear screen for watch mode
            if (watch_mode && !first_run) {
                std::cout << "\033[2J\033[H";  // Clear screen and move to top
            }

            // Show initialization status on first run
            if (first_run) {
                std::cout << Color::CYAN << "Initializing UMD telemetry..." << Color::RESET << std::endl;
            }

            // Discover devices
            std::vector<DeviceInfo> devices;

            // Use cluster descriptor to enumerate all devices
            ensure_cluster_descriptor_initialized();

            if (cluster_descriptor_) {
                // Use topology discovery to enumerate all devices
                auto all_chips = cluster_descriptor_->get_all_chips();

                // Sort chips so local ones are processed first
                auto chips_local_first = cluster_descriptor_->get_chips_local_first(all_chips);

                for (int chip_id : chips_local_first) {
                    DeviceInfo dev;
                    dev.device_id = chip_id;
                    dev.is_available = false;
                    dev.telemetry_busy = false;

                    // Check if this is a remote device
                    bool is_remote = cluster_descriptor_->is_chip_remote(chip_id);

                    if (is_remote) {
                        // Remote device - get arch from cluster descriptor
                        auto arch = cluster_descriptor_->get_arch(chip_id);
                        if (arch == tt::ARCH::WORMHOLE_B0) {
                            dev.arch_name = "Wormhole_B0";
                        } else if (arch == tt::ARCH::BLACKHOLE) {
                            dev.arch_name = "Blackhole";
                        } else if (arch == tt::ARCH::GRAYSKULL) {
                            dev.arch_name = "Grayskull";
                        } else {
                            dev.arch_name = "Unknown";
                        }
                        // No telemetry for remote devices
                        dev.telemetry = TelemetryData();
                        dev.is_available = true;  // Device exists but no telemetry
                    } else {
                        // Local device - try to read telemetry
                        dev.arch_name = "Unknown";
                        auto* tt_device = get_umd_device(chip_id, first_run);
                        if (tt_device) {
                            auto [telemetry, busy] = read_telemetry_from_cached_device(tt_device);
                            dev.telemetry = telemetry;
                            dev.telemetry_busy = busy;
                            // Get architecture name from device
                            auto arch = tt_device->get_arch();
                            dev.arch_name = get_arch_name(arch);
                            dev.is_available = true;
                        } else {
                            // Device initialization failed
                            dev.telemetry = TelemetryData();
                            dev.telemetry_busy = false;
                            dev.is_available = false;
                        }
                    }

                    devices.push_back(dev);
                }
            } else {
                // Fallback: try local PCIe devices only
                for (int i = 0; i < 8; ++i) {
                    DeviceInfo dev;
                    dev.device_id = i;
                    dev.arch_name = "Unknown";
                    dev.telemetry_busy = false;

                    // Try to read telemetry
                    auto* tt_device = get_umd_device(i, first_run);
                    if (tt_device) {
                        auto [telemetry, busy] = read_telemetry_from_cached_device(tt_device);
                        dev.telemetry = telemetry;
                        dev.telemetry_busy = busy;
                        dev.arch_name = get_arch_name(tt_device->get_arch());
                        dev.is_available = true;
                        devices.push_back(dev);
                    }
                }
            }

            // Check initialization status
            if (first_run) {
                int local_devices = 0;
                int remote_devices = 0;
                int failed_devices = 0;

                for (const auto& dev : devices) {
                    if (is_device_remote(dev.device_id)) {
                        remote_devices++;
                    } else if (dev.is_available) {
                        local_devices++;
                    } else {
                        failed_devices++;
                    }
                }

                if (failed_devices > 0) {
                    std::cout << Color::YELLOW
                              << "⚠  Some local devices couldn't be initialized (may be in use by another process)"
                              << Color::RESET << std::endl;
                }

                if (local_devices > 0) {
                    std::cout << Color::GREEN << "✓ Telemetry initialized for " << local_devices << " local device"
                              << (local_devices > 1 ? "s" : "") << Color::RESET << std::endl;
                }

                if (remote_devices > 0) {
                    std::cout << Color::CYAN << "ℹ  " << remote_devices << " remote device"
                              << (remote_devices > 1 ? "s" : "") << " detected (no telemetry)" << Color::RESET
                              << std::endl;
                }

                if (!devices.empty()) {
                    std::cout << std::endl;
                }

                first_run = false;
            }

            // Print device table
            print_device_header();

            for (const auto& dev : devices) {
                std::cout << Color::BOLD << "│ " << Color::RESET;
                std::cout << std::left << std::setw(4) << dev.device_id;
                std::cout << std::setw(14) << dev.arch_name;

                // Temperature
                if (dev.telemetry.asic_temperature >= 0) {
                    std::cout << std::fixed << std::setprecision(1) << std::setw(12)
                              << (std::to_string(static_cast<int>(dev.telemetry.asic_temperature)) + "°C");
                } else {
                    std::cout << std::setw(12) << "N/A";
                }

                // Power
                if (dev.telemetry.tdp.has_value()) {
                    std::cout << std::setw(10) << (std::to_string(dev.telemetry.tdp.value()) + "W");
                } else {
                    std::cout << std::setw(10) << "N/A";
                }

                // Current (amps)
                if (dev.telemetry.tdc.has_value()) {
                    std::cout << std::setw(8) << (std::to_string(dev.telemetry.tdc.value()) + "A");
                } else {
                    std::cout << std::setw(8) << "N/A";
                }

                // Voltage
                if (dev.telemetry.vcore.has_value()) {
                    std::cout << std::setw(10) << (std::to_string(dev.telemetry.vcore.value()) + "mV");
                } else {
                    std::cout << std::setw(10) << "N/A";
                }

                // AICLK
                if (dev.telemetry.aiclk.has_value()) {
                    std::cout << std::setw(12) << (std::to_string(dev.telemetry.aiclk.value()) + "M");
                } else {
                    std::cout << std::setw(12) << "N/A";
                }

                // Heartbeat - shows device status and telemetry availability
                bool has_telemetry =
                    (dev.telemetry.asic_temperature >= 0 || dev.telemetry.tdp.has_value() ||
                     dev.telemetry.aiclk.has_value());
                if (is_device_remote(dev.device_id)) {
                    std::cout << std::setw(8) << Color::BLUE << "●" << Color::RESET;
                } else if (dev.telemetry_busy) {
                    std::cout << std::setw(8) << Color::YELLOW << "◐" << Color::RESET;  // Busy indicator
                } else if (has_telemetry) {
                    std::cout << std::setw(8) << Color::GREEN << "●" << Color::RESET;
                } else {
                    std::cout << std::setw(8) << Color::YELLOW << "○" << Color::RESET;
                }

                // Status
                if (is_device_remote(dev.device_id)) {
                    std::cout << std::setw(12) << Color::CYAN << "Remote" << Color::RESET;
                } else if (dev.telemetry_busy) {
                    std::cout << std::setw(12) << Color::YELLOW << "Busy" << Color::RESET;
                } else if (dev.is_available) {
                    std::cout << std::setw(12) << Color::GREEN << "OK" << Color::RESET;
                } else {
                    std::cout << std::setw(12) << Color::RED << "Failed" << Color::RESET;
                }

                std::cout << " │" << std::endl;
            }

            print_device_footer();

            // Print footer info
            std::cout << "\n"
                      << Color::CYAN << "💡 Telemetry source: UMD (direct firmware access)" << Color::RESET
                      << std::endl;

            if (watch_mode) {
                std::cout << Color::CYAN << "📋 Press 'q' to quit, any other key to refresh" << Color::RESET
                          << std::endl;
                std::cout << std::flush;

                // Simple key input (non-blocking)
                struct timeval tv = {0, 0};
                fd_set fds;
                FD_ZERO(&fds);
                FD_SET(STDIN_FILENO, &fds);

                if (select(1, &fds, NULL, NULL, &tv) > 0) {
                    char c;
                    if (read(STDIN_FILENO, &c, 1) > 0) {
                        if (c == 'q' || c == 'Q') {
                            break;
                        }
                    }
                }

                std::this_thread::sleep_for(std::chrono::milliseconds(refresh_ms));
            }

        } while (watch_mode);
    }
};

int main(int argc, char* argv[]) {
    bool watch_mode = false;
    int refresh_ms = 1000;

    // Parse arguments
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "-w" || arg == "--watch") {
            watch_mode = true;
        } else if (arg == "-r" && i + 1 < argc) {
            refresh_ms = std::stoi(argv[++i]);
        } else if (arg == "-h" || arg == "--help") {
            std::cout << "tt-smi-simple: Simplified Tenstorrent System Management Interface" << std::endl;
            std::cout << "\nUsage: " << argv[0] << " [OPTIONS]" << std::endl;
            std::cout << "\nOptions:" << std::endl;
            std::cout << "  -w, --watch    Watch mode (refresh continuously)" << std::endl;
            std::cout << "  -r <ms>        Refresh interval in milliseconds (default: 1000)" << std::endl;
            std::cout << "  -h, --help     Show this help" << std::endl;
            std::cout << "\nExamples:" << std::endl;
            std::cout << "  " << argv[0] << "              # Show current telemetry" << std::endl;
            std::cout << "  " << argv[0] << " -w            # Watch mode with 1s refresh" << std::endl;
            std::cout << "  " << argv[0] << " -w -r 500     # Watch with 500ms refresh" << std::endl;
            std::cout << "\nFeatures:" << std::endl;
            std::cout << "  • Direct firmware telemetry via UMD (temperature, power, current, voltage, clocks)"
                      << std::endl;
            std::cout << "  • Chip status indicators:" << std::endl;
            std::cout << "    ● Green: Responding normally" << std::endl;
            std::cout << "    ◐ Yellow: Device busy (some telemetry may be N/A)" << std::endl;
            std::cout << "    ○ Yellow: Limited telemetry available" << std::endl;
            std::cout << "    ● Blue: Remote device" << std::endl;
            std::cout << "  • Works with local AND remote devices" << std::endl;
            std::cout << "  • Lightweight - no memory allocation tracking" << std::endl;
            std::cout << "  • Simple table format" << std::endl;
            return 0;
        }
    }

    try {
        TTSmiSimple smi;
        smi.run(watch_mode, refresh_ms);
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
