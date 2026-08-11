// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include <cstdlib>
#include <cstddef>
#include <chrono>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <thread>
#include <vector>

#include <fmt/core.h>

#include <tt-metalium/experimental/dispatch_telemetry.hpp>
#include <umd/device/pcie/pci_device.hpp>
#include <umd/device/tt_device/tt_device.hpp>

namespace {

std::string format_optional_float(const std::optional<float>& value) {
    if (!value.has_value()) {
        return "n/a";
    }
    return fmt::format("{:.6f}", value.value());
}

std::string_view format_bool(bool value) { return value ? "yes" : "no"; }

void print_telemetry_info(
    const tt::tt_metal::DispatchTelemetryDeviceInfo& info,
    uint32_t device_index,
    int pci_device_id,
    uint32_t telemetry_version) {
    fmt::print(
        "dispatch_telemetry_dump device_index={} pci_device_id={} telemetry_api_version={}\n",
        device_index,
        pci_device_id,
        telemetry_version);
    fmt::print(
        "device_core_efficiency_since_last_read: {}\n",
        format_optional_float(info.device_core_efficiency_since_last_read));

    for (const auto& cq_info : info.info_cqs) {
        fmt::print("cq_id: {}\n", cq_info.cq_id);
        fmt::print("prefetch_waiting_on_upstream: {}\n", format_bool(cq_info.prefetch_waiting_on_upstream));
        fmt::print("dispatch_waiting_on_upstream: {}\n", format_bool(cq_info.dispatch_waiting_on_upstream));
        fmt::print("program_count_since_last_read: {}\n", cq_info.program_count_since_last_read);
        fmt::print("prefetch_blocked_count_since_last_read: {}\n", cq_info.prefetch_blocked_count_since_last_read);
        fmt::print("dispatch_blocked_count_since_last_read: {}\n", cq_info.dispatch_blocked_count_since_last_read);
        fmt::print("prefetch_command_count_since_last_read: {}\n", cq_info.prefetch_command_count_since_last_read);
        fmt::print("utilization_since_last_read: {}\n", format_optional_float(cq_info.utilization_since_last_read));
    }
}

}  // namespace

int main(int argc, char** argv) {
    uint32_t device_index = 0;
    bool list_devices = false;
    bool monitor = false;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if ((arg == "--device" || arg == "-d") && i + 1 < argc) {
            const long value = std::atol(argv[++i]);
            if (value < 0) {
                fmt::print(stderr, "Device index must be >= 0; got {}.\n", value);
                return 1;
            }
            device_index = static_cast<uint32_t>(value);
        } else if (arg == "--list-devices") {
            list_devices = true;
        } else if (arg == "--monitor") {
            monitor = true;
        } else if (arg == "--help" || arg == "-h") {
            fmt::print("Usage: {} [--device INDEX] [--list-devices] [--monitor]\n", argv[0]);
            return 0;
        }
    }

    const std::vector<int> pci_device_ids = tt::umd::PCIDevice::enumerate_devices();
    if (pci_device_ids.empty()) {
        fmt::print(stderr, "No PCIe devices found.\n");
        return 1;
    }

    if (list_devices) {
        fmt::print("Enumerated UMD PCIe devices:\n");
        for (size_t i = 0; i < pci_device_ids.size(); ++i) {
            fmt::print("  index {} -> PCIe device id {}\n", i, pci_device_ids[i]);
        }
        return 0;
    }

    if (device_index >= pci_device_ids.size()) {
        fmt::print(
            stderr,
            "Requested device index {} but only {} PCIe device(s) were found.\n",
            device_index,
            pci_device_ids.size());
        return 1;
    }

    const int pci_device_id = pci_device_ids.at(device_index);
    std::unique_ptr<tt::umd::TTDevice> tt_device = tt::umd::TTDevice::create(pci_device_id);
    tt_device->init_tt_device();

    tt::tt_metal::DispatchTelemetry telemetry(*tt_device);
    while (true) {
        auto info = telemetry.read_info();
        if (info.has_value()) {
            print_telemetry_info(*info, device_index, pci_device_id, telemetry.version());
        } else {
            fmt::print(stderr, "Failed to read dispatch telemetry info; see log warnings for validation details.\n");
        }

        if (!monitor) {
            break;
        }

        fmt::print("\n");
        std::this_thread::sleep_for(std::chrono::seconds(1));
    }

    return 0;
}
