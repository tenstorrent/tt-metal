// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <memory>
#include <optional>
#include <cstdint>
#include <tt-metalium/device.hpp>

namespace tt::tt_metal {

struct DispatchTelemetryInfo {
    bool prefetch_waiting = false;
    uint64_t prefetch_blocked_count_since_last_read = 0;
    uint64_t prefetch_command_count_since_last_read = 0;
    bool dispatch_waiting = false;
    uint64_t dispatch_blocked_count_since_last_read = 0;
    uint64_t dispatch_program_count_since_last_read = 0;
};

class DispatchTelemetry {
public:
    DispatchTelemetry(const IDevice& device);
    ~DispatchTelemetry();

    /**
     * @brief Get the version of the dispatch telemetry API. This may mismatch with the version
     *        present on the device. If so, read_info will return std::nullopt and an error will be
     *        logged.
     *
     * @return The version of the dispatch telemetry API.
     */
    uint32_t version() const;

    /**
     * @brief Read the dispatch telemetry info from the device.
     *
     * @return The dispatch telemetry info on success, or std::nullopt if the telemetry buffer fails
     *         signature/version validation (a warning is logged in that case).
     */
    std::optional<DispatchTelemetryInfo> read_info();

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace tt::tt_metal
