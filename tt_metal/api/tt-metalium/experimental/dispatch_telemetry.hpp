// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <memory>
#include <vector>

#include <tt-metalium/device.hpp>

namespace tt::tt_metal {

struct DispatchTelemetryInfo {
    uint8_t cq_id = 0;
    bool prefetch_waiting_on_upstream = false;
    uint32_t prefetch_blocked_count_since_last_read = 0;
    uint32_t prefetch_command_count_since_last_read = 0;
    bool dispatch_waiting_on_upstream = false;
    uint32_t dispatch_blocked_count_since_last_read = 0;
    uint32_t dispatch_program_count_since_last_read = 0;
};

class DispatchTelemetry {
public:
    DispatchTelemetry(const IDevice& device);
    ~DispatchTelemetry();

    /**
     * @brief Get the version of the dispatch telemetry API. This may mismatch with the version
     *        present on the device. If so, read_info will return an empty vector and an error will be
     *        logged.
     *
     * @return The version of the dispatch telemetry API.
     */
    uint32_t version() const;

    /**
     * @brief Read the dispatch telemetry info from the device.
     *
     * @return Dispatch telemetry info for each command queue. If there is an issue reading telemetry
     *         from a command queue, a warning is logged and that entry will be absent.
     */
    std::vector<DispatchTelemetryInfo> read_info();

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace tt::tt_metal
