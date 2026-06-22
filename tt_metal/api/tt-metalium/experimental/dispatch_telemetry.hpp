// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <memory>
#include <optional>
#include <vector>

#include <tt-metalium/device.hpp>

namespace tt::tt_metal {

struct DispatchTelemetryCqInfo {
    uint8_t cq_id = 0;
    bool prefetch_waiting_on_upstream = false;
    bool dispatch_waiting_on_upstream = false;
    uint32_t program_count_since_last_read = 0;
    uint32_t prefetch_blocked_count_since_last_read = 0;
    uint32_t dispatch_blocked_count_since_last_read = 0;
    uint32_t prefetch_command_count_since_last_read = 0;

    /**
     * @brief   Sub device utilization describes the percentage of time the sub device is actively executing any work.
     * @note    sub_device_utilization = sub_device_runtime / uptime
     * @note    Normalized utilization ratio: 1.0 = 100%, 0.0 = 0%
     * @note    Requires worker dispatch
     */
    std::vector<float> sub_device_utilization_since_last_read;
};

struct DispatchTelemetryDeviceInfo {
    /**
     * @brief   Device core efficiency describes the percentage of time the cores are actively executing work.
     * @note    core_efficiency = avg_core_runtime / uptime
     * @note    Normalized utilization ratio: 1.0 = 100%, 0.0 = 0%
     * @note    Requires worker dispatch
     */
    std::optional<float> device_core_efficiency_since_last_read = std::nullopt;

    std::vector<DispatchTelemetryCqInfo> info_cqs;
};

class DispatchTelemetry {
public:
    DispatchTelemetry(const IDevice& device);
    ~DispatchTelemetry();

    /**
     * @brief   Get the version of the dispatch telemetry API. This may mismatch with the version
     *          present on the device. If so, read_info will return an empty vector and an error will be
     *          logged.
     *
     * @return  The version of the dispatch telemetry API.
     */
    uint32_t version() const;

    /**
     * @brief   Read device-wide dispatch telemetry derived from all command queues.
     *
     * @return  Device-wide telemetry info. If there is an issue reading telemetry from the device,
     *          a warning is logged and an empty optional is returned.
     */
    std::optional<DispatchTelemetryDeviceInfo> read_info();

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace tt::tt_metal
