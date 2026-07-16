// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <memory>
#include <optional>
#include <vector>

namespace tt::umd {
class TTDevice;
}  // namespace tt::umd

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
     * @brief   Utilization describes the percentage of time the cq is actively executing any work.
     *          Requires worker dispatch.
     * @note    utilization = work_runtime / uptime
     * @return  Normalized utilization since the last read, or std::nullopt if worker dispatch is
     *          disabled. 1.0 = 100%, 0.0 = 0%.
     */
    std::optional<float> utilization_since_last_read = std::nullopt;
};

struct DispatchTelemetryDeviceInfo {
    /**
     * @brief   Device core efficiency describes the percentage of time the cores are actively
     *          executing work. Requires worker dispatch.
     * @note    core_efficiency = avg_core_runtime / uptime
     * @return  Normalized core efficiency since the last read, or std::nullopt if worker dispatch is
     *          disabled. 1.0 = 100%, 0.0 = 0%.
     */
    std::optional<float> device_core_efficiency_since_last_read = std::nullopt;

    std::vector<DispatchTelemetryCqInfo> info_cqs;
};

class DispatchTelemetry {
public:
    DispatchTelemetry(tt::umd::TTDevice& device);
    ~DispatchTelemetry();

    /**
     * @brief   Get the version of the dispatch telemetry API. This may mismatch with the version
     *          present on the device. If so, read_info will return an empty optional and an error
     *          will be logged.
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
