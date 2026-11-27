// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>

#include <tt-metalium/core_coord.hpp>

#include <hostdevcommon/dispatch_telemetry_types.hpp>

namespace tt::umd {
class TTDevice;
}

namespace tt::tt_metal {

/**
 * @brief Read the SMC dispatch telemetry control block.
 *
 * @param tt_device Non-owning UMD device context.
 * @return Control block on success, or std::nullopt if discovery or validation fails.
 */
std::optional<dispatch_telemetry_types::SMCDispatchTelemetryControl> read_smc_dispatch_telemetry_control(
    tt::umd::TTDevice& tt_device);

/**
 * @brief Write the SMC dispatch telemetry control block.
 *
 * @param tt_device Non-owning UMD device context.
 * @param control Control block to write.
 * @return True if the write completed, false if the SMC buffer is unavailable.
 */
bool write_smc_dispatch_telemetry_control(
    tt::umd::TTDevice& tt_device, const dispatch_telemetry_types::SMCDispatchTelemetryControl& control);

/**
 * @brief Invalidate the SMC dispatch telemetry control block signature.
 *
 * @param tt_device Non-owning UMD device context.
 * @return True if the invalidation completed, false if the SMC buffer is unavailable.
 */
bool invalidate_smc_dispatch_telemetry_control(tt::umd::TTDevice& tt_device);

/**
 * @brief Read the DispatchCoreTelemetry block from a dispatch core's L1.
 *
 * @param tt_device Non-owning UMD device context.
 * @param virtual_core Virtual coord of the dispatch core to sample.
 * @return Telemetry data on success, or std::nullopt if the buffer fails signature/version
 *         validation (a warning is logged in that case).
 */
std::optional<dispatch_telemetry_types::DispatchCoreTelemetry> read_dispatch_core_telemetry(
    tt::umd::TTDevice& tt_device, CoreCoord virtual_core);

/**
 * @brief Read the PrefetchCoreTelemetry block from a prefetch core's L1.
 *
 * @param tt_device Non-owning UMD device context.
 * @param virtual_core Virtual coord of the prefetch core to sample.
 * @return Telemetry data on success, or std::nullopt if the buffer fails signature/version
 *         validation (a warning is logged in that case).
 */
std::optional<dispatch_telemetry_types::PrefetchCoreTelemetry> read_prefetch_core_telemetry(
    tt::umd::TTDevice& tt_device, CoreCoord virtual_core);

}  // namespace tt::tt_metal
