// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>

#include "core_coord.hpp"
#include "dispatch_telemetry_types.hpp"
namespace tt::tt_metal {
class IDevice;

/**
 * @brief Read the DispatchTelemetry block from a dispatch core's L1.
 *
 * @param device                Device that owns the dispatch core.
 * @param dispatch_logical_core Logical coord of the dispatch core to sample.
 * @param core_type             CoreType of the dispatch core (WORKER for tensix-based dispatch,
 *                              ETH for ethernet-based dispatch).
 * @return Telemetry data on success, or std::nullopt if the buffer fails signature/version
 *         validation (a warning is logged in that case).
 */
std::optional<DispatchTelemetry> read_dispatch_telemetry(
    IDevice* device, const CoreCoord& dispatch_logical_core, CoreType core_type = CoreType::WORKER);

/**
 * @brief Read the PrefetchTelemetry block from a prefetch core's L1.
 *
 * @param device                Device that owns the prefetch core.
 * @param prefetch_logical_core Logical coord of the prefetch core to sample.
 * @param core_type             CoreType of the prefetch core (WORKER for tensix-based prefetch,
 *                              ETH for ethernet-based prefetch).
 * @return Telemetry data on success, or std::nullopt if the buffer fails signature/version
 *         validation (a warning is logged in that case).
 */
std::optional<PrefetchTelemetry> read_prefetch_telemetry(
    IDevice* device, const CoreCoord& prefetch_logical_core, CoreType core_type = CoreType::WORKER);

}  // namespace tt::tt_metal
