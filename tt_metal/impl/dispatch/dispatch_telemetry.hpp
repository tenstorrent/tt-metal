// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>

#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/device_types.hpp>

#include <hostdevcommon/dispatch_telemetry_types.hpp>

namespace tt::tt_metal {

/**
 * @brief Read the DispatchCoreTelemetry block from a dispatch core's L1.
 *
 * @param chip         Chip that owns the dispatch core.
 * @param virtual_core Virtual, NOC-addressable coord of the dispatch core to sample.
 * @return Telemetry data on success, or std::nullopt if the buffer fails signature/version
 *         validation (a warning is logged in that case).
 */
std::optional<DispatchCoreTelemetry> read_dispatch_core_telemetry(ChipId chip, const CoreCoord& virtual_core);

/**
 * @brief Read the PrefetchCoreTelemetry block from a prefetch core's L1.
 *
 * @param chip         Chip that owns the prefetch core.
 * @param virtual_core Virtual, NOC-addressable coord of the prefetch core to sample.
 * @return Telemetry data on success, or std::nullopt if the buffer fails signature/version
 *         validation (a warning is logged in that case).
 */
std::optional<PrefetchCoreTelemetry> read_prefetch_core_telemetry(ChipId chip, const CoreCoord& virtual_core);

}  // namespace tt::tt_metal
