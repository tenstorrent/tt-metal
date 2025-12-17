// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <tt-metalium/profiler_types.hpp>
#include <tt-metalium/profiler_optional_metadata.hpp>
#include <tt-metalium/tt_metal_profiler.hpp>

namespace tt::tt_metal {
class IDevice;

namespace detail {

// clang-format off
/**
 * Clear profiler control buffer
 *
 * Return value: void
 *
 * | Argument | Description                                                         | Type     | Valid Range | Required |
 * |----------|---------------------------------------------------------------------|----------|-------------|----------|
 * | device   | Clear profiler control buffer before any core attempts to profiler  | IDevice* |             | True     |
 * */
// clang-format on
void ClearProfilerControlBuffer(IDevice* device);

// clang-format off
/**
 * Set the directory for device-side CSV logs produced by the profiler instance in the tt-metal module
 *
 * Return value: void
 *
 * | Argument   | Description                                              | Data type   | Valid range              | Required |
 * |------------|----------------------------------------------------------|-------------|--------------------------|----------|
 * | output_dir | The output directory that will hold the output CSV logs  | std::string | Any valid directory path | No       |
 * */
// clang-format on
void SetDeviceProfilerDir(const std::string& output_dir = "");

// clang-format off
/**
 * Start a fresh log for the device side profile results
 *
 * Return value: void
 *
 * | Argument | Description | Data type | Valid range | Required |
 * |----------|-------------|-----------|-------------|----------|
 * */
// clang-format on
void FreshProfilerDeviceLog();

}  // namespace detail
}  // namespace tt::tt_metal
