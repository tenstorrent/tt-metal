// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>
#include <tt-metalium/profiler_types.hpp>
#include <tt-metalium/profiler_optional_metadata.hpp>

namespace tt::tt_metal {
class IDevice;

namespace detail {

// clang-format off
/**
 * Initialize device profiling data buffers
 *
 * Return value: void
 *
 * | Argument | Description                                    | Type     | Valid Range | Required |
 * |----------|------------------------------------------------|----------|-------------|----------|
 * | device   | The device holding the program being profiled. | IDevice* |             | True     |
 * */
// clang-format on
void InitDeviceProfiler(IDevice* device);

// clang-format off
/**
 * Read device side profiler data for the device
 *
 * This function only works in PROFILER builds. Please refer to the "Device Program Profiler" section for more information.
 *
 * Return value: void
 *
 * | Argument      | Description                                           | Type                     | Valid Range               | Required |
 * |---------------|-------------------------------------------------------|--------------------------|---------------------------|----------|
 * | device        | The device to be profiled                             | IDevice*                 |                           | Yes      |
 * | state         | The state to use for this profiler read               | ProfilerReadState        |                           | No       |
 * | metadata      | Metadata to include in the profiler results           | ProfilerOptionalMetadata |                           | No       |
 * */
// clang-format on
void ReadDeviceProfilerResults(
    IDevice* device,
    ProfilerReadState = ProfilerReadState::NORMAL,
    const std::optional<ProfilerOptionalMetadata>& metadata = {});

// clang-format off
/**
 * Sync TT devices with host
 *
 * Return value: void
 *
 * | Argument | Description                     | Type               | Valid Range | Required |
 * |----------|---------------------------------|--------------------|-------------|----------|
 * | state    | The state to use for sync       | ProfilerSyncState  |             | Yes      |
 * */
// clang-format on
void ProfilerSync(ProfilerSyncState state);

}  // namespace detail
}  // namespace tt::tt_metal
