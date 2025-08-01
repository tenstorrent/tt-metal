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

/**
 * Clear profiler control buffer
 *
 * Return value: void
 *
 * | Argument      | Description                                                        | Type            | Valid Range
 * | Required |
 * |---------------|--------------------------------------------------------------------|-----------------|---------------------------|----------|
 * | device        | Clear profiler control buffer before any core attempts to profler  | IDevice*        | | True     |
 * */
void ClearProfilerControlBuffer(IDevice* device);

/**
 * Initialize device profiling data buffers
 *
 * Return value: void
 *
 * | Argument      | Description                                       | Type            | Valid Range               |
 * Required |
 * |---------------|---------------------------------------------------|-----------------|---------------------------|----------|
 * | device        | The device holding the program being profiled.    | IDevice*        |                           |
 * True     |
 * */
void InitDeviceProfiler(IDevice* device);

/**
 * Sync TT devices with host
 *
 * Return value: void
 *
 * | Argument      | Description                                       | Type            | Valid Range               |
 * Required |
 * |---------------|---------------------------------------------------|-----------------|---------------------------|----------|
 * */
void ProfilerSync(ProfilerSyncState state);

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

/**
 * Set the directory for device-side CSV logs produced by the profiler instance in the tt-metal module
 *
 * Return value: void
 *
 * | Argument     | Description                                             |  Data type  | Valid range              |
 * required |
 * |--------------|---------------------------------------------------------|-------------|--------------------------|----------|
 * | output_dir   | The output directory that will hold the output CSV logs  | std::string | Any valid directory path |
 * No       |
 * */
void SetDeviceProfilerDir(const std::string& output_dir = "");

/**
 * Start a fresh log for the device side profile results
 *
 * Return value: void
 *
 * | Argument     | Description                                             |  Data type  | Valid range              |
 * required |
 * |--------------|---------------------------------------------------------|-------------|--------------------------|----------|
 * */
void FreshProfilerDeviceLog();

}  // namespace detail
}  // namespace tt::tt_metal
