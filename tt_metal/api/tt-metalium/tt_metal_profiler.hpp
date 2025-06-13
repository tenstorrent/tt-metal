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

/**
 * Traverse all cores and read device side profiler data and dump results into device side CSV log
 *
 * Return value: void
 *
 * | Argument      | Description                                       | Type | Valid Range               | Required |
 * |---------------|---------------------------------------------------|--------------------------------------------------------------|---------------------------|----------|
 * | device        | The device holding the program being profiled.    | Device * |                           | True |
 * | satate        | Dumpprofiler various states                       | ProfilerDumpState |                  | False |
 * */
void DumpDeviceProfileResults(
    IDevice* device,
    ProfilerDumpState = ProfilerDumpState::NORMAL,
    const std::optional<ProfilerOptionalMetadata>& metadata = {});
/**
 * Traverse all cores and read device side profiler data and dump results into device side CSV log
 *
 * Return value: void
 *
 * | Argument      | Description                                       | Type | Valid Range               | Required |
 * |---------------|---------------------------------------------------|--------------------------------------------------------------|---------------------------|----------|
 * | device        | The device holding the program being profiled.    | Device * |                           | True |
 * | satate        | Dumpprofiler various states                       | ProfilerDumpState |                  | False |
 * */
void ShareTraceIDwithProfiler(chip_id_t device_id, uint32_t trace_id);

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
