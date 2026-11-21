// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <optional>
#include <string>

#include <tt-metalium/profiler_optional_metadata.hpp>

#include "impl/profiler/profiler_types.hpp"

namespace tt::tt_metal {
class IDevice;
class MetalContext;

namespace distributed {
class MeshDevice;
}

namespace detail {

void ClearProfilerControlBuffer(IDevice* device);

// Sync the devices of the given context with the host.
void ProfilerSync(MetalContext& ctx, ProfilerSyncState state);

// Initialize the profiler of the given device, which belongs to the given context.
void InitDeviceProfiler(MetalContext& ctx, IDevice* device);

// Read device side profiler data for the given device. Only works in PROFILER builds.
void ReadDeviceProfilerResults(
    IDevice* device,
    ProfilerReadState state = ProfilerReadState::NORMAL,
    const std::optional<ProfilerOptionalMetadata>& metadata = {});

// Set the output directory of the device profilers of the devices in the given mesh device.
void SetDeviceProfilerDir(distributed::MeshDevice& mesh_device, const std::string& output_dir = "");

// Delete the device log and perf report in the output directory of each device profiler in the given mesh device. All
// device profilers share the process-wide profiler log directory unless SetDeviceProfilerDir gave them their own, so by
// default this also deletes output written by other meshes.
void FreshProfilerDeviceLog(distributed::MeshDevice& mesh_device);

DeviceProgramId DecodePerDeviceProgramID(uint32_t device_program_id);

}  // namespace detail
}  // namespace tt::tt_metal
