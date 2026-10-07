// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <string>

#include <tt-metalium/tt_metal_profiler.hpp>

#include "profiler_types.hpp"

namespace tt::tt_metal {
class IDevice;
class MetalContext;

namespace distributed {
class MeshDevice;
}

namespace detail {

void ClearProfilerControlBuffer(IDevice* device);

// Sync the devices of the given context with the host. The public ProfilerSync(state) overload applies to the default
// context.
void ProfilerSync(MetalContext& ctx, ProfilerSyncState state);

// Apply to the device profilers of the devices in the given mesh device.
void SetDeviceProfilerDir(distributed::MeshDevice& mesh_device, const std::string& output_dir = "");

void FreshProfilerDeviceLog(distributed::MeshDevice& mesh_device);

DeviceProgramId DecodePerDeviceProgramID(uint32_t device_program_id);

}  // namespace detail
}  // namespace tt::tt_metal
