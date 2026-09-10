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

namespace detail {

void ClearProfilerControlBuffer(IDevice* device);

void SetDeviceProfilerDir(const std::string& output_dir = "");

void FreshProfilerDeviceLog();

DeviceProgramId DecodePerDeviceProgramID(uint32_t device_program_id);

}  // namespace detail
}  // namespace tt::tt_metal
