// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

namespace tt::tt_metal {

// Get global device profiling state based on build flag and environment variables
bool getDeviceProfilerState();

// Get if the device debug dump is enabled
bool getDeviceDebugDumpEnabled();

}  // namespace tt::tt_metal
