// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

namespace tt::tt_metal {

class MetalEnvImpl;

// Get whether device profiling is active for the given env. It is never active on a mock or emulated cluster.
bool getDeviceProfilerState(MetalEnvImpl& env);

// Get if the device debug dump is enabled for the given env.
bool getDeviceDebugDumpEnabled(MetalEnvImpl& env);

}  // namespace tt::tt_metal
