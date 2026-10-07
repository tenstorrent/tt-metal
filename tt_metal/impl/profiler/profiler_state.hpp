// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "impl/context/context_types.hpp"

namespace tt::tt_metal {

class MetalEnvImpl;

// Get whether device profiling is active for the given env. It is never active on a mock or emulated cluster.
bool getDeviceProfilerState(MetalEnvImpl& env);

// Get if the device debug dump is enabled for the given env.
bool getDeviceDebugDumpEnabled(MetalEnvImpl& env);

// TODO: Transitional overloads that look the env up from a context id. Remove once all callers pass the env.
bool getDeviceProfilerState(ContextId context_id = DEFAULT_CONTEXT_ID);
bool getDeviceDebugDumpEnabled(ContextId context_id = DEFAULT_CONTEXT_ID);

}  // namespace tt::tt_metal
