// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "impl/context/context_types.hpp"

namespace tt::tt_metal {

// Get global device profiling state based on build flag and environment variables
bool getDeviceProfilerState(ContextId context_id = DEFAULT_CONTEXT_ID);

// Get if the device debug dump is enabled
bool getDeviceDebugDumpEnabled(ContextId context_id = DEFAULT_CONTEXT_ID);

}  // namespace tt::tt_metal
