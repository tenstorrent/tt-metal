// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "context/context_types.hpp"

namespace tt::tt_metal {

class RealtimeProfilerService;

// Registers the profiler's built-in record consumers (currently the Tracy device-timeline sink) with the given
// service. Single point of ownership for the built-in sinks: the context calls this once during bring-up rather than
// wiring individual consumers inline. Built-in consumers live for the context's lifetime and are never unregistered.
void register_builtin_realtime_profiler_consumers(RealtimeProfilerService& service, ContextId context_id);

}  // namespace tt::tt_metal
