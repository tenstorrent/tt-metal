// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
#pragma once
//
// The extern-C symbol surface JIT kernels resolve by dlsym at dlopen (-rdynamic):
// the memory/NOC-address bridge and the fiber-scheduler thunks. Bodies in
// emule_noc_bridge.cpp; they keep C linkage + global scope so the mangled name
// never changes. Depends on emule_device_map (bank arrays, core-map, chip
// resolution) and emule_fiber_scheduler.

#include <cstdint>

#include <tt_stl/assert.hpp>

#include "emule_device_map.hpp"                // NUM_NOCS, NOC_* constants, bank arrays, get_sw_emulated_chip, core-map
#include "jit_hw/internal/emule_thread_ctx.h"  // __emule_self

// Contract guard for every C-linkage bridge/fabric hook: they run inside a kernel
// fiber, so __emule_self is set; a null means the hook ran outside a fiber. Kept at
// global scope (matches the original file-scope helper) and used across the whole
// emule surface, not just this file.
inline void emule_require_self(const char* fn) {
    TT_FATAL(__emule_self != nullptr, "{}: emule bridge call outside a kernel fiber context", fn);
}

// Per-core NOC coordinates — set per kernel fiber on swap-in (by the launch/engine
// code), read by __emule_multicast_write. Global/unmangled thread_local; defined in
// emule_noc_bridge.cpp.
extern thread_local uint8_t my_x[NUM_NOCS];
extern thread_local uint8_t my_y[NUM_NOCS];

// Declared for the runner's internal callers (e.g. the fabric deliver path calls
// __emule_fiber_wake). Kernels resolve all of these by dlsym, not via this header.
extern "C" void __emule_fiber_wake(const void* key);
