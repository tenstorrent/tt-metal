// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Umbrella header for the optional TRISC precompiled header (TT_METAL_JIT_PCH=1).
// These are the includes trisck.cc parses identically for every TRISC target;
// precompiling them removes roughly half of each target's compile time.
//
// Two headers trisck.cc includes are deliberately left out:
//  - chlkc_list.h pulls in the per-kernel generated kernel body, which is the
//    part that legitimately differs between targets.
//  - sanitizer/api.h reaches tt-llk's sanitizer/output.h, which expands
//    CTSTR(FULL_KERNEL_NAME). Baking that in would either tie the PCH to a
//    single kernel or silently freeze the "<unknown>" fallback into it.
//
// Ordering caveat: in trisck.cc, kernel_profiler.hpp and stack_usage.h parse
// after chlkc_list.h, whose defines_generated.h carries the kernel's defines --
// a kernel's FORCE_WATCHER_OFF is visible to them there, but not here. All of
// its consumers are inert unless WATCHER_ENABLED is set, so compile_one skips
// the PCH for TRISC targets whenever the watcher is on rather than precompile
// the wrong state.
//
// Keep this in sync with the prelude in trisck.cc.

#pragma once

#include "internal/firmware_common.h"
#include "tools/profiler/kernel_profiler.hpp"
#include "internal/debug/stack_usage.h"
