// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Shared precompiled prelude for every JIT-compiled firmware and kernel target.
//
// Two rules govern what may be listed here.
//
// Nothing may reference a project macro or a generated per-kernel header. That is what lets a
// single build of this file serve every target: GCC validates a precompiled header against the
// macros it recorded while building, so because this file records none, a consuming compile's
// own -D set (KERNEL_COMPILE_TIME_ARGS, UCK_CHLKC_*, COMPILE_FOR_*, ...) cannot invalidate it.
// One PCH per compiler flag set, not per kernel. It also means no per-kernel generated header
// (chlkc_descriptors.h and friends) can be baked in or go stale. Today the list is entirely
// standard library, but a toolchain header that references no project macro would also qualify.
//
// The list stays the union of what the core types actually parse, and nothing beyond it. A PCH
// is loaded in full by every compile that uses it, so its size is a fixed per-compile cost, and
// headers nothing parses are pure overhead: measured on a Blackhole workload, padding this list
// with standard headers no core reaches doubled the artifact and gave back most of the win.
// Conversely, a header missing from here is parsed from source by every target that needs it,
// so add one when it becomes commonly used.

#pragma once

#include <algorithm>
#include <array>
#include <atomic>
#include <bit>
#include <climits>
#include <compare>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <new>
#include <optional>
#include <tuple>
#include <type_traits>
#include <utility>
#include <variant>
