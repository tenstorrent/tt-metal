// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Common standard-library headers, independent of project macros and generated kernel headers.
// Keep this list small: every consuming compile loads the entire PCH.

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
