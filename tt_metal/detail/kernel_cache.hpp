// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "jit_build/build.hpp"

namespace tt::tt_metal::detail {

/**
 * Clear the current kernel compilation cache.
 *
 * Return value: void
 */
inline void ClearKernelCache() { jit_build_cache_clear(); }

}  // namespace tt::tt_metal::detail
