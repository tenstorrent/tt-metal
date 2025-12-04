// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

namespace tt::tt_metal::detail {

/**
 * Enable kernel compilation cache to be persistent across runs. When this is called, kernels will not be compiled if
 * the output binary path exists.
 *
 * Return value: void
 */
[[deprecated("Persistent kernel cache is no longer needed as the JIT build system now supports caching by default.  This API will be removed after December 11, 2025.")]]
void EnablePersistentKernelCache();

/**
 * Disables kernel compilation cache from being persistent across runs.
 *
 * Return value: void
 */
[[deprecated("Persistent kernel cache is no longer needed as the JIT build system now supports caching by default.  This API will be removed after December 11, 2025.")]]
void DisablePersistentKernelCache();

/**
 * Clear the in-memory kernel compilation hash lookup cache.
 *
 * This forces kernels to be recompiled on next use, even if they were previously
 * compiled in this process.
 *
 * Note: This only clears the in-memory HashLookup cache. To also clear disk-cached
 * kernel binaries, you must delete the files in:
 *   ~/.cache/tt-metal-cache/<git_hash>/<build_id>/kernels/
 *
 * Return value: void
 */
void ClearKernelCache();

}  // namespace tt::tt_metal::detail
