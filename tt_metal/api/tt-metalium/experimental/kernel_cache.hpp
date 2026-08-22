// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

namespace tt::tt_metal::experimental {

/**
 * Clear the in-memory kernel compilation hash lookup cache.
 *
 * This forces kernels to be recompiled on next use, even if they were previously
 * compiled in this process.
 *
 * Note: This only clears the in-memory HashLookup cache. To also clear disk-cached kernel
 * binaries, delete the per-build-key directories under the cache root, which is
 * $TT_METAL_CACHE/tt-metal-cache/ when that is set and ~/.cache/tt-metal-cache/ otherwise.
 *
 * The disk cache can also bound itself: setting TT_METAL_CACHE_MAX_SIZE evicts whole build-key
 * directories, least recently used first, once the root exceeds it.
 *
 * Return value: void
 */
void ClearKernelCache();

}  // namespace tt::tt_metal::experimental
