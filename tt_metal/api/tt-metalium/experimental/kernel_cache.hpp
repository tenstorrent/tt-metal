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
 * binaries, run `tt-metal-cache clear`, which evicts every entry in the cache root that no
 * live process is using. The cache root is <tt-metal-cache>/<build_key>/kernels/, but treat
 * that layout as an implementation detail and go through the CLI.
 *
 * The disk cache also bounds itself: entries are evicted least recently used first once it
 * exceeds TT_METAL_CACHE_MAX_SIZE.
 *
 * Return value: void
 */
void ClearKernelCache();

}  // namespace tt::tt_metal::experimental
