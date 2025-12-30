// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
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
 * Note: This only clears the in-memory HashLookup cache. To also clear disk-cached
 * kernel binaries, you must delete the files in:
 *   ~/.cache/tt-metal-cache/<git_hash>/<build_id>/kernels/
 *
 * Return value: void
 */
void ClearKernelCache();

}  // namespace tt::tt_metal::experimental
