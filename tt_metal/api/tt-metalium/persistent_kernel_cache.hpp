// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <atomic>

namespace tt::tt_metal::detail {

/**
 * Enable kernel compilation cache to be persistent across runs. When this is called, kernels will not be compiled if
 * the output binary path exists.
 *
 * @note The persistent kernel cache is keyed by the Metal build ID, the content of the kernel and build options
 * (including the core type and command queue). Included files changing _will not_ invalidate the cache.
 *
 * Return value: void
 */
void EnablePersistentKernelCache();

/**
 * Disables kernel compilation cache from being persistent across runs.
 *
 * Return value: void
 */
void DisablePersistentKernelCache();

}  // namespace tt::tt_metal::detail
