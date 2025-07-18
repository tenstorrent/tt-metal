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
 * Return value: void
 */
void EnablePersistentKernelCache();

/**
 * Disables kernel compilation cache from being persistent across runs.
 *
 * Return value: void
 */
void DisablePersistentKernelCache();

/**
 * Returns true if the kernel compilation cache is enabled.
 *
 * Return value: bool
 */
bool GetPersistentKernelCacheEnabled();

}  // namespace tt::tt_metal::detail
