// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#ifdef ARCH_QUASAR

#include <stdint.h>

// Per-processor kernel thread info, set by Quasar dm.cc/trisc.cc from kernel_config before kernel runs.
// Used by dmk.cc, trisc.cc, and runtime (e.g. CircularBuffers) via get_num_threads() / get_my_thread_id().
extern thread_local uint32_t num_sw_threads;
extern thread_local uint32_t my_thread_id;

// clang-format off
/**
 * Returns the number of threads (processors) in the kernel that this processor belongs to.
 * Set by Quasar firmware from kernel_config before the kernel runs. Valid only on ARCH_QUASAR.
 *
 * Return value: Number of kernel threads (num_processors_per_cluster for this kernel).
 */
// clang-format on
inline uint32_t get_num_threads() { return num_sw_threads; }

// clang-format off
/**
 * Returns this processor's thread ID within its kernel (0 to get_num_threads() - 1).
 * Set by Quasar firmware from kernel_config before the kernel runs. Valid only on ARCH_QUASAR.
 *
 * Return value: Thread ID for this processor.
 */
// clang-format on
inline uint32_t get_my_thread_id() { return my_thread_id; }

#endif
