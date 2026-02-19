// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#ifdef ARCH_QUASAR
// Per-processor kernel thread info, set by Quasar dm.cc from kernel_config before kernel runs.
// Used by dmk.cc and runtime (e.g. CircularBuffers) via get_num_kernel_threads() / get_my_thread_id().
extern thread_local uint32_t num_kernel_threads;
extern thread_local uint32_t my_thread_id;
#endif
