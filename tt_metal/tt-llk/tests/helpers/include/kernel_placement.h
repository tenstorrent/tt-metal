// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

// First in every TRISC unit with a build.h. TRISC timing depends on the code address, so run_kernel gets a 2 KiB
// aligned section that sections.ld places after the harness code, and harness changes cannot move it.
#if !defined(ARCH_QUASAR)
// The type RUNTIME_PARAMETERS expands to; build.h is not included because some kernels define RuntimeParams.
struct RuntimeParams;

__attribute__((noinline, section(".text.run_kernel"), aligned(2048))) void run_kernel(const struct RuntimeParams& params);
#endif
