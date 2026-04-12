// SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Wrapper for api/dataflow/dataflow_api.h (clang-tidy analysis only).
//
// In compute kernel translation units (TRISC_MATH / TRISC_UNPACK / TRISC_PACK)
// both api/compute/common.h and api/dataflow/dataflow_api.h define the same
// functions with the same signature (on riscv32 uintptr_t == uint32_t):
//
//   get_arg_addr, get_common_arg_addr, get_arg_val, get_common_arg_val
//   get_absolute_logical_x, get_absolute_logical_y
//   get_relative_logical_x, get_relative_logical_y
//
// C++ does not allow two definitions of the same function in one translation
// unit even if they are identical.  This conflict occurs because compute
// kernels that use experimental/noc.h pull in dataflow_api.h transitively:
//   experimental/circular_buffer.h → experimental/noc.h → dataflow_api.h
//
// Strategy: temporarily rename the conflicting symbols via preprocessor macros
// before including the real dataflow_api.h, then undefine the macros.  The
// renamed versions (kct_df_*) are harmless dead code in compute TUs; the
// original names resolve to the api/compute/common.h versions as expected.
// All other content of dataflow_api.h (barrier helpers, NOC wrappers,
// NocEventType, cb_push_back, etc.) is preserved unmodified.
//
// Dataflow kernel TUs (no TRISC_* define) get the real header unchanged.

#pragma once

#if defined(TRISC_MATH) || defined(TRISC_UNPACK) || defined(TRISC_PACK)

// ── Compute kernel path: rename the 8 conflicting functions ─────────────────
// The macros perform text substitution so the function DEFINITIONS in
// dataflow_api.h use kct_df_* names.  Cross-references inside that file
// (e.g., get_arg_val calling get_arg_addr) are consistently renamed too.

// clang-format off
#define get_arg_addr           kct_df_get_arg_addr_
#define get_common_arg_addr    kct_df_get_common_arg_addr_
#define get_arg_val            kct_df_get_arg_val_
#define get_common_arg_val     kct_df_get_common_arg_val_
#define get_absolute_logical_x kct_df_get_absolute_logical_x_
#define get_absolute_logical_y kct_df_get_absolute_logical_y_
#define get_relative_logical_x kct_df_get_relative_logical_x_
#define get_relative_logical_y kct_df_get_relative_logical_y_
// clang-format on

#include_next "api/dataflow/dataflow_api.h"

// Restore the original names so that compute kernel code continues to call
// the api/compute/common.h versions (which are already defined at this point
// via the prelude → compute_kernel_api → api/compute/common.h chain).
#undef get_arg_addr
#undef get_common_arg_addr
#undef get_arg_val
#undef get_common_arg_val
#undef get_absolute_logical_x
#undef get_absolute_logical_y
#undef get_relative_logical_x
#undef get_relative_logical_y

#else

// ── Dataflow kernel path — forward to the real header unchanged ──────────────
#include_next "api/dataflow/dataflow_api.h"

#endif
