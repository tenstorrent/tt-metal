// SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Wrapper for internal/firmware_common.h (clang-tidy analysis only).
//
// firmware_common.h conditionally includes api/dataflow/dataflow_api.h
// when KERNEL_BUILD && !COMPILE_FOR_TRISC.  That header defines get_arg_addr()
// returning uintptr_t, which conflicts with the uint32_t return type in
// api/compute/common.h (included by compute kernels).
//
// By temporarily defining COMPILE_FOR_TRISC we suppress the dataflow_api.h
// pull-in.  Dataflow kernels that need the dataflow API include it directly
// in their own source files, so nothing is lost for them.
//
// Headers that are also gated by !COMPILE_FOR_TRISC in transitively-included
// files (e.g. noc_nonblocking_api.h via risc_common.h) are already in the
// #pragma-once cache from the prelude's earlier includes, so the temporary
// define has no effect on them.

#pragma once

#ifndef COMPILE_FOR_TRISC
#define COMPILE_FOR_TRISC
#include_next "internal/firmware_common.h"
#undef COMPILE_FOR_TRISC
#else
// COMPILE_FOR_TRISC already set by the caller — forward unconditionally.
#include_next "internal/firmware_common.h"
#endif
