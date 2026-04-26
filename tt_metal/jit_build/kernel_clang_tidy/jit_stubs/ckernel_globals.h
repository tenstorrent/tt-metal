// SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Stub for ckernel_globals.h (clang-tidy analysis only).
//
// The real file (tt_metal/third_party/tt_llk/.../common/inc/ckernel_globals.h)
// contains extern declarations for compute-kernel global variables.  Those
// variables are already *defined* in kernel_clang_tidy_prelude.h, so the
// extern declarations here are intentionally omitted to avoid duplicate
// declaration noise.
//
// This stub is found before the LLK path because jit_stubs/ is first in the
// include list, allowing dataflow TUs (which lack the LLK common inc path) to
// parse firmware_common.h without a "file not found" error.

#pragma once
