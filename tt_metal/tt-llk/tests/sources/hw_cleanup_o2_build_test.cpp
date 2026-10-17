// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// hw_cleanup_test.cpp built at -O2 instead of the harness default -O3.
//
// tt-metal builds compute kernels at -O3 by default but lets a kernel choose O0/O1/O2/Os/Oz
// (KernelBuildOptLevel). The hw_cleanup headers must compile at every one of them: a Tensix
// instruction with an immediate operand (TTI_*) only compiles when its operands fold to
// constants, and -O3's full loop unrolling must not be what makes that happen.
//
// Build-only: test_hw_cleanup.py builds this driver and never runs it.

#pragma GCC optimize("O2")

#include "hw_cleanup_test.cpp"
