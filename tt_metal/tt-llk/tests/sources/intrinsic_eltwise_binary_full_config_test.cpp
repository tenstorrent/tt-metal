
// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

/* Full 3-thread compiler-config oracle: the same kernel as
 * intrinsic_eltwise_binary_test.cpp.  Since the compiler-emitted
 * config-declaration intrinsics were deleted, every thread's one-time setup is
 * now the LLK's _llk_*_hw_configure_ family, which issues the config through
 * the real config-write intrinsics (stallwait, rmwciB0..3, setc16, setdmareg, wrcfg)
 * that pass_rvtt_config consumes, coalesces, and re-emits -- the compiler's
 * config pass still owns the per-compute reconfig derived from the
 * elwmul/unpacr/pacr data-ops.  This TU is kept so the harness exercises the
 * same source through its own build; it no longer selects a distinct
 * compiler-config mode. */

#include "intrinsic_eltwise_binary_test.cpp"
