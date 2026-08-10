
// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

/* Pack-thread compiler-config oracle: the same kernel as
 * intrinsic_eltwise_binary_test.cpp.  Since the compiler-emitted
 * config-declaration intrinsics were deleted, the one-time config now comes
 * from the LLK's _llk_pack_hw_configure_wrapper_ / _llk_math_hw_configure_,
 * which issue the config through the real config-write intrinsics
 * (rmwciB*/setdmareg/wrcfg) that pass_rvtt_config consumes and coalesces.  This
 * TU is kept so the harness exercises the same source through its own build; it
 * no longer selects a distinct compiler-config mode. */

#include "intrinsic_eltwise_binary_test.cpp"
