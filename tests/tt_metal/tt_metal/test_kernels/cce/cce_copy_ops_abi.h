// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <stdint.h>

// cce_copy_ops kernel ABI. The kernel (cce_copy_ops.cpp) and the host (dram/mimir_access.hpp) both include this.
namespace cce_copy_ops_abi {
inline constexpr uint32_t kArgDoneOffset = 0, kArgNumOps = 1, kHeaderWords = 2;
inline constexpr uint32_t kOpSrcKind = 0, kOpSrcSelector = 1, kOpSrcOffsetLo = 2, kOpSrcOffsetHi = 3, kOpDstKind = 4,
                          kOpDstSelector = 5, kOpDstOffsetLo = 6, kOpDstOffsetHi = 7, kOpNumWords = 8, kOpWords = 9;
constexpr uint32_t op_arg_index(uint32_t op_index, uint32_t field) {
    return kHeaderWords + kOpWords * op_index + field;
}
inline constexpr uint32_t kKindOwnTl1 = 0, kKindGddr = 1, kKindSramSpa = 2;
// Gddr selector: (local Mimir << 16) | target Mimir.
inline constexpr uint32_t kGddrLocalShift = 16, kGddrTargetMask = 0xFFFFu;
constexpr uint32_t pack_gddr_selector(uint32_t local_mimir, uint32_t target_mimir) {
    return (local_mimir << kGddrLocalShift) | target_mimir;
}
constexpr uint32_t gddr_selector_local(uint32_t selector) { return selector >> kGddrLocalShift; }
constexpr uint32_t gddr_selector_target(uint32_t selector) { return selector & kGddrTargetMask; }
// Done block: after the last op the kernel writes the XOR, then the magic. An early return leaves the host's poison.
inline constexpr uint32_t kDoneMagicWord = 0, kDoneXorWord = 1, kDoneWords = 2;
inline constexpr uint32_t kDoneMagic = 0xD0E5C0DEu, kDonePoison = 0xBADD0E00u;
}  // namespace cce_copy_ops_abi
