// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <stdint.h>

#include "cce_test_status_abi.h"

// cce_copy_ops kernel ABI. The kernel (cce_copy_ops.cpp) and the host (dram/mimir_access.hpp) both include this.
namespace cce_copy_ops_abi {
inline constexpr uint32_t kArgDoneOffset = 0, kArgNumOps = 1, kHeaderWords = 2;
inline constexpr uint32_t kOpSrcKind = 0, kOpSrcSelector = 1, kOpSrcOffsetLo = 2, kOpSrcOffsetHi = 3, kOpDstKind = 4,
                          kOpDstSelector = 5, kOpDstOffsetLo = 6, kOpDstOffsetHi = 7, kOpNumWords = 8, kOpWords = 9;
constexpr uint32_t op_arg_index(uint32_t op_index, uint32_t field) {
    return kHeaderWords + kOpWords * op_index + field;
}
inline constexpr uint32_t kKindOwnTl1 = 0, kKindGddr = 1, kKindSramSpa = 2;
// Gddr selector: the target Mimir. The CCE's host-written route table picks the D2D port.
// Done block: after the last op the kernel writes the XOR, then the magic. An early return leaves the host's poison.
inline constexpr uint32_t kDoneMagicWord = 0, kDoneXorWord = 1, kDoneWords = 2;
inline constexpr uint32_t kDoneMagic = 0xD0E5C0DEu, kDonePoison = cce_test_status_abi::kStatusPoison;
}  // namespace cce_copy_ops_abi
