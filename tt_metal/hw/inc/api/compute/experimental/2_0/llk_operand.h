// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "api/compute/common_globals.h"                            // DataFormat enum
#include "api/compute/experimental/2_0/internal/llk_descriptor.h"  // LLKMemDescriptor (the ::descriptor NTTP)

// =====================================================================================================
// The ONLY public, id-free type kernel authors use: LLKOperand<Format, Shape>. It bundles the two halves of
// "an L1 tile" split by compile-time vs runtime:
//   * Format + Shape are NON-TYPE TEMPLATE PARAMETERS (-ftt-nttp) -- the compile-time "what". They build
//     ::descriptor (an LLKMemDescriptor), forwarded to the LLK as an NTTP so the per-format switches /
//     register writes / asserts fold and DCE away.
//   * l1_address is the ONLY runtime member -- the "where". A runtime value cannot be an NTTP, so the split
//     lives INSIDE the type (NTTP vs member).
// Bundling keeps an address welded to its own descriptor (a wrong pairing is unrepresentable), and lets an op
// derive per-tile addresses internally from the compile-time geometry (see tilize_block + tile_stride_words).
//
// The source (CB / DataflowBuffer / Scratchpad) is NOT known here. Source -> operand is done at the call site
// via the test-common CB helpers (cb_operand_helpers.h) or a future accessor's translator.
// =====================================================================================================

namespace ckernel {
namespace experimental {

template <DataFormat Format, TensorShape Shape>
struct LLKOperand {
    std::uint32_t l1_address;  // runtime "where"; Format/Shape are the compile-time "what"
    constexpr explicit LLKOperand(std::uint32_t addr) : l1_address(addr) {}

    // The descriptor the LLK APIs accept (buffer L1 format + geometry).
    static constexpr LLKMemDescriptor descriptor = LLKMemDescriptor{static_cast<std::uint8_t>(Format), Shape};
};

}  // namespace experimental
}  // namespace ckernel
