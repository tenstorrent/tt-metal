// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

// The experimental 2.0 compute API is Blackhole-only this phase. llk_operand.h is included by every 2.0 op
// header, so guarding it here hard-fails the compile of any 2.0 kernel built for another arch (rather than
// silently compiling an empty op body from the per-file `#ifdef ARCH_BLACKHOLE` guards).
#ifndef ARCH_BLACKHOLE
#error "The experimental 2.0 compute API (LLKOperand) is Blackhole-only; build with ARCH_BLACKHOLE defined."
#endif

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
// derive per-tile addresses internally from the compile-time geometry (see tile_stride_words).
//
// Namespace split: LLKOperand lives in ckernel::experimental (public); ckernel::experimental::detail holds
// internal-only helpers built on it (tile_address). The source (CB / DataflowBuffer / Scratchpad) is NOT
// known here -- source -> operand is done at the call site via the test-common CB helpers
// (cb_operand_helpers.h) or a future accessor's translator.
// =====================================================================================================

namespace ckernel {
namespace experimental {

// clang-format off
/**
 * The public id-free operand every 2.0 compute op consumes. Bundles the compile-time "what" (Format + Shape as
 * NTTPs) with the single runtime "where" (l1_address). ::descriptor packs Format+Shape into the LLKMemDescriptor
 * the LLK layer takes as an NTTP. Construct at the call site from an address (e.g. cb_read_address(cb, tile)).
 *
 * | Param Type | Name   | Description                               | Type        | Valid Range | Required |
 * |------------|--------|-------------------------------------------|-------------|-------------|----------|
 * | Template   | Format | Buffer L1 data format                     | DataFormat  | N/A         | True     |
 * | Template   | Shape  | Tile geometry                             | TensorShape | N/A         | True     |
 * | Function   | addr   | Absolute L1 tile base address (16B words) | uint32_t    | N/A         | True     |
 */
// clang-format on
template <DataFormat Format, TensorShape Shape>
struct LLKOperand {
    std::uint32_t l1_address;  // runtime "where"; Format/Shape are the compile-time "what"
    constexpr explicit LLKOperand(std::uint32_t addr) : l1_address(addr) {}

    // The descriptor the LLK APIs accept (buffer L1 format + geometry).
    static constexpr LLKMemDescriptor descriptor = LLKMemDescriptor{Format, Shape};
};

namespace detail {
// Absolute per-tile L1 base for operand `op` at `tile_index` (16B words). Stride folds to a compile-time
// constant via tile_stride_words (one tile's L1 size). Shared by every block/absolute-addressing op.
template <DataFormat Format, TensorShape Shape>
ALWI std::uint32_t tile_address(LLKOperand<Format, Shape> op, std::uint32_t tile_index) {
    constexpr std::uint32_t stride = tile_stride_words(Format, Shape);
    return op.l1_address + tile_index * stride;
}
}  // namespace detail

}  // namespace experimental
}  // namespace ckernel
