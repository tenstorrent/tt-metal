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
#ifdef ARCH_QUASAR
// clang-format off
/**
 * Quasar body of the operand. A Quasar unpack/pack addresses L1 as buffer-descriptor base + tile index, so the
 * runtime "where" is three numbers: the base an op init programs into its buffer descriptor (bd_base), the tile index
 * of the operand's first tile relative to that base (entry_idx), and the tile step between its consecutive tiles
 * (stride_tiles). Tile i of the operand is entry_idx + i * stride_tiles. DataflowBuffer::front/back fill all three
 * (bd_base = ring base, so one init covers the whole ring); a raw L1 address is {addr, 0, 1}.
 *
 * | Param Type | Name         | Description                                             | Type        | Valid Range | Required |
 * |------------|--------------|---------------------------------------------------------|-------------|-------------|----------|
 * | Template   | Format       | Buffer L1 data format                                   | DataFormat  | N/A         | True     |
 * | Template   | Shape        | Tile geometry                                           | TensorShape | N/A         | True     |
 * | Function   | addr         | Buffer-descriptor base address (16B words)              | uint32_t    | N/A         | True     |
 * | Function   | entry        | Tile index of the operand's first tile, from addr       | uint32_t    | < 65536     | False    |
 * | Function   | stride       | Tile step between consecutive tiles of the operand      | uint32_t    | 1 - 255     | False    |
 */
// clang-format on
template <DataFormat Format, TensorShape Shape>
struct LLKOperand {
    std::uint32_t l1_address;    // bd_base: buffer-descriptor base (16B words)
    std::uint16_t entry_idx;     // first tile, relative to l1_address
    std::uint8_t stride_tiles;   // tile step between consecutive tiles
    constexpr explicit LLKOperand(std::uint32_t addr, std::uint32_t entry = 0, std::uint32_t stride = 1) :
        l1_address(addr), entry_idx(static_cast<std::uint16_t>(entry)), stride_tiles(static_cast<std::uint8_t>(stride)) {}

    // L1 tile index (relative to l1_address) of the operand's tile i.
    constexpr std::uint32_t tile(std::uint32_t i) const { return entry_idx + i * stride_tiles; }

    // The descriptor the LLK APIs accept (buffer L1 format + geometry).
    static constexpr LLKMemDescriptor descriptor = LLKMemDescriptor{Format, Shape};
};
#else
template <DataFormat Format, TensorShape Shape>
struct LLKOperand {
    std::uint32_t l1_address;  // runtime "where"; Format/Shape are the compile-time "what"
    constexpr explicit LLKOperand(std::uint32_t addr) : l1_address(addr) {}

    // The descriptor the LLK APIs accept (buffer L1 format + geometry).
    static constexpr LLKMemDescriptor descriptor = LLKMemDescriptor{Format, Shape};
};
#endif

#ifndef ARCH_QUASAR  // Quasar addresses tiles through the buffer descriptor programmed at init (base + index).
namespace detail {
// Absolute per-tile L1 base for operand `op` at `tile_index` (16B words). Stride folds to a compile-time
// constant via tile_stride_words (one tile's L1 size). Shared by every block/absolute-addressing op.
template <DataFormat Format, TensorShape Shape>
ALWI std::uint32_t tile_address(LLKOperand<Format, Shape> op, std::uint32_t tile_index) {
    constexpr std::uint32_t stride = tile_stride_words(Format, Shape);
    return op.l1_address + tile_index * stride;
}
}  // namespace detail
#endif

}  // namespace experimental
}  // namespace ckernel
