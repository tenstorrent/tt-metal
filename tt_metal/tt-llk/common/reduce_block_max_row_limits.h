// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

namespace ckernel
{

/**
 * @brief Widest block, in tiles, the block reduce_max_row LLKs accept; one bound for both architectures.
 *
 * block_ct_dim goes straight into the MOP outer-loop count, which keeps 7 bits on Wormhole, so 128 expands to
 * no instructions and the packed row max is stale DEST. Blackhole's count keeps 10 bits, so there 127 is a
 * Wormhole-parity cap; its hard limit is the unpacker's tile counter, whose input address keeps 8 bits on both
 * architectures (tile 256 aliases tile 0). The compile-time and runtime LLKs, the Compute API documentation
 * and the SDPA host checks all use this one bound.
 */
constexpr std::uint32_t REDUCE_BLOCK_MAX_ROW_MAX_BLOCK_CT_DIM = 127;

/**
 * @brief Whether a block width is one the block reduce_max_row LLKs can process.
 *
 * Zero is rejected too: it gives both MOPs an outer-loop count of 0, the same stale-DEST result as a wrapped
 * count.
 *
 * @param block_ct_dim: Block width in tiles.
 */
constexpr bool is_valid_reduce_block_max_row_block_ct_dim(const std::uint32_t block_ct_dim)
{
    return block_ct_dim >= 1 && block_ct_dim <= REDUCE_BLOCK_MAX_ROW_MAX_BLOCK_CT_DIM;
}

} // namespace ckernel
