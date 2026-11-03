// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <vector>

namespace ttml::metal::ops::all_gather_weight_minimal_matmul_async {
struct StripeLayout {
    uint32_t ring_size = 1;
    uint32_t stripe_tiles = 0;
    uint32_t num_chains = 1;
    uint32_t piece_tiles = 0;
    uint32_t N_block_tiles = 1;
    std::vector<uint32_t> stripe_order;
    uint32_t hops_before = 0;

    uint32_t chain_offset(uint32_t c) const;
    uint32_t chain_width(uint32_t c) const;
    uint32_t n_blocks_per_core() const;
};

StripeLayout make_stripe_layout(
    uint32_t N_tiles, uint32_t ring_size, uint32_t ring_index, uint32_t num_chains, uint32_t N_block_tiles, bool ring);

struct RelayPiece {
    uint32_t chain = 0;
    uint32_t notify = 0;
};

struct RelaySegment {
    uint32_t stripe = 0;
    uint32_t distance = 0;
    std::vector<RelayPiece> pieces;
};

struct RelayPlan {
    std::vector<RelaySegment> forward;
    std::vector<RelaySegment> backward;
};

RelayPlan make_relay_plan(const StripeLayout& layout, uint32_t ring_index, bool ring, uint32_t parity_unit);
}  // namespace ttml::metal::ops::all_gather_weight_minimal_matmul_async
