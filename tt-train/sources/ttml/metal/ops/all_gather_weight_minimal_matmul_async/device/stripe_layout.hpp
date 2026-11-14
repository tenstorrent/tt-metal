// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <vector>

namespace ttml::metal::ops::all_gather_weight_minimal_matmul_async {
// The weight is split along N into one stripe per device. Every in1 chain owns the same slice (piece) of
// each stripe and consumes the stripes in hop order: local first, then 1 hop, 2 hops, ...
struct StripeLayout {
    uint32_t ring_size = 1;
    uint32_t stripe_tiles = 0;  // N tiles per device
    uint32_t num_chains = 1;
    uint32_t piece_tiles = 0;  // widest chain slice; narrower chains pad up to it
    uint32_t N_block_tiles = 1;
    std::vector<uint32_t> stripe_order;
    uint32_t hops_before = 0;  // how many of the other stripes belong to devices behind us (lower ring index)

    uint32_t chain_offset(uint32_t c) const;
    uint32_t chain_width(uint32_t c) const;
    uint32_t n_blocks_per_core() const;
};

StripeLayout make_stripe_layout(
    uint32_t N_tiles, uint32_t ring_size, uint32_t ring_index, uint32_t num_chains, uint32_t N_block_tiles, bool ring);

// What a device sends in one direction: segment j forwards the stripe that started j hops back (j = 0 is our
// own stripe). notify marks pieces the next device forwards again.
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
