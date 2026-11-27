// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include <algorithm>
#include <cstdint>
#include <vector>

#include "metal/ops/all_gather_minimal_matmul_outer_dim_async/device/stripe_layout.hpp"

namespace {
using ttml::metal::ops::all_gather_minimal_matmul_outer_dim_async::make_relay_plan;
using ttml::metal::ops::all_gather_minimal_matmul_outer_dim_async::make_stripe_layout;
using ttml::metal::ops::all_gather_minimal_matmul_outer_dim_async::RelayPlan;
using ttml::metal::ops::all_gather_minimal_matmul_outer_dim_async::StripeLayout;

constexpr uint32_t kNBlockTiles = 8;

const std::vector<uint32_t> kRingSizes = {1, 2, 3, 4, 5, 8, 16, 32};
const std::vector<uint32_t> kChainCounts = {1, 4, 8, 11, 13};
const std::vector<uint32_t> kStripeTiles = {1, 3, 7, 8, 16, 21, 64, 86, 128};

void expect_layout_covers_every_tile_once(const StripeLayout& layout, uint32_t N_tiles) {
    std::vector<uint32_t> owners(N_tiles, 0);
    for (uint32_t s : layout.stripe_order) {
        for (uint32_t c = 0; c < layout.num_chains; ++c) {
            ASSERT_LE(layout.chain_width(c), layout.piece_tiles);
            for (uint32_t j = 0; j < layout.chain_width(c); ++j) {
                owners.at((s * layout.stripe_tiles) + layout.chain_offset(c) + j)++;
            }
        }
    }
    for (uint32_t t = 0; t < N_tiles; ++t) {
        ASSERT_EQ(owners[t], 1u) << "tile " << t;
    }
}

void expect_relay_delivers_every_piece_once(const StripeLayout& layout, bool ring, uint32_t parity_unit) {
    const uint32_t r = layout.ring_size;
    const uint32_t chains = layout.num_chains;
    std::vector<RelayPlan> plans;
    plans.reserve(r);
    for (uint32_t d = 0; d < r; ++d) {
        plans.push_back(make_relay_plan(layout, d, ring, parity_unit));
    }
    std::vector<uint32_t> received(r * r * chains, 0);
    for (uint32_t d = 0; d < r; ++d) {
        for (const int32_t dir : {1, -1}) {
            for (const auto& seg : (dir > 0 ? plans[d].forward : plans[d].backward)) {
                const int32_t to = static_cast<int32_t>(d) + dir;
                ASSERT_TRUE(ring || (to >= 0 && to < static_cast<int32_t>(r)))
                    << "device " << d << " sends off the line";
                const uint32_t dst = static_cast<uint32_t>(
                    ((to % static_cast<int32_t>(r)) + static_cast<int32_t>(r)) % static_cast<int32_t>(r));
                const auto& onward = dir > 0 ? plans[dst].forward : plans[dst].backward;
                const auto next = std::find_if(onward.begin(), onward.end(), [&](const auto& s) {
                    return s.stripe == seg.stripe && s.distance == seg.distance + 1;
                });
                for (const auto& p : seg.pieces) {
                    received[(((dst * r) + seg.stripe) * chains) + p.chain]++;
                    const bool sent_on = next != onward.end() &&
                                         std::any_of(next->pieces.begin(), next->pieces.end(), [&](const auto& q) {
                                             return q.chain == p.chain;
                                         });
                    ASSERT_EQ(p.notify != 0, sent_on)
                        << "chain " << p.chain << "'s piece of stripe " << seg.stripe << " sent by device " << d;
                }
            }
        }
    }
    for (uint32_t d = 0; d < r; ++d) {
        for (uint32_t s = 0; s < r; ++s) {
            for (uint32_t c = 0; c < chains; ++c) {
                const uint32_t expected = (d == s || layout.chain_width(c) == 0) ? 0u : 1u;
                ASSERT_EQ(received[(((d * r) + s) * chains) + c], expected)
                    << "chain " << c << "'s piece of stripe " << s << " at device " << d;
            }
        }
    }
}

TEST(AllGatherMinimalMatmulOuterDimStripeLayout, CoversEveryOutputTileOnce) {
    for (const bool ring : {true, false}) {
        for (const uint32_t r : kRingSizes) {
            for (const uint32_t chains : kChainCounts) {
                for (const uint32_t stripe : kStripeTiles) {
                    for (uint32_t d = 0; d < r; ++d) {
                        SCOPED_TRACE(
                            testing::Message() << "ring=" << ring << " R=" << r << " chains=" << chains
                                               << " stripe=" << stripe << " device=" << d);
                        const StripeLayout layout = make_stripe_layout(stripe * r, r, d, chains, kNBlockTiles, ring);
                        expect_layout_covers_every_tile_once(layout, stripe * r);
                    }
                }
            }
        }
    }
}

TEST(AllGatherMinimalMatmulOuterDimStripeLayout, RelayDeliversEveryPieceOnce) {
    for (const bool ring : {true, false}) {
        for (const uint32_t r : kRingSizes) {
            if (r < 2) {
                continue;
            }
            for (const uint32_t chains : kChainCounts) {
                for (const uint32_t stripe : kStripeTiles) {
                    for (const uint32_t parity_unit : {1u, 2u, 6u}) {
                        SCOPED_TRACE(
                            testing::Message() << "ring=" << ring << " R=" << r << " chains=" << chains
                                               << " stripe=" << stripe << " parity_unit=" << parity_unit);
                        const StripeLayout layout = make_stripe_layout(stripe * r, r, 0, chains, kNBlockTiles, ring);
                        expect_relay_delivers_every_piece_once(layout, ring, parity_unit);
                    }
                }
            }
        }
    }
}
}  // namespace
