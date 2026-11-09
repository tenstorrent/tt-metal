// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include <cstdint>
#include <set>
#include <vector>

#include "metal/ops/minimal_matmul_reduce_scatter_async/device/slice_layout.hpp"

namespace {
using ttml::metal::ops::minimal_matmul_reduce_scatter_async::make_slice_layout;
using ttml::metal::ops::minimal_matmul_reduce_scatter_async::SliceLayout;

const std::vector<uint32_t> kRingSizes = {2, 3, 4, 8};
const std::vector<uint32_t> kCoreCounts = {1, 4, 11};
const std::vector<uint32_t> kSliceTiles = {1, 3, 9, 65};
const std::vector<uint32_t> kNTiles = {1, 13, 128};
const std::vector<uint32_t> kBlockTiles = {1, 3, 8};

void expect_steps_cover_output_once(const SliceLayout& layout, uint32_t ring_index) {
    const uint32_t M_tiles = layout.slice_tiles * layout.ring_size;
    std::vector<uint32_t> owners(static_cast<size_t>(M_tiles) * layout.N_tiles, 0);
    for (uint32_t step = 0; step < layout.ring_size; ++step) {
        for (uint32_t nb = 0; nb < layout.N_blocks; ++nb) {
            const uint32_t slice = mmrs::step_slice(step, ring_index, layout.ring_size);
            for (uint32_t mc = 0; mc < layout.m_cores; ++mc) {
                for (uint32_t nc = 0; nc < layout.n_cores; ++nc) {
                    const mmrs::Span m_share = layout.m_share(mc);
                    const mmrs::Span n_share = layout.n_share(nc);
                    ASSERT_LE(m_share.size, layout.m_piece);
                    ASSERT_LE(n_share.size, layout.n_piece);
                    const mmrs::Span n = layout.n_block(nc, nb);
                    ASSERT_LE(n.size, mmrs::balanced_span(nb, layout.n_piece, layout.N_blocks).size);
                    for (uint32_t mb = 0; mb < layout.M_blocks; ++mb) {
                        const mmrs::Span m = mmrs::clip(layout.m_block(mb), m_share.size);
                        for (uint32_t r = 0; r < m.size; ++r) {
                            for (uint32_t c = 0; c < n.size; ++c) {
                                const uint32_t row = (slice * layout.slice_tiles) + m_share.start + m.start + r;
                                const uint32_t col = n_share.start + n.start + c;
                                owners.at((static_cast<size_t>(row) * layout.N_tiles) + col)++;
                            }
                        }
                    }
                }
            }
        }
    }
    for (size_t t = 0; t < owners.size(); ++t) {
        ASSERT_EQ(owners[t], 1U) << "tile " << t << " was produced " << owners[t] << " times";
    }
}
}  // namespace

TEST(MinimalMatmulReduceScatterSliceLayoutTest, BalancedSpansTileTheRange) {
    for (const uint32_t total : {0U, 1U, 7U, 8U, 13U, 128U}) {
        for (const uint32_t parts : {1U, 2U, 3U, 10U, 11U}) {
            uint32_t next = 0;
            for (uint32_t i = 0; i < parts; ++i) {
                const mmrs::Span s = mmrs::balanced_span(i, total, parts);
                ASSERT_EQ(s.start, next);
                ASSERT_LE(s.size, (total + parts - 1) / parts);
                next += s.size;
            }
            ASSERT_EQ(next, total);
        }
    }
}

TEST(MinimalMatmulReduceScatterSliceLayoutTest, RingStepsChainAndEndOnOwnSlice) {
    for (const uint32_t r : kRingSizes) {
        for (uint32_t d = 0; d < r; ++d) {
            std::set<uint32_t> seen;
            for (uint32_t t = 0; t < r; ++t) {
                seen.insert(mmrs::step_slice(t, d, r));
            }
            EXPECT_EQ(seen.size(), r) << "ring " << r << " device " << d;
            EXPECT_EQ(mmrs::step_slice(r - 1, d, r), d);
            const uint32_t next = (d + 1) % r;
            for (uint32_t t = 0; t + 1 < r; ++t) {
                EXPECT_EQ(mmrs::step_slice(t + 1, next, r), mmrs::step_slice(t, d, r));
            }
        }
    }
}

TEST(MinimalMatmulReduceScatterSliceLayoutTest, StepsCoverEveryOutputTileOnce) {
    for (const uint32_t r : kRingSizes) {
        for (const uint32_t slice_tiles : kSliceTiles) {
            for (const uint32_t N_tiles : kNTiles) {
                for (const uint32_t cores : kCoreCounts) {
                    for (const uint32_t block : kBlockTiles) {
                        for (const bool narrow : {false, true}) {
                            const SliceLayout layout =
                                make_slice_layout(slice_tiles * r, N_tiles, r, cores, cores + 1, block, block, narrow);
                            SCOPED_TRACE(
                                testing::Message() << "ring " << r << " slice " << slice_tiles << " N " << N_tiles
                                                   << " cores " << cores << " block " << block << " narrow " << narrow);
                            expect_steps_cover_output_once(layout, r - 1);
                        }
                    }
                }
            }
        }
    }
}

TEST(MinimalMatmulReduceScatterSliceLayoutTest, NarrowFirstShareTakesAColumnPerBlock) {
    struct Case {
        uint32_t N_tiles;
        uint32_t n_cores;
        uint32_t n_piece;
        uint32_t n_first;
    };
    for (const Case& c : {Case{128, 11, 12, 10}, Case{128, 10, 13, 11}, Case{344, 11, 32, 28}}) {
        SCOPED_TRACE(testing::Message() << "N " << c.N_tiles << " cores " << c.n_cores);
        const SliceLayout plain = make_slice_layout(4 * 32, c.N_tiles, 4, 10, c.n_cores, 8, 8);
        const SliceLayout narrow = make_slice_layout(4 * 32, c.N_tiles, 4, 10, c.n_cores, 8, 8, true);
        EXPECT_EQ(plain.n_piece, c.n_piece);
        EXPECT_EQ(narrow.n_piece, c.n_piece);
        EXPECT_EQ(narrow.N_blocks, plain.N_blocks);
        EXPECT_EQ(narrow.n_share(0).size, c.n_first);
        for (uint32_t nb = 0; nb < narrow.N_blocks; ++nb) {
            EXPECT_EQ(narrow.n_block(0, nb).size + 1, mmrs::balanced_span(nb, narrow.n_piece, narrow.N_blocks).size);
        }
    }
}

TEST(MinimalMatmulReduceScatterSliceLayoutTest, RejectsRowsThatDoNotSplitOverTheRing) {
    EXPECT_ANY_THROW(make_slice_layout(10, 8, 4, 10, 11, 8, 8));
}
