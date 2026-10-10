// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include <internal/disaggregation/layer_ack_derivation.hpp>

namespace tt::tt_metal::internal {

namespace {

const AckLabel& label(const std::variant<AckLabel, AckDesync>& r) { return std::get<AckLabel>(r); }

}  // namespace

TEST(LayerAckDerivation, DenseRankLabelsChunkAndLayerFromTheCounter) {
    AckDerivation d(AckSpace{/*num_ack_layers=*/61, /*ack_first_idx=*/16, /*ack_local_count=*/15, {}});
    const AckIdentity chunk0{3, 0, 5120};
    for (uint32_t i = 0; i < 15; ++i) {
        const auto l = label(d.step(chunk0));
        EXPECT_EQ(l.chunk, 0u);
        EXPECT_EQ(l.ack_idx, 16u + i);
        EXPECT_EQ(l.layer, 16u + i);
        EXPECT_EQ(l.seq, 16u + i);
        EXPECT_EQ(l.identity, chunk0);
    }
    const auto l = label(d.step(AckIdentity{3, 5120, 10240}));
    EXPECT_EQ(l.chunk, 1u);
    EXPECT_EQ(l.ack_idx, 16u);
    EXPECT_EQ(l.seq, 61u + 16u);
}

TEST(LayerAckDerivation, HybridRankMapsAckIndexToTheGlobalLayer) {
    AckDerivation d(AckSpace{/*num_ack_layers=*/6, /*ack_first_idx=*/3, /*ack_local_count=*/3, {15, 19, 23}});
    const AckIdentity id{0, 0, 128};
    EXPECT_EQ(label(d.step(id)).layer, 15u);
    EXPECT_EQ(label(d.step(id)).layer, 19u);
    const auto last = label(d.step(id));
    EXPECT_EQ(last.layer, 23u);
    EXPECT_EQ(last.ack_idx, 5u);
    EXPECT_EQ(last.seq, 5u);
    EXPECT_EQ(label(d.step(AckIdentity{1, 0, 128})).seq, 6u + 3u);
}

TEST(LayerAckDerivation, AnIdentityChangeMidChunkIsADesync) {
    AckDerivation d(AckSpace{4, 0, 4, {}});
    const AckIdentity a{0, 0, 128};
    const AckIdentity b{1, 0, 128};
    d.step(a);
    d.step(a);
    const auto r = d.step(b);  // record 2 of 4: a record of chunk a was lost
    ASSERT_TRUE(std::holds_alternative<AckDesync>(r));
    EXPECT_EQ(std::get<AckDesync>(r).slice_idx, 2u);
    EXPECT_EQ(std::get<AckDesync>(r).record, 2u);
    EXPECT_EQ(std::get<AckDesync>(r).identity, b);
}

TEST(LayerAckDerivation, AnIdentityChangeOnTheChunkBoundaryIsNormal) {
    AckDerivation d(AckSpace{2, 0, 2, {}});
    d.step(AckIdentity{0, 0, 128});
    d.step(AckIdentity{0, 0, 128});
    EXPECT_TRUE(std::holds_alternative<AckLabel>(d.step(AckIdentity{5, 128, 256})));
}

TEST(LayerAckDerivation, NarrowRecordsNeverDesync) {
    AckDerivation d(AckSpace{2, 0, 2, {}});
    for (int i = 0; i < 5; ++i) {
        EXPECT_TRUE(std::holds_alternative<AckLabel>(d.step(std::nullopt)));
    }
    EXPECT_EQ(d.records(), 5u);
}

}  // namespace tt::tt_metal::internal
