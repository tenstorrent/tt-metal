// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include <internal/disaggregation/layer_ack_decoder.hpp>

namespace tt::tt_metal::internal {

namespace {

const DecodedAck& decoded(const std::variant<DecodedAck, LostAckRecord>& r) { return std::get<DecodedAck>(r); }

}  // namespace

TEST(LayerAckDecoder, DenseRankLabelsTheGlobalLayerFromTheCounter) {
    LayerAckDecoder decoder(LayerAckLayout{/*first_record=*/16, /*record_count=*/15, {}});
    const AckIdentity chunk0{3, 0, 5120};
    for (uint32_t i = 0; i < 15; ++i) {
        const auto l = decoded(decoder.decode(chunk0));
        EXPECT_EQ(l.layer, 16u + i);
        EXPECT_EQ(l.identity, chunk0);
    }
    const AckIdentity chunk1{3, 5120, 10240};
    const auto l = decoded(decoder.decode(chunk1));
    EXPECT_EQ(l.layer, 16u);
    EXPECT_EQ(l.identity, chunk1);
}

TEST(LayerAckDecoder, HybridRankMapsAckIndexToTheGlobalLayer) {
    LayerAckDecoder decoder(LayerAckLayout{/*first_record=*/3, /*record_count=*/3, {15, 19, 23}});
    const AckIdentity id{0, 0, 128};
    EXPECT_EQ(decoded(decoder.decode(id)).layer, 15u);
    EXPECT_EQ(decoded(decoder.decode(id)).layer, 19u);
    EXPECT_EQ(decoded(decoder.decode(id)).layer, 23u);
    EXPECT_EQ(decoded(decoder.decode(AckIdentity{1, 0, 128})).layer, 15u);
}

TEST(LayerAckDecoder, AnIdentityChangeMidChunkIsADesync) {
    LayerAckDecoder decoder(LayerAckLayout{0, 4, {}});
    const AckIdentity a{0, 0, 128};
    const AckIdentity b{1, 0, 128};
    decoder.decode(a);
    decoder.decode(a);
    const auto r = decoder.decode(b);  // record 2 of 4: a record of chunk a was lost
    ASSERT_TRUE(std::holds_alternative<LostAckRecord>(r));
    EXPECT_EQ(std::get<LostAckRecord>(r).slice_idx, 2u);
    EXPECT_EQ(std::get<LostAckRecord>(r).record, 2u);
    EXPECT_EQ(std::get<LostAckRecord>(r).identity, b);
}

TEST(LayerAckDecoder, AnIdentityChangeOnTheChunkBoundaryIsNormal) {
    LayerAckDecoder decoder(LayerAckLayout{0, 2, {}});
    decoder.decode(AckIdentity{0, 0, 128});
    decoder.decode(AckIdentity{0, 0, 128});
    EXPECT_TRUE(std::holds_alternative<DecodedAck>(decoder.decode(AckIdentity{5, 128, 256})));
}

}  // namespace tt::tt_metal::internal
