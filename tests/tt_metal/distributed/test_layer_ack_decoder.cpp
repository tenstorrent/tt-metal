// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include <internal/disaggregation/layer_ack_decoder.hpp>

namespace tt::tt_metal::internal {

namespace {

const DecodedAck& decoded(const DecodeResult& r) { return std::get<DecodedAck>(r); }

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

// A packed forward acks every chunk on a layer before the next layer.
TEST(LayerAckDecoder, PackedChunksInterleaveLayerByLayer) {
    LayerAckDecoder decoder(LayerAckLayout{/*first_record=*/3, /*record_count=*/3, {15, 19, 23}, /*open_chunks=*/4});
    const std::vector<AckIdentity> chunks{{0, 0, 128}, {1, 0, 128}, {2, 128, 256}, {3, 0, 128}};
    for (int forward = 0; forward < 2; ++forward) {
        for (const uint32_t layer : {15u, 19u, 23u}) {
            for (const auto& chunk : chunks) {
                const auto l = decoded(decoder.decode(chunk));
                EXPECT_EQ(l.layer, layer);
                EXPECT_EQ(l.identity, chunk);
            }
        }
    }
}

// A lost record shows when a later-started chunk gets ahead of its chunk.
TEST(LayerAckDecoder, AChunkGettingAheadOfAnOlderOneIsADesync) {
    LayerAckDecoder decoder(LayerAckLayout{0, /*record_count=*/3, {}, /*open_chunks=*/2});
    const AckIdentity a{0, 0, 128};
    const AckIdentity b{1, 0, 128};
    decoder.decode(a);
    decoder.decode(b);
    const auto r = decoder.decode(b);  // a's second record was lost: b is ahead of a
    ASSERT_TRUE(std::holds_alternative<LostAckRecord>(r));
    EXPECT_EQ(std::get<LostAckRecord>(r).slice_idx, 1u);
    EXPECT_EQ(std::get<LostAckRecord>(r).record, 2u);
    EXPECT_EQ(std::get<LostAckRecord>(r).identity, b);
}

// A chunk short a record blocks the next chunk start, even with open slots to spare.
TEST(LayerAckDecoder, AChunkStartingBesideAShortOneIsADesync) {
    LayerAckDecoder decoder(LayerAckLayout{0, /*record_count=*/3, {}, /*open_chunks=*/4});
    const AckIdentity a{3, 0, 128};
    decoder.decode(a);
    decoder.decode(a);  // a's third record was lost
    const auto r = decoder.decode(AckIdentity{3, 128, 256});
    ASSERT_TRUE(std::holds_alternative<LostAckRecord>(r));
    EXPECT_EQ(std::get<LostAckRecord>(r).slice_idx, 2u);
    EXPECT_EQ(std::get<LostAckRecord>(r).record, 2u);
}

// A chunk may send several records on one layer (an MTP level and its aliases) before the next chunk's.
TEST(LayerAckDecoder, AChunkMaySendSeveralRecordsBeforeTheNextChunk) {
    LayerAckDecoder decoder(LayerAckLayout{0, /*record_count=*/3, {}, /*open_chunks=*/2});
    const AckIdentity a{0, 0, 128};
    const AckIdentity b{1, 0, 128};
    for (const auto& id : {a, b, a, a, b, b}) {
        EXPECT_TRUE(std::holds_alternative<DecodedAck>(decoder.decode(id)));
    }
}

// A record naming no chunk is dropped when packed (open_chunks > 1), decoded when not.
TEST(LayerAckDecoder, ARecordNamingNoChunkIsDropped) {
    LayerAckDecoder decoder(LayerAckLayout{0, /*record_count=*/2, {}, /*open_chunks=*/2});
    const AckIdentity a{0, 0, 128};
    const AckIdentity none{0, 0, 0};
    EXPECT_EQ(decoded(decoder.decode(a)).layer, 0u);
    EXPECT_TRUE(std::holds_alternative<NoChunkAck>(decoder.decode(none)));
    EXPECT_EQ(decoded(decoder.decode(a)).layer, 1u);
    EXPECT_TRUE(std::holds_alternative<NoChunkAck>(decoder.decode(none)));
    EXPECT_EQ(decoded(decoder.decode(AckIdentity{1, 0, 128})).layer, 0u);
    LayerAckDecoder unpacked(LayerAckLayout{0, /*record_count=*/2, {}});
    EXPECT_EQ(decoded(unpacked.decode(none)).layer, 0u);
}

}  // namespace tt::tt_metal::internal
