// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Machine-free 1D route walk coverage: the router's per-hop advance, including the extension-word refill.

#include <gtest/gtest.h>

#include <array>
#include <cstdint>
#include <vector>

#include "hostdevcommon/fabric_common.h"

namespace tt::tt_fabric::route_1d_codec_tests {
namespace {

using HopAction = RoutingFieldsConstants::LowLatencyHopAction;
using routing_encoding::get_current_1d_hop_action;
using routing_encoding::refill_1d_route;

constexpr uint32_t kHopsPerWord = RoutingFieldsConstants::LowLatency::BASE_HOPS;
constexpr uint32_t kMaxWords = 4;

// Advances a route held as [live word, extension words...] past the current hop in place, the way the router's
// update_packet_header_for_next_hop does and decode will.
template <uint32_t NUM_EXT_WORDS>
constexpr void advance_in_place(std::array<uint32_t, kMaxWords>& words) {
    words[0] >>= RoutingFieldsConstants::LowLatency::FIELD_WIDTH;
    if constexpr (NUM_EXT_WORDS > 0) {
        words[0] = refill_1d_route<NUM_EXT_WORDS>(
            words[0],
            [&](uint32_t i) { return words[i + 1]; },
            [&](uint32_t i, uint32_t word) { words[i + 1] = word; });
    }
}

// The refill works in constant expressions.
static_assert([] {
    std::array<uint32_t, kMaxWords> words = {0b10, 0xA, 0xB, 0xC};
    advance_in_place<3>(words);
    return words;
}() == std::array<uint32_t, kMaxWords>{0xA, 0xB, 0xC, 0});

// Replays the route the way the routers do: each hop reads its action, stops after WRITE_ONLY (or a NOOP), and
// advances the route before forwarding.
template <uint32_t NUM_EXT_WORDS>
std::vector<HopAction> walk(std::array<uint32_t, kMaxWords> words) {
    std::vector<HopAction> actions;
    for (uint32_t hop = 0; hop <= kHopsPerWord * (NUM_EXT_WORDS + 1); ++hop) {
        const HopAction action = get_current_1d_hop_action(words[0]);
        actions.push_back(action);
        if (action == HopAction::WRITE_ONLY || action == HopAction::NOOP) {
            break;
        }
        advance_in_place<NUM_EXT_WORDS>(words);
    }
    return actions;
}

// Decode learns the word count at run time, so it picks the instantiation the same way.
std::vector<HopAction> walk(const std::array<uint32_t, kMaxWords>& words, uint32_t num_words) {
    switch (num_words) {
        case 1: return walk<0>(words);
        case 2: return walk<1>(words);
        case 3: return walk<2>(words);
        default: return walk<3>(words);
    }
}

std::vector<HopAction> expected_route(uint32_t forward_only, uint32_t write_and_forward) {
    std::vector<HopAction> actions(forward_only, HopAction::FORWARD_ONLY);
    actions.insert(actions.end(), write_and_forward, HopAction::WRITE_AND_FORWARD);
    actions.push_back(HopAction::WRITE_ONLY);
    return actions;
}

}  // namespace

TEST(Route1D, RefillKeepsALiveWordWithHopsLeft) {
    // Separate input and output storage, like the router's cached copy and packet header.
    const std::array<uint32_t, 1> ext_in = {0x1234};
    std::array<uint32_t, 1> ext_out = {};
    EXPECT_EQ(
        refill_1d_route<1>(
            0b01'10, [&](uint32_t i) { return ext_in[i]; }, [&](uint32_t i, uint32_t word) { ext_out[i] = word; }),
        0b01'10u);
    EXPECT_EQ(ext_out[0], 0x1234u);
}

TEST(Route1D, RefillReplacesASpentLiveWord) {
    const std::array<uint32_t, 3> ext_in = {0xA, 0xB, 0xC};
    std::array<uint32_t, 3> ext_out = {};
    EXPECT_EQ(
        refill_1d_route<3>(
            0, [&](uint32_t i) { return ext_in[i]; }, [&](uint32_t i, uint32_t word) { ext_out[i] = word; }),
        0xAu);
    EXPECT_EQ(ext_out, (std::array<uint32_t, 3>{0xB, 0xC, 0}));
}

TEST(Route1D, AdvanceRefillsInPlace) {
    std::array<uint32_t, kMaxWords> words = {0b10, 0xA, 0xB, 0xC};
    advance_in_place<3>(words);
    EXPECT_EQ(words, (std::array<uint32_t, kMaxWords>{0xA, 0xB, 0xC, 0}));
}

TEST(Route1D, AdvanceWithoutExtensionWordsOnlyShifts) {
    std::array<uint32_t, kMaxWords> words = {0b01'10'10, 0xA, 0xB, 0xC};
    advance_in_place<0>(words);
    EXPECT_EQ(words, (std::array<uint32_t, kMaxWords>{0b01'10, 0xA, 0xB, 0xC}));
}

TEST(Route1D, UnicastListsEveryHop) {
    for (uint32_t num_words = 1; num_words <= kMaxWords; ++num_words) {
        for (uint32_t hops = 1; hops <= kHopsPerWord * num_words; ++hops) {
            SCOPED_TRACE(testing::Message() << "num_words=" << num_words << " hops=" << hops);
            std::array<uint32_t, kMaxWords> words{};
            routing_encoding::encode_1d_unicast(static_cast<uint8_t>(hops), words.data(), num_words);
            EXPECT_EQ(walk(words, num_words), expected_route(hops - 1, 0));
        }
    }
}

TEST(Route1D, MulticastListsEveryHop) {
    for (uint32_t num_words = 1; num_words <= kMaxWords; ++num_words) {
        for (uint32_t start = 1; start <= kHopsPerWord * num_words; ++start) {
            // range == 0 is left out: the encoder writes no WRITE_ONLY for it.
            for (uint32_t range = 1; start + range - 1 <= kHopsPerWord * num_words; ++range) {
                SCOPED_TRACE(
                    testing::Message() << "num_words=" << num_words << " start=" << start << " range=" << range);
                std::array<uint32_t, kMaxWords> words{};
                routing_encoding::encode_1d_multicast(
                    static_cast<uint8_t>(start), static_cast<uint8_t>(range), words.data(), num_words);
                EXPECT_EQ(walk(words, num_words), expected_route(start - 1, range - 1));
            }
        }
    }
}

TEST(Route1D, SparseMulticastFollowsTheMask) {
    const auto encode = [](uint16_t mask) {
        std::array<uint32_t, kMaxWords> words{};
        routing_encoding::encode_1d_sparse_multicast(mask, words[0]);
        return walk(words, 1);
    };
    EXPECT_EQ(encode(0b1), (std::vector<HopAction>{HopAction::WRITE_ONLY}));
    EXPECT_EQ(
        encode(0b1010),
        (std::vector<HopAction>{
            HopAction::FORWARD_ONLY, HopAction::WRITE_AND_FORWARD, HopAction::FORWARD_ONLY, HopAction::WRITE_ONLY}));
    EXPECT_EQ(encode(0xFFFF), expected_route(0, kHopsPerWord - 1));
}

}  // namespace tt::tt_fabric::route_1d_codec_tests
