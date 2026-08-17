// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include <cstdint>
#include <string>
#include <vector>

#include "tt_metal/hw/ckernels/quasar/metal/llk_api/dest_order.h"

namespace {

using dest_order::bit_of;
using dest_order::chain;
using dest_order::client;

using dest_order::stage::fpu;
using dest_order::stage::pack;
using dest_order::stage::sfpu;
using dest_order::stage::unpack;

std::string name_of(client c) {
    switch (c) {
        case client::UNPACK: return "UNPACK";
        case client::FPU: return "FPU";
        case client::SFPU: return "SFPU";
        case client::PACK: return "PACK";
    }
    return "?";
}

template <typename CHAIN>
std::vector<std::string> stages_of() {
    std::vector<std::string> out;
    for (std::uint32_t i = 0; i < CHAIN::size; ++i) {
        out.push_back(name_of(CHAIN::at(i)));
    }
    return out;
}

template <typename CHAIN>
std::vector<std::string> visit_each() {
    std::vector<std::string> out;
    CHAIN::for_each([&out](auto s) { out.push_back(name_of(s.value)); });
    return out;
}

template <typename CHAIN>
std::vector<std::string> visit_runs() {
    std::vector<std::string> out;
    CHAIN::for_each_run([&out](auto s) { out.push_back(name_of(s.value)); });
    return out;
}

template <typename CHAIN>
std::vector<std::string> visit_distinct() {
    std::vector<std::string> out;
    CHAIN::for_each_distinct([&out](auto s) { out.push_back(name_of(s.value)); });
    return out;
}

using Pipeline = chain<unpack, fpu, sfpu, fpu, pack>;
using FpuLoop = chain<fpu, fpu, fpu, fpu, pack>;
using Ring = chain<unpack, fpu, sfpu, pack>;

}  // namespace

TEST(DestOrder, ClientEncodingMatchesHardwareDvalidBits) {
    EXPECT_EQ(static_cast<std::uint32_t>(client::UNPACK), 0u);
    EXPECT_EQ(static_cast<std::uint32_t>(client::FPU), 1u);
    EXPECT_EQ(static_cast<std::uint32_t>(client::SFPU), 2u);
    EXPECT_EQ(static_cast<std::uint32_t>(client::PACK), 3u);

    EXPECT_EQ(bit_of(client::UNPACK), 0b0001u);
    EXPECT_EQ(bit_of(client::FPU), 0b0010u);
    EXPECT_EQ(bit_of(client::SFPU), 0b0100u);
    EXPECT_EQ(bit_of(client::PACK), 0b1000u);
}

TEST(DestOrder, ChainPreservesDeclaredProgramOrder) {
    static_assert(Pipeline::size == 5);
    static_assert(Pipeline::at(0) == client::UNPACK);
    static_assert(Pipeline::at(1) == client::FPU);
    static_assert(Pipeline::at(2) == client::SFPU);
    static_assert(Pipeline::at(3) == client::FPU);
    static_assert(Pipeline::at(4) == client::PACK);
    static_assert(Pipeline::first() == client::UNPACK);
    static_assert(Pipeline::last() == client::PACK);

    EXPECT_EQ(stages_of<Pipeline>(), (std::vector<std::string>{"UNPACK", "FPU", "SFPU", "FPU", "PACK"}));
}

TEST(DestOrder, ParticipationQueries) {
    static_assert(Pipeline::distinct_count() == 4);
    static_assert(Pipeline::count_of(client::FPU) == 2);
    static_assert(Pipeline::count_of(client::SFPU) == 1);
    static_assert(Pipeline::index_of(client::FPU) == 1);
    static_assert(Pipeline::contains(client::UNPACK));
    static_assert(Pipeline::contains(client::PACK));

    static_assert(
        Pipeline::mask() ==
        (bit_of(client::UNPACK) | bit_of(client::FPU) | bit_of(client::SFPU) | bit_of(client::PACK)));

    static_assert(!FpuLoop::contains(client::UNPACK));
    static_assert(FpuLoop::mask() == (bit_of(client::FPU) | bit_of(client::PACK)));

    EXPECT_EQ(Pipeline::distinct_count(), 4u);
    EXPECT_EQ(Pipeline::count_of(client::FPU), 2u);
}

TEST(DestOrder, RingTopologyWrapsCyclically) {
    static_assert(Ring::successors_mask(client::UNPACK) == bit_of(client::FPU));
    static_assert(Ring::successors_mask(client::FPU) == bit_of(client::SFPU));
    static_assert(Ring::successors_mask(client::SFPU) == bit_of(client::PACK));
    static_assert(Ring::successors_mask(client::PACK) == bit_of(client::UNPACK));

    static_assert(Ring::successor(client::SFPU) == client::PACK);
    static_assert(Ring::successor(client::PACK) == client::UNPACK);
    static_assert(Ring::predecessor(client::UNPACK) == client::PACK);

    static_assert(Ring::is_simple_ring());

    EXPECT_EQ(Ring::successors_mask(client::PACK), bit_of(client::UNPACK));
}

TEST(DestOrder, RepeatedClientYieldsMultipleSuccessors) {
    static_assert(Pipeline::successors_mask(client::FPU) == (bit_of(client::SFPU) | bit_of(client::PACK)));
    static_assert(Pipeline::predecessors_mask(client::FPU) == (bit_of(client::UNPACK) | bit_of(client::SFPU)));
    static_assert(!Pipeline::is_simple_ring());

    EXPECT_FALSE(Pipeline::is_simple_ring());
}

TEST(DestOrder, SuccessorsSkipSelfAcrossConsecutiveRepeats) {
    static_assert(FpuLoop::successors_mask(client::FPU) == bit_of(client::PACK));
    static_assert(FpuLoop::successors_mask(client::PACK) == bit_of(client::FPU));
    static_assert(FpuLoop::predecessors_mask(client::FPU) == bit_of(client::PACK));
    static_assert(FpuLoop::predecessors_mask(client::PACK) == bit_of(client::FPU));

    static_assert(FpuLoop::is_simple_ring());
    static_assert(FpuLoop::collapsed::successors_mask(client::FPU) == FpuLoop::successors_mask(client::FPU));

    EXPECT_EQ(FpuLoop::successors_mask(client::FPU), bit_of(client::PACK));
    EXPECT_TRUE(FpuLoop::is_simple_ring());
}

TEST(DestOrder, ConsecutiveRunsCollapse) {
    static_assert(FpuLoop::size == 5);
    static_assert(!FpuLoop::is_collapsed());
    static_assert(FpuLoop::collapsed::size == 2);
    static_assert(FpuLoop::collapsed::at(0) == client::FPU);
    static_assert(FpuLoop::collapsed::at(1) == client::PACK);

    static_assert(chain<fpu, fpu, fpu>::collapsed::size == 1);

    EXPECT_EQ(stages_of<FpuLoop::collapsed>(), (std::vector<std::string>{"FPU", "PACK"}));
}

TEST(DestOrder, NonConsecutiveRepeatSurvivesCollapse) {
    static_assert(Pipeline::is_collapsed());
    static_assert(Pipeline::collapsed::size == 5);
    static_assert(Pipeline::collapsed::count_of(client::FPU) == 2);

    EXPECT_EQ(stages_of<Pipeline::collapsed>(), (std::vector<std::string>{"UNPACK", "FPU", "SFPU", "FPU", "PACK"}));
}

TEST(DestOrder, CollapseFoldsRingWrapDuplicate) {
    using Wrap = chain<pack, fpu, fpu, pack>;
    static_assert(Wrap::collapsed::size == 2);
    static_assert(Wrap::collapsed::at(0) == client::PACK);
    static_assert(Wrap::collapsed::at(1) == client::FPU);

    EXPECT_EQ(stages_of<Wrap::collapsed>(), (std::vector<std::string>{"PACK", "FPU"}));
}

TEST(DestOrder, CollapseIsIdempotentAndMinimalChainsUnchanged) {
    static_assert(FpuLoop::collapsed::collapsed::size == FpuLoop::collapsed::size);
    static_assert(Ring::is_collapsed());
    static_assert(Ring::collapsed::size == Ring::size);

    static_assert(FpuLoop::collapsed::is_simple_ring());
}

TEST(DestOrder, IterationModesDiffer) {
    EXPECT_EQ(visit_each<FpuLoop>(), (std::vector<std::string>{"FPU", "FPU", "FPU", "FPU", "PACK"}));
    EXPECT_EQ(visit_runs<FpuLoop>(), (std::vector<std::string>{"FPU", "PACK"}));
    EXPECT_EQ(visit_distinct<FpuLoop>(), (std::vector<std::string>{"FPU", "PACK"}));

    EXPECT_EQ(visit_each<Pipeline>(), (std::vector<std::string>{"UNPACK", "FPU", "SFPU", "FPU", "PACK"}));
    EXPECT_EQ(visit_runs<Pipeline>(), (std::vector<std::string>{"UNPACK", "FPU", "SFPU", "FPU", "PACK"}));
    EXPECT_EQ(visit_distinct<Pipeline>(), (std::vector<std::string>{"UNPACK", "FPU", "SFPU", "PACK"}));
}

TEST(DestOrder, TouchTracksMask) {
    dest_order::reset_touched();
    EXPECT_FALSE(dest_order::was_touched(client::FPU));
    EXPECT_FALSE(dest_order::was_touched(client::SFPU));

    dest_order::touch_fpu();
    EXPECT_TRUE(dest_order::was_touched(client::FPU));
    EXPECT_FALSE(dest_order::was_touched(client::SFPU));

    dest_order::touch_sfpu();
    EXPECT_TRUE(dest_order::was_touched(client::FPU));
    EXPECT_TRUE(dest_order::was_touched(client::SFPU));

    dest_order::reset_touched();
    EXPECT_FALSE(dest_order::was_touched(client::FPU));
    EXPECT_FALSE(dest_order::was_touched(client::SFPU));
}

TEST(DestOrder, TouchAccumulatesAllClients) {
    dest_order::reset_touched();
    dest_order::touch_unpack();
    dest_order::touch_fpu();
    dest_order::touch_sfpu();
    dest_order::touch_pack();
    EXPECT_TRUE(dest_order::was_touched(client::UNPACK));
    EXPECT_TRUE(dest_order::was_touched(client::FPU));
    EXPECT_TRUE(dest_order::was_touched(client::SFPU));
    EXPECT_TRUE(dest_order::was_touched(client::PACK));
    EXPECT_EQ(dest_order::touched_mask, Ring::mask());
    dest_order::reset_touched();
}

TEST(DestOrder, SupersetChainPassthroughDetection) {
    using Superset = chain<fpu, sfpu, pack>;

    dest_order::reset_touched();
    dest_order::touch_fpu();
    dest_order::touch_pack();

    EXPECT_TRUE(dest_order::was_touched(client::FPU));
    EXPECT_FALSE(dest_order::was_touched(client::SFPU));
    EXPECT_TRUE(dest_order::was_touched(client::PACK));

    EXPECT_TRUE(Superset::contains(client::SFPU));
    EXPECT_FALSE(dest_order::was_touched(client::SFPU));

    dest_order::reset_touched();
}
