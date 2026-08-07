// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Properties of the requirement-driven stream-register assignment: every configuration's total
// fits under the pinned region, the placement policy is deterministic, the pinned dual-use id has
// at most one live consumer, and the emitted name set is exactly the 38 the kernel declares (an
// omitted name is an unset CT arg; the out-of-range sentinel marks an inactive consumer). Includes
// the formerly failing {express, VC1 absent} configuration: the need-driven assignment fits at 20
// of 32 where the old delta encoding (baseline plus release) fatals -- the register shortage is
// fixed and pinned here; the standalone single-mesh express configuration itself remains
// unexercised on hardware.

#include <gtest/gtest.h>

#include <hostdevcommon/fabric_common.h>

#include <map>
#include <set>
#include <string>

#include "tt_metal/fabric/builder/fabric_stream_assignment.hpp"
#include "tt_metal/fabric/hw/inc/edm_fabric/fabric_connection_interface.hpp"

namespace tt::tt_fabric {
namespace {

// The placement inputs are the fabric's family maxima; single-family fabrics keep their own count.
StreamPlacementInputs legacy_2d_placement() {
    return StreamPlacementInputs{
        .max_sender_counts = {4, 3, 0},
        .max_receiver_counts = {1, 1, 0},
        .vc2_present = false,
        .tensix_relay_present = false};
}

// A legacy fabric with an intermesh boundary: the boundary family's five-wide VC0 is the fabric max.
StreamPlacementInputs legacy_with_boundary_placement() {
    return StreamPlacementInputs{
        .max_sender_counts = {5, 4, 0},
        .max_receiver_counts = {1, 1, 0},
        .vc2_present = false,
        .tensix_relay_present = false};
}

StreamPlacementInputs express_full_placement() {
    return StreamPlacementInputs{
        .max_sender_counts = {5, 4, 1},
        .max_receiver_counts = {1, 1, 1},
        .vc2_present = true,
        .tensix_relay_present = false};
}

StreamPlacementInputs express_vc1_absent_placement() {
    return StreamPlacementInputs{
        .max_sender_counts = {5, 0, 0},
        .max_receiver_counts = {1, 0, 0},
        .vc2_present = false,
        .tensix_relay_present = false};
}

std::set<std::string> emitted_names(const StreamAssignment& a) {
    std::set<std::string> names;
    for (const auto& [name, value] : a.named_args()) {
        names.insert(name);
    }
    return names;
}

TEST(StreamAssignmentTest, Legacy2DFitsAndPlacesDeterministically) {
    const auto a = make_stream_assignment(stream_requirements(legacy_2d_placement(), CreditTransportPlan{}));
    // The layout: the vc0 sender-free-slots base pinned at the worker-facing constant (the bottom
    // of the file), then one contiguous run (receivers, acked, completed, downstream, vc1
    // sender-free).
    // The worker-facing contract, read from the header worker-space and ControlPlane already read:
    EXPECT_EQ(a.id(StreamRole::SENDER_FREE_SLOTS, 0, 0), connection_interface::sender_channel_0_free_slots_stream_id);
    EXPECT_EQ(a.id(StreamRole::SENDER_FREE_SLOTS, 0, 3), 3u);
    EXPECT_EQ(a.id(StreamRole::RECEIVER_PKTS_SENT, 0, 0), 4u);
    EXPECT_EQ(a.id(StreamRole::RECEIVER_PKTS_SENT, 1, 0), 5u);
    EXPECT_EQ(a.id(StreamRole::SENDER_PKTS_ACKED, 0, 0), 6u);
    EXPECT_EQ(a.id(StreamRole::SENDER_PKTS_ACKED, 0, 3), 9u);
    EXPECT_EQ(a.id(StreamRole::SENDER_PKTS_COMPLETED, 0, 0), 10u);
    EXPECT_EQ(a.id(StreamRole::SENDER_PKTS_COMPLETED, 1, 0), 14u);
    EXPECT_EQ(a.id(StreamRole::SENDER_PKTS_COMPLETED, 1, 2), 16u);
    EXPECT_EQ(a.id(StreamRole::DOWNSTREAM_FREE_SLOTS, 0, 0), 17u);
    EXPECT_EQ(a.id(StreamRole::DOWNSTREAM_FREE_SLOTS, 1, 3), 24u);
    EXPECT_EQ(a.id(StreamRole::SENDER_FREE_SLOTS, 1, 2), 27u);
    // A four-wide VC0 has no fifth ack.
    EXPECT_FALSE(a.has(StreamRole::SENDER_PKTS_ACKED, 0, 4));
}

TEST(StreamAssignmentTest, ExpressFullFitsWithVc1OnCounters) {
    CreditTransportPlan plan{};
    plan.vc1_uses_counters = true;
    const auto a = make_stream_assignment(stream_requirements(express_full_placement(), plan));
    // 2 receivers + 5 acked + 5 completed + 8 downstream + 9 sender-free = 29, under the pinned
    // region, with the VC2 sender/receiver live on their pinned ids.
    EXPECT_EQ(a.id(StreamRole::SENDER_FREE_SLOTS, 1, 3), 28u);
    EXPECT_FALSE(a.has(StreamRole::SENDER_PKTS_COMPLETED, 1, 0));
    EXPECT_FALSE(a.has(StreamRole::SENDER_PKTS_ACKED, 1, 0));
}

TEST(StreamAssignmentTest, ExpressFullWithVc1OnRegistersOverruns) {
    // The maximal express configuration with VC1 on registers needs 33: the counters decision is
    // load-bearing, and the overrun now reads as a legible message instead of a late take_spare.
    EXPECT_ANY_THROW(make_stream_assignment(stream_requirements(express_full_placement(), CreditTransportPlan{})));
}

TEST(StreamAssignmentTest, BoundaryOnRegistersStatesFullNeedAndHitsTheBudgetWall) {
    // Legacy single-TXQ boundary fabric (5,4), both VCs on registers: with the cap removed the
    // need states the family's full width -- the fourth VC1 sender's completion register included
    // -- and the budget wall refuses it: 2 receivers + 5 acked + 9 completed + 8 downstream + 9
    // sender-free = 33, over the 30 below the pinned pair (and over 32 even with the pinned pair
    // freed). A fabric this wide on register-based credits is unservable; the fatal is the
    // protection, and it is what any credit-plan change toward registers hits.
    const auto need = stream_requirements(legacy_with_boundary_placement(), CreditTransportPlan{});
    uint32_t vc1_completed = 0;
    for (const auto& g : need.groups) {
        if (g.role == StreamRole::SENDER_PKTS_COMPLETED && g.vc == 1) {
            vc1_completed = g.count;
        }
    }
    EXPECT_EQ(vc1_completed, 4u);
    EXPECT_ANY_THROW(make_stream_assignment(need));
}

TEST(StreamAssignmentTest, NinthCompletedNameCoversFlat8WhenAVc1GroupIsAllocated) {
    // The completed table's declared extent now covers the full family position range (5+4 ->
    // positions 0..8): flat 8 -- the boundary family's fourth VC1 sender -- gets a real register
    // wherever a VC1 completed group is allocated and the budget fits. This plan (VC0 on counters,
    // VC1 on registers) is host-only: production derives vc0-on-counters == multi-TXQ ==
    // vc1-on-counters. The case pins table coverage, not a configuration any fabric runs.
    CreditTransportPlan plan{};
    plan.vc0_uses_counters = true;
    const auto a = make_stream_assignment(stream_requirements(legacy_with_boundary_placement(), plan));
    EXPECT_TRUE(a.has(StreamRole::SENDER_PKTS_COMPLETED, 1, 3));
    EXPECT_EQ(a.id(StreamRole::SENDER_PKTS_COMPLETED, 1, 3), 10u);
    std::map<std::string, uint32_t> values;
    for (const auto& [name, value] : a.named_args()) {
        values.emplace(name, value);
    }
    EXPECT_EQ(values["TO_SENDER_8_PKTS_COMPLETED_ID"], 10u);
}

TEST(StreamAssignmentTest, ExpressVc1AbsentFits) {
    CreditTransportPlan plan{};
    plan.vc1_uses_counters = true;

    // The configuration the old delta encoding could not serve: VC1 on counters frees nothing
    // (zero VC1 senders), so the five-wide VC0's fifth ack had no register to take. Need-driven:
    // 1 receiver (VC1 absent takes none) + 5 acked + 5 completed + 4 downstream + 5 sender-free = 20.
    const auto a = make_stream_assignment(stream_requirements(express_vc1_absent_placement(), plan));
    EXPECT_EQ(a.id(StreamRole::SENDER_PKTS_ACKED, 0, 4), 10u);  // the fifth ack exists
    EXPECT_EQ(a.id(StreamRole::SENDER_PKTS_COMPLETED, 0, 4), 15u);
    EXPECT_EQ(a.id(StreamRole::SENDER_FREE_SLOTS, 0, 4), 4u);
    // Nothing VC1 anywhere: no receiver, no senders, no downstream span.
    EXPECT_FALSE(a.has(StreamRole::RECEIVER_PKTS_SENT, 1, 0));
    EXPECT_FALSE(a.has(StreamRole::DOWNSTREAM_FREE_SLOTS, 1, 0));
}

TEST(StreamAssignmentTest, Vc0OnCountersNeedsNoAcks) {
    CreditTransportPlan plan{};
    plan.vc0_uses_counters = true;
    const auto a = make_stream_assignment(stream_requirements(legacy_2d_placement(), plan));
    EXPECT_FALSE(a.has(StreamRole::SENDER_PKTS_ACKED, 0, 0));
    EXPECT_FALSE(a.has(StreamRole::SENDER_PKTS_COMPLETED, 0, 0));
    EXPECT_TRUE(a.has(StreamRole::SENDER_PKTS_COMPLETED, 1, 0));
}

TEST(StreamAssignmentTest, PinnedIdThirtyHasAtMostOneLiveConsumer) {
    CreditTransportPlan plan{};
    plan.vc1_uses_counters = true;
    auto need = stream_requirements(express_full_placement(), plan);  // vc2_present == true
    need.tensix_relay_present = true;
    EXPECT_ANY_THROW(make_stream_assignment(need));
}

TEST(StreamAssignmentTest, EmittedNameSetIsExactlyTheKernelSet) {
    const auto a = make_stream_assignment(stream_requirements(legacy_2d_placement(), CreditTransportPlan{}));
    const auto names = emitted_names(a);
    // 2 receiver + 5 acked + 9 completed + 8 downstream + 10 sender-free + 2 pinned + 2 scratch,
    // derived from the declared extents rather than pinned as a count.
    EXPECT_EQ(names.size(), num_stream_register_names);
    EXPECT_TRUE(names.contains("TO_RECEIVER_0_PKTS_SENT_ID"));
    EXPECT_TRUE(names.contains("TO_SENDER_8_PKTS_COMPLETED_ID"));
    EXPECT_TRUE(names.contains("SENDER_CHANNEL_9_FREE_SLOTS_STREAM_ID"));
    EXPECT_TRUE(names.contains("VC2_RECEIVER_FREE_SLOTS_STREAM_ID"));
    EXPECT_TRUE(names.contains("TENSIX_RELAY_LOCAL_FREE_SLOTS_STREAM_ID"));
    EXPECT_TRUE(names.contains("MULTI_RISC_TEARDOWN_SYNC_STREAM_ID"));
    EXPECT_TRUE(names.contains("ETH_RETRAIN_LINK_SYNC_STREAM_ID"));
}

TEST(StreamAssignmentTest, BoundaryChipAgreementOnTheBoundaryOnlyChannel) {
    // The defect's own case: a legacy fabric with a boundary chip has a five-wide VC0 at the
    // fabric max, so the free-slots group covers channels 0-4 at ids 0-4 -- and a mesh router's
    // table position 4 holds the boundary router's channel-4 register, so a downstream lookup
    // resolves it identically on both. Placement follows the fabric max, not the narrower router.
    //
    // Note the plan (VC0 on counters, VC1 on registers) is host-only: production derives
    // vc0-on-counters == multi-TXQ == vc1-on-counters. It is used here to keep a VC1 completed
    // group in play for the placement pins.
    CreditTransportPlan plan{};
    plan.vc0_uses_counters = true;  // VC0 credits on counters, VC1 left on registers
    const auto a = make_stream_assignment(stream_requirements(legacy_with_boundary_placement(), plan));
    EXPECT_EQ(a.id(StreamRole::SENDER_FREE_SLOTS, 0, 4), 4u);
    EXPECT_EQ(a.id(StreamRole::SENDER_FREE_SLOTS, 0, 3), 3u);
    // The flat base of VC1 follows the fabric max (5), not a narrower router's own count (4).
    EXPECT_EQ(a.sender_flat_base(1), 5u);
    EXPECT_EQ(a.id(StreamRole::SENDER_FREE_SLOTS, 1, 0), 19u);
}

TEST(StreamAssignmentTest, PlacementIsFabricScopedAndDeterministic) {
    // The defect was a per-router divergence; now one fabric has one assignment. Two independent
    // derivations of the same fabric produce byte-identical placements.
    CreditTransportPlan plan{};
    plan.vc0_uses_counters = true;
    const auto a = make_stream_assignment(stream_requirements(legacy_with_boundary_placement(), plan));
    const auto b = make_stream_assignment(stream_requirements(legacy_with_boundary_placement(), plan));
    for (uint32_t ch = 0; ch < 5; ++ch) {
        EXPECT_EQ(a.id(StreamRole::SENDER_FREE_SLOTS, 0, ch), b.id(StreamRole::SENDER_FREE_SLOTS, 0, ch));
    }
    EXPECT_EQ(a.id(StreamRole::SENDER_FREE_SLOTS, 1, 0), b.id(StreamRole::SENDER_FREE_SLOTS, 1, 0));
}

TEST(StreamAssignmentTest, FreeSlotsEmitPlacementWhileAckedEmitsActivity) {
    // A router with 4 VC0 senders in a 5-wide fabric: the free-slots table still carries a real id
    // at index 4 (placement is fabric-scoped, so a downstream lookup resolves it), while the acked
    // table carries the sentinel there (activity is a transport-plan question, and VC0 is on
    // counters here).
    CreditTransportPlan plan{};
    plan.vc0_uses_counters = true;
    const auto a = make_stream_assignment(stream_requirements(legacy_with_boundary_placement(), plan));
    std::map<std::string, uint32_t> values;
    for (const auto& [name, value] : a.named_args()) {
        values.emplace(name, value);
    }
    EXPECT_EQ(values["SENDER_CHANNEL_4_FREE_SLOTS_STREAM_ID"], 4u);      // real id: placement, not activity
    EXPECT_EQ(values["TO_SENDER_4_PKTS_ACKED_ID"], k_unused_stream_id);  // sentinel: activity
}

TEST(StreamAssignmentTest, InactiveConsumersEmitTheSentinel) {
    CreditTransportPlan plan{};
    plan.vc1_uses_counters = true;
    const auto a = make_stream_assignment(stream_requirements(express_vc1_absent_placement(), plan));
    std::map<std::string, uint32_t> values;
    for (const auto& [name, value] : a.named_args()) {
        values.emplace(name, value);
    }
    EXPECT_EQ(values["TO_SENDER_4_PKTS_ACKED_ID"], 10u);  // the formerly missing fifth ack
    // The worker-facing pin, agreeing with the header worker-space and ControlPlane read:
    EXPECT_EQ(values["SENDER_CHANNEL_0_FREE_SLOTS_STREAM_ID"], 0u);
    // Everything inactive emits the out-of-range sentinel, so no real register is ever clobbered
    // by an absent consumer.
    EXPECT_EQ(values["TO_RECEIVER_1_PKTS_SENT_ID"], k_unused_stream_id);
    EXPECT_EQ(values["VC1_FREE_SLOTS_FROM_DOWNSTREAM_EDGE_1_STREAM_ID"], k_unused_stream_id);
    EXPECT_EQ(values["SENDER_CHANNEL_5_FREE_SLOTS_STREAM_ID"], k_unused_stream_id);
    EXPECT_EQ(values["VC2_RECEIVER_FREE_SLOTS_STREAM_ID"], k_unused_stream_id);
    EXPECT_EQ(values["TENSIX_RELAY_LOCAL_FREE_SLOTS_STREAM_ID"], k_unused_stream_id);
    // Scratch always emits the pinned ids.
    EXPECT_EQ(values["ETH_RETRAIN_LINK_SYNC_STREAM_ID"], 30u);
    EXPECT_EQ(values["MULTI_RISC_TEARDOWN_SYNC_STREAM_ID"], 31u);
}

}  // namespace
}  // namespace tt::tt_fabric
