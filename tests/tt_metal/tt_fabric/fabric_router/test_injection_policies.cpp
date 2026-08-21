// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// First host coverage for the non-express injection heuristic, extracted as
// NonExpressInjectionPolicy ahead of the unified slot walk. The expected vectors are derived from
// the pre-extraction function by hand, so an extraction that drifts from byte-identical behaviour
// fails here: on Torus the worker and the off-axis "turn" producers are injection channels; every
// other topology guards only the worker; Linear/Mesh and VC1 guard nothing. VC2's worker guard is
// pinned as the visible difference from the express policy's gate (which excludes VC2 entirely).

#include <gtest/gtest.h>

#include <enchantum/enchantum.hpp>
#include <vector>

#include "tt_metal/fabric/builder/fabric_builder_helpers.hpp"
#include "tt_metal/fabric/builder/injection_policy.hpp"

namespace tt::tt_fabric {
namespace {

using eth_chan_directions::EAST;
using eth_chan_directions::NORTH;
using eth_chan_directions::Z;

TEST(InjectionPoliciesTest, NonExpressTorusTurnsAreInjection) {
    const NonExpressInjectionPolicy policy(Topology::Torus, NORTH);

    // VC0, facing N: slots [worker, E, W, S]. The worker is always injection; off-axis producers
    // (E, W) are turn channels; the opposite-Y producer (S) is a straight continuation.
    const builder::RouterProducerSlots slots(NORTH, {4, 4, 0});
    EXPECT_EQ(
        compute_sender_channel_injection_flags(slots, /*vc=*/0, policy), std::vector<bool>({true, true, true, false}));

    // VC0, facing E: slots [worker, W, N, S]. W is the opposite-X continuation; N and S turn.
    const builder::RouterProducerSlots east_slots(EAST, {4, 4, 0});
    const NonExpressInjectionPolicy east_policy(Topology::Torus, EAST);
    EXPECT_EQ(
        compute_sender_channel_injection_flags(east_slots, /*vc=*/0, east_policy),
        std::vector<bool>({true, false, true, true}));
}

TEST(InjectionPoliciesTest, NonExpressZFacingRouterTurnsNothing) {
    // A Z-facing router's producers are all non-turn by the axis rule (preserved behaviour, not a
    // bug): only the worker is injection. Its VC0 count is the boundary family's five.
    const builder::RouterProducerSlots slots(Z, {5, 4, 0});
    const NonExpressInjectionPolicy policy(Topology::Torus, Z);
    EXPECT_EQ(
        compute_sender_channel_injection_flags(slots, /*vc=*/0, policy),
        std::vector<bool>({true, false, false, false, false}));
}

TEST(InjectionPoliciesTest, NonExpressNonTorusGuardsOnlyTheWorker) {
    const builder::RouterProducerSlots slots(NORTH, {4, 4, 0});

    // Ring/NeighborExchange: the worker is guarded, turn channels do not exist.
    const NonExpressInjectionPolicy ring_policy(Topology::Ring, NORTH);
    EXPECT_EQ(
        compute_sender_channel_injection_flags(slots, /*vc=*/0, ring_policy),
        std::vector<bool>({true, false, false, false}));

    // Linear and Mesh: no injection channels at all.
    for (const auto topology : {Topology::Linear, Topology::Mesh}) {
        const NonExpressInjectionPolicy policy(topology, NORTH);
        EXPECT_EQ(
            compute_sender_channel_injection_flags(slots, /*vc=*/0, policy),
            std::vector<bool>({false, false, false, false}))
            << "topology " << enchantum::to_string(topology);
    }
}

TEST(InjectionPoliciesTest, NonExpressVc1CarriesNoGuard) {
    const builder::RouterProducerSlots slots(NORTH, {4, 4, 0});
    const NonExpressInjectionPolicy policy(Topology::Torus, NORTH);
    EXPECT_EQ(
        compute_sender_channel_injection_flags(slots, /*vc=*/1, policy),
        std::vector<bool>({false, false, false, false}));
}

TEST(InjectionPoliciesTest, NonExpressVc2WorkerIsGuarded) {
    // The visible VC2 difference: the non-express gate excludes VC1 but not VC2, so the single
    // worker-type sender on VC2 is an injection channel here; the express policy excludes both.
    const builder::RouterProducerSlots slots(NORTH, {4, 4, 1});
    const NonExpressInjectionPolicy policy(Topology::Torus, NORTH);
    EXPECT_EQ(compute_sender_channel_injection_flags(slots, /*vc=*/2, policy), std::vector<bool>({true}));
}

}  // namespace
}  // namespace tt::tt_fabric
