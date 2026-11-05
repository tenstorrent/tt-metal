// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

/**
 * Channel Trimming Capture -> Replay Scenarios
 *
 * Each scenario captures a traffic pattern on a full fabric, rebuilds fabric from that capture, and checks
 * that the trimmed fabric still carries the same traffic.
 *
 *       src                  transit                            dst
 *    +--------+     +------------------------+            +-------------+
 *    | inject |-...>|  fwd sender ch N    ---|----------->|             |
 *    +--------+     |  worker sender ch 0 ---|----------->|  -> deliver |
 *                   +------------------------+            +-------------+
 *
 * Terminology:
 *   worker sender ch 0    traffic the chip injects itself
 *   fwd sender ch N       traffic the chip forwards for another chip
 *   speedy                a receiver that returns every credit to sender ch 0 only
 *   DA / ack              deadlock avoidance and first-level ack
 *
 * What the scenarios prove, each covering some of:
 *   - dst returns credits to every sender transit uses; dst is only speedy when transit sends on ch 0 alone
 *   - transit and dst run the same DA / ack setting
 *   - on a ring axis, DA / ack is only dropped when neither end of the link forwards
 *   - where it is safe, trimming still applies speedy and drops DA / ack
 *
 * Every scenario also checks that the replayed fabric loaded a non-empty profile.
 *
 * Every scenario also replays all of its captured flows with enough packets to exhaust all of a sender's credits
 * (a credit return is needed for traffic to flow so that a slot can be reused).
 *
 * Each scenario runs in two suites:
 *   ChannelTrimmingScenarioRulesTest   host only: the scenario's link rows through the trimming rules
 *   ChannelTrimmingReplayTest          on device: capture, validate the captured rows, replay the flows
 */

#include <gtest/gtest.h>
#include <cstdint>
#include <filesystem>
#include <map>
#include <optional>
#include <ostream>
#include <string>
#include <utility>
#include <vector>

#include <tt-metalium/host_api.hpp>
#include <tt-metalium/tt_metal.hpp>
#include <tt-metalium/experimental/fabric/fabric.hpp>
#include <tt-metalium/experimental/fabric/control_plane.hpp>
#include "impl/context/metal_context.hpp"
#include "tt_metal/fabric/fabric_builder_context.hpp"
#include "tt_metal/fabric/fabric_context.hpp"
#include "tt_metal/fabric/channel_trimming_import.hpp"
#include "tt_metal/fabric/channel_trimming_io.hpp"
#include "tt_metal/test_utils/env_vars.hpp"

#include "channel_trimming_test_utils.hpp"
#include "fabric_fixture.hpp"

namespace tt::tt_fabric::fabric_router_tests {
namespace {

// ============================================================================
// Scenario description
// ============================================================================
// src -> ... -> transit -> dst is a multi-hop route; beyond is a chip whose route from transit crosses dst next.
enum class Role { Src, Transit, Dst, Beyond };

struct Flow {
    Role from;
    Role to;
};

// Which senders a router used toward its link peer.
struct SenderUse {
    bool worker;     // sender ch 0
    bool forwarded;  // any other sender channel
};

// The traffic to record, and the shape it must leave in the captured rows of the transit <-> dst link.
struct CaptureSpec {
    std::vector<Flow> flows;    // capture order; replay runs the same flows
    SenderUse transit_senders;  // transit's router toward dst
    SenderUse dst_senders;      // dst's router toward transit
    bool dst_forwards_onward;   // dst's receiver passes transit's traffic on toward beyond
};

// What trimming must produce on the transit <-> dst link.
struct ReplaySpec {
    bool transit_speedy;      // transit's router toward dst runs the VC0 speedy path
    bool dst_speedy;          // dst's router toward transit runs the VC0 speedy path
    bool deadlock_avoidance;  // both ends keep DA / ack
};

// A scenario's capture and replay specifications.
struct ChannelTrimmingScenario {
    const char* name;  // gtest name suffix; keeps the scenario number from the scenario table
    tt::tt_fabric::FabricConfig fabric_config;
    bool last_hop_on_ring_axis;  // transit -> dst must run on an axis that keeps DA by default
    CaptureSpec capture;
    ReplaySpec replay;
};

void PrintTo(const ChannelTrimmingScenario& scenario, std::ostream* os) { *os << scenario.name; }

// ============================================================================
// Scenarios on a 2D mesh: no axis keeps DA / ack
// ============================================================================
const std::vector<ChannelTrimmingScenario> kMeshScenarios = {
    // ------------------------------------------------------------------------
    // Scenario 1 | Mesh | transit uses worker sender only | dst receive-only
    //
    //                            transit                            dst
    //                   +------------------------+            +-------------+
    //                   |  worker sender ch 0 ---|----------->|  -> deliver |
    //                   +------------------------+            +-------------+
    //
    //   transit sends on worker sender ch 0 alone, so a speedy receiver at dst still returns every credit
    //   transit needs.
    //
    //   Expect:  both ends speedy; no DA / ack on a mesh
    //   Proves:  trimming still applies speedy where it is safe
    // ------------------------------------------------------------------------
    {
        .name = "01_Mesh_Worker_DstReceiveOnly",
        .fabric_config = tt::tt_fabric::FabricConfig::FABRIC_2D,
        .last_hop_on_ring_axis = false,
        .capture =
            {
                .flows = {{Role::Transit, Role::Dst}},
                .transit_senders = {.worker = true, .forwarded = false},
                .dst_senders = {.worker = false, .forwarded = false},
                .dst_forwards_onward = false,
            },
        .replay = {.transit_speedy = true, .dst_speedy = true, .deadlock_avoidance = false},
    },

    // ------------------------------------------------------------------------
    // Scenario 2 | Mesh | transit uses fwd sender only | dst receive-only
    //
    //       src                  transit                            dst
    //    +--------+     +------------------------+            +-------------+
    //    | inject |-...>|  fwd sender ch N    ---|----------->|  -> deliver |
    //    +--------+     +------------------------+            +-------------+
    //
    //   transit only forwards src's traffic. A speedy receiver at dst would return every credit to sender ch 0,
    //   and fwd sender ch N would run out of slots.
    //
    //   Expect:  neither end speedy; no DA / ack on a mesh
    //   Proves:  dst returns credits to transit's fwd sender ch N
    // ------------------------------------------------------------------------
    {
        .name = "02_Mesh_Fwd_DstReceiveOnly",
        .fabric_config = tt::tt_fabric::FabricConfig::FABRIC_2D,
        .last_hop_on_ring_axis = false,
        .capture =
            {
                .flows = {{Role::Src, Role::Dst}},
                .transit_senders = {.worker = false, .forwarded = true},
                .dst_senders = {.worker = false, .forwarded = false},
                .dst_forwards_onward = false,
            },
        .replay = {.transit_speedy = false, .dst_speedy = false, .deadlock_avoidance = false},
    },

    // ------------------------------------------------------------------------
    // Scenario 3 | Mesh | transit uses worker + fwd sender | dst receive-only
    //
    //       src                  transit                            dst
    //    +--------+     +------------------------+            +-------------+
    //    | inject |-...>|  fwd sender ch N    ---|----------->|             |
    //    +--------+     |  worker sender ch 0 ---|----------->|  -> deliver |
    //                   +------------------------+            +-------------+
    //
    //   dst's router only receives, so judged from its own row it would run the terminal speedy receiver.
    //   transit also sends on fwd sender ch N, which that receiver would never credit.
    //
    //   Expect:  neither end speedy; no DA / ack on a mesh
    //   Proves:  dst returns credits to both of transit's senders (worker ch 0 and fwd ch N)
    // ------------------------------------------------------------------------
    {
        .name = "03_Mesh_WorkerAndFwd_DstReceiveOnly",
        .fabric_config = tt::tt_fabric::FabricConfig::FABRIC_2D,
        .last_hop_on_ring_axis = false,
        .capture =
            {
                .flows = {{Role::Transit, Role::Dst}, {Role::Src, Role::Dst}},
                .transit_senders = {.worker = true, .forwarded = true},
                .dst_senders = {.worker = false, .forwarded = false},
                .dst_forwards_onward = false,
            },
        .replay = {.transit_speedy = false, .dst_speedy = false, .deadlock_avoidance = false},
    },

    // ------------------------------------------------------------------------
    // Scenario 4 | Mesh | transit uses worker + fwd sender | dst also sends back
    //
    //       src                  transit                            dst
    //    +--------+     +------------------------+            +-------------+
    //    | inject |-...>|  fwd sender ch N    ---|----------->|             |
    //    +--------+     |  worker sender ch 0 ---|----------->|  -> deliver |
    //                   |                        |<-----------|  inject     |
    //                   +------------------------+  worker    +-------------+
    //                                               sender ch 0
    //
    //   dst's router sends on worker sender ch 0 only and forwards nothing, so judged from its own row it
    //   would go speedy. It also receives on fwd sender ch N, so it must return credits there too.
    //
    //   Expect:  dst is not speedy; no DA / ack on a mesh
    //   Proves:  dst returns credits to both of transit's senders (worker ch 0 and fwd ch N)
    // ------------------------------------------------------------------------
    {
        .name = "04_Mesh_WorkerAndFwd_DstSendsBack",
        .fabric_config = tt::tt_fabric::FabricConfig::FABRIC_2D,
        .last_hop_on_ring_axis = false,
        .capture =
            {
                .flows = {{Role::Transit, Role::Dst}, {Role::Dst, Role::Transit}, {Role::Src, Role::Dst}},
                .transit_senders = {.worker = true, .forwarded = true},
                .dst_senders = {.worker = true, .forwarded = false},
                .dst_forwards_onward = false,
            },
        .replay = {.transit_speedy = false, .dst_speedy = false, .deadlock_avoidance = false},
    },

    // ------------------------------------------------------------------------
    // Scenario 5 | Mesh | transit uses worker sender only | dst also sends back
    //
    //                            transit                            dst
    //                   +------------------------+            +-------------+
    //                   |  worker sender ch 0 ---|----------->|  -> deliver |
    //                   |  deliver <-            |<-----------|  inject     |
    //                   +------------------------+  worker    +-------------+
    //                                               sender ch 0
    //
    //   Each end sends on worker sender ch 0 alone, so each end's speedy receiver returns every credit the
    //   other needs.
    //
    //   Expect:  both ends speedy; no DA / ack on a mesh
    //   Proves:  trimming still applies speedy in both directions where it is safe
    // ------------------------------------------------------------------------
    {
        .name = "05_Mesh_Worker_DstSendsBack",
        .fabric_config = tt::tt_fabric::FabricConfig::FABRIC_2D,
        .last_hop_on_ring_axis = false,
        .capture =
            {
                .flows = {{Role::Transit, Role::Dst}, {Role::Dst, Role::Transit}},
                .transit_senders = {.worker = true, .forwarded = false},
                .dst_senders = {.worker = true, .forwarded = false},
                .dst_forwards_onward = false,
            },
        .replay = {.transit_speedy = true, .dst_speedy = true, .deadlock_avoidance = false},
    },
};

// ============================================================================
// Scenarios on a torus X ring axis: both ends of the E/W link start with DA / ack on
// ============================================================================
const std::vector<ChannelTrimmingScenario> kTorusXScenarios = {
    // ------------------------------------------------------------------------
    // Scenario 6 | Torus X ring | transit uses worker sender only | dst receive-only
    //
    //                            transit                            dst
    //                   +------------------------+            +-------------+
    //                   |  worker sender ch 0 ---|----------->|  -> deliver |
    //                   +------------------------+            +-------------+
    //
    //   Axis:    transit -> dst runs E/W on a torus X ring, so both ends start with DA / ack on.
    //
    //   Neither end forwards, so traffic on this link cannot be part of a cycle around the ring.
    //
    //   Expect:  DA / ack dropped at both ends; both ends speedy
    //   Proves:  on a ring axis, trimming still drops DA / ack and applies speedy where it is safe
    // ------------------------------------------------------------------------
    {
        .name = "06_TorusX_Worker_DstReceiveOnly",
        .fabric_config = tt::tt_fabric::FabricConfig::FABRIC_2D_TORUS_X,
        .last_hop_on_ring_axis = true,
        .capture =
            {
                .flows = {{Role::Transit, Role::Dst}},
                .transit_senders = {.worker = true, .forwarded = false},
                .dst_senders = {.worker = false, .forwarded = false},
                .dst_forwards_onward = false,
            },
        .replay = {.transit_speedy = true, .dst_speedy = true, .deadlock_avoidance = false},
    },

    // ------------------------------------------------------------------------
    // Scenario 7 | Torus X ring | transit uses fwd sender only | dst receive-only
    //
    //       src                  transit                            dst
    //    +--------+     +------------------------+            +-------------+
    //    | inject |-...>|  fwd sender ch N    ---|----------->|  -> deliver |
    //    +--------+     +------------------------+            +-------------+
    //
    //   Axis:    transit -> dst runs E/W on a torus X ring, so both ends start with DA / ack on.
    //
    //   dst's router only receives, so judged from its own row it would drop DA / ack. transit forwards src's
    //   traffic, keeps DA / ack, and waits for first-level acks from dst.
    //
    //   Expect:  DA / ack on at both ends; neither end speedy
    //   Proves:  transit and dst agree on DA / ack and keep it because transit forwards; dst returns credits
    //            to fwd sender ch N
    // ------------------------------------------------------------------------
    {
        .name = "07_TorusX_Fwd_DstReceiveOnly",
        .fabric_config = tt::tt_fabric::FabricConfig::FABRIC_2D_TORUS_X,
        .last_hop_on_ring_axis = true,
        .capture =
            {
                .flows = {{Role::Src, Role::Dst}},
                .transit_senders = {.worker = false, .forwarded = true},
                .dst_senders = {.worker = false, .forwarded = false},
                .dst_forwards_onward = false,
            },
        .replay = {.transit_speedy = false, .dst_speedy = false, .deadlock_avoidance = true},
    },

    // ------------------------------------------------------------------------
    // Scenario 8 | Torus X ring | transit uses worker + fwd sender | dst receive-only
    //
    //       src                  transit                            dst
    //    +--------+     +------------------------+            +-------------+
    //    | inject |-...>|  fwd sender ch N    ---|----------->|             |
    //    +--------+     |  worker sender ch 0 ---|----------->|  -> deliver |
    //                   +------------------------+            +-------------+
    //
    //   Axis:    transit -> dst runs E/W on a torus X ring, so both ends start with DA / ack on.
    //
    //   dst's router only receives and delivers locally, so judged from its own row it would drop DA / ack.
    //   transit forwards src's traffic, keeps DA / ack, and waits for first-level acks from dst.
    //
    //   Expect:  DA / ack on at both ends; dst is not speedy
    //   Proves:  transit and dst agree on DA / ack, and keep it because transit forwards src's traffic
    // ------------------------------------------------------------------------
    {
        .name = "08_TorusX_WorkerAndFwd_DstReceiveOnly",
        .fabric_config = tt::tt_fabric::FabricConfig::FABRIC_2D_TORUS_X,
        .last_hop_on_ring_axis = true,
        .capture =
            {
                .flows = {{Role::Transit, Role::Dst}, {Role::Src, Role::Dst}},
                .transit_senders = {.worker = true, .forwarded = true},
                .dst_senders = {.worker = false, .forwarded = false},
                .dst_forwards_onward = false,
            },
        .replay = {.transit_speedy = false, .dst_speedy = false, .deadlock_avoidance = true},
    },

    // ------------------------------------------------------------------------
    // Scenario 9 | Torus X ring | transit uses worker + fwd sender | dst also sends back
    //
    //       src                  transit                            dst
    //    +--------+     +------------------------+            +-------------+
    //    | inject |-...>|  fwd sender ch N    ---|----------->|             |
    //    +--------+     |  worker sender ch 0 ---|----------->|  -> deliver |
    //                   |                        |<-----------|  inject     |
    //                   +------------------------+  worker    +-------------+
    //                                               sender ch 0
    //
    //   Axis:    transit -> dst runs E/W on a torus X ring, so both ends start with DA / ack on.
    //
    //   dst's router sends on worker sender ch 0 only and forwards nothing, so judged from its own row it would
    //   go speedy and drop DA / ack. transit forwards, keeps DA / ack, and sends on fwd sender ch N.
    //
    //   Expect:  DA / ack on at both ends; neither end speedy
    //   Proves:  dst returns credits to both of transit's senders, and both ends keep DA / ack, when dst also
    //            sends
    // ------------------------------------------------------------------------
    {
        .name = "09_TorusX_WorkerAndFwd_DstSendsBack",
        .fabric_config = tt::tt_fabric::FabricConfig::FABRIC_2D_TORUS_X,
        .last_hop_on_ring_axis = true,
        .capture =
            {
                .flows = {{Role::Transit, Role::Dst}, {Role::Dst, Role::Transit}, {Role::Src, Role::Dst}},
                .transit_senders = {.worker = true, .forwarded = true},
                .dst_senders = {.worker = true, .forwarded = false},
                .dst_forwards_onward = false,
            },
        .replay = {.transit_speedy = false, .dst_speedy = false, .deadlock_avoidance = true},
    },

    // ------------------------------------------------------------------------
    // Scenario 10 | Torus X ring | transit uses worker sender only | dst also sends back
    //
    //                            transit                            dst
    //                   +------------------------+            +-------------+
    //                   |  worker sender ch 0 ---|----------->|  -> deliver |
    //                   |  deliver <-            |<-----------|  inject     |
    //                   +------------------------+  worker    +-------------+
    //                                               sender ch 0
    //
    //   Axis:    transit -> dst runs E/W on a torus X ring, so both ends start with DA / ack on.
    //
    //   Each end only injects and delivers locally; neither forwards.
    //
    //   Expect:  DA / ack dropped at both ends; both ends speedy
    //   Proves:  on a ring axis, trimming still drops DA / ack and applies speedy in both directions where it
    //            is safe
    // ------------------------------------------------------------------------
    {
        .name = "10_TorusX_Worker_DstSendsBack",
        .fabric_config = tt::tt_fabric::FabricConfig::FABRIC_2D_TORUS_X,
        .last_hop_on_ring_axis = true,
        .capture =
            {
                .flows = {{Role::Transit, Role::Dst}, {Role::Dst, Role::Transit}},
                .transit_senders = {.worker = true, .forwarded = false},
                .dst_senders = {.worker = true, .forwarded = false},
                .dst_forwards_onward = false,
            },
        .replay = {.transit_speedy = true, .dst_speedy = true, .deadlock_avoidance = false},
    },

    // ------------------------------------------------------------------------
    // Scenario 11 | Torus X ring | transit uses worker sender only | dst forwards onward
    //
    //                            transit                            dst                  beyond
    //                   +------------------------+            +---------------+     +-----------+
    //                   |  worker sender ch 0 ---|----------->|  -> deliver   |     |           |
    //                   |                        |            |  fwd sender --|---->| -> deliver |
    //                   +------------------------+            +---------------+     +-----------+
    //
    //   Axis:    transit -> dst runs E/W on a torus X ring, so both ends start with DA / ack on.
    //
    //   transit only injects, so judged from its own row it would drop DA / ack. dst's receiver forwards
    //   transit's traffic on toward beyond, so the link can be part of a cycle around the ring.
    //
    //   Expect:  DA / ack on at both ends; neither end speedy
    //   Proves:  on a ring axis, DA / ack is kept when dst forwards, even though transit only injects
    // ------------------------------------------------------------------------
    {
        .name = "11_TorusX_Worker_DstForwardsOnward",
        .fabric_config = tt::tt_fabric::FabricConfig::FABRIC_2D_TORUS_X,
        .last_hop_on_ring_axis = true,
        .capture =
            {
                .flows = {{Role::Transit, Role::Dst}, {Role::Transit, Role::Beyond}},
                .transit_senders = {.worker = true, .forwarded = false},
                .dst_senders = {.worker = false, .forwarded = false},
                .dst_forwards_onward = true,
            },
        .replay = {.transit_speedy = false, .dst_speedy = false, .deadlock_avoidance = true},
    },
};

std::vector<ChannelTrimmingScenario> all_scenarios() {
    std::vector<ChannelTrimmingScenario> scenarios = kMeshScenarios;
    scenarios.insert(scenarios.end(), kTorusXScenarios.begin(), kTorusXScenarios.end());
    return scenarios;
}

// ============================================================================
// The rows a scenario's capture should leave for the two routers on the transit <-> dst link. Only the
// distinctions the trimming rules make matter: sender ch 0 vs any other sender, and zero vs non-zero.
// ============================================================================
struct LinkRows {
    ChannelTrimmingOverrides transit;  // transit's router toward dst
    ChannelTrimmingOverrides dst;      // dst's router toward transit
};

// Stands in for whichever VC0 forwarded sender channel the router uses.
constexpr uint16_t kForwardedSenderBit = 1u << 3;

ChannelTrimmingOverrides expected_row(SenderUse senders, bool receives, bool forwards_onward) {
    ChannelTrimmingOverrides row{};
    row.sender_channel_used_bitfield_by_vc =
        (senders.worker ? 0x1u : 0x0u) | (senders.forwarded ? kForwardedSenderBit : 0x0u);
    row.receiver_channel_data_forwarded_bitfield_by_vc = receives ? 0x1u : 0x0u;
    row.sender_channel_forwarded_to_bitfield_by_vc[0] = forwards_onward ? kForwardedSenderBit : 0x0u;
    return row;
}

LinkRows expected_link_rows(const CaptureSpec& spec) {
    const bool transit_sends = spec.transit_senders.worker || spec.transit_senders.forwarded;
    const bool dst_sends = spec.dst_senders.worker || spec.dst_senders.forwarded;
    // transit delivers dst's traffic locally in every scenario, so its receiver never forwards onward.
    return LinkRows{
        .transit = expected_row(spec.transit_senders, dst_sends, /*forwards_onward=*/false),
        .dst = expected_row(spec.dst_senders, transit_sends, spec.dst_forwards_onward),
    };
}

// Scenarios only drive VC0 unicast traffic, so any sender bit other than ch 0 is a forwarded sender.
void expect_row_shape(
    const ChannelTrimmingOverrides& captured, const ChannelTrimmingOverrides& expected, const char* router) {
    const uint16_t captured_mask = captured.sender_channel_used_bitfield_by_vc;
    const uint16_t expected_mask = expected.sender_channel_used_bitfield_by_vc;
    EXPECT_EQ((captured_mask & 0x1u) != 0, (expected_mask & 0x1u) != 0)
        << router << ": worker sender ch 0 (captured mask " << captured_mask << ")";
    EXPECT_EQ((captured_mask & ~0x1u) != 0, (expected_mask & ~0x1u) != 0)
        << router << ": forwarded senders (captured mask " << captured_mask << ")";
    EXPECT_EQ(captured.is_receiver_channel_data_forwarded(0), expected.is_receiver_channel_data_forwarded(0))
        << router << ": receiver traffic";
    EXPECT_EQ(
        captured.sender_channel_forwarded_to_bitfield_by_vc[0] != 0,
        expected.sender_channel_forwarded_to_bitfield_by_vc[0] != 0)
        << router << ": receiver forwarding onward";
}

// ============================================================================
// Host only: run each scenario's link rows through the trimming rules the router builder uses.
// ============================================================================
class ChannelTrimmingScenarioRulesTest : public ::testing::TestWithParam<ChannelTrimmingScenario> {};

TEST_P(ChannelTrimmingScenarioRulesTest, ResolvesExpectedLinkSettings) {
    const auto& scenario = GetParam();
    // A 2D router's VC0 senders: the worker channel plus one per forwarding direction.
    constexpr std::size_t kVc0SenderChannels = 4;
    // Both ends of the link run on the same axis; only a torus ring axis starts with DA / ack on.
    const bool base_deadlock_avoidance = scenario.last_hop_on_ring_axis;
    // An intra-mesh 2D link that is neither UDM nor a dispatch link.
    constexpr bool allow_terminal_speedy_rx = true;

    const auto rows = expected_link_rows(scenario.capture);
    const auto transit_row = try_derive_vc0_trim_fast_path_info(rows.transit, kVc0SenderChannels, {});
    const auto dst_row = try_derive_vc0_trim_fast_path_info(rows.dst, kVc0SenderChannels, {});
    ASSERT_TRUE(transit_row.has_value());
    ASSERT_TRUE(dst_row.has_value());

    const auto transit = resolve_vc0_link_trim(
        *transit_row, dst_row, kVc0SenderChannels, base_deadlock_avoidance, allow_terminal_speedy_rx);
    const auto dst = resolve_vc0_link_trim(
        *dst_row, transit_row, kVc0SenderChannels, base_deadlock_avoidance, allow_terminal_speedy_rx);
    // As FabricEriscDatamoverBuilder: DA / ack stay on unless the link dropped them.
    const bool transit_deadlock_avoidance = base_deadlock_avoidance && !transit.info.terminal_or_source_only;
    const bool dst_deadlock_avoidance = base_deadlock_avoidance && !dst.info.terminal_or_source_only;

    EXPECT_EQ(transit.speedy, scenario.replay.transit_speedy) << "transit's router toward dst: speedy";
    EXPECT_EQ(dst.speedy, scenario.replay.dst_speedy) << "dst's router toward transit: speedy";
    EXPECT_EQ(transit_deadlock_avoidance, dst_deadlock_avoidance) << "both ends must agree on DA / ack";
    EXPECT_EQ(dst_deadlock_avoidance, scenario.replay.deadlock_avoidance) << "DA / ack on the link";
}

INSTANTIATE_TEST_SUITE_P(
    ChannelTrimmingScenarios,
    ChannelTrimmingScenarioRulesTest,
    ::testing::ValuesIn(all_scenarios()),
    [](const ::testing::TestParamInfo<ChannelTrimmingScenario>& info) { return std::string(info.param.name); });

// ============================================================================
// Chip selection: the last hop of a src -> ... -> transit -> dst route, on a ring axis when the scenario asks
// for one, plus beyond when the scenario's flows use it. Scenarios without src traffic only use the last hop.
// ============================================================================
bool uses_role(const ChannelTrimmingScenario& scenario, Role role) {
    for (const auto& flow : scenario.capture.flows) {
        if (flow.from == role || flow.to == role) {
            return true;
        }
    }
    return false;
}

// Map scenario nodes to actual chips on hardware.
std::optional<std::map<Role, FabricNodeId>> find_scenario_nodes(const ChannelTrimmingScenario& scenario) {
    const auto& control_plane = tt::tt_metal::MetalContext::instance().get_control_plane();
    const auto& fabric_context = control_plane.get_fabric_context();
    const uint32_t num_chips = control_plane.get_physical_mesh_shape(MeshId{0}).mesh_size();
    const bool needs_beyond = uses_role(scenario, Role::Beyond);

    for (uint32_t src = 0; src < num_chips; ++src) {
        for (uint32_t dst = 0; dst < num_chips; ++dst) {
            if (src == dst) {
                continue;
            }
            const FabricNodeId src_node(MeshId{0}, src);
            const FabricNodeId dst_node(MeshId{0}, dst);
            const auto route = control_plane.get_canonical_intramesh_route(src_node, dst_node);
            if (route.size() < 3 || !control_plane.get_forwarding_direction(src_node, dst_node).has_value()) {
                continue;
            }
            const FabricNodeId transit_node = route[route.size() - 2];
            if (scenario.last_hop_on_ring_axis) {
                const auto last_hop_direction = control_plane.get_forwarding_direction(transit_node, dst_node);
                if (!last_hop_direction.has_value() ||
                    !fabric_context.need_deadlock_avoidance_support(
                        control_plane,
                        transit_node,
                        control_plane.routing_direction_to_eth_direction(*last_hop_direction))) {
                    continue;
                }
            }
            std::map<Role, FabricNodeId> nodes = {
                {Role::Src, src_node}, {Role::Transit, transit_node}, {Role::Dst, dst_node}};
            if (needs_beyond) {
                for (uint32_t beyond = 0; beyond < num_chips && !nodes.contains(Role::Beyond); ++beyond) {
                    const FabricNodeId beyond_node(MeshId{0}, beyond);
                    if (beyond_node == transit_node || beyond_node == dst_node) {
                        continue;
                    }
                    const auto onward = control_plane.get_canonical_intramesh_route(transit_node, beyond_node);
                    if (onward.size() >= 3 && onward[1] == dst_node) {
                        nodes.emplace(Role::Beyond, beyond_node);
                    }
                }
                if (!nodes.contains(Role::Beyond)) {
                    continue;
                }
            }
            return nodes;
        }
    }
    return std::nullopt;
}

// ============================================================================
// Fixture: capture the scenario's flows on a full fabric, then rebuild fabric from that capture.
// ============================================================================
class ChannelTrimmingReplayTest : public BaseFabricFixture,
                                  public ::testing::WithParamInterface<ChannelTrimmingScenario> {
protected:
    static constexpr uint32_t kCapturePackets = 10;
    static constexpr uint32_t kReplayPackets = 256;

    // Fabric comes up per scenario, inside SetUp.
    static void SetUpTestSuite() {}
    static void TearDownTestSuite() {}

    void SetUp() override {
        const auto& scenario = GetParam();
        if (tt::get_arch_from_string(tt::test_utils::get_umd_arch_name()) != tt::ARCH::BLACKHOLE ||
            tt::tt_metal::GetNumAvailableDevices() < 4) {
            GTEST_SKIP() << "Channel trimming replay requires a Blackhole system with at least 4 chips";
        }

        // Capture: full fabric, recording on.
        auto& rtoptions = tt::tt_metal::MetalContext::instance().rtoptions();
        const auto export_path = tt::tt_fabric::get_channel_trimming_capture_path(rtoptions.get_logs_dir());
        std::filesystem::remove(export_path);
        rtoptions.set_enable_channel_trimming_capture(true);
        rtoptions.set_fabric_trimming_profile_path("");
        rtoptions.set_fabric_trimming_override_path("");
        bring_up_fabric(scenario.fabric_config);

        const auto& control_plane = tt::tt_metal::MetalContext::instance().get_control_plane();
        if (control_plane.express_routing_enabled(MeshId{0})) {
            GTEST_SKIP() << "Channel trimming replay requires a non-express mesh";
        }
        auto nodes = find_scenario_nodes(scenario);
        if (!nodes.has_value()) {
            GTEST_SKIP() << "No multi-hop route"
                         << (scenario.last_hop_on_ring_axis ? " whose last hop keeps deadlock avoidance" : "")
                         << (uses_role(scenario, Role::Beyond) ? " with a chip beyond dst" : "");
        }
        nodes_ = std::move(*nodes);

        run_flows(kCapturePackets);
        ASSERT_FALSE(HasFailure()) << "Capture traffic failed on the full fabric";

        // Fabric teardown exports the populated capture while its context and routing map are still active.
        tear_down_fabric();
        rtoptions.set_enable_channel_trimming_capture(false);
        ASSERT_TRUE(std::filesystem::exists(export_path)) << "Capture export not found: " << export_path;

        // Keep a per-scenario copy, so a failing scenario leaves its profile behind.
        profile_path_ = export_path.parent_path() / (std::string(scenario.name) + ".yaml");
        std::filesystem::copy_file(export_path, profile_path_, std::filesystem::copy_options::overwrite_existing);

        // Replay: rebuild fabric from the captured profile.
        rtoptions.set_fabric_trimming_profile_path(profile_path_.string());
        bring_up_fabric(scenario.fabric_config);
    }

    void TearDown() override {
        tear_down_fabric();
        auto& rtoptions = tt::tt_metal::MetalContext::instance().rtoptions();
        rtoptions.set_enable_channel_trimming_capture(false);
        rtoptions.set_fabric_trimming_profile_path("");
        rtoptions.set_fabric_trimming_override_path("");
        if (!HasFailure() && !profile_path_.empty()) {
            std::filesystem::remove(profile_path_);
        }
    }

    // Every captured flow, in capture order.
    void run_flows(uint32_t num_packets) {
        for (const auto& flow : GetParam().capture.flows) {
            const auto result = run_unicast_traffic_bw_nodes(nodes_.at(flow.from), nodes_.at(flow.to), 1, num_packets);
            EXPECT_FALSE(result.skipped) << "No fabric connection from chip " << nodes_.at(flow.from).chip_id
                                         << " to chip " << nodes_.at(flow.to).chip_id;
        }
    }

    // The replayed fabric must actually be trimmed; an empty profile would pass the traffic checks on a
    // full fabric.
    void expect_captured_profile_loaded() {
        const auto& overrides = trimming_overrides();
        ASSERT_TRUE(overrides.has_value());
        EXPECT_GT(overrides->size(), 0u) << "Replay loaded an empty profile from " << profile_path_;
    }

    // Precondition: the capture recorded the shape the scenario describes on the transit <-> dst link.
    // Otherwise replay passes while exercising a different scenario, e.g. when transit's own traffic and
    // src's forwarded traffic leave transit on different links.
    void expect_capture_shape() {
        const auto& spec = GetParam().capture;
        const auto& control_plane = tt::tt_metal::MetalContext::instance().get_control_plane();
        const auto& overrides = trimming_overrides();
        ASSERT_TRUE(overrides.has_value());

        const FabricNodeId transit = nodes_.at(Role::Transit);
        const FabricNodeId dst = nodes_.at(Role::Dst);
        const auto direction = control_plane.get_forwarding_direction(transit, dst);
        ASSERT_TRUE(direction.has_value());
        const auto links = get_forwarding_link_indices(
            *BaseFabricFixture::devices_map_.at(control_plane.get_physical_chip_id_from_fabric_node_id(transit)),
            transit,
            dst);
        ASSERT_FALSE(links.empty());
        const chan_id_t transit_chan =
            control_plane.get_active_fabric_eth_channels_in_direction(transit, *direction)[links[0]];
        const auto peer = control_plane.try_get_connected_mesh_chip_chan_ids(transit, transit_chan);
        ASSERT_TRUE(peer.has_value());
        ASSERT_EQ(peer->first, dst);
        const chan_id_t dst_chan = peer->second;

        const auto transit_row = overrides->find(
            make_override_key(control_plane.get_physical_chip_id_from_fabric_node_id(transit), transit_chan));
        ASSERT_NE(transit_row, overrides->end()) << "No captured row for transit's router toward dst";
        const auto dst_row =
            overrides->find(make_override_key(control_plane.get_physical_chip_id_from_fabric_node_id(dst), dst_chan));
        ASSERT_NE(dst_row, overrides->end()) << "No captured row for dst's router toward transit";

        const auto expected = expected_link_rows(spec);
        expect_row_shape(transit_row->second, expected.transit, "transit's router toward dst");
        expect_row_shape(dst_row->second, expected.dst, "dst's router toward transit");
    }

    std::map<Role, FabricNodeId> nodes_;
    std::filesystem::path profile_path_;

private:
    static const std::optional<ChannelTrimmingOverrideMap>& trimming_overrides() {
        const auto& control_plane = tt::tt_metal::MetalContext::instance().get_control_plane();
        return control_plane.get_fabric_context().get_builder_context().get_channel_trimming_overrides();
    }

    void bring_up_fabric(tt::tt_fabric::FabricConfig fabric_config) {
        BaseFabricFixture::DoSetUpTestSuite(fabric_config);
        fabric_up_ = true;
    }

    void tear_down_fabric() {
        if (fabric_up_) {
            BaseFabricFixture::DoTearDownTestSuite();
            fabric_up_ = false;
        }
    }

    bool fabric_up_ = false;
};

TEST_P(ChannelTrimmingReplayTest, CaptureThenReplay) {
    expect_captured_profile_loaded();
    expect_capture_shape();
    ASSERT_FALSE(HasFailure()) << "Capture did not record the scenario's shape; replay would test something else";
    run_flows(kReplayPackets);
}

INSTANTIATE_TEST_SUITE_P(
    ChannelTrimmingScenarios,
    ChannelTrimmingReplayTest,
    ::testing::ValuesIn(all_scenarios()),
    [](const ::testing::TestParamInfo<ChannelTrimmingScenario>& info) { return std::string(info.param.name); });

}  // namespace
}  // namespace tt::tt_fabric::fabric_router_tests
