// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include <algorithm>
#include <atomic>
#include <filesystem>
#include <fstream>
#include <set>
#include <string>
#include <vector>

#include <cabling_matcher/cabling_matcher.hpp>

namespace tt::scaleout_tools::matcher {
namespace {

constexpr std::string_view kDescriptorDir = "tools/tests/scaleout/cabling_descriptors/";
constexpr std::string_view kT3k = "t3k.textproto";
constexpr std::string_view kDualT3k = "dual_t3k.textproto";
constexpr std::string_view k5Lb = "5_n300_lb_superpod.textproto";
constexpr std::string_view kBhGalaxyMesh = "bh_galaxy_mesh.textproto";

std::string fixture(std::string_view name) { return std::string(kDescriptorDir) + std::string(name); }

MatchGraph load(std::string_view name, std::string template_name = "", TierScope tier = TierScope::Full) {
    return MatchGraph::load(fixture(name), "", template_name, tier, std::string(name));
}

// Writes a descriptor to a scratch file so a test can state exactly the cabling it is about.
std::string write_descriptor(const std::string& name, const std::string& contents) {
    static std::atomic<int> sequence{0};
    auto dir = std::filesystem::temp_directory_path() / ("cabling_matcher_" + std::to_string(sequence++));
    std::filesystem::remove_all(dir);
    std::filesystem::create_directories(dir);
    auto path = dir / (name + ".textproto");
    std::ofstream(path) << contents;
    return path.string();
}

// Two BH galaxy nodes with a single cable between them, on the given ports of tray 1.
std::string two_node_descriptor(uint32_t port_a, uint32_t port_b) {
    return R"(
graph_templates {
  key: "pair"
  value {
    children { name: "node1" node_ref { node_descriptor: "BH_GALAXY_REV_C" } }
    children { name: "node2" node_ref { node_descriptor: "BH_GALAXY_REV_C" } }
    internal_connections {
      key: "QSFP_DD"
      value {
        connections {
          port_a { path: ["node1"] tray_id: 1 port_id: )" +
           std::to_string(port_a) + R"( }
          port_b { path: ["node2"] tray_id: 1 port_id: )" +
           std::to_string(port_b) + R"( }
        }
      }
    }
  }
}
root_instance {
  template_name: "pair"
  child_mappings { key: "node1" value { host_id: 0 } }
  child_mappings { key: "node2" value { host_id: 1 } }
}
)";
}

// Four BH galaxy nodes cabled as two separate pairs, so nothing ties the halves together.
std::string two_pairs_descriptor() {
    return R"(
graph_templates {
  key: "two_pairs"
  value {
    children { name: "node1" node_ref { node_descriptor: "BH_GALAXY_REV_C" } }
    children { name: "node2" node_ref { node_descriptor: "BH_GALAXY_REV_C" } }
    children { name: "node3" node_ref { node_descriptor: "BH_GALAXY_REV_C" } }
    children { name: "node4" node_ref { node_descriptor: "BH_GALAXY_REV_C" } }
    internal_connections {
      key: "QSFP_DD"
      value {
        connections {
          port_a { path: ["node1"] tray_id: 1 port_id: 1 }
          port_b { path: ["node2"] tray_id: 1 port_id: 1 }
        }
        connections {
          port_a { path: ["node3"] tray_id: 1 port_id: 1 }
          port_b { path: ["node4"] tray_id: 1 port_id: 1 }
        }
      }
    }
  }
}
root_instance {
  template_name: "two_pairs"
  child_mappings { key: "node1" value { host_id: 0 } }
  child_mappings { key: "node2" value { host_id: 1 } }
  child_mappings { key: "node3" value { host_id: 2 } }
  child_mappings { key: "node4" value { host_id: 3 } }
}
)";
}

// Cables reduced to the ports they join, which is all that matching looks at. Where a cable was
// declared is deliberately left out: the same cabling reached through a synthesised root is
// declared at a different path.
std::vector<std::pair<CableEndpoint, CableEndpoint>> endpoints(const MatchGraph& graph) {
    std::vector<std::pair<CableEndpoint, CableEndpoint>> result;
    for (const auto& cable : graph.cables()) {
        result.emplace_back(cable.endpoint_a, cable.endpoint_b);
    }
    std::sort(result.begin(), result.end());
    return result;
}

std::vector<std::vector<uint32_t>> host_sets(const MatchResult& result) {
    std::vector<std::vector<uint32_t>> sets;
    for (const auto& component : result.components) {
        for (const auto& match : component.matches) {
            std::vector<uint32_t> hosts;
            for (uint32_t pattern_host : component.pattern_hosts) {
                hosts.push_back(match.host_map[pattern_host]);
            }
            std::sort(hosts.begin(), hosts.end());
            sets.push_back(std::move(hosts));
        }
    }
    std::sort(sets.begin(), sets.end());
    return sets;
}

MatchOptions strict() { return MatchOptions{.max_matches = 0}; }

// ---- Loading and pattern synthesis ----

TEST(CablingMatcher, ListsGraphTemplates) {
    EXPECT_EQ(list_graph_templates(fixture(kDualT3k)), std::vector<std::string>{"dual_t3k"});
}

TEST(CablingMatcher, TemplateAsPatternMatchesTheRootInstanceItDefines) {
    // dual_t3k's root instance is an instantiation of the dual_t3k template, so synthesising a root
    // for that template has to reproduce the same graph.
    MatchGraph from_root = load(kDualT3k);
    MatchGraph from_template = load(kDualT3k, "dual_t3k");
    EXPECT_EQ(from_template.hosts().size(), from_root.hosts().size());
    EXPECT_EQ(endpoints(from_template), endpoints(from_root));
}

TEST(CablingMatcher, UnknownTemplateNameIsRejectedWithTheAvailableOnes) {
    EXPECT_THROW(load(kDualT3k, "no_such_template"), std::exception);
}

TEST(CablingMatcher, OwnLevelTierKeepsOnlyTheCablesTheTemplateDeclares) {
    // dual_t3k declares 4 cables of its own and inherits the rest from the two nodes it contains.
    MatchGraph own_level = load(kDualT3k, "dual_t3k", TierScope::OwnLevel);
    MatchGraph full = load(kDualT3k, "dual_t3k", TierScope::Full);
    EXPECT_EQ(own_level.cables().size(), 4u);
    EXPECT_GT(full.cables().size(), own_level.cables().size());
    for (const auto& cable : own_level.cables()) {
        EXPECT_NE(cable.endpoint_a.host_id, cable.endpoint_b.host_id)
            << "a cable declared by dual_t3k itself has to cross between its two nodes";
    }
}

// ---- Matching ----

TEST(CablingMatcher, EveryDescriptorMatchesItselfExactlyAsTheIdentity) {
    for (std::string_view name : {kT3k, kDualT3k, k5Lb, kBhGalaxyMesh}) {
        MatchGraph graph = load(name);
        MatchOptions options = strict();
        options.mode = MatchMode::Exact;
        MatchResult result = match(graph, graph, options);
        ASSERT_TRUE(result.matched) << name;
        ASSERT_EQ(result.components.size(), 1u) << name;
        ASSERT_EQ(result.components[0].matches.size(), 1u) << name;
        const auto& host_map = result.components[0].matches[0].host_map;
        for (uint32_t host = 0; host < graph.hosts().size(); ++host) {
            EXPECT_EQ(host_map[host], host) << name << " host " << host;
        }
    }
}

TEST(CablingMatcher, RoleAssignmentsCountTheSymmetriesOfThePattern) {
    MatchOptions options = strict();
    options.mode = MatchMode::Exact;

    // dual_t3k joins its two nodes on the same tray and port at both ends, so exchanging the nodes
    // maps every cable onto a cable and the pattern fits its own cluster two ways.
    MatchGraph dual_t3k = load(kDualT3k);
    MatchResult symmetric = match(dual_t3k, dual_t3k, options);
    ASSERT_TRUE(symmetric.matched);
    ASSERT_EQ(symmetric.components[0].matches.size(), 1u) << "both ways use the same pair of hosts";
    EXPECT_EQ(symmetric.components[0].matches[0].role_assignments, 2u);

    // The 5lb star reaches each of its four outer nodes from a different port of the centre, so no
    // exchange of nodes preserves the cabling and the only way it fits is the one.
    MatchGraph star = load(k5Lb);
    MatchResult rigid = match(star, star, options);
    ASSERT_TRUE(rigid.matched);
    ASSERT_EQ(rigid.components[0].matches.size(), 1u);
    EXPECT_EQ(rigid.components[0].matches[0].role_assignments, 1u);
}

TEST(CablingMatcher, OneNodeSitsInEitherHalfOfATwoNodeCluster) {
    MatchResult result = match(load(kT3k), load(kDualT3k), strict());
    ASSERT_TRUE(result.matched);
    // A single node has no cable to any other, so it is placed by its boards alone and fits either
    // half. Both placements are reported, and neither is preferred.
    EXPECT_EQ(host_sets(result), (std::vector<std::vector<uint32_t>>{{0}, {1}}));
}

TEST(CablingMatcher, TwoNodesDoNotFitInOne) {
    MatchResult result = match(load(kDualT3k), load(kT3k), strict());
    EXPECT_FALSE(result.matched);
    EXPECT_FALSE(result.inconclusive());
}

TEST(CablingMatcher, HostsWithDifferentBoardsNeverStandInForEachOther) {
    // An N300 loopback node and a Blackhole galaxy have nothing in common to match on.
    EXPECT_FALSE(match(load(kT3k), load(kBhGalaxyMesh), strict()).matched);
    EXPECT_FALSE(match(load(kBhGalaxyMesh), load(kT3k), strict()).matched);
}

TEST(CablingMatcher, ExactMatchRejectsAPatternSmallerThanTheTarget) {
    MatchOptions options = strict();
    options.mode = MatchMode::Exact;
    MatchResult result = match(load(kT3k), load(kDualT3k), options);
    EXPECT_FALSE(result.matched);
    EXPECT_FALSE(result.exact_mismatch.empty()) << "the host counts alone rule this out, before any search";
}

// ---- Port identity modes ----

TEST(CablingMatcher, PortsOfABlackholeGalaxyReachTheAsicsTheBoardSays) {
    // The premise of the chip identity mode: some ports reach one ASIC, others either of two, and
    // that is what makes a cable on one port able to stand in for a cable on another.
    const auto& port_1 = asics_for_port(BoardType::UBB_BLACKHOLE, PortType::QSFP_DD, PortId(1));
    const auto& port_8 = asics_for_port(BoardType::UBB_BLACKHOLE, PortType::QSFP_DD, PortId(8));
    EXPECT_EQ(port_1.size(), 1u);
    EXPECT_EQ(port_8.size(), 2u);
    EXPECT_TRUE(std::includes(port_8.begin(), port_8.end(), port_1.begin(), port_1.end()))
        << "port 8 spans the ASIC port 1 reaches, so a port 1 cable can be carried by port 8";
}

TEST(CablingMatcher, ChipIdentityAcceptsADifferentPortReachingTheSameAsic) {
    // Port 1 reaches a single ASIC; port 8 reaches that one or its neighbour. A cable the pattern
    // puts on port 1 is therefore satisfied by a target cable on port 8, but only once port ids stop
    // being identities.
    MatchGraph pattern =
        MatchGraph::load(write_descriptor("pattern", two_node_descriptor(1, 1)), "", "", TierScope::Full, "pattern");
    MatchGraph target =
        MatchGraph::load(write_descriptor("target", two_node_descriptor(8, 8)), "", "", TierScope::Full, "target");

    EXPECT_FALSE(match(pattern, target, strict()).matched);

    MatchOptions chip = strict();
    chip.port_identity = PortIdentity::Chip;
    EXPECT_TRUE(match(pattern, target, chip).matched);
}

TEST(CablingMatcher, ChipIdentityStillRefusesPortsOnDifferentAsics) {
    // Port 1 and port 4 reach different ASICs, so no relaxation short of ignoring ports entirely
    // lets one carry the other.
    MatchGraph pattern =
        MatchGraph::load(write_descriptor("pattern", two_node_descriptor(1, 1)), "", "", TierScope::Full, "pattern");
    MatchGraph target =
        MatchGraph::load(write_descriptor("target", two_node_descriptor(4, 4)), "", "", TierScope::Full, "target");

    ASSERT_TRUE(
        asics_for_port(BoardType::UBB_BLACKHOLE, PortType::QSFP_DD, PortId(1)) !=
        asics_for_port(BoardType::UBB_BLACKHOLE, PortType::QSFP_DD, PortId(4)));

    MatchOptions chip = strict();
    chip.port_identity = PortIdentity::Chip;
    EXPECT_FALSE(match(pattern, target, chip).matched);

    MatchOptions relaxed = strict();
    relaxed.port_identity = PortIdentity::Relaxed;
    EXPECT_TRUE(match(pattern, target, relaxed).matched);
}

TEST(CablingMatcher, RelaxingPortIdentityNeverLosesAMatch) {
    // Each mode is a relaxation of the one before it, so a pattern that fits under a stricter reading
    // of ports has to keep fitting under a looser one.
    for (std::string_view pattern_name : {kT3k, kDualT3k, k5Lb}) {
        for (std::string_view target_name : {kT3k, kDualT3k, k5Lb}) {
            MatchGraph pattern = load(pattern_name);
            MatchGraph target = load(target_name);
            MatchOptions options = strict();
            options.max_matches = 1;
            bool strict_match = match(pattern, target, options).matched;
            options.port_identity = PortIdentity::Chip;
            bool chip_match = match(pattern, target, options).matched;
            options.port_identity = PortIdentity::Relaxed;
            bool relaxed_match = match(pattern, target, options).matched;
            EXPECT_TRUE(!strict_match || chip_match) << pattern_name << " in " << target_name;
            EXPECT_TRUE(!chip_match || relaxed_match) << pattern_name << " in " << target_name;
        }
    }
}

// ---- Diagnostics ----

TEST(CablingMatcher, AFailureNamesThePatternCableThatCouldNotBePlaced) {
    // Both clusters wire one cable between their two nodes, but on ports that reach different ASICs,
    // so the very first cable fails and the report has to say so in those terms.
    MatchGraph pattern =
        MatchGraph::load(write_descriptor("pattern", two_node_descriptor(1, 1)), "", "", TierScope::Full, "pattern");
    MatchGraph target =
        MatchGraph::load(write_descriptor("target", two_node_descriptor(4, 4)), "", "", TierScope::Full, "target");

    MatchResult result = match(pattern, target, strict());
    ASSERT_FALSE(result.matched);
    ASSERT_EQ(result.components.size(), 1u);
    const auto& diagnosis = result.components[0].diagnosis;
    ASSERT_TRUE(diagnosis.has_value());

    const auto& stuck = pattern.cables()[diagnosis->pattern_cable];
    EXPECT_NE(stuck.endpoint_a.host_id, stuck.endpoint_b.host_id);
    EXPECT_EQ(*stuck.endpoint_a.port_id, 1u);

    std::string report = format_result(pattern, target, result, strict());
    EXPECT_NE(report.find("NO MATCH"), std::string::npos);
    EXPECT_NE(report.find("Stuck on pattern cable"), std::string::npos);
}

// ---- Disconnected patterns ----

TEST(CablingMatcher, APatternInTwoPiecesIsRefusedUnlessAskedFor) {
    // Nothing ties the pieces to each other, so the matches would be the cross product of their
    // independent placements. The default is to say so rather than produce that.
    std::string path = write_descriptor("two_pairs", two_pairs_descriptor());
    MatchGraph pattern = MatchGraph::load(path, "", "", TierScope::Full, "two_pairs");
    ASSERT_EQ(pattern.components().size(), 2u);

    EXPECT_THROW(match(pattern, pattern, strict()), std::exception);

    MatchOptions options = strict();
    options.allow_disconnected = true;
    MatchResult result = match(pattern, pattern, options);
    ASSERT_TRUE(result.matched);
    ASSERT_EQ(result.components.size(), 2u);
    // Either pair can play either role, so each piece has both pairs as candidates.
    for (const auto& component : result.components) {
        EXPECT_EQ(component.num_host_sets, 2u);
    }
}

// ---- Search shortcuts ----

TEST(CablingMatcher, AbandoningEquivalentPlacementsDoesNotChangeTheAnswer) {
    // The search stops exploring the other ways of dealing out the same cables and trays once one
    // has been recorded. That is only sound if the host sets and the count of role assignments come
    // out the same as they do when every placement is visited.
    for (std::string_view name : {kT3k, kDualT3k, k5Lb}) {
        for (auto identity : {PortIdentity::Strict, PortIdentity::Chip}) {
            MatchGraph graph = load(name);
            MatchOptions pruned = strict();
            pruned.port_identity = identity;
            MatchOptions exhaustive = pruned;
            exhaustive.search_every_placement = true;

            MatchResult from_pruned = match(graph, graph, pruned);
            MatchResult from_exhaustive = match(graph, graph, exhaustive);
            ASSERT_FALSE(from_pruned.inconclusive()) << name;
            ASSERT_FALSE(from_exhaustive.inconclusive()) << name;
            EXPECT_EQ(host_sets(from_pruned), host_sets(from_exhaustive)) << name;
            ASSERT_EQ(from_pruned.components.size(), from_exhaustive.components.size()) << name;
            for (size_t index = 0; index < from_pruned.components.size(); ++index) {
                const auto& pruned_matches = from_pruned.components[index].matches;
                const auto& exhaustive_matches = from_exhaustive.components[index].matches;
                ASSERT_EQ(pruned_matches.size(), exhaustive_matches.size()) << name;
                for (size_t match_index = 0; match_index < pruned_matches.size(); ++match_index) {
                    EXPECT_EQ(
                        pruned_matches[match_index].role_assignments, exhaustive_matches[match_index].role_assignments)
                        << name << " match " << match_index;
                }
            }
        }
    }
}

TEST(CablingMatcher, StoppingAtTheReportingLimitIsReportedAsSuch) {
    MatchOptions options = strict();
    options.max_matches = 1;
    MatchResult result = match(load(kT3k), load(kDualT3k), options);
    ASSERT_TRUE(result.matched);
    ASSERT_EQ(result.components.size(), 1u);
    EXPECT_EQ(result.components[0].num_host_sets, 1u);
    EXPECT_TRUE(result.components[0].stopped_at_limit) << "the second placement exists and was not looked for";
}

}  // namespace
}  // namespace tt::scaleout_tools::matcher
