// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <filesystem>
#include <fstream>
#include <set>
#include <string>
#include <vector>

#include <fmt/format.h>
#include <google/protobuf/text_format.h>

#include <cabling_matcher/cabling_matcher.hpp>

#include "protobuf/factory_system_descriptor.pb.h"

namespace tt::scaleout_tools::matcher {
namespace {

constexpr std::string_view kDescriptorDir = "tools/tests/scaleout/cabling_descriptors/";
constexpr std::string_view kT3k = "t3k.textproto";
constexpr std::string_view kDualT3k = "dual_t3k.textproto";
constexpr std::string_view k5Lb = "5_n300_lb_superpod.textproto";
constexpr std::string_view kBhGalaxyMesh = "bh_galaxy_mesh.textproto";
constexpr std::string_view kWhGalaxyMesh = "wh_galaxy_mesh.textproto";
constexpr std::string_view k16Cluster = "16_n300_lb_cluster.textproto";

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

// One graph template of BH galaxy nodes cabled as given on tray 1. Nodes are numbered from 1, and a
// cable is (node, port, node, port).
using Cable = std::array<uint32_t, 4>;
std::string node_template(const std::string& name, size_t num_nodes, const std::vector<Cable>& cables) {
    std::string out = fmt::format("graph_templates {{\n  key: \"{}\"\n  value {{\n", name);
    for (size_t node = 1; node <= num_nodes; ++node) {
        out += fmt::format(
            "    children {{ name: \"node{}\" node_ref {{ node_descriptor: \"BH_GALAXY_REV_C\" }} }}\n", node);
    }
    out += "    internal_connections {\n      key: \"QSFP_DD\"\n      value {\n";
    for (const auto& [node_a, port_a, node_b, port_b] : cables) {
        out += fmt::format(
            "        connections {{ port_a {{ path: [\"node{}\"] tray_id: 1 port_id: {} }} "
            "port_b {{ path: [\"node{}\"] tray_id: 1 port_id: {} }} }}\n",
            node_a,
            port_a,
            node_b,
            port_b);
    }
    out += "      }\n    }\n  }\n}\n";
    return out;
}

// Instantiates a template's nodes onto hosts 0..n-1, which is what makes a descriptor loadable.
std::string root_instance_for(const std::string& name, size_t num_nodes) {
    std::string out = fmt::format("root_instance {{\n  template_name: \"{}\"\n", name);
    for (size_t node = 1; node <= num_nodes; ++node) {
        out += fmt::format("  child_mappings {{ key: \"node{}\" value {{ host_id: {} }} }}\n", node, node - 1);
    }
    return out + "}\n";
}

// Several descriptors in one directory, which is what a library path stands for.
std::string write_library(const std::vector<std::pair<std::string, std::string>>& descriptors) {
    static std::atomic<int> sequence{0};
    auto dir = std::filesystem::temp_directory_path() / ("cabling_library_" + std::to_string(sequence++));
    std::filesystem::remove_all(dir);
    std::filesystem::create_directories(dir);
    for (const auto& [name, contents] : descriptors) {
        std::ofstream(dir / (name + ".textproto")) << contents;
    }
    return dir.string();
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

// The factory system descriptor a cabling descriptor would be built into, with the given hostnames.
fsd::proto::FactorySystemDescriptor generate_fsd(std::string_view name, const std::vector<std::string>& hostnames) {
    return CablingGenerator(fixture(name), hostnames).generate_factory_system_descriptor();
}

std::vector<std::string> synthetic_hostnames(size_t count) {
    std::vector<std::string> hostnames;
    for (size_t index = 0; index < count; ++index) {
        hostnames.push_back(fmt::format("host_{}", index));
    }
    return hostnames;
}

std::string write_fsd(const fsd::proto::FactorySystemDescriptor& fsd) {
    std::string contents;
    if (!google::protobuf::TextFormat::PrintToString(fsd, &contents)) {
        throw std::runtime_error("failed to serialise the factory system descriptor");
    }
    return write_descriptor("fsd", contents);
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

// ---- Factory system descriptors as input ----

TEST(CablingMatcherFsd, BuiltSystemHasTheSameCablesAsTheDescriptorItWasBuiltFrom) {
    // An FSD records ethernet channels, a cabling descriptor records cables, and the FSD also carries
    // the traces inside each board. Reading one back has to recover the cables exactly: same count,
    // same ports, no traces among them.
    for (std::string_view name : {kT3k, kDualT3k, k5Lb, kBhGalaxyMesh}) {
        MatchGraph from_descriptor = load(name);
        fsd::proto::FactorySystemDescriptor fsd =
            generate_fsd(name, synthetic_hostnames(from_descriptor.hosts().size()));

        // A connection with both ends on one board is wiring inside that board rather than a cable.
        // Asserting these are present is what makes the comparison below evidence that they are
        // dropped, rather than evidence that there were none to drop.
        size_t within_a_board = 0;
        for (const auto& connection : fsd.eth_connections().connection()) {
            within_a_board += connection.endpoint_a().host_id() == connection.endpoint_b().host_id() &&
                                      connection.endpoint_a().tray_id() == connection.endpoint_b().tray_id()
                                  ? 1
                                  : 0;
        }
        EXPECT_GT(within_a_board, 0u) << name;

        MatchGraph from_fsd = MatchGraph::from_fsd(write_fsd(fsd), "fsd");

        EXPECT_EQ(from_fsd.hosts().size(), from_descriptor.hosts().size()) << name;
        EXPECT_EQ(endpoints(from_fsd), endpoints(from_descriptor)) << name;
        EXPECT_TRUE(from_fsd.notes().empty()) << name << ": " << fmt::format("{}", from_fsd.notes().front());
        for (const auto& cable : from_fsd.cables()) {
            EXPECT_NE(cable.endpoint_a.port_type, PortType::TRACE) << name;
            EXPECT_NE(cable.endpoint_b.port_type, PortType::TRACE) << name;
        }
    }
}

TEST(CablingMatcherFsd, ADescriptorMatchesTheSystemBuiltFromItExactly) {
    // The capability this is for: given what was built, does it implement the scheme it was meant to?
    for (std::string_view name : {kT3k, kDualT3k, k5Lb}) {
        MatchGraph pattern = load(name);
        MatchGraph target =
            MatchGraph::from_fsd(write_fsd(generate_fsd(name, synthetic_hostnames(pattern.hosts().size()))), "fsd");

        MatchOptions options = strict();
        options.mode = MatchMode::Exact;
        MatchResult result = match(pattern, target, options);
        ASSERT_TRUE(result.matched) << name;
        ASSERT_EQ(result.components[0].matches.size(), 1u) << name;
        const auto& host_map = result.components[0].matches[0].host_map;
        for (uint32_t host = 0; host < pattern.hosts().size(); ++host) {
            EXPECT_EQ(host_map[host], host) << name << " host " << host;
        }
    }
}

TEST(CablingMatcherFsd, ASchemeIsFoundInABuiltSystemLargerThanIt) {
    // Containment rather than equality: one T3K node is wired inside a dual-T3K system, twice over.
    MatchGraph target = MatchGraph::from_fsd(write_fsd(generate_fsd(kDualT3k, synthetic_hostnames(2))), "dual_t3k fsd");
    MatchResult result = match(load(kT3k), target, strict());
    ASSERT_TRUE(result.matched);
    EXPECT_EQ(host_sets(result), (std::vector<std::vector<uint32_t>>{{0}, {1}}));
}

TEST(CablingMatcherFsd, HostnamesComeFromTheDescriptor) {
    // An FSD names its hosts, unlike a bare cabling descriptor, and those names are what make the
    // report worth reading on a real system.
    std::vector<std::string> hostnames{"rack1-shelf3", "rack1-shelf5"};
    MatchGraph graph = MatchGraph::from_fsd(write_fsd(generate_fsd(kDualT3k, hostnames)), "fsd");
    ASSERT_EQ(graph.hosts().size(), 2u);
    EXPECT_EQ(graph.hosts()[0].name, hostnames[0]);
    EXPECT_EQ(graph.hosts()[1].name, hostnames[1]);
}

TEST(CablingMatcherFsd, ACableMissingAChannelIsStillACableAndIsReported) {
    // A dead lane leaves a cable with fewer channels than the boards call for. Whether that link is
    // usable is not this tool's question, so the cable stands, but the report says what was read.
    fsd::proto::FactorySystemDescriptor fsd = generate_fsd(kDualT3k, synthetic_hostnames(2));
    auto* connections = fsd.mutable_eth_connections()->mutable_connection();
    auto crossing = std::find_if(connections->begin(), connections->end(), [](const auto& connection) {
        return connection.endpoint_a().host_id() != connection.endpoint_b().host_id();
    });
    ASSERT_NE(crossing, connections->end()) << "dual_t3k joins its two hosts, so some channel must cross";
    connections->erase(crossing);

    MatchGraph graph = MatchGraph::from_fsd(write_fsd(fsd), "fsd");
    MatchGraph whole = load(kDualT3k);
    EXPECT_EQ(graph.cables().size(), whole.cables().size()) << "the cable is short a channel, not absent";
    ASSERT_EQ(graph.notes().size(), 1u);
    EXPECT_NE(graph.notes()[0].find("missing channels"), std::string::npos) << graph.notes()[0];
    EXPECT_NE(graph.notes()[0].find("1 of 2 channels"), std::string::npos) << graph.notes()[0];

    // And the scheme is still recognised in it, which is the point of not dropping the cable.
    MatchOptions options = strict();
    options.mode = MatchMode::Exact;
    MatchResult result = match(whole, graph, options);
    EXPECT_TRUE(result.matched);
    EXPECT_NE(format_result(whole, graph, result, options).find("Note (Target)"), std::string::npos)
        << "a match reached by reading the input this way has to say so";
}

TEST(CablingMatcherFsd, AChannelListedTwiceIsReported) {
    // Nothing about matching goes wrong, but a descriptor that counts a channel twice is not one to
    // trust silently.
    fsd::proto::FactorySystemDescriptor fsd = generate_fsd(kDualT3k, synthetic_hostnames(2));
    const auto& connections = fsd.eth_connections().connection();
    auto crossing = std::find_if(connections.begin(), connections.end(), [](const auto& connection) {
        return connection.endpoint_a().host_id() != connection.endpoint_b().host_id();
    });
    ASSERT_NE(crossing, connections.end());
    *fsd.mutable_eth_connections()->add_connection() = *crossing;

    MatchGraph graph = MatchGraph::from_fsd(write_fsd(fsd), "fsd");
    EXPECT_EQ(graph.cables().size(), load(kDualT3k).cables().size());
    ASSERT_EQ(graph.notes().size(), 1u);
    EXPECT_NE(graph.notes()[0].find("lists some twice"), std::string::npos) << graph.notes()[0];
}

TEST(CablingMatcherFsd, AConnectionOnATrayWithNoBoardIsRefused) {
    // Without a board type there is no channel-to-port map, so the connection cannot be read at all
    // and guessing would be worse than stopping.
    fsd::proto::FactorySystemDescriptor fsd = generate_fsd(kT3k, synthetic_hostnames(1));
    ASSERT_GT(fsd.eth_connections().connection_size(), 0);
    fsd.mutable_eth_connections()->mutable_connection(0)->mutable_endpoint_a()->set_tray_id(99);

    EXPECT_THROW(MatchGraph::from_fsd(write_fsd(fsd), "fsd"), std::exception);
}

TEST(CablingMatcherFsd, ABoardTypeSpelledByItsOtherEnumNameIsAccepted) {
    // The board type enum spells a wormhole galaxy tray both UBB and UBB_WORMHOLE. Reflection
    // surfaces only the first, and descriptors written elsewhere use the second, so reading one
    // through reflection alone rejects a system it otherwise understands completely.
    ASSERT_EQ(get_board_type_from_string("UBB_WORMHOLE"), get_board_type_from_string("UBB"));

    fsd::proto::FactorySystemDescriptor fsd = generate_fsd(kWhGalaxyMesh, synthetic_hostnames(1));
    size_t respelled = 0;
    for (auto& location : *fsd.mutable_board_types()->mutable_board_locations()) {
        if (get_board_type_from_string(location.board_type()) == BoardType::UBB_WORMHOLE) {
            location.set_board_type("UBB_WORMHOLE");
            ++respelled;
        }
    }
    ASSERT_GT(respelled, 0u) << "a wormhole galaxy is built from UBB trays, so there is something to respell";

    EXPECT_EQ(endpoints(MatchGraph::from_fsd(write_fsd(fsd), "fsd")), endpoints(load(kWhGalaxyMesh)));
}

TEST(CablingMatcherFsd, OneBuiltSystemIsComparedAgainstAnother) {
    // Both sides read from FSDs, which asks whether one system's wiring is reproduced in another.
    MatchGraph pattern = MatchGraph::from_fsd(write_fsd(generate_fsd(kT3k, {"node-a"})), "one node");
    MatchGraph target = MatchGraph::from_fsd(write_fsd(generate_fsd(kDualT3k, {"node-b", "node-c"})), "two nodes");
    MatchResult result = match(pattern, target, strict());
    ASSERT_TRUE(result.matched);
    EXPECT_EQ(host_sets(result), (std::vector<std::vector<uint32_t>>{{0}, {1}}));
}

// ---- Scoring a library of schemes ----

TEST(CablingMatcherLibrary, TheSchemesInsideAMatchedOneAreReportedAsCoveredByIt) {
    // The cluster fixture is built from four superpods, so both fit, and saying the superpod fits in
    // four places is only worth reading if it also says all four are inside the cluster.
    std::string path = fixture(k16Cluster);
    LibraryResult result = score_library({path}, load(k16Cluster), strict(), TierScope::Full);

    ASSERT_EQ(result.entries.size(), 2u);
    ASSERT_TRUE(result.skipped.empty()) << result.skipped.front().second;

    EXPECT_EQ(result.entries[0].template_name, "n300_lb_cluster") << "the larger scheme is reported first";
    EXPECT_EQ(result.findings[0].host_sets.size(), 1u);
    EXPECT_TRUE(result.findings[0].covered_by.empty()) << "nothing in the library is bigger than the cluster";

    EXPECT_EQ(result.entries[1].template_name, "n300_lb_superpod");
    EXPECT_EQ(result.findings[1].host_sets.size(), 4u);
    EXPECT_EQ(result.findings[1].covered_by, std::vector<size_t>{0});
}

TEST(CablingMatcherLibrary, ASchemeAlsoFoundOutsideTheLargerOneIsNotCoveredByIt) {
    // The discriminating case for coverage. A three node chain, and a pair that sits inside it but
    // also once more on hosts the chain does not reach. That second placement is exactly what a
    // reader wants to know about, so the pair must be reported in its own right.
    std::string library = write_library(
        {{"schemes",
          node_template("chain", 3, {{1, 1, 2, 1}, {2, 2, 3, 2}}) + node_template("pair", 2, {{1, 1, 2, 1}})}});
    MatchGraph target = MatchGraph::load(
        write_descriptor(
            "target",
            node_template("target", 5, {{1, 1, 2, 1}, {2, 2, 3, 2}, {4, 1, 5, 1}}) + root_instance_for("target", 5)),
        "",
        "",
        TierScope::Full,
        "chain and a spare pair");

    LibraryResult result = score_library({library}, target, strict(), TierScope::Full);
    ASSERT_EQ(result.entries.size(), 2u);
    ASSERT_TRUE(result.skipped.empty()) << result.skipped.front().second;

    EXPECT_EQ(result.entries[0].template_name, "chain");
    EXPECT_EQ(result.findings[0].host_sets, (std::vector<std::vector<uint32_t>>{{0, 1, 2}}));

    EXPECT_EQ(result.entries[1].template_name, "pair");
    EXPECT_EQ(result.findings[1].host_sets, (std::vector<std::vector<uint32_t>>{{0, 1}, {3, 4}}));
    EXPECT_TRUE(result.findings[1].covered_by.empty()) << "one of its two placements is outside the chain";
}

TEST(CablingMatcherLibrary, ASchemeOnlyFoundInsideTheLargerOneIsCoveredByIt) {
    // The same library, with the spare pair taken away: now every placement of the pair is inside
    // the chain, and repeating it as a finding of its own would add nothing.
    std::string library = write_library(
        {{"schemes",
          node_template("chain", 3, {{1, 1, 2, 1}, {2, 2, 3, 2}}) + node_template("pair", 2, {{1, 1, 2, 1}})}});
    MatchGraph target = MatchGraph::load(
        write_descriptor(
            "target", node_template("target", 3, {{1, 1, 2, 1}, {2, 2, 3, 2}}) + root_instance_for("target", 3)),
        "",
        "",
        TierScope::Full,
        "chain alone");

    LibraryResult result = score_library({library}, target, strict(), TierScope::Full);
    ASSERT_EQ(result.entries.size(), 2u);
    EXPECT_EQ(result.findings[1].host_sets, (std::vector<std::vector<uint32_t>>{{0, 1}}));
    EXPECT_EQ(result.findings[1].covered_by, std::vector<size_t>{0});
}

TEST(CablingMatcherLibrary, SameNamedTemplatesInDifferentFilesStayDistinct) {
    // A library is a set of files, each free to name its templates as it likes, and both of these
    // call theirs "pair". They are different schemes and the report has to tell them apart.
    std::string library =
        write_library({{"on_port_1", two_node_descriptor(1, 1)}, {"on_port_4", two_node_descriptor(4, 4)}});
    MatchGraph target =
        MatchGraph::load(write_descriptor("target", two_node_descriptor(1, 1)), "", "", TierScope::Full, "target");

    LibraryResult result = score_library({library}, target, strict(), TierScope::Full);
    ASSERT_EQ(result.entries.size(), 2u);
    EXPECT_NE(result.entries[0].name, result.entries[1].name);

    // Same size, so the order between them is by name, and only the one cabled like the target fits.
    size_t on_port_1 = result.entries[0].name.find("on_port_1") != std::string::npos ? 0 : 1;
    EXPECT_FALSE(result.findings[on_port_1].host_sets.empty());
    EXPECT_TRUE(result.findings[1 - on_port_1].host_sets.empty());
}

TEST(CablingMatcherLibrary, EqualSizedSchemesDoNotCoverEachOther) {
    // Both fit the same two hosts under relaxed ports, but neither contains the other: they are
    // alternative descriptions of that wiring, and calling one covered would hide it.
    std::string library =
        write_library({{"on_port_1", two_node_descriptor(1, 1)}, {"on_port_4", two_node_descriptor(4, 4)}});
    MatchOptions options = strict();
    options.port_identity = PortIdentity::Relaxed;

    LibraryResult result = score_library(
        {library},
        MatchGraph::load(write_descriptor("target", two_node_descriptor(1, 1)), "", "", TierScope::Full, "target"),
        options,
        TierScope::Full);
    ASSERT_EQ(result.entries.size(), 2u);
    ASSERT_FALSE(result.findings[0].host_sets.empty());
    ASSERT_FALSE(result.findings[1].host_sets.empty());
    EXPECT_EQ(result.findings[0].host_sets, result.findings[1].host_sets);
    EXPECT_TRUE(result.findings[1].covered_by.empty());
}

TEST(CablingMatcherLibrary, ASchemeThatDoesNotFitSaysWhereItGotStuck) {
    LibraryResult result = score_library({fixture(kDualT3k)}, load(kT3k), strict(), TierScope::Full);
    ASSERT_EQ(result.entries.size(), 1u);
    EXPECT_TRUE(result.findings[0].host_sets.empty());
    EXPECT_FALSE(result.findings[0].stuck_at.empty());
    EXPECT_FALSE(result.findings[0].inconclusive);

    std::string report = format_library_result(load(kT3k), result, strict());
    EXPECT_NE(report.find("NOTHING IN THE LIBRARY FITS"), std::string::npos);
    EXPECT_NE(report.find("Does not fit"), std::string::npos);
}

TEST(CablingMatcherLibrary, ASchemeWithNoCablesOfItsOwnIsSkippedRatherThanMatched) {
    // A single-node template declares no cables itself, so at own-level scope there is nothing to
    // look for and any host would satisfy it. That is not an answer worth reporting as a match.
    LibraryResult result = score_library({fixture(kWhGalaxyMesh)}, load(kWhGalaxyMesh), strict(), TierScope::OwnLevel);
    EXPECT_TRUE(result.entries.empty());
    ASSERT_FALSE(result.skipped.empty());
    EXPECT_NE(result.skipped.front().second.find("no cables"), std::string::npos) << result.skipped.front().second;
}

TEST(CablingMatcherLibrary, HostsBelongingToNoKnownSchemeAreCalledOut) {
    // Half of a two-superpod target is wired as a known superpod and half is not. A report that only
    // listed what matched would leave the other half unaccounted for.
    LibraryResult result = score_library({fixture(k5Lb)}, load(k16Cluster), strict(), TierScope::Full);
    ASSERT_EQ(result.entries.size(), 1u);
    EXPECT_TRUE(result.findings[0].host_sets.empty()) << "the 5lb star is not how the cluster wires its superpods";

    std::string report = format_library_result(load(k16Cluster), result, strict());
    EXPECT_EQ(report.find("Largest scheme each host belongs to"), std::string::npos)
        << "with nothing matched there is no coverage to summarise";
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
