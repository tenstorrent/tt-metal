// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>
#include <filesystem>
#include <fstream>
#include <google/protobuf/text_format.h>

#include <cabling_generator/cabling_generator.hpp>
#include "protobuf/cluster_config.pb.h"
#include "protobuf/deployment.pb.h"

namespace tt::scaleout_tools {

class HostRemovalTest : public ::testing::Test {
protected:
    static std::string create_test_dir(const std::string& test_name) {
        const std::string dir = "generated/tests/" + test_name + "/";
        if (std::filesystem::exists(dir)) {
            std::filesystem::remove_all(dir);
        }
        std::filesystem::create_directories(dir);
        return dir;
    }

    static void write_textproto(const std::string& path, const std::string& content) {
        std::ofstream ofs(path);
        ofs << content;
    }

    // Builds a flat single-template cabling descriptor where the hostnames are used as both
    // the graph `children.name` and the deployment `host:` field. `connections` is a list of
    // (from_name, to_name, tray_a, port_a, tray_b, port_b) tuples expressed as
    // `internal_connections` of type QSFP_DD.
    struct Conn {
        std::string from;
        std::string to;
        uint32_t tray_a = 1;
        uint32_t port_a = 1;
        uint32_t tray_b = 1;
        uint32_t port_b = 1;
    };
    static void write_flat_cabling(
        const std::string& path,
        const std::string& template_name,
        const std::vector<std::string>& hostnames,
        const std::string& node_descriptor,
        const std::vector<Conn>& connections = {}) {
        std::string content = "graph_templates {\n  key: \"" + template_name + "\"\n  value {\n";
        for (const auto& h : hostnames) {
            content +=
                "    children { name: \"" + h + "\" node_ref { node_descriptor: \"" + node_descriptor + "\" } }\n";
        }
        if (!connections.empty()) {
            content += "    internal_connections {\n      key: \"QSFP_DD\"\n      value {\n";
            for (const auto& c : connections) {
                content += "        connections {\n";
                content += "          port_a { path: [\"" + c.from + "\"] tray_id: " + std::to_string(c.tray_a) +
                           " port_id: " + std::to_string(c.port_a) + " }\n";
                content += "          port_b { path: [\"" + c.to + "\"] tray_id: " + std::to_string(c.tray_b) +
                           " port_id: " + std::to_string(c.port_b) + " }\n";
                content += "        }\n";
            }
            content += "      }\n    }\n";
        }
        content += "  }\n}\n";
        content += "root_instance {\n  template_name: \"" + template_name + "\"\n";
        for (size_t i = 0; i < hostnames.size(); ++i) {
            content +=
                "  child_mappings { key: \"" + hostnames[i] + "\" value { host_id: " + std::to_string(i) + " } }\n";
        }
        content += "}\n";
        write_textproto(path, content);
    }

    static void write_simple_deployment(
        const std::string& path, const std::vector<std::string>& hostnames, const std::string& node_type) {
        std::string content;
        for (const auto& h : hostnames) {
            content += "hosts {\n";
            content += "  host: \"" + h + "\"\n";
            content += "  node_type: \"" + node_type + "\"\n";
            content += "}\n";
        }
        write_textproto(path, content);
    }

    static cabling_generator::proto::ClusterDescriptor load_cluster(const std::string& path) {
        std::ifstream f(path);
        EXPECT_TRUE(f.is_open()) << path;
        std::string content((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
        cabling_generator::proto::ClusterDescriptor desc;
        EXPECT_TRUE(google::protobuf::TextFormat::ParseFromString(content, &desc)) << path;
        return desc;
    }

    static deployment::proto::DeploymentDescriptor load_deployment(const std::string& path) {
        std::ifstream f(path);
        EXPECT_TRUE(f.is_open()) << path;
        std::string content((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
        deployment::proto::DeploymentDescriptor desc;
        EXPECT_TRUE(google::protobuf::TextFormat::ParseFromString(content, &desc)) << path;
        return desc;
    }

    // Writes the generator out to cabling + deployment textprotos in `dir` and returns their paths.
    static std::pair<std::string, std::string> emit_pair(const CablingGenerator& gen, const std::string& dir) {
        const std::string cabling_out = dir + "emitted_cabling.textproto";
        const std::string deployment_out = dir + "emitted_deployment.textproto";
        gen.emit_cabling_descriptor(cabling_out);
        gen.emit_deployment_descriptor(deployment_out);
        return {cabling_out, deployment_out};
    }

    // Collects all host_ids referenced in the root_instance child_mappings (single level only).
    static std::vector<uint32_t> collect_root_host_ids(const cabling_generator::proto::ClusterDescriptor& desc) {
        std::vector<uint32_t> ids;
        for (const auto& [_, mapping] : desc.root_instance().child_mappings()) {
            if (mapping.mapping_case() == cabling_generator::proto::ChildMapping::kHostId) {
                ids.push_back(mapping.host_id());
            }
        }
        std::sort(ids.begin(), ids.end());
        return ids;
    }

    static std::vector<std::string> hostnames_in_deployment(const deployment::proto::DeploymentDescriptor& d) {
        std::vector<std::string> names;
        for (const auto& h : d.hosts()) {
            names.push_back(h.host());
        }
        return names;
    }
};

// Core case: drop a single leaf. Graph shrinks, deployment shrinks, host_ids stay contiguous 0..N-1.
TEST_F(HostRemovalTest, RemoveLeafHostRenumbersRemaining) {
    const std::string dir = create_test_dir("remove_leaf_host");
    const std::vector<std::string> hosts = {"node_a", "node_b", "node_c", "node_d"};
    write_flat_cabling(dir + "cabling.textproto", "cluster", hosts, "WH_GALAXY");
    write_simple_deployment(dir + "deployment.textproto", hosts, "WH_GALAXY");

    CablingGenerator gen(dir + "cabling.textproto", dir + "deployment.textproto");
    gen.remove_host("node_b");

    auto [cabling_out, deployment_out] = emit_pair(gen, dir);
    auto cabling_desc = load_cluster(cabling_out);
    auto deployment_desc = load_deployment(deployment_out);

    // 3 hosts survive in both descriptors; "node_b" is gone; host_ids are contiguous 0..2.
    EXPECT_EQ(cabling_desc.root_instance().child_mappings().size(), 3u);
    EXPECT_FALSE(cabling_desc.root_instance().child_mappings().contains("node_b"));
    EXPECT_EQ(collect_root_host_ids(cabling_desc), (std::vector<uint32_t>{0, 1, 2}));

    const auto names = hostnames_in_deployment(deployment_desc);
    EXPECT_EQ(names.size(), 3u);
    EXPECT_EQ(std::count(names.begin(), names.end(), "node_b"), 0);
}

// Connection touching the removed host must be dropped from the emitted descriptor.
TEST_F(HostRemovalTest, RemoveHostDropsTouchingConnections) {
    const std::string dir = create_test_dir("remove_host_drops_connections");
    const std::vector<std::string> hosts = {"node_a", "node_b"};
    write_flat_cabling(
        dir + "cabling.textproto",
        "cluster",
        hosts,
        "WH_GALAXY",
        {Conn{"node_a", "node_b", 1, 1, 1, 1}});
    write_simple_deployment(dir + "deployment.textproto", hosts, "WH_GALAXY");

    CablingGenerator gen(dir + "cabling.textproto", dir + "deployment.textproto");
    gen.remove_host("node_b");

    auto [cabling_out, _] = emit_pair(gen, dir);
    auto desc = load_cluster(cabling_out);

    // After removal, no internal_connections should reference node_b (or survive at all,
    // since the only connection we declared touches it).
    const auto& templates = desc.graph_templates();
    ASSERT_TRUE(templates.contains("cluster"));
    size_t total_conns = 0;
    for (const auto& [port_type, conns] : templates.at("cluster").internal_connections()) {
        total_conns += conns.connections_size();
        for (const auto& conn : conns.connections()) {
            for (const auto& seg : conn.port_a().path()) {
                EXPECT_NE(seg, "node_b") << "connection still references node_b";
            }
            for (const auto& seg : conn.port_b().path()) {
                EXPECT_NE(seg, "node_b") << "connection still references node_b";
            }
        }
    }
    EXPECT_EQ(total_conns, 0u);
}

// remove_hosts drops several in one call.
TEST_F(HostRemovalTest, RemoveMultipleHostsInOneCall) {
    const std::string dir = create_test_dir("remove_multiple_hosts");
    const std::vector<std::string> hosts = {"node_a", "node_b", "node_c", "node_d"};
    write_flat_cabling(dir + "cabling.textproto", "cluster", hosts, "WH_GALAXY");
    write_simple_deployment(dir + "deployment.textproto", hosts, "WH_GALAXY");

    CablingGenerator gen(dir + "cabling.textproto", dir + "deployment.textproto");
    gen.remove_hosts({"node_b", "node_d"});

    auto [cabling_out, deployment_out] = emit_pair(gen, dir);
    auto cabling_desc = load_cluster(cabling_out);
    auto deployment_desc = load_deployment(deployment_out);

    EXPECT_EQ(cabling_desc.root_instance().child_mappings().size(), 2u);
    EXPECT_FALSE(cabling_desc.root_instance().child_mappings().contains("node_b"));
    EXPECT_FALSE(cabling_desc.root_instance().child_mappings().contains("node_d"));
    EXPECT_EQ(collect_root_host_ids(cabling_desc), (std::vector<uint32_t>{0, 1}));
    EXPECT_EQ(deployment_desc.hosts_size(), 2);
}

// Unknown hostname produces no mutation and no throw.
TEST_F(HostRemovalTest, RemoveUnknownHostIsNoOp) {
    const std::string dir = create_test_dir("remove_unknown_host");
    const std::vector<std::string> hosts = {"node_a", "node_b"};
    write_flat_cabling(dir + "cabling.textproto", "cluster", hosts, "WH_GALAXY");
    write_simple_deployment(dir + "deployment.textproto", hosts, "WH_GALAXY");

    CablingGenerator gen(dir + "cabling.textproto", dir + "deployment.textproto");
    EXPECT_NO_THROW(gen.remove_host("does-not-exist"));

    auto [cabling_out, deployment_out] = emit_pair(gen, dir);
    auto cabling_desc = load_cluster(cabling_out);
    auto deployment_desc = load_deployment(deployment_out);

    EXPECT_EQ(cabling_desc.root_instance().child_mappings().size(), 2u);
    EXPECT_EQ(deployment_desc.hosts_size(), 2);
}

// After removal, emit + reload must produce a valid generator with the reduced cluster.
TEST_F(HostRemovalTest, RemovedClusterRoundTripsThroughEmit) {
    const std::string dir = create_test_dir("remove_round_trip");
    const std::vector<std::string> hosts = {"node_a", "node_b", "node_c"};
    write_flat_cabling(
        dir + "cabling.textproto",
        "cluster",
        hosts,
        "WH_GALAXY",
        {Conn{"node_a", "node_b", 1, 1, 1, 1}, Conn{"node_b", "node_c", 1, 2, 1, 2}});
    write_simple_deployment(dir + "deployment.textproto", hosts, "WH_GALAXY");

    CablingGenerator gen(dir + "cabling.textproto", dir + "deployment.textproto");
    gen.remove_host("node_b");
    auto [cabling_out, deployment_out] = emit_pair(gen, dir);

    // Reload with a fresh generator — must not error and must describe 2 hosts.
    EXPECT_NO_THROW({
        CablingGenerator reloaded(cabling_out, deployment_out);
        EXPECT_EQ(reloaded.get_deployment_hosts().size(), 2u);
    });
}

// Removing the very first host (host_id 0) — the renumbering must shift everyone down.
TEST_F(HostRemovalTest, RemoveFirstHostShiftsRemaining) {
    const std::string dir = create_test_dir("remove_first_host");
    const std::vector<std::string> hosts = {"node_a", "node_b", "node_c"};
    write_flat_cabling(dir + "cabling.textproto", "cluster", hosts, "WH_GALAXY");
    write_simple_deployment(dir + "deployment.textproto", hosts, "WH_GALAXY");

    CablingGenerator gen(dir + "cabling.textproto", dir + "deployment.textproto");
    gen.remove_host("node_a");

    // In-memory state: 2 hosts, contiguous host_ids 0..1.
    EXPECT_EQ(gen.get_deployment_hosts().size(), 2u);

    auto [cabling_out, deployment_out] = emit_pair(gen, dir);
    auto cabling_desc = load_cluster(cabling_out);
    auto deployment_desc = load_deployment(deployment_out);

    EXPECT_EQ(cabling_desc.root_instance().child_mappings().size(), 2u);
    EXPECT_FALSE(cabling_desc.root_instance().child_mappings().contains("node_a"));
    EXPECT_EQ(collect_root_host_ids(cabling_desc), (std::vector<uint32_t>{0, 1}));
    EXPECT_EQ(deployment_desc.hosts_size(), 2);
    EXPECT_EQ(deployment_desc.hosts(0).host(), "node_b");
    EXPECT_EQ(deployment_desc.hosts(1).host(), "node_c");
}

// Removing the last host (highest host_id) — boundary case for renumbering.
TEST_F(HostRemovalTest, RemoveLastHostLeavesContiguousIds) {
    const std::string dir = create_test_dir("remove_last_host");
    const std::vector<std::string> hosts = {"node_a", "node_b", "node_c"};
    write_flat_cabling(dir + "cabling.textproto", "cluster", hosts, "WH_GALAXY");
    write_simple_deployment(dir + "deployment.textproto", hosts, "WH_GALAXY");

    CablingGenerator gen(dir + "cabling.textproto", dir + "deployment.textproto");
    gen.remove_host("node_c");

    auto [cabling_out, deployment_out] = emit_pair(gen, dir);
    auto cabling_desc = load_cluster(cabling_out);
    auto deployment_desc = load_deployment(deployment_out);

    EXPECT_EQ(cabling_desc.root_instance().child_mappings().size(), 2u);
    EXPECT_FALSE(cabling_desc.root_instance().child_mappings().contains("node_c"));
    EXPECT_EQ(collect_root_host_ids(cabling_desc), (std::vector<uint32_t>{0, 1}));
    EXPECT_EQ(deployment_desc.hosts_size(), 2);
}

// Surviving connections must reference the renumbered host_ids of the surviving hosts,
// AND their hostnames in the emitted descriptor's path must match the survivors.
TEST_F(HostRemovalTest, SurvivingConnectionsUseRenumberedIds) {
    const std::string dir = create_test_dir("surviving_connections_renumber");
    const std::vector<std::string> hosts = {"node_a", "node_b", "node_c", "node_d"};
    // Build connections: a-b (drop), b-c (drop), a-c (keep), a-d (keep), c-d (keep).
    write_flat_cabling(
        dir + "cabling.textproto",
        "cluster",
        hosts,
        "WH_GALAXY",
        {
            Conn{"node_a", "node_b", 1, 1, 1, 1},
            Conn{"node_b", "node_c", 1, 2, 1, 2},
            Conn{"node_a", "node_c", 2, 1, 2, 1},
            Conn{"node_a", "node_d", 3, 1, 3, 1},
            Conn{"node_c", "node_d", 4, 1, 4, 1},
        });
    write_simple_deployment(dir + "deployment.textproto", hosts, "WH_GALAXY");

    CablingGenerator gen(dir + "cabling.textproto", dir + "deployment.textproto");
    gen.remove_host("node_b");

    auto [cabling_out, _] = emit_pair(gen, dir);
    auto desc = load_cluster(cabling_out);

    // Surviving connections: a-c, a-d, c-d (three).
    ASSERT_TRUE(desc.graph_templates().contains("cluster"));
    size_t total_conns = 0;
    std::set<std::pair<std::string, std::string>> survivors;
    for (const auto& [port_type, conns] : desc.graph_templates().at("cluster").internal_connections()) {
        for (const auto& c : conns.connections()) {
            ASSERT_EQ(c.port_a().path_size(), 1);
            ASSERT_EQ(c.port_b().path_size(), 1);
            const std::string a = c.port_a().path(0);
            const std::string b = c.port_b().path(0);
            EXPECT_NE(a, "node_b");
            EXPECT_NE(b, "node_b");
            // Normalize as a sorted pair for set membership.
            survivors.insert(a < b ? std::make_pair(a, b) : std::make_pair(b, a));
            ++total_conns;
        }
    }
    EXPECT_EQ(total_conns, 3u);
    EXPECT_TRUE(survivors.count({"node_a", "node_c"}));
    EXPECT_TRUE(survivors.count({"node_a", "node_d"}));
    EXPECT_TRUE(survivors.count({"node_c", "node_d"}));
}

// Calling remove_host twice on the same hostname must be safe — the second call no-ops.
TEST_F(HostRemovalTest, IdempotentDoubleRemove) {
    const std::string dir = create_test_dir("idempotent_double_remove");
    const std::vector<std::string> hosts = {"node_a", "node_b", "node_c"};
    write_flat_cabling(dir + "cabling.textproto", "cluster", hosts, "WH_GALAXY");
    write_simple_deployment(dir + "deployment.textproto", hosts, "WH_GALAXY");

    CablingGenerator gen(dir + "cabling.textproto", dir + "deployment.textproto");
    gen.remove_host("node_b");
    EXPECT_EQ(gen.get_deployment_hosts().size(), 2u);
    EXPECT_NO_THROW(gen.remove_host("node_b"));  // already gone -> warn + no-op
    EXPECT_EQ(gen.get_deployment_hosts().size(), 2u);
}

// chip_connections_ must be regenerated (not stale) after removal. We can't easily assert
// an exact post-removal count because chip_connections includes intra-node inter-board
// connections too, but we can verify (a) it shrinks and (b) no surviving entry references
// the removed host_id.
TEST_F(HostRemovalTest, ChipConnectionsShrinkAfterRemoval) {
    const std::string dir = create_test_dir("chip_connections_shrink");
    const std::vector<std::string> hosts = {"node_a", "node_b", "node_c"};
    write_flat_cabling(
        dir + "cabling.textproto",
        "cluster",
        hosts,
        "WH_GALAXY",
        {Conn{"node_a", "node_b", 1, 1, 1, 1}, Conn{"node_b", "node_c", 1, 2, 1, 2}});
    write_simple_deployment(dir + "deployment.textproto", hosts, "WH_GALAXY");

    CablingGenerator gen(dir + "cabling.textproto", dir + "deployment.textproto");
    const size_t before = gen.get_chip_connections().size();
    ASSERT_GT(before, 0u);

    // The host_id of node_b before removal — chip_connections referencing it must be gone afterwards.
    HostId removed_id{0};
    for (const auto& h : gen.get_deployment_hosts()) {
        if (h.hostname == "node_b") {
            // deployment_hosts_ is in DFS host_id order; index gives the host_id.
            removed_id = HostId(static_cast<uint32_t>(&h - &gen.get_deployment_hosts().front()));
        }
    }

    gen.remove_host("node_b");
    const size_t after = gen.get_chip_connections().size();
    EXPECT_LT(after, before) << "removing a connected host must shrink chip_connections";

    // After renumber, host_ids 0..N-2. We can't compare to the pre-renumber removed_id directly,
    // but we can assert that no surviving entry points at any host that's no longer in the cluster.
    std::set<HostId> live_ids;
    for (const auto& h : gen.get_deployment_hosts()) {
        live_ids.insert(HostId(static_cast<uint32_t>(&h - &gen.get_deployment_hosts().front())));
    }
    for (const auto& [a, b] : gen.get_chip_connections()) {
        EXPECT_TRUE(live_ids.contains(a.host_id)) << "stale endpoint host_id " << *a.host_id;
        EXPECT_TRUE(live_ids.contains(b.host_id)) << "stale endpoint host_id " << *b.host_id;
    }
    (void)removed_id;
}

// Sequential remove + remove must keep state consistent (no stale connections, contiguous ids).
TEST_F(HostRemovalTest, SequentialRemovalsKeepStateConsistent) {
    const std::string dir = create_test_dir("sequential_removals");
    const std::vector<std::string> hosts = {"node_a", "node_b", "node_c", "node_d"};
    write_flat_cabling(dir + "cabling.textproto", "cluster", hosts, "WH_GALAXY");
    write_simple_deployment(dir + "deployment.textproto", hosts, "WH_GALAXY");

    CablingGenerator gen(dir + "cabling.textproto", dir + "deployment.textproto");
    gen.remove_host("node_a");
    gen.remove_host("node_d");

    auto [cabling_out, deployment_out] = emit_pair(gen, dir);
    auto cabling_desc = load_cluster(cabling_out);
    auto deployment_desc = load_deployment(deployment_out);

    EXPECT_EQ(cabling_desc.root_instance().child_mappings().size(), 2u);
    EXPECT_EQ(collect_root_host_ids(cabling_desc), (std::vector<uint32_t>{0, 1}));
    EXPECT_EQ(deployment_desc.hosts_size(), 2);

    // Both surviving descriptors must round-trip cleanly through a fresh generator.
    EXPECT_NO_THROW({
        CablingGenerator reloaded(cabling_out, deployment_out);
        EXPECT_EQ(reloaded.get_deployment_hosts().size(), 2u);
    });
}

// remove_host with an empty hostname must not match anything (i.e. behave as a no-op).
TEST_F(HostRemovalTest, RemoveEmptyHostnameIsNoOp) {
    const std::string dir = create_test_dir("remove_empty_hostname");
    const std::vector<std::string> hosts = {"node_a", "node_b"};
    write_flat_cabling(dir + "cabling.textproto", "cluster", hosts, "WH_GALAXY");
    write_simple_deployment(dir + "deployment.textproto", hosts, "WH_GALAXY");

    CablingGenerator gen(dir + "cabling.textproto", dir + "deployment.textproto");
    EXPECT_NO_THROW(gen.remove_host(""));
    EXPECT_EQ(gen.get_deployment_hosts().size(), 2u);
}

// Invariant: deployment_hosts_ ordering matches DFS host_id order after removal.
// (We construct a fresh generator on the emitted output and verify deployment.hosts(i).host()
// matches the cabling's child whose host_id == i.)
TEST_F(HostRemovalTest, DeploymentOrderingMatchesDfsHostIds) {
    const std::string dir = create_test_dir("deployment_ordering_dfs");
    const std::vector<std::string> hosts = {"node_a", "node_b", "node_c", "node_d", "node_e"};
    write_flat_cabling(dir + "cabling.textproto", "cluster", hosts, "WH_GALAXY");
    write_simple_deployment(dir + "deployment.textproto", hosts, "WH_GALAXY");

    CablingGenerator gen(dir + "cabling.textproto", dir + "deployment.textproto");
    gen.remove_host("node_c");  // remove a middle node

    auto [cabling_out, deployment_out] = emit_pair(gen, dir);
    auto cabling_desc = load_cluster(cabling_out);
    auto deployment_desc = load_deployment(deployment_out);

    // Build hostname -> host_id from the cabling root_instance child_mappings.
    std::unordered_map<std::string, uint32_t> hostname_to_id;
    for (const auto& [name, mapping] : cabling_desc.root_instance().child_mappings()) {
        ASSERT_EQ(mapping.mapping_case(), cabling_generator::proto::ChildMapping::kHostId);
        hostname_to_id[name] = mapping.host_id();
    }

    // For each entry in deployment.hosts(i), the cabling's child with hostname == hosts(i).host()
    // must have host_id == i. (i.e. the deployment is positionally indexed by host_id.)
    ASSERT_EQ(deployment_desc.hosts_size(), 4);
    for (int i = 0; i < deployment_desc.hosts_size(); ++i) {
        const std::string& hostname = deployment_desc.hosts(i).host();
        ASSERT_TRUE(hostname_to_id.contains(hostname)) << "deployment host '" << hostname << "' has no cabling entry";
        EXPECT_EQ(hostname_to_id[hostname], static_cast<uint32_t>(i))
            << "deployment[" << i << "].host = '" << hostname << "' but cabling host_id = " << hostname_to_id[hostname];
    }
}

// Nested case: removing the only leaf of a sub_instance collapses it out of the parent.
TEST_F(HostRemovalTest, RemoveLastNodeInSubgraphCollapsesIt) {
    const std::string dir = create_test_dir("remove_last_node_collapses_subgraph");
    // Two templates: `outer` has child `group` (graph_ref) + sibling leaf `node_top`,
    // and `group` has one leaf `node_inner`. Removing `node_inner` should collapse `group`.
    write_textproto(
        dir + "cabling.textproto",
        R"(
graph_templates {
  key: "outer"
  value {
    children { name: "node_top" node_ref { node_descriptor: "WH_GALAXY" } }
    children { name: "group" graph_ref { graph_template: "group" } }
  }
}
graph_templates {
  key: "group"
  value {
    children { name: "node_inner" node_ref { node_descriptor: "WH_GALAXY" } }
  }
}
root_instance {
  template_name: "outer"
  child_mappings {
    key: "node_top"
    value { host_id: 0 }
  }
  child_mappings {
    key: "group"
    value {
      sub_instance {
        template_name: "group"
        child_mappings {
          key: "node_inner"
          value { host_id: 1 }
        }
      }
    }
  }
}
)");
    write_simple_deployment(dir + "deployment.textproto", {"node_top", "node_inner"}, "WH_GALAXY");

    CablingGenerator gen(dir + "cabling.textproto", dir + "deployment.textproto");
    gen.remove_host("node_inner");

    auto [cabling_out, deployment_out] = emit_pair(gen, dir);
    auto desc = load_cluster(cabling_out);

    // `group` should no longer appear as a child mapping of the root instance; only node_top remains.
    EXPECT_EQ(desc.root_instance().child_mappings().size(), 1u);
    EXPECT_TRUE(desc.root_instance().child_mappings().contains("node_top"));
    EXPECT_FALSE(desc.root_instance().child_mappings().contains("group"));

    // And the surviving host is at host_id 0.
    EXPECT_EQ(collect_root_host_ids(desc), (std::vector<uint32_t>{0}));

    // Round-trip the reduced descriptor — must still load cleanly.
    EXPECT_NO_THROW({ CablingGenerator reloaded(cabling_out, deployment_out); });
}

}  // namespace tt::scaleout_tools
