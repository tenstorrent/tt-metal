// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>
#include <algorithm>
#include <atomic>
#include <filesystem>
#include <fstream>
#include <set>
#include <google/protobuf/text_format.h>

#include <cabling_generator/cabling_generator.hpp>
#include "protobuf/cluster_config.pb.h"
#include "protobuf/factory_system_descriptor.pb.h"

namespace tt::scaleout_tools {

static constexpr std::string_view k5NodeCabling =
    "tools/tests/scaleout/cabling_descriptors/5_wh_galaxy_y_torus_superpod.textproto";
static constexpr std::string_view k16NodeCabling =
    "tools/tests/scaleout/cabling_descriptors/16_n300_lb_cluster.textproto";
static constexpr std::string_view k16NodeDeploy =
    "tools/tests/scaleout/deployment_descriptors/16_lb_deployment.textproto";

// ---- Helpers ----

static std::string tmp_dir(const std::string& name) {
    static std::atomic<int> seq{0};
    auto p = std::filesystem::temp_directory_path() / ("host_id_" + name + "_" + std::to_string(seq++));
    std::filesystem::remove_all(p);
    std::filesystem::create_directories(p);
    return p.string() + "/";
}

static void write(const std::string& path, const std::string& content) {
    std::ofstream(path) << content;
}

static void write_deploy(const std::string& path, const std::vector<std::string>& hosts,
                         const std::string& node_type = "WH_GALAXY_Y_TORUS") {
    std::string s;
    for (const auto& h : hosts)
        s += "hosts { host: \"" + h + "\" node_type: \"" + node_type + "\" }\n";
    write(path, s);
}

static std::string read_file(const std::string& path) {
    std::ifstream f(path);
    return {std::istreambuf_iterator<char>(f), {}};
}

static void replace_first(std::string& s, const std::string& from, const std::string& to) {
    auto pos = s.find(from);
    if (pos != std::string::npos) s.replace(pos, from.size(), to);
}

// ---- Single-file tests ----

// Real 5-node cabling with node1<->node5 and node2<->node4 host_ids swapped.
// DFS must restore template order (node1..5) regardless of scrambled initial IDs.
// Also checks the emitted cabling carries the DFS-corrected IDs, not the originals.
TEST(HostIdAssignmentTest, ScrambledHostIds_TemplateOrderRestored) {
    auto dir = tmp_dir("scrambled");

    auto cabling = read_file(std::string(k5NodeCabling));
    for (const auto& [key, from, to] : std::vector<std::tuple<std::string, int, int>>{
             {"node1", 0, 4}, {"node2", 1, 3}, {"node4", 3, 1}, {"node5", 4, 0}}) {
        replace_first(cabling,
            "\"" + key + "\"\n    value { host_id: " + std::to_string(from) + " }",
            "\"" + key + "\"\n    value { host_id: " + std::to_string(to) + " }");
    }
    write(dir + "c.textproto", cabling);
    write_deploy(dir + "dep.textproto", {"node1", "node2", "node3", "node4", "node5"});

    CablingGenerator gen(dir + "c.textproto", dir + "dep.textproto");
    const auto hosts = gen.generate_factory_system_descriptor().hosts();

    ASSERT_EQ(hosts.size(), 5u);
    EXPECT_EQ(hosts[0].hostname(), "node1");
    EXPECT_EQ(hosts[1].hostname(), "node2");
    EXPECT_EQ(hosts[2].hostname(), "node3");
    EXPECT_EQ(hosts[3].hostname(), "node4");
    EXPECT_EQ(hosts[4].hostname(), "node5");

    gen.emit_cabling_descriptor(dir + "out.textproto");
    std::ifstream ef(dir + "out.textproto");
    std::string emitted((std::istreambuf_iterator<char>(ef)), {});
    cabling_generator::proto::ClusterDescriptor desc;
    ASSERT_TRUE(google::protobuf::TextFormat::ParseFromString(emitted, &desc));
    const auto& m = desc.root_instance().child_mappings();
    EXPECT_EQ(m.at("node1").host_id(), 0u);
    EXPECT_EQ(m.at("node5").host_id(), 4u);

    std::filesystem::remove_all(dir);
}

// Real 5-node cabling + deployment with hosts listed in reverse template order.
// DFS must reorder deployment_hosts_ to match template traversal regardless of file order.
TEST(HostIdAssignmentTest, ReversedDeployment_TemplateOrderRestored) {
    auto dir = tmp_dir("reversed_deploy");
    write_deploy(dir + "dep.textproto", {"node5", "node4", "node3", "node2", "node1"});

    CablingGenerator gen(std::string(k5NodeCabling), dir + "dep.textproto");
    const auto hosts = gen.generate_factory_system_descriptor().hosts();

    ASSERT_EQ(hosts.size(), 5u);
    EXPECT_EQ(hosts[0].hostname(), "node1");
    EXPECT_EQ(hosts[4].hostname(), "node5");

    std::filesystem::remove_all(dir);
}

// Real 16-node cabling (4 superpods of 4 nodes each) with its real deployment.
// All FSD connection endpoints must use host_ids in [0..15], confirming DFS traversed
// the full nested hierarchy correctly.
TEST(HostIdAssignmentTest, ComplexHierarchy_AllConnectionsUseValidHostIds) {
    CablingGenerator gen{std::string(k16NodeCabling), std::string(k16NodeDeploy)};
    const auto fsd = gen.generate_factory_system_descriptor();

    ASSERT_EQ(fsd.hosts().size(), 16u);
    for (const auto& conn : fsd.eth_connections().connection()) {
        EXPECT_LT(conn.endpoint_a().host_id(), 16u);
        EXPECT_LT(conn.endpoint_b().host_id(), 16u);
    }
    EXPECT_GT(fsd.eth_connections().connection().size(), 0u);
}

// ---- Merge tests ----

// Real 5-node cabling as file A; file B adds node6+node7 to the same template.
// After merge + DFS, all 7 nodes must be sequential and in template order.
TEST(MergeTest, RealBaseWithExtension_AllNodesSequential) {
    auto dir = tmp_dir("merge_extend");
    std::string cdir = dir + "c/";
    std::filesystem::create_directories(cdir);
    std::filesystem::copy(k5NodeCabling, cdir + "a.textproto");
    write(cdir + "b.textproto", R"(
graph_templates { key: "5_wh_galaxy_y_torus_superpod" value {
  children { name: "node6" node_ref { node_descriptor: "WH_GALAXY_Y_TORUS" } }
  children { name: "node7" node_ref { node_descriptor: "WH_GALAXY_Y_TORUS" } }
}}
root_instance { template_name: "5_wh_galaxy_y_torus_superpod"
  child_mappings { key: "node6" value { host_id: 0 } }
  child_mappings { key: "node7" value { host_id: 1 } }
}
)");
    write_deploy(dir + "dep.textproto", {"node1", "node2", "node3", "node4", "node5", "node6", "node7"});

    CablingGenerator gen(cdir, dir + "dep.textproto");
    const auto fsd = gen.generate_factory_system_descriptor();

    ASSERT_EQ(fsd.hosts().size(), 7u);
    EXPECT_EQ(fsd.hosts()[0].hostname(), "node1");
    EXPECT_EQ(fsd.hosts()[5].hostname(), "node6");
    EXPECT_EQ(fsd.hosts()[6].hostname(), "node7");

    std::filesystem::remove_all(dir);
}

// Real 5-node cabling as file A; file B re-declares node1..5 (shared) and adds node6.
// Shared nodes collapse to a single entry: 6 unique hosts, not 11.
TEST(MergeTest, SharedNodesCollapse_NoDuplication) {
    auto dir = tmp_dir("merge_shared");
    std::string cdir = dir + "c/";
    std::filesystem::create_directories(cdir);
    std::filesystem::copy(k5NodeCabling, cdir + "a.textproto");
    write(cdir + "b.textproto", R"(
graph_templates { key: "5_wh_galaxy_y_torus_superpod" value {
  children { name: "node1" node_ref { node_descriptor: "WH_GALAXY_Y_TORUS" } }
  children { name: "node2" node_ref { node_descriptor: "WH_GALAXY_Y_TORUS" } }
  children { name: "node3" node_ref { node_descriptor: "WH_GALAXY_Y_TORUS" } }
  children { name: "node4" node_ref { node_descriptor: "WH_GALAXY_Y_TORUS" } }
  children { name: "node5" node_ref { node_descriptor: "WH_GALAXY_Y_TORUS" } }
  children { name: "node6" node_ref { node_descriptor: "WH_GALAXY_Y_TORUS" } }
}}
root_instance { template_name: "5_wh_galaxy_y_torus_superpod"
  child_mappings { key: "node1" value { host_id: 0 } }
  child_mappings { key: "node2" value { host_id: 1 } }
  child_mappings { key: "node3" value { host_id: 2 } }
  child_mappings { key: "node4" value { host_id: 3 } }
  child_mappings { key: "node5" value { host_id: 4 } }
  child_mappings { key: "node6" value { host_id: 5 } }
}
)");
    write_deploy(dir + "dep.textproto", {"node1", "node2", "node3", "node4", "node5", "node6"});

    CablingGenerator gen(cdir, dir + "dep.textproto");
    const auto fsd = gen.generate_factory_system_descriptor();

    ASSERT_EQ(fsd.hosts().size(), 6u);
    EXPECT_EQ(fsd.hosts()[5].hostname(), "node6");

    std::filesystem::remove_all(dir);
}

// File B declares n2+n3 with local ids 0,1 and a connection between them.
// After merge, n2 and n3 get post-merge ids 1,2; connection endpoints must reflect that.
TEST(MergeTest, ConnectionHostIdsRemappedAfterMerge) {
    auto dir = tmp_dir("merge_conns");
    std::string cdir = dir + "c/";
    std::filesystem::create_directories(cdir);
    write(cdir + "a.textproto", R"(
graph_templates { key: "root" value {
  children { name: "n1" node_ref { node_descriptor: "WH_GALAXY" } }
}}
root_instance { template_name: "root"
  child_mappings { key: "n1" value { host_id: 0 } }
}
)");
    write(cdir + "b.textproto", R"(
graph_templates { key: "root" value {
  children { name: "n2" node_ref { node_descriptor: "WH_GALAXY" } }
  children { name: "n3" node_ref { node_descriptor: "WH_GALAXY" } }
  internal_connections { key: "QSFP_DD" value {
    connections {
      port_a { path: ["n2"] tray_id: 1 port_id: 1 }
      port_b { path: ["n3"] tray_id: 1 port_id: 1 }
    }
  }}
}}
root_instance { template_name: "root"
  child_mappings { key: "n2" value { host_id: 0 } }
  child_mappings { key: "n3" value { host_id: 1 } }
}
)");
    write(dir + "dep.textproto", "hosts{host:\"n1\" node_type:\"WH_GALAXY\"}\nhosts{host:\"n2\" node_type:\"WH_GALAXY\"}\nhosts{host:\"n3\" node_type:\"WH_GALAXY\"}\n");

    CablingGenerator gen(cdir, dir + "dep.textproto");
    bool stale = false, n2_n3 = false;
    for (const auto& [ep_a, ep_b] : gen.get_chip_connections()) {
        if (*ep_a.host_id == *ep_b.host_id) continue;
        auto a = *ep_a.host_id, b = *ep_b.host_id;
        if (a == 0u || b == 0u) stale = true;
        if ((a == 1u && b == 2u) || (a == 2u && b == 1u)) n2_n3 = true;
    }
    EXPECT_FALSE(stale) << "n1 (id=0) must not appear in inter-node connections";
    EXPECT_TRUE(n2_n3) << "n2 (id=1) and n3 (id=2) must be connected after remap";

    std::filesystem::remove_all(dir);
}

}  // namespace tt::scaleout_tools
