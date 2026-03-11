// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>
#include <atomic>
#include <filesystem>
#include <fstream>
#include <google/protobuf/text_format.h>

#include <cabling_generator/cabling_generator.hpp>
#include "protobuf/cluster_config.pb.h"
#include "protobuf/factory_system_descriptor.pb.h"

namespace tt::scaleout_tools {

// All tests use the deployment-descriptor constructor so FSD hosts[i].hostname()
// directly reflects which node received host_id=i after DFS reassignment.
class HostIdAssignmentTest : public ::testing::Test {
protected:
    std::string create_test_dir(const std::string& name) {
        static std::atomic<int> seq{0};
        auto dir = std::filesystem::temp_directory_path() /
                   ("host_id_" + name + "_test_" + std::to_string(seq++));
        std::filesystem::remove_all(dir);
        std::filesystem::create_directories(dir);
        dirs_.push_back(dir.string());
        return dir.string() + "/";
    }

    void TearDown() override {
        for (const auto& d : dirs_) {
            std::filesystem::remove_all(d);
        }
        dirs_.clear();
    }

    std::vector<std::string> dirs_;

    static void write(const std::string& path, const std::string& content) {
        std::ofstream(path) << content;
    }

    // Flat cabling descriptor: root template -> direct WH_GALAXY nodes.
    static void flat(
        const std::string& path,
        const std::vector<std::string>& names,
        const std::vector<uint32_t>& ids) {
        std::string s = "graph_templates { key: \"root\" value {\n";
        for (const auto& n : names) {
            s += "  children { name: \"" + n + "\" node_ref { node_descriptor: \"WH_GALAXY\" } }\n";
        }
        s += "}}\nroot_instance { template_name: \"root\"\n";
        for (size_t i = 0; i < names.size(); ++i) {
            s += "  child_mappings { key: \"" + names[i] + "\" value { host_id: " +
                 std::to_string(ids[i]) + " } }\n";
        }
        s += "}\n";
        write(path, s);
    }

    // Deployment descriptor with one WH_GALAXY entry per hostname.
    static void deployment(const std::string& path, const std::vector<std::string>& hosts) {
        std::string s;
        for (const auto& h : hosts) {
            s += "hosts { host: \"" + h + "\" node_type: \"WH_GALAXY\" }\n";
        }
        write(path, s);
    }

    // Read back FSD and return hosts vector for easy indexed access.
    static auto fsd_hosts(const CablingGenerator& gen) {
        return gen.generate_factory_system_descriptor().hosts();
    }
};

// Template order determines host_id assignment, not initial child_mapping values.
// node_a is first in the template -> must get host_id 0 after DFS reassignment.
TEST_F(HostIdAssignmentTest, TemplateOrderDeterminesAssignment) {
    auto dir = create_test_dir("template_order");
    flat(dir + "c.textproto", {"node_a", "node_b", "node_c"}, {2, 0, 1});  // scrambled
    deployment(dir + "d.textproto", {"node_a", "node_b", "node_c"});

    CablingGenerator gen(dir + "c.textproto", dir + "d.textproto");
    const auto hosts = fsd_hosts(gen);

    ASSERT_EQ(hosts.size(), 3);
    EXPECT_EQ(hosts[0].hostname(), "node_a");
    EXPECT_EQ(hosts[1].hostname(), "node_b");
    EXPECT_EQ(hosts[2].hostname(), "node_c");
}

// Template order must be respected even when names are non-alphabetical.
// Template: zebra, yak, apple -> host_ids 0, 1, 2 respectively.
TEST_F(HostIdAssignmentTest, TemplateOrderNotAlphabetical) {
    auto dir = create_test_dir("not_alphabetical");
    flat(dir + "c.textproto", {"zebra", "yak", "apple"}, {0, 1, 2});
    deployment(dir + "d.textproto", {"zebra", "yak", "apple"});

    CablingGenerator gen(dir + "c.textproto", dir + "d.textproto");
    const auto hosts = fsd_hosts(gen);

    ASSERT_EQ(hosts.size(), 3);
    EXPECT_EQ(hosts[0].hostname(), "zebra");
    EXPECT_EQ(hosts[1].hostname(), "yak");
    EXPECT_EQ(hosts[2].hostname(), "apple");
}

// DFS enters a subgraph before visiting siblings that come after it in the template.
// Template: node1, subgraph1(sub1,sub2), node2 -> DFS order: node1, sub1, sub2, node2.
TEST_F(HostIdAssignmentTest, SubgraphChildrenVisitedBeforeSiblings) {
    auto dir = create_test_dir("nested_dfs");
    write(dir + "c.textproto", R"(
graph_templates { key: "sub" value {
  children { name: "sub1" node_ref { node_descriptor: "WH_GALAXY" } }
  children { name: "sub2" node_ref { node_descriptor: "WH_GALAXY" } }
}}
graph_templates { key: "root" value {
  children { name: "node1"     node_ref  { node_descriptor: "WH_GALAXY" } }
  children { name: "subgraph1" graph_ref { graph_template: "sub" } }
  children { name: "node2"     node_ref  { node_descriptor: "WH_GALAXY" } }
}}
root_instance { template_name: "root"
  child_mappings { key: "node1" value { host_id: 3 } }
  child_mappings { key: "subgraph1" value { sub_instance {
    template_name: "sub"
    child_mappings { key: "sub1" value { host_id: 0 } }
    child_mappings { key: "sub2" value { host_id: 2 } }
  }}}
  child_mappings { key: "node2" value { host_id: 1 } }
}
)");
    deployment(dir + "d.textproto", {"node1", "sub1", "sub2", "node2"});

    CablingGenerator gen(dir + "c.textproto", dir + "d.textproto");
    const auto hosts = fsd_hosts(gen);

    ASSERT_EQ(hosts.size(), 4);
    EXPECT_EQ(hosts[0].hostname(), "node1");
    EXPECT_EQ(hosts[1].hostname(), "sub1");
    EXPECT_EQ(hosts[2].hostname(), "sub2");
    EXPECT_EQ(hosts[3].hostname(), "node2");
}

// When a subgraph appears first in the template, all its children are
// assigned lower host_ids than any node listed after it.
TEST_F(HostIdAssignmentTest, SubgraphFirstInTemplate_GetsLowerIds) {
    auto dir = create_test_dir("subgraph_first");
    write(dir + "c.textproto", R"(
graph_templates { key: "sub" value {
  children { name: "sub1" node_ref { node_descriptor: "WH_GALAXY" } }
  children { name: "sub2" node_ref { node_descriptor: "WH_GALAXY" } }
}}
graph_templates { key: "root" value {
  children { name: "subgraph1" graph_ref { graph_template: "sub" } }
  children { name: "node1"     node_ref  { node_descriptor: "WH_GALAXY" } }
}}
root_instance { template_name: "root"
  child_mappings { key: "subgraph1" value { sub_instance {
    template_name: "sub"
    child_mappings { key: "sub1" value { host_id: 2 } }
    child_mappings { key: "sub2" value { host_id: 0 } }
  }}}
  child_mappings { key: "node1" value { host_id: 1 } }
}
)");
    deployment(dir + "d.textproto", {"sub1", "sub2", "node1"});

    CablingGenerator gen(dir + "c.textproto", dir + "d.textproto");
    const auto hosts = fsd_hosts(gen);

    ASSERT_EQ(hosts.size(), 3);
    EXPECT_EQ(hosts[0].hostname(), "sub1");
    EXPECT_EQ(hosts[1].hostname(), "sub2");
    EXPECT_EQ(hosts[2].hostname(), "node1");
}

// Template order is [third, first, second] with scrambled initial ids - exported ids must match template position.
TEST_F(HostIdAssignmentTest, ScrambledInitialIdsReassignedSequentially) {
    auto dir = create_test_dir("scrambled");
    write(dir + "c.textproto", R"(
graph_templates { key: "t" value {
  children { name: "third"  node_ref { node_descriptor: "WH_GALAXY" } }
  children { name: "first"  node_ref { node_descriptor: "WH_GALAXY" } }
  children { name: "second" node_ref { node_descriptor: "WH_GALAXY" } }
}}
root_instance { template_name: "t"
  child_mappings { key: "third"  value { host_id: 1 } }
  child_mappings { key: "first"  value { host_id: 2 } }
  child_mappings { key: "second" value { host_id: 0 } }
}
)");
    deployment(dir + "d.textproto", {"third", "first", "second"});

    CablingGenerator gen(dir + "c.textproto", dir + "d.textproto");
    gen.emit_cabling_descriptor(dir + "out.textproto");

    std::ifstream f(dir + "out.textproto");
    std::string content((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
    cabling_generator::proto::ClusterDescriptor desc;
    ASSERT_TRUE(google::protobuf::TextFormat::ParseFromString(content, &desc));

    const auto& m = desc.root_instance().child_mappings();
    EXPECT_EQ(m.at("third").host_id(), 0);
    EXPECT_EQ(m.at("first").host_id(), 1);
    EXPECT_EQ(m.at("second").host_id(), 2);
}

// Template order wins over alphabetical order.
// Template: charlie, alpha, bravo -> must get host_ids 0, 1, 2 respectively.
TEST_F(HostIdAssignmentTest, TemplateOrderNotAlphabetical_ExportedIds) {
    auto dir = create_test_dir("non_alpha_export");
    write(dir + "c.textproto", R"(
graph_templates { key: "t" value {
  children { name: "charlie" node_ref { node_descriptor: "WH_GALAXY" } }
  children { name: "alpha"   node_ref { node_descriptor: "WH_GALAXY" } }
  children { name: "bravo"   node_ref { node_descriptor: "WH_GALAXY" } }
}}
root_instance { template_name: "t"
  child_mappings { key: "charlie" value { host_id: 2 } }
  child_mappings { key: "alpha"   value { host_id: 0 } }
  child_mappings { key: "bravo"   value { host_id: 1 } }
}
)");
    deployment(dir + "d.textproto", {"charlie", "alpha", "bravo"});

    CablingGenerator gen(dir + "c.textproto", dir + "d.textproto");
    const auto hosts = fsd_hosts(gen);

    ASSERT_EQ(hosts.size(), 3);
    EXPECT_EQ(hosts[0].hostname(), "charlie");
    EXPECT_EQ(hosts[1].hostname(), "alpha");
    EXPECT_EQ(hosts[2].hostname(), "bravo");
}

class MergeChildrenOrderTest : public ::testing::Test {
protected:
    std::string create_test_dir(const std::string& name) {
        static std::atomic<int> seq{0};
        auto dir = std::filesystem::temp_directory_path() /
                   ("merge_children_order_" + name + "_test_" + std::to_string(seq++));
        std::filesystem::remove_all(dir);
        std::filesystem::create_directories(dir);
        dirs_.push_back(dir.string());
        return dir.string() + "/";
    }

    void TearDown() override {
        for (const auto& d : dirs_) {
            std::filesystem::remove_all(d);
        }
        dirs_.clear();
    }

    std::vector<std::string> dirs_;

    static void write_textproto(const std::string& path, const std::string& content) {
        std::ofstream(path) << content;
    }

    static void write_deployment(const std::string& path, const std::vector<std::string>& host_names) {
        std::string s;
        for (const auto& h : host_names) {
            s += "hosts { host: \"" + h + "\" node_type: \"WH_GALAXY\" }\n";
        }
        write_textproto(path, s);
    }

    static void write_flat_descriptor(
        const std::string& path,
        const std::string& root_tmpl,
        const std::vector<std::string>& node_names,
        const std::vector<uint32_t>& host_ids) {
        std::string s;
        s += "graph_templates { key: \"" + root_tmpl + "\" value {\n";
        for (const auto& n : node_names) {
            s += "  children { name: \"" + n + "\" node_ref { node_descriptor: \"WH_GALAXY\" } }\n";
        }
        s += "}}\n";
        s += "root_instance { template_name: \"" + root_tmpl + "\"\n";
        for (size_t i = 0; i < node_names.size(); ++i) {
            s += "  child_mappings { key: \"" + node_names[i] + "\" value { host_id: " +
                 std::to_string(host_ids[i]) + " } }\n";
        }
        s += "}\n";
        write_textproto(path, s);
    }

    static void write_subgraph_descriptor(
        const std::string& path,
        const std::string& root_tmpl,
        const std::string& group_name,
        const std::string& sub_tmpl,
        const std::vector<std::string>& node_names,
        const std::vector<uint32_t>& host_ids) {
        std::string s;
        s += "graph_templates { key: \"" + sub_tmpl + "\" value {\n";
        for (const auto& n : node_names) {
            s += "  children { name: \"" + n + "\" node_ref { node_descriptor: \"WH_GALAXY\" } }\n";
        }
        s += "}}\n";
        s += "graph_templates { key: \"" + root_tmpl + "\" value {\n";
        s += "  children { name: \"" + group_name +
             "\" graph_ref { graph_template: \"" + sub_tmpl + "\" } }\n";
        s += "}}\n";
        s += "root_instance { template_name: \"" + root_tmpl + "\"\n";
        s += "  child_mappings { key: \"" + group_name + "\" value { sub_instance {\n";
        s += "    template_name: \"" + sub_tmpl + "\"\n";
        for (size_t i = 0; i < node_names.size(); ++i) {
            s += "    child_mappings { key: \"" + node_names[i] + "\" value { host_id: " +
                 std::to_string(host_ids[i]) + " } }\n";
        }
        s += "  }}}\n}\n";
        write_textproto(path, s);
    }
};

// Two files each contributing a disjoint direct node - both must appear in the merged FSD.
TEST_F(MergeChildrenOrderTest, NewDirectNodeFromMergeGetsHostId) {
    auto base = create_test_dir("new_direct_node");
    std::string cluster_dir = base + "cluster/";
    std::filesystem::create_directories(cluster_dir);
    std::string deployment_path = base + "deployment.textproto";

    write_deployment(deployment_path, {"h1", "h2"});
    write_flat_descriptor(cluster_dir + "a.textproto", "root", {"h1"}, {0});
    write_flat_descriptor(cluster_dir + "b.textproto", "root", {"h2"}, {1});

    CablingGenerator gen(cluster_dir, deployment_path);

    const auto& hosts = gen.get_deployment_hosts();
    ASSERT_EQ(hosts.size(), 2u);
    EXPECT_EQ(hosts[0].hostname, "h1");
    EXPECT_EQ(hosts[1].hostname, "h2");
}

// Two files each contributing a disjoint subgraph - all nodes must appear in the merged FSD.
TEST_F(MergeChildrenOrderTest, NewSubgraphFromMergeGetsHostIds) {
    auto base = create_test_dir("new_subgraph");
    std::string cluster_dir = base + "cluster/";
    std::filesystem::create_directories(cluster_dir);
    std::string deployment_path = base + "deployment.textproto";

    write_deployment(deployment_path, {"h1", "h2", "h3", "h4"});
    write_subgraph_descriptor(cluster_dir + "a.textproto", "root", "group_a", "sub", {"h1", "h2"}, {0, 1});
    write_subgraph_descriptor(cluster_dir + "b.textproto", "root", "group_b", "sub", {"h3", "h4"}, {2, 3});

    CablingGenerator gen(cluster_dir, deployment_path);

    const auto& hosts = gen.get_deployment_hosts();
    ASSERT_EQ(hosts.size(), 4u);
    EXPECT_EQ(hosts[0].hostname, "h1");
    EXPECT_EQ(hosts[1].hostname, "h2");
    EXPECT_EQ(hosts[2].hostname, "h3");
    EXPECT_EQ(hosts[3].hostname, "h4");
}

// Per-file host_ids are irrelevant - the FSD must reflect DFS order after merge.
TEST_F(MergeChildrenOrderTest, MergedSubgraphHostIdsAreSequentialAfterExport) {
    auto base = create_test_dir("subgraph_sequential_export");
    std::string cluster_dir = base + "cluster/";
    std::filesystem::create_directories(cluster_dir);
    std::string deployment_path = base + "deployment.textproto";

    write_deployment(deployment_path, {"h1", "h2", "h3", "h4"});
    write_subgraph_descriptor(cluster_dir + "a.textproto", "root", "group_a", "sub", {"h1", "h2"}, {2, 3});
    write_subgraph_descriptor(cluster_dir + "b.textproto", "root", "group_b", "sub", {"h3", "h4"}, {0, 1});

    CablingGenerator gen(cluster_dir, deployment_path);
    const auto fsd = gen.generate_factory_system_descriptor();

    ASSERT_EQ(fsd.hosts().size(), 4);
    EXPECT_EQ(fsd.hosts()[0].hostname(), "h1");
    EXPECT_EQ(fsd.hosts()[1].hostname(), "h2");
    EXPECT_EQ(fsd.hosts()[2].hostname(), "h3");
    EXPECT_EQ(fsd.hosts()[3].hostname(), "h4");
}

// Template order in file B is [rack_b(subgraph), n2(node)] - subgraph first.
// DFS must visit rack_b's children before n2, so the merged order is n1, n3, n4, n2.
TEST_F(MergeChildrenOrderTest, MergePreservesTemplateOrderForMixedNewChildren) {
    auto base = create_test_dir("template_order_mixed");
    std::string cluster_dir = base + "cluster/";
    std::filesystem::create_directories(cluster_dir);
    std::string deployment_path = base + "deployment.textproto";

    write_deployment(deployment_path, {"n1", "n3", "n4", "n2"});
    write_flat_descriptor(cluster_dir + "a.textproto", "root", {"n1"}, {0});
    write_textproto(
        cluster_dir + "b.textproto",
        R"(graph_templates { key: "sub" value {
  children { name: "n3" node_ref { node_descriptor: "WH_GALAXY" } }
  children { name: "n4" node_ref { node_descriptor: "WH_GALAXY" } }
}}
graph_templates { key: "root" value {
  children { name: "rack_b" graph_ref { graph_template: "sub" } }
  children { name: "n2" node_ref { node_descriptor: "WH_GALAXY" } }
}}
root_instance { template_name: "root"
  child_mappings { key: "rack_b" value { sub_instance {
    template_name: "sub"
    child_mappings { key: "n3" value { host_id: 1 } }
    child_mappings { key: "n4" value { host_id: 2 } }
  }}}
  child_mappings { key: "n2" value { host_id: 3 } }
}
)");

    CablingGenerator gen(cluster_dir, deployment_path);
    const auto fsd = gen.generate_factory_system_descriptor();

    ASSERT_EQ(fsd.hosts().size(), 4);
    EXPECT_EQ(fsd.hosts()[0].hostname(), "n1");
    EXPECT_EQ(fsd.hosts()[1].hostname(), "n3");
    EXPECT_EQ(fsd.hosts()[2].hostname(), "n4");
    EXPECT_EQ(fsd.hosts()[3].hostname(), "n2");
}

// Merged direct nodes must be reassigned sequential host_ids in the emitted descriptor.
TEST_F(MergeChildrenOrderTest, MergedDirectNodeHostIdsAreSequentialAfterExport) {
    auto base = create_test_dir("direct_node_sequential_export");
    std::string cluster_dir = base + "cluster/";
    std::filesystem::create_directories(cluster_dir);
    std::string deployment_path = base + "deployment.textproto";

    write_deployment(deployment_path, {"h1", "h2"});
    write_flat_descriptor(cluster_dir + "a.textproto", "root", {"h1"}, {1});
    write_flat_descriptor(cluster_dir + "b.textproto", "root", {"h2"}, {0});

    CablingGenerator gen(cluster_dir, deployment_path);
    gen.emit_cabling_descriptor(base + "output.textproto");

    std::ifstream f(base + "output.textproto");
    std::string content((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());

    cabling_generator::proto::ClusterDescriptor desc;
    ASSERT_TRUE(google::protobuf::TextFormat::ParseFromString(content, &desc));

    const auto& mappings = desc.root_instance().child_mappings();
    EXPECT_EQ(mappings.at("h1").host_id(), 0);
    EXPECT_EQ(mappings.at("h2").host_id(), 1);
}

// Three-file merge with mixed node/subgraph contributions - DFS order must be n1, n2, n3, n4.
TEST_F(MergeChildrenOrderTest, ThreeFileMergeAllGetSequentialHostIds) {
    auto base = create_test_dir("three_file_merge");
    std::string cluster_dir = base + "cluster/";
    std::filesystem::create_directories(cluster_dir);
    std::string deployment_path = base + "deployment.textproto";

    write_deployment(deployment_path, {"n1", "n2", "n3", "n4"});
    write_flat_descriptor(cluster_dir + "a.textproto", "root", {"n1"}, {0});
    write_subgraph_descriptor(cluster_dir + "b.textproto", "root", "group_b", "sub", {"n2", "n3"}, {1, 2});
    write_flat_descriptor(cluster_dir + "c.textproto", "root", {"n4"}, {3});

    CablingGenerator gen(cluster_dir, deployment_path);
    const auto fsd = gen.generate_factory_system_descriptor();

    ASSERT_EQ(fsd.hosts().size(), 4);
    EXPECT_EQ(fsd.hosts()[0].hostname(), "n1");
    EXPECT_EQ(fsd.hosts()[1].hostname(), "n2");
    EXPECT_EQ(fsd.hosts()[2].hostname(), "n3");
    EXPECT_EQ(fsd.hosts()[3].hostname(), "n4");
}

// Two files share a subgraph name - new nodes are appended in file B's template order.
TEST_F(MergeChildrenOrderTest, NewNodesAddedToExistingSubgraphGetSequentialHostIds) {
    auto base = create_test_dir("new_nodes_existing_subgraph");
    std::string cluster_dir = base + "cluster/";
    std::filesystem::create_directories(cluster_dir);
    std::string deployment_path = base + "deployment.textproto";

    write_deployment(deployment_path, {"n1", "n2", "n3", "n4"});
    write_subgraph_descriptor(cluster_dir + "a.textproto", "root", "group_a", "sub", {"n1", "n2"}, {0, 1});
    write_subgraph_descriptor(cluster_dir + "b.textproto", "root", "group_a", "sub", {"n3", "n4"}, {2, 3});

    CablingGenerator gen(cluster_dir, deployment_path);
    const auto fsd = gen.generate_factory_system_descriptor();

    ASSERT_EQ(fsd.hosts().size(), 4);
    EXPECT_EQ(fsd.hosts()[0].hostname(), "n1");
    EXPECT_EQ(fsd.hosts()[1].hostname(), "n2");
    EXPECT_EQ(fsd.hosts()[2].hostname(), "n3");
    EXPECT_EQ(fsd.hosts()[3].hostname(), "n4");
}

// Two-level nesting in file A (root->outer->inner->{n1,n2}); file B adds n3 at root level.
// DFS must fully descend before visiting n3.
TEST_F(MergeChildrenOrderTest, DeepNestingThroughMergeGetsCorrectDFSOrder) {
    auto base = create_test_dir("deep_nesting_merge");
    std::string cluster_dir = base + "cluster/";
    std::filesystem::create_directories(cluster_dir);
    std::string deployment_path = base + "deployment.textproto";

    write_deployment(deployment_path, {"n1", "n2", "n3"});
    write_textproto(
        cluster_dir + "a.textproto",
        R"(graph_templates { key: "inner_tmpl" value {
  children { name: "n1" node_ref { node_descriptor: "WH_GALAXY" } }
  children { name: "n2" node_ref { node_descriptor: "WH_GALAXY" } }
}}
graph_templates { key: "outer_tmpl" value {
  children { name: "inner_group" graph_ref { graph_template: "inner_tmpl" } }
}}
graph_templates { key: "root" value {
  children { name: "outer_group" graph_ref { graph_template: "outer_tmpl" } }
}}
root_instance { template_name: "root"
  child_mappings { key: "outer_group" value { sub_instance {
    template_name: "outer_tmpl"
    child_mappings { key: "inner_group" value { sub_instance {
      template_name: "inner_tmpl"
      child_mappings { key: "n1" value { host_id: 0 } }
      child_mappings { key: "n2" value { host_id: 1 } }
    }}}
  }}}
}
)");
    write_flat_descriptor(cluster_dir + "b.textproto", "root", {"n3"}, {2});

    CablingGenerator gen(cluster_dir, deployment_path);
    const auto fsd = gen.generate_factory_system_descriptor();

    ASSERT_EQ(fsd.hosts().size(), 3);
    EXPECT_EQ(fsd.hosts()[0].hostname(), "n1");
    EXPECT_EQ(fsd.hosts()[1].hostname(), "n2");
    EXPECT_EQ(fsd.hosts()[2].hostname(), "n3");
}

// File B's local host_ids (n2=0, n3=1) must be remapped to post-merge ids (n2=1, n3=2).
// n1 (id=0) is standalone and must not appear in any inter-node connection.
TEST_F(MergeChildrenOrderTest, ConnectionHostIdsCorrectlyRemappedAfterMerge) {
    auto base = create_test_dir("connection_remap");
    std::string cluster_dir = base + "cluster/";
    std::filesystem::create_directories(cluster_dir);
    std::string deployment_path = base + "deployment.textproto";

    write_deployment(deployment_path, {"n1", "n2", "n3"});
    write_flat_descriptor(cluster_dir + "a.textproto", "root", {"n1"}, {0});
    write_textproto(
        cluster_dir + "b.textproto",
        R"(graph_templates { key: "root" value {
  children { name: "n2" node_ref { node_descriptor: "WH_GALAXY" } }
  children { name: "n3" node_ref { node_descriptor: "WH_GALAXY" } }
  internal_connections {
    key: "QSFP_DD"
    value {
      connections {
        port_a { path: ["n2"] tray_id: 1 port_id: 1 }
        port_b { path: ["n3"] tray_id: 1 port_id: 1 }
      }
    }
  }
}}
root_instance { template_name: "root"
  child_mappings { key: "n2" value { host_id: 0 } }
  child_mappings { key: "n3" value { host_id: 1 } }
}
)");

    CablingGenerator gen(cluster_dir, deployment_path);
    const auto& connections = gen.get_chip_connections();

    bool stale_connection_found = false;
    bool n2_n3_connected = false;
    for (const auto& [ep_a, ep_b] : connections) {
        if (*ep_a.host_id == *ep_b.host_id) {
            continue;  // intra-node connection, skip
        }
        auto a = *ep_a.host_id, b = *ep_b.host_id;
        if (a == 0u || b == 0u) {
            stale_connection_found = true;
        }
        if ((a == 1u && b == 2u) || (a == 2u && b == 1u)) {
            n2_n3_connected = true;
        }
    }
    EXPECT_FALSE(stale_connection_found) << "n1 (host_id=0) must not appear in any inter-node connection";
    EXPECT_TRUE(n2_n3_connected) << "n2 (host_id=1) and n3 (host_id=2) must be connected";
}

// Deployment file order is irrelevant - FSD hosts[i] must reflect DFS template order.
TEST_F(MergeChildrenOrderTest, DeploymentOutOfOrder_SingleFile_FsdReflectsDfsOrder) {
    auto base = create_test_dir("dep_ooo_single");
    write_deployment(base + "dep.textproto", {"h3", "h2", "h1"});
    write_flat_descriptor(base + "cabling.textproto", "root", {"h1", "h2", "h3"}, {2, 1, 0});

    CablingGenerator gen(base + "cabling.textproto", base + "dep.textproto");
    const auto fsd = gen.generate_factory_system_descriptor();

    ASSERT_EQ(fsd.hosts().size(), 3);
    EXPECT_EQ(fsd.hosts()[0].hostname(), "h1");
    EXPECT_EQ(fsd.hosts()[1].hostname(), "h2");
    EXPECT_EQ(fsd.hosts()[2].hostname(), "h3");
}

// Same as above but across a two-file merge with fully reversed deployment order.
TEST_F(MergeChildrenOrderTest, DeploymentOutOfOrder_MergedFiles_FsdReflectsDfsOrder) {
    auto base = create_test_dir("dep_ooo_merge");
    std::string cdir = base + "c/";
    std::filesystem::create_directories(cdir);
    write_deployment(base + "dep.textproto", {"h4", "h3", "h2", "h1"});
    write_flat_descriptor(cdir + "a.textproto", "root", {"h1", "h2"}, {3, 2});
    write_flat_descriptor(cdir + "b.textproto", "root", {"h3", "h4"}, {1, 0});

    CablingGenerator gen(cdir, base + "dep.textproto");
    const auto fsd = gen.generate_factory_system_descriptor();

    ASSERT_EQ(fsd.hosts().size(), 4);
    EXPECT_EQ(fsd.hosts()[0].hostname(), "h1");
    EXPECT_EQ(fsd.hosts()[1].hostname(), "h2");
    EXPECT_EQ(fsd.hosts()[2].hostname(), "h3");
    EXPECT_EQ(fsd.hosts()[3].hostname(), "h4");
}

// Both files declare connections; file B's local ids (0<->1) must become (2<->3) after merge.
TEST_F(MergeChildrenOrderTest, BothFilesHaveConnections_PostMergeIdsCorrect) {
    auto base = create_test_dir("both_connected");
    std::string cdir = base + "c/";
    std::filesystem::create_directories(cdir);
    write_deployment(base + "dep.textproto", {"h1", "h2", "h3", "h4"});
    write_textproto(cdir + "a.textproto", R"(
graph_templates { key: "root" value {
  children { name: "h1" node_ref { node_descriptor: "WH_GALAXY" } }
  children { name: "h2" node_ref { node_descriptor: "WH_GALAXY" } }
  internal_connections { key: "QSFP_DD" value {
    connections {
      port_a { path: ["h1"] tray_id: 1 port_id: 1 }
      port_b { path: ["h2"] tray_id: 1 port_id: 1 }
    }
  }}
}}
root_instance { template_name: "root"
  child_mappings { key: "h1" value { host_id: 0 } }
  child_mappings { key: "h2" value { host_id: 1 } }
}
)");
    write_textproto(cdir + "b.textproto", R"(
graph_templates { key: "root" value {
  children { name: "h3" node_ref { node_descriptor: "WH_GALAXY" } }
  children { name: "h4" node_ref { node_descriptor: "WH_GALAXY" } }
  internal_connections { key: "QSFP_DD" value {
    connections {
      port_a { path: ["h3"] tray_id: 1 port_id: 1 }
      port_b { path: ["h4"] tray_id: 1 port_id: 1 }
    }
  }}
}}
root_instance { template_name: "root"
  child_mappings { key: "h3" value { host_id: 0 } }
  child_mappings { key: "h4" value { host_id: 1 } }
}
)");

    CablingGenerator gen(cdir, base + "dep.textproto");
    std::set<std::pair<uint32_t, uint32_t>> pairs;
    for (const auto& [ep_a, ep_b] : gen.get_chip_connections()) {
        if (*ep_a.host_id != *ep_b.host_id) {
            pairs.insert({std::min(*ep_a.host_id, *ep_b.host_id), std::max(*ep_a.host_id, *ep_b.host_id)});
        }
    }
    EXPECT_EQ(pairs, (std::set<std::pair<uint32_t, uint32_t>>{{0u, 1u}, {2u, 3u}}));
}

// File B's fan-out (h3->h4, h3->h5) must use post-merge ids after remapping.
TEST_F(MergeChildrenOrderTest, FanOut_SecondFile_AllConnectionsRemapped) {
    auto base = create_test_dir("fanout_second_file");
    std::string cdir = base + "c/";
    std::filesystem::create_directories(cdir);
    write_deployment(base + "dep.textproto", {"h1", "h2", "h3", "h4", "h5"});
    write_flat_descriptor(cdir + "a.textproto", "root", {"h1", "h2"}, {0, 1});
    write_textproto(cdir + "b.textproto", R"(
graph_templates { key: "root" value {
  children { name: "h3" node_ref { node_descriptor: "WH_GALAXY" } }
  children { name: "h4" node_ref { node_descriptor: "WH_GALAXY" } }
  children { name: "h5" node_ref { node_descriptor: "WH_GALAXY" } }
  internal_connections { key: "QSFP_DD" value {
    connections {
      port_a { path: ["h3"] tray_id: 1 port_id: 1 }
      port_b { path: ["h4"] tray_id: 1 port_id: 1 }
    }
    connections {
      port_a { path: ["h3"] tray_id: 1 port_id: 2 }
      port_b { path: ["h5"] tray_id: 1 port_id: 1 }
    }
  }}
}}
root_instance { template_name: "root"
  child_mappings { key: "h3" value { host_id: 0 } }
  child_mappings { key: "h4" value { host_id: 1 } }
  child_mappings { key: "h5" value { host_id: 2 } }
}
)");

    CablingGenerator gen(cdir, base + "dep.textproto");
    const auto fsd = gen.generate_factory_system_descriptor();
    ASSERT_EQ(fsd.hosts().size(), 5);
    EXPECT_EQ(fsd.hosts()[2].hostname(), "h3");

    std::set<std::pair<uint32_t, uint32_t>> pairs;
    for (const auto& [ep_a, ep_b] : gen.get_chip_connections()) {
        if (*ep_a.host_id != *ep_b.host_id) {
            pairs.insert({std::min(*ep_a.host_id, *ep_b.host_id), std::max(*ep_a.host_id, *ep_b.host_id)});
        }
    }
    EXPECT_EQ(pairs, (std::set<std::pair<uint32_t, uint32_t>>{{2u, 3u}, {2u, 4u}}));
}

}  // namespace tt::scaleout_tools
