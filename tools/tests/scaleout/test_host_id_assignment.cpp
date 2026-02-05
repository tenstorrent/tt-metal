// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>
#include <filesystem>
#include <fstream>
#include <google/protobuf/text_format.h>

#include <cabling_generator/cabling_generator.hpp>
#include "protobuf/cluster_config.pb.h"
#include "protobuf/factory_system_descriptor.pb.h"

namespace tt::scaleout_tools {

class HostIdAssignmentTest : public ::testing::Test {
protected:
    static std::string create_test_dir(const std::string& name) {
        std::string dir = "generated/tests/host_id_" + name + "/";
        if (std::filesystem::exists(dir)) {
            std::filesystem::remove_all(dir);
        }
        std::filesystem::create_directories(dir);
        return dir;
    }

    static std::vector<std::string> hostnames(int count) {
        std::vector<std::string> result;
        for (int i = 0; i < count; ++i) {
            result.push_back("host" + std::to_string(i));
        }
        return result;
    }

    static void write_textproto(const std::string& path, const std::string& content) {
        std::ofstream(path) << content;
    }

    static void create_descriptor(
        const std::string& path,
        const std::string& tmpl,
        const std::vector<std::string>& children,
        const std::vector<uint32_t>& ids) {
        std::string s = "graph_templates {\n  key: \"" + tmpl + "\"\n  value {\n";
        for (const auto& c : children) {
            s += "    children { name: \"" + c + "\" node_ref { node_descriptor: \"WH_GALAXY\" } }\n";
        }
        s += "  }\n}\nroot_instance {\n  template_name: \"" + tmpl + "\"\n";
        for (size_t i = 0; i < children.size(); ++i) {
            s += "  child_mappings { key: \"" + children[i] + "\" value { host_id: " + std::to_string(ids[i]) + " } }\n";
        }
        s += "}\n";
        write_textproto(path, s);
    }

    static void create_nested_descriptor(
        const std::string& path,
        const std::vector<std::string>& root_children,
        const std::vector<bool>& is_subgraph,
        const std::vector<std::string>& sub_children,
        const std::vector<uint32_t>& ids) {
        std::string s;
        s += "graph_templates { key: \"sub\" value {\n";
        for (const auto& c : sub_children) {
            s += "  children { name: \"" + c + "\" node_ref { node_descriptor: \"WH_GALAXY\" } }\n";
        }
        s += "}}\n";
        s += "graph_templates { key: \"root\" value {\n";
        for (size_t i = 0; i < root_children.size(); ++i) {
            if (is_subgraph[i]) {
                s += "  children { name: \"" + root_children[i] + "\" graph_ref { graph_template: \"sub\" } }\n";
            } else {
                s += "  children { name: \"" + root_children[i] + "\" node_ref { node_descriptor: \"WH_GALAXY\" } }\n";
            }
        }
        s += "}}\n";
        s += "root_instance { template_name: \"root\"\n";
        size_t idx = 0;
        for (size_t i = 0; i < root_children.size(); ++i) {
            if (is_subgraph[i]) {
                s += "  child_mappings { key: \"" + root_children[i] + "\" value { sub_instance {\n";
                s += "    template_name: \"sub\"\n";
                for (const auto& c : sub_children) {
                    s += "    child_mappings { key: \"" + c + "\" value { host_id: " + std::to_string(ids[idx++]) + " } }\n";
                }
                s += "  }}}\n";
            } else {
                s += "  child_mappings { key: \"" + root_children[i] + "\" value { host_id: " + std::to_string(ids[idx++]) + " } }\n";
            }
        }
        s += "}\n";
        write_textproto(path, s);
    }
};

TEST_F(HostIdAssignmentTest, FollowsTemplateOrder) {
    auto dir = create_test_dir("template_order");
    create_descriptor(dir + "test.textproto", "cluster", {"node_a", "node_b", "node_c"}, {5, 3, 7});
    CablingGenerator gen(dir + "test.textproto", hostnames(3));

    const auto& hosts = gen.get_deployment_hosts();
    ASSERT_EQ(hosts.size(), 3);
    EXPECT_EQ(hosts[0].hostname, "host0");
    EXPECT_EQ(hosts[1].hostname, "host1");
    EXPECT_EQ(hosts[2].hostname, "host2");
}

TEST_F(HostIdAssignmentTest, TemplateOrderNotAlphabetical) {
    // Template: zebra, yak, apple (not alphabetical)
    auto dir = create_test_dir("not_alphabetical");
    create_descriptor(dir + "test.textproto", "cluster", {"zebra", "yak", "apple"}, {0, 1, 2});
    CablingGenerator gen(dir + "test.textproto", hostnames(3));

    const auto& hosts = gen.get_deployment_hosts();
    ASSERT_EQ(hosts.size(), 3);
}

TEST_F(HostIdAssignmentTest, NestedGraphDFS) {
    // Structure: node1 -> subgraph(sub1, sub2) -> node2
    // DFS order: node1=0, sub1=1, sub2=2, node2=3
    auto dir = create_test_dir("nested");
    create_nested_descriptor(
        dir + "test.textproto",
        {"node1", "subgraph1", "node2"},
        {false, true, false},
        {"sub1", "sub2"},
        {10, 20, 30, 40}  // arbitrary, will be reassigned
    );
    CablingGenerator gen(dir + "test.textproto", hostnames(4));

    const auto& hosts = gen.get_deployment_hosts();
    ASSERT_EQ(hosts.size(), 4);
}

TEST_F(HostIdAssignmentTest, SubgraphFirst) {
    // subgraph comes before node
    auto dir = create_test_dir("subgraph_first");
    create_nested_descriptor(
        dir + "test.textproto",
        {"subgraph1", "node1"},
        {true, false},
        {"sub1", "sub2"},
        {5, 6, 7}
    );
    CablingGenerator gen(dir + "test.textproto", hostnames(3));
    EXPECT_EQ(gen.get_deployment_hosts().size(), 3);
}

TEST_F(HostIdAssignmentTest, MergeProducesConsistentOrder) {
    auto dir = create_test_dir("merge");
    create_descriptor(dir + "a.textproto", "cluster", {"n1", "n2", "n3"}, {0, 1, 2});
    create_descriptor(dir + "b.textproto", "cluster", {"n1", "n2", "n3"}, {0, 1, 2});
    CablingGenerator gen(dir, hostnames(3));

    const auto& hosts = gen.get_deployment_hosts();
    ASSERT_EQ(hosts.size(), 3);
}

TEST_F(HostIdAssignmentTest, ReloadIdentical) {
    auto dir = create_test_dir("reload");
    create_descriptor(dir + "test.textproto", "cluster", {"x", "m", "a"}, {0, 1, 2});

    CablingGenerator gen1(dir + "test.textproto", hostnames(3));
    CablingGenerator gen2(dir + "test.textproto", hostnames(3));
    EXPECT_EQ(gen1, gen2);
}

TEST_F(HostIdAssignmentTest, FileOrderDoesNotMatter) {
    auto dir1 = create_test_dir("order1");
    auto dir2 = create_test_dir("order2");

    std::string content = R"(
graph_templates { key: "t" value {
  children { name: "n1" node_ref { node_descriptor: "WH_GALAXY" } }
  children { name: "n2" node_ref { node_descriptor: "WH_GALAXY" } }
}}
root_instance { template_name: "t"
  child_mappings { key: "n1" value { host_id: 0 } }
  child_mappings { key: "n2" value { host_id: 1 } }
}
)";
    write_textproto(dir1 + "a.textproto", content);
    write_textproto(dir1 + "z.textproto", content);
    write_textproto(dir2 + "z.textproto", content);
    write_textproto(dir2 + "a.textproto", content);

    CablingGenerator gen1(dir1, hostnames(2));
    CablingGenerator gen2(dir2, hostnames(2));
    EXPECT_EQ(gen1, gen2);
}

TEST_F(HostIdAssignmentTest, ExportedHostIdsAreSequential) {
    auto dir = create_test_dir("export");
    write_textproto(dir + "input.textproto", R"(
graph_templates { key: "t" value {
  children { name: "first" node_ref { node_descriptor: "WH_GALAXY" } }
  children { name: "second" node_ref { node_descriptor: "WH_GALAXY" } }
  children { name: "third" node_ref { node_descriptor: "WH_GALAXY" } }
}}
root_instance { template_name: "t"
  child_mappings { key: "first" value { host_id: 100 } }
  child_mappings { key: "second" value { host_id: 50 } }
  child_mappings { key: "third" value { host_id: 200 } }
}
)");

    CablingGenerator gen(dir + "input.textproto", hostnames(3));
    gen.emit_cabling_descriptor(dir + "output.textproto");

    std::ifstream f(dir + "output.textproto");
    std::string content((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());

    cabling_generator::proto::ClusterDescriptor desc;
    ASSERT_TRUE(google::protobuf::TextFormat::ParseFromString(content, &desc));

    const auto& mappings = desc.root_instance().child_mappings();
    EXPECT_EQ(mappings.at("first").host_id(), 0);
    EXPECT_EQ(mappings.at("second").host_id(), 1);
    EXPECT_EQ(mappings.at("third").host_id(), 2);
}

TEST_F(HostIdAssignmentTest, TemplateOrderTakesPrecedence) {
    auto dir = create_test_dir("precedence");
    write_textproto(dir + "input.textproto", R"(
graph_templates { key: "t" value {
  children { name: "charlie" node_ref { node_descriptor: "WH_GALAXY" } }
  children { name: "alpha" node_ref { node_descriptor: "WH_GALAXY" } }
  children { name: "bravo" node_ref { node_descriptor: "WH_GALAXY" } }
}}
root_instance { template_name: "t"
  child_mappings { key: "charlie" value { host_id: 99 } }
  child_mappings { key: "alpha" value { host_id: 98 } }
  child_mappings { key: "bravo" value { host_id: 97 } }
}
)");

    CablingGenerator gen(dir + "input.textproto", hostnames(3));
    gen.emit_cabling_descriptor(dir + "output.textproto");

    std::ifstream f(dir + "output.textproto");
    std::string content((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());

    cabling_generator::proto::ClusterDescriptor desc;
    ASSERT_TRUE(google::protobuf::TextFormat::ParseFromString(content, &desc));

    const auto& mappings = desc.root_instance().child_mappings();
    // Template order: charlie, alpha, bravo -> host_ids: 0, 1, 2
    EXPECT_EQ(mappings.at("charlie").host_id(), 0);
    EXPECT_EQ(mappings.at("alpha").host_id(), 1);
    EXPECT_EQ(mappings.at("bravo").host_id(), 2);
}

TEST_F(HostIdAssignmentTest, SingleNode) {
    auto dir = create_test_dir("single");
    create_descriptor(dir + "test.textproto", "t", {"only"}, {42});
    CablingGenerator gen(dir + "test.textproto", hostnames(1));
    EXPECT_EQ(gen.get_deployment_hosts().size(), 1);
}

TEST_F(HostIdAssignmentTest, ManyNodes) {
    auto dir = create_test_dir("many");
    std::vector<std::string> names;
    std::vector<uint32_t> ids;
    for (int i = 0; i < 50; ++i) {
        names.push_back("n" + std::to_string(i));
        ids.push_back(49 - i);  // reverse order
    }
    create_descriptor(dir + "test.textproto", "t", names, ids);
    CablingGenerator gen(dir + "test.textproto", hostnames(50));
    EXPECT_EQ(gen.get_deployment_hosts().size(), 50);
}

}  // namespace tt::scaleout_tools
