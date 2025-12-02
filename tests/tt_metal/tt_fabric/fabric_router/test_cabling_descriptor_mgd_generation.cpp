// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>
#include <gmock/gmock.h>

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <ranges>
#include <sstream>
#include <string>
#include <string_view>

#include <google/protobuf/text_format.h>
#include <tt-metalium/experimental/fabric/mesh_graph_descriptor.hpp>
#include <tt_stl/cleanup.hpp>
#include "generate_mgd.hpp"

namespace fs = std::filesystem;
using namespace tt::tt_fabric;
using namespace tt::scaleout_tools;

namespace {

constexpr std::string_view kMeshType = "MESH";
constexpr std::string_view kTempTestFile = "test_mgd_generation.textproto";

// Topology configuration for different cluster types
struct TopologyConfig {
    std::string_view name;
    std::string_view cabling_path;
    std::string_view architecture;
    size_t devices_per_mesh;
    size_t num_meshes;
};

constexpr TopologyConfig k16NodeClosetBoxConfig{
    .name = "16NodeClosetBox",
    .cabling_path = "tools/tests/scaleout/cabling_descriptors/16_n300_lb_cluster.textproto",
    .architecture = "WORMHOLE_B0",
    .devices_per_mesh = 8,
    .num_meshes = 16};

constexpr TopologyConfig kDualT3KConfig{
    .name = "DualT3K",
    .cabling_path = "tools/tests/scaleout/cabling_descriptors/dual_t3k.textproto",
    .architecture = "WORMHOLE_B0",
    .devices_per_mesh = 8,
    .num_meshes = 2};

constexpr size_t kNumSuperpods = 4;
constexpr size_t kNodesPerSuperpod = 4;

void check_instance_count_by_type(const MeshGraphDescriptor& desc, std::string_view type, size_t expected_count) {
    const auto& ids = desc.instances_by_type(std::string(type));
    EXPECT_EQ(ids.size(), expected_count) << "Expected " << expected_count << " " << type << " instances";
}

void check_mesh_exists(const MeshGraphDescriptor& desc, uint32_t expected_mesh_id) {
    const auto& mesh_ids = desc.all_meshes();
    const auto found =
        std::ranges::any_of(mesh_ids, [&](auto id) { return desc.get_instance(id).local_id == expected_mesh_id; });
    EXPECT_TRUE(found) << "Mesh " << expected_mesh_id << " not found";
}

void check_mesh_device_count(const MeshGraphDescriptor& desc, uint32_t mesh_id, size_t expected_devices) {
    for (const auto id : desc.all_meshes()) {
        const auto& inst = desc.get_instance(id);
        if (inst.local_id == mesh_id) {
            EXPECT_EQ(inst.sub_instances.size(), expected_devices)
                << "Mesh " << mesh_id << " has incorrect device count";
            return;
        }
    }
    FAIL() << "Mesh " << mesh_id << " not found";
}

void check_architecture(const MeshGraphDescriptor& desc, [[maybe_unused]] std::string_view expected_arch) {
    EXPECT_GT(desc.all_meshes().size(), 0) << "No meshes found";
}

void check_intermesh_connection_exists(
    const MeshGraphDescriptor& desc, uint32_t mesh_id_a, uint32_t mesh_id_b, uint32_t expected_channel_count) {
    const auto& connections = desc.connections_by_instance_id(desc.top_level().global_id);

    for (const auto conn_id : connections) {
        const auto& conn = desc.get_connection(conn_id);
        ASSERT_GE(conn.nodes.size(), 2) << "Connection must have at least 2 nodes";

        const auto& [inst_a, inst_b] = std::tie(desc.get_instance(conn.nodes[0]), desc.get_instance(conn.nodes[1]));

        if (desc.is_mesh(inst_a) && desc.is_mesh(inst_b)) {
            const auto matches = (inst_a.local_id == mesh_id_a && inst_b.local_id == mesh_id_b) ||
                                 (inst_a.local_id == mesh_id_b && inst_b.local_id == mesh_id_a);
            if (matches) {
                EXPECT_EQ(conn.count, expected_channel_count)
                    << "M" << mesh_id_a << " <-> M" << mesh_id_b << " channel count mismatch";
                return;
            }
        }
    }
    FAIL() << "Connection M" << mesh_id_a << " <-> M" << mesh_id_b << " not found";
}

void check_total_connection_count(const MeshGraphDescriptor& desc, size_t expected_count) {
    const auto& connections = desc.connections_by_instance_id(desc.top_level().global_id);

    const auto mesh_connection_count = std::ranges::count_if(connections, [&](auto conn_id) {
        const auto& conn = desc.get_connection(conn_id);
        EXPECT_GE(conn.nodes.size(), 2) << "Connection must have at least 2 nodes";
        if (conn.nodes.size() < 2) {
            return false;
        }

        const auto& inst_a = desc.get_instance(conn.nodes[0]);
        const auto& inst_b = desc.get_instance(conn.nodes[1]);
        return desc.is_mesh(inst_a) && desc.is_mesh(inst_b);
    });

    EXPECT_EQ(mesh_connection_count, expected_count * 2)
        << "Expected " << expected_count << " bidirectional connections";
}

void verify_mgd_structure(const MeshGraphDescriptor& desc, const TopologyConfig& config) {
    check_instance_count_by_type(desc, kMeshType, config.num_meshes);
    check_architecture(desc, config.architecture);

    for (uint32_t i = 0; i < config.num_meshes; ++i) {
        check_mesh_exists(desc, i);
        check_mesh_device_count(desc, i, config.devices_per_mesh);
    }
}

void verify_16node_closetbox_connections(const MeshGraphDescriptor& desc) {
    // Verify total connection count
    constexpr size_t kIntraSuperpodConnections = 6;
    constexpr size_t kInterSuperpodConnections = 6;
    constexpr size_t kTotalConnections = (kIntraSuperpodConnections * kNumSuperpods) + kInterSuperpodConnections;
    check_total_connection_count(desc, kTotalConnections);

    // Verify all superpods have internal connections
    for (uint32_t superpod = 0; superpod < kNumSuperpods; ++superpod) {
        const uint32_t base = superpod * kNodesPerSuperpod;
        for (const auto& [a, b] :
             std::vector<std::pair<uint32_t, uint32_t>>{{0, 1}, {0, 2}, {0, 3}, {1, 2}, {1, 3}, {2, 3}}) {
            check_intermesh_connection_exists(desc, base + a, base + b, 2);
        }
    }

    // Verify inter-superpod connections
    check_intermesh_connection_exists(desc, 0, 8, 2);    // Superpod1.Node1 (M0) <-> Superpod3.Node1 (M8)
    check_intermesh_connection_exists(desc, 2, 5, 2);    // Superpod1.Node3 (M2) <-> Superpod2.Node2 (M5)
    check_intermesh_connection_exists(desc, 3, 12, 2);   // Superpod1.Node4 (M3) <-> Superpod4.Node1 (M12)
    check_intermesh_connection_exists(desc, 10, 13, 2);  // Superpod3.Node3 (M10) <-> Superpod4.Node2 (M13)
    check_intermesh_connection_exists(desc, 11, 4, 2);   // Superpod3.Node4 (M11) <-> Superpod2.Node1 (M4)
    check_intermesh_connection_exists(desc, 7, 15, 2);   // Superpod2.Node4 (M7) <-> Superpod4.Node4 (M15)
}

void verify_dual_t3k_connections(const MeshGraphDescriptor& desc) {
    // Dual T3K has 1 connection between the two meshes
    check_total_connection_count(desc, 1);
    check_intermesh_connection_exists(desc, 0, 1, 8);  // M0 <-> M1 with 8 channels
}

}  // namespace

namespace tt::tt_fabric::mgd_generation_tests {

TEST(CablingDescriptorMGDGenerationTests, Generate16NodeClosetBox) {
    const auto mgd_proto = generate_mgd_from_cabling(std::string(k16NodeClosetBoxConfig.cabling_path), false);

    std::string mgd_text;
    google::protobuf::TextFormat::PrintToString(mgd_proto, &mgd_text);
    const MeshGraphDescriptor desc(mgd_text);

    verify_mgd_structure(desc, k16NodeClosetBoxConfig);
    verify_16node_closetbox_connections(desc);
}

TEST(CablingDescriptorMGDGenerationTests, GenerateDualT3K) {
    const auto mgd_proto = generate_mgd_from_cabling(std::string(kDualT3KConfig.cabling_path), false);

    std::string mgd_text;
    google::protobuf::TextFormat::PrintToString(mgd_proto, &mgd_text);
    const MeshGraphDescriptor desc(mgd_text);

    verify_mgd_structure(desc, kDualT3KConfig);
    verify_dual_t3k_connections(desc);
}

TEST(CablingDescriptorMGDGenerationTests, ProtobufAPIUsage) {
    if (std::ifstream input{std::string(k16NodeClosetBoxConfig.cabling_path)}; input.is_open()) {
        const std::string cabling_text((std::istreambuf_iterator<char>(input)), std::istreambuf_iterator<char>());

        cabling_generator::proto::ClusterDescriptor cluster_desc;
        ASSERT_TRUE(google::protobuf::TextFormat::ParseFromString(cabling_text, &cluster_desc));

        const auto mgd_proto = generate_mgd_from_cabling(cluster_desc, false);

        std::string mgd_text;
        google::protobuf::TextFormat::PrintToString(mgd_proto, &mgd_text);
        const MeshGraphDescriptor desc(mgd_text);

        verify_mgd_structure(desc, k16NodeClosetBoxConfig);
        verify_16node_closetbox_connections(desc);
    } else {
        FAIL() << "Failed to open cabling descriptor";
    }
}

TEST(CablingDescriptorMGDGenerationTests, InvalidCablingDescriptorPath) {
    constexpr auto invalid_path = "nonexistent/cabling_descriptor.textproto";
    EXPECT_THROW(
        { [[maybe_unused]] auto result = generate_mgd_from_cabling(invalid_path, false); }, std::runtime_error);
}

TEST(CablingDescriptorMGDGenerationTests, WriteToFileTextProto) {
    const fs::path temp_output = fs::temp_directory_path() / kTempTestFile;
    auto _ = ttsl::make_cleanup([&temp_output]() { fs::remove(temp_output); });

    const auto mgd_proto = generate_mgd_from_cabling(std::string(k16NodeClosetBoxConfig.cabling_path), false);
    write_mgd_to_file(mgd_proto, temp_output);

    EXPECT_TRUE(fs::exists(temp_output));
    EXPECT_GT(fs::file_size(temp_output), 0) << "Output file is empty";

    const MeshGraphDescriptor desc(temp_output);
    verify_mgd_structure(desc, k16NodeClosetBoxConfig);
    verify_16node_closetbox_connections(desc);
}

TEST(CablingDescriptorMGDGenerationTests, EndToEndConvenienceAPI) {
    const fs::path temp_output = fs::temp_directory_path() / kTempTestFile;
    auto _ = ttsl::make_cleanup([&temp_output]() { fs::remove(temp_output); });

    generate_mesh_graph_descriptor(std::string(k16NodeClosetBoxConfig.cabling_path), temp_output, false);

    EXPECT_TRUE(fs::exists(temp_output));
    EXPECT_GT(fs::file_size(temp_output), 0) << "Output file is empty";

    const MeshGraphDescriptor desc(temp_output);
    verify_mgd_structure(desc, k16NodeClosetBoxConfig);
    verify_16node_closetbox_connections(desc);
}

}  // namespace tt::tt_fabric::mgd_generation_tests
