// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <memory>
#include <vector>
#include <unordered_map>
#include <random>

#include <tt-metalium/distributed.hpp>
#include <tt-metalium/fabric.hpp>

#include "tests/tt_metal/multihost/fabric_tests/mesh_socket_yaml_parser.hpp"
#include "tt_metal/fabric/fabric_context.hpp"
#include "impl/context/metal_context.hpp"

using MeshId = tt::tt_fabric::MeshId;
using ControlPlane = tt::tt_fabric::ControlPlane;

namespace tt::tt_fabric::mesh_socket_tests {

class MeshSocketTestContext {
public:
    explicit MeshSocketTestContext(const MeshSocketTestConfiguration& config);

    ~MeshSocketTestContext();

    void initialize();
    void run_all_tests();
    void run_test_by_name(const std::string& test_name);
    void cleanup();
    const std::shared_ptr<tt::tt_metal::distributed::multihost::DistributedContext>& get_distributed_context() const;
    const std::shared_ptr<tt::tt_metal::distributed::MeshDevice>& get_mesh_device() const;
    const tt::tt_fabric::MeshGraph& get_mesh_graph() const;
    const std::unordered_map<Rank, tt::tt_fabric::MeshId>& get_rank_to_mesh_mapping() const;
    std::mt19937 get_rng() const { return gen_; }

private:
    void initialize_and_validate_custom_physical_config(const PhysicalMeshConfig& physical_mesh_config);

    void initialize_mesh_device();
    void setup_fabric_configuration();
    void share_seed();
    std::unordered_map<Rank, tt::tt_fabric::MeshId> create_rank_to_mesh_mapping();
    void expand_test_configurations();

    std::vector<tt::tt_metal::distributed::MeshSocket> create_sockets_for_test(const ParsedTestConfig& test);
    tt::tt_metal::distributed::SocketConfig convert_to_socket_config(
        const TestSocketConfig& socket_config, const ParsedMemoryConfig& memory_config);

    tt::tt_metal::distributed::SocketConnection convert_to_socket_connection(
        const SocketConnectionConfig& connection_config);

    void run_test(const ParsedTestConfig& test);
    void execute_socket_test(tt::tt_metal::distributed::MeshSocket& socket, const ParsedTestConfig& test);
    bool should_participate_in_test(const ParsedTestConfig& test) const;

    // Configuration and state
    MeshSocketTestConfiguration config_;
    std::vector<ParsedTestConfig> expanded_tests_;
    std::shared_ptr<tt::tt_metal::distributed::MeshDevice> mesh_device_;
    tt::tt_metal::distributed::multihost::Rank local_rank_;
    std::shared_ptr<tt::tt_metal::distributed::multihost::DistributedContext> distributed_context_;

    // Control plane and mesh configuration
    ControlPlane* control_plane_ptr_;
    MeshId local_mesh_id_;
    std::unordered_map<Rank, tt::tt_fabric::MeshId> rank_to_mesh_mapping_;

    // Random number generation
    std::mt19937 gen_;

    // Default test parameters
    static constexpr uint32_t DEFAULT_NUM_ITERATIONS = 1;
};

}  // namespace tt::tt_fabric::mesh_socket_tests
