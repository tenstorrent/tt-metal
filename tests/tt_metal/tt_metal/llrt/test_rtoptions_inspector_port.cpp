// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdlib>
#include <stdexcept>
#include <string>

#include "gtest/gtest.h"
#include "llrt/rtoptions.hpp"

namespace tt::llrt {

class InspectorPortTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Save any existing environment variables that might interfere
        saved_ompi_rank_ = safe_getenv("OMPI_COMM_WORLD_RANK");
        saved_pmi_rank_ = safe_getenv("PMI_RANK");
        saved_mesh_rank_ = safe_getenv("TT_MESH_HOST_RANK");
        saved_inspector_addr_ = safe_getenv("TT_METAL_INSPECTOR_RPC_SERVER_ADDRESS");

        // Clear all rank-related env vars for a clean slate
        unsetenv("OMPI_COMM_WORLD_RANK");
        unsetenv("PMI_RANK");
        unsetenv("TT_MESH_HOST_RANK");
        unsetenv("TT_METAL_INSPECTOR_RPC_SERVER_ADDRESS");
    }

    void TearDown() override {
        // Restore original environment
        restore_env("OMPI_COMM_WORLD_RANK", saved_ompi_rank_);
        restore_env("PMI_RANK", saved_pmi_rank_);
        restore_env("TT_MESH_HOST_RANK", saved_mesh_rank_);
        restore_env("TT_METAL_INSPECTOR_RPC_SERVER_ADDRESS", saved_inspector_addr_);
    }

    static std::string safe_getenv(const char* name) {
        const char* val = std::getenv(name);
        return val ? val : "";
    }

    static void restore_env(const char* name, const std::string& value) {
        if (value.empty()) {
            unsetenv(name);
        } else {
            setenv(name, value.c_str(), 1);
        }
    }

private:
    std::string saved_ompi_rank_;
    std::string saved_pmi_rank_;
    std::string saved_mesh_rank_;
    std::string saved_inspector_addr_;
};

TEST_F(InspectorPortTest, DefaultPortWithNoRank) {
    // When no MPI rank is set, should return the default base port (50051)
    RunTimeOptions options;
    EXPECT_EQ(options.get_inspector_rpc_server_port(), 50051);
}

TEST_F(InspectorPortTest, PortWithOMPIRank) {
    // OMPI_COMM_WORLD_RANK should be used for port offset
    setenv("OMPI_COMM_WORLD_RANK", "3", 1);
    RunTimeOptions options;
    EXPECT_EQ(options.get_inspector_rpc_server_port(), 50054);  // 50051 + 3
}

TEST_F(InspectorPortTest, PortWithPMIRank) {
    // PMI_RANK should be used when OMPI rank is not set
    setenv("PMI_RANK", "5", 1);
    RunTimeOptions options;
    EXPECT_EQ(options.get_inspector_rpc_server_port(), 50056);  // 50051 + 5
}

TEST_F(InspectorPortTest, PortWithMeshRank) {
    // TT_MESH_HOST_RANK should be used when other rank vars are not set
    setenv("TT_MESH_HOST_RANK", "2", 1);
    RunTimeOptions options;
    EXPECT_EQ(options.get_inspector_rpc_server_port(), 50053);  // 50051 + 2
}

TEST_F(InspectorPortTest, OMPIRankTakesPrecedence) {
    // OMPI_COMM_WORLD_RANK should take precedence over other rank variables
    setenv("OMPI_COMM_WORLD_RANK", "1", 1);
    setenv("PMI_RANK", "100", 1);
    setenv("TT_MESH_HOST_RANK", "200", 1);
    RunTimeOptions options;
    EXPECT_EQ(options.get_inspector_rpc_server_port(), 50052);  // 50051 + 1
}

TEST_F(InspectorPortTest, PMIRankOverMeshRank) {
    // PMI_RANK should take precedence over TT_MESH_HOST_RANK
    setenv("PMI_RANK", "4", 1);
    setenv("TT_MESH_HOST_RANK", "200", 1);
    RunTimeOptions options;
    EXPECT_EQ(options.get_inspector_rpc_server_port(), 50055);  // 50051 + 4
}

TEST_F(InspectorPortTest, PortOverflowThrows) {
    // When base_port + rank exceeds 65535, runtime should fail fast.
    setenv("OMPI_COMM_WORLD_RANK", "20000", 1);
    RunTimeOptions options;
    EXPECT_THROW(static_cast<void>(options.get_inspector_rpc_server_port()), std::runtime_error);
}

TEST_F(InspectorPortTest, CustomBasePortWithRank) {
    // Custom base port via TT_METAL_INSPECTOR_RPC_SERVER_ADDRESS
    setenv("TT_METAL_INSPECTOR_RPC_SERVER_ADDRESS", "localhost:60000", 1);
    setenv("OMPI_COMM_WORLD_RANK", "10", 1);
    RunTimeOptions options;
    EXPECT_EQ(options.get_inspector_rpc_server_port(), 60010);  // 60000 + 10
}

TEST_F(InspectorPortTest, CustomBasePortOverflowThrows) {
    // Custom base port close to max with rank should fail fast.
    setenv("TT_METAL_INSPECTOR_RPC_SERVER_ADDRESS", "localhost:65530", 1);
    setenv("OMPI_COMM_WORLD_RANK", "100", 1);
    RunTimeOptions options;
    EXPECT_THROW(static_cast<void>(options.get_inspector_rpc_server_port()), std::runtime_error);
}

TEST_F(InspectorPortTest, InvalidRankIgnored) {
    // Invalid rank string should result in no rank offset
    setenv("OMPI_COMM_WORLD_RANK", "invalid", 1);
    RunTimeOptions options;
    EXPECT_EQ(options.get_inspector_rpc_server_port(), 50051);  // Base port, no offset
}

TEST_F(InspectorPortTest, RankZeroUsesBasePort) {
    // Rank 0 should use exactly the base port
    setenv("OMPI_COMM_WORLD_RANK", "0", 1);
    RunTimeOptions options;
    EXPECT_EQ(options.get_inspector_rpc_server_port(), 50051);  // 50051 + 0
}

TEST_F(InspectorPortTest, ServerAddressIncludesRankAwarePort) {
    // get_inspector_rpc_server_address should use the rank-aware port
    setenv("OMPI_COMM_WORLD_RANK", "7", 1);
    RunTimeOptions options;
    std::string expected = "localhost:50058";  // 50051 + 7
    EXPECT_EQ(options.get_inspector_rpc_server_address(), expected);
}

}  // namespace tt::llrt
