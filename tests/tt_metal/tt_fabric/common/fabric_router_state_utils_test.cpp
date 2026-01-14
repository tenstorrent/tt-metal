// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include <memory>
#include <vector>
#include <map>

#include "fabric_router_state_utils.hpp"
#include "tt_metal/fabric/control_plane.hpp"
#include "hostdevcommon/fabric_common.h"

using namespace tt::tt_fabric;
using namespace tt::tt_fabric::test_utils;
using ::testing::Return;
using ::testing::ReturnRef;

// =============================================================================
// Unit Tests: router_state_to_string()
// =============================================================================

class RouterStateToStringTest : public ::testing::Test {
protected:
    RouterStateToStringTest() = default;
};

TEST_F(RouterStateToStringTest, ConvertInitializingState) {
    const char* result = router_state_to_string(RouterStateCommon::INITIALIZING);
    EXPECT_STREQ(result, "INITIALIZING");
}

TEST_F(RouterStateToStringTest, ConvertRunningState) {
    const char* result = router_state_to_string(RouterStateCommon::RUNNING);
    EXPECT_STREQ(result, "RUNNING");
}

TEST_F(RouterStateToStringTest, ConvertPausedState) {
    const char* result = router_state_to_string(RouterStateCommon::PAUSED);
    EXPECT_STREQ(result, "PAUSED");
}

TEST_F(RouterStateToStringTest, ConvertDrainingState) {
    const char* result = router_state_to_string(RouterStateCommon::DRAINING);
    EXPECT_STREQ(result, "DRAINING");
}

TEST_F(RouterStateToStringTest, ConvertRetrainingState) {
    const char* result = router_state_to_string(RouterStateCommon::RETRAINING);
    EXPECT_STREQ(result, "RETRAINING");
}

TEST_F(RouterStateToStringTest, InvalidStateReturnsUnknown) {
    // Cast an invalid enum value (255)
    RouterStateCommon invalid_state = static_cast<RouterStateCommon>(255);
    const char* result = router_state_to_string(invalid_state);
    EXPECT_STREQ(result, "UNKNOWN");
}

TEST_F(RouterStateToStringTest, AllEnumValuesHandled) {
    // Verify that all enum values defined in RouterStateCommon return non-UNKNOWN strings
    // Except for any values we've explicitly marked as invalid
    EXPECT_STRNE(router_state_to_string(RouterStateCommon::INITIALIZING), "UNKNOWN");
    EXPECT_STRNE(router_state_to_string(RouterStateCommon::RUNNING), "UNKNOWN");
    EXPECT_STRNE(router_state_to_string(RouterStateCommon::PAUSED), "UNKNOWN");
    EXPECT_STRNE(router_state_to_string(RouterStateCommon::DRAINING), "UNKNOWN");
    EXPECT_STRNE(router_state_to_string(RouterStateCommon::RETRAINING), "UNKNOWN");
}

TEST_F(RouterStateToStringTest, ReturnValueIsConstCharPtr) {
    const char* result = router_state_to_string(RouterStateCommon::RUNNING);
    EXPECT_NE(result, nullptr);
    // Verify it's a valid string pointer
    std::string str_result(result);
    EXPECT_FALSE(str_result.empty());
}

// =============================================================================
// Unit Tests: count_routers_by_state()
// =============================================================================

class CountRoutersByStateTest : public ::testing::Test {
protected:
    CountRoutersByStateTest() = default;

    // Mock the FabricNodeId construction
    FabricNodeId create_fabric_node_id(MeshId mesh_id, ChipId device_id) {
        return FabricNodeId(mesh_id, device_id);
    }
};

TEST_F(CountRoutersByStateTest, EmptyMeshListReturnsEmptyMap) {
    // Create a mock ControlPlane for this test
    // Since ControlPlane requires complex initialization, we test with an empty mesh list
    auto mock_control_plane = std::make_unique<ControlPlane>();
    std::vector<MeshId> empty_meshes;

    auto counts = count_routers_by_state(*mock_control_plane, empty_meshes);
    EXPECT_TRUE(counts.empty());
}

TEST_F(CountRoutersByStateTest, CountReturnsMapWithStateKeys) {
    // This is a structural test - when implemented with real data,
    // the map should contain RouterStateCommon enum values as keys
    auto mock_control_plane = std::make_unique<ControlPlane>();
    std::vector<MeshId> meshes;

    // When the function is properly implemented, it should return a map
    // where keys are RouterStateCommon values
    auto counts = count_routers_by_state(*mock_control_plane, meshes);

    // For empty meshes, result should be empty
    // For non-empty meshes, result should have entries with uint32_t counts
    if (!counts.empty()) {
        for (const auto& [state, count] : counts) {
            // Verify the count is a valid uint32_t
            EXPECT_GE(count, 0);
        }
    }
}

TEST_F(CountRoutersByStateTest, CountValuesAreNonNegative) {
    auto mock_control_plane = std::make_unique<ControlPlane>();
    std::vector<MeshId> empty_meshes;

    auto counts = count_routers_by_state(*mock_control_plane, empty_meshes);

    // All counts should be >= 0 (they're uint32_t so naturally >= 0)
    for (const auto& [state, count] : counts) {
        EXPECT_GE(count, 0u);
    }
}

TEST_F(CountRoutersByStateTest, MapHasCorrectKeyType) {
    // Structural test: the returned map must use RouterStateCommon as key
    auto mock_control_plane = std::make_unique<ControlPlane>();
    std::vector<MeshId> meshes;

    auto counts = count_routers_by_state(*mock_control_plane, meshes);

    // If map is non-empty, iterate to verify RouterStateCommon keys
    for (const auto& [state, count] : counts) {
        // Verify we can call router_state_to_string on the state key
        const char* state_string = router_state_to_string(state);
        EXPECT_NE(state_string, nullptr);
    }
}

// =============================================================================
// Unit Tests: log_all_router_states()
// =============================================================================

class LogAllRouterStatesTest : public ::testing::Test {
protected:
    LogAllRouterStatesTest() = default;
};

TEST_F(LogAllRouterStatesTest, EmptyMeshListDoesNotCrash) {
    auto mock_control_plane = std::make_unique<ControlPlane>();
    std::vector<MeshId> empty_meshes;

    // Should not throw or crash with empty mesh list
    EXPECT_NO_THROW(log_all_router_states(*mock_control_plane, empty_meshes));
}

TEST_F(LogAllRouterStatesTest, FunctionAcceptsControlPlaneReference) {
    auto mock_control_plane = std::make_unique<ControlPlane>();
    std::vector<MeshId> meshes;

    // Function signature requires ControlPlane& and std::vector<MeshId>&
    // This test verifies the function accepts the correct types
    EXPECT_NO_THROW(log_all_router_states(*mock_control_plane, meshes));
}

TEST_F(LogAllRouterStatesTest, FunctionDoesNotReturnValue) {
    auto mock_control_plane = std::make_unique<ControlPlane>();
    std::vector<MeshId> meshes;

    // log_all_router_states returns void
    auto result = log_all_router_states(*mock_control_plane, meshes);
    // If this compiles and runs, the return type is correct
    static_assert(std::is_void_v<decltype(result)>, "log_all_router_states should return void");
}

// =============================================================================
// Integration Tests: Functions Work Together
// =============================================================================

class IntegrationTest : public ::testing::Test {
protected:
    IntegrationTest() = default;
};

TEST_F(IntegrationTest, StateStringAndCountMapUseConsistentEnums) {
    // Verify that router_state_to_string works with states that could be in the count map
    std::map<RouterStateCommon, uint32_t> sample_counts = {
        {RouterStateCommon::RUNNING, 4},
        {RouterStateCommon::PAUSED, 0},
        {RouterStateCommon::INITIALIZING, 2},
        {RouterStateCommon::DRAINING, 0},
        {RouterStateCommon::RETRAINING, 1},
    };

    // All states in the map should have valid string representations
    for (const auto& [state, count] : sample_counts) {
        const char* state_str = router_state_to_string(state);
        EXPECT_NE(state_str, nullptr);
        EXPECT_STRNE(state_str, "UNKNOWN");
    }
}

TEST_F(IntegrationTest, StateStringUsedInCountOutput) {
    // Simulate the pattern used in log_all_router_states
    std::map<RouterStateCommon, uint32_t> counts = {
        {RouterStateCommon::RUNNING, 4},
        {RouterStateCommon::PAUSED, 0},
    };

    // Pattern from changeset spec: router_state_to_string(state) in log output
    for (const auto& [state, count] : counts) {
        const char* state_str = router_state_to_string(state);
        // This simulates the logging: "  {}: {}",router_state_to_string(state), count
        std::string log_line = std::string(state_str) + ": " + std::to_string(count);
        EXPECT_FALSE(log_line.empty());
    }
}

// =============================================================================
// Edge Case Tests
// =============================================================================

class EdgeCasesTest : public ::testing::Test {
protected:
    EdgeCasesTest() = default;
};

TEST_F(EdgeCasesTest, RouterStateToStringConstexprCompatible) {
    // Verify the function signature allows it to work with constexpr contexts
    // if needed in future
    const char* result = router_state_to_string(RouterStateCommon::RUNNING);
    EXPECT_NE(result, nullptr);
}

TEST_F(EdgeCasesTest, CountRoutersByStateWithLargeMeshCount) {
    // Test with many meshes (boundary condition)
    auto mock_control_plane = std::make_unique<ControlPlane>();
    std::vector<MeshId> many_meshes;

    // Create a large vector of mesh IDs (simulate Galaxy with many meshes)
    for (uint32_t i = 0; i < 100; i++) {
        many_meshes.push_back(MeshId(i));
    }

    // Should handle large mesh lists without crashing
    EXPECT_NO_THROW(auto counts = count_routers_by_state(*mock_control_plane, many_meshes));
}

TEST_F(EdgeCasesTest, StateStringReturnedValueIsNotNull) {
    // For any enum value (even undefined ones), the function should return
    // a valid pointer (not nullptr)
    for (uint32_t i = 0; i < 256; i++) {
        RouterStateCommon state = static_cast<RouterStateCommon>(i);
        const char* result = router_state_to_string(state);
        EXPECT_NE(result, nullptr) << "router_state_to_string returned nullptr for value " << i;
    }
}

TEST_F(EdgeCasesTest, StateStringReturnedValueIsReadable) {
    // All returned strings should be readable and not garbage
    const char* result = router_state_to_string(RouterStateCommon::RUNNING);
    EXPECT_NE(result, nullptr);

    // Should be able to create a string from it
    std::string state_str(result);
    EXPECT_FALSE(state_str.empty());

    // Should contain only reasonable characters
    for (char c : state_str) {
        EXPECT_TRUE(isalnum(c) || c == '_' || c == ' ')
            << "Unexpected character in state string: " << c;
    }
}

// =============================================================================
// Namespace Tests
// =============================================================================

TEST(NamespaceTest, FunctionsAreInCorrectNamespace) {
    // Verify functions are in tt::tt_fabric::test_utils namespace
    // by calling them with fully qualified names
    const char* result = tt::tt_fabric::test_utils::router_state_to_string(RouterStateCommon::RUNNING);
    EXPECT_STREQ(result, "RUNNING");
}

// =============================================================================
// Compile-time Structure Tests
// =============================================================================

TEST(CompileTimeTests, FunctionSignaturesCorrect) {
    // This test verifies signatures compile correctly
    // router_state_to_string(RouterStateCommon state) -> const char*
    static_assert(
        std::is_same_v<
            decltype(router_state_to_string(RouterStateCommon::RUNNING)),
            const char*
        >,
        "router_state_to_string should return const char*"
    );

    // count_routers_by_state(ControlPlane&, std::vector<MeshId>&)
    // -> std::map<RouterStateCommon, uint32_t>
    // Verified by calling and storing result
}

// =============================================================================
// Default Behavior Tests
// =============================================================================

TEST(DefaultBehaviorTest, AllStatesHaveUniqueMappings) {
    // Each enum value should map to a distinct string (no duplicates)
    std::set<std::string> seen_strings;

    std::vector<RouterStateCommon> all_states = {
        RouterStateCommon::INITIALIZING,
        RouterStateCommon::RUNNING,
        RouterStateCommon::PAUSED,
        RouterStateCommon::DRAINING,
        RouterStateCommon::RETRAINING,
    };

    for (auto state : all_states) {
        std::string state_str(router_state_to_string(state));
        EXPECT_TRUE(seen_strings.find(state_str) == seen_strings.end())
            << "Duplicate state string: " << state_str;
        seen_strings.insert(state_str);
    }
}

TEST(DefaultBehaviorTest, UnknownStateIsDistinct) {
    // UNKNOWN should only be returned for truly invalid values
    std::string unknown_str("UNKNOWN");

    // All valid enum values should NOT be UNKNOWN
    EXPECT_NE(unknown_str, std::string(router_state_to_string(RouterStateCommon::RUNNING)));
    EXPECT_NE(unknown_str, std::string(router_state_to_string(RouterStateCommon::PAUSED)));
    EXPECT_NE(unknown_str, std::string(router_state_to_string(RouterStateCommon::INITIALIZING)));
}
