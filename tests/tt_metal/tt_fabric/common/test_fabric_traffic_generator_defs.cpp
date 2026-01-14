// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>
#include <chrono>
#include <cstdint>
#include <type_traits>
#include "fabric_traffic_generator_defs.hpp"

namespace tt::tt_fabric::test_utils {

// Test suite for fabric_traffic_generator_defs.hpp scaffolding
class FabricTrafficGeneratorDefsTest : public ::testing::Test {};

// ============================================================================
// Constant Value Verification Tests
// ============================================================================

TEST_F(FabricTrafficGeneratorDefsTest, WorkerKeepRunningConstantValue) {
    // Verify WORKER_KEEP_RUNNING is correctly defined as 0
    EXPECT_EQ(WORKER_KEEP_RUNNING, 0u);
}

TEST_F(FabricTrafficGeneratorDefsTest, WorkerTeardownConstantValue) {
    // Verify WORKER_TEARDOWN is correctly defined as 1
    EXPECT_EQ(WORKER_TEARDOWN, 1u);
}

TEST_F(FabricTrafficGeneratorDefsTest, WorkerStateConstantsAreDistinct) {
    // Verify that WORKER_KEEP_RUNNING and WORKER_TEARDOWN are different
    EXPECT_NE(WORKER_KEEP_RUNNING, WORKER_TEARDOWN);
}

TEST_F(FabricTrafficGeneratorDefsTest, WorkerStateConstantsAreUint32) {
    // Verify constants are uint32_t compatible
    static_assert(std::is_same_v<decltype(WORKER_KEEP_RUNNING), const uint32_t>);
    static_assert(std::is_same_v<decltype(WORKER_TEARDOWN), const uint32_t>);
}

// ============================================================================
// Timing Constant Verification Tests
// ============================================================================

TEST_F(FabricTrafficGeneratorDefsTest, DefaultTrafficSampleIntervalValue) {
    // Verify DEFAULT_TRAFFIC_SAMPLE_INTERVAL is correctly set to 100ms
    EXPECT_EQ(DEFAULT_TRAFFIC_SAMPLE_INTERVAL.count(), 100);
}

TEST_F(FabricTrafficGeneratorDefsTest, DefaultTrafficSampleIntervalType) {
    // Verify DEFAULT_TRAFFIC_SAMPLE_INTERVAL is std::chrono::milliseconds
    static_assert(
        std::is_same_v<decltype(DEFAULT_TRAFFIC_SAMPLE_INTERVAL), const std::chrono::milliseconds>);
}

TEST_F(FabricTrafficGeneratorDefsTest, DefaultPauseTimeoutValue) {
    // Verify DEFAULT_PAUSE_TIMEOUT is correctly set to 5000ms
    EXPECT_EQ(DEFAULT_PAUSE_TIMEOUT.count(), 5000);
}

TEST_F(FabricTrafficGeneratorDefsTest, DefaultPauseTimeoutType) {
    // Verify DEFAULT_PAUSE_TIMEOUT is std::chrono::milliseconds
    static_assert(std::is_same_v<decltype(DEFAULT_PAUSE_TIMEOUT), const std::chrono::milliseconds>);
}

TEST_F(FabricTrafficGeneratorDefsTest, DefaultPollIntervalValue) {
    // Verify DEFAULT_POLL_INTERVAL is correctly set to 100ms
    EXPECT_EQ(DEFAULT_POLL_INTERVAL.count(), 100);
}

TEST_F(FabricTrafficGeneratorDefsTest, DefaultPollIntervalType) {
    // Verify DEFAULT_POLL_INTERVAL is std::chrono::milliseconds
    static_assert(std::is_same_v<decltype(DEFAULT_POLL_INTERVAL), const std::chrono::milliseconds>);
}

TEST_F(FabricTrafficGeneratorDefsTest, TimingConstantsHaveCorrectRelationships) {
    // Verify that timing constants have sensible relationships
    // DEFAULT_PAUSE_TIMEOUT should be larger than DEFAULT_TRAFFIC_SAMPLE_INTERVAL
    EXPECT_GT(DEFAULT_PAUSE_TIMEOUT, DEFAULT_TRAFFIC_SAMPLE_INTERVAL);
    // DEFAULT_POLL_INTERVAL should be equal to DEFAULT_TRAFFIC_SAMPLE_INTERVAL
    EXPECT_EQ(DEFAULT_POLL_INTERVAL, DEFAULT_TRAFFIC_SAMPLE_INTERVAL);
}

// ============================================================================
// TrafficGeneratorCompileArgs Structure Tests
// ============================================================================

TEST_F(FabricTrafficGeneratorDefsTest, TrafficGeneratorCompileArgsIsDefinable) {
    // Verify the struct can be instantiated
    TrafficGeneratorCompileArgs args;
    EXPECT_TRUE(sizeof(args) > 0);
}

TEST_F(FabricTrafficGeneratorDefsTest, TrafficGeneratorCompileArgsHasCorrectFields) {
    // Verify all required fields are present in TrafficGeneratorCompileArgs
    TrafficGeneratorCompileArgs args{};

    // These should compile and initialize without error
    args.source_buffer_address = 0x1000;
    args.packet_payload_size_bytes = 64;
    args.target_noc_encoding = 0;
    args.teardown_signal_address = 0x2000;
    args.is_2d_fabric = 1;

    EXPECT_EQ(args.source_buffer_address, 0x1000u);
    EXPECT_EQ(args.packet_payload_size_bytes, 64u);
    EXPECT_EQ(args.target_noc_encoding, 0u);
    EXPECT_EQ(args.teardown_signal_address, 0x2000u);
    EXPECT_EQ(args.is_2d_fabric, 1u);
}

TEST_F(FabricTrafficGeneratorDefsTest, TrafficGeneratorCompileArgsAllFieldsAreUint32) {
    // Verify all fields in TrafficGeneratorCompileArgs are uint32_t
    static_assert(std::is_same_v<decltype(TrafficGeneratorCompileArgs::source_buffer_address), uint32_t>);
    static_assert(
        std::is_same_v<decltype(TrafficGeneratorCompileArgs::packet_payload_size_bytes), uint32_t>);
    static_assert(std::is_same_v<decltype(TrafficGeneratorCompileArgs::target_noc_encoding), uint32_t>);
    static_assert(
        std::is_same_v<decltype(TrafficGeneratorCompileArgs::teardown_signal_address), uint32_t>);
    static_assert(std::is_same_v<decltype(TrafficGeneratorCompileArgs::is_2d_fabric), uint32_t>);
}

TEST_F(FabricTrafficGeneratorDefsTest, TrafficGeneratorCompileArgsDefaultConstruction) {
    // Verify default construction works
    TrafficGeneratorCompileArgs args{};
    // Values should be zero-initialized
    EXPECT_EQ(args.source_buffer_address, 0u);
    EXPECT_EQ(args.packet_payload_size_bytes, 0u);
    EXPECT_EQ(args.target_noc_encoding, 0u);
    EXPECT_EQ(args.teardown_signal_address, 0u);
    EXPECT_EQ(args.is_2d_fabric, 0u);
}

TEST_F(FabricTrafficGeneratorDefsTest, TrafficGeneratorCompileArgsAggregateConstruction) {
    // Verify aggregate/list initialization works
    TrafficGeneratorCompileArgs args{0x1000, 64, 0, 0x2000, 1};
    EXPECT_EQ(args.source_buffer_address, 0x1000u);
    EXPECT_EQ(args.packet_payload_size_bytes, 64u);
    EXPECT_EQ(args.target_noc_encoding, 0u);
    EXPECT_EQ(args.teardown_signal_address, 0x2000u);
    EXPECT_EQ(args.is_2d_fabric, 1u);
}

TEST_F(FabricTrafficGeneratorDefsTest, TrafficGeneratorCompileArgsSize) {
    // Verify struct size is reasonable (5 uint32_t fields = 20 bytes)
    EXPECT_EQ(sizeof(TrafficGeneratorCompileArgs), 5 * sizeof(uint32_t));
}

// ============================================================================
// TrafficGeneratorRuntimeArgs Structure Tests
// ============================================================================

TEST_F(FabricTrafficGeneratorDefsTest, TrafficGeneratorRuntimeArgsIsDefinable) {
    // Verify the struct can be instantiated
    TrafficGeneratorRuntimeArgs args;
    EXPECT_TRUE(sizeof(args) > 0);
}

TEST_F(FabricTrafficGeneratorDefsTest, TrafficGeneratorRuntimeArgsHasCorrectFields) {
    // Verify all required fields are present in TrafficGeneratorRuntimeArgs
    TrafficGeneratorRuntimeArgs args{};

    // These should compile and initialize without error
    args.dest_chip_id = 1;
    args.dest_mesh_id = 0;
    args.random_seed = 42;

    EXPECT_EQ(args.dest_chip_id, 1u);
    EXPECT_EQ(args.dest_mesh_id, 0u);
    EXPECT_EQ(args.random_seed, 42u);
}

TEST_F(FabricTrafficGeneratorDefsTest, TrafficGeneratorRuntimeArgsAllFieldsAreUint32) {
    // Verify all fields in TrafficGeneratorRuntimeArgs are uint32_t
    static_assert(std::is_same_v<decltype(TrafficGeneratorRuntimeArgs::dest_chip_id), uint32_t>);
    static_assert(std::is_same_v<decltype(TrafficGeneratorRuntimeArgs::dest_mesh_id), uint32_t>);
    static_assert(std::is_same_v<decltype(TrafficGeneratorRuntimeArgs::random_seed), uint32_t>);
}

TEST_F(FabricTrafficGeneratorDefsTest, TrafficGeneratorRuntimeArgsDefaultConstruction) {
    // Verify default construction works
    TrafficGeneratorRuntimeArgs args{};
    // Values should be zero-initialized
    EXPECT_EQ(args.dest_chip_id, 0u);
    EXPECT_EQ(args.dest_mesh_id, 0u);
    EXPECT_EQ(args.random_seed, 0u);
}

TEST_F(FabricTrafficGeneratorDefsTest, TrafficGeneratorRuntimeArgsAggregateConstruction) {
    // Verify aggregate/list initialization works
    TrafficGeneratorRuntimeArgs args{1, 0, 42};
    EXPECT_EQ(args.dest_chip_id, 1u);
    EXPECT_EQ(args.dest_mesh_id, 0u);
    EXPECT_EQ(args.random_seed, 42u);
}

TEST_F(FabricTrafficGeneratorDefsTest, TrafficGeneratorRuntimeArgsSize) {
    // Verify struct size is reasonable (3 uint32_t fields = 12 bytes)
    EXPECT_EQ(sizeof(TrafficGeneratorRuntimeArgs), 3 * sizeof(uint32_t));
}

// ============================================================================
// Namespace Resolution Tests
// ============================================================================

TEST_F(FabricTrafficGeneratorDefsTest, NamespaceResolution) {
    // Verify we can access elements through the correct namespace
    using tt::tt_fabric::test_utils::WORKER_KEEP_RUNNING;
    using tt::tt_fabric::test_utils::WORKER_TEARDOWN;
    using tt::tt_fabric::test_utils::DEFAULT_TRAFFIC_SAMPLE_INTERVAL;
    using tt::tt_fabric::test_utils::DEFAULT_PAUSE_TIMEOUT;
    using tt::tt_fabric::test_utils::DEFAULT_POLL_INTERVAL;
    using tt::tt_fabric::test_utils::TrafficGeneratorCompileArgs;
    using tt::tt_fabric::test_utils::TrafficGeneratorRuntimeArgs;

    EXPECT_EQ(WORKER_KEEP_RUNNING, 0u);
    EXPECT_EQ(WORKER_TEARDOWN, 1u);
    EXPECT_EQ(DEFAULT_TRAFFIC_SAMPLE_INTERVAL.count(), 100);
    EXPECT_EQ(DEFAULT_PAUSE_TIMEOUT.count(), 5000);
    EXPECT_EQ(DEFAULT_POLL_INTERVAL.count(), 100);
}

// ============================================================================
// Type Safety and Trait Tests
// ============================================================================

TEST_F(FabricTrafficGeneratorDefsTest, TrafficGeneratorCompileArgsIsStandardLayout) {
    // Verify struct is standard layout (can be used for memory mapping)
    static_assert(std::is_standard_layout_v<TrafficGeneratorCompileArgs>);
}

TEST_F(FabricTrafficGeneratorDefsTest, TrafficGeneratorRuntimeArgsIsStandardLayout) {
    // Verify struct is standard layout (can be used for memory mapping)
    static_assert(std::is_standard_layout_v<TrafficGeneratorRuntimeArgs>);
}

TEST_F(FabricTrafficGeneratorDefsTest, TrafficGeneratorCompileArgsIsTriviallyCopyable) {
    // Verify struct is trivially copyable (can be memcpy'd)
    static_assert(std::is_trivially_copyable_v<TrafficGeneratorCompileArgs>);
}

TEST_F(FabricTrafficGeneratorDefsTest, TrafficGeneratorRuntimeArgsIsTriviallyCopyable) {
    // Verify struct is trivially copyable (can be memcpy'd)
    static_assert(std::is_trivially_copyable_v<TrafficGeneratorRuntimeArgs>);
}

TEST_F(FabricTrafficGeneratorDefsTest, TrafficGeneratorCompileArgsIsAggregate) {
    // Verify struct can be used with aggregate initialization
    static_assert(std::is_aggregate_v<TrafficGeneratorCompileArgs>);
}

TEST_F(FabricTrafficGeneratorDefsTest, TrafficGeneratorRuntimeArgsIsAggregate) {
    // Verify struct can be used with aggregate initialization
    static_assert(std::is_aggregate_v<TrafficGeneratorRuntimeArgs>);
}

// ============================================================================
// Integration Tests - Multiple Structs
// ============================================================================

TEST_F(FabricTrafficGeneratorDefsTest, CanCreateBothStructsTogether) {
    // Verify both compile and runtime arg structs can coexist and be used together
    TrafficGeneratorCompileArgs compile_args{0x1000, 64, 0, 0x2000, 1};
    TrafficGeneratorRuntimeArgs runtime_args{1, 0, 42};

    EXPECT_NE(static_cast<void*>(&compile_args), static_cast<void*>(&runtime_args));
    EXPECT_EQ(sizeof(TrafficGeneratorCompileArgs), 20u);
    EXPECT_EQ(sizeof(TrafficGeneratorRuntimeArgs), 12u);
}

TEST_F(FabricTrafficGeneratorDefsTest, StructsCanBeStoredInArrays) {
    // Verify structs can be stored in arrays
    std::array<TrafficGeneratorCompileArgs, 5> compile_args_array{};
    std::array<TrafficGeneratorRuntimeArgs, 3> runtime_args_array{};

    compile_args_array[0] = {0x1000, 64, 0, 0x2000, 1};
    runtime_args_array[0] = {1, 0, 42};

    EXPECT_EQ(compile_args_array[0].source_buffer_address, 0x1000u);
    EXPECT_EQ(runtime_args_array[0].dest_chip_id, 1u);
}

TEST_F(FabricTrafficGeneratorDefsTest, StructsCanBeStoredInVectors) {
    // Verify structs can be stored in vectors
    std::vector<TrafficGeneratorCompileArgs> compile_args_vec;
    std::vector<TrafficGeneratorRuntimeArgs> runtime_args_vec;

    compile_args_vec.push_back({0x1000, 64, 0, 0x2000, 1});
    runtime_args_vec.push_back({1, 0, 42});

    EXPECT_EQ(compile_args_vec.size(), 1u);
    EXPECT_EQ(runtime_args_vec.size(), 1u);
    EXPECT_EQ(compile_args_vec[0].source_buffer_address, 0x1000u);
    EXPECT_EQ(runtime_args_vec[0].dest_chip_id, 1u);
}

// ============================================================================
// Edge Case and Boundary Tests
// ============================================================================

TEST_F(FabricTrafficGeneratorDefsTest, TrafficGeneratorCompileArgsMaxValues) {
    // Verify struct can hold maximum uint32_t values
    TrafficGeneratorCompileArgs args{
        std::numeric_limits<uint32_t>::max(),
        std::numeric_limits<uint32_t>::max(),
        std::numeric_limits<uint32_t>::max(),
        std::numeric_limits<uint32_t>::max(),
        std::numeric_limits<uint32_t>::max()};

    EXPECT_EQ(args.source_buffer_address, std::numeric_limits<uint32_t>::max());
    EXPECT_EQ(args.packet_payload_size_bytes, std::numeric_limits<uint32_t>::max());
    EXPECT_EQ(args.target_noc_encoding, std::numeric_limits<uint32_t>::max());
    EXPECT_EQ(args.teardown_signal_address, std::numeric_limits<uint32_t>::max());
    EXPECT_EQ(args.is_2d_fabric, std::numeric_limits<uint32_t>::max());
}

TEST_F(FabricTrafficGeneratorDefsTest, TrafficGeneratorRuntimeArgsMaxValues) {
    // Verify struct can hold maximum uint32_t values
    TrafficGeneratorRuntimeArgs args{
        std::numeric_limits<uint32_t>::max(),
        std::numeric_limits<uint32_t>::max(),
        std::numeric_limits<uint32_t>::max()};

    EXPECT_EQ(args.dest_chip_id, std::numeric_limits<uint32_t>::max());
    EXPECT_EQ(args.dest_mesh_id, std::numeric_limits<uint32_t>::max());
    EXPECT_EQ(args.random_seed, std::numeric_limits<uint32_t>::max());
}

TEST_F(FabricTrafficGeneratorDefsTest, TrafficGeneratorCompileArgsZeroValues) {
    // Verify struct can hold zero values
    TrafficGeneratorCompileArgs args{0, 0, 0, 0, 0};

    EXPECT_EQ(args.source_buffer_address, 0u);
    EXPECT_EQ(args.packet_payload_size_bytes, 0u);
    EXPECT_EQ(args.target_noc_encoding, 0u);
    EXPECT_EQ(args.teardown_signal_address, 0u);
    EXPECT_EQ(args.is_2d_fabric, 0u);
}

TEST_F(FabricTrafficGeneratorDefsTest, TrafficGeneratorRuntimeArgsZeroValues) {
    // Verify struct can hold zero values
    TrafficGeneratorRuntimeArgs args{0, 0, 0};

    EXPECT_EQ(args.dest_chip_id, 0u);
    EXPECT_EQ(args.dest_mesh_id, 0u);
    EXPECT_EQ(args.random_seed, 0u);
}

// ============================================================================
// Compile-Time Verification Tests
// ============================================================================

TEST_F(FabricTrafficGeneratorDefsTest, ConstantsAreCompileTimeConstant) {
    // Verify constants can be used in constexpr contexts
    constexpr uint32_t keep_running = WORKER_KEEP_RUNNING;
    constexpr uint32_t teardown = WORKER_TEARDOWN;
    constexpr auto sample_interval = DEFAULT_TRAFFIC_SAMPLE_INTERVAL;
    constexpr auto pause_timeout = DEFAULT_PAUSE_TIMEOUT;
    constexpr auto poll_interval = DEFAULT_POLL_INTERVAL;

    EXPECT_EQ(keep_running, 0u);
    EXPECT_EQ(teardown, 1u);
    EXPECT_EQ(sample_interval.count(), 100);
    EXPECT_EQ(pause_timeout.count(), 5000);
    EXPECT_EQ(poll_interval.count(), 100);
}

TEST_F(FabricTrafficGeneratorDefsTest, TimingConstantsCanBeUsedInConstexprComparisons) {
    // Verify timing constants can be used in compile-time comparisons
    static_assert(DEFAULT_PAUSE_TIMEOUT > DEFAULT_TRAFFIC_SAMPLE_INTERVAL);
    static_assert(DEFAULT_POLL_INTERVAL == DEFAULT_TRAFFIC_SAMPLE_INTERVAL);
    static_assert(DEFAULT_PAUSE_TIMEOUT.count() == 5000);
    static_assert(DEFAULT_TRAFFIC_SAMPLE_INTERVAL.count() == 100);
    static_assert(DEFAULT_POLL_INTERVAL.count() == 100);
}

}  // namespace tt::tt_fabric::test_utils
