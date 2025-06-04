// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <fmt/base.h>
#include <stddef.h>
#include <tt-metalium/logger.hpp>
#include <array>
#include <cstdint>
#include <optional>
#include <set>
#include <sstream>
#include <string>
#include <tuple>
#include <unordered_map>
#include <utility>

#include <tt-metalium/buffer.hpp>
#include <tt-metalium/buffer_types.hpp>
#include <tt-metalium/core_coord.hpp>
#include "gtest/gtest.h"
#include <tt-metalium/shape.hpp>
#include <tt-metalium/shape_base.hpp>
#include "impl/context/metal_context.hpp"
#include "ttnn/decorators.hpp"
#include "ttnn/graph/graph_query_op_constraints.hpp"
#include "ttnn/graph/graph_trace_utils.hpp"
#include "ttnn/operations/conv/conv2d/conv2d.hpp"
#include "ttnn/operations/eltwise/binary/binary.hpp"
#include "ttnn/operations/eltwise/unary/common/unary_op_types.hpp"
#include "ttnn/operations/eltwise/unary/unary.hpp"
#include "ttnn/operations/functions.hpp"
#include "ttnn/operations/matmul/device/matmul_op.hpp"
#include "ttnn/operations/matmul/matmul.hpp"
#include "ttnn/operations/normalization/softmax/softmax.hpp"
#include "ttnn/tensor/enum_types.hpp"
#include "ttnn/tensor/layout/page_config.hpp"
#include "ttnn/tensor/layout/tensor_layout.hpp"
#include "ttnn/tensor/shape/shape.hpp"
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/tensor/tensor_spec.hpp"
#include "ttnn/tensor/types.hpp"
#include "ttnn/types.hpp"
#include "ttnn_test_fixtures.hpp"
#include "umd/device/types/cluster_descriptor_types.h"

namespace tt {
namespace tt_metal {
class IDevice;
}  // namespace tt_metal
}  // namespace tt

namespace ttnn {
namespace operations {
namespace binary {
namespace test {

using ResourceUsageMap = std::unordered_map<BoardType, ttnn::graph::ResourceUsage>;

namespace detail {
static std::ostream& operator<<(std::ostream& os, const tt::tt_metal::TensorMemoryLayout& tensor_memory_layout) {
    switch (tensor_memory_layout) {
        case TensorMemoryLayout::INTERLEAVED: os << "I"; break;
        case TensorMemoryLayout::WIDTH_SHARDED: os << "WS"; break;
        case TensorMemoryLayout::HEIGHT_SHARDED: os << "HS"; break;
        case TensorMemoryLayout::BLOCK_SHARDED: os << "BS"; break;
        default: os << "U"; break;
    }
    return os;
}
static std::ostream& operator<<(std::ostream& os, const tt::tt_metal::BufferType& buffer_type) {
    switch (buffer_type) {
        case BufferType::DRAM: os << "DRAM"; break;
        case BufferType::L1: os << "L1"; break;
        default: os << "U"; break;
    }
    return os;
}

static std::ostream& operator<<(std::ostream& os, const tt::tt_metal::TensorLayout& tensor_layout) {
    os << tensor_layout.get_memory_config().buffer_type() << "_" << tensor_layout.get_memory_config().memory_layout();
    return os;
}

static std::ostream& operator<<(std::ostream& os, const ttnn::Shape& shape) {
    for (size_t i = 0; i < shape.rank(); i++) {
        if (i != 0) {
            os << "x";
        }
        os << std::to_string(shape[i]);
    }
    return os;
}

}  // namespace detail

// ============================================================================
// Test data
// ============================================================================

const auto g_height_shard_3_1_1024_1024_tiled_to_16_cores = ttnn::TensorSpec(
    ttnn::Shape(tt::tt_metal::Array4D{3, 1, 32 * 32, 32 * 32}),
    tt::tt_metal::TensorLayout(
        tt::tt_metal::DataType::BFLOAT16,
        tt::tt_metal::PageConfig(tt::tt_metal::Layout::TILE),
        tt::tt_metal::MemoryConfig{
            TensorMemoryLayout::HEIGHT_SHARDED,
            tt::tt_metal::BufferType::L1,
            tt::tt_metal::ShardSpec{
                CoreRangeSet{std::set<CoreRange>{CoreRange{CoreCoord{0, 0}, CoreCoord{3, 3}}}},
                {6 * 32, 32 * 32},
                ShardOrientation::ROW_MAJOR}}));

const auto g_height_shard_1_1_1024_32_tiled_to_32_cores = ttnn::TensorSpec(
    ttnn::Shape(tt::tt_metal::Array4D{1, 1, 32 * 32, 64}),
    tt::tt_metal::TensorLayout(
        tt::tt_metal::DataType::BFLOAT16,
        tt::tt_metal::PageConfig(tt::tt_metal::Layout::TILE),
        tt::tt_metal::MemoryConfig{
            TensorMemoryLayout::HEIGHT_SHARDED,
            tt::tt_metal::BufferType::L1,
            tt::tt_metal::ShardSpec{
                CoreRangeSet{std::set<CoreRange>{CoreRange{CoreCoord{0, 0}, CoreCoord{7, 3}}}},
                {32, 64},
                ShardOrientation::ROW_MAJOR}}));

const auto g_interleaved_1_1_1024_1024_tiled = ttnn::TensorSpec(
    ttnn::Shape(tt::tt_metal::Array4D{1, 1, 64, 128}),
    tt::tt_metal::TensorLayout(
        tt::tt_metal::DataType::BFLOAT16,
        tt::tt_metal::PageConfig(tt::tt_metal::Layout::TILE),
        ttnn::L1_MEMORY_CONFIG));

const auto g_interleave_4_2_160_244_tiled = ttnn::TensorSpec(
    ttnn::Shape(tt::tt_metal::Array4D{4, 2, 5 * 32, 7 * 32}),
    tt::tt_metal::TensorLayout(
        tt::tt_metal::DataType::BFLOAT16,
        tt::tt_metal::PageConfig(tt::tt_metal::Layout::TILE),
        ttnn::L1_MEMORY_CONFIG));

const auto g_interleave_1_2_160_244_tiled = ttnn::TensorSpec(
    ttnn::Shape(tt::tt_metal::Array4D{1, 1, 5 * 32, 7 * 32}),
    tt::tt_metal::TensorLayout(
        tt::tt_metal::DataType::BFLOAT16,
        tt::tt_metal::PageConfig(tt::tt_metal::Layout::TILE),
        ttnn::L1_MEMORY_CONFIG));

const auto g_interleave_1_1_160_244_tiled = ttnn::TensorSpec(
    ttnn::Shape(tt::tt_metal::Array4D{1, 1, 5 * 32, 7 * 32}),
    tt::tt_metal::TensorLayout(
        tt::tt_metal::DataType::BFLOAT16,
        tt::tt_metal::PageConfig(tt::tt_metal::Layout::TILE),
        ttnn::L1_MEMORY_CONFIG));

const auto g_interleaved_1_1_2048_64_tiled = ttnn::TensorSpec(
    ttnn::Shape(tt::tt_metal::Array4D{1, 1, 2048, 64}),
    tt::tt_metal::TensorLayout(
        tt::tt_metal::DataType::BFLOAT16,
        tt::tt_metal::PageConfig(tt::tt_metal::Layout::TILE),
        ttnn::L1_MEMORY_CONFIG));

const auto g_interleaved_1_1_245_1024_tiled = ttnn::TensorSpec(
    ttnn::Shape(tt::tt_metal::Array4D{1, 1, 256, 1024}),
    tt::tt_metal::TensorLayout(
        tt::tt_metal::DataType::BFLOAT16,
        tt::tt_metal::PageConfig(tt::tt_metal::Layout::TILE),
        ttnn::L1_MEMORY_CONFIG));

const auto g_width_shard_1_1_64_2048_tiled_to_32_cores = ttnn::TensorSpec(
    ttnn::Shape(tt::tt_metal::Array4D{1, 1, 64, 2048}),
    tt::tt_metal::TensorLayout(
        tt::tt_metal::DataType::BFLOAT16,
        tt::tt_metal::PageConfig(tt::tt_metal::Layout::TILE),
        tt::tt_metal::MemoryConfig{
            tt::tt_metal::TensorMemoryLayout::WIDTH_SHARDED,
            tt::tt_metal::BufferType::L1,
            tt::tt_metal::ShardSpec{
                CoreRangeSet{std::set<CoreRange>{CoreRange{CoreCoord{0, 0}, CoreCoord{7, 3}}}},
                {64, 64},
                ShardOrientation::ROW_MAJOR}}));

const auto g_block_shard_1_1_1600_256_tiled_to_32_cores = ttnn::TensorSpec(
    ttnn::Shape(tt::tt_metal::Array4D{1, 1, 1600, 256}),
    tt::tt_metal::TensorLayout(
        tt::tt_metal::DataType::BFLOAT16,
        tt::tt_metal::PageConfig(tt::tt_metal::Layout::TILE),
        tt::tt_metal::MemoryConfig{
            tt::tt_metal::TensorMemoryLayout::BLOCK_SHARDED,
            tt::tt_metal::BufferType::L1,
            tt::tt_metal::ShardSpec{
                CoreRangeSet{std::set<CoreRange>{CoreRange{CoreCoord{0, 0}, CoreCoord{7, 4}}}},
                {320, 32},
                ShardOrientation::ROW_MAJOR}}));

// ============================================================================
// Unary tests
// ============================================================================

class EltwiseUnaryOpIfTest : public TTNNFixtureWithDevice,
                             public testing::WithParamInterface<std::tuple<ttnn::TensorSpec, ResourceUsageMap>> {};

TEST_P(EltwiseUnaryOpIfTest, UnaryRelu) {
    const auto& input_spec = std::get<ttnn::TensorSpec>(GetParam());
    const auto& expected_resource_usage_map = std::get<ResourceUsageMap>(GetParam());
    const BoardType board_type = tt::tt_metal::MetalContext::instance().get_cluster().get_board_type(0);
    if (expected_resource_usage_map.count(board_type) == 0) {
        GTEST_SKIP();
    }
    const auto& expected_resource_usage = expected_resource_usage_map.at(board_type);

    // Run the test
    {
        tt::tt_metal::IDevice* device = device_;
        const auto& output_spec = input_spec;
        auto query = ttnn::graph::query_op_constraints(
            ttnn::relu, device, input_spec, output_spec.tensor_layout().get_memory_config());

        EXPECT_EQ(query.status, ttnn::graph::ExecutionStatus::Success);
        EXPECT_EQ(query.resource_usage.cb_peak_size_per_core, expected_resource_usage.cb_peak_size_per_core);
        EXPECT_EQ(query.resource_usage.l1_buffers_peak_per_core, expected_resource_usage.l1_buffers_peak_per_core);
        EXPECT_EQ(query.resource_usage.l1_output_buffer_per_core, expected_resource_usage.l1_output_buffer_per_core);
        ASSERT_TRUE(query.output_tensor_spec.has_value());
        EXPECT_EQ(query.output_tensor_spec.value(), input_spec);
    }
}

INSTANTIATE_TEST_SUITE_P(
    QueryOpConstraints,    // Prefix for the instantiated test suite
    EltwiseUnaryOpIfTest,  // Test suite name
    ::testing::Values(
        std::make_tuple(
            g_height_shard_3_1_1024_1024_tiled_to_16_cores,
            ResourceUsageMap{
                {BoardType::N300,
                 ttnn::graph::ResourceUsage{
                     .cb_peak_size_per_core = 0,
                     .l1_buffers_peak_per_core = 2 * (3 * 32 * 32 * 32 * 32) / 16,
                     .l1_output_buffer_per_core = 2 * (3 * 32 * 32 * 32 * 32) / 16}},
                {BoardType::E150,
                 ttnn::graph::ResourceUsage{
                     .cb_peak_size_per_core = 0,
                     .l1_buffers_peak_per_core = 2 * (3 * 32 * 32 * 32 * 32) / 16,
                     .l1_output_buffer_per_core = 2 * (3 * 32 * 32 * 32 * 32) / 16}}}),
        std::make_tuple(
            g_interleave_4_2_160_244_tiled,
            ResourceUsageMap{
                {BoardType::N300,
                 ttnn::graph::ResourceUsage{
                     .cb_peak_size_per_core = (2 * 2 * 2 * 32 * 32),
                     .l1_buffers_peak_per_core = 10240,
                     .l1_output_buffer_per_core = 10240}},
                {BoardType::E150,
                 ttnn::graph::ResourceUsage{
                     .cb_peak_size_per_core = (2 * 2 * 2 * 32 * 32),
                     .l1_buffers_peak_per_core = 6144,
                     .l1_output_buffer_per_core = 6144}}})),
    [](const testing::TestParamInfo<std::tuple<ttnn::TensorSpec, ResourceUsageMap>>& info) {
        std::stringstream ss;

        // print unique id for each test case
        static int uid = 0;
        ss << uid++;

        // print tensor layout
        using detail::operator<<;
        ss << "_" << std::get<ttnn::TensorSpec>(info.param).tensor_layout();

        // print tensor shape; operator<< exists but is too long to be used here
        ss << "_";
        detail::operator<<(ss, std::get<ttnn::TensorSpec>(info.param).logical_shape());
        return ss.str();
    });

// ============================================================================
// Softmax tests
// ============================================================================

class SoftmaxOpIfTest : public TTNNFixtureWithDevice,
                        public testing::WithParamInterface<std::tuple<ttnn::TensorSpec, int, ResourceUsageMap>> {};

TEST_P(SoftmaxOpIfTest, Softmax) {
    const auto& input_spec = std::get<ttnn::TensorSpec>(GetParam());
    const auto& dim_arg = std::get<int>(GetParam());
    const auto& expected_resource_usage_map = std::get<ResourceUsageMap>(GetParam());
    const BoardType board_type = tt::tt_metal::MetalContext::instance().get_cluster().get_board_type(0);
    if (expected_resource_usage_map.count(board_type) == 0) {
        GTEST_SKIP();
    }
    const auto& expected_resource_usage = expected_resource_usage_map.at(board_type);

    // Run the test
    {
        tt::tt_metal::IDevice* device = device_;
        const auto& output_spec = input_spec;
        auto query = ttnn::graph::query_op_constraints(
            ttnn::softmax, device, input_spec, dim_arg, output_spec.tensor_layout().get_memory_config());

        EXPECT_EQ(query.status, ttnn::graph::ExecutionStatus::Success);
        EXPECT_EQ(query.resource_usage.cb_peak_size_per_core, expected_resource_usage.cb_peak_size_per_core);
        EXPECT_EQ(query.resource_usage.l1_buffers_peak_per_core, expected_resource_usage.l1_buffers_peak_per_core);
        EXPECT_EQ(query.resource_usage.l1_output_buffer_per_core, expected_resource_usage.l1_output_buffer_per_core);
        EXPECT_EQ(query.output_tensor_spec.value(), input_spec);
    }
}

INSTANTIATE_TEST_SUITE_P(
    QueryOpConstraints,  // Prefix for the instantiated test suite
    SoftmaxOpIfTest,     // Test suite name
    ::testing::Values(
        std::make_tuple(
            g_height_shard_3_1_1024_1024_tiled_to_16_cores,
            -1,

            ResourceUsageMap{
                {BoardType::N300,
                 ttnn::graph::ResourceUsage{
                     .cb_peak_size_per_core = 2 * (1 * 32 * 32 * 32 * 32) / 16 + 3 * (2 * 32 * 32),
                     .l1_buffers_peak_per_core = 2 * (3 * 32 * 32 * 32 * 32) / 16,
                     .l1_output_buffer_per_core = 2 * (3 * 32 * 32 * 32 * 32) / 16}},
                {BoardType::E150,
                 ttnn::graph::ResourceUsage{
                     .cb_peak_size_per_core = 2 * (1 * 32 * 32 * 32 * 32) / 16 + 3 * (2 * 32 * 32),
                     .l1_buffers_peak_per_core = 2 * (3 * 32 * 32 * 32 * 32) / 16,
                     .l1_output_buffer_per_core = 2 * (3 * 32 * 32 * 32 * 32) / 16}}}),
        std::make_tuple(
            g_interleave_4_2_160_244_tiled,
            -1,
            ResourceUsageMap{
                {BoardType::N300,
                 ttnn::graph::ResourceUsage{
                     .cb_peak_size_per_core = 7 * (2 * 2 * 32 * 32),
                     .l1_buffers_peak_per_core = 10240,
                     .l1_output_buffer_per_core = 10240}},
                {BoardType::E150,
                 ttnn::graph::ResourceUsage{
                     .cb_peak_size_per_core = 7 * (2 * 2 * 32 * 32),
                     .l1_buffers_peak_per_core = 6144,
                     .l1_output_buffer_per_core = 6144}}})),
    [](const testing::TestParamInfo<std::tuple<ttnn::TensorSpec, int, ResourceUsageMap>>& info) {
        std::stringstream ss;

        // print unique id for each test case
        static int uid = 0;
        ss << uid++;

        // print tensor layout
        using detail::operator<<;
        ss << "_" << std::get<ttnn::TensorSpec>(info.param).tensor_layout();

        // print tensor shape; operator<< exists but is too long to be used here
        ss << "_";
        detail::operator<<(ss, std::get<ttnn::TensorSpec>(info.param).logical_shape());
        return ss.str();
    });

// ============================================================================
// Binary tests
// ============================================================================

class EltwiseBinaryOpIfTest
    : public TTNNFixtureWithDevice,
      public testing::WithParamInterface<std::tuple<ttnn::TensorSpec, ttnn::TensorSpec, ResourceUsageMap>> {};

TEST_P(EltwiseBinaryOpIfTest, BinaryAdd) {
    const auto& input_spec_a = std::get<0>(GetParam());
    const auto& input_spec_b = std::get<1>(GetParam());
    const auto& expected_resource_usage_map = std::get<ResourceUsageMap>(GetParam());
    const BoardType board_type = tt::tt_metal::MetalContext::instance().get_cluster().get_board_type(0);
    if (expected_resource_usage_map.count(board_type) == 0) {
        GTEST_SKIP();
    }
    const auto& expected_resource_usage = expected_resource_usage_map.at(board_type);

    // Run the test
    {
        tt::tt_metal::IDevice* device = device_;
        const auto& output_spec = input_spec_a;
        constexpr tt::stl::Span<const ttnn::operations::unary::UnaryWithParam> none{};

        auto query = ttnn::graph::query_op_constraints(
            ttnn::add,
            device,
            input_spec_a,
            input_spec_b,
            output_spec.data_type(),
            output_spec.tensor_layout().get_memory_config(),
            std::nullopt,
            none,
            none,
            none,
            false);

        EXPECT_EQ(query.status, ttnn::graph::ExecutionStatus::Success);
        EXPECT_EQ(query.resource_usage.cb_peak_size_per_core, expected_resource_usage.cb_peak_size_per_core);
        EXPECT_EQ(query.resource_usage.l1_buffers_peak_per_core, expected_resource_usage.l1_buffers_peak_per_core);
        EXPECT_EQ(query.resource_usage.l1_output_buffer_per_core, expected_resource_usage.l1_output_buffer_per_core);
    }
}

INSTANTIATE_TEST_SUITE_P(
    QueryOpConstraints,     // Prefix for the instantiated test suite
    EltwiseBinaryOpIfTest,  // Test suite name
    ::testing::Values(
        std::make_tuple(  // sharded
            g_height_shard_3_1_1024_1024_tiled_to_16_cores,
            g_height_shard_3_1_1024_1024_tiled_to_16_cores,
            ResourceUsageMap{
                {BoardType::N300,
                 ttnn::graph::ResourceUsage{
                     .cb_peak_size_per_core = 0,
                     .l1_buffers_peak_per_core = 2 * (3 * 32 * 32 * 32 * 32) / 16,
                     .l1_output_buffer_per_core = 2 * (3 * 32 * 32 * 32 * 32) / 16}},
                {BoardType::E150,
                 ttnn::graph::ResourceUsage{
                     .cb_peak_size_per_core = 0,
                     .l1_buffers_peak_per_core = 2 * (3 * 32 * 32 * 32 * 32) / 16,
                     .l1_output_buffer_per_core = 2 * (3 * 32 * 32 * 32 * 32) / 16}}}),
        std::make_tuple(  // l1 interleaved
            g_interleave_4_2_160_244_tiled,
            g_interleave_4_2_160_244_tiled,
            ResourceUsageMap{
                {BoardType::N300,
                 ttnn::graph::ResourceUsage{
                     .cb_peak_size_per_core = 3 * (2 * 2 * 32 * 32),
                     .l1_buffers_peak_per_core = 10240,
                     .l1_output_buffer_per_core = 10240}},
                {BoardType::E150,
                 ttnn::graph::ResourceUsage{
                     .cb_peak_size_per_core = 3 * (2 * 2 * 32 * 32),
                     .l1_buffers_peak_per_core = 6144,
                     .l1_output_buffer_per_core = 6144}}}),
        std::make_tuple(  // broadcast
            g_interleave_4_2_160_244_tiled,
            g_interleave_1_1_160_244_tiled,
            ResourceUsageMap{
                {BoardType::N300,
                 ttnn::graph::ResourceUsage{
                     .cb_peak_size_per_core = 3 * (2 * 2 * 32 * 32),
                     .l1_buffers_peak_per_core = 10240,
                     .l1_output_buffer_per_core = 10240}},
                {BoardType::E150,
                 ttnn::graph::ResourceUsage{
                     .cb_peak_size_per_core = 3 * (2 * 2 * 32 * 32),
                     .l1_buffers_peak_per_core = 6144,
                     .l1_output_buffer_per_core = 6144}}}),
        std::make_tuple(  // broadcast
            g_interleave_4_2_160_244_tiled,
            g_interleave_1_2_160_244_tiled,
            ResourceUsageMap{
                {BoardType::N300,
                 ttnn::graph::ResourceUsage{
                     .cb_peak_size_per_core = 3 * (2 * 2 * 32 * 32),
                     .l1_buffers_peak_per_core = 10240,
                     .l1_output_buffer_per_core = 10240}},
                {BoardType::E150,
                 ttnn::graph::ResourceUsage{
                     .cb_peak_size_per_core = 3 * (2 * 2 * 32 * 32),
                     .l1_buffers_peak_per_core = 6144,
                     .l1_output_buffer_per_core = 6144}}})),
    [](const testing::TestParamInfo<std::tuple<ttnn::TensorSpec, ttnn::TensorSpec, ResourceUsageMap>>& info) {
        std::stringstream ss;

        // print unique id for each test case
        static int uid = 0;
        ss << uid++;

        // print tensor layout
        using detail::operator<<;
        ss << "_" << std::get<0>(info.param).tensor_layout();

        // print tensor shape; operator<< exists but is too long to be used here
        ss << "_";
        detail::operator<<(ss, std::get<0>(info.param).logical_shape());

        ss << "_";

        // print tensor layout
        using detail::operator<<;
        ss << "_" << std::get<1>(info.param).tensor_layout();

        // print tensor shape; operator<< exists but is too long to be used here
        ss << "_";
        detail::operator<<(ss, std::get<1>(info.param).logical_shape());

        return ss.str();
    });

// ============================================================================
// Matmul tests
// ============================================================================

class MatmulOpIfTest : public TTNNFixtureWithDevice,
                       public testing::WithParamInterface<std::tuple<
                           ttnn::TensorSpec,
                           ttnn::TensorSpec,
                           std::optional<ttnn::operations::matmul::MatmulProgramConfig>,
                           ResourceUsageMap>> {};

TEST_P(MatmulOpIfTest, Matmul) {
    const auto& input_spec_a = std::get<0>(GetParam());
    const auto& input_spec_b = std::get<1>(GetParam());
    const auto& matmul_program_config =
        std::get<std::optional<ttnn::operations::matmul::MatmulProgramConfig>>(GetParam());
    const auto& expected_resource_usage_map = std::get<ResourceUsageMap>(GetParam());
    const BoardType board_type = tt::tt_metal::MetalContext::instance().get_cluster().get_board_type(0);
    if (expected_resource_usage_map.count(board_type) == 0) {
        GTEST_SKIP();
    }
    const auto& expected_resource_usage = expected_resource_usage_map.at(board_type);

    // Run the test
    {
        tt::tt_metal::IDevice* device = device_;

        const auto output_spec = ttnn::TensorSpec(
            ttnn::Shape(tt::tt_metal::Array4D{
                input_spec_a.logical_shape()[0],
                input_spec_a.logical_shape()[1],
                input_spec_a.logical_shape()[-2],
                input_spec_b.logical_shape()[-1]}),
            tt::tt_metal::TensorLayout(
                tt::tt_metal::DataType::BFLOAT16,
                tt::tt_metal::PageConfig(tt::tt_metal::Layout::TILE),
                ttnn::L1_MEMORY_CONFIG));

        tt::log_info(
            "input_a_shape = {}, input_b_shape = {}, output_shape = {}",
            input_spec_a.logical_shape(),
            input_spec_b.logical_shape(),
            output_spec.logical_shape());

        auto query = ttnn::graph::query_op_constraints(
            ttnn::matmul,
            device,
            input_spec_a,
            input_spec_b,
            false,  // transpose_a
            false,  // transpose_b
            output_spec.tensor_layout().get_memory_config(),
            output_spec.data_type(),
            matmul_program_config);

        tt::log_info("query status = {}, error_message = {}", query.status, query.error_message.value_or("none"));

        EXPECT_EQ(query.status, ttnn::graph::ExecutionStatus::Success);
        EXPECT_EQ(query.resource_usage.cb_peak_size_per_core, expected_resource_usage.cb_peak_size_per_core);
        EXPECT_EQ(query.resource_usage.l1_buffers_peak_per_core, expected_resource_usage.l1_buffers_peak_per_core);
        EXPECT_EQ(query.resource_usage.l1_output_buffer_per_core, expected_resource_usage.l1_output_buffer_per_core);
        ASSERT_TRUE(query.output_tensor_spec.has_value());
        EXPECT_EQ(query.output_tensor_spec.value(), output_spec);
    }
}

INSTANTIATE_TEST_SUITE_P(
    QueryOpConstraints,  // Prefix for the instantiated test suite
    MatmulOpIfTest,      // Test suite name
    ::testing::Values(
        std::make_tuple(  // default
            g_height_shard_1_1_1024_32_tiled_to_32_cores,
            g_interleaved_1_1_1024_1024_tiled,
            std::nullopt,
            ResourceUsageMap{
                {BoardType::N300,
                 ttnn::graph::ResourceUsage{
                     .cb_peak_size_per_core = 24576,
                     .l1_buffers_peak_per_core = 6144,
                     .l1_output_buffer_per_core = 6144}},
                {BoardType::E150,
                 ttnn::graph::ResourceUsage{
                     .cb_peak_size_per_core = 24576,
                     .l1_buffers_peak_per_core = 2048,
                     .l1_output_buffer_per_core = 2048}}}),

        std::make_tuple(  // REUSE_MCAST_1D_IN0
            g_interleaved_1_1_2048_64_tiled,
            g_width_shard_1_1_64_2048_tiled_to_32_cores,
            ttnn::operations::matmul::MatmulMultiCoreReuseMultiCast1DProgramConfig{
                .compute_with_storage_grid_size = CoreCoord(8, 4),
                .in0_block_w = 2,
                .out_subblock_h = 1,
                .out_subblock_w = 1,
                .out_block_h = 64,
                .out_block_w = 2,
                .per_core_M = 64,
                .per_core_N = 2,
                .fuse_batch = true,
                .fused_activation = std::nullopt,
                .mcast_in0 = true},
            ResourceUsageMap{
                {BoardType::N300,
                 ttnn::graph::ResourceUsage{
                     .cb_peak_size_per_core = 524288,
                     .l1_buffers_peak_per_core = 151552,
                     .l1_output_buffer_per_core = 151552}},
                {BoardType::E150,
                 ttnn::graph::ResourceUsage{
                     .cb_peak_size_per_core = 524288,
                     .l1_buffers_peak_per_core = 65536,
                     .l1_output_buffer_per_core = 65536}}}),
        std::make_tuple(  // REUSE_MCAST_2D_BLOCK_SHARDED
            g_block_shard_1_1_1600_256_tiled_to_32_cores,
            g_interleaved_1_1_245_1024_tiled,
            ttnn::operations::matmul::MatmulMultiCoreReuseMultiCastProgramConfig{
                .compute_with_storage_grid_size = CoreCoord(8, 8),
                .in0_block_w = 1,
                .out_subblock_h = 1,
                .out_subblock_w = 4,
                .out_block_h = 1,
                .out_block_w = 4,
                .per_core_M = 10,
                .per_core_N = 4,
                .transpose_mcast = false,
                .fused_activation = std::nullopt},
            ResourceUsageMap{
                {BoardType::N300,
                 ttnn::graph::ResourceUsage{
                     .cb_peak_size_per_core = 28736,
                     .l1_buffers_peak_per_core = 59392,
                     .l1_output_buffer_per_core = 59392}},
                {BoardType::E150,
                 ttnn::graph::ResourceUsage{
                     .cb_peak_size_per_core = 28736,
                     .l1_buffers_peak_per_core = 26624,
                     .l1_output_buffer_per_core = 26624}}})));

class Conv2dOpIfTest : public ttnn::TTNNFixtureWithDevice {};
TEST_F(Conv2dOpIfTest, Conv2d) {
    const auto input_spec = ttnn::TensorSpec(
        ttnn::Shape{1, 1, 50176, 3},
        tt::tt_metal::TensorLayout(
            tt::tt_metal::DataType::BFLOAT16,
            tt::tt_metal::PageConfig(tt::tt_metal::Layout::TILE),
            ttnn::DRAM_MEMORY_CONFIG));
    const auto weight_spec = ttnn::TensorSpec(
        ttnn::Shape{1, 1, 1568, 64},
        tt::tt_metal::TensorLayout(
            tt::tt_metal::DataType::BFLOAT16,
            tt::tt_metal::PageConfig(tt::tt_metal::Layout::TILE),
            ttnn::DRAM_MEMORY_CONFIG));
    const auto output_spec = ttnn::TensorSpec(
        ttnn::Shape{1, 1, 12544, 64},
        tt::tt_metal::TensorLayout(
            tt::tt_metal::DataType::BFLOAT16,
            tt::tt_metal::PageConfig(tt::tt_metal::Layout::TILE),
            ttnn::DRAM_MEMORY_CONFIG));
    const uint32_t in_channels = 3;
    const uint32_t out_channels = 64;
    const uint32_t batch_size = 1;
    const uint32_t input_height = 224;
    const uint32_t input_width = 224;
    const std::array<uint32_t, 2> kernel_size{7, 7};
    const std::array<uint32_t, 2> stride{2, 2};
    const std::array<uint32_t, 2> padding{3, 3};
    const std::array<uint32_t, 2> dilation{1, 1};
    const uint32_t groups = 1;

    const auto expected_resource_usage_map = ResourceUsageMap{
        {BoardType::N300,
         ttnn::graph::ResourceUsage{
             .cb_peak_size_per_core = 229440, .l1_buffers_peak_per_core = 190568, .l1_output_buffer_per_core = 0}}};
    const BoardType board_type = tt::tt_metal::MetalContext::instance().get_cluster().get_board_type(0);
    if (expected_resource_usage_map.count(board_type) == 0) {
        GTEST_SKIP();
    }
    const auto& expected_resource_usage = expected_resource_usage_map.at(board_type);

    // Run the test
    {
        tt::tt_metal::IDevice* device = device_;
        auto query = ttnn::graph::query_op_constraints(
            ttnn::conv2d,
            device,
            input_spec,
            weight_spec,
            device,
            in_channels,
            out_channels,
            batch_size,
            input_height,
            input_width,
            kernel_size,
            stride,
            padding,
            dilation,
            groups,
            std::nullopt,
            std::nullopt,
            std::nullopt,
            output_spec.tensor_layout().get_memory_config());

        EXPECT_EQ(query.status, ttnn::graph::ExecutionStatus::Success);
        EXPECT_EQ(query.resource_usage.cb_peak_size_per_core, expected_resource_usage.cb_peak_size_per_core);
        EXPECT_EQ(query.resource_usage.l1_buffers_peak_per_core, expected_resource_usage.l1_buffers_peak_per_core);
        EXPECT_EQ(query.resource_usage.l1_output_buffer_per_core, expected_resource_usage.l1_output_buffer_per_core);
        ASSERT_TRUE(query.output_tensor_spec.has_value());
        EXPECT_EQ(query.output_tensor_spec.value(), output_spec);
    }
}

}  // namespace test
}  // namespace binary
}  // namespace operations
}  // namespace ttnn
