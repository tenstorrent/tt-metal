// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#include <nlohmann/json.hpp>
#include "ttnn/tensor/tensor_ops.hpp"
#include <array>
#include <algorithm>
#include <map>
#include <vector>

#include <tt-metalium/graph_tracking.hpp>
#include "gtest/gtest.h"
#include <tt-metalium/shape.hpp>
#include "ttnn/graph/graph_processor.hpp"
#include "ttnn/graph/graph_trace_utils.hpp"
#include "ttnn/operations/data_movement/transpose/transpose.hpp"
#include "ttnn/operations/data_movement/permute/permute.hpp"
#include "ttnn/tensor/layout/page_config.hpp"
#include "ttnn/tensor/layout/tensor_layout.hpp"
#include "ttnn/tensor/shape/shape.hpp"
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/tensor/types.hpp"
#include "ttnn/types.hpp"
#include "ttnn_test_fixtures.hpp"

namespace ttnn::graph::arguments::test {
namespace {

using TestGraphCaptureArgumentsTranspose = TTNNFixtureWithDevice;

TensorSpec make_nd_sharded_tensor_spec(const ttnn::Shape& shape, const ttnn::Shape& shard_shape) {
    const auto cores = tt::tt_metal::CoreRangeSet(
        tt::tt_metal::CoreRange(tt::tt_metal::CoreCoord{0, 0}, tt::tt_metal::CoreCoord{0, 0}));
    const auto memory_config = tt::tt_metal::MemoryConfig(
        tt::tt_metal::BufferType::L1,
        tt::tt_metal::NdShardSpec{shard_shape, cores, tt::tt_metal::ShardOrientation::ROW_MAJOR});
    return TensorSpec(
        shape, TensorLayout(tt::tt_metal::DataType::BFLOAT16, PageConfig(tt::tt_metal::Layout::TILE), memory_config));
}

TensorSpec make_nd_sharded_dram_tensor_spec(
    const ttnn::Shape& shape,
    const ttnn::Shape& shard_shape,
    tt::tt_metal::CoreCoord grid_end = tt::tt_metal::CoreCoord{0, 0}) {
    const auto cores = tt::tt_metal::CoreRangeSet(tt::tt_metal::CoreRange(tt::tt_metal::CoreCoord{0, 0}, grid_end));
    const auto memory_config = tt::tt_metal::MemoryConfig(
        tt::tt_metal::BufferType::DRAM,
        tt::tt_metal::NdShardSpec{shard_shape, cores, tt::tt_metal::ShardOrientation::ROW_MAJOR});
    return TensorSpec(
        shape, TensorLayout(tt::tt_metal::DataType::BFLOAT16, PageConfig(tt::tt_metal::Layout::TILE), memory_config));
}

TensorSpec make_legacy_height_sharded_tensor_spec(const ttnn::Shape& shape) {
    const auto cores = tt::tt_metal::CoreRangeSet(
        tt::tt_metal::CoreRange(tt::tt_metal::CoreCoord{0, 0}, tt::tt_metal::CoreCoord{0, 0}));
    const auto memory_config = tt::tt_metal::MemoryConfig(
        tt::tt_metal::TensorMemoryLayout::HEIGHT_SHARDED,
        tt::tt_metal::BufferType::L1,
        tt::tt_metal::ShardSpec(cores, std::array<uint32_t, 2>{32, 64}, tt::tt_metal::ShardOrientation::ROW_MAJOR));
    return TensorSpec(
        shape, TensorLayout(tt::tt_metal::DataType::BFLOAT16, PageConfig(tt::tt_metal::Layout::TILE), memory_config));
}

const auto has_nd_provenance = [](const std::string& args) {
    // MemoryConfig reflection prints an empty nd_shard_spec as std::nullopt and a
    // populated one as a JSON object, so a populated spec is any non-nullopt value.
    return args.find("created_with_nd_shard_spec=1") != std::string::npos &&
           args.find("nd_shard_spec=std::nullopt") == std::string::npos;
};

const auto find_create_device_tensor = [](const auto& operations) {
    return std::find_if(operations.begin(), operations.end(), [](const auto& op) {
        return op.operation_name == "tt::tt_metal::create_device_tensor";
    });
};

TEST_F(TestGraphCaptureArgumentsTranspose, Transpose) {
    tt::tt_metal::TensorSpec tensor_spec(
        ttnn::Shape({1, 1, 2048, 512}),
        TensorLayout(tt::tt_metal::DataType::BFLOAT16, PageConfig(tt::tt_metal::Layout::ROW_MAJOR), L1_MEMORY_CONFIG));
    auto tt_input = ttnn::create_device_tensor(tensor_spec, device_);

    ttnn::graph::GraphProcessor::begin_graph_capture(tt::tt_metal::IGraphProcessor::RunMode::NORMAL);
    ttnn::transpose(tt_input, 1, 2);
    auto trace = ttnn::graph::GraphProcessor::end_graph_capture();
    auto operations = ttnn::graph::extract_arguments(trace);

    // operations[0]: PermuteDeviceOperation (device operation)
    const auto& operation0 = operations[0];
    EXPECT_EQ(operation0.operation_name, "PermuteDeviceOperation");
    EXPECT_EQ(operation0.arguments.size(), 2);

    // arguments[0]: operation_attributes_t with permutation, memory config, padding value
    EXPECT_TRUE(operation0.arguments[0].find("SmallVector([0, 2, 1, 3])") != std::string::npos);
    EXPECT_TRUE(operation0.arguments[0].find("MemoryConfig(") != std::string::npos);
    EXPECT_TRUE(operation0.arguments[0].find("TensorMemoryLayout::INTERLEAVED") != std::string::npos);
    EXPECT_TRUE(operation0.arguments[0].find("BufferType::L1") != std::string::npos);

    // arguments[1]: vector of input tensors with full tensor info
    EXPECT_TRUE(operation0.arguments[1].find("Tensor(") != std::string::npos);
    EXPECT_TRUE(operation0.arguments[1].find("Shape([1, 1, 2048, 512])") != std::string::npos);
    EXPECT_TRUE(operation0.arguments[1].find("DataType::BFLOAT16") != std::string::npos);
    EXPECT_TRUE(operation0.arguments[1].find("RowMajorPageConfig") != std::string::npos);
    EXPECT_TRUE(operation0.arguments[1].find("DeviceStorage()") != std::string::npos);

    // Find tt::tt_metal::create_device_tensor operation (output tensor creation)
    auto it = std::find_if(operations.begin(), operations.end(), [](const auto& op) {
        return op.operation_name == "tt::tt_metal::create_device_tensor";
    });
    ASSERT_NE(it, operations.end()) << "create_device_tensor operation not found";
    const auto& create_tensor_op = *it;
    EXPECT_EQ(create_tensor_op.arguments.size(), 5);
    EXPECT_EQ(create_tensor_op.arguments[0], "Shape([1, 2048, 1, 512])");
    EXPECT_EQ(create_tensor_op.arguments[1], "DataType::BFLOAT16");
    EXPECT_EQ(create_tensor_op.arguments[2], "Layout::ROW_MAJOR");
}

TEST_F(TestGraphCaptureArgumentsTranspose, PermuteImplicitOutputConfigPreservesNdProvenanceFor4DShardedFallback) {
    auto tt_input = create_device_tensor(
        make_nd_sharded_tensor_spec(ttnn::Shape({1, 1, 64, 64}), ttnn::Shape({1, 1, 32, 32})), device_);

    ttnn::graph::GraphProcessor::begin_graph_capture(tt::tt_metal::IGraphProcessor::RunMode::NO_DISPATCH);
    ttnn::permute(tt_input, ttnn::SmallVector<int64_t>({1, 0, 3, 2}));
    auto trace = ttnn::graph::GraphProcessor::end_graph_capture();
    auto operations = ttnn::graph::extract_arguments(trace);

    auto it = std::find_if(operations.begin(), operations.end(), [](const auto& op) {
        return op.operation_name == "PermuteDeviceOperation";
    });
    ASSERT_NE(it, operations.end()) << "PermuteDeviceOperation not found";
    EXPECT_TRUE(has_nd_provenance(it->arguments[0])) << it->arguments[0];

    auto create_tensor_it = find_create_device_tensor(operations);
    ASSERT_NE(create_tensor_it, operations.end()) << "create_device_tensor operation not found";
    EXPECT_EQ(create_tensor_it->arguments[0], "Shape([1, 1, 64, 64])");
    EXPECT_EQ(create_tensor_it->arguments[2], "Layout::TILE");
    ASSERT_EQ(create_tensor_it->arguments.size(), 5);
    EXPECT_TRUE(has_nd_provenance(create_tensor_it->arguments[4])) << create_tensor_it->arguments[4];
}

TEST_F(TestGraphCaptureArgumentsTranspose, PermuteImplicitOutputConfigRecomputesLegacyShardSpecForShardedFallback) {
    auto tt_input = create_device_tensor(make_legacy_height_sharded_tensor_spec(ttnn::Shape({1, 1, 32, 64})), device_);

    ttnn::graph::GraphProcessor::begin_graph_capture(tt::tt_metal::IGraphProcessor::RunMode::NO_DISPATCH);
    ttnn::permute(tt_input, ttnn::SmallVector<int64_t>({3, 2, 1, 0}));
    auto trace = ttnn::graph::GraphProcessor::end_graph_capture();
    auto operations = ttnn::graph::extract_arguments(trace);

    auto create_tensor_it = find_create_device_tensor(operations);
    ASSERT_NE(create_tensor_it, operations.end()) << "create_device_tensor operation not found";
    EXPECT_EQ(create_tensor_it->arguments[0], "Shape([64, 32, 1, 1])");
    EXPECT_EQ(create_tensor_it->arguments[2], "Layout::TILE");
    ASSERT_EQ(create_tensor_it->arguments.size(), 5);
    EXPECT_TRUE(create_tensor_it->arguments[4].find("TensorMemoryLayout::HEIGHT_SHARDED") != std::string::npos)
        << create_tensor_it->arguments[4];
}

TEST_F(TestGraphCaptureArgumentsTranspose, PermuteImplicitOutputConfigPreservesNdProvenanceForRank5ShardedFallback) {
    auto tt_input = create_device_tensor(
        make_nd_sharded_tensor_spec(ttnn::Shape({1, 2, 2, 32, 64}), ttnn::Shape({1, 1, 2, 32, 64})), device_);

    ttnn::graph::GraphProcessor::begin_graph_capture(tt::tt_metal::IGraphProcessor::RunMode::NO_DISPATCH);
    ttnn::permute(tt_input, ttnn::SmallVector<int64_t>({0, 2, 1, 4, 3}));
    auto trace = ttnn::graph::GraphProcessor::end_graph_capture();
    auto operations = ttnn::graph::extract_arguments(trace);

    auto it = std::find_if(operations.begin(), operations.end(), [](const auto& op) {
        return op.operation_name == "PermuteDeviceOperation";
    });
    ASSERT_NE(it, operations.end()) << "PermuteDeviceOperation not found";
    EXPECT_TRUE(has_nd_provenance(it->arguments[0])) << it->arguments[0];

    auto create_tensor_it = find_create_device_tensor(operations);
    ASSERT_NE(create_tensor_it, operations.end()) << "create_device_tensor operation not found";
    EXPECT_EQ(create_tensor_it->arguments[0], "Shape([1, 2, 2, 64, 32])");
    EXPECT_EQ(create_tensor_it->arguments[2], "Layout::TILE");
    ASSERT_EQ(create_tensor_it->arguments.size(), 5);
    EXPECT_TRUE(has_nd_provenance(create_tensor_it->arguments[4])) << create_tensor_it->arguments[4];
}

TEST_F(
    TestGraphCaptureArgumentsTranspose, TransposeImplicitOutputConfigPreservesNdProvenanceForNonNativeShardedFallback) {
    // DRAM-sharded inputs are non-native for transpose (side_native() rejects DRAM), so this
    // exercises the non-native sharded fallback in detail::transpose_(), which must preserve the
    // input's ND-sharding provenance instead of rebuilding a legacy MemoryConfig from only
    // memory_layout()/buffer_type(). This also covers the second boundary hole:
    // TransposeDeviceOperation::derive_effective_output_memory_config() (called from
    // compute_output_specs/select_program_factory) separately re-synthesizes the *final* shard
    // spec, and must re-wrap it as an NdShardSpec (via nd_shard_spec_from_legacy /
    // adjust_nd_shard_spec_for_transpose) instead of dropping provenance at that later stage — so
    // we assert provenance on both operation_attributes and the final create_device_tensor spec.
    auto tt_input = create_device_tensor(
        make_nd_sharded_dram_tensor_spec(ttnn::Shape({1, 1, 64, 64}), ttnn::Shape({1, 1, 32, 32})), device_);

    ttnn::graph::GraphProcessor::begin_graph_capture(tt::tt_metal::IGraphProcessor::RunMode::NO_DISPATCH);
    ttnn::transpose(tt_input, 2, 3);
    auto trace = ttnn::graph::GraphProcessor::end_graph_capture();
    auto operations = ttnn::graph::extract_arguments(trace);

    auto it = std::find_if(operations.begin(), operations.end(), [](const auto& op) {
        return op.operation_name == "TransposeDeviceOperation";
    });
    ASSERT_NE(it, operations.end()) << "TransposeDeviceOperation not found";
    EXPECT_TRUE(has_nd_provenance(it->arguments[0])) << it->arguments[0];

    auto create_tensor_it = find_create_device_tensor(operations);
    ASSERT_NE(create_tensor_it, operations.end()) << "create_device_tensor operation not found";
    EXPECT_EQ(create_tensor_it->arguments[0], "Shape([1, 1, 64, 64])");
    EXPECT_EQ(create_tensor_it->arguments[2], "Layout::TILE");
    ASSERT_EQ(create_tensor_it->arguments.size(), 5);
    EXPECT_TRUE(has_nd_provenance(create_tensor_it->arguments[4])) << create_tensor_it->arguments[4];
}

TEST_F(
    TestGraphCaptureArgumentsTranspose,
    TransposeImplicitOutputConfigReindexesNdShardShapeForAsymmetricNonNativeShardedFallback) {
    // Asymmetric shard/tensor shape (unlike the 64x64 case above) so that a bug in
    // adjust_nd_shard_spec_for_transpose (e.g. forgetting to swap the last two shard_shape entries
    // for a WH transpose) would show up as a shape mismatch instead of trivially passing. Both
    // shard extents must stay tile-aligned (multiples of 32) since the tensor is TILE layout.
    // 2 shards along H (64/32) x 2 shards along W (128/64) = 4 shards; DRAM shard grids are 1D
    // (bank_id == logical x-coordinate), so all shard cores must stay on row y == 0.
    auto tt_input = create_device_tensor(
        make_nd_sharded_dram_tensor_spec(
            ttnn::Shape({1, 1, 64, 128}), ttnn::Shape({1, 1, 32, 64}), tt::tt_metal::CoreCoord{3, 0}),
        device_);

    ttnn::graph::GraphProcessor::begin_graph_capture(tt::tt_metal::IGraphProcessor::RunMode::NO_DISPATCH);
    ttnn::transpose(tt_input, 2, 3);
    auto trace = ttnn::graph::GraphProcessor::end_graph_capture();
    auto operations = ttnn::graph::extract_arguments(trace);

    auto it = std::find_if(operations.begin(), operations.end(), [](const auto& op) {
        return op.operation_name == "TransposeDeviceOperation";
    });
    ASSERT_NE(it, operations.end()) << "TransposeDeviceOperation not found";
    EXPECT_TRUE(has_nd_provenance(it->arguments[0])) << it->arguments[0];
    // operation_attributes.output_mem_config is the *implicit* config built in transpose.cpp's
    // fallback, which intentionally mirrors the input's nd_shard_spec verbatim (unswapped) —
    // reindexing for the specific transpose dim only happens later, in
    // TransposeDeviceOperation::derive_effective_output_memory_config(), when synthesizing the
    // *final* output spec below. So this still shows the pre-transpose shard_shape.
    EXPECT_TRUE(it->arguments[0].find("\"shard_shape\":[1, 1, 32, 64]") != std::string::npos) << it->arguments[0];

    auto create_tensor_it = find_create_device_tensor(operations);
    ASSERT_NE(create_tensor_it, operations.end()) << "create_device_tensor operation not found";
    EXPECT_EQ(create_tensor_it->arguments[0], "Shape([1, 1, 128, 64])");
    EXPECT_EQ(create_tensor_it->arguments[2], "Layout::TILE");
    ASSERT_EQ(create_tensor_it->arguments.size(), 5);
    EXPECT_TRUE(has_nd_provenance(create_tensor_it->arguments[4])) << create_tensor_it->arguments[4];
    // The final synthesized shard geometry must have the last two shard_shape entries swapped
    // relative to the pre-transpose [1, 1, 32, 64] — this is what actually exercises
    // adjust_nd_shard_spec_for_transpose.
    EXPECT_TRUE(create_tensor_it->arguments[4].find("\"shard_shape\":[1, 1, 64, 32]") != std::string::npos)
        << create_tensor_it->arguments[4];
}

}  // namespace
}  // namespace ttnn::graph::arguments::test
