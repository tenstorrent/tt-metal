// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#include <nlohmann/json.hpp>
#include "ttnn/tensor/tensor_ops.hpp"
#include <array>
#include <algorithm>
#include <map>
#include <vector>

#include <tt-metalium/experimental/per_core_allocation/memory_config.hpp>
#include <tt-metalium/experimental/range_lockstep_allocation/memory_config.hpp>
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

TensorSpec make_nd_sharded_tensor_spec(
    const ttnn::Shape& shape,
    const ttnn::Shape& shard_shape,
    tt::tt_metal::CoreCoord grid_end = tt::tt_metal::CoreCoord{0, 0},
    bool range_lockstep_allocation = false) {
    const auto cores = tt::tt_metal::CoreRangeSet(tt::tt_metal::CoreRange(tt::tt_metal::CoreCoord{0, 0}, grid_end));
    auto memory_config = tt::tt_metal::MemoryConfig(
        tt::tt_metal::BufferType::L1,
        tt::tt_metal::NdShardSpec{shard_shape, cores, tt::tt_metal::ShardOrientation::ROW_MAJOR});
    tt::tt_metal::experimental::range_lockstep_allocation::set_range_lockstep_allocation(
        memory_config, range_lockstep_allocation);
    return TensorSpec(
        shape, TensorLayout(tt::tt_metal::DataType::BFLOAT16, PageConfig(tt::tt_metal::Layout::TILE), memory_config));
}

// GRID_2D maps shards onto the grid's x/y extents rather than linearizing the core list, and only
// accepts a rank <= 2 shard shape (trailing-aligned to the tensor's rank).
TensorSpec make_nd_sharded_grid_2d_tensor_spec(
    const ttnn::Shape& shape, const ttnn::Shape& shard_shape, tt::tt_metal::CoreCoord grid_end) {
    const auto cores = tt::tt_metal::CoreRangeSet(tt::tt_metal::CoreRange(tt::tt_metal::CoreCoord{0, 0}, grid_end));
    const auto memory_config = tt::tt_metal::MemoryConfig(
        tt::tt_metal::BufferType::L1,
        tt::tt_metal::NdShardSpec{
            shard_shape,
            cores,
            tt::tt_metal::ShardOrientation::ROW_MAJOR,
            tt::tt_metal::ShardDistributionStrategy::GRID_2D});
    return TensorSpec(
        shape, TensorLayout(tt::tt_metal::DataType::BFLOAT16, PageConfig(tt::tt_metal::Layout::TILE), memory_config));
}

// CONTIGUOUS_1D has no legacy equivalent, so TensorSpec leaves the tensor ND-sharding-only (see
// populate_legacy_shard_spec_from_nd) — which is exactly the case where a synthesis fall-through has
// no legacy shard_spec to read an orientation from.
TensorSpec make_nd_sharded_contiguous_col_major_tensor_spec(
    const ttnn::Shape& shape, const ttnn::Shape& shard_shape, tt::tt_metal::CoreCoord grid_end) {
    const auto cores = tt::tt_metal::CoreRangeSet(tt::tt_metal::CoreRange(tt::tt_metal::CoreCoord{0, 0}, grid_end));
    const auto memory_config = tt::tt_metal::MemoryConfig(
        tt::tt_metal::BufferType::L1,
        tt::tt_metal::NdShardSpec{
            shard_shape,
            cores,
            tt::tt_metal::ShardOrientation::COL_MAJOR,
            tt::tt_metal::ShardDistributionStrategy::CONTIGUOUS_1D});
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

TensorSpec make_nd_sharded_dram_rm_tensor_spec(
    const ttnn::Shape& shape, const ttnn::Shape& shard_shape, tt::tt_metal::CoreCoord grid_end) {
    const auto cores = tt::tt_metal::CoreRangeSet(tt::tt_metal::CoreRange(tt::tt_metal::CoreCoord{0, 0}, grid_end));
    const auto memory_config = tt::tt_metal::MemoryConfig(
        tt::tt_metal::BufferType::DRAM,
        tt::tt_metal::NdShardSpec{shard_shape, cores, tt::tt_metal::ShardOrientation::ROW_MAJOR});
    return TensorSpec(
        shape,
        TensorLayout(tt::tt_metal::DataType::BFLOAT16, PageConfig(tt::tt_metal::Layout::ROW_MAJOR), memory_config));
}

// Legacy height-sharded L1 config with the experimental allocation flags set. Every public
// MemoryConfig constructor resets them, so they are what's at risk whenever the output config is
// rebuilt from derived shard geometry.
TensorSpec make_allocation_flagged_height_sharded_tensor_spec(
    const ttnn::Shape& shape, std::array<uint32_t, 2> shard_shape, tt::tt_metal::CoreCoord grid_end) {
    const auto cores = tt::tt_metal::CoreRangeSet(tt::tt_metal::CoreRange(tt::tt_metal::CoreCoord{0, 0}, grid_end));
    auto memory_config = tt::tt_metal::MemoryConfig(
        tt::tt_metal::TensorMemoryLayout::HEIGHT_SHARDED,
        tt::tt_metal::BufferType::L1,
        tt::tt_metal::ShardSpec(cores, shard_shape, tt::tt_metal::ShardOrientation::ROW_MAJOR));
    tt::tt_metal::experimental::range_lockstep_allocation::set_range_lockstep_allocation(memory_config, true);
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

TEST_F(TestGraphCaptureArgumentsTranspose, TransposeGrid2dNdShardRegeneratesGridInsteadOfSwappingExtents) {
    // A valid ROW_MAJOR 3x2 GRID_2D grid: 3 shards along width (192/64) fit grid.x, 2 along height
    // (64/32) fit grid.y. Swapping the extents in place would leave the 192x64 output needing 3
    // shard rows against grid.y == 2, which BufferDistributionSpec rejects — so GRID_2D must
    // regenerate the geometry and grid while still preserving ND provenance.
    auto tt_input = create_device_tensor(
        make_nd_sharded_grid_2d_tensor_spec(
            ttnn::Shape({1, 1, 64, 192}), ttnn::Shape({32, 64}), tt::tt_metal::CoreCoord{2, 1}),
        device_);

    ttnn::graph::GraphProcessor::begin_graph_capture(tt::tt_metal::IGraphProcessor::RunMode::NO_DISPATCH);
    ttnn::transpose(tt_input, 2, 3);
    auto trace = ttnn::graph::GraphProcessor::end_graph_capture();
    auto operations = ttnn::graph::extract_arguments(trace);

    auto create_tensor_it = find_create_device_tensor(operations);
    ASSERT_NE(create_tensor_it, operations.end()) << "create_device_tensor operation not found";
    EXPECT_EQ(create_tensor_it->arguments[0], "Shape([1, 1, 192, 64])");
    ASSERT_EQ(create_tensor_it->arguments.size(), 5);
    EXPECT_TRUE(has_nd_provenance(create_tensor_it->arguments[4])) << create_tensor_it->arguments[4];
    // Reaching this line at all is the core regression guard: BufferDistributionSpec aborts while
    // building the buffer if the GRID_2D geometry doesn't fit its grid, which is what the naive
    // extent swap produced. These pin what regeneration actually yields.
    EXPECT_NE(
        create_tensor_it->arguments[4].find("\"shard_distribution_strategy\":\"ShardDistributionStrategy::GRID_2D\""),
        std::string::npos)
        << create_tensor_it->arguments[4];
    EXPECT_EQ(create_tensor_it->arguments[4].find("\"end\":{\"x\":2,\"y\":1}"), std::string::npos)
        << "the input's 3x2 grid must not be retained: " << create_tensor_it->arguments[4];

    // generate_transpose_shard_spec() sizes the output from the device's compute grid, so the exact
    // geometry is arch-dependent — pin it concretely for the grid this runs on. BLOCK synthesis over
    // an 8x8 grid for the 192x64 output gives 32x32 shards: 6 along height, 2 along width, on a 2x6
    // core grid (vs. the 3 shard rows against grid.y == 2 the naive swap needed).
    const auto compute_grid = device_->compute_with_storage_grid_size();
    if (compute_grid.x == 8 && compute_grid.y == 8) {
        EXPECT_NE(create_tensor_it->arguments[4].find("\"shard_shape\":[32, 32]"), std::string::npos)
            << create_tensor_it->arguments[4];
        EXPECT_NE(
            create_tensor_it->arguments[4].find("\"grid\":[{\"start\":{\"x\":0,\"y\":0},\"end\":{\"x\":1,\"y\":5}}]"),
            std::string::npos)
            << create_tensor_it->arguments[4];
    }
}

TEST_F(TestGraphCaptureArgumentsTranspose, TransposeExplicitNdOutputConfigIsHonoredVerbatim) {
    // An ND-only output config carries no legacy shard_spec, so it used to be misread as a bare
    // "sharded, geometry unspecified" request and overwritten with input-derived geometry. It is
    // already expressed in the output frame and must reach the device op untouched.
    auto tt_input = create_device_tensor(
        make_nd_sharded_tensor_spec(
            ttnn::Shape({1, 1, 64, 32}), ttnn::Shape({1, 1, 32, 32}), tt::tt_metal::CoreCoord{1, 0}),
        device_);
    ASSERT_TRUE(tt_input.memory_config().shard_spec().has_value())
        << "input should be native (legacy spec back-filled)";

    // Deliberately distinct from anything the input-derived path would produce: the whole 32x64
    // output in one shard on one core, vs. the input's two 32x32 shards on two cores.
    const auto output_cores = tt::tt_metal::CoreRangeSet(
        tt::tt_metal::CoreRange(tt::tt_metal::CoreCoord{0, 0}, tt::tt_metal::CoreCoord{0, 0}));
    const auto explicit_output_config = tt::tt_metal::MemoryConfig(
        tt::tt_metal::BufferType::L1,
        tt::tt_metal::NdShardSpec{
            ttnn::Shape({1, 1, 32, 64}), output_cores, tt::tt_metal::ShardOrientation::ROW_MAJOR});

    ttnn::graph::GraphProcessor::begin_graph_capture(tt::tt_metal::IGraphProcessor::RunMode::NO_DISPATCH);
    ttnn::transpose(tt_input, 2, 3, explicit_output_config);
    auto trace = ttnn::graph::GraphProcessor::end_graph_capture();
    auto operations = ttnn::graph::extract_arguments(trace);

    auto create_tensor_it = find_create_device_tensor(operations);
    ASSERT_NE(create_tensor_it, operations.end()) << "create_device_tensor operation not found";
    EXPECT_EQ(create_tensor_it->arguments[0], "Shape([1, 1, 32, 64])");
    ASSERT_EQ(create_tensor_it->arguments.size(), 5);
    EXPECT_TRUE(has_nd_provenance(create_tensor_it->arguments[4])) << create_tensor_it->arguments[4];
    EXPECT_NE(create_tensor_it->arguments[4].find("\"shard_shape\":[1, 1, 32, 64]"), std::string::npos)
        << create_tensor_it->arguments[4];
}

TEST_F(TestGraphCaptureArgumentsTranspose, TransposeCnReindexesNdShardShape) {
    // CN exercises adjust_nd_shard_spec_for_transpose()'s leading-axis pair (shard indices 0 and 1
    // for a rank-4 shard shape), which nothing else covers. Asymmetric extents so a missing swap
    // shows up as a mismatch: 2x2 shards along N/C = 4 shards over 4 DRAM banks.
    auto tt_input = create_device_tensor(
        make_nd_sharded_dram_tensor_spec(
            ttnn::Shape({2, 4, 32, 32}), ttnn::Shape({1, 2, 32, 32}), tt::tt_metal::CoreCoord{3, 0}),
        device_);

    ttnn::graph::GraphProcessor::begin_graph_capture(tt::tt_metal::IGraphProcessor::RunMode::NO_DISPATCH);
    ttnn::transpose(tt_input, 0, 1);
    auto trace = ttnn::graph::GraphProcessor::end_graph_capture();
    auto operations = ttnn::graph::extract_arguments(trace);

    auto create_tensor_it = find_create_device_tensor(operations);
    ASSERT_NE(create_tensor_it, operations.end()) << "create_device_tensor operation not found";
    EXPECT_EQ(create_tensor_it->arguments[0], "Shape([4, 2, 32, 32])");
    ASSERT_EQ(create_tensor_it->arguments.size(), 5);
    EXPECT_TRUE(has_nd_provenance(create_tensor_it->arguments[4])) << create_tensor_it->arguments[4];
    EXPECT_NE(create_tensor_it->arguments[4].find("\"shard_shape\":[2, 1, 32, 32]"), std::string::npos)
        << create_tensor_it->arguments[4];
}

TEST_F(TestGraphCaptureArgumentsTranspose, TransposeHcRowMajorReindexesNdShardShape) {
    // HC only reindexes on ROW_MAJOR (the TILE padded-shape contract is asymmetric), so this is the
    // only shape of input that reaches that branch. Shard indices 1 and 2 swap: [1, 2, 1, 32] ->
    // [1, 1, 2, 32], 4 shards over 4 DRAM banks either way.
    auto tt_input = create_device_tensor(
        make_nd_sharded_dram_rm_tensor_spec(
            ttnn::Shape({1, 4, 2, 32}), ttnn::Shape({1, 2, 1, 32}), tt::tt_metal::CoreCoord{3, 0}),
        device_);

    ttnn::graph::GraphProcessor::begin_graph_capture(tt::tt_metal::IGraphProcessor::RunMode::NO_DISPATCH);
    ttnn::transpose(tt_input, 1, 2);
    auto trace = ttnn::graph::GraphProcessor::end_graph_capture();
    auto operations = ttnn::graph::extract_arguments(trace);

    auto create_tensor_it = find_create_device_tensor(operations);
    ASSERT_NE(create_tensor_it, operations.end()) << "create_device_tensor operation not found";
    EXPECT_EQ(create_tensor_it->arguments[0], "Shape([1, 2, 4, 32])");
    ASSERT_EQ(create_tensor_it->arguments.size(), 5);
    EXPECT_TRUE(has_nd_provenance(create_tensor_it->arguments[4])) << create_tensor_it->arguments[4];
    EXPECT_NE(create_tensor_it->arguments[4].find("\"shard_shape\":[1, 1, 2, 32]"), std::string::npos)
        << create_tensor_it->arguments[4];
}

TEST_F(TestGraphCaptureArgumentsTranspose, TransposeHcTileFallsThroughToSynthesisKeepingRoundRobin1d) {
    // HC on TILE can't be reindexed (asymmetric padded-shape contract), so it takes the
    // nd_shard_spec_from_legacy() fall-through. The 96-row tensor with 64-row shards divides
    // unevenly, which is what keeps this L1 height-sharded input non-native and routes it here.
    auto tt_input = create_device_tensor(
        make_nd_sharded_tensor_spec(
            ttnn::Shape({1, 1, 96, 32}), ttnn::Shape({1, 1, 64, 32}), tt::tt_metal::CoreCoord{1, 0}),
        device_);
    ASSERT_EQ(tt_input.memory_config().memory_layout(), tt::tt_metal::TensorMemoryLayout::HEIGHT_SHARDED)
        << "input should flatten to a legacy height-sharded layout for the synthesis step to mirror";

    ttnn::graph::GraphProcessor::begin_graph_capture(tt::tt_metal::IGraphProcessor::RunMode::NO_DISPATCH);
    ttnn::transpose(tt_input, 1, 2);
    auto trace = ttnn::graph::GraphProcessor::end_graph_capture();
    auto operations = ttnn::graph::extract_arguments(trace);

    auto create_tensor_it = find_create_device_tensor(operations);
    ASSERT_NE(create_tensor_it, operations.end()) << "create_device_tensor operation not found";
    EXPECT_EQ(create_tensor_it->arguments[0], "Shape([1, 96, 1, 32])");
    ASSERT_EQ(create_tensor_it->arguments.size(), 5);
    EXPECT_TRUE(has_nd_provenance(create_tensor_it->arguments[4])) << create_tensor_it->arguments[4];
    // The synthesized shard height is compute-grid dependent, so assert the parts that aren't: the
    // strategy is carried through rather than inferred, and the spec was re-wrapped from a
    // flattened 2D shard (rank 2), not reused from the input's rank-4 shape.
    EXPECT_NE(
        create_tensor_it->arguments[4].find(
            "\"shard_distribution_strategy\":\"ShardDistributionStrategy::ROUND_ROBIN_1D\""),
        std::string::npos)
        << create_tensor_it->arguments[4];
    EXPECT_EQ(create_tensor_it->arguments[4].find("\"shard_shape\":[1, 1,"), std::string::npos)
        << create_tensor_it->arguments[4];
}

TEST_F(TestGraphCaptureArgumentsTranspose, TransposeNativeShardedPreservesAllocationFlags) {
    // Native WH: N=C=1 height-sharded with full width promotes to WIDTH_SHARDED, rebuilding the
    // output config through a public MemoryConfig constructor — which resets the experimental
    // allocation flags. derive_effective_output_memory_config() can't restore them either, since it
    // early-returns once shard_spec() is set, so transpose_() has to reapply them itself.
    auto tt_input = create_device_tensor(
        make_allocation_flagged_height_sharded_tensor_spec(
            ttnn::Shape({1, 1, 64, 128}), {32, 128}, tt::tt_metal::CoreCoord{1, 0}),
        device_);
    ASSERT_TRUE(
        tt::tt_metal::experimental::range_lockstep_allocation::is_range_lockstep_allocation(tt_input.memory_config()));

    ttnn::graph::GraphProcessor::begin_graph_capture(tt::tt_metal::IGraphProcessor::RunMode::NO_DISPATCH);
    ttnn::transpose(tt_input, 2, 3);
    auto trace = ttnn::graph::GraphProcessor::end_graph_capture();
    auto operations = ttnn::graph::extract_arguments(trace);

    auto create_tensor_it = find_create_device_tensor(operations);
    ASSERT_NE(create_tensor_it, operations.end()) << "create_device_tensor operation not found";
    EXPECT_EQ(create_tensor_it->arguments[0], "Shape([1, 1, 128, 64])");
    ASSERT_EQ(create_tensor_it->arguments.size(), 5);
    // Confirms the promote-to-width-sharded rebuild really ran, so the flag assertion below isn't
    // trivially passing on a config that was never reconstructed.
    EXPECT_NE(create_tensor_it->arguments[4].find("TensorMemoryLayout::WIDTH_SHARDED"), std::string::npos)
        << create_tensor_it->arguments[4];
    EXPECT_NE(create_tensor_it->arguments[4].find("range_lockstep_allocation=1"), std::string::npos)
        << create_tensor_it->arguments[4];
}

TEST_F(TestGraphCaptureArgumentsTranspose, TransposeNonNativeNdShardedPreservesAllocationFlags) {
    // Complements the native case above, where the output config keeps a legacy shard_spec and so
    // short-circuits derive_effective_output_memory_config() on its first line. Here the 96-row
    // tensor with 64-row shards divides unevenly, which makes this L1 ND-sharded input non-native
    // and routes the output through two *further* reconstructions that both reset the flags:
    // transpose_()'s non-native ND branch, and the device op's ND reindex path.
    auto tt_input = create_device_tensor(
        make_nd_sharded_tensor_spec(
            ttnn::Shape({1, 1, 96, 32}),
            ttnn::Shape({1, 1, 64, 32}),
            tt::tt_metal::CoreCoord{1, 0},
            /*range_lockstep_allocation=*/true),
        device_);
    ASSERT_TRUE(
        tt::tt_metal::experimental::range_lockstep_allocation::is_range_lockstep_allocation(tt_input.memory_config()));

    ttnn::graph::GraphProcessor::begin_graph_capture(tt::tt_metal::IGraphProcessor::RunMode::NO_DISPATCH);
    ttnn::transpose(tt_input, 2, 3);
    auto trace = ttnn::graph::GraphProcessor::end_graph_capture();
    auto operations = ttnn::graph::extract_arguments(trace);

    // transpose_()'s non-native ND branch built this one.
    auto it = std::find_if(operations.begin(), operations.end(), [](const auto& op) {
        return op.operation_name == "TransposeDeviceOperation";
    });
    ASSERT_NE(it, operations.end()) << "TransposeDeviceOperation not found";
    EXPECT_NE(it->arguments[0].find("range_lockstep_allocation=1"), std::string::npos) << it->arguments[0];

    auto create_tensor_it = find_create_device_tensor(operations);
    ASSERT_NE(create_tensor_it, operations.end()) << "create_device_tensor operation not found";
    EXPECT_EQ(create_tensor_it->arguments[0], "Shape([1, 1, 32, 96])");
    ASSERT_EQ(create_tensor_it->arguments.size(), 5);
    // Confirms the device op really did reconstruct the config (ND reindex swapped the last two
    // shard extents), so the flag assertion below isn't passing on an untouched config.
    EXPECT_TRUE(has_nd_provenance(create_tensor_it->arguments[4])) << create_tensor_it->arguments[4];
    EXPECT_NE(create_tensor_it->arguments[4].find("\"shard_shape\":[1, 1, 32, 64]"), std::string::npos)
        << create_tensor_it->arguments[4];
    EXPECT_NE(create_tensor_it->arguments[4].find("range_lockstep_allocation=1"), std::string::npos)
        << create_tensor_it->arguments[4];
}

TEST_F(TestGraphCaptureArgumentsTranspose, TransposeHcTileFallThroughPreservesNdOrientation) {
    // A CONTIGUOUS_1D input gets no legacy shard_spec at all, so the TILE HC fall-through has
    // nothing to infer an orientation from: synthesize_output_shard_spec() resolves to ROW_MAJOR
    // unless the ND spec's own orientation is passed through as the hint, and
    // nd_shard_spec_from_legacy() would then carry that wrong orientation back out.
    auto tt_input = create_device_tensor(
        make_nd_sharded_contiguous_col_major_tensor_spec(
            ttnn::Shape({1, 1, 96, 32}), ttnn::Shape({1, 1, 64, 32}), tt::tt_metal::CoreCoord{1, 0}),
        device_);
    ASSERT_FALSE(tt_input.memory_config().shard_spec().has_value())
        << "CONTIGUOUS_1D should stay ND-sharding-only, with no legacy orientation to fall back on";

    ttnn::graph::GraphProcessor::begin_graph_capture(tt::tt_metal::IGraphProcessor::RunMode::NO_DISPATCH);
    ttnn::transpose(tt_input, 1, 2);
    auto trace = ttnn::graph::GraphProcessor::end_graph_capture();
    auto operations = ttnn::graph::extract_arguments(trace);

    auto create_tensor_it = find_create_device_tensor(operations);
    ASSERT_NE(create_tensor_it, operations.end()) << "create_device_tensor operation not found";
    EXPECT_EQ(create_tensor_it->arguments[0], "Shape([1, 96, 1, 32])");
    ASSERT_EQ(create_tensor_it->arguments.size(), 5);
    EXPECT_TRUE(has_nd_provenance(create_tensor_it->arguments[4])) << create_tensor_it->arguments[4];
    // The output stays ND-only too, so the only quoted orientation here is the ND spec's.
    EXPECT_NE(create_tensor_it->arguments[4].find("\"orientation\":\"ShardOrientation::COL_MAJOR\""), std::string::npos)
        << create_tensor_it->arguments[4];
    EXPECT_NE(
        create_tensor_it->arguments[4].find(
            "\"shard_distribution_strategy\":\"ShardDistributionStrategy::CONTIGUOUS_1D\""),
        std::string::npos)
        << create_tensor_it->arguments[4];
}

TEST_F(TestGraphCaptureArgumentsTranspose, TransposeNativeCnReindexesBackFilledNdShardShape) {
    // Native CN copies the input's config through verbatim, and TensorSpec has back-filled a legacy
    // shard_spec onto it — which used to short-circuit derive_effective_output_memory_config()
    // before any reindexing, leaving the authoritative ND spec in the input frame. L1 (not DRAM)
    // is what keeps this on the native path.
    auto tt_input = create_device_tensor(
        make_nd_sharded_tensor_spec(
            ttnn::Shape({2, 4, 32, 32}), ttnn::Shape({1, 2, 32, 32}), tt::tt_metal::CoreCoord{3, 0}),
        device_);
    ASSERT_TRUE(tt_input.memory_config().shard_spec().has_value())
        << "input must be flattenable so TensorSpec back-fills the legacy spec this test is about";
    ASSERT_EQ(tt_input.memory_config().memory_layout(), tt::tt_metal::TensorMemoryLayout::HEIGHT_SHARDED)
        << "input must be native (L1, non-block, evenly sharded)";

    ttnn::graph::GraphProcessor::begin_graph_capture(tt::tt_metal::IGraphProcessor::RunMode::NO_DISPATCH);
    ttnn::transpose(tt_input, 0, 1);
    auto trace = ttnn::graph::GraphProcessor::end_graph_capture();
    auto operations = ttnn::graph::extract_arguments(trace);

    auto create_tensor_it = find_create_device_tensor(operations);
    ASSERT_NE(create_tensor_it, operations.end()) << "create_device_tensor operation not found";
    EXPECT_EQ(create_tensor_it->arguments[0], "Shape([4, 2, 32, 32])");
    ASSERT_EQ(create_tensor_it->arguments.size(), 5);
    EXPECT_TRUE(has_nd_provenance(create_tensor_it->arguments[4])) << create_tensor_it->arguments[4];
    // Swapped: the input frame's [1, 2, ...] maps 1 shard along N=4 and 2 along C=2, not the
    // 2-along-N / 1-along-C the output needs.
    EXPECT_NE(create_tensor_it->arguments[4].find("\"shard_shape\":[2, 1, 32, 32]"), std::string::npos)
        << create_tensor_it->arguments[4];
}

}  // namespace
}  // namespace ttnn::graph::arguments::test
