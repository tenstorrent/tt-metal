// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include <functional>
#include <optional>
#include <ostream>
#include <string>
#include <tuple>
#include <variant>
#include <vector>

#include "common/constants.hpp"
#include "gtest/gtest.h"
#include "impl/event/event.hpp"
#include "impl/program/program.hpp"
#include "tests/tt_metal/tt_metal/unit_tests_common/common/common_fixture.hpp"
#include "third_party/json/json.hpp"
#include "tt_metal/common/logger.hpp"
#include "ttnn/device.hpp"
#include "ttnn/graph/graph_operation_queries.hpp"
#include "ttnn/graph/graph_processor.hpp"
#include "ttnn/graph/graph_trace_utils.hpp"
#include "ttnn/operations/common/l1_interface_common.hpp"
#include "ttnn/operations/core/core.hpp"
#include "ttnn/operations/creation.hpp"
#include "ttnn/operations/eltwise/binary/binary.hpp"
#include "ttnn/operations/eltwise/binary/binary_constraints.hpp"
#include "ttnn/operations/eltwise/binary/binary_l1_interface.hpp"
#include "ttnn/operations/eltwise/unary/unary.hpp"
#include "ttnn/operations/eltwise/unary/unary_l1_interface.hpp"
#include "ttnn/operations/matmul/device/matmul_types.hpp"
#include "ttnn/operations/matmul/matmul.hpp"
#include "ttnn/operations/matmul/matmul_l1_interface.hpp"
#include "ttnn/operations/normalization/softmax/softmax.hpp"
#include "ttnn/operations/normalization/softmax/softmax_l1_interface.hpp"
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/tensor/tensor_utils.hpp"
#include "ttnn/tensor/types.hpp"
#include "ttnn/types.hpp"
#include "ttnn_test_fixtures.hpp"

namespace ttnn {
namespace operations {
namespace binary {
namespace test {

struct OperandShapeTestParam {
    ttnn::types::Shape shape;
    tt::tt_metal::MemoryConfig memory_config;
    tt::tt_metal::DataType data_type = tt::tt_metal::DataType::BFLOAT16;
    tt::tt_metal::Layout layout = tt::tt_metal::Layout::TILE;
};

ttnn::Shape pad_shape_to_tile(const ttnn::Shape& shape) {
    std::vector<uint32_t> shape_og;
    std::vector<uint32_t> shape_padded;

    auto rank = shape.rank();
    for (auto i = 0; i < rank; i++) {
        shape_og.push_back(shape[i]);

        if (i >= rank - 2) {
            shape_padded.push_back((shape[i] + 31) / 32 * 32);
        } else {
            shape_padded.push_back(shape[i]);
        }
    }
    return ttnn::Shape(shape_og, shape_padded);
}

void compare_l1_circular_buffer_allocations(
    const std::vector<std::tuple<uint32_t, uint32_t>>& usage_estimator_result, const nlohmann::json& json_trace) {
    auto graph_circular_buffer_allocations = graph::extract_circular_buffer_allocations_per_core(json_trace);
    EXPECT_EQ(usage_estimator_result.size(), graph_circular_buffer_allocations.size());

    for (const auto& [size, cores] : usage_estimator_result) {
        std::cout << size << " ";
    }
    std::cout << std::endl;
    for (int size : graph_circular_buffer_allocations) {
        std::cout << size << " ";
    }
    std::cout << std::endl;

    for (int i = 0; i < graph_circular_buffer_allocations.size(); i++) {
        std::cout << "DBG cb[" << i << "] " << std::get<0>(usage_estimator_result[i]) << " "
                  << graph_circular_buffer_allocations[i] << std::endl;

        EXPECT_EQ(std::get<0>(usage_estimator_result[i]), graph_circular_buffer_allocations[i]);
    }
}

void compare_l1_tensor_allocations(
    const std::vector<std::tuple<uint32_t, uint32_t>>& usage_estimator_result, const nlohmann::json& json_trace) {
    auto graph_l1_buffer_allocations = graph::extract_l1_buffer_allocations(json_trace);  // total
    EXPECT_GE(usage_estimator_result.size(), graph_l1_buffer_allocations.size());
    int l = 0, r = 0;
    const auto read_graph_capture_buffer = [&](int idx) {
        return idx >= graph_l1_buffer_allocations.size() ? 0 : graph_l1_buffer_allocations[idx];
    };

    // Skip comparing dram allocated buffers which graph capture doesn't report.
    while (l < usage_estimator_result.size()) {
        if (std::get<0>(usage_estimator_result[l]) == 0) {
            l++;
            continue;
        }

        std::cout << "DBG l1[" << l << "]" << std::get<0>(usage_estimator_result[l])
                  << std::endl;  // << " " << graph_l1_buffer_allocations[i] << std::endl;
        EXPECT_EQ(
            std::get<0>(usage_estimator_result[l]) * std::get<1>(usage_estimator_result[l]),
            read_graph_capture_buffer(r));
        l++;
        r++;
    }

    // Make sure we compared all the elements in the two arrays.
    EXPECT_EQ(l, usage_estimator_result.size());
    EXPECT_GE(r, graph_l1_buffer_allocations.size());
}

OperandShapeTestParam select_larger_input(const OperandShapeTestParam& a, const OperandShapeTestParam& b) {
    return a.shape.volume() >= b.shape.volume() ? a : b;
}

class EltwiseUnaryOpInterfaceTestFixture : public TTNNFixtureWithDevice,
                                           public testing::WithParamInterface<OperandShapeTestParam> {};

class EltwiseBinaryOpInterfaceTestFixture
    : public TTNNFixtureWithDevice,
      public testing::WithParamInterface<std::tuple<OperandShapeTestParam, OperandShapeTestParam>> {};

class SoftmaxOpInterfaceTestFixture : public TTNNFixtureWithDevice,
                                      public testing::WithParamInterface<std::tuple<OperandShapeTestParam, int>> {};

class MatmulOpInterfaceTestFixture
    : public TTNNFixtureWithDevice,
      public testing::WithParamInterface<
          std::tuple<OperandShapeTestParam, OperandShapeTestParam, ttnn::operations::matmul::MatmulProgramConfig>> {};

TEST_P(MatmulOpInterfaceTestFixture, MlirInterfaceTest) {
    auto param_combination = GetParam();
    auto input_a = std::get<0>(param_combination);
    auto input_b = std::get<1>(param_combination);
    auto program_config = std::get<2>(param_combination);

    // pad input shapes (this isn't happening automagically)
    input_a.shape = pad_shape_to_tile(input_a.shape);
    input_b.shape = pad_shape_to_tile(input_b.shape);
    std::cout << "OP = matmul(" << input_a.shape << ", " << input_b.shape << ")" << std::endl;

    // TODO: Test constraints

    // Run the test
    {
        auto input_tensor_a =
            ttnn::zeros(input_a.shape, input_a.data_type, input_a.layout, this->getDevice(), input_a.memory_config);
        auto input_tensor_b =
            ttnn::zeros(input_b.shape, input_b.data_type, input_b.layout, this->getDevice(), input_b.memory_config);

        auto call = [&] {
            const auto output_tensor = ttnn::matmul(
                input_tensor_a,
                input_tensor_b,
                false /* transpose_a */,
                false /* transpose_b */,
                std::nullopt /* memory_config */,
                std::nullopt /* dtype */,
                program_config);

            return output_tensor;
        };

        // // get graph trace for ground truth
        auto json_trace = graph::query_trace(call);
        // tt::log_info("Trace: {}", json_trace.dump(4));

        // L1 interface calls and checks against graph trace
        {
            const auto shape_a = input_a.shape.value;
            const auto shape_b = input_b.shape.value;

            auto l1_input_a = std::make_tuple(input_a.shape, input_a.data_type, input_a.layout, input_a.memory_config);
            auto l1_input_b = std::make_tuple(input_b.shape, input_b.data_type, input_b.layout, input_b.memory_config);

            // If tt-mlir doesn't specify output memory config, the default is dram interleaved
            auto l1_output = std::make_tuple(
                ttnn::Shape(tt::tt_metal::Array4D{shape_a[0], shape_a[1], shape_a[-2], shape_b[-1]}),
                input_a.data_type,
                tt::tt_metal::Layout::TILE,
                ttnn::DRAM_MEMORY_CONFIG);

            auto l1_usage = MatmulOpL1UsageFactory::Make(l1_input_a, l1_input_a, l1_output, program_config);

            compare_l1_circular_buffer_allocations(l1_usage->get_circular_buffer_l1_allocations_per_core(), json_trace);
            compare_l1_tensor_allocations(l1_usage->get_tensor_l1_allocations_per_core(), json_trace);
        }
    }
}

INSTANTIATE_TEST_SUITE_P(
    MlirInterfaceTests_REUSE_MCAST_1D_IN0,  // Prefix for the instantiated test suite
    MatmulOpInterfaceTestFixture,           // Test suite name
    ::testing::Combine(
        ::testing::Values(
            OperandShapeTestParam{
                .shape = ttnn::Shape(tt::tt_metal::Array4D{1, 1, 64, 2048}),
                .memory_config = ttnn::L1_MEMORY_CONFIG,
            },
            OperandShapeTestParam{
                .shape = ttnn::Shape(tt::tt_metal::Array4D{1, 1, 64, 2048}),
                .memory_config =
                    {.memory_layout = tt::tt_metal::TensorMemoryLayout::WIDTH_SHARDED,
                     .buffer_type = tt::tt_metal::BufferType::L1,
                     .shard_spec =
                         tt::tt_metal::ShardSpec{
                             CoreRangeSet{std::set<CoreRange>{CoreRange{CoreCoord{0, 0}, CoreCoord{7, 3}}}},
                             {64, 64},
                             ShardOrientation::ROW_MAJOR}},
            }),

        ::testing::Values(OperandShapeTestParam{
            .shape = ttnn::Shape(tt::tt_metal::Array4D{1, 1, 2048, 1024}),
            .memory_config = ttnn::L1_MEMORY_CONFIG,
        }),

        ::testing::Values(ttnn::operations::matmul::MatmulMultiCoreReuseMultiCast1DProgramConfig{
            .compute_with_storage_grid_size = CoreCoord(8, 4),
            .in0_block_w = 2,
            .out_subblock_h = 1,
            .out_subblock_w = 1,
            .per_core_M = 2,
            .per_core_N = 1,
            .fuse_batch = true,
            .fused_activation = std::nullopt,
            .mcast_in0 = true}))

);

INSTANTIATE_TEST_SUITE_P(
    MlirInterfaceTests_REUSE_MCAST_1D_IN1,  // Prefix for the instantiated test suite
    MatmulOpInterfaceTestFixture,           // Test suite name
    ::testing::Combine(
        ::testing::Values(
            OperandShapeTestParam{
                .shape = ttnn::Shape(tt::tt_metal::Array4D{1, 1, 4096, 64}),
                .memory_config = ttnn::DRAM_MEMORY_CONFIG,
            },
            OperandShapeTestParam{
                .shape = ttnn::Shape(tt::tt_metal::Array4D{1, 1, 4096, 64}),
                .memory_config =
                    {.memory_layout = tt::tt_metal::TensorMemoryLayout::HEIGHT_SHARDED,
                     .buffer_type = tt::tt_metal::BufferType::L1,
                     .shard_spec =
                         tt::tt_metal::ShardSpec{
                             CoreRangeSet{std::set<CoreRange>{CoreRange{CoreCoord{0, 0}, CoreCoord{7, 3}}}},
                             {128, 64},
                             ShardOrientation::ROW_MAJOR}},
            }),

        ::testing::Values(OperandShapeTestParam{
            .shape = ttnn::Shape(tt::tt_metal::Array4D{1, 1, 64, 256}),
            .memory_config = ttnn::DRAM_MEMORY_CONFIG,
        }),

        ::testing::Values(ttnn::operations::matmul::MatmulMultiCoreReuseMultiCast1DProgramConfig{
            .compute_with_storage_grid_size = CoreCoord(8, 4),
            .in0_block_w = 2,
            .out_subblock_h = 1,
            .out_subblock_w = 1,
            .per_core_M = 4,
            .per_core_N = 8,
            .fuse_batch = true,
            .fused_activation = std::nullopt,
            .mcast_in0 = false}))

);

INSTANTIATE_TEST_SUITE_P(
    MlirInterfaceTests_REUSE_MCAST_2D,  // Prefix for the instantiated test suite
    MatmulOpInterfaceTestFixture,       // Test suite name
    ::testing::Combine(
        ::testing::Values(OperandShapeTestParam{
            .shape = ttnn::Shape(tt::tt_metal::Array4D{1, 1, 4 * 32, 8 * 32}),
            .memory_config = ttnn::L1_MEMORY_CONFIG,
        }),

        ::testing::Values(OperandShapeTestParam{
            .shape = ttnn::Shape(tt::tt_metal::Array4D{1, 1, 8 * 32, 4 * 32}),
            .memory_config = ttnn::L1_MEMORY_CONFIG,
        }),

        ::testing::Values(ttnn::operations::matmul::MatmulMultiCoreReuseMultiCastProgramConfig{
            .compute_with_storage_grid_size = CoreCoord(2, 2),
            .in0_block_w = 4,
            .out_subblock_h = 2,
            .out_subblock_w = 2,
            .per_core_M = 2,
            .per_core_N = 4,
            .transpose_mcast = false,
            .fused_activation = std::nullopt}))

);

INSTANTIATE_TEST_SUITE_P(
    MlirInterfaceTests_REUSE_MCAST_2D_BATCHED,  // Prefix for the instantiated test suite
    MatmulOpInterfaceTestFixture,               // Test suite name
    ::testing::Combine(
        ::testing::Values(OperandShapeTestParam{
            .shape = ttnn::Shape(tt::tt_metal::Array4D{3, 1, 4 * 32, 8 * 32}),
            .memory_config = ttnn::L1_MEMORY_CONFIG,
        }),

        ::testing::Values(OperandShapeTestParam{
            .shape = ttnn::Shape(tt::tt_metal::Array4D{3, 1, 8 * 32, 4 * 32}),
            .memory_config = ttnn::L1_MEMORY_CONFIG,
        }),

        ::testing::Values(ttnn::operations::matmul::MatmulMultiCoreReuseMultiCastProgramConfig{
            .compute_with_storage_grid_size = CoreCoord(4, 4),
            .in0_block_w = 4,
            .out_subblock_h = 3,
            .out_subblock_w = 1,
            .per_core_M = 3,
            .per_core_N = 1,
            .transpose_mcast = false,
            .fused_activation = std::nullopt}))

);

INSTANTIATE_TEST_SUITE_P(
    MlirInterfaceTests_REUSE_MCAST_2D_BLOCK_SHARDED,  // Prefix for the instantiated test suite
    MatmulOpInterfaceTestFixture,                     // Test suite name
    ::testing::Combine(
        ::testing::Values(OperandShapeTestParam{
            .shape = ttnn::Shape(tt::tt_metal::Array4D{1, 1, 1600, 256}),
            .memory_config =
                {.memory_layout = tt::tt_metal::TensorMemoryLayout::BLOCK_SHARDED,
                 .buffer_type = tt::tt_metal::BufferType::L1,
                 .shard_spec =
                     tt::tt_metal::ShardSpec{
                         CoreRangeSet{std::set<CoreRange>{CoreRange{CoreCoord{0, 0}, CoreCoord{7, 4}}}},
                         {320, 32},
                         ShardOrientation::ROW_MAJOR}},
        }),

        ::testing::Values(OperandShapeTestParam{
            .shape = ttnn::Shape(tt::tt_metal::Array4D{1, 1, 256, 1024}),
            .memory_config = ttnn::L1_MEMORY_CONFIG,
        }),

        ::testing::Values(ttnn::operations::matmul::MatmulMultiCoreReuseMultiCastProgramConfig{
            .compute_with_storage_grid_size = CoreCoord(8, 8),
            .in0_block_w = 1,
            .out_subblock_h = 1,
            .out_subblock_w = 4,
            .per_core_M = 10,
            .per_core_N = 4,
            .transpose_mcast = false,
            .fused_activation = std::nullopt}))

);

TEST_P(SoftmaxOpInterfaceTestFixture, MlirInterfaceTest) {
    auto param_combination = GetParam();
    auto input = std::get<0>(param_combination);
    auto dim_arg = std::get<1>(param_combination);

    // pad input shapes (this isn't happening automagically)
    input.shape = pad_shape_to_tile(input.shape);
    std::cout << "OP = softmax(" << input.shape << ", dim=" << dim_arg << ")" << std::endl;

    // TODO: Test constraints

    // Run the test
    {
        auto input_tensor =
            ttnn::zeros(input.shape, input.data_type, input.layout, this->getDevice(), input.memory_config);

        auto call = [&] {
            const auto output_tensor = ttnn::softmax(input_tensor, dim_arg);
            return output_tensor;
        };

        // // get graph trace for ground truth
        auto json_trace = graph::query_trace(call);
        tt::log_info("Trace: {}", json_trace.dump(4));

        // L1 interface calls and checks against graph trace
        {
            auto l1_input = std::make_tuple(input.shape, input.data_type, input.layout, input.memory_config);
            auto l1_usage = SoftmaxOpL1UsageFactory::Make(l1_input, dim_arg);

            compare_l1_circular_buffer_allocations(l1_usage->get_circular_buffer_l1_allocations_per_core(), json_trace);
            compare_l1_tensor_allocations(l1_usage->get_tensor_l1_allocations_per_core(), json_trace);
        }
    }
}

INSTANTIATE_TEST_SUITE_P(
    MlirInterfaceTests,             // Prefix for the instantiated test suite
    SoftmaxOpInterfaceTestFixture,  // Test suite name
    ::testing::Combine(
        ::testing::Values(
            OperandShapeTestParam{
                .shape = ttnn::Shape(tt::tt_metal::Array4D{4, 2, 5 * 32, 7 * 32}),
                .memory_config = ttnn::L1_MEMORY_CONFIG,
            },
            OperandShapeTestParam{
                .shape = ttnn::Shape(tt::tt_metal::Array4D{4, 2, 5 * 32, 7 * 32}),
                .memory_config = ttnn::L1_MEMORY_CONFIG,
                .layout = ttnn::ROW_MAJOR_LAYOUT},
            OperandShapeTestParam{
                .shape = ttnn::Shape(tt::tt_metal::Array4D{3, 1, 32 * 32, 32 * 32}),
                .memory_config =
                    {.memory_layout = tt::tt_metal::TensorMemoryLayout::HEIGHT_SHARDED,
                     .buffer_type = tt::tt_metal::BufferType::L1,
                     .shard_spec =
                         tt::tt_metal::ShardSpec{
                             CoreRangeSet{std::set<CoreRange>{CoreRange{CoreCoord{0, 0}, CoreCoord{3, 3}}}},
                             {6 * 32, 32 * 32},
                             ShardOrientation::COL_MAJOR}},
            }),

        ::testing::Values(-1))

);

TEST_P(EltwiseUnaryOpInterfaceTestFixture, MlirInterfaceTest) {
    auto input = GetParam();

    // pad input shapes (this isn't happening automagically)
    input.shape = pad_shape_to_tile(input.shape);
    std::cout << "OP = relu(" << input.shape << ")" << std::endl;

    // TODO: Test constraints

    // Run the test
    {
        auto input_tensor =
            ttnn::zeros(input.shape, input.data_type, input.layout, this->getDevice(), input.memory_config);

        auto call = [&] {
            const auto output_tensor = ttnn::relu(input_tensor);
            return output_tensor;
        };

        // // get graph trace for ground truth
        auto json_trace = graph::query_trace(call);
        tt::log_info("Trace: {}", json_trace.dump(4));

        // L1 interface calls and checks against graph trace
        {
            auto l1_input = std::make_tuple(input.shape, input.data_type, input.layout, input.memory_config);
            auto l1_usage = UnaryOpL1UsageFactory::Make(l1_input);

            compare_l1_circular_buffer_allocations(l1_usage->get_circular_buffer_l1_allocations_per_core(), json_trace);
            compare_l1_tensor_allocations(l1_usage->get_tensor_l1_allocations_per_core(), json_trace);
        }
    }
}

INSTANTIATE_TEST_SUITE_P(
    MlirInterfaceTests,                  // Prefix for the instantiated test suite
    EltwiseUnaryOpInterfaceTestFixture,  // Test suite name
    ::testing::Values(
        OperandShapeTestParam{
            .shape = ttnn::Shape(tt::tt_metal::Array4D{3, 1, 32 * 32, 32 * 32}),
            .memory_config =
                {.memory_layout = tt::tt_metal::TensorMemoryLayout::HEIGHT_SHARDED,
                 .buffer_type = tt::tt_metal::BufferType::L1,
                 .shard_spec =
                     tt::tt_metal::ShardSpec{
                         CoreRangeSet{std::set<CoreRange>{CoreRange{CoreCoord{0, 0}, CoreCoord{3, 3}}}},
                         {6 * 32, 32 * 32},
                         ShardOrientation::COL_MAJOR}},
        },
        OperandShapeTestParam{
            .shape = ttnn::Shape(tt::tt_metal::Array4D{4, 2, 5 * 32, 7 * 32}),
            .memory_config = ttnn::L1_MEMORY_CONFIG,
        },
        OperandShapeTestParam{
            .shape = ttnn::Shape(tt::tt_metal::Array4D{3, 1, 32 * 32, 32 * 32}),
            .memory_config =
                {.memory_layout = tt::tt_metal::TensorMemoryLayout::HEIGHT_SHARDED,
                 .buffer_type = tt::tt_metal::BufferType::L1,
                 .shard_spec =
                     tt::tt_metal::ShardSpec{
                         CoreRangeSet{std::set<CoreRange>{CoreRange{CoreCoord{0, 0}, CoreCoord{3, 3}}}},
                         {6 * 32, 32 * 32},
                         ShardOrientation::COL_MAJOR}},
            .layout = tt::tt_metal::Layout::ROW_MAJOR,
        })

);

TEST_P(EltwiseBinaryOpInterfaceTestFixture, MlirInterfaceTest) {
    auto param_combination = GetParam();
    auto input_a = std::get<0>(param_combination);
    auto input_b = std::get<1>(param_combination);
    auto output = select_larger_input(input_a, input_b);

    // pad input shapes (this isn't happening automagically)
    input_a.shape = pad_shape_to_tile(input_a.shape);
    input_b.shape = pad_shape_to_tile(input_b.shape);
    std::cout << "OP = " << input_a.shape << " + " << input_b.shape << std::endl;
    const CoreCoord chip_size{8, 8};
    // Check input params against op constraints
    try {
        std::unique_ptr<EltwiseOpConstraintsBuilder> builder = EltwiseOpConstraintsFactory::Make(
            input_a.shape,
            input_a.memory_config,
            input_b.shape,
            input_b.memory_config,
            output.memory_config,
            chip_size);
        if (builder) {
            const auto op_constraints =
                (*builder)
                    .setDataTypeA(input_b.data_type)
                    .setDataTypeB(input_b.data_type)
                    .setDataTypeO(input_a.data_type)  // assuming output data type is the same as input_a
                    .build_constraints();
            std::cout << "size(op_contraints) =  " << op_constraints.size() << std::endl;

            if (op_constraints.size() == 0) {
                std::cout << "op_constraints is empty" << std::endl;
                GTEST_SKIP();
            }
        } else {
            std::cout << "builder is nullptr" << std::endl;
            GTEST_SKIP();
        }
    } catch (const std::exception& e) {
        std::cout << e.what() << std::endl;
        GTEST_FAIL();
    }

    // Run the test
    {
        auto input_tensor_a =
            ttnn::zeros(input_a.shape, input_a.data_type, input_a.layout, this->getDevice(), input_a.memory_config);
        auto input_tensor_b =
            ttnn::zeros(input_b.shape, input_b.data_type, input_b.layout, this->getDevice(), input_b.memory_config);

        auto call = [&] {
            const auto output_tensor = ttnn::add(input_tensor_a, input_tensor_b);
            return output_tensor;
        };

        // // get graph trace for ground truth
        auto json_trace = graph::query_trace(call);
        tt::log_info("Trace: {}", json_trace.dump(4));

        // L1 interface calls and checks against graph trace
        {
            auto l1_input_a = std::make_tuple(input_a.shape, input_a.data_type, input_a.layout, input_a.memory_config);
            auto l1_input_b = std::make_tuple(input_b.shape, input_b.data_type, input_b.layout, input_b.memory_config);
            auto l1_output = std::make_tuple(output.shape, output.data_type, output.layout, output.memory_config);

            auto l1_usage = EltwiseOpL1UsageFactory::Make(l1_input_a, l1_input_b, l1_output);

            compare_l1_circular_buffer_allocations(l1_usage->get_circular_buffer_l1_allocations_per_core(), json_trace);
            compare_l1_tensor_allocations(l1_usage->get_tensor_l1_allocations_per_core(), json_trace);
        }
    }
}

INSTANTIATE_TEST_SUITE_P(
    MlirInterfaceTestsREPEAT_MAX_BLOCK_SCALE,  // Prefix for the instantiated test suite
    EltwiseBinaryOpInterfaceTestFixture,       // Test suite name
    ::testing::Combine(
        ::testing::Values(OperandShapeTestParam{
            .shape = ttnn::Shape(tt::tt_metal::Array4D{3, 1, 32 * 32, 32 * 32}),
            .memory_config =
                {.memory_layout = tt::tt_metal::TensorMemoryLayout::HEIGHT_SHARDED,
                 .buffer_type = tt::tt_metal::BufferType::L1,
                 .shard_spec =
                     tt::tt_metal::ShardSpec{
                         CoreRangeSet{std::set<CoreRange>{CoreRange{CoreCoord{0, 0}, CoreCoord{3, 3}}}},
                         {6 * 32, 32 * 32},
                         ShardOrientation::COL_MAJOR}},
        }),

        ::testing::Values(OperandShapeTestParam{
            .shape = ttnn::Shape(tt::tt_metal::Array4D{1, 1, 32 * 32, 32 * 32}),
            .memory_config = ttnn::L1_MEMORY_CONFIG,
        }))

);

INSTANTIATE_TEST_SUITE_P(
    MlirInterfaceTests,                   // Prefix for the instantiated test suite
    EltwiseBinaryOpInterfaceTestFixture,  // Test suite name
    ::testing::Combine(
        ::testing::Values(
            OperandShapeTestParam{
                .shape = ttnn::Shape(tt::tt_metal::Array4D{1, 5, 32 * 64, 32}),
                .memory_config =
                    {.memory_layout = tt::tt_metal::TensorMemoryLayout::HEIGHT_SHARDED,
                     .buffer_type = tt::tt_metal::BufferType::L1,
                     .shard_spec =
                         tt::tt_metal::ShardSpec{
                             CoreRangeSet{std::set<CoreRange>{CoreRange{CoreCoord{0, 0}, CoreCoord{7, 7}}}},
                             {160, 32},
                             ShardOrientation::COL_MAJOR}},
            },
            OperandShapeTestParam{
                .shape = ttnn::Shape(tt::tt_metal::Array4D{1, 5, 32, 32 * 64}),
                .memory_config =
                    {.memory_layout = tt::tt_metal::TensorMemoryLayout::WIDTH_SHARDED,
                     .buffer_type = tt::tt_metal::BufferType::L1,
                     .shard_spec =
                         tt::tt_metal::ShardSpec{
                             CoreRangeSet{std::set<CoreRange>{CoreRange{CoreCoord{0, 0}, CoreCoord{7, 7}}}},
                             {5 * 32, 160},
                             ShardOrientation::COL_MAJOR}},
            },
            OperandShapeTestParam{
                .shape = ttnn::Shape(tt::tt_metal::Array4D{1, 1, 32, 32}),
                .memory_config = ttnn::L1_MEMORY_CONFIG,
            },
            OperandShapeTestParam{
                .shape = ttnn::Shape(tt::tt_metal::Array4D{4, 2, 5 * 32, 5 * 32}),
                .memory_config = ttnn::L1_MEMORY_CONFIG,
            }),

        ::testing::Values(
            OperandShapeTestParam{
                .shape = ttnn::Shape(tt::tt_metal::Array4D{1, 5, 32 * 64, 32}),
                .memory_config =
                    {.memory_layout = tt::tt_metal::TensorMemoryLayout::HEIGHT_SHARDED,
                     .buffer_type = tt::tt_metal::BufferType::L1,
                     .shard_spec =
                         tt::tt_metal::ShardSpec{
                             CoreRangeSet{std::set<CoreRange>{CoreRange{CoreCoord{0, 0}, CoreCoord{7, 7}}}},
                             {160, 32},
                             ShardOrientation::COL_MAJOR}},
            },
            OperandShapeTestParam{
                .shape = ttnn::Shape(tt::tt_metal::Array4D{1, 5, 32, 32 * 64}),
                .memory_config =
                    {.memory_layout = tt::tt_metal::TensorMemoryLayout::WIDTH_SHARDED,
                     .buffer_type = tt::tt_metal::BufferType::L1,
                     .shard_spec =
                         tt::tt_metal::ShardSpec{
                             CoreRangeSet{std::set<CoreRange>{CoreRange{CoreCoord{0, 0}, CoreCoord{7, 7}}}},
                             {5 * 32, 160},
                             ShardOrientation::COL_MAJOR}},
            },
            OperandShapeTestParam{
                .shape = ttnn::Shape(tt::tt_metal::Array4D{1, 1, 32, 32}),
                .memory_config = ttnn::L1_MEMORY_CONFIG,
            },
            OperandShapeTestParam{
                .shape = ttnn::Shape(tt::tt_metal::Array4D{1, 4, 32, 32}),
                .memory_config = ttnn::L1_MEMORY_CONFIG,
            },
            OperandShapeTestParam{
                .shape = ttnn::Shape(tt::tt_metal::Array4D{3, 1, 32, 32}),
                .memory_config = ttnn::L1_MEMORY_CONFIG,
            },
            OperandShapeTestParam{
                .shape = ttnn::Shape(tt::tt_metal::Array4D{3, 4, 32, 32}),
                .memory_config = ttnn::L1_MEMORY_CONFIG,
            },
            OperandShapeTestParam{
                .shape = ttnn::Shape(tt::tt_metal::Array4D{1, 1, 1, 32}),
                .memory_config = ttnn::L1_MEMORY_CONFIG,
            },
            OperandShapeTestParam{
                .shape = ttnn::Shape(tt::tt_metal::Array4D{1, 1, 32, 1}),
                .memory_config = ttnn::L1_MEMORY_CONFIG,
            },
            OperandShapeTestParam{
                .shape = ttnn::Shape(tt::tt_metal::Array4D{1, 1, 1, 1}),
                .memory_config = ttnn::L1_MEMORY_CONFIG,
            },
            OperandShapeTestParam{
                .shape = ttnn::Shape(tt::tt_metal::Array4D{4, 2, 5 * 32, 5 * 32}),
                .memory_config = ttnn::L1_MEMORY_CONFIG,
            },
            OperandShapeTestParam{
                .shape = ttnn::Shape(tt::tt_metal::Array4D{4, 2, 5 * 32, 1}),
                .memory_config = ttnn::L1_MEMORY_CONFIG,
            }))

);

INSTANTIATE_TEST_SUITE_P(
    MlirInterfaceTestsHEIGHT_SHARDED,     // Prefix for the instantiated test suite
    EltwiseBinaryOpInterfaceTestFixture,  // Test suite name
    ::testing::Combine(
        ::testing::Values(
            OperandShapeTestParam{
                .shape = ttnn::Shape(tt::tt_metal::Array4D{1, 5, 32 * 64, 32}),
                .memory_config =
                    {.memory_layout = tt::tt_metal::TensorMemoryLayout::HEIGHT_SHARDED,
                     .buffer_type = tt::tt_metal::BufferType::L1,
                     .shard_spec =
                         tt::tt_metal::ShardSpec{
                             CoreRangeSet{std::set<CoreRange>{CoreRange{CoreCoord{0, 0}, CoreCoord{7, 7}}}},
                             {160, 32},
                             ShardOrientation::COL_MAJOR}},
            },
            OperandShapeTestParam{
                .shape = ttnn::Shape(tt::tt_metal::Array4D{1, 1, 32 * 64, 32}),
                .memory_config =
                    {.memory_layout = tt::tt_metal::TensorMemoryLayout::HEIGHT_SHARDED,
                     .buffer_type = tt::tt_metal::BufferType::L1,
                     .shard_spec =
                         tt::tt_metal::ShardSpec{
                             CoreRangeSet{std::set<CoreRange>{CoreRange{CoreCoord{0, 0}, CoreCoord{7, 7}}}},
                             {160, 32},
                             ShardOrientation::COL_MAJOR}},
            }),

        ::testing::Values(
            OperandShapeTestParam{
                .shape = ttnn::Shape(tt::tt_metal::Array4D{1, 5, 32 * 64, 32}),
                .memory_config =
                    {.memory_layout = tt::tt_metal::TensorMemoryLayout::HEIGHT_SHARDED,
                     .buffer_type = tt::tt_metal::BufferType::L1,
                     .shard_spec =
                         tt::tt_metal::ShardSpec{
                             CoreRangeSet{std::set<CoreRange>{CoreRange{CoreCoord{0, 0}, CoreCoord{7, 7}}}},
                             {160, 32},
                             ShardOrientation::COL_MAJOR}},
            },
            OperandShapeTestParam{
                .shape = ttnn::Shape(tt::tt_metal::Array4D{1, 1, 32 * 64, 32}),
                .memory_config =
                    {.memory_layout = tt::tt_metal::TensorMemoryLayout::HEIGHT_SHARDED,
                     .buffer_type = tt::tt_metal::BufferType::L1,
                     .shard_spec =
                         tt::tt_metal::ShardSpec{
                             CoreRangeSet{std::set<CoreRange>{CoreRange{CoreCoord{0, 0}, CoreCoord{7, 7}}}},
                             {160, 32},
                             ShardOrientation::COL_MAJOR}},
            }

            ))

);

}  // namespace test
}  // namespace binary
}  // namespace operations
}  // namespace ttnn
