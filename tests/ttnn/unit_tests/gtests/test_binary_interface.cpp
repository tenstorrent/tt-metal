// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include <functional>
#include <optional>
#include <string>
#include <tuple>
#include <variant>
#include <vector>

#include "common/constants.hpp"
#include "gtest/gtest.h"
#include "impl/event/event.hpp"
#include "impl/program/program.hpp"
#include "tests/tt_metal/tt_metal/unit_tests_common/common/common_fixture.hpp"
#include "tt_metal/common/logger.hpp"
#include "ttnn/cpp/ttnn/operations/eltwise/unary/unary_constraints.hpp"
#include "ttnn/device.hpp"
#include "ttnn/graph/graph_operation_queries.hpp"
#include "ttnn/graph/graph_processor.hpp"
#include "ttnn/graph/graph_trace_utils.hpp"
#include "ttnn/operations/core/core.hpp"
#include "ttnn/operations/creation.hpp"
#include "ttnn/operations/eltwise/binary/binary.hpp"
#include "ttnn/operations/eltwise/binary/binary_constraints.hpp"
#include "ttnn/operations/eltwise/binary/binary_l1_interface.hpp"
#include "ttnn/operations/matmul/matmul_constraints.hpp"
#include "ttnn/operations/normalization/softmax/softmax.hpp"
#include "ttnn/operations/normalization/softmax/softmax_constraints.hpp"
#include "ttnn/operations/reduction/generic/generic_reductions.hpp"
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/tensor/types.hpp"
#include "ttnn/types.hpp"
#include "ttnn_test_fixtures.hpp"

namespace ttnn {
namespace operations {
namespace binary {
namespace test {

struct InputShapeTestParam {
    ttnn::types::Shape shape;
    tt::tt_metal::MemoryConfig memory_config;
    tt::tt_metal::DataType data_type = tt::tt_metal::DataType::BFLOAT16;
};

class BinaryInterfaceTestFixture : public TTNNFixtureWithDevice,
                                   public testing::WithParamInterface<std::tuple<
                                       InputShapeTestParam,
                                       InputShapeTestParam,
                                       InputShapeTestParam,
                                       tt::tt_metal::IGraphProcessor::RunMode>> {};

TEST_P(BinaryInterfaceTestFixture, BinaryInterfaceTest) {
    auto param_combination = GetParam();
    auto input_a = std::get<0>(param_combination);
    auto input_b = std::get<1>(param_combination);
    auto input_o = std::get<2>(param_combination);
    auto run_mode = std::get<3>(param_combination);

    // pad input shapes (this isn't happening automagically)
    auto pad_shape_to_tile = [](const ttnn::Shape& shape) {
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
    };
    input_a.shape = pad_shape_to_tile(input_a.shape);
    input_b.shape = pad_shape_to_tile(input_b.shape);
    std::cout << "OP = " << input_a.shape << " + " << input_b.shape << std::endl;

    // Check input params against op constraints
    try {
        std::unique_ptr<EltwiseOpConstraintsBuilder> builder = EltwiseOpConstraintsFactory::Make(
            input_a.shape, input_a.memory_config, input_b.shape, input_b.memory_config, input_o.memory_config);
        if (builder) {
            const auto op_constraints =
                (*builder)
                    .setDataTypeA(input_a.data_type)
                    .setDataTypeB(input_b.data_type)
                    .setDataTypeO(input_a.data_type)  // assuming output data type is the same as input_a
                    .build_constraints();
            std::cout << "size(op_contraints) =  " << op_constraints.size() << std::endl;

            if (op_constraints.size() == 0) {
                std::cout << "op_constraints is empty" << std::endl;
                GTEST_SKIP();
            }
            // Run the test
            {
                for (const auto& op_constraint : op_constraints) {
                    auto input_tensor_a = ttnn::zeros(
                        input_a.shape,
                        op_constraint.getDataTypeA(),
                        op_constraint.getTileLayoutA(),
                        this->getDevice(),
                        input_a.memory_config);
                    auto input_tensor_b = ttnn::zeros(
                        input_b.shape,
                        op_constraint.getDataTypeB(),
                        op_constraint.getTileLayoutB(),
                        this->getDevice(),
                        input_b.memory_config);

                    auto call = [&] {
                        const auto output_tensor = ttnn::add(input_tensor_a, input_tensor_b);
                        return output_tensor;
                    };

                    // // get graph trace for ground truth
                    auto json_trace = graph::query_trace(call);
                    tt::log_info("Trace: {}", json_trace.dump(4));
                }
            }
        } else {
            std::cout << "builder is nullptr" << std::endl;
            GTEST_SKIP();
        }
    } catch (const std::exception& e) {
        std::cout << e.what() << std::endl;
        GTEST_FAIL();
    }
}

INSTANTIATE_TEST_SUITE_P(
    BinaryInterfaceTests,        // Prefix for the instantiated test suite
    BinaryInterfaceTestFixture,  // Test suite name
    ::testing::Combine(
        ::testing::Values(
            InputShapeTestParam{
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
            InputShapeTestParam{
                .shape = ttnn::Shape(tt::tt_metal::Array4D{1, 5, 32, 32 * 64}),
                .memory_config =
                    {.memory_layout = tt::tt_metal::TensorMemoryLayout::WIDTH_SHARDED,
                     .buffer_type = tt::tt_metal::BufferType::L1,
                     .shard_spec =
                         tt::tt_metal::ShardSpec{
                             CoreRangeSet{std::set<CoreRange>{CoreRange{CoreCoord{0, 0}, CoreCoord{7, 7}}}},
                             {32, 160},
                             ShardOrientation::COL_MAJOR}},
            },
            InputShapeTestParam{
                .shape = ttnn::Shape(tt::tt_metal::Array4D{1, 1, 32, 32}),
                .memory_config = ttnn::L1_MEMORY_CONFIG,
            },
            InputShapeTestParam{
                .shape = ttnn::Shape(tt::tt_metal::Array4D{4, 2, 5 * 32, 5 * 32}),
                .memory_config = ttnn::L1_MEMORY_CONFIG,
            }),

        ::testing::Values(
            InputShapeTestParam{
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
            InputShapeTestParam{
                .shape = ttnn::Shape(tt::tt_metal::Array4D{1, 5, 32, 32 * 64}),
                .memory_config =
                    {.memory_layout = tt::tt_metal::TensorMemoryLayout::WIDTH_SHARDED,
                     .buffer_type = tt::tt_metal::BufferType::L1,
                     .shard_spec =
                         tt::tt_metal::ShardSpec{
                             CoreRangeSet{std::set<CoreRange>{CoreRange{CoreCoord{0, 0}, CoreCoord{7, 7}}}},
                             {32, 160},
                             ShardOrientation::COL_MAJOR}},
            },
            InputShapeTestParam{
                .shape = ttnn::Shape(tt::tt_metal::Array4D{1, 1, 32, 32}),
                .memory_config = ttnn::L1_MEMORY_CONFIG,
            },
            InputShapeTestParam{
                .shape = ttnn::Shape(tt::tt_metal::Array4D{1, 4, 32, 32}),
                .memory_config = ttnn::L1_MEMORY_CONFIG,
            },
            InputShapeTestParam{
                .shape = ttnn::Shape(tt::tt_metal::Array4D{3, 1, 32, 32}),
                .memory_config = ttnn::L1_MEMORY_CONFIG,
            },
            InputShapeTestParam{
                .shape = ttnn::Shape(tt::tt_metal::Array4D{3, 4, 32, 32}),
                .memory_config = ttnn::L1_MEMORY_CONFIG,
            },
            InputShapeTestParam{
                .shape = ttnn::Shape(tt::tt_metal::Array4D{1, 1, 1, 32}),
                .memory_config = ttnn::L1_MEMORY_CONFIG,
            },
            InputShapeTestParam{
                .shape = ttnn::Shape(tt::tt_metal::Array4D{1, 1, 32, 1}),
                .memory_config = ttnn::L1_MEMORY_CONFIG,
            },
            InputShapeTestParam{
                .shape = ttnn::Shape(tt::tt_metal::Array4D{1, 1, 1, 1}),
                .memory_config = ttnn::L1_MEMORY_CONFIG,
            },
            InputShapeTestParam{
                .shape = ttnn::Shape(tt::tt_metal::Array4D{4, 2, 5 * 32, 5 * 32}),
                .memory_config = ttnn::L1_MEMORY_CONFIG,
            },
            InputShapeTestParam{
                .shape = ttnn::Shape(tt::tt_metal::Array4D{4, 2, 5 * 32, 1}),
                .memory_config = ttnn::L1_MEMORY_CONFIG,
            }),
        ::testing::Values(
            InputShapeTestParam{
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
            InputShapeTestParam{
                .shape = ttnn::Shape(tt::tt_metal::Array4D{1, 5, 32, 32 * 64}),
                .memory_config =
                    {.memory_layout = tt::tt_metal::TensorMemoryLayout::WIDTH_SHARDED,
                     .buffer_type = tt::tt_metal::BufferType::L1,
                     .shard_spec =
                         tt::tt_metal::ShardSpec{
                             CoreRangeSet{std::set<CoreRange>{CoreRange{CoreCoord{0, 0}, CoreCoord{7, 7}}}},
                             {32, 160},
                             ShardOrientation::COL_MAJOR}},
            },
            InputShapeTestParam{
                .shape = ttnn::Shape(tt::tt_metal::Array4D{1, 1, 32, 32}),
                .memory_config = ttnn::L1_MEMORY_CONFIG,
            },
            InputShapeTestParam{
                .shape = ttnn::Shape(tt::tt_metal::Array4D{4, 2, 5 * 32, 5 * 32}),
                .memory_config = ttnn::L1_MEMORY_CONFIG,
            }),

        // ::testing::Values(tt::tt_metal::IGraphProcessor::RunMode::NO_DISPATCH,
        // tt::tt_metal::IGraphProcessor::RunMode::NORMAL)
        ::testing::Values(tt::tt_metal::IGraphProcessor::RunMode::NO_DISPATCH)));

INSTANTIATE_TEST_SUITE_P(
    BinaryInterfaceTestsHEIGHT_SHARDED,  // Prefix for the instantiated test suite
    BinaryInterfaceTestFixture,          // Test suite name
    ::testing::Combine(
        ::testing::Values(
            InputShapeTestParam{
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
            InputShapeTestParam{
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
            InputShapeTestParam{
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
            InputShapeTestParam{
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

            ),

        ::testing::Values(
            InputShapeTestParam{
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
            InputShapeTestParam{
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

        // ::testing::Values(tt::tt_metal::IGraphProcessor::RunMode::NO_DISPATCH,
        // tt::tt_metal::IGraphProcessor::RunMode::NORMAL)
        ::testing::Values(tt::tt_metal::IGraphProcessor::RunMode::NO_DISPATCH)));

}  // namespace test
}  // namespace binary
}  // namespace operations
}  // namespace ttnn
