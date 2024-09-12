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
#include "tt_arch_types.h"
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
#include "ttnn/operations/eltwise/unary/unary.hpp"
#include "ttnn/operations/matmul/matmul_constraints.hpp"
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
    tt::tt_metal::Layout layout = tt::tt_metal::Layout::TILE;
};

class UnaryInterfaceTestFixture
    : public TTNNFixtureWithDevice,
      public testing::WithParamInterface<
          std::tuple<InputShapeTestParam, InputShapeTestParam, tt::tt_metal::IGraphProcessor::RunMode>> {};

TEST_P(UnaryInterfaceTestFixture, UnaryInterfaceTest) {
    auto param_combination = GetParam();
    auto input_a = std::get<0>(param_combination);
    auto input_o = std::get<1>(param_combination);
    auto run_mode = std::get<2>(param_combination);

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
    std::cout << "OP = " << input_a.shape << std::endl;
    ttnn::operations::unary::UnaryOpType unary_op_type = ttnn::operations::unary::UnaryOpType::RELU;
    tt::ARCH arch = tt::ARCH::WORMHOLE_B0;
    // Check input params against op constraints
    try {
        std::unique_ptr<unary::UnaryOpConstraintsBuilder> builder = unary::UnaryOpConstraintsFactory::Make(
            unary_op_type, arch, input_a.shape, input_a.memory_config, input_o.shape, input_o.memory_config);
        if (builder) {
            const auto op_constraints =
                (*builder)
                    .setDataTypeA(input_a.data_type)
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

        auto call = [&] {
            const auto output_tensor = ttnn::relu(input_tensor_a);
            return output_tensor;
        };
    }
}

INSTANTIATE_TEST_SUITE_P(
    UnaryInterfaceTests,        // Prefix for the instantiated test suite
    UnaryInterfaceTestFixture,  // Test suite name
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
                .shape = ttnn::Shape(tt::tt_metal::Array4D{4, 2, 5 * 32, 5 * 32}),
                .memory_config = ttnn::L1_MEMORY_CONFIG,
            }),

        ::testing::Values(tt::tt_metal::IGraphProcessor::RunMode::NO_DISPATCH)));

}  // namespace test
}  // namespace binary
}  // namespace operations
}  // namespace ttnn
