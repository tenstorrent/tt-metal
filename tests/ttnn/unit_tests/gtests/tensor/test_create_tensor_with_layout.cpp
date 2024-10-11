// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "gtest/gtest.h"

#include "tt_metal/common/bfloat16.hpp"
#include "ttnn/device.hpp"
#include "ttnn/operations/core/core.hpp"
#include "ttnn/async_runtime.hpp"
#include "ttnn/operations/numpy/functions.hpp"
#include "tt_metal/common/logger.hpp"

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/tensor/tensor_layout.hpp"
#include "ttnn/tensor/types.hpp"
#include "ttnn_test_fixtures.hpp"

namespace {
struct Inputs {
    ttnn::SimpleShape shape;
    TensorLayout layout;
};

struct Expected {
    ttnn::SimpleShape padded_shape;
};

struct CreateTensorParams {
    Inputs inputs;
    Expected expected;
};

}

class CreateTensorWithLayoutTest : public ttnn::TTNNFixtureWithDevice, public ::testing::WithParamInterface<CreateTensorParams> {};

TEST_P(CreateTensorWithLayoutTest, Tile) {
    CreateTensorParams params = GetParam();

    auto tensor = tt::tt_metal::create_device_tensor(params.inputs.shape, params.inputs.layout, device_);
    EXPECT_EQ(tensor.get_padded_shape(), params.expected.padded_shape);
    EXPECT_EQ(tensor.get_logical_shape(), params.inputs.shape);
}

namespace {
const tt::tt_metal::MemoryConfig DefaultMemoryConfig{tt::tt_metal::TensorMemoryLayout::INTERLEAVED, tt::tt_metal::BufferType::DRAM, std::nullopt};
}

INSTANTIATE_TEST_SUITE_P(
    CreateTensorWithLayoutTestWithShape,
    CreateTensorWithLayoutTest,
    ::testing::Values(
        CreateTensorParams{
            Inputs{
                .shape=ttnn::SimpleShape({1, 1, 32, 32}),
                .layout=TensorLayout(DataType::BFLOAT16, Layout::TILE, DefaultMemoryConfig)
            },
            Expected{
                .padded_shape=ttnn::SimpleShape({1, 1, 32, 32})
            }
        },

        CreateTensorParams{
            Inputs{
                .shape=ttnn::SimpleShape({1, 1, 16, 10}),
                .layout=TensorLayout(DataType::BFLOAT16, Layout::TILE, DefaultMemoryConfig)
            },
            Expected{
                .padded_shape=ttnn::SimpleShape({1, 1, 32, 32})
            }
        },

        CreateTensorParams{
            Inputs{
                .shape=ttnn::SimpleShape({1, 1, 16, 10}),
                .layout=TensorLayout(DataType::BFLOAT16, Layout::ROW_MAJOR, DefaultMemoryConfig)
            },
            Expected{
                .padded_shape=ttnn::SimpleShape({1, 1, 16, 10})
            }
        }
    )
);
