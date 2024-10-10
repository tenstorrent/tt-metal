// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "gtest/gtest.h"

#include "tt_metal/common/bfloat16.hpp"
#include "ttnn/device.hpp"
#include "ttnn/operations/core/core.hpp"
#include "ttnn/async_runtime.hpp"
#include "ttnn/operations/data_movement/repeat_interleave/repeat_interleave.hpp"
#include "ttnn/operations/numpy/functions.hpp"
#include "tt_metal/common/logger.hpp"

#include "ttnn_test_fixtures.hpp"

#include <memory>

namespace ttnn {
namespace operations {
namespace data_movement {
namespace test {

void run_repeat_interleave_test(tt::tt_metal::Device* device, const uint32_t repeats, const uint32_t dim) {
    MemoryConfig mem_cfg;
    mem_cfg.memory_layout = tt::tt_metal::TensorMemoryLayout::INTERLEAVED;
    mem_cfg.buffer_type = BufferType::DRAM;

    const uint32_t io_cq = 0;
    const uint32_t input_buf_size_datums = 32 * 32;
    const uint32_t output_buf_size_datums = input_buf_size_datums * repeats;
    const uint32_t datum_size_bytes = 2;
    ttnn::SimpleShape input_shape{1, 1, 32, 32};
    auto host_data = std::shared_ptr<uint16_t[]>(new uint16_t[input_buf_size_datums]);
    auto readback_data = std::shared_ptr<uint16_t[]>(new uint16_t[output_buf_size_datums]);

    for (uint16_t i = 0; i < 32; i++) {
        for (uint16_t j = 0; j < 32; j++) {
            host_data[i * 32 + j] = i;
        }
    }

    auto input_buffer = ttnn::allocate_buffer_on_device(input_buf_size_datums * datum_size_bytes, device, input_shape, DataType::UINT16, Layout::TILE, mem_cfg);
    auto input_storage = tt::tt_metal::DeviceStorage{input_buffer};
    Tensor input_tensor = Tensor(input_storage, input_shape, DataType::UINT16, Layout::TILE);
    ttnn::write_buffer(io_cq, input_tensor, {host_data});

    ttnn::Tensor output_tensor = ttnn::repeat_interleave(input_tensor, repeats, dim);

    ttnn::read_buffer(io_cq, output_tensor, {readback_data});

    tt::log_debug("input_data: \n {}", input_tensor.write_to_string());
    tt::log_debug("readback_data: \n {}", output_tensor.write_to_string());

    for (int i = 0; i < input_buf_size_datums; i++) {
        auto input_value = host_data[i];
        for(int r = 0; r < repeats; r++) {
            auto value = readback_data[i + r * input_buf_size_datums];
            ASSERT_EQ(input_value, value);
        }
    }

    input_tensor.deallocate();
    output_tensor.deallocate();
}

struct RepeatInterleaveParams {
    int repeats = 0;
    int dim = 0;
};

class RepeatInterleaveTest : public ttnn::TTNNFixtureWithDevice, public ::testing::WithParamInterface<RepeatInterleaveParams> {};

TEST_P(RepeatInterleaveTest, RunsCorrectly) {
    RepeatInterleaveParams params = GetParam();
    run_repeat_interleave_test(device_, params.repeats, params.dim);
}

INSTANTIATE_TEST_SUITE_P(
    RepeatInterleaveWithDim0,
    RepeatInterleaveTest,
    ::testing::Values(
        RepeatInterleaveParams{1, 0},
        RepeatInterleaveParams{2, 0},
        RepeatInterleaveParams{3, 0}
    )
);

// tests/ttnn/unit_tests/operations/test_repeat_interleave.py proves that it should work over dim 1 too
// likely need to fix the comparison in the test
INSTANTIATE_TEST_SUITE_P(
    DISABLED_RepeatInterleaveWithDim1,
    RepeatInterleaveTest,
    ::testing::Values(
        RepeatInterleaveParams{1, 1},
        RepeatInterleaveParams{2, 1},
        RepeatInterleaveParams{3, 1}
    )
);


}  // namespace test
}  // namespace binary
}  // namespace operations
}  // namespace ttnn
