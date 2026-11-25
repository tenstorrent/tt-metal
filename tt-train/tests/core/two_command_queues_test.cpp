// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include <vector>

#include "autograd/auto_context.hpp"
#include "ttnn/events.hpp"
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/tensor/tensor_ops.hpp"

TEST(TwoCommandQueuesTest, Cq0ReadsTheCq1WriteAfterTheEvent) {
    auto& ctx = ttml::autograd::ctx();
    ctx.open_device(tt::tt_metal::distributed::MeshShape(1, 1), {}, /*num_command_queues=*/2);
    auto* device = &ctx.get_device();
    EXPECT_EQ(device->num_hw_cqs(), 2);

    std::vector<float> values(32 * 64);
    for (size_t i = 0; i < values.size(); ++i) {
        values[i] = static_cast<float>(i % 128);
    }
    const tt::tt_metal::TensorSpec spec(
        ttnn::Shape({1, 1, 32, 64}),
        tt::tt_metal::TensorLayout(
            ttnn::DataType::BFLOAT16, ttnn::PageConfig(ttnn::Layout::TILE), ttnn::DRAM_MEMORY_CONFIG));
    auto on_device = ttnn::create_device_tensor(spec, device);
    ttnn::copy_to_device(ttnn::Tensor::from_vector(values, spec), on_device, ttnn::QueueId(1));
    const auto written = ttnn::events::record_mesh_event(device, ttnn::QueueId(1));
    ttnn::events::wait_for_mesh_event(ttnn::QueueId(0), written);
    EXPECT_EQ(on_device.cpu(/*blocking=*/true, ttnn::QueueId(0)).to_vector<float>(), values);

    ctx.close_device();
}
