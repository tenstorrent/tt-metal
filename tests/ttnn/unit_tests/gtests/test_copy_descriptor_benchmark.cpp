// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <chrono>
#include <cstdint>
#include <iostream>
#include <memory>
#include <vector>

#include <gtest/gtest.h>

#include <tt-metalium/buffer_types.hpp>
#include <tt-metalium/experimental/tensor/mesh_tensor.hpp>

#include "tests/tt_metal/tt_metal/common/multi_device_fixture.hpp"

#include "ttnn/async_runtime.hpp"
#include "ttnn/common/queue_id.hpp"
#include "ttnn/operations/core/core.hpp"
#include "ttnn/operations/data_movement/copy/copy.hpp"
#include "ttnn/operations/data_movement/copy_new/copy_new.hpp"
#include "ttnn/operations/functions.hpp"
#include "ttnn/tensor/layout/page_config.hpp"
#include "ttnn/tensor/layout/tensor_layout.hpp"
#include "ttnn/tensor/shape/shape.hpp"
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/tensor/tensor_spec.hpp"

namespace {

using tt::tt_metal::BufferType;
using tt::tt_metal::DataType;
using tt::tt_metal::Layout;
using tt::tt_metal::MemoryConfig;
using tt::tt_metal::MeshTensor;
using tt::tt_metal::TensorMemoryLayout;
using tt::tt_metal::TensorSpec;
using tt::tt_metal::TensorTopology;

ttnn::Tensor allocate_tensor(tt::tt_metal::distributed::MeshDevice& mesh, const TensorSpec& spec) {
    return ttnn::Tensor(MeshTensor::allocate_on_device(mesh, spec, TensorTopology{}));
}

}  // namespace

class CopyDescriptorBenchmark : public tt::tt_metal::GenericMeshDeviceFixture {};

TEST_F(CopyDescriptorBenchmark, CorrectnessNonCached) {
    auto mesh_device = get_mesh_device();
    const ttnn::Shape shape{1, 1, 32, 32};
    MemoryConfig mem_cfg{TensorMemoryLayout::INTERLEAVED, BufferType::DRAM};
    TensorSpec spec(
        shape, tt::tt_metal::TensorLayout(DataType::BFLOAT16, tt::tt_metal::PageConfig(Layout::TILE), mem_cfg));

    ttnn::Tensor input = allocate_tensor(*mesh_device, spec);
    ttnn::Tensor out_legacy = allocate_tensor(*mesh_device, spec);
    ttnn::Tensor out_descriptor = allocate_tensor(*mesh_device, spec);

    const auto count = static_cast<size_t>(shape.volume());
    auto host_data = std::make_shared<std::vector<uint16_t>>(count, 0);
    for (size_t i = 0; i < count; ++i) {
        (*host_data)[i] = static_cast<uint16_t>((static_cast<uint32_t>(i) * 17 + 31) & 0xFFFF);
    }

    const ttnn::QueueId cq{0};
    ttnn::write_buffer(cq, input, {host_data});

    ttnn::copy(input, out_legacy);
    ttnn::copy_new(input, out_descriptor);

    EXPECT_TRUE(
        ttnn::allclose<::bfloat16>(ttnn::from_device(out_legacy), ttnn::from_device(out_descriptor), 1e-1f, 1e-5f));
}

TEST_F(CopyDescriptorBenchmark, CorrectnessCached) {
    auto mesh_device = get_mesh_device();
    const ttnn::Shape shape{1, 1, 32, 32};
    MemoryConfig mem_cfg{TensorMemoryLayout::INTERLEAVED, BufferType::DRAM};
    TensorSpec spec(
        shape, tt::tt_metal::TensorLayout(DataType::BFLOAT16, tt::tt_metal::PageConfig(Layout::TILE), mem_cfg));

    ttnn::Tensor input = allocate_tensor(*mesh_device, spec);
    ttnn::Tensor out_legacy = allocate_tensor(*mesh_device, spec);
    ttnn::Tensor out_descriptor = allocate_tensor(*mesh_device, spec);

    const auto count = static_cast<size_t>(shape.volume());
    auto host_data = std::make_shared<std::vector<uint16_t>>(count, 7);
    const ttnn::QueueId cq{0};
    ttnn::write_buffer(cq, input, {host_data});

    ttnn::copy(input, out_legacy);
    ttnn::copy_new(input, out_descriptor);

    ttnn::copy(input, out_legacy);
    ttnn::copy_new(input, out_descriptor);

    EXPECT_TRUE(
        ttnn::allclose<::bfloat16>(ttnn::from_device(out_legacy), ttnn::from_device(out_descriptor), 1e-1f, 1e-5f));
}

TEST_F(CopyDescriptorBenchmark, DispatchPerformance) {
    auto mesh_device = get_mesh_device();
    const ttnn::Shape shape{1, 1, 32, 32};
    MemoryConfig mem_cfg{TensorMemoryLayout::INTERLEAVED, BufferType::DRAM};
    TensorSpec spec(
        shape, tt::tt_metal::TensorLayout(DataType::BFLOAT16, tt::tt_metal::PageConfig(Layout::TILE), mem_cfg));

    ttnn::Tensor input = allocate_tensor(*mesh_device, spec);
    ttnn::Tensor out_legacy = allocate_tensor(*mesh_device, spec);
    ttnn::Tensor out_descriptor = allocate_tensor(*mesh_device, spec);

    const auto count = static_cast<size_t>(shape.volume());
    auto host_data = std::make_shared<std::vector<uint16_t>>(count, 42);
    const ttnn::QueueId cq{0};
    ttnn::write_buffer(cq, input, {host_data});
    ` constexpr int k_iters = 50000;

    const auto start_new = std::chrono::steady_clock::now();
    for (int i = 0; i < k_iters; ++i) {
        ttnn::copy_new(input, out_descriptor);
    }
    const auto mid = std::chrono::steady_clock::now();
    for (int i = 0; i < k_iters; ++i) {
        ttnn::copy(input, out_legacy);
    }
    const auto end = std::chrono::steady_clock::now();

    const double t_new = std::chrono::duration<double>(mid - start_new).count();
    const double t_old = std::chrono::duration<double>(end - mid).count();
    const double overhead_pct = (t_new / t_old - 1.0) * 100.0;
    std::cout << "copy_new vs copy dispatch overhead (%): " << overhead_pct << std::endl;

    EXPECT_LT(overhead_pct, 5.0);
}
