// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <benchmark/benchmark.h>

#include <ttnn/tensor/tensor.hpp>
#include <ttnn/cpp/ttnn/operations/core/core.hpp>
#include "bfloat16.hpp"
#include "constants.hpp"
#include "ttnn/distributed/types.hpp"
#include "ttnn/tensor/tensor_spec.hpp"
#include "ttnn/tensor/layout/tensor_layout.hpp"
#include "ttnn/distributed/api.hpp"

#include <list>

namespace tt::tt_metal {
namespace {

ttnn::distributed::MeshDevice* device = nullptr;

void BM_host_bfloat8_conversion(benchmark::State& state) {
    const auto tensor_shape = ttnn::Shape{state.range(0), constants::TILE_HEIGHT, constants::TILE_WIDTH};

    std::vector<Tensor> tensors;
    for (int i = 0; i < device->num_devices(); i++) {
        std::vector<bfloat16> host_data;
        host_data.reserve(tensor_shape.volume());
        for (int j = 0; j < tensor_shape.volume(); j++) {
            host_data.push_back(bfloat16(j));
        }
        tensors.push_back(Tensor::from_vector(
            std::move(host_data),
            TensorSpec(tensor_shape, TensorLayout(DataType::BFLOAT16, Layout::ROW_MAJOR, MemoryConfig{}))));
    }
    auto tensor = ttnn::distributed::from_host_shards(tensors, device->shape());

    for ([[maybe_unused]] auto _ : state) {
        auto converted_tensor = ttnn::to_dtype(tensor, ttnn::DataType::BFLOAT8_B);
        benchmark::DoNotOptimize(converted_tensor);
    }
}

}  // namespace

BENCHMARK(tt::tt_metal::BM_host_bfloat8_conversion)
    ->Unit(benchmark::kMicrosecond)
    ->RangeMultiplier(2)
    ->Range(1 << 10, 1 << 14);

}  // namespace tt::tt_metal

int main(int argc, char** argv) {
    auto mesh_device = ttnn::distributed::open_mesh_device(
        ttnn::distributed::MeshShape{2, 4},
        DEFAULT_L1_SMALL_SIZE,
        DEFAULT_TRACE_REGION_SIZE,
        1,
        tt::tt_metal::DispatchCoreConfig{});

    tt::tt_metal::device = mesh_device.get();

    ::benchmark::Initialize(&argc, argv);
    ::benchmark::RunSpecifiedBenchmarks();
    ::benchmark::Shutdown();

    return 0;
}
