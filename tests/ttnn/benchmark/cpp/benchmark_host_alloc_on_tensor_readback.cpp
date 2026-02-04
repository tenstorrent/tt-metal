// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <benchmark/benchmark.h>

#include <ttnn/tensor/tensor.hpp>
#include "ttnn/tensor/tensor_ops.hpp"
#include "ttnn/distributed/types.hpp"
#include "ttnn/tensor/tensor_spec.hpp"
#include "ttnn/tensor/layout/tensor_layout.hpp"
#include "ttnn/distributed/api.hpp"

namespace {

ttnn::distributed::MeshDevice* device = nullptr;

void BM_host_alloc_on_tensor_readback(benchmark::State& state) {
    const size_t global_tensor_size = state.range(0);
    tt::tt_metal::Tensor device_tensor = tt::tt_metal::create_device_tensor(
        tt::tt_metal::TensorSpec(
            ttnn::Shape({global_tensor_size / sizeof(float) / device->num_devices()}),
            tt::tt_metal::TensorLayout(
                tt::tt_metal::DataType::FLOAT32,
                tt::tt_metal::PageConfig(tt::tt_metal::Layout::ROW_MAJOR),
                tt::tt_metal::MemoryConfig())),
        device);

    // Note we are reading garbage data from the device, but it is not important for this benchmark.
    for ([[maybe_unused]] auto _ : state) {
        auto host_tensor = device_tensor.cpu(/*blocking=*/true);
        benchmark::DoNotOptimize(host_tensor);
    }

    device_tensor.deallocate(/*force=*/true);
}

BENCHMARK(BM_host_alloc_on_tensor_readback)
    ->Unit(benchmark::kMicrosecond)
    ->RangeMultiplier(2)
    ->Range(1 << 10, 1 << 30);  // 1KB to 1GB, powers of 2

}  // namespace

int main(int argc, char** argv) {
    auto mesh_device = ttnn::distributed::open_mesh_device(
        ttnn::distributed::MeshShape{2, 4},
        DEFAULT_L1_SMALL_SIZE,
        DEFAULT_TRACE_REGION_SIZE,
        1,
        tt::tt_metal::DispatchCoreConfig{});

    device = mesh_device.get();

    ::benchmark::Initialize(&argc, argv);
    ::benchmark::RunSpecifiedBenchmarks();
    ::benchmark::Shutdown();

    return 0;
}
