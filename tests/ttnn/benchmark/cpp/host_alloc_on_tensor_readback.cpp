// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <benchmark/benchmark.h>

#include <ttnn/tensor/tensor.hpp>
#include "ttnn/tensor/tensor_impl.hpp"
#include "ttnn/distributed/types.hpp"
#include "ttnn/tensor/tensor_spec.hpp"
#include "ttnn/tensor/layout/tensor_layout.hpp"
#include "ttnn/distributed/api.hpp"

namespace {

ttnn::distributed::MeshDevice* device = nullptr;

void BM_host_alloc_on_tensor_readback(benchmark::State& state) {
    const size_t global_tensor_size = state.range(0);
    tt::tt_metal::Tensor device_tensor = tt::tt_metal::allocate_tensor_on_mesh(
        tt::tt_metal::TensorSpec(
            ttnn::Shape({global_tensor_size / sizeof(float) / device->num_devices()}),
            tt::tt_metal::TensorLayout(
                tt::tt_metal::DataType::FLOAT32,
                tt::tt_metal::PageConfig(tt::tt_metal::Layout::ROW_MAJOR),
                tt::tt_metal::MemoryConfig())),
        device);

    // Note we are reading garbage data from the device, but it is not important for this benchmark.
    for (auto _ : state) {
        auto host_tensor = device_tensor.cpu(/*blocking=*/true);
        benchmark::DoNotOptimize(host_tensor);
    }

    device_tensor.deallocate(/*force=*/true);
}

BENCHMARK(BM_host_alloc_on_tensor_readback)
    ->Unit(benchmark::kMicrosecond)
    ->Iterations(5)
    ->Arg(1 << 10)    // 1KB
    ->Arg(2 << 10)    // 2KB
    ->Arg(4 << 10)    // 4KB
    ->Arg(8 << 10)    // 8KB
    ->Arg(16 << 10)   // 16KB
    ->Arg(32 << 10)   // 32KB
    ->Arg(64 << 10)   // 64KB
    ->Arg(128 << 10)  // 128KB
    ->Arg(256 << 10)  // 256KB
    ->Arg(512 << 10)  // 512KB
    ->Arg(1 << 20)    // 1MB
    ->Arg(2 << 20)    // 2MB
    ->Arg(4 << 20)    // 4MB
    ->Arg(8 << 20)    // 8MB
    ->Arg(16 << 20)   // 16MB
    ->Arg(32 << 20)   // 32MB
    ->Arg(64 << 20)   // 64MB
    ->Arg(128 << 20)  // 128MB
    ->Arg(256 << 20)  // 256MB
    ->Arg(512 << 20)  // 512MB
    ->Arg(1 << 30);   // 1GB

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
