// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <benchmark/benchmark.h>

#include <chrono>
#include <cstdint>
#include <optional>
#include <span>
#include <string>
#include <tt-metalium/distributed.hpp>
#include <vector>

#include "autograd/auto_context.hpp"
#include "autograd/tensor.hpp"
#include "core/random.hpp"
#include "core/tt_tensor_utils.hpp"
#include "metal/ops/rmsnorm_bw/rmsnorm_bw.hpp"
#include "metal/ops/rmsnorm_fw/rmsnorm_fw.hpp"
#include "ops/rmsnorm_op.hpp"
#include "ttnn/device.hpp"
#include "ttnn/tensor/shape/shape.hpp"

namespace {

struct RmsNormBwShape {
    std::vector<uint32_t> shape;  // [B, 1, S, C]
    std::string name;
};

struct TestConfig {
    int num_warmup_iterations = 5;
    int num_measurement_iterations = 50;
};

const TestConfig test_config = {
    .num_warmup_iterations = 5,
    .num_measurement_iterations = 50,
};

// Shapes aligned with polynorm / Llama-style workloads (batch, 1, sequence, hidden).
const std::vector<RmsNormBwShape> k_rmsnorm_bw_shapes = {
    {{1, 1, 256, 384}, "B1_S256_C384"},
    {{1, 1, 256, 2048}, "B1_S256_C2048"},
    {{4, 1, 256, 2048}, "B4_S256_C2048"},
    {{16, 1, 256, 2048}, "B16_S256_C2048"},
    {{1, 1, 2048, 2048}, "B1_S2048_C2048"},
};

constexpr float k_rmsnorm_epsilon = 0.0078125F;

ttnn::Tensor make_random_tensor(
    const ttnn::Shape& shape, ttnn::DataType dtype, ttnn::distributed::MeshDevice* device, uint32_t seed) {
    std::vector<float> data(shape.volume());
    ttml::core::parallel_generate(
        std::span{data.data(), data.size()}, []() { return std::uniform_real_distribution<float>(-1.0f, 1.0f); }, seed);
    return ttnn::Tensor::from_vector(
        data,
        ttnn::TensorSpec(
            shape, tt::tt_metal::TensorLayout(dtype, tt::tt_metal::Layout::TILE, ttnn::DRAM_MEMORY_CONFIG)),
        device);
}

void deallocate_grads(std::vector<std::optional<ttnn::Tensor>>& grads) {
    for (auto& g : grads) {
        if (g.has_value()) {
            g.value().deallocate();
            g.reset();
        }
    }
}

// One fused forward (with RMS intermediate) plus fused backward; frees RMS and grads after sync.
[[nodiscard]] bool run_kernel_rmsnorm_fw_bw_step(
    const ttnn::Tensor& input,
    const ttnn::Tensor& gamma,
    const ttnn::Tensor& dL_dout,
    ttnn::distributed::MeshDevice* device,
    float epsilon) {
    auto fw = ttml::metal::rmsnorm_fw(input, gamma, /*return_intermediates=*/true, epsilon);
    if (fw.size() < 2U || !fw[1].has_value()) {
        return false;
    }
    if (fw[0].has_value()) {
        fw[0].value().deallocate();
        fw[0].reset();
    }
    ttnn::Tensor rms = std::move(fw[1].value());
    fw[1].reset();

    auto grads = ttml::metal::rmsnorm_bw(input, gamma, rms, dL_dout);
    tt::tt_metal::distributed::Synchronize(device, std::nullopt);
    deallocate_grads(grads);
    rms.deallocate();
    return true;
}

// Backward-only 
void BM_RmsNormBw(benchmark::State& state) {
    const int shape_index = static_cast<int>(state.range(0));
    const auto& cfg = k_rmsnorm_bw_shapes[static_cast<size_t>(shape_index)];

    const auto device_id = 0;
    auto device = ttnn::device::open_mesh_device(device_id);
    device->enable_program_cache();

    const auto dtype = ttnn::DataType::BFLOAT16;
    const ttnn::Shape x_shape(cfg.shape);
    const uint32_t c_dim = cfg.shape[3];
    const ttnn::Shape gamma_shape({1, 1, 1, c_dim});
    const uint32_t seed = static_cast<uint32_t>(std::hash<std::string>{}(cfg.name));

    auto input = make_random_tensor(x_shape, dtype, device.get(), seed);
    auto gamma = make_random_tensor(gamma_shape, dtype, device.get(), seed + 1U);
    auto dL_dout = make_random_tensor(x_shape, dtype, device.get(), seed + 2U);

    auto fw = ttml::metal::rmsnorm_fw(input, gamma, /*return_intermediates=*/true, k_rmsnorm_epsilon);
    if (fw.size() < 2U || !fw[1].has_value()) {
        state.SkipWithError("rmsnorm_fw did not return RMS intermediate");
        return;
    }
    if (fw[0].has_value()) {
        fw[0].value().deallocate();
        fw[0].reset();
    }
    ttnn::Tensor rms = std::move(fw[1].value());
    fw[1].reset();

    const uint64_t elems_in = static_cast<uint64_t>(x_shape.volume());
    const uint64_t elems_gamma = static_cast<uint64_t>(gamma_shape.volume());
    const uint64_t elems_rms = static_cast<uint64_t>(rms.logical_shape().volume());
    const uint64_t elems_dout = elems_in;
    const uint64_t elems_din = elems_in;
    const uint64_t elems_dgamma = elems_gamma;
    const uint64_t total_elems = elems_in + elems_gamma + elems_rms + elems_dout + elems_din + elems_dgamma;
    const uint64_t total_dram_bytes = total_elems * sizeof(uint16_t);  // bf16

    for (int i = 0; i < test_config.num_warmup_iterations; ++i) {
        auto grads = ttml::metal::rmsnorm_bw(input, gamma, rms, dL_dout);
        tt::tt_metal::distributed::Synchronize(device.get(), std::nullopt);
        deallocate_grads(grads);
    }

    for ([[maybe_unused]] auto _ : state) {
        auto total_time = std::chrono::duration<double>::zero();

        for (int iter = 0; iter < test_config.num_measurement_iterations; ++iter) {
            const auto start = std::chrono::high_resolution_clock::now();
            auto grads = ttml::metal::rmsnorm_bw(input, gamma, rms, dL_dout);
            tt::tt_metal::distributed::Synchronize(device.get(), std::nullopt);
            const auto end = std::chrono::high_resolution_clock::now();
            total_time += end - start;
            deallocate_grads(grads);
        }

        const double avg_time_s = total_time.count() / static_cast<double>(test_config.num_measurement_iterations);
        const double time_us = avg_time_s * 1e6;
        const double gb_per_s = static_cast<double>(total_dram_bytes) / avg_time_s / 1e9;

        state.SetIterationTime(avg_time_s);
        state.SetLabel(cfg.name);
        state.counters["Time_us"] = time_us;
        state.counters["GB_per_s"] = gb_per_s;
        state.counters["Elems_M"] = static_cast<double>(elems_in) / 1e6;
    }

    rms.deallocate();
    dL_dout.deallocate();
    gamma.deallocate();
    input.deallocate();

    device->close();
}

// Kernel forward + kernel backward each iteration (same scope as `RmsNormBw_Composite` for fair timing).
void BM_RmsNormBwKernelFwBw(benchmark::State& state) {
    const int shape_index = static_cast<int>(state.range(0));
    const auto& cfg = k_rmsnorm_bw_shapes[static_cast<size_t>(shape_index)];

    const auto device_id = 0;
    auto device = ttnn::device::open_mesh_device(device_id);
    device->enable_program_cache();

    const auto dtype = ttnn::DataType::BFLOAT16;
    const ttnn::Shape x_shape(cfg.shape);
    const uint32_t c_dim = cfg.shape[3];
    const ttnn::Shape gamma_shape({1, 1, 1, c_dim});
    const uint32_t seed = static_cast<uint32_t>(std::hash<std::string>{}(cfg.name));

    auto input = make_random_tensor(x_shape, dtype, device.get(), seed);
    auto gamma = make_random_tensor(gamma_shape, dtype, device.get(), seed + 1U);
    auto dL_dout = make_random_tensor(x_shape, dtype, device.get(), seed + 2U);

    const uint64_t elems_in = static_cast<uint64_t>(x_shape.volume());

    for (int i = 0; i < test_config.num_warmup_iterations; ++i) {
        if (!run_kernel_rmsnorm_fw_bw_step(input, gamma, dL_dout, device.get(), k_rmsnorm_epsilon)) {
            state.SkipWithError("rmsnorm_fw did not return RMS intermediate");
            dL_dout.deallocate();
            gamma.deallocate();
            input.deallocate();
            device->close();
            return;
        }
    }

    for ([[maybe_unused]] auto _ : state) {
        auto total_time = std::chrono::duration<double>::zero();

        for (int iter = 0; iter < test_config.num_measurement_iterations; ++iter) {
            const auto start = std::chrono::high_resolution_clock::now();
            if (!run_kernel_rmsnorm_fw_bw_step(input, gamma, dL_dout, device.get(), k_rmsnorm_epsilon)) {
                state.SkipWithError("rmsnorm_fw did not return RMS intermediate");
                dL_dout.deallocate();
                gamma.deallocate();
                input.deallocate();
                device->close();
                return;
            }
            const auto end = std::chrono::high_resolution_clock::now();
            total_time += end - start;
        }

        const double avg_time_s = total_time.count() / static_cast<double>(test_config.num_measurement_iterations);
        const double time_us = avg_time_s * 1e6;

        state.SetIterationTime(avg_time_s);
        state.SetLabel(cfg.name + "_KernelFwBw");
        state.counters["Time_us"] = time_us;
        state.counters["Elems_M"] = static_cast<double>(elems_in) / 1e6;
    }

    dL_dout.deallocate();
    gamma.deallocate();
    input.deallocate();

    device->close();
}

// Composite path: full `rmsnorm_composite` forward (eltwise decomposition) plus backward with fixed upstream grad.
// Compare latency to `RmsNormBw_KernelFwBw` (kernel FW+BW per iteration), not to backward-only `RmsNormBw`.
void BM_RmsNormBwComposite(benchmark::State& state) {
    const int shape_index = static_cast<int>(state.range(0));
    const auto& cfg = k_rmsnorm_bw_shapes[static_cast<size_t>(shape_index)];

    const tt::tt_metal::distributed::MeshShape mesh(1, 1);
    ttml::autograd::ctx().open_device(mesh);
    auto* mesh_device = &ttml::autograd::ctx().get_device();
    mesh_device->enable_program_cache();

    const ttnn::Shape x_shape(cfg.shape);
    const uint32_t c_dim = cfg.shape[3];
    const ttnn::Shape gamma_shape({1, 1, 1, c_dim});
    const uint32_t seed = static_cast<uint32_t>(std::hash<std::string>{}(cfg.name));

    std::vector<float> x_host(x_shape.volume());
    std::vector<float> gamma_host(gamma_shape.volume());
    std::vector<float> dout_host(x_shape.volume());
    ttml::core::parallel_generate(
        std::span{x_host.data(), x_host.size()},
        []() { return std::uniform_real_distribution<float>(-1.0f, 1.0f); },
        seed);
    ttml::core::parallel_generate(
        std::span{gamma_host.data(), gamma_host.size()},
        []() { return std::uniform_real_distribution<float>(-1.0f, 1.0f); },
        seed + 1U);
    ttml::core::parallel_generate(
        std::span{dout_host.data(), dout_host.size()},
        []() { return std::uniform_real_distribution<float>(-1.0f, 1.0f); },
        seed + 2U);

    auto input_tt = ttml::core::from_vector(x_host, x_shape, mesh_device, ttnn::Layout::TILE);
    auto gamma_tt = ttml::core::from_vector(gamma_host, gamma_shape, mesh_device, ttnn::Layout::TILE);
    auto dL_dout_tt = ttml::core::from_vector(dout_host, x_shape, mesh_device, ttnn::Layout::TILE);

    const uint64_t elems_in = static_cast<uint64_t>(x_shape.volume());

    const auto run_composite_step = [&]() {
        ttml::autograd::ctx().reset_graph();
        auto x = ttml::autograd::create_tensor(input_tt, /*requires_grad=*/true);
        auto gamma = ttml::autograd::create_tensor(gamma_tt, /*requires_grad=*/true);
        auto out = ttml::ops::rmsnorm_composite(x, gamma, k_rmsnorm_epsilon);
        out->set_grad(dL_dout_tt);
        out->backward();
    };

    for (int i = 0; i < test_config.num_warmup_iterations; ++i) {
        run_composite_step();
        tt::tt_metal::distributed::Synchronize(mesh_device, std::nullopt);
    }

    for ([[maybe_unused]] auto _ : state) {
        auto total_time = std::chrono::duration<double>::zero();

        for (int iter = 0; iter < test_config.num_measurement_iterations; ++iter) {
            const auto start = std::chrono::high_resolution_clock::now();
            run_composite_step();
            tt::tt_metal::distributed::Synchronize(mesh_device, std::nullopt);
            const auto end = std::chrono::high_resolution_clock::now();
            total_time += end - start;
        }

        const double avg_time_s = total_time.count() / static_cast<double>(test_config.num_measurement_iterations);
        const double time_us = avg_time_s * 1e6;

        state.SetIterationTime(avg_time_s);
        state.SetLabel(cfg.name + "_Composite");
        state.counters["Time_us"] = time_us;
        state.counters["Elems_M"] = static_cast<double>(elems_in) / 1e6;
    }

    dL_dout_tt.deallocate();
    gamma_tt.deallocate();
    input_tt.deallocate();
    ttml::autograd::ctx().reset_graph();
    ttml::autograd::ctx().close_device();
}

}  // namespace

BENCHMARK(BM_RmsNormBw)
    ->DenseRange(0, static_cast<int>(k_rmsnorm_bw_shapes.size()) - 1, 1)
    ->Unit(benchmark::kMillisecond)
    ->UseManualTime()
    ->Iterations(1)
    ->Name("RmsNormBw");

BENCHMARK(BM_RmsNormBwComposite)
    ->DenseRange(0, static_cast<int>(k_rmsnorm_bw_shapes.size()) - 1, 1)
    ->Unit(benchmark::kMillisecond)
    ->UseManualTime()
    ->Iterations(1)
    ->Name("RmsNormBw_Composite");

BENCHMARK(BM_RmsNormBwKernelFwBw)
    ->DenseRange(0, static_cast<int>(k_rmsnorm_bw_shapes.size()) - 1, 1)
    ->Unit(benchmark::kMillisecond)
    ->UseManualTime()
    ->Iterations(1)
    ->Name("RmsNormBw_KernelFwBw");

BENCHMARK_MAIN();
