// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// ============================================================================
// Distributed RMSNorm op-level benchmark
//
// Measures the average wall-clock time of a forward + synthetic-grad backward step of RMSNorm over a
// hidden dim sharded across a 1 x TP device line, for TP in {2, 4} (whichever the machine has).
//
// Three configurations are compared per TP:
//   gather        — all_gather(x), all_gather(gamma) -> ops::rmsnorm on the full row -> scatter the output;
//                   the backward runs the same collectives in reverse.
//   distributed   — ops::distributed::rmsnorm: pre/post all-gather forward, LocalSum -> all_gather ->
//                   ExternalScale backward (this PR).
//   local         — ops::rmsnorm on the C/TP shard with no communication; a lower bound, not a valid norm.
//
// It also times the one ttnn::sum that reduces the per-core fp32 dgamma partials of the distributed backward,
// to show how much of the step the host-side dgamma reduction costs.
//
// Timed regions enqueue multiple steps and synchronize once at the end; this amortizes dispatch/sync
// latency and is closer to how model-level runs queue TTNN operations.
//
// RMSNORM_BENCH_TP (comma separated, default "2,4") selects the TP sizes; sizes above the device count are
// skipped. RMSNORM_BENCH_S and RMSNORM_BENCH_C override the sequence length and hidden dim. Each mesh shape needs a
// matching mesh graph descriptor, so on a Blackhole QuietBox 2 (2 x P300) run:
//   TT_VISIBLE_DEVICES=0,1 TT_MESH_GRAPH_DESC_PATH=.../p300_mesh_graph_descriptor.textproto \
//       RMSNORM_BENCH_TP=2 rmsnorm_distributed_benchmark
//   TT_MESH_GRAPH_DESC_PATH=.../p300_x2_mesh_graph_descriptor.textproto RMSNORM_BENCH_TP=4 \
//       rmsnorm_distributed_benchmark
// ============================================================================

#include <benchmark/benchmark.h>
#include <fmt/format.h>
#include <fmt/ranges.h>

#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <sstream>
#include <stdexcept>
#include <string>
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/host_api.hpp>
#include <vector>

#include "autograd/auto_context.hpp"
#include "autograd/tensor.hpp"
#include "benchmark_utils.hpp"
#include "core/compute_kernel_config.hpp"
#include "core/tt_tensor_utils.hpp"
#include "ops/distributed/comm_ops.hpp"
#include "ops/distributed/rmsnorm_op.hpp"
#include "ops/rmsnorm_op.hpp"
#include "test_utils/random_data.hpp"
#include "ttnn/distributed/distributed_tensor.hpp"
#include "ttnn_fixed/distributed/tt_metal.hpp"

namespace {

struct SweepConfig {
    std::vector<uint32_t> tensor_parallel_sizes = {2, 4};
    uint32_t batch_size = 1;
    uint32_t sequence_length = 8192;
    uint32_t hidden_dim = 5120;
    uint32_t num_warmup = 2;
    uint32_t num_measure = 5;
};

enum class Mode { Gather, Distributed, Local };

constexpr float kEpsilon = 1e-5F;

std::string mode_name(Mode mode) {
    switch (mode) {
        case Mode::Gather: return "gather";
        case Mode::Distributed: return "distributed";
        case Mode::Local: return "local";
    }
    return "unknown";
}

ttnn::Tensor shard_last_dim(const std::vector<float>& values, const ttnn::Shape& shape) {
    auto* device = &ttml::autograd::ctx().get_device();
    auto mapper = ttnn::distributed::shard_tensor_to_mesh_mapper(*device, 3);
    return ttml::core::from_vector<float, ttnn::DataType::BFLOAT16>(
        values, shape, device, ttnn::Layout::TILE, mapper.get());
}

template <typename Fn>
double time_steps_ms(tt::tt_metal::distributed::MeshDevice& device, const SweepConfig& cfg, Fn&& run_step) {
    for (uint32_t step = 0; step < cfg.num_warmup; ++step) {
        run_step();
    }
    tt::tt_metal::distributed::Synchronize(device, std::nullopt);

    const auto t0 = std::chrono::high_resolution_clock::now();
    for (uint32_t step = 0; step < cfg.num_measure; ++step) {
        run_step();
    }
    tt::tt_metal::distributed::Synchronize(device, std::nullopt);
    const auto t1 = std::chrono::high_resolution_clock::now();
    return std::chrono::duration<double, std::milli>(t1 - t0).count() / cfg.num_measure;
}

struct TpResult {
    double gather_ms = 0.0;
    double distributed_ms = 0.0;
    double local_ms = 0.0;
    double dgamma_sum_ms = 0.0;
};

double run_mode(const SweepConfig& cfg, Mode mode) {
    auto& device = ttml::autograd::ctx().get_device();
    device.clear_program_cache();

    const ttnn::Shape x_shape({cfg.batch_size, 1U, cfg.sequence_length, cfg.hidden_dim});
    const ttnn::Shape gamma_shape({1U, 1U, 1U, cfg.hidden_dim});
    const size_t x_count = static_cast<size_t>(cfg.batch_size) * cfg.sequence_length * cfg.hidden_dim;
    const uint32_t seed = ttml::benchmark_utils::seed_from_name(mode_name(mode));
    const auto x_host = ttml::test_utils::make_uniform_vector<float>(x_count, -1.0F, 1.0F, seed);
    const auto gamma_host = ttml::test_utils::make_uniform_vector<float>(cfg.hidden_dim, 0.5F, 1.5F, seed + 1U);

    const auto x_value = shard_last_dim(x_host, x_shape);
    const auto gamma_value = shard_last_dim(gamma_host, gamma_shape);
    const auto grad_value = ttml::core::ones_like(x_value);
    tt::tt_metal::distributed::Synchronize(device, std::nullopt);

    const auto run_step = [&]() {
        auto x = ttml::autograd::create_tensor(x_value, /*requires_grad=*/true);
        auto gamma = ttml::autograd::create_tensor(gamma_value, /*requires_grad=*/true);
        ttml::autograd::TensorPtr out;
        switch (mode) {
            case Mode::Gather: {
                auto x_full = ttml::ops::distributed::all_gather(x, /* dim */ 3);
                auto gamma_full = ttml::ops::distributed::all_gather(gamma, /* dim */ 3);
                out = ttml::ops::distributed::scatter(ttml::ops::rmsnorm(x_full, gamma_full, kEpsilon), /* dim */ 3);
                break;
            }
            case Mode::Distributed: out = ttml::ops::distributed::rmsnorm(x, gamma, kEpsilon); break;
            case Mode::Local: out = ttml::ops::rmsnorm(x, gamma, kEpsilon); break;
        }
        out->set_grad(grad_value);
        out->backward();
        ttml::autograd::ctx().reset_graph();
    };

    const double step_ms = time_steps_ms(device, cfg, run_step);
    ttml::autograd::ctx().reset_graph();
    tt::tt_metal::distributed::Synchronize(device, std::nullopt);
    return step_ms;
}

// The distributed backward leaves dgamma as [1,1,num_cores*32,C/TP] fp32 per-core partials; this is the cost of the
// ttnn::sum that finishes the reduction.
double run_dgamma_sum(const SweepConfig& cfg, uint32_t tp) {
    auto& device = ttml::autograd::ctx().get_device();
    const auto grid = device.compute_with_storage_grid_size();
    const uint32_t num_rows = cfg.batch_size * cfg.sequence_length / 32U;
    const uint32_t num_cores = std::min<uint32_t>(grid.x * grid.y, num_rows);
    const ttnn::Shape partials_shape({1U, 1U, num_cores * 32U, cfg.hidden_dim / tp});
    const auto partials = ttml::core::zeros(partials_shape, &device, ttnn::DataType::FLOAT32);
    tt::tt_metal::distributed::Synchronize(device, std::nullopt);

    return time_steps_ms(device, cfg, [&]() {
        (void)ttnn::sum(
            partials,
            ttsl::SmallVector<int>{0, 1, 2},
            /* keep_dim */ true,
            /* output_mem_config */ std::nullopt,
            ttml::core::ComputeKernelConfig::precise());
    });
}

TpResult run_tp(const SweepConfig& cfg, uint32_t tp) {
    ttml::ttnn_fixed::distributed::enable_fabric(tp);
    ttml::autograd::ctx().open_device(tt::tt_metal::distributed::MeshShape(1, tp));
    ttml::autograd::ctx().set_seed(42);

    TpResult result{};
    auto timed = [&](Mode mode) {
        fmt::print(stderr, "[TP={}] running {}...\n", tp, mode_name(mode));
        const double ms = run_mode(cfg, mode);
        fmt::print(stderr, "[TP={}] {}: {:.2f} ms/step\n", tp, mode_name(mode), ms);
        return ms;
    };
    result.local_ms = timed(Mode::Local);
    result.distributed_ms = timed(Mode::Distributed);
    result.gather_ms = timed(Mode::Gather);
    result.dgamma_sum_ms = run_dgamma_sum(cfg, tp);
    fmt::print(stderr, "[TP={}] dgamma ttnn::sum: {:.2f} ms\n", tp, result.dgamma_sum_ms);

    ttml::autograd::ctx().close_device();
    return result;
}

struct RowSummary {
    uint32_t tp = 0;
    TpResult result;
};

void print_table(const SweepConfig& cfg, const std::vector<RowSummary>& rows) {
    fmt::print("\n");
    fmt::print(
        "RMSNorm fwd+bw step, B={} S={} C={} (ms per step, lower is better)\n"
        "| TP | (a) gather->rmsnorm->scatter | (b) distributed rmsnorm | (c) local rmsnorm at C/TP | (b) vs (a) | "
        "(b) dgamma ttnn::sum |\n",
        cfg.batch_size,
        cfg.sequence_length,
        cfg.hidden_dim);
    fmt::print(
        "|----|------------------------------|-------------------------|---------------------------|------------|------"
        "---------------|\n");
    for (const auto& row : rows) {
        const auto& r = row.result;
        fmt::print(
            "| {:>2} | {:>28.2f} | {:>23.2f} | {:>25.2f} | {:>8.2f}x | {:>10.2f} ({:>4.1f}%) |\n",
            row.tp,
            r.gather_ms,
            r.distributed_ms,
            r.local_ms,
            ttml::benchmark_utils::speedup_x(r.gather_ms, r.distributed_ms),
            r.dgamma_sum_ms,
            r.distributed_ms == 0.0 ? 0.0 : r.dgamma_sum_ms / r.distributed_ms * 100.0);
    }
}

SweepConfig g_cfg;
std::vector<uint32_t> g_tps;
std::vector<RowSummary> g_rows;

void BM_RMSNormDistributed(benchmark::State& state) {
    const uint32_t tp = g_tps.at(static_cast<size_t>(state.range(0)));
    for ([[maybe_unused]] auto _ : state) {
        const auto result = run_tp(g_cfg, tp);
        g_rows.push_back(RowSummary{.tp = tp, .result = result});

        state.SetIterationTime(result.distributed_ms / 1000.0);
        state.SetLabel(
            fmt::format("TP={} B={} S={} C={}", tp, g_cfg.batch_size, g_cfg.sequence_length, g_cfg.hidden_dim));
        state.counters["gather_ms"] = result.gather_ms;
        state.counters["distributed_ms"] = result.distributed_ms;
        state.counters["local_ms"] = result.local_ms;
        state.counters["dgamma_sum_ms"] = result.dgamma_sum_ms;
    }
}

}  // namespace

int main(int argc, char** argv) {
    try {
        benchmark::Initialize(&argc, argv);
        if (benchmark::ReportUnrecognizedArguments(argc, argv)) {
            return 1;
        }

        if (const char* seq_env = std::getenv("RMSNORM_BENCH_S"); seq_env != nullptr) {
            g_cfg.sequence_length = static_cast<uint32_t>(std::stoul(seq_env));
        }
        if (const char* hidden_env = std::getenv("RMSNORM_BENCH_C"); hidden_env != nullptr) {
            g_cfg.hidden_dim = static_cast<uint32_t>(std::stoul(hidden_env));
        }
        if (const char* tp_env = std::getenv("RMSNORM_BENCH_TP"); tp_env != nullptr) {
            g_cfg.tensor_parallel_sizes.clear();
            std::stringstream tp_list(tp_env);
            for (std::string item; std::getline(tp_list, item, ',');) {
                g_cfg.tensor_parallel_sizes.push_back(static_cast<uint32_t>(std::stoul(item)));
            }
        }
        const size_t num_devices = tt::tt_metal::GetNumAvailableDevices();
        for (const auto tp : g_cfg.tensor_parallel_sizes) {
            if (tp <= num_devices) {
                g_tps.push_back(tp);
            }
        }
        if (g_tps.empty()) {
            throw std::invalid_argument(
                fmt::format("Distributed RMSNorm benchmark needs at least 2 devices, found {}", num_devices));
        }

        fmt::print("Distributed RMSNorm op-level benchmark (gather vs distributed vs local)\n");
        fmt::print(
            "B={} S={} C={} TP={} warmup={} measure={}\n",
            g_cfg.batch_size,
            g_cfg.sequence_length,
            g_cfg.hidden_dim,
            fmt::join(g_tps, ","),
            g_cfg.num_warmup,
            g_cfg.num_measure);
        fmt::print(
            "Runs op-level timings on BF16 tensors with synthetic upstream gradient; no loss/optimizer.\n"
            "Each measured sample enqueues multiple FW+BW steps and synchronizes once at the end.\n");

        benchmark::RegisterBenchmark("RMSNormDistributed", BM_RMSNormDistributed)
            ->DenseRange(0, static_cast<int>(g_tps.size()) - 1, 1)
            ->Unit(benchmark::kMillisecond)
            ->UseManualTime()
            ->Iterations(1);
        benchmark::RunSpecifiedBenchmarks();
        benchmark::Shutdown();

        print_table(g_cfg, g_rows);
        return 0;
    } catch (const std::exception& e) {
        fmt::print(stderr, "rmsnorm_distributed_benchmark failed: {}\n", e.what());
        return 1;
    }
}
