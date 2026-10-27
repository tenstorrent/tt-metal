// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// ============================================================================
// RMSNorm op-level benchmark
//
// Single device, for a sweep of [B,1,S,C] shapes:
//   fw  — ttml::metal::rmsnorm_fw returning rms
//   bw  — ttml::metal::rmsnorm_bw (device op + dgamma reduction), given that rms
// plus the layout rmsnorm_bw picks (cores, whole row in L1 or streamed, dgamma on chip or as components) and the
// DRAM peak of one backward call.
//
// Hidden dim sharded across a 1 x TP line, for each TP in RMSNORM_BENCH_TP the machine has devices for:
//   (a) gather      — all_gather(x, REPLICATED) -> ops::rmsnorm on the full row with a replicated full-width
//                     gamma -> scatter the output. The backward all-gathers dy, and the REPLICATED gather keeps
//                     the local slice of dx.
//   (b) distributed — ops::distributed::rmsnorm.
//   (c) local       — ops::rmsnorm on the C/TP shard with no communication; a lower bound, not a valid norm.
// Each arm reports the forward (recording the backward node), the forward + synthetic-grad backward step, their
// difference as the backward, and the DRAM peak of one step. A mesh buffer is tracked once at its per-device
// size, so the peak is per device.
// Every op of (b) is also timed on its own, on the local shard.
//
// Times are host µs per call over RMSNORM_BENCH_MEASURE calls between two device syncs, after
// RMSNORM_BENCH_WARMUP calls.
//
// Environment:
//   RMSNORM_BENCH_TP       comma separated TP sizes (default "2,4"); sizes above the system mesh are skipped
//   RMSNORM_BENCH_B/S/C    shape of the sharded runs (default 1 x 8192 x 5120)
//   RMSNORM_BENCH_SINGLE   0 skips the single-device sweep
// Each mesh shape needs a matching mesh graph descriptor, so run one TP per process, e.g. on a Blackhole QuietBox 2:
//   TT_MESH_GRAPH_DESC_PATH=$TT_METAL_HOME/tt-train/configs/mgd/bh_1_2_line_line.textproto RMSNORM_BENCH_TP=2 \
//       rmsnorm_benchmark
//   MGD=$TT_METAL_HOME/tt_metal/fabric/mesh_graph_descriptors
//   TT_MESH_GRAPH_DESC_PATH=$MGD/p300_x2_mesh_graph_descriptor.textproto RMSNORM_BENCH_TP=4 RMSNORM_BENCH_SINGLE=0 \
//       rmsnorm_benchmark
// The ring descriptor tt-train/configs/mgd/bh_qb_1_4_ring_ring.textproto needs a chip 0 <-> chip 3 link that not
// every QuietBox 2 has; setting up its fabric then fails with "Expected 4 eth links from physical chip 0 to 3".
// ============================================================================

#include <fmt/format.h>
#include <fmt/ranges.h>

#include <cstdint>
#include <cstdlib>
#include <enchantum/enchantum.hpp>
#include <functional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/experimental/fabric/fabric.hpp>
#include <tt-metalium/system_mesh.hpp>
#include <vector>

#include "autograd/auto_context.hpp"
#include "autograd/tensor.hpp"
#include "benchmark_utils.hpp"
#include "core/compute_kernel_config.hpp"
#include "core/tt_tensor_utils.hpp"
#include "metal/ops/rmsnorm_bw/device/rmsnorm_bw_program_factory.hpp"
#include "metal/ops/rmsnorm_bw/rmsnorm_bw.hpp"
#include "metal/ops/rmsnorm_fw/rmsnorm_fw.hpp"
#include "ops/distributed/comm_ops.hpp"
#include "ops/distributed/rmsnorm_op.hpp"
#include "ops/rmsnorm_op.hpp"
#include "test_utils/random_data.hpp"
#include "ttnn/distributed/distributed_tensor.hpp"
#include "ttnn/operations/normalization/rmsnorm_distributed/rmsnorm_post_all_gather.hpp"
#include "ttnn/operations/normalization/rmsnorm_distributed/rmsnorm_pre_all_gather.hpp"
#include "ttnn_fixed/distributed/tt_metal.hpp"
#include "ttnn_fixed/distributed/ttnn_ops.hpp"
#include "ttnn_fixed/trivial_ttnn_ops.hpp"

namespace {

using ttml::benchmark_utils::capture_dram_peak;
using ttml::benchmark_utils::time_device_avg_us;

constexpr float kEpsilon = 1e-6F;
constexpr double kBytesPerMb = 1024.0 * 1024.0;

struct Shape {
    uint32_t b;
    uint32_t s;
    uint32_t c;
};

struct Config {
    uint32_t num_warmup = 3;
    uint32_t num_measure = 20;
    bool single_device = true;
    std::vector<uint32_t> tensor_parallel_sizes = {2, 4};
    Shape sharded_shape{1, 8192, 5120};
};

uint32_t env_or(const char* name, uint32_t fallback) {
    const char* value = std::getenv(name);
    return value == nullptr ? fallback : static_cast<uint32_t>(std::stoul(value));
}

std::vector<float> uniform(size_t count, float lo, float hi, uint32_t seed) {
    return ttml::test_utils::make_uniform_vector<float>(count, lo, hi, seed);
}

ttnn::Tensor to_device(
    const std::vector<float>& values,
    const ttnn::Shape& shape,
    const ttnn::distributed::TensorToMesh* mapper = nullptr) {
    auto* device = &ttml::autograd::ctx().get_device();
    return ttml::core::from_vector<float, ttnn::DataType::BFLOAT16>(values, shape, device, ttnn::Layout::TILE, mapper);
}

// ---------------------------------------------------------------------------
// Single device
// ---------------------------------------------------------------------------

std::string layout_name(const ttml::metal::ops::rmsnorm_bw::device::RMSNormBackwardLayout& layout) {
    return fmt::format(
        "{} cores, {}, {}",
        layout.num_cores,
        layout.everything_fits_in_l1 ? "fits" : "stream",
        layout.dgamma_on_chip ? "on-chip" : "components");
}

void run_single_device(const Config& cfg) {
    const std::vector<Shape> shapes = {
        {64, 256, 384},
        {8, 1024, 768},
        {4, 2048, 2048},
        {16, 2048, 2048},
        {4, 2048, 4096},
        {16, 2048, 4096},
        {1, 8192, 5120},
        {4, 8192, 5120},
        {4, 2048, 8192},
        {16, 2048, 8192},
        {1, 1024, 16384},
        {4, 2000, 2048},
        {4, 2048, 2000},
    };

    ttml::autograd::ctx().open_device();
    auto& device = ttml::autograd::ctx().get_device();

    fmt::print("\nSingle-device RMSNorm (µs per call, DRAM MB)\n");
    fmt::print("| B | S | C | bw layout | fw µs | bw µs | bw DRAM peak MB |\n");
    fmt::print("|---|---|---|---|---|---|---|\n");
    for (const auto& shape : shapes) {
        device.clear_program_cache();
        const ttnn::Shape x_shape({shape.b, 1U, shape.s, shape.c});
        const size_t count = static_cast<size_t>(shape.b) * shape.s * shape.c;
        const auto x = to_device(uniform(count, -1.0F, 1.0F, 1U), x_shape);
        const auto dy = to_device(uniform(count, -1.0F, 1.0F, 2U), x_shape);
        const auto gamma = to_device(uniform(shape.c, 0.5F, 1.5F, 3U), ttnn::Shape({1U, 1U, 1U, shape.c}));
        const auto rms = ttml::metal::rmsnorm_fw(x, gamma, true, kEpsilon)[1].value();

        using namespace ttml::metal::ops::rmsnorm_bw::device;
        const auto layout = compute_layout(
            operation_attributes_t{}, tensor_args_t{.input = x, .gamma = gamma, .rms = rms, .dL_dout = dy});

        const double fw_us = time_device_avg_us(device, cfg.num_warmup, cfg.num_measure, [&]() {
            (void)ttml::metal::rmsnorm_fw(x, gamma, true, kEpsilon);
        });
        const double bw_us = time_device_avg_us(
            device, cfg.num_warmup, cfg.num_measure, [&]() { (void)ttml::metal::rmsnorm_bw(x, gamma, rms, dy); });
        const size_t bw_dram =
            capture_dram_peak("RMSNORM_BW", [&]() { (void)ttml::metal::rmsnorm_bw(x, gamma, rms, dy); });

        fmt::print(
            "| {} | {} | {} | {} | {:.1f} | {:.1f} | {:.1f} |\n",
            shape.b,
            shape.s,
            shape.c,
            layout_name(layout),
            fw_us,
            bw_us,
            static_cast<double>(bw_dram) / kBytesPerMb);
        std::fflush(stdout);
    }

    ttml::autograd::ctx().close_device();
}

// ---------------------------------------------------------------------------
// Hidden dim sharded across 1 x TP
// ---------------------------------------------------------------------------

struct ShardedInputs {
    ttnn::Tensor x;           // [B,1,S,C/TP] per device
    ttnn::Tensor gamma;       // [1,1,1,C/TP] per device
    ttnn::Tensor gamma_full;  // [1,1,1,C] replicated
    ttnn::Tensor dy;          // [B,1,S,C/TP] per device
};

ShardedInputs make_sharded_inputs(const Shape& shape) {
    auto& device = ttml::autograd::ctx().get_device();
    const auto mapper = ttnn::distributed::shard_tensor_to_mesh_mapper(device, 3);
    const ttnn::Shape x_shape({shape.b, 1U, shape.s, shape.c});
    const ttnn::Shape gamma_shape({1U, 1U, 1U, shape.c});
    const size_t count = static_cast<size_t>(shape.b) * shape.s * shape.c;
    const auto gamma_host = uniform(shape.c, 0.5F, 1.5F, 3U);
    return ShardedInputs{
        .x = to_device(uniform(count, -1.0F, 1.0F, 1U), x_shape, mapper.get()),
        .gamma = to_device(gamma_host, gamma_shape, mapper.get()),
        .gamma_full = to_device(gamma_host, gamma_shape),
        .dy = to_device(uniform(count, -1.0F, 1.0F, 2U), x_shape, mapper.get()),
    };
}

enum class Arm { Gather, Distributed, Local };

std::string arm_name(Arm arm) {
    switch (arm) {
        case Arm::Gather: return "(a) gather -> rmsnorm -> scatter";
        case Arm::Distributed: return "(b) distributed rmsnorm";
        case Arm::Local: return "(c) local rmsnorm at C/TP";
    }
    return "unknown";
}

ttml::autograd::TensorPtr forward(Arm arm, const ShardedInputs& in) {
    auto x = ttml::autograd::create_tensor(in.x, /* requires_grad */ true);
    switch (arm) {
        case Arm::Gather: {
            auto gamma = ttml::autograd::create_tensor(in.gamma_full, /* requires_grad */ true);
            auto x_full = ttml::ops::distributed::all_gather(
                x, /* dim */ 3, /* cluster_axis */ std::nullopt, ttml::ops::distributed::GradOutputType::REPLICATED);
            return ttml::ops::distributed::scatter(ttml::ops::rmsnorm(x_full, gamma, kEpsilon), /* dim */ 3);
        }
        case Arm::Distributed: {
            auto gamma = ttml::autograd::create_tensor(in.gamma, /* requires_grad */ true);
            return ttml::ops::distributed::rmsnorm(x, gamma, kEpsilon);
        }
        case Arm::Local: {
            auto gamma = ttml::autograd::create_tensor(in.gamma, /* requires_grad */ true);
            return ttml::ops::rmsnorm(x, gamma, kEpsilon);
        }
    }
    throw std::logic_error("unknown arm");
}

struct ArmResult {
    double fw_us = 0.0;
    double step_us = 0.0;
    double dram_peak_mb = 0.0;
};

ArmResult run_arm(const Config& cfg, Arm arm, const ShardedInputs& in) {
    auto& device = ttml::autograd::ctx().get_device();
    const auto fw = [&]() {
        (void)forward(arm, in);
        ttml::autograd::ctx().reset_graph();
    };
    const auto step = [&]() {
        auto out = forward(arm, in);
        out->set_grad(in.dy);
        out->backward();
        ttml::autograd::ctx().reset_graph();
    };
    return ArmResult{
        .fw_us = time_device_avg_us(device, cfg.num_warmup, cfg.num_measure, fw),
        .step_us = time_device_avg_us(device, cfg.num_warmup, cfg.num_measure, step),
        .dram_peak_mb = static_cast<double>(capture_dram_peak("RMSNORM_STEP", step)) / kBytesPerMb,
    };
}

struct OpRow {
    std::string name;
    double us = 0.0;
};

std::vector<OpRow> run_distributed_ops(const Config& cfg, const ShardedInputs& in, uint32_t full_width) {
    auto& device = ttml::autograd::ctx().get_device();
    const auto precise = ttml::core::ComputeKernelConfig::precise();

    const auto local_stats = ttnn::rms_norm_pre_all_gather(in.x, ttnn::DataType::FLOAT32, std::nullopt, precise);
    const auto stats = ttml::ttnn_fixed::distributed::all_gather(local_stats, /* dim */ 3);
    const auto rms = ttml::ops::distributed::rms_from_gathered_stats(stats, full_width, kEpsilon);
    const auto local_sums = ttml::metal::rmsnorm_bw_local_sum(in.x, in.gamma, in.dy);
    const auto gathered_sums = ttml::ttnn_fixed::distributed::all_gather(local_sums, /* dim */ 3);

    using namespace ttml::metal::ops::rmsnorm_bw::device;
    const auto layout = compute_layout(
        operation_attributes_t{.mode = RMSNormBackwardMode::ExternalScale, .full_width = full_width},
        tensor_args_t{.input = in.x, .gamma = in.gamma, .rms = rms, .dL_dout = in.dy, .external_sums = gathered_sums});
    const auto& x_shape = in.x.logical_shape();
    const auto dgamma_partials =
        layout.dgamma_on_chip
            ? ttml::core::zeros(ttnn::Shape({1U, 1U, layout.num_cores, x_shape[-1]}), &device, ttnn::DataType::FLOAT32)
            : ttml::core::zeros(x_shape, &device, ttnn::DataType::BFLOAT16);

    const std::vector<std::pair<std::string, std::function<void()>>> ops = {
        {"fw: rms_norm_pre_all_gather",
         [&]() { (void)ttnn::rms_norm_pre_all_gather(in.x, ttnn::DataType::FLOAT32, std::nullopt, precise); }},
        {"fw: stats all_gather", [&]() { (void)ttml::ttnn_fixed::distributed::all_gather(local_stats, 3); }},
        {"fw: rms_from_gathered_stats",
         [&]() { (void)ttml::ops::distributed::rms_from_gathered_stats(stats, full_width, kEpsilon); }},
        {"fw: rms_norm_post_all_gather",
         [&]() {
             (void)ttnn::rms_norm_post_all_gather(
                 in.x, stats, kEpsilon, in.gamma, std::nullopt, std::nullopt, precise, std::nullopt, in.x.dtype());
         }},
        {"bw: rmsnorm_bw LocalSum", [&]() { (void)ttml::metal::rmsnorm_bw_local_sum(in.x, in.gamma, in.dy); }},
        {"bw: sums all_gather", [&]() { (void)ttml::ttnn_fixed::distributed::all_gather(local_sums, 3); }},
        {"bw: rmsnorm_bw ExternalScale (incl. dgamma reduce)",
         [&]() {
             (void)ttml::metal::rmsnorm_bw_external_scale(in.x, in.gamma, rms, in.dy, gathered_sums, full_width);
         }},
        {fmt::format("bw: dgamma sum_over_rows ({})", layout_name(layout)),
         [&]() { (void)ttml::ttnn_fixed::sum_over_rows(dgamma_partials); }},
    };

    std::vector<OpRow> rows;
    rows.reserve(ops.size());
    for (const auto& [name, op] : ops) {
        rows.push_back(OpRow{.name = name, .us = time_device_avg_us(device, cfg.num_warmup, cfg.num_measure, op)});
    }
    return rows;
}

void run_tensor_parallel(const Config& cfg, uint32_t tp) {
    const Shape& shape = cfg.sharded_shape;
    if (shape.c % tp != 0U) {
        throw std::invalid_argument(fmt::format("C={} is not divisible by TP={}", shape.c, tp));
    }

    const auto mgd_path = ttml::ttnn_fixed::distributed::get_mgd_path(tp);
    ttml::ttnn_fixed::distributed::enable_fabric(tp);
    ttml::autograd::ctx().open_device(tt::tt_metal::distributed::MeshShape(1, tp));
    ttml::autograd::ctx().set_seed(42);
    auto& device = ttml::autograd::ctx().get_device();

    fmt::print(
        "\nRMSNorm, hidden dim sharded across 1 x {}: B={} S={} C={} (C/TP={})\n",
        tp,
        shape.b,
        shape.s,
        shape.c,
        shape.c / tp);
    fmt::print(
        "mesh graph descriptor: {}\nfabric config: {}, topology: {}\n",
        mgd_path.value_or("none (default FABRIC_2D)"),
        enchantum::to_string(tt::tt_fabric::GetFabricConfig()),
        enchantum::to_string(tt::tt_fabric::get_fabric_topology()));

    const auto in = make_sharded_inputs(shape);
    tt::tt_metal::distributed::Synchronize(device, std::nullopt);

    fmt::print("| arm | fw µs | bw µs (step - fw) | fw+bw step µs | step DRAM peak MB/device |\n");
    fmt::print("|---|---|---|---|---|\n");
    double gather_step_us = 0.0;
    for (const Arm arm : {Arm::Gather, Arm::Distributed, Arm::Local}) {
        device.clear_program_cache();
        const ArmResult r = run_arm(cfg, arm, in);
        if (arm == Arm::Gather) {
            gather_step_us = r.step_us;
        }
        const std::string vs_gather =
            arm == Arm::Gather
                ? ""
                : fmt::format(" ({:.2f}x vs (a))", ttml::benchmark_utils::speedup_x(gather_step_us, r.step_us));
        fmt::print(
            "| {} | {:.1f} | {:.1f} | {:.1f}{} | {:.1f} |\n",
            arm_name(arm),
            r.fw_us,
            r.step_us - r.fw_us,
            r.step_us,
            vs_gather,
            r.dram_peak_mb);
        std::fflush(stdout);
    }

    fmt::print("\n(b) per op on the local shard\n| op | µs |\n|---|---|\n");
    for (const auto& row : run_distributed_ops(cfg, in, shape.c)) {
        fmt::print("| {} | {:.1f} |\n", row.name, row.us);
    }
    std::fflush(stdout);

    ttml::autograd::ctx().close_device();
}

}  // namespace

int main() {
    try {
        Config cfg;
        cfg.num_warmup = env_or("RMSNORM_BENCH_WARMUP", cfg.num_warmup);
        cfg.num_measure = env_or("RMSNORM_BENCH_MEASURE", cfg.num_measure);
        cfg.single_device = env_or("RMSNORM_BENCH_SINGLE", 1U) != 0U;
        cfg.sharded_shape = Shape{
            env_or("RMSNORM_BENCH_B", cfg.sharded_shape.b),
            env_or("RMSNORM_BENCH_S", cfg.sharded_shape.s),
            env_or("RMSNORM_BENCH_C", cfg.sharded_shape.c)};
        if (const char* tp_env = std::getenv("RMSNORM_BENCH_TP"); tp_env != nullptr) {
            cfg.tensor_parallel_sizes.clear();
            std::stringstream tp_list(tp_env);
            for (std::string item; std::getline(tp_list, item, ',');) {
                cfg.tensor_parallel_sizes.push_back(static_cast<uint32_t>(std::stoul(item)));
            }
        }

        fmt::print("RMSNorm benchmark, warmup={} measure={}\n", cfg.num_warmup, cfg.num_measure);
        if (cfg.single_device) {
            run_single_device(cfg);
        }

        // Bounded by the mesh graph descriptor, not by the chip count.
        const size_t num_devices = tt::tt_metal::distributed::SystemMesh::instance().shape().mesh_size();
        for (const uint32_t tp : cfg.tensor_parallel_sizes) {
            if (tp > num_devices) {
                fmt::print("\nSkipping TP={}: the system mesh has {} devices\n", tp, num_devices);
                continue;
            }
            run_tensor_parallel(cfg, tp);
        }
        return 0;
    } catch (const std::exception& e) {
        fmt::print(stderr, "rmsnorm_benchmark failed: {}\n", e.what());
        return 1;
    }
}
