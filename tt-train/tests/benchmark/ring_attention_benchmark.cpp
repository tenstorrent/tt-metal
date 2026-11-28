// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <fmt/format.h>
#include <fmt/ranges.h>

#include <CLI/CLI.hpp>
#include <chrono>
#include <core/xtensor_utils.hpp>
#include <string>
#include <tools/profiler/op_profiler.hpp>
#include <tt-metalium/discovery/system_mesh.hpp>
#include <tt-metalium/execution/distributed.hpp>
#include <tt-metalium/observability/profiler/device_profiler.hpp>
#include <vector>

#include "../ops/distributed/ring_sdpa_test_utils.hpp"
#include "autograd/auto_context.hpp"
#include "benchmark_utils.hpp"
#include "core/tt_tensor_utils.hpp"
#include "ops/distributed/ring_attention_sdpa.hpp"
#include "test_utils/random_data.hpp"
#include "test_utils/relative_l2.hpp"
#include "ttnn/distributed/create_socket.hpp"
#include "ttnn/distributed/distributed_tensor.hpp"
#include "ttnn_fixed/distributed/tt_metal.hpp"

namespace {

using ttml::benchmark_utils::median;
using ttml::core::distributed::CpSharding;
using ttml::test_utils::relative_l2;
using ttml::test_utils::ring_sdpa::gather_cp_sequence;
using ttml::test_utils::ring_sdpa::permute_cp_order;

struct Config {
    uint32_t batch = 1U;
    std::vector<uint32_t> seqs{32768U};
    uint32_t heads = 8U;
    uint32_t kv_heads = 2U;
    uint32_t head_dim = 128U;
    uint32_t iters = 5U;
    bool check = false;
    bool reuse_forward_kv = false;
    bool profile_read = false;
    std::vector<std::string> shardings{"contiguous", "zigzag"};
};

struct RunResult {
    std::vector<double> fw_ms;
    std::vector<double> bw_ms;
    xt::xarray<float> out, dq, dk, dv;
};

RunResult run_sharding(
    CpSharding sharding,
    const std::string& name,
    uint32_t seq,
    const Config& cfg,
    const xt::xarray<float>& q_host,
    const xt::xarray<float>& k_host,
    const xt::xarray<float>& v_host,
    const xt::xarray<float>& do_host) {
    using namespace ttml;
    auto* device = &autograd::ctx().get_device();
    const auto& pctx = autograd::ctx().get_parallelism_context();
    const uint32_t cp_axis = pctx.get_cp_axis().value();
    const uint32_t ring_size = pctx.get_cp_size();
    const auto to_device_order = [&](const xt::xarray<float>& x) {
        return permute_cp_order(x, sharding, ring_size, /*to_device=*/true);
    };

    const auto mapper = ttnn::distributed::shard_tensor_to_mesh_mapper(*device, /*dim=*/2, cp_axis);
    const auto upload = [&](const xt::xarray<float>& x) {
        return core::from_xtensor<float, ttnn::DataType::BFLOAT16>(
            to_device_order(x), device, ttnn::Layout::TILE, mapper.get());
    };
    const ttnn::Tensor q_tt = upload(q_host);
    const ttnn::Tensor k_tt = upload(k_host);
    const ttnn::Tensor v_tt = upload(v_host);
    const ttnn::Tensor do_tt = upload(do_host);

    RunResult result;
    for (uint32_t iter = 0; iter <= cfg.iters; ++iter) {
        auto q = autograd::create_tensor(q_tt, /*requires_grad=*/true);
        auto k = autograd::create_tensor(k_tt, /*requires_grad=*/true);
        auto v = autograd::create_tensor(v_tt, /*requires_grad=*/true);

        tt::tt_metal::distributed::Synchronize(*device, std::nullopt);
        tt::tt_metal::op_profiler::tracy_message(fmt::format("`TT_SIGNPOST: {}_s{}_fw_it{}`", name, seq, iter));
        const auto t0 = std::chrono::steady_clock::now();
        auto out = ops::distributed::ring_attention_sdpa(
            q,
            k,
            v,
            /*mask=*/std::nullopt,
            ttml::metal::AttentionMaskType::Causal,
            cfg.reuse_forward_kv,
            sharding);
        tt::tt_metal::distributed::Synchronize(*device, std::nullopt);
        const auto t1 = std::chrono::steady_clock::now();
        tt::tt_metal::op_profiler::tracy_message(fmt::format("`TT_SIGNPOST: {}_s{}_bw_it{}`", name, seq, iter));
        out->set_grad(do_tt);
        out->backward();
        tt::tt_metal::distributed::Synchronize(*device, std::nullopt);
        const auto t2 = std::chrono::steady_clock::now();

        if (iter > 0U) {
            result.fw_ms.push_back(std::chrono::duration<double, std::milli>(t1 - t0).count());
            result.bw_ms.push_back(std::chrono::duration<double, std::milli>(t2 - t1).count());
        }
        if (cfg.check && iter == 0U) {
            result.out = gather_cp_sequence(out->get_value(), cp_axis, sharding);
            result.dq = gather_cp_sequence(q->get_grad(), cp_axis, sharding);
            result.dk = gather_cp_sequence(k->get_grad(), cp_axis, sharding);
            result.dv = gather_cp_sequence(v->get_grad(), cp_axis, sharding);
        }
        autograd::ctx().reset_graph();
    }
    if (cfg.profile_read) {
        tt::tt_metal::ReadMeshDeviceProfilerResults(*device);
    }
    return result;
}

void run_sequence(const Config& cfg, uint32_t seq) {
    const auto mesh_shape = ttml::autograd::ctx().get_device().shape();
    const auto& pctx = ttml::autograd::ctx().get_parallelism_context();
    const uint32_t ring_size = pctx.get_cp_size();
    TT_FATAL(seq % (2U * ring_size * 32U) == 0U, "--seq {} must be a multiple of 2 * ring_size * 32", seq);
    fmt::print(
        "mesh {} cp_axis {} ring {} | B {} S {} (S_local {}) Hq {} Hkv {} D {} | iters {} reuse_forward_kv {}\n",
        mesh_shape,
        pctx.get_cp_axis().value(),
        ring_size,
        cfg.batch,
        seq,
        seq / ring_size,
        cfg.heads,
        cfg.kv_heads,
        cfg.head_dim,
        cfg.iters,
        cfg.reuse_forward_kv);

    const std::array<size_t, 4> q_shape{cfg.batch, cfg.heads, seq, cfg.head_dim};
    const std::array<size_t, 4> kv_shape{cfg.batch, cfg.kv_heads, seq, cfg.head_dim};
    const auto q_host = ttml::test_utils::make_uniform_xarray<float>(q_shape, -1.0F, 1.0F, 1U);
    const auto k_host = ttml::test_utils::make_uniform_xarray<float>(kv_shape, -1.0F, 1.0F, 2U);
    const auto v_host = ttml::test_utils::make_uniform_xarray<float>(kv_shape, -1.0F, 1.0F, 3U);
    const auto do_host = ttml::test_utils::make_uniform_xarray<float>(q_shape, -1.0F, 1.0F, 4U);

    std::vector<std::pair<std::string, RunResult>> results;
    for (const std::string& name : cfg.shardings) {
        const CpSharding sharding = ttml::core::distributed::read_cp_sharding(name);
        auto result = run_sharding(sharding, name, seq, cfg, q_host, k_host, v_host, do_host);
        fmt::print(
            "[RING_ATTN_BENCH] cp_sharding={} S={} S_local={} fw_ms={:.2f} bw_ms={:.2f} fw_all={{{:.2f}}} "
            "bw_all={{{:.2f}}}\n",
            name,
            seq,
            seq / ring_size,
            median(result.fw_ms),
            median(result.bw_ms),
            fmt::join(result.fw_ms, ","),
            fmt::join(result.bw_ms, ","));
        results.emplace_back(name, std::move(result));
    }

    if (cfg.check && results.size() == 2U) {
        const auto& base = results[0].second;
        const auto& other = results[1].second;
        fmt::print(
            "[RING_ATTN_BENCH] check relative_l2({}, {}): out {:.2e} dQ {:.2e} dK {:.2e} dV {:.2e}\n",
            results[1].first,
            results[0].first,
            relative_l2(other.out, base.out),
            relative_l2(other.dq, base.dq),
            relative_l2(other.dk, base.dk),
            relative_l2(other.dv, base.dv));
    }
}

}  // namespace

int main(int argc, char** argv) {
    Config cfg;
    CLI::App app{"Ring attention benchmark: one attention layer fw+bw per CP sharding"};
    app.add_option("--batch", cfg.batch, "Batch size");
    app.add_option("--seq", cfg.seqs, "Global sequence length(s), comma-separated to sweep")->delimiter(',');
    app.add_option("--heads", cfg.heads, "Query heads per device");
    app.add_option("--kv-heads", cfg.kv_heads, "Key/value heads per device");
    app.add_option("--head-dim", cfg.head_dim, "Head dimension");
    app.add_option("--iters", cfg.iters, "Timed iterations per CP sharding (after one untimed warm-up)");
    app.add_option("--cp-sharding", cfg.shardings, "CP shardings to run: contiguous, zigzag")->delimiter(',');
    app.add_flag("--check", cfg.check, "Compare zig-zag against contiguous (outputs and gradients)");
    app.add_flag(
        "--reuse-forward-kv", cfg.reuse_forward_kv, "Backward reads the forward's K/V instead of shifting them again");
    app.add_flag("--profile-read", cfg.profile_read, "Read device profiler results after every (S, sharding)");
    CLI11_PARSE(app, argc, argv);

    auto& ctx = ttml::autograd::ctx();
    ctx.initialize_distributed_context(argc, argv);
    const auto mesh_shape = tt::tt_metal::distributed::SystemMesh::instance().shape();
    ttml::ttnn_fixed::distributed::enable_fabric(static_cast<uint32_t>(mesh_shape.mesh_size()));
    ctx.open_device(mesh_shape);
    ctx.initialize_socket_manager(ttnn::distributed::SocketType::FABRIC);
    ttml::test_utils::ring_sdpa::initialize_cp_parallelism_context();

    for (const uint32_t seq : cfg.seqs) {
        run_sequence(cfg, seq);
    }
    ctx.close_device();
    return 0;
}
