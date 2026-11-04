// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Ring attention (context parallelism) benchmark: contiguous vs zig-zag sequence layout.
//
// Runs ttml::ops::distributed::ring_attention_sdpa forward + backward for one attention layer on the
// whole system mesh, with CP along the longest mesh axis (LoudBox 1x8 -> CP8; Galaxy 8x4 -> CP8 on
// axis 0, the other axis only replicates). Both layouts see the same global Q/K/V/dO; the zig-zag
// run permutes the sequence on the host so rank d holds chunks [d | 2N-1-d]. Reports the mean
// wall time of forward and backward (device-synchronized) over the timed iterations, and with
// --check compares the zig-zag outputs and gradients (un-permuted) against the contiguous ones.
//
// Example (BH Galaxy, 8B shape after TP4: 8 q heads, 2 kv heads, head dim 128):
//   ring_attention_benchmark --seq 131072 --heads 8 --kv-heads 2 --head-dim 128 --iters 3
// Set TTML_RING_ATTN_TIMING=1 to also print the per-phase breakdown of every call.
// --seq takes a comma-separated list to sweep several lengths in one process. Every timed pass is
// preceded by a `TT_SIGNPOST: <layout>_s<S>_<fw|bw>_it<i>` tracy message, so a device-profiler run
// (python -m tracy -r -p ...) can be sliced per pass; --profile-read drains the device profiler
// buffers after every (seq, layout) so long sweeps do not overflow them.

#include <fmt/format.h>

#include <CLI/CLI.hpp>
#include <chrono>
#include <cmath>
#include <core/xtensor_utils.hpp>
#include <string>
#include <tools/profiler/op_profiler.hpp>
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/system_mesh.hpp>
#include <vector>

#include "autograd/auto_context.hpp"
#include "core/tt_tensor_utils.hpp"
#include "ops/distributed/ring_attention_sdpa.hpp"
#include "test_utils/random_data.hpp"
#include "ttnn/distributed/create_socket.hpp"
#include "ttnn/distributed/distributed_tensor.hpp"
#include "ttnn_fixed/distributed/tt_metal.hpp"

namespace {

using ttml::ops::distributed::RingAttentionLayout;

struct Config {
    uint32_t batch = 1;
    uint32_t seq = 131072;  // current length of the sweep
    std::vector<uint32_t> seqs{131072};
    uint32_t heads = 8;
    uint32_t kv_heads = 2;
    uint32_t head_dim = 128;
    uint32_t iters = 3;
    bool check = false;
    bool profile_read = false;
    std::string layouts = "contiguous,zigzag";
};

// Global chunk held at each position of the zig-zag order: rank d holds [d | 2N-1-d].
std::vector<uint32_t> zigzag_chunk_order(uint32_t ring_size) {
    std::vector<uint32_t> order;
    for (uint32_t d = 0; d < ring_size; ++d) {
        order.push_back(d);
        order.push_back(2U * ring_size - 1U - d);
    }
    return order;
}

// Reorders (B, H, S, D) along S in 2N chunks. forward: out chunk i = in chunk order[i];
// inverse: out chunk order[i] = in chunk i.
xt::xarray<float> permute_chunks(const xt::xarray<float>& x, const std::vector<uint32_t>& order, bool forward) {
    const size_t B = x.shape()[0], H = x.shape()[1], S = x.shape()[2], D = x.shape()[3];
    const size_t chunk = S / order.size();
    xt::xarray<float> out = xt::empty<float>(x.shape());
    const float* src = x.data();
    float* dst = out.data();
    for (size_t bh = 0; bh < B * H; ++bh) {
        for (size_t i = 0; i < order.size(); ++i) {
            const size_t from = forward ? order[i] : i;
            const size_t to = forward ? i : order[i];
            std::copy_n(src + (bh * S + from * chunk) * D, chunk * D, dst + (bh * S + to * chunk) * D);
        }
    }
    return out;
}

// Gathers a CP-sharded (B, H, S_local, D) tensor back to (B, H, S, D) from the ring at index 0 of
// every non-CP axis (the other axes only hold replicas).
xt::xarray<float> gather_ring(const ttnn::Tensor& tensor, uint32_t cp_axis) {
    const auto shards = ttml::core::to_xtensor<float>(tensor, ttml::core::IdentityComposer{});
    const auto mesh_shape = ttml::autograd::ctx().get_device().shape();
    const uint32_t ring_size = mesh_shape[cp_axis];
    uint32_t stride = 1;
    for (uint32_t i = mesh_shape.dims(); i-- > cp_axis + 1;) {
        stride *= mesh_shape[i];
    }
    // Copy shard r into rows [r * S_local, (r + 1) * S_local). (Assigning xt::concatenate(out, ...)
    // back into `out` would alias the lazy expression with its own destination.)
    const auto& first = shards[0];
    const size_t B = first.shape()[0], H = first.shape()[1], S_local = first.shape()[2], D = first.shape()[3];
    xt::xarray<float> out = xt::empty<float>({B, H, S_local * ring_size, D});
    for (uint32_t r = 0; r < ring_size; ++r) {
        const auto& shard = shards[r * stride];
        for (size_t bh = 0; bh < B * H; ++bh) {
            std::copy_n(
                shard.data() + bh * S_local * D,
                S_local * D,
                out.data() + (bh * S_local * ring_size + r * S_local) * D);
        }
    }
    return out;
}

double rel_l2(const xt::xarray<float>& actual, const xt::xarray<float>& expected) {
    double diff = 0.0;
    double norm = 0.0;
    for (size_t i = 0; i < expected.size(); ++i) {
        const double e = expected.flat(i);
        const double a = actual.flat(i);
        diff += (a - e) * (a - e);
        norm += e * e;
    }
    return std::sqrt(diff / std::max(norm, 1e-30));
}

void signpost(const std::string& name) {
    tt::tt_metal::op_profiler::tracy_message(std::string("`TT_SIGNPOST: ") + name + "`");
}

struct RunResult {
    std::vector<double> fw_ms;
    std::vector<double> bw_ms;
    xt::xarray<float> out, dq, dk, dv;  // global, natural order (only with --check)
};

RunResult run_layout(
    RingAttentionLayout layout,
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
    const auto order = zigzag_chunk_order(ring_size);
    const bool zigzag = layout == RingAttentionLayout::ZigZag;
    auto to_layout = [&](const xt::xarray<float>& x) { return zigzag ? permute_chunks(x, order, true) : x; };
    auto from_layout = [&](const xt::xarray<float>& x) { return zigzag ? permute_chunks(x, order, false) : x; };

    const auto mapper = ttnn::distributed::shard_tensor_to_mesh_mapper(*device, /*dim=*/2, cp_axis);
    auto upload = [&](const xt::xarray<float>& x) {
        return core::from_xtensor<float, ttnn::DataType::BFLOAT16>(
            to_layout(x), device, ttnn::Layout::TILE, mapper.get());
    };
    const ttnn::Tensor q_tt = upload(q_host);
    const ttnn::Tensor k_tt = upload(k_host);
    const ttnn::Tensor v_tt = upload(v_host);
    const ttnn::Tensor do_tt = upload(do_host);

    RunResult result;
    for (uint32_t iter = 0; iter < cfg.iters + 1U; ++iter) {  // iteration 0 compiles and is not timed
        auto q = autograd::create_tensor(q_tt, /*requires_grad=*/true);
        auto k = autograd::create_tensor(k_tt, /*requires_grad=*/true);
        auto v = autograd::create_tensor(v_tt, /*requires_grad=*/true);

        const std::string tag = fmt::format("{}_s{}_{{}}_it{}", zigzag ? "zigzag" : "contiguous", cfg.seq, iter);
        tt::tt_metal::distributed::Synchronize(*device, std::nullopt);
        signpost(fmt::format(fmt::runtime(tag), "fw"));
        const auto t0 = std::chrono::steady_clock::now();
        auto out = ops::distributed::ring_attention_sdpa(
            q, k, v, /*mask=*/std::nullopt, ttml::metal::AttentionMaskType::Causal, layout);
        tt::tt_metal::distributed::Synchronize(*device, std::nullopt);
        const auto t1 = std::chrono::steady_clock::now();
        signpost(fmt::format(fmt::runtime(tag), "bw"));
        out->set_grad(do_tt);
        out->backward();
        tt::tt_metal::distributed::Synchronize(*device, std::nullopt);
        const auto t2 = std::chrono::steady_clock::now();

        if (iter > 0) {
            result.fw_ms.push_back(std::chrono::duration<double, std::milli>(t1 - t0).count());
            result.bw_ms.push_back(std::chrono::duration<double, std::milli>(t2 - t1).count());
        }
        if (cfg.check && iter == 0) {
            result.out = from_layout(gather_ring(out->get_value(), cp_axis));
            result.dq = from_layout(gather_ring(q->get_grad(), cp_axis));
            result.dk = from_layout(gather_ring(k->get_grad(), cp_axis));
            result.dv = from_layout(gather_ring(v->get_grad(), cp_axis));
        }
        autograd::ctx().reset_graph();
    }
    if (cfg.profile_read) {
        tt::tt_metal::ReadMeshDeviceProfilerResults(*device);
    }
    return result;
}

double mean(const std::vector<double>& values) {
    double sum = 0.0;
    for (double v : values) {
        sum += v;
    }
    return values.empty() ? 0.0 : sum / static_cast<double>(values.size());
}

}  // namespace

int main(int argc, char** argv) {
    Config cfg;
    CLI::App app{"Ring attention benchmark: contiguous vs zig-zag CP layout"};
    app.add_option("--batch", cfg.batch, "Batch size");
    app.add_option("--seq", cfg.seqs, "Global sequence length(s), comma-separated to sweep")->delimiter(',');
    app.add_option("--heads", cfg.heads, "Query heads per device");
    app.add_option("--kv-heads", cfg.kv_heads, "Key/value heads per device");
    app.add_option("--head-dim", cfg.head_dim, "Head dimension");
    app.add_option("--iters", cfg.iters, "Timed iterations per layout (after one untimed warm-up)");
    app.add_option("--layouts", cfg.layouts, "Comma-separated layouts: contiguous,zigzag");
    app.add_flag("--check", cfg.check, "Compare zig-zag against contiguous (outputs and gradients)");
    app.add_flag("--profile-read", cfg.profile_read, "Read device profiler results after every (seq, layout)");
    CLI11_PARSE(app, argc, argv);

    auto& ctx = ttml::autograd::ctx();
    ctx.initialize_distributed_context(argc, argv);
    const auto mesh_shape = tt::tt_metal::distributed::SystemMesh::instance().shape();
    ttml::ttnn_fixed::distributed::enable_fabric(static_cast<uint32_t>(mesh_shape.mesh_size()));
    ctx.open_device(mesh_shape);
    ctx.initialize_socket_manager(ttnn::distributed::SocketType::FABRIC);
    uint32_t non_trivial_axes = 0;
    for (uint32_t i = 0; i < mesh_shape.dims(); ++i) {
        non_trivial_axes += mesh_shape[i] > 1U ? 1U : 0U;
    }
    // On a 2D mesh two parallelisms are required; CP takes axis 0 (DP -> CP -> TP order) and TP only
    // replicates here because every device already holds its per-device heads.
    ctx.initialize_parallelism_context({.enable_ddp = false, .enable_tp = non_trivial_axes > 1U, .enable_cp = true});
    const auto& pctx = ctx.get_parallelism_context();
    const uint32_t ring_size = pctx.get_cp_size();
    for (const uint32_t seq : cfg.seqs) {
        cfg.seq = seq;
        TT_FATAL(cfg.seq % (2U * ring_size * 32U) == 0U, "seq must be a multiple of 2 * ring_size * 32");

        fmt::print(
            "mesh {} cp_axis {} ring {} | B {} S {} (S_local {}) Hq {} Hkv {} D {} | iters {}\n",
            mesh_shape,
            pctx.get_cp_axis().value(),
            ring_size,
            cfg.batch,
            cfg.seq,
            cfg.seq / ring_size,
            cfg.heads,
            cfg.kv_heads,
            cfg.head_dim,
            cfg.iters);

        const std::array<size_t, 4> q_shape{cfg.batch, cfg.heads, cfg.seq, cfg.head_dim};
        const std::array<size_t, 4> kv_shape{cfg.batch, cfg.kv_heads, cfg.seq, cfg.head_dim};
        const auto q_host = ttml::test_utils::make_uniform_xarray<float>(q_shape, -1.0F, 1.0F, 1U);
        const auto k_host = ttml::test_utils::make_uniform_xarray<float>(kv_shape, -1.0F, 1.0F, 2U);
        const auto v_host = ttml::test_utils::make_uniform_xarray<float>(kv_shape, -1.0F, 1.0F, 3U);
        const auto do_host = ttml::test_utils::make_uniform_xarray<float>(q_shape, -1.0F, 1.0F, 4U);

        std::vector<std::pair<std::string, RunResult>> results;
        for (const std::string name : {"contiguous", "zigzag"}) {
            if (cfg.layouts.find(name) == std::string::npos) {
                continue;
            }
            const auto layout = name == "zigzag" ? RingAttentionLayout::ZigZag : RingAttentionLayout::Contiguous;
            auto result = run_layout(layout, cfg, q_host, k_host, v_host, do_host);
            fmt::print(
                "[RING_ATTN_BENCH] layout={} S={} S_local={} fw_ms={:.1f} bw_ms={:.1f} fw_all={} bw_all={}\n",
                name,
                cfg.seq,
                cfg.seq / ring_size,
                mean(result.fw_ms),
                mean(result.bw_ms),
                result.fw_ms,
                result.bw_ms);
            results.emplace_back(name, std::move(result));
        }

        if (results.size() == 2U) {
            const auto& base = results[0].second;
            const auto& zig = results[1].second;
            fmt::print(
                "[RING_ATTN_BENCH] speedup zigzag vs contiguous: fw {:.2f}x bw {:.2f}x fw+bw {:.2f}x\n",
                mean(base.fw_ms) / mean(zig.fw_ms),
                mean(base.bw_ms) / mean(zig.bw_ms),
                (mean(base.fw_ms) + mean(base.bw_ms)) / (mean(zig.fw_ms) + mean(zig.bw_ms)));
            if (cfg.check) {
                auto norm = [](const xt::xarray<float>& x) { return std::sqrt(static_cast<double>(xt::sum(x * x)())); };
                fmt::print(
                    "[RING_ATTN_BENCH] check norms contiguous/zigzag: out {:.3e}/{:.3e} dQ {:.3e}/{:.3e} dK "
                    "{:.3e}/{:.3e} "
                    "dV {:.3e}/{:.3e}\n",
                    norm(base.out),
                    norm(zig.out),
                    norm(base.dq),
                    norm(zig.dq),
                    norm(base.dk),
                    norm(zig.dk),
                    norm(base.dv),
                    norm(zig.dv));
                fmt::print(
                    "[RING_ATTN_BENCH] check rel_l2(zigzag, contiguous): out {:.2e} dQ {:.2e} dK {:.2e} dV {:.2e}\n",
                    rel_l2(zig.out, base.out),
                    rel_l2(zig.dq, base.dq),
                    rel_l2(zig.dk, base.dk),
                    rel_l2(zig.dv, base.dv));
            }
        }
    }
    ctx.close_device();
    return 0;
}
