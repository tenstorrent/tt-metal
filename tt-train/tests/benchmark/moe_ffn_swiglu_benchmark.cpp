// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <fmt/format.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <numeric>
#include <random>
#include <sstream>
#include <string>
#include <tt-metalium/distributed.hpp>
#include <vector>

#include "autograd/auto_context.hpp"
#include "autograd/tensor.hpp"
#include "core/tt_tensor_utils.hpp"
#include "ops/moe_ffn_swiglu_op.hpp"
#include "test_utils/random_data.hpp"

namespace {

// ---------------------------------------------------------------------------
// Case definition
// ---------------------------------------------------------------------------

struct Case {
    std::string name;
    uint32_t E;                    // E_local
    uint32_t H;                    // hidden_dim
    uint32_t I;                    // intermediate_dim
    std::vector<uint32_t> counts;  // active rows per expert (must be tile-aligned multiples of 32)
    // Extra tile-rows of trailing slack between offsets[-1] and T_cap, mirroring
    // moe_group's worst-case allocation slack. 0 = no trailing pad path triggered.
    uint32_t slack_tiles = 0U;
};

struct Stats {
    double avg_us = 0.0;
    double min_us = 0.0;
    double max_us = 0.0;
    double p50_us = 0.0;
};

struct CaseResult {
    Stats forward;
    Stats forward_backward;
};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

std::vector<uint32_t> compute_offsets(const std::vector<uint32_t>& counts) {
    constexpr uint32_t kTile = 32U;
    std::vector<uint32_t> offsets(counts.size() + 1U, 0U);
    for (std::size_t e = 0; e < counts.size(); ++e) {
        const uint32_t padded = ((counts[e] + kTile - 1U) / kTile) * kTile;
        offsets[e + 1U] = offsets[e] + padded;
    }
    return offsets;
}

std::vector<ttml::autograd::TensorPtr> build_expert_weight_list(
    uint32_t E, uint32_t K, uint32_t N, ttnn::distributed::MeshDevice* device, std::mt19937& rng) {
    std::vector<ttml::autograd::TensorPtr> out;
    out.reserve(E);
    const std::array<std::size_t, 4U> shape{1U, 1U, K, N};
    for (uint32_t e = 0; e < E; ++e) {
        const auto w = ttml::test_utils::make_uniform_xarray<float>(shape, -0.05F, 0.05F, rng());
        out.push_back(ttml::autograd::create_tensor(ttml::core::from_xtensor(w, device), /*requires_grad=*/true));
    }
    return out;
}

ttml::autograd::TensorPtr build_grouped_tensor(
    uint32_t T_cap,
    uint32_t H,
    const std::vector<uint32_t>& offsets,
    const std::vector<uint32_t>& counts,
    ttnn::distributed::MeshDevice* device,
    std::mt19937& rng) {
    xt::xarray<float> grouped = xt::zeros<float>(std::vector<std::size_t>{1U, 1U, T_cap, H});
    for (uint32_t e = 0; e < counts.size(); ++e) {
        if (counts[e] == 0U) {
            continue;
        }
        const std::array<std::size_t, 4U> slice_shape{1U, 1U, counts[e], H};
        const auto slice = ttml::test_utils::make_uniform_xarray<float>(slice_shape, -1.0F, 1.0F, rng());
        xt::view(grouped, 0, 0, xt::range(offsets[e], offsets[e] + counts[e]), xt::all()) = xt::view(slice, 0, 0);
    }
    return ttml::autograd::create_tensor(ttml::core::from_xtensor(grouped, device), /*requires_grad=*/true);
}

Stats summarize(const std::vector<double>& times_us) {
    Stats s;
    if (times_us.empty()) {
        return s;
    }
    s.avg_us = std::accumulate(times_us.begin(), times_us.end(), 0.0) / static_cast<double>(times_us.size());
    s.min_us = *std::min_element(times_us.begin(), times_us.end());
    s.max_us = *std::max_element(times_us.begin(), times_us.end());
    std::vector<double> sorted = times_us;
    const auto mid = sorted.begin() + sorted.size() / 2;
    std::nth_element(sorted.begin(), mid, sorted.end());
    s.p50_us = *mid;
    return s;
}

// ---------------------------------------------------------------------------
// Timing loop
// ---------------------------------------------------------------------------

CaseResult run_case(const Case& c, uint32_t num_warmup, uint32_t num_measure) {
    auto* device = &ttml::autograd::ctx().get_device();

    auto& rng_global = ttml::autograd::ctx().get_generator();
    std::mt19937 rng(rng_global());

    const auto offsets_host = compute_offsets(c.counts);
    constexpr uint32_t kTile = 32U;
    const uint32_t T_cap = offsets_host.back() + c.slack_tiles * kTile;

    const auto offsets_tensor = ttml::core::from_vector<uint32_t, ttnn::DataType::UINT32>(
        offsets_host, ttnn::Shape({static_cast<uint32_t>(offsets_host.size())}), device, ttnn::Layout::ROW_MAJOR);

    // Build weights once and reuse across iterations.
    const auto w_gate = build_expert_weight_list(c.E, c.I, c.H, device, rng);
    const auto w_up = build_expert_weight_list(c.E, c.I, c.H, device, rng);
    const auto w_down = build_expert_weight_list(c.E, c.H, c.I, device, rng);

    auto build_grouped = [&]() { return build_grouped_tensor(T_cap, c.H, offsets_host, c.counts, device, rng); };

    // Forward-only timing pass.
    std::vector<double> fwd_times;
    fwd_times.reserve(num_measure);

    auto run_forward = [&]() -> double {
        const auto grouped = build_grouped();
        const auto t0 = std::chrono::high_resolution_clock::now();
        const auto out = ttml::ops::moe_ffn_swiglu_fw(grouped, offsets_tensor, w_gate, w_up, w_down);
        tt::tt_metal::distributed::Synchronize(device, std::nullopt);
        const auto t1 = std::chrono::high_resolution_clock::now();
        ttml::autograd::ctx().reset_graph();
        return std::chrono::duration<double, std::micro>(t1 - t0).count();
    };

    for (uint32_t i = 0; i < num_warmup; ++i) {
        (void)run_forward();
    }
    for (uint32_t i = 0; i < num_measure; ++i) {
        fwd_times.push_back(run_forward());
    }

    // Forward+backward timing pass.
    std::vector<double> fb_times;
    fb_times.reserve(num_measure);

    auto run_fwd_bwd = [&]() -> double {
        const auto grouped = build_grouped();
        const auto t0 = std::chrono::high_resolution_clock::now();
        const auto out = ttml::ops::moe_ffn_swiglu_fw(grouped, offsets_tensor, w_gate, w_up, w_down);
        out->set_grad(ttml::core::ones_like(out->get_value()));
        out->backward();
        tt::tt_metal::distributed::Synchronize(device, std::nullopt);
        const auto t1 = std::chrono::high_resolution_clock::now();
        ttml::autograd::ctx().reset_graph();
        return std::chrono::duration<double, std::micro>(t1 - t0).count();
    };

    for (uint32_t i = 0; i < num_warmup; ++i) {
        (void)run_fwd_bwd();
    }
    for (uint32_t i = 0; i < num_measure; ++i) {
        fb_times.push_back(run_fwd_bwd());
    }

    CaseResult r;
    r.forward = summarize(fwd_times);
    r.forward_backward = summarize(fb_times);
    return r;
}

// ---------------------------------------------------------------------------
// Case generators
// ---------------------------------------------------------------------------

// Uniform top-K dispatch: every active expert receives ~tokens·K/E rows.
std::vector<uint32_t> uniform_counts(uint32_t E, uint32_t tokens, uint32_t K) {
    const uint32_t per_expert = (tokens * K) / E;
    return std::vector<uint32_t>(E, per_expert);
}

// Skewed dispatch: one hot expert gets `hot_frac` share of (tokens·K), rest split evenly.
std::vector<uint32_t> skewed_counts(uint32_t E, uint32_t tokens, uint32_t K, float hot_frac) {
    const uint32_t total_active = tokens * K;
    const uint32_t hot = static_cast<uint32_t>(static_cast<float>(total_active) * hot_frac);
    const uint32_t rest = (total_active - hot) / (E - 1U);
    std::vector<uint32_t> counts(E, rest);
    counts[0] = hot;
    return counts;
}

// ---------------------------------------------------------------------------
// Output
// ---------------------------------------------------------------------------

void print_header() {
    fmt::print("\n");
    fmt::print("+------------------------------------+--------+--------+--------+--------+--------+--------+\n");
    fmt::print("| case                               | fwd    | fwd    | fwd    | fwd+bw | fwd+bw | fwd+bw |\n");
    fmt::print("|                                    | avg µs | min µs | p50 µs | avg µs | min µs | p50 µs |\n");
    fmt::print("+------------------------------------+--------+--------+--------+--------+--------+--------+\n");
}

void print_row(const std::string& name, const CaseResult& r) {
    fmt::print(
        "| {:<34} | {:>6.0f} | {:>6.0f} | {:>6.0f} | {:>6.0f} | {:>6.0f} | {:>6.0f} |\n",
        name,
        r.forward.avg_us,
        r.forward.min_us,
        r.forward.p50_us,
        r.forward_backward.avg_us,
        r.forward_backward.min_us,
        r.forward_backward.p50_us);
}

void print_footer() {
    fmt::print("+------------------------------------+--------+--------+--------+--------+--------+--------+\n");
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

}  // namespace

int main() {
    try {
        constexpr uint32_t num_warmup = 2U;
        constexpr uint32_t num_measure = 10U;

        const tt::tt_metal::distributed::MeshShape mesh(1, 1);
        ttml::autograd::ctx().open_device(mesh);

        fmt::print("MoE FFN SwiGLU benchmark — warmup={} measure={}\n", num_warmup, num_measure);

        // Cases. Names follow `<H>x<I>_<E>e_<dispatch>_<batch>tok`.
        // Total token count = batch_size · sequence (a synthetic value here).
        const std::vector<Case> cases = {
            // Mixtral-like 8x7B microbatch: H=4096, I=14336, E_local=8, top-K=2
            {"mixtral_h4k_i14k_e8_uniform_4ktok", 8, 4096, 14336, uniform_counts(8, /*tokens=*/4096, /*K=*/2)},
            {"mixtral_h4k_i14k_e8_skewed40_4ktok", 8, 4096, 14336, skewed_counts(8, /*tokens=*/4096, /*K=*/2, 0.4F)},

            // DeepSeek-like with high E: H=2048, I=1408, E_local=12, top-K=6 (EP=8 from 96 experts)
            {"dsv2_h2k_i1408_e12_uniform_2ktok", 12, 2048, 1408, uniform_counts(12, /*tokens=*/2048, /*K=*/6)},
            {"dsv2_h2k_i1408_e12_skewed40_2ktok", 12, 2048, 1408, skewed_counts(12, /*tokens=*/2048, /*K=*/6, 0.4F)},

            // H=4096, I=512, E_local=12, top-K=8, seq_len=4096
            {"h4096_i512_e12_uniform_4ktok", 12, 4096, 512, uniform_counts(12, /*tokens=*/4096, /*K=*/8)},
            {"h4096_i512_e12_skewed40_4ktok", 12, 4096, 512, skewed_counts(12, /*tokens=*/4096, /*K=*/8, 0.4F)},

            // Smaller debug shape that fits comfortably and runs fast.
            {"debug_h512_i1024_e4_uniform_512tok", 4, 512, 1024, uniform_counts(4, /*tokens=*/512, /*K=*/2)},

            // Trailing-pad path comparison: same shape, with vs without slack.
            // Slack mirrors moe_group's worst-case formula: per-expert (32 + 7·N) rows
            // for WH N=72 → ~17 tile-rows per expert. E=12 → 12·17 ≈ 204 tile-rows.
            {"h4k_i512_e12_uniform_4ktok_pad0", 12, 4096, 512, uniform_counts(12, 4096, 8), /*slack_tiles=*/0U},
            {"h4k_i512_e12_uniform_4ktok_pad204", 12, 4096, 512, uniform_counts(12, 4096, 8), /*slack_tiles=*/204U},
            {"mixtral_h4k_i14k_e8_uniform_4ktok_pad0", 8, 4096, 14336, uniform_counts(8, 4096, 2), /*slack_tiles=*/0U},
            {"mixtral_h4k_i14k_e8_uniform_4ktok_pad136",
             8,
             4096,
             14336,
             uniform_counts(8, 4096, 2),
             /*slack_tiles=*/136U},
        };

        print_header();
        for (const auto& c : cases) {
            const auto r = run_case(c, num_warmup, num_measure);
            print_row(c.name, r);
        }
        print_footer();

        ttml::autograd::ctx().close_device();
        return 0;
    } catch (const std::exception& e) {
        fmt::print(stderr, "moe_ffn_swiglu_benchmark failed: {}\n", e.what());
        return 1;
    }
}
