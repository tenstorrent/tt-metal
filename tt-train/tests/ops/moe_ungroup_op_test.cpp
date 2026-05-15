// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include <algorithm>
#include <cstdint>
#include <random>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/hal.hpp>
#include <vector>

#include "autograd/auto_context.hpp"
#include "core/tt_tensor_utils.hpp"
#include "metal/ops/moe_group/moe_group.hpp"
#include "metal/ops/moe_ungroup/moe_ungroup.hpp"

namespace {

constexpr uint32_t kSentinel = 0xFFFFFFFFU;

class MoeUngroupTest : public ::testing::Test {
protected:
    static void SetUpTestSuite() {
        ttml::autograd::ctx().open_device();
        ttml::autograd::ctx().set_seed(42);
    }
    static void TearDownTestSuite() {
        ttml::autograd::ctx().close_device();
    }
};

// Input builders ------------------------------------------------------------

xt::xarray<float> make_dispatched(uint32_t D, uint32_t B, uint32_t S, uint32_t H, uint32_t seed = 0) {
    std::mt19937 rng(seed + 5);
    std::uniform_real_distribution<float> dist(-1.0F, 1.0F);
    xt::xarray<float> out = xt::zeros<float>({D, B, S, H});
    for (auto it = out.begin(); it != out.end(); ++it) *it = dist(rng);
    // bf16 round-trip so reference matches device exactly.
    auto& dev = ttml::autograd::ctx().get_device();
    auto t = ttml::core::from_xtensor<float, ttnn::DataType::BFLOAT16>(out, &dev, ttnn::Layout::ROW_MAJOR);
    return ttml::core::to_xtensor(t);
}

xt::xarray<uint32_t> make_metadata(uint32_t D, uint32_t B, uint32_t S, uint32_t K, uint32_t E, uint32_t seed = 0) {
    xt::xarray<uint32_t> out = xt::zeros<uint32_t>({D, B, S, K});
    std::mt19937 rng(seed + 1);
    std::vector<uint32_t> all(E);
    for (uint32_t e = 0; e < E; ++e) all[e] = e;
    for (uint32_t d = 0; d < D; ++d) {
        for (uint32_t b = 0; b < B; ++b) {
            for (uint32_t s = 0; s < S; ++s) {
                std::shuffle(all.begin(), all.end(), rng);
                for (uint32_t ki = 0; ki < K; ++ki) out(d, b, s, ki) = all[ki];
            }
        }
    }
    return out;
}

xt::xarray<float> make_scores(uint32_t D, uint32_t B, uint32_t S, uint32_t K, uint32_t seed = 0) {
    xt::xarray<float> out = xt::zeros<float>({D, B, S, K});
    std::mt19937 rng(seed + 13);
    std::uniform_real_distribution<float> dist(0.0F, 0.5F);
    for (auto it = out.begin(); it != out.end(); ++it) *it = dist(rng);
    auto& dev = ttml::autograd::ctx().get_device();
    auto t = ttml::core::from_xtensor<float, ttnn::DataType::BFLOAT16>(out, &dev, ttnn::Layout::ROW_MAJOR);
    return ttml::core::to_xtensor(t);
}

// Reference: for each tile-row i with plan[i]=src (!= SENTINEL), add
// grouped_scores[i] * expert_out[i] into ungrouped[src].
xt::xarray<float> moe_ungroup_reference(
    const xt::xarray<float>& expert_out,       // [T_cap, H]
    const std::vector<uint32_t>& plan,         // [T_cap]
    const std::vector<float>& grouped_scores,  // [T_cap]
    uint32_t D,
    uint32_t B,
    uint32_t S) {
    const uint32_t T_cap = static_cast<uint32_t>(expert_out.shape(0));
    const uint32_t H = static_cast<uint32_t>(expert_out.shape(1));
    xt::xarray<float> ungrouped = xt::zeros<float>({D, B, S, H});
    for (uint32_t i = 0; i < T_cap; ++i) {
        const uint32_t src = plan[i];
        if (src == kSentinel)
            continue;
        const float w = grouped_scores[i];
        const uint32_t flat_row = src;  // flat (d,b,s) index
        const uint32_t d = flat_row / (B * S);
        const uint32_t b = (flat_row / S) % B;
        const uint32_t s = flat_row % S;
        for (uint32_t hh = 0; hh < H; ++hh) {
            ungrouped(d, b, s, hh) += w * expert_out(i, hh);
        }
    }
    return ungrouped;
}

struct GroupOutputs {
    ttnn::Tensor expert_out;  // we reuse `grouped` as expert_out (identity FFN)
    ttnn::Tensor plan;
    ttnn::Tensor offsets;
    ttnn::Tensor grouped_scores;
    std::vector<uint32_t> plan_host;
    std::vector<float> grouped_scores_host;
    xt::xarray<float> expert_out_host;  // [T_cap, H]
};

GroupOutputs build_group_inputs(
    const xt::xarray<float>& dispatched,
    const xt::xarray<uint32_t>& metadata,
    const xt::xarray<float>& scores,
    const std::vector<uint16_t>& local_expert_ids,
    uint32_t k) {
    auto& dev = ttml::autograd::ctx().get_device();
    const uint32_t E_local = static_cast<uint32_t>(local_expert_ids.size());

    auto disp_tt = ttml::core::from_xtensor<float, ttnn::DataType::BFLOAT16>(dispatched, &dev, ttnn::Layout::ROW_MAJOR);
    xt::xarray<uint16_t> md16 = xt::cast<uint16_t>(metadata);
    auto md_tt = ttml::core::from_xtensor<uint16_t, ttnn::DataType::UINT16>(md16, &dev, ttnn::Layout::ROW_MAJOR);
    auto sc_tt = ttml::core::from_xtensor<float, ttnn::DataType::BFLOAT16>(scores, &dev, ttnn::Layout::ROW_MAJOR);
    xt::xarray<uint16_t> leids_arr = xt::adapt(local_expert_ids, std::vector<size_t>{local_expert_ids.size()});
    auto leids_tt =
        ttml::core::from_xtensor<uint16_t, ttnn::DataType::UINT16>(leids_arr, &dev, ttnn::Layout::ROW_MAJOR);
    auto [grouped, grouped_scores, k_slot, counts, offsets, plan] =
        ttml::metal::moe_group(disp_tt, md_tt, sc_tt, leids_tt, E_local, k);

    GroupOutputs g{std::move(grouped), std::move(plan), std::move(offsets), std::move(grouped_scores), {}, {}, {}};
    auto plan_xt = ttml::core::to_xtensor<uint32_t>(g.plan);
    g.plan_host.assign(plan_xt.begin(), plan_xt.end());
    auto gs_xt = ttml::core::to_xtensor(g.grouped_scores);
    g.grouped_scores_host.assign(gs_xt.begin(), gs_xt.end());
    auto grouped_rm = ttnn::to_layout(g.expert_out, ttnn::ROW_MAJOR_LAYOUT);
    auto grouped_xt = ttml::core::to_xtensor(grouped_rm);  // [1, 1, T_cap, H]
    g.expert_out_host = xt::squeeze(grouped_xt, std::array<int, 2>{0, 1});
    return g;
}

void run_and_check(
    const xt::xarray<float>& dispatched,
    const xt::xarray<uint32_t>& metadata,
    const xt::xarray<float>& scores,
    const std::vector<uint16_t>& local_expert_ids,
    uint32_t k) {
    const uint32_t D = static_cast<uint32_t>(dispatched.shape(0));
    const uint32_t B = static_cast<uint32_t>(dispatched.shape(1));
    const uint32_t S = static_cast<uint32_t>(dispatched.shape(2));
    const uint32_t E_local = static_cast<uint32_t>(local_expert_ids.size());

    auto g = build_group_inputs(dispatched, metadata, scores, local_expert_ids, k);

    auto out_tt = ttml::metal::moe_ungroup(g.expert_out, g.plan, g.offsets, g.grouped_scores, E_local, D, B, S);
    auto out_xt = ttml::core::to_xtensor(out_tt);

    auto ref = moe_ungroup_reference(g.expert_out_host, g.plan_host, g.grouped_scores_host, D, B, S);
    EXPECT_TRUE(xt::allclose(out_xt, ref, 5e-2F, 1e-2F));
}

}  // namespace

TEST_F(MoeUngroupTest, Basic) {
    constexpr uint32_t D = 2, B = 1, S = 32, H = 64;
    constexpr uint32_t E = 4, K = 2;
    const std::vector<uint16_t> leids = {0, 1};
    auto dispatched = make_dispatched(D, B, S, H);
    auto metadata = make_metadata(D, B, S, K, E);
    auto scores = make_scores(D, B, S, K);
    run_and_check(dispatched, metadata, scores, leids, K);
}

TEST_F(MoeUngroupTest, LargerH) {
    constexpr uint32_t D = 2, B = 1, S = 32, H = 256;
    constexpr uint32_t E = 4, K = 2;
    const std::vector<uint16_t> leids = {0, 1};
    auto dispatched = make_dispatched(D, B, S, H);
    auto metadata = make_metadata(D, B, S, K, E);
    auto scores = make_scores(D, B, S, K);
    run_and_check(dispatched, metadata, scores, leids, K);
}

TEST_F(MoeUngroupTest, NonTileAlignedH) {
    constexpr uint32_t D = 2, B = 1, S = 32, H = 80;
    constexpr uint32_t E = 4, K = 2;
    const std::vector<uint16_t> leids = {0, 1};
    auto dispatched = make_dispatched(D, B, S, H);
    auto metadata = make_metadata(D, B, S, K, E);
    auto scores = make_scores(D, B, S, K);
    run_and_check(dispatched, metadata, scores, leids, K);
}

TEST_F(MoeUngroupTest, AllTokensActive) {
    constexpr uint32_t D = 2, B = 1, S = 32, H = 64;
    constexpr uint32_t E = 2, K = 2;
    const std::vector<uint16_t> leids = {0, 1};
    auto dispatched = make_dispatched(D, B, S, H);
    auto metadata = make_metadata(D, B, S, K, E);
    auto scores = make_scores(D, B, S, K);
    run_and_check(dispatched, metadata, scores, leids, K);
}

TEST_F(MoeUngroupTest, LargeELocal) {
    constexpr uint32_t D = 2, B = 1, S = 64, H = 64;
    constexpr uint32_t E = 64, K = 4;
    std::vector<uint16_t> leids;
    leids.reserve(32);
    for (uint16_t i = 0; i < 32; ++i) leids.push_back(i);
    auto dispatched = make_dispatched(D, B, S, H);
    auto metadata = make_metadata(D, B, S, K, E);
    auto scores = make_scores(D, B, S, K);
    run_and_check(dispatched, metadata, scores, leids, K);
}

// End-to-end integration: moe_group -> identity FFN (expert_out = grouped) ->
// moe_ungroup should recover sum_e sum_k score_{t,k}[leid match] * dispatched[t].
TEST_F(MoeUngroupTest, GroupUngroupRoundTrip) {
    constexpr uint32_t D = 2, B = 1, S = 32, H = 64;
    constexpr uint32_t E = 4, K = 2;
    const std::vector<uint16_t> leids = {0, 1, 2, 3};  // all experts local
    auto dispatched = make_dispatched(D, B, S, H);
    auto metadata = make_metadata(D, B, S, K, E);
    auto scores = make_scores(D, B, S, K);

    auto g = build_group_inputs(dispatched, metadata, scores, leids, K);
    auto out_tt = ttml::metal::moe_ungroup(
        g.expert_out, g.plan, g.offsets, g.grouped_scores, static_cast<uint32_t>(leids.size()), D, B, S);
    auto out_xt = ttml::core::to_xtensor(out_tt);

    // Closed-form expected: for each token t, sum over k of score[t,k] * dispatched[t]
    // (since every expert in metadata is local here and the FFN is identity).
    xt::xarray<float> expected = xt::zeros<float>({D, B, S, H});
    for (uint32_t d = 0; d < D; ++d) {
        for (uint32_t b = 0; b < B; ++b) {
            for (uint32_t s = 0; s < S; ++s) {
                float w_sum = 0.0F;
                for (uint32_t ki = 0; ki < K; ++ki) w_sum += scores(d, b, s, ki);
                for (uint32_t hh = 0; hh < H; ++hh) expected(d, b, s, hh) = w_sum * dispatched(d, b, s, hh);
            }
        }
    }
    EXPECT_TRUE(xt::allclose(out_xt, expected, 5e-2F, 1e-2F));
}
