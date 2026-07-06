// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include <array>
#include <cstdint>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/hal.hpp>
#include <vector>

#include "autograd/auto_context.hpp"
#include "core/tt_tensor_utils.hpp"
#include "metal/ops/moe_group/moe_group.hpp"
#include "metal/ops/moe_ungroup/moe_ungroup.hpp"
#include "moe_test_utils.hpp"

namespace {

constexpr uint32_t kSentinel = 0xFFFFFFFFU;

// Float-comparison tolerance for bf16-derived outputs. Single source of truth so
// rtol/atol stay consistent across all checks in this test file.
constexpr float kRtol = 5e-2F;
constexpr float kAtol = 1e-2F;

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

// Build host inputs with dispatched + scores already bf16-roundtripped on the
// test device, so reference comparisons match what the device actually sees.
ttml::test_utils::moe::MoeHostInputs make_inputs(
    uint32_t D, uint32_t B, uint32_t S, uint32_t H, uint32_t E, uint32_t K) {
    return ttml::test_utils::moe::make_moe_host_inputs({
        .D = D,
        .B = B,
        .S = S,
        .H = H,
        .E = E,
        .K = K,
        .roundtrip_device = &ttml::autograd::ctx().get_device(),
    });
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
    const ttml::test_utils::moe::MoeHostInputs& host, const std::vector<uint16_t>& local_expert_ids, uint32_t k) {
    auto& dev = ttml::autograd::ctx().get_device();
    const uint32_t E_local = static_cast<uint32_t>(local_expert_ids.size());

    auto dev_in = ttml::test_utils::moe::to_device_inputs(host, local_expert_ids, &dev);
    auto [grouped, grouped_scores, k_slot, counts, offsets, plan] = ttml::metal::moe_group(
        dev_in.dispatched_bf16, dev_in.metadata_u16, dev_in.scores_bf16, dev_in.leids_u16, E_local, k);

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
    const ttml::test_utils::moe::MoeHostInputs& host, const std::vector<uint16_t>& local_expert_ids, uint32_t k) {
    const uint32_t D = static_cast<uint32_t>(host.dispatched.shape(0));
    const uint32_t B = static_cast<uint32_t>(host.dispatched.shape(1));
    const uint32_t S = static_cast<uint32_t>(host.dispatched.shape(2));
    const uint32_t E_local = static_cast<uint32_t>(local_expert_ids.size());

    auto g = build_group_inputs(host, local_expert_ids, k);

    auto out_tt = ttml::metal::moe_ungroup(g.expert_out, g.plan, g.offsets, g.grouped_scores, E_local, D, B, S);
    auto out_xt = ttml::core::to_xtensor(out_tt);

    auto ref = moe_ungroup_reference(g.expert_out_host, g.plan_host, g.grouped_scores_host, D, B, S);
    EXPECT_TRUE(xt::allclose(out_xt, ref, kRtol, kAtol));
}

}  // namespace

TEST_F(MoeUngroupTest, Basic) {
    constexpr uint32_t D = 2, B = 1, S = 32, H = 64;
    constexpr uint32_t E = 4, K = 2;
    const std::vector<uint16_t> leids = {0, 1};
    run_and_check(make_inputs(D, B, S, H, E, K), leids, K);
}

TEST_F(MoeUngroupTest, LargerH) {
    constexpr uint32_t D = 2, B = 1, S = 32, H = 256;
    constexpr uint32_t E = 4, K = 2;
    const std::vector<uint16_t> leids = {0, 1};
    run_and_check(make_inputs(D, B, S, H, E, K), leids, K);
}

TEST_F(MoeUngroupTest, NonTileAlignedH) {
    constexpr uint32_t D = 2, B = 1, S = 32, H = 80;
    constexpr uint32_t E = 4, K = 2;
    const std::vector<uint16_t> leids = {0, 1};
    run_and_check(make_inputs(D, B, S, H, E, K), leids, K);
}

TEST_F(MoeUngroupTest, AllTokensActive) {
    constexpr uint32_t D = 2, B = 1, S = 32, H = 64;
    constexpr uint32_t E = 2, K = 2;
    const std::vector<uint16_t> leids = {0, 1};
    run_and_check(make_inputs(D, B, S, H, E, K), leids, K);
}

TEST_F(MoeUngroupTest, LargeELocal) {
    constexpr uint32_t D = 2, B = 1, S = 64, H = 64;
    constexpr uint32_t E = 64, K = 4;
    std::vector<uint16_t> leids;
    leids.reserve(32);
    for (uint32_t i = 0; i < 32; ++i) leids.push_back(static_cast<uint16_t>(i));
    run_and_check(make_inputs(D, B, S, H, E, K), leids, K);
}

// End-to-end integration: moe_group -> identity FFN (expert_out = grouped) ->
// moe_ungroup should recover sum_e sum_k score_{t,k}[leid match] * dispatched[t].
//
// Requires leids = [0, E) so that every metadata id is local, which makes
// `w_sum` the simple sum over all K scores per token.
void check_group_ungroup_roundtrip(uint32_t D, uint32_t B, uint32_t S, uint32_t H, uint32_t E, uint32_t K) {
    std::vector<uint16_t> leids;
    leids.reserve(E);
    for (uint32_t i = 0; i < E; ++i) leids.push_back(static_cast<uint16_t>(i));
    auto host = make_inputs(D, B, S, H, E, K);

    auto g = build_group_inputs(host, leids, K);
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
                for (uint32_t ki = 0; ki < K; ++ki) w_sum += host.scores(d, b, s, ki);
                for (uint32_t hh = 0; hh < H; ++hh) expected(d, b, s, hh) = w_sum * host.dispatched(d, b, s, hh);
            }
        }
    }
    EXPECT_TRUE(xt::allclose(out_xt, expected, kRtol, kAtol));
}

TEST_F(MoeUngroupTest, GroupUngroupRoundTrip) {
    check_group_ungroup_roundtrip(/*D=*/2, /*B=*/1, /*S=*/32, /*H=*/64, /*E=*/4, /*K=*/2);
}

TEST_F(MoeUngroupTest, GroupUngroupRoundTripLargerH) {
    check_group_ungroup_roundtrip(/*D=*/2, /*B=*/1, /*S=*/32, /*H=*/256, /*E=*/4, /*K=*/2);
}

TEST_F(MoeUngroupTest, GroupUngroupRoundTripNonTileAlignedH) {
    check_group_ungroup_roundtrip(/*D=*/2, /*B=*/1, /*S=*/32, /*H=*/80, /*E=*/4, /*K=*/2);
}

TEST_F(MoeUngroupTest, GroupUngroupRoundTripLargerS) {
    check_group_ungroup_roundtrip(/*D=*/2, /*B=*/1, /*S=*/128, /*H=*/64, /*E=*/4, /*K=*/2);
}

TEST_F(MoeUngroupTest, GroupUngroupRoundTripKEqualsE) {
    // K == E → every token hits every expert; w_sum is the full row sum of scores.
    check_group_ungroup_roundtrip(/*D=*/2, /*B=*/1, /*S=*/32, /*H=*/64, /*E=*/4, /*K=*/4);
}

TEST_F(MoeUngroupTest, GroupUngroupRoundTripLargerD) {
    check_group_ungroup_roundtrip(/*D=*/4, /*B=*/1, /*S=*/32, /*H=*/64, /*E=*/4, /*K=*/2);
}

TEST_F(MoeUngroupTest, GroupUngroupRoundTripLargerELocal) {
    check_group_ungroup_roundtrip(/*D=*/2, /*B=*/1, /*S=*/64, /*H=*/64, /*E=*/8, /*K=*/4);
}

// NIGHTLY_ prefix keeps this off the per-PR run (CI filters out *NIGHTLY*),
// matching the convention in swiglu_op_test.cpp. Production-like shape: D=8,
// large E=96, multi-thousand T_cap, wide H. Full S=4096/H=4096 would allocate
// several ~0.5 GB host float arrays for the reference; S=1024/H=2048 keeps host
// RAM modest while still exercising the large-E routing path and tile counts
// that small per-PR cases don't. Profiling for the full roofline shape lives in
// moe_profile_sweep_test.cpp.
TEST_F(MoeUngroupTest, NIGHTLY_RealShapeLargeE) {
    check_group_ungroup_roundtrip(/*D=*/8, /*B=*/1, /*S=*/1024, /*H=*/2048, /*E=*/96, /*K=*/8);
}
