// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Tests for ttml::metal::ring_lse_merge, the fused online-softmax (logsumexp) merge of ring attention.
// The reference is the ttnn composite it replaces in ring_attention_sdpa (slice, maximum, 2x subtract+exp,
// add, log, add, 2x subtract+exp, typecast, 2x multiply, add), run on the same device inputs in FP32.

#include <gtest/gtest.h>

#include <array>
#include <cmath>
#include <core/ttnn_all_includes.hpp>
#include <cstdint>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

#include "autograd/auto_context.hpp"
#include "core/tt_tensor_utils.hpp"
#include "metal/operations.hpp"
#include "test_utils/program_cache.hpp"
#include "test_utils/random_data.hpp"
#include "ttnn/operations/copy/typecast/typecast.hpp"
#include "ttnn/operations/data_movement/slice/slice.hpp"
#include "ttnn/operations/eltwise/binary/binary.hpp"
#include "ttnn/operations/eltwise/binary/binary_composite.hpp"
#include "ttnn/operations/eltwise/unary/unary.hpp"

namespace {

using Shape4 = std::array<uint32_t, 4>;  // (B, H, S, D)

constexpr float kNegInf = -std::numeric_limits<float>::infinity();
constexpr uint32_t kLseWidth = 32U;

// Every tile row of one (b, h) pair whose index is a multiple of `period` gets lse_i = -inf.
enum class NegInfRows { None, EveryThird, All };

struct MergeInputs {
    xt::xarray<float> accum;               // (B, H, S, D)
    xt::xarray<float> running;             // (B, H, S, 32), lse in column 0, zeros elsewhere
    xt::xarray<float> step_output;         // (B, H, S, D), bf16-representable values
    xt::xarray<float> step_intermediates;  // (B, H, S, 32), lse in column 0, zeros elsewhere
};

xt::xarray<float> lse_layout(const xt::xarray<float>& lse_col) {
    const auto& s = lse_col.shape();
    xt::xarray<float> out = xt::zeros<float>({s[0], s[1], s[2], static_cast<std::size_t>(kLseWidth)});
    xt::view(out, xt::all(), xt::all(), xt::all(), xt::range(0, 1)) = lse_col;
    return out;
}

xt::xarray<float> round_to_bf16(const xt::xarray<float>& x) {
    auto t = ttml::core::from_xtensor<float, ttnn::DataType::BFLOAT16>(x, &ttml::autograd::ctx().get_device());
    return ttml::core::to_xtensor(t);
}

MergeInputs make_inputs(
    const Shape4& shape, uint32_t seed, NegInfRows step_neg_inf, bool running_neg_inf, bool zero_accum = false) {
    const auto [B, H, S, D] = shape;
    MergeInputs in;
    in.accum = zero_accum ? xt::xarray<float>(xt::zeros<float>({B, H, S, D}))
                          : ttml::test_utils::make_uniform_xarray<float>(shape, -2.0F, 2.0F, seed);
    in.step_output = round_to_bf16(ttml::test_utils::make_uniform_xarray<float>(shape, -2.0F, 2.0F, seed + 1U));

    const Shape4 col_shape{B, H, S, 1U};
    xt::xarray<float> running = ttml::test_utils::make_uniform_xarray<float>(col_shape, -3.0F, 9.0F, seed + 2U);
    xt::xarray<float> step = ttml::test_utils::make_uniform_xarray<float>(col_shape, -3.0F, 9.0F, seed + 3U);
    if (running_neg_inf) {
        running.fill(kNegInf);
    }
    for (uint32_t b = 0; b < B; ++b) {
        for (uint32_t h = 0; h < H; ++h) {
            for (uint32_t s = 0; s < S; ++s) {
                const uint32_t tile_row = s / tt::constants::TILE_HEIGHT;
                const bool neg_inf = step_neg_inf == NegInfRows::All ||
                                     (step_neg_inf == NegInfRows::EveryThird && (tile_row + b + h) % 3U == 0U);
                if (neg_inf) {
                    step(b, h, s, 0) = kNegInf;
                }
            }
        }
    }
    in.running = lse_layout(running);
    in.step_intermediates = lse_layout(step);
    return in;
}

struct DeviceInputs {
    ttnn::Tensor accum;
    ttnn::Tensor running;
    ttnn::Tensor step_output;
    ttnn::Tensor step_intermediates;
};

DeviceInputs to_device(const MergeInputs& in) {
    auto* device = &ttml::autograd::ctx().get_device();
    return {
        ttml::core::from_xtensor<float, ttnn::DataType::FLOAT32>(in.accum, device),
        ttml::core::from_xtensor<float, ttnn::DataType::FLOAT32>(in.running, device),
        ttml::core::from_xtensor<float, ttnn::DataType::BFLOAT16>(in.step_output, device),
        ttml::core::from_xtensor<float, ttnn::DataType::FLOAT32>(in.step_intermediates, device),
    };
}

// The 17-op composite that ring_attention_sdpa used before the fused op, on a (B, H, S, 1) running lse.
// Returns {accum, running (B, H, S, 1)}.
std::pair<ttnn::Tensor, ttnn::Tensor> composite_merge(
    const ttnn::Tensor& output_accum,
    const ttnn::Tensor& global_lse,
    const ttnn::Tensor& output_tensor,
    const ttnn::Tensor& intermediate_tensor) {
    const auto [batch_num, heads, seq_len_local, width] = intermediate_tensor.logical_shape().to_array_4D();
    const ttsl::SmallVector<uint32_t> slice_step = {1, 1, 1, 1};
    const ttsl::SmallVector<uint32_t> lse_start = {0, 0, 0, 0};
    const ttsl::SmallVector<uint32_t> lse_end = {batch_num, heads, seq_len_local, 1};
    ttnn::Tensor lse_chunk = ttnn::slice(intermediate_tensor, lse_start, lse_end, slice_step);

    ttnn::Tensor m = ttnn::maximum(global_lse, lse_chunk);
    ttnn::Tensor exp_global = ttnn::exp(ttnn::subtract(global_lse, m));
    ttnn::Tensor exp_chunk = ttnn::exp(ttnn::subtract(lse_chunk, m));
    ttnn::Tensor new_lse = ttnn::add(m, ttnn::log(ttnn::add(exp_global, exp_chunk)));

    ttnn::Tensor old_weight = ttnn::exp(ttnn::subtract(global_lse, new_lse));
    ttnn::Tensor new_weight = ttnn::exp(ttnn::subtract(lse_chunk, new_lse));

    ttnn::Tensor step_output_fp32 = ttnn::typecast(output_tensor, ttnn::DataType::FLOAT32);
    ttnn::Tensor accum =
        ttnn::add(ttnn::multiply(output_accum, old_weight), ttnn::multiply(step_output_fp32, new_weight));
    return {accum, new_lse};
}

ttnn::Tensor lse_column(const ttnn::Tensor& lse_layout_tensor) {
    const auto [b, h, s, w] = lse_layout_tensor.logical_shape().to_array_4D();
    return ttnn::slice(
        lse_layout_tensor,
        ttsl::SmallVector<uint32_t>{0, 0, 0, 0},
        ttsl::SmallVector<uint32_t>{b, h, s, 1},
        ttsl::SmallVector<uint32_t>{1, 1, 1, 1});
}

// Double-precision host reference of the merge (column 0 of the lse tensors only).
std::pair<xt::xarray<double>, xt::xarray<double>> host_merge(const MergeInputs& in) {
    const auto& shape = in.accum.shape();
    xt::xarray<double> accum = xt::cast<double>(in.accum);
    xt::xarray<double> lse = xt::zeros<double>({shape[0], shape[1], shape[2], std::size_t{1}});
    for (std::size_t b = 0; b < shape[0]; ++b) {
        for (std::size_t h = 0; h < shape[1]; ++h) {
            for (std::size_t s = 0; s < shape[2]; ++s) {
                const double a = in.running(b, h, s, 0);
                const double c = in.step_intermediates(b, h, s, 0);
                const double m = std::max(a, c);
                const double new_lse = m + std::log(std::exp(a - m) + std::exp(c - m));
                const double w_old = std::exp(a - new_lse);
                const double w_new = std::exp(c - new_lse);
                lse(b, h, s, 0) = new_lse;
                for (std::size_t d = 0; d < shape[3]; ++d) {
                    accum(b, h, s, d) = accum(b, h, s, d) * w_old + double(in.step_output(b, h, s, d)) * w_new;
                }
            }
        }
    }
    return {accum, lse};
}

struct ErrorStats {
    double max_abs = 0.0;
    double max_rel = 0.0;  // relative to max(|expected|, 1e-3)
    std::size_t non_finite_mismatch = 0;
};

template <typename E>
ErrorStats error_stats(const xt::xarray<float>& actual, const E& expected) {
    ErrorStats st;
    for (std::size_t i = 0; i < actual.size(); ++i) {
        const double a = actual.flat(i);
        const double e = expected.flat(i);
        if (!std::isfinite(a) || !std::isfinite(e)) {
            st.non_finite_mismatch += static_cast<std::size_t>(!(a == e || (std::isnan(a) && std::isnan(e))));
            continue;
        }
        const double abs_err = std::abs(a - e);
        st.max_abs = std::max(st.max_abs, abs_err);
        st.max_rel = std::max(st.max_rel, abs_err / std::max(std::abs(e), 1e-3));
    }
    return st;
}

void report(const std::string& label, const ErrorStats& st) {
    std::cout << "[ring_lse_merge] " << label << ": max_abs=" << st.max_abs << " max_rel=" << st.max_rel
              << " non_finite_mismatch=" << st.non_finite_mismatch << std::endl;
}

xt::xarray<float> col0(const xt::xarray<float>& lse_layout_values) {
    return xt::view(lse_layout_values, xt::all(), xt::all(), xt::all(), xt::range(0, 1));
}

// FP32 tolerances against the composite (same SFPU operation sequence) and against the exact float64 result.
constexpr double kCompositeAtol = 1e-5;
constexpr double kHostAtol = 1e-4;

// Runs the fused op and the composite on the same inputs and checks them against each other and the host.
void run_and_check(const std::string& label, const MergeInputs& in) {
    const DeviceInputs dev = to_device(in);

    auto [ref_accum_t, ref_lse_t] =
        composite_merge(dev.accum, lse_column(dev.running), dev.step_output, dev.step_intermediates);
    const auto ref_accum = ttml::core::to_xtensor(ref_accum_t);
    const auto ref_lse = ttml::core::to_xtensor(ref_lse_t);

    ttml::metal::ring_lse_merge(dev.accum, dev.running, dev.step_output, dev.step_intermediates);
    const auto accum = ttml::core::to_xtensor(dev.accum);
    const auto running = ttml::core::to_xtensor(dev.running);

    const auto accum_vs_composite = error_stats(accum, ref_accum);
    const auto lse_vs_composite = error_stats(col0(running), ref_lse);
    const auto [host_accum, host_lse] = host_merge(in);
    const auto accum_vs_host = error_stats(accum, host_accum);
    const auto lse_vs_host = error_stats(col0(running), host_lse);
    const auto composite_vs_host = error_stats(ref_accum, host_accum);
    report(label + " accum vs composite", accum_vs_composite);
    report(label + " lse vs composite", lse_vs_composite);
    report(label + " accum vs float64", accum_vs_host);
    report(label + " lse vs float64", lse_vs_host);
    report(label + " composite accum vs float64", composite_vs_host);

    EXPECT_EQ(accum_vs_composite.non_finite_mismatch, 0U) << label;
    EXPECT_EQ(lse_vs_composite.non_finite_mismatch, 0U) << label;
    EXPECT_LE(accum_vs_composite.max_abs, kCompositeAtol) << label;
    EXPECT_LE(lse_vs_composite.max_abs, kCompositeAtol) << label;
    EXPECT_LE(accum_vs_host.max_abs, kHostAtol) << label;
    EXPECT_LE(lse_vs_host.max_abs, kHostAtol) << label;

    // The running lse keeps the intermediates layout: columns 1..31 stay zero.
    const xt::xarray<float> padding_cols = xt::view(running, xt::all(), xt::all(), xt::all(), xt::range(1, kLseWidth));
    EXPECT_TRUE(xt::all(xt::equal(padding_cols, 0.0F))) << label << ": running lse columns 1..31 must be zero";
}

}  // namespace

class RingLseMergeTest : public ttml::test_utils::DeviceSuiteTest {
protected:
    void TearDown() override {
        ttml::autograd::ctx().reset_graph();
    }
};

struct MergeCase {
    Shape4 shape;
    std::string name;
};

class RingLseMergeShapeTest : public RingLseMergeTest, public ::testing::WithParamInterface<MergeCase> {};

TEST_P(RingLseMergeShapeTest, MatchesComposite) {
    const auto& c = GetParam();
    run_and_check(c.name, make_inputs(c.shape, 1234U, NegInfRows::None, /*running_neg_inf=*/false));
}

TEST_P(RingLseMergeShapeTest, NegInfStepLeavesStateUnchanged) {
    const auto& c = GetParam();
    const auto in = make_inputs(c.shape, 99U, NegInfRows::EveryThird, /*running_neg_inf=*/false);
    run_and_check(c.name + " -inf rows", in);

    // Bit-exact: rows with lse_i = -inf keep their accumulator and running lse.
    const DeviceInputs dev = to_device(in);
    ttml::metal::ring_lse_merge(dev.accum, dev.running, dev.step_output, dev.step_intermediates);
    const auto accum = ttml::core::to_xtensor(dev.accum);
    const auto running = ttml::core::to_xtensor(dev.running);
    const auto [B, H, S, D] = c.shape;
    std::size_t mismatches = 0;
    std::size_t neg_inf_rows = 0;
    for (uint32_t b = 0; b < B; ++b) {
        for (uint32_t h = 0; h < H; ++h) {
            for (uint32_t s = 0; s < S; ++s) {
                if (in.step_intermediates(b, h, s, 0) != kNegInf) {
                    continue;
                }
                ++neg_inf_rows;
                mismatches += !ttml::test_utils::same_bits(running(b, h, s, 0), in.running(b, h, s, 0));
                for (uint32_t d = 0; d < D; ++d) {
                    mismatches += !ttml::test_utils::same_bits(accum(b, h, s, d), in.accum(b, h, s, d));
                }
            }
        }
    }
    EXPECT_GT(neg_inf_rows, 0U);
    EXPECT_EQ(mismatches, 0U) << "rows with lse_i = -inf must be left bit-identical";
}

TEST_P(RingLseMergeShapeTest, NegInfRunningTakesStep) {
    // The first ring step: running lse = -inf, accumulator zeros -> accum = float(O_i), lse = lse_i, exactly.
    const auto& c = GetParam();
    const auto in = make_inputs(c.shape, 7U, NegInfRows::None, /*running_neg_inf=*/true, /*zero_accum=*/true);
    run_and_check(c.name + " -inf running", in);

    const DeviceInputs dev = to_device(in);
    ttml::metal::ring_lse_merge(dev.accum, dev.running, dev.step_output, dev.step_intermediates);
    const auto accum = ttml::core::to_xtensor(dev.accum);
    const auto running = ttml::core::to_xtensor(dev.running);
    EXPECT_EQ(ttml::test_utils::count_bit_mismatches(in.step_output, accum), 0U);
    EXPECT_EQ(ttml::test_utils::count_bit_mismatches(in.step_intermediates, running), 0U);
}

static std::string CaseName(const ::testing::TestParamInfo<MergeCase>& info) {
    return info.param.name;
}

// Several (b, h), D = 64 and D = 128 (2 and 4 tiles per row, block sizes 2 and 4), D = 96 (3 tiles, block 3),
// and a row count that gives split_work_to_cores two core groups.
INSTANTIATE_TEST_SUITE_P(
    All,
    RingLseMergeShapeTest,
    ::testing::Values(
        MergeCase{{1, 1, 32, 32}, "SingleTile"},
        MergeCase{{2, 3, 64, 64}, "B2H3S64D64"},
        MergeCase{{1, 8, 256, 128}, "B1H8S256D128"},
        MergeCase{{2, 2, 96, 96}, "B2H2S96D96"},
        MergeCase{{1, 2, 1024, 128}, "B1H2S1024D128"}),
    CaseName);

TEST_F(RingLseMergeTest, TwoCoreGroups) {
    const uint32_t rows = ttml::test_utils::two_core_group_units(3U);
    run_and_check("TwoCoreGroups", make_inputs({1U, 1U, rows * 32U, 64U}, 5U, NegInfRows::EveryThird, false));
}

TEST_F(RingLseMergeTest, RingSequenceMatchesComposite) {
    // A whole causal-style ring: step 0 from -inf, later steps with some -inf rows, chained in place.
    const Shape4 shape{1U, 4U, 128U, 128U};
    constexpr uint32_t kSteps = 8U;
    auto* device = &ttml::autograd::ctx().get_device();

    MergeInputs first = make_inputs(shape, 11U, NegInfRows::None, /*running_neg_inf=*/true, /*zero_accum=*/true);
    DeviceInputs fused = to_device(first);
    ttnn::Tensor ref_accum = ttml::core::from_xtensor<float, ttnn::DataType::FLOAT32>(first.accum, device);
    ttnn::Tensor ref_lse = lse_column(fused.running);

    for (uint32_t step = 0; step < kSteps; ++step) {
        const auto rows = step == 0 ? NegInfRows::None : (step % 2 == 0 ? NegInfRows::EveryThird : NegInfRows::None);
        const auto in = make_inputs(shape, 100U + step, rows, false);
        const DeviceInputs dev = to_device(in);
        std::tie(ref_accum, ref_lse) = composite_merge(ref_accum, ref_lse, dev.step_output, dev.step_intermediates);
        ttml::metal::ring_lse_merge(fused.accum, fused.running, dev.step_output, dev.step_intermediates);
    }
    const auto accum_err = error_stats(ttml::core::to_xtensor(fused.accum), ttml::core::to_xtensor(ref_accum));
    const auto lse_err = error_stats(col0(ttml::core::to_xtensor(fused.running)), ttml::core::to_xtensor(ref_lse));
    report("RingSequence(8 steps) accum vs composite", accum_err);
    report("RingSequence(8 steps) lse vs composite", lse_err);
    EXPECT_EQ(accum_err.non_finite_mismatch, 0U);
    EXPECT_LE(accum_err.max_abs, 8 * kCompositeAtol);
    EXPECT_LE(lse_err.max_abs, 8 * kCompositeAtol);
}

TEST_F(RingLseMergeTest, AllNegInfStepIsNoOp) {
    const auto in = make_inputs({1U, 2U, 64U, 64U}, 3U, NegInfRows::All, false);
    const DeviceInputs dev = to_device(in);
    ttml::metal::ring_lse_merge(dev.accum, dev.running, dev.step_output, dev.step_intermediates);
    EXPECT_EQ(ttml::test_utils::count_bit_mismatches(in.accum, ttml::core::to_xtensor(dev.accum)), 0U);
    EXPECT_EQ(ttml::test_utils::count_bit_mismatches(in.running, ttml::core::to_xtensor(dev.running)), 0U);
}

// ============================================================================
// Program-cache hit
// ============================================================================
// The second launch on new buffers with the same specs must reuse the cached program, read and write its own
// buffers, and leave the first launch's tensors alone.

class RingLseMergeCacheHitTest : public ttml::test_utils::ProgramCacheSuiteTest {};

TEST_F(RingLseMergeCacheHitTest, SecondCallHitsAndIsCorrect) {
    using ttml::test_utils::CacheExpectation;
    const Shape4 shape{1U, 3U, 96U, 128U};

    const auto in1 = make_inputs(shape, 21U, NegInfRows::EveryThird, false);
    const auto in2 = make_inputs(shape, 42U, NegInfRows::None, false);
    const DeviceInputs dev1 = to_device(in1);
    const DeviceInputs dev2 = to_device(in2);
    ttml::test_utils::expect_new_buffers(
        {dev1.accum, dev1.running, dev1.step_output, dev1.step_intermediates},
        {dev2.accum, dev2.running, dev2.step_output, dev2.step_intermediates},
        "second launch");

    auto entries = ttml::autograd::ctx().get_device().num_program_cache_entries();
    ttml::metal::ring_lse_merge(dev1.accum, dev1.running, dev1.step_output, dev1.step_intermediates);
    ttml::test_utils::expect_cache(entries, CacheExpectation::Miss, "first launch");
    const auto accum1 = ttml::core::to_xtensor(dev1.accum);
    const auto running1 = ttml::core::to_xtensor(dev1.running);

    entries = ttml::autograd::ctx().get_device().num_program_cache_entries();
    ttml::metal::ring_lse_merge(dev2.accum, dev2.running, dev2.step_output, dev2.step_intermediates);
    ttml::test_utils::expect_cache(entries, CacheExpectation::Hit, "second launch");

    ttml::test_utils::expect_unchanged(dev1.accum, accum1, "first accumulator");
    ttml::test_utils::expect_unchanged(dev1.running, running1, "first running lse");

    const auto [host_accum1, host_lse1] = host_merge(in1);
    const auto [host_accum2, host_lse2] = host_merge(in2);
    const auto err1 = error_stats(accum1, host_accum1);
    const auto err2 = error_stats(ttml::core::to_xtensor(dev2.accum), host_accum2);
    const auto lse_err2 = error_stats(col0(ttml::core::to_xtensor(dev2.running)), host_lse2);
    report("CacheHit first launch accum vs float64", err1);
    report("CacheHit second launch accum vs float64", err2);
    EXPECT_LE(err1.max_abs, kHostAtol);
    EXPECT_LE(err2.max_abs, kHostAtol);
    EXPECT_LE(lse_err2.max_abs, kHostAtol);
}

TEST_F(RingLseMergeTest, RejectsNonDramInputs) {
    const auto in = make_inputs({1U, 1U, 32U, 64U}, 1U, NegInfRows::None, false);
    const DeviceInputs dev = to_device(in);
    const auto l1_running = ttnn::to_memory_config(dev.running, ttnn::L1_MEMORY_CONFIG);
    EXPECT_ANY_THROW(ttml::metal::ring_lse_merge(dev.accum, l1_running, dev.step_output, dev.step_intermediates));
}
