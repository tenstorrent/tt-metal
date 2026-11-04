// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Single-device tests for the online-softmax merge that ring attention runs after every ring
// step (ttml::ops::distributed::detail::merge_partial_into), against a float64 host reference.
// Covers the "no contribution" lse of -inf on either side, lse values large enough that a naive
// log(exp(a) + exp(b)) overflows FP32, large gaps between the two lse values, and a full ring of
// merges compared with the direct softmax-weighted combination.

#include <fmt/format.h>
#include <gtest/gtest.h>

#include <array>
#include <cmath>
#include <core/xtensor_utils.hpp>
#include <cstdint>
#include <limits>
#include <random>
#include <vector>

#include "autograd/auto_context.hpp"
#include "core/tt_tensor_utils.hpp"
#include "ops/distributed/ring_attention_sdpa.hpp"
#include "test_utils/random_data.hpp"

namespace {

constexpr float kNegInf = -std::numeric_limits<float>::infinity();
constexpr uint32_t kIntermediatesWidth = 32U;

// Row kinds, cycled over the (B, H, S) rows so every tile holds all of them.
enum RowKind : uint32_t {
    kFirstMerge = 0,      // running lse = -inf, accumulator = 0 (the first ring step)
    kSkippedStep = 1,     // step lse = -inf (a step the causal schedule skips)
    kLargeLse = 2,        // both lse ~ 100..200: exp() of either overflows FP32
    kStepNegligible = 3,  // step lse 40 below the running one
    kStepDominates = 4,   // step lse 40 above the running one
    kNumRowKinds = 8,     // remaining kinds: both lse uniform in [-5, 15]
};

struct MergeInputs {
    xt::xarray<float> accum;          // (B, H, S, D)
    xt::xarray<float> running_lse;    // (B, H, S, 1)
    xt::xarray<float> step_output;    // (B, H, S, D), bf16-representable
    xt::xarray<float> intermediates;  // (B, H, S, 32), lse in column 0
};

struct DeviceMergeInputs {
    ttnn::Tensor accum;
    ttnn::Tensor running_lse;
    ttnn::Tensor step_output;
    ttnn::Tensor intermediates;
};

ttnn::Tensor to_device_fp32(const xt::xarray<float>& x) {
    return ttml::core::from_xtensor<float, ttnn::DataType::FLOAT32>(x, &ttml::autograd::ctx().get_device());
}

ttnn::Tensor to_device_bf16(const xt::xarray<float>& x) {
    return ttml::core::from_xtensor<float, ttnn::DataType::BFLOAT16>(x, &ttml::autograd::ctx().get_device());
}

MergeInputs make_inputs(uint32_t B, uint32_t H, uint32_t S, uint32_t D, uint32_t seed) {
    using ttml::test_utils::make_uniform_xarray;
    const std::array<size_t, 4> full_shape{B, H, S, D};
    const std::array<size_t, 4> lse_shape{B, H, S, 1};
    const std::array<size_t, 4> inter_shape{B, H, S, kIntermediatesWidth};
    MergeInputs in{
        .accum = make_uniform_xarray<float>(full_shape, -1.0F, 1.0F, seed),
        .running_lse = make_uniform_xarray<float>(lse_shape, -5.0F, 15.0F, seed + 1U),
        .step_output = make_uniform_xarray<float>(full_shape, -1.0F, 1.0F, seed + 2U),
        // Columns 1..31 are padding the merge must ignore; fill them with large junk.
        .intermediates = make_uniform_xarray<float>(inter_shape, -1.0e6F, 1.0e6F, seed + 3U),
    };
    const auto step_lse = make_uniform_xarray<float>(lse_shape, -5.0F, 15.0F, seed + 4U);
    const auto large = make_uniform_xarray<float>(lse_shape, 100.0F, 200.0F, seed + 5U);
    for (uint32_t b = 0; b < B; ++b) {
        for (uint32_t h = 0; h < H; ++h) {
            for (uint32_t s = 0; s < S; ++s) {
                float& running = in.running_lse(b, h, s, 0);
                float& step = in.intermediates(b, h, s, 0);
                step = step_lse(b, h, s, 0);
                switch (s % kNumRowKinds) {
                    case kFirstMerge:
                        running = kNegInf;
                        for (uint32_t d = 0; d < D; ++d) {
                            in.accum(b, h, s, d) = 0.0F;
                        }
                        break;
                    case kSkippedStep: step = kNegInf; break;
                    case kLargeLse:
                        running = large(b, h, s, 0);
                        step = running + (step_lse(b, h, s, 0) - 5.0F) * 0.5F;
                        break;
                    case kStepNegligible: step = running - 40.0F; break;
                    case kStepDominates: step = running + 40.0F; break;
                    default: break;
                }
            }
        }
    }
    return in;
}

// Uploads the inputs and replaces the host step output with the bf16 values the device holds, so
// the reference sees exactly what the merge sees.
DeviceMergeInputs upload(MergeInputs& in) {
    DeviceMergeInputs dev{
        .accum = to_device_fp32(in.accum),
        .running_lse = to_device_fp32(in.running_lse),
        .step_output = to_device_bf16(in.step_output),
        .intermediates = to_device_fp32(in.intermediates),
    };
    in.step_output = ttml::core::to_xtensor<float>(dev.step_output);
    return dev;
}

double logaddexp(double a, double b) {
    if (std::isinf(a) && a < 0.0) {
        return b;
    }
    if (std::isinf(b) && b < 0.0) {
        return a;
    }
    return std::max(a, b) + std::log1p(std::exp(-std::abs(a - b)));
}

struct MergeResult {
    xt::xarray<double> accum;
    xt::xarray<double> lse;
};

// float64 reference of one merge: O' = O * exp(L - L') + O_i * exp(l_i - L'), L' = logaddexp(L, l_i),
// with exp(-inf - L') = 0.
MergeResult reference_merge(const MergeInputs& in) {
    const auto& shape = in.accum.shape();
    const size_t B = shape[0], H = shape[1], S = shape[2], D = shape[3];
    MergeResult ref{xt::zeros<double>({B, H, S, D}), xt::zeros<double>({B, H, S, size_t{1}})};
    for (size_t b = 0; b < B; ++b) {
        for (size_t h = 0; h < H; ++h) {
            for (size_t s = 0; s < S; ++s) {
                const double running = in.running_lse(b, h, s, 0);
                const double step = in.intermediates(b, h, s, 0);
                const double merged = logaddexp(running, step);
                const double old_weight = std::isinf(running) ? 0.0 : std::exp(running - merged);
                const double new_weight = std::isinf(step) ? 0.0 : std::exp(step - merged);
                ref.lse(b, h, s, 0) = merged;
                for (size_t d = 0; d < D; ++d) {
                    ref.accum(b, h, s, d) = static_cast<double>(in.accum(b, h, s, d)) * old_weight +
                                            static_cast<double>(in.step_output(b, h, s, d)) * new_weight;
                }
            }
        }
    }
    return ref;
}

struct ErrorStats {
    double max_abs = 0.0;
    double max_rel = 0.0;  // |a - e| / max(|e|, 1)
};

ErrorStats compare(const xt::xarray<float>& actual, const xt::xarray<double>& expected) {
    EXPECT_EQ(actual.size(), expected.size());
    ErrorStats stats;
    for (size_t i = 0; i < expected.size(); ++i) {
        const double a = actual.flat(i);
        const double e = expected.flat(i);
        EXPECT_TRUE(std::isfinite(a)) << "non-finite result at flat index " << i << ": " << a;
        const double err = std::abs(a - e);
        stats.max_abs = std::max(stats.max_abs, err);
        stats.max_rel = std::max(stats.max_rel, err / std::max(std::abs(e), 1.0));
    }
    return stats;
}

// Largest |actual - expected| / bound over the accumulator. The merge weight exp(l_i - L') inherits
// the absolute FP32 rounding of the lse values, which grows with their magnitude (ulp(200) ~ 1.5e-5),
// so the bound for a row is a few FP32 ulps of O plus a few ulps of that row's lse.
double accum_error_over_bound(const xt::xarray<float>& actual, const MergeResult& ref) {
    const auto& shape = ref.accum.shape();
    const size_t B = shape[0], H = shape[1], S = shape[2], D = shape[3];
    double worst = 0.0;
    for (size_t b = 0; b < B; ++b) {
        for (size_t h = 0; h < H; ++h) {
            for (size_t s = 0; s < S; ++s) {
                const double bound =
                    1.0e-6 + 4.0 * std::numeric_limits<float>::epsilon() * std::max(1.0, std::abs(ref.lse(b, h, s, 0)));
                for (size_t d = 0; d < D; ++d) {
                    worst = std::max(worst, std::abs(actual(b, h, s, d) - ref.accum(b, h, s, d)) / bound);
                }
            }
        }
    }
    return worst;
}

}  // namespace

class RingAttentionMergeTest : public ::testing::Test {
protected:
    static void SetUpTestSuite() {
        ttml::autograd::ctx().open_device();
    }

    static void TearDownTestSuite() {
        ttml::autograd::ctx().close_device();
    }
};

// One merge over rows of every kind, against the float64 reference.
TEST_F(RingAttentionMergeTest, SingleMergeMatchesReference) {
    constexpr uint32_t B = 1U, H = 2U, S = 128U, D = 128U;
    MergeInputs in = make_inputs(B, H, S, D, /*seed=*/7U);
    DeviceMergeInputs dev = upload(in);
    const MergeResult ref = reference_merge(in);

    ttml::ops::distributed::detail::merge_partial_into(dev.accum, dev.running_lse, dev.step_output, dev.intermediates);
    ASSERT_EQ(dev.accum.dtype(), ttnn::DataType::FLOAT32);
    ASSERT_EQ(dev.running_lse.dtype(), ttnn::DataType::FLOAT32);
    ASSERT_EQ(dev.accum.logical_shape(), ttnn::Shape({B, H, S, D}));
    ASSERT_EQ(dev.running_lse.logical_shape(), ttnn::Shape({B, H, S, 1U}));

    const auto accum = ttml::core::to_xtensor<float>(dev.accum);
    const auto lse = ttml::core::to_xtensor<float>(dev.running_lse);
    const ErrorStats accum_err = compare(accum, ref.accum);
    const ErrorStats lse_err = compare(lse, ref.lse);
    const double accum_over_bound = accum_error_over_bound(accum, ref);
    fmt::print(
        "[merge] accum max_abs {:.3e} max_rel {:.3e} (error/bound {:.2f}) | lse max_abs {:.3e} max_rel {:.3e}\n",
        accum_err.max_abs,
        accum_err.max_rel,
        accum_over_bound,
        lse_err.max_abs,
        lse_err.max_rel);
    // lse up to ~200 here: both errors are measured relative to its magnitude.
    EXPECT_LE(accum_over_bound, 1.0);
    EXPECT_LT(lse_err.max_rel, 1.0e-6);

    // "No contribution" rows must be passed through untouched, bit for bit.
    for (uint32_t h = 0; h < H; ++h) {
        for (uint32_t s = kSkippedStep; s < S; s += kNumRowKinds) {
            ASSERT_EQ(lse(0, h, s, 0), in.running_lse(0, h, s, 0)) << "h " << h << " s " << s;
            for (uint32_t d = 0; d < D; ++d) {
                ASSERT_EQ(accum(0, h, s, d), in.accum(0, h, s, d)) << "h " << h << " s " << s << " d " << d;
            }
        }
    }
}

// A full ring of merges starting from (O = 0, L = -inf), some steps skipped, against the direct
// combination O = sum_i exp(l_i - L) * O_i with L = logsumexp_i(l_i).
TEST_F(RingAttentionMergeTest, RingOfMergesMatchesDirectCombination) {
    constexpr uint32_t B = 1U, H = 2U, S = 64U, D = 128U, kSteps = 8U;
    const std::array<size_t, 4> full_shape{B, H, S, D};
    const std::array<size_t, 4> lse_shape{B, H, S, 1};

    ttnn::Tensor accum = to_device_fp32(xt::zeros<float>(full_shape));
    ttnn::Tensor running_lse = to_device_fp32(xt::xarray<float>(xt::ones<float>(lse_shape) * kNegInf));

    std::vector<xt::xarray<float>> step_outputs;
    std::vector<xt::xarray<float>> step_lses;
    for (uint32_t step = 0; step < kSteps; ++step) {
        MergeInputs in = make_inputs(B, H, S, D, /*seed=*/100U + 10U * step);
        // Only the first step contributes to every row; later steps skip a row with
        // probability ~1/4, the way the causal schedule skips whole steps per device.
        auto skip = ttml::test_utils::make_uniform_xarray<float>(lse_shape, 0.0F, 1.0F, 1000U + step);
        auto step_lse = ttml::test_utils::make_uniform_xarray<float>(lse_shape, -5.0F, 15.0F, 2000U + step);
        for (size_t i = 0; i < step_lse.size(); ++i) {
            if (step > 0U && skip.flat(i) < 0.25F) {
                step_lse.flat(i) = kNegInf;
            }
        }
        xt::view(in.intermediates, xt::all(), xt::all(), xt::all(), xt::range(0, 1)) = step_lse;
        DeviceMergeInputs dev = upload(in);
        step_outputs.push_back(in.step_output);
        step_lses.push_back(step_lse);
        ttml::ops::distributed::detail::merge_partial_into(accum, running_lse, dev.step_output, dev.intermediates);
    }

    xt::xarray<double> ref_accum = xt::zeros<double>(full_shape);
    xt::xarray<double> ref_lse = xt::zeros<double>(lse_shape);
    for (uint32_t h = 0; h < H; ++h) {
        for (uint32_t s = 0; s < S; ++s) {
            double total = -std::numeric_limits<double>::infinity();
            for (uint32_t step = 0; step < kSteps; ++step) {
                total = logaddexp(total, step_lses[step](0, h, s, 0));
            }
            ref_lse(0, h, s, 0) = total;
            for (uint32_t step = 0; step < kSteps; ++step) {
                const double l = step_lses[step](0, h, s, 0);
                if (std::isinf(l)) {
                    continue;
                }
                const double w = std::exp(l - total);
                for (uint32_t d = 0; d < D; ++d) {
                    ref_accum(0, h, s, d) += w * step_outputs[step](0, h, s, d);
                }
            }
        }
    }

    const ErrorStats accum_err = compare(ttml::core::to_xtensor<float>(accum), ref_accum);
    const ErrorStats lse_err = compare(ttml::core::to_xtensor<float>(running_lse), ref_lse);
    fmt::print(
        "[ring of {} merges] accum max_abs {:.3e} max_rel {:.3e} | lse max_abs {:.3e} max_rel {:.3e}\n",
        kSteps,
        accum_err.max_abs,
        accum_err.max_rel,
        lse_err.max_abs,
        lse_err.max_rel);
    // Eight FP32 merges with lse in [-5, 15]: a few ulps of drift at most.
    EXPECT_LT(accum_err.max_abs, 1.0e-5);
    EXPECT_LT(lse_err.max_rel, 1.0e-6);
}
