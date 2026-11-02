// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "core/tt_tensor_utils.hpp"
#include "metal/optimizers/adamw/adamw.hpp"
#include "test_utils/program_cache.hpp"
#include "ttnn/tensor/tensor.hpp"
#include "xtensor/containers/xarray.hpp"

// Launch tensors and checks shared by the AdamW and AdamWFullPrecision tests.

namespace ttml::test_utils::adamw {

// The mean and max absolute differences of `actual` from `reference`; both non-finite if any difference is.
struct ErrorMetrics {
    float mean_error;
    float max_error;
};

inline ErrorMetrics error_metrics(const xt::xarray<float>& reference, const xt::xarray<float>& actual) {
    float sum_error = 0.0F;
    float max_error = 0.0F;
    for (std::size_t i = 0; i < reference.size(); ++i) {
        const float error = std::abs(reference.flat(i) - actual.flat(i));
        if (!std::isfinite(error)) {
            return {error, error};  // std::max would skip a NaN
        }
        sum_error += error;
        max_error = std::max(max_error, error);
    }
    return {sum_error / static_cast<float>(reference.size()), max_error};
}

// The arguments of ttml::metal::adamw that are left out of the program hash. The optimizers keep the beta powers apart
// from the betas, so each is a value of its own that a cache hit could keep.
struct AdamWStepScalars {
    float lr;
    float beta1;
    float beta2;
    float beta1_pow;
    float beta2_pow;
    float epsilon;
    float weight_decay;

    bool operator==(const AdamWStepScalars&) const = default;
};

// The scalars of the step after `steps` completed ones, with beta^(steps + 1) as the beta powers.
inline AdamWStepScalars step_scalars(
    float lr, float beta1, float beta2, size_t steps, float epsilon, float weight_decay) {
    const auto t = static_cast<float>(steps + 1U);
    return {lr, beta1, beta2, std::pow(beta1, t), std::pow(beta2, t), epsilon, weight_decay};
}

// `scalars` with one of them set to its value in `cached`, named, for each scalar the two differ in: what a hit that
// kept that one value of the cached program would compute with.
inline std::vector<std::pair<const char*, AdamWStepScalars>> stale_scalar_variants(
    const AdamWStepScalars& scalars, const AdamWStepScalars& cached) {
    auto with = [&](auto take_cached) {
        auto stale = scalars;
        take_cached(stale);
        return stale;
    };
    const std::array<std::pair<const char*, AdamWStepScalars>, 7> variants = {{
        {"lr", with([&](auto& s) { s.lr = cached.lr; })},
        {"beta1", with([&](auto& s) { s.beta1 = cached.beta1; })},
        {"beta2", with([&](auto& s) { s.beta2 = cached.beta2; })},
        {"beta1_pow", with([&](auto& s) { s.beta1_pow = cached.beta1_pow; })},
        {"beta2_pow", with([&](auto& s) { s.beta2_pow = cached.beta2_pow; })},
        {"epsilon", with([&](auto& s) { s.epsilon = cached.epsilon; })},
        {"weight_decay", with([&](auto& s) { s.weight_decay = cached.weight_decay; })},
    }};
    std::vector<std::pair<const char*, AdamWStepScalars>> changed;
    for (const auto& variant : variants) {
        if (variant.second != scalars) {
            changed.push_back(variant);
        }
    }
    return changed;
}

// A device tensor and the values the kernels read from it.
struct StoredTensor {
    ttnn::Tensor tensor;
    xt::xarray<float> before;
};

inline StoredTensor store(ttnn::Tensor tensor) {
    auto before = ttml::core::to_xtensor(tensor);
    return {std::move(tensor), std::move(before)};
}

// The tensors of one AdamW launch.
struct AdamWLaunchTensors {
    StoredTensor param;
    StoredTensor grad;
    StoredTensor exp_avg;
    StoredTensor exp_avg_sq;
    std::optional<StoredTensor> max_exp_avg_sq;  // AMSGrad only
};

inline ttnn::Tensor launch_adamw(
    const AdamWLaunchTensors& tensors,
    const AdamWStepScalars& scalars,
    ttml::metal::StochasticRounding stochastic_rounding = ttml::metal::StochasticRounding::Disabled,
    std::optional<uint32_t> stochastic_rounding_seed = std::nullopt) {
    std::optional<ttnn::Tensor> max_exp_avg_sq;
    if (tensors.max_exp_avg_sq.has_value()) {
        max_exp_avg_sq = tensors.max_exp_avg_sq->tensor;
    }
    return ttml::metal::adamw(
        tensors.param.tensor,
        tensors.grad.tensor,
        tensors.exp_avg.tensor,
        tensors.exp_avg_sq.tensor,
        max_exp_avg_sq,
        scalars.lr,
        scalars.beta1,
        scalars.beta2,
        scalars.beta1_pow,
        scalars.beta2_pow,
        scalars.epsilon,
        scalars.weight_decay,
        stochastic_rounding,
        stochastic_rounding_seed);
}

// Every tensor of a launch, as read back from the device or as a CPU reference computes it.
struct AdamWValues {
    xt::xarray<float> param;
    xt::xarray<float> grad;
    xt::xarray<float> exp_avg;
    xt::xarray<float> exp_avg_sq;
    std::optional<xt::xarray<float>> max_exp_avg_sq;  // AMSGrad only
};

inline AdamWValues read_back(const AdamWLaunchTensors& tensors) {
    AdamWValues values{
        ttml::core::to_xtensor(tensors.param.tensor),
        ttml::core::to_xtensor(tensors.grad.tensor),
        ttml::core::to_xtensor(tensors.exp_avg.tensor),
        ttml::core::to_xtensor(tensors.exp_avg_sq.tensor),
        std::nullopt};
    if (tensors.max_exp_avg_sq.has_value()) {
        values.max_exp_avg_sq = ttml::core::to_xtensor(tensors.max_exp_avg_sq->tensor);
    }
    return values;
}

inline void expect_bitwise_equal(const AdamWValues& expected, const AdamWValues& actual, const std::string& label) {
    EXPECT_EQ(count_bit_mismatches(expected.param, actual.param), 0U) << label << ": param";
    EXPECT_EQ(count_bit_mismatches(expected.grad, actual.grad), 0U) << label << ": grad";
    EXPECT_EQ(count_bit_mismatches(expected.exp_avg, actual.exp_avg), 0U) << label << ": exp_avg";
    EXPECT_EQ(count_bit_mismatches(expected.exp_avg_sq, actual.exp_avg_sq), 0U) << label << ": exp_avg_sq";
    ASSERT_EQ(expected.max_exp_avg_sq.has_value(), actual.max_exp_avg_sq.has_value()) << label;
    if (expected.max_exp_avg_sq.has_value()) {
        EXPECT_EQ(count_bit_mismatches(*expected.max_exp_avg_sq, *actual.max_exp_avg_sq), 0U)
            << label << ": max_exp_avg_sq";
    }
}

inline void expect_distinct_buffers(const AdamWLaunchTensors& a, const AdamWLaunchTensors& b) {
    EXPECT_NE(a.param.tensor.buffer()->address(), b.param.tensor.buffer()->address());
    EXPECT_NE(a.grad.tensor.buffer()->address(), b.grad.tensor.buffer()->address());
    EXPECT_NE(a.exp_avg.tensor.buffer()->address(), b.exp_avg.tensor.buffer()->address());
    EXPECT_NE(a.exp_avg_sq.tensor.buffer()->address(), b.exp_avg_sq.tensor.buffer()->address());
    if (a.max_exp_avg_sq.has_value() && b.max_exp_avg_sq.has_value()) {
        EXPECT_NE(a.max_exp_avg_sq->tensor.buffer()->address(), b.max_exp_avg_sq->tensor.buffer()->address());
    }
}

}  // namespace ttml::test_utils::adamw
