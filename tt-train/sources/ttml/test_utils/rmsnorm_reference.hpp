// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cmath>
#include <cstdint>
#include <tt-metalium/bfloat16.hpp>

#include "xtensor/containers/xarray.hpp"
#include "xtensor/core/xmath.hpp"

namespace ttml::test_utils {

// Round-to-nearest-even to bfloat16, which is what the device holds after from_xtensor.
inline xt::xarray<float> round_to_bf16(const xt::xarray<float>& values) {
    xt::xarray<float> rounded = values;
    for (auto& value : rounded) {
        value = static_cast<float>(bfloat16(value));
    }
    return rounded;
}

struct RMSNormReference {
    xt::xarray<float> out;
    xt::xarray<float> dx;
    xt::xarray<float> dgamma;
    // Per-row sum(x * gamma * dy) over the last dim, the value LocalSum computes for a shard.
    xt::xarray<float> local_sum;
};

// RMSNorm over the last dim of [B, N, S, C] in fp32, on bf16-rounded inputs so that a comparison measures only the
// device arithmetic and not input rounding.
inline RMSNormReference rmsnorm_reference(
    const xt::xarray<float>& x_values,
    const xt::xarray<float>& gamma_values,
    const xt::xarray<float>& dy_values,
    float epsilon) {
    const xt::xarray<float> x = round_to_bf16(x_values);
    const xt::xarray<float> gamma = round_to_bf16(gamma_values);
    const xt::xarray<float> dy = round_to_bf16(dy_values);
    const xt::xarray<float> rms = xt::sqrt(xt::mean(x * x, {3}, xt::keep_dims) + epsilon);
    const xt::xarray<float> gained = gamma * dy / rms;
    const xt::xarray<float> scale = xt::mean(x * gained, {3}, xt::keep_dims);
    return RMSNormReference{
        .out = gamma * x / rms,
        .dx = gained - x * scale / (rms * rms),
        .dgamma = xt::sum(dy * x / rms, {0, 1, 2}, xt::keep_dims),
        .local_sum = xt::sum(x * gamma * dy, {3}, xt::keep_dims),
    };
}

// dgamma sums num_rows independent per-row errors, so its absolute error grows with sqrt(num_rows).
inline float dgamma_atol(uint32_t num_rows, float per_row_atol) {
    return per_row_atol * std::sqrt(static_cast<float>(num_rows));
}

}  // namespace ttml::test_utils
