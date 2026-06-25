// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cmath>
#include <algorithm>
#include <string>

namespace activation_reference {

// Reference implementations of activation functions for error calculation
// These use high-precision float/double computation as ground truth

inline float sigmoid(float x) { return 1.0f / (1.0f + std::exp(-x)); }

inline float tanh_fn(float x) { return std::tanh(x); }

inline float relu(float x) { return std::max(0.0f, x); }

inline float gelu(float x) {
    // GELU(x) = x * Φ(x) where Φ is the CDF of standard normal distribution
    // Approximation: GELU(x) ≈ 0.5 * x * (1 + tanh(√(2/π) * (x + 0.044715 * x^3)))
    constexpr float sqrt_2_over_pi = 0.7978845608f;
    constexpr float coeff = 0.044715f;
    float x_cubed = x * x * x;
    return 0.5f * x * (1.0f + std::tanh(sqrt_2_over_pi * (x + coeff * x_cubed)));
}

inline float swish(float x) {
    // Swish(x) = x * sigmoid(x)
    return x * sigmoid(x);
}

inline float elu(float x, float alpha = 1.0f) { return x >= 0.0f ? x : alpha * (std::exp(x) - 1.0f); }

inline float selu(float x) {
    constexpr float alpha = 1.67326324f;
    constexpr float scale = 1.05070098f;
    return scale * (x >= 0.0f ? x : alpha * (std::exp(x) - 1.0f));
}

inline float leaky_relu(float x, float alpha = 0.01f) { return x >= 0.0f ? x : alpha * x; }

inline float prelu(float x, float alpha = 0.25f) { return x >= 0.0f ? x : alpha * x; }

inline float softplus(float x) {
    // softplus(x) = log(1 + exp(x))
    // Numerically stable version
    if (x > 20.0f) {
        return x;
    }
    return std::log(1.0f + std::exp(x));
}

inline float softsign(float x) { return x / (1.0f + std::abs(x)); }

inline float mish(float x) {
    // Mish(x) = x * tanh(softplus(x))
    return x * std::tanh(softplus(x));
}

inline float hardtanh(float x, float min_val = -1.0f, float max_val = 1.0f) {
    return std::max(min_val, std::min(max_val, x));
}

inline float hardsigmoid(float x) {
    // hardsigmoid(x) = max(0, min(1, (x + 3) / 6))
    return std::max(0.0f, std::min(1.0f, (x + 3.0f) / 6.0f));
}

inline float hardswish(float x) {
    // hardswish(x) = x * hardsigmoid(x)
    return x * hardsigmoid(x);
}

inline float relu6(float x) { return std::max(0.0f, std::min(6.0f, x)); }

inline float celu(float x, float alpha = 1.0f) { return x >= 0.0f ? x : alpha * (std::exp(x / alpha) - 1.0f); }

inline float threshold(float x, float threshold = 0.0f, float value = 0.0f) { return x > threshold ? x : value; }

inline float softshrink(float x, float lambda = 0.5f) {
    if (x > lambda) {
        return x - lambda;
    }
    if (x < -lambda) {
        return x + lambda;
    }
    return 0.0f;
}

inline float hardshrink(float x, float lambda = 0.5f) {
    if (std::abs(x) > lambda) {
        return x;
    }
    return 0.0f;
}

inline float tanhshrink(float x) { return x - std::tanh(x); }

inline float logsigmoid(float x) {
    // logsigmoid(x) = log(sigmoid(x))
    // Numerically stable version
    if (x >= 0.0f) {
        return -std::log(1.0f + std::exp(-x));
    } else {
        return x - std::log(1.0f + std::exp(x));
    }
}

inline float exp_fn(float x) { return std::exp(x); }

inline float sin_fn(float x) { return std::sin(x); }

inline float cos_fn(float x) { return std::cos(x); }

inline float sinh_fn(float x) { return std::sinh(x); }

inline float cosh_fn(float x) { return std::cosh(x); }

inline float atanh_fn(float x) { return std::atanh(x); }

inline float erf_fn(float x) { return std::erf(x); }

// Dispatch function that calls the appropriate activation based on name
inline float compute_reference(const std::string& activation, float x) {
    if (activation == "sigmoid") {
        return sigmoid(x);
    }
    if (activation == "tanh") {
        return tanh_fn(x);
    }
    if (activation == "relu") {
        return relu(x);
    }
    if (activation == "gelu") {
        return gelu(x);
    }
    if (activation == "swish") {
        return swish(x);
    }
    if (activation == "elu") {
        return elu(x);
    }
    if (activation == "selu") {
        return selu(x);
    }
    if (activation == "leaky_relu") {
        return leaky_relu(x);
    }
    if (activation == "prelu") {
        return prelu(x);
    }
    if (activation == "softplus") {
        return softplus(x);
    }
    if (activation == "softsign") {
        return softsign(x);
    }
    if (activation == "mish") {
        return mish(x);
    }
    if (activation == "hardtanh") {
        return hardtanh(x);
    }
    if (activation == "hardsigmoid") {
        return hardsigmoid(x);
    }
    if (activation == "hardswish") {
        return hardswish(x);
    }
    if (activation == "relu6") {
        return relu6(x);
    }
    if (activation == "celu") {
        return celu(x);
    }
    if (activation == "threshold") {
        return threshold(x);
    }
    if (activation == "softshrink") {
        return softshrink(x);
    }
    if (activation == "hardshrink") {
        return hardshrink(x);
    }
    if (activation == "tanhshrink") {
        return tanhshrink(x);
    }
    if (activation == "logsigmoid") {
        return logsigmoid(x);
    }
    if (activation == "exp") {
        return exp_fn(x);
    }
    if (activation == "sin") {
        return sin_fn(x);
    }
    if (activation == "cos") {
        return cos_fn(x);
    }
    if (activation == "sinh") {
        return sinh_fn(x);
    }
    if (activation == "cosh") {
        return cosh_fn(x);
    }
    if (activation == "atanh") {
        return atanh_fn(x);
    }
    if (activation == "erf") {
        return erf_fn(x);
    }

    // Default: return x (identity)
    return x;
}

}  // namespace activation_reference
