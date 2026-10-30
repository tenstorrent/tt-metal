// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// The SFPU activations the Compute API's vec_ calls name, and what the vec Resolver takes in place of a plain
// modifier struct. Arch-independent: the Compute API (vec.h) names a function, each arch's Resolver puts it in
// MathPlan.function, and each arch's llk-lib picks the SFPI code for it.

#pragma once

#include <cstdint>

#include "compute_api/modifiers.h"

namespace llk2
{

/// One SFPU activation. The value is what MathPlan.function holds; 0 is no function.
enum class VecFunction : std::uint32_t
{
    None,
    Silu,        ///< x * sigmoid(x)
    Tanh,        ///< tanh(x)
    Gelu,        ///< x * Phi(x), the erf form
    GeluTanh,    ///< 0.5 x (1 + tanh(sqrt(2 / pi) (x + 0.044715 x^3)))
    ReluMax,     ///< min(max(x, 0), max): relu6 with max = 6
    Sigmoid,     ///< 1 / (1 + exp(-x))
    Hardsigmoid, ///< min(max(x / 6 + 1 / 2, 0), 1)
    Hardtanh,    ///< min(max(x, min), max)
    Selu,        ///< scale * (x > 0 ? x : alpha * (exp(x) - 1))
    Softplus     ///< log(1 + exp(beta x)) / beta, and x itself above threshold
};

/// What a vec_ call hands the Resolver: the function it names and the caller's modifiers.
struct vec_call
{
    VecFunction function; ///< which activation
    vec_modifiers mods;   ///< the caller's modifiers
};

} // namespace llk2
