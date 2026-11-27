// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// The SFPU activations the Compute API's vec_ calls name, the forms a function can have and the regions of a tile it
// can be run on. Arch-independent: the Compute API (vec.h) names a function, each arch's Resolver says which forms and
// regions it has there and puts them in the Plan (MathPlan::vec, vec_form, vec_region), and each arch's llk-lib picks
// the SFPI code for them.

#pragma once

#include <cstdint>

namespace llk2
{

/// One SFPU activation; None is no function.
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
    Softplus,    ///< log(1 + exp(beta x)) / beta, and x itself above threshold
    Exp,         ///< exp(input_scale x)
    Reciprocal   ///< 1 / x
};

/// Which of a function's algorithms runs. Every function has Exact; the Resolver says which others it has.
enum class VecForm : std::uint32_t
{
    Exact,     ///< the function's accurate form
    Fast,      ///< its approximate form (LLK 1.0's fast_and_approx / APPROX template argument)
    Polynomial ///< range reduction and a polynomial, a few instructions per iteration with no lookup table or replayed body
};

/// Which datums of each tile a function rewrites; the others keep their values.
enum class VecRegion : std::uint32_t
{
    All,                 ///< every datum
    LeftFaces,           ///< faces 0 and 2: tile columns 0..15, every row (LLK 1.0's VectorMode::C)
    LeftFacesEvenColumns ///< the even columns of faces 0 and 2: half the iterations of LeftFaces
};

} // namespace llk2
