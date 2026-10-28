// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Compute API: the SFPU activations. Single thread, arch-independent. Every TRISC runs these same calls; llk-lib's
// per-thread call executes only on the thread that drives the SFPU.
//
//   vec_silu(acc);                                   // acc = silu(acc), every tile, in place
//   vec_tanh<vec_modifiers{.fast = true}>(acc);      // the approximate form
//   vec_relu_max(acc, max);                          // scalars are FP32 bit patterns, as today's *_tile calls take
//
// The calls, with their scalars:
//   vec_silu(acc)                vec_tanh(acc)               vec_gelu(acc)              vec_gelu_tanh(acc)
//   vec_sigmoid(acc)             vec_hardsigmoid(acc)        vec_relu_max(acc, max)     vec_hardtanh(acc, min, max)
//   vec_selu(acc, scale, alpha)  vec_softplus(acc, beta, beta_reciprocal, threshold)
//
// acc is a Dest tensor the caller holds (from the Dest sync API). Which thread may touch it when is the sync API's
// job, not the op's. vec_modifiers::thread picks the thread that drives the SFPU: Thread::Math (the default) runs
// it right after the FPU wrote Dest; Thread::Pack runs it after math has handed Dest to the packer, as the target
// matmul kernel does (llk_lib/vec.h describes the hand-over). Each call resolves itself as a chain of one
// (resolver/resolver.h): accepts is checked at compile time, then the Plan drives llk-lib's vec on the thread
// vec_modifiers::thread names.

#pragma once

#include "compute_api/modifiers.h"
#include "llk_lib/vec.h"
#include "resolver/vec.h"

namespace llk2
{

/// Any SFPU activation F on every tile of acc, in place. The named calls below are the API; this is what they share.
template <VecFunction F, vec_modifiers Mods, const TensorSpec<tensor_rank>& D, class... Scalars>
inline __attribute__((always_inline)) void vec_apply(const Tensor<D>& acc, Scalars... scalars)
{
    constexpr vec_call call {F, Mods};
    static_assert(D.storage == Storage::Dest, "the SFPU activations work on Dest");
    static_assert(
        accepted<vec_op, D, D, call>, "the SFPU activations take 32x32 tiles in Dest, in the kernel's Dest format, driven from the math or the pack thread");

    constexpr const Resolved& r = resolved<vec_op, D, D, call>;
    llk_lib::vec<Mods.thread, r.plan>(acc.address, scalars...);
}

/// acc = x * sigmoid(x).
template <vec_modifiers Mods = vec_modifiers {}, const TensorSpec<tensor_rank>& D>
inline __attribute__((always_inline)) void vec_silu(const Tensor<D>& acc)
{
    vec_apply<VecFunction::Silu, Mods>(acc);
}

/// acc = tanh(x).
template <vec_modifiers Mods = vec_modifiers {}, const TensorSpec<tensor_rank>& D>
inline __attribute__((always_inline)) void vec_tanh(const Tensor<D>& acc)
{
    vec_apply<VecFunction::Tanh, Mods>(acc);
}

/// acc = gelu(x), the erf form.
template <vec_modifiers Mods = vec_modifiers {}, const TensorSpec<tensor_rank>& D>
inline __attribute__((always_inline)) void vec_gelu(const Tensor<D>& acc)
{
    vec_apply<VecFunction::Gelu, Mods>(acc);
}

/// acc = gelu(x), the tanh form.
template <vec_modifiers Mods = vec_modifiers {}, const TensorSpec<tensor_rank>& D>
inline __attribute__((always_inline)) void vec_gelu_tanh(const Tensor<D>& acc)
{
    vec_apply<VecFunction::GeluTanh, Mods>(acc);
}

/// acc = min(max(x, 0), max). max is an FP32 bit pattern (relu6: 0x40c00000, 6.0).
template <vec_modifiers Mods = vec_modifiers {}, const TensorSpec<tensor_rank>& D>
inline __attribute__((always_inline)) void vec_relu_max(const Tensor<D>& acc, std::uint32_t max)
{
    vec_apply<VecFunction::ReluMax, Mods>(acc, max);
}

/// acc = sigmoid(x).
template <vec_modifiers Mods = vec_modifiers {}, const TensorSpec<tensor_rank>& D>
inline __attribute__((always_inline)) void vec_sigmoid(const Tensor<D>& acc)
{
    vec_apply<VecFunction::Sigmoid, Mods>(acc);
}

/// acc = min(max(x / 6 + 1 / 2, 0), 1).
template <vec_modifiers Mods = vec_modifiers {}, const TensorSpec<tensor_rank>& D>
inline __attribute__((always_inline)) void vec_hardsigmoid(const Tensor<D>& acc)
{
    vec_apply<VecFunction::Hardsigmoid, Mods>(acc);
}

/// acc = min(max(x, min), max). min and max are FP32 bit patterns.
template <vec_modifiers Mods = vec_modifiers {}, const TensorSpec<tensor_rank>& D>
inline __attribute__((always_inline)) void vec_hardtanh(const Tensor<D>& acc, std::uint32_t min, std::uint32_t max)
{
    vec_apply<VecFunction::Hardtanh, Mods>(acc, min, max);
}

/// acc = scale * (x > 0 ? x : alpha * (exp(x) - 1)). scale and alpha are FP32 bit patterns.
template <vec_modifiers Mods = vec_modifiers {}, const TensorSpec<tensor_rank>& D>
inline __attribute__((always_inline)) void vec_selu(const Tensor<D>& acc, std::uint32_t scale, std::uint32_t alpha)
{
    vec_apply<VecFunction::Selu, Mods>(acc, scale, alpha);
}

/// acc = log(1 + exp(beta x)) / beta, and x above threshold. beta, 1 / beta and threshold are FP32 bit patterns.
template <vec_modifiers Mods = vec_modifiers {}, const TensorSpec<tensor_rank>& D>
inline __attribute__((always_inline)) void vec_softplus(const Tensor<D>& acc, std::uint32_t beta, std::uint32_t beta_reciprocal, std::uint32_t threshold)
{
    vec_apply<VecFunction::Softplus, Mods>(acc, beta, beta_reciprocal, threshold);
}

} // namespace llk2
