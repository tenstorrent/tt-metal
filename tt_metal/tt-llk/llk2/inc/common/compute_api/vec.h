// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Compute API: the SFPU activations. Single thread, arch-independent. Every TRISC runs these same calls; llk-lib's
// per-thread call executes only on the thread that drives the SFPU.
//
//   z = vec_silu(y);                                 // z = silu(y), every tile, in place in y's rows
//   z = vec_tanh<vec_modifiers{.fast = true}>(y);    // the approximate form
//   z = vec_relu_max(y, max);                        // scalars are FP32 bit patterns, as today's *_tile calls take
//
// The calls, with their scalars:
//   vec_silu(acc)                vec_tanh(acc)               vec_gelu(acc)              vec_gelu_tanh(acc)
//   vec_sigmoid(acc)             vec_hardsigmoid(acc)        vec_relu_max(acc, max)     vec_hardtanh(acc, min, max)
//   vec_selu(acc, scale, alpha)  vec_softplus(acc, beta, beta_reciprocal, threshold)
//
// y is a Dest tensor (compute_api/dest.h); the result z is y's rows and release, now the SFPU's on the thread that
// drove it. vec_modifiers::thread picks that thread: Thread::Math (the default) or Thread::Pack. The call takes y over
// for the SFPU on that thread first (compute_api/sync.h), so moving it to the pack thread needs no other change:
//   math:  y the FPU's -> a STALLWAIT for the FPU; pack(z, ...) later: math drains the SFPU and posts, pack waits
//   pack:  y the FPU's -> math drains the FPU and posts, the pack thread waits; pack(z, ...): a STALLWAIT for the SFPU
// Moving the SFPU between threads in the middle of a kernel: migrate (compute_api/sync.h). Each call resolves itself
// as a chain of one (resolver/resolver.h): accepts is checked at compile time, then the Plan drives llk-lib's vec on
// the thread vec_modifiers::thread names.

#pragma once

#include "compute_api/modifiers.h"
#include "llk_lib/sync.h"
#include "llk_lib/vec.h"
#include "resolver/vec.h"

namespace llk2
{

/// Any SFPU activation F on every tile of acc, in place, returned as the SFPU's. The named calls below are the API;
/// this is what they share.
template <VecFunction F, vec_modifiers Mods, const TensorSpec<tensor_rank>& D, class R, class... Scalars>
inline __attribute__((always_inline)) auto vec_apply(const Tensor<D, R>& acc, Scalars... scalars)
{
    constexpr vec_call call {F, Mods};
    static_assert(D.storage == Storage::Dest, "the SFPU activations work on Dest");
    static_assert(
        accepted<vec_op, D, D, call>, "the SFPU activations take 32x32 tiles in Dest, in the kernel's Dest format, driven from the math or the pack thread");

    constexpr const Resolved& r = resolved<vec_op, D, D, call>;
    llk_lib::acquire<Mods.thread, Exu::Sfpu, D>();
    llk_lib::vec<Mods.thread, r.plan>(acc.address, scalars...);
    return as_owned<Mods.thread, Exu::Sfpu>(acc);
}

/// x * sigmoid(x), in place; returns acc as the SFPU's.
template <vec_modifiers Mods = vec_modifiers {}, const TensorSpec<tensor_rank>& D, class R>
inline __attribute__((always_inline)) auto vec_silu(const Tensor<D, R>& acc)
{
    return vec_apply<VecFunction::Silu, Mods>(acc);
}

/// tanh(x), in place; returns acc as the SFPU's.
template <vec_modifiers Mods = vec_modifiers {}, const TensorSpec<tensor_rank>& D, class R>
inline __attribute__((always_inline)) auto vec_tanh(const Tensor<D, R>& acc)
{
    return vec_apply<VecFunction::Tanh, Mods>(acc);
}

/// gelu(x), the erf form, in place; returns acc as the SFPU's.
template <vec_modifiers Mods = vec_modifiers {}, const TensorSpec<tensor_rank>& D, class R>
inline __attribute__((always_inline)) auto vec_gelu(const Tensor<D, R>& acc)
{
    return vec_apply<VecFunction::Gelu, Mods>(acc);
}

/// gelu(x), the tanh form, in place; returns acc as the SFPU's.
template <vec_modifiers Mods = vec_modifiers {}, const TensorSpec<tensor_rank>& D, class R>
inline __attribute__((always_inline)) auto vec_gelu_tanh(const Tensor<D, R>& acc)
{
    return vec_apply<VecFunction::GeluTanh, Mods>(acc);
}

/// min(max(x, 0), max). max is an FP32 bit pattern (relu6: 0x40c00000, 6.0), in place; returns acc as the SFPU's.
template <vec_modifiers Mods = vec_modifiers {}, const TensorSpec<tensor_rank>& D, class R>
inline __attribute__((always_inline)) auto vec_relu_max(const Tensor<D, R>& acc, std::uint32_t max)
{
    return vec_apply<VecFunction::ReluMax, Mods>(acc, max);
}

/// sigmoid(x), in place; returns acc as the SFPU's.
template <vec_modifiers Mods = vec_modifiers {}, const TensorSpec<tensor_rank>& D, class R>
inline __attribute__((always_inline)) auto vec_sigmoid(const Tensor<D, R>& acc)
{
    return vec_apply<VecFunction::Sigmoid, Mods>(acc);
}

/// min(max(x / 6 + 1 / 2, 0), 1), in place; returns acc as the SFPU's.
template <vec_modifiers Mods = vec_modifiers {}, const TensorSpec<tensor_rank>& D, class R>
inline __attribute__((always_inline)) auto vec_hardsigmoid(const Tensor<D, R>& acc)
{
    return vec_apply<VecFunction::Hardsigmoid, Mods>(acc);
}

/// min(max(x, min), max). min and max are FP32 bit patterns, in place; returns acc as the SFPU's.
template <vec_modifiers Mods = vec_modifiers {}, const TensorSpec<tensor_rank>& D, class R>
inline __attribute__((always_inline)) auto vec_hardtanh(const Tensor<D, R>& acc, std::uint32_t min, std::uint32_t max)
{
    return vec_apply<VecFunction::Hardtanh, Mods>(acc, min, max);
}

/// scale * (x > 0 ? x : alpha * (exp(x) - 1)). scale and alpha are FP32 bit patterns, in place; returns acc as the SFPU's.
template <vec_modifiers Mods = vec_modifiers {}, const TensorSpec<tensor_rank>& D, class R>
inline __attribute__((always_inline)) auto vec_selu(const Tensor<D, R>& acc, std::uint32_t scale, std::uint32_t alpha)
{
    return vec_apply<VecFunction::Selu, Mods>(acc, scale, alpha);
}

/// log(1 + exp(beta x)) / beta, and x above threshold. beta, 1 / beta and threshold are FP32 bit patterns, in place; returns acc as the SFPU's.
template <vec_modifiers Mods = vec_modifiers {}, const TensorSpec<tensor_rank>& D, class R>
inline __attribute__((always_inline)) auto vec_softplus(const Tensor<D, R>& acc, std::uint32_t beta, std::uint32_t beta_reciprocal, std::uint32_t threshold)
{
    return vec_apply<VecFunction::Softplus, Mods>(acc, beta, beta_reciprocal, threshold);
}

} // namespace llk2
