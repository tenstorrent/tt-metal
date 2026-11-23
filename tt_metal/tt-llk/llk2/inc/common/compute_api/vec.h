// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Compute API: the SFPU functions, the vec_ calls (the form of an op: compute_api/modifiers.h). Thread: the one that
// drives the SFPU, math or pack (vec_modifiers::thread). No L1 address is read.
//
//   z = vec_silu(y);                                            // z = silu(y), every tile, in place in y's rows
//   z = vec_tanh<vec_modifiers{.form = VecForm::Fast}>(y);      // the approximate form
//   z = vec_recip<vec_modifiers{.region = VecRegion::LeftFaces}>(y);   // tile columns 0..15 only
//   z = vec_relu_max(y, max);                                   // scalars are FP32 bit patterns, as 1.0's *_tile calls take
//
// The calls, with their scalars:
//   vec_silu(acc)                vec_tanh(acc)               vec_gelu(acc)              vec_gelu_tanh(acc)
//   vec_sigmoid(acc)             vec_hardsigmoid(acc)        vec_relu_max(acc, max)     vec_hardtanh(acc, min, max)
//   vec_selu(acc, scale, alpha)  vec_softplus(acc, beta, beta_reciprocal, threshold)
//   vec_exp(acc)                 vec_recip(acc)
//
// Modifiers: vec_modifiers (resolver/modifiers.h), the same struct for every call. What a function does not have is
// refused at compile time (resolver/vec.h: vec_has_form, vec_has_region, vec_has_clamping).
//   .form         Exact (the default), Fast for the functions with an approximate form (tanh, gelu, gelu_tanh,
//                 sigmoid, exp, recip: 1.0's fast_and_approx / APPROX), Polynomial for exp on LeftFacesEvenColumns
//   .region       the datums of each tile the function rewrites; the others keep their values.
//                 All (the default); LeftFaces: faces 0 and 2, tile columns 0..15 (1.0's VectorMode::C), so column
//                 0, where a reduce leaves its values; LeftFacesEvenColumns: their even columns, half the iterations
//                 (exp only: 1.0's exp_tile_first_column)
//   .clamp_input  true (the default): the result is right for every input. The fast exp has only false: Schraudolph's
//                 approximation with no input clamping (1.0's InputClamping::None), right for input_scale x in
//                 [-88, 0.72], so for values with their maximum subtracted; below -88 it returns negative values, so
//                 pack the result with pack_modifiers::relu. The call says so by naming .clamp_input = false
//   .thread       the thread that drives the SFPU (below)
//   .input_scale  exp only: y = exp(input_scale x), an FP32 bit pattern (default 1.0). A modifier and not a run-time
//                 argument because the fast exp folds it into its constants when the SFPU is configured, so it has
//                 to be known at compile time. The other forms round it to bf16 (the top 16 bits), as 1.0 does
//
// The exp and reciprocal calls against 1.0's:
//   vec_exp<vec_modifiers{.form = VecForm::Fast, .clamp_input = false, .thread = Thread::Pack, .input_scale = s}>(y)
//                                    exp_packthread_tile<true, false, InputClamping::None, 32>, s in
//                                    exp_packthread_tile_init: the whole tile in one pass
//   vec_exp<vec_modifiers{.input_scale = s}>(y)
//                                    exp_tile<false, true>(i, RC, s >> 16): the exact exp, face by face
//   vec_exp<vec_modifiers{.form = VecForm::Polynomial, .region = VecRegion::LeftFacesEvenColumns, .input_scale = s}>(y)
//                                    exp_tile_first_column<true, s >> 16>: range reduction and a polynomial;
//                                    with .form = VecForm::Exact, exp_tile_first_column<false, s >> 16>
//   vec_recip(y)                     recip_tile(i): 1 / x on every datum; Fast is 1.0's approximate (7-bit) form
//   vec_recip<vec_modifiers{.region = VecRegion::LeftFaces}>(y)
//                                    recip_tile(i, VectorMode::C)
//
// Result and owners. y is a Dest tensor (compute_api/dest.h); the result z is y's rows and release, now the SFPU's on
// the thread that drove it. vec_modifiers::thread picks that thread: Thread::Math (the default) or Thread::Pack. The
// call takes y over for the SFPU on that thread first (compute_api/sync.h), so moving it to the pack thread needs no
// other change:
//   math:  y the FPU's -> a STALLWAIT for the FPU; pack(z, ...) later: math drains the SFPU and posts, pack waits
//   pack:  y the FPU's -> math drains the FPU and posts, the pack thread waits; pack(z, ...): a STALLWAIT for the SFPU
// Moving the SFPU between threads in the middle of a kernel: migrate (compute_api/sync.h).
//
// One thread at a time, and nothing enforces it. The functions that run from SFPLOADMACRO schedules (the fast exp, the
// reciprocals, the exact fp32 exp) count their units' delays in issued SFPU instructions, and the hardware counts an
// instruction to the Vector Unit from ANY thread (tt-isa-documentation, SFPLOADMACRO.md: the delay counters are
// "decremented by one every time a thread issues an instruction to the Vector Unit"). An SFPU instruction from the
// other thread while such a schedule is in flight moves every pending round / set-sign / store of it by one. So a
// kernel's vec_ calls that may overlap in time must all name the same thread; to change threads, migrate first (it
// waits for the SFPU to be idle). This is a rule for the kernel author: no compile-time check ties the calls of a
// kernel together.
//
// Resolver and llk-lib. vec_op (resolver/vec.h); llk-lib's vec on the thread vec_modifiers::thread names. The calls
// are one family: they share one OpApi, so what a call hands the Resolver is a vec_call (resolver/modifiers.h: the
// function it names and its modifiers), and they share the body detail::vec.
// A vec_ call has one tensor: the Resolver's second operand is acc again, and is not read.
//
// Run-time arguments (the FP32 scalars) come after the tensor, as every op's do (compute_api/modifiers.h).

#pragma once

#include <cstdint>

#include "compute_api/counts.h"
#include "compute_api/modifiers.h"
#include "llk_lib/sync.h"
#include "llk_lib/vec.h"
#include "resolver/vec.h"

namespace llk2
{

namespace detail
{
/// The body every vec_ call shares: the SFPU function Call names (the function and its modifiers) on the region of
/// every tile of acc, in place, returned as the SFPU's.
template <vec_call Call, const auto& D, class RD, class... Scalars>
inline __attribute__((always_inline)) auto vec(const Tensor<D, RD>& acc, Scalars... scalars)
{
    constexpr vec_modifiers Mods = Call.mods;
    constexpr const Accepts& ok = accepted_parts<vec_op, D, D, Call>;
    static_assert(ok.storage, "the SFPU functions (the vec_ calls) take a Dest tensor and write it");
    static_assert(ok.tiles, "the SFPU functions (the vec_ calls) take 32x32 tiles. Not ported: other tile sizes");
    static_assert(
        ok.formats,
        "the SFPU functions (the vec_ calls) take a Dest tensor in Float16_b, or in Float32 with a 32-bit Dest. Not ported: FP16 and integer Dest tensors");
    static_assert(
        ok.modifiers,
        "the SFPU functions (the vec_ calls) are driven from the math or the pack thread, in a form the function has (Fast: tanh, gelu, gelu_tanh, "
        "sigmoid, exp, recip; Polynomial: exp on LeftFacesEvenColumns), on a region that form runs on (the fast exp: All only; LeftFacesEvenColumns: "
        "exp's Exact and Polynomial), with .clamp_input = false for the fast exp and for nothing else, and .input_scale for exp only");
    static_assert(ok.runtime, "the SFPU functions (the vec_ calls) take a run-time size as tile rows, or as the tile columns of one tile row");

    constexpr const Resolved& r = resolved<vec_op, D, D, Call>;
    llk_lib::acquire<Mods.thread, Exu::Sfpu, D>();
    const auto counts = work_counts<r.work>(acc, acc); // the Work's run-time counts; empty when every size is compile time
    llk_lib::vec<Mods.thread, r.plan>(counts, acc.address, scalars...);
    return as_owned<Mods.thread, Exu::Sfpu>(acc); // in place: acc already holds the count the function ran on
}

} // namespace detail

/// x * sigmoid(x), in place; returns acc as the SFPU's.
template <vec_modifiers Mods = vec_modifiers {}, const auto& D, class RD>
inline __attribute__((always_inline)) auto vec_silu(const Tensor<D, RD>& acc)
{
    return detail::vec<vec_call {VecFunction::Silu, Mods}>(acc);
}

/// tanh(x), in place; returns acc as the SFPU's.
template <vec_modifiers Mods = vec_modifiers {}, const auto& D, class RD>
inline __attribute__((always_inline)) auto vec_tanh(const Tensor<D, RD>& acc)
{
    return detail::vec<vec_call {VecFunction::Tanh, Mods}>(acc);
}

/// gelu(x), the erf form, in place; returns acc as the SFPU's.
template <vec_modifiers Mods = vec_modifiers {}, const auto& D, class RD>
inline __attribute__((always_inline)) auto vec_gelu(const Tensor<D, RD>& acc)
{
    return detail::vec<vec_call {VecFunction::Gelu, Mods}>(acc);
}

/// gelu(x), the tanh form, in place; returns acc as the SFPU's.
template <vec_modifiers Mods = vec_modifiers {}, const auto& D, class RD>
inline __attribute__((always_inline)) auto vec_gelu_tanh(const Tensor<D, RD>& acc)
{
    return detail::vec<vec_call {VecFunction::GeluTanh, Mods}>(acc);
}

/// min(max(x, 0), max). max is an FP32 bit pattern (relu6: 0x40c00000, 6.0), in place; returns acc as the SFPU's.
template <vec_modifiers Mods = vec_modifiers {}, const auto& D, class RD>
inline __attribute__((always_inline)) auto vec_relu_max(const Tensor<D, RD>& acc, std::uint32_t max)
{
    return detail::vec<vec_call {VecFunction::ReluMax, Mods}>(acc, max);
}

/// sigmoid(x), in place; returns acc as the SFPU's.
template <vec_modifiers Mods = vec_modifiers {}, const auto& D, class RD>
inline __attribute__((always_inline)) auto vec_sigmoid(const Tensor<D, RD>& acc)
{
    return detail::vec<vec_call {VecFunction::Sigmoid, Mods}>(acc);
}

/// min(max(x / 6 + 1 / 2, 0), 1), in place; returns acc as the SFPU's.
template <vec_modifiers Mods = vec_modifiers {}, const auto& D, class RD>
inline __attribute__((always_inline)) auto vec_hardsigmoid(const Tensor<D, RD>& acc)
{
    return detail::vec<vec_call {VecFunction::Hardsigmoid, Mods}>(acc);
}

/// min(max(x, min), max). min and max are FP32 bit patterns, in place; returns acc as the SFPU's.
template <vec_modifiers Mods = vec_modifiers {}, const auto& D, class RD>
inline __attribute__((always_inline)) auto vec_hardtanh(const Tensor<D, RD>& acc, std::uint32_t min, std::uint32_t max)
{
    return detail::vec<vec_call {VecFunction::Hardtanh, Mods}>(acc, min, max);
}

/// scale * (x > 0 ? x : alpha * (exp(x) - 1)). scale and alpha are FP32 bit patterns, in place; returns acc as the SFPU's.
template <vec_modifiers Mods = vec_modifiers {}, const auto& D, class RD>
inline __attribute__((always_inline)) auto vec_selu(const Tensor<D, RD>& acc, std::uint32_t scale, std::uint32_t alpha)
{
    return detail::vec<vec_call {VecFunction::Selu, Mods}>(acc, scale, alpha);
}

/// log(1 + exp(beta x)) / beta, and x above threshold. beta, 1 / beta and threshold are FP32 bit patterns, in place; returns acc as the SFPU's.
template <vec_modifiers Mods = vec_modifiers {}, const auto& D, class RD>
inline __attribute__((always_inline)) auto vec_softplus(const Tensor<D, RD>& acc, std::uint32_t beta, std::uint32_t beta_reciprocal, std::uint32_t threshold)
{
    return detail::vec<vec_call {VecFunction::Softplus, Mods}>(acc, beta, beta_reciprocal, threshold);
}

/// exp(Mods.input_scale x), in place; returns acc as the SFPU's. Its forms, and the input range of the fast one: the
/// file comment.
template <vec_modifiers Mods = vec_modifiers {}, const auto& D, class RD>
inline __attribute__((always_inline)) auto vec_exp(const Tensor<D, RD>& acc)
{
    return detail::vec<vec_call {VecFunction::Exp, Mods}>(acc);
}

/// 1 / x, in place; returns acc as the SFPU's.
template <vec_modifiers Mods = vec_modifiers {}, const auto& D, class RD>
inline __attribute__((always_inline)) auto vec_recip(const Tensor<D, RD>& acc)
{
    return detail::vec<vec_call {VecFunction::Reciprocal, Mods}>(acc);
}

} // namespace llk2
