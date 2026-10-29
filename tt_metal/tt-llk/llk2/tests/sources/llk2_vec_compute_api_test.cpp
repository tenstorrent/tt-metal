// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Compute API test: the SFPU activations (llk2::vec_*) on Tensors, one kernel body for all three threads. The
// activation is a compile-time parameter (VEC_FUNCTION, an llk2::VecFunction; VEC_FAST), and so is the
// thread that drives the SFPU (VEC_THREAD: llk2::Thread::Math or, as the target matmul kernel does, Thread::Pack).
// Only Tensors, the Compute API (copy, vec_*, pack) and a Dest ring: no LLK, init or per-thread code.

#include <cstdint>

#include "ckernel.h"
#include "ckernel_defs.h"
#include "llk_defs.h"
#include "llk_memory_checks.h"
#include "params.h"

// ckernel_globals.h declares these extern; unp_cfg_context is the unpack context unpack.h flips.
std::uint32_t unp_cfg_context          = 0;
std::uint32_t pack_sync_tile_dst_ptr   = 0;
std::uint32_t math_sync_tile_dst_index = 0;

#define LLK2_DEST_FP32 is_fp32_dest_acc_en
#include "compute_api/copy.h"
#include "compute_api/dest.h"
#include "compute_api/pack.h"
#include "compute_api/vec.h"

constexpr DataFormat in_format  = static_cast<DataFormat>(UNPACK_A_IN);
constexpr DataFormat out_format = static_cast<DataFormat>(PACK_OUT);

inline constexpr auto& in_spec       = llk2::l1_spec<ROWS, COLS, in_format>;
inline constexpr auto& out_spec      = llk2::l1_spec<ROWS, COLS, out_format>;
inline constexpr auto& acc_spec      = llk2::dst_spec<ROWS, COLS, llk2::kernel_dest_format>; // the whole tensor in Dest
inline constexpr llk2::Thread thread = VEC_THREAD;

// The scalars each activation takes, as FP32 bit patterns. The goldens in test_llk2_vec.py use the same values.
constexpr std::uint32_t relu_max_max        = 0x40a00000u; // 5.0 (RELU_MAX_THRESHOLD)
constexpr std::uint32_t hardtanh_min        = 0xbf800000u; // -1.0 (CLAMP_MIN)
constexpr std::uint32_t hardtanh_max        = 0x3f800000u; // 1.0 (CLAMP_MAX)
constexpr std::uint32_t selu_scale          = 0x3f867d5fu; // 1.0507 (torch selu)
constexpr std::uint32_t selu_alpha          = 0x3fd62d7du; // 1.6733 (torch selu)
constexpr std::uint32_t softplus_beta       = 0x3f800000u; // 1.0 (SOFTPLUS_BETA)
constexpr std::uint32_t softplus_beta_recip = 0x3f800000u; // 1 / 1.0
constexpr std::uint32_t softplus_threshold  = 0x41a00000u; // 20.0 (SOFTPLUS_THRESHOLD)

/// VEC_FUNCTION on acc, driven from VEC_THREAD; returns acc as that thread's SFPU's.
template <const auto& Acc, class R>
inline auto apply(const llk2::Tensor<Acc, R>& acc)
{
    using llk2::VecFunction;
    constexpr llk2::vec_modifiers mods {.fast = VEC_FAST, .thread = thread};
    constexpr VecFunction f = VEC_FUNCTION;
    if constexpr (f == VecFunction::Silu)
    {
        return llk2::vec_silu<mods>(acc);
    }
    else if constexpr (f == VecFunction::Tanh)
    {
        return llk2::vec_tanh<mods>(acc);
    }
    else if constexpr (f == VecFunction::Gelu)
    {
        return llk2::vec_gelu<mods>(acc);
    }
    else if constexpr (f == VecFunction::GeluTanh)
    {
        return llk2::vec_gelu_tanh<mods>(acc);
    }
    else if constexpr (f == VecFunction::ReluMax)
    {
        return llk2::vec_relu_max<mods>(acc, relu_max_max);
    }
    else if constexpr (f == VecFunction::Sigmoid)
    {
        return llk2::vec_sigmoid<mods>(acc);
    }
    else if constexpr (f == VecFunction::Hardsigmoid)
    {
        return llk2::vec_hardsigmoid<mods>(acc);
    }
    else if constexpr (f == VecFunction::Hardtanh)
    {
        return llk2::vec_hardtanh<mods>(acc, hardtanh_min, hardtanh_max);
    }
    else if constexpr (f == VecFunction::Selu)
    {
        return llk2::vec_selu<mods>(acc, selu_scale, selu_alpha);
    }
    else if constexpr (f == VecFunction::Softplus)
    {
        return llk2::vec_softplus<mods>(acc, softplus_beta, softplus_beta_recip, softplus_threshold);
    }
    else
    {
        static_assert(f != f, "unknown VEC_FUNCTION");
    }
}

void run_kernel(RUNTIME_PARAMETERS params)
{
    const auto x   = llk2::Tensor<in_spec>(params.buffer_A[0]);
    const auto out = llk2::Tensor<out_spec>(params.buffer_Res[0]);

    // The same body whichever thread drives the SFPU: the hand-offs follow the owners. With VEC_THREAD = Math, apply
    // stalls for the FPU and pack takes z from math; with Pack, apply takes y from math and pack stalls for the SFPU.
    using AccRing = llk2::DestRing<acc_spec>;
    const auto y  = llk2::copy(x, llk2::reserve_back<AccRing>()); // y = x, math's
    const auto z  = apply(y);                                     // z = f(y), every tile, the SFPU's on VEC_THREAD
    llk2::pack(z, out);
    llk2::release(z);
}
