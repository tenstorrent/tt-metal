// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Compute API test: the SFPU activations (llk::vec_*) on Tensors, one kernel body for all three threads. The
// activation is a compile-time parameter (VEC_FUNCTION, an llk::VecFunction; VEC_FAST), and so is the
// thread that drives the SFPU (VEC_THREAD: llk::OwnerThread::Compute1 or, as the target matmul kernel does, OwnerThread::Compute2).
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

inline constexpr auto& in_spec       = llk::l1_spec<ROWS, COLS, in_format>;
inline constexpr auto& out_spec      = llk::l1_spec<ROWS, COLS, out_format>;
inline constexpr auto& acc_spec      = llk::dst_spec<ROWS, COLS, llk::kernel_dest_format>; // the whole tensor in Dest
inline constexpr llk::OwnerThread thread = VEC_THREAD;

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
template <class Acc, class R>
inline auto apply(const llk::Tensor<Acc, R>& acc)
{
    using llk::VecFunction;
    constexpr llk::vec_modifiers mods {.fast = VEC_FAST, .thread = thread};
    constexpr VecFunction f = VEC_FUNCTION;
    if constexpr (f == VecFunction::Silu)
    {
        return llk::vec_silu<mods>(acc);
    }
    else if constexpr (f == VecFunction::Tanh)
    {
        return llk::vec_tanh<mods>(acc);
    }
    else if constexpr (f == VecFunction::Gelu)
    {
        return llk::vec_gelu<mods>(acc);
    }
    else if constexpr (f == VecFunction::GeluTanh)
    {
        return llk::vec_gelu_tanh<mods>(acc);
    }
    else if constexpr (f == VecFunction::ReluMax)
    {
        return llk::vec_relu_max<mods>(acc, relu_max_max);
    }
    else if constexpr (f == VecFunction::Sigmoid)
    {
        return llk::vec_sigmoid<mods>(acc);
    }
    else if constexpr (f == VecFunction::Hardsigmoid)
    {
        return llk::vec_hardsigmoid<mods>(acc);
    }
    else if constexpr (f == VecFunction::Hardtanh)
    {
        return llk::vec_hardtanh<mods>(acc, hardtanh_min, hardtanh_max);
    }
    else if constexpr (f == VecFunction::Selu)
    {
        return llk::vec_selu<mods>(acc, selu_scale, selu_alpha);
    }
    else if constexpr (f == VecFunction::Softplus)
    {
        return llk::vec_softplus<mods>(acc, softplus_beta, softplus_beta_recip, softplus_threshold);
    }
    else
    {
        static_assert(f != f, "unknown VEC_FUNCTION");
    }
}

void run_kernel(RUNTIME_PARAMETERS params)
{
    const auto x   = llk::Tensor(in_spec, params.buffer_A[0]);
    const auto out = llk::Tensor(out_spec, params.buffer_Res[0]);

    // The same body whichever thread drives the SFPU: the hand-offs follow the owners. With VEC_THREAD = Math, apply
    // stalls for the FPU and pack takes z from math; with Pack, apply takes y from math and pack stalls for the SFPU.
    using AccRing = llk::DestRing<decltype(acc_spec)>;
    const auto y  = llk::copy(x, llk::reserve_back<AccRing>()); // y = x, math's
    const auto z  = apply(y);                                     // z = f(y), every tile, the SFPU's on VEC_THREAD
    llk::pack(z, out);
    llk::release(z);
}
