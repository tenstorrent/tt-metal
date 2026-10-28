// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Compute API test: the SFPU activations (llk2::vec_*) on Tensors, one kernel body for all three threads. The
// activation is a compile-time parameter (VEC_FUNCTION, the value of llk2::VecFunction; VEC_FAST), and so is the
// thread that drives the SFPU (VEC_THREAD: llk2::Thread::Math or, as the target matmul kernel does, Thread::Pack).
// Only Tensors, the Compute API (copy, vec_*, pack) and the Dest sync API: no LLK, init or per-thread code.

#include <cstdint>

#include "ckernel.h"
#include "ckernel_defs.h"
#include "llk_defs.h"
#include "llk_memory_checks.h"
#include "params.h"

// Globals
std::uint32_t unp_cfg_context          = 0;
std::uint32_t pack_sync_tile_dst_ptr   = 0;
std::uint32_t math_sync_tile_dst_index = 0;

#define LLK2_DEST_FP32 is_fp32_dest_acc_en
#include "compute_api/copy.h"
#include "compute_api/dest.h"
#include "compute_api/pack.h"
#include "compute_api/vec.h"

constexpr DataFormat in_format   = static_cast<DataFormat>(UNPACK_A_IN);
constexpr DataFormat out_format  = static_cast<DataFormat>(PACK_OUT);
constexpr DataFormat dest_format = llk2::kernel_settings.dest_fp32 ? DataFormat::Float32 : DataFormat::Float16_b;

inline constexpr auto& in_spec  = llk2::l1_spec<ROWS, COLS, in_format>;
inline constexpr auto& out_spec = llk2::l1_spec<ROWS, COLS, out_format>;
inline constexpr auto& acc_spec = llk2::dst_spec<ROWS, COLS, dest_format>; // the whole tensor in Dest
inline constexpr auto thread    = static_cast<llk2::Thread>(VEC_THREAD);

// The scalars each activation takes, as FP32 bit patterns. The goldens in test_llk2_vec.py use the same values.
constexpr std::uint32_t relu_max_max        = 0x40a00000u; // 5.0 (RELU_MAX_THRESHOLD)
constexpr std::uint32_t hardtanh_min        = 0xbf800000u; // -1.0 (CLAMP_MIN)
constexpr std::uint32_t hardtanh_max        = 0x3f800000u; // 1.0 (CLAMP_MAX)
constexpr std::uint32_t selu_scale          = 0x3f867d5fu; // 1.0507 (torch selu)
constexpr std::uint32_t selu_alpha          = 0x3fd62d7du; // 1.6733 (torch selu)
constexpr std::uint32_t softplus_beta       = 0x3f800000u; // 1.0 (SOFTPLUS_BETA)
constexpr std::uint32_t softplus_beta_recip = 0x3f800000u; // 1 / 1.0
constexpr std::uint32_t softplus_threshold  = 0x41a00000u; // 20.0 (SOFTPLUS_THRESHOLD)

template <const auto& Acc>
inline void apply(const llk2::Tensor<Acc>& acc)
{
    using llk2::VecFunction;
    constexpr llk2::vec_modifiers mods {.fast = VEC_FAST, .thread = thread};
    constexpr VecFunction f = static_cast<VecFunction>(VEC_FUNCTION);
    if constexpr (f == VecFunction::Silu)
    {
        llk2::vec_silu<mods>(acc);
    }
    else if constexpr (f == VecFunction::Tanh)
    {
        llk2::vec_tanh<mods>(acc);
    }
    else if constexpr (f == VecFunction::Gelu)
    {
        llk2::vec_gelu<mods>(acc);
    }
    else if constexpr (f == VecFunction::GeluTanh)
    {
        llk2::vec_gelu_tanh<mods>(acc);
    }
    else if constexpr (f == VecFunction::ReluMax)
    {
        llk2::vec_relu_max<mods>(acc, relu_max_max);
    }
    else if constexpr (f == VecFunction::Sigmoid)
    {
        llk2::vec_sigmoid<mods>(acc);
    }
    else if constexpr (f == VecFunction::Hardsigmoid)
    {
        llk2::vec_hardsigmoid<mods>(acc);
    }
    else if constexpr (f == VecFunction::Hardtanh)
    {
        llk2::vec_hardtanh<mods>(acc, hardtanh_min, hardtanh_max);
    }
    else if constexpr (f == VecFunction::Selu)
    {
        llk2::vec_selu<mods>(acc, selu_scale, selu_alpha);
    }
    else if constexpr (f == VecFunction::Softplus)
    {
        llk2::vec_softplus<mods>(acc, softplus_beta, softplus_beta_recip, softplus_threshold);
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

    const auto acc = llk2::dest_acquire<acc_spec>();
    llk2::copy(x, acc); // acc = x
    if constexpr (thread == llk2::Thread::Math)
    {
        apply(acc); // acc = f(acc), every tile, before math hands Dest over
    }
    llk2::dest_commit(acc);
    if constexpr (thread == llk2::Thread::Pack)
    {
        apply(acc); // acc = f(acc), every tile, after math handed Dest to the packer
    }
    llk2::pack(acc, out);
    llk2::dest_release(acc);
}
