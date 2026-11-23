// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Hardware probe, not an API test: can ALU_ACC_CTRL_Zero_Flag_disabled_src change what the FPU moves, and when is it
// read?
//
// A copy through SrcA moves each tile to Dest with MOVA2Ds (four cycles in the FPU). MOVA2D flushes a SrcA datum whose
// low eight bits are 0 unless the flag is set (tt-isa-documentation, MOVA2D.md). Every repetition copies the same
// tiles, half of them BF16 denormals, in three forms, each packed to its own output tiles:
//   0  Default   llk2::copy as it is: the flag is clear
//   1  FlagOn    the flag is set (with a wait for the FPU) before the MOVA2Ds
//   2  Behind    llk2::copy as it is, then the flag is set rep * NOP_STEP NOPs behind the last MOVA2D with no wait
//                for the FPU, then a wait
// Behind must equal Default. Measured: so does FlagOn (a BF16 denormal comes out as 0 whatever the flag); see
// test_llk2_fpu_zero_flag_latch_probe.py. reduce.h writes the flag right behind its last pool and its last MOVB2D
// in the same way.

#include <cstdint>

#include "ckernel.h"
#include "ckernel_defs.h"
#include "ckernel_ops.h"
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

constexpr DataFormat in_format  = static_cast<DataFormat>(UNPACK_A_IN);
constexpr DataFormat out_format = static_cast<DataFormat>(PACK_OUT);

inline constexpr auto& in_spec  = llk2::l1_spec<llk2::default_tile_dim, TILES * llk2::default_tile_dim, in_format>;
inline constexpr auto& out_spec = llk2::l1_spec<llk2::default_tile_dim, TILES * llk2::default_tile_dim, out_format>;
inline constexpr auto& acc_spec = llk2::dst_spec<llk2::default_tile_dim, TILES * llk2::default_tile_dim, llk2::kernel_dest_format>;
using AccRing                   = llk2::DestRing<acc_spec>;

constexpr const llk2::Resolved& resolved_copy = llk2::resolved<llk2::copy_op, in_spec, in_spec, llk2::copy_modifiers {}>;
static_assert(resolved_copy.plan.in0.to == llk2::Where::SrcA, "the copy goes through SrcA (MOVA2D)");
constexpr llk2::llk_lib::copy_params copy_p = llk2::llk_lib::copy_params_from(resolved_copy.plan);

inline __attribute__((always_inline)) void set_src_zero_flag_disabled()
{
    ckernel::cfg_reg_rmw_tensix<ALU_ACC_CTRL_Zero_Flag_disabled_src_RMW>(1);
}

/// llk2::copy (compute_api/copy.h) with the flag set before the math thread's MOVA2Ds.
template <class Src, class Acc>
inline __attribute__((always_inline)) auto copy_flag_on(const Src& src, const Acc& acc)
{
    using namespace llk2;
    llk_lib::acquire<Thread::Unpack, Exu::Unpacker, in_spec>();
    llk_lib::acquire<Thread::Math, Exu::Fpu, acc_spec>();
    llk_lib::unpack<Thread::Unpack, resolved_copy.plan>(WorkCounts<resolved_copy.work> {}, src.address);
    if constexpr (runs_on<Thread::Math>)
    {
        llk_lib::copy_configure<copy_p>();
        TTI_STALLWAIT(ckernel::p_stall::STALL_CFG, ckernel::p_stall::MATH | ckernel::p_stall::WAIT_SFPU);
        set_src_zero_flag_disabled();
        llk_lib::copy_execute<copy_p>(acc.address);
    }
    return as_owned<Thread::Math, Exu::Fpu>(acc);
}

void run_kernel(RUNTIME_PARAMETERS params)
{
    const auto src = llk2::Tensor<in_spec>(params.buffer_A[0]);
    for (std::uint32_t rep = 0; rep < REPS; rep++)
    {
        {
            const auto y = llk2::copy(src, llk2::reserve_back<AccRing>());
            llk2::pack(y, llk2::Tensor<out_spec>(params.buffer_Res[(3 * rep + 0) * TILES]));
            llk2::release(y);
        }
        {
            const auto y = copy_flag_on(src, llk2::reserve_back<AccRing>());
            llk2::pack(y, llk2::Tensor<out_spec>(params.buffer_Res[(3 * rep + 1) * TILES]));
            llk2::release(y);
        }
        {
            const auto y = llk2::copy(src, llk2::reserve_back<AccRing>());
#ifdef LLK_TRISC_MATH
            for (std::uint32_t nop = 0; nop < rep * NOP_STEP; nop++)
            {
                TTI_NOP;
            }
            set_src_zero_flag_disabled();
            TTI_STALLWAIT(ckernel::p_stall::STALL_THREAD, ckernel::p_stall::MATH);
#endif
            llk2::pack(y, llk2::Tensor<out_spec>(params.buffer_Res[(3 * rep + 2) * TILES]));
            llk2::release(y);
        }
    }
}
