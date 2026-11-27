// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Hardware probe, not an API test: does the timing of reduce's Src zero-flag writes matter?
//
// llk_lib/reduce.h writes ALU_ACC_CTRL_Zero_Flag_disabled_src = 1 right behind its last pool (GMPOOL / GAPOOL, five
// cycles in the FPU) and back to the default right behind its last MOVB2D (four cycles), with no wait for the FPU.
// That is right only if an FPU instruction takes the flag when it leaves the Wait Gate, not later in its pipeline.
//
// The kernel runs the same reduce REPS times in each of four forms and packs every result to its own output tile:
//   0  Library        llk_lib::reduce_execute as it is
//   1  Waited         the same instructions, with a wait for the FPU (STALLWAIT, Config, MathIdle) in front of each
//                     of the two flag writes: the form that cannot race
//   2  PoolsFlagOn    the flag is 1 before the pools start (written with a wait), then the Library form: what the
//                     pools would give if they did read a flag written behind them
//   3  MovesFlagOff   the Waited form without the flag writes: the moves run with the default flag, which is what
//                     the last MOVB2Ds would see if they read a flag restored behind them
// Library must equal Waited bit for bit. Where form 2 or 3 differs from Waited on the same data, the flag changes
// what the pools or the moves produce, and Library == Waited then shows that the instruction issued just before
// the write did not see it.
//
// The math thread's part is llk-lib called directly (reduce_configure, then one of the forms); the unpack and pack
// threads run what llk2::reduce and llk2::pack run.

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

#define LLK2_DEST_FP32     is_fp32_dest_acc_en
#define LLK2_MATH_FIDELITY static_cast<std::uint32_t>(MATH_FIDELITY)
#include "compute_api/dest.h"
#include "compute_api/pack.h"
#include "compute_api/reduce.h"

constexpr DataFormat in_format  = static_cast<DataFormat>(UNPACK_A_IN);
constexpr DataFormat out_format = static_cast<DataFormat>(PACK_OUT);
constexpr llk2::reduce_modifiers mods {.kind = IS_SUM ? llk2::ReduceKind::Sum : llk2::ReduceKind::Max};

inline constexpr auto& in_spec     = llk2::l1_spec<llk2::default_tile_dim, KT_DIM * llk2::default_tile_dim, in_format>;
inline constexpr auto& scaler_spec = llk2::l1_spec<llk2::default_tile_dim, llk2::default_tile_dim, in_format>;
inline constexpr auto& out_spec    = llk2::l1_spec<llk2::default_tile_dim, llk2::default_tile_dim, out_format>;
inline constexpr auto& acc_spec    = llk2::dst_spec<llk2::default_tile_dim, llk2::default_tile_dim, llk2::kernel_dest_format>;
using AccRing                      = llk2::DestRing<acc_spec>;

constexpr const llk2::Resolved& resolved_reduce = llk2::resolved<llk2::reduce_op, in_spec, scaler_spec, mods>;
constexpr llk2::llk_lib::reduce_params reduce_p = llk2::llk_lib::reduce_params_from(resolved_reduce.plan);
static_assert(reduce_p.tile_rows == 1, "one output tile per reduce");

enum class Form
{
    Library,
    Waited,
    PoolsFlagOn,
    MovesFlagOff,
};
constexpr std::uint32_t forms = 4;

/// reduce_execute (llk_lib/reduce.h) for one output tile, with a wait for the FPU in front of each flag write;
/// WriteFlag false leaves the flag at its default through the moves.
template <bool WriteFlag>
inline __attribute__((always_inline)) void reduce_execute_waited(std::uint32_t dest_row)
{
    using namespace llk2::llk_lib;
    using namespace hal::counters::rwc;
    constexpr auto& P                         = reduce_p;
    constexpr std::uint32_t default_zero_flag = src_zero_flag_disabled(P.src_a_format, P.src_b_format) ? 1u : 0u;
    constexpr MathCounter release_scaler {.src_a = Set(0), .src_b = Set(0), .dest = Set(0), .flip = Src::B, .clear_fidelity = true};

    set_math_dest_row(dest_row);
    hal::dst::run<hal::dst::Zero {.scope = hal::dst::ZeroScope::SingleRow, .index = reduce_top_row, .address_mode = addr_mod_hold}>();
    hal::dst::run<hal::dst::Zero {.scope = hal::dst::ZeroScope::SingleRow, .index = reduce_bottom_row, .address_mode = addr_mod_hold}>();
    for (std::uint32_t k = 0; k < P.k_tiles; k++)
    {
        hal::replay::run<reduce_tile_replay(P)>();
    }
    hal::sync::wait::stall<StallTarget::Config, StallCondition::MathIdle>();
    if constexpr (WriteFlag)
    {
        reduce_set_src_zero_flag<1 /*Disabled*/>();
    }
    reduce_row_to_column<reduce_top_row, 0>();
    reduce_row_to_column<reduce_bottom_row, 2>();
    hal::sync::wait::stall<StallTarget::Config, StallCondition::MathIdle>();
    reduce_set_src_zero_flag<default_zero_flag>();
    set<release_scaler>();
    reduce_zero_face<P>(dest_row, 1);
    reduce_zero_face<P>(dest_row, 3);
}

/// llk2::reduce (compute_api/reduce.h) with the math thread's execute in form F.
template <Form F, class In, class Scaler, class Acc>
inline __attribute__((always_inline)) auto probe_reduce(const In& in, const Scaler& scaler, const Acc& acc)
{
    using namespace llk2;
    using hal::sync::StallCondition;
    using hal::sync::StallTarget;
    llk_lib::acquire<Thread::Unpack, Exu::Unpacker, in_spec>();
    llk_lib::acquire<Thread::Unpack, Exu::Unpacker, scaler_spec>();
    llk_lib::acquire<Thread::Math, Exu::Fpu, acc_spec>();
    const auto counts = work_counts<resolved_reduce.work>(in, scaler);
    llk_lib::unpack<Thread::Unpack, resolved_reduce.plan>(counts, in.address, scaler.address);
    if constexpr (runs_on<Thread::Math>)
    {
        llk_lib::reduce_configure<reduce_p>();
        if constexpr (F == Form::Library)
        {
            llk_lib::reduce_execute<reduce_p>(acc.address, plan_count<WorkCount::K, resolved_reduce.plan.work.k_tiles>(counts));
        }
        else if constexpr (F == Form::PoolsFlagOn)
        {
            hal::sync::wait::stall<StallTarget::Config, StallCondition::MathIdle | StallCondition::SfpuIdle>();
            llk_lib::reduce_set_src_zero_flag<1 /*Disabled*/>();
            llk_lib::reduce_execute<reduce_p>(acc.address, plan_count<WorkCount::K, resolved_reduce.plan.work.k_tiles>(counts));
        }
        else
        {
            reduce_execute_waited<F == Form::Waited>(acc.address);
        }
    }
    return as_owned<Thread::Math, Exu::Fpu>(acc);
}

template <Form F, class In, class Scaler, class Out>
inline __attribute__((always_inline)) void run_form(const In& in, const Scaler& scaler, const Out& out)
{
    const auto y = probe_reduce<F>(in, scaler, llk2::reserve_back<AccRing>());
    llk2::pack(y, out);
    llk2::release(y);
}

void run_kernel(RUNTIME_PARAMETERS params)
{
    const auto in     = llk2::Tensor<in_spec>(params.buffer_A[0]);
    const auto scaler = llk2::Tensor<scaler_spec>(params.buffer_B[0]);
    for (std::uint32_t rep = 0; rep < REPS; rep++)
    {
        run_form<Form::Library>(in, scaler, llk2::Tensor<out_spec>(params.buffer_Res[rep * forms + 0]));
        run_form<Form::Waited>(in, scaler, llk2::Tensor<out_spec>(params.buffer_Res[rep * forms + 1]));
        run_form<Form::PoolsFlagOn>(in, scaler, llk2::Tensor<out_spec>(params.buffer_Res[rep * forms + 2]));
        run_form<Form::MovesFlagOff>(in, scaler, llk2::Tensor<out_spec>(params.buffer_Res[rep * forms + 3]));
    }
}
