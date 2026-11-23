// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// HAL-based implementation of the Blackhole FPU elementwise binary LLK (SrcA [+, -, *] SrcB -> Dest).

#pragma once

#include <cstdint>

#include "ckernel.h"
#include "ckernel_defs.h"
#include "ckernel_globals.h"
#include "hal/cfg.h"
#include "hal/dst.h"
#include "hal/fpu.h"
#include "hal/math_counters.h"
#include "hal/mop.h"
#include "hal/move.h"
#include "hal/sync.h"
#include "llk_assert.h"
#include "llk_defs.h"
#include "llk_math_common.h"
#include "tensor_shape.h"

namespace hal_llk::math
{

// Math ADDR_MOD slots used by the eltwise binary sequences.
constexpr std::uint32_t AM_NEXT_BLOCK = 0; // Next 8-row block of the same face.
constexpr std::uint32_t AM_NEXT_FACE  = 1; // Last block of a face: rewind sources, step Dest to the next face.
constexpr std::uint32_t AM_FIDELITY   = 2; // Next fidelity phase of the same block.
constexpr std::uint32_t AM_MOVE_ROWS  = 3; // Dest -> Src move: next four rows.
constexpr std::uint32_t AM_MOVE_DONE  = 4; // Dest -> Src move finished: rewind Dest and the source counters.

// Rows produced by one elementwise instruction.
constexpr std::uint32_t ROWS_PER_OP = 8;

constexpr hal::fpu::ElementwiseOperation to_hal_operation(const EltwiseBinaryType type)
{
    return (type == EltwiseBinaryType::ELWADD)   ? hal::fpu::ElementwiseOperation::Add
           : (type == EltwiseBinaryType::ELWSUB) ? hal::fpu::ElementwiseOperation::Subtract
                                                 : hal::fpu::ElementwiseOperation::Multiply;
}

constexpr hal::fpu::SrcBBroadcast to_hal_broadcast(const BroadcastType type)
{
    return (type == BroadcastType::COL)      ? hal::fpu::SrcBBroadcast::Column
           : (type == BroadcastType::ROW)    ? hal::fpu::SrcBBroadcast::Row
           : (type == BroadcastType::SCALAR) ? hal::fpu::SrcBBroadcast::Scalar
                                             : hal::fpu::SrcBBroadcast::None;
}

constexpr std::uint32_t fidelity_phases(const MathFidelity fidelity)
{
    return (fidelity == MathFidelity::LoFi) ? 1 : static_cast<std::uint32_t>(fidelity);
}

// Number of consecutive faces that share one unpacked SrcB bank.
inline std::uint32_t faces_per_srcb_bank(const BroadcastType bcast, const ckernel::TensorShape tensor_shape)
{
    if (bcast == BroadcastType::SCALAR)
    {
        return tensor_shape.total_num_faces();
    }
    if (bcast == BroadcastType::COL)
    {
        return tensor_shape.num_faces_c_dim;
    }
    return 1;
}

// Number of 8-row elementwise blocks needed to cover one face.
inline std::uint32_t blocks_per_face(const std::uint32_t face_r_dim)
{
    return (face_r_dim > ROWS_PER_OP) ? (face_r_dim / ROWS_PER_OP) : 1;
}

template <EltwiseBinaryType type, BroadcastType bcast, MathFidelity math_fidelity>
inline constexpr std::uint32_t elementwise_word(const std::uint32_t address_modifier, const hal::fpu::SourceRelease release, const std::uint32_t acc_to_dest)
{
    return hal::fpu::get_operation(hal::fpu::Elementwise {
        .operation        = to_hal_operation(type),
        .broadcast        = to_hal_broadcast(bcast),
        .dest_write       = acc_to_dest ? hal::fpu::DestWriteMode::Accumulate : hal::fpu::DestWriteMode::Overwrite,
        .address_modifier = static_cast<std::uint8_t>(address_modifier),
        .release          = release,
    });
}

// Program the math MOP for one SrcB bank worth of faces: each face is blocks_per_face blocks of
// fidelity_phases instructions; the last instruction of every face releases SrcA, and the last face
// of the group also releases SrcB.
template <EltwiseBinaryType type, BroadcastType bcast, MathFidelity math_fidelity>
inline void program_face_group_mop(const std::uint32_t faces_in_group, const std::uint32_t face_r_dim, const std::uint32_t acc_to_dest)
{
    using hal::fpu::SourceRelease;
    using hal::mop::MopConfig;
    using hal::mop::MopTemplate;

    constexpr std::uint32_t phases = (type == EltwiseBinaryType::ELWMUL) ? fidelity_phases(math_fidelity) : 1;
    const std::uint32_t blocks     = blocks_per_face(face_r_dim);

    MopConfig<MopTemplate::Template1> config {};
    if constexpr (phases == 1)
    {
        config.outer_loop.count                                = faces_in_group;
        config.inner_loop.count                                = blocks;
        config.inner_loop.body_op                              = elementwise_word<type, bcast, math_fidelity>(AM_NEXT_BLOCK, SourceRelease::None, acc_to_dest);
        config.inner_loop.last_op_on_nonfinal_outer_iteration = elementwise_word<type, bcast, math_fidelity>(AM_NEXT_FACE, SourceRelease::SrcA, acc_to_dest);
        config.inner_loop.last_op_on_final_outer_iteration    = elementwise_word<type, bcast, math_fidelity>(AM_NEXT_FACE, SourceRelease::Both, acc_to_dest);
    }
    else
    {
        // High fidelity: one MOP run per face, the inner loop walks the fidelity phases of one block.
        config.outer_loop.count                                = blocks;
        config.inner_loop.count                                = phases;
        config.inner_loop.body_op                              = elementwise_word<type, bcast, math_fidelity>(AM_FIDELITY, SourceRelease::None, acc_to_dest);
        config.inner_loop.last_op_on_nonfinal_outer_iteration = elementwise_word<type, bcast, math_fidelity>(AM_NEXT_BLOCK, SourceRelease::None, acc_to_dest);
        config.inner_loop.last_op_on_final_outer_iteration =
            elementwise_word<type, bcast, math_fidelity>(AM_NEXT_FACE, (faces_in_group == 1) ? SourceRelease::Both : SourceRelease::SrcA, acc_to_dest);
    }
    hal::mop::program(config);
}

} // namespace hal_llk::math

namespace hal_llk::math
{

template <BroadcastType bcast_type>
inline void configure_addrmod(const std::uint32_t face_r_dim)
{
    namespace cfg = hal::cfg;

    // Row/scalar broadcast read one SrcB row selected by the (fixed) SrcB counter.
    constexpr std::uint32_t srcb_step = (bcast_type == BroadcastType::NONE || bcast_type == BroadcastType::COL) ? ROWS_PER_OP : 0;
    // Every face occupies FACE_R_DIM Dest rows regardless of how many rows it carries.
    const std::uint32_t blocks          = blocks_per_face(face_r_dim);
    const std::uint32_t dest_face_step  = FACE_R_DIM - ROWS_PER_OP * (blocks - 1);

    cfg::write<cfg::Access::TensixCfgUnit>(
        // AM_NEXT_BLOCK: advance sources and Dest by one block.
        cfg::set<cfg::AddrMod[cfg::SrcA].Incr, cfg::Sec::S0, ROWS_PER_OP>(),
        cfg::set<cfg::AddrMod[cfg::SrcA].CR, cfg::Sec::S0, 0>(),
        cfg::set<cfg::AddrMod[cfg::SrcA].Clear, cfg::Sec::S0, 0>(),
        cfg::set<cfg::AddrMod[cfg::SrcB].Incr, cfg::Sec::S0, srcb_step>(),
        cfg::set<cfg::AddrMod[cfg::SrcB].CR, cfg::Sec::S0, 0>(),
        cfg::set<cfg::AddrMod[cfg::SrcB].Clear, cfg::Sec::S0, 0>(),
        cfg::set<cfg::AddrMod[cfg::Dest].Incr, cfg::Sec::S0, ROWS_PER_OP>(),
        cfg::set<cfg::AddrMod[cfg::Dest].CR, cfg::Sec::S0, 0>(),
        cfg::set<cfg::AddrMod[cfg::Dest].Clear, cfg::Sec::S0, 0>(),
        cfg::set<cfg::AddrMod[cfg::Dest].CToCR, cfg::Sec::S0, 0>(),
        cfg::set<cfg::AddrMod[cfg::Fidelity].Incr, cfg::Sec::S0, 0>(),
        cfg::set<cfg::AddrMod[cfg::Fidelity].Clear, cfg::Sec::S0, 1>(),
        // AM_NEXT_FACE: rewind sources, move Dest to the start of the next face.
        cfg::set<cfg::AddrMod[cfg::SrcA].Incr, cfg::Sec::S1, 0>(),
        cfg::set<cfg::AddrMod[cfg::SrcA].CR, cfg::Sec::S1, 0>(),
        cfg::set<cfg::AddrMod[cfg::SrcA].Clear, cfg::Sec::S1, 1>(),
        cfg::set<cfg::AddrMod[cfg::SrcB].Incr, cfg::Sec::S1, 0>(),
        cfg::set<cfg::AddrMod[cfg::SrcB].CR, cfg::Sec::S1, 0>(),
        cfg::set<cfg::AddrMod[cfg::SrcB].Clear, cfg::Sec::S1, 1>(),
        cfg::set<cfg::AddrMod[cfg::Dest].Incr, cfg::Sec::S1>(dest_face_step),
        cfg::set<cfg::AddrMod[cfg::Dest].CR, cfg::Sec::S1, 0>(),
        cfg::set<cfg::AddrMod[cfg::Dest].Clear, cfg::Sec::S1, 0>(),
        cfg::set<cfg::AddrMod[cfg::Dest].CToCR, cfg::Sec::S1, 0>(),
        cfg::set<cfg::AddrMod[cfg::Fidelity].Incr, cfg::Sec::S1, 0>(),
        cfg::set<cfg::AddrMod[cfg::Fidelity].Clear, cfg::Sec::S1, 1>(),
        // AM_FIDELITY: same block, next fidelity phase.
        cfg::set<cfg::AddrMod[cfg::SrcA].Incr, cfg::Sec::S2, 0>(),
        cfg::set<cfg::AddrMod[cfg::SrcA].CR, cfg::Sec::S2, 0>(),
        cfg::set<cfg::AddrMod[cfg::SrcA].Clear, cfg::Sec::S2, 0>(),
        cfg::set<cfg::AddrMod[cfg::SrcB].Incr, cfg::Sec::S2, 0>(),
        cfg::set<cfg::AddrMod[cfg::SrcB].CR, cfg::Sec::S2, 0>(),
        cfg::set<cfg::AddrMod[cfg::SrcB].Clear, cfg::Sec::S2, 0>(),
        cfg::set<cfg::AddrMod[cfg::Dest].Incr, cfg::Sec::S2, 0>(),
        cfg::set<cfg::AddrMod[cfg::Dest].CR, cfg::Sec::S2, 0>(),
        cfg::set<cfg::AddrMod[cfg::Dest].Clear, cfg::Sec::S2, 0>(),
        cfg::set<cfg::AddrMod[cfg::Dest].CToCR, cfg::Sec::S2, 0>(),
        cfg::set<cfg::AddrMod[cfg::Fidelity].Incr, cfg::Sec::S2, 1>(),
        cfg::set<cfg::AddrMod[cfg::Fidelity].Clear, cfg::Sec::S2, 0>(),
        // AM_MOVE_ROWS: four-row Dest -> Src move fragments.
        cfg::set<cfg::AddrMod[cfg::SrcA].Incr, cfg::Sec::S3, 4>(),
        cfg::set<cfg::AddrMod[cfg::SrcA].CR, cfg::Sec::S3, 0>(),
        cfg::set<cfg::AddrMod[cfg::SrcA].Clear, cfg::Sec::S3, 0>(),
        cfg::set<cfg::AddrMod[cfg::SrcB].Incr, cfg::Sec::S3, 4>(),
        cfg::set<cfg::AddrMod[cfg::SrcB].CR, cfg::Sec::S3, 0>(),
        cfg::set<cfg::AddrMod[cfg::SrcB].Clear, cfg::Sec::S3, 0>(),
        cfg::set<cfg::AddrMod[cfg::Dest].Incr, cfg::Sec::S3, 4>(),
        cfg::set<cfg::AddrMod[cfg::Dest].CR, cfg::Sec::S3, 0>(),
        cfg::set<cfg::AddrMod[cfg::Dest].Clear, cfg::Sec::S3, 0>(),
        cfg::set<cfg::AddrMod[cfg::Dest].CToCR, cfg::Sec::S3, 0>(),
        cfg::set<cfg::AddrMod[cfg::Fidelity].Incr, cfg::Sec::S3, 0>(),
        cfg::set<cfg::AddrMod[cfg::Fidelity].Clear, cfg::Sec::S3, 0>(),
        // AM_MOVE_DONE: rewind every counter for the compute pass of the same face.
        cfg::set<cfg::AddrMod[cfg::SrcA].Incr, cfg::Sec::S4, 0>(),
        cfg::set<cfg::AddrMod[cfg::SrcA].CR, cfg::Sec::S4, 0>(),
        cfg::set<cfg::AddrMod[cfg::SrcA].Clear, cfg::Sec::S4, 1>(),
        cfg::set<cfg::AddrMod[cfg::SrcB].Incr, cfg::Sec::S4, 0>(),
        cfg::set<cfg::AddrMod[cfg::SrcB].CR, cfg::Sec::S4, 0>(),
        cfg::set<cfg::AddrMod[cfg::SrcB].Clear, cfg::Sec::S4, 1>(),
        cfg::set<cfg::AddrMod[cfg::Dest].Incr, cfg::Sec::S4, 0>(),
        cfg::set<cfg::AddrMod[cfg::Dest].CR, cfg::Sec::S4, 0>(),
        cfg::set<cfg::AddrMod[cfg::Dest].Clear, cfg::Sec::S4, 1>(),
        cfg::set<cfg::AddrMod[cfg::Dest].CToCR, cfg::Sec::S4, 0>(),
        cfg::set<cfg::AddrMod[cfg::Fidelity].Incr, cfg::Sec::S4, 0>(),
        cfg::set<cfg::AddrMod[cfg::Fidelity].Clear, cfg::Sec::S4, 0>());
}

} // namespace hal_llk::math

template <EltwiseBinaryType eltwise_binary_type, BroadcastType bcast_type, MathFidelity math_fidelity>
inline void eltwise_binary_configure_addrmod()
{
    // Full-height faces; the init functions program the slots for the actual face height.
    hal_llk::math::configure_addrmod<bcast_type>(FACE_R_DIM);
}

template <EltwiseBinaryType eltwise_binary_type, BroadcastType bcast_type, MathFidelity math_fidelity = MathFidelity::LoFi>
inline void eltwise_binary_configure_mop_standard(const std::uint32_t acc_to_dest, const ckernel::TensorShape tensor_shape)
{
    hal_llk::math::program_face_group_mop<eltwise_binary_type, bcast_type, math_fidelity>(
        hal_llk::math::faces_per_srcb_bank(bcast_type, tensor_shape), tensor_shape.face_r_dim, acc_to_dest);
}

template <EltwiseBinaryType eltwise_binary_type, BroadcastType src_b_bcast_type, MathFidelity math_fidelity = MathFidelity::LoFi>
inline void _llk_math_eltwise_binary_standard_init_(const ckernel::TensorShape tensor_shape, const std::uint32_t acc_to_dest)
{
    namespace cfg = hal::cfg;
    // Every bank is consumed with the format the unpacker recorded for it.
    cfg::write<cfg::Access::TensixCfgUnit, cfg::DisableImpliedFmt[cfg::SrcA], cfg::Sec::S0, 0>();
    cfg::write<cfg::Access::TensixCfgUnit, cfg::DisableImpliedFmt[cfg::SrcB], cfg::Sec::S0, 0>();
    hal_llk::math::configure_addrmod<src_b_bcast_type>(tensor_shape.face_r_dim);
    eltwise_binary_configure_mop_standard<eltwise_binary_type, src_b_bcast_type, math_fidelity>(acc_to_dest, tensor_shape);
    hal::math_counters.set<hal::rwc::Counters::All, 0>().clear_fidelity().apply();
}

template <EltwiseBinaryType eltwise_binary_type, BroadcastType src_b_bcast_type, DstSync Dst, bool is_fp32_dest_acc_en, MathFidelity math_fidelity = MathFidelity::LoFi>
inline void _llk_math_eltwise_binary_standard_(const ckernel::TensorShape tensor_shape, std::uint32_t dst_index)
{
    using hal::fpu::SourceRelease;

    hal_llk::math::set_dst_write_addr(dst_index);
    hal::math_counters.set<hal::rwc::Counters::Dest, 0>().apply();

    const std::uint32_t num_faces      = tensor_shape.total_num_faces();
    const std::uint32_t faces_per_bank = hal_llk::math::faces_per_srcb_bank(src_b_bcast_type, tensor_shape);

    constexpr bool high_fidelity = (eltwise_binary_type == EltwiseBinaryType::ELWMUL) && (hal_llk::math::fidelity_phases(math_fidelity) > 1);
    if constexpr (!high_fidelity)
    {
        for (std::uint32_t group = 0; group < num_faces / faces_per_bank; group++)
        {
            hal::mop::Runner<hal::mop::MopTemplate::Template1>::run();
        }
    }
    else
    {
        for (std::uint32_t face = 0; face < num_faces; face++)
        {
            const bool last_in_group = ((face + 1) % faces_per_bank) == 0;
            hal::mop::write<hal::mop::Template1Field::LastOpOnFinalOuterIteration>(
                hal_llk::math::elementwise_word<eltwise_binary_type, src_b_bcast_type, math_fidelity>(
                    hal_llk::math::AM_NEXT_FACE, last_in_group ? SourceRelease::Both : SourceRelease::SrcA, 0));
            hal::mop::Runner<hal::mop::MopTemplate::Template1>::run();
        }
    }
}

template <EltwiseBinaryReuseDestType binary_reuse_dest>
inline void eltwise_binary_reuse_dest_as_src()
{
    namespace sync = hal::sync;
    static_assert(binary_reuse_dest != EltwiseBinaryReuseDestType::NONE);

    // Dest -> Src moves do not wait for bank ownership: the unpacked bank must be valid and math idle.
    if constexpr (binary_reuse_dest == EltwiseBinaryReuseDestType::DEST_TO_SRCA)
    {
        sync::wait::stall<sync::StallTarget::Math, sync::StallCondition::SrcAValid | sync::StallCondition::MathIdle>();
        constexpr hal::move::Transfer<hal::move::Dst, hal::move::SrcA> move_face {
            .source_row         = 0,
            .source_addressing  = hal::move::RowAddressing::CounterRelative,
            .destination_row    = 0,
            .address_mode       = hal_llk::math::AM_MOVE_ROWS,
            .number_of_rows     = hal::move::rows::Face,
            .final_address_mode = hal_llk::math::AM_MOVE_DONE,
        };
        hal::move::run<move_face>();
    }
    else
    {
        sync::wait::stall<sync::StallTarget::Math, sync::StallCondition::SrcBValid | sync::StallCondition::MathIdle>();
        constexpr hal::move::Transfer<hal::move::Dst, hal::move::SrcB> move_face {
            .source_row         = 0,
            .source_addressing  = hal::move::RowAddressing::CounterRelative,
            .destination_row    = 0,
            .address_mode       = hal_llk::math::AM_MOVE_ROWS,
            .number_of_rows     = hal::move::rows::Face,
            .final_address_mode = hal_llk::math::AM_MOVE_DONE,
        };
        hal::move::run<move_face>();
    }
}

template <EltwiseBinaryType eltwise_binary_type, BroadcastType src_b_bcast_type, MathFidelity math_fidelity = MathFidelity::LoFi>
inline void eltwise_binary_configure_mop_with_dest_reuse(const std::uint32_t acc_to_dest, const ckernel::TensorShape tensor_shape)
{
    // One face per MOP run: every face first pulls its Dest rows into a source register.
    hal_llk::math::program_face_group_mop<eltwise_binary_type, src_b_bcast_type, math_fidelity>(1, tensor_shape.face_r_dim, acc_to_dest);
}

template <
    EltwiseBinaryType eltwise_binary_type,
    BroadcastType src_b_bcast_type,
    MathFidelity math_fidelity                   = MathFidelity::LoFi,
    EltwiseBinaryReuseDestType binary_reuse_dest = EltwiseBinaryReuseDestType::DEST_TO_SRCA>
inline void _llk_math_eltwise_binary_with_dest_reuse_init_(const ckernel::TensorShape tensor_shape, const std::uint32_t acc_to_dest)
{
    namespace cfg = hal::cfg;
    static_assert(binary_reuse_dest != EltwiseBinaryReuseDestType::NONE, "Dest reuse init requires DEST_TO_SRCA or DEST_TO_SRCB");
    static_assert(src_b_bcast_type == BroadcastType::NONE, "Dest reuse supports no SrcB broadcast");

    // The Dest move overwrites a bank the unpacker has already published, so the format it recorded for
    // that bank stays valid: implied-format inference is left enabled.
    cfg::write<cfg::Access::TensixCfgUnit, cfg::DisableImpliedFmt[cfg::SrcA], cfg::Sec::S0, 0>();
    cfg::write<cfg::Access::TensixCfgUnit, cfg::DisableImpliedFmt[cfg::SrcB], cfg::Sec::S0, 0>();

    hal_llk::math::configure_addrmod<src_b_bcast_type>(tensor_shape.face_r_dim);
    eltwise_binary_configure_mop_with_dest_reuse<eltwise_binary_type, src_b_bcast_type, math_fidelity>(acc_to_dest, tensor_shape);
    hal::math_counters.set<hal::rwc::Counters::All, 0>().clear_fidelity().apply();
}

template <bool is_fp32_dest_acc_en, EltwiseBinaryReuseDestType binary_reuse_dest, MathFidelity math_fidelity = MathFidelity::LoFi>
inline void eltwise_binary_run_with_dest_reuse(
    const std::uint32_t loop_count,
    const std::uint32_t face_offset,
    [[maybe_unused]] const bool clear_fp32_dst_acc,
    const std::uint32_t dst_index,
    [[maybe_unused]] const std::uint32_t face_r_dim)
{
    for (std::uint32_t face = face_offset; face < face_offset + loop_count; face++)
    {
        // Each face is processed with all counters at zero and the Dest window on that face.
        hal_llk::math::set_dest_offset(hal_llk::math::dest_section_base() + dst_index * hal_llk::math::DEST_TILE_ROWS + face * FACE_R_DIM);
        hal::math_counters.set<hal::rwc::Counters::All, 0>().apply();

        eltwise_binary_reuse_dest_as_src<binary_reuse_dest>();

        hal::mop::Runner<hal::mop::MopTemplate::Template1>::run();
    }
}

inline void zeroacc_face_by_row()
{
    // Mark the 16 rows of the face at the current Dest window/counter undefined (read back as zero).
    hal::dst::run<hal::dst::Zero {.scope = hal::dst::ZeroScope::SingleRow, .index = 0}>();
    hal::dst::run<hal::dst::Zero {.scope = hal::dst::ZeroScope::SingleRow, .index = 1}>();
    hal::dst::run<hal::dst::Zero {.scope = hal::dst::ZeroScope::SingleRow, .index = 2}>();
    hal::dst::run<hal::dst::Zero {.scope = hal::dst::ZeroScope::SingleRow, .index = 3}>();
    hal::dst::run<hal::dst::Zero {.scope = hal::dst::ZeroScope::SingleRow, .index = 4}>();
    hal::dst::run<hal::dst::Zero {.scope = hal::dst::ZeroScope::SingleRow, .index = 5}>();
    hal::dst::run<hal::dst::Zero {.scope = hal::dst::ZeroScope::SingleRow, .index = 6}>();
    hal::dst::run<hal::dst::Zero {.scope = hal::dst::ZeroScope::SingleRow, .index = 7}>();
    hal::dst::run<hal::dst::Zero {.scope = hal::dst::ZeroScope::SingleRow, .index = 8}>();
    hal::dst::run<hal::dst::Zero {.scope = hal::dst::ZeroScope::SingleRow, .index = 9}>();
    hal::dst::run<hal::dst::Zero {.scope = hal::dst::ZeroScope::SingleRow, .index = 10}>();
    hal::dst::run<hal::dst::Zero {.scope = hal::dst::ZeroScope::SingleRow, .index = 11}>();
    hal::dst::run<hal::dst::Zero {.scope = hal::dst::ZeroScope::SingleRow, .index = 12}>();
    hal::dst::run<hal::dst::Zero {.scope = hal::dst::ZeroScope::SingleRow, .index = 13}>();
    hal::dst::run<hal::dst::Zero {.scope = hal::dst::ZeroScope::SingleRow, .index = 14}>();
    hal::dst::run<hal::dst::Zero {.scope = hal::dst::ZeroScope::SingleRow, .index = 15}>();
}

template <
    EltwiseBinaryType eltwise_binary_type,
    BroadcastType src_b_bcast_type,
    DstSync Dst,
    bool is_fp32_dest_acc_en,
    MathFidelity math_fidelity,
    EltwiseBinaryReuseDestType binary_reuse_dest>
inline void _llk_math_eltwise_binary_with_dest_reuse_(const ckernel::TensorShape tensor_shape, std::uint32_t dst_index, const bool clear_fp32_dst_acc)
{
    eltwise_binary_run_with_dest_reuse<is_fp32_dest_acc_en, binary_reuse_dest, math_fidelity>(
        tensor_shape.total_num_faces(), 0, clear_fp32_dst_acc, dst_index, tensor_shape.face_r_dim);
    // Leave the Dest window on the tile slot for any following standard operation.
    hal_llk::math::set_dst_write_addr(dst_index);
    hal::math_counters.set<hal::rwc::Counters::All, 0>().apply();
}

template <
    EltwiseBinaryType eltwise_binary_type,
    BroadcastType src_b_bcast_type,
    MathFidelity math_fidelity                   = MathFidelity::LoFi,
    EltwiseBinaryReuseDestType binary_reuse_dest = EltwiseBinaryReuseDestType::NONE>
inline void _llk_math_eltwise_binary_init_(const ckernel::TensorShape tensor_shape, const std::uint32_t acc_to_dest)
{
    if constexpr (binary_reuse_dest == EltwiseBinaryReuseDestType::NONE)
    {
        _llk_math_eltwise_binary_standard_init_<eltwise_binary_type, src_b_bcast_type, math_fidelity>(tensor_shape, acc_to_dest);
    }
    else
    {
        _llk_math_eltwise_binary_with_dest_reuse_init_<eltwise_binary_type, src_b_bcast_type, math_fidelity, binary_reuse_dest>(tensor_shape, acc_to_dest);
    }
}

template <
    EltwiseBinaryType eltwise_binary_type,
    BroadcastType src_b_bcast_type,
    DstSync Dst,
    bool is_fp32_dest_acc_en,
    MathFidelity math_fidelity                   = MathFidelity::LoFi,
    EltwiseBinaryReuseDestType binary_reuse_dest = EltwiseBinaryReuseDestType::NONE>
inline void _llk_math_eltwise_binary_(const ckernel::TensorShape tensor_shape, std::uint32_t dst_index, const bool clear_fp32_dst_acc = false)
{
    if constexpr (binary_reuse_dest == EltwiseBinaryReuseDestType::NONE)
    {
        _llk_math_eltwise_binary_standard_<eltwise_binary_type, src_b_bcast_type, Dst, is_fp32_dest_acc_en, math_fidelity>(tensor_shape, dst_index);
    }
    else
    {
        _llk_math_eltwise_binary_with_dest_reuse_<eltwise_binary_type, src_b_bcast_type, Dst, is_fp32_dest_acc_en, math_fidelity, binary_reuse_dest>(
            tensor_shape, dst_index, clear_fp32_dst_acc);
    }
}

inline void _llk_math_eltwise_binary_uninit_()
{
}
