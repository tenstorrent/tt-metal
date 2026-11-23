// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "hal/dst.h"
#include "hal/fpu.h"
#include "hal/math_counters.h"
#include "hal/mop.h"
#include "hal/move.h"
#include "hal/src.h"
#include "llk_hal_common.h"
#include "llk_math_common.h"

namespace hal_llk
{

// Math ADDR_MOD slots used by eltwise binary.
inline constexpr std::uint8_t AM_STEP      = 0; // next 8 rows of SrcA/SrcB/Dest
inline constexpr std::uint8_t AM_FACE_END  = 1; // rewind SrcA/SrcB to row 0, Dest to the next 8 rows
inline constexpr std::uint8_t AM_TILE_END  = 2; // rewind SrcA/SrcB/Dest to row 0
inline constexpr std::uint8_t AM_HOLD      = 3; // leave every counter unchanged
inline constexpr std::uint8_t AM_FID_NEXT  = 4; // replay the face with the next fidelity phase
inline constexpr std::uint8_t AM_FID_DONE  = 5; // last fidelity phase done: Dest to the next face

template <EltwiseBinaryType type>
constexpr hal::fpu::ElementwiseOperation elementwise_operation()
{
    static_assert(type == EltwiseBinaryType::ELWADD || type == EltwiseBinaryType::ELWSUB || type == EltwiseBinaryType::ELWMUL, "Unsupported eltwise binary type");
    if constexpr (type == EltwiseBinaryType::ELWADD)
    {
        return hal::fpu::ElementwiseOperation::Add;
    }
    else if constexpr (type == EltwiseBinaryType::ELWSUB)
    {
        return hal::fpu::ElementwiseOperation::Subtract;
    }
    else
    {
        return hal::fpu::ElementwiseOperation::Multiply;
    }
}

template <BroadcastType bcast>
constexpr hal::fpu::SrcBBroadcast srcb_broadcast()
{
    if constexpr (bcast == BroadcastType::COL)
    {
        return hal::fpu::SrcBBroadcast::Column;
    }
    else if constexpr (bcast == BroadcastType::ROW)
    {
        return hal::fpu::SrcBBroadcast::Row;
    }
    else if constexpr (bcast == BroadcastType::SCALAR)
    {
        return hal::fpu::SrcBBroadcast::Scalar;
    }
    else
    {
        return hal::fpu::SrcBBroadcast::None;
    }
}

constexpr std::uint32_t fidelity_phases(const MathFidelity math_fidelity)
{
    return static_cast<std::uint32_t>(math_fidelity) == 0 ? 1 : static_cast<std::uint32_t>(math_fidelity);
}

// SrcB rows advance with SrcA unless one SrcB row is broadcast.
template <BroadcastType bcast>
inline constexpr std::uint32_t SRCB_ROW_STEP = (bcast == BroadcastType::NONE || bcast == BroadcastType::COL) ? FPU_ROWS : 0;

template <EltwiseBinaryType type, BroadcastType bcast>
inline std::uint32_t elementwise_word(const std::uint32_t acc_to_dest, const std::uint8_t address_modifier, const hal::fpu::SourceRelease release)
{
    return hal::fpu::get_operation(hal::fpu::Elementwise {
        .operation        = elementwise_operation<type>(),
        .broadcast        = srcb_broadcast<bcast>(),
        .dest_write       = acc_to_dest ? hal::fpu::DestWriteMode::Accumulate : hal::fpu::DestWriteMode::Overwrite,
        .address_modifier = address_modifier,
        .release          = release,
    });
}

// Source banks released when a face is finished. A broadcast SrcB bank stays with math until
// every face that uses it is done.
template <BroadcastType bcast>
inline constexpr hal::fpu::SourceRelease FACE_RELEASE =
    (bcast == BroadcastType::SCALAR || bcast == BroadcastType::COL) ? hal::fpu::SourceRelease::SrcA : hal::fpu::SourceRelease::Both;

template <BroadcastType bcast, MathFidelity math_fidelity>
inline void eltwise_binary_configure_addrmods()
{
    constexpr std::uint32_t srcb_step = SRCB_ROW_STEP<bcast>;
    set_math_addr_mod<hal::cfg::Sec::S0, MathAddrMod {.srca_incr = FPU_ROWS, .srcb_incr = srcb_step, .dest_incr = FPU_ROWS}>();
    set_math_addr_mod<hal::cfg::Sec::S1, MathAddrMod {.srca_clear = 1, .srcb_clear = 1, .dest_incr = FPU_ROWS}>();
    set_math_addr_mod<hal::cfg::Sec::S2, MathAddrMod {.srca_clear = 1, .srcb_clear = 1, .dest_clear = 1}>();
    set_math_addr_mod<hal::cfg::Sec::S3, MathAddrMod {}>();
    if constexpr (fidelity_phases(math_fidelity) > 1)
    {
        set_math_addr_mod<hal::cfg::Sec::S4, MathAddrMod {.srca_clear = 1, .srcb_clear = 1, .dest_cr = 1, .fid_incr = 1}>();
        set_math_addr_mod<hal::cfg::Sec::S5, MathAddrMod {.srca_clear = 1, .srcb_clear = 1, .dest_incr = FPU_ROWS, .dest_c2cr = 1, .fid_clear = 1}>();
    }
}

// Whole-tile MOP: every face is two 8-row operations; the last operation of each face releases
// the consumed banks and the last operation of the tile rewinds Dest for the next tile.
template <EltwiseBinaryType type, BroadcastType bcast>
inline void eltwise_binary_configure_tile_mop(const std::uint32_t acc_to_dest, const std::uint32_t num_faces)
{
    using namespace hal::mop;
    using hal::fpu::SourceRelease;

    const std::uint32_t step      = elementwise_word<type, bcast>(acc_to_dest, AM_STEP, SourceRelease::None);
    const std::uint32_t face_end  = elementwise_word<type, bcast>(acc_to_dest, AM_FACE_END, FACE_RELEASE<bcast>);
    const std::uint32_t tile_end  = elementwise_word<type, bcast>(acc_to_dest, AM_TILE_END, SourceRelease::Both);
    const std::uint32_t inner_ops = FACE_R_DIM / FPU_ROWS;

    if constexpr (bcast == BroadcastType::COL)
    {
        if (num_faces >= 2)
        {
            // Each SrcB bank serves a pair of faces: the first face of the pair releases only SrcA,
            // the second releases both banks.
            const std::uint32_t pair_first_face_end = elementwise_word<type, bcast>(acc_to_dest, AM_FACE_END, SourceRelease::SrcA);
            const std::uint32_t pair_end            = elementwise_word<type, bcast>(acc_to_dest, AM_FACE_END, SourceRelease::Both);
            program(MopConfig<MopTemplate::Template1> {
                .outer_loop = {.count = num_faces / 2},
                .inner_loop =
                    {.count                               = inner_ops,
                     .body_op                             = step,
                     .alternating_op                      = pair_first_face_end,
                     .last_op_on_final_outer_iteration    = tile_end,
                     .last_op_on_nonfinal_outer_iteration = pair_end},
            });
            return;
        }
    }

    program(MopConfig<MopTemplate::Template1> {
        .outer_loop = {.count = num_faces},
        .inner_loop =
            {.count                               = inner_ops,
             .body_op                             = step,
             .last_op_on_final_outer_iteration    = tile_end,
             .last_op_on_nonfinal_outer_iteration = face_end},
    });
}

// Single-face MOP, run once per face: used with dest reuse and for high fidelity.
template <EltwiseBinaryType type, BroadcastType bcast, MathFidelity math_fidelity>
inline void eltwise_binary_configure_face_mop(const std::uint32_t acc_to_dest, const hal::fpu::SourceRelease release)
{
    using namespace hal::mop;
    using hal::fpu::SourceRelease;

    const std::uint32_t step      = elementwise_word<type, bcast>(acc_to_dest, AM_STEP, SourceRelease::None);
    const std::uint32_t inner_ops = FACE_R_DIM / FPU_ROWS;

    if constexpr (fidelity_phases(math_fidelity) > 1)
    {
        program(MopConfig<MopTemplate::Template1> {
            .outer_loop = {.count = fidelity_phases(math_fidelity)},
            .inner_loop =
                {.count                               = inner_ops,
                 .body_op                             = step,
                 .last_op_on_final_outer_iteration    = elementwise_word<type, bcast>(acc_to_dest, AM_FID_DONE, release),
                 .last_op_on_nonfinal_outer_iteration = elementwise_word<type, bcast>(acc_to_dest, AM_FID_NEXT, SourceRelease::None)},
        });
    }
    else
    {
        program(MopConfig<MopTemplate::Template1> {
            .outer_loop = {.count = 1},
            .inner_loop = {.count = inner_ops, .body_op = step, .last_op_on_final_outer_iteration = elementwise_word<type, bcast>(acc_to_dest, AM_FACE_END, release)},
        });
    }
}

template <EltwiseBinaryType type, MathFidelity math_fidelity, EltwiseBinaryReuseDestType reuse>
inline constexpr bool USES_FACE_MOP = (reuse != EltwiseBinaryReuseDestType::NONE) || (type == EltwiseBinaryType::ELWMUL && fidelity_phases(math_fidelity) > 1);

// Copy the current Dest face into the source register that reuse_dest selects (rows 0-15).
template <EltwiseBinaryReuseDestType reuse>
inline void move_dest_face_to_src()
{
    namespace sync = hal::sync;
    namespace move = hal::move;
    if constexpr (reuse == EltwiseBinaryReuseDestType::DEST_TO_SRCA)
    {
        // MOVD2A does not wait for SrcA ownership; also let earlier Dest writes settle.
        sync::wait::stall<sync::StallTarget::Math, sync::StallCondition::SrcAValid | sync::StallCondition::MathIdle>();
        // SrcA rows are in the math view (0-15), the same rows the FPU reads.
        move::run<move::Transfer<move::Dst, move::SrcA> {.source_row = 0, .destination_row = 0, .address_mode = AM_HOLD, .number_of_rows = move::rows::Four}>();
        move::run<move::Transfer<move::Dst, move::SrcA> {.source_row = 4, .destination_row = 4, .address_mode = AM_HOLD, .number_of_rows = move::rows::Four}>();
        move::run<move::Transfer<move::Dst, move::SrcA> {.source_row = 8, .destination_row = 8, .address_mode = AM_HOLD, .number_of_rows = move::rows::Four}>();
        move::run<move::Transfer<move::Dst, move::SrcA> {.source_row = 12, .destination_row = 12, .address_mode = AM_HOLD, .number_of_rows = move::rows::Four}>();
    }
    else if constexpr (reuse == EltwiseBinaryReuseDestType::DEST_TO_SRCB)
    {
        sync::wait::stall<sync::StallTarget::Math, sync::StallCondition::SrcBValid | sync::StallCondition::MathIdle>();
        move::run<move::Transfer<move::Dst, move::SrcB> {.source_row = 0, .destination_row = 0, .address_mode = AM_HOLD, .number_of_rows = move::rows::Four}>();
        move::run<move::Transfer<move::Dst, move::SrcB> {.source_row = 4, .destination_row = 4, .address_mode = AM_HOLD, .number_of_rows = move::rows::Four}>();
        move::run<move::Transfer<move::Dst, move::SrcB> {.source_row = 8, .destination_row = 8, .address_mode = AM_HOLD, .number_of_rows = move::rows::Four}>();
        move::run<move::Transfer<move::Dst, move::SrcB> {.source_row = 12, .destination_row = 12, .address_mode = AM_HOLD, .number_of_rows = move::rows::Four}>();
    }
}

} // namespace hal_llk

template <
    EltwiseBinaryType eltwise_binary_type,
    BroadcastType src_b_bcast_type,
    MathFidelity math_fidelity                   = MathFidelity::LoFi,
    EltwiseBinaryReuseDestType binary_reuse_dest = EltwiseBinaryReuseDestType::NONE>
inline void _llk_math_eltwise_binary_init_(const ckernel::TensorShape tensor_shape, const std::uint32_t acc_to_dest)
{
    namespace cfg = hal::cfg;
    using hal::fpu::SourceRelease;
    static_assert(
        binary_reuse_dest == EltwiseBinaryReuseDestType::NONE || src_b_bcast_type == BroadcastType::NONE, "Dest reuse is implemented without SrcB broadcast");

    hal_llk::eltwise_binary_configure_addrmods<src_b_bcast_type, math_fidelity>();

    if constexpr (hal_llk::USES_FACE_MOP<eltwise_binary_type, math_fidelity, binary_reuse_dest>)
    {
        hal_llk::eltwise_binary_configure_face_mop<eltwise_binary_type, src_b_bcast_type, math_fidelity>(
            acc_to_dest, hal_llk::FACE_RELEASE<src_b_bcast_type>);
    }
    else
    {
        hal_llk::eltwise_binary_configure_tile_mop<eltwise_binary_type, src_b_bcast_type>(acc_to_dest, tensor_shape.total_num_faces());
    }

    // Releasing a bank must also clear its data-valid so the unpacker can refill it.
    cfg::write<cfg::Access::TensixCfgUnit>(cfg::set<cfg::ClrDvalid::SrcA_Disable, cfg::Sec::S0, 0>(), cfg::set<cfg::ClrDvalid::SrcB_Disable, cfg::Sec::S0, 0>());

    hal::math_counters.set<hal::rwc::Counters::All, 0>().clear_fidelity().apply();
}

template <
    EltwiseBinaryType eltwise_binary_type,
    BroadcastType src_b_bcast_type,
    DstSync Dst,
    bool is_fp32_dest_acc_en,
    MathFidelity math_fidelity                   = MathFidelity::LoFi,
    EltwiseBinaryReuseDestType binary_reuse_dest = EltwiseBinaryReuseDestType::NONE>
inline void _llk_math_eltwise_binary_(const ckernel::TensorShape tensor_shape, std::uint32_t dst_index, [[maybe_unused]] const bool clear_fp32_dst_acc = false)
{
    using Runner = hal::mop::Runner<hal::mop::MopTemplate::Template1>;

    const std::uint32_t tile_row = hal_llk::dest_section_base() + dst_index * hal_llk::DEST_ROWS_PER_TILE;
    hal_llk::set_math_dest_offset(tile_row);

    if constexpr (!hal_llk::USES_FACE_MOP<eltwise_binary_type, math_fidelity, binary_reuse_dest>)
    {
        Runner::run();
    }
    else
    {
        const std::uint32_t num_faces = tensor_shape.total_num_faces();
        for (std::uint32_t face = 0; face < num_faces; face++)
        {
            if constexpr (binary_reuse_dest != EltwiseBinaryReuseDestType::NONE)
            {
                hal_llk::move_dest_face_to_src<binary_reuse_dest>();
                if constexpr (eltwise_binary_type == EltwiseBinaryType::ELWMUL)
                {
                    // ELWMUL accumulates; the face it overwrites must read back as zero.
                    hal::dst::run(hal::dst::Zero {
                        .scope   = hal::dst::ZeroScope::Face,
                        .index   = tile_row / FACE_R_DIM + face,
                        .width   = is_fp32_dest_acc_en ? hal::dst::DestWidth::Bits32 : hal::dst::DestWidth::Bits16,
                    });
                }
            }
            Runner::run();
            if constexpr (src_b_bcast_type == BroadcastType::SCALAR || src_b_bcast_type == BroadcastType::COL)
            {
                const bool srcb_done = (face + 1 == num_faces) || (src_b_bcast_type == BroadcastType::COL && (face % 2) == 1);
                if (srcb_done)
                {
                    hal::src<hal::SrcB>.release();
                }
            }
        }
        hal::math_counters.set<hal::rwc::Counters::Dest, 0>().apply();
    }
}

inline void _llk_math_eltwise_binary_uninit_()
{
}
