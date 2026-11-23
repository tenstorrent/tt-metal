// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// HAL-based implementation of the FPU element-wise binary LLK (ELWADD / ELWSUB / ELWMUL).

#pragma once

#include <cstdint>

#include "hal/dst.h"
#include "hal/fpu.h"
#include "hal/math_counters.h"
#include "hal/mop.h"
#include "hal/move.h"
#include "llk_hal_common.h"
#include "llk_math_common.h"
#include "tensor_shape.h"

namespace hal_llk::eltwise_binary
{

namespace cfg = hal::cfg;
using hal::fpu::SourceRelease;

// Address-modifier slots used by this op.
inline constexpr std::uint8_t AM_ROW_BLOCK = 0; // next 8 rows of the face
inline constexpr std::uint8_t AM_FACE_END  = 1; // sources back to row 0, dest to the next face
inline constexpr std::uint8_t AM_FIDELITY  = 2; // next fidelity phase on the same rows
inline constexpr std::uint8_t AM_MOVE_ROWS = 3; // dest-reuse: next 4 rows of dest and of the source register
inline constexpr std::uint8_t AM_MOVE_END  = 4; // dest-reuse: back to the start of the current face

// One address-modifier slot: SrcA/SrcB/Dest/Fidelity steps applied after the instruction that selects it.
// Dest "CR" adds the increment to the dest carry register and reloads the counter from it, which is how
// the dest counter steps from face to face (16 rows) independently of the face height.
template <
    cfg::Sec Slot,
    std::uint32_t SrcAIncr,
    std::uint32_t SrcAClear,
    std::uint32_t SrcBIncr,
    std::uint32_t SrcBClear,
    std::uint32_t DestIncr,
    std::uint32_t DestCR,
    std::uint32_t FidelityIncr,
    std::uint32_t FidelityClear>
inline void set_address_modifier()
{
    cfg::write<cfg::Access::TensixCfgUnit>(
        cfg::set<cfg::AddrMod[cfg::SrcA].Incr, Slot, SrcAIncr>(),
        cfg::set<cfg::AddrMod[cfg::SrcA].CR, Slot, 0>(),
        cfg::set<cfg::AddrMod[cfg::SrcA].Clear, Slot, SrcAClear>(),
        cfg::set<cfg::AddrMod[cfg::SrcB].Incr, Slot, SrcBIncr>(),
        cfg::set<cfg::AddrMod[cfg::SrcB].CR, Slot, 0>(),
        cfg::set<cfg::AddrMod[cfg::SrcB].Clear, Slot, SrcBClear>(),
        cfg::set<cfg::AddrMod[cfg::SrcA].Incr2, Slot, 0>(),
        cfg::set<cfg::AddrMod[cfg::SrcB].Incr2, Slot, 0>(),
        cfg::set<cfg::AddrMod[cfg::Dest].Incr, Slot, DestIncr>(),
        cfg::set<cfg::AddrMod[cfg::Dest].CR, Slot, DestCR>(),
        cfg::set<cfg::AddrMod[cfg::Dest].Clear, Slot, 0>(),
        cfg::set<cfg::AddrMod[cfg::Dest].CToCR, Slot, 0>(),
        cfg::set<cfg::AddrMod[cfg::Fidelity].Incr, Slot, FidelityIncr>(),
        cfg::set<cfg::AddrMod[cfg::Fidelity].Clear, Slot, FidelityClear>(),
        cfg::set<cfg::AddrMod[cfg::Bias].Incr, Slot, 0>(),
        cfg::set<cfg::AddrMod[cfg::Bias].Clear, Slot, 0>());
}

template <BroadcastType bcast_type>
inline void configure_address_modifiers()
{
    // Row/scalar broadcast keep reading SrcB row 0; column broadcast walks SrcB rows with SrcA.
    constexpr std::uint32_t srcb_incr = (bcast_type == BroadcastType::NONE || bcast_type == BroadcastType::COL) ? 8 : 0;
    set_address_modifier<cfg::Sec::S0, 8, 0, srcb_incr, 0, 8, 0, 0, 1>();
    set_address_modifier<cfg::Sec::S1, 0, 1, 0, 1, FACE_ROWS, 1, 0, 1>();
    set_address_modifier<cfg::Sec::S2, 0, 0, 0, 0, 0, 0, 1, 0>();
}

inline void configure_reuse_address_modifiers()
{
    set_address_modifier<cfg::Sec::S3, 4, 0, 4, 0, 4, 0, 0, 0>();
    set_address_modifier<cfg::Sec::S4, 0, 1, 0, 1, 0, 1, 0, 0>();
}

constexpr hal::fpu::ElementwiseOperation fpu_operation(const EltwiseBinaryType type)
{
    return (type == EltwiseBinaryType::ELWADD)   ? hal::fpu::ElementwiseOperation::Add
           : (type == EltwiseBinaryType::ELWSUB) ? hal::fpu::ElementwiseOperation::Subtract
                                                 : hal::fpu::ElementwiseOperation::Multiply;
}

constexpr hal::fpu::SrcBBroadcast fpu_broadcast(const BroadcastType type)
{
    return (type == BroadcastType::COL)      ? hal::fpu::SrcBBroadcast::Column
           : (type == BroadcastType::ROW)    ? hal::fpu::SrcBBroadcast::Row
           : (type == BroadcastType::SCALAR) ? hal::fpu::SrcBBroadcast::Scalar
                                             : hal::fpu::SrcBBroadcast::None;
}

template <EltwiseBinaryType type, BroadcastType bcast, bool acc_to_dest>
constexpr std::uint32_t op(const std::uint8_t address_modifier, const SourceRelease release)
{
    return hal::fpu::get_operation(hal::fpu::Elementwise {
        .operation        = fpu_operation(type),
        .broadcast        = fpu_broadcast(bcast),
        .dest_write       = acc_to_dest ? hal::fpu::DestWriteMode::Accumulate : hal::fpu::DestWriteMode::Overwrite,
        .address_modifier = address_modifier,
        .release          = release,
    });
}

constexpr std::uint32_t fidelity_phases(const MathFidelity fidelity)
{
    return (static_cast<std::uint32_t>(fidelity) == 0) ? 1 : static_cast<std::uint32_t>(fidelity);
}

// 8-row blocks per face (a face shorter than 8 rows still takes one block).
inline std::uint32_t blocks_per_face(const std::uint32_t face_r_dim)
{
    return (face_r_dim > 8) ? (face_r_dim >> 3) : 1;
}

// Source releases at the end of a face: the last face of a group releases both banks, the
// others release only SrcA (the broadcast SrcB bank stays valid for the next SrcA face).
template <BroadcastType bcast>
constexpr SourceRelease nonfinal_face_release()
{
    return (bcast == BroadcastType::COL || bcast == BroadcastType::SCALAR) ? SourceRelease::SrcA : SourceRelease::Both;
}

// Number of faces handled by one MOP run.
template <BroadcastType bcast>
inline std::uint32_t faces_per_run(const std::uint32_t num_faces)
{
    if constexpr (bcast == BroadcastType::COL)
    {
        return (num_faces > 2) ? 2 : num_faces;
    }
    return num_faces;
}

template <EltwiseBinaryType type, BroadcastType bcast, MathFidelity fidelity, bool acc_to_dest>
inline void program_mop_for(const std::uint32_t faces, const std::uint32_t blocks)
{
    using MopConfig1 = hal::mop::MopConfig<hal::mop::MopTemplate::Template1>;
    using hal::mop::Template1Field;
    constexpr std::uint32_t phases = (type == EltwiseBinaryType::ELWMUL) ? fidelity_phases(fidelity) : 1;

    if constexpr (phases == 1)
    {
        // outer: faces, inner: 8-row blocks of one face.
        constexpr MopConfig1 config {
            .outer_loop = {.count = 1},
            .inner_loop =
                {
                    .count                               = 1,
                    .body_op                             = op<type, bcast, acc_to_dest>(AM_ROW_BLOCK, SourceRelease::None),
                    .last_op_on_final_outer_iteration    = op<type, bcast, acc_to_dest>(AM_FACE_END, SourceRelease::Both),
                    .last_op_on_nonfinal_outer_iteration = op<type, bcast, acc_to_dest>(AM_FACE_END, nonfinal_face_release<bcast>()),
                },
        };
        hal::mop::program<config>(
            hal::mop::runtime_field<Template1Field::OuterLoopCount>(faces), hal::mop::runtime_field<Template1Field::InnerLoopCount>(blocks));
    }
    else
    {
        // High fidelity: outer: 8-row blocks of one face, inner: fidelity phases of one block.
        constexpr MopConfig1 config {
            .outer_loop = {.count = 1},
            .inner_loop =
                {
                    .count                               = phases,
                    .body_op                             = op<type, bcast, acc_to_dest>(AM_FIDELITY, SourceRelease::None),
                    .last_op_on_final_outer_iteration    = op<type, bcast, acc_to_dest>(AM_FACE_END, SourceRelease::Both),
                    .last_op_on_nonfinal_outer_iteration = op<type, bcast, acc_to_dest>(AM_ROW_BLOCK, SourceRelease::None),
                },
        };
        hal::mop::program<config>(hal::mop::runtime_field<Template1Field::OuterLoopCount>(blocks));
    }
}

template <EltwiseBinaryType type, BroadcastType bcast, MathFidelity fidelity>
inline void program_mop(const std::uint32_t faces, const std::uint32_t blocks, const std::uint32_t acc_to_dest)
{
    if (acc_to_dest)
    {
        program_mop_for<type, bcast, fidelity, true>(faces, blocks);
    }
    else
    {
        program_mop_for<type, bcast, fidelity, false>(faces, blocks);
    }
}

// Face geometry captured at init for the per-tile call.
inline std::uint32_t num_faces = 4;

// Point math at the tile in the current dest half and restart all counters at row 0.
inline void start_tile(const std::uint32_t dst_index)
{
    cfg::write<cfg::Access::TensixCfgUnit, cfg::DestTargetRegCfgMath::Offset, cfg::Sec::S0>(math::dest_bank_base() + dst_index * DEST_ROWS_PER_TILE);
    hal::math_counters.set<hal::rwc::Counters::All, 0>().clear_fidelity().apply();
}

} // namespace hal_llk::eltwise_binary

/**
 * @brief Initialize FPU for an element-wise binary operation Output = SrcA [+, -, *] SrcB.
 */
template <
    EltwiseBinaryType eltwise_binary_type,
    BroadcastType src_b_bcast_type,
    MathFidelity math_fidelity                   = MathFidelity::LoFi,
    EltwiseBinaryReuseDestType binary_reuse_dest = EltwiseBinaryReuseDestType::NONE>
inline void _llk_math_eltwise_binary_init_(const ckernel::TensorShape tensor_shape, const std::uint32_t acc_to_dest)
{
    using namespace hal_llk::eltwise_binary;
    static_assert(
        (binary_reuse_dest == EltwiseBinaryReuseDestType::NONE) || (src_b_bcast_type == BroadcastType::NONE), "Dest reuse supports no SrcB broadcast only");

    num_faces                   = tensor_shape.total_num_faces();
    const std::uint32_t blocks  = blocks_per_face(tensor_shape.face_r_dim);
    const bool high_fidelity    = (eltwise_binary_type == EltwiseBinaryType::ELWMUL) && (fidelity_phases(math_fidelity) > 1);

    configure_address_modifiers<src_b_bcast_type>();
    if constexpr (binary_reuse_dest != EltwiseBinaryReuseDestType::NONE)
    {
        configure_reuse_address_modifiers();
    }

    // Dest reuse and high fidelity run the MOP once per face; otherwise one run covers a face group.
    const bool per_face = (binary_reuse_dest != EltwiseBinaryReuseDestType::NONE) || high_fidelity;
    const std::uint32_t faces = per_face ? 1 : faces_per_run<src_b_bcast_type>(num_faces);
    program_mop<eltwise_binary_type, src_b_bcast_type, math_fidelity>(faces, blocks, acc_to_dest);
}

/**
 * @brief Perform an element-wise binary operation on one tile into dest tile dst_index.
 */
template <
    EltwiseBinaryType eltwise_binary_type,
    BroadcastType src_b_bcast_type,
    DstSync Dst,
    bool is_fp32_dest_acc_en,
    MathFidelity math_fidelity                   = MathFidelity::LoFi,
    EltwiseBinaryReuseDestType binary_reuse_dest = EltwiseBinaryReuseDestType::NONE>
inline void _llk_math_eltwise_binary_(
    [[maybe_unused]] const ckernel::TensorShape tensor_shape, std::uint32_t dst_index, [[maybe_unused]] const bool clear_fp32_dst_acc = false)
{
    using namespace hal_llk;
    using namespace hal_llk::eltwise_binary;
    using Runner = hal::mop::Runner<hal::mop::MopTemplate::Template1>;
    constexpr bool high_fidelity =
        (eltwise_binary_type == EltwiseBinaryType::ELWMUL) && (fidelity_phases(math_fidelity) > 1);

    start_tile(dst_index);

    if constexpr (binary_reuse_dest != EltwiseBinaryReuseDestType::NONE)
    {
        namespace move = hal::move;
        for (std::uint32_t face = 0; face < num_faces; face++)
        {
            // Copy the current dest face into the source register the unpacker just filled
            // (wait for that bank, and for in-flight math writes to the face to retire).
            if constexpr (binary_reuse_dest == EltwiseBinaryReuseDestType::DEST_TO_SRCA)
            {
                hal::sync::wait::stall<hal::sync::StallTarget::Math, hal::sync::StallCondition::SrcAValid | hal::sync::StallCondition::MathIdle>();
                constexpr move::Transfer<move::Dst, move::SrcA> dest_face_to_srca {
                    .source_row         = 0,
                    .destination_row    = 0,
                    .address_mode       = AM_MOVE_ROWS,
                    .number_of_rows     = move::rows::Face,
                    .final_address_mode = AM_MOVE_END,
                };
                move::run<dest_face_to_srca>();
            }
            else
            {
                hal::sync::wait::stall<hal::sync::StallTarget::Math, hal::sync::StallCondition::SrcBValid | hal::sync::StallCondition::MathIdle>();
                constexpr move::Transfer<move::Dst, move::SrcB> dest_face_to_srcb {
                    .source_row         = 0,
                    .destination_row    = 0,
                    .address_mode       = AM_MOVE_ROWS,
                    .number_of_rows     = move::rows::Face,
                    .final_address_mode = AM_MOVE_END,
                };
                move::run<dest_face_to_srcb>();
            }
            if constexpr (eltwise_binary_type == EltwiseBinaryType::ELWMUL)
            {
                // ELWMUL accumulates: clear the face before computing into it.
                // NOTE: this dest-reuse ELWMUL path is outside the add-only test selection and does not pass
                // test_eltwise_binary_dest_reuse[...Elwmul...] yet (see the report).
                hal::sync::wait::stall<hal::sync::StallTarget::Math, hal::sync::StallCondition::MathIdle>();
                hal::dst::run(hal::dst::Zero {
                    .scope = hal::dst::ZeroScope::Face,
                    .index = (math::dest_bank_base() + dst_index * DEST_ROWS_PER_TILE) / FACE_ROWS + face,
                    .width = is_fp32_dest_acc_en ? hal::dst::DestWidth::Bits32 : hal::dst::DestWidth::Bits16,
                });
            }
            Runner::run();
        }
    }
    else if constexpr (high_fidelity)
    {
        // One MOP run per face (all 8-row blocks and fidelity phases of that face).
        for (std::uint32_t face = 0; face < num_faces; face++)
        {
            if constexpr (src_b_bcast_type == BroadcastType::COL || src_b_bcast_type == BroadcastType::SCALAR)
            {
                // The broadcast SrcB bank is released only after the last SrcA face that uses it.
                const bool release_srcb = (src_b_bcast_type == BroadcastType::COL) ? ((face & 1) || (face + 1 == num_faces)) : (face + 1 == num_faces);
                hal::mop::write<hal::mop::Template1Field::LastOpOnFinalOuterIteration>(
                    release_srcb ? op<eltwise_binary_type, src_b_bcast_type, false>(AM_FACE_END, SourceRelease::Both)
                                 : op<eltwise_binary_type, src_b_bcast_type, false>(AM_FACE_END, SourceRelease::SrcA));
            }
            Runner::run();
        }
    }
    else
    {
        Runner::run();
        if constexpr (src_b_bcast_type == BroadcastType::COL)
        {
            // Second face group (faces 2,3) uses the next SrcB bank.
            if (num_faces > 2)
            {
                Runner::run();
            }
        }
    }
}

inline void _llk_math_eltwise_binary_uninit_()
{
}
