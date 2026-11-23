// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// HAL-based implementation of the Blackhole FPU elementwise binary LLK functions.

#pragma once

#include <cstdint>

#include "ckernel.h"
#include "ckernel_defs.h"
#include "ckernel_globals.h"
#include "hal/cfg.h"
#include "hal/fpu.h"
#include "hal/math_counters.h"
#include "hal/mop.h"
#include "hal/move.h"
#include "hal/sync.h"
#include "llk_defs.h"
#include "llk_math_common.h"
#include "tensor_shape.h"

namespace hal_llk::eltwise_binary
{

using hal::cfg::Sec;

/// ADDR_MOD slots used by the eltwise binary MOPs.
inline constexpr std::uint8_t AddrModRowStep  = 0; // next 8-row block of the same face
inline constexpr std::uint8_t AddrModFaceEnd  = 1; // last block of a face: rewind sources, step Dest to the next face
inline constexpr std::uint8_t AddrModMoveStep = 2; // dest reuse: next 4 rows of the Dest -> Src move
inline constexpr std::uint8_t AddrModMoveEnd  = 3; // dest reuse: rewind Dest and the reused source after the move

/// Number of 8-row FPU blocks that cover one face.
inline std::uint32_t blocks_per_face(const std::uint32_t face_r_dim)
{
    return (face_r_dim > 8) ? 2 : 1;
}

template <EltwiseBinaryType type>
inline constexpr hal::fpu::ElementwiseOperation fpu_operation()
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
inline constexpr hal::fpu::SrcBBroadcast fpu_broadcast()
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

/// One 8x16 elementwise operation word.
template <EltwiseBinaryType type, BroadcastType bcast>
inline std::uint32_t operation(const std::uint32_t acc_to_dest, const std::uint8_t address_modifier, const hal::fpu::SourceRelease release)
{
    return hal::fpu::get_operation(hal::fpu::Elementwise {
        .operation        = fpu_operation<type>(),
        .broadcast        = fpu_broadcast<bcast>(),
        .dest_write       = acc_to_dest ? hal::fpu::DestWriteMode::Accumulate : hal::fpu::DestWriteMode::Overwrite,
        .address_modifier = address_modifier,
        .release          = release,
    });
}

/// Program one math ADDR_MOD slot (SrcA/SrcB/Dest/fidelity counter updates applied after an instruction).
template <Sec S>
inline void set_addr_mod(
    const std::uint32_t srca_incr,
    const std::uint32_t srca_clear,
    const std::uint32_t srcb_incr,
    const std::uint32_t srcb_clear,
    const std::uint32_t dest_incr,
    const std::uint32_t dest_clear)
{
    namespace cfg = hal::cfg;
    cfg::write<cfg::Access::TensixCfgUnit>(
        cfg::set<cfg::AddrMod[cfg::SrcA].Incr, S>(srca_incr),
        cfg::set<cfg::AddrMod[cfg::SrcA].CR, S, 0>(),
        cfg::set<cfg::AddrMod[cfg::SrcA].Clear, S>(srca_clear),
        cfg::set<cfg::AddrMod[cfg::SrcB].Incr, S>(srcb_incr),
        cfg::set<cfg::AddrMod[cfg::SrcB].CR, S, 0>(),
        cfg::set<cfg::AddrMod[cfg::SrcB].Clear, S>(srcb_clear));
    cfg::write<cfg::Access::TensixCfgUnit>(
        cfg::set<cfg::AddrMod[cfg::Dest].Incr, S>(dest_incr),
        cfg::set<cfg::AddrMod[cfg::Dest].CR, S, 0>(),
        cfg::set<cfg::AddrMod[cfg::Dest].Clear, S>(dest_clear),
        cfg::set<cfg::AddrMod[cfg::Dest].CToCR, S, 0>(),
        cfg::set<cfg::AddrMod[cfg::Fidelity].Incr, S, 0>(),
        cfg::set<cfg::AddrMod[cfg::Fidelity].Clear, S, 0>());
}

/// Source-register format of data moved back from a 16-bit Dest: block-float formats become the 16-bit float of
/// the same exponent family; every other format is kept.
inline std::uint32_t dest_float_format(const std::uint32_t format)
{
    switch (format)
    {
        case static_cast<std::uint32_t>(DataFormat::Bfp8_b):
        case static_cast<std::uint32_t>(DataFormat::Bfp4_b):
        case static_cast<std::uint32_t>(DataFormat::Bfp2_b):
            return static_cast<std::uint32_t>(DataFormat::Float16_b);
        case static_cast<std::uint32_t>(DataFormat::Bfp8):
        case static_cast<std::uint32_t>(DataFormat::Bfp4):
        case static_cast<std::uint32_t>(DataFormat::Bfp2):
            return static_cast<std::uint32_t>(DataFormat::Float16);
        default:
            return format;
    }
}

/// Rewind SrcA, SrcB and Dest counters and the fidelity phase.
inline void reset_counters()
{
    hal::math_counters.set<hal::rwc::Counters::All, 0>().clear_fidelity().apply();
}

} // namespace hal_llk::eltwise_binary

template <EltwiseBinaryType eltwise_binary_type, BroadcastType src_b_bcast_type, MathFidelity math_fidelity = MathFidelity::LoFi>
inline void _llk_math_eltwise_binary_standard_init_(const ckernel::TensorShape tensor_shape, const std::uint32_t acc_to_dest)
{
    using namespace hal_llk::eltwise_binary;
    using hal::fpu::SourceRelease;
    static_assert(math_fidelity == MathFidelity::LoFi, "HAL eltwise binary implements LoFi only (no fidelity phases)");

    const std::uint32_t face_r_dim  = tensor_shape.face_r_dim;
    const std::uint32_t num_faces_c = tensor_shape.num_faces_c_dim;
    const std::uint32_t num_faces   = tensor_shape.total_num_faces();
    const std::uint32_t blocks      = blocks_per_face(face_r_dim);

    // Sources are consumed in normal (non-reused) mode: let the unpacker-implied formats apply, flush src zero flags.
    hal::sync::wait::stall<hal::sync::StallTarget::Config, hal::sync::StallCondition::MathIdle>();
    hal::cfg::write<hal::cfg::Access::TensixCfgUnit, hal::cfg::DisableImpliedFmt[hal::cfg::SrcA], Sec::S0, 0>();
    hal::cfg::write<hal::cfg::Access::TensixCfgUnit, hal::cfg::DisableImpliedFmt[hal::cfg::SrcB], Sec::S0, 0>();
    hal::cfg::write<hal::cfg::Access::TensixCfgUnit, hal::cfg::AluAccCtrl::Zero_Flag_disabled_src, Sec::S0, 0>();

    // Row/scalar broadcast read one SrcB row for the whole face, so SrcB does not advance.
    constexpr std::uint32_t srcb_row_step = (src_b_bcast_type == BroadcastType::NONE || src_b_bcast_type == BroadcastType::COL) ? 8 : 0;
    set_addr_mod<Sec::S0>(8, 0, srcb_row_step, 0, 8, 0);
    // Every face occupies 16 Dest rows, whatever its height.
    set_addr_mod<Sec::S1>(0, 1, 0, 1, (blocks == 2) ? 8 : 16, 0);

    using Config = hal::mop::MopConfig<hal::mop::MopTemplate::Template1>;
    Config config {};
    config.inner_loop.count   = blocks;
    config.inner_loop.body_op = operation<eltwise_binary_type, src_b_bcast_type>(acc_to_dest, AddrModRowStep, SourceRelease::None);
    if constexpr (src_b_bcast_type == BroadcastType::COL)
    {
        // One SrcB face serves a whole row of faces: run once per face row.
        config.outer_loop.count = num_faces_c;
        config.inner_loop.last_op_on_nonfinal_outer_iteration =
            operation<eltwise_binary_type, src_b_bcast_type>(acc_to_dest, AddrModFaceEnd, SourceRelease::SrcA);
        config.inner_loop.last_op_on_final_outer_iteration = operation<eltwise_binary_type, src_b_bcast_type>(acc_to_dest, AddrModFaceEnd, SourceRelease::Both);
    }
    else if constexpr (src_b_bcast_type == BroadcastType::SCALAR)
    {
        // One SrcB face serves the whole tile.
        config.outer_loop.count = num_faces;
        config.inner_loop.last_op_on_nonfinal_outer_iteration =
            operation<eltwise_binary_type, src_b_bcast_type>(acc_to_dest, AddrModFaceEnd, SourceRelease::SrcA);
        config.inner_loop.last_op_on_final_outer_iteration = operation<eltwise_binary_type, src_b_bcast_type>(acc_to_dest, AddrModFaceEnd, SourceRelease::Both);
    }
    else
    {
        // Every SrcA face comes with its own SrcB face.
        const std::uint32_t face_end = operation<eltwise_binary_type, src_b_bcast_type>(acc_to_dest, AddrModFaceEnd, SourceRelease::Both);
        config.outer_loop.count                                = num_faces;
        config.inner_loop.last_op_on_nonfinal_outer_iteration = face_end;
        config.inner_loop.last_op_on_final_outer_iteration    = face_end;
    }
    hal::mop::program(config);

    reset_counters();
}

template <
    EltwiseBinaryType eltwise_binary_type,
    BroadcastType src_b_bcast_type,
    DstSync Dst,
    bool is_fp32_dest_acc_en,
    MathFidelity math_fidelity = MathFidelity::LoFi>
inline void _llk_math_eltwise_binary_standard_(const ckernel::TensorShape tensor_shape, std::uint32_t dst_index)
{
    using namespace hal;
    // Thread-CFG word: one SETC16 per tile.
    cfg::write<cfg::Access::TensixCfgUnit, cfg::DestTargetRegCfgMath::Offset, cfg::Sec::S0>(
        hal_llk::math::dest_section_base() + dst_index * hal_llk::math::DestRowsPerTile);

    if constexpr (src_b_bcast_type == BroadcastType::COL)
    {
        for (std::uint32_t face_row = 0; face_row < tensor_shape.num_faces_r_dim; face_row++)
        {
            hal::mop::Runner<hal::mop::MopTemplate::Template1>::run();
        }
    }
    else
    {
        hal::mop::Runner<hal::mop::MopTemplate::Template1>::run();
    }

    hal_llk::eltwise_binary::reset_counters();
}

template <
    EltwiseBinaryType eltwise_binary_type,
    BroadcastType src_b_bcast_type,
    MathFidelity math_fidelity                   = MathFidelity::LoFi,
    EltwiseBinaryReuseDestType binary_reuse_dest = EltwiseBinaryReuseDestType::DEST_TO_SRCA>
inline void _llk_math_eltwise_binary_with_dest_reuse_init_(const ckernel::TensorShape tensor_shape, const std::uint32_t acc_to_dest)
{
    using namespace hal_llk::eltwise_binary;
    using hal::fpu::SourceRelease;
    static_assert(binary_reuse_dest != EltwiseBinaryReuseDestType::NONE, "Use the standard init without dest reuse");
    static_assert(src_b_bcast_type == BroadcastType::NONE, "HAL dest reuse implements no-broadcast only");
    static_assert(eltwise_binary_type != EltwiseBinaryType::ELWMUL, "HAL dest reuse implements ELWADD/ELWSUB (overwrite) only");
    static_assert(math_fidelity == MathFidelity::LoFi, "HAL eltwise binary implements LoFi only");

    constexpr bool to_srca           = binary_reuse_dest == EltwiseBinaryReuseDestType::DEST_TO_SRCA;
    const std::uint32_t face_r_dim   = tensor_shape.face_r_dim;
    const std::uint32_t blocks       = blocks_per_face(face_r_dim);
    const std::uint32_t move_quads   = (face_r_dim > 4) ? (face_r_dim / 4) : 1;

    // Data moved from Dest carries no unpacker-implied format: the ALU uses the configured source format instead.
    // Dest holds 16-bit floats for block-float operands, so a BFP source format becomes the matching 16-bit float.
    hal::sync::wait::stall<hal::sync::StallTarget::Config, hal::sync::StallCondition::MathIdle>();
    if constexpr (to_srca)
    {
        hal::cfg::write<hal::cfg::Access::TensixCfgUnit, hal::cfg::AluFormatSpecReg0::SrcA, Sec::S0>(dest_float_format(hal_llk::math::srca_format));
    }
    else
    {
        hal::cfg::write<hal::cfg::Access::TensixCfgUnit, hal::cfg::AluFormatSpecReg1::SrcB, Sec::S0>(dest_float_format(hal_llk::math::srcb_format));
    }
    hal::cfg::write<hal::cfg::Access::TensixCfgUnit, hal::cfg::DisableImpliedFmt[hal::cfg::SrcA], Sec::S0, to_srca ? 1 : 0>();
    hal::cfg::write<hal::cfg::Access::TensixCfgUnit, hal::cfg::DisableImpliedFmt[hal::cfg::SrcB], Sec::S0, to_srca ? 0 : 1>();
    hal::cfg::write<hal::cfg::Access::TensixCfgUnit, hal::cfg::AluAccCtrl::Zero_Flag_disabled_src, Sec::S0, 0>();

    // Each face is computed at Dest counter 0 (the face base is in the math Dest offset).
    set_addr_mod<Sec::S0>(8, 0, 8, 0, 8, 0);
    set_addr_mod<Sec::S1>(0, 1, 0, 1, 0, 1);
    set_addr_mod<Sec::S2>(to_srca ? 4 : 0, 0, to_srca ? 0 : 4, 0, 4, 0);
    set_addr_mod<Sec::S3>(0, to_srca ? 1 : 0, 0, to_srca ? 0 : 1, 0, 1);

    std::uint32_t move_step;
    std::uint32_t move_end;
    if constexpr (to_srca)
    {
        hal::move::Transfer<hal::move::Dst, hal::move::SrcA> move {
            .source_row      = 0,
            .destination_row = 0,
            .address_mode    = AddrModMoveStep,
            .number_of_rows  = hal::move::rows::Four,
        };
        move_step         = move.get_operation();
        move.address_mode = AddrModMoveEnd;
        move_end          = move.get_operation();
    }
    else
    {
        hal::move::Transfer<hal::move::Dst, hal::move::SrcB> move {
            .source_row      = 0,
            .destination_row = 0,
            .address_mode    = AddrModMoveStep,
            .number_of_rows  = hal::move::rows::Four,
        };
        move_step         = move.get_operation();
        move.address_mode = AddrModMoveEnd;
        move_end          = move.get_operation();
    }

    // Per face: copy the Dest face into the reused source, then compute it back into the same Dest rows.
    using Config = hal::mop::MopConfig<hal::mop::MopTemplate::Template1>;
    Config config {};
    config.outer_loop.count                                = 1;
    config.inner_loop.count                                = move_quads;
    config.inner_loop.body_op                              = move_step;
    config.inner_loop.last_op_on_final_outer_iteration    = move_end;
    config.inner_loop.last_op_on_nonfinal_outer_iteration = move_end;
    const std::uint32_t face_end = operation<eltwise_binary_type, src_b_bcast_type>(acc_to_dest, AddrModFaceEnd, SourceRelease::Both);
    if (blocks == 2)
    {
        config.outer_loop.end_op        = operation<eltwise_binary_type, src_b_bcast_type>(acc_to_dest, AddrModRowStep, SourceRelease::None);
        config.outer_loop.second_end_op = face_end;
    }
    else
    {
        config.outer_loop.end_op = face_end;
    }
    hal::mop::program(config);

    reset_counters();
}

template <
    EltwiseBinaryType eltwise_binary_type,
    BroadcastType src_b_bcast_type,
    DstSync Dst,
    bool is_fp32_dest_acc_en,
    MathFidelity math_fidelity,
    EltwiseBinaryReuseDestType binary_reuse_dest>
inline void _llk_math_eltwise_binary_with_dest_reuse_(
    const ckernel::TensorShape tensor_shape, std::uint32_t dst_index, [[maybe_unused]] const bool clear_fp32_dst_acc)
{
    using namespace hal;
    constexpr bool to_srca             = binary_reuse_dest == EltwiseBinaryReuseDestType::DEST_TO_SRCA;
    constexpr auto source_valid        = to_srca ? sync::StallCondition::SrcAValid : sync::StallCondition::SrcBValid;
    const std::uint32_t num_faces      = tensor_shape.total_num_faces();
    const std::uint32_t tile_dest_base = hal_llk::math::dest_section_base() + dst_index * hal_llk::math::DestRowsPerTile;

    for (std::uint32_t face = 0; face < num_faces; face++)
    {
        cfg::write<cfg::Access::TensixCfgUnit, cfg::DestTargetRegCfgMath::Offset, cfg::Sec::S0>(tile_dest_base + face * FACE_R_DIM);
        // The move neither waits for the unpacked bank nor for the previous result in Dest.
        sync::wait::stall<sync::StallTarget::Math, source_valid | sync::StallCondition::MathIdle>();
        hal::mop::Runner<hal::mop::MopTemplate::Template1>::run();
    }
    hal_llk::eltwise_binary::reset_counters();
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
