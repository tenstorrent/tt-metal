// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// HAL-based implementation of the FPU elementwise binary operations (Blackhole).

#pragma once

#include <cstdint>
#include <utility>

#include "hal/dst.h"
#include "hal/fpu.h"
#include "hal/math_counters.h"
#include "hal/mop.h"
#include "hal/move.h"
#include "hal_llk_common.h"
#include "llk_math_common.h"
#include "tensor_shape.h"

namespace hal_llk
{

// Math address-modifier slots used by the eltwise binary MOPs.
inline constexpr std::uint8_t AM_ROW     = 0; // next 8 rows of SrcA/SrcB/Dest (clears the fidelity phase)
inline constexpr std::uint8_t AM_FACE    = 1; // rewind SrcA/SrcB, move Dest to the next face, save it to carry/reset
inline constexpr std::uint8_t AM_FIDEL   = 2; // next fidelity phase, same rows
inline constexpr std::uint8_t AM_NONE    = 3; // no counter change
inline constexpr std::uint8_t AM_MOV     = 4; // dest reuse: next 4 rows of Dest and of the reused source
inline constexpr std::uint8_t AM_MOV_END = 5; // dest reuse: rewind the reused source, reload Dest from carry/reset

template <EltwiseBinaryType eltwise_binary_type>
constexpr hal::fpu::ElementwiseOperation fpu_operation()
{
    if constexpr (eltwise_binary_type == EltwiseBinaryType::ELWADD)
    {
        return hal::fpu::ElementwiseOperation::Add;
    }
    else if constexpr (eltwise_binary_type == EltwiseBinaryType::ELWSUB)
    {
        return hal::fpu::ElementwiseOperation::Subtract;
    }
    else
    {
        return hal::fpu::ElementwiseOperation::Multiply;
    }
}

template <BroadcastType bcast_type>
constexpr hal::fpu::SrcBBroadcast fpu_broadcast()
{
    if constexpr (bcast_type == BroadcastType::COL)
    {
        return hal::fpu::SrcBBroadcast::Column;
    }
    else if constexpr (bcast_type == BroadcastType::ROW)
    {
        return hal::fpu::SrcBBroadcast::Row;
    }
    else if constexpr (bcast_type == BroadcastType::SCALAR)
    {
        return hal::fpu::SrcBBroadcast::Scalar;
    }
    else
    {
        return hal::fpu::SrcBBroadcast::None;
    }
}

template <EltwiseBinaryType eltwise_binary_type, BroadcastType bcast_type>
inline std::uint32_t eltwise_operation(const std::uint32_t acc_to_dest, const std::uint8_t addr_mod, const hal::fpu::SourceRelease release)
{
    return hal::fpu::get_operation(hal::fpu::Elementwise {
        .operation        = fpu_operation<eltwise_binary_type>(),
        .broadcast        = fpu_broadcast<bcast_type>(),
        .dest_write       = acc_to_dest ? hal::fpu::DestWriteMode::Accumulate : hal::fpu::DestWriteMode::Overwrite,
        .address_modifier = addr_mod,
        .release          = release,
    });
}

template <MathFidelity math_fidelity>
constexpr std::uint32_t fidelity_phases()
{
    return (math_fidelity == MathFidelity::LoFi) ? 1 : static_cast<std::uint32_t>(math_fidelity);
}

// Source banks released after face `face` of a tile with `num_faces` faces.
template <BroadcastType bcast_type>
inline hal::fpu::SourceRelease face_release(const std::uint32_t face, const std::uint32_t num_faces)
{
    if constexpr (bcast_type == BroadcastType::SCALAR)
    {
        return (face + 1 == num_faces) ? hal::fpu::SourceRelease::Both : hal::fpu::SourceRelease::SrcA;
    }
    else if constexpr (bcast_type == BroadcastType::COL)
    {
        // SrcB holds column 0 of face 0 (faces 0/1) or face 2 (faces 2/3).
        return ((face & 1) == 0 && (face + 1) < num_faces) ? hal::fpu::SourceRelease::SrcA : hal::fpu::SourceRelease::Both;
    }
    else
    {
        return hal::fpu::SourceRelease::Both;
    }
}

// Dest increment of the face step so every face starts on a 16-row boundary.
inline std::uint32_t face_dest_step(const std::uint32_t face_r_dim)
{
    return FACE_R_DIM - 8 * (face_row_blocks(face_r_dim) - 1);
}

template <EltwiseBinaryType eltwise_binary_type, BroadcastType bcast_type, MathFidelity math_fidelity>
inline void program_eltwise_binary_addrmods(const std::uint32_t face_r_dim)
{
    constexpr bool high_fidelity    = fidelity_phases<math_fidelity>() > 1;
    constexpr std::uint32_t srcb_8  = (bcast_type == BroadcastType::NONE || bcast_type == BroadcastType::COL) ? 8 : 0;
    using hal::cfg::Sec;
    program_math_addrmod<Sec::S0>({.srca_incr = 8, .srcb_incr = srcb_8, .dest_incr = 8, .fidel_clear = high_fidelity});
    program_math_addrmod<Sec::S1>(
        {.srca_clear = true, .srcb_clear = true, .dest_incr = face_dest_step(face_r_dim), .dest_c_to_cr = true, .fidel_clear = high_fidelity});
    program_math_addrmod<Sec::S2>({.fidel_incr = 1});
    program_math_addrmod<Sec::S3>({});
}

// Program the MOP covering one face: row blocks in the outer loop, fidelity phases in the inner loop.
template <EltwiseBinaryType eltwise_binary_type, BroadcastType bcast_type, MathFidelity math_fidelity>
inline void program_eltwise_binary_face_mop(const std::uint32_t acc_to_dest, const std::uint32_t face_r_dim, const hal::fpu::SourceRelease release)
{
    hal::mop::MopConfig<hal::mop::MopTemplate::Template1> config {};
    config.outer_loop.count                               = face_row_blocks(face_r_dim);
    config.inner_loop.count                               = fidelity_phases<math_fidelity>();
    config.inner_loop.body_op                             = eltwise_operation<eltwise_binary_type, bcast_type>(acc_to_dest, AM_FIDEL, hal::fpu::SourceRelease::None);
    config.inner_loop.last_op_on_nonfinal_outer_iteration = eltwise_operation<eltwise_binary_type, bcast_type>(acc_to_dest, AM_ROW, hal::fpu::SourceRelease::None);
    config.inner_loop.last_op_on_final_outer_iteration    = eltwise_operation<eltwise_binary_type, bcast_type>(acc_to_dest, AM_FACE, release);
    hal::mop::program(config);
}

template <EltwiseBinaryType eltwise_binary_type, BroadcastType bcast_type, MathFidelity math_fidelity>
inline void patch_eltwise_binary_face_release(const std::uint32_t acc_to_dest, const hal::fpu::SourceRelease release)
{
    hal::mop::write<hal::mop::Template1Field::LastOpOnFinalOuterIteration>(
        eltwise_operation<eltwise_binary_type, bcast_type>(acc_to_dest, AM_FACE, release));
}

template <EltwiseBinaryType eltwise_binary_type>
inline void configure_src_zero_flag_and_formats()
{
    namespace cfg = hal::cfg;
    // FP/INT8 compute: SrcA/SrcB zero substitution stays enabled.
    cfg::write<cfg::Access::TensixCfgUnit, cfg::AluAccCtrl::Zero_Flag_disabled_src, cfg::Sec::S0, 0>();
}

// Rewind SrcA/SrcB/Dest counters (live and carry/reset) to the start of a tile.
inline void reset_math_counters()
{
    hal::math_counters.set<hal::rwc::Counters::All, 0>().apply();
}

inline void set_math_dest_tile(const std::uint32_t dst_index)
{
    namespace cfg = hal::cfg;
    cfg::write<cfg::Access::TensixCfgUnit, cfg::DestTargetRegCfgMath::Offset, cfg::Sec::S0>(
        math_dest_section_base() + (dst_index << DEST_TILE_ROWS_LOG2));
}

} // namespace hal_llk

/*************************************************************************
 * Common Helpers
 *************************************************************************/

template <EltwiseBinaryType eltwise_binary_type, BroadcastType bcast_type, MathFidelity math_fidelity>
inline void eltwise_binary_configure_addrmod()
{
    hal_llk::program_eltwise_binary_addrmods<eltwise_binary_type, bcast_type, math_fidelity>(FACE_R_DIM);
}

template <EltwiseBinaryType eltwise_binary_type>
inline auto eltwise_binary_func(std::uint8_t clr_src, std::uint8_t acc_to_dest, std::uint8_t broadcast_type, std::uint8_t addr_mod)
{
    const hal::fpu::SourceRelease release = clr_src == 0   ? hal::fpu::SourceRelease::None
                                            : clr_src == 1 ? hal::fpu::SourceRelease::SrcA
                                            : clr_src == 2 ? hal::fpu::SourceRelease::SrcB
                                                           : hal::fpu::SourceRelease::Both;
    const hal::fpu::SrcBBroadcast broadcast = broadcast_type == 1   ? hal::fpu::SrcBBroadcast::Column
                                              : broadcast_type == 2 ? hal::fpu::SrcBBroadcast::Row
                                              : broadcast_type == 3 ? hal::fpu::SrcBBroadcast::Scalar
                                                                    : hal::fpu::SrcBBroadcast::None;
    return hal::fpu::get_operation(hal::fpu::Elementwise {
        .operation        = hal_llk::fpu_operation<eltwise_binary_type>(),
        .broadcast        = broadcast,
        .dest_write       = acc_to_dest ? hal::fpu::DestWriteMode::Accumulate : hal::fpu::DestWriteMode::Overwrite,
        .address_modifier = addr_mod,
        .release          = release,
    });
}

/*************************************************************************
 * Eltwise Binary Standard (No Dest Reuse)
 *************************************************************************/

template <EltwiseBinaryType eltwise_binary_type, BroadcastType bcast_type, MathFidelity math_fidelity = MathFidelity::LoFi>
inline void eltwise_binary_configure_mop_standard(const std::uint32_t acc_to_dest, const ckernel::TensorShape tensor_shape)
{
    using hal::fpu::SourceRelease;
    const std::uint32_t num_faces  = tensor_shape.total_num_faces();
    const std::uint32_t row_blocks = hal_llk::face_row_blocks(tensor_shape.face_r_dim);

    if constexpr (hal_llk::fidelity_phases<math_fidelity>() > 1)
    {
        // High fidelity: one MOP run per face (see _llk_math_eltwise_binary_standard_).
        hal_llk::program_eltwise_binary_face_mop<eltwise_binary_type, bcast_type, math_fidelity>(
            acc_to_dest, tensor_shape.face_r_dim, hal_llk::face_release<bcast_type>(0, num_faces));
        return;
    }

    const auto elw = [&](const std::uint8_t addr_mod, const SourceRelease release)
    { return hal_llk::eltwise_operation<eltwise_binary_type, bcast_type>(acc_to_dest, addr_mod, release); };

    // Low fidelity: the whole tile in one MOP run; faces in the outer loop, 8-row blocks in the inner loop.
    hal::mop::MopConfig<hal::mop::MopTemplate::Template1> config {};
    if (bcast_type == BroadcastType::COL && num_faces > 1)
    {
        // Two faces share one SrcB bank: release SrcA after the first face of the pair, both after the second.
        config.outer_loop.count = num_faces / 2;
        config.inner_loop.count = 2;
        if (row_blocks == 2)
        {
            config.inner_loop.body_op        = elw(hal_llk::AM_ROW, SourceRelease::None);
            config.inner_loop.alternating_op = elw(hal_llk::AM_FACE, SourceRelease::SrcA);
        }
        else
        {
            config.inner_loop.body_op = elw(hal_llk::AM_FACE, SourceRelease::SrcA);
        }
        config.inner_loop.last_op_on_final_outer_iteration    = elw(hal_llk::AM_FACE, SourceRelease::Both);
        config.inner_loop.last_op_on_nonfinal_outer_iteration = elw(hal_llk::AM_FACE, SourceRelease::Both);
    }
    else
    {
        const SourceRelease nonfinal_release = (bcast_type == BroadcastType::SCALAR) ? SourceRelease::SrcA : SourceRelease::Both;
        config.outer_loop.count                               = num_faces;
        config.inner_loop.count                               = row_blocks;
        config.inner_loop.body_op                             = elw(hal_llk::AM_ROW, SourceRelease::None);
        config.inner_loop.last_op_on_final_outer_iteration    = elw(hal_llk::AM_FACE, SourceRelease::Both);
        config.inner_loop.last_op_on_nonfinal_outer_iteration = elw(hal_llk::AM_FACE, nonfinal_release);
    }
    hal::mop::program(config);
}

template <EltwiseBinaryType eltwise_binary_type, BroadcastType src_b_bcast_type, MathFidelity math_fidelity = MathFidelity::LoFi>
inline void _llk_math_eltwise_binary_standard_init_(const ckernel::TensorShape tensor_shape, const std::uint32_t acc_to_dest)
{
    namespace cfg = hal::cfg;
    using namespace hal::sync;

    // Reprogramming address modifiers / MOP must not race with FPU work still in flight.
    wait::stall<StallTarget::Config, StallCondition::MathIdle>();

    cfg::write<cfg::Access::TensixCfgUnit, cfg::DisableImpliedFmt[cfg::SrcA], cfg::Sec::S0, 0>();
    cfg::write<cfg::Access::TensixCfgUnit, cfg::DisableImpliedFmt[cfg::SrcB], cfg::Sec::S0, 0>();
    hal_llk::configure_src_zero_flag_and_formats<eltwise_binary_type>();

    hal_llk::program_eltwise_binary_addrmods<eltwise_binary_type, src_b_bcast_type, math_fidelity>(tensor_shape.face_r_dim);
    eltwise_binary_configure_mop_standard<eltwise_binary_type, src_b_bcast_type, math_fidelity>(acc_to_dest, tensor_shape);
    hal_llk::reset_math_counters();
}

template <
    EltwiseBinaryType eltwise_binary_type,
    BroadcastType src_b_bcast_type,
    DstSync Dst,
    bool is_fp32_dest_acc_en,
    MathFidelity math_fidelity = MathFidelity::LoFi>
inline void _llk_math_eltwise_binary_standard_(const ckernel::TensorShape tensor_shape, std::uint32_t dst_index)
{
    hal_llk::set_math_dest_tile(dst_index);

    if constexpr (hal_llk::fidelity_phases<math_fidelity>() > 1)
    {
        const std::uint32_t num_faces = tensor_shape.total_num_faces();
        hal::fpu::SourceRelease programmed = hal_llk::face_release<src_b_bcast_type>(0, num_faces);
        for (std::uint32_t face = 0; face < num_faces; face++)
        {
            const hal::fpu::SourceRelease release = hal_llk::face_release<src_b_bcast_type>(face, num_faces);
            if (release != programmed)
            {
                // acc_to_dest is not tracked here: high-fidelity ELWMUL always accumulates onto Dest.
                hal_llk::patch_eltwise_binary_face_release<eltwise_binary_type, src_b_bcast_type, math_fidelity>(0, release);
                programmed = release;
            }
            hal::mop::Runner<hal::mop::MopTemplate::Template1>::run();
        }
        if (programmed != hal_llk::face_release<src_b_bcast_type>(0, num_faces))
        {
            hal_llk::patch_eltwise_binary_face_release<eltwise_binary_type, src_b_bcast_type, math_fidelity>(
                0, hal_llk::face_release<src_b_bcast_type>(0, num_faces));
        }
    }
    else
    {
        hal::mop::Runner<hal::mop::MopTemplate::Template1>::run();
    }

    hal_llk::reset_math_counters();
}

/*************************************************************************
 * Eltwise Binary WITH Dest Reuse
 *************************************************************************/

template <EltwiseBinaryReuseDestType binary_reuse_dest>
inline void eltwise_binary_reuse_dest_as_src()
{
    using namespace hal::sync;
    if constexpr (binary_reuse_dest == EltwiseBinaryReuseDestType::DEST_TO_SRCA)
    {
        // MOVD2A does not wait for the SrcA bank: wait for the unpacked bank and for Dest writes to settle.
        wait::stall<StallTarget::Math, StallCondition::SrcAValid | StallCondition::MathIdle>();
        constexpr hal::move::Transfer<hal::move::Dst, hal::move::SrcA> move_face {
            .source_row         = 0,
            .destination_row    = 0,
            .address_mode       = hal_llk::AM_MOV,
            .number_of_rows     = hal::move::rows::Face,
            .final_address_mode = hal_llk::AM_MOV_END,
        };
        hal::move::run<move_face>();
    }
    else
    {
        wait::stall<StallTarget::Math, StallCondition::SrcBValid | StallCondition::MathIdle>();
        constexpr hal::move::Transfer<hal::move::Dst, hal::move::SrcB> move_face {
            .source_row         = 0,
            .destination_row    = 0,
            .address_mode       = hal_llk::AM_MOV,
            .number_of_rows     = hal::move::rows::Face,
            .final_address_mode = hal_llk::AM_MOV_END,
        };
        hal::move::run<move_face>();
    }
}

namespace hal_llk
{
template <std::size_t... Rows>
inline void zero_dest_rows(std::index_sequence<Rows...>)
{
    (hal::dst::run<hal::dst::Zero {.scope = hal::dst::ZeroScope::SingleRow, .index = Rows, .address_mode = AM_NONE}>(), ...);
}
} // namespace hal_llk

inline void zeroacc_face_by_row()
{
    // Dest counter points at the face start; each ZEROACC clears one row relative to it.
    hal_llk::zero_dest_rows(std::make_index_sequence<FACE_R_DIM> {});
}

template <EltwiseBinaryType eltwise_binary_type, BroadcastType bcast_type, MathFidelity math_fidelity = MathFidelity::LoFi>
inline void eltwise_binary_configure_mop_with_dest_reuse(const std::uint32_t acc_to_dest, const ckernel::TensorShape tensor_shape)
{
    hal_llk::program_eltwise_binary_face_mop<eltwise_binary_type, bcast_type, math_fidelity>(
        acc_to_dest, tensor_shape.face_r_dim, hal::fpu::SourceRelease::Both);
}

template <
    EltwiseBinaryType eltwise_binary_type,
    BroadcastType src_b_bcast_type,
    MathFidelity math_fidelity                   = MathFidelity::LoFi,
    EltwiseBinaryReuseDestType binary_reuse_dest = EltwiseBinaryReuseDestType::DEST_TO_SRCA>
inline void _llk_math_eltwise_binary_with_dest_reuse_init_(const ckernel::TensorShape tensor_shape, const std::uint32_t acc_to_dest)
{
    static_assert(binary_reuse_dest != EltwiseBinaryReuseDestType::NONE, "Dest reuse init requires DEST_TO_SRCA or DEST_TO_SRCB");
    namespace cfg = hal::cfg;
    using namespace hal::sync;
    using hal::cfg::Sec;

    wait::stall<StallTarget::Config, StallCondition::MathIdle>();

    // The reused source bank holds Dest data: use the configured ALU format instead of the unpacker's implied one.
    constexpr std::uint32_t reuse_a = binary_reuse_dest == EltwiseBinaryReuseDestType::DEST_TO_SRCA ? 1 : 0;
    constexpr std::uint32_t reuse_b = binary_reuse_dest == EltwiseBinaryReuseDestType::DEST_TO_SRCB ? 1 : 0;
    cfg::write<cfg::Access::TensixCfgUnit, cfg::DisableImpliedFmt[cfg::SrcA], Sec::S0, reuse_a>();
    cfg::write<cfg::Access::TensixCfgUnit, cfg::DisableImpliedFmt[cfg::SrcB], Sec::S0, reuse_b>();
    hal_llk::configure_src_zero_flag_and_formats<eltwise_binary_type>();

    hal_llk::program_eltwise_binary_addrmods<eltwise_binary_type, src_b_bcast_type, math_fidelity>(tensor_shape.face_r_dim);
    hal_llk::program_math_addrmod<Sec::S4>({.srca_incr = 4 * reuse_a, .srcb_incr = 4 * reuse_b, .dest_incr = 4});
    hal_llk::program_math_addrmod<Sec::S5>({.srca_clear = reuse_a != 0, .srcb_clear = reuse_b != 0, .dest_cr = true});

    eltwise_binary_configure_mop_with_dest_reuse<eltwise_binary_type, src_b_bcast_type, math_fidelity>(acc_to_dest, tensor_shape);
    hal_llk::reset_math_counters();
}

template <bool is_fp32_dest_acc_en, EltwiseBinaryReuseDestType binary_reuse_dest, MathFidelity math_fidelity = MathFidelity::LoFi>
inline void eltwise_binary_run_with_dest_reuse(
    const std::uint32_t loop_count,
    [[maybe_unused]] const std::uint32_t face_offset,
    [[maybe_unused]] const bool clear_fp32_dst_acc,
    [[maybe_unused]] const std::uint32_t dst_index,
    [[maybe_unused]] const std::uint32_t face_r_dim)
{
    for (std::uint32_t face = 0; face < loop_count; face++)
    {
        eltwise_binary_reuse_dest_as_src<binary_reuse_dest>();
        hal::mop::Runner<hal::mop::MopTemplate::Template1>::run();
    }
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
    hal_llk::set_math_dest_tile(dst_index);
    const std::uint32_t num_faces = tensor_shape.total_num_faces();
    for (std::uint32_t face = 0; face < num_faces; face++)
    {
        eltwise_binary_reuse_dest_as_src<binary_reuse_dest>();
        if constexpr (eltwise_binary_type == EltwiseBinaryType::ELWMUL)
        {
            // ELWMUL always accumulates: the face was moved out, clear it before the product is written back.
            zeroacc_face_by_row();
        }
        hal::mop::Runner<hal::mop::MopTemplate::Template1>::run();
    }
    hal_llk::reset_math_counters();
    (void)clear_fp32_dst_acc;
}

/*************************************************************************
 * Public API
 *************************************************************************/

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
        (void)clear_fp32_dst_acc;
    }
    else
    {
        _llk_math_eltwise_binary_with_dest_reuse_<eltwise_binary_type, src_b_bcast_type, Dst, is_fp32_dest_acc_en, math_fidelity, binary_reuse_dest>(
            tensor_shape, dst_index, clear_fp32_dst_acc);
    }
}

inline void _llk_math_eltwise_binary_uninit_()
{
    namespace cfg = hal::cfg;
    cfg::write<cfg::Access::TensixCfgUnit, cfg::DisableImpliedFmt[cfg::SrcA], cfg::Sec::S0, 0>();
    cfg::write<cfg::Access::TensixCfgUnit, cfg::DisableImpliedFmt[cfg::SrcB], cfg::Sec::S0, 0>();
}
