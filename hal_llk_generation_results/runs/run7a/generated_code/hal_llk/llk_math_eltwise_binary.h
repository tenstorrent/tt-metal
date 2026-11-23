// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// HAL-based implementation of the Blackhole FPU eltwise binary LLK functions.

#pragma once

#include <cstdint>

#include "hal/dst.h"
#include "hal/fpu.h"
#include "hal/math_counters.h"
#include "hal/move.h"
#include "llk_hal_common.h"
#include "llk_math_common.h"

namespace hal_llk::math
{

using hal::rwc::Counters;
using MopConfig = hal::mop::MopConfig<hal::mop::MopTemplate::Template1>;

// Math ADDR_MOD slots used by eltwise binary.
constexpr std::uint8_t ADDR_MOD_ROWS     = 0; // advance to the next 8-row block
constexpr std::uint8_t ADDR_MOD_HOLD     = 1; // leave all counters unchanged
constexpr std::uint8_t ADDR_MOD_FIDELITY = 2; // next fidelity phase: rewind sources and Dest
constexpr std::uint8_t ADDR_MOD_FACE_END = 3; // last fidelity phase: next block, reset fidelity

// First SrcA/SrcB row read by the FPU at counter 0.
constexpr std::uint32_t SRCA_FIRST_ROW = 0;
constexpr std::uint32_t SRCB_FIRST_ROW = 0;

struct AddrModSpec
{
    std::uint8_t srca_incr      = 0;
    std::uint8_t srca_clear     = 0;
    std::uint8_t srcb_incr      = 0;
    std::uint8_t srcb_clear     = 0;
    std::uint16_t dest_incr     = 0;
    std::uint8_t dest_cr        = 0;
    std::uint8_t dest_clear     = 0;
    std::uint8_t dest_c_to_cr   = 0;
    std::uint8_t fidelity_incr  = 0;
    std::uint8_t fidelity_clear = 0;
};

// Program one math ADDR_MOD slot. Every field of the slot is written so no stale bits survive.
template <cfg::Sec S, AddrModSpec M>
inline void program_addr_mod()
{
    cfg::write<cfg::Access::TensixCfgUnit>(
        cfg::set<cfg::AddrMod[cfg::SrcA].Incr, S, M.srca_incr>(),
        cfg::set<cfg::AddrMod[cfg::SrcA].CR, S, 0>(),
        cfg::set<cfg::AddrMod[cfg::SrcA].Clear, S, M.srca_clear>(),
        cfg::set<cfg::AddrMod[cfg::SrcB].Incr, S, M.srcb_incr>(),
        cfg::set<cfg::AddrMod[cfg::SrcB].CR, S, 0>(),
        cfg::set<cfg::AddrMod[cfg::SrcB].Clear, S, M.srcb_clear>(),
        cfg::set<cfg::AddrMod[cfg::SrcA].Incr2, S, 0>(),
        cfg::set<cfg::AddrMod[cfg::SrcB].Incr2, S, 0>(),
        cfg::set<cfg::AddrMod[cfg::Dest].Incr, S, M.dest_incr>(),
        cfg::set<cfg::AddrMod[cfg::Dest].CR, S, M.dest_cr>(),
        cfg::set<cfg::AddrMod[cfg::Dest].Clear, S, M.dest_clear>(),
        cfg::set<cfg::AddrMod[cfg::Dest].CToCR, S, M.dest_c_to_cr>(),
        cfg::set<cfg::AddrMod[cfg::Fidelity].Incr, S, M.fidelity_incr>(),
        cfg::set<cfg::AddrMod[cfg::Fidelity].Clear, S, M.fidelity_clear>(),
        cfg::set<cfg::AddrMod[cfg::Bias].Incr, S, 0>(),
        cfg::set<cfg::AddrMod[cfg::Bias].Clear, S, 0>());
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

constexpr std::uint32_t num_fidelity_phases(const EltwiseBinaryType type, const MathFidelity fidelity)
{
    return (type == EltwiseBinaryType::ELWMUL && fidelity != MathFidelity::LoFi) ? static_cast<std::uint32_t>(fidelity) : 1;
}

// One 8x16 block of SrcA op SrcB into Dest.
template <EltwiseBinaryType type, BroadcastType bcast, std::uint8_t address_modifier>
constexpr std::uint32_t block_operation(const bool acc_to_dest)
{
    return hal::fpu::get_operation(hal::fpu::Elementwise {
        .operation        = fpu_operation(type),
        .broadcast        = fpu_broadcast(bcast),
        .dest_write       = acc_to_dest ? hal::fpu::DestWriteMode::Accumulate : hal::fpu::DestWriteMode::Overwrite,
        .address_modifier = address_modifier,
        .release          = hal::fpu::SourceRelease::None,
    });
}

// Release the consumed source bank(s) and rewind both source row counters.
constexpr std::uint32_t RELEASE_SRCA_AND_SRCB = hal::math_counters.set<Counters::SrcA | Counters::SrcB, 0>().release<Counters::SrcA | Counters::SrcB>().get_operation();
constexpr std::uint32_t RELEASE_SRCA          = hal::math_counters.set<Counters::SrcA | Counters::SrcB, 0>().release<Counters::SrcA>().get_operation();

// Move one 16-row Dest face (at the current Dest counter) into the SrcA or SrcB bank owned by math.
template <EltwiseBinaryReuseDestType reuse>
inline void move_dest_face_to_source()
{
    namespace move = hal::move;
    if constexpr (reuse == EltwiseBinaryReuseDestType::DEST_TO_SRCA)
    {
        // MOVD2A does not wait for SrcA ownership.
        sync::wait::stall<sync::StallTarget::Math, sync::StallCondition::SrcAValid | sync::StallCondition::MathIdle>();
        move::run<move::Transfer<move::Dst, move::SrcA> {
            .source_row = 0, .destination_row = SRCA_FIRST_ROW + 0, .address_mode = ADDR_MOD_HOLD, .number_of_rows = move::rows::Four}>();
        move::run<move::Transfer<move::Dst, move::SrcA> {
            .source_row = 4, .destination_row = SRCA_FIRST_ROW + 4, .address_mode = ADDR_MOD_HOLD, .number_of_rows = move::rows::Four}>();
        move::run<move::Transfer<move::Dst, move::SrcA> {
            .source_row = 8, .destination_row = SRCA_FIRST_ROW + 8, .address_mode = ADDR_MOD_HOLD, .number_of_rows = move::rows::Four}>();
        move::run<move::Transfer<move::Dst, move::SrcA> {
            .source_row = 12, .destination_row = SRCA_FIRST_ROW + 12, .address_mode = ADDR_MOD_HOLD, .number_of_rows = move::rows::Four}>();
    }
    else if constexpr (reuse == EltwiseBinaryReuseDestType::DEST_TO_SRCB)
    {
        // MOVD2B does not wait for SrcB ownership.
        sync::wait::stall<sync::StallTarget::Math, sync::StallCondition::SrcBValid | sync::StallCondition::MathIdle>();
        move::run<move::Transfer<move::Dst, move::SrcB> {
            .source_row = 0, .destination_row = SRCB_FIRST_ROW + 0, .address_mode = ADDR_MOD_HOLD, .number_of_rows = move::rows::Four}>();
        move::run<move::Transfer<move::Dst, move::SrcB> {
            .source_row = 4, .destination_row = SRCB_FIRST_ROW + 4, .address_mode = ADDR_MOD_HOLD, .number_of_rows = move::rows::Four}>();
        move::run<move::Transfer<move::Dst, move::SrcB> {
            .source_row = 8, .destination_row = SRCB_FIRST_ROW + 8, .address_mode = ADDR_MOD_HOLD, .number_of_rows = move::rows::Four}>();
        move::run<move::Transfer<move::Dst, move::SrcB> {
            .source_row = 12, .destination_row = SRCB_FIRST_ROW + 12, .address_mode = ADDR_MOD_HOLD, .number_of_rows = move::rows::Four}>();
    }
}

// Mark one Dest face as zero (ELWMUL accumulates into Dest).
template <bool is_fp32_dest_acc_en>
inline void zero_dest_face(const std::uint32_t dest_row)
{
    hal::dst::run(hal::dst::Zero {
        .scope = hal::dst::ZeroScope::Face,
        .index = dest_row / DEST_ROWS_PER_FACE,
        .width = is_fp32_dest_acc_en ? hal::dst::DestWidth::Bits32 : hal::dst::DestWidth::Bits16,
    });
}

} // namespace hal_llk::math

/**
 * @brief Program the math ADDR_MOD slots used by the eltwise binary MOP.
 */
template <EltwiseBinaryType eltwise_binary_type, BroadcastType bcast_type, MathFidelity math_fidelity>
inline void eltwise_binary_configure_addrmod()
{
    namespace cfg = hal_llk::cfg;
    using namespace hal_llk::math;

    // ROW and SCALAR broadcast keep reading the selected SrcB row.
    constexpr std::uint8_t srcb_incr = (bcast_type == BroadcastType::ROW || bcast_type == BroadcastType::SCALAR) ? 0 : 8;
    constexpr std::uint8_t fidelity_incr = (num_fidelity_phases(eltwise_binary_type, math_fidelity) > 1) ? 1 : 0;

    program_addr_mod<cfg::Sec::S0, AddrModSpec {.srca_incr = 8, .srcb_incr = srcb_incr, .dest_incr = 8}>();
    program_addr_mod<cfg::Sec::S1, AddrModSpec {}>();
    program_addr_mod<cfg::Sec::S2, AddrModSpec {.srca_clear = 1, .srcb_clear = 1, .dest_cr = 1, .fidelity_incr = fidelity_incr}>();
    program_addr_mod<cfg::Sec::S3, AddrModSpec {.srca_clear = 1, .srcb_clear = 1, .dest_incr = 8, .dest_c_to_cr = 1, .fidelity_clear = 1}>();
}

template <EltwiseBinaryType eltwise_binary_type>
inline auto eltwise_binary_func(std::uint8_t clr_src, std::uint8_t acc_to_dest, std::uint8_t broadcast_type, std::uint8_t addr_mod)
{
    return hal::fpu::get_operation(hal::fpu::Elementwise {
        .operation        = hal_llk::math::fpu_operation(eltwise_binary_type),
        .broadcast        = static_cast<hal::fpu::SrcBBroadcast>(broadcast_type),
        .dest_write       = acc_to_dest ? hal::fpu::DestWriteMode::Accumulate : hal::fpu::DestWriteMode::Overwrite,
        .address_modifier = addr_mod,
        .release          = clr_src ? hal::fpu::SourceRelease::Both : hal::fpu::SourceRelease::None,
    });
}

namespace hal_llk::math
{

// Program the MOP that processes faces of one tile. With dest reuse, one MOP run covers one face.
template <EltwiseBinaryType type, BroadcastType bcast, MathFidelity fidelity, bool per_face>
inline void configure_mop(const std::uint32_t acc_to_dest, const ckernel::TensorShape tensor_shape)
{
    constexpr std::uint32_t phases    = num_fidelity_phases(type, fidelity);
    constexpr std::uint32_t BLOCKS    = DEST_ROWS_PER_FACE / 8; // 8-row FPU blocks per face
    const std::uint32_t num_faces     = per_face ? 1 : tensor_shape.total_num_faces();
    const std::uint32_t row_block     = block_operation<type, bcast, ADDR_MOD_ROWS>(acc_to_dest);

    if constexpr (phases > 1)
    {
        // Each outer iteration is one fidelity phase over one face; the caller runs it per face.
        hal::mop::program(MopConfig {
            .outer_loop = {.count = phases},
            .inner_loop =
                {.count                               = BLOCKS,
                 .body_op                             = row_block,
                 .last_op_on_final_outer_iteration    = block_operation<type, bcast, ADDR_MOD_FACE_END>(acc_to_dest),
                 .last_op_on_nonfinal_outer_iteration = block_operation<type, bcast, ADDR_MOD_FIDELITY>(acc_to_dest)},
        });
    }
    else if constexpr (bcast == BroadcastType::COL)
    {
        // One face row per run: every SrcA face of the row shares the SrcB face, released by the caller.
        hal::mop::program(MopConfig {
            .outer_loop = {.count = per_face ? 1u : tensor_shape.num_faces_c_dim, .end_op = RELEASE_SRCA},
            .inner_loop = {.count = BLOCKS, .body_op = row_block},
        });
    }
    else if constexpr (bcast == BroadcastType::SCALAR)
    {
        hal::mop::program(MopConfig {
            .outer_loop = {.count = num_faces, .end_op = RELEASE_SRCA},
            .inner_loop = {.count = BLOCKS, .body_op = row_block},
        });
    }
    else
    {
        hal::mop::program(MopConfig {
            .outer_loop = {.count = num_faces, .end_op = RELEASE_SRCA_AND_SRCB},
            .inner_loop = {.count = BLOCKS, .body_op = row_block},
        });
    }
}

} // namespace hal_llk::math

template <EltwiseBinaryType eltwise_binary_type, BroadcastType bcast_type, MathFidelity math_fidelity = MathFidelity::LoFi>
inline void eltwise_binary_configure_mop_standard(const std::uint32_t acc_to_dest, const ckernel::TensorShape tensor_shape)
{
    hal_llk::math::configure_mop<eltwise_binary_type, bcast_type, math_fidelity, false>(acc_to_dest, tensor_shape);
}

template <EltwiseBinaryType eltwise_binary_type, BroadcastType bcast_type, MathFidelity math_fidelity = MathFidelity::LoFi>
inline void eltwise_binary_configure_mop_with_dest_reuse(const std::uint32_t acc_to_dest, const ckernel::TensorShape tensor_shape)
{
    hal_llk::math::configure_mop<eltwise_binary_type, bcast_type, math_fidelity, true>(acc_to_dest, tensor_shape);
}

template <EltwiseBinaryType eltwise_binary_type, BroadcastType src_b_bcast_type, MathFidelity math_fidelity = MathFidelity::LoFi>
inline void _llk_math_eltwise_binary_standard_init_(const ckernel::TensorShape tensor_shape, const std::uint32_t acc_to_dest)
{
    eltwise_binary_configure_addrmod<eltwise_binary_type, src_b_bcast_type, math_fidelity>();
    eltwise_binary_configure_mop_standard<eltwise_binary_type, src_b_bcast_type, math_fidelity>(acc_to_dest, tensor_shape);
    hal::math_counters.set<hal::rwc::Counters::All, 0>().clear_fidelity().apply();
}

template <
    EltwiseBinaryType eltwise_binary_type,
    BroadcastType src_b_bcast_type,
    MathFidelity math_fidelity                   = MathFidelity::LoFi,
    EltwiseBinaryReuseDestType binary_reuse_dest = EltwiseBinaryReuseDestType::DEST_TO_SRCA>
inline void _llk_math_eltwise_binary_with_dest_reuse_init_(const ckernel::TensorShape tensor_shape, const std::uint32_t acc_to_dest)
{
    static_assert(binary_reuse_dest != EltwiseBinaryReuseDestType::NONE, "Dest reuse init requires DEST_TO_SRCA or DEST_TO_SRCB");
    eltwise_binary_configure_addrmod<eltwise_binary_type, src_b_bcast_type, math_fidelity>();
    eltwise_binary_configure_mop_with_dest_reuse<eltwise_binary_type, src_b_bcast_type, math_fidelity>(acc_to_dest, tensor_shape);
    hal::math_counters.set<hal::rwc::Counters::All, 0>().clear_fidelity().apply();
}

template <
    EltwiseBinaryType eltwise_binary_type,
    BroadcastType src_b_bcast_type,
    DstSync Dst,
    bool is_fp32_dest_acc_en,
    MathFidelity math_fidelity = MathFidelity::LoFi>
inline void _llk_math_eltwise_binary_standard_(const ckernel::TensorShape tensor_shape, std::uint32_t dst_index)
{
    namespace cfg = hal_llk::cfg;
    using hal_llk::math::Counters;
    using Runner = hal::mop::Runner<hal::mop::MopTemplate::Template1>;

    constexpr bool per_face_runs = hal_llk::math::num_fidelity_phases(eltwise_binary_type, math_fidelity) > 1;
    const std::uint32_t dest_row = hal_llk::dest_section_base() + dst_index * hal_llk::DEST_ROWS_PER_TILE;
    cfg::write<cfg::Access::TensixCfgUnit, cfg::DestTargetRegCfgMath::Offset, cfg::Sec::S0>(dest_row);

    if constexpr (eltwise_binary_type == EltwiseBinaryType::ELWMUL)
    {
        // ELWMUL accumulates, so every face starts from zero.
        for (std::uint32_t face = 0; face < tensor_shape.total_num_faces(); face++)
        {
            hal_llk::math::zero_dest_face<is_fp32_dest_acc_en>(dest_row + face * hal_llk::DEST_ROWS_PER_FACE);
        }
    }

    if constexpr (per_face_runs)
    {
        // High fidelity: one MOP run (all phases) per face, then hand back the consumed banks.
        for (std::uint32_t face = 0; face < tensor_shape.total_num_faces(); face++)
        {
            Runner::run();
            if constexpr (src_b_bcast_type == BroadcastType::NONE || src_b_bcast_type == BroadcastType::ROW)
            {
                hal::math_counters.release<Counters::SrcA | Counters::SrcB>().apply();
            }
            else
            {
                hal::math_counters.release<Counters::SrcA>().apply();
                const bool last_in_row = ((face + 1) % tensor_shape.num_faces_c_dim) == 0;
                if (src_b_bcast_type == BroadcastType::COL && last_in_row)
                {
                    hal::math_counters.release<Counters::SrcB>().apply();
                }
            }
        }
        if constexpr (src_b_bcast_type == BroadcastType::SCALAR)
        {
            hal::math_counters.release<Counters::SrcB>().apply();
        }
    }
    else if constexpr (src_b_bcast_type == BroadcastType::COL)
    {
        for (std::uint32_t face_row = 0; face_row < tensor_shape.num_faces_r_dim; face_row++)
        {
            Runner::run();
            hal::math_counters.release<Counters::SrcB>().apply();
        }
    }
    else if constexpr (src_b_bcast_type == BroadcastType::SCALAR)
    {
        Runner::run();
        hal::math_counters.release<Counters::SrcB>().apply();
    }
    else
    {
        Runner::run();
    }

    hal::math_counters.set<Counters::Dest, 0>().apply();
}

template <bool is_fp32_dest_acc_en, EltwiseBinaryReuseDestType binary_reuse_dest, MathFidelity math_fidelity = MathFidelity::LoFi>
inline void eltwise_binary_run_with_dest_reuse(
    const std::uint32_t loop_count,
    const std::uint32_t face_offset,
    [[maybe_unused]] const bool clear_fp32_dst_acc,
    const std::uint32_t dst_index,
    [[maybe_unused]] const std::uint32_t face_r_dim)
{
    using hal_llk::math::Counters;
    const std::uint32_t dest_row = hal_llk::dest_section_base() + dst_index * hal_llk::DEST_ROWS_PER_TILE;

    // Dest counter addresses the face being processed (the offset register holds the tile base).
    hal::math_counters.set<Counters::Dest, 0>().apply();
    for (std::uint32_t face = 0; face < face_offset; face++)
    {
        hal::math_counters.increment<Counters::Dest, 8>().apply();
        hal::math_counters.increment<Counters::Dest, 8>().apply();
    }

    for (std::uint32_t face = face_offset; face < face_offset + loop_count; face++)
    {
        hal_llk::math::move_dest_face_to_source<binary_reuse_dest>();
        if constexpr (math_fidelity != MathFidelity::LoFi)
        {
            hal_llk::math::zero_dest_face<is_fp32_dest_acc_en>(dest_row + face * hal_llk::DEST_ROWS_PER_FACE);
        }
        hal::mop::Runner<hal::mop::MopTemplate::Template1>::run();
        if constexpr (math_fidelity != MathFidelity::LoFi)
        {
            hal::math_counters.release<Counters::SrcA | Counters::SrcB>().apply();
        }
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
    static_assert(binary_reuse_dest != EltwiseBinaryReuseDestType::NONE, "Dest reuse requires DEST_TO_SRCA or DEST_TO_SRCB");
    static_assert(src_b_bcast_type == BroadcastType::NONE, "Dest reuse supports BroadcastType::NONE only");
    namespace cfg = hal_llk::cfg;

    const std::uint32_t dest_row = hal_llk::dest_section_base() + dst_index * hal_llk::DEST_ROWS_PER_TILE;
    cfg::write<cfg::Access::TensixCfgUnit, cfg::DestTargetRegCfgMath::Offset, cfg::Sec::S0>(dest_row);

    constexpr MathFidelity effective_fidelity = (eltwise_binary_type == EltwiseBinaryType::ELWMUL) ? math_fidelity : MathFidelity::LoFi;
    eltwise_binary_run_with_dest_reuse<is_fp32_dest_acc_en, binary_reuse_dest, effective_fidelity>(
        tensor_shape.total_num_faces(), 0, clear_fp32_dst_acc, 0, tensor_shape.face_r_dim);

    hal::math_counters.set<hal_llk::math::Counters::Dest, 0>().apply();
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
