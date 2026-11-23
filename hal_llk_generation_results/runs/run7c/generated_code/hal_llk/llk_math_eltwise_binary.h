// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// HAL-based implementation of the elementwise binary FPU LLK (Blackhole).
//
// Model: the MOP processes exactly one face (face_r_dim rows, eight rows per FPU instruction, repeated once per
// fidelity phase). The execute call runs it once per face; the last instruction of each face releases the
// consumed source banks and steps the Dest counter to the next 16-row face slot.

#pragma once

#include <cstdint>

#include "ckernel.h"
#include "ckernel_defs.h"
#include "hal/cfg.h"
#include "hal/dst.h"
#include "hal/fpu.h"
#include "hal/math_counters.h"
#include "hal/mop.h"
#include "hal/move.h"
#include "hal/sync.h"
#include "llk_assert.h"
#include "llk_defs.h"
#include "llk_hal_common.h"
#include "llk_math_common.h"
#include "tensor_shape.h"

namespace hal_llk::eltwise_binary_detail
{

// ADDR_MOD slots used by this op.
constexpr std::uint8_t ADDR_MOD_STEP     = 0; // next eight rows of the face
constexpr std::uint8_t ADDR_MOD_HOLD     = 1; // no counter change
constexpr std::uint8_t ADDR_MOD_FIDELITY = 2; // next fidelity phase of the same rows
constexpr std::uint8_t ADDR_MOD_FACE     = 3; // next face slot, source counters back to row 0

constexpr std::uint32_t ROWS_PER_INSTRUCTION = 8;

constexpr std::uint32_t fidelity_phases(const MathFidelity fidelity)
{
    return (fidelity == MathFidelity::LoFi) ? 1 : static_cast<std::uint32_t>(fidelity);
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

// SrcB keeps its rows for row/scalar broadcast; column broadcast and no broadcast walk SrcB with SrcA.
constexpr std::uint8_t srcb_row_step(const BroadcastType type)
{
    return (type == BroadcastType::NONE || type == BroadcastType::COL) ? ROWS_PER_INSTRUCTION : 0;
}

// Column/scalar broadcast keep one SrcB bank across several faces: the face only returns SrcA.
constexpr hal::fpu::SourceRelease face_release(const BroadcastType type)
{
    return (type == BroadcastType::COL || type == BroadcastType::SCALAR) ? hal::fpu::SourceRelease::SrcA : hal::fpu::SourceRelease::Both;
}

inline std::uint32_t instructions_per_face(const std::uint32_t face_r_dim)
{
    return (face_r_dim <= ROWS_PER_INSTRUCTION) ? 1 : (face_r_dim / ROWS_PER_INSTRUCTION);
}

template <EltwiseBinaryType type, BroadcastType bcast>
inline std::uint32_t elementwise_op(const std::uint8_t addr_mod, const hal::fpu::SourceRelease release, const std::uint32_t acc_to_dest)
{
    return hal::fpu::get_operation(hal::fpu::Elementwise {
        .operation        = fpu_operation(type),
        .broadcast        = fpu_broadcast(bcast),
        .dest_write       = acc_to_dest ? hal::fpu::DestWriteMode::Accumulate : hal::fpu::DestWriteMode::Overwrite,
        .address_modifier = addr_mod,
        .release          = release,
    });
}

// SrcB bank boundaries for broadcasts that share one SrcB bank across faces.
template <BroadcastType bcast>
inline bool face_ends_srcb_bank(const std::uint32_t face, const ckernel::TensorShape tensor_shape)
{
    if constexpr (bcast == BroadcastType::COL)
    {
        return ((face + 1) % tensor_shape.num_faces_c_dim) == 0;
    }
    else if constexpr (bcast == BroadcastType::SCALAR)
    {
        return (face + 1) == tensor_shape.total_num_faces();
    }
    return false;
}

// Select the tile in DEST and restart the SrcA/SrcB/Dest counters and the fidelity phase.
inline void select_dest_tile(const std::uint32_t dst_index)
{
    using hal::rwc::Counters;
    hal_llk::math_detail::set_dest_window(hal_llk::math_detail::dest_section_base() + dst_index * hal_llk::DEST_ROWS_PER_TILE);
    hal::math_counters.set<Counters::All, 0>().clear_fidelity().apply();
}

template <BroadcastType bcast>
inline void release_shared_srcb(const std::uint32_t face, const ckernel::TensorShape tensor_shape)
{
    if (face_ends_srcb_bank<bcast>(face, tensor_shape))
    {
        hal::math_counters.release<hal::rwc::Counters::SrcB>().apply();
    }
}

} // namespace hal_llk::eltwise_binary_detail

/**
 * @brief Program the address-mod slots used by eltwise binary MOPs (per-row, no-op, fidelity-step, face-step).
 */
template <EltwiseBinaryType eltwise_binary_type, BroadcastType bcast_type, MathFidelity math_fidelity>
inline void eltwise_binary_configure_addrmod()
{
    using namespace hal_llk;
    using namespace hal_llk::eltwise_binary_detail;
    namespace cfg = hal::cfg;

    constexpr bool high_fidelity = fidelity_phases(math_fidelity) > 1;
    constexpr std::uint8_t srcb  = srcb_row_step(bcast_type);

    // Within a face the live Dest counter steps by eight rows while its carry shadow keeps the face start.
    // The face step advances the carry by one 16-row face slot and reloads the live counter from it, so the
    // next face starts on its slot whatever the face height; SrcA/SrcB restart at row 0.
    program_math_addr_mod<cfg::Sec::S0>(
        MathAddrMod {.srca_incr = ROWS_PER_INSTRUCTION, .srcb_incr = srcb, .dest_incr = ROWS_PER_INSTRUCTION, .fid_clear = high_fidelity});
    program_math_addr_mod<cfg::Sec::S1>(MathAddrMod {});
    program_math_addr_mod<cfg::Sec::S2>(MathAddrMod {.fid_incr = 1});
    program_math_addr_mod<cfg::Sec::S3>(
        MathAddrMod {.srca_clear = true, .srcb_clear = true, .dest_incr = FACE_R_DIM, .dest_cr = true, .fid_clear = high_fidelity});
}

/**
 * @brief Configure the one-face MOP shared by the standard and the dest-reuse eltwise binary paths.
 */
template <EltwiseBinaryType eltwise_binary_type, BroadcastType bcast_type, MathFidelity math_fidelity = MathFidelity::LoFi>
inline void eltwise_binary_configure_mop_standard(const std::uint32_t acc_to_dest, const ckernel::TensorShape tensor_shape)
{
    using namespace hal_llk::eltwise_binary_detail;
    using hal::fpu::SourceRelease;

    constexpr std::uint32_t phases = fidelity_phases(math_fidelity);

    // Outer loop: eight-row blocks of the face. Inner loop: fidelity phases of one block.
    hal::mop::MopConfig<hal::mop::MopTemplate::Template1> mop {};
    mop.outer_loop.count                               = instructions_per_face(tensor_shape.face_r_dim);
    mop.inner_loop.count                               = phases;
    mop.inner_loop.body_op                             = elementwise_op<eltwise_binary_type, bcast_type>(ADDR_MOD_FIDELITY, SourceRelease::None, acc_to_dest);
    mop.inner_loop.last_op_on_nonfinal_outer_iteration = elementwise_op<eltwise_binary_type, bcast_type>(ADDR_MOD_STEP, SourceRelease::None, acc_to_dest);
    mop.inner_loop.last_op_on_final_outer_iteration    = elementwise_op<eltwise_binary_type, bcast_type>(ADDR_MOD_FACE, face_release(bcast_type), acc_to_dest);
    hal::mop::program(mop);
}

template <EltwiseBinaryType eltwise_binary_type, BroadcastType bcast_type, MathFidelity math_fidelity = MathFidelity::LoFi>
inline void eltwise_binary_configure_mop_with_dest_reuse(const std::uint32_t acc_to_dest, const ckernel::TensorShape tensor_shape)
{
    eltwise_binary_configure_mop_standard<eltwise_binary_type, bcast_type, math_fidelity>(acc_to_dest, tensor_shape);
}

/**
 * @brief Initialize FPU for standard elementwise binary operations (no dest reuse).
 */
template <EltwiseBinaryType eltwise_binary_type, BroadcastType src_b_bcast_type, MathFidelity math_fidelity = MathFidelity::LoFi>
inline void _llk_math_eltwise_binary_standard_init_(const ckernel::TensorShape tensor_shape, const std::uint32_t acc_to_dest)
{
    namespace cfg = hal::cfg;
    using namespace hal_llk;

    // FP compute: Src zero substitution stays enabled unless an operand is 16-bit integer.
    const bool keep_src_zeros =
        (math_detail::srca_format == fmt(DataFormat::UInt16)) || (math_detail::srcb_format == fmt(DataFormat::UInt16));
    hal::sync::mutex::acquire<hal::sync::Mutex::M0>();
    cfg::write<cfg::Access::TensixCfgUnit, cfg::AluAccCtrl::Zero_Flag_disabled_src, cfg::Sec::S0>(keep_src_zeros ? 1 : 0);
    hal::sync::mutex::release<hal::sync::Mutex::M0>();

    eltwise_binary_configure_addrmod<eltwise_binary_type, src_b_bcast_type, math_fidelity>();
    eltwise_binary_configure_mop_standard<eltwise_binary_type, src_b_bcast_type, math_fidelity>(acc_to_dest, tensor_shape);
    hal::math_counters.set<hal::rwc::Counters::All, 0>().clear_fidelity().apply();
}

/**
 * @brief Perform standard elementwise binary operation (no dest reuse): Output = SrcA [+, -, *] SrcB.
 */
template <
    EltwiseBinaryType eltwise_binary_type,
    BroadcastType src_b_bcast_type,
    DstSync Dst,
    bool is_fp32_dest_acc_en,
    MathFidelity math_fidelity = MathFidelity::LoFi>
inline void _llk_math_eltwise_binary_standard_(const ckernel::TensorShape tensor_shape, std::uint32_t dst_index)
{
    using namespace hal_llk::eltwise_binary_detail;

    select_dest_tile(dst_index);
    const std::uint32_t num_faces = tensor_shape.total_num_faces();
    for (std::uint32_t face = 0; face < num_faces; face++)
    {
        hal::mop::Runner<hal::mop::MopTemplate::Template1>::run();
        release_shared_srcb<src_b_bcast_type>(face, tensor_shape);
    }
}

/**
 * @brief Move one face of the destination register into a source register (SrcA or SrcB) for dest-reuse ops.
 *
 * The Dest counter points at the face; four four-row moves copy it into rows 0..15 of the current source bank.
 */
template <EltwiseBinaryReuseDestType binary_reuse_dest>
inline void eltwise_binary_reuse_dest_as_src()
{
    using namespace hal_llk::eltwise_binary_detail;
    namespace sync = hal::sync;

    if constexpr (binary_reuse_dest == EltwiseBinaryReuseDestType::DEST_TO_SRCA)
    {
        // MOVD2A does not wait for bank ownership: wait for the unpacked bank and for Dest writes to settle.
        sync::wait::stall<sync::StallTarget::Math, sync::StallCondition::SrcAValid | sync::StallCondition::MathIdle>();
        hal::move::run<hal::move::Transfer<hal::move::Dst, hal::move::SrcA> {.source_row = 0, .destination_row = 0, .address_mode = ADDR_MOD_HOLD, .number_of_rows = hal::move::rows::Four}>();
        hal::move::run<hal::move::Transfer<hal::move::Dst, hal::move::SrcA> {.source_row = 4, .destination_row = 4, .address_mode = ADDR_MOD_HOLD, .number_of_rows = hal::move::rows::Four}>();
        hal::move::run<hal::move::Transfer<hal::move::Dst, hal::move::SrcA> {.source_row = 8, .destination_row = 8, .address_mode = ADDR_MOD_HOLD, .number_of_rows = hal::move::rows::Four}>();
        hal::move::run<hal::move::Transfer<hal::move::Dst, hal::move::SrcA> {.source_row = 12, .destination_row = 12, .address_mode = ADDR_MOD_HOLD, .number_of_rows = hal::move::rows::Four}>();
    }
    else if constexpr (binary_reuse_dest == EltwiseBinaryReuseDestType::DEST_TO_SRCB)
    {
        sync::wait::stall<sync::StallTarget::Math, sync::StallCondition::SrcBValid | sync::StallCondition::MathIdle>();
        hal::move::run<hal::move::Transfer<hal::move::Dst, hal::move::SrcB> {.source_row = 0, .destination_row = 0, .address_mode = ADDR_MOD_HOLD, .number_of_rows = hal::move::rows::Four}>();
        hal::move::run<hal::move::Transfer<hal::move::Dst, hal::move::SrcB> {.source_row = 4, .destination_row = 4, .address_mode = ADDR_MOD_HOLD, .number_of_rows = hal::move::rows::Four}>();
        hal::move::run<hal::move::Transfer<hal::move::Dst, hal::move::SrcB> {.source_row = 8, .destination_row = 8, .address_mode = ADDR_MOD_HOLD, .number_of_rows = hal::move::rows::Four}>();
        hal::move::run<hal::move::Transfer<hal::move::Dst, hal::move::SrcB> {.source_row = 12, .destination_row = 12, .address_mode = ADDR_MOD_HOLD, .number_of_rows = hal::move::rows::Four}>();
    }
}

/**
 * @brief Clear the 16 DEST rows of the face the Dest counter points at (one-row ZEROACC each).
 */
template <bool is_fp32_dest_acc_en>
inline void zeroacc_face_by_row()
{
    constexpr hal::dst::DestWidth width = is_fp32_dest_acc_en ? hal::dst::DestWidth::Bits32 : hal::dst::DestWidth::Bits16;
    for (std::uint32_t row = 0; row < FACE_R_DIM; row++)
    {
        hal::dst::run(hal::dst::Zero {
            .scope        = hal::dst::ZeroScope::SingleRow,
            .index        = row,
            .width        = width,
            .address_mode = hal_llk::eltwise_binary_detail::ADDR_MOD_HOLD,
        });
    }
}

template <
    EltwiseBinaryType eltwise_binary_type,
    BroadcastType src_b_bcast_type,
    MathFidelity math_fidelity                   = MathFidelity::LoFi,
    EltwiseBinaryReuseDestType binary_reuse_dest = EltwiseBinaryReuseDestType::DEST_TO_SRCA>
inline void _llk_math_eltwise_binary_with_dest_reuse_init_(const ckernel::TensorShape tensor_shape, const std::uint32_t acc_to_dest)
{
    static_assert(binary_reuse_dest != EltwiseBinaryReuseDestType::NONE, "dest reuse init requires DEST_TO_SRCA or DEST_TO_SRCB");
    static_assert(src_b_bcast_type == BroadcastType::NONE, "dest reuse supports no SrcB broadcast");
    _llk_math_eltwise_binary_standard_init_<eltwise_binary_type, src_b_bcast_type, math_fidelity>(tensor_shape, acc_to_dest);
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
        zeroacc_face_by_row<is_fp32_dest_acc_en>();
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
    hal_llk::eltwise_binary_detail::select_dest_tile(dst_index);
    eltwise_binary_run_with_dest_reuse<is_fp32_dest_acc_en, binary_reuse_dest, math_fidelity>(
        tensor_shape.total_num_faces(), 0, clear_fp32_dst_acc, dst_index, tensor_shape.face_r_dim);
}

/**
 * @brief Initialize FPU to perform an elementwise binary operation where Output = SrcA [+, -, *] SrcB.
 */
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

/**
 * @brief Perform an elementwise binary operation where Output = SrcA [+, -, *] SrcB.
 */
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
