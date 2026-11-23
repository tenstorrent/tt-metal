// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// HAL-based implementation of the FPU elementwise binary LLK (ELWADD / ELWSUB / ELWMUL).
//
// The unpacker hands over one (SrcA, SrcB) bank pair per output face. For every face the math thread
// issues one elementwise instruction per 8-row block (repeated once per fidelity phase for multiply),
// writing the face at its absolute Dest row, and releases both banks on the face's final instruction.
// With Dest reuse, the face currently held in Dest is first moved into SrcA or SrcB and replaces the
// unpacked operand.

#pragma once

#include <cstdint>

#include "ckernel.h"
#include "ckernel_globals.h"
#include "hal/cfg.h"
#include "hal/dst.h"
#include "hal/fpu.h"
#include "hal/math_counters.h"
#include "hal/move.h"
#include "hal/sync.h"
#include "hal_llk_common.h"
#include "llk_defs.h"
#include "llk_math_common.h"
#include "tensor_shape.h"

using namespace ckernel;

namespace hal_llk::math_detail
{

namespace cfg = hal::cfg;

// ADDR_MOD slots used by the elementwise binary sequence.
constexpr std::uint8_t ADDR_MOD_NEXT_BLOCK  = 0; // next 8-row block of the same face
constexpr std::uint8_t ADDR_MOD_FACE_DONE   = 1; // source banks restart for the next face
constexpr std::uint8_t ADDR_MOD_NEXT_PHASE  = 2; // same block, next fidelity phase
constexpr std::uint8_t ADDR_MOD_MOVE_STEP   = 3; // Dest -> Src move: next 4 rows
constexpr std::uint8_t ADDR_MOD_MOVE_DONE   = 4; // Dest -> Src move finished: counters back to zero
constexpr std::uint32_t MOVE_FRAGMENT_ROWS  = 4;

// Geometry of the tile configured by the last _llk_math_eltwise_binary_init_.
struct BinaryGeometry
{
    std::uint32_t num_faces   = 4;
    std::uint32_t face_blocks = 2; // 8-row blocks per face
    bool acc_to_dest          = false;
};

inline BinaryGeometry binary_geometry {};

constexpr hal::fpu::ElementwiseOperation fpu_operation(const EltwiseBinaryType type)
{
    return type == EltwiseBinaryType::ELWADD   ? hal::fpu::ElementwiseOperation::Add
           : type == EltwiseBinaryType::ELWSUB ? hal::fpu::ElementwiseOperation::Subtract
                                               : hal::fpu::ElementwiseOperation::Multiply;
}

constexpr hal::fpu::SrcBBroadcast fpu_broadcast(const BroadcastType type)
{
    return type == BroadcastType::COL      ? hal::fpu::SrcBBroadcast::Column
           : type == BroadcastType::ROW    ? hal::fpu::SrcBBroadcast::Row
           : type == BroadcastType::SCALAR ? hal::fpu::SrcBBroadcast::Scalar
                                           : hal::fpu::SrcBBroadcast::None;
}

template <EltwiseBinaryType type, MathFidelity math_fidelity>
constexpr std::uint32_t fidelity_phases()
{
    constexpr std::uint32_t phases = static_cast<std::uint32_t>(math_fidelity);
    return (type == EltwiseBinaryType::ELWMUL && phases > 1) ? phases : 1;
}

template <BroadcastType bcast_type>
inline void configure_addrmod()
{
    // SrcB stays on its first row when a single row (or datum) is broadcast.
    constexpr std::uint32_t srcb_block_step = (bcast_type == BroadcastType::ROW || bcast_type == BroadcastType::SCALAR) ? 0 : FPU_BLOCK_ROWS;

    cfg::write<cfg::Access::TensixCfgUnit>(
        // Next block: advance both sources by one block, keep Dest (rows are addressed by the instruction).
        cfg::set<cfg::AddrMod[cfg::SrcA].Incr, cfg::Sec::S0, FPU_BLOCK_ROWS>(),
        cfg::set<cfg::AddrMod[cfg::SrcB].Incr, cfg::Sec::S0, srcb_block_step>(),
        cfg::set<cfg::AddrMod[cfg::Fidelity].Clear, cfg::Sec::S0, 1>(),
        // Face done: both sources restart at row 0 of the next bank.
        cfg::set<cfg::AddrMod[cfg::SrcA].Clear, cfg::Sec::S1, 1>(),
        cfg::set<cfg::AddrMod[cfg::SrcB].Clear, cfg::Sec::S1, 1>(),
        cfg::set<cfg::AddrMod[cfg::Fidelity].Clear, cfg::Sec::S1, 1>(),
        // Next fidelity phase: same rows, next phase.
        cfg::set<cfg::AddrMod[cfg::SrcA].Incr, cfg::Sec::S2, 0>(),
        cfg::set<cfg::AddrMod[cfg::Fidelity].Incr, cfg::Sec::S2, 1>(),
        // Dest -> Src move fragment: advance Dest and both sources by four rows.
        cfg::set<cfg::AddrMod[cfg::SrcA].Incr, cfg::Sec::S3, MOVE_FRAGMENT_ROWS>(),
        cfg::set<cfg::AddrMod[cfg::SrcB].Incr, cfg::Sec::S3, MOVE_FRAGMENT_ROWS>(),
        cfg::set<cfg::AddrMod[cfg::Dest].Incr, cfg::Sec::S3, MOVE_FRAGMENT_ROWS>(),
        // Dest -> Src move done: every counter back to zero.
        cfg::set<cfg::AddrMod[cfg::SrcA].Clear, cfg::Sec::S4, 1>(),
        cfg::set<cfg::AddrMod[cfg::SrcB].Clear, cfg::Sec::S4, 1>(),
        cfg::set<cfg::AddrMod[cfg::Dest].Clear, cfg::Sec::S4, 1>());

    // Thread ADDR_MOD words are replaced whole; the extension word (increment bit 6) must read zero.
    cfg::write<cfg::Access::TensixCfgUnit>(
        cfg::set<cfg::AddrMod[cfg::SrcA].Incr2, cfg::Sec::S0, 0>(),
        cfg::set<cfg::AddrMod[cfg::SrcA].Incr2, cfg::Sec::S1, 0>(),
        cfg::set<cfg::AddrMod[cfg::SrcA].Incr2, cfg::Sec::S2, 0>(),
        cfg::set<cfg::AddrMod[cfg::SrcA].Incr2, cfg::Sec::S3, 0>(),
        cfg::set<cfg::AddrMod[cfg::SrcA].Incr2, cfg::Sec::S4, 0>());
}

inline void configure_geometry(const ckernel::TensorShape tensor_shape, const std::uint32_t acc_to_dest)
{
    binary_geometry = BinaryGeometry {
        .num_faces   = tensor_shape.total_num_faces(),
        .face_blocks = (tensor_shape.face_r_dim > FPU_BLOCK_ROWS) ? 2u : 1u,
        .acc_to_dest = acc_to_dest != 0,
    };
}

inline void reset_counters()
{
    hal::math_counters.set<hal::rwc::Counters::All, 0>().clear_fidelity().apply();
}

// Issue the elementwise instructions of one face whose (SrcA, SrcB) banks are valid.
template <EltwiseBinaryType eltwise_binary_type, BroadcastType bcast_type, MathFidelity math_fidelity>
inline void run_face(const std::uint32_t face_dest_row, const BinaryGeometry& geometry)
{
    constexpr std::uint32_t phases = fidelity_phases<eltwise_binary_type, math_fidelity>();
    const hal::fpu::DestWriteMode write_mode = geometry.acc_to_dest ? hal::fpu::DestWriteMode::Accumulate : hal::fpu::DestWriteMode::Overwrite;

    for (std::uint32_t block = 0; block < geometry.face_blocks; block++)
    {
        const bool last_block = (block + 1) == geometry.face_blocks;
        for (std::uint32_t phase = 0; phase < phases; phase++)
        {
            const bool last_phase = (phase + 1) == phases;
            const std::uint8_t addr_mod =
                !last_phase ? ADDR_MOD_NEXT_PHASE : (last_block ? ADDR_MOD_FACE_DONE : ADDR_MOD_NEXT_BLOCK);
            hal::fpu::run(hal::fpu::Elementwise {
                .operation        = fpu_operation(eltwise_binary_type),
                .broadcast        = fpu_broadcast(bcast_type),
                .dest_write       = write_mode,
                .address_modifier = addr_mod,
                .dest_row_offset  = static_cast<std::uint16_t>(face_dest_row + block * FPU_BLOCK_ROWS),
                .release          = (last_phase && last_block) ? hal::fpu::SourceRelease::Both : hal::fpu::SourceRelease::None,
            });
        }
    }
}

// Replace the unpacked SrcA (or SrcB) face with the face currently held in Dest.
template <EltwiseBinaryReuseDestType binary_reuse_dest>
inline void move_dest_face_to_source(const std::uint32_t face_dest_row)
{
    using namespace hal::sync;
    if constexpr (binary_reuse_dest == EltwiseBinaryReuseDestType::DEST_TO_SRCA)
    {
        // MOVD2A does not wait for bank ownership: wait for the unpacked bank and for earlier Dest writes.
        wait::stall<StallTarget::Math, StallCondition::SrcAValid | StallCondition::MathIdle>();
        hal::move::run(hal::move::Transfer<hal::move::Dst, hal::move::SrcA> {
            .source_row         = face_dest_row,
            .destination_row    = 0, // FPU-relative: the unpacker's 4-row SrcA write offset does not apply
            .address_mode       = ADDR_MOD_MOVE_STEP,
            .number_of_rows     = hal::move::rows::Face,
            .final_address_mode = ADDR_MOD_MOVE_DONE,
        });
    }
    else
    {
        wait::stall<StallTarget::Math, StallCondition::SrcBValid | StallCondition::MathIdle>();
        hal::move::run(hal::move::Transfer<hal::move::Dst, hal::move::SrcB> {
            .source_row         = face_dest_row,
            .destination_row    = 0,
            .address_mode       = ADDR_MOD_MOVE_STEP,
            .number_of_rows     = hal::move::rows::Face,
            .final_address_mode = ADDR_MOD_MOVE_DONE,
        });
    }
}

template <EltwiseBinaryReuseDestType binary_reuse_dest>
inline void configure_implied_formats()
{
    // Sources written by Dest moves carry no unpacker-implied format: use the configured ALU format instead.
    constexpr std::uint32_t srca_disable = binary_reuse_dest == EltwiseBinaryReuseDestType::DEST_TO_SRCA ? 1 : 0;
    constexpr std::uint32_t srcb_disable = binary_reuse_dest == EltwiseBinaryReuseDestType::DEST_TO_SRCB ? 1 : 0;
    hal::sync::wait::stall<hal::sync::StallTarget::Config, hal::sync::StallCondition::MathIdle>();
    cfg::write<cfg::Access::TensixCfgUnit, cfg::DisableImpliedFmt[cfg::SrcA], cfg::Sec::S0, srca_disable>();
    cfg::write<cfg::Access::TensixCfgUnit, cfg::DisableImpliedFmt[cfg::SrcB], cfg::Sec::S0, srcb_disable>();
}

} // namespace hal_llk::math_detail

/**
 * @brief Program the address-mod slots used by eltwise binary operations.
 */
template <EltwiseBinaryType eltwise_binary_type, BroadcastType bcast_type, MathFidelity math_fidelity>
inline void eltwise_binary_configure_addrmod()
{
    hal_llk::math_detail::configure_addrmod<bcast_type>();
}

/**
 * @brief Initialize FPU for standard elementwise binary operations (no dest reuse).
 */
template <EltwiseBinaryType eltwise_binary_type, BroadcastType src_b_bcast_type, MathFidelity math_fidelity = MathFidelity::LoFi>
inline void _llk_math_eltwise_binary_standard_init_(const ckernel::TensorShape tensor_shape, const std::uint32_t acc_to_dest)
{
    using namespace hal_llk::math_detail;
    configure_implied_formats<EltwiseBinaryReuseDestType::NONE>();
    configure_addrmod<src_b_bcast_type>();
    configure_geometry(tensor_shape, acc_to_dest);
    reset_counters();
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
inline void _llk_math_eltwise_binary_standard_([[maybe_unused]] const ckernel::TensorShape tensor_shape, std::uint32_t dst_index)
{
    using namespace hal_llk::math_detail;
    const BinaryGeometry geometry = binary_geometry;
    const std::uint32_t tile_row  = dst_index * hal_llk::DEST_ROWS_PER_TILE;
    for (std::uint32_t face = 0; face < geometry.num_faces; face++)
    {
        run_face<eltwise_binary_type, src_b_bcast_type, math_fidelity>(tile_row + face * hal_llk::FACE_ROWS, geometry);
    }
}

/**
 * @brief Initialize FPU for elementwise binary operations with dest reuse.
 */
template <
    EltwiseBinaryType eltwise_binary_type,
    BroadcastType src_b_bcast_type,
    MathFidelity math_fidelity                   = MathFidelity::LoFi,
    EltwiseBinaryReuseDestType binary_reuse_dest = EltwiseBinaryReuseDestType::DEST_TO_SRCA>
inline void _llk_math_eltwise_binary_with_dest_reuse_init_(const ckernel::TensorShape tensor_shape, const std::uint32_t acc_to_dest)
{
    static_assert(binary_reuse_dest != EltwiseBinaryReuseDestType::NONE, "Dest reuse init requires DEST_TO_SRCA or DEST_TO_SRCB");
    using namespace hal_llk::math_detail;
    configure_implied_formats<binary_reuse_dest>();
    configure_addrmod<src_b_bcast_type>();
    configure_geometry(tensor_shape, acc_to_dest);
    reset_counters();
}

/**
 * @brief Perform elementwise binary operation with dest reuse.
 */
template <
    EltwiseBinaryType eltwise_binary_type,
    BroadcastType src_b_bcast_type,
    DstSync Dst,
    bool is_fp32_dest_acc_en,
    MathFidelity math_fidelity,
    EltwiseBinaryReuseDestType binary_reuse_dest>
inline void _llk_math_eltwise_binary_with_dest_reuse_(
    [[maybe_unused]] const ckernel::TensorShape tensor_shape, std::uint32_t dst_index, [[maybe_unused]] const bool clear_fp32_dst_acc)
{
    using namespace hal_llk::math_detail;
    const BinaryGeometry geometry = binary_geometry;
    const std::uint32_t tile_row  = dst_index * hal_llk::DEST_ROWS_PER_TILE;
    for (std::uint32_t face = 0; face < geometry.num_faces; face++)
    {
        const std::uint32_t face_row = tile_row + face * hal_llk::FACE_ROWS;
        move_dest_face_to_source<binary_reuse_dest>(face_row);
        if constexpr (eltwise_binary_type == EltwiseBinaryType::ELWMUL)
        {
            // Multiply always accumulates: the face must start from zero once its value sits in the source bank.
            hal::dst::run(hal::dst::Zero {
                .scope = hal::dst::ZeroScope::Face,
                .index = face_row / hal_llk::FACE_ROWS,
                .width = is_fp32_dest_acc_en ? hal::dst::DestWidth::Bits32 : hal::dst::DestWidth::Bits16,
            });
        }
        run_face<eltwise_binary_type, src_b_bcast_type, math_fidelity>(face_row, geometry);
    }
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

/**
 * @brief Uninitialize/cleanup after elementwise binary operations.
 */
inline void _llk_math_eltwise_binary_uninit_()
{
}
