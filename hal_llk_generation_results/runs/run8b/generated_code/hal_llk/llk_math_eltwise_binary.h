// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// HAL implementation of FPU elementwise binary operations (Blackhole).

#pragma once

#include <cstdint>

#include "llk_hal_common.h"
#include "llk_math_common.h"

using namespace ckernel;

namespace hal_llk
{

// Math address-modifier slots used by elementwise binary.
constexpr std::uint8_t ELTWISE_ADDR_MOD_STEP      = 0; // next 8 rows of SrcA/SrcB/Dest
constexpr std::uint8_t ELTWISE_ADDR_MOD_HOLD      = 1; // no counter change
constexpr std::uint8_t ELTWISE_ADDR_MOD_FACE_DONE = 2; // restart Src rows, Dest to the next face

struct MathAddrMod
{
    std::uint32_t srca_incr  = 0;
    bool srca_clear          = false;
    std::uint32_t srcb_incr  = 0;
    bool srcb_clear          = false;
    std::uint32_t dest_incr  = 0;
    bool dest_clear          = false;
    bool dest_carry_reload   = false;
    bool dest_copy_to_carry  = false;
    std::uint32_t fidelity_incr = 0;
    bool fidelity_clear      = false;
};

// Program every field of one math ADDR_MOD slot (each 16-bit thread-CFG word is written whole).
template <cfg::Sec Slot, MathAddrMod M>
inline void math_configure_addr_mod()
{
    cfg::write<cfg::Access::TensixCfgUnit>(
        cfg::set<cfg::AddrMod[cfg::SrcA].Incr, Slot, M.srca_incr & 0x3f>(),
        cfg::set<cfg::AddrMod[cfg::SrcA].CR, Slot, 0>(),
        cfg::set<cfg::AddrMod[cfg::SrcA].Clear, Slot, M.srca_clear ? 1 : 0>(),
        cfg::set<cfg::AddrMod[cfg::SrcB].Incr, Slot, M.srcb_incr & 0x3f>(),
        cfg::set<cfg::AddrMod[cfg::SrcB].CR, Slot, 0>(),
        cfg::set<cfg::AddrMod[cfg::SrcB].Clear, Slot, M.srcb_clear ? 1 : 0>(),
        cfg::set<cfg::AddrMod[cfg::SrcA].Incr2, Slot, (M.srca_incr >> 6) & 0x1>(),
        cfg::set<cfg::AddrMod[cfg::SrcB].Incr2, Slot, (M.srcb_incr >> 6) & 0x1>(),
        cfg::set<cfg::AddrMod[cfg::Dest].Incr, Slot, M.dest_incr & 0x3ff>(),
        cfg::set<cfg::AddrMod[cfg::Dest].CR, Slot, M.dest_carry_reload ? 1 : 0>(),
        cfg::set<cfg::AddrMod[cfg::Dest].Clear, Slot, M.dest_clear ? 1 : 0>(),
        cfg::set<cfg::AddrMod[cfg::Dest].CToCR, Slot, M.dest_copy_to_carry ? 1 : 0>(),
        cfg::set<cfg::AddrMod[cfg::Fidelity].Incr, Slot, M.fidelity_incr>(),
        cfg::set<cfg::AddrMod[cfg::Fidelity].Clear, Slot, M.fidelity_clear ? 1 : 0>(),
        cfg::set<cfg::AddrMod[cfg::Bias].Incr, Slot, 0>(),
        cfg::set<cfg::AddrMod[cfg::Bias].Clear, Slot, 0>());
}

constexpr hal::fpu::ElementwiseOperation elementwise_operation(const EltwiseBinaryType type)
{
    return (type == EltwiseBinaryType::ELWADD)   ? hal::fpu::ElementwiseOperation::Add
           : (type == EltwiseBinaryType::ELWSUB) ? hal::fpu::ElementwiseOperation::Subtract
                                                 : hal::fpu::ElementwiseOperation::Multiply;
}

constexpr hal::fpu::SrcBBroadcast elementwise_broadcast(const BroadcastType bcast)
{
    return (bcast == BroadcastType::COL)      ? hal::fpu::SrcBBroadcast::Column
           : (bcast == BroadcastType::ROW)    ? hal::fpu::SrcBBroadcast::Row
           : (bcast == BroadcastType::SCALAR) ? hal::fpu::SrcBBroadcast::Scalar
                                              : hal::fpu::SrcBBroadcast::None;
}

// ELW instructions process 8 rows; a face takes face_r_dim / 8 of them (at least one).
inline std::uint32_t eltwise_blocks_per_face(const ckernel::TensorShape tensor_shape)
{
    return (tensor_shape.face_r_dim <= 8) ? 1 : (tensor_shape.face_r_dim >> 3);
}

// Faces that consume one SrcB bank (COL: one face row, SCALAR: the whole tile, otherwise one face).
template <BroadcastType bcast_type>
inline std::uint32_t eltwise_faces_per_srcb_bank(const ckernel::TensorShape tensor_shape)
{
    const std::uint32_t num_faces = tensor_shape.total_num_faces();
    if constexpr (bcast_type == BroadcastType::COL)
    {
        return (num_faces < 2) ? 1 : 2;
    }
    else if constexpr (bcast_type == BroadcastType::SCALAR)
    {
        return num_faces;
    }
    else
    {
        return 1;
    }
}

inline std::uint32_t eltwise_mop_runs_per_tile = 1;

template <EltwiseBinaryType eltwise_binary_type, BroadcastType bcast_type, MathFidelity math_fidelity>
inline void eltwise_binary_configure_addrmod_impl()
{
    static_assert(math_fidelity == MathFidelity::LoFi || eltwise_binary_type == EltwiseBinaryType::ELWMUL, "Only ELWMUL has fidelity phases");
    constexpr std::uint32_t srcb_incr = (bcast_type == BroadcastType::NONE || bcast_type == BroadcastType::COL) ? 8 : 0;
    math_configure_addr_mod<cfg::Sec::S0, MathAddrMod {.srca_incr = 8, .srcb_incr = srcb_incr, .dest_incr = 8}>();
    math_configure_addr_mod<cfg::Sec::S1, MathAddrMod {}>();
}

// Face end: restart the Src rows and hand the consumed bank(s) back to the unpacker.
constexpr std::uint32_t FACE_DONE_RELEASE_BOTH =
    hal::math_counters.set<hal::rwc::Counters::SrcA | hal::rwc::Counters::SrcB, 0>().release<hal::rwc::Counters::SrcA | hal::rwc::Counters::SrcB>().get_operation();
constexpr std::uint32_t FACE_DONE_RELEASE_SRCA =
    hal::math_counters.set<hal::rwc::Counters::SrcA | hal::rwc::Counters::SrcB, 0>().release<hal::rwc::Counters::SrcA>().get_operation();
// Faces of up to 8 rows take one 8-row block; Dest still advances by a whole face.
constexpr std::uint32_t PARTIAL_FACE_DEST_PAD = hal::math_counters.increment<hal::rwc::Counters::Dest, 8>().get_operation();

// Program the MOP: outer loop over the faces sharing one SrcB bank, inner loop over 8-row blocks.
template <EltwiseBinaryType eltwise_binary_type, BroadcastType bcast_type>
inline void eltwise_binary_program_mop(const std::uint32_t acc_to_dest, const std::uint32_t faces_per_run, const std::uint32_t blocks_per_face, const bool srcb_per_face)
{
    using namespace hal::fpu;
    const DestWriteMode dest_write = acc_to_dest ? DestWriteMode::Accumulate : DestWriteMode::Overwrite;
    constexpr ElementwiseOperation op = elementwise_operation(eltwise_binary_type);
    constexpr SrcBBroadcast bcast     = elementwise_broadcast(bcast_type);

    hal::mop::MopConfig<hal::mop::MopTemplate::Template1> config {};
    config.outer_loop.count  = faces_per_run;
    config.outer_loop.end_op = srcb_per_face ? FACE_DONE_RELEASE_BOTH : FACE_DONE_RELEASE_SRCA;
    if (blocks_per_face * 8 < DEST_FACE_ROWS)
    {
        config.outer_loop.second_end_op = PARTIAL_FACE_DEST_PAD;
    }
    config.inner_loop.count = blocks_per_face;
    config.inner_loop.body_op =
        get_operation(Elementwise {.operation = op, .broadcast = bcast, .dest_write = dest_write, .address_modifier = ELTWISE_ADDR_MOD_STEP, .release = SourceRelease::None});
    hal::mop::program(config);
}

// Zero the Dest faces a multiply accumulates into (ELWMUL always accumulates).
template <bool is_fp32_dest_acc_en>
inline void eltwise_binary_zero_dest_faces(const std::uint32_t dst_index, const std::uint32_t first_face, const std::uint32_t num_faces)
{
    for (std::uint32_t face = first_face; face < first_face + num_faces; face++)
    {
        hal::dst::run(hal::dst::Zero {
            .scope = hal::dst::ZeroScope::Face,
            .index = (math_dest_bank_base() + dst_index * DEST_TILE_ROWS) / DEST_FACE_ROWS + face,
            .width = is_fp32_dest_acc_en ? hal::dst::DestWidth::Bits32 : hal::dst::DestWidth::Bits16});
    }
}

// Copy the current Dest face (Dest counter) into the face rows of SrcA or SrcB.
template <typename Destination, std::uint32_t Row>
constexpr hal::move::Transfer<hal::move::Dst, Destination> dest_rows_to_src()
{
    return {
        .source_row      = Row,
        .destination_row = Row,
        .address_mode    = ELTWISE_ADDR_MOD_HOLD,
        .number_of_rows  = hal::move::rows::Four,
    };
}

template <EltwiseBinaryReuseDestType binary_reuse_dest>
inline void eltwise_binary_move_dest_face_to_src()
{
    namespace sync = hal::sync;
    // MOVD2A / MOVD2B do not wait for Src bank ownership: wait for the unpacked bank and settled Dest.
    if constexpr (binary_reuse_dest == EltwiseBinaryReuseDestType::DEST_TO_SRCA)
    {
        using Destination = hal::move::SrcA;
        sync::wait::stall<sync::StallTarget::Math, sync::StallCondition::SrcAValid | sync::StallCondition::MathIdle>();
        hal::move::run<dest_rows_to_src<Destination, 0>()>();
        hal::move::run<dest_rows_to_src<Destination, 4>()>();
        hal::move::run<dest_rows_to_src<Destination, 8>()>();
        hal::move::run<dest_rows_to_src<Destination, 12>()>();
    }
    else
    {
        using Destination = hal::move::SrcB;
        sync::wait::stall<sync::StallTarget::Math, sync::StallCondition::SrcBValid | sync::StallCondition::MathIdle>();
        hal::move::run<dest_rows_to_src<Destination, 0>()>();
        hal::move::run<dest_rows_to_src<Destination, 4>()>();
        hal::move::run<dest_rows_to_src<Destination, 8>()>();
        hal::move::run<dest_rows_to_src<Destination, 12>()>();
    }
}

// Select Dest tile `dst_index` of the current section (the Dest counter is at row 0 of the tile window).
inline void eltwise_binary_select_dest_tile(const std::uint32_t dst_index)
{
    math_set_dest_offset(math_dest_bank_base() + dst_index * DEST_TILE_ROWS);
}

// Return the Dest counter to row 0 once a tile is done.
inline void eltwise_binary_tile_done()
{
    hal::math_counters.set<hal::rwc::Counters::Dest, 0>().apply();
}

} // namespace hal_llk

/*************************************************************************
 * Common Helpers
 *************************************************************************/

template <EltwiseBinaryType eltwise_binary_type, BroadcastType bcast_type, MathFidelity math_fidelity>
inline void eltwise_binary_configure_addrmod()
{
    hal_llk::eltwise_binary_configure_addrmod_impl<eltwise_binary_type, bcast_type, math_fidelity>();
}

template <EltwiseBinaryType eltwise_binary_type>
inline auto eltwise_binary_func(std::uint8_t clr_src, std::uint8_t acc_to_dest, std::uint8_t broadcast_type, std::uint8_t addr_mod)
{
    using namespace hal::fpu;
    const SourceRelease release = (clr_src == 3) ? SourceRelease::Both : (clr_src == 1) ? SourceRelease::SrcA : (clr_src == 2) ? SourceRelease::SrcB : SourceRelease::None;
    return get_operation(Elementwise {
        .operation        = hal_llk::elementwise_operation(eltwise_binary_type),
        .broadcast        = hal_llk::elementwise_broadcast(static_cast<BroadcastType>(broadcast_type)),
        .dest_write       = acc_to_dest ? DestWriteMode::Accumulate : DestWriteMode::Overwrite,
        .address_modifier = addr_mod,
        .release          = release});
}

/*************************************************************************
 * Eltwise Binary Standard (No Dest Reuse)
 *************************************************************************/

template <EltwiseBinaryType eltwise_binary_type, BroadcastType bcast_type, MathFidelity math_fidelity = MathFidelity::LoFi>
inline void eltwise_binary_configure_mop_standard(const std::uint32_t acc_to_dest, const ckernel::TensorShape tensor_shape)
{
    const std::uint32_t faces_per_bank = hal_llk::eltwise_faces_per_srcb_bank<bcast_type>(tensor_shape);
    const bool srcb_per_face           = (bcast_type == BroadcastType::NONE) || (bcast_type == BroadcastType::ROW);
    // NONE/ROW release both banks after every face, so one MOP run covers the whole tile.
    const std::uint32_t faces_per_run    = srcb_per_face ? tensor_shape.total_num_faces() : faces_per_bank;
    hal_llk::eltwise_mop_runs_per_tile   = tensor_shape.total_num_faces() / faces_per_run;
    hal_llk::eltwise_binary_program_mop<eltwise_binary_type, bcast_type>(acc_to_dest, faces_per_run, hal_llk::eltwise_blocks_per_face(tensor_shape), srcb_per_face);
}

template <EltwiseBinaryType eltwise_binary_type, BroadcastType src_b_bcast_type, MathFidelity math_fidelity = MathFidelity::LoFi>
inline void _llk_math_eltwise_binary_standard_init_(const ckernel::TensorShape tensor_shape, const std::uint32_t acc_to_dest)
{
    eltwise_binary_configure_addrmod<eltwise_binary_type, src_b_bcast_type, math_fidelity>();
    eltwise_binary_configure_mop_standard<eltwise_binary_type, src_b_bcast_type, math_fidelity>(acc_to_dest, tensor_shape);
    hal::math_counters.set<hal::rwc::Counters::All, 0>().apply();
}

template <
    EltwiseBinaryType eltwise_binary_type,
    BroadcastType src_b_bcast_type,
    DstSync Dst,
    bool is_fp32_dest_acc_en,
    MathFidelity math_fidelity = MathFidelity::LoFi>
inline void _llk_math_eltwise_binary_standard_(const ckernel::TensorShape tensor_shape, std::uint32_t dst_index)
{
    if constexpr (eltwise_binary_type == EltwiseBinaryType::ELWMUL)
    {
        hal_llk::eltwise_binary_zero_dest_faces<is_fp32_dest_acc_en>(dst_index, 0, tensor_shape.total_num_faces());
    }
    hal_llk::eltwise_binary_select_dest_tile(dst_index);
    for (std::uint32_t run = 0; run < hal_llk::eltwise_mop_runs_per_tile; run++)
    {
        hal::mop::Runner<hal::mop::MopTemplate::Template1>::run();
        if constexpr (src_b_bcast_type == BroadcastType::COL || src_b_bcast_type == BroadcastType::SCALAR)
        {
            hal::math_counters.release<hal::rwc::Counters::SrcB>().apply();
        }
    }
    hal_llk::eltwise_binary_tile_done();
}

/*************************************************************************
 * Eltwise Binary WITH Dest Reuse
 *************************************************************************/

template <EltwiseBinaryReuseDestType binary_reuse_dest>
inline void eltwise_binary_reuse_dest_as_src()
{
    hal_llk::eltwise_binary_move_dest_face_to_src<binary_reuse_dest>();
}

template <EltwiseBinaryType eltwise_binary_type, BroadcastType bcast_type, MathFidelity math_fidelity = MathFidelity::LoFi>
inline void eltwise_binary_configure_mop_with_dest_reuse(const std::uint32_t acc_to_dest, const ckernel::TensorShape tensor_shape)
{
    // One face per MOP run: the Dest face has to be copied into a Src bank before it is computed.
    hal_llk::eltwise_binary_program_mop<eltwise_binary_type, bcast_type>(acc_to_dest, 1, hal_llk::eltwise_blocks_per_face(tensor_shape), true);
}

template <
    EltwiseBinaryType eltwise_binary_type,
    BroadcastType src_b_bcast_type,
    MathFidelity math_fidelity                   = MathFidelity::LoFi,
    EltwiseBinaryReuseDestType binary_reuse_dest = EltwiseBinaryReuseDestType::DEST_TO_SRCA>
inline void _llk_math_eltwise_binary_with_dest_reuse_init_(const ckernel::TensorShape tensor_shape, const std::uint32_t acc_to_dest)
{
    static_assert(binary_reuse_dest != EltwiseBinaryReuseDestType::NONE, "Dest reuse init requires DEST_TO_SRCA or DEST_TO_SRCB");
    static_assert(src_b_bcast_type == BroadcastType::NONE, "Dest reuse supports no SrcB broadcast");
    eltwise_binary_configure_addrmod<eltwise_binary_type, src_b_bcast_type, math_fidelity>();
    eltwise_binary_configure_mop_with_dest_reuse<eltwise_binary_type, src_b_bcast_type, math_fidelity>(acc_to_dest, tensor_shape);
    hal::math_counters.set<hal::rwc::Counters::All, 0>().apply();
}

template <bool is_fp32_dest_acc_en, EltwiseBinaryReuseDestType binary_reuse_dest, MathFidelity math_fidelity = MathFidelity::LoFi>
inline void eltwise_binary_run_with_dest_reuse(
    const std::uint32_t loop_count,
    const std::uint32_t face_offset,
    [[maybe_unused]] const bool clear_fp32_dst_acc,
    [[maybe_unused]] const std::uint32_t dst_index,
    [[maybe_unused]] const std::uint32_t face_r_dim)
{
    for (std::uint32_t face = face_offset; face < face_offset + loop_count; face++)
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
    hal_llk::eltwise_binary_select_dest_tile(dst_index);
    const std::uint32_t num_faces = tensor_shape.total_num_faces();
    if constexpr (eltwise_binary_type == EltwiseBinaryType::ELWMUL)
    {
        // The multiply accumulates: each face is copied to Src first, then its Dest rows are zeroed.
        for (std::uint32_t face = 0; face < num_faces; face++)
        {
            eltwise_binary_reuse_dest_as_src<binary_reuse_dest>();
            hal_llk::eltwise_binary_zero_dest_faces<is_fp32_dest_acc_en>(dst_index, face, 1);
            hal::mop::Runner<hal::mop::MopTemplate::Template1>::run();
        }
    }
    else
    {
        eltwise_binary_run_with_dest_reuse<is_fp32_dest_acc_en, binary_reuse_dest, math_fidelity>(num_faces, 0, clear_fp32_dst_acc, dst_index, tensor_shape.face_r_dim);
    }
    hal_llk::eltwise_binary_tile_done();
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
