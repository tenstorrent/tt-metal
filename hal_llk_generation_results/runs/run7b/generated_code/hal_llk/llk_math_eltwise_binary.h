// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// HAL-based implementation of the FPU elementwise binary operation (spec: llk_lib/llk_math_eltwise_binary.h).
//
// Model: the unpacker hands over one face per SrcA bank (and one face per SrcB bank, except for
// column/scalar broadcast where one SrcB face serves several SrcA faces). The math thread runs one
// MOP expansion per face. Faces are written to Dest at a 16-row pitch and tiles at a 64-row pitch,
// which is the layout the packer reads back.

#pragma once

#include <cstdint>

#include "llk_math_common.h"

namespace hal_llk::math
{

using hal::fpu::Elementwise;
using hal::fpu::ElementwiseOperation;
using hal::fpu::SourceRelease;

// ADDR_MOD slots used by the eltwise binary MOP.
constexpr std::uint8_t ADDR_MOD_BLOCK    = 0; // next 8-row block of the face, fidelity phase reset
constexpr std::uint8_t ADDR_MOD_HOLD     = 1; // no counter change (Dest -> Src moves)
constexpr std::uint8_t ADDR_MOD_FIDELITY = 2; // next fidelity phase on the same block
constexpr std::uint8_t ADDR_MOD_FACE_END = 3; // Src counters back to row 0, Dest to the next face

struct MathAddrMod
{
    std::uint32_t srca_incr = 0;
    std::uint32_t srca_clr  = 0;
    std::uint32_t srcb_incr = 0;
    std::uint32_t srcb_clr  = 0;
    std::uint32_t dest_incr = 0;
    std::uint32_t dest_cr   = 0;
    std::uint32_t dest_c2cr = 0;
    std::uint32_t fid_incr  = 0;
    std::uint32_t fid_clr   = 0;
};

// Program one math ADDR_MOD slot. Every field of the slot is written so no stale state survives.
template <cfg::Sec S>
inline void program_addr_mod(const MathAddrMod m)
{
    cfg::write<cfg::Access::TensixCfgUnit>(
        cfg::set<cfg::AddrMod[cfg::SrcA].Incr, S>(m.srca_incr),
        cfg::set<cfg::AddrMod[cfg::SrcA].CR, S, 0>(),
        cfg::set<cfg::AddrMod[cfg::SrcA].Clear, S>(m.srca_clr),
        cfg::set<cfg::AddrMod[cfg::SrcB].Incr, S>(m.srcb_incr),
        cfg::set<cfg::AddrMod[cfg::SrcB].CR, S, 0>(),
        cfg::set<cfg::AddrMod[cfg::SrcB].Clear, S>(m.srcb_clr),
        cfg::set<cfg::AddrMod[cfg::SrcA].Incr2, S, 0>(),
        cfg::set<cfg::AddrMod[cfg::SrcB].Incr2, S, 0>(),
        cfg::set<cfg::AddrMod[cfg::Dest].Incr, S>(m.dest_incr),
        cfg::set<cfg::AddrMod[cfg::Dest].CR, S>(m.dest_cr),
        cfg::set<cfg::AddrMod[cfg::Dest].Clear, S, 0>(),
        cfg::set<cfg::AddrMod[cfg::Dest].CToCR, S>(m.dest_c2cr),
        cfg::set<cfg::AddrMod[cfg::Fidelity].Incr, S>(m.fid_incr),
        cfg::set<cfg::AddrMod[cfg::Fidelity].Clear, S>(m.fid_clr),
        cfg::set<cfg::AddrMod[cfg::Bias].Incr, S, 0>(),
        cfg::set<cfg::AddrMod[cfg::Bias].Clear, S, 0>());
}

constexpr ElementwiseOperation to_hal_operation(const EltwiseBinaryType type)
{
    return type == EltwiseBinaryType::ELWADD   ? ElementwiseOperation::Add
           : type == EltwiseBinaryType::ELWSUB ? ElementwiseOperation::Subtract
                                               : ElementwiseOperation::Multiply;
}

constexpr hal::fpu::SrcBBroadcast to_hal_broadcast(const BroadcastType type)
{
    return type == BroadcastType::COL      ? hal::fpu::SrcBBroadcast::Column
           : type == BroadcastType::ROW    ? hal::fpu::SrcBBroadcast::Row
           : type == BroadcastType::SCALAR ? hal::fpu::SrcBBroadcast::Scalar
                                           : hal::fpu::SrcBBroadcast::None;
}

constexpr std::uint32_t fidelity_phases(const MathFidelity fidelity)
{
    return fidelity == MathFidelity::LoFi ? 1 : static_cast<std::uint32_t>(fidelity);
}

constexpr std::uint32_t blocks_per_face(const std::uint32_t face_r_dim)
{
    return face_r_dim > FPU_ROWS ? 2 : 1;
}

// Last-instruction words of the per-face MOP: the face either releases only SrcA or both banks.
inline std::uint32_t face_end_release_a    = 0;
inline std::uint32_t face_end_release_both = 0;
inline std::uint32_t programmed_face_end   = 0;

template <EltwiseBinaryType eltwise_binary_type, BroadcastType bcast_type, MathFidelity math_fidelity>
inline void configure_addrmod(const ckernel::TensorShape tensor_shape)
{
    constexpr bool srcb_follows_rows = bcast_type == BroadcastType::NONE || bcast_type == BroadcastType::COL;
    constexpr bool high_fidelity     = fidelity_phases(math_fidelity) > 1;
    const std::uint32_t blocks       = blocks_per_face(tensor_shape.face_r_dim);

    program_addr_mod<cfg::Sec::S0>(MathAddrMod {
        .srca_incr = FPU_ROWS, .srcb_incr = srcb_follows_rows ? FPU_ROWS : 0, .dest_incr = FPU_ROWS, .fid_clr = high_fidelity ? 1u : 0u});
    program_addr_mod<cfg::Sec::S1>(MathAddrMod {});
    program_addr_mod<cfg::Sec::S2>(MathAddrMod {.fid_incr = 1});
    // A face always occupies 16 Dest rows, also when only one 8-row block of it is computed.
    program_addr_mod<cfg::Sec::S3>(MathAddrMod {
        .srca_clr  = 1,
        .srcb_clr  = 1,
        .dest_incr = FACE_R_DIM - FPU_ROWS * (blocks - 1),
        .dest_c2cr = 1,
        .fid_clr   = high_fidelity ? 1u : 0u});
}

// Program the per-face MOP: `blocks` outer iterations of `phases` fidelity phases each.
template <EltwiseBinaryType eltwise_binary_type, BroadcastType bcast_type, MathFidelity math_fidelity>
inline void configure_mop(const std::uint32_t acc_to_dest, const ckernel::TensorShape tensor_shape)
{
    using hal::mop::MopConfig;
    using hal::mop::MopTemplate;

    constexpr std::uint32_t phases = fidelity_phases(math_fidelity);
    const hal::fpu::DestWriteMode write_mode =
        (acc_to_dest && eltwise_binary_type != EltwiseBinaryType::ELWMUL) ? hal::fpu::DestWriteMode::Accumulate : hal::fpu::DestWriteMode::Overwrite;

    auto op = [&](const std::uint8_t addr_mod, const SourceRelease release)
    {
        return hal::fpu::get_operation(Elementwise {
            .operation        = to_hal_operation(eltwise_binary_type),
            .broadcast        = to_hal_broadcast(bcast_type),
            .dest_write       = write_mode,
            .address_modifier = addr_mod,
            .release          = release,
        });
    };

    face_end_release_a    = op(ADDR_MOD_FACE_END, SourceRelease::SrcA);
    face_end_release_both = op(ADDR_MOD_FACE_END, SourceRelease::Both);
    programmed_face_end   = face_end_release_both;

    MopConfig<MopTemplate::Template1> config {};
    config.outer_loop.count                               = blocks_per_face(tensor_shape.face_r_dim);
    config.inner_loop.count                               = phases;
    config.inner_loop.body_op                             = op(ADDR_MOD_FIDELITY, SourceRelease::None);
    config.inner_loop.last_op_on_nonfinal_outer_iteration = op(ADDR_MOD_BLOCK, SourceRelease::None);
    config.inner_loop.last_op_on_final_outer_iteration    = programmed_face_end;
    hal::mop::program(config);
}

// Run the per-face MOP, releasing SrcB only when `release_srcb` (its face is no longer needed).
inline void run_face(const bool release_srcb)
{
    const std::uint32_t face_end = release_srcb ? face_end_release_both : face_end_release_a;
    if (face_end != programmed_face_end)
    {
        hal::mop::write<hal::mop::Template1Field::LastOpOnFinalOuterIteration>(face_end);
        programmed_face_end = face_end;
    }
    hal::mop::Runner<hal::mop::MopTemplate::Template1>::run();
}

template <BroadcastType bcast_type>
inline bool face_releases_srcb(const ckernel::TensorShape tensor_shape, const std::uint32_t face)
{
    const std::uint32_t num_faces = tensor_shape.total_num_faces();
    if constexpr (bcast_type == BroadcastType::SCALAR)
    {
        return face == num_faces - 1;
    }
    else if constexpr (bcast_type == BroadcastType::COL)
    {
        const std::uint32_t faces_per_b = (tensor_shape.num_faces_c_dim == 1) ? 1 : tensor_shape.num_faces_c_dim;
        return (face % faces_per_b) == faces_per_b - 1;
    }
    else
    {
        return true;
    }
}

template <std::uint32_t... Rows>
inline void zero_dest_face_rows(std::integer_sequence<std::uint32_t, Rows...>)
{
    (hal::dst::run<hal::dst::Zero {.scope = hal::dst::ZeroScope::SingleRow, .index = Rows, .address_mode = ADDR_MOD_HOLD}>(), ...);
}

// Move four rows of the current Dest face (Dest counter + Row) into the Src register rows read by the FPU.
// MOVD2A/MOVD2B rows use the same math-side Src row numbering as the FPU (no unpacker row offset).
template <typename Src, std::uint32_t... Rows>
inline void move_dest_rows_to_src(std::integer_sequence<std::uint32_t, Rows...>)
{
    (hal::move::run<hal::move::Transfer<hal::move::Dst, Src> {
         .source_row        = Rows,
         .source_addressing = hal::move::RowAddressing::CounterRelative,
         .destination_row   = Rows,
         .address_mode      = ADDR_MOD_HOLD,
         .number_of_rows    = hal::move::rows::Four}>(),
     ...);
}

// Copy the current Dest face into the SrcA or SrcB bank that the FPU reads next.
template <EltwiseBinaryReuseDestType binary_reuse_dest>
inline void move_dest_face_to_src()
{
    using namespace hal::sync;
    using FaceQuarters = std::integer_sequence<std::uint32_t, 0, 4, 8, 12>;
    if constexpr (binary_reuse_dest == EltwiseBinaryReuseDestType::DEST_TO_SRCA)
    {
        // MOVD2A does not wait for SrcA ownership: wait for the unpacked bank and for pending Dest writes.
        wait::stall<StallTarget::Math, StallCondition::SrcAValid | StallCondition::MathIdle>();
        move_dest_rows_to_src<hal::move::SrcA>(FaceQuarters {});
    }
    else
    {
        wait::stall<StallTarget::Math, StallCondition::SrcBValid | StallCondition::MathIdle>();
        move_dest_rows_to_src<hal::move::SrcB>(FaceQuarters {});
    }
}

} // namespace hal_llk::math

template <EltwiseBinaryType eltwise_binary_type, BroadcastType bcast_type, MathFidelity math_fidelity>
inline void eltwise_binary_configure_addrmod()
{
    hal_llk::math::configure_addrmod<eltwise_binary_type, bcast_type, math_fidelity>(ckernel::DEFAULT_TENSOR_SHAPE);
}

template <EltwiseBinaryType eltwise_binary_type, BroadcastType bcast_type, MathFidelity math_fidelity = MathFidelity::LoFi>
inline void eltwise_binary_configure_mop_standard(const std::uint32_t acc_to_dest, const ckernel::TensorShape tensor_shape)
{
    hal_llk::math::configure_mop<eltwise_binary_type, bcast_type, math_fidelity>(acc_to_dest, tensor_shape);
}

template <
    EltwiseBinaryType eltwise_binary_type,
    BroadcastType src_b_bcast_type,
    MathFidelity math_fidelity                   = MathFidelity::LoFi,
    EltwiseBinaryReuseDestType binary_reuse_dest = EltwiseBinaryReuseDestType::NONE>
inline void _llk_math_eltwise_binary_init_(const ckernel::TensorShape tensor_shape, const std::uint32_t acc_to_dest)
{
    namespace cfg = hal::cfg;
    using hal::rwc::Counters;
    static_assert(
        eltwise_binary_type == EltwiseBinaryType::ELWADD || eltwise_binary_type == EltwiseBinaryType::ELWSUB || eltwise_binary_type == EltwiseBinaryType::ELWMUL,
        "Unsupported eltwise binary type");
    static_assert(binary_reuse_dest == EltwiseBinaryReuseDestType::NONE || src_b_bcast_type == BroadcastType::NONE, "Dest reuse requires no broadcast");

    hal::sync::wait::stall<hal::sync::StallTarget::Config, hal::sync::StallCondition::MathIdle>();
    hal_llk::math::configure_addrmod<eltwise_binary_type, src_b_bcast_type, math_fidelity>(tensor_shape);
    hal_llk::math::configure_mop<eltwise_binary_type, src_b_bcast_type, math_fidelity>(acc_to_dest, tensor_shape);

    // Releases must clear data-valid so the unpackers can refill the banks; FP compute flushes Src zero flags.
    cfg::write<cfg::Access::TensixCfgUnit>(cfg::set<cfg::ClrDvalid::SrcA_Disable, cfg::Sec::S0, 0>(), cfg::set<cfg::ClrDvalid::SrcB_Disable, cfg::Sec::S0, 0>());
    cfg::write<cfg::Access::TensixCfgUnit, cfg::AluAccCtrl::Zero_Flag_disabled_src, cfg::Sec::S0, 0>();

    hal::math_counters.set<Counters::All, 0>().clear_fidelity().apply();
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
    using hal::rwc::Counters;

    hal_llk::math::set_dest_offset(hal_llk::dest_half_base() + dst_index * hal_llk::DEST_ROWS_PER_TILE);
    hal::math_counters.set<Counters::All, 0>().apply();

    const std::uint32_t num_faces = tensor_shape.total_num_faces();
    for (std::uint32_t face = 0; face < num_faces; face++)
    {
        if constexpr (binary_reuse_dest != EltwiseBinaryReuseDestType::NONE)
        {
            hal_llk::math::move_dest_face_to_src<binary_reuse_dest>();
            if constexpr (eltwise_binary_type == EltwiseBinaryType::ELWMUL)
            {
                // ELWMUL accumulates onto Dest: clear the face that is being replaced.
                hal_llk::math::zero_dest_face_rows(std::make_integer_sequence<std::uint32_t, FACE_R_DIM> {});
            }
        }
        hal_llk::math::run_face(hal_llk::math::face_releases_srcb<src_b_bcast_type>(tensor_shape, face));
    }
}

inline void _llk_math_eltwise_binary_uninit_()
{
}
