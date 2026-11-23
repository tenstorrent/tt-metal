// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// HAL-based implementation of the Blackhole AB (two operand) unpack LLK functions.

#pragma once

#include <cstdint>

#include "hal/nop.h"
#include "hal/unpack.h"
#include "llk_hal_common.h"
#include "llk_unpack_common.h"

namespace hal_llk::unpack
{

// One face from L1 into the next SrcA (Unpacker0) or SrcB (Unpacker1) bank, handed to math on completion.
constexpr hal::unpack::DataTransfer face_transfer(const hal::unpack::Engine engine, const std::uint8_t z_increment)
{
    return {
        .engine     = engine,
        .increments = {.channel0 = {.z = z_increment}},
        .handoff    = hal::unpack::SourceHandoff::FlipAndSetDataValid,
    };
}

// Set the input-face (channel-0 Z) counter of one unpacker.
template <AdcClient Client, std::uint8_t Z>
constexpr std::uint32_t set_input_face_operation()
{
    return hal::address_counters.client<Client>().template channel<AdcChannel::Channel0>().template Z<Z>().template get_operation<hal::GetOpType::SETTER>();
}

} // namespace hal_llk::unpack

/**
 * @brief Program the unpack MOP for the face order and SrcB broadcast mode.
 */
template <BroadcastType BType = BroadcastType::NONE>
inline void _llk_unpack_AB_mop_config_(const bool transpose_of_faces, const ckernel::TensorShape tensor_shape)
{
    using hal::unpack::Engine;
    using hal_llk::unpack::AdcClient;
    using hal_llk::unpack::face_transfer;
    using MopConfig = hal::mop::MopConfig<hal::mop::MopTemplate::Template1>;

    const std::uint32_t num_faces_r = tensor_shape.num_faces_r_dim;
    const std::uint32_t num_faces_c = tensor_shape.num_faces_c_dim;
    const std::uint32_t num_faces   = tensor_shape.total_num_faces();

    constexpr std::uint32_t unpack_srcb = hal::unpack::get_operation<face_transfer(Engine::Unpacker1, 1)>();

    // With SrcA face transpose (2x2 faces only) SrcA faces are read column-major (0, 2, 1, 3): step two faces,
    // then restart at face 1 for the second output face row.
    const bool transposed_faces              = transpose_of_faces && num_faces_r > 1 && num_faces_c > 1;
    constexpr std::uint32_t srca_next_column = hal_llk::unpack::set_input_face_operation<AdcClient::Unpacker0, 1>();
    const std::uint32_t unpack_srca          = transposed_faces ? hal::unpack::get_operation<face_transfer(Engine::Unpacker0, 2)>()
                                                                : hal::unpack::get_operation<face_transfer(Engine::Unpacker0, 1)>();
    const std::uint32_t srca_row_end_op      = transposed_faces ? srca_next_column : hal::nop::thread_operation();

    if constexpr (BType == BroadcastType::COL)
    {
        // Per face row: one SrcB face (its column 0 is broadcast), then every SrcA face of that row.
        // SrcB skips to the first face of the next face row.
        const std::uint32_t unpack_srcb_next_row = hal::unpack::get_operation(face_transfer(Engine::Unpacker1, static_cast<std::uint8_t>(num_faces_c)));
        hal::mop::program(MopConfig {
            .outer_loop = {.count = num_faces_r, .start_op = unpack_srcb_next_row, .end_op = srca_row_end_op},
            .inner_loop = {.count = num_faces_c, .body_op = unpack_srca},
        });
    }
    else if constexpr (BType == BroadcastType::ROW)
    {
        // Every SrcA face is paired with the SrcB face of the first face row (which holds the broadcast row).
        constexpr std::uint32_t srcb_first_face = hal_llk::unpack::set_input_face_operation<AdcClient::Unpacker1, 0>();
        hal::mop::program(MopConfig {
            .outer_loop = {.count = num_faces_r, .end_op = srcb_first_face, .second_end_op = srca_row_end_op},
            .inner_loop = {.count = num_faces_c, .body_op = unpack_srcb, .alternating_op = unpack_srca},
        });
    }
    else if constexpr (BType == BroadcastType::SCALAR)
    {
        // One SrcB face (datum 0 is broadcast), then all SrcA faces.
        hal::mop::program(MopConfig {
            .outer_loop = {.count = 1, .start_op = unpack_srcb},
            .inner_loop = {.count = num_faces, .body_op = unpack_srca},
        });
    }
    else
    {
        // SrcA and SrcB faces in pairs; SrcB faces always in natural order.
        hal::mop::program(MopConfig {
            .outer_loop = {.count = num_faces_r, .end_op = srca_row_end_op},
            .inner_loop = {.count = num_faces_c, .body_op = unpack_srca, .alternating_op = unpack_srcb},
        });
    }
}

/**
 * @brief Initialize the unpackers to feed SrcA/SrcB face by face for an eltwise binary op.
 */
template <BroadcastType BType = BroadcastType::NONE>
inline void _llk_unpack_AB_init_(const ckernel::TensorShape tensor_shape, const ckernel::Transpose transpose)
{
    namespace cfg  = hal_llk::cfg;
    namespace sync = hal_llk::sync;
    using hal_llk::unpack::AdcChannel;
    using hal_llk::unpack::AdcClient;

    const bool transpose_within_face = (transpose == ckernel::Transpose::IntraFace) || (transpose == ckernel::Transpose::Both);
    const bool transpose_of_faces    = (transpose == ckernel::Transpose::InterFace) || (transpose == ckernel::Transpose::Both);

    // SrcA within-face (XY) transpose is done by unpacker 0.
    sync::wait::stall<sync::StallTarget::Config, sync::StallCondition::Unpacker0Idle>();
    cfg::write<cfg::Access::TensixCfgUnit, cfg::Thcon[cfg::Reg2].Haloize_mode, cfg::Sec::S0>(transpose_within_face ? 1 : 0);

    // Datums per face: last datum index in the channel-0 X counter, first datum (0) in channel-1 X.
    // (This is the SETADCXX x_end2/x_start pair; see the report about the HAL channel naming.)
    hal::runtime_address_counters.client<AdcClient::Unpacker0, AdcClient::Unpacker1>()
        .channel<AdcChannel::Channel0>()
        .X(tensor_shape.face_r_dim * FACE_C_DIM - 1)
        .channel<AdcChannel::Channel1>()
        .X(0)
        .apply();

    _llk_unpack_AB_mop_config_<BType>(transpose_of_faces, tensor_shape);
}

template <BroadcastType BType = BroadcastType::NONE>
inline void _llk_unpack_AB_init_(const ckernel::TensorShape tensor_shape = ckernel::DEFAULT_TENSOR_SHAPE)
{
    _llk_unpack_AB_init_<BType>(tensor_shape, ckernel::Transpose::None);
}

template <BroadcastType BType = BroadcastType::NONE>
inline void _llk_unpack_AB_init_(const ckernel::TensorShape tensor_shape, const std::uint32_t transpose)
{
    _llk_unpack_AB_init_<BType>(tensor_shape, transpose ? ckernel::Transpose::Both : ckernel::Transpose::None);
}

inline void _llk_unpack_AB_uninit_()
{
}

/**
 * @brief Unpack one tile of A into SrcA and one tile of B into SrcB.
 */
template <BroadcastType BType = BroadcastType::NONE>
inline void _llk_unpack_AB_(
    const std::uint32_t address_a,
    std::uint32_t address_b,
    [[maybe_unused]] const std::uint32_t bcast_row_idx = 0,
    [[maybe_unused]] const std::uint32_t srcb_format   = 0)
{
    namespace cfg  = hal_llk::cfg;
    namespace sync = hal_llk::sync;
    using hal_llk::unpack::AdcChannel;
    using hal_llk::unpack::AdcClient;

    // Rewind the input/output face counters for the new tile.
    hal::address_counters.client<AdcClient::Unpacker0, AdcClient::Unpacker1>()
        .channel<AdcChannel::Channel0>()
        .Z<0>()
        .W<0>()
        .channel<AdcChannel::Channel1>()
        .Z<0>()
        .W<0>()
        .apply();

    // The previous tile's transfers must finish before their base addresses are replaced.
    sync::wait::stall<sync::StallTarget::Config, sync::StallCondition::Unpacker0Idle | sync::StallCondition::Unpacker1Idle>();
    cfg::write<cfg::Access::TensixCfgUnit>(
        cfg::set<cfg::Unpacker[0].Cntx[0].Base, cfg::Sec::S0>(address_a), cfg::set<cfg::Unpacker[1].Cntx[0].Base, cfg::Sec::S0>(address_b));

    hal::mop::Runner<hal::mop::MopTemplate::Template1>::run();
}
