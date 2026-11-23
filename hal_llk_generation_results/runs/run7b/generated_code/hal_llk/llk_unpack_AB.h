// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// HAL-based implementation of the two-operand (A -> SrcA, B -> SrcB) unpack (spec: llk_lib/llk_unpack_AB.h).

#pragma once

#include <cstdint>

#include "llk_unpack_common.h"

namespace hal_llk::unpack
{

using hal::unpack::DataTransfer;
using hal::unpack::Engine;
using hal::unpack::SourceHandoff;

// One face of operand A/B into SrcA/SrcB, advancing the L1 face (channel-0 Z) by `z_step` and handing the bank to math.
constexpr DataTransfer face_transfer(const Engine engine, const std::uint8_t z_step)
{
    return DataTransfer {
        .engine     = engine,
        .increments = {.channel0 = {.z = z_step}},
        .handoff    = SourceHandoff::FlipAndSetDataValid,
    };
}

constexpr std::uint32_t UNPACK_A           = hal::unpack::get_operation<face_transfer(Engine::Unpacker0, 1)>();
constexpr std::uint32_t UNPACK_A_SKIP_FACE = hal::unpack::get_operation<face_transfer(Engine::Unpacker0, 2)>();
constexpr std::uint32_t UNPACK_B           = hal::unpack::get_operation<face_transfer(Engine::Unpacker1, 1)>();
constexpr std::uint32_t UNPACK_B_SAME_FACE = hal::unpack::get_operation<face_transfer(Engine::Unpacker1, 0)>();

// Re-point operand A at face 1 (second face column) for the face-transposed order 0, 2, 1, 3.
constexpr std::uint32_t SET_A_FACE_1 =
    hal::address_counters.client<hal::AddressCounterClient::Unpacker0>().channel<hal::AddressChannel::Channel0>().Z<1>().get_operation<hal::GetOpType::SETTER>();
// Re-point operand B at face 0 / face 2 (row / column broadcast reuse of B faces).
constexpr std::uint32_t SET_B_FACE_0 =
    hal::address_counters.client<hal::AddressCounterClient::Unpacker1>().channel<hal::AddressChannel::Channel0>().Z<0>().get_operation<hal::GetOpType::SETTER>();
constexpr std::uint32_t SET_B_FACE_2 =
    hal::address_counters.client<hal::AddressCounterClient::Unpacker1>().channel<hal::AddressChannel::Channel0>().Z<2>().get_operation<hal::GetOpType::SETTER>();

} // namespace hal_llk::unpack

template <BroadcastType BType = BroadcastType::NONE>
inline void _llk_unpack_AB_mop_config_(const bool transpose_of_faces, const ckernel::TensorShape tensor_shape)
{
    using namespace hal_llk::unpack;
    using hal::mop::MopConfig;
    using hal::mop::MopTemplate;

    const std::uint32_t num_faces = tensor_shape.total_num_faces();
    const bool narrow_tile        = tensor_shape.num_faces_c_dim == 1;
    const bool two_face_rows      = tensor_shape.num_faces_r_dim > 1;
    // Face-transposed A (order 0, 2, 1, 3) only differs from the natural order for 2x2-face tiles.
    const bool transposed_a       = transpose_of_faces && num_faces == 4;
    const std::uint32_t unpack_a  = transposed_a ? UNPACK_A_SKIP_FACE : UNPACK_A;

    MopConfig<MopTemplate::Template1> config {};

    if constexpr (BType == BroadcastType::COL)
    {
        // One B face feeds every A face of the same output face row.
        const std::uint32_t a_faces_per_b = narrow_tile ? 1 : tensor_shape.num_faces_c_dim;
        config.outer_loop.count           = num_faces / a_faces_per_b;
        config.outer_loop.start_op        = UNPACK_B;
        config.inner_loop.count           = a_faces_per_b;
        config.inner_loop.body_op         = unpack_a;
        if (!narrow_tile && two_face_rows)
        {
            config.outer_loop.end_op = SET_B_FACE_2;
            if (transposed_a)
            {
                config.outer_loop.second_end_op = SET_A_FACE_1;
            }
        }
    }
    else if constexpr (BType == BroadcastType::ROW)
    {
        // The B faces of the first face row are re-used for every output face row.
        config.outer_loop.count           = tensor_shape.num_faces_r_dim;
        config.inner_loop.count           = tensor_shape.num_faces_c_dim;
        config.inner_loop.body_op         = narrow_tile ? UNPACK_B_SAME_FACE : UNPACK_B;
        config.inner_loop.alternating_op  = unpack_a;
        if (!narrow_tile)
        {
            config.outer_loop.end_op = SET_B_FACE_0;
            if (transposed_a)
            {
                config.outer_loop.second_end_op = SET_A_FACE_1;
            }
        }
    }
    else if constexpr (BType == BroadcastType::SCALAR)
    {
        config.outer_loop.count    = 1;
        config.outer_loop.start_op = UNPACK_B;
        config.inner_loop.count    = num_faces;
        config.inner_loop.body_op  = UNPACK_A;
    }
    else
    {
        if (transpose_of_faces && num_faces == 4)
        {
            // A faces in order 0, 2, 1, 3; B faces in natural order.
            config.outer_loop.count          = 2;
            config.outer_loop.end_op         = SET_A_FACE_1;
            config.inner_loop.count          = 2;
            config.inner_loop.body_op        = UNPACK_A_SKIP_FACE;
            config.inner_loop.alternating_op = UNPACK_B;
        }
        else
        {
            config.outer_loop.count          = 1;
            config.inner_loop.count          = num_faces;
            config.inner_loop.body_op        = UNPACK_A;
            config.inner_loop.alternating_op = UNPACK_B;
        }
    }

    hal::mop::program(config);
}

template <BroadcastType BType = BroadcastType::NONE>
inline void _llk_unpack_AB_init_(const ckernel::TensorShape tensor_shape, const ckernel::Transpose transpose)
{
    namespace cfg                  = hal::cfg;
    const bool transpose_of_faces  = transpose == ckernel::Transpose::InterFace || transpose == ckernel::Transpose::Both;
    const bool transpose_in_face   = transpose == ckernel::Transpose::IntraFace || transpose == ckernel::Transpose::Both;

    hal_llk::unpack::wait_unpackers_idle_for_config();
    // Within-face transpose of operand A is done by the unpacker 0 XY transpose (haloize) mode.
    cfg::write<cfg::Access::TensixCfgUnit, cfg::Thcon[cfg::Reg2].Haloize_mode, cfg::Sec::S0>(transpose_in_face ? 1 : 0);

    hal_llk::unpack::configure_unpacker_x_end(tensor_shape.face_r_dim);
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

template <BroadcastType BType = BroadcastType::NONE>
inline void _llk_unpack_AB_(
    const std::uint32_t address_a,
    std::uint32_t address_b,
    [[maybe_unused]] const std::uint32_t bcast_row_idx = 0,
    [[maybe_unused]] const std::uint32_t srcb_format   = 0)
{
    namespace cfg = hal::cfg;

    // The previous tile must be fully read before its base addresses are replaced.
    hal_llk::unpack::wait_unpackers_idle_for_config();
    cfg::write<cfg::Access::TensixCfgUnit>(
        cfg::set<cfg::Thcon[cfg::Reg3].Base_address, cfg::Sec::S0>(address_a), cfg::set<cfg::Thcon[cfg::Reg3].Base_address, cfg::Sec::S1>(address_b));

    hal_llk::unpack::reset_unpacker_zw_counters();
    hal::mop::Runner<hal::mop::MopTemplate::Template1>::run();
}
