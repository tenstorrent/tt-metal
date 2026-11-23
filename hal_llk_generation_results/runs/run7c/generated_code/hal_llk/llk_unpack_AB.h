// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// HAL-based implementation of the two-operand (A/B) unpack LLK (Blackhole).

#pragma once

#include <cstdint>

#include "ckernel.h"
#include "ckernel_defs.h"
#include "hal/address_counters.h"
#include "hal/cfg.h"
#include "hal/mop.h"
#include "hal/sync.h"
#include "hal/unpack.h"
#include "llk_assert.h"
#include "llk_defs.h"
#include "llk_hal_common.h"
#include "llk_unpack_common.h"
#include "tensor_shape.h"

namespace hal_llk::unpack_detail
{

// One face transfer of unpacker `engine`, advancing the channel-0 (L1) Z counter by `z_step` faces and
// handing the completed SrcA/SrcB bank to the math thread.
constexpr hal::unpack::DataTransfer face_transfer(const hal::unpack::Engine engine, const std::uint8_t z_step)
{
    return {
        .engine     = engine,
        .increments = {.channel0 = {.z = z_step}},
        .handoff    = hal::unpack::SourceHandoff::FlipAndSetDataValid,
    };
}

template <hal::unpack::Engine engine, std::uint8_t z_step>
constexpr std::uint32_t face_transfer_op()
{
    return hal::unpack::get_operation<face_transfer(engine, z_step)>();
}

template <hal::AddressCounterClient client, std::uint8_t z>
constexpr std::uint32_t set_l1_face_op()
{
    return hal::address_counters.client<client>()
        .template channel<hal::AddressChannel::Channel0>()
        .template Z<z>()
        .template get_operation<hal::GetOpType::SETTER>();
}

} // namespace hal_llk::unpack_detail

/**
 * @brief Configure the MOP for unpacking SrcA/SrcB, one tile per MOP run.
 */
template <BroadcastType BType = BroadcastType::NONE>
inline void _llk_unpack_AB_mop_config_(const bool transpose_of_faces, const ckernel::TensorShape tensor_shape)
{
    using hal::unpack::Engine;
    using namespace hal_llk::unpack_detail;
    using Config = hal::mop::MopConfig<hal::mop::MopTemplate::Template1>;

    const std::uint32_t face_rows = tensor_shape.num_faces_r_dim;
    const std::uint32_t face_cols = tensor_shape.num_faces_c_dim;
    const std::uint32_t num_faces = tensor_shape.total_num_faces();

    constexpr std::uint32_t unpack_b      = face_transfer_op<Engine::Unpacker1, 1>();
    constexpr std::uint32_t unpack_b_hold = face_transfer_op<Engine::Unpacker1, 0>();

    // Transposed face order (2x2 faces only): SrcA visits faces 0, 2 on the first face row and 1, 3 on the
    // second, so it steps two faces at a time and restarts at face 1 after the first face row.
    const bool swap_faces         = transpose_of_faces && (face_rows == 2) && (face_cols == 2);
    const std::uint32_t unpack_a  = swap_faces ? face_transfer_op<Engine::Unpacker0, 2>() : face_transfer_op<Engine::Unpacker0, 1>();
    const std::uint32_t restart_a = set_l1_face_op<hal::AddressCounterClient::Unpacker0, 1>();

    Config mop {};
    if constexpr (BType == BroadcastType::COL)
    {
        // One SrcB face (its column 0) serves every face of the same face row:
        // B(face row r), then A for each face of that row.
        mop.outer_loop.count    = face_rows;
        mop.outer_loop.start_op = (face_cols == 2) ? face_transfer_op<Engine::Unpacker1, 2>() : unpack_b;
        if (swap_faces)
        {
            mop.outer_loop.end_op = restart_a;
        }
        mop.inner_loop.count   = face_cols;
        mop.inner_loop.body_op = unpack_a;
    }
    else if constexpr (BType == BroadcastType::ROW)
    {
        // Every face column reads row 0 of the matching top SrcB face: B(0, c), A(r, c); B restarts per face row.
        mop.outer_loop.count  = face_rows;
        mop.outer_loop.end_op = set_l1_face_op<hal::AddressCounterClient::Unpacker1, 0>();
        if (swap_faces)
        {
            mop.outer_loop.second_end_op = restart_a;
        }
        mop.inner_loop.count          = face_cols;
        mop.inner_loop.body_op        = unpack_b;
        mop.inner_loop.alternating_op = unpack_a;
    }
    else if constexpr (BType == BroadcastType::SCALAR)
    {
        // One SrcB face (its first datum) serves the whole tile.
        mop.outer_loop.count    = 1;
        mop.outer_loop.start_op = unpack_b_hold;
        mop.inner_loop.count    = num_faces;
        mop.inner_loop.body_op  = unpack_a;
    }
    else
    {
        // A and B face pairs, one face row per outer iteration.
        mop.outer_loop.count = face_rows;
        if (swap_faces)
        {
            mop.outer_loop.end_op = restart_a;
        }
        mop.inner_loop.count          = face_cols;
        mop.inner_loop.body_op        = unpack_a;
        mop.inner_loop.alternating_op = unpack_b;
    }
    hal::mop::program(mop);
}

/**
 * @brief Initialize unpacker to unpack two source operands A and B into SrcA and SrcB registers.
 */
template <BroadcastType BType = BroadcastType::NONE>
inline void _llk_unpack_AB_init_(const ckernel::TensorShape tensor_shape, const ckernel::Transpose transpose)
{
    namespace cfg  = hal::cfg;
    namespace sync = hal::sync;

    const bool within_face = (transpose == ckernel::Transpose::IntraFace) || (transpose == ckernel::Transpose::Both);
    const bool of_faces    = (transpose == ckernel::Transpose::InterFace) || (transpose == ckernel::Transpose::Both);

    sync::wait::stall<sync::StallTarget::Config, sync::StallCondition::Unpacker0Idle | sync::StallCondition::Unpacker1Idle>();
    cfg::write<cfg::Access::TensixCfgUnit, cfg::Thcon[cfg::Reg2].Haloize_mode, cfg::Sec::S0>(within_face ? 1 : 0);

    hal_llk::unpack_detail::set_face_datum_count(tensor_shape.face_r_dim);
    _llk_unpack_AB_mop_config_<BType>(of_faces, tensor_shape);
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
 * @brief Unpack two tiles from L1 memory into SrcA and SrcB registers.
 */
template <BroadcastType BType = BroadcastType::NONE>
inline void _llk_unpack_AB_(
    const std::uint32_t address_a, std::uint32_t address_b, [[maybe_unused]] const std::uint32_t bcast_row_idx = 0, [[maybe_unused]] const std::uint32_t srcb_format = 0)
{
    namespace cfg  = hal::cfg;
    namespace sync = hal::sync;

    // The previous tile's transfers must have read their base addresses before they are replaced.
    sync::wait::stall<sync::StallTarget::Config, sync::StallCondition::Unpacker0Idle | sync::StallCondition::Unpacker1Idle>();

    // Restart face (Z) and W positions of both unpackers.
    hal::address_counters.client<hal::AddressCounterClient::Unpacker0, hal::AddressCounterClient::Unpacker1>()
        .channel<hal::AddressChannel::Channel0>()
        .Z<0>()
        .W<0>()
        .channel<hal::AddressChannel::Channel1>()
        .Z<0>()
        .W<0>()
        .apply();

    cfg::write<cfg::Access::TensixCfgUnit>(
        cfg::set<cfg::Thcon[cfg::Reg3].Base_address, cfg::Sec::S0>(address_a), cfg::set<cfg::Thcon[cfg::Reg3].Base_address, cfg::Sec::S1>(address_b));

    hal::mop::Runner<hal::mop::MopTemplate::Template1>::run();
}
