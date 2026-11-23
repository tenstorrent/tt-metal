// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// HAL implementation of the two-operand (A -> SrcA, B -> SrcB) unpack (Blackhole).

#pragma once

#include <cstdint>

#include "llk_hal_common.h"
#include "llk_unpack_common.h"

using namespace ckernel;

namespace hal_llk
{

// One face of an operand per UNPACR; the transfer that completes a face hands the bank to math.
constexpr hal::unpack::DataTransfer unpack_face(const hal::unpack::Engine engine, const std::uint8_t z_increment)
{
    return {
        .engine     = engine,
        .increments = {.channel0 = {.z = z_increment}},
        .handoff    = hal::unpack::SourceHandoff::FlipAndSetDataValid,
    };
}

constexpr std::uint32_t UNPACK_A_FACE         = hal::unpack::get_operation<unpack_face(hal::unpack::Engine::Unpacker0, 1)>();
constexpr std::uint32_t UNPACK_A_FACE_SKIP    = hal::unpack::get_operation<unpack_face(hal::unpack::Engine::Unpacker0, 2)>();
constexpr std::uint32_t UNPACK_B_FACE         = hal::unpack::get_operation<unpack_face(hal::unpack::Engine::Unpacker1, 1)>();

// Set the L1 face (channel-0 Z) of one unpacker.
template <hal::AddressCounterClient Client, std::uint8_t Face>
constexpr std::uint32_t unpack_select_face_operation()
{
    return hal::address_counters.client<Client>().template channel<hal::AddressChannel::Channel0>().template Z<Face>().template get_operation<hal::GetOpType::SETTER>();
}

} // namespace hal_llk

/**
 * @brief Configure the MOP for unpacking two source operands A and B.
 */
template <BroadcastType BType = BroadcastType::NONE>
inline void _llk_unpack_AB_mop_config_(const bool transpose_of_faces, const ckernel::TensorShape tensor_shape)
{
    using namespace hal_llk;
    using hal::mop::MopConfig;
    using hal::mop::MopTemplate;

    const std::uint32_t num_faces = tensor_shape.total_num_faces();
    // Faces are arranged num_faces_r_dim x num_faces_c_dim; broadcasts pair B faces per face row / column.
    const std::uint32_t face_rows = (num_faces < 4) ? 1 : 2;
    const std::uint32_t face_cols = (num_faces < 2) ? 1 : 2;

    // With transposed faces, A is read column-major (0, 2, then 1, 3): step two faces and restart at face 1.
    const bool a_column_major = transpose_of_faces && num_faces == 4;
    const std::uint32_t unpack_a = a_column_major ? UNPACK_A_FACE_SKIP : UNPACK_A_FACE;

    MopConfig<MopTemplate::Template1> config {};
    if constexpr (BType == BroadcastType::COL)
    {
        // B face of each face row (0, then 2) is unpacked once and reused by every A face of that row.
        config.outer_loop.count    = face_rows;
        config.outer_loop.start_op = UNPACK_B_FACE;
        config.outer_loop.end_op   = unpack_select_face_operation<hal::AddressCounterClient::Unpacker1, 2>();
        if (a_column_major)
        {
            config.outer_loop.second_end_op = unpack_select_face_operation<hal::AddressCounterClient::Unpacker0, 1>();
        }
        config.inner_loop.count   = face_cols;
        config.inner_loop.body_op = unpack_a;
    }
    else if constexpr (BType == BroadcastType::ROW)
    {
        // B faces of the first face row are unpacked again for every face row of A.
        config.outer_loop.count  = face_rows;
        config.outer_loop.end_op = unpack_select_face_operation<hal::AddressCounterClient::Unpacker1, 0>();
        if (a_column_major)
        {
            config.outer_loop.second_end_op = unpack_select_face_operation<hal::AddressCounterClient::Unpacker0, 1>();
        }
        config.inner_loop.count          = face_cols;
        config.inner_loop.body_op        = UNPACK_B_FACE;
        config.inner_loop.alternating_op = unpack_a;
    }
    else if constexpr (BType == BroadcastType::SCALAR)
    {
        // The first B face is unpacked once for the whole A tile.
        config.outer_loop.count    = 1;
        config.outer_loop.start_op = UNPACK_B_FACE;
        config.inner_loop.count    = num_faces;
        config.inner_loop.body_op  = UNPACK_A_FACE;
    }
    else if (transpose_of_faces && num_faces == 4)
    {
        // A faces in transposed order 0, 2, 1, 3 paired with B faces 0, 1, 2, 3.
        config.outer_loop.count          = face_rows;
        config.outer_loop.end_op         = unpack_select_face_operation<hal::AddressCounterClient::Unpacker0, 1>();
        config.inner_loop.count          = face_cols;
        config.inner_loop.body_op        = UNPACK_A_FACE_SKIP;
        config.inner_loop.alternating_op = UNPACK_B_FACE;
    }
    else
    {
        config.outer_loop.count          = num_faces;
        config.inner_loop.count          = 1;
        config.inner_loop.body_op        = UNPACK_A_FACE;
        config.inner_loop.alternating_op = UNPACK_B_FACE;
    }
    hal::mop::program(config);
}

/**
 * @brief Initialize the unpacker to unpack operands A and B into SrcA and SrcB.
 */
template <BroadcastType BType = BroadcastType::NONE>
inline void _llk_unpack_AB_init_(const ckernel::TensorShape tensor_shape, const ckernel::Transpose transpose)
{
    namespace cfg  = hal::cfg;
    namespace sync = hal::sync;

    const bool transpose_within_face = (transpose == ckernel::Transpose::IntraFace) || (transpose == ckernel::Transpose::Both);
    const bool transpose_of_faces    = (transpose == ckernel::Transpose::InterFace) || (transpose == ckernel::Transpose::Both);

    // Within-face (XY) transpose of SrcA is an unpacker-0 mode.
    sync::wait::stall<sync::StallTarget::Config, sync::StallCondition::Unpacker0Idle>();
    cfg::write<cfg::Access::TensixCfgUnit, cfg::Thcon[cfg::Reg2].Haloize_mode, cfg::Sec::S0>(transpose_within_face ? 1 : 0);

    // Each UNPACR moves one face: face_r_dim rows of FACE_C_DIM datums. With the HAL builder the
    // datum-count end is the Channel0 X value and the start is the Channel1 X value.
    const std::uint32_t x_end = static_cast<std::uint32_t>(tensor_shape.face_r_dim) * FACE_C_DIM - 1;
    hal::runtime_address_counters.client<hal::AddressCounterClient::Unpacker0, hal::AddressCounterClient::Unpacker1>()
        .channel<hal::AddressChannel::Channel0>()
        .X(x_end)
        .channel<hal::AddressChannel::Channel1>()
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

/**
 * @brief No-op teardown after AB unpacking (datum-count state is reprogrammed by every init).
 */
inline void _llk_unpack_AB_uninit_()
{
}

/**
 * @brief Unpack one tile of A into SrcA and one tile of B into SrcB.
 *
 * The tile base addresses go to the unpacker configuration context the unpacker is not using, through
 * MMIO, and the UNPACK_SYNC semaphore tracks how many contexts are owned by pending tiles.
 */
template <BroadcastType BType = BroadcastType::NONE>
inline void _llk_unpack_AB_(
    const std::uint32_t address_a,
    std::uint32_t address_b,
    [[maybe_unused]] const std::uint32_t bcast_row_idx = 0,
    [[maybe_unused]] const std::uint32_t srcb_format   = 0)
{
    namespace cfg  = hal::cfg;
    namespace sync = hal::sync;
    using hal_llk::UNPACK_SYNC_SEM;

    // Start every tile from face 0 of both operands.
    hal::address_counters.client<hal::AddressCounterClient::Unpacker0, hal::AddressCounterClient::Unpacker1>()
        .channel<hal::AddressChannel::Channel0>()
        .Z<0>()
        .W<0>()
        .channel<hal::AddressChannel::Channel1>()
        .Z<0>()
        .W<0>()
        .apply();

    // Wait until the context this tile uses has been released by the tile that used it before.
    while (sync::semaphore::read<UNPACK_SYNC_SEM>() >= hal_llk::NUM_UNPACK_CFG_CONTEXTS)
    {
    }

    // The UNPACRs below must not be issued before both stores have reached the CFG registers:
    // reading back the last store completes only after both have landed.
    if (hal_llk::unpack_cfg_context == 0)
    {
        cfg::write<cfg::Access::MMIO, cfg::Unpacker[0].Cntx[0].Base, cfg::Sec::S0>(address_a);
        cfg::write<cfg::Access::MMIO, cfg::Unpacker[1].Cntx[0].Base, cfg::Sec::S0>(address_b);
        while (cfg::read<cfg::Access::MMIO, cfg::Unpacker[1].Cntx[0].Base, cfg::Sec::S0>() != address_b)
        {
        }
    }
    else
    {
        cfg::write<cfg::Access::MMIO, cfg::Unpacker[0].Cntx[1].Base, cfg::Sec::S0>(address_a);
        cfg::write<cfg::Access::MMIO, cfg::Unpacker[1].Cntx[1].Base, cfg::Sec::S0>(address_b);
        while (cfg::read<cfg::Access::MMIO, cfg::Unpacker[1].Cntx[1].Base, cfg::Sec::S0>() != address_b)
        {
        }
    }

    // The context is now owned by this tile.
    sync::semaphore::post<sync::Access::MMIO, UNPACK_SYNC_SEM>();

    hal::mop::Runner<hal::mop::MopTemplate::Template1>::run();

    // Release the context once the unpacks of this tile have been issued.
    sync::semaphore::get<UNPACK_SYNC_SEM>();

    // The next tile uses the other configuration context.
    hal_llk::unpack_cfg_context = 1 - hal_llk::unpack_cfg_context;
    if (hal_llk::unpack_cfg_context == 0)
    {
        cfg::write<cfg::Access::TensixCfgUnit>(
            cfg::set<cfg::UnpackMiscCfg::CfgContextOffset_0, cfg::Sec::S0, 0>(), cfg::set<cfg::UnpackMiscCfg::CfgContextOffset_1, cfg::Sec::S0, 0>());
    }
    else
    {
        cfg::write<cfg::Access::TensixCfgUnit>(
            cfg::set<cfg::UnpackMiscCfg::CfgContextOffset_0, cfg::Sec::S0, 1>(), cfg::set<cfg::UnpackMiscCfg::CfgContextOffset_1, cfg::Sec::S0, 1>());
    }
}
