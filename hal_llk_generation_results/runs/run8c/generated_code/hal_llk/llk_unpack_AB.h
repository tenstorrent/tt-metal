// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// HAL-based implementation of the two-operand (A/B) unpack (Blackhole).

#pragma once

#include <cstdint>

#include "hal/mop.h"
#include "hal/unpack.h"
#include "hal_llk_common.h"
#include "llk_memory_checks.h"
#include "llk_unpack_common.h"
#include "tensor_shape.h"

namespace hal_llk
{

// One face per UNPACR: advance the L1 face (channel-0 Z) and hand the completed bank to math.
constexpr hal::unpack::DataTransfer unpack_face(const hal::unpack::Engine engine, const std::uint8_t z_increment)
{
    return {
        .engine     = engine,
        .increments = {.channel0 = {.z = z_increment}},
        .context    = hal::unpack::ContextSelection::explicit_context(), // configuration context 0 + UnpackMiscCfg offset
        .handoff    = hal::unpack::SourceHandoff::FlipAndSetDataValid,
    };
}

inline constexpr std::uint32_t UNPACK_A_NEXT_FACE = hal::unpack::get_operation<unpack_face(hal::unpack::Engine::Unpacker0, 1)>();
inline constexpr std::uint32_t UNPACK_A_SKIP_FACE = hal::unpack::get_operation<unpack_face(hal::unpack::Engine::Unpacker0, 2)>();
inline constexpr std::uint32_t UNPACK_B_NEXT_FACE = hal::unpack::get_operation<unpack_face(hal::unpack::Engine::Unpacker1, 1)>();

// Channel-0 Z (L1 face index) rewinds used inside the MOPs.
inline constexpr std::uint32_t UNPACK_A_SET_FACE_1 =
    hal::address_counters.client<hal::AddressCounterClient::Unpacker0>().channel<hal::AddressChannel::Channel0>().Z<1>().get_operation<hal::GetOpType::SETTER>();
inline constexpr std::uint32_t UNPACK_B_SET_FACE_0 =
    hal::address_counters.client<hal::AddressCounterClient::Unpacker1>().channel<hal::AddressChannel::Channel0>().Z<0>().get_operation<hal::GetOpType::SETTER>();
inline constexpr std::uint32_t UNPACK_B_SET_FACE_2 =
    hal::address_counters.client<hal::AddressCounterClient::Unpacker1>().channel<hal::AddressChannel::Channel0>().Z<2>().get_operation<hal::GetOpType::SETTER>();

} // namespace hal_llk

/**
 * @brief Configure the MOP for unpacking two source operands A and B.
 *
 * Every UNPACR moves one face and hands its bank to math. Broadcast modes unpack only the SrcB faces
 * that math consumes: row broadcast reuses faces 0/1, column broadcast faces 0/2, scalar face 0.
 */
template <BroadcastType BType = BroadcastType::NONE>
inline void _llk_unpack_AB_mop_config_(const bool transpose_of_faces, const ckernel::TensorShape tensor_shape)
{
    using hal::mop::MopConfig;
    using hal::mop::MopTemplate;

    const std::uint32_t num_faces = tensor_shape.total_num_faces();
    const std::uint32_t outerloop = num_faces < 4 ? 1 : 2;
    const std::uint32_t innerloop = num_faces < 2 ? 1 : 2;

    // Transposed SrcA faces are read in the order 0, 2, 1, 3: step Z by 2 and restart at face 1.
    const bool transpose_a = transpose_of_faces && num_faces == 4;
    const std::uint32_t unpack_a = transpose_a ? hal_llk::UNPACK_A_SKIP_FACE : hal_llk::UNPACK_A_NEXT_FACE;

    MopConfig<MopTemplate::Template1> config {};
    if constexpr (BType == BroadcastType::COL)
    {
        // B(col 0 of face 0) A A, B(face 2) A A
        config.outer_loop.count    = outerloop;
        config.outer_loop.start_op = hal_llk::UNPACK_B_NEXT_FACE;
        config.outer_loop.end_op   = hal_llk::UNPACK_B_SET_FACE_2;
        if (transpose_a)
        {
            config.outer_loop.second_end_op = hal_llk::UNPACK_A_SET_FACE_1;
        }
        config.inner_loop.count   = innerloop;
        config.inner_loop.body_op = unpack_a;
    }
    else if constexpr (BType == BroadcastType::ROW)
    {
        // B(0) A B(1) A, B(0) A B(1) A
        config.outer_loop.count  = outerloop;
        config.outer_loop.end_op = hal_llk::UNPACK_B_SET_FACE_0;
        if (transpose_a)
        {
            config.outer_loop.second_end_op = hal_llk::UNPACK_A_SET_FACE_1;
        }
        config.inner_loop.count          = innerloop;
        config.inner_loop.body_op        = hal_llk::UNPACK_B_NEXT_FACE;
        config.inner_loop.alternating_op = unpack_a;
    }
    else if constexpr (BType == BroadcastType::SCALAR)
    {
        // B(0) A(0) A(1) A(2) A(3)
        config.outer_loop.count    = 1;
        config.outer_loop.start_op = hal_llk::UNPACK_B_NEXT_FACE;
        config.inner_loop.count    = num_faces;
        config.inner_loop.body_op  = hal_llk::UNPACK_A_NEXT_FACE;
    }
    else if (transpose_of_faces && num_faces == 4)
    {
        // A faces in transposed order 0, 2, 1, 3; B faces in order.
        config.outer_loop.count          = 2;
        config.outer_loop.end_op         = hal_llk::UNPACK_A_SET_FACE_1;
        config.inner_loop.count          = 2;
        config.inner_loop.body_op        = hal_llk::UNPACK_A_SKIP_FACE;
        config.inner_loop.alternating_op = hal_llk::UNPACK_B_NEXT_FACE;
    }
    else
    {
        config.outer_loop.count          = 1;
        config.inner_loop.count          = num_faces;
        config.inner_loop.body_op        = hal_llk::UNPACK_A_NEXT_FACE;
        config.inner_loop.alternating_op = hal_llk::UNPACK_B_NEXT_FACE;
    }
    hal::mop::program(config);
}

/**
 * @brief Initialize the unpacker to unpack operands A and B into SrcA and SrcB.
 */
template <BroadcastType BType = BroadcastType::NONE>
inline void _llk_unpack_AB_init_(const ckernel::TensorShape tensor_shape, const ckernel::Transpose transpose)
{
    namespace cfg = hal::cfg;
    using namespace hal::sync;

    const bool within_face = (transpose == ckernel::Transpose::IntraFace) || (transpose == ckernel::Transpose::Both);
    const bool of_faces    = (transpose == ckernel::Transpose::InterFace) || (transpose == ckernel::Transpose::Both);

    wait::stall<StallTarget::Config, StallCondition::Unpacker0Idle | StallCondition::Unpacker1Idle>();

    // Within-face (XY) transpose of SrcA is done by unpacker 0.
    cfg::write<cfg::Access::TensixCfgUnit, cfg::Thcon[cfg::Reg2].Haloize_mode, cfg::Sec::S0>(within_face ? 1 : 0);

    if constexpr (BType == BroadcastType::NONE)
    {
        hal_llk::program_unpack_x_end<hal::AddressCounterClient::Unpacker0, hal::AddressCounterClient::Unpacker1>(tensor_shape.face_r_dim);
    }
    else
    {
        hal_llk::program_unpack_x_end<hal::AddressCounterClient::Unpacker0>(tensor_shape.face_r_dim);
    }

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
 * @brief Unpack one tile of A and one tile of B from L1 into SrcA and SrcB.
 *
 * The L1 addresses are written by the RISC-V (MMIO) into the configuration context that the unpacker is not
 * using; the UNPACK_SYNC semaphore counts the contexts in flight so at most two tiles are outstanding.
 */
template <BroadcastType BType = BroadcastType::NONE>
inline void _llk_unpack_AB_(
    const std::uint32_t address_a,
    std::uint32_t address_b,
    [[maybe_unused]] const std::uint32_t bcast_row_idx = 0,
    [[maybe_unused]] const std::uint32_t srcb_format   = 0)
{
    namespace cfg = hal::cfg;
    using namespace hal::sync;

    LLK_ASSERT(is_valid_L1_address(address_a), "L1 address_a must be in valid L1 memory region");
    LLK_ASSERT(is_valid_L1_address(address_b), "L1 address_b must be in valid L1 memory region");

    // Rewind the face (Z) and W counters of both unpackers.
    hal::address_counters.client<hal::AddressCounterClient::Unpacker0, hal::AddressCounterClient::Unpacker1>()
        .channel<hal::AddressChannel::Channel0>()
        .Z<0>()
        .W<0>()
        .channel<hal::AddressChannel::Channel1>()
        .Z<0>()
        .W<0>()
        .apply();

    // Wait for a free configuration context, then program it.
    while (semaphore::read<hal_llk::UNPACK_SYNC_SEMAPHORE>() >= hal_llk::UNPACK_NUM_CONTEXTS)
    {
    }
    if (unp_cfg_context == 0)
    {
        cfg::write<cfg::Access::MMIO, cfg::Unpacker[0].Cntx[0].Base, cfg::Sec::S0>(address_a);
        cfg::write<cfg::Access::MMIO, cfg::Unpacker[1].Cntx[0].Base, cfg::Sec::S0>(address_b);
    }
    else
    {
        cfg::write<cfg::Access::MMIO, cfg::Unpacker[0].Cntx[1].Base, cfg::Sec::S0>(address_a);
        cfg::write<cfg::Access::MMIO, cfg::Unpacker[1].Cntx[1].Base, cfg::Sec::S0>(address_b);
    }
    semaphore::post<Access::MMIO, hal_llk::UNPACK_SYNC_SEMAPHORE>();

    // The UNPACRs must observe the RISC-V configuration writes above.
    // NOT_IN_HAL: hal::sync::StallCondition has no documented name for the "TRISC config writes done" condition;
    // its numeric value is taken from ckernel::p_stall::TRISC_CFG (the HAL enum values equal the STALLWAIT bits).
    wait::stall<StallTarget::Unpack, static_cast<StallCondition>(ckernel::p_stall::TRISC_CFG)>();

    hal::mop::Runner<hal::mop::MopTemplate::Template1>::run();

    // Release the context once the unpacker has consumed it, then switch to the other one.
    semaphore::get<hal_llk::UNPACK_SYNC_SEMAPHORE>();
    unp_cfg_context = 1 - unp_cfg_context;
    hal_llk::select_unpack_config_context(unp_cfg_context);
}
