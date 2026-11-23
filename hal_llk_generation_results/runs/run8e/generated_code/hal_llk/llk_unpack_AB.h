// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// HAL-based implementation of the Blackhole two-operand (A/B) unpack LLK functions.

#pragma once

#include <cstdint>

#include "ckernel.h"
#include "ckernel_defs.h"
#include "ckernel_globals.h"
#include "hal/address_counters.h"
#include "hal/cfg.h"
#include "hal/mop.h"
#include "hal/sync.h"
#include "hal/unpack.h"
#include "llk_defs.h"
#include "llk_unpack_common.h"
#include "tensor_shape.h"

namespace hal_llk::unpack
{

/// One face transfer into SrcA (unpacker 0) or SrcB (unpacker 1) that hands the bank to math.
inline constexpr std::uint32_t face_transfer(const hal::unpack::Engine engine, const std::uint8_t z_increment)
{
    return hal::unpack::get_operation(hal::unpack::DataTransfer {
        .engine     = engine,
        .increments = {.channel0 = {.z = z_increment}},
        .handoff    = hal::unpack::SourceHandoff::FlipAndSetDataValid,
    });
}

} // namespace hal_llk::unpack

template <BroadcastType BType = BroadcastType::NONE>
inline void _llk_unpack_AB_mop_config_(const bool transpose_of_faces, const ckernel::TensorShape tensor_shape)
{
    using hal::unpack::Engine;
    using hal_llk::unpack::face_transfer;
    using Config = hal::mop::MopConfig<hal::mop::MopTemplate::Template1>;

    const std::uint32_t num_faces_r = tensor_shape.num_faces_r_dim;
    const std::uint32_t num_faces_c = tensor_shape.num_faces_c_dim;
    const std::uint32_t num_faces   = num_faces_r * num_faces_c;

    // With transposed faces (32x32 tiles only) SrcA faces are read in order 0,2,1,3: step Z by 2 within a face row,
    // then restart the second face row at face 1.
    constexpr std::uint32_t srca_restart_second_row = hal::address_counters.client<hal::AddressCounterClient::Unpacker0>()
                                                          .channel<hal::AddressChannel::Channel0>()
                                                          .Z<1>()
                                                          .get_operation<hal::GetOpType::SETTER>();
    const std::uint8_t srca_z_step = transpose_of_faces ? 2 : 1;

    Config config {};
    if constexpr (BType == BroadcastType::COL)
    {
        // One SrcB face per row of faces (its first column carries the broadcast values), then every SrcA face of that row.
        // Advancing SrcB Z by the row width lands on the first face of the next face row.
        config.outer_loop.count    = num_faces_r;
        config.outer_loop.start_op = face_transfer(Engine::Unpacker1, static_cast<std::uint8_t>(num_faces_c));
        if (transpose_of_faces)
        {
            config.outer_loop.end_op = srca_restart_second_row;
        }
        config.inner_loop.count   = num_faces_c;
        config.inner_loop.body_op = face_transfer(Engine::Unpacker0, srca_z_step);
    }
    else if constexpr (BType == BroadcastType::ROW)
    {
        // SrcB faces of the first face row are reused for every face row: B0,A0,B1,A1 then rewind SrcB to face 0.
        config.outer_loop.count  = num_faces_r;
        config.outer_loop.end_op = hal::address_counters.client<hal::AddressCounterClient::Unpacker1>()
                                       .channel<hal::AddressChannel::Channel0>()
                                       .Z<0>()
                                       .get_operation<hal::GetOpType::SETTER>();
        if (transpose_of_faces)
        {
            config.outer_loop.second_end_op = srca_restart_second_row;
        }
        config.inner_loop.count          = num_faces_c;
        config.inner_loop.body_op        = face_transfer(Engine::Unpacker1, 1);
        config.inner_loop.alternating_op = face_transfer(Engine::Unpacker0, srca_z_step);
    }
    else if constexpr (BType == BroadcastType::SCALAR)
    {
        // A single SrcB face (its first datum) is broadcast over the whole tile.
        config.outer_loop.count    = 1;
        config.outer_loop.start_op = face_transfer(Engine::Unpacker1, 1);
        config.inner_loop.count    = num_faces;
        config.inner_loop.body_op  = face_transfer(Engine::Unpacker0, 1);
    }
    else if (transpose_of_faces)
    {
        // SrcA faces in transposed order 0,2,1,3 paired with SrcB faces 0,1,2,3.
        config.outer_loop.count          = 2;
        config.outer_loop.end_op         = srca_restart_second_row;
        config.inner_loop.count          = 2;
        config.inner_loop.body_op        = face_transfer(Engine::Unpacker0, 2);
        config.inner_loop.alternating_op = face_transfer(Engine::Unpacker1, 1);
    }
    else
    {
        // Faces in order, A then B.
        config.outer_loop.count          = 1;
        config.inner_loop.count          = num_faces;
        config.inner_loop.body_op        = face_transfer(Engine::Unpacker0, 1);
        config.inner_loop.alternating_op = face_transfer(Engine::Unpacker1, 1);
    }
    hal::mop::program(config);
}

template <BroadcastType BType = BroadcastType::NONE>
inline void _llk_unpack_AB_init_(const ckernel::TensorShape tensor_shape, const ckernel::Transpose transpose)
{
    using namespace hal;

    const bool within_face = (transpose == ckernel::Transpose::IntraFace) || (transpose == ckernel::Transpose::Both);
    const bool of_faces    = (transpose == ckernel::Transpose::InterFace) || (transpose == ckernel::Transpose::Both);

    // Within-face transpose of SrcA is done by unpacker 0 while it writes the face.
    sync::wait::stall<sync::StallTarget::Config, sync::StallCondition::Unpacker0Idle>();
    cfg::write<cfg::Access::TensixCfgUnit, cfg::Thcon[cfg::Reg2].Haloize_mode, cfg::Sec::S0>(within_face ? 1 : 0);

    // Each transfer moves one face: X runs from datum 0 to face_r_dim * FACE_C_DIM - 1.
    // The X-on-both-channels builder (SETADCXX) places the Channel0 value in the X-end field and the Channel1
    // value in the X-start field, so the end goes on Channel0 here.
    const std::uint32_t x_end = static_cast<std::uint32_t>(tensor_shape.face_r_dim) * FACE_C_DIM - 1;
    hal::runtime_address_counters.client<hal::AddressCounterClient::Unpacker0, hal::AddressCounterClient::Unpacker1>()
        .channel<hal::AddressChannel::Channel0>()
        .X(x_end)
        .channel<hal::AddressChannel::Channel1>()
        .X(0)
        .apply();

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

template <BroadcastType BType = BroadcastType::NONE>
inline void _llk_unpack_AB_(
    const std::uint32_t address_a,
    std::uint32_t address_b,
    [[maybe_unused]] const std::uint32_t bcast_row_idx = 0,
    [[maybe_unused]] const std::uint32_t srcb_format   = 0)
{
    using namespace hal;
    using hal_llk::unpack::ContextSemaphore;

    // Rewind the face (Z) counters of both unpackers to the start of the tile.
    hal::address_counters.client<hal::AddressCounterClient::Unpacker0, hal::AddressCounterClient::Unpacker1>()
        .channel<hal::AddressChannel::Channel0>()
        .Z<0>()
        .W<0>()
        .channel<hal::AddressChannel::Channel1>()
        .Z<0>()
        .W<0>()
        .apply();

    // Wait until the configuration context this tile will use is no longer used by the unpackers.
    while (sync::semaphore::read<ContextSemaphore>() >= hal_llk::unpack::NumContexts)
    {
    }

    // Write the tile addresses into the idle context; the unpackers keep working from the other one.
    // Reading the last written word back makes the RISC wait until both stores have reached the CFG
    // registers, so the MOP issued below cannot observe stale addresses.
    LLK_ASSERT(is_valid_L1_address(address_a), "L1 address_a must be in valid L1 memory region");
    LLK_ASSERT(is_valid_L1_address(address_b), "L1 address_b must be in valid L1 memory region");
    std::uint32_t written_b;
    if (unp_cfg_context == 0)
    {
        cfg::write<cfg::Access::MMIO, cfg::Unpacker[0].Cntx[0].Base, cfg::Sec::S0>(address_a);
        cfg::write<cfg::Access::MMIO, cfg::Unpacker[1].Cntx[0].Base, cfg::Sec::S0>(address_b);
        written_b = cfg::read<cfg::Access::MMIO, cfg::Unpacker[1].Cntx[0].Base, cfg::Sec::S0>();
    }
    else
    {
        cfg::write<cfg::Access::MMIO, cfg::Unpacker[0].Cntx[1].Base, cfg::Sec::S0>(address_a);
        cfg::write<cfg::Access::MMIO, cfg::Unpacker[1].Cntx[1].Base, cfg::Sec::S0>(address_b);
        written_b = cfg::read<cfg::Access::MMIO, cfg::Unpacker[1].Cntx[1].Base, cfg::Sec::S0>();
    }
    // NOT_IN_HAL: hal::sync has no stall condition for "TRISC CFG writes landed" (p_stall::TRISC_CFG), so the RISC
    // itself waits: this RISC-V instruction consumes the read-back value and blocks until the load has returned.
    asm volatile("andi x0, %0, 0" : : "r"(written_b) : "memory");

    // Mark the context busy and unpack the tile.
    sync::semaphore::post<sync::Access::MMIO, ContextSemaphore>();
    hal::mop::Runner<hal::mop::MopTemplate::Template1>::run();

    // Release the context once the unpack instructions for this tile have been issued.
    sync::semaphore::get<ContextSemaphore>();

    // Switch both unpackers to the other configuration context for the next tile.
    unp_cfg_context = 1 - unp_cfg_context;
    if (unp_cfg_context == 0)
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
