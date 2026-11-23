// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// HAL-based implementation of the two-operand (A/B) unpack LLK.

#pragma once

#include <cstdint>

#include "hal/mop.h"
#include "hal/unpack.h"
#include "llk_hal_common.h"
#include "llk_unpack_common.h"
#include "tensor_shape.h"

namespace hal_llk::unpack_ab
{

using hal::unpack::Engine;

// One face per UNPACR: advance the L1 read position by one face (channel-0 Y; the tile descriptor
// has one face-sized row per Y step) and hand the filled bank to math.
constexpr hal::unpack::DataTransfer face_transfer(const Engine engine, const std::uint8_t face_step)
{
    return {
        .engine     = engine,
        .increments = {.channel0 = {.y = face_step}},
        .handoff    = hal::unpack::SourceHandoff::FlipAndSetDataValid,
    };
}

inline constexpr std::uint32_t UNPACK_A      = hal::unpack::get_operation<face_transfer(Engine::Unpacker0, 1)>();
inline constexpr std::uint32_t UNPACK_A_SKIP = hal::unpack::get_operation<face_transfer(Engine::Unpacker0, 2)>();
inline constexpr std::uint32_t UNPACK_B      = hal::unpack::get_operation<face_transfer(Engine::Unpacker1, 1)>();

// Reposition the L1 face counter (channel-0 Y) of one unpacker.
inline constexpr std::uint32_t SRCA_FACE_1 =
    hal::address_counters.client<hal::AddressCounterClient::Unpacker0>().channel<hal::AddressChannel::Channel0>().Y<1>().get_operation<hal::GetOpType::SETTER>();
inline constexpr std::uint32_t SRCB_FACE_0 =
    hal::address_counters.client<hal::AddressCounterClient::Unpacker1>().channel<hal::AddressChannel::Channel0>().Y<0>().get_operation<hal::GetOpType::SETTER>();
inline constexpr std::uint32_t SRCB_FACE_2 =
    hal::address_counters.client<hal::AddressCounterClient::Unpacker1>().channel<hal::AddressChannel::Channel0>().Y<2>().get_operation<hal::GetOpType::SETTER>();

using MopConfig1 = hal::mop::MopConfig<hal::mop::MopTemplate::Template1>;
using hal::mop::Template1Field;

// A0 B0 A1 B1 ...
inline constexpr MopConfig1 MOP_NO_BCAST {
    .outer_loop = {.count = 1},
    .inner_loop = {.count = 4, .body_op = UNPACK_A, .alternating_op = UNPACK_B},
};

// A faces in order 0,2,1,3 (inter-face transpose), B faces in order 0,1,2,3.
inline constexpr MopConfig1 MOP_TRANSPOSE_FACES {
    .outer_loop = {.count = 2, .end_op = SRCA_FACE_1},
    .inner_loop = {.count = 2, .body_op = UNPACK_A_SKIP, .alternating_op = UNPACK_B},
};

// Column broadcast: B face 0 serves A faces 0,1; B face 2 serves A faces 2,3.
inline constexpr MopConfig1 MOP_COL_BCAST {
    .outer_loop = {.count = 2, .start_op = UNPACK_B, .end_op = SRCB_FACE_2},
    .inner_loop = {.count = 2, .body_op = UNPACK_A},
};

// Row broadcast: B faces 0,1 (tile row 0) serve both A face rows.
inline constexpr MopConfig1 MOP_ROW_BCAST {
    .outer_loop = {.count = 2, .end_op = SRCB_FACE_0},
    .inner_loop = {.count = 2, .body_op = UNPACK_B, .alternating_op = UNPACK_A},
};

// Column broadcast with inter-face transpose of A: A faces 0,2 with B face 0, A faces 1,3 with B face 2.
inline constexpr MopConfig1 MOP_COL_BCAST_TRANSPOSE {
    .outer_loop = {.count = 2, .start_op = UNPACK_B, .end_op = SRCB_FACE_2, .second_end_op = SRCA_FACE_1},
    .inner_loop = {.count = 2, .body_op = UNPACK_A_SKIP},
};

// Row broadcast with inter-face transpose of A: A faces 0,2 then 1,3, each pair with B faces 0,1.
inline constexpr MopConfig1 MOP_ROW_BCAST_TRANSPOSE {
    .outer_loop = {.count = 2, .end_op = SRCB_FACE_0, .second_end_op = SRCA_FACE_1},
    .inner_loop = {.count = 2, .body_op = UNPACK_B, .alternating_op = UNPACK_A_SKIP},
};

// Scalar broadcast: B face 0 once, then every A face.
inline constexpr MopConfig1 MOP_SCALAR_BCAST {
    .outer_loop = {.count = 1, .start_op = UNPACK_B},
    .inner_loop = {.count = 4, .body_op = UNPACK_A},
};

template <BroadcastType BType>
inline void program_mop(const bool transpose_of_faces, const std::uint32_t num_faces)
{
    const std::uint32_t outer = (num_faces < 4) ? 1 : 2;
    const std::uint32_t inner = (num_faces < 2) ? 1 : 2;
    const bool reorder_a_faces = transpose_of_faces && (num_faces == 4);
    if constexpr (BType == BroadcastType::COL)
    {
        if (reorder_a_faces)
        {
            hal::mop::program<MOP_COL_BCAST_TRANSPOSE>();
        }
        else
        {
            hal::mop::program<MOP_COL_BCAST>(
                hal::mop::runtime_field<Template1Field::OuterLoopCount>(outer), hal::mop::runtime_field<Template1Field::InnerLoopCount>(inner));
        }
    }
    else if constexpr (BType == BroadcastType::ROW)
    {
        if (reorder_a_faces)
        {
            hal::mop::program<MOP_ROW_BCAST_TRANSPOSE>();
        }
        else
        {
            hal::mop::program<MOP_ROW_BCAST>(
                hal::mop::runtime_field<Template1Field::OuterLoopCount>(outer), hal::mop::runtime_field<Template1Field::InnerLoopCount>(inner));
        }
    }
    else if constexpr (BType == BroadcastType::SCALAR)
    {
        hal::mop::program<MOP_SCALAR_BCAST>(hal::mop::runtime_field<Template1Field::InnerLoopCount>(num_faces));
    }
    else
    {
        if (transpose_of_faces && (num_faces == 4))
        {
            hal::mop::program<MOP_TRANSPOSE_FACES>();
        }
        else
        {
            hal::mop::program<MOP_NO_BCAST>(hal::mop::runtime_field<Template1Field::InnerLoopCount>(num_faces));
        }
    }
}

} // namespace hal_llk::unpack_ab

/**
 * @brief Configure the unpack MOP for operands A and B (broadcast / face-transpose aware).
 */
template <BroadcastType BType = BroadcastType::NONE>
inline void _llk_unpack_AB_mop_config_(const bool transpose_of_faces, const ckernel::TensorShape tensor_shape)
{
    hal_llk::unpack_ab::program_mop<BType>(transpose_of_faces, tensor_shape.total_num_faces());
}

/**
 * @brief Initialize unpacker to unpack two source operands A and B into SrcA and SrcB.
 */
template <BroadcastType BType = BroadcastType::NONE>
inline void _llk_unpack_AB_init_(const ckernel::TensorShape tensor_shape, const ckernel::Transpose transpose)
{
    namespace cfg                 = hal::cfg;
    const bool transpose_of_faces = (transpose == ckernel::Transpose::InterFace) || (transpose == ckernel::Transpose::Both);
    const bool transpose_xy_srca  = (transpose == ckernel::Transpose::IntraFace) || (transpose == ckernel::Transpose::Both);

    // Within-face transpose is a mode of unpacker 0; it must not change under a running unpack.
    hal::sync::wait::stall<hal::sync::StallTarget::Config, hal::sync::StallCondition::Unpacker0Idle>();
    cfg::write<cfg::Access::TensixCfgUnit, cfg::Thcon[cfg::Reg2].Haloize_mode, cfg::Sec::S0>(transpose_xy_srca ? 1u : 0u);

    hal_llk::unpack::config_unpacker_x_end(tensor_shape.face_r_dim);
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
 * @brief Unpack one tile of A and one tile of B from L1 into SrcA/SrcB.
 *
 * The RISC programs the tile addresses into the configuration context the unpacker is not using
 * (MMIO, no Tensix instructions), then switches the unpacker to that context in-stream.
 */
template <BroadcastType BType = BroadcastType::NONE>
inline void _llk_unpack_AB_(
    const std::uint32_t address_a,
    std::uint32_t address_b,
    [[maybe_unused]] const std::uint32_t bcast_row_idx = 0,
    [[maybe_unused]] const std::uint32_t srcb_format   = 0)
{
    using namespace hal_llk;
    namespace cfg = hal::cfg;

    // Restart both unpackers at the first face of the tile (channel-1 X keeps the datum range).
    hal::address_counters.client<hal::AddressCounterClient::Unpacker0, hal::AddressCounterClient::Unpacker1>()
        .channel<hal::AddressChannel::Channel0>()
        .Y<0>()
        .channel<hal::AddressChannel::Channel1>()
        .Y<0>()
        .apply();

    // At most two tiles in flight: wait until the context we are about to program is released.
    while (hal::sync::semaphore::read<SEM_UNPACK_SYNC>() >= 2)
    {
    }

    if (unpack::next_context == 0)
    {
        cfg::write<cfg::Access::MMIO, cfg::Unpacker[0].Cntx[0].Base, cfg::Sec::S0>(address_a);
        cfg::write<cfg::Access::MMIO, cfg::Unpacker[1].Cntx[0].Base, cfg::Sec::S0>(address_b);
    }
    else
    {
        cfg::write<cfg::Access::MMIO, cfg::Unpacker[0].Cntx[1].Base, cfg::Sec::S0>(address_a);
        cfg::write<cfg::Access::MMIO, cfg::Unpacker[1].Cntx[1].Base, cfg::Sec::S0>(address_b);
    }

    // Context acquired by the RISC; released by the unpack thread once this tile's UNPACRs are issued.
    hal::sync::semaphore::post<hal::sync::Access::MMIO, SEM_UNPACK_SYNC>();

    // The MMIO stores above are not ordered with the instruction stream: hold the unpacker until they landed.
    hal::sync::wait::stall<hal::sync::StallTarget::Unpack, hal::sync::StallCondition::RiscvAccessProcessed>();

    hal::mop::Runner<hal::mop::MopTemplate::Template1>::run();

    hal::sync::semaphore::get<SEM_UNPACK_SYNC>();

    // Switch the unpackers to the context programmed for the next tile.
    if (unpack::next_context == 0)
    {
        unpack::next_context = 1;
        cfg::write<cfg::Access::TensixCfgUnit>(
            cfg::set<cfg::UnpackMiscCfg::CfgContextOffset_0, cfg::Sec::S0, 1>(), cfg::set<cfg::UnpackMiscCfg::CfgContextOffset_1, cfg::Sec::S0, 1>());
    }
    else
    {
        unpack::next_context = 0;
        cfg::write<cfg::Access::TensixCfgUnit>(
            cfg::set<cfg::UnpackMiscCfg::CfgContextOffset_0, cfg::Sec::S0, 0>(), cfg::set<cfg::UnpackMiscCfg::CfgContextOffset_1, cfg::Sec::S0, 0>());
    }
}
