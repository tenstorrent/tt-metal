// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// HAL-based implementation of the Blackhole two-operand (A/B) unpack LLK.

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
#include "llk_assert.h"
#include "llk_defs.h"
#include "llk_unpack_common.h"
#include "tensor_shape.h"

namespace hal_llk::unpack
{

// One face unpack into SrcA (unpacker 0) or SrcB (unpacker 1). The input Z counter (face index in L1)
// advances by z_increment, and the completed bank is handed to the math thread.
constexpr hal::unpack::DataTransfer face_transfer(const hal::unpack::Engine engine, const std::uint8_t z_increment)
{
    return {
        .engine     = engine,
        .increments = {.channel0 = {.z = z_increment}},
        .handoff    = hal::unpack::SourceHandoff::FlipAndSetDataValid,
    };
}

constexpr std::uint32_t unpack_srca_next_face   = hal::unpack::get_operation<face_transfer(hal::unpack::Engine::Unpacker0, 1)>();
constexpr std::uint32_t unpack_srca_skip_face   = hal::unpack::get_operation<face_transfer(hal::unpack::Engine::Unpacker0, 2)>();
constexpr std::uint32_t unpack_srcb_next_face   = hal::unpack::get_operation<face_transfer(hal::unpack::Engine::Unpacker1, 1)>();
constexpr std::uint32_t unpack_srcb_skip_face   = hal::unpack::get_operation<face_transfer(hal::unpack::Engine::Unpacker1, 2)>();
constexpr std::uint32_t unpack_srcb_same_face   = hal::unpack::get_operation<face_transfer(hal::unpack::Engine::Unpacker1, 0)>();

// Input-face (channel 0 Z) counter rewinds.
constexpr std::uint32_t srca_face_to_1 = hal::address_counters.client<hal::AddressCounterClient::Unpacker0>()
                                             .channel<hal::AddressChannel::Channel0>()
                                             .Z<1>()
                                             .get_operation<hal::GetOpType::SETTER>();
constexpr std::uint32_t srcb_face_to_0 = hal::address_counters.client<hal::AddressCounterClient::Unpacker1>()
                                             .channel<hal::AddressChannel::Channel0>()
                                             .Z<0>()
                                             .get_operation<hal::GetOpType::SETTER>();

} // namespace hal_llk::unpack

template <BroadcastType BType = BroadcastType::NONE>
inline void _llk_unpack_AB_mop_config_(const bool transpose_of_faces, const ckernel::TensorShape tensor_shape)
{
    using namespace hal_llk::unpack;
    using hal::mop::MopConfig;
    using hal::mop::MopTemplate;

    const std::uint32_t num_faces       = tensor_shape.total_num_faces();
    const std::uint32_t num_faces_r_dim = tensor_shape.num_faces_r_dim;
    const std::uint32_t num_faces_c_dim = tensor_shape.num_faces_c_dim;

    // Face transpose (A faces 0,2,1,3) is only defined for a full 2x2-face tile.
    const bool transpose_faces = transpose_of_faces && num_faces_r_dim == 2 && num_faces_c_dim == 2;
    // A faces are visited row by row; with face transpose the A face index steps by 2 and rewinds to 1.
    const std::uint32_t srca_op = transpose_faces ? unpack_srca_skip_face : unpack_srca_next_face;

    MopConfig<MopTemplate::Template1> config {};

    if constexpr (BType == BroadcastType::NONE)
    {
        // Pair every A face with the B face at the same position.
        config.outer_loop.count = transpose_faces ? 2 : 1;
        if (transpose_faces)
        {
            config.outer_loop.end_op = srca_face_to_1;
        }
        config.inner_loop.count          = transpose_faces ? 2 : num_faces;
        config.inner_loop.body_op        = srca_op;
        config.inner_loop.alternating_op = unpack_srcb_next_face;
    }
    else if constexpr (BType == BroadcastType::COL)
    {
        // Column 0 lives in the first face of each face row: one B face per face row, shared by its A faces.
        config.outer_loop.count    = num_faces_r_dim;
        config.outer_loop.start_op = (num_faces_c_dim == 2) ? unpack_srcb_skip_face : unpack_srcb_next_face;
        if (transpose_faces)
        {
            config.outer_loop.end_op = srca_face_to_1;
        }
        config.inner_loop.count   = num_faces_c_dim;
        config.inner_loop.body_op = srca_op;
    }
    else if constexpr (BType == BroadcastType::ROW)
    {
        // Row 0 lives in the face-row-0 faces: every A face row reuses B faces 0..c_dim-1.
        config.outer_loop.count  = num_faces_r_dim;
        config.outer_loop.end_op = srcb_face_to_0;
        if (transpose_faces)
        {
            config.outer_loop.second_end_op = srca_face_to_1;
        }
        config.inner_loop.count          = num_faces_c_dim;
        config.inner_loop.body_op        = srca_op;
        config.inner_loop.alternating_op = unpack_srcb_next_face;
    }
    else
    {
        LLK_ASSERT(!transpose_faces, "SrcA face transpose is not supported with scalar broadcast");
        // The scalar lives in B face 0: unpack it once, then every A face.
        config.outer_loop.count    = 1;
        config.outer_loop.start_op = unpack_srcb_same_face;
        config.inner_loop.count    = num_faces;
        config.inner_loop.body_op  = unpack_srca_next_face;
    }

    hal::mop::program(config);
}

template <BroadcastType BType = BroadcastType::NONE>
inline void _llk_unpack_AB_init_(const ckernel::TensorShape tensor_shape, const ckernel::Transpose transpose)
{
    namespace cfg = hal::cfg;

    const bool transpose_within_face = (transpose == ckernel::Transpose::IntraFace) || (transpose == ckernel::Transpose::Both);
    const bool transpose_of_faces    = (transpose == ckernel::Transpose::InterFace) || (transpose == ckernel::Transpose::Both);

    // The unpackers must be idle before their configuration changes.
    hal::sync::wait::stall<hal::sync::StallTarget::Config, hal::sync::StallCondition::Unpacker0Idle | hal::sync::StallCondition::Unpacker1Idle>();

    // SrcA within-face transpose is an unpacker 0 output mode.
    cfg::write<cfg::Access::TensixCfgUnit, cfg::Thcon[cfg::Reg2].Haloize_mode, cfg::Sec::S0>(transpose_within_face ? 1 : 0);

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

template <BroadcastType BType = BroadcastType::NONE>
inline void _llk_unpack_AB_(
    const std::uint32_t address_a,
    std::uint32_t address_b,
    [[maybe_unused]] const std::uint32_t bcast_row_idx = 0,
    [[maybe_unused]] const std::uint32_t srcb_format   = 0)
{
    namespace cfg  = hal::cfg;
    namespace sync = hal::sync;

    // Start every tile from input face 0 of both operands.
    hal_llk::unpack::reset_unpacker_yzw_counters();

    // Wait for a free configuration context (two are in flight at most).
    while (sync::semaphore::read<sync::Semaphore::S5>() >= 2)
    {
    }

    LLK_ASSERT(is_valid_L1_address(address_a), "L1 address_a must be in valid L1 memory region");
    LLK_ASSERT(is_valid_L1_address(address_b), "L1 address_b must be in valid L1 memory region");

    // Source tile addresses of the selected context, written in order with the unpack instructions.
    if (unp_cfg_context == 0)
    {
        cfg::write<cfg::Access::TensixCfgUnit>(
            cfg::set<cfg::Thcon[cfg::Reg3].Base_address, cfg::Sec::S0>(address_a), cfg::set<cfg::Thcon[cfg::Reg3].Base_address, cfg::Sec::S1>(address_b));
    }
    else
    {
        cfg::write<cfg::Access::TensixCfgUnit>(
            cfg::set<cfg::Thcon[cfg::Reg3].Base_cntx1_address, cfg::Sec::S0>(address_a),
            cfg::set<cfg::Thcon[cfg::Reg3].Base_cntx1_address, cfg::Sec::S1>(address_b));
    }

    // Acquire the context for the unpackers; it is released by the Tensix side once the MOP has been issued.
    sync::semaphore::post<sync::Access::MMIO, sync::Semaphore::S5>();

    // Unpack only after the configuration writes above have landed.
    sync::wait::stall<sync::StallTarget::Unpack, sync::StallCondition::ConfigUnitIdle>();

    hal::mop::Runner<hal::mop::MopTemplate::Template1>::run();

    sync::semaphore::get<sync::Semaphore::S5>();

    hal_llk::unpack::switch_config_context();
}
