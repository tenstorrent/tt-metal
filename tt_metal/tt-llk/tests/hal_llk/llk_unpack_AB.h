// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "hal/mop.h"
#include "hal/unpack.h"
#include "llk_hal_common.h"
#include "llk_unpack_common.h"

namespace hal_llk
{

// One face from L1 into the next SrcA/SrcB bank; hands the bank to math.
constexpr hal::unpack::DataTransfer unpack_face(const hal::unpack::Engine engine, const std::uint8_t z_increment)
{
    return {
        .engine     = engine,
        .increments = {.channel0 = {.z = z_increment}},
        .handoff    = hal::unpack::SourceHandoff::FlipAndSetDataValid,
    };
}

inline constexpr std::uint32_t UNPACK_A_NEXT_FACE = hal::unpack::get_operation<unpack_face(hal::unpack::Engine::Unpacker0, 1)>();
inline constexpr std::uint32_t UNPACK_A_SKIP_FACE = hal::unpack::get_operation<unpack_face(hal::unpack::Engine::Unpacker0, 2)>();
inline constexpr std::uint32_t UNPACK_B_NEXT_FACE = hal::unpack::get_operation<unpack_face(hal::unpack::Engine::Unpacker1, 1)>();
inline constexpr std::uint32_t UNPACK_B_SAME_FACE = hal::unpack::get_operation<unpack_face(hal::unpack::Engine::Unpacker1, 0)>();

// Jump the L1 face (channel-0 Z) counter of one unpacker.
template <hal::AddressCounterClient Client, std::uint8_t Face>
inline constexpr std::uint32_t UNPACK_SET_FACE = hal::address_counters.client<Client>()
                                                      .template channel<hal::AddressChannel::Channel0>()
                                                      .template Z<Face>()
                                                      .template get_operation<hal::GetOpType::SETTER>();

} // namespace hal_llk

/**
 * @brief Program the unpack MOP for one tile of operands A and B, honouring broadcast and face transpose.
 */
template <BroadcastType BType = BroadcastType::NONE>
inline void _llk_unpack_AB_mop_config_(const bool transpose_of_faces, const ckernel::TensorShape tensor_shape)
{
    using namespace hal::mop;
    using hal::AddressCounterClient;

    const std::uint32_t num_faces = tensor_shape.total_num_faces();
    const bool narrow_tile        = (tensor_shape.num_faces_c_dim == 1) && (num_faces > 1);
    const std::uint32_t outerloop = num_faces < 4 ? 1 : 2;
    const std::uint32_t innerloop = num_faces < 2 ? 1 : 2;

    // With face transpose, SrcA faces go 0, 2 in the first half and 1, 3 in the second: step A's
    // face counter by 2 and rewind it to face 1 between the halves.
    const bool reorder_a         = transpose_of_faces && num_faces == 4;
    const std::uint32_t unpack_a = reorder_a ? hal_llk::UNPACK_A_SKIP_FACE : hal_llk::UNPACK_A_NEXT_FACE;
    const std::uint32_t rewind_a = hal_llk::UNPACK_SET_FACE<AddressCounterClient::Unpacker0, 1>;

    if constexpr (BType == BroadcastType::COL)
    {
        // One B face feeds a row of output faces: B0, A0, A1, then B2, A2, A3.
        const std::uint32_t next_b = narrow_tile ? hal_llk::UNPACK_B_NEXT_FACE : hal_llk::UNPACK_SET_FACE<AddressCounterClient::Unpacker1, 2>;
        if (reorder_a)
        {
            program(MopConfig<MopTemplate::Template1> {
                .outer_loop = {.count = outerloop, .start_op = hal_llk::UNPACK_B_NEXT_FACE, .end_op = next_b, .second_end_op = rewind_a},
                .inner_loop = {.count = innerloop, .body_op = unpack_a},
            });
        }
        else
        {
            program(MopConfig<MopTemplate::Template1> {
                .outer_loop = {.count = outerloop, .start_op = hal_llk::UNPACK_B_NEXT_FACE, .end_op = next_b},
                .inner_loop = {.count = innerloop, .body_op = unpack_a},
            });
        }
    }
    else if constexpr (BType == BroadcastType::ROW)
    {
        // B faces 0/1 pair with the first two output faces and again with the last two.
        const std::uint32_t unpack_b = narrow_tile ? hal_llk::UNPACK_B_SAME_FACE : hal_llk::UNPACK_B_NEXT_FACE;
        const std::uint32_t rewind_b = hal_llk::UNPACK_SET_FACE<AddressCounterClient::Unpacker1, 0>;
        if (reorder_a)
        {
            program(MopConfig<MopTemplate::Template1> {
                .outer_loop = {.count = outerloop, .end_op = rewind_b, .second_end_op = rewind_a},
                .inner_loop = {.count = innerloop, .body_op = unpack_b, .alternating_op = unpack_a},
            });
        }
        else
        {
            program(MopConfig<MopTemplate::Template1> {
                .outer_loop = {.count = outerloop, .end_op = rewind_b},
                .inner_loop = {.count = innerloop, .body_op = unpack_b, .alternating_op = unpack_a},
            });
        }
    }
    else if constexpr (BType == BroadcastType::SCALAR)
    {
        // One B face, then every A face.
        program(MopConfig<MopTemplate::Template1> {
            .outer_loop = {.count = 1, .start_op = hal_llk::UNPACK_B_NEXT_FACE},
            .inner_loop = {.count = num_faces, .body_op = hal_llk::UNPACK_A_NEXT_FACE},
        });
    }
    else
    {
        if (transpose_of_faces)
        {
            // A faces in order 0, 2, 1, 3 paired with B faces 0, 1, 2, 3.
            program(MopConfig<MopTemplate::Template1> {
                .outer_loop = {.count = outerloop, .end_op = hal_llk::UNPACK_SET_FACE<AddressCounterClient::Unpacker0, 1>},
                .inner_loop =
                    {.count          = innerloop,
                     .body_op        = num_faces < 4 ? hal_llk::UNPACK_A_NEXT_FACE : hal_llk::UNPACK_A_SKIP_FACE,
                     .alternating_op = hal_llk::UNPACK_B_NEXT_FACE},
            });
        }
        else
        {
            program(MopConfig<MopTemplate::Template1> {
                .outer_loop = {.count = 1},
                .inner_loop = {.count = num_faces, .body_op = hal_llk::UNPACK_A_NEXT_FACE, .alternating_op = hal_llk::UNPACK_B_NEXT_FACE},
            });
        }
    }
}

template <BroadcastType BType = BroadcastType::NONE>
inline void _llk_unpack_AB_init_(const ckernel::TensorShape tensor_shape, const ckernel::Transpose transpose)
{
    namespace cfg  = hal::cfg;
    namespace sync = hal::sync;

    const bool within_face = (transpose == ckernel::Transpose::IntraFace) || (transpose == ckernel::Transpose::Both);
    const bool of_faces    = (transpose == ckernel::Transpose::InterFace) || (transpose == ckernel::Transpose::Both);

    sync::wait::stall<sync::StallTarget::Config, sync::StallCondition::Unpacker0Idle | sync::StallCondition::Unpacker1Idle>();

    // SrcA transpose within each face.
    cfg::write<cfg::Access::TensixCfgUnit, cfg::Thcon[cfg::Reg2].Haloize_mode, cfg::Sec::S0>(within_face ? 1 : 0);

    // Datums per UNPACR: one face of face_r_dim rows (X start 0, X end face_dim - 1). The two
    // channels are written separately: the combined X/X builder form (SETADCXX) hung the unpacker.
    hal::address_counters.client<hal::AddressCounterClient::Unpacker0, hal::AddressCounterClient::Unpacker1>()
        .channel<hal::AddressChannel::Channel0>()
        .X<0>()
        .apply();
    hal::runtime_address_counters.client<hal::AddressCounterClient::Unpacker0, hal::AddressCounterClient::Unpacker1>()
        .channel<hal::AddressChannel::Channel1>()
        .X(tensor_shape.face_r_dim * FACE_C_DIM - 1)
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
    _llk_unpack_AB_init_<BType>(tensor_shape, transpose != 0 ? ckernel::Transpose::Both : ckernel::Transpose::None);
}

inline void _llk_unpack_AB_uninit_()
{
}

/**
 * @brief Unpack one A tile into SrcA and one B tile into SrcB.
 *
 * The L1 addresses are written through MMIO into the configuration context the unpacker is not
 * using; the UNPACK_SYNC semaphore bounds the number of contexts in flight.
 */
template <BroadcastType BType = BroadcastType::NONE>
inline void _llk_unpack_AB_(
    const std::uint32_t address_a, std::uint32_t address_b, [[maybe_unused]] const std::uint32_t bcast_row_idx = 0, [[maybe_unused]] const std::uint32_t srcb_format = 0)
{
    namespace cfg  = hal::cfg;
    namespace sync = hal::sync;

    // Restart both unpackers at face 0 of the new tiles.
    hal::address_counters.client<hal::AddressCounterClient::Unpacker0, hal::AddressCounterClient::Unpacker1>()
        .channel<hal::AddressChannel::Channel0>()
        .Z<0>()
        .apply();

    // Wait for a free configuration context, then fill it with this tile's addresses.
    while (sync::semaphore::read<hal_llk::UNPACK_SYNC_SEM>() >= hal_llk::UNPACK_NUM_CONTEXTS)
    {
    }
    // The unpacker must observe these RISC writes before the MOP is issued. hal::sync has no
    // stall condition for pending TRISC CFG writes, so read the last store back: the stores are
    // performed in order, so once B's address is visible A's is as well.
    if (unp_cfg_context == 0)
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
    sync::semaphore::post<sync::Access::MMIO, hal_llk::UNPACK_SYNC_SEM>();

    hal::mop::Runner<hal::mop::MopTemplate::Template1>::run();

    // Release the context and switch to the other one for the next tile.
    sync::semaphore::get<hal_llk::UNPACK_SYNC_SEM>();
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
