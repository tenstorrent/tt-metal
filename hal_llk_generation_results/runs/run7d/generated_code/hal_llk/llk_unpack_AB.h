// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// HAL-based implementation of the two-operand (A/B) unpack LLK.
//
// Every output face is fed as one SrcA face plus one SrcB face, each published to the FPU with its own
// data-valid handoff. The SrcB face is chosen according to the broadcast type and the SrcA face according
// to the inter-face transpose, so the math thread always consumes exactly one (SrcA, SrcB) bank pair per face.

#pragma once

#include <cstdint>

#include "ckernel.h"
#include "hal/address_counters.h"
#include "hal/cfg.h"
#include "hal/mop.h"
#include "hal/sync.h"
#include "hal/unpack.h"
#include "hal_llk_common.h"
#include "llk_defs.h"
#include "llk_unpack_common.h"
#include "tensor_shape.h"

using namespace ckernel;

namespace hal_llk::unpack_detail
{

// Geometry of the tile currently configured by _llk_unpack_AB_init_.
struct AbTileGeometry
{
    std::uint32_t num_faces       = 4;
    std::uint32_t num_faces_c_dim = 2;
    bool transpose_faces          = false;
};

inline AbTileGeometry ab_geometry {};

constexpr hal::unpack::DataTransfer face_transfer(const hal::unpack::Engine engine)
{
    return {
        .engine  = engine,
        .handoff = hal::unpack::SourceHandoff::FlipAndSetDataValid,
    };
}

constexpr hal::unpack::DataTransfer unpack_srca_face = face_transfer(hal::unpack::Engine::Unpacker0);
constexpr hal::unpack::DataTransfer unpack_srcb_face = face_transfer(hal::unpack::Engine::Unpacker1);

// Face of operand A that lands in output face `face` (faces are reordered 0,2,1,3 by the inter-face transpose).
inline std::uint32_t srca_face(const std::uint32_t face, const AbTileGeometry& geometry)
{
    if (!geometry.transpose_faces)
    {
        return face;
    }
    const std::uint32_t num_faces_r_dim = geometry.num_faces / geometry.num_faces_c_dim;
    const std::uint32_t row             = face / geometry.num_faces_c_dim;
    const std::uint32_t col             = face % geometry.num_faces_c_dim;
    return col * num_faces_r_dim + row;
}

// Face of operand B combined with output face `face` for the given broadcast type.
template <BroadcastType BType>
inline std::uint32_t srcb_face(const std::uint32_t face, const AbTileGeometry& geometry)
{
    if constexpr (BType == BroadcastType::SCALAR)
    {
        return 0;
    }
    else if constexpr (BType == BroadcastType::ROW)
    {
        // Row broadcast takes the first row of the tile: the top face of the same face column.
        return face % geometry.num_faces_c_dim;
    }
    else if constexpr (BType == BroadcastType::COL)
    {
        // Column broadcast takes the first column of the tile: the left face of the same face row.
        return face - (face % geometry.num_faces_c_dim);
    }
    else
    {
        return face;
    }
}

} // namespace hal_llk::unpack_detail

/**
 * @brief Initialize unpacker to unpack two source operands A and B into SrcA and SrcB registers
 */
template <BroadcastType BType = BroadcastType::NONE>
inline void _llk_unpack_AB_init_(const ckernel::TensorShape tensor_shape, const ckernel::Transpose transpose)
{
    namespace cfg = hal::cfg;
    using namespace hal_llk::unpack_detail;

    const bool transpose_faces  = (transpose == ckernel::Transpose::InterFace) || (transpose == ckernel::Transpose::Both);
    const bool transpose_within = (transpose == ckernel::Transpose::IntraFace) || (transpose == ckernel::Transpose::Both);

    ab_geometry = AbTileGeometry {
        .num_faces       = tensor_shape.total_num_faces(),
        .num_faces_c_dim = tensor_shape.num_faces_c_dim,
        .transpose_faces = transpose_faces,
    };

    wait_for_unpackers_idle();

    // Within-face transpose of SrcA is done by the unpacker while writing the face.
    cfg::write<cfg::Access::TensixCfgUnit, cfg::Thcon[cfg::Reg2].Haloize_mode, cfg::Sec::S0>(transpose_within ? 1 : 0);

    // Each transfer moves one face: input datums [0, face_r_dim * FACE_C_DIM - 1].
    // Through the HAL builder the last datum goes on channel 0 X and the first datum on channel 1 X
    // (observed on hardware: the reverse assignment never completes the transfer).
    const std::uint32_t x_end = tensor_shape.face_r_dim * FACE_C_DIM - 1;
    hal::runtime_address_counters.client<hal::AddressCounterClient::Unpacker0, hal::AddressCounterClient::Unpacker1>()
        .channel<hal::AddressChannel::Channel0>()
        .X(x_end)
        .channel<hal::AddressChannel::Channel1>()
        .X(0)
        .apply();
}

/**
 * @brief Initialize unpacker to unpack operands A and B with no transpose.
 */
template <BroadcastType BType = BroadcastType::NONE>
inline void _llk_unpack_AB_init_(const ckernel::TensorShape tensor_shape = ckernel::DEFAULT_TENSOR_SHAPE)
{
    _llk_unpack_AB_init_<BType>(tensor_shape, ckernel::Transpose::None);
}

/**
 * @brief Initialize unpacker to unpack operands A and B with a boolean transpose flag.
 */
template <BroadcastType BType = BroadcastType::NONE>
inline void _llk_unpack_AB_init_(const ckernel::TensorShape tensor_shape, const std::uint32_t transpose)
{
    _llk_unpack_AB_init_<BType>(tensor_shape, transpose ? ckernel::Transpose::Both : ckernel::Transpose::None);
}

/**
 * @brief No-op teardown after AB (two-operand) unpacking.
 */
inline void _llk_unpack_AB_uninit_()
{
}

/**
 * @brief Unpack two tiles from L1 memory into SrcA and SrcB registers
 */
template <BroadcastType BType = BroadcastType::NONE>
inline void _llk_unpack_AB_(
    const std::uint32_t address_a,
    std::uint32_t address_b,
    [[maybe_unused]] const std::uint32_t bcast_row_idx = 0,
    [[maybe_unused]] const std::uint32_t srcb_format   = 0)
{
    namespace cfg = hal::cfg;
    using namespace hal_llk::unpack_detail;

    // The previous tile's transfers still read the base addresses; let them finish first.
    wait_for_unpackers_idle();
    cfg::write<cfg::Access::TensixCfgUnit>(
        cfg::set<cfg::Unpacker[0].Cntx[0].Base, cfg::Sec::S0>(address_a), cfg::set<cfg::Unpacker[1].Cntx[0].Base, cfg::Sec::S0>(address_b));

    const AbTileGeometry geometry = ab_geometry;
    for (std::uint32_t face = 0; face < geometry.num_faces; face++)
    {
        // The channel-0 Z counter selects the input face in L1.
        hal::runtime_address_counters.client<hal::AddressCounterClient::Unpacker0>()
            .channel<hal::AddressChannel::Channel0>()
            .Z(srca_face(face, geometry))
            .apply();
        hal::unpack::run<unpack_srca_face>();

        hal::runtime_address_counters.client<hal::AddressCounterClient::Unpacker1>()
            .channel<hal::AddressChannel::Channel0>()
            .Z(srcb_face<BType>(face, geometry))
            .apply();
        hal::unpack::run<unpack_srcb_face>();
    }
}
