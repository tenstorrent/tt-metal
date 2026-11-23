// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "ckernel_gpr_map.h"

// Which Tensix GPR each thread uses for what. The numbers are ckernel's (ckernel_gpr_map.h), so code on this HAL and
// code on the older LLKs agree on who owns a GPR.
namespace llk2::platform::gpr_map
{

namespace unpack
{
inline constexpr std::uint32_t scratch0             = ckernel::p_gpr_unpack::TMP0;        ///< scratch
inline constexpr std::uint32_t scratch1             = ckernel::p_gpr_unpack::TMP1;        ///< scratch
inline constexpr std::uint32_t unpacker0_tile_words = ckernel::p_gpr_unpack::TILE_SIZE_A; ///< unpacker 0's L1 address step per tile
inline constexpr std::uint32_t unpacker1_tile_words = ckernel::p_gpr_unpack::TILE_SIZE_B; ///< unpacker 1's L1 address step per tile
} // namespace unpack

namespace pack
{
inline constexpr std::uint32_t scratch0       = ckernel::p_gpr_pack::TMP0;        ///< scratch
inline constexpr std::uint32_t output_address = ckernel::p_gpr_pack::OUTPUT_ADDR; ///< the L1 address the packer writes to
} // namespace pack

} // namespace llk2::platform::gpr_map
