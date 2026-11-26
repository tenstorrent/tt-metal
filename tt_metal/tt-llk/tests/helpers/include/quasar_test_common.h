// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "ckernel_trisc_common.h"
#include "llk_math_common.h"
#include "perf.h"
#include "tensor_shape.h"

/**
 * @brief Tiles per Dest section when a run of tile_cnt 32x32 tiles is split into full sections.
 *
 * @tparam DST_SYNC: Destination synchronization mode, values = <SyncHalf/SyncFull>
 * @tparam EN_32BIT_DEST: Whether the Dest register is in 32-bit mode
 * @param tile_cnt: Tiles in the run; a run longer than one section must be a multiple of it.
 */
template <ckernel::DstSync DST_SYNC, bool EN_32BIT_DEST>
inline std::uint32_t tiles_per_dest_section(const std::uint32_t tile_cnt)
{
    constexpr std::uint32_t section_tiles = ckernel::trisc::get_dest_max_tiles<DST_SYNC, EN_32BIT_DEST, ckernel::trisc::DstTileShape::Tile32x32>();
    return tile_cnt < section_tiles ? tile_cnt : section_tiles;
}

/**
 * @brief Dest sections in a run of tile_cnt tiles split into sections of section_tiles.
 *
 * @param tile_cnt: Tiles in the run; must be a whole number of sections.
 * @param section_tiles: Tiles per section, e.g. from @ref tiles_per_dest_section.
 */
inline std::uint32_t dest_section_count(const std::uint32_t tile_cnt, const std::uint32_t section_tiles)
{
    LLK_ASSERT(section_tiles == 0 || tile_cnt % section_tiles == 0, "A run longer than one Dest section must be a whole number of sections");
    return section_tiles == 0 ? 0 : tile_cnt / section_tiles;
}

/**
 * @brief Matmul output rows per Dest section: the tallest divisor of rt_dim whose rows x ct_dim fit one section.
 *
 * @tparam DST_SYNC: Destination synchronization mode, values = <SyncHalf/SyncFull>
 * @tparam EN_32BIT_DEST: Whether the Dest register is in 32-bit mode
 * @param rt_dim: Output row tiles.
 * @param ct_dim: Output column tiles; one output row must fit a Dest section.
 */
template <ckernel::DstSync DST_SYNC, bool EN_32BIT_DEST>
inline std::uint32_t matmul_rows_per_dest_section(const std::uint32_t rt_dim, const std::uint32_t ct_dim)
{
    constexpr std::uint32_t section_tiles = ckernel::trisc::get_dest_max_tiles<DST_SYNC, EN_32BIT_DEST, ckernel::trisc::DstTileShape::Tile32x32>();
    LLK_ASSERT(ct_dim <= section_tiles, "One matmul output row must fit a Dest section");
    std::uint32_t rows = ct_dim == 0 ? rt_dim : section_tiles / ct_dim;
    rows               = rows < rt_dim ? rows : rt_dim;
    while (rows > 1 && rt_dim % rows != 0)
    {
        rows--;
    }
    return rows;
}

inline void perf_unpack_set_srcb_once_then_srca_per_face(std::uint32_t srcb_once_count, std::uint32_t srca_per_face_count)
{
    _perf_unpack_loop_set_valid<false /*set_a*/, true /*set_b*/>(srcb_once_count);
    _perf_unpack_loop_set_valid<true /*set_a*/, false /*set_b*/>(srca_per_face_count);
}

inline void perf_math_clear_srca_per_face_then_srcb_once(std::uint32_t srca_per_face_count, std::uint32_t srcb_once_count)
{
    _perf_math_loop_clear_valid<true /*clear_a*/, false /*clear_b*/>(srca_per_face_count);
    _perf_math_loop_clear_valid<false /*clear_a*/, true /*clear_b*/>(srcb_once_count);
}

template <ckernel::dest_dvalid_client thread>
inline void set_up_fpu_to_pack_dest_dvalid_chain()
{
    ckernel::set_up_dest_dvalid_per_thread<thread>({ckernel::dest_dvalid_client::FPU, ckernel::dest_dvalid_client::PACK});
}

template <ckernel::dest_dvalid_client thread>
inline void set_up_unpack_to_pack_dest_dvalid_chain()
{
    ckernel::set_up_dest_dvalid_per_thread<thread>({ckernel::dest_dvalid_client::UNPACK, ckernel::dest_dvalid_client::PACK});
}

template <ckernel::dest_dvalid_client thread>
inline void set_up_unpack_to_fpu_to_pack_dest_dvalid_chain()
{
    ckernel::set_up_dest_dvalid_per_thread<thread>({ckernel::dest_dvalid_client::UNPACK, ckernel::dest_dvalid_client::FPU, ckernel::dest_dvalid_client::PACK});
}

template <ckernel::dest_dvalid_client thread>
inline void set_up_fpu_to_sfpu_to_pack_dest_dvalid_chain()
{
    ckernel::set_up_dest_dvalid_per_thread<thread>({ckernel::dest_dvalid_client::FPU, ckernel::dest_dvalid_client::SFPU, ckernel::dest_dvalid_client::PACK});
}

template <ckernel::dest_dvalid_client thread>
inline void set_up_unpack_to_sfpu_to_pack_dest_dvalid_chain()
{
    ckernel::set_up_dest_dvalid_per_thread<thread>({ckernel::dest_dvalid_client::UNPACK, ckernel::dest_dvalid_client::SFPU, ckernel::dest_dvalid_client::PACK});
}

inline void clear_dest_dvalid_wait_mask(std::uint32_t wait_mask_addr)
{
    volatile std::uint32_t* cfg = reinterpret_cast<volatile std::uint32_t*>(TENSIX_CFG_BASE);
    cfg[wait_mask_addr]         = 0;
}

/**
 * @brief Disables the DEST DVALID handshake for UNPACK.
 */
inline void set_up_zero_dest_dvalid_handshake_for_unpack()
{
    clear_dest_dvalid_wait_mask(UNPACK_TO_DEST_DVALID_CTRL_wait_mask_ADDR32);
}

/**
 * @brief Disables the DEST DVALID handshake for MATH.
 */
inline void set_up_zero_dest_dvalid_handshake_for_math()
{
    clear_dest_dvalid_wait_mask(MATH_DEST_DVALID_CTRL_wait_mask_ADDR32);
}

/**
 * @brief Disables the DEST DVALID handshake for SFPU.
 */
inline void set_up_zero_dest_dvalid_handshake_for_sfpu()
{
    clear_dest_dvalid_wait_mask(SFPU_DEST_DVALID_CTRL_wait_mask_ADDR32);
}

/**
 * @brief Disables the DEST DVALID handshake for PACK.
 */
inline void set_up_zero_dest_dvalid_handshake_for_pack()
{
    clear_dest_dvalid_wait_mask(PACK_DEST_DVALID_CTRL_wait_mask_ADDR32);
}

/**
 * @brief Populates TensorShape from explicit dimensions.
 */
inline ckernel::TensorShape tensor_shape_from_dimensions(
    std::uint32_t face_r_dim, std::uint32_t face_c_dim, std::uint32_t num_faces_r_dim, std::uint32_t num_faces_c_dim)
{
    return {
        static_cast<std::uint8_t>(face_r_dim),
        static_cast<std::uint8_t>(face_c_dim),
        static_cast<std::uint8_t>(num_faces_r_dim),
        static_cast<std::uint8_t>(num_faces_c_dim),
    };
}

#ifdef SPEED_OF_LIGHT
#define TENSOR_SHAPE_FROM_PARAMS(params) tensor_shape_from_dimensions(TEST_FACE_R_DIM, TEST_FACE_C_DIM, num_faces_r_dim_A, num_faces_c_dim_A)
#else
/**
 * @brief Populates TensorShape struct args from runtime test parameters.
 *
 * @param params: Runtime parameters passed through pytest.
 */
template <typename Params>
inline ckernel::TensorShape tensor_shape_from_params(const Params& params)
{
    return tensor_shape_from_dimensions(params.TEST_FACE_R_DIM, params.TEST_FACE_C_DIM, params.num_faces_r_dim_A, params.num_faces_c_dim_A);
}

#define TENSOR_SHAPE_FROM_PARAMS(params) tensor_shape_from_params(params)
#endif
