// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

// These wrappers are intended exclusively for LLK tests and are only available
// when the LLK infrastructure is enabled.
#ifdef ENV_LLK_INFRA

#include <cstdint>

#include "llk_unpack_tilize.h"

#ifdef ARCH_WORMHOLE

inline void _llk_unpack_tilize_init_wrapper_(
    const std::uint32_t unpack_src_format   = 0,
    const std::uint32_t unpack_dst_format   = 0,
    const std::uint32_t ct_dim              = 0,
    const ckernel::TensorShape tensor_shape = ckernel::DEFAULT_TENSOR_SHAPE)
{
    _llk_unpack_tilize_init_(unpack_src_format, unpack_dst_format, ct_dim, tensor_shape);
}

inline std::uint32_t _llk_unpack_tilize_block_ct_dim_wrapper_(const std::uint32_t block_ct_dim)
{
    // Wormhole uses block_ct_dim to compute tilize source offsets.
    return block_ct_dim;
}

inline std::uint32_t _llk_unpack_tilize_num_dvalids_wrapper_(const std::uint32_t tile_count, const std::uint32_t tile_num_faces)
{
    // Wormhole tracks dvalids per tile face.
    return tile_count * tile_num_faces;
}

inline void _llk_unpack_tilize_wrapper_(
    const std::uint32_t base_address,
    const std::uint32_t tile_index,
    const std::uint32_t unpack_src_format   = 0,
    const std::uint32_t unpack_dst_format   = 0,
    const std::uint32_t block_ct_dim        = 0,
    const ckernel::TensorShape tensor_shape = ckernel::DEFAULT_TENSOR_SHAPE)
{
    _llk_unpack_tilize_(base_address, tile_index, unpack_src_format, unpack_dst_format, block_ct_dim, tensor_shape);
}

template <bool neginf_srcA = false, std::uint32_t reload_srcB = false, bool zero_srcA = false, bool zero_srcA_reduce = false>
inline void _llk_unpack_tilizeA_B_init_wrapper_(
    const std::uint32_t unpack_src_format,
    const std::uint32_t unpack_dst_format,
    const std::uint32_t ct_dim,
    const ckernel::TensorShape tensor_shape = ckernel::DEFAULT_TENSOR_SHAPE)
{
    // The tests give operand B the same tile shape as operand A.
    _llk_unpack_tilizeA_B_init_<neginf_srcA, reload_srcB, zero_srcA, zero_srcA_reduce>(
        unpack_src_format, unpack_dst_format, ct_dim, tensor_shape, tensor_shape);
}

// Wormhole bakes neginf_srcA / reload_srcB / zero_srcA_reduce into the MOP at init; only zero_srcA is an execute template.
template <bool /*neginf_srcA*/ = false, std::uint32_t /*reload_srcB*/ = false, bool zero_srcA = false, bool /*zero_srcA_reduce*/ = false>
inline void _llk_unpack_tilizeA_B_wrapper_(
    const std::uint32_t unpack_src_format,
    const std::uint32_t base_address_a,
    const std::uint32_t address_b,
    const std::uint32_t tile_index_a,
    const std::uint32_t block_ct_dim,
    const ckernel::TensorShape tensor_shape = ckernel::DEFAULT_TENSOR_SHAPE)
{
    _llk_unpack_tilizeA_B_<zero_srcA>(unpack_src_format, base_address_a, address_b, tile_index_a, block_ct_dim, tensor_shape);
}

#elif defined(ARCH_BLACKHOLE)

inline void _llk_unpack_tilize_init_wrapper_(
    const std::uint32_t unpack_src_format   = 0,
    const std::uint32_t unpack_dst_format   = 0,
    const std::uint32_t ct_dim              = 0,
    const ckernel::TensorShape tensor_shape = ckernel::DEFAULT_TENSOR_SHAPE)
{
    _llk_unpack_tilize_init_(unpack_src_format, unpack_dst_format, ct_dim, tensor_shape);
}

inline std::uint32_t _llk_unpack_tilize_block_ct_dim_wrapper_([[maybe_unused]] const std::uint32_t block_ct_dim)
{
    // Blackhole unpack_tilize does not use block_ct_dim and expects zero.
    return 0;
}

inline std::uint32_t _llk_unpack_tilize_num_dvalids_wrapper_(const std::uint32_t tile_count, [[maybe_unused]] const std::uint32_t tile_num_faces)
{
    // Blackhole unpack_tilize emits 1 DVALID per tile: 8-bit via inline path (SetDvalid on
    // last face of active pair), non-8-bit via the whole-tile BH workaround.
    return tile_count;
}

inline void _llk_unpack_tilize_wrapper_(
    const std::uint32_t base_address,
    const std::uint32_t tile_index,
    const std::uint32_t unpack_src_format             = 0,
    const std::uint32_t unpack_dst_format             = 0,
    [[maybe_unused]] const std::uint32_t block_ct_dim = 0,
    const ckernel::TensorShape tensor_shape           = ckernel::DEFAULT_TENSOR_SHAPE)
{
    _llk_unpack_tilize_(base_address, tile_index, unpack_src_format, unpack_dst_format, tensor_shape);
}

template <bool neginf_srcA = false, std::uint32_t reload_srcB = false, bool zero_srcA = false, bool zero_srcA_reduce = false>
inline void _llk_unpack_tilizeA_B_init_wrapper_(
    const std::uint32_t unpack_src_format,
    const std::uint32_t unpack_dst_format,
    const std::uint32_t ct_dim,
    const ckernel::TensorShape tensor_shape = ckernel::DEFAULT_TENSOR_SHAPE)
{
    // The tests give operand B the same tile shape as operand A.
    _llk_unpack_tilizeA_B_init_<neginf_srcA, reload_srcB, zero_srcA, zero_srcA_reduce>(
        unpack_src_format, unpack_dst_format, ct_dim, tensor_shape, tensor_shape);
}

template <bool neginf_srcA = false, std::uint32_t reload_srcB = false, bool zero_srcA = false, bool zero_srcA_reduce = false>
inline void _llk_unpack_tilizeA_B_wrapper_(
    const std::uint32_t unpack_src_format,
    const std::uint32_t base_address_a,
    const std::uint32_t address_b,
    const std::uint32_t tile_index_a,
    const std::uint32_t block_ct_dim,
    const ckernel::TensorShape tensor_shape = ckernel::DEFAULT_TENSOR_SHAPE)
{
    _llk_unpack_tilizeA_B_<neginf_srcA, reload_srcB, zero_srcA, zero_srcA_reduce>(
        unpack_src_format, base_address_a, address_b, tile_index_a, block_ct_dim, tensor_shape);
}

#else
#error "Unsupported architecture for LLK unpack tilize wrappers"
#endif

inline void _llk_unpack_tilize_uninit_wrapper_(const std::uint32_t unpack_dst_format, const ckernel::TensorShape tensor_shape = ckernel::DEFAULT_TENSOR_SHAPE)
{
    _llk_unpack_tilize_uninit_(unpack_dst_format, tensor_shape);
}

inline void _llk_unpack_tilizeA_B_uninit_wrapper_(
    const std::uint32_t unpack_dst_format, const ckernel::TensorShape tensor_shape = ckernel::DEFAULT_TENSOR_SHAPE)
{
    _llk_unpack_tilizeA_B_uninit_(unpack_dst_format, tensor_shape);
}

// Isolate handshake count shared by unpack and math. The two TRISC threads deadlock
// if they disagree. Tilize posts architecture-specific dvalids (BH: 1/tile, WH: 1/face);
// unpack_A posts one per face.
inline std::uint32_t _perf_src_handshake_iters_(const bool tilize_en, const std::uint32_t num_tiles, const std::uint32_t num_faces)
{
    return tilize_en ? _llk_unpack_tilize_num_dvalids_wrapper_(num_tiles, num_faces) : num_tiles * num_faces;
}

#endif // ENV_LLK_INFRA
