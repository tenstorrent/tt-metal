// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

// The block loop of the pack sweep kernels through pack_api_shim.h (include it first): production's pack_block, pack init
// and reconfigure per arm. pack_sweep_state 0 steady (one output), 1 a pack init before every block, 2 outputs A and B
// alternating with a reconfigure before each block. Arms: 'T', 'B', 'W' (pack_block's arm forms), 0 the form tables, 'M'
// main's per-tile loop; on main's LLK every arm is main's loop.

#if defined(ARCH_BLACKHOLE) && defined(LLK_TRISC_PACK)

#include <cstdint>

#include "perf.h"

namespace pack_api_sweep
{

// The output page of a tile of face_r rows per face, in bytes.
inline std::uint32_t page_bytes(const std::uint32_t fmt, const std::uint32_t r, const std::uint32_t faces)
{
    switch (static_cast<DataFormat>(fmt))
    {
        case DataFormat::Float32:
        case DataFormat::Int32:
        case DataFormat::UInt32:
        case DataFormat::Tf32:
            return faces * r * 16 * 4;
        case DataFormat::Int8:
        case DataFormat::UInt8:
        case DataFormat::Lf8:
        case DataFormat::Fp8_e4m3:
            return faces * r * 16;
        case DataFormat::Bfp8_b:
        case DataFormat::Bfp8:
            return faces * (r * 16 + 16);
        case DataFormat::Bfp4_b:
        case DataFormat::Bfp4:
            return faces * (r * 8 + 16);
        case DataFormat::Bfp2_b:
        case DataFormat::Bfp2:
            return faces * (r * 4 + 16);
        default:
            return faces * r * 16 * 2;
    }
}

// A block's DEST start: PACK_API_DST, a compile-time constant as production's constant pack_block start, else the run-time index.
#if defined(PACK_API_DST)
inline constexpr std::uint32_t dest_start(const int)
{
    return PACK_API_DST;
}
#else
inline std::uint32_t dest_start(const int dst)
{
    return static_cast<std::uint32_t>(dst);
}
#endif

// The pack thread's context marks: matmul_block_init's PACK_MATH_CONTEXT(1), and behind it a pack-thread SFPU init's
// PACK_SFPU_CONTEXT(). Main's LLK has no pack contexts.
inline void mark_matmul()
{
#if PACK_API_SHIM_PR
    ckernel::packer::llk_pack_math_context(1);
#endif
}

inline void mark_sfpu()
{
#if PACK_API_SHIM_PR
    ckernel::packer::llk_pack_sfpu_context();
#endif
}

template <char arm, PackMode pack_mode = PackMode::Default>
inline void init(const std::uint32_t out)
{
    if constexpr (arm == 'M')
    {
        pack_api_shim::main_pack_init<pack_mode>(out);
    }
    else
    {
        pack_api_shim::pack_init<pack_mode>(out);
    }
}

template <std::uint32_t ntiles, bool is_fp32_dest_acc_en, std::uint32_t kt, char arm, std::uint32_t out>
__attribute__((always_inline)) inline void pack(const std::uint32_t dst)
{
#if PACK_API_SHIM_PR
    if constexpr (arm != 'M')
    {
        pack_api_shim::pack_block<ntiles, is_fp32_dest_acc_en, kt, arm>(dst, out);
    }
    else
#endif
    {
        pack_api_shim::main_pack_block<is_fp32_dest_acc_en>(dst, out, ntiles);
    }
}

// One block, as a compute kernel packs it: wait for math, the state's reconfigure or init, the kernel's pack-thread work,
// pack_block, release DEST, cb_push_back.
template <std::uint32_t ntiles, bool is_fp32_dest_acc_en, std::uint32_t kt, std::uint32_t state, char arm, std::uint32_t out, class BlockInit, class Tail>
__attribute__((always_inline)) inline void block(const std::uint32_t dst, BlockInit& block_init, Tail& tail)
{
    if constexpr (PERF_RUN_TYPE != PerfRunType::PACK_ISOLATE)
    {
        _llk_packer_wait_for_math_done_();
    }
    if constexpr (state == 2)
    {
        pack_api_shim::pack_reconfig_data_format<is_fp32_dest_acc_en>(out);
    }
    if constexpr (state == 1)
    {
        block_init();
    }
    tail();
    pack<ntiles, is_fp32_dest_acc_en, kt, arm, out>(dst);
    if constexpr (PERF_RUN_TYPE != PerfRunType::PACK_ISOLATE)
    {
        _llk_pack_dest_section_done_<DST_SYNC_MODE, is_fp32_dest_acc_en>();
    }
    pack_api_shim::cb_push_back(out, ntiles);
}

// The TILE_LOOP of a sweep arm; state 2 packs blocks in pairs, A then B, so every pack_block has a constant output.
template <
    std::uint32_t ntiles,
    bool is_fp32_dest_acc_en,
    std::uint32_t kt,
    std::uint32_t state,
    char arm,
    char arm_b,
    std::uint32_t out_a,
    std::uint32_t out_b,
    class BlockInit,
    class Tail>
inline void run(const std::uint32_t loops, const std::uint32_t blocks, const std::uint32_t dst, BlockInit block_init, Tail tail)
{
    if constexpr (PERF_RUN_TYPE == PerfRunType::UNPACK_ISOLATE || PERF_RUN_TYPE == PerfRunType::MATH_ISOLATE)
    {
        return;
    }
    for (std::uint32_t loop = 0; loop < loops; ++loop)
    {
        for (std::uint32_t b = 0; b < blocks; b += (state == 2 ? 2 : 1))
        {
            block<ntiles, is_fp32_dest_acc_en, kt, state, arm, out_a>(dst, block_init, tail);
            if constexpr (state == 2)
            {
                block<ntiles, is_fp32_dest_acc_en, kt, state, arm_b, out_b>(dst, block_init, tail);
            }
        }
    }
}

// Binds output A over the result buffer from base (16 byte words) and, for state 2, output B behind it at the next 1 KB;
// cbd: A's CB depth in blocks, 0 every block A packs in a run. Returns B's base.
template <std::uint32_t ntiles, std::uint32_t state, std::uint32_t cbd, std::uint32_t out_a, std::uint32_t out_b>
inline std::uint32_t bind(const std::uint32_t base, const std::uint32_t blocks, const std::uint32_t page_a, const std::uint32_t page_b)
{
    const std::uint32_t blocks_a = state == 2 ? (blocks + 1) / 2 : blocks;
    const std::uint32_t tiles_a  = cbd != 0 ? cbd * ntiles : blocks_a * ntiles;
    const std::uint32_t base_b   = base + (((blocks + 1) / 2 * ntiles * page_a + 1023) & ~1023u) / 16;
    pack_api_shim::cb_bind(out_a, base + 1, page_a / 16, tiles_a);
    if constexpr (state == 2)
    {
        pack_api_shim::cb_bind(out_b, base_b + 1, page_b / 16, blocks / 2 * ntiles);
    }
    return base_b;
}

} // namespace pack_api_sweep

#endif
