// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Perf twin of the api header api/compute/experimental/sdpa_weighted_reduce.h as the DSA indexer runs it: per iteration
// NUM_CHUNKS chunks into DEST slots 0 to NUM_CHUNKS - 1 of one section, then a two-PACR row pack per chunk. BLOCK false
// unpacks every chunk with its own context transaction (weighted_reduce), BLOCK true all of them with one
// (weighted_reduce_block). The unit is one chunk.

#include <cstdint>

#include "ckernel.h"
#include "ckernel_defs.h"
#include "llk_defs.h"
#include "params.h"
#include "perf.h"

#include "counters.h"
#include "profiler.h"

using namespace ckernel;

// Globals
std::uint32_t unp_cfg_context          = 0;
std::uint32_t pack_sync_tile_dst_ptr   = 0;
std::uint32_t math_sync_tile_dst_index = 0;

constexpr std::uint32_t QK_FACE_R_DIM      = 16;
constexpr std::uint32_t WEIGHTS_FACE_R_DIM = 8;
constexpr std::uint32_t QK_NUM_FACES       = 2;
constexpr std::uint32_t QK_TILE_WORDS      = QK_NUM_FACES * QK_FACE_R_DIM * FACE_C_DIM * 2 / 16;

#ifdef LLK_TRISC_UNPACK

#include "experimental/llk_unpack_AB_sdpa_weighted_reduce.h"
#include "llk_unpack_common.h"

// weighted_reduce_unpack_impl of the api header, with L1 addresses in place of CB reads.
inline void chunk_unpack(const std::uint32_t address_qk, const std::uint32_t address_weights)
{
    volatile std::uint32_t tt_reg_ptr* cfg = get_cfg_pointer();
    TTI_SETADCZW(0b011, 0, 0, 0, 0, 0b1111);
    wait_for_next_context(2);
    _llk_unpack_configure_addresses_(address_qk, address_weights, cfg);
    semaphore_post(semaphore::UNPACK_SYNC);
    TTI_STALLWAIT(p_stall::STALL_UNPACK, p_stall::TRISC_CFG);
    TTI_UNPACR(SrcA, 0b00010001, 0, 0, 0, 1, 0, p_unpacr::RAREFYB_DISABLE, 0, 0, 0, 0, 1);
    TTI_UNPACR(SrcB, 0b00000000, 0, 0, 0, 1, 1, p_unpacr::RAREFYB_DISABLE, 0, 0, 0, 0, 1);
    TTI_UNPACR(SrcA, 0b00000000, 0, 0, 0, 1, 1, p_unpacr::RAREFYB_DISABLE, 0, 0, 0, 0, 1);
    t6_semaphore_get(semaphore::UNPACK_SYNC);
    switch_config_context(unp_cfg_context);
    TTI_SETADCZW(0b011, 0, 0, 0, 0, 0b1111);
}

void run_kernel(RUNTIME_PARAMETERS params)
{
#if defined(RUNTIME_FORMATS) && !defined(SPEED_OF_LIGHT)
    const FormatConfig& formats = params.formats;
#endif
    const std::uint32_t LOOP_FACTOR = params.LOOP_FACTOR;
    {
        START_PERF_MEASURE("INIT")
        _llk_unpack_hw_configure_<is_fp32_dest_acc_en>(
            formats.unpack_B_src,
            formats.unpack_A_src,
            formats.unpack_B_dst,
            formats.unpack_A_dst,
            QK_FACE_R_DIM,
            WEIGHTS_FACE_R_DIM,
            QK_NUM_FACES,
            2,
            params.TILE_SIZE_UNPACK_B,
            params.TILE_SIZE_UNPACK_A);
        // weighted_reduce_init_short: no haloize, unpacker 0 reads one 16x16 face per UNPACR; unpacker 1 keeps the
        // indexer's 8-row Q faces.
        cfg_reg_rmw_tensix<THCON_SEC0_REG2_Haloize_mode_RMW>(0);
        TTI_SETADCXX(p_setadc::UNP_A, QK_FACE_R_DIM * FACE_C_DIM - 1, 0x0);
        TTI_SETADCXX(p_setadc::UNP_B, WEIGHTS_FACE_R_DIM * FACE_C_DIM - 1, 0x0);
        PROFILER_SYNC();
    }
    {
        START_PERF_MEASURE("TILE_LOOP")
        if constexpr (PERF_RUN_TYPE == PerfRunType::PACK_ISOLATE)
        {
        }
        else if constexpr (PERF_RUN_TYPE == PerfRunType::MATH_ISOLATE)
        {
            for (std::uint32_t loop = 0; loop < LOOP_FACTOR; ++loop)
            {
                for (std::uint32_t c = 0; c < NUM_CHUNKS; ++c)
                {
                    _perf_unpack_set_valid(ckernel::SrcB);
                    _perf_unpack_set_valid(ckernel::SrcA);
                }
            }
        }
        else
        {
            const std::uint32_t address_qk      = L1_ADDRESS(params.buffer_B[0]);
            const std::uint32_t address_weights = L1_ADDRESS(params.buffer_A[0]);
            for (std::uint32_t loop = 0; loop < LOOP_FACTOR; ++loop)
            {
                if constexpr (BLOCK)
                {
                    _llk_unpack_AB_sdpa_weighted_reduce_block_(address_qk, address_weights, QK_NUM_FACES, NUM_CHUNKS);
                }
                else
                {
                    for (std::uint32_t c = 0; c < NUM_CHUNKS; ++c)
                    {
                        chunk_unpack(address_qk + c * QK_TILE_WORDS, address_weights);
                    }
                }
            }
        }
        PROFILER_SYNC();
    }
}

#endif

#ifdef LLK_TRISC_MATH

#include "experimental/llk_math_matmul_custom_no_mop.h"
#include "llk_math_common.h"

// weighted_reduce_addrmod_init and weighted_reduce_math_impl of the api header (ADDR_MOD_3 stands in for
// sdpa_custom_mm_init's).
inline void weighted_reduce_addrmod_init_math()
{
    addr_mod_t {
        .srca = {.incr = 16, .clr = 0, .cr = 0},
        .srcb = {.incr = 0, .clr = 0, .cr = 0},
        .dest = {.incr = 8, .clr = 0, .cr = 0},
    }
        .set(ADDR_MOD_6);
    addr_mod_t {
        .srca = {.incr = 0, .clr = 0, .cr = 0},
        .srcb = {.incr = 0, .clr = 0, .cr = 0},
        .dest = {.incr = 0, .clr = 0, .cr = 0},
    }
        .set(ADDR_MOD_3);
}

inline void weighted_reduce_math_impl(const std::uint32_t dst_slot)
{
    constexpr std::uint32_t weighted_dest_slot_rows = 16;
    math::reset_counters(p_setrwc::SET_ABD_F);
    TT_SETC16(DEST_TARGET_REG_CFG_MATH_Offset_ADDR32, get_dest_buffer_base() + dst_slot * weighted_dest_slot_rows);
    TTI_MVMUL(p_setrwc::CLR_NONE, 0, ADDR_MOD_6, 0);
    TTI_MVMUL(p_setrwc::CLR_AB, 0, ADDR_MOD_3, 0);
}

void run_kernel(RUNTIME_PARAMETERS params)
{
#if defined(RUNTIME_FORMATS) && !defined(SPEED_OF_LIGHT)
    const FormatConfig& formats = params.formats;
#endif
    const std::uint32_t LOOP_FACTOR = params.LOOP_FACTOR;
    {
        START_PERF_MEASURE("INIT")
        _llk_math_matmul_init_no_mop_<ckernel::MathFidelity::LoFi, 0>(TILE_R_DIM, TILE_C_DIM, TILE_R_DIM, TILE_C_DIM, false, 0, 1, 1);
        _llk_math_pack_sync_init_<DstSync::SyncHalf, is_fp32_dest_acc_en>();
        _llk_math_hw_configure_<is_fp32_dest_acc_en>(formats.math, formats.math);
        weighted_reduce_addrmod_init_math();
        PROFILER_SYNC();
    }
    {
        START_PERF_MEASURE("TILE_LOOP")
        if constexpr (PERF_RUN_TYPE == PerfRunType::PACK_ISOLATE)
        {
        }
        else if constexpr (PERF_RUN_TYPE == PerfRunType::UNPACK_ISOLATE)
        {
            for (std::uint32_t loop = 0; loop < LOOP_FACTOR; ++loop)
            {
                for (std::uint32_t c = 0; c < NUM_CHUNKS; ++c)
                {
                    _perf_math_clear_valid(ckernel::SrcA);
                    _perf_math_clear_valid(ckernel::SrcB);
                }
            }
        }
        else
        {
            for (std::uint32_t loop = 0; loop < LOOP_FACTOR; ++loop)
            {
                if constexpr (PERF_RUN_TYPE != PerfRunType::MATH_ISOLATE)
                {
                    _llk_math_wait_for_dest_available_<DstSync::SyncHalf>();
                }
                for (std::uint32_t c = 0; c < NUM_CHUNKS; ++c)
                {
                    weighted_reduce_math_impl(c);
                }
                if constexpr (PERF_RUN_TYPE != PerfRunType::MATH_ISOLATE)
                {
                    _llk_math_dest_section_done_<DstSync::SyncHalf, is_fp32_dest_acc_en>();
                }
            }
        }
        PROFILER_SYNC();
    }
    _llk_math_matmul_uninit_no_mop_();
}

#endif

#ifdef LLK_TRISC_PACK

#include "llk_lib_pack_wrappers.h"
#include "llk_pack_common.h"

using namespace ckernel::packer;

// weighted_reduce_pack_impl of the api header: DEST slot row 0 (one row of each of two faces) as two raw PACRs into one
// 1x32 row of the output tile.
inline void row_pack(const std::uint32_t dst_slot, const std::uint32_t row_address)
{
    set_dst_write_addr(dst_slot);
    program_packer_destination(row_address);
    TTI_PACR(
        p_pacr::CFG_CTXT_0,
        p_pacr::NO_ROW_PAD_ZERO,
        p_pacr::DST_ACCESS_NORMAL_MODE,
        ADDR_MOD_3,
        p_pacr::ADDR_CNT_CTXT_0,
        p_pacr::P_ZERO_OUTPUT_DISABLED,
        p_pacr::SINGLE_INTF_ACTIVE,
        0,
        0,
        0,
        0,
        0);
    TTI_PACR(
        p_pacr::CFG_CTXT_0,
        p_pacr::NO_ROW_PAD_ZERO,
        p_pacr::DST_ACCESS_NORMAL_MODE,
        ADDR_MOD_1,
        p_pacr::ADDR_CNT_CTXT_0,
        p_pacr::P_ZERO_OUTPUT_DISABLED,
        p_pacr::SINGLE_INTF_ACTIVE,
        0,
        0,
        0,
        0,
        1);
    TTI_SETADCZW(p_setadc::PAC, 0, 0, 0, 0, 0b0101);
}

// weighted_reduce_pack_block: one DEST base and one destination for the section's rows, two PACRs per chunk
inline void block_pack(const std::uint32_t address)
{
    set_dst_write_addr(0);
    program_packer_destination(address);
    for (std::uint32_t i = 0; i < 2 * NUM_CHUNKS - 1; i++)
    {
        TTI_PACR(
            p_pacr::CFG_CTXT_0,
            p_pacr::NO_ROW_PAD_ZERO,
            p_pacr::DST_ACCESS_NORMAL_MODE,
            ADDR_MOD_3,
            p_pacr::ADDR_CNT_CTXT_0,
            p_pacr::P_ZERO_OUTPUT_DISABLED,
            p_pacr::SINGLE_INTF_ACTIVE,
            0,
            0,
            0,
            0,
            0);
    }
    TTI_PACR(
        p_pacr::CFG_CTXT_0,
        p_pacr::NO_ROW_PAD_ZERO,
        p_pacr::DST_ACCESS_NORMAL_MODE,
        ADDR_MOD_1,
        p_pacr::ADDR_CNT_CTXT_0,
        p_pacr::P_ZERO_OUTPUT_DISABLED,
        p_pacr::SINGLE_INTF_ACTIVE,
        0,
        0,
        0,
        0,
        1);
    TTI_SETADCZW(p_setadc::PAC, 0, 0, 0, 0, 0b0101);
}

inline void pack_chunks(RUNTIME_PARAMETERS params)
{
    constexpr std::uint32_t row_1x32_size_words = 4;
    if constexpr (BLOCK_PACK)
    {
        block_pack(L1_ADDRESS(params.buffer_Res[0]));
        return;
    }
    for (std::uint32_t c = 0; c < NUM_CHUNKS; ++c)
    {
        row_pack(c, L1_ADDRESS(params.buffer_Res[0]) + c * row_1x32_size_words);
    }
}

void run_kernel(RUNTIME_PARAMETERS params)
{
#if defined(RUNTIME_FORMATS) && !defined(SPEED_OF_LIGHT)
    const FormatConfig& formats = params.formats;
#endif
    const std::uint32_t LOOP_FACTOR = params.LOOP_FACTOR;
    {
        START_PERF_MEASURE("INIT")
        // The indexer's [8, 32] output layout: two 8-row faces, so DEST slots sit 16 rows apart.
        _llk_pack_hw_configure_wrapper_<is_fp32_dest_acc_en, PackMode::Default>(
            formats.pack_src, formats.pack_dst, params.TILE_SIZE_PACK, WEIGHTS_FACE_R_DIM, TILE_C_DIM, 2, true);
        _llk_pack_init_<PackMode::Default, false, false, false>(formats.pack_src, WEIGHTS_FACE_R_DIM, TILE_C_DIM, 2, 1, false);
        _llk_pack_dest_init_<DstSync::SyncHalf, is_fp32_dest_acc_en>();
        addr_mod_pack_t {
            .y_src = {.incr = 0, .clr = 0, .cr = 0},
            .y_dst = {.incr = 1, .clr = 0, .cr = 0},
            .z_src = {.incr = 1, .clr = 0},
            .z_dst = {.incr = 0, .clr = 0},
        }
            .set(ADDR_MOD_3);
        // The DEST-read strides sdpa_custom_mm's pack init leaves for the row packs: faces 8 rows apart, slots 16
        cfg_reg_rmw_tensix<PCK0_ADDR_CTRL_ZW_REG_0_Zstride_RMW>(FACE_C_DIM * 8 * 2);
        cfg_reg_rmw_tensix<PCK0_ADDR_CTRL_ZW_REG_0_Wstride_RMW>((TILE_NUM_FACES / 2) * FACE_C_DIM * 8 * 2);
        PROFILER_SYNC();
    }
    {
        START_PERF_MEASURE("TILE_LOOP")
        if constexpr (PERF_RUN_TYPE == PerfRunType::UNPACK_ISOLATE || PERF_RUN_TYPE == PerfRunType::MATH_ISOLATE)
        {
        }
        else if constexpr (PERF_RUN_TYPE == PerfRunType::PACK_ISOLATE)
        {
            for (std::uint32_t loop = 0; loop < LOOP_FACTOR; ++loop)
            {
                pack_chunks(params);
            }
        }
        else
        {
            for (std::uint32_t loop = 0; loop < LOOP_FACTOR; ++loop)
            {
                _llk_packer_wait_for_math_done_();
                pack_chunks(params);
                _llk_pack_dest_section_done_<DstSync::SyncHalf, is_fp32_dest_acc_en>();
            }
        }
        PROFILER_SYNC();
    }
}

#endif
