// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
//
// FPU kernel that needs Dest: Out[block] = Partial[block] + A x B, one SyncHalf bank per block, 32-bit Dest.
// Each bank first gets the fp32 partial (buffer_S, CT tiles per bank), then TRISC1 accumulates KT matmul steps on top.
//   A (DWP_SRCS_WRITER = false): TRISC0 interleaves per bank: UNP_DEST the partial, then reprograms Unpacker 0 and the
//      matmul unpack MOP for the bf16 SrcA/SrcB stream. Dest chain UNPACK -> FPU -> PACK.
//   B (DWP_SRCS_WRITER = true):  TRISC3 copies the partial Unpacker 2 -> SrcS -> SFPU -> Dest while TRISC0 only feeds
//      SrcA/SrcB. Dest chain SFPU -> FPU -> PACK.
//   L1-acc (DWP_PACK_L1_ACC = true): no Dest preload. TRISC1 copies the partials into the output buffer at init and
//      PACK0 accumulates each bank's A x B onto them in L1. Dest chain FPU -> PACK.
// DWP_HW_BANK_SWITCH picks the partial writer's bank hand-off, as in dest_write_path_quasar_test.cpp.

#include <cstdint>

#include "ckernel.h"
#include "counters.h"
#include "dest_write_path_quasar.h"
#include "llk_defs.h"
#include "llk_memory_checks.h"
#include "params.h"
#include "perf.h"
#include "profiler.h"
#include "quasar_test_common.h"

// Globals
std::uint32_t unp_cfg_context          = 0;
std::uint32_t pack_sync_tile_dst_ptr   = 0;
std::uint32_t math_sync_tile_dst_index = 0;

constexpr std::uint32_t NUM_BANKS      = DWP_NUM_BLOCKS * DWP_LOOP_FACTOR;
constexpr std::uint32_t MM_CT_DIM      = DWP_TILES_PER_BANK;
constexpr std::uint32_t MM_RT_DIM      = 1;
constexpr std::uint32_t MM_KT_DIM      = DWP_KT_DIM;
constexpr DataFormat PARTIAL_FORMAT    = DataFormat::Float32;
constexpr std::uint8_t NO_BUF_DESC_ID  = 0;
constexpr bool MATMUL_UNPACK_TRANSPOSE = false;

static_assert(dest_sync == ckernel::DstSync::SyncHalf, "the partial writer and the FPU alternate the two SyncHalf Dest banks");
static_assert(is_fp32_dest_acc_en, "the partial and the accumulation live in a 32-bit Dest");
static_assert(MM_CT_DIM >= 1 && MM_CT_DIM <= 4, "a 32-bit SyncHalf Dest bank holds 1-4 output tiles");
static_assert(NUM_BANKS % 2 == 0, "an even bank count returns every Dest client to bank 0 for the next run");
static_assert(NUM_BANKS <= dwp::STAMP_MAX_BANKS, "per-bank stamps must fit one buffer_C tile");
static_assert(!DWP_PACK_L1_ACC || (!DWP_SRCS_WRITER && !DWP_HW_BANK_SWITCH), "L1-acc has no Dest preload to hand off");

// Dest is preloaded by TRISC0 (path A) or TRISC3 (path B); L1-acc leaves it to the FPU alone.
constexpr bool UNPACK_PRELOADS_DEST = !DWP_SRCS_WRITER && !DWP_PACK_L1_ACC;

// Path B issue options, as in dest_write_path_quasar_test.cpp.
static_assert(!(DWP_SRCS_ADDR_MOD_STEP || DWP_SRCS_BANK_UNPACK) || DWP_SRCS_WRITER, "SrcS issue options are path B only");
static_assert(!DWP_SRCS_MULTI_BANK_UNPACK || DWP_SRCS_BANK_UNPACK, "DWP_SRCS_MULTI_BANK_UNPACK extends DWP_SRCS_BANK_UNPACK");
constexpr std::uint32_t SRCS_BANKS_PER_UNPACK = DWP_SRCS_MULTI_BANK_UNPACK ? dwp::srcs_banks_per_unpack(MM_CT_DIM) : 1;
constexpr std::uint32_t SRCS_TILES_PER_UNPACK = DWP_SRCS_BANK_UNPACK ? MM_CT_DIM * SRCS_BANKS_PER_UNPACK : 1;
constexpr dwp::SrcsUnpack SRCS_UNPACK =
    DWP_SRCS_MULTI_BANK_UNPACK ? dwp::SrcsUnpack::Caller : (DWP_SRCS_BANK_UNPACK ? dwp::SrcsUnpack::PerBank : dwp::SrcsUnpack::PerTile);
static_assert(DWP_NUM_BLOCKS % SRCS_BANKS_PER_UNPACK == 0, "a multi-bank UNPACR2 must not wrap past the last L1 block");

template <ckernel::dest_dvalid_client CLIENT>
inline void set_up_dest_chain()
{
    if constexpr (DWP_PACK_L1_ACC)
    {
        set_up_fpu_to_pack_dest_dvalid_chain<CLIENT>();
    }
    else if constexpr (DWP_SRCS_WRITER)
    {
        ckernel::set_up_dest_dvalid_per_thread<CLIENT>(
            {ckernel::dest_dvalid_client::SFPU, ckernel::dest_dvalid_client::FPU, ckernel::dest_dvalid_client::PACK});
    }
    else
    {
        set_up_unpack_to_fpu_to_pack_dest_dvalid_chain<CLIENT>();
    }
}

#ifdef LLK_TRISC_UNPACK

#include "llk_bfd_alloc.h"
#include "llk_unpack_common.h"
#include "llk_unpack_matmul.h"

void run_kernel(RUNTIME_PARAMETERS params)
{
#if defined(RUNTIME_FORMATS) && !defined(SPEED_OF_LIGHT)
    const FormatConfig& formats = params.formats;
#endif
#ifndef SPEED_OF_LIGHT
    const Operand& buffer_A = params.buffer_A;
    const Operand& buffer_B = params.buffer_B;
    const Operand& buffer_S = params.buffer_S;
#endif
    const DataFormat srca_format = static_cast<DataFormat>(formats.unpack_B_dst);
    std::uint8_t bfd_a           = NO_BUF_DESC_ID;
    std::uint8_t bfd_b           = NO_BUF_DESC_ID;
    std::uint8_t bfd_partial     = NO_BUF_DESC_ID;
    {
        START_PERF_MEASURE("INIT")
        set_ttsync_enables<TRACK_ALL>(ckernel::TRISC_ID);
        // Matmul flips the unpacker roles: operand A drives UNPACR1/SrcB, operand B drives UNPACR0/SrcA.
        bfd_a = ckernel::trisc::bfd_alloc_and_program<ckernel::trisc::BfdResource::Unp1>(
            ckernel::DEFAULT_TENSOR_SHAPE, L1_ADDRESS(buffer_A[0]), formats.unpack_A_src);
        bfd_b = ckernel::trisc::bfd_alloc_and_program<ckernel::trisc::BfdResource::Unp0>(
            ckernel::DEFAULT_TENSOR_SHAPE, L1_ADDRESS(buffer_B[0]), formats.unpack_B_src);
        _llk_unpack_hw_configure_<ckernel::p_unpacr::UNP_B>(static_cast<DataFormat>(formats.unpack_A_dst));
        _llk_unpack_hw_configure_<ckernel::p_unpacr::UNP_A>(srca_format);
        if constexpr (UNPACK_PRELOADS_DEST)
        {
            set_up_dest_chain<ckernel::dest_dvalid_client::UNPACK>();
            bfd_partial = ckernel::trisc::bfd_alloc_and_program<ckernel::trisc::BfdResource::Unp0>(
                ckernel::DEFAULT_TENSOR_SHAPE, L1_ADDRESS(buffer_S[0]), static_cast<std::uint32_t>(PARTIAL_FORMAT));
            _llk_unpack_dest_init_();
        }
        _llk_unpack_matmul_init_<MATMUL_UNPACK_TRANSPOSE>(bfd_a, bfd_b, MM_CT_DIM, MM_RT_DIM, MM_KT_DIM);
        PROFILER_SYNC();
    }
    {
        START_PERF_MEASURE("TILE_LOOP")
        for (std::uint32_t bank = 0; bank < NUM_BANKS; ++bank)
        {
            if constexpr (UNPACK_PRELOADS_DEST)
            {
                // Unpacker 0 serves both the partial (to Dest, fp32) and SrcA (bf16): swap its format and MOP per bank.
                _llk_unpack_hw_configure_<ckernel::p_unpacr::UNP_DEST>(PARTIAL_FORMAT);
                dwp::program_u2d_bank_mop<DWP_HW_BANK_SWITCH>(bfd_partial, MM_CT_DIM);
                dwp::unpack_bank_to_dest<DWP_HW_BANK_SWITCH>((bank % DWP_NUM_BLOCKS) * MM_CT_DIM);
                _llk_unpack_hw_configure_<ckernel::p_unpacr::UNP_A>(srca_format);
                _llk_unpack_matmul_init_<MATMUL_UNPACK_TRANSPOSE>(bfd_a, bfd_b, MM_CT_DIM, MM_RT_DIM, MM_KT_DIM);
            }
            for (std::uint32_t k = 0; k < MM_KT_DIM; ++k)
            {
                _llk_unpack_matmul_(MM_CT_DIM, MM_RT_DIM, MM_KT_DIM, k, k * MM_CT_DIM);
            }
        }
        wait_unpack_idle();
        PROFILER_SYNC();
    }
}

#endif

#ifdef LLK_TRISC_MATH

#include "llk_math_common.h"
#include "llk_math_matmul.h"

// Copies the fp32 partials into the output tiles that PACK0 then accumulates onto. Reading every word back before the
// fence makes the stores land in L1 before this thread issues the first matmul, so ahead of the packer's first
// accumulating read. The copy runs here, not on TRISC2: the FPU heads the FPU -> PACK dvalid chain and is not held
// back by a packer that starts late, so a slow pack init lets the matmul overrun the two banks and the pack hangs.
inline void copy_partials_to_output(const Operand& partials, const Operand& output)
{
    constexpr std::uint32_t WORDS_PER_TILE = 32 * 32;
    for (std::uint32_t tile = 0; tile < DWP_NUM_BLOCKS * MM_CT_DIM; ++tile)
    {
        const volatile std::uint32_t* src = reinterpret_cast<const volatile std::uint32_t*>(partials[tile]);
        volatile std::uint32_t* dst       = reinterpret_cast<volatile std::uint32_t*>(output[tile]);
        for (std::uint32_t i = 0; i < WORDS_PER_TILE; ++i)
        {
            dst[i] = src[i];
        }
        for (std::uint32_t i = 0; i < WORDS_PER_TILE; ++i)
        {
            (void)dst[i];
        }
    }
    asm volatile("fence" ::: "memory");
}

void run_kernel(RUNTIME_PARAMETERS params)
{
#if defined(RUNTIME_FORMATS) && !defined(SPEED_OF_LIGHT)
    const FormatConfig& formats = params.formats;
#endif
#ifndef SPEED_OF_LIGHT
    const Operand& buffer_S   = params.buffer_S;
    const Operand& buffer_Res = params.buffer_Res;
#endif
    {
        START_PERF_MEASURE("INIT")
        if constexpr (DWP_PACK_L1_ACC)
        {
            copy_partials_to_output(buffer_S, buffer_Res);
        }
        else
        {
            (void)buffer_S;
            (void)buffer_Res;
        }
        set_up_dest_chain<ckernel::dest_dvalid_client::FPU>();
        if constexpr (DWP_PACK_L1_ACC)
        {
            // The matmul accumulates onto Dest and nothing preloads it, so both banks must start at zero. Pack's
            // section done re-zeroes each bank after that. ADDR_MOD_5 is reprogrammed by the matmul init below.
            ckernel::addr_mod_t {.srca = {.incr = 0}, .srcb = {.incr = 0}, .dest = {.incr = 0}}.set(ADDR_MOD_5);
            TTI_ZEROACC(p_zeroacc::CLR_ALL, is_fp32_dest_acc_en, 0, ADDR_MOD_5, 0);
        }
        const DataFormat math_format = static_cast<DataFormat>(formats.math);
        _llk_math_srcAB_hw_configure_<is_fp32_dest_acc_en>(math_format, math_format);
        _llk_math_matmul_init_<MATH_FIDELITY>(math_format, math_format, MM_CT_DIM, MM_RT_DIM);
        PROFILER_SYNC();
    }
    {
        START_PERF_MEASURE("TILE_LOOP")
        for (std::uint32_t bank = 0; bank < NUM_BANKS; ++bank)
        {
            for (std::uint32_t k = 0; k < MM_KT_DIM; ++k)
            {
                _llk_math_matmul_block_(MM_CT_DIM, MM_RT_DIM);
            }
            _llk_math_set_dvalid_<p_cleardvalid::FPU, dest_sync>();
        }
        wait_fpu_idle();
        PROFILER_SYNC();
    }
}

#endif

#ifdef LLK_TRISC_ISOLATE_SFPU

void run_kernel(RUNTIME_PARAMETERS params)
{
#ifndef SPEED_OF_LIGHT
    const Operand& buffer_S = params.buffer_S;
#endif
    std::uint8_t bfd_srcs = NO_BUF_DESC_ID;
    {
        START_PERF_MEASURE("INIT")
        // TRISC1 owns the shared ALU format config here; this thread only touches SrcS, the SFPU and its own Dest base.
        if constexpr (DWP_SRCS_WRITER)
        {
            set_up_dest_chain<ckernel::dest_dvalid_client::SFPU>();
            bfd_srcs = dwp::init_srcs_to_dest<SRCS_TILES_PER_UNPACK, DWP_SRCS_ADDR_MOD_STEP>(L1_ADDRESS(buffer_S[0]), PARTIAL_FORMAT, PARTIAL_FORMAT);
        }
        PROFILER_SYNC();
    }
    {
        START_PERF_MEASURE("TILE_LOOP")
        if constexpr (DWP_SRCS_WRITER)
        {
            for (std::uint32_t bank = 0; bank < NUM_BANKS; ++bank)
            {
                const std::uint32_t l1_tile_idx = (bank % DWP_NUM_BLOCKS) * MM_CT_DIM;
                if constexpr (SRCS_UNPACK == dwp::SrcsUnpack::Caller)
                {
                    if (bank % SRCS_BANKS_PER_UNPACK == 0)
                    {
                        _llk_unpack_srcs_<1>(bfd_srcs, l1_tile_idx * dwp::SRCS_SLICES_PER_TILE);
                    }
                }
                dwp::srcs_bank_to_dest<DWP_HW_BANK_SWITCH, SRCS_UNPACK>(
                    bfd_srcs,
                    l1_tile_idx,
                    MM_CT_DIM,
                    [](auto last_slice) { dwp::copy_srcs_slice_to_dest<decltype(last_slice)::value, DWP_SRCS_ADDR_MOD_STEP>(); });
            }
            wait_unpack_idle();
            wait_sfpu_idle();
        }
        PROFILER_SYNC();
    }
}

#endif

#ifdef LLK_TRISC_PACK

#include "llk_bfd_alloc.h"
#include "llk_pack.h"
#include "llk_pack_common.h"
#include "llk_pack_matmul.h"

void run_kernel(RUNTIME_PARAMETERS params)
{
#if defined(RUNTIME_FORMATS) && !defined(SPEED_OF_LIGHT)
    const FormatConfig& formats = params.formats;
#endif
#ifndef SPEED_OF_LIGHT
    const Operand& buffer_Res = params.buffer_Res;
    const Operand& buffer_C   = params.buffer_C;
#endif
    {
        START_PERF_MEASURE("INIT")
        set_up_dest_chain<ckernel::dest_dvalid_client::PACK>();
        const auto bfd_pack = ckernel::trisc::bfd_alloc_and_program<ckernel::trisc::BfdResource::Pack0>(
            ckernel::DEFAULT_TENSOR_SHAPE, L1_ADDRESS(buffer_Res[0]), formats.pack_dst);
        _llk_pack_hw_configure_<p_pacr::PACK0, is_fp32_dest_acc_en>(static_cast<DataFormat>(formats.pack_src), ckernel::ReluConfig::none());
        _llk_pack_matmul_init_(bfd_pack, MM_RT_DIM, MM_CT_DIM, 1 /*num_subblocks_c_dim*/, ckernel::DEFAULT_TENSOR_SHAPE);
        _llk_pack_set_l1_acc_<p_pacr::PACK0>(DWP_PACK_L1_ACC);
        PROFILER_SYNC();
    }
    {
        START_PERF_MEASURE("TILE_LOOP")
        dwp::BankStamps<DWP_STAMP_SYNC> stamps(buffer_C[0], NUM_BANKS);
        stamps.start();
        for (std::uint32_t bank = 0; bank < NUM_BANKS; ++bank)
        {
            _llk_pack_matmul_(0 /*start_math_dest_tile_idx*/, (bank % DWP_NUM_BLOCKS) * MM_CT_DIM, ckernel::DEFAULT_TENSOR_SHAPE);
            _llk_pack_dest_dvalid_section_done_<dest_sync, is_fp32_dest_acc_en>();
            stamps.bank_done(bank);
        }
        stamps.end();
        // L1 acc persists in config across runs; the other paths must overwrite.
        _llk_pack_set_l1_acc_<p_pacr::PACK0>(false);
        PROFILER_SYNC();
    }
}

#endif
