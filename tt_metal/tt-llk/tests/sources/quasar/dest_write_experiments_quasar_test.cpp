// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
//
// 32-bit Dest write paths on Quasar, L1 -> Dest -> PACK0 -> L1 over NUM_BLOCKS * LOOP_FACTOR SyncHalf banks:
//   A (DWP_SRCS_WRITER = false): TRISC0 UNPACR_DEST fills each bank; for SQUARE/EXP TRISC3 then runs the SFPU op on Dest.
//   B (DWP_SRCS_WRITER = true):  TRISC3 streams Unpacker 2 -> SrcS and the SFPU writes each slice (copied, squared or
//                                exponentiated) into Dest.
// DWP_HW_BANK_SWITCH picks the bank hand-off of the writer: drain (STALLWAIT + CLEARDVALID) or the done bit on its
// last write. TRISC2 stamps the wall clock after every bank into buffer_C (layout in dest_write_path_quasar.h).
// DWP_WRITER_ONLY shrinks each bank's pack to one face to reduce output traffic; this still includes pack and handoff overhead.

#include "dest_write_experiments_quasar.h"

#include <cstdint>
#include <type_traits>

#include "ckernel.h"
#include "counters.h"
#include "llk_defs.h"
#include "llk_memory_checks.h"
#include "params.h"
#include "perf.h"
#include "profiler.h"
#include "quasar_test_common.h"

constexpr std::uint32_t NUM_BANKS     = DWP_NUM_BLOCKS * DWP_LOOP_FACTOR;
constexpr bool SFPU_IN_DEST_CHAIN     = DWP_SRCS_WRITER || DWP_KERNEL_OP != dwp::OP_COPY;
constexpr std::uint8_t NO_BUF_DESC_ID = 0;

static_assert(dest_sync == ckernel::DstSync::SyncHalf, "the write paths alternate the two SyncHalf Dest banks");
static_assert(is_fp32_dest_acc_en, "both write paths target a 32-bit Dest");
static_assert(DWP_TILES_PER_BANK >= 1 && DWP_TILES_PER_BANK <= 4, "a 32-bit SyncHalf Dest bank holds 1-4 tiles");
static_assert(NUM_BANKS % 2 == 0, "an even bank count returns every Dest client to bank 0 for the next run");
static_assert(NUM_BANKS <= dwp::STAMP_MAX_BANKS, "per-bank stamps must fit one buffer_C tile");
static_assert((DWP_KERNEL_OP <= dwp::OP_EXP || DWP_KERNEL_OP == dwp::OP_CHAIN), "unknown DWP_KERNEL_OP");
static_assert(!DWP_SRCS_BANK_UNPACK || DWP_SRCS_WRITER, "DWP_SRCS_BANK_UNPACK is a path B option");
static_assert(!DWP_SRCS_MULTI_BANK_UNPACK || DWP_SRCS_BANK_UNPACK, "DWP_SRCS_MULTI_BANK_UNPACK extends DWP_SRCS_BANK_UNPACK");
static_assert(!DWP_SRCS_ADDR_MOD_STEP || (DWP_SRCS_WRITER && DWP_KERNEL_OP == dwp::OP_COPY), "DWP_SRCS_ADDR_MOD_STEP only reaches the path B copy");

// DWP_SRCS_BANK_UNPACK: one UNPACR2 auto-loop per bank instead of per tile; DWP_SRCS_MULTI_BANK_UNPACK: per group of
// SRCS_BANKS_PER_UNPACK banks.
constexpr std::uint32_t SRCS_BANKS_PER_UNPACK = DWP_SRCS_MULTI_BANK_UNPACK ? dwp::srcs_banks_per_unpack(DWP_TILES_PER_BANK) : 1;
constexpr std::uint32_t SRCS_TILES_PER_UNPACK = DWP_SRCS_BANK_UNPACK ? DWP_TILES_PER_BANK * SRCS_BANKS_PER_UNPACK : 1;
constexpr dwp::SrcsUnpack SRCS_UNPACK =
    DWP_SRCS_MULTI_BANK_UNPACK ? dwp::SrcsUnpack::Caller : (DWP_SRCS_BANK_UNPACK ? dwp::SrcsUnpack::PerBank : dwp::SrcsUnpack::PerTile);
static_assert(DWP_NUM_BLOCKS % SRCS_BANKS_PER_UNPACK == 0, "a multi-bank UNPACR2 must not wrap past the last L1 block");

// DWP_WIDEN_TO_FP32: L1 holds a narrower float (e.g. Float16_b) that the unpacker widens into Float32 SrcS / Dest, so
// every register-side format is Float32 and only the L1 formats come from the harness.
inline DataFormat dest_register_format(const std::uint32_t inferred)
{
    return DWP_WIDEN_TO_FP32 ? DataFormat::Float32 : static_cast<DataFormat>(inferred);
}

template <ckernel::dest_dvalid_client CLIENT>
inline void set_up_dest_chain()
{
    if constexpr (DWP_SRCS_WRITER)
    {
        ckernel::set_up_dest_dvalid_per_thread<CLIENT>({ckernel::dest_dvalid_client::SFPU, ckernel::dest_dvalid_client::PACK});
    }
    else if constexpr (DWP_KERNEL_OP == dwp::OP_COPY)
    {
        set_up_unpack_to_pack_dest_dvalid_chain<CLIENT>();
    }
    else
    {
        set_up_unpack_to_sfpu_to_pack_dest_dvalid_chain<CLIENT>();
    }
}

#ifdef LLK_TRISC_UNPACK

#include "llk_bfd_alloc.h"
#include "llk_unpack_common.h"
#include "llk_unpack_unary_operand_to_dest.h"

void run_kernel(RUNTIME_PARAMETERS params)
{
#if defined(RUNTIME_FORMATS) && !defined(SPEED_OF_LIGHT)
    const FormatConfig& formats = params.formats;
#endif
#ifndef SPEED_OF_LIGHT
    const Operand& buffer_A = params.buffer_A;
#endif
    {
        START_PERF_MEASURE("INIT")
        if constexpr (!DWP_SRCS_WRITER)
        {
            set_up_dest_chain<ckernel::dest_dvalid_client::UNPACK>();
            const auto bfd_unpack = ckernel::trisc::bfd_alloc_and_program<ckernel::trisc::BfdResource::Unp0>(
                ckernel::DEFAULT_TENSOR_SHAPE, L1_ADDRESS(buffer_A[0]), formats.unpack_A_src);
            _llk_unpack_configure_unary_<p_unpacr::UNP_DEST>(dest_register_format(formats.unpack_A_dst));
            _llk_unpack_dest_init_();
            _llk_unpack_unary_operand_to_dest_init_(bfd_unpack);
            dwp::program_u2d_bank_mop<DWP_HW_BANK_SWITCH>(bfd_unpack, DWP_TILES_PER_BANK);
        }
        PROFILER_SYNC();
    }
    {
        START_PERF_MEASURE("TILE_LOOP")
        if constexpr (!DWP_SRCS_WRITER)
        {
            for (std::uint32_t bank = 0; bank < NUM_BANKS; ++bank)
            {
                dwp::issue_gap(DWP_PRODUCER_GAP);
                if constexpr (DWP_INPUT_STRIDE == 1)
                {
                    dwp::unpack_bank_to_dest<DWP_HW_BANK_SWITCH>((bank % DWP_NUM_BLOCKS) * DWP_TILES_PER_BANK);
                }
                else
                {
                    const auto bfd = ckernel::trisc::bfd_current<ckernel::trisc::BfdResource::Unp0>();
                    for (unsigned t = 0; t < DWP_TILES_PER_BANK; ++t)
                    {
                        TT_SET_SRC_TILE_FACE_ROW_IDX(
                            p_set_inc_sel::TILE_SEL, p_unpacr::UNP_A, ((bank % DWP_NUM_BLOCKS) * DWP_TILES_PER_BANK + t) * DWP_INPUT_STRIDE);
                        TT_SET_DST_TILE_FACE_ROW_IDX(p_set_inc_sel::TILE_SEL, p_unpacr::UNP_A, t);
                        if (DWP_HW_BANK_SWITCH && t + 1 == DWP_TILES_PER_BANK)
                        {
                            TT_UNPACR_DEST_TILE_INC(0, 0, bfd, 1);
                        }
                        else
                        {
                            TT_UNPACR_DEST_TILE_INC(0, 0, bfd, 0);
                        }
                    }
                    if constexpr (!DWP_HW_BANK_SWITCH)
                    {
                        _llk_unpack_dest_dvalid_section_done_<ckernel::DstSync::SyncHalf>();
                    }
                }
            }
            wait_unpack_idle();
        }
        PROFILER_SYNC();
    }
}

#endif

#ifdef LLK_TRISC_MATH

void run_kernel(RUNTIME_PARAMETERS params)
{
    (void)params;
    // No FPU work, but every zone boundary is a four-thread rendezvous.
    {
        START_PERF_MEASURE("INIT")
    }
    {
        START_PERF_MEASURE("TILE_LOOP")
    }
}

#endif

#ifdef LLK_TRISC_ISOLATE_SFPU

#include "cmath_common.h"
#include "llk_math_common.h"
#include "llk_sfpu/ckernel_sfpu_exp.h"
#include "llk_sfpu/ckernel_sfpu_square.h"
#include "llk_sfpu/llk_math_eltwise_unary_sfpu_macros.h"

using namespace ckernel;
using namespace ckernel::math;
using namespace ckernel::sfpu;

// Same math on both paths: accurate fp32 exp (SFPU-heavy) or a single multiply (SFPU-light).
using SrcsMath = std::conditional_t<
    DWP_KERNEL_OP == dwp::OP_EXP,
    ExpAlgo<false, true>,
    std::conditional_t<DWP_KERNEL_OP == dwp::OP_CHAIN, dwp::AffineChain<DWP_MATH_DEPTH>, SquareMath>>;

inline void sfpu_op_on_dest_tile(const std::uint32_t dst_tile)
{
    _llk_math_eltwise_sfpu_start_<ckernel::trisc::DstTileShape::Tile32x32>(dst_tile);
    for (unsigned slice = 0; slice < dwp::SRCS_SLICES_PER_TILE; ++slice)
    {
        dwp::apply_dest_slice<SrcsMath>();
    }
    // The slice walk advances RWC_D by 64 rows. Changing the next tile's
    // section base does not reset RWC_D; finish each tile before selecting
    // another base, including when the next bank starts at tile zero.
    _llk_math_eltwise_sfpu_done_();
}

void run_kernel(RUNTIME_PARAMETERS params)
{
#if defined(RUNTIME_FORMATS) && !defined(SPEED_OF_LIGHT)
    const FormatConfig& formats = params.formats;
#endif
#ifndef SPEED_OF_LIGHT
    const Operand& buffer_A = params.buffer_A;
#endif
    std::uint8_t bfd_srcs = NO_BUF_DESC_ID;
    {
        START_PERF_MEASURE("INIT")
        if constexpr (SFPU_IN_DEST_CHAIN)
        {
            set_up_dest_chain<dest_dvalid_client::SFPU>();
            const DataFormat math_format = dest_register_format(formats.math);
            _llk_math_srcAB_hw_configure_<is_fp32_dest_acc_en>(math_format, math_format);
        }
        if constexpr (DWP_SRCS_WRITER)
        {
            bfd_srcs = dwp::init_srcs_to_dest<SRCS_TILES_PER_UNPACK, DWP_SRCS_ADDR_MOD_STEP>(
                L1_ADDRESS(buffer_A[0]), static_cast<DataFormat>(formats.unpack_A_src), dest_register_format(formats.unpack_A_dst));
        }
        else if constexpr (SFPU_IN_DEST_CHAIN)
        {
            _llk_math_eltwise_sfpu_init_();
            if constexpr (DWP_KERNEL_OP == dwp::OP_SQUARE)
            {
                init_square();
            }
        }
        PROFILER_SYNC();
    }
    {
        START_PERF_MEASURE("TILE_LOOP")
        if constexpr (DWP_SRCS_WRITER)
        {
            for (std::uint32_t bank = 0; bank < NUM_BANKS; ++bank)
            {
                dwp::issue_gap(DWP_PRODUCER_GAP);
                const std::uint32_t l1_tile_idx = (bank % DWP_NUM_BLOCKS) * DWP_TILES_PER_BANK * DWP_INPUT_STRIDE;
                if constexpr (SRCS_UNPACK == dwp::SrcsUnpack::Caller)
                {
                    if (bank % SRCS_BANKS_PER_UNPACK == 0)
                    {
                        _llk_unpack_srcs_<1>(bfd_srcs, l1_tile_idx * dwp::SRCS_SLICES_PER_TILE);
                    }
                }
                if constexpr (DWP_KERNEL_OP == dwp::OP_COPY)
                {
                    dwp::srcs_bank_to_dest<DWP_HW_BANK_SWITCH, SRCS_UNPACK>(
                        bfd_srcs,
                        l1_tile_idx,
                        DWP_TILES_PER_BANK,
                        [](auto last_slice) { dwp::copy_srcs_slice_to_dest<decltype(last_slice)::value, DWP_SRCS_ADDR_MOD_STEP>(); },
                        DWP_INPUT_STRIDE);
                }
                else
                {
                    dwp::srcs_bank_to_dest<DWP_HW_BANK_SWITCH, SRCS_UNPACK>(
                        bfd_srcs,
                        l1_tile_idx,
                        DWP_TILES_PER_BANK,
                        [](auto last_slice) { dwp::apply_srcs_slice_to_dest<SrcsMath, decltype(last_slice)::value>(); },
                        DWP_INPUT_STRIDE);
                }
            }
        }
        else if constexpr (SFPU_IN_DEST_CHAIN)
        {
            // Path A's SFPU only computes; the Dest fill and its bank switch belong to TRISC0.
            for (std::uint32_t bank = 0; bank < NUM_BANKS; ++bank)
            {
                for (std::uint32_t tile = 0; tile < DWP_TILES_PER_BANK; ++tile)
                {
                    sfpu_op_on_dest_tile(tile);
                }
                _llk_math_set_dvalid_<p_cleardvalid::SFPU, dest_sync>();
            }
        }
        wait_unpack_idle();
        wait_sfpu_idle();
        PROFILER_SYNC();
    }
}

#endif

#ifdef LLK_TRISC_PACK

#include "llk_bfd_alloc.h"
#include "llk_pack.h"
#include "llk_pack_common.h"

// DWP_WRITER_ONLY: still wait for and release every bank, but pack only face 0 of its first tile (into L1 face slot
// bank % DWP_NUM_BLOCKS), so the bank stamps time the Dest writer instead of PACK0's full-bank L1 writes.
constexpr ckernel::TensorShape PACK_SHAPE   = DWP_WRITER_ONLY ? ckernel::make_tensor_shape(16, 16, 1, 1) : ckernel::DEFAULT_TENSOR_SHAPE;
constexpr std::uint32_t PACK_TILES_PER_BANK = DWP_WRITER_ONLY ? 1 : DWP_TILES_PER_BANK;

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
        const auto bfd_pack =
            ckernel::trisc::bfd_alloc_and_program<ckernel::trisc::BfdResource::Pack0>(PACK_SHAPE, L1_ADDRESS(buffer_Res[0]), formats.pack_dst);
        _llk_pack_hw_configure_<p_pacr::PACK0, is_fp32_dest_acc_en>(dest_register_format(formats.pack_src), ckernel::ReluConfig::none());
        _llk_pack_init_(bfd_pack, PACK_SHAPE, PACK_TILES_PER_BANK);
        dwp::snapshot_hash(buffer_C[0]);
        PROFILER_SYNC();
    }
    {
        START_PERF_MEASURE("TILE_LOOP")
        dwp::BankStamps<DWP_STAMP_SYNC, DWP_STAMP_EVERY> stamps(buffer_C[0], NUM_BANKS);
        stamps.start();
        for (std::uint32_t bank = 0; bank < NUM_BANKS; ++bank)
        {
            const std::uint32_t l1_tile_idx = DWP_WRITER_ONLY ? bank % DWP_NUM_BLOCKS : (bank % DWP_NUM_BLOCKS) * DWP_TILES_PER_BANK;
            dwp::issue_gap(DWP_CONSUMER_GAP);
            _llk_pack_(0 /*start_math_dest_tile_idx*/, l1_tile_idx, PACK_SHAPE);
            _llk_pack_dest_dvalid_section_done_<dest_sync, is_fp32_dest_acc_en>();
            stamps.bank_done(bank);
        }
        stamps.end();
        PROFILER_SYNC();
    }
}

#endif
