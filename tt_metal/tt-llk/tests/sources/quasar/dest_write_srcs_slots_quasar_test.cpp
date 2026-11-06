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

#include <cstdint>
#include <type_traits>
#include <utility>

#include "ckernel.h"
#include "counters.h"
#include "dest_write_experiments_quasar.h"
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
#include "llk_srcs.h"
using namespace ckernel;
using namespace ckernel::math;
static_assert(SLOT_COUNT >= 1 && SLOT_COUNT <= 3);
constexpr unsigned SLICES = DWP_TILES_PER_BANK * 16;
constexpr unsigned GROUPS = SLICES / SLOT_COUNT;
constexpr unsigned TAIL   = SLICES % SLOT_COUNT;
constexpr unsigned PASSES = SLOT_TWO_PASS ? 2 : GROUPS;
// SLOT_PER_INSTR: slots one UNPACR2 fills, through a descriptor of 4 * SLOT_PER_INSTR FP32 rows.
constexpr unsigned INSTRS_PER_GROUP = SLOT_COUNT / SLOT_PER_INSTR;
static_assert(SLOT_PER_INSTR == 1 || SLOT_PER_INSTR == SLOT_COUNT, "one UNPACR2 fills one slot or the whole hand-off");
// FP32 descriptors must be a legal tile size (16x4 or 16x8 here); a 16x12 UNPACR2 is dropped and never sets dvalid.
static_assert(SLOT_PER_INSTR <= 2, "no legal FP32 tile shape covers three SrcS slots");
static_assert(SLOT_PER_INSTR == 1 || SLOT_STREAM_ONCE, "multi-slot descriptors only run in the configure-once stream");
// SLOT_STREAM_ONCE: program the auto-loop once and issue one stream per bank without draining U2 first. Safe because
// the previous bank's last SrcS load, issued in order before the next stream, waited for U2's last slice.
static_assert(!SLOT_STREAM_ONCE || (TAIL == 0 && !SLOT_TWO_PASS), "the configure-once stream needs whole groups");
static_assert(GROUPS <= 256, "UNPACR2 auto-loop count is 8 bits");
// SLOT_SFPU_ONLY: U2 fills each SrcS bank once and the SFPU copies the first one into every Dest slice without
// releasing it, so the TILE_LOOP times the SFPU copy alone. Dest holds input slice 0 everywhere.
static_assert(!SLOT_SFPU_ONLY || (SLOT_COUNT == 1 && SLOT_PER_INSTR == 1 && !SLOT_STREAM_ONCE && !SLOT_TWO_PASS));
// SLOT_COPY_MODE: COPY_DRAIN hands each bank over with _llk_math_set_dvalid_; COPY_DONE and COPY_LOADMACRO hand it
// over with done on the bank's last SFPSTORE. COPY_LOADMACRO copies the middle slices with SFPLOADMACRO.
constexpr unsigned COPY_DRAIN     = 0;
constexpr unsigned COPY_DONE      = 1;
constexpr unsigned COPY_LOADMACRO = 2;
static_assert(SLOT_COPY_MODE <= COPY_LOADMACRO, "unknown SLOT_COPY_MODE");
static_assert(SLOT_COPY_MODE == COPY_DRAIN || (SLOT_COUNT == 1 && SLOT_PER_INSTR == 1 && (SLOT_SFPU_ONLY || SLOT_STREAM_ONCE)));
static_assert(SLOT_COPY_MODE != COPY_LOADMACRO || SLICES >= 3, "a LOADMACRO bank starts and ends with a plain slice");

inline void issue_bank_stream(const unsigned bfd)
{
    for (unsigned i = 0; i + 1 < INSTRS_PER_GROUP; ++i)
    {
        TT_UNPACR2_TILE_INC(1, 1, bfd, 0);
    }
    TT_UNPACR2_TILE_INC(0, 1, bfd, 1);
}

// Same descriptor, but each captured word has its own slot increment/dvalid.
template <unsigned N, unsigned LOOPS>
inline void issue_group_stream(const unsigned bfd)
{
    // All previous SrcS groups have already had their reads queued before this
    // drain. Never change live capture registers while U2 is executing.
    wait_unpack_idle();
    _llk_unpack_srcs_config_<N, LOOPS>();
    for (unsigned slot = 0; slot < N; ++slot)
    {
        if (slot + 1 == N)
        {
            TT_UNPACR2_TILE_INC(0, 1, bfd, 1);
        }
        else
        {
            TT_UNPACR2_TILE_INC(1, 1, bfd, 0);
        }
    }
}

template <unsigned SLOT, unsigned N, bool RELEASE = true>
inline void copy_group()
{
    constexpr unsigned SRC = SFPU_SRCS_BASE_ADDR + 4 * SLOT;
    constexpr unsigned DST = SFPU_DEST_BASE_ADDR;
    TTI_SFPLOAD(p_sfpu::LREG0, p_sfpu::sfpmem::INT32, ADDR_MOD_7, 0, SRC);
    TTI_SFPSTORE(p_sfpu::LREG0, p_sfpu::sfpmem::INT32, ADDR_MOD_7, 0, DST);
    TTI_SFPLOAD(p_sfpu::LREG1, p_sfpu::sfpmem::INT32, ADDR_MOD_7, RELEASE && SLOT + 1 == N, SRC + 2);
    TTI_SFPSTORE(p_sfpu::LREG1, p_sfpu::sfpmem::INT32, ADDR_MOD_6, 0, DST + 2);
    if constexpr (SLOT + 1 < N)
    {
        copy_group<SLOT + 1, N, RELEASE>();
    }
}

// LOADMACRO copy of slot-0 slices. A macro store goes to the register file named by its programmed SFPSTORE's own
// address bit 10 (clear here, so Dest), at the LOADMACRO's address + the programmed store's address without RWC_D,
// so the Dest row has to come from those two immediates. Sequence i stores 4 * i rows on; slice k loads SrcS row
// 16 * (k / 4) with sequence k % 4, so it lands on Dest row 4 * k. Assumes SrcS ignores row bits above its 16 rows,
// so every such load still reads slot 0.
constexpr std::uint32_t MACRO_SEQS              = 4;
constexpr std::uint32_t MACRO_STORE_ADDR_OFFSET = 0x80;
static_assert(SLOT_COPY_MODE != COPY_LOADMACRO || SLICES * dwp::SRCS_SLICE_ROWS <= 256, "macro stores must not reach the bank-select bit");

// LOADMACRO register CONFIG_DEST = WORD through LREG0 and SFPCONFIG; sequence registers have no backdoor.
template <std::uint32_t CONFIG_DEST, std::uint32_t WORD>
inline void program_macro_reg()
{
    TTI_SFPLOADI(p_sfpu::LREG0, sfpi::SFPLOADI_MOD0_LOWER, WORD & 0xFFFF);
    TTI_SFPLOADI(p_sfpu::LREG0, sfpi::SFPLOADI_MOD0_UPPER, WORD >> 16);
    TTI_SFPNOP(0, 0, 0); // TEN-4581: keep one instruction between an LREG0 write and SFPCONFIG reading LREG0
    TTI_SFPCONFIG(0, CONFIG_DEST, 0);
    TTI_SFPNOP(0, 0, 0); // SFPCONFIG hazard: nothing may issue the cycle after SFPCONFIG
}

// Instr reg 4 + I: a Dest store 4 * I rows past the load address; ADDR_MOD_7 leaves RWC_D alone. Captured through the
// backdoor, whose register numbering is documented; the SFPCONFIG one for instruction registers is not. Its Dest
// address keeps it clear of TEN-3645 (a captured SrcS store hangs issue). The LREG field is replaced by the
// LOADMACRO's. Sequence I ([31:24] store byte: address offset, no staging so it stores the loaded LREG, delay 0, instr
// reg 4 + I; select 0 elsewhere passes issued instructions through) runs that store.
template <std::uint32_t I>
inline void init_macro_store()
{
    TTI_SFPSTORE(p_sfpu::MACRO_CAPTURE_INSTR4 + I, p_sfpu::sfpmem::INT32, ADDR_MOD_7, 0, 4);
    program_macro_reg<p_sfpconfig::MACRO_SEQ0 + I, (MACRO_STORE_ADDR_OFFSET | (0 << 3) | (4 + I)) << 24>();
}

inline void init_copy_loadmacro()
{
    constexpr std::uint32_t shift = TT_OP_SFPSTORE(0, p_sfpu::sfpmem::INT32, ADDR_MOD_7, 0, 4);
    program_macro_reg<0, shift>();
    program_macro_reg<1, shift>();
    program_macro_reg<2, shift>();
    program_macro_reg<3, shift>();
    TTI_SFPSHFT2((-1) & 0xfff, 0, p_sfpu::MACRO_CAPTURE_INSTR4, sfpi::SFPSHFT2_MOD1_SHFT_IMM);
    program_macro_reg<p_sfpconfig::MACRO_SEQ0, 0xCEC40000>();
    program_macro_reg<p_sfpconfig::MACRO_SEQ1, 0xCEC40000>();
    program_macro_reg<p_sfpconfig::MACRO_SEQ2, 0xCEC40000>();
    program_macro_reg<p_sfpconfig::MACRO_SEQ3, 0xCEC40000>();
    // [3:0] default store format INT32, [7:4] every sequence inherits the LOADMACRO's format anyway, [11:8] no
    // dependent units, so the sequencer never waits for later SFPU instructions to drain.
    constexpr std::uint32_t macro_ctrl = p_sfpu::sfpmem::INT32 | (0xF << 4);
    TTI_SFPCONFIG(macro_ctrl, p_sfpconfig::MACRO_CTRL, 1);
    TTI_SFPNOP(0, 0, 0);
}

template <unsigned SHIFT>
inline void probe_offset()
{
    constexpr std::uint32_t word = 0x72000000u | (4u << SHIFT);
    program_macro_reg<0, word>();
    program_macro_reg<1, word>();
    program_macro_reg<2, word>();
    program_macro_reg<3, word>();
    TTI_SFPSHFT2((-1) & 0xfff, 0, p_sfpu::MACRO_CAPTURE_INSTR4, sfpi::SFPSHFT2_MOD1_SHFT_IMM);
}

// Slot-0 slice at Dest RWC_D + DEST_ROW: done = RELEASE on the odd load frees the SrcS bank, done = DEST_DONE on the
// odd store hands Dest over.
template <bool RELEASE, bool DEST_DONE, std::uint32_t DEST_ROW = 0>
inline void copy_slice()
{
    TTI_SFPLOAD(p_sfpu::LREG0, p_sfpu::sfpmem::INT32, ADDR_MOD_7, 0, SFPU_SRCS_BASE_ADDR);
    TTI_SFPSTORE(p_sfpu::LREG0, p_sfpu::sfpmem::INT32, ADDR_MOD_7, 0, SFPU_DEST_BASE_ADDR + DEST_ROW);
    TTI_SFPLOAD(p_sfpu::LREG1, p_sfpu::sfpmem::INT32, ADDR_MOD_7, RELEASE, SFPU_SRCS_BASE_ADDR + 2);
    TTI_SFPSTORE(p_sfpu::LREG1, p_sfpu::sfpmem::INT32, ADDR_MOD_6, DEST_DONE, SFPU_DEST_BASE_ADDR + DEST_ROW + 2);
}

// Slice K of the bank into Dest row 4 * K. A LOADMACRO from SrcS must keep done = 0: its done would also ride the Dest
// store. The SFPNOP frees the SrcS bank instead; without RELEASE it stays as a plain SFPNOP so the issue count matches
// the streaming copy.
template <bool RELEASE, std::uint32_t K>
inline void copy_slice_loadmacro()
{
    constexpr std::uint32_t SEQ = K % MACRO_SEQS;
    constexpr std::uint32_t SRC = SFPU_DEST_BASE_ADDR;
    TTI_SFPLOADMACRO(SEQ, p_sfpu::LREG0, p_sfpu::sfpmem::INT32, ADDR_MOD_7, 0, SRC >> 1, 0);
    TTI_SFPLOADMACRO(SEQ, p_sfpu::LREG1, p_sfpu::sfpmem::INT32, ADDR_MOD_7, 0, (SRC + 2) >> 1, 0);
    TTI_SFPNOP(0, RELEASE ? 1 : 0, 0);
}

template <bool RELEASE, std::uint32_t... K>
inline void copy_slices_loadmacro(std::integer_sequence<std::uint32_t, K...>)
{
    (copy_slice_loadmacro<RELEASE, K + 1>(), ...);
}

// One Dest bank from SrcS slot 0, then the bank hand-off. LOADMACRO stores neither wait for the Dest bank nor order the
// hand-off, so the bank starts with a plain slice (its SFPSTORE waits for the bank) and ends with one whose done
// store comes at least 3 SFPU instructions after the last macro.
template <bool RELEASE>
inline void copy_bank()
{
    if constexpr (SLOT_COPY_MODE == COPY_DRAIN)
    {
        for (unsigned slice = 0; slice < SLICES; ++slice)
        {
            copy_group<0, 1, RELEASE>();
        }
        _llk_math_set_dvalid_<p_cleardvalid::SFPU, dest_sync>();
    }
    // Unrolled alike, so TRISC3's loop overhead stays off the SFPU issue rate in both modes.
    else if constexpr (SLOT_COPY_MODE == COPY_DONE)
    {
#pragma GCC unroll 16
        for (unsigned slice = 0; slice + 1 < SLICES; ++slice)
        {
            copy_slice<RELEASE, false>();
        }
        copy_slice<RELEASE, true>();
    }
    else
    {
        // The first slice leaves RWC_D = 4 and the macros leave it there, so the last slice names its row itself.
        copy_slice<RELEASE, false>();
        _reset_counters_<p_setrwc::SET_D>();
        copy_slices_loadmacro<RELEASE>(std::make_integer_sequence<std::uint32_t, 1> {});
        copy_slice<RELEASE, true, (SLICES - 1) * dwp::SRCS_SLICE_ROWS>();
    }
}

void run_kernel(RUNTIME_PARAMETERS params)
{
    unsigned bfd = 0;
    {
        START_PERF_MEASURE("INIT")
        set_up_dest_chain<dest_dvalid_client::SFPU>();
        _llk_math_srcAB_hw_configure_<true>(DataFormat::Float32, DataFormat::Float32);
        bfd = dwp::init_srcs_to_dest<1, true>(L1_ADDRESS(params.buffer_A[0]), DataFormat::Float32, DataFormat::Float32);
        if constexpr (SLOT_PER_INSTR > 1)
        {
            const auto desc = ckernel::trisc::construct_buf_desc<ckernel::trisc::L1AccessMode::Continuous>(
                ckernel::make_tensor_shape(static_cast<std::uint8_t>(dwp::SRCS_SLICE_ROWS * SLOT_PER_INSTR), 16, 1, 1),
                L1_ADDRESS(params.buffer_A[0]),
                static_cast<std::uint32_t>(DataFormat::Float32));
            _configure_buf_desc_table_(bfd, desc.words[0], desc.words[1], desc.words[2]);
        }
        if constexpr (SLOT_STREAM_ONCE)
        {
            _llk_unpack_srcs_config_<INSTRS_PER_GROUP, GROUPS>();
        }
        if constexpr (SLOT_COPY_MODE == COPY_LOADMACRO)
        {
            init_copy_loadmacro();
        }
        PROFILER_SYNC();
    }
    {
        START_PERF_MEASURE("TILE_LOOP")
        if constexpr (SLOT_SFPU_ONLY)
        {
            TT_SET_SRC_TILE_FACE_ROW_IDX(p_set_inc_sel::TILE_SEL, p_unpacr::UNP_S, 0);
            TTI_SET_DST_TILE_FACE_ROW_IDX(p_set_inc_sel::TILE_SEL, p_unpacr::UNP_S, 0);
            issue_group_stream<1, 2>(bfd);
            for (unsigned bank = 0; bank < NUM_BANKS; ++bank)
            {
                _reset_counters_<p_setrwc::SET_D>();
                switch (bank % 16)
                {
                    case 0:
                        probe_offset<0>();
                        break;
                    case 1:
                        probe_offset<1>();
                        break;
                    case 2:
                        probe_offset<2>();
                        break;
                    case 3:
                        probe_offset<3>();
                        break;
                    case 4:
                        probe_offset<4>();
                        break;
                    case 5:
                        probe_offset<5>();
                        break;
                    case 6:
                        probe_offset<6>();
                        break;
                    case 7:
                        probe_offset<7>();
                        break;
                    case 8:
                        probe_offset<8>();
                        break;
                    case 9:
                        probe_offset<9>();
                        break;
                    case 10:
                        probe_offset<10>();
                        break;
                    case 11:
                        probe_offset<11>();
                        break;
                    case 12:
                        probe_offset<12>();
                        break;
                    case 13:
                        probe_offset<13>();
                        break;
                    case 14:
                        probe_offset<14>();
                        break;
                    case 15:
                        probe_offset<15>();
                        break;
                }
                copy_bank<false>();
            }
            // Release both SrcS banks so every client starts the next run on bank 0.
            TTI_SFPLOAD(p_sfpu::LREG0, p_sfpu::sfpmem::INT32, ADDR_MOD_7, 1, SFPU_SRCS_BASE_ADDR);
            TTI_SFPLOAD(p_sfpu::LREG0, p_sfpu::sfpmem::INT32, ADDR_MOD_7, 1, SFPU_SRCS_BASE_ADDR);
        }
        else if constexpr (SLOT_STREAM_ONCE)
        {
            for (unsigned bank = 0; bank < NUM_BANKS; ++bank)
            {
                _reset_counters_<p_setrwc::SET_D>();
                TT_SET_SRC_TILE_FACE_ROW_IDX(p_set_inc_sel::TILE_SEL, p_unpacr::UNP_S, (bank % DWP_NUM_BLOCKS) * SLICES / SLOT_PER_INSTR);
                TTI_SET_DST_TILE_FACE_ROW_IDX(p_set_inc_sel::TILE_SEL, p_unpacr::UNP_S, 0);
                issue_bank_stream(bfd);
                if constexpr (SLOT_COPY_MODE == COPY_DRAIN)
                {
                    for (unsigned group = 0; group < GROUPS; ++group)
                    {
                        copy_group<0, SLOT_COUNT>();
                    }
                    _llk_math_set_dvalid_<p_cleardvalid::SFPU, dest_sync>();
                }
                else
                {
                    copy_bank<true>();
                }
            }
        }
        else
        {
            for (unsigned bank = 0; bank < NUM_BANKS; ++bank)
            {
                wait_unpack_idle(); // Retire the previous replay before resetting its indices.
                _reset_counters_<p_setrwc::SET_D>();
                TT_SET_SRC_TILE_FACE_ROW_IDX(p_set_inc_sel::TILE_SEL, p_unpacr::UNP_S, (bank % DWP_NUM_BLOCKS) * SLICES);
                TTI_SET_DST_TILE_FACE_ROW_IDX(p_set_inc_sel::TILE_SEL, p_unpacr::UNP_S, 0);
                for (unsigned group = 0; group < GROUPS;)
                {
                    if (group + PASSES <= GROUPS)
                    {
                        issue_group_stream<SLOT_COUNT, PASSES>(bfd);
                        for (unsigned i = 0; i < PASSES; ++i)
                        {
                            copy_group<0, SLOT_COUNT>();
                        }
                        group += PASSES;
                    }
                    else
                    {
                        issue_group_stream<SLOT_COUNT, 1>(bfd);
                        copy_group<0, SLOT_COUNT>();
                        ++group;
                    }
                }
                if constexpr (TAIL != 0)
                {
                    issue_group_stream<TAIL, 1>(bfd);
                    copy_group<0, TAIL>();
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
