// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// RUN: %split-file %s %t
// RUN: %{blackhole_tensix_compile} %{blackhole_unpack_thread} -c %t/compile_time.cpp -o %t/compile_time.o
// RUN: %{blackhole_objdump} -d %t/compile_time.o | FileCheck %s
// RUN: %{blackhole_tensix_compile} %{blackhole_unpack_thread} -UENABLE_LLK_ASSERT -fno-ipa-icf -c %t/codegen.cpp -o %t/codegen.o
// RUN: %{blackhole_objdump} -drz %t/codegen.o > %t/codegen.dump
// RUN: %{blackhole_compare_codegen} %t/codegen.dump

// Compare each HAL function in codegen.cpp with its adjacent reference_
// function. Disable identical-code folding so both bodies are emitted
// independently.
// UNPACR operands are (unpacker, AddrMode, CfgContextCntInc, CfgContextId, AddrCntContextId, OvrdThreadId,
// SetDatValid, srcb_bcast, ZeroWrite2, AutoIncContextID, RowSearch, SearchCacheFlush, Last).
// UNPACR_NOP operands are (unpacker, Stream_Id, Msg_Clr_Cnt, Set_Dvalid, Clr_to1_fmt_Ctrl, Stall_Clr_Cntrl,
// Bank_Clr_Ctrl, Src_ClrVal_Ctrl, Unpack_Pop).

//--- common.h
#pragma once

#include <cstdint>

#include "hal/unpack.h"

namespace unpack = hal::unpack;

//--- compile_time.cpp
#include "common.h"

extern "C" __attribute__((noinline, used)) void transfer_flip_source()
{
    unpack::Unpacker0.transfer<unpack::Transfer {
        .increments = {.channel0 = {.z = 1}},
        .handoff    = unpack::SrcHandoff::FlipAndSetDataValid,
    }>();
}

// CHECK-LABEL: <transfer_flip_source>:
// CHECK-NEXT: ttunpacr 0,1,0,0,0,1,1,0,0,0,0,0,1{{$}}
// CHECK-NEXT: ret

extern "C" __attribute__((noinline, used)) void transfer_unpacker1_all_increments()
{
    unpack::Unpacker1.transfer<unpack::Transfer {
        .increments = {.channel0 = {.y = 1, .z = 2}, .channel1 = {.y = 3, .z = 1}},
        .context    = unpack::Context::explicit_context(1, 2),
        .handoff    = unpack::SrcHandoff::Keep,
    }>();
}

// CHECK-LABEL: <transfer_unpacker1_all_increments>:
// CHECK-NEXT: ttunpacr 1,214,0,1,2,1,0,0,0,0,0,0,1{{$}}
// CHECK-NEXT: ret

extern "C" __attribute__((noinline, used)) void transfer_thread_default_context()
{
    unpack::Unpacker0.transfer<unpack::Transfer {
        .context = unpack::Context::thread_default(),
        .handoff = unpack::SrcHandoff::Keep,
    }>();
}

// CHECK-LABEL: <transfer_thread_default_context>:
// CHECK-NEXT: ttunpacr 0,0,0,0,0,0,0,0,0,0,0,0,1{{$}}
// CHECK-NEXT: ret

extern "C" __attribute__((noinline, used)) void transfer_explicit_context()
{
    unpack::Unpacker0.transfer<unpack::Transfer {
        .context = unpack::Context::explicit_context(7, 2),
        .handoff = unpack::SrcHandoff::Keep,
    }>();
}

// CHECK-LABEL: <transfer_explicit_context>:
// CHECK-NEXT: ttunpacr 0,0,0,7,2,1,0,0,0,0,0,0,1{{$}}
// CHECK-NEXT: ret

extern "C" __attribute__((noinline, used)) void transfer_counter_context()
{
    unpack::Unpacker0.transfer<unpack::Transfer {
        .context = unpack::Context::counter(2),
        .handoff = unpack::SrcHandoff::Keep,
    }>();
}

// CHECK-LABEL: <transfer_counter_context>:
// CHECK-NEXT: ttunpacr 0,0,0,0,2,1,0,0,0,1,0,0,1{{$}}
// CHECK-NEXT: ret

// Thread-default ignores the hand-written context IDs; counter mode ignores the configuration ID.
extern "C" __attribute__((noinline, used)) void transfer_aggregate_contexts()
{
    unpack::Unpacker0.transfer<unpack::Transfer {
        .context = {unpack::ContextSource::ThreadDefault, 5, 2},
        .handoff = unpack::SrcHandoff::Keep,
    }>();
    unpack::Unpacker1.transfer<unpack::Transfer {
        .context = {unpack::ContextSource::Counter, 5, 1},
        .handoff = unpack::SrcHandoff::Keep,
    }>();
}

// CHECK-LABEL: <transfer_aggregate_contexts>:
// CHECK-NEXT: ttunpacr 0,0,0,0,0,0,0,0,0,0,0,0,1{{$}}
// CHECK-NEXT: ttunpacr 1,0,0,0,1,1,0,0,0,1,0,0,1{{$}}
// CHECK-NEXT: ret

extern "C" __attribute__((noinline, used)) void transfer_zero_row_continue()
{
    unpack::Unpacker0.transfer<unpack::Transfer {
        .handoff        = unpack::SrcHandoff::Keep,
        .datum_override = unpack::DatumOverride::Zero,
        .search         = unpack::SearchMode::Row,
        .accumulation   = unpack::AccumulationAction::Continue,
    }>();
}

// CHECK-LABEL: <transfer_zero_row_continue>:
// CHECK-NEXT: ttunpacr 0,0,0,0,0,1,0,0,1,0,1,0,0{{$}}
// CHECK-NEXT: ret

extern "C" __attribute__((noinline, used)) void increment_context_counters()
{
    unpack::Unpacker0.increment_context_counter();
    unpack::Unpacker1.increment_context_counter();
}

// CHECK-LABEL: <increment_context_counters>:
// CHECK-NEXT: ttunpacr 0,0,1,0,0,0,0,0,0,0,0,0,0{{$}}
// CHECK-NEXT: ttunpacr 1,0,1,0,0,0,0,0,0,0,0,0,0{{$}}
// CHECK-NEXT: ret

extern "C" __attribute__((noinline, used)) void flush_row_start_caches()
{
    unpack::Unpacker0.flush_row_start_cache<unpack::CacheScope::CurrentThread>();
    unpack::Unpacker1.flush_row_start_cache<unpack::CacheScope::AllEntries>();
}

// CHECK-LABEL: <flush_row_start_caches>:
// CHECK-NEXT: ttunpacr 0,0,0,0,0,0,0,0,0,0,0,1,0{{$}}
// CHECK-NEXT: ttunpacr 1,0,0,0,0,1,0,0,0,0,0,1,0{{$}}
// CHECK-NEXT: ret

// The default clear writes zero after waiting for the unpacker's own bank.
extern "C" __attribute__((noinline, used)) void clear_zero_handoffs()
{
    unpack::Unpacker0.clear<unpack::Clear {.handoff = unpack::SrcHandoff::FlipAndSetDataValid}>();
    unpack::Unpacker1.clear<unpack::Clear {.handoff = unpack::SrcHandoff::Keep}>();
}

// CHECK-LABEL: <clear_zero_handoffs>:
// CHECK-NEXT: ttunpacr_nop 0,0,0,1,0,1,0,0,1{{$}}
// CHECK-NEXT: ttunpacr_nop 1,0,0,0,0,1,0,0,1{{$}}
// CHECK-NEXT: ret

extern "C" __attribute__((noinline, used)) void clear_values()
{
    unpack::Unpacker0.clear<unpack::Clear {.value = unpack::ClearValue::NegativeInfinity, .handoff = unpack::SrcHandoff::FlipAndSetDataValid}>();
    unpack::Unpacker0.clear<unpack::Clear {.value = unpack::ClearValue::One, .handoff = unpack::SrcHandoff::Keep}>();
    unpack::Unpacker1.clear<unpack::Clear {.value = unpack::ClearValue::One, .handoff = unpack::SrcHandoff::FlipAndSetDataValid}>();
}

// CHECK-LABEL: <clear_values>:
// CHECK-NEXT: ttunpacr_nop 0,0,0,1,0,1,0,1,1{{$}}
// CHECK-NEXT: ttunpacr_nop 0,0,0,0,0,1,0,2,1{{$}}
// CHECK-NEXT: ttunpacr_nop 1,0,0,1,0,1,0,2,1{{$}}
// CHECK-NEXT: ret

// Waiting for the Matrix Unit's bank clears Stall_Clr_Cntrl.
extern "C" __attribute__((noinline, used)) void clear_after_matrix_bank()
{
    unpack::Unpacker0.clear<unpack::Clear {.handoff = unpack::SrcHandoff::Keep, .wait = unpack::ClearWait::MatrixBank}>();
    unpack::Unpacker0.clear<unpack::Clear {.value = unpack::ClearValue::One, .handoff = unpack::SrcHandoff::Keep, .wait = unpack::ClearWait::MatrixBank}>();
}

// CHECK-LABEL: <clear_after_matrix_bank>:
// CHECK-NEXT: ttunpacr_nop 0,0,0,0,0,0,0,0,1{{$}}
// CHECK-NEXT: ttunpacr_nop 0,0,0,0,0,0,0,2,1{{$}}
// CHECK-NEXT: ret

// The experimental both-bank clear is one UNPACR_NOP with Bank_Clr_Ctrl set and no data-valid handoff.
extern "C" __attribute__((noinline, used)) void clear_both_banks()
{
    unpack::experimental::clear_both_banks<unpack::experimental::BothBanksClear {.wait = unpack::ClearWait::MatrixBank}>(unpack::Unpacker1);
    unpack::experimental::clear_both_banks<unpack::experimental::BothBanksClear {.value = unpack::ClearValue::NegativeInfinity}>(unpack::Unpacker0);
}

// CHECK-LABEL: <clear_both_banks>:
// CHECK-NEXT: ttunpacr_nop 1,0,0,0,0,0,1,0,1{{$}}
// CHECK-NEXT: ttunpacr_nop 0,0,0,0,0,1,1,1,1{{$}}
// CHECK-NEXT: ret

//--- codegen.cpp
#include "common.h"

using namespace ckernel;

// Encoded operations are plain words that an expander such as a MOP stores.
extern "C" __attribute__((noinline, used)) void store_encoded_operations(std::uint32_t* words)
{
    words[0] = unpack::Unpacker1.operation().transfer<unpack::Transfer {.handoff = unpack::SrcHandoff::FlipAndSetDataValid}>();
    words[1] = unpack::Unpacker1.operation().increment_context_counter();
    words[2] = unpack::Unpacker0.operation().flush_row_start_cache<unpack::CacheScope::AllEntries>();
    words[3] = unpack::Unpacker1.operation().clear<unpack::Clear {.handoff = unpack::SrcHandoff::FlipAndSetDataValid}>();
    words[4] =
        unpack::experimental::clear_both_banks_operation<unpack::experimental::BothBanksClear {.wait = unpack::ClearWait::MatrixBank}>(unpack::Unpacker1);
}

extern "C" __attribute__((noinline, used)) void reference_store_encoded_operations(std::uint32_t* words)
{
    words[0] = TT_OP_UNPACR(
        1 /*Unpack_block_selection*/,
        0 /*AddrMode*/,
        0 /*CfgContextCntInc*/,
        0 /*CfgContextId*/,
        0 /*AddrCntContextId*/,
        1 /*OvrdThreadId*/,
        1 /*SetDatValid*/,
        0 /*srcb_bcast*/,
        0 /*ZeroWrite2*/,
        0 /*AutoIncContextID*/,
        0 /*RowSearch*/,
        0 /*SearchCacheFlush*/,
        1 /*Last*/);
    words[1] = TT_OP_UNPACR(
        1 /*Unpack_block_selection*/,
        0 /*AddrMode*/,
        1 /*CfgContextCntInc*/,
        0 /*CfgContextId*/,
        0 /*AddrCntContextId*/,
        0 /*OvrdThreadId*/,
        0 /*SetDatValid*/,
        0 /*srcb_bcast*/,
        0 /*ZeroWrite2*/,
        0 /*AutoIncContextID*/,
        0 /*RowSearch*/,
        0 /*SearchCacheFlush*/,
        0 /*Last*/);
    words[2] = TT_OP_UNPACR(
        0 /*Unpack_block_selection*/,
        0 /*AddrMode*/,
        0 /*CfgContextCntInc*/,
        0 /*CfgContextId*/,
        0 /*AddrCntContextId*/,
        1 /*OvrdThreadId*/,
        0 /*SetDatValid*/,
        0 /*srcb_bcast*/,
        0 /*ZeroWrite2*/,
        0 /*AutoIncContextID*/,
        0 /*RowSearch*/,
        1 /*SearchCacheFlush*/,
        0 /*Last*/);
    words[3] = TT_OP_UNPACR_NOP(
        SrcB,
        0 /*Stream_Id*/,
        0 /*Msg_Clr_Cnt*/,
        p_unpacr_nop::SET_DVALID,
        0 /*Clr_to1_fmt_Ctrl*/,
        p_unpacr_nop::WAIT_LIKE_UNPACR,
        0 /*Bank_Clr_Ctrl*/,
        p_unpacr_nop::CLR_SRC_0,
        p_unpacr_nop::CLR_SRC);
    words[4] = TT_OP_UNPACR_NOP(
        SrcB,
        0 /*Stream_Id*/,
        0 /*Msg_Clr_Cnt*/,
        0 /*Set_Dvalid*/,
        0 /*Clr_to1_fmt_Ctrl*/,
        0 /*Stall_Clr_Cntrl*/,
        1 /*Bank_Clr_Ctrl*/,
        p_unpacr_nop::CLR_SRC_0,
        p_unpacr_nop::CLR_SRC);
}

// The pairs below follow production Blackhole kernels, each written with the HAL and with the raw instruction.

// llk_unpack_A.h dest_reuse_dummy_unpack: a constexpr condition selects the unpacker.
template <bool ToSrcA>
constexpr std::uint32_t dest_reuse_dummy_unpack()
{
    constexpr unpack::Unpacker source = ToSrcA ? unpack::Unpacker0 : unpack::Unpacker1;
    return source.operation().clear<unpack::Clear {.handoff = unpack::SrcHandoff::FlipAndSetDataValid}>();
}

template <bool ToSrcA>
constexpr std::uint32_t reference_dest_reuse_dummy_unpack()
{
    constexpr std::uint32_t source = ToSrcA ? SrcA : SrcB;
    return TT_OP_UNPACR_NOP(
        source,
        0 /*Stream_Id*/,
        0 /*Msg_Clr_Cnt*/,
        p_unpacr_nop::SET_DVALID,
        0 /*Clr_to1_fmt_Ctrl*/,
        p_unpacr_nop::WAIT_LIKE_UNPACR,
        0 /*Bank_Clr_Ctrl*/,
        0 /*Src_ClrVal_Ctrl*/,
        p_unpacr_nop::UNP_ZEROSRC);
}

extern "C" __attribute__((noinline, used)) void store_selected_unpacker(std::uint32_t* words)
{
    words[0] = dest_reuse_dummy_unpack<true>();
    words[1] = dest_reuse_dummy_unpack<false>();
}

extern "C" __attribute__((noinline, used)) void reference_store_selected_unpacker(std::uint32_t* words)
{
    words[0] = reference_dest_reuse_dummy_unpack<true>();
    words[1] = reference_dest_reuse_dummy_unpack<false>();
}

// cunpack_common.h flush_xsearch_cache: the unpacker arrives as an argument and folds after inlining.
inline __attribute__((always_inline)) void flush_xsearch_cache(const unpack::Engine engine)
{
    unpack::Unpacker {engine}.flush_row_start_cache(unpack::CacheScope::CurrentThread);
}

inline __attribute__((always_inline)) void reference_flush_xsearch_cache(const std::uint32_t unpacker)
{
    TTI_UNPACR(unpacker, 0, 0, 0, 0, 0, 0, p_unpacr::RAREFYB_DISABLE, 0, 0, 0, 1 /*SearchCacheFlush*/, 0);
}

extern "C" __attribute__((noinline, used)) void flush_argument_unpackers()
{
    flush_xsearch_cache(unpack::Engine::Unpacker0);
    flush_xsearch_cache(unpack::Engine::Unpacker1);
}

extern "C" __attribute__((noinline, used)) void reference_flush_argument_unpackers()
{
    reference_flush_xsearch_cache(SrcA);
    reference_flush_xsearch_cache(SrcB);
}

// llk_unpack_tilize.h: a bool selects the address increment.
inline __attribute__((always_inline)) void unpack_tilize_srcb(const bool reload_srcB)
{
    unpack::Unpacker1.transfer({
        .increments = {.channel0 = {.z = static_cast<std::uint8_t>(reload_srcB ? 0 : 1)}},
        .handoff    = unpack::SrcHandoff::FlipAndSetDataValid,
    });
}

inline __attribute__((always_inline)) void reference_unpack_tilize_srcb(const bool reload_srcB)
{
    TTI_UNPACR(SrcB, reload_srcB ? 0b0 : 0b1, 0, 0, 0, 1, 1, p_unpacr::RAREFYB_DISABLE, 0, 0, 0, 0, 1);
}

extern "C" __attribute__((noinline, used)) void unpack_tilize_srcb_both()
{
    unpack_tilize_srcb(true);
    unpack_tilize_srcb(false);
}

extern "C" __attribute__((noinline, used)) void reference_unpack_tilize_srcb_both()
{
    reference_unpack_tilize_srcb(true);
    reference_unpack_tilize_srcb(false);
}

// llk_unpack_AB_reduce.h: template constants select the unpacker and the clear value.
template <bool NegativeInfinity>
inline __attribute__((always_inline)) void clear_reduce_data()
{
    constexpr unpack::Unpacker data = NegativeInfinity ? unpack::Unpacker0 : unpack::Unpacker1;
    data.clear<unpack::Clear {
        .value = NegativeInfinity ? unpack::ClearValue::NegativeInfinity : unpack::ClearValue::Zero, .handoff = unpack::SrcHandoff::Keep}>();
}

template <bool NegativeInfinity>
inline __attribute__((always_inline)) void reference_clear_reduce_data()
{
    constexpr std::uint32_t data_unpacker  = NegativeInfinity ? SrcA : SrcB;
    constexpr std::uint32_t data_clear_val = NegativeInfinity ? p_unpacr_nop::CLR_SRC_NEGINF : p_unpacr_nop::CLR_SRC_0;
    TTI_UNPACR_NOP(data_unpacker, 0, 0, 0, 0, p_unpacr_nop::WAIT_LIKE_UNPACR, 0, data_clear_val, p_unpacr_nop::CLR_SRC);
}

extern "C" __attribute__((noinline, used)) void clear_reduce_data_both()
{
    clear_reduce_data<true>();
    clear_reduce_data<false>();
}

extern "C" __attribute__((noinline, used)) void reference_clear_reduce_data_both()
{
    reference_clear_reduce_data<true>();
    reference_clear_reduce_data<false>();
}

// llk_unpack_dummy.h and llk_unpack_hadamard.h: clear SrcA after waiting for the Matrix Unit's bank.
extern "C" __attribute__((noinline, used)) void clear_srca_after_matrix_bank()
{
    unpack::Unpacker0.clear<unpack::Clear {.handoff = unpack::SrcHandoff::Keep, .wait = unpack::ClearWait::MatrixBank}>();
}

extern "C" __attribute__((noinline, used)) void reference_clear_srca_after_matrix_bank()
{
    TTI_UNPACR_NOP(SrcA, 0, 0, 0, 0, 0, 0, p_unpacr_nop::CLR_SRC_0, p_unpacr_nop::CLR_SRC);
}

// llk_unpack_AB_custom_mm.h: the both-bank SrcB clear between the stalls its contract requires.
extern "C" __attribute__((noinline, used)) void clear_srcb_both_banks()
{
    TTI_STALLWAIT(p_stall::STALL_UNPACK, p_stall::UNPACK);
    unpack::experimental::clear_both_banks<unpack::experimental::BothBanksClear {.wait = unpack::ClearWait::MatrixBank}>(unpack::Unpacker1);
    TTI_STALLWAIT(p_stall::STALL_UNPACK, p_stall::UNPACK1);
}

extern "C" __attribute__((noinline, used)) void reference_clear_srcb_both_banks()
{
    TTI_STALLWAIT(p_stall::STALL_UNPACK, p_stall::UNPACK);
    TTI_UNPACR_NOP(SrcB, 0, 0, 0, 0, 0, 1, 0, p_unpacr_nop::CLR_SRC);
    TTI_STALLWAIT(p_stall::STALL_UNPACK, p_stall::UNPACK1);
}
