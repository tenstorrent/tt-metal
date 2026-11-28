// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// RUN: %split-file %s %t
// RUN: %{blackhole_tensix_compile} %{blackhole_unpack_thread} -UENABLE_LLK_ASSERT -fno-ipa-icf -c %t/codegen.cpp -o %t/codegen.o
// RUN: %{blackhole_objdump} -drz %t/codegen.o > %t/codegen.dump
// RUN: %{blackhole_compare_codegen} %t/codegen.dump
// RUN: %{blackhole_tensix_compile} %{blackhole_unpack_thread} -DENABLE_LLK_ASSERT -c %t/asserts.cpp -o %t/assert.o
// RUN: %{blackhole_objdump} -d %t/assert.o | FileCheck %s --check-prefix=ASSERT

// Compare each HAL function with its adjacent reference_ function. Disable
// identical-code folding so both bodies are emitted independently.
// codegen.cpp is compiled without LLK_ASSERT so runtime paths contain only the
// issued instruction; asserts.cpp checks the runtime operand asserts.

//--- common.h
#pragma once

#include <cstdint>

#include "hal/unpack.h"

namespace unpack = hal::unpack;

//--- codegen.cpp
#include "common.h"

// Constant options passed to the runtime overloads still fold to immediate instructions.
extern "C" __attribute__((noinline, used)) void run_constant_options()
{
    unpack::Unpacker1.transfer({.handoff = unpack::SrcHandoff::Keep});
    unpack::Unpacker0.clear({.value = unpack::ClearValue::NegativeInfinity, .handoff = unpack::SrcHandoff::Keep});
}

extern "C" __attribute__((noinline, used)) void reference_run_constant_options()
{
    TTI_UNPACR(
        1 /*Unpack_block_selection*/,
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
        0 /*SearchCacheFlush*/,
        1 /*Last*/);
    TTI_UNPACR_NOP(
        0 /*Unpacker_Select*/,
        0 /*Stream_Id*/,
        0 /*Msg_Clr_Cnt*/,
        0 /*Set_Dvalid*/,
        0 /*Clr_to1_fmt_Ctrl*/,
        1 /*Stall_Clr_Cntrl*/,
        0 /*Bank_Clr_Ctrl*/,
        1 /*Src_ClrVal_Ctrl*/,
        1 /*Unpack_Pop*/);
}

// Runtime unpackers and options are encoded in a register and pushed through the
// instruction buffer.

extern "C" __attribute__((noinline, used)) void run_runtime_engine(const unpack::Engine engine)
{
    unpack::Unpacker {engine}.transfer({.handoff = unpack::SrcHandoff::FlipAndSetDataValid});
}

extern "C" __attribute__((noinline, used)) void reference_run_runtime_engine(const unpack::Engine engine)
{
    TT_UNPACR(
        static_cast<std::uint32_t>(engine),
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
}

extern "C" __attribute__((noinline, used)) void run_runtime_context_counter_increment(const unpack::Engine engine)
{
    unpack::Unpacker {engine}.increment_context_counter();
}

extern "C" __attribute__((noinline, used)) void reference_run_runtime_context_counter_increment(const unpack::Engine engine)
{
    TT_UNPACR(
        static_cast<std::uint32_t>(engine),
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
}

// The flush scope is the multi-context-mode field.
extern "C" __attribute__((noinline, used)) void run_runtime_row_start_cache_flush(const unpack::CacheScope scope)
{
    unpack::Unpacker1.flush_row_start_cache(scope);
}

extern "C" __attribute__((noinline, used)) void reference_run_runtime_row_start_cache_flush(const unpack::CacheScope scope)
{
    TT_UNPACR(
        1 /*Unpack_block_selection*/,
        0 /*AddrMode*/,
        0 /*CfgContextCntInc*/,
        0 /*CfgContextId*/,
        0 /*AddrCntContextId*/,
        static_cast<std::uint32_t>(scope),
        0 /*SetDatValid*/,
        0 /*srcb_bcast*/,
        0 /*ZeroWrite2*/,
        0 /*AutoIncContextID*/,
        0 /*RowSearch*/,
        1 /*SearchCacheFlush*/,
        0 /*Last*/);
}

extern "C" __attribute__((noinline, used)) void run_runtime_clear(const unpack::Engine engine)
{
    unpack::Unpacker {engine}.clear({.handoff = unpack::SrcHandoff::FlipAndSetDataValid});
}

extern "C" __attribute__((noinline, used)) void reference_run_runtime_clear(const unpack::Engine engine)
{
    TT_UNPACR_NOP(
        static_cast<std::uint32_t>(engine),
        0 /*Stream_Id*/,
        0 /*Msg_Clr_Cnt*/,
        1 /*Set_Dvalid*/,
        0 /*Clr_to1_fmt_Ctrl*/,
        1 /*Stall_Clr_Cntrl*/,
        0 /*Bank_Clr_Ctrl*/,
        0 /*Src_ClrVal_Ctrl*/,
        1 /*Unpack_Pop*/);
}

// The clear value is the Src_ClrVal_Ctrl field.
extern "C" __attribute__((noinline, used)) void run_runtime_clear_value(const unpack::ClearValue value)
{
    unpack::Unpacker0.clear({.value = value, .handoff = unpack::SrcHandoff::Keep});
}

extern "C" __attribute__((noinline, used)) void reference_run_runtime_clear_value(const unpack::ClearValue value)
{
    TT_UNPACR_NOP(
        0 /*Unpacker_Select*/,
        0 /*Stream_Id*/,
        0 /*Msg_Clr_Cnt*/,
        0 /*Set_Dvalid*/,
        0 /*Clr_to1_fmt_Ctrl*/,
        1 /*Stall_Clr_Cntrl*/,
        0 /*Bank_Clr_Ctrl*/,
        static_cast<std::uint32_t>(value),
        1 /*Unpack_Pop*/);
}

// The both-bank clear value is the Src_ClrVal_Ctrl field, always with the Matrix-bank wait.
extern "C" __attribute__((noinline, used)) void run_runtime_both_banks_clear(const unpack::ClearValue value)
{
    unpack::experimental::clear_both_banks(unpack::Unpacker1, {.value = value});
}

extern "C" __attribute__((noinline, used)) void reference_run_runtime_both_banks_clear(const unpack::ClearValue value)
{
    TT_UNPACR_NOP(
        1 /*Unpacker_Select*/,
        0 /*Stream_Id*/,
        0 /*Msg_Clr_Cnt*/,
        0 /*Set_Dvalid*/,
        0 /*Clr_to1_fmt_Ctrl*/,
        0 /*Stall_Clr_Cntrl*/,
        1 /*Bank_Clr_Ctrl*/,
        static_cast<std::uint32_t>(value),
        1 /*Unpack_Pop*/);
}

//--- asserts.cpp
#include "common.h"

// Invalid runtime operands must trap before the instruction is issued; valid
// boundaries must not trap.

extern "C" void check_runtime_engine(const unpack::Engine engine)
{
    unpack::Unpacker {engine}.transfer({.handoff = unpack::SrcHandoff::FlipAndSetDataValid});
}

// ASSERT-LABEL: <check_runtime_engine>:
// ASSERT: li [[LIMIT:a[0-7]]],1{{$}}
// ASSERT-NEXT: bltu [[LIMIT]],a0,
// ASSERT: ebreak

extern "C" void reject_invalid_engine()
{
    unpack::Unpacker {static_cast<unpack::Engine>(2)}.transfer({.handoff = unpack::SrcHandoff::FlipAndSetDataValid});
}

// ASSERT-LABEL: <reject_invalid_engine>:
// ASSERT-NOT: {{[[:space:]]tt}}
// ASSERT: ebreak
// ASSERT: ret

extern "C" void check_runtime_context_counter_increment(const unpack::Engine engine)
{
    unpack::Unpacker {engine}.increment_context_counter();
}

// ASSERT-LABEL: <check_runtime_context_counter_increment>:
// ASSERT: li [[LIMIT:a[0-7]]],1{{$}}
// ASSERT-NEXT: bltu [[LIMIT]],a0,
// ASSERT: ebreak

extern "C" void reject_invalid_context_counter_engine()
{
    unpack::Unpacker {static_cast<unpack::Engine>(2)}.increment_context_counter();
}

// ASSERT-LABEL: <reject_invalid_context_counter_engine>:
// ASSERT-NOT: {{[[:space:]]tt}}
// ASSERT: ebreak
// ASSERT: ret

// A bool-backed scope has no invalid runtime value, so it adds no assert.
extern "C" void accept_runtime_row_start_cache_flush(const unpack::CacheScope scope)
{
    unpack::Unpacker1.flush_row_start_cache(scope);
}

// ASSERT-LABEL: <accept_runtime_row_start_cache_flush>:
// ASSERT-NOT: ebreak
// ASSERT: ret

extern "C" void accept_runtime_engine_boundaries()
{
    unpack::Unpacker1.transfer({.handoff = unpack::SrcHandoff::FlipAndSetDataValid});
    unpack::Unpacker1.increment_context_counter();
    unpack::experimental::clear(unpack::Unpacker1, {.value = unpack::ClearValue::One, .handoff = unpack::SrcHandoff::Keep});
}

// ASSERT-LABEL: <accept_runtime_engine_boundaries>:
// ASSERT-NOT: ebreak
// ASSERT: ret

// Unpacker-specific rules are checked where the unpacker is only known at run time.
extern "C" void check_runtime_negative_infinity_clear(const unpack::Engine engine)
{
    unpack::Unpacker {engine}.clear({.value = unpack::ClearValue::NegativeInfinity, .handoff = unpack::SrcHandoff::Keep});
}

// ASSERT-LABEL: <check_runtime_negative_infinity_clear>:
// ASSERT: ebreak

// Issued with a known unpacker but outside a constant expression, the rule becomes an assert.
extern "C" void reject_srcb_negative_infinity_clear()
{
    unpack::Unpacker1.clear<unpack::Clear {.value = unpack::ClearValue::NegativeInfinity, .handoff = unpack::SrcHandoff::Keep}>();
}

// ASSERT-LABEL: <reject_srcb_negative_infinity_clear>:
// ASSERT-NOT: {{[[:space:]]tt}}
// ASSERT: ebreak
// ASSERT: ret

extern "C" void reject_unpacker1_explicit_context()
{
    unpack::Unpacker1.transfer<unpack::Transfer {.context = unpack::Context::explicit_context(2), .handoff = unpack::SrcHandoff::Keep}>();
}

// ASSERT-LABEL: <reject_unpacker1_explicit_context>:
// ASSERT-NOT: {{[[:space:]]tt}}
// ASSERT: ebreak
// ASSERT: ret

extern "C" void check_runtime_clear_value(const unpack::ClearValue value)
{
    unpack::experimental::clear(unpack::Unpacker0, {.value = value, .handoff = unpack::SrcHandoff::Keep});
}

// ASSERT-LABEL: <check_runtime_clear_value>:
// ASSERT: li [[LIMIT:a[0-7]]],2{{$}}
// ASSERT-NEXT: bltu [[LIMIT]],a0,
// ASSERT: ebreak

// The non-experimental clear rejects ClearValue::One.
extern "C" void reject_runtime_one_clear()
{
    unpack::Unpacker0.clear({.value = unpack::ClearValue::One, .handoff = unpack::SrcHandoff::Keep});
}

// ASSERT-LABEL: <reject_runtime_one_clear>:
// ASSERT-NOT: {{[[:space:]]tt}}
// ASSERT: ebreak
// ASSERT: ret

extern "C" void check_runtime_clear_handoff(const unpack::ClearWait wait)
{
    unpack::Unpacker0.clear({.handoff = unpack::SrcHandoff::FlipAndSetDataValid, .wait = wait});
}

// ASSERT-LABEL: <check_runtime_clear_handoff>:
// ASSERT: ebreak

extern "C" void accept_clear_boundaries()
{
    unpack::Unpacker0.clear({.value = unpack::ClearValue::NegativeInfinity, .handoff = unpack::SrcHandoff::FlipAndSetDataValid});
    unpack::experimental::clear(
        unpack::Unpacker0, {.value = unpack::ClearValue::One, .handoff = unpack::SrcHandoff::Keep, .wait = unpack::ClearWait::MatrixBank});
    unpack::experimental::clear_both_banks(unpack::Unpacker1, {.value = unpack::ClearValue::One});
}

// ASSERT-LABEL: <accept_clear_boundaries>:
// ASSERT-NOT: ebreak
// ASSERT: ret
