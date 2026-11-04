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

// A constant descriptor passed to the runtime overload still folds to an
// immediate instruction.
extern "C" __attribute__((noinline, used)) void run_constant_descriptor()
{
    unpack::run(unpack::DataTransfer {.engine = unpack::Engine::Unpacker1, .handoff = unpack::SourceHandoff::Keep});
}

extern "C" __attribute__((noinline, used)) void reference_run_constant_descriptor()
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
}

// Runtime descriptors are encoded in a register and pushed through the
// instruction buffer.

extern "C" __attribute__((noinline, used)) void run_runtime_engine(const unpack::Engine engine)
{
    unpack::run(unpack::DataTransfer {.engine = engine, .handoff = unpack::SourceHandoff::FlipAndSetDataValid});
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
    unpack::run(unpack::ContextCounterIncrement {engine});
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
    unpack::run(unpack::RowStartCacheFlush {unpack::Engine::Unpacker1, scope});
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

//--- asserts.cpp
#include "common.h"

// Invalid runtime operands must trap before the instruction is issued; valid
// boundaries must not trap.

extern "C" void check_runtime_engine(const unpack::Engine engine)
{
    unpack::run(unpack::DataTransfer {.engine = engine, .handoff = unpack::SourceHandoff::FlipAndSetDataValid});
}

// ASSERT-LABEL: <check_runtime_engine>:
// ASSERT: li [[LIMIT:a[0-7]]],1{{$}}
// ASSERT-NEXT: bltu [[LIMIT]],a0,
// ASSERT: ebreak

extern "C" void reject_invalid_engine()
{
    unpack::run(unpack::DataTransfer {.engine = static_cast<unpack::Engine>(2), .handoff = unpack::SourceHandoff::FlipAndSetDataValid});
}

// ASSERT-LABEL: <reject_invalid_engine>:
// ASSERT-NOT: {{[[:space:]]tt}}
// ASSERT: ebreak
// ASSERT: ret

extern "C" void check_runtime_context_counter_increment(const unpack::Engine engine)
{
    unpack::run(unpack::ContextCounterIncrement {engine});
}

// ASSERT-LABEL: <check_runtime_context_counter_increment>:
// ASSERT: li [[LIMIT:a[0-7]]],1{{$}}
// ASSERT-NEXT: bltu [[LIMIT]],a0,
// ASSERT: ebreak

extern "C" void reject_invalid_context_counter_engine()
{
    unpack::run(unpack::ContextCounterIncrement {static_cast<unpack::Engine>(2)});
}

// ASSERT-LABEL: <reject_invalid_context_counter_engine>:
// ASSERT-NOT: {{[[:space:]]tt}}
// ASSERT: ebreak
// ASSERT: ret

// A bool-backed scope has no invalid runtime value, so it adds no assert.
extern "C" void accept_runtime_row_start_cache_flush(const unpack::CacheScope scope)
{
    unpack::run(unpack::RowStartCacheFlush {unpack::Engine::Unpacker1, scope});
}

// ASSERT-LABEL: <accept_runtime_row_start_cache_flush>:
// ASSERT-NOT: ebreak
// ASSERT: ret

extern "C" void accept_runtime_engine_boundaries()
{
    unpack::run(unpack::DataTransfer {.engine = unpack::Engine::Unpacker1, .handoff = unpack::SourceHandoff::FlipAndSetDataValid});
    unpack::run(unpack::ContextCounterIncrement {unpack::Engine::Unpacker1});
}

// ASSERT-LABEL: <accept_runtime_engine_boundaries>:
// ASSERT-NOT: ebreak
// ASSERT: ret
