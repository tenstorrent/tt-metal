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

//--- common.h
#pragma once

#include <cstdint>

#include "hal/unpack.h"

namespace unpack = hal::unpack;

//--- compile_time.cpp
#include "common.h"

extern "C" __attribute__((noinline, used)) void transfer_flip_source()
{
    unpack::run<unpack::DataTransfer {
        .engine     = unpack::Engine::Unpacker0,
        .increments = {.channel0 = {.z = 1}},
        .handoff    = unpack::SourceHandoff::FlipAndSetDataValid,
    }>();
}

// CHECK-LABEL: <transfer_flip_source>:
// CHECK-NEXT: ttunpacr 0,1,0,0,0,1,1,0,0,0,0,0,1{{$}}
// CHECK-NEXT: ret

extern "C" __attribute__((noinline, used)) void transfer_unpacker1_all_increments()
{
    unpack::run<unpack::DataTransfer {
        .engine     = unpack::Engine::Unpacker1,
        .increments = {.channel0 = {.y = 1, .z = 2}, .channel1 = {.y = 3, .z = 1}},
        .context    = unpack::ContextSelection::explicit_context(1, 2),
        .handoff    = unpack::SourceHandoff::Keep,
    }>();
}

// CHECK-LABEL: <transfer_unpacker1_all_increments>:
// CHECK-NEXT: ttunpacr 1,214,0,1,2,1,0,0,0,0,0,0,1{{$}}
// CHECK-NEXT: ret

extern "C" __attribute__((noinline, used)) void transfer_thread_default_context()
{
    unpack::run<unpack::DataTransfer {
        .engine  = unpack::Engine::Unpacker0,
        .context = unpack::ContextSelection::thread_default(),
        .handoff = unpack::SourceHandoff::Keep,
    }>();
}

// CHECK-LABEL: <transfer_thread_default_context>:
// CHECK-NEXT: ttunpacr 0,0,0,0,0,0,0,0,0,0,0,0,1{{$}}
// CHECK-NEXT: ret

extern "C" __attribute__((noinline, used)) void transfer_explicit_context()
{
    unpack::run<unpack::DataTransfer {
        .engine  = unpack::Engine::Unpacker0,
        .context = unpack::ContextSelection::explicit_context(7, 2),
        .handoff = unpack::SourceHandoff::Keep,
    }>();
}

// CHECK-LABEL: <transfer_explicit_context>:
// CHECK-NEXT: ttunpacr 0,0,0,7,2,1,0,0,0,0,0,0,1{{$}}
// CHECK-NEXT: ret

extern "C" __attribute__((noinline, used)) void transfer_counter_context()
{
    unpack::run<unpack::DataTransfer {
        .engine  = unpack::Engine::Unpacker0,
        .context = unpack::ContextSelection::counter(2),
        .handoff = unpack::SourceHandoff::Keep,
    }>();
}

// CHECK-LABEL: <transfer_counter_context>:
// CHECK-NEXT: ttunpacr 0,0,0,0,2,1,0,0,0,1,0,0,1{{$}}
// CHECK-NEXT: ret

// Thread-default ignores the hand-written context IDs; counter mode ignores the configuration ID.
extern "C" __attribute__((noinline, used)) void transfer_aggregate_contexts()
{
    unpack::run<unpack::DataTransfer {
        .engine  = unpack::Engine::Unpacker0,
        .context = {unpack::ContextSource::ThreadDefault, 5, 2},
        .handoff = unpack::SourceHandoff::Keep,
    }>();
    unpack::run<unpack::DataTransfer {
        .engine  = unpack::Engine::Unpacker1,
        .context = {unpack::ContextSource::Counter, 5, 1},
        .handoff = unpack::SourceHandoff::Keep,
    }>();
}

// CHECK-LABEL: <transfer_aggregate_contexts>:
// CHECK-NEXT: ttunpacr 0,0,0,0,0,0,0,0,0,0,0,0,1{{$}}
// CHECK-NEXT: ttunpacr 1,0,0,0,1,1,0,0,0,1,0,0,1{{$}}
// CHECK-NEXT: ret

extern "C" __attribute__((noinline, used)) void transfer_zero_row_continue()
{
    unpack::run<unpack::DataTransfer {
        .engine         = unpack::Engine::Unpacker0,
        .handoff        = unpack::SourceHandoff::Keep,
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
    unpack::run<unpack::ContextCounterIncrement {unpack::Engine::Unpacker0}>();
    unpack::run<unpack::ContextCounterIncrement {unpack::Engine::Unpacker1}>();
}

// CHECK-LABEL: <increment_context_counters>:
// CHECK-NEXT: ttunpacr 0,0,1,0,0,0,0,0,0,0,0,0,0{{$}}
// CHECK-NEXT: ttunpacr 1,0,1,0,0,0,0,0,0,0,0,0,0{{$}}
// CHECK-NEXT: ret

extern "C" __attribute__((noinline, used)) void flush_row_start_caches()
{
    unpack::run<unpack::RowStartCacheFlush {unpack::Engine::Unpacker0}>();
    unpack::run<unpack::RowStartCacheFlush {unpack::Engine::Unpacker1, unpack::CacheScope::AllEntries}>();
}

// CHECK-LABEL: <flush_row_start_caches>:
// CHECK-NEXT: ttunpacr 0,0,0,0,0,0,0,0,0,0,0,1,0{{$}}
// CHECK-NEXT: ttunpacr 1,0,0,0,0,1,0,0,0,0,0,1,0{{$}}
// CHECK-NEXT: ret

//--- codegen.cpp
#include "common.h"

// Encoded operations are plain words that an expander such as a MOP stores.
extern "C" __attribute__((noinline, used)) void store_encoded_operations(std::uint32_t* words)
{
    words[0] = unpack::get_operation<unpack::DataTransfer {.engine = unpack::Engine::Unpacker1, .handoff = unpack::SourceHandoff::FlipAndSetDataValid}>();
    words[1] = unpack::get_operation<unpack::ContextCounterIncrement {unpack::Engine::Unpacker1}>();
    words[2] = unpack::get_operation<unpack::RowStartCacheFlush {unpack::Engine::Unpacker0, unpack::CacheScope::AllEntries}>();
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
}
