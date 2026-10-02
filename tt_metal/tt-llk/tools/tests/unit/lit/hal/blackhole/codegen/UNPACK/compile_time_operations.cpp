// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// RUN: %{blackhole_tensix_compile} %{blackhole_unpack_thread} -c %s -o %t.o
// RUN: %{blackhole_objdump} -d %t.o | FileCheck %s

#include <cstdint>

#include "hal/unpack.h"

namespace unpack = hal::unpack;

extern "C" __attribute__((noinline, used)) void transfer_flip_source()
{
    unpack::run<unpack::DataTransfer {
        .engine     = unpack::Engine::Unpacker0,
        .increments = {.channel0 = {.z = 1}},
        .handoff    = unpack::SourceHandoff::FlipAndSetDataValid,
    }>();
}

// CHECK-LABEL: <transfer_flip_source>:
// CHECK-NEXT: ttunpacr 0,1,0,0,0,1,1,0,0,0,0,0,1
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
// CHECK-NEXT: ttunpacr 1,214,0,1,2,1,0,0,0,0,0,0,1
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
// CHECK-NEXT: ttunpacr 0,0,0,0,0,0,0,0,0,0,0,0,1
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
// CHECK-NEXT: ttunpacr 0,0,0,7,2,1,0,0,0,0,0,0,1
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
// CHECK-NEXT: ttunpacr 0,0,0,0,2,1,0,0,0,1,0,0,1
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
// CHECK-NEXT: ttunpacr 0,0,0,0,0,0,0,0,0,0,0,0,1
// CHECK-NEXT: ttunpacr 1,0,0,0,1,1,0,0,0,1,0,0,1
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
// CHECK-NEXT: ttunpacr 0,0,0,0,0,1,0,0,1,0,1,0,0
// CHECK-NEXT: ret

extern "C" __attribute__((noinline, used)) void increment_context_counters()
{
    unpack::run<unpack::ContextCounterIncrement {unpack::Engine::Unpacker0}>();
    unpack::run<unpack::ContextCounterIncrement {unpack::Engine::Unpacker1}>();
}

// CHECK-LABEL: <increment_context_counters>:
// CHECK-NEXT: ttunpacr 0,0,1,0,0,0,0,0,0,0,0,0,0
// CHECK-NEXT: ttunpacr 1,0,1,0,0,0,0,0,0,0,0,0,0
// CHECK-NEXT: ret

extern "C" __attribute__((noinline, used)) void flush_row_start_caches()
{
    unpack::run<unpack::RowStartCacheFlush {unpack::Engine::Unpacker0}>();
    unpack::run<unpack::RowStartCacheFlush {unpack::Engine::Unpacker1, unpack::CacheScope::AllEntries}>();
}

// CHECK-LABEL: <flush_row_start_caches>:
// CHECK-NEXT: ttunpacr 0,0,0,0,0,0,0,0,0,0,0,1,0
// CHECK-NEXT: ttunpacr 1,0,0,0,0,1,0,0,0,0,0,1,0
// CHECK-NEXT: ret

// Encoded operations are plain words that an expander such as a MOP stores.
extern "C" __attribute__((noinline, used)) void store_encoded_operations(std::uint32_t* words)
{
    words[0] = unpack::get_operation<unpack::DataTransfer {.engine = unpack::Engine::Unpacker1, .handoff = unpack::SourceHandoff::FlipAndSetDataValid}>();
    words[1] = unpack::get_operation<unpack::ContextCounterIncrement {unpack::Engine::Unpacker1}>();
    words[2] = unpack::get_operation<unpack::RowStartCacheFlush {unpack::Engine::Unpacker0, unpack::CacheScope::AllEntries}>();
}

// CHECK-LABEL: <store_encoded_operations>:
// CHECK-DAG: lui [[TRANSFER:a[0-7]]],0x42800
// CHECK-DAG: addi [[TRANSFER]],[[TRANSFER]],193
// CHECK-DAG: lui [[INCREMENT:a[0-7]]],0x42802
// CHECK-DAG: lui [[FLUSH:a[0-7]]],0x42000
// CHECK-DAG: addi [[FLUSH]],[[FLUSH]],130
// CHECK-DAG: sw [[TRANSFER]],0(a0)
// CHECK-DAG: sw [[INCREMENT]],4(a0)
// CHECK-DAG: sw [[FLUSH]],8(a0)
// CHECK: ret
