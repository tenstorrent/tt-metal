// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// RUN: %{blackhole_tensix_compile} %{blackhole_unpack_thread} -c %s -o %t.o
// RUN: %{blackhole_objdump} -dr %t.o | FileCheck %s --enable-var-scope

#include "hal/unpack.h"

namespace unpack = hal::unpack;

// A runtime descriptor is encoded in a register and pushed through the
// instruction buffer; the UNPACR engine select lands in bit 23.

extern "C" __attribute__((noinline, used)) void run_constant_descriptor()
{
    unpack::run(unpack::DataTransfer {.engine = unpack::Engine::Unpacker1, .handoff = unpack::SourceHandoff::Keep});
}

// CHECK-LABEL: <run_constant_descriptor>:
// CHECK-DAG: lui [[OP:a[0-7]]],0x42800
// CHECK-DAG: addi [[OP]],[[OP]],129
// CHECK-DAG: R_RISCV_HI20 __instrn_buffer
// CHECK: sw [[OP]],0({{a[0-7]}})
// CHECK-NEXT: ret

// An out-of-range engine reaches the LLK assertion before the instruction is issued.
extern "C" __attribute__((noinline, used)) void run_runtime_engine(unpack::Engine engine)
{
    unpack::run(unpack::DataTransfer {.engine = engine, .handoff = unpack::SourceHandoff::FlipAndSetDataValid});
}

// CHECK-LABEL: <run_runtime_engine>:
// CHECK: li [[MAX:a[0-7]]],1
// CHECK-NEXT: bltu [[MAX]],a0,{{[0-9a-f]+}} <[[ASSERT:.L[0-9]+]]>
// CHECK-DAG: lui [[BASE:a[0-7]]],0x42000
// CHECK-DAG: addi [[OP:a[0-7]]],[[BASE]],193
// CHECK-DAG: slli [[ENGINE:a[0-7]]],a0,0x17
// CHECK-DAG: add [[WORD:a[0-7]]],[[ENGINE]],[[OP]]
// CHECK-DAG: R_RISCV_HI20 __instrn_buffer
// CHECK: sw [[WORD]],0({{a[0-7]}})
// CHECK-NEXT: ret
// CHECK: <[[ASSERT]]>:
// CHECK-NEXT: ebreak

extern "C" __attribute__((noinline, used)) void run_runtime_context_counter_increment(unpack::Engine engine)
{
    unpack::run(unpack::ContextCounterIncrement {engine});
}

// CHECK-LABEL: <run_runtime_context_counter_increment>:
// CHECK: li [[MAX:a[0-7]]],1
// CHECK-NEXT: bltu [[MAX]],a0,{{[0-9a-f]+}} <[[ASSERT:.L[0-9]+]]>
// CHECK-DAG: slli [[ENGINE:a[0-7]]],a0,0x17
// CHECK-DAG: lui [[OP:a[0-7]]],0x42002
// CHECK-DAG: add [[WORD:a[0-7]]],[[ENGINE]],[[OP]]
// CHECK-DAG: R_RISCV_HI20 __instrn_buffer
// CHECK: sw [[WORD]],0({{a[0-7]}})
// CHECK-NEXT: ret
// CHECK: <[[ASSERT]]>:
// CHECK-NEXT: ebreak

// The flush scope is the multi-context-mode field at bit 7.
extern "C" __attribute__((noinline, used)) void run_runtime_row_start_cache_flush(unpack::CacheScope scope)
{
    unpack::run(unpack::RowStartCacheFlush {unpack::Engine::Unpacker1, scope});
}

// CHECK-LABEL: <run_runtime_row_start_cache_flush>:
// CHECK-DAG: lui [[BASE:a[0-7]]],0x42800
// CHECK-DAG: addi [[OP:a[0-7]]],[[BASE]],2
// CHECK-DAG: slli [[SCOPE:a[0-7]]],a0,0x7
// CHECK-DAG: add [[WORD:a[0-7]]],[[SCOPE]],[[OP]]
// CHECK-DAG: R_RISCV_HI20 __instrn_buffer
// CHECK: sw [[WORD]],0({{a[0-7]}})
// CHECK-NEXT: ret
