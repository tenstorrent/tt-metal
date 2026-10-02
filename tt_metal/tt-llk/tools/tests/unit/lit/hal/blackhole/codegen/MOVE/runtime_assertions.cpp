// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// RUN: %{blackhole_tensix_compile} %{blackhole_math_thread} -c %s -o %t.o
// RUN: %{blackhole_objdump} -dr %t.o | FileCheck %s --enable-var-scope

// With LLK_ASSERT enabled, runtime descriptors are validated before issue: a
// descriptor the compiler proves invalid reduces to an unconditional ebreak,
// and a valid one carries no check at all.

#include <cstdint>

#include "hal/move.h"

namespace move = hal::move;

extern "C" __attribute__((noinline, used)) void valid_runtime_descriptors()
{
    move::run(move::Transfer<move::SrcA, move::Dst> {.source_row = 0, .destination_row = 0, .address_mode = 2, .number_of_rows = move::rows::Eight});
    move::run(move::Transfer<move::Gpr, move::Mmio> {
        .source = hal::gpr<4>(), .destination = {.addressing = move::MmioAddressing::Immediate, .byte_address = 0xffb11000u}});
    move::run(move::Transfer<move::L1, move::L1> {.source = {.block = 0}, .destination = {.block = 1}, .size = {.value = 1}});
}

// CHECK-LABEL: <valid_runtime_descriptors>:
// CHECK-NOT: ebreak
// CHECK: lui {{a[0-7]}},0x40000
// CHECK-NOT: ebreak
// CHECK: ret

extern "C" __attribute__((noinline, used)) void srca_non_canonical_rows()
{
    move::run(move::Transfer<move::SrcA, move::Dst> {.source_row = 0, .destination_row = 0, .address_mode = 0, .number_of_rows = {2}});
}

// CHECK-LABEL: <srca_non_canonical_rows>:
// CHECK-NEXT: ebreak

extern "C" __attribute__((noinline, used)) void srcb_runtime_address_mode(const std::uint8_t address_mode)
{
    move::run(move::Transfer<move::SrcB, move::Dst> {.source_row = 0, .destination_row = 0, .address_mode = address_mode, .number_of_rows = move::rows::One});
}

// CHECK-LABEL: <srcb_runtime_address_mode>:
// CHECK-NEXT: li [[LIMIT:a[0-7]]],7
// CHECK-NEXT: bltu [[LIMIT]],a0,{{[0-9a-f]+}}
// CHECK: <.L{{[0-9]+}}>:
// CHECK-NEXT: ebreak

extern "C" __attribute__((noinline, used)) void dst_absolute_span_past_dst()
{
    move::run(move::Transfer<move::Dst, move::SrcB> {.source_row = 1020, .destination_row = 0, .address_mode = 0, .number_of_rows = move::rows::Eight});
}

// CHECK-LABEL: <dst_absolute_span_past_dst>:
// CHECK-NEXT: ebreak

extern "C" __attribute__((noinline, used)) void loadind_misaligned_quad()
{
    move::run(move::Transfer<move::L1, move::Gpr> {
        .source = {.base = hal::gpr<0>(), .offset = {.gpr = hal::gpr<1>()}}, .destination = hal::gpr<1>(), .size = move::ScalarSize::Bytes16});
}

// CHECK-LABEL: <loadind_misaligned_quad>:
// CHECK-NEXT: ebreak

extern "C" __attribute__((noinline, used)) void storereg_unaligned_address()
{
    move::run(move::Transfer<move::Gpr, move::Mmio> {
        .source = hal::gpr<0>(), .destination = {.addressing = move::MmioAddressing::Immediate, .byte_address = 0xffb11002u}});
}

// CHECK-LABEL: <storereg_unaligned_address>:
// CHECK-NEXT: ebreak

extern "C" __attribute__((noinline, used)) void xmov_overlapping_spans()
{
    move::run(move::Transfer<move::L1, move::L1> {.source = {.block = 0x100}, .destination = {.block = 0x108}, .size = {.value = 16}});
}

// CHECK-LABEL: <xmov_overlapping_spans>:
// CHECK-NEXT: ebreak

extern "C" __attribute__((noinline, used)) std::uint32_t multi_instruction_get_operation(const std::uint32_t source_row)
{
    return move::Transfer<move::SrcA, move::Dst> {.source_row = source_row, .destination_row = 0, .address_mode = 0, .number_of_rows = move::rows::Face}
        .get_operation();
}

// Even a valid face descriptor reaches the single-instruction assertion.
// CHECK-LABEL: <multi_instruction_get_operation>:
// CHECK: ebreak
// CHECK-NEXT: lui {{a[0-7]}},0x12002
