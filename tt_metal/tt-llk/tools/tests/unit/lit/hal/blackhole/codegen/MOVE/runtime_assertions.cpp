// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// RUN: %{blackhole_tensix_compile} %{blackhole_math_thread} -DENABLE_LLK_ASSERT -c %s -o %t.o
// RUN: %{blackhole_objdump} -d %t.o | FileCheck %s --check-prefix=ASSERT

// With LLK_ASSERT enabled, runtime descriptors are validated before issue: a
// descriptor the compiler proves invalid reduces to an unconditional ebreak,
// and a valid one carries no check at all. Each invalid probe must trap before
// its instruction is issued.

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

// ASSERT-LABEL: <valid_runtime_descriptors>:
// ASSERT-NOT: ebreak
// ASSERT: ttmova2d 0,0,2,2,0{{$}}
// ASSERT-NOT: ebreak
// ASSERT: ttstorereg 4,17408{{$}}
// ASSERT-NOT: ebreak
// ASSERT: ttxmov 0,0{{$}}
// ASSERT-NOT: ebreak
// ASSERT: ret

// Each descriptor sits on the valid side of a limit an invalid probe below crosses.
extern "C" __attribute__((noinline, used)) void valid_runtime_boundaries()
{
    move::run(move::Transfer<move::SrcB, move::Dst> {.source_row = 0, .destination_row = 0, .address_mode = 7, .number_of_rows = move::rows::One});
    move::run(move::Transfer<move::Dst, move::SrcB> {.source_row = 1016, .destination_row = 0, .address_mode = 0, .number_of_rows = move::rows::Eight});
    move::run(move::Transfer<move::L1, move::Gpr> {
        .source = {.base = hal::gpr<0>(), .offset = {.gpr = hal::gpr<1>()}}, .destination = hal::gpr<60>(), .size = move::ScalarSize::Bytes16});
    move::run(move::Transfer<move::Gpr, move::Mmio> {
        .source = hal::gpr<0>(), .destination = {.addressing = move::MmioAddressing::Immediate, .byte_address = 0xffbffffcu}});
    move::run(move::Transfer<move::L1, move::L1> {.source = {.block = 0x100}, .destination = {.block = 0x110}, .size = {.value = 16}});
}

// ASSERT-LABEL: <valid_runtime_boundaries>:
// ASSERT-NOT: ebreak
// ASSERT: ret

extern "C" __attribute__((noinline, used)) void srca_non_canonical_rows()
{
    move::run(move::Transfer<move::SrcA, move::Dst> {.source_row = 0, .destination_row = 0, .address_mode = 0, .number_of_rows = {2}});
}

// ASSERT-LABEL: <srca_non_canonical_rows>:
// ASSERT-NOT: {{[[:space:]]tt}}
// ASSERT: ebreak
// ASSERT: ret

extern "C" __attribute__((noinline, used)) void srcb_runtime_address_mode(const std::uint8_t address_mode)
{
    move::run(move::Transfer<move::SrcB, move::Dst> {.source_row = 0, .destination_row = 0, .address_mode = address_mode, .number_of_rows = move::rows::One});
}

// ASSERT-LABEL: <srcb_runtime_address_mode>:
// ASSERT-NEXT: li [[LIMIT:a[0-7]]],7{{$}}
// ASSERT-NEXT: bltu [[LIMIT]],a0,
// ASSERT: <.L{{[0-9]+}}>:
// ASSERT-NEXT: ebreak
// ASSERT: ret

extern "C" __attribute__((noinline, used)) void srcb_address_mode_past_limit()
{
    move::run(move::Transfer<move::SrcB, move::Dst> {.source_row = 0, .destination_row = 0, .address_mode = 8, .number_of_rows = move::rows::One});
}

// ASSERT-LABEL: <srcb_address_mode_past_limit>:
// ASSERT-NOT: {{[[:space:]]tt}}
// ASSERT: ebreak
// ASSERT: ret

extern "C" __attribute__((noinline, used)) void dst_absolute_span_past_dst()
{
    move::run(move::Transfer<move::Dst, move::SrcB> {.source_row = 1020, .destination_row = 0, .address_mode = 0, .number_of_rows = move::rows::Eight});
}

// ASSERT-LABEL: <dst_absolute_span_past_dst>:
// ASSERT-NOT: {{[[:space:]]tt}}
// ASSERT: ebreak
// ASSERT: ret

extern "C" __attribute__((noinline, used)) void loadind_misaligned_quad()
{
    move::run(move::Transfer<move::L1, move::Gpr> {
        .source = {.base = hal::gpr<0>(), .offset = {.gpr = hal::gpr<1>()}}, .destination = hal::gpr<1>(), .size = move::ScalarSize::Bytes16});
}

// ASSERT-LABEL: <loadind_misaligned_quad>:
// ASSERT-NOT: {{[[:space:]]tt}}
// ASSERT: ebreak
// ASSERT: ret

extern "C" __attribute__((noinline, used)) void storereg_unaligned_address()
{
    move::run(move::Transfer<move::Gpr, move::Mmio> {
        .source = hal::gpr<0>(), .destination = {.addressing = move::MmioAddressing::Immediate, .byte_address = 0xffb11002u}});
}

// ASSERT-LABEL: <storereg_unaligned_address>:
// ASSERT-NOT: {{[[:space:]]tt}}
// ASSERT: ebreak
// ASSERT: ret

extern "C" __attribute__((noinline, used)) void xmov_overlapping_spans()
{
    move::run(move::Transfer<move::L1, move::L1> {.source = {.block = 0x100}, .destination = {.block = 0x108}, .size = {.value = 16}});
}

// ASSERT-LABEL: <xmov_overlapping_spans>:
// ASSERT-NOT: {{[[:space:]]tt}}
// ASSERT: ebreak
// ASSERT: ret

extern "C" __attribute__((noinline, used)) std::uint32_t multi_instruction_get_operation(const std::uint32_t source_row)
{
    return move::Transfer<move::SrcA, move::Dst> {.source_row = source_row, .destination_row = 0, .address_mode = 0, .number_of_rows = move::rows::Face}
        .get_operation();
}

// Even a valid face descriptor reaches the single-instruction assertion before
// its encoding is built.
// ASSERT-LABEL: <multi_instruction_get_operation>:
// ASSERT-NOT: lui {{a[0-7]}},0x12002{{$}}
// ASSERT: ebreak
// ASSERT-NEXT: lui {{a[0-7]}},0x12002{{$}}
// ASSERT: ret

extern "C" __attribute__((noinline, used)) std::uint32_t single_instruction_get_operation_boundary()
{
    return move::Transfer<move::SrcA, move::Dst> {.source_row = 56, .destination_row = 0, .address_mode = 0, .number_of_rows = move::rows::Eight}
        .get_operation();
}

// ASSERT-LABEL: <single_instruction_get_operation_boundary>:
// ASSERT-NOT: ebreak
// ASSERT: ret
