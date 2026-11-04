// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// RUN: %split-file %s %t
// RUN: %{blackhole_tensix_compile} %{blackhole_math_thread} -UENABLE_LLK_ASSERT -fno-ipa-icf -c %t/codegen.cpp -o %t/codegen.o
// RUN: %{blackhole_objdump} -drz %t/codegen.o > %t/codegen.dump
// RUN: %{blackhole_compare_codegen} %t/codegen.dump
// RUN: %{blackhole_tensix_compile} %{blackhole_math_thread} -UENABLE_LLK_ASSERT -c %t/compile_time.cpp -o %t/compile_time.o
// RUN: %{blackhole_objdump} -d %t/compile_time.o | FileCheck %s

// Compare each runtime HAL function with its adjacent reference_ function.
// Disable identical-code folding so both bodies are emitted independently.
// LLK_ASSERT is disabled so runtime paths contain only the issued instructions.

//--- common.h
#pragma once

#include <cstdint>

#include "hal/move.h"

namespace move = hal::move;

using SrcAToDst = move::Transfer<move::SrcA, move::Dst>;

//--- codegen.cpp
#include "common.h"

// TT_OP_ macros add their fields, but the HAL ORs a runtime row into the
// encoding, so the references OR it into a zero dst field.

extern "C" __attribute__((noinline, used)) void srca_face_runtime_destination(const std::uint32_t destination_row)
{
    move::run(SrcAToDst {
        .source_row         = 0,
        .destination_row    = destination_row,
        .address_mode       = 2,
        .number_of_rows     = move::rows::Face,
        .final_address_mode = 3,
    });
}

extern "C" __attribute__((noinline, used)) void reference_srca_face_runtime_destination(const std::uint32_t destination_row)
{
    TT_INSN(TT_OP_MOVA2D(0 /*dest_32b_lo*/, 0 /*src*/, 2 /*addr_mode*/, 2 /*instr_mod*/, 0 /*dst*/) | destination_row);
    TT_INSN(TT_OP_MOVA2D(0 /*dest_32b_lo*/, 0 /*src*/, 3 /*addr_mode*/, 2 /*instr_mod*/, 0 /*dst*/) | destination_row);
}

extern "C" __attribute__((noinline, used)) std::uint32_t srca_encode_runtime_destination(const std::uint32_t destination_row)
{
    return SrcAToDst {.source_row = 3, .destination_row = destination_row, .address_mode = 7, .number_of_rows = move::rows::One}.get_operation();
}

extern "C" __attribute__((noinline, used)) std::uint32_t reference_srca_encode_runtime_destination(const std::uint32_t destination_row)
{
    return TT_OP_MOVA2D(0 /*dest_32b_lo*/, 3 /*src*/, 7 /*addr_mode*/, 0 /*instr_mod*/, 0 /*dst*/) | destination_row;
}

//--- compile_time.cpp
#include "common.h"

constexpr SrcAToDst one_row {
    .source_row      = 3,
    .destination_row = 17,
    .address_mode    = 7,
    .number_of_rows  = move::rows::One,
};

constexpr SrcAToDst eight_rows_ignore_validity {
    .source_row            = 8,
    .destination_row       = 24,
    .address_mode          = 2,
    .number_of_rows        = move::rows::Eight,
    .destination_32bit_low = true,
    .source_validity       = move::SourceValidity::Ignore,
};

constexpr SrcAToDst four_rows {
    .source_row      = 4,
    .destination_row = 100,
    .address_mode    = 1,
    .number_of_rows  = move::rows::Four,
};

constexpr SrcAToDst face_final_mode {
    .source_row         = 0,
    .destination_row    = move::dst_layout::tile_row(1, 2),
    .address_mode       = 2,
    .number_of_rows     = move::rows::Face,
    .final_address_mode = 3,
};

constexpr SrcAToDst four_faces {
    .source_row         = 0,
    .destination_row    = 0,
    .address_mode       = 0,
    .number_of_rows     = move::rows::faces<4>(),
    .final_address_mode = 7,
};

constexpr SrcAToDst counter_relative {
    .source_row             = 16,
    .destination_row        = move::relative_row<-8>(),
    .destination_addressing = move::RowAddressing::CounterRelative,
    .address_mode           = 5,
    .number_of_rows         = move::rows::Eight,
};

extern "C" __attribute__((noinline, used)) void srca_one_row()
{
    move::run<one_row>();
}

// CHECK-LABEL: <srca_one_row>:
// CHECK-NEXT: ttmova2d 0,3,7,0,17{{$}}
// CHECK-NEXT: ret

extern "C" __attribute__((noinline, used)) void srca_eight_rows_ignore_validity()
{
    move::run<eight_rows_ignore_validity>();
}

// CHECK-LABEL: <srca_eight_rows_ignore_validity>:
// CHECK-NEXT: ttmovdbga2d 1,8,2,2,24{{$}}
// CHECK-NEXT: ret

extern "C" __attribute__((noinline, used)) void srca_four_rows()
{
    move::run<four_rows>();
}

// CHECK-LABEL: <srca_four_rows>:
// CHECK-NEXT: ttmova2d 0,4,1,0,100{{$}}
// CHECK-NEXT: ttmova2d 0,4,1,0,100{{$}}
// CHECK-NEXT: ttmova2d 0,4,1,0,100{{$}}
// CHECK-NEXT: ttmova2d 0,4,1,0,100{{$}}
// CHECK-NEXT: ret

extern "C" __attribute__((noinline, used)) void srca_face_final_mode()
{
    move::run<face_final_mode>();
}

// CHECK-LABEL: <srca_face_final_mode>:
// CHECK-NEXT: ttmova2d 0,0,2,2,96{{$}}
// CHECK-NEXT: ttmova2d 0,0,3,2,96{{$}}
// CHECK-NEXT: ret

extern "C" __attribute__((noinline, used)) void srca_four_faces()
{
    move::run<four_faces>();
}

// CHECK-LABEL: <srca_four_faces>:
// CHECK-NEXT: ttmova2d 0,0,0,2,0{{$}}
// CHECK-NEXT: ttmova2d 0,0,0,2,0{{$}}
// CHECK-NEXT: ttmova2d 0,0,0,2,0{{$}}
// CHECK-NEXT: ttmova2d 0,0,0,2,0{{$}}
// CHECK-NEXT: ttmova2d 0,0,0,2,0{{$}}
// CHECK-NEXT: ttmova2d 0,0,0,2,0{{$}}
// CHECK-NEXT: ttmova2d 0,0,0,2,0{{$}}
// CHECK-NEXT: ttmova2d 0,0,7,2,0{{$}}
// CHECK-NEXT: ret

extern "C" __attribute__((noinline, used)) void srca_counter_relative()
{
    move::run<counter_relative>();
}

// CHECK-LABEL: <srca_counter_relative>:
// CHECK-NEXT: ttmova2d 0,16,5,2,1016{{$}}
// CHECK-NEXT: ret

extern "C" __attribute__((noinline, used)) void srca_issue_encoded_operations()
{
    TTI_INSN(one_row.get_operation());
    TTI_INSN(eight_rows_ignore_validity.get_operation());
}

// CHECK-LABEL: <srca_issue_encoded_operations>:
// CHECK-NEXT: ttmova2d 0,3,7,0,17{{$}}
// CHECK-NEXT: ttmovdbga2d 1,8,2,2,24{{$}}
// CHECK-NEXT: ret
