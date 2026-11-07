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

using DstToSrcB = move::Transfer<move::Dst, move::SrcB>;

//--- codegen.cpp
#include "common.h"

// TT_OP_ macros add their fields, but the HAL ORs a runtime row into the
// encoding, so the references OR it into a zero dst field.

extern "C" __attribute__((noinline, used)) void dst_face_runtime_source(const std::uint32_t source_row)
{
    DstToSrcB extract {
        .source_row         = source_row,
        .destination_row    = 16,
        .address_mode       = 0,
        .number_of_rows     = move::rows::Face,
        .final_address_mode = 3,
    };
    move::run(extract);
    extract.destination_row  = 0;
    extract.source_32bit_low = true;
    move::run(extract);
}

extern "C" __attribute__((noinline, used)) void reference_dst_face_runtime_source(const std::uint32_t source_row)
{
    TT_INSN(TT_OP_MOVD2B(0 /*dest_32b_lo*/, 16 /*src*/, 0 /*addr_mode*/, 2 /*instr_mod*/, 0 /*dst*/) | source_row);
    TT_INSN(TT_OP_MOVD2B(0 /*dest_32b_lo*/, 16 /*src*/, 0 /*addr_mode*/, 2 /*instr_mod*/, 0 /*dst*/) | source_row);
    TT_INSN(TT_OP_MOVD2B(0 /*dest_32b_lo*/, 16 /*src*/, 0 /*addr_mode*/, 2 /*instr_mod*/, 0 /*dst*/) | source_row);
    TT_INSN(TT_OP_MOVD2B(0 /*dest_32b_lo*/, 16 /*src*/, 3 /*addr_mode*/, 2 /*instr_mod*/, 0 /*dst*/) | source_row);
    TT_INSN(TT_OP_MOVD2B(1 /*dest_32b_lo*/, 0 /*src*/, 0 /*addr_mode*/, 2 /*instr_mod*/, 0 /*dst*/) | source_row);
    TT_INSN(TT_OP_MOVD2B(1 /*dest_32b_lo*/, 0 /*src*/, 0 /*addr_mode*/, 2 /*instr_mod*/, 0 /*dst*/) | source_row);
    TT_INSN(TT_OP_MOVD2B(1 /*dest_32b_lo*/, 0 /*src*/, 0 /*addr_mode*/, 2 /*instr_mod*/, 0 /*dst*/) | source_row);
    TT_INSN(TT_OP_MOVD2B(1 /*dest_32b_lo*/, 0 /*src*/, 3 /*addr_mode*/, 2 /*instr_mod*/, 0 /*dst*/) | source_row);
}

extern "C" __attribute__((noinline, used)) std::uint32_t dst_encode_runtime_source(const std::uint32_t source_row)
{
    return DstToSrcB {.source_row = source_row, .destination_row = 16, .address_mode = 3, .number_of_rows = move::rows::One, .source_32bit_low = true}
        .get_operation();
}

extern "C" __attribute__((noinline, used)) std::uint32_t reference_dst_encode_runtime_source(const std::uint32_t source_row)
{
    return TT_OP_MOVD2B(1 /*dest_32b_lo*/, 16 /*src*/, 3 /*addr_mode*/, 0 /*instr_mod*/, 0 /*dst*/) | source_row;
}

//--- compile_time.cpp
#include "common.h"

constexpr DstToSrcB one_row_low {
    .source_row       = 19,
    .destination_row  = 16,
    .address_mode     = 3,
    .number_of_rows   = move::rows::One,
    .source_32bit_low = true,
};

constexpr DstToSrcB four_rows {
    .source_row      = 200,
    .destination_row = 60,
    .address_mode    = 1,
    .number_of_rows  = move::rows::Four,
};

constexpr DstToSrcB eight_rows {
    .source_row         = 8,
    .destination_row    = 0,
    .address_mode       = 4,
    .number_of_rows     = move::rows::Eight,
    .final_address_mode = 5,
};

constexpr DstToSrcB face_final_mode {
    .source_row         = move::dst_layout::tile_row(1),
    .destination_row    = 16,
    .address_mode       = 0,
    .number_of_rows     = move::rows::Face,
    .final_address_mode = 3,
};

constexpr DstToSrcB two_faces {
    .source_row         = 0,
    .destination_row    = 0,
    .address_mode       = 0,
    .number_of_rows     = move::rows::faces<2>(),
    .final_address_mode = 3,
};

constexpr DstToSrcB counter_relative {
    .source_row        = move::relative_row<-512>(),
    .source_addressing = move::RowAddressing::CounterRelative,
    .destination_row   = 4,
    .address_mode      = 2,
    .number_of_rows    = move::rows::Four,
};

extern "C" __attribute__((noinline, used)) void dst_one_row_low()
{
    move::run<one_row_low>();
}

// CHECK-LABEL: <dst_one_row_low>:
// CHECK-NEXT: ttmovd2b 1,16,3,0,19{{$}}
// CHECK-NEXT: ret

extern "C" __attribute__((noinline, used)) void dst_four_rows()
{
    move::run<four_rows>();
}

// CHECK-LABEL: <dst_four_rows>:
// CHECK-NEXT: ttmovd2b 0,60,1,2,200{{$}}
// CHECK-NEXT: ret

extern "C" __attribute__((noinline, used)) void dst_eight_rows()
{
    move::run<eight_rows>();
}

// CHECK-LABEL: <dst_eight_rows>:
// CHECK-NEXT: ttmovd2b 0,0,4,2,8{{$}}
// CHECK-NEXT: ttmovd2b 0,0,5,2,8{{$}}
// CHECK-NEXT: ret

extern "C" __attribute__((noinline, used)) void dst_face_final_mode()
{
    move::run<face_final_mode>();
}

// CHECK-LABEL: <dst_face_final_mode>:
// CHECK-NEXT: ttmovd2b 0,16,0,2,64{{$}}
// CHECK-NEXT: ttmovd2b 0,16,0,2,64{{$}}
// CHECK-NEXT: ttmovd2b 0,16,0,2,64{{$}}
// CHECK-NEXT: ttmovd2b 0,16,3,2,64{{$}}
// CHECK-NEXT: ret

extern "C" __attribute__((noinline, used)) void dst_two_faces()
{
    move::run<two_faces>();
}

// CHECK-LABEL: <dst_two_faces>:
// CHECK-NEXT: ttmovd2b 0,0,0,2,0{{$}}
// CHECK-NEXT: ttmovd2b 0,0,0,2,0{{$}}
// CHECK-NEXT: ttmovd2b 0,0,0,2,0{{$}}
// CHECK-NEXT: ttmovd2b 0,0,0,2,0{{$}}
// CHECK-NEXT: ttmovd2b 0,0,0,2,0{{$}}
// CHECK-NEXT: ttmovd2b 0,0,0,2,0{{$}}
// CHECK-NEXT: ttmovd2b 0,0,0,2,0{{$}}
// CHECK-NEXT: ttmovd2b 0,0,3,2,0{{$}}
// CHECK-NEXT: ret

extern "C" __attribute__((noinline, used)) void dst_counter_relative()
{
    move::run<counter_relative>();
}

// CHECK-LABEL: <dst_counter_relative>:
// CHECK-NEXT: ttmovd2b 0,4,2,2,512{{$}}
// CHECK-NEXT: ret

extern "C" __attribute__((noinline, used)) void dst_issue_encoded_operations()
{
    TTI_INSN(one_row_low.get_operation());
    TTI_INSN(four_rows.get_operation());
}

// CHECK-LABEL: <dst_issue_encoded_operations>:
// CHECK-NEXT: ttmovd2b 1,16,3,0,19{{$}}
// CHECK-NEXT: ttmovd2b 0,60,1,2,200{{$}}
// CHECK-NEXT: ret
