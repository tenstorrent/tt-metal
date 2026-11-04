// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// RUN: %split-file %s %t
// RUN: %{blackhole_tensix_compile} %{blackhole_math_thread} -UENABLE_LLK_ASSERT -fno-ipa-icf -c %t/codegen.cpp -o %t/codegen.o
// RUN: %{blackhole_objdump} -drz %t/codegen.o > %t/codegen.dump
// RUN: %{blackhole_compare_codegen} %t/codegen.dump
// RUN: %{blackhole_tensix_compile} %{blackhole_math_thread} -UENABLE_LLK_ASSERT -c %t/compile_time.cpp -o %t/compile_time.o
// RUN: %{blackhole_objdump} -d %t/compile_time.o | FileCheck %s
// RUN: %{blackhole_tensix_compile} %{blackhole_math_thread} -UENABLE_LLK_ASSERT -c %t/runtime_rows.cpp -o %t/runtime_rows.o
// RUN: %{blackhole_objdump} -d %t/runtime_rows.o | FileCheck %s --check-prefix=ROWS

// Compare each runtime HAL function with its adjacent reference_ function.
// Disable identical-code folding so both bodies are emitted independently.
// LLK_ASSERT is disabled so runtime paths contain only the issued instructions.

//--- common.h
#pragma once

#include <cstdint>

#include "hal/move.h"

namespace move = hal::move;

using SrcBToDst = move::Transfer<move::SrcB, move::Dst>;

//--- codegen.cpp
#include "common.h"

// TT_OP_ macros add their fields, but the HAL ORs a runtime row into the
// encoding, so the reference ORs it into a zero src field.

extern "C" __attribute__((noinline, used)) void srcb_eight_rows_runtime_source(const std::uint32_t source_row)
{
    move::run(SrcBToDst {
        .source_row         = source_row,
        .destination_row    = 0,
        .address_mode       = 0,
        .number_of_rows     = move::rows::Eight,
        .final_address_mode = 3,
    });
}

extern "C" __attribute__((noinline, used)) void reference_srcb_eight_rows_runtime_source(const std::uint32_t source_row)
{
    const std::uint32_t source_field = source_row << 17;
    TT_INSN(TT_OP_MOVB2D(0 /*dest_32b_lo*/, 0 /*src*/, 0 /*addr_mode*/, 4 /*movb2d_instr_mod*/, 0 /*dst*/) | source_field);
    TT_INSN(TT_OP_MOVB2D(0 /*dest_32b_lo*/, 0 /*src*/, 3 /*addr_mode*/, 4 /*movb2d_instr_mod*/, 0 /*dst*/) | source_field);
}

//--- runtime_rows.cpp
#include "common.h"

// No TT_ reference reproduces GCC's block layout for the runtime fragment
// count, so check that both fragment sizes build their intermediate and final
// (address mode 3) encodings: one-row 0x13000/0x1300c, four-row 0x13002/0x1300e.

extern "C" __attribute__((noinline, used)) void srcb_runtime_rows(const std::uint16_t number_of_rows)
{
    move::run(SrcBToDst {
        .source_row         = 0,
        .destination_row    = 0,
        .address_mode       = 0,
        .number_of_rows     = {number_of_rows},
        .final_address_mode = 3,
    });
}

// ROWS-LABEL: <srcb_runtime_rows>:
// ROWS-DAG: lui {{a[0-7]}},0x13000{{$}}
// ROWS-DAG: lui {{a[0-7]}},0x1300c{{$}}
// ROWS-DAG: lui {{a[0-7]}},0x13002{{$}}
// ROWS-DAG: lui {{a[0-7]}},0x1300e{{$}}

//--- compile_time.cpp
#include "common.h"

template <move::Broadcast B, std::uint16_t Rows>
constexpr SrcBToDst shape {
    .source_row      = 5,
    .destination_row = 32,
    .address_mode    = 1,
    .number_of_rows  = {Rows},
    .broadcast       = B,
};

extern "C" __attribute__((noinline, used)) void srcb_one_row()
{
    move::run<shape<move::Broadcast::None, 1>>();
    move::run<shape<move::Broadcast::Column0, 1>>();
    move::run<shape<move::Broadcast::Scalar, 1>>();
}

// CHECK-LABEL: <srcb_one_row>:
// CHECK-NEXT: ttmovb2d 0,5,1,0,32{{$}}
// CHECK-NEXT: ttmovb2d 0,5,1,1,32{{$}}
// CHECK-NEXT: ttmovb2d 0,5,1,1,32{{$}}
// CHECK-NEXT: ret

extern "C" __attribute__((noinline, used)) void srcb_four_rows()
{
    move::run<shape<move::Broadcast::None, 4>>();
    move::run<shape<move::Broadcast::Column0, 4>>();
}

// CHECK-LABEL: <srcb_four_rows>:
// CHECK-NEXT: ttmovb2d 0,5,1,4,32{{$}}
// CHECK-NEXT: ttmovb2d 0,5,1,5,32{{$}}
// CHECK-NEXT: ret

extern "C" __attribute__((noinline, used)) void srcb_four_rows_row_broadcast()
{
    move::run<shape<move::Broadcast::Row, 4>>();
}

// CHECK-LABEL: <srcb_four_rows_row_broadcast>:
// CHECK-NEXT: ttmovb2d 0,5,1,0,32{{$}}
// CHECK-NEXT: ttmovb2d 0,5,1,0,32{{$}}
// CHECK-NEXT: ttmovb2d 0,5,1,0,32{{$}}
// CHECK-NEXT: ttmovb2d 0,5,1,0,32{{$}}
// CHECK-NEXT: ret

extern "C" __attribute__((noinline, used)) void srcb_eight_rows()
{
    move::run<shape<move::Broadcast::None, 8>>();
    move::run<shape<move::Broadcast::Row, 8>>();
    move::run<shape<move::Broadcast::Scalar, 8>>();
}

// CHECK-LABEL: <srcb_eight_rows>:
// CHECK-NEXT: ttmovb2d 0,5,1,4,32{{$}}
// CHECK-NEXT: ttmovb2d 0,5,1,4,32{{$}}
// CHECK-NEXT: ttmovb2d 0,5,1,2,32{{$}}
// CHECK-NEXT: ttmovb2d 0,5,1,3,32{{$}}
// CHECK-NEXT: ret

constexpr SrcBToDst scalar_ignore_validity {
    .source_row            = 63,
    .destination_row       = 40,
    .address_mode          = 4,
    .number_of_rows        = move::rows::Eight,
    .broadcast             = move::Broadcast::Scalar,
    .destination_32bit_low = true,
    .source_validity       = move::SourceValidity::Ignore,
};

extern "C" __attribute__((noinline, used)) void srcb_scalar_ignore_validity()
{
    move::run<scalar_ignore_validity>();
}

// CHECK-LABEL: <srcb_scalar_ignore_validity>:
// CHECK-NEXT: ttmovdbgb2d 1,63,4,3,40{{$}}
// CHECK-NEXT: ret

constexpr SrcBToDst face_final_mode {
    .source_row         = 16,
    .destination_row    = move::dst_layout::tile_row(2),
    .address_mode       = 0,
    .number_of_rows     = move::rows::Face,
    .final_address_mode = 3,
};

constexpr SrcBToDst two_faces_column {
    .source_row         = 0,
    .destination_row    = 0,
    .address_mode       = 2,
    .number_of_rows     = move::rows::faces<2>(),
    .broadcast          = move::Broadcast::Column0,
    .final_address_mode = 6,
};

constexpr SrcBToDst four_faces_row {
    .source_row         = 63,
    .destination_row    = 0,
    .address_mode       = 0,
    .number_of_rows     = move::rows::faces<4>(),
    .broadcast          = move::Broadcast::Row,
    .final_address_mode = 3,
};

constexpr SrcBToDst face_scalar {
    .source_row      = 7,
    .destination_row = 0,
    .address_mode    = 1,
    .number_of_rows  = move::rows::Face,
    .broadcast       = move::Broadcast::Scalar,
};

extern "C" __attribute__((noinline, used)) void srcb_face_final_mode()
{
    move::run<face_final_mode>();
}

// CHECK-LABEL: <srcb_face_final_mode>:
// CHECK-NEXT: ttmovb2d 0,16,0,4,128{{$}}
// CHECK-NEXT: ttmovb2d 0,16,0,4,128{{$}}
// CHECK-NEXT: ttmovb2d 0,16,0,4,128{{$}}
// CHECK-NEXT: ttmovb2d 0,16,3,4,128{{$}}
// CHECK-NEXT: ret

extern "C" __attribute__((noinline, used)) void srcb_two_faces_column()
{
    move::run<two_faces_column>();
}

// CHECK-LABEL: <srcb_two_faces_column>:
// CHECK-NEXT: ttmovb2d 0,0,2,5,0{{$}}
// CHECK-NEXT: ttmovb2d 0,0,2,5,0{{$}}
// CHECK-NEXT: ttmovb2d 0,0,2,5,0{{$}}
// CHECK-NEXT: ttmovb2d 0,0,2,5,0{{$}}
// CHECK-NEXT: ttmovb2d 0,0,2,5,0{{$}}
// CHECK-NEXT: ttmovb2d 0,0,2,5,0{{$}}
// CHECK-NEXT: ttmovb2d 0,0,2,5,0{{$}}
// CHECK-NEXT: ttmovb2d 0,0,6,5,0{{$}}
// CHECK-NEXT: ret

extern "C" __attribute__((noinline, used)) void srcb_four_faces_row()
{
    move::run<four_faces_row>();
}

// CHECK-LABEL: <srcb_four_faces_row>:
// CHECK-NEXT: ttmovb2d 0,63,0,2,0{{$}}
// CHECK-NEXT: ttmovb2d 0,63,0,2,0{{$}}
// CHECK-NEXT: ttmovb2d 0,63,0,2,0{{$}}
// CHECK-NEXT: ttmovb2d 0,63,0,2,0{{$}}
// CHECK-NEXT: ttmovb2d 0,63,0,2,0{{$}}
// CHECK-NEXT: ttmovb2d 0,63,0,2,0{{$}}
// CHECK-NEXT: ttmovb2d 0,63,0,2,0{{$}}
// CHECK-NEXT: ttmovb2d 0,63,3,2,0{{$}}
// CHECK-NEXT: ret

extern "C" __attribute__((noinline, used)) void srcb_face_scalar()
{
    move::run<face_scalar>();
}

// CHECK-LABEL: <srcb_face_scalar>:
// CHECK-NEXT: ttmovb2d 0,7,1,3,0{{$}}
// CHECK-NEXT: ttmovb2d 0,7,1,3,0{{$}}
// CHECK-NEXT: ret

constexpr SrcBToDst counter_relative {
    .source_row             = 2,
    .destination_row        = move::relative_row<511>(),
    .destination_addressing = move::RowAddressing::CounterRelative,
    .address_mode           = 3,
    .number_of_rows         = move::rows::Four,
};

extern "C" __attribute__((noinline, used)) void srcb_counter_relative()
{
    move::run<counter_relative>();
}

// CHECK-LABEL: <srcb_counter_relative>:
// CHECK-NEXT: ttmovb2d 0,2,3,4,511{{$}}
// CHECK-NEXT: ret

extern "C" __attribute__((noinline, used)) void srcb_issue_encoded_operations()
{
    TTI_INSN((shape<move::Broadcast::Column0, 4>.get_operation()));
    TTI_INSN(scalar_ignore_validity.get_operation());
}

// CHECK-LABEL: <srcb_issue_encoded_operations>:
// CHECK-NEXT: ttmovb2d 0,5,1,5,32{{$}}
// CHECK-NEXT: ttmovdbgb2d 1,63,4,3,40{{$}}
// CHECK-NEXT: ret
