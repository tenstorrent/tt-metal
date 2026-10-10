// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// RUN: %{blackhole_tensix_compile} %{blackhole_math_thread} -UENABLE_LLK_ASSERT -c %s -o %t.o
// RUN: %{blackhole_objdump} -dr %t.o | FileCheck %s --enable-var-scope

// LLK_ASSERT is disabled so runtime paths contain only the issued instructions.

#include <cstdint>

#include "hal/move.h"

namespace move = hal::move;

using SrcBToDst = move::Transfer<move::SrcB, move::Dst>;

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
// CHECK-NEXT: ttmovb2d 0,5,1,0,32
// CHECK-NEXT: ttmovb2d 0,5,1,1,32
// CHECK-NEXT: ttmovb2d 0,5,1,1,32
// CHECK-NEXT: ret

extern "C" __attribute__((noinline, used)) void srcb_four_rows()
{
    move::run<shape<move::Broadcast::None, 4>>();
    move::run<shape<move::Broadcast::Column0, 4>>();
}

// CHECK-LABEL: <srcb_four_rows>:
// CHECK-NEXT: ttmovb2d 0,5,1,4,32
// CHECK-NEXT: ttmovb2d 0,5,1,5,32
// CHECK-NEXT: ret

extern "C" __attribute__((noinline, used)) void srcb_four_rows_row_broadcast()
{
    move::run<shape<move::Broadcast::Row, 4>>();
}

// CHECK-LABEL: <srcb_four_rows_row_broadcast>:
// CHECK-NEXT: ttmovb2d 0,5,1,0,32
// CHECK-NEXT: ttmovb2d 0,5,1,0,32
// CHECK-NEXT: ttmovb2d 0,5,1,0,32
// CHECK-NEXT: ttmovb2d 0,5,1,0,32
// CHECK-NEXT: ret

extern "C" __attribute__((noinline, used)) void srcb_eight_rows()
{
    move::run<shape<move::Broadcast::None, 8>>();
    move::run<shape<move::Broadcast::Row, 8>>();
    move::run<shape<move::Broadcast::Scalar, 8>>();
}

// CHECK-LABEL: <srcb_eight_rows>:
// CHECK-NEXT: ttmovb2d 0,5,1,4,32
// CHECK-NEXT: ttmovb2d 0,5,1,4,32
// CHECK-NEXT: ttmovb2d 0,5,1,2,32
// CHECK-NEXT: ttmovb2d 0,5,1,3,32
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
// CHECK-NEXT: ttmovdbgb2d 1,63,4,3,40
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
// CHECK-NEXT: ttmovb2d 0,16,0,4,128
// CHECK-NEXT: ttmovb2d 0,16,0,4,128
// CHECK-NEXT: ttmovb2d 0,16,0,4,128
// CHECK-NEXT: ttmovb2d 0,16,3,4,128
// CHECK-NEXT: ret

extern "C" __attribute__((noinline, used)) void srcb_two_faces_column()
{
    move::run<two_faces_column>();
}

// CHECK-LABEL: <srcb_two_faces_column>:
// CHECK-NEXT: ttmovb2d 0,0,2,5,0
// CHECK-NEXT: ttmovb2d 0,0,2,5,0
// CHECK-NEXT: ttmovb2d 0,0,2,5,0
// CHECK-NEXT: ttmovb2d 0,0,2,5,0
// CHECK-NEXT: ttmovb2d 0,0,2,5,0
// CHECK-NEXT: ttmovb2d 0,0,2,5,0
// CHECK-NEXT: ttmovb2d 0,0,2,5,0
// CHECK-NEXT: ttmovb2d 0,0,6,5,0
// CHECK-NEXT: ret

extern "C" __attribute__((noinline, used)) void srcb_four_faces_row()
{
    move::run<four_faces_row>();
}

// CHECK-LABEL: <srcb_four_faces_row>:
// CHECK-NEXT: ttmovb2d 0,63,0,2,0
// CHECK-NEXT: ttmovb2d 0,63,0,2,0
// CHECK-NEXT: ttmovb2d 0,63,0,2,0
// CHECK-NEXT: ttmovb2d 0,63,0,2,0
// CHECK-NEXT: ttmovb2d 0,63,0,2,0
// CHECK-NEXT: ttmovb2d 0,63,0,2,0
// CHECK-NEXT: ttmovb2d 0,63,0,2,0
// CHECK-NEXT: ttmovb2d 0,63,3,2,0
// CHECK-NEXT: ret

extern "C" __attribute__((noinline, used)) void srcb_face_scalar()
{
    move::run<face_scalar>();
}

// CHECK-LABEL: <srcb_face_scalar>:
// CHECK-NEXT: ttmovb2d 0,7,1,3,0
// CHECK-NEXT: ttmovb2d 0,7,1,3,0
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
// CHECK-NEXT: ttmovb2d 0,2,3,4,511
// CHECK-NEXT: ret

extern "C" __attribute__((noinline, used)) void srcb_issue_encoded_operations()
{
    TTI_INSN((shape<move::Broadcast::Column0, 4>.get_operation()));
    TTI_INSN(scalar_ignore_validity.get_operation());
}

// CHECK-LABEL: <srcb_issue_encoded_operations>:
// CHECK-NEXT: ttmovb2d 0,5,1,5,32
// CHECK-NEXT: ttmovdbgb2d 1,63,4,3,40
// CHECK-NEXT: ret

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

// CHECK-LABEL: <srcb_eight_rows_runtime_source>:
// CHECK-NEXT: slli a0,a0,0x11
// CHECK-NEXT: lui [[R0:a[0-7]]],0x13002
// CHECK-NEXT: lui [[R1:a[0-7]]],0x0
// CHECK-NEXT: R_RISCV_HI20 __instrn_buffer
// CHECK-NEXT: R_RISCV_RELAX
// CHECK-NEXT: or [[R0]],a0,[[R0]]
// CHECK-NEXT: mv [[R1]],[[R1]]
// CHECK-NEXT: R_RISCV_LO12_I __instrn_buffer
// CHECK-NEXT: R_RISCV_RELAX
// CHECK-NEXT: lui [[R2:a[0-7]]],0x1300e
// CHECK-NEXT: sw [[R0]],0([[R1]])
// CHECK-NEXT: or a0,a0,[[R2]]
// CHECK-NEXT: sw a0,0([[R1]])
// CHECK-NEXT: ret

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

// A runtime row count selects one- or four-row fragments; intermediate and
// final (address mode 3) encodings are both built for each fragment size.
// CHECK-LABEL: <srcb_runtime_rows>:
// CHECK-DAG: lui {{a[0-7]}},0x13000
// CHECK-DAG: lui {{a[0-7]}},0x1300c
// CHECK-DAG: lui {{a[0-7]}},0x13002
// CHECK-DAG: lui {{a[0-7]}},0x1300e
