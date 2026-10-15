// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// RUN: %{blackhole_tensix_compile} %{blackhole_math_thread} -UENABLE_LLK_ASSERT -c %s -o %t.o
// RUN: %{blackhole_objdump} -dr %t.o | FileCheck %s --enable-var-scope

// LLK_ASSERT is disabled so runtime paths contain only the issued instructions.

#include <cstdint>

#include "hal/move.h"

namespace move = hal::move;

using SrcAToDst = move::Transfer<move::SrcA, move::Dst>;

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
// CHECK-NEXT: ttmova2d 0,3,7,0,17
// CHECK-NEXT: ret

extern "C" __attribute__((noinline, used)) void srca_eight_rows_ignore_validity()
{
    move::run<eight_rows_ignore_validity>();
}

// CHECK-LABEL: <srca_eight_rows_ignore_validity>:
// CHECK-NEXT: ttmovdbga2d 1,8,2,2,24
// CHECK-NEXT: ret

extern "C" __attribute__((noinline, used)) void srca_four_rows()
{
    move::run<four_rows>();
}

// CHECK-LABEL: <srca_four_rows>:
// CHECK-NEXT: ttmova2d 0,4,1,0,100
// CHECK-NEXT: ttmova2d 0,4,1,0,100
// CHECK-NEXT: ttmova2d 0,4,1,0,100
// CHECK-NEXT: ttmova2d 0,4,1,0,100
// CHECK-NEXT: ret

extern "C" __attribute__((noinline, used)) void srca_face_final_mode()
{
    move::run<face_final_mode>();
}

// CHECK-LABEL: <srca_face_final_mode>:
// CHECK-NEXT: ttmova2d 0,0,2,2,96
// CHECK-NEXT: ttmova2d 0,0,3,2,96
// CHECK-NEXT: ret

extern "C" __attribute__((noinline, used)) void srca_four_faces()
{
    move::run<four_faces>();
}

// CHECK-LABEL: <srca_four_faces>:
// CHECK-NEXT: ttmova2d 0,0,0,2,0
// CHECK-NEXT: ttmova2d 0,0,0,2,0
// CHECK-NEXT: ttmova2d 0,0,0,2,0
// CHECK-NEXT: ttmova2d 0,0,0,2,0
// CHECK-NEXT: ttmova2d 0,0,0,2,0
// CHECK-NEXT: ttmova2d 0,0,0,2,0
// CHECK-NEXT: ttmova2d 0,0,0,2,0
// CHECK-NEXT: ttmova2d 0,0,7,2,0
// CHECK-NEXT: ret

extern "C" __attribute__((noinline, used)) void srca_counter_relative()
{
    move::run<counter_relative>();
}

// CHECK-LABEL: <srca_counter_relative>:
// CHECK-NEXT: ttmova2d 0,16,5,2,1016
// CHECK-NEXT: ret

extern "C" __attribute__((noinline, used)) void srca_issue_encoded_operations()
{
    TTI_INSN(one_row.get_operation());
    TTI_INSN(eight_rows_ignore_validity.get_operation());
}

// CHECK-LABEL: <srca_issue_encoded_operations>:
// CHECK-NEXT: ttmova2d 0,3,7,0,17
// CHECK-NEXT: ttmovdbga2d 1,8,2,2,24
// CHECK-NEXT: ret

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

// CHECK-LABEL: <srca_face_runtime_destination>:
// CHECK-NEXT: lui [[R0:a[0-7]]],0x1200a
// CHECK-NEXT: lui [[R1:a[0-7]]],0x0
// CHECK-NEXT: R_RISCV_HI20 __instrn_buffer
// CHECK-NEXT: R_RISCV_RELAX
// CHECK-NEXT: or [[R0]],a0,[[R0]]
// CHECK-NEXT: mv [[R1]],[[R1]]
// CHECK-NEXT: R_RISCV_LO12_I __instrn_buffer
// CHECK-NEXT: R_RISCV_RELAX
// CHECK-NEXT: lui [[R2:a[0-7]]],0x1200e
// CHECK-NEXT: sw [[R0]],0([[R1]])
// CHECK-NEXT: or a0,a0,[[R2]]
// CHECK-NEXT: sw a0,0([[R1]])
// CHECK-NEXT: ret

extern "C" __attribute__((noinline, used)) std::uint32_t srca_encode_runtime_destination(const std::uint32_t destination_row)
{
    return SrcAToDst {.source_row = 3, .destination_row = destination_row, .address_mode = 7, .number_of_rows = move::rows::One}.get_operation();
}

// CHECK-LABEL: <srca_encode_runtime_destination>:
// CHECK-NEXT: lui [[R0:a[0-7]]],0x1207c
// CHECK-NEXT: or a0,a0,[[R0]]
// CHECK-NEXT: ret
