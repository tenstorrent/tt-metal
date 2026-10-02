// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// RUN: %{blackhole_tensix_compile} %{blackhole_math_thread} -UENABLE_LLK_ASSERT -c %s -o %t.o
// RUN: %{blackhole_objdump} -dr %t.o | FileCheck %s --enable-var-scope

// LLK_ASSERT is disabled so runtime paths contain only the issued instructions.

#include <cstdint>

#include "hal/move.h"

namespace move = hal::move;

using DstToSrcB = move::Transfer<move::Dst, move::SrcB>;

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
// CHECK-NEXT: ttmovd2b 1,16,3,0,19
// CHECK-NEXT: ret

extern "C" __attribute__((noinline, used)) void dst_four_rows()
{
    move::run<four_rows>();
}

// CHECK-LABEL: <dst_four_rows>:
// CHECK-NEXT: ttmovd2b 0,60,1,2,200
// CHECK-NEXT: ret

extern "C" __attribute__((noinline, used)) void dst_eight_rows()
{
    move::run<eight_rows>();
}

// CHECK-LABEL: <dst_eight_rows>:
// CHECK-NEXT: ttmovd2b 0,0,4,2,8
// CHECK-NEXT: ttmovd2b 0,0,5,2,8
// CHECK-NEXT: ret

extern "C" __attribute__((noinline, used)) void dst_face_final_mode()
{
    move::run<face_final_mode>();
}

// CHECK-LABEL: <dst_face_final_mode>:
// CHECK-NEXT: ttmovd2b 0,16,0,2,64
// CHECK-NEXT: ttmovd2b 0,16,0,2,64
// CHECK-NEXT: ttmovd2b 0,16,0,2,64
// CHECK-NEXT: ttmovd2b 0,16,3,2,64
// CHECK-NEXT: ret

extern "C" __attribute__((noinline, used)) void dst_two_faces()
{
    move::run<two_faces>();
}

// CHECK-LABEL: <dst_two_faces>:
// CHECK-NEXT: ttmovd2b 0,0,0,2,0
// CHECK-NEXT: ttmovd2b 0,0,0,2,0
// CHECK-NEXT: ttmovd2b 0,0,0,2,0
// CHECK-NEXT: ttmovd2b 0,0,0,2,0
// CHECK-NEXT: ttmovd2b 0,0,0,2,0
// CHECK-NEXT: ttmovd2b 0,0,0,2,0
// CHECK-NEXT: ttmovd2b 0,0,0,2,0
// CHECK-NEXT: ttmovd2b 0,0,3,2,0
// CHECK-NEXT: ret

extern "C" __attribute__((noinline, used)) void dst_counter_relative()
{
    move::run<counter_relative>();
}

// CHECK-LABEL: <dst_counter_relative>:
// CHECK-NEXT: ttmovd2b 0,4,2,2,512
// CHECK-NEXT: ret

extern "C" __attribute__((noinline, used)) void dst_issue_encoded_operations()
{
    TTI_INSN(one_row_low.get_operation());
    TTI_INSN(four_rows.get_operation());
}

// CHECK-LABEL: <dst_issue_encoded_operations>:
// CHECK-NEXT: ttmovd2b 1,16,3,0,19
// CHECK-NEXT: ttmovd2b 0,60,1,2,200
// CHECK-NEXT: ret

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

// CHECK-LABEL: <dst_face_runtime_source>:
// CHECK-NEXT: lui [[R0:a[0-7]]],0xa202
// CHECK-NEXT: lui [[R1:a[0-7]]],0x0
// CHECK-NEXT: R_RISCV_HI20 __instrn_buffer
// CHECK-NEXT: R_RISCV_RELAX
// CHECK-NEXT: mv [[R1]],[[R1]]
// CHECK-NEXT: R_RISCV_LO12_I __instrn_buffer
// CHECK-NEXT: R_RISCV_RELAX
// CHECK-NEXT: or [[R0]],a0,[[R0]]
// CHECK-NEXT: sw [[R0]],0([[R1]])
// CHECK-NEXT: sw [[R0]],0([[R1]])
// CHECK-NEXT: lui [[R2:a[0-7]]],0xa20e
// CHECK-NEXT: sw [[R0]],0([[R1]])
// CHECK-NEXT: or [[R2]],a0,[[R2]]
// CHECK-NEXT: lui [[R0]],0xa802
// CHECK-NEXT: sw [[R2]],0([[R1]])
// CHECK-NEXT: or [[R0]],a0,[[R0]]
// CHECK-NEXT: sw [[R0]],0([[R1]])
// CHECK-NEXT: sw [[R0]],0([[R1]])
// CHECK-NEXT: lui [[R2]],0xa80e
// CHECK-NEXT: sw [[R0]],0([[R1]])
// CHECK-NEXT: or a0,a0,[[R2]]
// CHECK-NEXT: sw a0,0([[R1]])
// CHECK-NEXT: ret

extern "C" __attribute__((noinline, used)) std::uint32_t dst_encode_runtime_source(const std::uint32_t source_row)
{
    return DstToSrcB {.source_row = source_row, .destination_row = 16, .address_mode = 3, .number_of_rows = move::rows::One, .source_32bit_low = true}
        .get_operation();
}

// CHECK-LABEL: <dst_encode_runtime_source>:
// CHECK-NEXT: lui [[R0:a[0-7]]],0xaa0c
// CHECK-NEXT: or a0,a0,[[R0]]
// CHECK-NEXT: ret
