// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// RUN: %{blackhole_tensix_compile} %{blackhole_unpack_thread} -UENABLE_LLK_ASSERT -c %s -o %t.o
// RUN: %{blackhole_objdump} -dr %t.o | FileCheck %s --enable-var-scope

// LLK_ASSERT is disabled so runtime paths contain only the issued instructions.

#include <cstdint>

#include "hal/move.h"

namespace move = hal::move;

using L1ToGpr   = move::Transfer<move::L1, move::Gpr>;
using GprToL1   = move::Transfer<move::Gpr, move::L1>;
using GprToSrcA = move::Transfer<move::Gpr, move::SrcA>;
using GprToSrcB = move::Transfer<move::Gpr, move::SrcB>;

template <move::ScalarSize Size, move::OffsetIncrement Increment, move::GprHalf Half>
constexpr L1ToGpr load {
    .source      = {.base = hal::gpr<1>(), .offset = {.gpr = hal::gpr<2>(), .half = Half}, .offset_increment = Increment},
    .destination = hal::gpr<4>(),
    .size        = Size,
};

extern "C" __attribute__((noinline, used)) void l1_to_gpr_sizes()
{
    move::run<load<move::ScalarSize::Bytes16, move::OffsetIncrement::None, move::GprHalf::Low>>();
    move::run<load<move::ScalarSize::Bytes4, move::OffsetIncrement::Bytes4, move::GprHalf::High>>();
    move::run<load<move::ScalarSize::Bytes2, move::OffsetIncrement::Bytes2, move::GprHalf::Low>>();
    move::run<load<move::ScalarSize::Bytes1, move::OffsetIncrement::Bytes16, move::GprHalf::High>>();
}

// CHECK-LABEL: <l1_to_gpr_sizes>:
// CHECK-NEXT: ttloadind 0,4,0,4,1
// CHECK-NEXT: ttloadind 1,5,2,4,1
// CHECK-NEXT: ttloadind 2,4,1,4,1
// CHECK-NEXT: ttloadind 3,5,3,4,1
// CHECK-NEXT: ret

constexpr L1ToGpr load_wait {
    .source      = {.base = hal::gpr<63>(), .offset = {.gpr = hal::gpr<63>(), .half = move::GprHalf::High}},
    .destination = hal::gpr<60>(),
    .completion  = move::Completion::Wait,
};

extern "C" __attribute__((noinline, used)) void l1_to_gpr_wait()
{
    move::run<load_wait>();
}

// CHECK-LABEL: <l1_to_gpr_wait>:
// CHECK-NEXT: ttloadind 0,127,0,60,63
// CHECK-NEXT: ttstallwait 511,1
// CHECK-NEXT: ret

template <move::ScalarSize Size, move::OffsetIncrement Increment>
constexpr GprToL1 store {
    .source      = hal::gpr<8>(),
    .destination = {.base = hal::gpr<5>(), .offset = {.gpr = hal::gpr<6>(), .half = move::GprHalf::Low}, .offset_increment = Increment},
    .size        = Size,
};

extern "C" __attribute__((noinline, used)) void gpr_to_l1_sizes()
{
    move::run<store<move::ScalarSize::Bytes16, move::OffsetIncrement::Bytes16>>();
    move::run<store<move::ScalarSize::Bytes4, move::OffsetIncrement::None>>();
    move::run<store<move::ScalarSize::Bytes2, move::OffsetIncrement::Bytes2>>();
    move::run<store<move::ScalarSize::Bytes1, move::OffsetIncrement::Bytes4>>();
}

// CHECK-LABEL: <gpr_to_l1_sizes>:
// CHECK-NEXT: ttstoreind 1,0,0,12,3,8,5
// CHECK-NEXT: ttstoreind 1,0,1,12,0,8,5
// CHECK-NEXT: ttstoreind 1,1,0,12,1,8,5
// CHECK-NEXT: ttstoreind 1,1,1,12,2,8,5
// CHECK-NEXT: ret

constexpr GprToL1 store_wait {
    .source      = hal::gpr<4>(),
    .destination = {.base = hal::gpr<5>(), .offset = {.gpr = hal::gpr<6>(), .half = move::GprHalf::Low}, .offset_increment = move::OffsetIncrement::Bytes16},
    .size        = move::ScalarSize::Bytes2,
    .completion  = move::Completion::Wait,
};

extern "C" __attribute__((noinline, used)) void gpr_to_l1_wait()
{
    move::run<store_wait>();
}

// CHECK-LABEL: <gpr_to_l1_wait>:
// CHECK-NEXT: ttstoreind 1,1,0,12,3,4,5
// CHECK-NEXT: ttstallwait 511,1
// CHECK-NEXT: ret

constexpr GprToSrcA store_srca {
    .source      = hal::gpr<8>(),
    .destination = {.base = hal::gpr<9>(), .offset = {.gpr = hal::gpr<10>(), .half = move::GprHalf::High}, .offset_increment = move::OffsetIncrement::Bytes2},
};

constexpr GprToSrcB store_srcb {
    .source      = hal::gpr<12>(),
    .destination = {.base = hal::gpr<13>(), .offset = {.gpr = hal::gpr<14>(), .half = move::GprHalf::Low}},
};

extern "C" __attribute__((noinline, used)) void gpr_to_src()
{
    move::run<store_srca>();
    move::run<store_srcb>();
}

// CHECK-LABEL: <gpr_to_src>:
// CHECK-NEXT: ttstoreind 0,0,0,21,1,8,9
// CHECK-NEXT: ttstoreind 0,0,1,28,0,12,13
// CHECK-NEXT: ret

extern "C" __attribute__((noinline, used)) void scalar_issue_encoded_operations()
{
    TTI_INSN((load<move::ScalarSize::Bytes4, move::OffsetIncrement::Bytes4, move::GprHalf::High>.get_operation()));
    TTI_INSN((store<move::ScalarSize::Bytes1, move::OffsetIncrement::Bytes4>.get_operation()));
    TTI_INSN(store_srca.get_operation());
    TTI_INSN(store_srcb.get_operation());
}

// CHECK-LABEL: <scalar_issue_encoded_operations>:
// CHECK-NEXT: ttloadind 1,5,2,4,1
// CHECK-NEXT: ttstoreind 1,1,1,12,2,8,5
// CHECK-NEXT: ttstoreind 0,0,0,21,1,8,9
// CHECK-NEXT: ttstoreind 0,0,1,28,0,12,13
// CHECK-NEXT: ret

extern "C" __attribute__((noinline, used)) void l1_to_gpr_runtime(const std::uint32_t base, const std::uint32_t offset, const std::uint32_t destination)
{
    move::run(L1ToGpr {
        .source =
            {
                .base             = hal::gpr(base),
                .offset           = {.gpr = hal::gpr(offset), .half = move::GprHalf::High},
                .offset_increment = move::OffsetIncrement::Bytes4,
            },
        .destination = hal::gpr(destination),
        .size        = move::ScalarSize::Bytes4,
    });
}

// CHECK-LABEL: <l1_to_gpr_runtime>:
// CHECK-NEXT: slli a1,a1,0x1
// CHECK-NEXT: addi a1,a1,1
// CHECK-NEXT: slli a1,a1,0xe
// CHECK-NEXT: slli a2,a2,0x6
// CHECK-NEXT: or a1,a1,a2
// CHECK-NEXT: or a1,a1,a0
// CHECK-NEXT: lui [[R0:a[0-7]]],0x49402
// CHECK-NEXT: lui [[R1:a[0-7]]],0x0
// CHECK-NEXT: R_RISCV_HI20 __instrn_buffer
// CHECK-NEXT: R_RISCV_RELAX
// CHECK-NEXT: or a1,a1,[[R0]]
// CHECK-NEXT: mv [[R1]],[[R1]]
// CHECK-NEXT: R_RISCV_LO12_I __instrn_buffer
// CHECK-NEXT: R_RISCV_RELAX
// CHECK-NEXT: sw a1,0([[R1]])
// CHECK-NEXT: ret

extern "C" __attribute__((noinline, used)) void gpr_to_l1_runtime_completion(const std::uint32_t source, const move::Completion completion)
{
    move::run(GprToL1 {
        .source      = hal::gpr(source),
        .destination = {.base = hal::gpr<5>(), .offset = {.gpr = hal::gpr<6>()}},
        .size        = move::ScalarSize::Bytes2,
        .completion  = completion,
    });
}

// CHECK-LABEL: <gpr_to_l1_runtime_completion>:
// CHECK-NEXT: lui [[R0:a[0-7]]],0x66c30
// CHECK-NEXT: addi [[R1:a[0-7]]],[[R0]],5
// CHECK-NEXT: slli a0,a0,0x6
// CHECK-NEXT: lui [[R0]],0x0
// CHECK-NEXT: R_RISCV_HI20 __instrn_buffer
// CHECK-NEXT: R_RISCV_RELAX
// CHECK-NEXT: or a0,a0,[[R1]]
// CHECK-NEXT: mv [[R0]],[[R0]]
// CHECK-NEXT: R_RISCV_LO12_I __instrn_buffer
// CHECK-NEXT: R_RISCV_RELAX
// CHECK-NEXT: sw a0,0([[R0]])
// CHECK-NEXT: li [[R1]],1
// CHECK-NEXT: beq a1,[[R1]],{{[0-9a-f]+}}
// CHECK-NEXT: R_RISCV_BRANCH
// CHECK-NEXT: ret
// CHECK-EMPTY:
// CHECK-NEXT: <.L{{[0-9]+}}>:
// CHECK-NEXT: lui [[R1]],0xa2ff8
// CHECK-NEXT: add [[R1]],[[R1]],a1
// CHECK-NEXT: sw [[R1]],0([[R0]])
// CHECK-NEXT: ret

extern "C" __attribute__((noinline, used)) void gpr_to_srcb_runtime(const std::uint32_t source)
{
    move::run(GprToSrcB {.source = hal::gpr(source), .destination = {.base = hal::gpr<13>(), .offset = {.gpr = hal::gpr<14>()}}});
}

// CHECK-LABEL: <gpr_to_srcb_runtime>:
// CHECK-NEXT: lui [[R0:a[0-7]]],0x66270
// CHECK-NEXT: addi [[R1:a[0-7]]],[[R0]],13
// CHECK-NEXT: slli a0,a0,0x6
// CHECK-NEXT: lui [[R0]],0x0
// CHECK-NEXT: R_RISCV_HI20 __instrn_buffer
// CHECK-NEXT: R_RISCV_RELAX
// CHECK-NEXT: or a0,a0,[[R1]]
// CHECK-NEXT: mv [[R0]],[[R0]]
// CHECK-NEXT: R_RISCV_LO12_I __instrn_buffer
// CHECK-NEXT: R_RISCV_RELAX
// CHECK-NEXT: sw a0,0([[R0]])
// CHECK-NEXT: ret

extern "C" __attribute__((noinline, used)) std::uint32_t l1_to_gpr_encode_runtime(const std::uint32_t destination)
{
    return L1ToGpr {.source = {.base = hal::gpr<1>(), .offset = {.gpr = hal::gpr<2>()}}, .destination = hal::gpr(destination), .size = move::ScalarSize::Bytes1}
        .get_operation();
}

// CHECK-LABEL: <l1_to_gpr_encode_runtime>:
// CHECK-NEXT: lui [[R0:a[0-7]]],0x49c10
// CHECK-NEXT: addi [[R0]],[[R0]],1
// CHECK-NEXT: slli a0,a0,0x6
// CHECK-NEXT: or a0,a0,[[R0]]
// CHECK-NEXT: ret
