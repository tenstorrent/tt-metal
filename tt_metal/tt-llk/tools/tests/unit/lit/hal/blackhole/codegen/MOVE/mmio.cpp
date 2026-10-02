// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// RUN: %{blackhole_tensix_compile} %{blackhole_pack_thread} -UENABLE_LLK_ASSERT -c %s -o %t.o
// RUN: %{blackhole_objdump} -dr %t.o | FileCheck %s --enable-var-scope

// LLK_ASSERT is disabled so runtime paths contain only the issued instructions.

#include <cstdint>

#include "hal/move.h"

namespace move = hal::move;

using GprToMmio = move::Transfer<move::Gpr, move::Mmio>;
using MmioToGpr = move::Transfer<move::Mmio, move::Gpr>;

constexpr GprToMmio store_first {
    .source      = hal::gpr<15>(),
    .destination = {.addressing = move::MmioAddressing::Immediate, .byte_address = 0xffb11000u},
};

constexpr GprToMmio store_last_wait {
    .source      = hal::gpr<63>(),
    .destination = {.addressing = move::MmioAddressing::Immediate, .byte_address = 0xffbffffcu},
    .completion  = move::Completion::Wait,
};

constexpr GprToMmio store_indirect {
    .source = hal::gpr<16>(),
    .destination =
        {
            .addressing       = move::MmioAddressing::Indirect,
            .base             = hal::gpr<17>(),
            .offset           = {.gpr = hal::gpr<18>(), .half = move::GprHalf::High},
            .offset_increment = move::OffsetIncrement::Bytes4,
        },
};

constexpr GprToMmio store_indirect_wait {
    .source      = hal::gpr<0>(),
    .destination = {.addressing = move::MmioAddressing::Indirect, .base = hal::gpr<1>(), .offset = {.gpr = hal::gpr<2>()}},
    .completion  = move::Completion::Wait,
};

constexpr MmioToGpr load_last {
    .source      = {.addressing = move::MmioAddressing::Immediate, .byte_address = 0xffbffffcu},
    .destination = hal::gpr<19>(),
};

constexpr MmioToGpr load_first_wait {
    .source      = {.addressing = move::MmioAddressing::Immediate, .byte_address = 0xffb11000u},
    .destination = hal::gpr<0>(),
    .completion  = move::Completion::Wait,
};

extern "C" __attribute__((noinline, used)) void gpr_to_mmio_immediate()
{
    move::run<store_first>();
    move::run<store_last_wait>();
}

// CHECK-LABEL: <gpr_to_mmio_immediate>:
// CHECK-NEXT: ttstorereg 15,17408
// CHECK-NEXT: ttstorereg 63,262143
// CHECK-NEXT: ttstallwait 511,1
// CHECK-NEXT: ret

extern "C" __attribute__((noinline, used)) void gpr_to_mmio_indirect()
{
    move::run<store_indirect>();
    move::run<store_indirect_wait>();
}

// CHECK-LABEL: <gpr_to_mmio_indirect>:
// CHECK-NEXT: ttstoreind 0,1,0,37,2,16,17
// CHECK-NEXT: ttstoreind 0,1,0,4,0,0,1
// CHECK-NEXT: ttstallwait 511,1
// CHECK-NEXT: ret

extern "C" __attribute__((noinline, used)) void mmio_to_gpr()
{
    move::run<load_last>();
    move::run<load_first_wait>();
}

// CHECK-LABEL: <mmio_to_gpr>:
// CHECK-NEXT: ttloadreg 19,262143
// CHECK-NEXT: ttloadreg 0,17408
// CHECK-NEXT: ttstallwait 511,1
// CHECK-NEXT: ret

extern "C" __attribute__((noinline, used)) void mmio_issue_encoded_operations()
{
    TTI_INSN(store_first.get_operation());
    TTI_INSN(store_indirect.get_operation());
    TTI_INSN(load_last.get_operation());
}

// CHECK-LABEL: <mmio_issue_encoded_operations>:
// CHECK-NEXT: ttstorereg 15,17408
// CHECK-NEXT: ttstoreind 0,1,0,37,2,16,17
// CHECK-NEXT: ttloadreg 19,262143
// CHECK-NEXT: ret

extern "C" __attribute__((noinline, used)) void gpr_to_mmio_runtime_address(const std::uint32_t byte_address)
{
    move::run(GprToMmio {.source = hal::gpr<15>(), .destination = {.addressing = move::MmioAddressing::Immediate, .byte_address = byte_address}});
}

// CHECK-LABEL: <gpr_to_mmio_runtime_address>:
// CHECK-NEXT: lui [[R0:a[0-7]]],0x500
// CHECK-NEXT: add a0,a0,[[R0]]
// CHECK-NEXT: srli a0,a0,0x2
// CHECK-NEXT: lui [[R1:a[0-7]]],0x673c0
// CHECK-NEXT: lui [[R0]],0x0
// CHECK-NEXT: R_RISCV_HI20 __instrn_buffer
// CHECK-NEXT: R_RISCV_RELAX
// CHECK-NEXT: or a0,a0,[[R1]]
// CHECK-NEXT: mv [[R0]],[[R0]]
// CHECK-NEXT: R_RISCV_LO12_I __instrn_buffer
// CHECK-NEXT: R_RISCV_RELAX
// CHECK-NEXT: sw a0,0([[R0]])
// CHECK-NEXT: ret

extern "C" __attribute__((noinline, used)) void gpr_to_mmio_runtime_addressing(const move::MmioAddressing addressing)
{
    move::run(GprToMmio {
        .source      = hal::gpr<16>(),
        .destination = {.addressing = addressing, .byte_address = 0xffb11000u, .base = hal::gpr<17>(), .offset = {.gpr = hal::gpr<18>()}},
    });
}

// CHECK-LABEL: <gpr_to_mmio_runtime_addressing>:
// CHECK-NEXT: lui [[R0:a[0-7]]],0x66490
// CHECK-NEXT: li [[R1:a[0-7]]],1
// CHECK-NEXT: addi [[R0]],[[R0]],1041
// CHECK-NEXT: bne a0,[[R1]],{{[0-9a-f]+}}
// CHECK-NEXT: R_RISCV_BRANCH
// CHECK-NEXT: lui [[R0]],0x67404
// CHECK-NEXT: addi [[R0]],[[R0]],1024
// CHECK-EMPTY:
// CHECK-NEXT: <.L{{[0-9]+}}>:
// CHECK-NEXT: lui [[R1]],0x0
// CHECK-NEXT: R_RISCV_HI20 __instrn_buffer
// CHECK-NEXT: R_RISCV_RELAX
// CHECK-NEXT: mv [[R1]],[[R1]]
// CHECK-NEXT: R_RISCV_LO12_I __instrn_buffer
// CHECK-NEXT: R_RISCV_RELAX
// CHECK-NEXT: sw [[R0]],0([[R1]])
// CHECK-NEXT: ret

extern "C" __attribute__((noinline, used)) void mmio_to_gpr_runtime_wait(const std::uint32_t destination)
{
    move::run(MmioToGpr {
        .source      = {.addressing = move::MmioAddressing::Immediate, .byte_address = 0xffb12000u},
        .destination = hal::gpr(destination),
        .completion  = move::Completion::Wait,
    });
}

// CHECK-LABEL: <mmio_to_gpr_runtime_wait>:
// CHECK-NEXT: lui [[R0:a[0-7]]],0x68005
// CHECK-NEXT: addi [[R1:a[0-7]]],[[R0]],-2048
// CHECK-NEXT: slli a0,a0,0x12
// CHECK-NEXT: lui [[R0]],0x0
// CHECK-NEXT: R_RISCV_HI20 __instrn_buffer
// CHECK-NEXT: R_RISCV_RELAX
// CHECK-NEXT: or a0,a0,[[R1]]
// CHECK-NEXT: mv [[R0]],[[R0]]
// CHECK-NEXT: R_RISCV_LO12_I __instrn_buffer
// CHECK-NEXT: R_RISCV_RELAX
// CHECK-NEXT: lui [[R1]],0xa2ff8
// CHECK-NEXT: sw a0,0([[R0]])
// CHECK-NEXT: addi [[R1]],[[R1]],1
// CHECK-NEXT: sw [[R1]],0([[R0]])
// CHECK-NEXT: ret
