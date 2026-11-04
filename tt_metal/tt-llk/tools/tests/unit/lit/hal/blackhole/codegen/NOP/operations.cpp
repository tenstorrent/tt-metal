// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// RUN: %split-file %s %t
// RUN: %{blackhole_tensix_compile} %{blackhole_unpack_thread} -c %t/compile_time.cpp -o %t/compile_time.o
// RUN: %{blackhole_objdump} -d %t/compile_time.o | FileCheck %s
// RUN: %{blackhole_tensix_compile} %{blackhole_unpack_thread} -UENABLE_LLK_ASSERT -fno-ipa-icf -c %t/codegen.cpp -o %t/codegen.o
// RUN: %{blackhole_objdump} -drz %t/codegen.o > %t/codegen.dump
// RUN: %{blackhole_compare_codegen} %t/codegen.dump

// Compare each HAL function in codegen.cpp with its adjacent reference_
// function. Disable identical-code folding so both bodies are emitted
// independently.

//--- common.h
#pragma once

#include <cstdint>

#include "hal/nop.h"

namespace nop = hal::nop;

//--- compile_time.cpp
#include "common.h"

extern "C" __attribute__((noinline, used)) void issue_every_nop()
{
    nop::thread();
    nop::scalar();
    nop::vector();
    nop::unpacker<hal::unpack::Engine::Unpacker0>();
    nop::unpacker<hal::unpack::Engine::Unpacker1>();
}

// CHECK-LABEL: <issue_every_nop>:
// CHECK-NEXT: ttnop
// CHECK-NEXT: ttdmanop
// CHECK-NEXT: sfpnop
// CHECK-NEXT: ttunpacr_nop 0,0,0,0,0,0,0,0,2{{$}}
// CHECK-NEXT: ttunpacr_nop 1,0,0,0,0,0,0,0,2{{$}}
// CHECK-NEXT: ret

//--- codegen.cpp
#include "common.h"

extern "C" __attribute__((noinline, used)) void store_encoded_nops(std::uint32_t* words)
{
    words[0] = nop::thread_operation();
    words[1] = nop::scalar_operation();
    words[2] = nop::vector_operation();
    words[3] = nop::unpacker_operation<hal::unpack::Engine::Unpacker1>();
}

extern "C" __attribute__((noinline, used)) void reference_store_encoded_nops(std::uint32_t* words)
{
    words[0] = TT_OP_NOP;
    words[1] = TT_OP_DMANOP;
    words[2] = TT_OP_SFPNOP;
    words[3] = TT_OP_UNPACR_NOP(
        1 /*Unpacker_Select*/,
        0 /*Stream_Id*/,
        0 /*Msg_Clr_Cnt*/,
        0 /*Set_Dvalid*/,
        0 /*Clr_to1_fmt_Ctrl*/,
        0 /*Stall_Clr_Cntrl*/,
        0 /*Bank_Clr_Ctrl*/,
        0 /*Src_ClrVal_Ctrl*/,
        2 /*Unpack_Pop*/);
}
