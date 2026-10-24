// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// RUN: %{blackhole_tensix_compile} %{blackhole_unpack_thread} -c %s -o %t.o
// RUN: %{blackhole_objdump} -d %t.o | FileCheck %s --enable-var-scope

#include <cstdint>

#include "hal/nop.h"

namespace nop = hal::nop;

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
// CHECK-NEXT: ttunpacr_nop 0,0,0,0,0,0,0,0,2
// CHECK-NEXT: ttunpacr_nop 1,0,0,0,0,0,0,0,2
// CHECK-NEXT: ret

extern "C" __attribute__((noinline, used)) void store_encoded_nops(std::uint32_t* words)
{
    words[0] = nop::thread_operation();
    words[1] = nop::scalar_operation();
    words[2] = nop::vector_operation();
    words[3] = nop::unpacker_operation<hal::unpack::Engine::Unpacker1>();
}

// CHECK-LABEL: <store_encoded_nops>:
// CHECK-DAG: lui [[THREAD:a[0-7]]],0x2000
// CHECK-DAG: lui [[SCALAR:a[0-7]]],0x60000
// CHECK-DAG: lui [[VECTOR:a[0-7]]],0x8f000
// CHECK-DAG: lui [[UNPACKER:a[0-7]]],0x43800
// CHECK-DAG: addi [[UNPACKER]],[[UNPACKER]],2
// CHECK-DAG: sw [[THREAD]],0(a0)
// CHECK-DAG: sw [[SCALAR]],4(a0)
// CHECK-DAG: sw [[VECTOR]],8(a0)
// CHECK-DAG: sw [[UNPACKER]],12(a0)
// CHECK: ret
