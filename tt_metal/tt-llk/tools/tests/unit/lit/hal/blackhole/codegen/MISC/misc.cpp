// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// RUN: %{blackhole_tensix_compile} %{blackhole_math_thread} -UENABLE_LLK_ASSERT -c %s -o %t.o
// RUN: %{blackhole_objdump} -dr %t.o | FileCheck %s --enable-var-scope

#include "hal/misc.h"

namespace misc = hal::misc;

extern "C" __attribute__((noinline, used)) void flush_tdma_scopes()
{
    misc::flush_tdma();
    misc::flush_tdma<misc::FlushScope::ThreadController>();
    misc::flush_tdma<misc::FlushScope::Unpacker0 | misc::FlushScope::Unpacker1>();
    misc::flush_tdma<misc::FlushScope::Packer>();
}

// CHECK-LABEL: <flush_tdma_scopes>:
// CHECK-NEXT: ttflushdma 0
// CHECK-NEXT: ttflushdma 1
// CHECK-NEXT: ttflushdma 6
// CHECK-NEXT: ttflushdma 8
// CHECK-NEXT: ret

extern "C" __attribute__((noinline, used)) void reset_and_tbuf()
{
    misc::reset_tdma();
    misc::tbuf_command();
}

// CHECK-LABEL: <reset_and_tbuf>:
// CHECK-NEXT: ttrstdma
// CHECK-NEXT: tttbufcmd
// CHECK-NEXT: ret

extern "C" __attribute__((noinline, used)) void issue_encoded_operations()
{
    TTI_INSN(misc::flush_tdma_operation<misc::FlushScope::Unpacker0 | misc::FlushScope::Packer>());
    TTI_INSN(misc::reset_tdma_operation());
    TTI_INSN(misc::tbuf_command_operation());
    TTI_INSN((misc::ResourceDeclaration {5, 0x101, 9}.get_operation()));
}

// CHECK-LABEL: <issue_encoded_operations>:
// CHECK-NEXT: ttflushdma 10
// CHECK-NEXT: ttrstdma
// CHECK-NEXT: tttbufcmd
// CHECK-NEXT: ttresourcedecl 9,257,5
// CHECK-NEXT: ret

extern "C" __attribute__((noinline, used)) void declare_resources()
{
    misc::run<misc::ResourceDeclaration {}>();
    misc::run<misc::ResourceDeclaration {3, 0x1ff, 7}>();
    misc::run<misc::ResourceDeclaration {15, 0, 15}>();
}

// The default declaration is class 0, no resources, and a one-cycle linger.
// CHECK-LABEL: <declare_resources>:
// CHECK-NEXT: ttresourcedecl 1,0,0
// CHECK-NEXT: ttresourcedecl 7,511,3
// CHECK-NEXT: ttresourcedecl 15,0,15
// CHECK-NEXT: ret

static_assert(misc::is_valid(misc::ResourceDeclaration {15, 0x1ff, 15}));
static_assert(!misc::is_valid(misc::ResourceDeclaration {16, 0, 1}));
static_assert(!misc::is_valid(misc::ResourceDeclaration {0, 0x200, 1}));
static_assert(!misc::is_valid(misc::ResourceDeclaration {0, 0, 16}));
