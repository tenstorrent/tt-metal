// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// RUN: %{blackhole_tensix_compile} %{blackhole_unpack_thread} -fno-ipa-icf -c %s -o %t.unpack.o
// RUN: %{blackhole_objdump} -t --special-syms -drz %t.unpack.o > %t.unpack.dump
// RUN: %{blackhole_compare_codegen} %t.unpack.dump
// RUN: %{blackhole_tensix_compile} %{blackhole_math_thread} -fno-ipa-icf -c %s -o %t.math.o
// RUN: %{blackhole_objdump} -t --special-syms -drz %t.math.o > %t.math.dump
// RUN: %{blackhole_compare_codegen} %t.math.dump
// RUN: %{blackhole_tensix_compile} %{blackhole_pack_thread} -fno-ipa-icf -c %s -o %t.pack.o
// RUN: %{blackhole_objdump} -t --special-syms -drz %t.pack.o > %t.pack.dump
// RUN: %{blackhole_compare_codegen} %t.pack.dump
// RUN: %{blackhole_tensix_compile} %{blackhole_unpack_thread} -DENABLE_LLK_ASSERT -c %s -o %t.assert.o
// RUN: %{blackhole_objdump} -d %t.assert.o | FileCheck %s --check-prefix=ASSERT

#include <cstdint>

#include "hal/misc.h"
#include "utils/helpers.h"

namespace hm = hal::misc;

// Encoding must match raw instruction words without issuing instructions or MMIO.

extern "C" __attribute__((noinline, used)) std::uint32_t encode_flush_tdma(hm::FlushScope scope)
{
    return hm::FlushTdma {scope}.operation();
}

extern "C" __attribute__((noinline, used)) std::uint32_t reference_encode_flush_tdma(hm::FlushScope scope)
{
    return TT_OP_FLUSHDMA(hal::to_underlying(scope));
}

extern "C" __attribute__((noinline, used)) std::uint32_t encode_reset_tdma()
{
    return hm::ResetTdma {}.operation();
}

extern "C" __attribute__((noinline, used)) std::uint32_t reference_encode_reset_tdma()
{
    return TT_OP_RSTDMA;
}

extern "C" __attribute__((noinline, used)) std::uint32_t encode_tbuf_command()
{
    return hm::TbufCommand {}.operation();
}

extern "C" __attribute__((noinline, used)) std::uint32_t reference_encode_tbuf_command()
{
    return TT_OP_TBUFCMD;
}

extern "C" __attribute__((noinline, used)) std::uint32_t encode_resource_declaration(hm::ResourceDeclaration descriptor)
{
    return descriptor.operation();
}

extern "C" __attribute__((noinline, used)) std::uint32_t reference_encode_resource_declaration(hm::ResourceDeclaration descriptor)
{
    return TT_OP_RESOURCEDECL(descriptor.linger_time, descriptor.resources, descriptor.instruction_class);
}

extern "C" __attribute__((noinline, used)) void issue_misc_descriptors()
{
    hm::run<hm::FlushTdma {hm::FlushScope::Packer}>();
    hm::run<hm::ResetTdma {}>();
    hm::run<hm::TbufCommand {}>();
    hm::run<hm::ResourceDeclaration {3, 5, 7}>();
}

extern "C" __attribute__((noinline, used)) void reference_issue_misc_descriptors()
{
    TTI_FLUSHDMA(8);
    TTI_RSTDMA;
    TTI_TBUFCMD;
    TTI_RESOURCEDECL(7, 5, 3);
}

extern "C" __attribute__((noinline, used)) void issue_runtime_misc_descriptor(hm::ResourceDeclaration descriptor)
{
    hm::run(descriptor);
}

extern "C" __attribute__((noinline, used)) void reference_issue_runtime_misc_descriptor(hm::ResourceDeclaration descriptor)
{
    TT_RESOURCEDECL(descriptor.linger_time, descriptor.resources, descriptor.instruction_class);
}

#ifdef ENABLE_LLK_ASSERT

extern "C" void reject_invalid_runtime_descriptors()
{
    (void)hm::FlushTdma {static_cast<hm::FlushScope>(16)}.operation();
    (void)hm::ResourceDeclaration {16, 0, 1}.operation();
    (void)hm::ResourceDeclaration {0, 512, 1}.operation();
    (void)hm::ResourceDeclaration {0, 0, 2048}.operation();
}

// ASSERT-LABEL: <reject_invalid_runtime_descriptors>:
// ASSERT-COUNT-4: ebreak
// ASSERT-NOT: ebreak
// ASSERT-NOT: sw
// ASSERT: ret

#endif
