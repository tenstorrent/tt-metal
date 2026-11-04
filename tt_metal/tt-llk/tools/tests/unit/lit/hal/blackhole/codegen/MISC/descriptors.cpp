// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// RUN: %split-file %s %t
// RUN: %{blackhole_tensix_compile} %{blackhole_unpack_thread} -fno-ipa-icf -c %t/codegen.cpp -o %t/unpack.o
// RUN: %{blackhole_objdump} -drz %t/unpack.o > %t/unpack.dump
// RUN: %{blackhole_compare_codegen} %t/unpack.dump
// RUN: %{blackhole_tensix_compile} %{blackhole_math_thread} -fno-ipa-icf -c %t/codegen.cpp -o %t/math.o
// RUN: %{blackhole_objdump} -drz %t/math.o > %t/math.dump
// RUN: %{blackhole_compare_codegen} %t/math.dump
// RUN: %{blackhole_tensix_compile} %{blackhole_pack_thread} -fno-ipa-icf -c %t/codegen.cpp -o %t/pack.o
// RUN: %{blackhole_objdump} -drz %t/pack.o > %t/pack.dump
// RUN: %{blackhole_compare_codegen} %t/pack.dump
// RUN: %{blackhole_tensix_compile} %{blackhole_unpack_thread} -DENABLE_LLK_ASSERT -c %t/asserts.cpp -o %t/assert.o
// RUN: %{blackhole_objdump} -d %t/assert.o | FileCheck %s --check-prefix=ASSERT

//--- common.h
#pragma once

#include <cstdint>

#include "hal/misc.h"
#include "utils/helpers.h"

namespace hm = hal::misc;

//--- codegen.cpp
#include "common.h"

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
    TTI_FLUSHDMA(0x8 /*FlushSpec*/);
    TTI_RSTDMA;
    TTI_TBUFCMD;
    TTI_RESOURCEDECL(7 /*linger_time*/, 5 /*resources*/, 3 /*op_class*/);
}

extern "C" __attribute__((noinline, used)) void issue_runtime_misc_descriptor(hm::ResourceDeclaration descriptor)
{
    hm::run(descriptor);
}

extern "C" __attribute__((noinline, used)) void reference_issue_runtime_misc_descriptor(hm::ResourceDeclaration descriptor)
{
    TT_RESOURCEDECL(descriptor.linger_time, descriptor.resources, descriptor.instruction_class);
}

//--- asserts.cpp
#include "common.h"

// Each invalid runtime operand must emit its own ebreak. One probe per check
// keeps the expectation independent of how many checks an operand violates.

extern "C" void reject_invalid_flush_scope()
{
    (void)hm::FlushTdma {static_cast<hm::FlushScope>(16)}.operation();
}

// ASSERT-LABEL: <reject_invalid_flush_scope>:
// ASSERT: ebreak
// ASSERT: ret

extern "C" void reject_all_combined_with_component()
{
    (void)(hm::FlushScope::All | hm::FlushScope::Packer);
}

// ASSERT-LABEL: <reject_all_combined_with_component>:
// ASSERT: ebreak
// ASSERT: ret

extern "C" void reject_component_combined_with_all()
{
    (void)(hm::FlushScope::Packer | hm::FlushScope::All);
}

// ASSERT-LABEL: <reject_component_combined_with_all>:
// ASSERT: ebreak
// ASSERT: ret

extern "C" void reject_invalid_instruction_class()
{
    (void)hm::ResourceDeclaration {16, 0, 1}.operation();
}

// ASSERT-LABEL: <reject_invalid_instruction_class>:
// ASSERT: ebreak
// ASSERT: ret

extern "C" void reject_invalid_resources()
{
    (void)hm::ResourceDeclaration {0, 512, 1}.operation();
}

// ASSERT-LABEL: <reject_invalid_resources>:
// ASSERT: ebreak
// ASSERT: ret

extern "C" void reject_invalid_linger_time()
{
    (void)hm::ResourceDeclaration {0, 0, 16}.operation();
}

// ASSERT-LABEL: <reject_invalid_linger_time>:
// ASSERT: ebreak
// ASSERT: ret
