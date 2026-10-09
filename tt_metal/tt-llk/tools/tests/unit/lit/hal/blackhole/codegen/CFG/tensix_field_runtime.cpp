// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// RUN: %{blackhole_tensix_compile} %{blackhole_unpack_thread} -fno-ipa-icf -c %s -o %t.o
// RUN: %{blackhole_objdump} -t --special-syms -drz %t.o > %t.dump
// RUN: %{blackhole_compare_codegen} %t.dump
// RUN: %{blackhole_tensix_compile} %{blackhole_unpack_thread} -DENABLE_LLK_ASSERT -c %s -o %t.assert.o
// RUN: %{blackhole_objdump} -d %t.assert.o | FileCheck %s --check-prefix=ASSERT

// Each HAL function is compared with its adjacent reference_ function.
// Disable identical-code folding so both bodies must be emitted independently.

#include <cstdint>

#include "hal/cfg.h"

namespace cfg = hal::cfg;

// A runtime value cannot be an immediate, so its RMWCIB/SETC16 is built in a
// register and pushed through the instruction buffer. The operation word holds
// the opcode, mask, and address; the runtime data lands in bits 15:8 (RMWCIB)
// or 15:0 (SETC16).

// RMWCIB0, mask 0x0f, word 64.
extern "C" __attribute__((noinline, used)) void write_runtime_state_byte_0(std::uint32_t format)
{
    cfg::write<cfg::Access::TensixCfgUnit, cfg::Thcon[cfg::Reg0].TileDescriptor.InDataFormat, cfg::Sec::S0>(format);
}

extern "C" __attribute__((noinline, used)) void reference_write_runtime_state_byte_0(std::uint32_t format)
{
    TT_RMWCIB0(0x0f, format & 0xffu, 64);
}

// SrcB_val (bits 8:5) needs RMWCIB0 with mask 0xe0 and RMWCIB1 with mask 0x01, both at word 0.
extern "C" __attribute__((noinline, used)) void write_runtime_state_byte_straddle(std::uint32_t format)
{
    cfg::write<cfg::Access::TensixCfgUnit, cfg::AluFormatSpecReg::SrcB_val, cfg::Sec::S0>(format);
}

extern "C" __attribute__((noinline, used)) void reference_write_runtime_state_byte_straddle(std::uint32_t format)
{
    TT_RMWCIB0(0xe0, (format << 5) & 0xffu, 0);
    TT_RMWCIB1(0x01, (format >> 3) & 0xffu, 0);
}

// Each byte lane gets its own RMWCIB at word 186.
extern "C" __attribute__((noinline, used)) void write_runtime_state_full_word(std::uint32_t seed)
{
    cfg::write<cfg::Access::TensixCfgUnit, cfg::PrngSeed::Seed_Val, cfg::Sec::S0>(seed);
}

extern "C" __attribute__((noinline, used)) void reference_write_runtime_state_full_word(std::uint32_t seed)
{
    TT_RMWCIB0(0xff, seed & 0xffu, 186);
    TT_RMWCIB1(0xff, (seed >> 8) & 0xffu, 186);
    TT_RMWCIB2(0xff, (seed >> 16) & 0xffu, 186);
    TT_RMWCIB3(0xff, seed >> 24, 186);
}

// SETC16 to thread word 19 with the value shifted to bits 13:8.
extern "C" __attribute__((noinline, used)) void write_runtime_thread_section(std::uint32_t increment)
{
    cfg::write<cfg::Access::TensixCfgUnit, cfg::AddrMod[cfg::SrcB].Incr, cfg::Sec::S7>(increment);
}

extern "C" __attribute__((noinline, used)) void reference_write_runtime_thread_section(std::uint32_t increment)
{
    TT_SETC16(19, (increment << 8) & 0xffffu);
}

// Word 64 is emitted once with both runtime fields, before the later constant word.
// Only InDataFormat is masked; Uncompressed is the top field and the RMWCIB byte
// lane clips it.
extern "C" __attribute__((noinline, used)) void write_shuffled_runtime_group(std::uint32_t format, std::uint32_t uncompressed)
{
    cfg::write<cfg::Access::TensixCfgUnit>(
        cfg::set<cfg::Thcon[cfg::Reg0].TileDescriptor.InDataFormat, cfg::Sec::S0>(format),
        cfg::set<cfg::Thcon[cfg::Reg2].Out_data_format, cfg::Sec::S0, 5>(),
        cfg::from_gpr<cfg::Thcon[cfg::Reg3].Base_address, cfg::Sec::S0, cfg::GprTransferSize::Bits128>(hal::gpr<4>()),
        cfg::set<cfg::Thcon[cfg::Reg0].TileDescriptor.Uncompressed, cfg::Sec::S0>(uncompressed),
        cfg::set<cfg::Thcon[cfg::Reg2].Throttle_mode, cfg::Sec::S0, 1>());
}

extern "C" __attribute__((noinline, used)) void reference_write_shuffled_runtime_group(std::uint32_t format, std::uint32_t uncompressed)
{
    TT_RMWCIB0(0x1f, ((format & 0x0fu) | (uncompressed << 4)) & 0xffu, 64);
    TTI_RMWCIB0(0x3f, 0x15, 72);
    TTI_WRCFG(4, 1, 76);
}

// The constant Uncompressed bit (0x1000 = 1 << 4 << 8) joins the runtime format.
extern "C" __attribute__((noinline, used)) void write_mixed_constant_runtime_word(std::uint32_t format)
{
    cfg::write<cfg::Access::TensixCfgUnit>(
        cfg::set<cfg::Thcon[cfg::Reg0].TileDescriptor.Uncompressed, cfg::Sec::S0, 1>(),
        cfg::from_gpr<cfg::Thcon[cfg::Reg3].Base_address, cfg::Sec::S0>(hal::gpr<4>()),
        cfg::set<cfg::Thcon[cfg::Reg0].TileDescriptor.InDataFormat, cfg::Sec::S0>(format));
}

extern "C" __attribute__((noinline, used)) void reference_write_mixed_constant_runtime_word(std::uint32_t format)
{
    TT_RMWCIB0(0x1f, (format & 0x0fu) | 0x10u, 64);
    TTI_WRCFG(4, 0, 76);
}

// Word 2: byte 0 mixes runtime mode with threshold bits. Bytes 1 and 2 are constant.
extern "C" __attribute__((noinline, used)) void write_mixed_relu_bytes(std::uint32_t mode)
{
    cfg::write<cfg::Access::TensixCfgUnit>(
        cfg::set<cfg::StaccRelu::ApplyRelu, cfg::Sec::S0>(mode), cfg::set<cfg::StaccRelu::ReluThreshold, cfg::Sec::S0, 0x1234>());
}

extern "C" __attribute__((noinline, used)) void reference_write_mixed_relu_bytes(std::uint32_t mode)
{
    TT_RMWCIB0(0xfc, (mode & 0x0fu) << 2, 2);
    TTI_RMWCIB1(0xff, 0x8d, 2);
    TTI_RMWCIB2(0x3f, 0x04, 2);
}

// Word 112: preserve byte order with constant bytes on both sides of runtime byte 1.
// Byte 0 must still clear its field when the constant data is zero.
extern "C" __attribute__((noinline, used)) void write_mixed_descriptor_bytes(std::uint32_t blobs)
{
    cfg::write<cfg::Access::TensixCfgUnit>(
        cfg::set<cfg::Thcon[cfg::Reg0].TileDescriptor.XDim, cfg::Sec::S1, 0x1234>(),
        cfg::set<cfg::Thcon[cfg::Reg0].TileDescriptor.BlobsPerXyPlane, cfg::Sec::S1>(blobs),
        cfg::set<cfg::Thcon[cfg::Reg0].TileDescriptor.InDataFormat, cfg::Sec::S1, 0>());
}

extern "C" __attribute__((noinline, used)) void reference_write_mixed_descriptor_bytes(std::uint32_t blobs)
{
    TTI_RMWCIB0(0x0f, 0, 112);
    TT_RMWCIB1(0x0f, blobs & 0x0fu, 112);
    TTI_RMWCIB2(0xff, 0x34, 112);
    TTI_RMWCIB3(0xff, 0x12, 112);
}

// Thread word 5 and state word 5 are separate: one SETC16 for both thread fields
// even across the GPR transfer, and one RMWCIB for the state field.
// SetOvrdWithAddr is the top field, so only the SETC16 group mask clips it.
extern "C" __attribute__((noinline, used)) void write_thread_and_state_same_address(std::uint32_t base, std::uint32_t override_address)
{
    cfg::write<cfg::Access::TensixCfgUnit>(
        cfg::set<cfg::SrcASet::Base, cfg::Sec::S0>(base),
        cfg::set<cfg::DestOffset::Enable, cfg::Sec::S0, 1>(),
        cfg::from_gpr<cfg::Thcon[cfg::Reg3].Base_address, cfg::Sec::S0>(hal::gpr<4>()),
        cfg::set<cfg::SrcASet::SetOvrdWithAddr, cfg::Sec::S0>(override_address));
}

extern "C" __attribute__((noinline, used)) void reference_write_thread_and_state_same_address(std::uint32_t base, std::uint32_t override_address)
{
    TT_SETC16(5, ((base & 0x03u) | (override_address << 2)) & 0x07u);
    TTI_RMWCIB0(0x01, 1, 5);
    TTI_WRCFG(4, 0, 76);
}

// A runtime field is masked only when another field of its group lies above it:
// ApplyRelu (bits 5:2) keeps its mask, while ReluThreshold (bits 21:6) relies on
// the byte-lane masks to clip its value.
extern "C" __attribute__((noinline, used)) void write_runtime_fields_stacked(std::uint32_t mode, std::uint32_t threshold)
{
    cfg::write<cfg::Access::TensixCfgUnit>(
        cfg::set<cfg::StaccRelu::ApplyRelu, cfg::Sec::S0>(mode), cfg::set<cfg::StaccRelu::ReluThreshold, cfg::Sec::S0>(threshold));
}

extern "C" __attribute__((noinline, used)) void reference_write_runtime_fields_stacked(std::uint32_t mode, std::uint32_t threshold)
{
    const std::uint32_t data = ((mode << 2) & 0x3cu) | (threshold << 6);
    TT_RMWCIB0(0xfc, data & 0xffu, 2);
    TT_RMWCIB1(0xff, (data >> 8) & 0xffu, 2);
    TT_RMWCIB2(0x3f, (data >> 16) & 0xffu, 2);
}

#ifdef ENABLE_LLK_ASSERT

extern "C" void write_prepacked_thread_word_overflow()
{
    cfg::write<cfg::Access::TensixCfgUnit, cfg::AddrMod[cfg::SrcA].Incr>(0, 0x10000);
}

// ASSERT-LABEL: <write_prepacked_thread_word_overflow>:
// ASSERT: ebreak
// ASSERT: ret

// Runtime assignments must validate values before masking, even when a group
// has only one member or the invalid assignment is not its first member.
extern "C" void write_runtime_group_single_overflow()
{
    cfg::write<cfg::Access::TensixCfgUnit>(cfg::set<cfg::AluAccCtrl::Fp32_enabled, cfg::Sec::S0>(2));
}

// ASSERT-LABEL: <write_runtime_group_single_overflow>:
// ASSERT-NOT: sw
// ASSERT: ebreak
// ASSERT: ret

extern "C" void write_runtime_group_later_overflow()
{
    cfg::write<cfg::Access::TensixCfgUnit>(
        cfg::set<cfg::StaccRelu::ReluThreshold, cfg::Sec::S0, 0x1234>(), cfg::set<cfg::StaccRelu::ApplyRelu, cfg::Sec::S0>(16));
}

// ASSERT-LABEL: <write_runtime_group_later_overflow>:
// ASSERT-NOT: sw
// ASSERT-NOT: ttrmwcib
// ASSERT: ebreak
// ASSERT: ret

extern "C" void write_runtime_thread_group_overflow()
{
    cfg::write<cfg::Access::TensixCfgUnit>(cfg::set<cfg::AddrMod[cfg::SrcA].Incr, cfg::Sec::S0>(64));
}

// ASSERT-LABEL: <write_runtime_thread_group_overflow>:
// ASSERT-NOT: sw
// ASSERT: ebreak
// ASSERT: ret

extern "C" void write_runtime_group_valid_boundaries()
{
    cfg::write<cfg::Access::TensixCfgUnit>(
        cfg::set<cfg::PrngSeed::Seed_Val, cfg::Sec::S0>(0xffffffffu),
        cfg::set<cfg::AluAccCtrl::Fp32_enabled, cfg::Sec::S0>(1),
        cfg::set<cfg::AddrMod[cfg::SrcA].Incr, cfg::Sec::S0>(63),
        cfg::set<cfg::AddrMod[cfg::SrcA].CR, cfg::Sec::S0>(1));
}

// ASSERT-LABEL: <write_runtime_group_valid_boundaries>:
// ASSERT-NOT: ebreak
// ASSERT: ret

#endif
