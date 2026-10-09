// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// RUN: %{blackhole_tensix_compile} %{blackhole_unpack_thread} -fno-ipa-icf -c %s -o %t.t0.o
// RUN: %{blackhole_compare_codegen} %t.t0.o
// RUN: %{blackhole_tensix_compile} %{blackhole_pack_thread} -fno-ipa-icf -c %s -o %t.t2.o
// RUN: %{blackhole_compare_codegen} %t.t2.o
// RUN: %{blackhole_tensix_compile} %{blackhole_unpack_thread} -DENABLE_LLK_ASSERT -S %s -o %t.assert.s
// RUN: FileCheck %s --check-prefix=ASSERT < %t.assert.s

// Each HAL function is compared with its adjacent reference_ function.
// Disable identical-code folding so both bodies must be emitted independently.

#include <array>
#include <cstdint>

#include "hal/cfg.h"

namespace cfg = hal::cfg;

// Explicit bank addressing for the reference MMIO operations.
inline volatile std::uint32_t tt_reg_ptr* reference_cfg_bank()
{
    const std::uint32_t bank_offset = (ckernel::cfg_state_id == 0) ? 0u : 224u;
    return reinterpret_cast<volatile std::uint32_t tt_reg_ptr*>(TENSIX_CFG_BASE) + bank_offset;
}

// State words are addressed from the bank selected by cfg_state_id (see
// mmio_compile_time.cpp), so each byte offset below is 4 * word. Thread words
// are read through the debug CREG_READ (88) and CREG_RDDATA (120) registers
// with index 672 + 68 * thread + word.

inline constexpr cfg::Field state_first {cfg::RegisterScope::State, 32, 0, 0, 0, 32, 1, 0};

// The same 64-bit field spans words 64-66 in S0 (bit 16) and 67-68 in S1 (bit 0).
inline constexpr cfg::Field state_sectioned_wide {cfg::RegisterScope::State, 32, 64, 0, 16, 64, 2, 80};

// Single-field writes.

// Word 0 read-modify-write of bits 8:5 (mask 480): (old & ~mask) | (new & mask).
extern "C" __attribute__((noinline, used)) void write_runtime_state_field(std::uint32_t format)
{
    cfg::write<cfg::Access::MMIO, cfg::AluFormatSpecReg::SrcB_val, cfg::Sec::S0>(format);
}

extern "C" __attribute__((noinline, used)) void reference_write_runtime_state_field(std::uint32_t format)
{
    auto* bank = reference_cfg_bank();
    bank[0]    = (bank[0] & ~0x1e0u) | ((format << 5) & 0x1e0u);
}

// S1 lives in word 112 at bit 0, so no shift is needed.
extern "C" __attribute__((noinline, used)) void write_runtime_state_field_section(std::uint32_t format)
{
    cfg::write<cfg::Access::MMIO, cfg::Thcon[cfg::Reg0].TileDescriptor.InDataFormat, cfg::Sec::S1>(format);
}

extern "C" __attribute__((noinline, used)) void reference_write_runtime_state_field_section(std::uint32_t format)
{
    auto* bank = reference_cfg_bank();
    bank[112]  = (format & 0x0fu) | (bank[112] & ~0x0fu);
}

// A 32-bit field replaces word 186 without a read.
extern "C" __attribute__((noinline, used)) void write_runtime_state_full_word(std::uint32_t seed)
{
    cfg::write<cfg::Access::MMIO, cfg::PrngSeed::Seed_Val, cfg::Sec::S0>(seed);
}

extern "C" __attribute__((noinline, used)) void reference_write_runtime_state_full_word(std::uint32_t seed)
{
    reference_cfg_bank()[186] = seed;
}

// Grouped field writes.

// Both fields of word 64 merge into one read-modify-write.
extern "C" __attribute__((noinline, used)) void write_field_group(std::uint32_t format)
{
    cfg::write<cfg::Access::MMIO>(
        cfg::set<cfg::Thcon[cfg::Reg0].TileDescriptor.InDataFormat, cfg::Sec::S0>(format),
        cfg::set<cfg::Thcon[cfg::Reg0].TileDescriptor.Uncompressed, cfg::Sec::S0, 1>());
}

extern "C" __attribute__((noinline, used)) void reference_write_field_group(std::uint32_t format)
{
    auto* bank               = reference_cfg_bank();
    const std::uint32_t data = (format & 0x0fu) | 0x10u;
    const std::uint32_t old  = bank[64];
    bank[64]                 = (old & ~0x1fu) | data;
}

// One bank lookup, then one read-modify-write per word (0, then 5).
extern "C" __attribute__((noinline, used)) void write_runtime_group_two_words(std::uint32_t format, std::uint32_t enable)
{
    cfg::write<cfg::Access::MMIO>(cfg::set<cfg::AluFormatSpecReg::SrcA_val, cfg::Sec::S0>(format), cfg::set<cfg::DestOffset::Enable, cfg::Sec::S0>(enable));
}

extern "C" __attribute__((noinline, used)) void reference_write_runtime_group_two_words(std::uint32_t format, std::uint32_t enable)
{
    auto* bank = reference_cfg_bank();
    bank[0]    = (format & 0x0fu) | (bank[0] & ~0x0fu);
    bank[5]    = (enable & 0x01u) | (bank[5] & ~0x01u);
}

// Unlike RMWCIB byte lanes, MMIO clips the merged data to the group mask
// 0x3ffffc; ApplyRelu (bits 5:2) is still masked before ReluThreshold joins it.
extern "C" __attribute__((noinline, used)) void write_runtime_fields_stacked(std::uint32_t mode, std::uint32_t threshold)
{
    cfg::write<cfg::Access::MMIO>(cfg::set<cfg::StaccRelu::ApplyRelu, cfg::Sec::S0>(mode), cfg::set<cfg::StaccRelu::ReluThreshold, cfg::Sec::S0>(threshold));
}

extern "C" __attribute__((noinline, used)) void reference_write_runtime_fields_stacked(std::uint32_t mode, std::uint32_t threshold)
{
    auto* bank                 = reference_cfg_bank();
    const std::uint32_t data   = ((mode << 2) & 0x3cu) | (threshold << 6);
    const std::uint32_t masked = data & 0x3ffffcu;
    const std::uint32_t old    = bank[2];
    bank[2]                    = (old & ~0x3ffffcu) | masked;
}

// Array writes.

// Runtime values must pass directly to the three configure_pack() stores without a batch.
extern "C" __attribute__((noinline, used)) void write_pack_words_runtime(std::uint32_t counter, std::uint32_t edge, std::uint32_t mapping)
{
    const std::array<std::uint32_t, 1> counters = {counter};
    const std::array<std::uint32_t, 1> edges    = {edge};
    const std::array<std::uint32_t, 1> mappings = {mapping};

    cfg::write<cfg::Access::MMIO, cfg::PackCounters::pack_per_xy_plane, cfg::Sec::S0, 1>(counters);
    cfg::write<cfg::Access::MMIO, cfg::PckEdgeOffsetSec0::mask, cfg::Sec::S0, 1>(edges);
    cfg::write<cfg::Access::MMIO, cfg::TileRowSetMapping[0][0], cfg::Sec::S0, 1>(mappings);
}

extern "C" __attribute__((noinline, used)) void reference_write_pack_words_runtime(std::uint32_t counter, std::uint32_t edge, std::uint32_t mapping)
{
    auto* bank = reference_cfg_bank();
    bank[28]   = counter;
    bank[24]   = edge;
    bank[20]   = mapping;
}

// Words 222-223 end exactly at the bank boundary.
extern "C" __attribute__((noinline, used)) void write_last_state_words(const std::array<std::uint32_t, 2>& values)
{
    cfg::write<cfg::Access::MMIO, cfg::ChickenBits::sfpu_scbd_disable, cfg::Sec::S0, 2>(values);
}

extern "C" __attribute__((noinline, used)) void reference_write_last_state_words(const std::array<std::uint32_t, 2>& values)
{
    auto* bank = reference_cfg_bank();
    bank[222]  = values[0];
    bank[223]  = values[1];
}

extern "C" __attribute__((noinline, used)) void write_full_state_bank(const std::array<std::uint32_t, 224>& values)
{
    cfg::write<cfg::Access::MMIO, state_first, cfg::Sec::S0, 224>(values);
}

extern "C" __attribute__((noinline, used)) void reference_write_full_state_bank(const std::array<std::uint32_t, 224>& values)
{
    auto* bank = reference_cfg_bank();
    for (std::uint32_t i = 0; i < 224; ++i)
    {
        bank[i] = values[i];
    }
}

// A field group anchors whole-word access on its Raw field (S1 at word 112).
extern "C" __attribute__((noinline, used)) void write_field_group_words(const std::array<std::uint32_t, 4>& values)
{
    cfg::write<cfg::Access::MMIO, cfg::Thcon[cfg::Reg0].TileDescriptor, cfg::Sec::S1, 4>(values);
}

extern "C" __attribute__((noinline, used)) void reference_write_field_group_words(const std::array<std::uint32_t, 4>& values)
{
    auto* bank = reference_cfg_bank();
    bank[112]  = values[0];
    bank[113]  = values[1];
    bank[114]  = values[2];
    bank[115]  = values[3];
}

// Whole-word access covers every word the selected section occupies.
extern "C" __attribute__((noinline, used)) void write_shifted_wide_field_words(const std::array<std::uint32_t, 3>& values)
{
    cfg::write<cfg::Access::MMIO, state_sectioned_wide, cfg::Sec::S0, 3>(values);
}

extern "C" __attribute__((noinline, used)) void reference_write_shifted_wide_field_words(const std::array<std::uint32_t, 3>& values)
{
    auto* bank = reference_cfg_bank();
    bank[64]   = values[0];
    bank[65]   = values[1];
    bank[66]   = values[2];
}

extern "C" __attribute__((noinline, used)) void write_aligned_section_words(const std::array<std::uint32_t, 2>& values)
{
    cfg::write<cfg::Access::MMIO, state_sectioned_wide, cfg::Sec::S1, 2>(values);
}

extern "C" __attribute__((noinline, used)) void reference_write_aligned_section_words(const std::array<std::uint32_t, 2>& values)
{
    auto* bank = reference_cfg_bank();
    bank[67]   = values[0];
    bank[68]   = values[1];
}

// State reads.

extern "C" __attribute__((noinline, used)) std::uint32_t read_state_field()
{
    return cfg::read<cfg::Access::MMIO, cfg::AluFormatSpecReg::SrcB_val, cfg::Sec::S0>();
}

extern "C" __attribute__((noinline, used)) std::uint32_t reference_read_state_field()
{
    return (reference_cfg_bank()[0] >> 5) & 0x0fu;
}

extern "C" __attribute__((noinline, used)) std::uint32_t read_last_state_word()
{
    return cfg::read_word<cfg::Access::MMIO, cfg::PackGlobalCfgCtl::pack_disable_fast_tile_end_drain, cfg::Sec::S0, 183>();
}

extern "C" __attribute__((noinline, used)) std::uint32_t reference_read_last_state_word()
{
    return reference_cfg_bank()[223];
}

extern "C" __attribute__((noinline, used)) std::uint32_t read_field_group_word()
{
    return cfg::read_word<cfg::Access::MMIO, cfg::Thcon[cfg::Reg0].TileDescriptor, cfg::Sec::S0, 3>();
}

extern "C" __attribute__((noinline, used)) std::uint32_t reference_read_field_group_word()
{
    return reference_cfg_bank()[67];
}

extern "C" __attribute__((noinline, used)) std::uint32_t read_shifted_wide_field_word()
{
    return cfg::read_word<cfg::Access::MMIO, state_sectioned_wide, cfg::Sec::S0, 2>();
}

extern "C" __attribute__((noinline, used)) std::uint32_t reference_read_shifted_wide_field_word()
{
    return reference_cfg_bank()[66];
}

extern "C" __attribute__((noinline, used)) std::uint32_t read_aligned_section_word()
{
    return cfg::read_word<cfg::Access::MMIO, state_sectioned_wide, cfg::Sec::S1, 1>();
}

extern "C" __attribute__((noinline, used)) std::uint32_t reference_read_aligned_section_word()
{
    return reference_cfg_bank()[68];
}

// Thread reads.

// Word 5 of the current thread's bank.
extern "C" __attribute__((noinline, used)) std::uint32_t read_current_thread_field()
{
    return cfg::read<cfg::Access::MMIO, cfg::SrcASet::Base, cfg::Sec::S0>();
}

extern "C" __attribute__((noinline, used)) std::uint32_t reference_read_current_thread_field()
{
    ckernel::reg_write(RISCV_DEBUG_REG_TENSIX_CREG_READ, 672 + 68 * COMPILE_FOR_TRISC + 5);
    ckernel::wait(1);
    return ckernel::reg_read(RISCV_DEBUG_REG_TENSIX_CREG_RDDATA) & 0x03u;
}

// Word 67 is the last word of the current thread's bank.
extern "C" __attribute__((noinline, used)) std::uint32_t read_last_thread_word()
{
    return cfg::read_word<cfg::Access::MMIO, cfg::SrcASet::Base, cfg::Sec::S0, 62>();
}

extern "C" __attribute__((noinline, used)) std::uint32_t reference_read_last_thread_word()
{
    ckernel::reg_write(RISCV_DEBUG_REG_TENSIX_CREG_READ, 672 + 68 * COMPILE_FOR_TRISC + 67);
    ckernel::wait(1);
    return ckernel::reg_read(RISCV_DEBUG_REG_TENSIX_CREG_RDDATA) & 0xffffu;
}

// ThreadTarget::T2 selects thread 2's bank from any thread.
extern "C" __attribute__((noinline, used)) std::uint32_t read_other_thread_field()
{
    return cfg::read<cfg::Access::MMIO, cfg::SrcASet::Base, cfg::Sec::S0, cfg::ThreadTarget::T2>();
}

extern "C" __attribute__((noinline, used)) std::uint32_t reference_read_other_thread_field()
{
    ckernel::reg_write(RISCV_DEBUG_REG_TENSIX_CREG_READ, 813);
    ckernel::wait(1);
    return ckernel::reg_read(RISCV_DEBUG_REG_TENSIX_CREG_RDDATA) & 0x03u;
}

#ifdef ENABLE_LLK_ASSERT

extern "C" void write_runtime_mmio_group_single_overflow()
{
    cfg::write<cfg::Access::MMIO>(cfg::set<cfg::AluAccCtrl::Fp32_enabled, cfg::Sec::S0>(2));
}

// ASSERT-LABEL: {{^}}write_runtime_mmio_group_single_overflow:
// ASSERT-NOT: sw {{[a-z0-9]+}},
// ASSERT: ebreak
// ASSERT: ret

extern "C" void write_runtime_mmio_group_later_overflow()
{
    cfg::write<cfg::Access::MMIO>(cfg::set<cfg::AluAccCtrl::SFPU_Fp32_enabled, cfg::Sec::S0, 1>(), cfg::set<cfg::AluAccCtrl::Fp32_enabled, cfg::Sec::S0>(2));
}

// ASSERT-LABEL: {{^}}write_runtime_mmio_group_later_overflow:
// ASSERT-NOT: sw {{[a-z0-9]+}},
// ASSERT: ebreak
// ASSERT: ret

#endif
