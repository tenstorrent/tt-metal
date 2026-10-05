// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// RUN: %{blackhole_tensix_compile} %{blackhole_unpack_thread} -S %s -o %t.t0.s
// RUN: FileCheck %s --enable-var-scope --check-prefixes=CHECK,T0 < %t.t0.s
// RUN: %{blackhole_tensix_compile} %{blackhole_pack_thread} -S %s -o %t.t2.s
// RUN: FileCheck %s --enable-var-scope --check-prefixes=CHECK,T2 < %t.t2.s

#include <array>
#include <cstdint>

#include "hal/cfg.h"

namespace cfg = hal::cfg;

// State words are addressed from the bank selected by cfg_state_id (see
// mmio_compile_time.cpp), so each byte offset below is 4 * word. Thread words
// are read through the debug CREG_READ (88) and CREG_RDDATA (120) registers
// with index 672 + 68 * thread + word.

inline constexpr cfg::Field state_first {cfg::RegisterScope::State, 32, 0, 0, 0, 32, 1, 0};

// The same 64-bit field spans words 64-66 in S0 (bit 16) and 67-68 in S1 (bit 0).
inline constexpr cfg::Field state_sectioned_wide {cfg::RegisterScope::State, 32, 64, 0, 16, 64, 2, 80};

// Single-field writes.

extern "C" __attribute__((noinline, used)) void write_runtime_state_field(std::uint32_t format)
{
    cfg::write<cfg::Access::MMIO, cfg::AluFormatSpecReg::SrcB_val, cfg::Sec::S0>(format);
}

// Word 0 read-modify-write of bits 8:5 (mask 480): old ^ ((new ^ old) & mask).
// A value wider than four bits reaches ebreak.
// CHECK-LABEL: {{^}}write_runtime_state_field:
// CHECK-NEXT: li [[MAX:a[0-7]]],15
// CHECK-NEXT: bgtu a0,[[MAX]],[[FAIL:\.L[0-9]+]]
// CHECK: %lo(_ZN7ckernel12cfg_state_idE)
// CHECK: lw [[OLD:a[0-7]]],0([[BANK:a[0-7]]])
// CHECK-NEXT: slli a0,a0,5
// CHECK-NEXT: xor a0,a0,[[OLD]]
// CHECK-NEXT: andi a0,a0,480
// CHECK-NEXT: xor a0,a0,[[OLD]]
// CHECK-NEXT: sw a0,0([[BANK]])
// CHECK-NEXT: ret
// CHECK-NEXT: [[FAIL]]:
// CHECK: ebreak

extern "C" __attribute__((noinline, used)) void write_runtime_state_field_section(std::uint32_t format)
{
    cfg::write<cfg::Access::MMIO, cfg::Thcon[cfg::Reg0].TileDescriptor.InDataFormat, cfg::Sec::S1>(format);
}

// S1 lives in word 112 at bit 0, so no shift is needed.
// CHECK-LABEL: {{^}}write_runtime_state_field_section:
// CHECK-NEXT: li [[MAX:a[0-7]]],15
// CHECK-NEXT: bgtu a0,[[MAX]],[[FAIL:\.L[0-9]+]]
// CHECK: %lo(_ZN7ckernel12cfg_state_idE)
// CHECK: lw [[OLD:a[0-7]]],448([[BANK:a[0-7]]])
// CHECK-NEXT: xor a0,a0,[[OLD]]
// CHECK-NEXT: andi a0,a0,15
// CHECK-NEXT: xor a0,a0,[[OLD]]
// CHECK-NEXT: sw a0,448([[BANK]])
// CHECK-NEXT: ret
// CHECK-NEXT: [[FAIL]]:
// CHECK: ebreak

extern "C" __attribute__((noinline, used)) void write_runtime_state_full_word(std::uint32_t seed)
{
    cfg::write<cfg::Access::MMIO, cfg::PrngSeed::Seed_Val, cfg::Sec::S0>(seed);
}

// A 32-bit field replaces word 186 without a range check or read.
// CHECK-LABEL: {{^}}write_runtime_state_full_word:
// CHECK-NOT: lw {{a[0-7]}},744(
// CHECK: [[SELECTED:\.L[0-9]+]]:
// CHECK-NEXT: sw a0,744({{a[0-7]}})
// CHECK-NEXT: ret
// CHECK-NOT: ebreak

// Grouped field writes.

extern "C" __attribute__((noinline, used)) void write_field_group(std::uint32_t format)
{
    cfg::write<cfg::Access::MMIO>(
        cfg::set<cfg::Thcon[cfg::Reg0].TileDescriptor.InDataFormat, cfg::Sec::S0>(format),
        cfg::set<cfg::Thcon[cfg::Reg0].TileDescriptor.Uncompressed, cfg::Sec::S0, 1>());
}

// Both fields of word 64 merge into one read-modify-write.
// CHECK-LABEL: {{^}}write_field_group:
// CHECK: %lo(_ZN7ckernel12cfg_state_idE)
// CHECK-DAG: lw [[OLD:a[0-7]]],256([[BANK:a[0-7]]])
// CHECK-DAG: andi a0,a0,15
// CHECK: andi [[OLD]],[[OLD]],-32
// CHECK: or [[DATA:a[0-7]]],{{a[0-7]}},{{a[0-7]}}
// CHECK: ori [[DATA]],[[DATA]],16
// CHECK: sw [[DATA]],256([[BANK]])
// CHECK-NOT: sw
// CHECK: ret

extern "C" __attribute__((noinline, used)) void write_runtime_group_two_words(std::uint32_t format, std::uint32_t enable)
{
    cfg::write<cfg::Access::MMIO>(cfg::set<cfg::AluFormatSpecReg::SrcA_val, cfg::Sec::S0>(format), cfg::set<cfg::DestOffset::Enable, cfg::Sec::S0>(enable));
}

// One bank lookup, then one read-modify-write per word (0, then 5).
// CHECK-LABEL: {{^}}write_runtime_group_two_words:
// CHECK: %lo(_ZN7ckernel12cfg_state_idE)
// CHECK: [[SELECTED:\.L[0-9]+]]:
// CHECK-NEXT: lw [[W0:a[0-7]]],0([[BANK:a[0-7]]])
// CHECK-NEXT: andi a0,a0,15
// CHECK-NEXT: andi [[W0]],[[W0]],-16
// CHECK-NEXT: or a0,a0,[[W0]]
// CHECK-NEXT: sw a0,0([[BANK]])
// CHECK-NEXT: lw [[W5:a[0-7]]],20([[BANK]])
// CHECK-NEXT: andi a1,a1,1
// CHECK-NEXT: andi [[W5]],[[W5]],-2
// CHECK-NEXT: or a1,a1,[[W5]]
// CHECK-NEXT: sw a1,20([[BANK]])
// CHECK-NEXT: ret

extern "C" __attribute__((noinline, used)) void write_runtime_fields_stacked(std::uint32_t mode, std::uint32_t threshold)
{
    cfg::write<cfg::Access::MMIO>(cfg::set<cfg::StaccRelu::ApplyRelu, cfg::Sec::S0>(mode), cfg::set<cfg::StaccRelu::ReluThreshold, cfg::Sec::S0>(threshold));
}

// Unlike RMWCIB byte lanes, MMIO clips the merged data to the group mask
// 0x3ffffc; ApplyRelu (bits 5:2) is still masked before ReluThreshold joins it.
// CHECK-LABEL: {{^}}write_runtime_fields_stacked:
// CHECK: %lo(_ZN7ckernel12cfg_state_idE)
// CHECK-DAG: lw {{a[0-7]}},8([[BANK:a[0-7]]])
// CHECK-DAG: andi a0,a0,60
// CHECK-DAG: li [[MASK:a[0-7]]],4194304
// CHECK: addi [[MASK]],[[MASK]],-4
// CHECK: and a0,a0,[[MASK]]
// CHECK: sw a0,8([[BANK]])
// CHECK-NEXT: ret

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

// CHECK-LABEL: {{^}}write_pack_words_runtime:
// CHECK-NEXT: lui [[STATE:a[0-7]]],%hi(_ZN7ckernel12cfg_state_idE)
// CHECK-NEXT: lw [[ID:a[0-7]]],%lo(_ZN7ckernel12cfg_state_idE)([[STATE]])
// CHECK-NEXT: li [[BASE:a[0-7]]],-1114112
// CHECK-NEXT: addi [[BANK:a[0-7]]],[[BASE]],896
// CHECK-NEXT: bne [[ID]],zero,[[SELECTED:\.L[0-9]+]]
// CHECK-NEXT: mv [[BANK]],[[BASE]]
// CHECK-NEXT: [[SELECTED]]:
// CHECK-NEXT: sw a0,112([[BANK]])
// CHECK-NEXT: sw a1,96([[BANK]])
// CHECK-NEXT: sw a2,80([[BANK]])
// CHECK-NEXT: ret
// CHECK-NEXT: .size write_pack_words_runtime,

extern "C" __attribute__((noinline, used)) void write_last_state_words(const std::array<std::uint32_t, 2>& values)
{
    cfg::write<cfg::Access::MMIO, cfg::ChickenBits::sfpu_scbd_disable, cfg::Sec::S0, 2>(values);
}

// Words 222-223 end exactly at the bank boundary.
// CHECK-LABEL: {{^}}write_last_state_words:
// CHECK: %lo(_ZN7ckernel12cfg_state_idE)
// CHECK: sw {{a[0-7]}},888([[BANK:a[0-7]]])
// CHECK: sw {{a[0-7]}},892([[BANK]])
// CHECK-NOT: sw
// CHECK: ret

extern "C" __attribute__((noinline, used)) void write_full_state_bank(const std::array<std::uint32_t, 224>& values)
{
    cfg::write<cfg::Access::MMIO, state_first, cfg::Sec::S0, 224>(values);
}

// CHECK-LABEL: {{^}}write_full_state_bank:
// CHECK: %lo(_ZN7ckernel12cfg_state_idE)
// CHECK: addi [[END:a[0-7]]],a0,896
// CHECK: [[LOOP:\.L[0-9]+]]:
// CHECK: sw {{a[0-7]}},0({{a[0-7]}})
// CHECK: bne a0,[[END]],[[LOOP]]
// CHECK-NOT: sw
// CHECK: ret

extern "C" __attribute__((noinline, used)) void write_field_group_words(const std::array<std::uint32_t, 4>& values)
{
    cfg::write<cfg::Access::MMIO, cfg::Thcon[cfg::Reg0].TileDescriptor, cfg::Sec::S1, 4>(values);
}

// A field group anchors whole-word access on its Raw field (S1 at word 112).
// CHECK-LABEL: {{^}}write_field_group_words:
// CHECK: %lo(_ZN7ckernel12cfg_state_idE)
// CHECK: sw {{a[0-7]}},448([[BANK:a[0-7]]])
// CHECK: sw {{a[0-7]}},452([[BANK]])
// CHECK: sw {{a[0-7]}},456([[BANK]])
// CHECK: sw {{a[0-7]}},460([[BANK]])
// CHECK-NOT: sw
// CHECK: ret

extern "C" __attribute__((noinline, used)) void write_shifted_wide_field_words(const std::array<std::uint32_t, 3>& values)
{
    cfg::write<cfg::Access::MMIO, state_sectioned_wide, cfg::Sec::S0, 3>(values);
}

// Whole-word access covers every word the selected section occupies.
// CHECK-LABEL: {{^}}write_shifted_wide_field_words:
// CHECK: %lo(_ZN7ckernel12cfg_state_idE)
// CHECK: sw {{a[0-7]}},256([[BANK:a[0-7]]])
// CHECK: sw {{a[0-7]}},260([[BANK]])
// CHECK: sw {{a[0-7]}},264([[BANK]])
// CHECK-NOT: sw
// CHECK: ret

extern "C" __attribute__((noinline, used)) void write_aligned_section_words(const std::array<std::uint32_t, 2>& values)
{
    cfg::write<cfg::Access::MMIO, state_sectioned_wide, cfg::Sec::S1, 2>(values);
}

// CHECK-LABEL: {{^}}write_aligned_section_words:
// CHECK: %lo(_ZN7ckernel12cfg_state_idE)
// CHECK: sw {{a[0-7]}},268([[BANK:a[0-7]]])
// CHECK: sw {{a[0-7]}},272([[BANK]])
// CHECK-NOT: sw
// CHECK: ret

// State reads.

extern "C" __attribute__((noinline, used)) std::uint32_t read_state_field()
{
    return cfg::read<cfg::Access::MMIO, cfg::AluFormatSpecReg::SrcB_val, cfg::Sec::S0>();
}

// CHECK-LABEL: {{^}}read_state_field:
// CHECK: %lo(_ZN7ckernel12cfg_state_idE)
// CHECK: lw a0,0({{a[0-7]}})
// CHECK-NEXT: srli a0,a0,5
// CHECK-NEXT: andi a0,a0,15
// CHECK-NEXT: ret

extern "C" __attribute__((noinline, used)) std::uint32_t read_last_state_word()
{
    return cfg::read_word<cfg::Access::MMIO, cfg::PackGlobalCfgCtl::pack_disable_fast_tile_end_drain, cfg::Sec::S0, 183>();
}

// CHECK-LABEL: {{^}}read_last_state_word:
// CHECK: %lo(_ZN7ckernel12cfg_state_idE)
// CHECK: lw a0,892({{a[0-7]}})
// CHECK-NEXT: ret

extern "C" __attribute__((noinline, used)) std::uint32_t read_field_group_word()
{
    return cfg::read_word<cfg::Access::MMIO, cfg::Thcon[cfg::Reg0].TileDescriptor, cfg::Sec::S0, 3>();
}

// CHECK-LABEL: {{^}}read_field_group_word:
// CHECK: %lo(_ZN7ckernel12cfg_state_idE)
// CHECK: lw a0,268({{a[0-7]}})
// CHECK-NEXT: ret

extern "C" __attribute__((noinline, used)) std::uint32_t read_shifted_wide_field_word()
{
    return cfg::read_word<cfg::Access::MMIO, state_sectioned_wide, cfg::Sec::S0, 2>();
}

// CHECK-LABEL: {{^}}read_shifted_wide_field_word:
// CHECK: %lo(_ZN7ckernel12cfg_state_idE)
// CHECK: lw a0,264({{a[0-7]}})
// CHECK-NEXT: ret

extern "C" __attribute__((noinline, used)) std::uint32_t read_aligned_section_word()
{
    return cfg::read_word<cfg::Access::MMIO, state_sectioned_wide, cfg::Sec::S1, 1>();
}

// CHECK-LABEL: {{^}}read_aligned_section_word:
// CHECK: %lo(_ZN7ckernel12cfg_state_idE)
// CHECK: lw a0,272({{a[0-7]}})
// CHECK-NEXT: ret

// Thread reads.

extern "C" __attribute__((noinline, used)) std::uint32_t read_current_thread_field()
{
    return cfg::read<cfg::Access::MMIO, cfg::SrcASet::Base, cfg::Sec::S0>();
}

// Word 5 of the current thread's bank.
// CHECK-LABEL: {{^}}read_current_thread_field:
// CHECK: li [[DEBUG:a[0-7]]],-5169152
// T0: li [[INDEX:a[0-7]]],677
// T2: li [[INDEX:a[0-7]]],813
// CHECK: sw [[INDEX]],88([[DEBUG]])
// CHECK: lw a0,120({{a[0-7]}})
// CHECK-NEXT: andi a0,a0,3
// CHECK-NEXT: ret

extern "C" __attribute__((noinline, used)) std::uint32_t read_last_thread_word()
{
    return cfg::read_word<cfg::Access::MMIO, cfg::SrcASet::Base, cfg::Sec::S0, 62>();
}

// Word 67 is the last word of the current thread's bank.
// CHECK-LABEL: {{^}}read_last_thread_word:
// CHECK: li [[DEBUG:a[0-7]]],-5169152
// T0: li [[INDEX:a[0-7]]],739
// T2: li [[INDEX:a[0-7]]],875
// CHECK: sw [[INDEX]],88([[DEBUG]])
// CHECK: lw a0,120({{a[0-7]}})
// CHECK-NEXT: zext.h a0,a0
// CHECK-NEXT: ret

extern "C" __attribute__((noinline, used)) std::uint32_t read_other_thread_field()
{
    return cfg::read<cfg::Access::MMIO, cfg::SrcASet::Base, cfg::Sec::S0, cfg::ThreadTarget::T2>();
}

// ThreadTarget::T2 selects thread 2's bank from any thread.
// CHECK-LABEL: {{^}}read_other_thread_field:
// CHECK: li [[DEBUG:a[0-7]]],-5169152
// CHECK: li [[INDEX:a[0-7]]],813
// CHECK: sw [[INDEX]],88([[DEBUG]])
// CHECK: lw a0,120({{a[0-7]}})
// CHECK-NEXT: andi a0,a0,3
// CHECK-NEXT: ret
