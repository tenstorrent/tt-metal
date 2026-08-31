// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// RUN: %{blackhole_tensix_compile} %{blackhole_math_thread} -fsyntax-only %s

#include <cstdint>
#include <type_traits>

#include "hal/cfg.h"

namespace cfg = hal::cfg;

static_assert(static_cast<std::uint32_t>(cfg::Access::MMIO) == 0);
static_assert(static_cast<std::uint32_t>(cfg::Access::TensixCfgUnit) == 1);
static_assert(static_cast<std::uint32_t>(cfg::Access::TensixScalarUnit) == 2);
static_assert(static_cast<std::uint32_t>(cfg::RegisterFile::Thread) == 0);
static_assert(static_cast<std::uint32_t>(cfg::RegisterFile::State) == 1);
static_assert(static_cast<std::uint32_t>(cfg::Sec::S0) == 0);
static_assert(static_cast<std::uint32_t>(cfg::Sec::S7) == 7);
static_assert(static_cast<std::uint32_t>(cfg::ThreadTarget::Current) == 0);
static_assert(static_cast<std::uint32_t>(cfg::ThreadTarget::T2) == 3);
static_assert(static_cast<std::uint32_t>(cfg::GprTransferSize::Bits32) == 0);
static_assert(static_cast<std::uint32_t>(cfg::GprTransferSize::Bits128) == 1);
static_assert(static_cast<std::uint32_t>(cfg::WrcfgCompletion::Wait) == 0);
static_assert(static_cast<std::uint32_t>(cfg::WrcfgCompletion::Deferred) == 1);

// Field arithmetic: state/thread words, repeated sections, narrow/full-width
// masks, and multi-word descriptors.
constexpr cfg::Field state_field {cfg::RegisterFile::State, 32, 10, 1, 3, 5, 3, 40};
static_assert(state_field.abs0() == 355);
static_assert(state_field.addr32(cfg::Sec::S0) == 11);
static_assert(state_field.addr32(cfg::Sec::S1) == 12);
static_assert(state_field.shamt(cfg::Sec::S1) == 11);
static_assert(state_field.mask(cfg::Sec::S1) == 0xf800u);
static_assert(state_field.words() == 1);

constexpr cfg::Field full_word {cfg::RegisterFile::State, 32, 20, 0, 0, 32, 1, 0};
static_assert(full_word.mask(cfg::Sec::S0) == 0xffffffffu);
static_assert(full_word.words() == 1);
static_assert(cfg::Thcon[cfg::Reg0].TileDescriptor.Raw.words() == 4);

constexpr cfg::Field thread_field {cfg::RegisterFile::Thread, 16, 7, 0, 8, 4, 2, 16};
static_assert(thread_field.abs0() == 120);
static_assert(thread_field.addr32(cfg::Sec::S1) == 8);
static_assert(thread_field.shamt(cfg::Sec::S1) == 8);
static_assert(thread_field.mask(cfg::Sec::S1) == 0x0f00u);

// Every typed THCON selector overload resolves a representative field.
static_assert(cfg::Thcon[cfg::Reg0].TileDescriptor.Raw.addr32(cfg::Sec::S0) == 64);
static_assert(cfg::Thcon[cfg::Reg1].L1_Dest_addr.addr32(cfg::Sec::S0) == 69);
static_assert(cfg::Thcon[cfg::Reg2].Out_data_format.addr32(cfg::Sec::S0) == 72);
static_assert(cfg::Thcon[cfg::Reg3].Base_address.addr32(cfg::Sec::S0) == 76);
static_assert(cfg::Thcon[cfg::Reg4].Base_cntx4_address.addr32(cfg::Sec::S0) == 80);
static_assert(cfg::Thcon[cfg::Reg5].Dest_cntx0_address.addr32(cfg::Sec::S0) == 84);
static_assert(cfg::Thcon[cfg::Reg6].Source_address.addr32(cfg::Sec::S0) == 88);
static_assert(cfg::Thcon[cfg::Reg7].Offset_address.addr32(cfg::Sec::S0) == 92);
static_assert(cfg::Thcon[cfg::Reg8].L1_Dest_addr.addr32(cfg::Sec::S0) == 97);
static_assert(cfg::Thcon[cfg::Reg9].Pack_0_2_limit_address.addr32(cfg::Sec::S0) == 100);
static_assert(cfg::Thcon[cfg::Reg10].Unpack_limit_address.addr32(cfg::Sec::S0) == 104);
static_assert(cfg::Thcon[cfg::Reg11].Metadata_l1_addr.addr32(cfg::Sec::S0) == 108);

// Selector-only descriptor entries.
static_assert(cfg::DisableImpliedFmt[cfg::SrcA].addr32(cfg::Sec::S0) == 2);
static_assert(cfg::DisableImpliedFmt[cfg::SrcB].addr32(cfg::Sec::S0) == 3);
static_assert(cfg::AddrMod[cfg::SrcA].Incr.shamt(cfg::Sec::S0) == 0);
static_assert(cfg::AddrMod[cfg::SrcB].Incr.shamt(cfg::Sec::S0) == 8);
static_assert(cfg::AddrMod[cfg::Src][cfg::Y].Incr.shamt(cfg::Sec::S0) == 0);
static_assert(cfg::AddrMod[cfg::Src][cfg::Z].Incr.shamt(cfg::Sec::S0) == 12);
static_assert(cfg::AddrMod[cfg::Dest].Incr.shamt(cfg::Sec::S0) == 0);
static_assert(cfg::AddrMod[cfg::Dest][cfg::Y].Incr.shamt(cfg::Sec::S0) == 6);
static_assert(cfg::AddrMod[cfg::Dest][cfg::Z].Incr.shamt(cfg::Sec::S0) == 14);
static_assert(cfg::AddrMod[cfg::Fidelity].Incr.shamt(cfg::Sec::S0) == 13);
static_assert(cfg::AddrMod[cfg::Bias].Incr.shamt(cfg::Sec::S0) == 0);

constexpr bool packer_descriptors_are_complete()
{
    const auto packer           = cfg::Packer[std::integral_constant<std::uint32_t, 0> {}];
    std::uint32_t control_count = 0;
    std::uint32_t control_sum   = 0;
    packer.AddrCtrl.forEach(
        [&](auto reg)
        {
            ++control_count;
            control_sum += packer.AddrCtrl[reg].Xstride.addr32(cfg::Sec::S0);
        });

    std::uint32_t base_count = 0;
    std::uint32_t base_sum   = 0;
    packer.AddrBase.forEach(
        [&](auto reg)
        {
            ++base_count;
            base_sum += packer.AddrBase[reg].addr32(cfg::Sec::S0);
        });

    return control_count == 2 && control_sum == 26 && base_count == 2 && base_sum == 33 &&
           packer.AddrCtrl[cfg::PackerReg::Reg0].Zstride.addr32(cfg::Sec::S0) == 13 &&
           packer.AddrCtrl[cfg::PackerReg::Reg1].Zstride.addr32(cfg::Sec::S0) == 15 && packer.AddrBase[cfg::PackerReg::Reg0].addr32(cfg::Sec::S0) == 16 &&
           packer.AddrBase[cfg::PackerReg::Reg1].addr32(cfg::Sec::S0) == 17;
}

static_assert(packer_descriptors_are_complete());

constexpr bool unpacker_descriptors_are_complete()
{
    std::uint32_t unpacker_count = 0;
    std::uint32_t register_count = 0;
    std::uint32_t context_count  = 0;
    std::uint32_t address_sum    = 0;

    cfg::Unpacker.forEach(
        [&](auto unpacker)
        {
            ++unpacker_count;
            const auto selected = cfg::Unpacker[unpacker];
            selected.AddrCtrl.forEach(
                [&](auto reg)
                {
                    ++register_count;
                    address_sum += selected.AddrCtrl[reg].Xstride.addr32(cfg::Sec::S0);
                });
            selected.AddrBase.forEach(
                [&](auto reg)
                {
                    ++register_count;
                    address_sum += selected.AddrBase[reg].addr32(cfg::Sec::S0);
                });
            selected.Cntx.forEach(
                [&](auto context)
                {
                    ++context_count;
                    address_sum += selected.Cntx[context].Base.addr32(cfg::Sec::S0);
                });
        });

    return unpacker_count == 2 && register_count == 8 && context_count == 10 && address_sum == 1307 &&
           cfg::Unpacker[0].AddrCtrl[cfg::UnpackerReg::Reg0].Xstride.addr32(cfg::Sec::S0) == 44 &&
           cfg::Unpacker[0].AddrCtrl[cfg::UnpackerReg::Reg1].Xstride.addr32(cfg::Sec::S0) == 56 &&
           cfg::Unpacker[1].AddrBase[cfg::UnpackerReg::Reg0].addr32(cfg::Sec::S0) == 60 &&
           cfg::Unpacker[1].AddrBase[cfg::UnpackerReg::Reg1].addr32(cfg::Sec::S0) == 61 &&
           cfg::Unpacker[0].BlobsYStart[cfg::BlobContext::Context01].addr32(cfg::Sec::S0) == 51 &&
           cfg::Unpacker[0].BlobsYStart[cfg::BlobContext::Context23].addr32(cfg::Sec::S0) == 52;
}

static_assert(unpacker_descriptors_are_complete());

constexpr bool mapping_descriptors_are_complete()
{
    std::uint32_t row_fields  = 0;
    std::uint32_t face_fields = 0;
    std::uint32_t row_sum     = 0;
    std::uint32_t face_sum    = 0;

    cfg::TileRowSetMapping.forEach(
        [&](auto mapping)
        {
            cfg::TileRowSetMapping[mapping].forEach(
                [&](auto set)
                {
                    ++row_fields;
                    const auto& field = cfg::TileRowSetMapping[mapping][set];
                    row_sum += field.addr32(cfg::Sec::S0) + field.shamt(cfg::Sec::S0);
                });
        });
    cfg::TileFaceSetMapping.forEach(
        [&](auto mapping)
        {
            cfg::TileFaceSetMapping[mapping].forEach(
                [&](auto set)
                {
                    ++face_fields;
                    const auto& field = cfg::TileFaceSetMapping[mapping][set];
                    face_sum += field.addr32(cfg::Sec::S0) + field.shamt(cfg::Sec::S0);
                });
        });

    return row_fields == 64 && face_fields == 64 && row_sum == 2336 && face_sum == 3360;
}

static_assert(mapping_descriptors_are_complete());

static_assert(cfg::PerfCntCmd[0].Start.shamt(cfg::Sec::S0) == 0);
static_assert(cfg::PerfCntCmd[1].Start.shamt(cfg::Sec::S0) == 2);
static_assert(cfg::PerfCntCmd[2].Stop.shamt(cfg::Sec::S0) == 5);
static_assert(cfg::PerfCntCmd[3].Stop.shamt(cfg::Sec::S0) == 7);
constexpr auto first_perf_counter = cfg::PerfCntCmdEntry::make<0>();
constexpr auto last_perf_counter  = cfg::PerfCntCmdEntry::make<3>();
static_assert(first_perf_counter.Start.shamt(cfg::Sec::S0) == 0);
static_assert(last_perf_counter.Stop.shamt(cfg::Sec::S0) == 7);

// Compile-time builders and their public descriptor results.
constexpr auto common_gpr = hal::gpr<5>();
static_assert(decltype(common_gpr)::index == 5);

constexpr auto cfg_gpr = cfg::gpr<16, cfg::GprTransferSize::Bits128, cfg::WrcfgCompletion::Deferred>();
static_assert(decltype(cfg_gpr)::index == 16);
static_assert(decltype(cfg_gpr)::size == cfg::GprTransferSize::Bits128);
static_assert(decltype(cfg_gpr)::completion == cfg::WrcfgCompletion::Deferred);

constexpr auto constant_assignment = cfg::set<cfg::AluAccCtrl::Fp32_enabled, cfg::Sec::S0, 1>();
static_assert(decltype(constant_assignment)::file == cfg::RegisterFile::State);
static_assert(decltype(constant_assignment)::addr == 1);
static_assert(decltype(constant_assignment)::shift == 29);
static_assert(decltype(constant_assignment)::mask == 0x20000000u);
static_assert(decltype(constant_assignment)::value == 1);

constexpr auto gpr_write = cfg::from_gpr<cfg::Thcon[cfg::Reg0].TileDescriptor.Raw, cfg::Sec::S1>(cfg_gpr);
static_assert(decltype(gpr_write)::file == cfg::RegisterFile::State);
static_assert(decltype(gpr_write)::addr == 112);
static_assert(decltype(gpr_write)::words == 4);
static_assert(decltype(gpr_write.source)::index == 16);

constexpr auto common_gpr_write = cfg::from_gpr<cfg::Thcon[cfg::Reg3].Base_address, cfg::Sec::S0>(common_gpr);
static_assert(decltype(common_gpr_write)::addr == 76);
static_assert(decltype(common_gpr_write)::words == 1);
static_assert(cfg::word_addr<cfg::Thcon[cfg::Reg0].TileDescriptor.Raw, cfg::Sec::S1> == 112);
