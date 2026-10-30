// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "tt_metal/tt_metal/dram/mimir_hw_state.hpp"

#include <array>
#include <cstddef>
#include <map>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include <fmt/format.h>
#include <fmt/ranges.h>
#include <tt-logger/tt-logger.hpp>

#include "impl/context/metal_context.hpp"
#include "tt_metal/tt_metal/dram/mimir_access.hpp"

// dev_mem_map.h defines CCE_RESET_VECTOR_BASE as a macro; read UMD's constant of the same name with the macro hidden.
#pragma push_macro("CCE_RESET_VECTOR_BASE")
#undef CCE_RESET_VECTOR_BASE
#include "umd/device/arch/grendel_implementation.hpp"
namespace tt::tt_metal::mimir_test::hw_state_impl {
inline constexpr uint64_t kCceResetVectorBaseAddress = tt::umd::grendel::CCE_RESET_VECTOR_BASE;
}
#pragma pop_macro("CCE_RESET_VECTOR_BASE")

namespace tt::tt_metal::mimir_test {

namespace hw_state_impl {

inline constexpr uint64_t kSsBarBaseAddress = 0x04300000ULL;
inline constexpr uint64_t kSsBarBase10Offset = 0x40;
inline constexpr uint64_t kAttBaseAddress = 0x04290000ULL;
inline constexpr uint64_t kAttEnableOffset = 0x00;
inline constexpr uint64_t kAttDebugLastSrcOffset = 0x08;
inline constexpr uint64_t kAttDebugLastDestOffset = 0x10;
inline constexpr uint64_t kAttDebugLastTransSrcOffset = 0x18;
inline constexpr uint64_t kAttDebugLastTransDestOffset = 0x20;
inline constexpr uint64_t kAttDebugLastTranslationOffset = 0x28;
inline constexpr uint64_t kAttDebugMiscOffset = 0x2C;
inline constexpr uint64_t kAttEndpointTableOffset = 0x2000;
inline constexpr uint64_t kD2dTileStride = 0x00200000ULL;
inline constexpr uint64_t kCceResetBaseAddress = tt::umd::grendel::CCE_PF_CTRL_RESET_BASE;
inline constexpr uint64_t kCceResetVectorHartStride = tt::umd::grendel::CCE_HART_RESET_VECTOR_STRIDE;
inline constexpr uint64_t kCceUncoreReleased = tt::umd::grendel::CCE_UNCORE_RELEASED;
inline constexpr uint64_t kCceAxiReadErrorOffset = 0x98;
inline constexpr uint64_t kCceAxiWriteErrorOffset = 0xA0;
inline constexpr uint64_t kCceRemapBaseAddress = tt::umd::grendel::CCE_DMRISC_REMAP_BASE;
inline constexpr uint64_t kCceRemapEntryStride = tt::umd::grendel::CCE_DMRISC_REMAP_ENTRY_STRIDE;
inline constexpr uint64_t kCceRemapStartOffset = 0x00;
inline constexpr uint64_t kCceRemapEndOffset = 0x08;
inline constexpr uint64_t kCceRemapRemapStartOffset = 0x10;
inline constexpr uint64_t kCceRemapAttrsOffset = 0x18;
inline constexpr uint64_t kCceRemapEntriesPerCce = 3;
inline constexpr uint64_t kGddrIntlvFlagOutStatus0Address = 0x04004058ULL;
inline constexpr uint64_t kGddrCapFlagOutStatus0Address = 0x04004358ULL;
inline constexpr uint64_t kGddrIntlvExpected = 1;  // TILE
inline constexpr uint64_t kGddrCapExpected = 2;    // realignment code 2 (8 GB Mimir)
inline constexpr uint64_t kGddrIntlvCompareMask = 0x7;
inline constexpr uint64_t kGddrCapCompareMask = 0xF;
inline constexpr uint64_t kCceAttrsValidMask = 0x1;
inline constexpr uint64_t kAttEnableValidMask = 0x1;
inline constexpr uint64_t kDebugMiscErrorBitsMask = 0xE00;  // bits 9-11
inline constexpr uint64_t kAxiErrorBitsMask = 0x3F0FFF0FFFULL;
inline constexpr uint32_t kIdentityEndpointEntry = 64;
inline constexpr uint64_t kIdentityEndpointValue = 0x184;  // (4,6)
inline constexpr uint64_t kIdentityBase10Base = 0x80000000ULL;
inline constexpr uint64_t kIdentityBase10Stride = 0x800000ULL;

// d2d_index 0/1; D2D1 registers are +0x200000.
struct D2dTile {
    uint32_t mimir_index = 0;
    uint32_t d2d_index = 0;
};

struct AttEndpointExpected {
    uint32_t entry_index;
    uint64_t value;
};

// Entry word = x | (y << 6), same on both D2D tiles (chippy noc2axi_m2k.py _MIMIR_ET_ENDPOINTS).
inline constexpr AttEndpointExpected kAttEndpointsExpected[] = {
    {64, 0x184},
    {65, 0x144},
    {66, 0x104},
    {67, 0xC4},
    {96, 0x184},
    {97, 0x104},
    {112, 0x144},
    {113, 0xC4},
};

// ATT entry 2 routes to endpoint 64 + destination CCE index.
inline constexpr uint32_t kFirstRemoteSramEntry = 64;

struct RemoteGddrEndpointEntry {
    uint32_t source_mimir;
    uint32_t d2d_port;
    uint32_t entry_index;
};

// ATT endpoints for remote DRAM in a two-Mimir package: Mimir 0 reaches Mimir 1 over D2D1, Mimir 1 reaches 0 over D2D0.
inline constexpr RemoteGddrEndpointEntry kRemoteGddrEndpointEntries[] = {
    {0, 1, 113},
    {1, 0, 96},
};

AttEndpointPrediction expected_att_endpoint(uint32_t entry) {
    for (const AttEndpointExpected& endpoint : kAttEndpointsExpected) {
        if (endpoint.entry_index == entry) {
            return {endpoint.entry_index, endpoint.value};
        }
    }
    TT_THROW("ATT endpoint entry {} is not in the expected table", entry);
}

uint32_t mimir_count(const metal_SocDescriptor& soc) { return static_cast<uint32_t>(soc.get_num_dram_channels()); }

// Live D2D tiles: 1x1 -> only M0 D2D0; 2x -> all four. Never read a non-live tile.
std::vector<D2dTile> live_d2d_tiles(const metal_SocDescriptor& soc) {
    // A 1x1 package has one Mimir and only its D2D0 tile is live. A 2x package has two Mimirs with two tiles each.
    const uint32_t num_mimirs = mimir_count(soc);
    const bool has_two_d2d_tiles = num_mimirs > 1;
    std::vector<D2dTile> tiles;
    for (uint32_t mimir_index = 0; mimir_index < num_mimirs; ++mimir_index) {
        tiles.push_back(D2dTile{mimir_index, 0});
        if (has_two_d2d_tiles) {
            tiles.push_back(D2dTile{mimir_index, 1});
        }
    }
    return tiles;
}

std::string tile_prefix(const D2dTile& tile) { return fmt::format("M{}.D2D{}", tile.mimir_index, tile.d2d_index); }

uint64_t d2d_tile_offset(const D2dTile& tile) { return static_cast<uint64_t>(tile.d2d_index) * kD2dTileStride; }

uint64_t cce_register_address(const metal_SocDescriptor& soc, uint32_t location, uint64_t register_address) {
    return tt::umd::mimir_local_address_windows(soc).cce_control_address(0, location, register_address);
}

uint64_t ss_bar_address(const D2dTile& tile, uint64_t offset) {
    return kSsBarBaseAddress + d2d_tile_offset(tile) + offset;
}

uint64_t att_address(const D2dTile& tile, uint64_t offset) { return kAttBaseAddress + d2d_tile_offset(tile) + offset; }

uint64_t att_endpoint_address(const D2dTile& tile, uint32_t entry_index) {
    return att_address(tile, kAttEndpointTableOffset + 4 * static_cast<uint64_t>(entry_index));
}

MimirRegister make_register(std::string name, uint32_t mimir_index, uint64_t address, uint64_t value, bool is_64bit) {
    MimirRegister reg;
    reg.name = std::move(name);
    reg.mimir_index = mimir_index;
    reg.address = address;
    reg.value = value;
    reg.is_64bit = is_64bit;
    return reg;
}

// Cluster::read_reg moves 4 bytes per call. 64-bit values are lo at addr, then hi at addr + 4.
uint32_t read_smc_word(
    const Cluster& cluster, const metal_SocDescriptor& soc, ChipId chip, uint32_t mimir_index, uint64_t address) {
    const CoreCoord smc_core = mimir_smc_core(soc, mimir_index);
    uint32_t word = 0;
    cluster.read_reg(&word, tt_cxy_pair(static_cast<std::size_t>(chip), smc_core.x, smc_core.y), address);
    return word;
}

uint64_t read_smc_value(
    const Cluster& cluster,
    const metal_SocDescriptor& soc,
    ChipId chip,
    uint32_t mimir_index,
    uint64_t address,
    bool is_64bit) {
    const uint64_t low = read_smc_word(cluster, soc, chip, mimir_index, address);
    if (!is_64bit) {
        return low;
    }
    const uint64_t high = read_smc_word(cluster, soc, chip, mimir_index, address + 4);
    return (high << 32) | low;
}

MimirRegister read_named(
    const Cluster& cluster,
    const metal_SocDescriptor& soc,
    ChipId chip,
    const std::string& name,
    uint32_t mimir_index,
    uint64_t address,
    bool is_64bit) {
    return make_register(
        name, mimir_index, address, read_smc_value(cluster, soc, chip, mimir_index, address, is_64bit), is_64bit);
}

// Same names, addresses and order as the layout. Only the values come from the readback.
std::vector<MimirRegister> read_layout(
    const Cluster& cluster, const metal_SocDescriptor& soc, ChipId chip, const std::vector<MimirRegister>& layout) {
    std::vector<MimirRegister> readback;
    readback.reserve(layout.size());
    for (const MimirRegister& layout_reg : layout) {
        MimirRegister reg = layout_reg;
        reg.value = read_smc_value(cluster, soc, chip, layout_reg.mimir_index, layout_reg.address, layout_reg.is_64bit);
        readback.push_back(reg);
    }
    return readback;
}

// Bits that are compared. Other bits of CCE remap attrs and ATT.ENABLE are logged in diff_d2d_config, not compared.
uint64_t compare_mask_for_name(const std::string& name) {
    if (name.ends_with(".ATTRS")) {
        return kCceAttrsValidMask;
    }
    if (name.ends_with(".ATT.ENABLE")) {
        return kAttEnableValidMask;
    }
    if (name.ends_with(".INTLV_FLAGOUTSTATUS0")) {
        return kGddrIntlvCompareMask;
    }
    if (name.ends_with(".CAP_FLAGOUTSTATUS0")) {
        return kGddrCapCompareMask;
    }
    return ~0ULL;
}

void add_ss_bar_expected(
    std::vector<MimirRegister>& registers, const D2dTile& tile, const char* name, uint64_t offset, uint64_t value) {
    registers.push_back(make_register(
        fmt::format("{}.SS_BAR.{}", tile_prefix(tile), name),
        tile.mimir_index,
        ss_bar_address(tile, offset),
        value,
        false));
}

// Same values as chippy noc2axi_m2k.py noc2axi_mimir_bar_cfg.
void append_ss_bar_expected(std::vector<MimirRegister>& registers, const D2dTile& tile) {
    const uint64_t inst = tile.mimir_index;
    const bool via_d2d1 = tile.d2d_index != 0;
    // BAR0: SMN config space.
    add_ss_bar_expected(registers, tile, "BASE_00", 0x20, inst * 0x08000000ULL);
    add_ss_bar_expected(registers, tile, "BASE_01", 0x24, 0x13);
    add_ss_bar_expected(registers, tile, "LIMIT_00", 0x28, 0x08000000ULL + inst * 0x08000000ULL);
    add_ss_bar_expected(registers, tile, "LIMIT_01", 0x2C, 0x13);
    add_ss_bar_expected(registers, tile, "OFFSET_00", 0x30, 0);
    add_ss_bar_expected(registers, tile, "OFFSET_01", 0x34, 0);
    // BAR1: Mimir CCE SRAM.
    add_ss_bar_expected(registers, tile, "BASE_10", 0x40, kIdentityBase10Base + inst * kIdentityBase10Stride);
    add_ss_bar_expected(registers, tile, "BASE_11", 0x44, 0x12);
    add_ss_bar_expected(registers, tile, "LIMIT_10", 0x48, 0x80800000ULL + inst * 0x00800000ULL);
    add_ss_bar_expected(registers, tile, "LIMIT_11", 0x4C, 0x12);
    add_ss_bar_expected(registers, tile, "OFFSET_10", 0x50, 0x40000000ULL);
    add_ss_bar_expected(registers, tile, "OFFSET_11", 0x54, 0);
    // BAR2: GDDR, density code 2. Only BASE_21 and LIMIT_21 differ between D2D0 and D2D1.
    add_ss_bar_expected(registers, tile, "BASE_20", 0x60, 0);
    add_ss_bar_expected(registers, tile, "BASE_21", 0x64, (via_d2d1 ? 0x10020ULL : 0x10000ULL) + 2 * inst);
    add_ss_bar_expected(registers, tile, "LIMIT_20", 0x68, 0);
    add_ss_bar_expected(registers, tile, "LIMIT_21", 0x6C, (via_d2d1 ? 0x10022ULL : 0x10002ULL) + 2 * inst);
    add_ss_bar_expected(registers, tile, "OFFSET_20", 0x70, 0);
    add_ss_bar_expected(registers, tile, "OFFSET_21", 0x74, 0x8);
    // BAR3: spare.
    add_ss_bar_expected(registers, tile, "BASE_30", 0x80, 0);
    add_ss_bar_expected(registers, tile, "BASE_31", 0x84, 0);
    add_ss_bar_expected(registers, tile, "LIMIT_30", 0x88, 0x10000000ULL);
    add_ss_bar_expected(registers, tile, "LIMIT_31", 0x8C, 0);
    add_ss_bar_expected(registers, tile, "OFFSET_30", 0x90, 0);
    add_ss_bar_expected(registers, tile, "OFFSET_31", 0x94, 0);
}

void append_att_expected(std::vector<MimirRegister>& registers, const D2dTile& tile) {
    registers.push_back(make_register(
        tile_prefix(tile) + ".ATT.ENABLE", tile.mimir_index, att_address(tile, kAttEnableOffset), 1, false));
    for (const AttEndpointExpected& endpoint : kAttEndpointsExpected) {
        registers.push_back(make_register(
            fmt::format("{}.ATT.ENDPOINT{}", tile_prefix(tile), endpoint.entry_index),
            tile.mimir_index,
            att_endpoint_address(tile, endpoint.entry_index),
            endpoint.value,
            false));
    }
}

void append_cce_remap_expected(
    std::vector<MimirRegister>& registers, const metal_SocDescriptor& soc, uint32_t mimir_index) {
    // Chippy programs the remap; compare it with UMD's package windows.
    const std::array<tt::umd::CceDmriscRemapEntry, kCceRemapEntriesPerCce> expected_entries =
        tt::umd::cce_dmrisc_remap_entries(tt::umd::GrendelAddressWindows{}, mimir_index);
    for (uint32_t location = 0; location < locations_per_channel(soc); ++location) {
        for (uint32_t entry_index = 0; entry_index < kCceRemapEntriesPerCce; ++entry_index) {
            const tt::umd::CceDmriscRemapEntry& entry = expected_entries[entry_index];
            const uint64_t entry_base = cce_register_address(
                soc, location, kCceRemapBaseAddress + static_cast<uint64_t>(entry_index) * kCceRemapEntryStride);
            const std::string prefix = fmt::format("M{}.CCE{}.DMRISC_REMAP{}", mimir_index, location, entry_index);
            registers.push_back(make_register(
                prefix + ".START", mimir_index, entry_base + kCceRemapStartOffset, entry.region_start, true));
            registers.push_back(
                make_register(prefix + ".END", mimir_index, entry_base + kCceRemapEndOffset, entry.region_end, true));
            registers.push_back(make_register(
                prefix + ".REMAP_START", mimir_index, entry_base + kCceRemapRemapStartOffset, entry.local_base, true));
            registers.push_back(
                make_register(prefix + ".ATTRS", mimir_index, entry_base + kCceRemapAttrsOffset, 1, true));
        }
    }
}

void append_gddr_interleave_expected(std::vector<MimirRegister>& registers, uint32_t mimir_index) {
    registers.push_back(make_register(
        fmt::format("M{}.GDDR.INTLV_FLAGOUTSTATUS0", mimir_index),
        mimir_index,
        kGddrIntlvFlagOutStatus0Address,
        kGddrIntlvExpected,
        false));
    registers.push_back(make_register(
        fmt::format("M{}.GDDR.CAP_FLAGOUTSTATUS0", mimir_index),
        mimir_index,
        kGddrCapFlagOutStatus0Address,
        kGddrCapExpected,
        false));
}

std::string debug_misc_bit_label(uint32_t bit) {
    switch (bit) {
        case 9: return "routing_table_more_than_one_match";
        case 10: return "more_than_one_match";
        case 11: return "no_match";
        default: return "";
    }
}

std::string axi_error_bit_label(uint32_t bit) {
    if (bit <= 11) {
        return fmt::format("idma_slv_error_{}", bit);
    }
    if (bit >= 16 && bit <= 27) {
        return fmt::format("idma_dec_error_{}", bit - 16);
    }
    switch (bit) {
        case 32: return "dmrisc_spm_slv_error";
        case 33: return "dmrisc_spm_dec_error";
        case 34: return "dmrisc_mmio_slv_error";
        case 35: return "dmrisc_mmio_dec_error";
        case 36: return "msi_slv_error";
        case 37: return "msi_dec_error";
        default: return "";
    }
}

void append_bit_changes(
    D2dErrorDelta& delta,
    const std::vector<MimirRegister>& before,
    const std::vector<MimirRegister>& after,
    uint64_t mask,
    std::string (*label_for_bit)(uint32_t)) {
    for (const MimirRegister& after_reg : after) {
        uint64_t before_value = 0;
        for (const MimirRegister& before_reg : before) {
            if (before_reg.name == after_reg.name) {
                before_value = before_reg.value;
            }
        }
        for (uint32_t bit = 0; bit < 64; ++bit) {
            const uint64_t bit_mask = 1ULL << bit;
            if ((mask & bit_mask) == 0 || (after_reg.value & bit_mask) == 0) {
                continue;
            }
            const std::string description = fmt::format("{} bit {} ({})", after_reg.name, bit, label_for_bit(bit));
            if (before_value & bit_mask) {
                delta.inconclusive_bits.push_back(description);
            } else {
                delta.new_bits.push_back(description);
            }
        }
    }
}

std::vector<MimirRegister> expected_d2d_setup_config(const metal_SocDescriptor& soc) {
    std::vector<MimirRegister> registers;
    for (const D2dTile& tile : live_d2d_tiles(soc)) {
        append_ss_bar_expected(registers, tile);
        append_att_expected(registers, tile);
    }
    for (uint32_t mimir_index = 0; mimir_index < mimir_count(soc); ++mimir_index) {
        append_gddr_interleave_expected(registers, mimir_index);
    }
    return registers;
}

std::vector<MimirRegister> read_d2d_setup_config(const Cluster& cluster, const metal_SocDescriptor& soc, ChipId chip) {
    return read_layout(cluster, soc, chip, expected_d2d_setup_config(soc));
}

std::vector<MimirRegister> expected_cce_remap_config(const metal_SocDescriptor& soc) {
    std::vector<MimirRegister> registers;
    for (uint32_t mimir_index = 0; mimir_index < mimir_count(soc); ++mimir_index) {
        append_cce_remap_expected(registers, soc, mimir_index);
    }
    return registers;
}

// Chippy and UMD program the same remap values, so the host and remote checks share one layout.
std::vector<MimirRegister> read_cce_remap_config(const Cluster& cluster, const metal_SocDescriptor& soc, ChipId chip) {
    return read_layout(cluster, soc, chip, expected_cce_remap_config(soc));
}

std::vector<MimirRegister> expected_d2d_config(const metal_SocDescriptor& soc) {
    std::vector<MimirRegister> registers = expected_d2d_setup_config(soc);
    const std::vector<MimirRegister> remap = expected_cce_remap_config(soc);
    registers.insert(registers.end(), remap.begin(), remap.end());
    return registers;
}

std::vector<MimirRegister> read_d2d_config(const Cluster& cluster, const metal_SocDescriptor& soc, ChipId chip) {
    std::vector<MimirRegister> registers = read_d2d_setup_config(cluster, soc, chip);
    const std::vector<MimirRegister> remap = read_cce_remap_config(cluster, soc, chip);
    registers.insert(registers.end(), remap.begin(), remap.end());
    return registers;
}

std::vector<std::string> diff_d2d_config(
    const std::vector<MimirRegister>& expected, const std::vector<MimirRegister>& actual) {
    std::map<std::string, const MimirRegister*> expected_by_name;
    for (const MimirRegister& expected_reg : expected) {
        expected_by_name[expected_reg.name] = &expected_reg;
    }
    std::map<std::string, const MimirRegister*> actual_by_name;
    for (const MimirRegister& actual_reg : actual) {
        actual_by_name[actual_reg.name] = &actual_reg;
    }

    std::vector<std::string> mismatches;
    for (const MimirRegister& expected_reg : expected) {
        std::map<std::string, const MimirRegister*>::const_iterator found = actual_by_name.find(expected_reg.name);
        if (found == actual_by_name.end()) {
            mismatches.push_back(fmt::format("{}: missing from readback", expected_reg.name));
            continue;
        }
        const MimirRegister& actual_reg = *found->second;
        const uint64_t mask = compare_mask_for_name(expected_reg.name);
        if ((expected_reg.value & mask) != (actual_reg.value & mask)) {
            mismatches.push_back(fmt::format(
                "{}: expected 0x{:x}, read 0x{:x}", expected_reg.name, expected_reg.value, actual_reg.value));
        }
        if (expected_reg.name.ends_with(".ATTRS") || expected_reg.name.ends_with(".ATT.ENABLE")) {
            const uint64_t other_bits = ~compare_mask_for_name(expected_reg.name);
            if ((expected_reg.value & other_bits) != (actual_reg.value & other_bits)) {
                log_info(
                    tt::LogTest,
                    "{}: 0x{:x} differs from expected 0x{:x} outside the compared bits (not a mismatch)",
                    actual_reg.name,
                    actual_reg.value,
                    expected_reg.value);
            }
        }
    }
    for (const MimirRegister& actual_reg : actual) {
        if (expected_by_name.find(actual_reg.name) == expected_by_name.end()) {
            mismatches.push_back(fmt::format("{}: not in expected layout", actual_reg.name));
        }
    }
    return mismatches;
}

// Identity check: "" when ATT endpoint entry 64 == (4,6) and SS_BAR1 base == 0x8000_0000 + inst*8MiB on every
// live tile; otherwise the mismatch description.
std::string mimir_identity_mismatch(const Cluster& cluster, const metal_SocDescriptor& soc, ChipId chip) {
    std::vector<std::string> mismatches;
    for (const D2dTile& tile : live_d2d_tiles(soc)) {
        const std::string prefix = tile_prefix(tile);
        const uint64_t endpoint_value = read_smc_value(
            cluster, soc, chip, tile.mimir_index, att_endpoint_address(tile, kIdentityEndpointEntry), false);
        if (endpoint_value != kIdentityEndpointValue) {
            mismatches.push_back(fmt::format(
                "{}.ATT.ENDPOINT{} is 0x{:x}, expected 0x{:x}",
                prefix,
                kIdentityEndpointEntry,
                endpoint_value,
                kIdentityEndpointValue));
        }
        const uint64_t base10_value =
            read_smc_value(cluster, soc, chip, tile.mimir_index, ss_bar_address(tile, kSsBarBase10Offset), false);
        const uint64_t base10_expected = kIdentityBase10Base + tile.mimir_index * kIdentityBase10Stride;
        if (base10_value != base10_expected) {
            mismatches.push_back(
                fmt::format("{}.SS_BAR.BASE_10 is 0x{:x}, expected 0x{:x}", prefix, base10_value, base10_expected));
        }
    }
    return fmt::format("{}", fmt::join(mismatches, "; "));
}

std::string layout_mismatch_text(
    const std::string& prefix, const std::vector<MimirRegister>& expected, const std::vector<MimirRegister>& actual) {
    const std::vector<std::string> diffs = diff_d2d_config(expected, actual);
    if (diffs.empty()) {
        return "";
    }
    return prefix + fmt::format("{}", fmt::join(diffs, "; "));
}

std::string identity_mismatch_text(const Cluster& cluster, const metal_SocDescriptor& soc) {
    const std::string mismatch = mimir_identity_mismatch(cluster, soc, 0);
    if (mismatch.empty()) {
        return "";
    }
    return "identity mismatch: " + mismatch;
}

// Host suites: identity, then every setup and CCE remap register against Chippy.
std::string identity_result() {
    const Cluster& cluster = MetalContext::instance().get_cluster();
    const metal_SocDescriptor& soc = cluster.get_soc_desc(0);
    const std::string identity = identity_mismatch_text(cluster, soc);
    if (!identity.empty()) {
        return identity;
    }
    return layout_mismatch_text(
        "D2D setup readback mismatch: ", expected_d2d_config(soc), read_d2d_config(cluster, soc, 0));
}

// Remote suites, before the device is opened: identity, then SS_BAR, ATT and interleave. No DMRISC registers.
std::string preflight_result() {
    const Cluster& cluster = MetalContext::instance().get_cluster();
    const metal_SocDescriptor& soc = cluster.get_soc_desc(0);
    const std::string identity = identity_mismatch_text(cluster, soc);
    if (!identity.empty()) {
        return identity;
    }
    return layout_mismatch_text(
        "D2D setup readback mismatch: ", expected_d2d_setup_config(soc), read_d2d_setup_config(cluster, soc, 0));
}

// Remote suites, after the device is opened: the DMRISC remap against UMD's values.
std::string remap_result() {
    const Cluster& cluster = MetalContext::instance().get_cluster();
    const metal_SocDescriptor& soc = cluster.get_soc_desc(0);
    return layout_mismatch_text(
        "CCE DMRISC remap mismatch: ", expected_cce_remap_config(soc), read_cce_remap_config(cluster, soc, 0));
}

}  // namespace hw_state_impl

CoreCoord mimir_smc_core(const metal_SocDescriptor& soc, uint32_t mimir_index) {
    const std::vector<tt::umd::CoreCoord> smc_cores = soc.get_cores(CoreType::SMC, CoordSystem::TRANSLATED);
    TT_FATAL(mimir_index < smc_cores.size(), "Mimir {} has no SMC core in the descriptor", mimir_index);
    return {smc_cores[mimir_index].x, smc_cores[mimir_index].y};
}

std::string mimir_watcher_skip_reason() {
    if (!MetalContext::instance().rtoptions().get_watcher_enabled()) {
        return "";
    }
    return "watcher enabled: the host NOC sanitizer rejects SMC cores, so SMC register reads are skipped";
}

std::vector<CceResetState> read_cce_reset_states(const Cluster& cluster, const metal_SocDescriptor& soc, ChipId chip) {
    std::vector<CceResetState> states;
    const uint32_t num_cces = num_cces_from_soc(soc);
    for (uint32_t cce_index = 0; cce_index < num_cces; ++cce_index) {
        const uint32_t mimir_index = mimir_index_for_cce(cce_index);
        const uint64_t reset_address =
            hw_state_impl::cce_register_address(soc, location_for_cce(cce_index), hw_state_impl::kCceResetBaseAddress);
        CceResetState state;
        state.cce_index = cce_index;
        state.raw_lo = hw_state_impl::read_smc_word(cluster, soc, chip, mimir_index, reset_address);
        state.uncore_released = (state.raw_lo & hw_state_impl::kCceUncoreReleased) != 0;
        state.core_reset = (state.raw_lo >> 1) & 0xFF;
        states.push_back(state);
    }
    return states;
}

// Returns the raw 64-bit value. The vector occupies bits 51:0; upper bits are not masked.
std::vector<uint64_t> read_cce_reset_vectors(
    const Cluster& cluster, const metal_SocDescriptor& soc, ChipId chip, uint32_t cce_index) {
    std::vector<uint64_t> vectors(cce_hart_count());
    const uint32_t mimir_index = mimir_index_for_cce(cce_index);
    const uint32_t location = location_for_cce(cce_index);
    for (uint32_t hart = 0; hart < vectors.size(); ++hart) {
        vectors[hart] = hw_state_impl::read_smc_value(
            cluster,
            soc,
            chip,
            mimir_index,
            hw_state_impl::cce_register_address(
                soc,
                location,
                hw_state_impl::kCceResetVectorBaseAddress + hw_state_impl::kCceResetVectorHartStride * hart),
            true);
    }
    return vectors;
}

std::string host_only_guard_reason(
    const Cluster& cluster, const metal_SocDescriptor& soc, ChipId chip, bool require_uncore_released) {
    const std::string watcher_reason = mimir_watcher_skip_reason();
    if (!watcher_reason.empty()) {
        return watcher_reason;
    }
    for (const CceResetState& state : read_cce_reset_states(cluster, soc, chip)) {
        if (state.core_reset != 0) {
            return fmt::format(
                "CCE {} has harts out of reset (core_reset 0x{:x}); host-only test skipped",
                state.cce_index,
                state.core_reset);
        }
        if (require_uncore_released && !state.uncore_released) {
            return fmt::format("CCE {} uncore is held in reset; test skipped", state.cce_index);
        }
    }
    return "";
}

AttEndpointPrediction att_endpoint_for_remote_sram(uint32_t dest_cce) {
    return hw_state_impl::expected_att_endpoint(hw_state_impl::kFirstRemoteSramEntry + dest_cce);
}

std::optional<AttEndpointPrediction> att_endpoint_for_remote_gddr(uint32_t source_channel, uint32_t d2d_port) {
    for (const hw_state_impl::RemoteGddrEndpointEntry& entry : hw_state_impl::kRemoteGddrEndpointEntries) {
        if (entry.source_mimir == source_channel && entry.d2d_port == d2d_port) {
            return hw_state_impl::expected_att_endpoint(entry.entry_index);
        }
    }
    return std::nullopt;
}

D2dErrorSnapshot snapshot_d2d_errors(const Cluster& cluster, const metal_SocDescriptor& soc, ChipId chip) {
    D2dErrorSnapshot snapshot;
    for (const hw_state_impl::D2dTile& tile : hw_state_impl::live_d2d_tiles(soc)) {
        // DEBUG_MISC bits 9-11 are sticky until chip reset per the register description, with no software clear.
        snapshot.debug_misc.push_back(hw_state_impl::read_named(
            cluster,
            soc,
            chip,
            hw_state_impl::tile_prefix(tile) + ".ATT.DEBUG_MISC",
            tile.mimir_index,
            hw_state_impl::att_address(tile, hw_state_impl::kAttDebugMiscOffset),
            false));
    }
    for (uint32_t cce_index = 0; cce_index < num_cces_from_soc(soc); ++cce_index) {
        const uint32_t mimir_index = mimir_index_for_cce(cce_index);
        const uint32_t location = location_for_cce(cce_index);
        const std::string prefix = fmt::format("M{}.CCE{}", mimir_index, location);
        snapshot.axi_errors.push_back(hw_state_impl::read_named(
            cluster,
            soc,
            chip,
            prefix + ".AXI_READ_ERROR",
            mimir_index,
            hw_state_impl::cce_register_address(
                soc, location, hw_state_impl::kCceResetBaseAddress + hw_state_impl::kCceAxiReadErrorOffset),
            true));
        snapshot.axi_errors.push_back(hw_state_impl::read_named(
            cluster,
            soc,
            chip,
            prefix + ".AXI_WRITE_ERROR",
            mimir_index,
            hw_state_impl::cce_register_address(
                soc, location, hw_state_impl::kCceResetBaseAddress + hw_state_impl::kCceAxiWriteErrorOffset),
            true));
    }
    return snapshot;
}

D2dErrorDelta new_error_bits(const D2dErrorSnapshot& before, const D2dErrorSnapshot& after) {
    D2dErrorDelta delta;
    hw_state_impl::append_bit_changes(
        delta,
        before.debug_misc,
        after.debug_misc,
        hw_state_impl::kDebugMiscErrorBitsMask,
        hw_state_impl::debug_misc_bit_label);
    hw_state_impl::append_bit_changes(
        delta,
        before.axi_errors,
        after.axi_errors,
        hw_state_impl::kAxiErrorBitsMask,
        hw_state_impl::axi_error_bit_label);
    return delta;
}

bool expect_no_new_d2d_error_bits(
    const D2dErrorSnapshot& before, const D2dErrorSnapshot& after, const std::string& context) {
    const D2dErrorDelta delta = new_error_bits(before, after);
    for (const std::string& bit : delta.inconclusive_bits) {
        log_info(tt::LogTest, "{}: D2D error bit already set before the test (inconclusive): {}", context, bit);
    }
    for (const std::string& bit : delta.new_bits) {
        ADD_FAILURE() << context << ": new D2D error bit " << bit;
    }
    return delta.new_bits.empty();
}

std::vector<MimirRegister> read_debug_last(const Cluster& cluster, const metal_SocDescriptor& soc, ChipId chip) {
    std::vector<MimirRegister> registers;
    for (const hw_state_impl::D2dTile& tile : hw_state_impl::live_d2d_tiles(soc)) {
        const std::string prefix = hw_state_impl::tile_prefix(tile) + ".ATT.DEBUG_LAST_";
        registers.push_back(hw_state_impl::read_named(
            cluster,
            soc,
            chip,
            prefix + "SRC_ADDR",
            tile.mimir_index,
            hw_state_impl::att_address(tile, hw_state_impl::kAttDebugLastSrcOffset),
            true));
        registers.push_back(hw_state_impl::read_named(
            cluster,
            soc,
            chip,
            prefix + "DEST_ADDR",
            tile.mimir_index,
            hw_state_impl::att_address(tile, hw_state_impl::kAttDebugLastDestOffset),
            true));
        registers.push_back(hw_state_impl::read_named(
            cluster,
            soc,
            chip,
            prefix + "TRANS_SRC_ADDR",
            tile.mimir_index,
            hw_state_impl::att_address(tile, hw_state_impl::kAttDebugLastTransSrcOffset),
            true));
        registers.push_back(hw_state_impl::read_named(
            cluster,
            soc,
            chip,
            prefix + "TRANS_DEST_ADDR",
            tile.mimir_index,
            hw_state_impl::att_address(tile, hw_state_impl::kAttDebugLastTransDestOffset),
            true));
        registers.push_back(hw_state_impl::read_named(
            cluster,
            soc,
            chip,
            prefix + "TRANSLATION",
            tile.mimir_index,
            hw_state_impl::att_address(tile, hw_state_impl::kAttDebugLastTranslationOffset),
            false));
    }
    return registers;
}

void dump_debug_last(const Cluster& cluster, const metal_SocDescriptor& soc, ChipId chip) {
    for (const MimirRegister& reg : read_debug_last(cluster, soc, chip)) {
        log_warning(tt::LogTest, "{} = 0x{:x}", reg.name, reg.value);
    }
}

const std::string& mimir_identity_result() {
    static const std::string result = hw_state_impl::identity_result();
    return result;
}

const std::string& mimir_remote_preflight_result() {
    static const std::string result = hw_state_impl::preflight_result();
    return result;
}

const std::string& mimir_remote_remap_result() {
    static const std::string result = hw_state_impl::remap_result();
    return result;
}

}  // namespace tt::tt_metal::mimir_test
