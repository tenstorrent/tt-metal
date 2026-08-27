// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Host-only verification that the temporary bring-up ATT register images are
// bit-for-bit consistent with the transcribed map configurations: every
// kernel-visible mask-table entry in the generated QSR1 image is decoded
// through the hardware control-word layout and compared field-by-field with
// the config windows, the endpoint-table writes are compared row-for-row with
// the transcribed endpoint tables, and the compact aether program is checked
// the same way. This mechanizes the by-hand image verification so a
// regenerated image or edited config cannot silently disagree. No device is
// opened.

#include <gtest/gtest.h>

#include <cstdint>
#include <optional>

#include "internal/tt-2xx/quasar/noc/att/temporary_programming/grendel_qsr1_att_data.h"
#include "internal/tt-2xx/quasar/noc/att/temporary_programming/quasar_aether_2x3_att_data.h"

namespace {

using noc_att::Window;

// The value a register holds after the full replay: the last write wins, and
// the delivery overrides replay after the main image.
std::optional<std::uint32_t> replayed_value(std::uint32_t address) {
    std::optional<std::uint32_t> value;
    const noc_att::ProgramImage& image = grendel_qsr1_att_program::PROGRAM_IMAGE;
    for (std::uint32_t i = 0; i < image.write_count; ++i) {
        if (image.writes[i].address == address) {
            value = image.writes[i].data;
        }
    }
    for (std::uint32_t i = 0; i < image.override_count; ++i) {
        if (image.overrides[i].address == address) {
            value = image.overrides[i].data;
        }
    }
    return value;
}

void expect_mask_slot_matches_window(
    std::uint32_t slot, const Window& window, std::uint64_t expected_bar, const char* name) {
    SCOPED_TRACE(name);

    const auto control = replayed_value(
        noc_att::mask_register_address(slot, NOC_ADDRESS_TRANSLATION_TABLE_A_MASK_TABLE_ENTRY_REG_OFFSET));
    ASSERT_TRUE(control.has_value());
    NOC_ADDRESS_TRANSLATION_TABLE_MASK_TABLE_ENTRY_reg_u decoded;
    decoded.val = *control;
    EXPECT_EQ(decoded.f.mask, window.mask_bits);
    EXPECT_EQ(decoded.f.ep_id_idx, window.endpoint_shift);
    EXPECT_EQ(decoded.f.ep_id_size, window.endpoint_size);
    EXPECT_EQ(decoded.f.table_offset, window.endpoint_table_offset);
    EXPECT_EQ(decoded.f.translate_addr, window.translate_address ? 1u : 0u);
    // The union decode and the encode helper must agree on the layout.
    EXPECT_EQ(*control, noc_att::mask_entry_control_word(window));

    const auto compare_lo = replayed_value(
        noc_att::mask_register_address(slot, NOC_ADDRESS_TRANSLATION_TABLE_A_MASK_TABLE_EP_LO_REG_OFFSET));
    const auto compare_hi = replayed_value(
        noc_att::mask_register_address(slot, NOC_ADDRESS_TRANSLATION_TABLE_A_MASK_TABLE_EP_HI_REG_OFFSET));
    ASSERT_TRUE(compare_lo.has_value());
    ASSERT_TRUE(compare_hi.has_value());
    EXPECT_EQ((std::uint64_t{*compare_hi} << 32) | *compare_lo, window.compare);

    const auto bar_lo = replayed_value(
        noc_att::mask_register_address(slot, NOC_ADDRESS_TRANSLATION_TABLE_A_MASK_TABLE_BAR_LO_REG_OFFSET));
    const auto bar_hi = replayed_value(
        noc_att::mask_register_address(slot, NOC_ADDRESS_TRANSLATION_TABLE_A_MASK_TABLE_BAR_HI_REG_OFFSET));
    ASSERT_TRUE(bar_lo.has_value());
    ASSERT_TRUE(bar_hi.has_value());
    EXPECT_EQ((std::uint64_t{*bar_hi} << 32) | *bar_lo, expected_bar);
}

// ---------------------------------------------------------------------------
// QSR1 generated image vs the transcribed configuration
// ---------------------------------------------------------------------------

TEST(QuasarAttImageQsr1, MaskSlotsMatchTheConfigWindows) {
    expect_mask_slot_matches_window(4, grendel_qsr1_att_config::WORKER_WINDOW, 0, "worker slot 4");
    // The GDDR window rebases through Mimir at BAR 0x8_0000_0000 (dump slot 5).
    expect_mask_slot_matches_window(5, grendel_qsr1_att_config::DRAM_WINDOW, 0x800000000ull, "dram slot 5");
    expect_mask_slot_matches_window(
        13, grendel_qsr1_att_config::LOOPBACK_SCRATCH_WINDOW, 0, "loopback-scratch slot 13");
    expect_mask_slot_matches_window(14, grendel_qsr1_att_config::TILE_WINDOW, 0, "tile slot 14");
}

TEST(QuasarAttImageQsr1, WorkerEndpointRowsMatchTheConfigTable) {
    for (std::uint32_t i = 0; i < 32; ++i) {
        const std::uint32_t endpoint_index = grendel_qsr1_att_config::WORKER_WINDOW.endpoint_table_offset + i;
        const auto value = replayed_value(noc_att::endpoint_register_address(endpoint_index));
        ASSERT_TRUE(value.has_value()) << "no write for worker endpoint " << endpoint_index;
        EXPECT_EQ(*value, grendel_qsr1_att_config::ATT_WORKER_ENDPOINT_WORDS[i]) << "worker selector " << i;
    }
}

TEST(QuasarAttImageQsr1, FullTileEndpointRowsMatchTheConfigTable) {
    for (std::uint32_t i = 0; i < 60; ++i) {
        const std::uint32_t endpoint_index = grendel_qsr1_att_config::TILE_WINDOW.endpoint_table_offset + i;
        const auto value = replayed_value(noc_att::endpoint_register_address(endpoint_index));
        ASSERT_TRUE(value.has_value()) << "no write for full-tile endpoint " << endpoint_index;
        EXPECT_EQ(*value, grendel_qsr1_att_config::ATT_FULL_TILE_ENDPOINT_WORDS[i]) << "tile selector " << i;
    }
}

TEST(QuasarAttImageQsr1, ReplayContractHolds) {
    const noc_att::ProgramImage& image = grendel_qsr1_att_program::PROGRAM_IMAGE;

    // The replay withholds enable-table writes and enables once at the end;
    // that is only sound if the image never intends the register disabled.
    for (std::uint32_t i = 0; i < image.write_count; ++i) {
        if (image.writes[i].address == image.enable_tables_address) {
            EXPECT_EQ(image.writes[i].data, 1u);
        }
    }

    // The per-initiator self patch targets the boot-patched ep256 row.
    EXPECT_EQ(
        image.local_endpoint_address,
        noc_att::endpoint_register_address(grendel_qsr1_att_config::TILE_WINDOW.endpoint_table_offset));
    EXPECT_EQ(image.debug_misc_address, +NOC_ADDRESS_TRANSLATION_TABLE_A_DEBUG_MISC_REG_ADDR);

    // Every write stays inside this NIU's translation-table register block.
    constexpr std::uint32_t block_begin = NOC_ADDRESS_TRANSLATION_TABLE_A_REG_MAP_BASE_ADDR;
    constexpr std::uint32_t block_end = noc_att::endpoint_register_address(1024);
    for (std::uint32_t i = 0; i < image.write_count; ++i) {
        EXPECT_GE(image.writes[i].address, block_begin);
        EXPECT_LT(image.writes[i].address, block_end);
    }
    for (std::uint32_t i = 0; i < image.override_count; ++i) {
        EXPECT_GE(image.overrides[i].address, block_begin);
        EXPECT_LT(image.overrides[i].address, block_end);
    }
}

// ---------------------------------------------------------------------------
// Aether compact program vs the transcribed configuration
// ---------------------------------------------------------------------------

TEST(QuasarAttImageAether, MaskEntriesMatchTheConfigWindows) {
    const noc_att::Program& program = quasar_aether_2x3_att_program::PROGRAM_IMAGE;
    ASSERT_EQ(program.mask_count, 2u);

    EXPECT_EQ(program.masks[0].slot, 0);
    EXPECT_EQ(
        noc_att::mask_entry_control_word(program.masks[0].window),
        noc_att::mask_entry_control_word(quasar_aether_2x3_att_config::LOCAL_WINDOW));
    EXPECT_EQ(program.masks[0].window.compare, quasar_aether_2x3_att_config::LOCAL_WINDOW.compare);
    EXPECT_EQ(program.masks[0].bar, 0u);

    EXPECT_EQ(program.masks[1].slot, 1);
    EXPECT_EQ(
        noc_att::mask_entry_control_word(program.masks[1].window),
        noc_att::mask_entry_control_word(quasar_aether_2x3_att_config::REMOTE_WINDOW));
    EXPECT_EQ(program.masks[1].window.compare, quasar_aether_2x3_att_config::REMOTE_WINDOW.compare);
    EXPECT_EQ(program.masks[1].bar, 0u);
}

TEST(QuasarAttImageAether, EndpointEntriesMatchTheConfigTable) {
    const noc_att::Program& program = quasar_aether_2x3_att_program::PROGRAM_IMAGE;
    const std::uint16_t remote_offset = quasar_aether_2x3_att_config::REMOTE_WINDOW.endpoint_table_offset;

    ASSERT_EQ(
        program.endpoint_count,
        sizeof(quasar_aether_2x3_att_config::ATT_FULL_TILE_ENDPOINT_WORDS) / sizeof(std::uint16_t));
    for (std::uint32_t i = 0; i < program.endpoint_count; ++i) {
        const noc_att::EndpointEntry& entry = program.endpoints[i];
        ASSERT_GE(entry.index, remote_offset);
        const std::uint32_t selector = entry.index - remote_offset;
        // Endpoint words encode (y << 6) | x.
        EXPECT_EQ(
            (std::uint32_t{entry.y} << 6) | entry.x,
            quasar_aether_2x3_att_config::ATT_FULL_TILE_ENDPOINT_WORDS[selector])
            << "remote selector " << selector;
    }

    // The pass-through local window resolves through the per-initiator entry 0.
    EXPECT_EQ(program.local_endpoint_index, quasar_aether_2x3_att_config::LOCAL_WINDOW.endpoint_table_offset);
}

}  // namespace
