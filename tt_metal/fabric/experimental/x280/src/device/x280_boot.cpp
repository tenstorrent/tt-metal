// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// TileImpl's resident-idle image, per-hart protocol, trap, and reset-vector operations.
// Shared PLL and reset-register sequencing is on ChipImpl.

#include "x280_tile_impl.hpp"

#include "x280_format.hpp"
#include "x280_hw_addrs.hpp"

#include "x280_address_map.h"
#include "x280_hart_protocol.h"

#include <array>
#include <fstream>

namespace x280::detail {

HartState TileImpl::read_hart_state_unlocked(uint32_t hart) const {
    HartState observation;
    observation.hart_id = hart;
    const auto tile = core();
    const uint64_t control_address =
        X280_HART_PROTOCOL_BASE + offsetof(x280_hart_protocol_t, control) + hart * sizeof(observation.control);
    const uint64_t state_address =
        X280_HART_PROTOCOL_BASE + offsetof(x280_hart_protocol_t, state) + hart * sizeof(observation.state);
    const uint64_t trap_address =
        X280_HART_PROTOCOL_BASE + offsetof(x280_hart_protocol_t, trap) + hart * sizeof(observation.trap);
    transport().read_block(id().chip_id, tile, control_address, &observation.control, sizeof(observation.control));
    transport().read_block(id().chip_id, tile, state_address, &observation.state, sizeof(observation.state));
    transport().read_block(id().chip_id, tile, trap_address, &observation.trap, sizeof(observation.trap));

    if (is_all_ones(observation.state.phase) && is_all_ones(observation.control.command) &&
        is_all_ones(observation.state.heartbeat)) {
        observation.invalid_reason = "phase, command, and heartbeat read all ones";
        return observation;
    }
    observation.observation_valid = true;
    return observation;
}

std::array<HartState, kHartsPerTile> TileImpl::read_hart_states_unlocked() const {
    std::array<HartState, kHartsPerTile> harts;
    for (uint32_t hart = 0; hart < kHartsPerTile; ++hart) {
        harts[hart] = read_hart_state_unlocked(hart);
    }
    return harts;
}

std::vector<uint8_t> TileImpl::read_image_file(const std::string& path, const char* what, const char* build_hint) {
    if (path.empty()) {
        fail(std::string("no ") + what + " path was supplied");
    }
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file) {
        fail(std::string("cannot open ") + what + " '" + path + "'. Build one with: " + build_hint);
    }
    const std::streamsize size = file.tellg();
    if (size < 0) {
        fail(std::string("cannot size ") + what + " '" + path + "'");
    }
    file.seekg(0, std::ios::beg);

    std::vector<uint8_t> image(static_cast<size_t>(size));
    if (size > 0 && !file.read(reinterpret_cast<char*>(image.data()), size)) {
        fail(std::string("short read on ") + what + " '" + path + "'");
    }
    return image;
}

void TileImpl::load_idle_image_unlocked(const std::vector<uint8_t>& image) {
    const auto check = validate_idle_image(X280_IDLE_FW_LOAD_ADDR, image.size());
    if (!check.ok) {
        fail(check.reason);
    }

    // The block path takes 4-byte units; pad rather than issue a partial tail.
    std::vector<uint8_t> padded = image;
    padded.resize((padded.size() + 3) & ~static_cast<size_t>(3), 0);

    const auto tile = core();
    transport().write_block(id().chip_id, tile, X280_IDLE_FW_LOAD_ADDR, padded.data(), padded.size());

    std::vector<uint8_t> readback(padded.size(), 0);
    transport().read_block(id().chip_id, tile, X280_IDLE_FW_LOAD_ADDR, readback.data(), readback.size());
    if (readback != padded) {
        size_t first = 0;
        while (first < padded.size() && readback[first] == padded[first]) {
            first++;
        }
        fail(
            "resident-idle image readback mismatch at LIM+" + hex(first) + ": " +
            format_readback_values(padded[first], readback[first]) + ". Reset was not released.");
    }
}

void TileImpl::set_reset_vectors_unlocked(uint64_t entry_addr) {
    const uint32_t lo = static_cast<uint32_t>(entry_addr & 0xFFFFFFFFull);
    const uint32_t hi = static_cast<uint32_t>(entry_addr >> 32);
    const auto tile = core();

    for (uint32_t hart = 0; hart < kHartsPerTile; hart++) {
        const uint64_t base = kL2cpuRegBase + hart * kResetVectorStride;
        transport().write_reg32(id().chip_id, tile, base + 0, lo);
        transport().write_reg32(id().chip_id, tile, base + 4, hi);
    }

    for (uint32_t hart = 0; hart < kHartsPerTile; hart++) {
        const uint64_t base = kL2cpuRegBase + hart * kResetVectorStride;
        const uint32_t seen_lo = transport().read_reg32(id().chip_id, tile, base + 0);
        const uint32_t seen_hi = transport().read_reg32(id().chip_id, tile, base + 4);
        if (seen_lo != lo || seen_hi != hi) {
            fail(
                "reset vector for hart " + std::to_string(hart) + " did not take: " +
                format_readback_values(entry_addr, (static_cast<uint64_t>(seen_hi) << 32) | seen_lo) + " at " +
                hex(base) +
                ". The L2CPU register file is reached through the dynamic TLB path rather than the static window at "
                "LIM base; refusing to release reset onto unverified vectors, since that cannot be undone without an "
                "ASIC reset.");
        }
    }
}

void TileImpl::set_rnmi_handlers_unlocked(bool install) {
    const auto tile = core();
    const uint64_t entry = install ? X280_IDLE_RNMI_ENTRY_ADDR : 0;
    const uint64_t exc = install ? X280_IDLE_RNMI_EXC_ADDR : 0;
    const std::array<uint32_t, 4> words = {
        static_cast<uint32_t>(entry),
        static_cast<uint32_t>(entry >> 32),
        static_cast<uint32_t>(exc),
        static_cast<uint32_t>(exc >> 32)};

    transport().write_reg32(id().chip_id, tile, kRnmiTrigger, 0);
    for (uint32_t hart = 0; hart < kHartsPerTile; hart++) {
        const uint64_t base = kRnmiHandlerBase + hart * kRnmiHandlerStride;
        for (uint32_t i = 0; i < words.size(); i++) {
            transport().write_reg32(id().chip_id, tile, base + i * sizeof(uint32_t), words[i]);
        }
    }

    const uint32_t trigger = transport().read_reg32(id().chip_id, tile, kRnmiTrigger);
    if (trigger != 0) {
        fail("RNMI trigger word did not clear: " + format_readback_values(0, trigger) + ". Reset was not released.");
    }
    for (uint32_t hart = 0; hart < kHartsPerTile; hart++) {
        const uint64_t base = kRnmiHandlerBase + hart * kRnmiHandlerStride;
        for (uint32_t i = 0; i < words.size(); i++) {
            const uint32_t seen = transport().read_reg32(id().chip_id, tile, base + i * sizeof(uint32_t));
            if (seen != words[i]) {
                fail(
                    "RNMI handler word " + std::to_string(i) + " of hart " + std::to_string(hart) +
                    " did not take: " + format_readback_values(words[i], seen) +
                    ". Reset was not released: a stray RNMI would jump to an unverified address.");
            }
        }
    }
}

std::string TileImpl::rnmi_not_installed_unlocked() const {
    const auto tile = core();
    for (uint32_t hart = 0; hart < kHartsPerTile; hart++) {
        const uint64_t base = kRnmiHandlerBase + hart * kRnmiHandlerStride;
        const uint64_t entry = transport().read_reg32(id().chip_id, tile, base + 0) |
                               (static_cast<uint64_t>(transport().read_reg32(id().chip_id, tile, base + 4)) << 32);
        const uint64_t exc = transport().read_reg32(id().chip_id, tile, base + 8) |
                             (static_cast<uint64_t>(transport().read_reg32(id().chip_id, tile, base + 12)) << 32);
        if (entry != X280_IDLE_RNMI_ENTRY_ADDR || exc != X280_IDLE_RNMI_EXC_ADDR) {
            return "hart " + std::to_string(hart) + " RNMI handlers are " + hex(entry) + " / " + hex(exc) +
                   ", not the resident idle's " + hex(X280_IDLE_RNMI_ENTRY_ADDR) + " / " +
                   hex(X280_IDLE_RNMI_EXC_ADDR) + ": the tile was booted without them";
        }
    }
    const uint64_t marker = transport().read_u64(id().chip_id, tile, X280_IDLE_RNMI_MAGIC_ADDR);
    if (!idle_rnmi_marker_matches(marker)) {
        return "the resident idle's RNMI marker reads " + hex(marker) + ", not " + hex(X280_IDLE_RNMI_MAGIC) +
               ": the idle in LIM does not carry this runtime's handler";
    }
    return {};
}

void TileImpl::zero_hart_protocol_unlocked() {
    const auto tile = core();
    const x280_hart_protocol_t protocol{};
    transport().write_block(id().chip_id, tile, X280_HART_PROTOCOL_BASE, &protocol, sizeof(protocol));
}

}  // namespace x280::detail
