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

void TileImpl::zero_hart_protocol_unlocked() {
    const auto tile = core();
    const x280_hart_protocol_t protocol{};
    transport().write_block(id().chip_id, tile, X280_HART_PROTOCOL_BASE, &protocol, sizeof(protocol));
}

}  // namespace x280::detail
