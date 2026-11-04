// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "x280_chip_impl.hpp"
#include "x280_format.hpp"
#include "x280_hw_addrs.hpp"

#include <algorithm>

namespace x280::detail {

ChipImpl::ChipImpl(int chip_id, std::shared_ptr<X280MetalTransport> transport) :
    chip_id_(chip_id), transport_(std::move(transport)) {
    if (!transport_) {
        fail("cannot construct a chip context without the runtime's Metal transport");
    }
    if (!transport_->has_chip(chip_id_)) {
        fail("chip " + std::to_string(chip_id_) + " is not present in the cluster Metal opened");
    }
    if (!transport_->is_mmio_local(chip_id_)) {
        fail(
            "chip " + std::to_string(chip_id_) +
            " is not MMIO-local. A remote chip has no TLB manager and its register accesses degrade to plain "
            "reads, so the component refuses to drive one.");
    }
    arc_ = transport_->arc_core(chip_id_);
    l2cpu_cores_ = transport_->l2cpu_cores(chip_id_);
    if (l2cpu_cores_.empty()) {
        fail("chip " + std::to_string(chip_id_) + " reports no unharvested L2CPU tiles");
    }

    // UMD constructs this list from its fixed Blackhole L2CPU table after
    // applying ENABLED_L2CPU as the harvesting mask.
    std::vector<int> hardware_indices;
    std::vector<TileIdentity> identities;
    for (const auto& core : l2cpu_cores_) {
        const int hw_index = hw_l2cpu_from_noc0(static_cast<uint32_t>(core.x), static_cast<uint32_t>(core.y));
        if (hw_index < 0) {
            fail(
                "chip " + std::to_string(chip_id_) + " reports L2CPU " + label_noc0_xy(core.x, core.y) +
                " outside the documented X280 tile set");
        }
        if (std::find(hardware_indices.begin(), hardware_indices.end(), hw_index) != hardware_indices.end()) {
            fail(
                "chip " + std::to_string(chip_id_) + " reports hardware L2CPU index " + std::to_string(hw_index) +
                " more than once");
        }
        hardware_indices.push_back(hw_index);

        TileIdentity identity;
        identity.chip_id = chip_id_;
        identity.hw_index = hw_index;
        identity.noc_x = static_cast<uint32_t>(core.x);
        identity.noc_y = static_cast<uint32_t>(core.y);
        identity.num_harts = kHartsPerTile;
        identities.push_back(identity);
    }
    std::sort(identities.begin(), identities.end(), [](const TileIdentity& a, const TileIdentity& b) {
        return a.hw_index < b.hw_index;
    });
    tile_identities_ = std::move(identities);
}

tt::umd::CoreCoord ChipImpl::core_for(int hw_index) const {
    for (const auto& core : l2cpu_cores_) {
        const int core_hw_index = hw_l2cpu_from_noc0(static_cast<uint32_t>(core.x), static_cast<uint32_t>(core.y));
        if (core_hw_index == hw_index) {
            return core;
        }
    }
    fail(
        "no unharvested L2CPU with hardware index " + std::to_string(hw_index) + " on chip " +
        std::to_string(chip_id_));
}

const TileIdentity& ChipImpl::tile_identity(int hw_index) const {
    for (const auto& identity : tile_identities_) {
        if (identity.hw_index == hw_index) {
            return identity;
        }
    }
    fail("no tile identity for L2CPU " + std::to_string(hw_index) + " on chip " + std::to_string(chip_id_));
}

std::vector<int> ChipImpl::tile_hw_indices() const {
    std::vector<int> indices;
    indices.reserve(tile_identities_.size());
    for (const auto& identity : tile_identities_) {
        indices.push_back(identity.hw_index);
    }
    return indices;
}

uint32_t ChipImpl::read_l2cpu_reset_raw() const { return transport_->read_reg32(chip_id_, arc_, kL2cpuResetReg); }

bool ChipImpl::tile_reset_released(uint32_t reset_register, int hw_index) {
    const int bit = l2cpu_reset_bit(hw_index);
    return bit >= 0 && (reset_register & (1u << bit)) != 0;
}

uint32_t ChipImpl::read_l2cpu_reset_checked() const {
    return transport_->read_reg32_checked(chip_id_, arc_, kL2cpuResetReg, "L2CPU_RESET");
}

bool ChipImpl::all_tiles_held_in_reset() const { return (read_l2cpu_reset_checked() & 0xF0u) == 0; }

void ChipImpl::release_tile_reset(int hw_index) {
    const int bit = l2cpu_reset_bit(hw_index);
    if (bit < 0) {
        fail("no reset bit for L2CPU index " + std::to_string(hw_index));
    }

    const uint32_t before = read_l2cpu_reset_checked();
    transport_->write_reg32(chip_id_, arc_, kL2cpuResetReg, before | (1u << bit));
    const uint32_t after = read_l2cpu_reset_checked();
    if ((after & (1u << bit)) == 0) {
        fail("L2CPU " + std::to_string(hw_index) + " reset release did not take: L2CPU_RESET is " + hex(after));
    }
}

void ChipImpl::release_tile_resets(const std::vector<int>& hw_indices) {
    if (hw_indices.empty()) {
        fail("no L2CPU indices given to release");
    }

    uint32_t mask = 0;
    for (int hw_index : hw_indices) {
        const int bit = l2cpu_reset_bit(hw_index);
        if (bit < 0) {
            fail("no reset bit for L2CPU index " + std::to_string(hw_index));
        }
        mask |= (1u << bit);
    }

    const uint32_t before = read_l2cpu_reset_checked();
    transport_->write_reg32(chip_id_, arc_, kL2cpuResetReg, before | mask);
    const uint32_t after = read_l2cpu_reset_checked();
    if ((after & mask) != mask) {
        fail("L2CPU reset release did not take for every selected tile: L2CPU_RESET is " + hex(after));
    }
}

PllEstablishResult ChipImpl::establish_pll(uint32_t mhz, std::chrono::milliseconds confirm_timeout) {
    return establish_l2cpu_pll(*transport_, chip_id_, arc_, tile_hw_indices(), mhz, confirm_timeout);
}

}  // namespace x280::detail
