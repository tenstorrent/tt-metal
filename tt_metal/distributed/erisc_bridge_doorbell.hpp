// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// Ringing a router's receiver doorbell from the host over MMIO: stream registers are mapped NOC
// addresses, so H2E needs no router change. Three silent-zero traps -- contract §7.1a-c.
#pragma once

#include <cstdint>
#include <string>

// The Hal CLASS lives here; <tt-metalium/hal.hpp> is the free-function namespace and does not
// declare it.
#include "llrt/hal.hpp"
#include <tt-metalium/device.hpp>

namespace tt::tt_fabric::erisc_bridge {

// STREAM_REG_ADDR(stream_id, reg_id) from noc_overlay_parameters.h, host side.
inline std::uint32_t stream_reg_addr(const tt::tt_metal::Hal& hal, std::uint32_t stream_id, std::uint32_t reg_id) {
    return hal.get_noc_overlay_start_addr() + stream_id * hal.get_noc_stream_reg_space_size() + (reg_id << 2);
}

// What remote_update_ptr_val writes. The hardware ACCUMULATES what lands here: writing 1 adds
// one. Reading it back gives the UPDATE register, not the count -- see available_addr().
inline std::uint32_t doorbell_addr(const tt::tt_metal::Hal& hal, std::uint32_t stream_id) {
    return stream_reg_addr(hal, stream_id, hal.get_noc_stream_remote_dest_buf_space_available_update_reg_index());
}

// What get_ptr_val() reads: the current value, a DIFFERENT register from the one written above.
inline std::uint32_t available_addr(const tt::tt_metal::Hal& hal, std::uint32_t stream_id) {
    return stream_reg_addr(hal, stream_id, hal.get_noc_stream_remote_dest_buf_space_available_reg_index());
}

// ARMS THE ACCUMULATOR, and nothing counts until it is written: AVAILABLE is documented
// "updated automatically to maximum value when STREAM_REMOTE_DEST_BUF_SIZE_REG is updated".
inline std::uint32_t buf_size_addr(const tt::tt_metal::Hal& hal, std::uint32_t stream_id) {
    return stream_reg_addr(hal, stream_id, hal.get_noc_stream_remote_dest_buf_size_reg_index());
}

// get_ptr_val() masks on Blackhole; a host reading the raw register sees fields the router does
// not. REMOTE_DEST_WORDS_FREE_WIDTH is MEM_WORD_ADDR_WIDTH = 17.
inline constexpr std::uint32_t kWordsFreeWidthBlackhole = 17;
constexpr std::uint32_t available_value(std::uint32_t raw, std::uint32_t words_free_width) {
    return raw & ((1u << words_free_width) - 1u);
}

// The shift is NOT in the HAL. On Blackhole it is REMOTE_DEST_BUF_SPACE_AVAILABLE_UPDATE_DEST_NUM
// (0) + ..._DEST_NUM_WIDTH (6) = 6, so it stays a parameter rather than a baked-in constant.
inline constexpr std::uint32_t kWordsFreeIncShiftBlackhole = 6;
constexpr std::uint32_t doorbell_value(std::int32_t val, std::uint32_t words_free_inc_shift) {
    return static_cast<std::uint32_t>(val) << words_free_inc_shift;
}

// An ERISC has 32 streams (ETH_NOC_NUM_STREAMS) and a Tensix 64. An id past the end reads and
// writes as zero, which looks exactly like an unreachable core -- range-check before trusting a 0.
inline constexpr std::uint32_t kEthNumStreams = 32;
constexpr bool stream_id_valid_for_eth(std::uint32_t stream_id) { return stream_id < kEthNumStreams; }

// THE ACCUMULATOR DOES NOT REST AT ZERO: a running total since fabric init, read as 131066
// (-5 in 17 bits) where a 4-chip box reads 0. Baseline at open, modular difference -- §7.2.
constexpr std::uint32_t occupancy_since(std::uint32_t now, std::uint32_t baseline, std::uint32_t width) {
    return (now - baseline) & ((1u << width) - 1u);
}

// STREAM REGISTERS ARE NOT L1: WriteToDeviceL1 reaches cluster.write_core, never 0xFFB4xxxx --
// it does not fail, it reads and writes zeros (§7.1a). Defined in erisc_bridge_reg_access.cpp.
std::uint32_t reg_read(
    tt::tt_metal::IDevice* dev, const tt::tt_metal::CoreCoord& logical, tt::CoreType ct, std::uint32_t addr);
bool reg_write(
    tt::tt_metal::IDevice* dev,
    const tt::tt_metal::CoreCoord& logical,
    tt::CoreType ct,
    std::uint32_t addr,
    std::uint32_t val);

// L1 from a BARE POINTER. WriteToDeviceL1 takes a vector, which would force a copy of bytes
// that already sit in the arena -- the one copy this leg exists to avoid.
bool write_bytes_to_l1(
    tt::tt_metal::IDevice* dev,
    const tt::tt_metal::CoreCoord& logical,
    std::uint32_t addr,
    const void* src,
    std::uint32_t bytes,
    tt::CoreType ct = tt::CoreType::ETH);

// Clock anchor: the two clocks share no base, so a bracketed read (t1, cycle, t2) pairs them to
// within half the round trip. Keep the tightest sample; AICLK sets the rate and moves at runtime.
struct ClockAnchor {
    bool ok = false;
    std::uint32_t cyc = 0;        // the ERISC wall clock, low 32 bits, as the sender stamps it
    double host_us = 0.0;         // steady_clock microseconds corresponding to cyc
    double cyc_per_us = 0.0;      // AICLK, from the device, never assumed
    double uncertainty_us = 0.0;  // half the tightest bracket achieved
    std::string why;              // set when !ok
};

// Blackhole: RISCV_DEBUG_REGS_START_ADDR | 0x1F0, the register get_timestamp() reads on device.
// In the 0xFFBxxxxx window reg_read reaches the stream registers, not L1.
inline constexpr std::uint32_t kEriscWallClockLo = 0xFFB121F0u;

ClockAnchor measure_clock_anchor(
    tt::tt_metal::IDevice* dev, const tt::tt_metal::CoreCoord& eth_logical, std::uint32_t samples = 64);

// The read-back half: writing alone cannot tell "bytes arrived" from "bytes were dropped and the
// counters are literals" -- which is what H2E reported before this existed.
bool read_bytes_from_l1(
    tt::tt_metal::IDevice* dev,
    const tt::tt_metal::CoreCoord& logical,
    std::uint32_t addr,
    void* dst,
    std::uint32_t bytes,
    tt::CoreType ct = tt::CoreType::ETH);

// Host +1 per inject, router -1 per packet processed, so AVAILABLE is the channel's occupancy.
// The margin is deliberate: "processed" is a proxy for "reusable", so leave slack.
struct Credit {
    std::uint32_t slots = 0;   // the receiver channel's slot count
    std::uint32_t margin = 2;  // slack against the decrement being a proxy
    std::uint64_t stalls = 0;  // times the host had to wait for the router

    std::uint32_t usable() const { return slots > margin ? slots - margin : 1; }
    bool has_room(std::uint32_t outstanding) const { return outstanding < usable(); }
};

}  // namespace tt::tt_fabric::erisc_bridge
