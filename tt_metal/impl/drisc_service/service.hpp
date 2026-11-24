// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Host side of the DRISC data mover service. The device-side contract is hostdev/drisc_service/abi.hpp. Needs no
// metal device: a bare UMD cluster is enough.

#pragma once

#include <array>
#include <cstdint>
#include <string>

#include <tt-metalium/vocabulary/device_types.hpp>
#include <tt-umd/types/core_coordinates.hpp>

#include "hostdev/drisc_service/abi.hpp"
#include "impl/drisc_service/seat_lock.hpp"
#include "internal/tt-1xx/blackhole/noc/noc_parameters.h"

struct metal_SocDescriptor;

namespace tt::umd {
class Cluster;
}  // namespace tt::umd

namespace tt::llrt {
class RunTimeOptions;
}  // namespace tt::llrt

namespace tt::tt_metal::internal::drisc_service {

// One DRISC the service runs on, and the NoC its kernel drives.
struct Seat {
    ChipId chip;
    uint32_t bank;
    uint32_t noc;
    tt::umd::CoreCoord translated;
    SeatId id;
    uint8_t noc2axi_mask;  // NIUs DRISC firmware leaves in NOC2AXI mode; the kernel never drives these
    uint64_t bank_size;    // NoC-addressable bytes of the bank (dram_view_size)
    std::array<uint16_t, abi::BANK_ENDPOINTS> bank_noc_xy_addr;  // every endpoint of the bank, TRANSLATED
};

// The two seats of DRAM view `bank`, as named by dram_sender_logical_cores: [0] is the bank's NOC1-endpoint DRISC,
// running on NoC0; [1] is its free DRISC, running on NoC1. Throws when `bank` is not one of the chip's DRAM views,
// whatever dram_sender_logical_cores throws, or when a seat's NoC is one its NIU must keep in NOC2AXI mode.
std::array<Seat, NUM_NOCS> bank_seats(
    tt::umd::Cluster& cluster, ChipId chip, const metal_SocDescriptor& soc, uint32_t bank);

// This process's RunTimeOptions, built once: construction reads every TT_METAL_* variable. Its root dir is found the
// way metal finds its own firmware.
const tt::llrt::RunTimeOptions& runtime_options();

// The metal SoC descriptor of `chip`, built from UMD and metal's SoC descriptor YAML without a metal device.
metal_SocDescriptor soc_descriptor(tt::umd::Cluster& cluster, ChipId chip);

}  // namespace tt::tt_metal::internal::drisc_service
