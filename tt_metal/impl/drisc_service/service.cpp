// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "impl/drisc_service/service.hpp"

#include <vector>

#include <fmt/format.h>
#include <tt_stl/assert.hpp>
#include <tt-umd/cluster_descriptor.hpp>
#include <tt-umd-workload/cluster.hpp>

#include "impl/buffers/dram_sender_topology.hpp"
#include "llrt/metal_soc_descriptor.hpp"
#include "llrt/rtoptions.hpp"

namespace tt::tt_metal::internal::drisc_service {

namespace {

uint16_t noc_xy_addr(const tt::umd::CoreCoord& core) { return static_cast<uint16_t>(abi::noc_xy(core.x, core.y)); }

// Every NoC endpoint of DRAM view `bank`, TRANSLATED.
std::array<uint16_t, abi::BANK_ENDPOINTS> bank_endpoints(const metal_SocDescriptor& soc, ChipId chip, uint32_t bank) {
    TT_FATAL(
        soc.get_grid_size(tt::CoreType::DRAM).y == abi::BANK_ENDPOINTS,
        "drisc_service: chip {} has {} endpoints per DRAM bank, not {}",
        chip,
        soc.get_grid_size(tt::CoreType::DRAM).y,
        abi::BANK_ENDPOINTS);
    const int channel = static_cast<int>(soc.get_channel_for_dram_view(static_cast<int>(bank)));
    std::array<uint16_t, abi::BANK_ENDPOINTS> endpoints{};
    for (uint32_t sub = 0; sub < endpoints.size(); sub++) {
        endpoints[sub] =
            noc_xy_addr(soc.get_dram_core_for_channel(channel, static_cast<int>(sub), tt::CoordSystem::TRANSLATED));
    }
    return endpoints;
}

// `bank_fields` placed on the DRISC at `logical`, driving `noc`. Throws when DRISC firmware keeps that NoC's NIU in
// NOC2AXI mode: only a stream-mode NIU lets the DRISC start NoC transfers.
Seat place_seat(const metal_SocDescriptor& soc, Seat bank_fields, uint32_t noc, const CoreCoord& logical) {
    const CoreCoord translated = soc.get_physical_dram_core_from_logical(logical);
    const tt::umd::CoreCoord core{translated.x, translated.y, tt::CoreType::DRAM, tt::CoordSystem::TRANSLATED};
    const uint8_t mask = soc.get_dram_endpoint_noc_mask(translated);
    TT_FATAL(
        ((mask >> noc) & 1u) == 0,
        "drisc_service: chip {} bank {} NoC{} seat {} must keep that NIU in NOC2AXI mode (mask {:#x})",
        bank_fields.chip,
        bank_fields.bank,
        noc,
        core.str(),
        mask);
    const tt::umd::CoreCoord noc0 = soc.translate_coord_to(core, tt::CoordSystem::NOC0);
    Seat seat = bank_fields;
    seat.noc = noc;
    seat.translated = core;
    seat.id.noc0_x = static_cast<uint32_t>(noc0.x);
    seat.id.noc0_y = static_cast<uint32_t>(noc0.y);
    seat.noc2axi_mask = mask;
    return seat;
}

}  // namespace

std::array<Seat, NUM_NOCS> bank_seats(
    tt::umd::Cluster& cluster, ChipId chip, const metal_SocDescriptor& soc, uint32_t bank) {
    TT_FATAL(
        bank < soc.get_num_dram_views(),
        "drisc_service: bank {} is out of range; chip {} has {} DRAM banks",
        bank,
        chip,
        soc.get_num_dram_views());
    const std::vector<CoreCoord> logical = dram_sender_logical_cores(soc, bank, chip);
    const Seat bank_fields{
        .chip = chip,
        .bank = bank,
        .id = {.asic_id = cluster.get_cluster_description()->get_chip_unique_ids().at(chip)},
        .bank_size = soc.dram_view_size,
        .bank_noc_xy_addr = bank_endpoints(soc, chip, bank)};
    std::array<Seat, NUM_NOCS> seats{};
    for (uint32_t noc = 0; noc < seats.size(); noc++) {
        seats[noc] = place_seat(soc, bank_fields, noc, logical.at(1 - noc));
    }
    return seats;
}

const tt::llrt::RunTimeOptions& runtime_options() {
    static const tt::llrt::RunTimeOptions options;
    return options;
}

metal_SocDescriptor soc_descriptor(tt::umd::Cluster& cluster, ChipId chip) {
    tt::umd::SocDescriptor soc = cluster.get_soc_descriptor(chip);
    TT_FATAL(soc.arch == tt::ARCH::BLACKHOLE, "drisc_service: chip {} is not Blackhole", chip);
    soc.device_descriptor_file_path =
        get_soc_description_file(tt::ARCH::BLACKHOLE, tt::TargetDevice::Silicon, runtime_options());
    return metal_SocDescriptor(soc, tt::BoardType::UNKNOWN, read_mrisc_noc2axi_ports(cluster.get_tt_device(chip)));
}

}  // namespace tt::tt_metal::internal::drisc_service
