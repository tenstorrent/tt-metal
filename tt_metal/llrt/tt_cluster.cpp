// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "tt_cluster.hpp"

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <map>
#include <memory>
#include <set>
#include <stdexcept>
#include <string>
#include <tuple>                                                     // for get
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "fmt/base.h"
#include "tt_metal/common/base.hpp"
#include "tt_metal/common/logger.hpp"
#include "tt_metal/common/metal_soc_descriptor.h"
#include "tt_metal/common/test_common.hpp"
#include "tt_metal/common/tt_backend_api_types.hpp"
#include "umd/device/types/arch.h"
#include "umd/device/tt_cluster_descriptor.h"
#include "umd/device/types/cluster_descriptor_types.h"
#include "umd/device/cluster.h"
#include "umd/device/tt_soc_descriptor.h"
#include "umd/device/tt_xy_pair.h"
#include "umd/device/types/xy_pair.h"
#include "umd/device/hugepage.h"

// TODO: ARCH_NAME specific, must remove
#include "eth_l1_address_map.h"

#include "dev_msgs.h"

#include "llrt/hal.hpp"

#include "tracy/Tracy.hpp"
#include "umd/device/tt_simulation_device.h"

#include "tt_metal/impl/debug/sanitize_noc_host.hpp"
#include "tt_metal/llrt/rtoptions.hpp"
#include "tt_metal/llrt/tlb_config.hpp"
#include "tt_metal/common/core_coord.hpp"

#include "get_platform_architecture.hpp"

static constexpr uint32_t HOST_MEM_CHANNELS = 4;
static constexpr uint32_t HOST_MEM_CHANNELS_MASK = HOST_MEM_CHANNELS - 1;

namespace tt {

const Cluster &Cluster::instance() {
    static Cluster inst;
    return inst;
}

Cluster::Cluster() {
    ZoneScoped;
    log_info(tt::LogDevice, "Opening user mode device driver");

    this->detect_arch_and_target();

    this->generate_cluster_descriptor();

    this->initialize_device_drivers();

    this->reserve_ethernet_cores_for_tunneling();

    this->initialize_ethernet_sockets();

    this->set_tunnels_from_mmio_device();

    this->assert_risc_reset();
}

void Cluster::detect_arch_and_target() {

    this->target_type_ = (std::getenv("TT_METAL_SIMULATOR_EN")) ? TargetDevice::Simulator : TargetDevice::Silicon;

    this->arch_ = tt_metal::get_platform_architecture();

#ifdef ARCH_GRAYSKULL
    TT_FATAL(
        this->arch_ == tt::ARCH::GRAYSKULL,
        "Arch={} doesn't match compile-time build for GRAYSKULL",
        get_string(this->arch_));
#endif
#ifdef ARCH_WORMHOLE
    TT_FATAL(
        (this->arch_ == tt::ARCH::WORMHOLE_B0) || (this->arch_ == tt::ARCH::WORMHOLE),
        "Arch={} doesn't match compile-time build for WORMHOLE",
        get_string(this->arch_));
#endif
#ifdef ARCH_BLACKHOLE
    TT_FATAL(
        this->arch_ == tt::ARCH::BLACKHOLE,
        "Arch={} doesn't match compile-time build for BLACKHOLE",
        get_string(this->arch_));
#endif

    TT_FATAL(
        this->target_type_ == TargetDevice::Silicon or this->target_type_ == TargetDevice::Simulator,
        "Target type={} is not supported",
        this->target_type_);
}

bool Cluster::is_galaxy_cluster() const {
    return this->is_tg_cluster_;
}

BoardType Cluster::get_board_type(chip_id_t chip_id) const {
  return this->cluster_desc_->get_board_type(chip_id);
}

void Cluster::generate_cluster_descriptor() {
    // Cluster descriptor yaml not available for Blackhole bring up
    if (this->target_type_ == TargetDevice::Simulator) {
        // Passing simulator reported physical devices as logical devices.
        this->cluster_desc_ = tt_ClusterDescriptor::create_mock_cluster(tt_SimulationDevice::detect_available_device_ids(), this->arch_);
    } else {
        this->cluster_desc_ = tt_ClusterDescriptor::create_from_yaml(tt_ClusterDescriptor::get_cluster_descriptor_file_path());
        for (const auto &chip_id : this->cluster_desc_->get_all_chips()) {
            if (this->cluster_desc_->get_board_type(chip_id) == BoardType::GALAXY) {
                this->is_tg_cluster_ = true;
                break;
            }
        }
    }

    // Use cluster descriptor to map MMIO device id to all devices on the same card (including the MMIO device)
    if (this->target_type_ == TargetDevice::Simulator) {
        std::set<chip_id_t> dummy_card = {0};
        this->devices_grouped_by_assoc_mmio_device_[0] = dummy_card;
        this->device_to_mmio_device_[0] = 0;
    } else {
        for (chip_id_t device_id : this->cluster_desc_->get_all_chips()) {
            chip_id_t closest_mmio_device_id = this->cluster_desc_->get_closest_mmio_capable_chip(device_id);
            std::set<chip_id_t> &device_ids = this->devices_grouped_by_assoc_mmio_device_[closest_mmio_device_id];
            device_ids.insert(device_id);
            this->device_to_mmio_device_[device_id] = closest_mmio_device_id;
        }
    }

    uint32_t total_num_hugepages = tt::umd::get_num_hugepages();
    if (this->is_tg_cluster_) {
        // TODO: don't think this check is correct, we want to have total num hugepages == num chips even for Galaxy
        TT_FATAL(
            this->arch_ == tt::ARCH::BLACKHOLE or total_num_hugepages >= this->cluster_desc_->get_all_chips().size()/4,
            "Machine setup error: Insufficient number of hugepages available, expected >= {} for {} devices but have {}. "
            "Increase number of hugepages!",
            this->cluster_desc_->get_all_chips().size()/4,
            this->cluster_desc_->get_all_chips().size(),
            total_num_hugepages);
    } else if (this->target_type_ != TargetDevice::Simulator){
    // TODO (abhullar): ignore hugepage set up for BH bringup
        TT_FATAL(
            this->arch_ == tt::ARCH::BLACKHOLE or total_num_hugepages >= this->cluster_desc_->get_all_chips().size(),
            "Machine setup error: Insufficient number of hugepages available, expected one per device ({}) but have {}. "
            "Increase number of hugepages!",
            this->cluster_desc_->get_all_chips().size(),
            total_num_hugepages);
    }
}

void Cluster::initialize_device_drivers() {
    for (const auto &[mmio_device_id, controlled_devices] : this->devices_grouped_by_assoc_mmio_device_) {
        this->assign_mem_channels_to_devices(mmio_device_id, controlled_devices);
    }

    this->open_driver();

    tt_device_params default_params;
    this->start_driver(default_params);
    this->generate_virtual_to_umd_coord_mapping();
    this->generate_logical_to_virtual_coord_mapping();
    this->generate_virtual_to_profiler_flat_id_mapping();
}

void Cluster::assert_risc_reset() {
    this->driver_->assert_risc_reset();
}

void Cluster::assign_mem_channels_to_devices(
    chip_id_t mmio_device_id, const std::set<chip_id_t> &controlled_device_ids) {
    // g_MAX_HOST_MEM_CHANNELS (4) is defined in tt::umd::Cluster and denotes the max number of host memory channels per
    // MMIO device Metal currently assigns 1 channel per device. See https://github.com/tenstorrent/tt-metal/issues/4087
    // One WH gateway should have 8 remote deivces in its control group.
    TT_ASSERT(controlled_device_ids.size() <= 9, "Unable to assign each device to its own host memory channel!");
    uint16_t channel = 0;
    this->device_to_host_mem_channel_[mmio_device_id] = channel++;
    for (const chip_id_t &device_id : controlled_device_ids) {
        if (device_id == mmio_device_id) {
            continue;
        }
        this->device_to_host_mem_channel_[device_id] = channel++;
        if ((channel + 1) % 4 == 0) channel++;
    }
}

void Cluster::get_metal_desc_from_tt_desc(
    const std::unordered_map<chip_id_t, tt_SocDescriptor> &input,
    const std::unordered_map<chip_id_t, uint32_t> &per_chip_id_harvesting_masks) {
    for (const auto& it : input) {
        chip_id_t id = it.first;
        this->sdesc_per_chip_.emplace(id, metal_SocDescriptor(it.second, per_chip_id_harvesting_masks.at(id), this->cluster_desc_->get_board_type(id)));
    }
}

const std::unordered_map<CoreCoord, int32_t>& Cluster::get_virtual_routing_to_profiler_flat_id(chip_id_t chip_id) const {
    return this->virtual_routing_to_profiler_flat_id_.at(this->get_board_type(chip_id));
}

void Cluster::open_driver(const bool &skip_driver_allocs) {
    const std::string sdesc_path = get_soc_description_file(this->arch_, this->target_type_);

    std::unique_ptr<tt_device> device_driver;
    if (this->target_type_ == TargetDevice::Silicon) {
        std::unordered_set<chip_id_t> all_chips = this->cluster_desc_->get_all_chips();
        std::set<chip_id_t> all_chips_set(all_chips.begin(), all_chips.end());
        // This is the target/desired number of mem channels per arch/device.
        // Silicon driver will attempt to open this many hugepages as channels per mmio chip,
        // and assert if workload uses more than available.
        uint32_t num_host_mem_ch_per_mmio_device = std::min(HOST_MEM_CHANNELS, (uint32_t)all_chips_set.size());
        // This will remove harvested rows from the soc descriptor
        const bool perform_harvesting = true;
        const bool clean_system_resources = true;
        device_driver = std::make_unique<tt::umd::Cluster>(
            sdesc_path,
            all_chips_set,
            num_host_mem_ch_per_mmio_device,
            skip_driver_allocs,
            clean_system_resources,
            perform_harvesting);
        if (this->arch_ == tt::ARCH::WORMHOLE_B0 and not this->is_galaxy_cluster()) {
            // Give UMD Limited access to eth cores 8 and 9 for Non-Galaxy Wormhole Clusters
            for (const auto &[mmio_device_id, _]: this->cluster_desc_->get_chips_with_mmio()) {
                device_driver->configure_active_ethernet_cores_for_mmio_device(mmio_device_id, {});
            }
        }

        // Adding this check is a workaround for current UMD bug that only uses this getter to populate private metadata
        // that is later expected to be populated by unrelated APIs
        // TT_FATAL(device_driver->get_target_mmio_device_ids().size() == 1, "Only one target mmio device id allowed.");
    } else if (this->target_type_ == TargetDevice::Simulator) {
        device_driver = std::make_unique<tt_SimulationDevice>(sdesc_path);
    }

    barrier_address_params barrier_params;
    barrier_params.tensix_l1_barrier_base = tt_metal::hal.get_dev_addr(tt_metal::HalProgrammableCoreType::TENSIX, tt_metal::HalL1MemAddrType::BARRIER);
    barrier_params.dram_barrier_base = tt_metal::hal.get_dev_addr(tt_metal::HalDramMemAddrType::DRAM_BARRIER);

    if (tt_metal::hal.get_arch() != tt::ARCH::GRAYSKULL) {
        barrier_params.eth_l1_barrier_base = tt_metal::hal.get_dev_addr(tt_metal::HalProgrammableCoreType::ACTIVE_ETH, tt_metal::HalL1MemAddrType::BARRIER);
    }
    device_driver->set_barrier_address_params(barrier_params);

    this->get_metal_desc_from_tt_desc(
        device_driver->get_virtual_soc_descriptors(), device_driver->get_harvesting_masks_for_soc_descriptors());
    this->driver_ = std::move(device_driver);
}

void Cluster::start_driver(tt_device_params &device_params) const {
    device_params.init_device = true;

    TT_FATAL(this->sdesc_per_chip_.size(), "Descriptor must be loaded. Try open_driver()");

    if (this->target_type_ == TargetDevice::Silicon && device_params.init_device) {
        for (const auto [mmio_device_id, _]: this->cluster_desc_->get_chips_with_mmio()) {
            ll_api::configure_static_tlbs(
                this->arch_, mmio_device_id, this->get_soc_desc(mmio_device_id), *this->driver_);
        }
    }

    this->driver_->start_device(device_params);
}

Cluster::~Cluster() {
    log_info(tt::LogDevice, "Closing user mode device drivers");
    this->driver_->close_device();

    this->sdesc_per_chip_.clear();
    this->devices_grouped_by_assoc_mmio_device_.clear();
    this->device_to_mmio_device_.clear();
    this->device_to_host_mem_channel_.clear();
    this->device_eth_routing_info_.clear();
    this->tunnels_from_mmio_device.clear();
}

std::unordered_map<chip_id_t, eth_coord_t> Cluster::get_user_chip_ethernet_coordinates() const {
    auto user_chip_ethernet_coordinates = this->cluster_desc_->get_chip_locations();
    if (this->is_galaxy_cluster()) {
        std::erase_if(user_chip_ethernet_coordinates, [this](const auto& entry) {
            return this->cluster_desc_->get_board_type(entry.first) != BoardType::GALAXY;
        });
    }
    return user_chip_ethernet_coordinates;
}

const metal_SocDescriptor &Cluster::get_soc_desc(chip_id_t chip) const {
    if (this->sdesc_per_chip_.find(chip) == this->sdesc_per_chip_.end()) {
        TT_THROW(
            "Cannot access soc descriptor for {} before device driver is initialized! Call "
            "initialize_device_driver({}) first",
            chip,
            chip);
    }
    return this->sdesc_per_chip_.at(chip);
}

void Cluster::generate_virtual_to_umd_coord_mapping() {
    // UMD APIs currently use a coordinate system that is not Physical, Virtual or Logical.
    // TT-Metal uses Virtual Coordinates when programming txns on device.
    // This mapping allows Cluster APIs to be consistent with the rest of TT-Metal, while correctly
    // using UMD under the hood.
    // This will be kept around until UMD supports generic coordinates in its APIs, at which point TT-Metal
    // virtual coordinates can be passed to UMD directly.
    for (auto chip_id : this->cluster_desc_->get_all_chips()) {
        this->virtual_worker_cores_[chip_id] = {};
        this->virtual_eth_cores_[chip_id] = {};
        for (auto& core_desc : this->get_soc_desc(chip_id).physical_cores) {
            if (core_desc.second.type != CoreType::HARVESTED) {
                CoreCoord virtual_coords = this->get_virtual_coordinate_from_physical_coordinates(chip_id, core_desc.first, core_desc.second.type);
                tt_cxy_pair virtual_core = tt_cxy_pair(chip_id, virtual_coords.x, virtual_coords.y);
                tt_cxy_pair umd_core = this->get_soc_desc(chip_id).convert_to_umd_coordinates(tt_cxy_pair(chip_id, core_desc.first.x, core_desc.first.y));
                this->virtual_to_umd_coord_mapping_[virtual_core] = umd_core;
                if (core_desc.second.type == CoreType::WORKER) {
                    this->virtual_worker_cores_[chip_id].insert(virtual_coords);
                } else if (core_desc.second.type == CoreType::ETH) {
                    this->virtual_eth_cores_[chip_id].insert(virtual_coords);
                }
            }
        }
    }
}

void Cluster::generate_logical_to_virtual_coord_mapping() {
    for (auto chip_id : this->cluster_desc_->get_all_chips()) {
        auto board_type = this->get_board_type(chip_id);
        if (this->worker_logical_to_virtual_x_.find(board_type) != this->worker_logical_to_virtual_x_.end()) {
            continue;
        }
        auto& soc_desc = this->get_soc_desc(chip_id);
        this->worker_logical_to_virtual_x_.insert({board_type, {}});
        this->worker_logical_to_virtual_y_.insert({board_type, {}});
        this->eth_logical_to_virtual_.insert({board_type, {}});
        for (auto x_coords : soc_desc.worker_log_to_routing_x) {
            CoreCoord phys_core = soc_desc.get_physical_core_from_logical_core(CoreCoord(x_coords.first, 0), CoreType::WORKER);
            CoreCoord virtual_coords = this->get_virtual_coordinate_from_physical_coordinates(chip_id, phys_core, CoreType::WORKER);
            this->worker_logical_to_virtual_x_.at(board_type).insert({x_coords.first, virtual_coords.x});
        }
        for (auto y_coords : soc_desc.worker_log_to_routing_y) {
            CoreCoord phys_core = soc_desc.get_physical_core_from_logical_core(CoreCoord(0, y_coords.first), CoreType::WORKER);
            CoreCoord virtual_coords = this->get_virtual_coordinate_from_physical_coordinates(chip_id, phys_core, CoreType::WORKER);
            this->worker_logical_to_virtual_y_.at(board_type).insert({y_coords.first, virtual_coords.y});
        }
        for (std::size_t log_eth_core_y = 0; log_eth_core_y < soc_desc.physical_ethernet_cores.size(); log_eth_core_y++) {
            CoreCoord logical_eth_core = {0, log_eth_core_y};
            CoreCoord virtual_coords = this->get_virtual_coordinate_from_physical_coordinates(chip_id, soc_desc.physical_ethernet_cores.at(log_eth_core_y), CoreType::ETH);
            this->eth_logical_to_virtual_.at(board_type).insert({logical_eth_core, virtual_coords});
        }
    }

}

void Cluster::generate_virtual_to_profiler_flat_id_mapping() {
#if defined(TRACY_ENABLE)
    for (auto chip_id : this->cluster_desc_->get_all_chips()) {
        auto board_type = this->get_board_type(chip_id);
        if (this->virtual_routing_to_profiler_flat_id_.find(board_type) != this->virtual_routing_to_profiler_flat_id_.end()) {
            continue;
        }
        this->virtual_routing_to_profiler_flat_id_.insert({board_type, {}});
        auto& soc_desc = this->get_soc_desc(chip_id);
        for (const auto& core_to_profiler_id : soc_desc.physical_routing_to_profiler_flat_id) {
            if (std::find(soc_desc.physical_workers.begin(), soc_desc.physical_workers.end(), core_to_profiler_id.first) != soc_desc.physical_workers.end()) {
                this->virtual_routing_to_profiler_flat_id_.at(board_type).insert({this->get_virtual_coordinate_from_physical_coordinates(chip_id, core_to_profiler_id.first, CoreType::WORKER), core_to_profiler_id.second});
            } else {
                this->virtual_routing_to_profiler_flat_id_.at(board_type).insert({this->get_virtual_coordinate_from_physical_coordinates(chip_id, core_to_profiler_id.first, CoreType::ETH), core_to_profiler_id.second});
            }
        }
    }
#endif
}

bool Cluster::is_worker_core(const CoreCoord &core, chip_id_t chip_id) const {
    return this->virtual_worker_cores_.at(chip_id).find(core) != this->virtual_worker_cores_.at(chip_id).end();
}

bool Cluster::is_ethernet_core(const CoreCoord &core, chip_id_t chip_id) const {

    return this->virtual_eth_cores_.find(chip_id) != this->virtual_eth_cores_.end() and
           this->virtual_eth_cores_.at(chip_id).find(core) != this->virtual_eth_cores_.at(chip_id).end();
}

const std::unordered_set<CoreCoord>& Cluster::get_virtual_worker_cores(chip_id_t chip_id) const {
    return this->virtual_worker_cores_.at(chip_id);
}

const std::unordered_set<CoreCoord>& Cluster::get_virtual_eth_cores(chip_id_t chip_id) const {
    return this->virtual_eth_cores_.at(chip_id);
}

CoreCoord Cluster::get_virtual_coordinate_from_logical_coordinates(chip_id_t chip_id, CoreCoord logical_coord, const CoreType& core_type) const {
    auto board_type = this->get_board_type(chip_id);
    if (core_type == CoreType::WORKER) {
        return CoreCoord(this->worker_logical_to_virtual_x_.at(board_type).at(logical_coord.x), this->worker_logical_to_virtual_y_.at(board_type).at(logical_coord.y));
    } else if (core_type == CoreType::ETH) {
        return this->eth_logical_to_virtual_.at(board_type).at(logical_coord);
    }
    auto& soc_desc = this->get_soc_desc(chip_id);
    return soc_desc.get_physical_core_from_logical_core(logical_coord, core_type);
}

tt_cxy_pair Cluster::get_virtual_coordinate_from_logical_coordinates(tt_cxy_pair logical_coordinate, const CoreType& core_type) const {
    auto xy_virtual_coord = this->get_virtual_coordinate_from_logical_coordinates(logical_coordinate.chip, CoreCoord(logical_coordinate.x, logical_coordinate.y), core_type);
    return tt_cxy_pair(logical_coordinate.chip, xy_virtual_coord);
}
CoreCoord Cluster::get_virtual_coordinate_from_physical_coordinates(chip_id_t chip_id, CoreCoord physical_coord, const CoreType& core_type) const {
    auto& soc_desc = this->get_soc_desc(chip_id);
    if ((not (core_type == CoreType::WORKER or core_type == CoreType::ETH)) or this->target_type_ == TargetDevice::Simulator) {
        return physical_coord;
    }
    tt_cxy_pair virtual_chip_coord = soc_desc.convert_to_umd_coordinates(tt_cxy_pair(chip_id, physical_coord.x, physical_coord.y));
    std::size_t c = virtual_chip_coord.x;
    std::size_t r = virtual_chip_coord.y;
    this->driver_->translate_to_noc_table_coords(chip_id, r, c);
    return CoreCoord{c, r};
}

CoreCoord Cluster::get_logical_ethernet_core_from_virtual(chip_id_t chip, CoreCoord core) const {
    const metal_SocDescriptor &soc_desc = tt::Cluster::instance().get_soc_desc(chip);
    auto phys_eth_core = this->virtual_to_umd_coord_mapping_.at(tt_cxy_pair(chip, core.x, core.y));
    return soc_desc.get_logical_ethernet_core_from_physical(phys_eth_core);
}

uint32_t Cluster::get_harvested_rows(chip_id_t chip) const {
    if (this->target_type_ == TargetDevice::Simulator) {
        return 0;
    } else {
        return this->driver_->harvested_rows_per_target.at(chip);
    }
}

int Cluster::get_device_aiclk(const chip_id_t &chip_id) const {
    if (this->arch_ == tt::ARCH::BLACKHOLE) {
        // For Blackhole bring up remove AICLK query due to lack of ARC message support
        log_info(tt::LogDevice, "For Blackhole hardcode AICLK to 800 MHz due to lack of ARC message support");
        return 800;
    }
    if (this->device_to_mmio_device_.find(chip_id) != this->device_to_mmio_device_.end()) {
        // get_clocks returns MMIO device ID -> clock frequency
        // There is one driver per MMIO device, so we use that to index returned map
        chip_id_t mmio_device_id = this->device_to_mmio_device_.at(chip_id);
        return this->driver_->get_clocks().at(mmio_device_id);
    }
    TT_THROW("Cannot get frequency for device {} that is not initialized!", chip_id);
    return 0;
}

void Cluster::deassert_risc_reset_at_core(const tt_cxy_pair &core) const {
    const metal_SocDescriptor &soc_desc = this->get_soc_desc(core.chip);
    tt_cxy_pair umd_core = this->virtual_to_umd_coord_mapping_.at(core);
    this->driver_->deassert_risc_reset_at_core(umd_core);
}

void Cluster::assert_risc_reset_at_core(const tt_cxy_pair &core) const {
    const metal_SocDescriptor &soc_desc = this->get_soc_desc(core.chip);
    tt_cxy_pair umd_core = this->virtual_to_umd_coord_mapping_.at(core);
    this->driver_->assert_risc_reset_at_core(umd_core);
}

void Cluster::write_dram_vec(std::vector<uint32_t> &vec, tt_target_dram dram, uint64_t addr, bool small_access) const {
    int chip_id, d_chan, d_subchannel;
    std::tie(chip_id, d_chan, d_subchannel) = dram;
    const metal_SocDescriptor &desc_to_use = get_soc_desc(chip_id);
    TT_FATAL(
        d_chan < desc_to_use.dram_cores.size(),
        "Bounds-Error -- dram_channel={} is outside of num_dram_channels={}",
        d_chan,
        desc_to_use.dram_cores.size());
    TT_ASSERT(
        d_subchannel < desc_to_use.dram_cores.at(d_chan).size(),
        "Trying to address dram sub channel that doesnt exist in the device descriptor");
    tt_cxy_pair dram_core = tt_cxy_pair(chip_id, desc_to_use.get_core_for_dram_channel(d_chan, d_subchannel));
    size_t offset = desc_to_use.get_address_offset(d_chan);
    write_core(vec.data(), vec.size() * sizeof(uint32_t), dram_core, addr + offset, small_access);
}

void Cluster::read_dram_vec(
    std::vector<uint32_t> &vec, uint32_t sz_in_bytes, tt_target_dram dram, uint64_t addr, bool small_access) const {
    int chip_id, d_chan, d_subchannel;
    std::tie(chip_id, d_chan, d_subchannel) = dram;
    const metal_SocDescriptor &desc_to_use = get_soc_desc(chip_id);
    TT_FATAL(
        d_chan < desc_to_use.dram_cores.size(),
        "Bounds-Error -- dram_channel={} is outside of num_dram_channels={}",
        d_chan,
        desc_to_use.dram_cores.size());
    TT_ASSERT(
        d_subchannel < desc_to_use.dram_cores.at(d_chan).size(),
        "Trying to address dram sub channel that doesnt exist in the device descriptor");
    tt_cxy_pair dram_core = tt_cxy_pair(chip_id, desc_to_use.get_core_for_dram_channel(d_chan, d_subchannel));
    size_t offset = desc_to_use.get_address_offset(d_chan);
    read_core(vec, sz_in_bytes, dram_core, addr + offset, small_access);
}

void Cluster::write_core(
    const void *mem_ptr, uint32_t sz_in_bytes, tt_cxy_pair core, uint64_t addr, bool small_access) const {
    chip_id_t chip_id = core.chip;
    const metal_SocDescriptor &soc_desc = this->get_soc_desc(chip_id);
    if (tt::llrt::RunTimeOptions::get_instance().get_watcher_enabled()) {
        tt::watcher_sanitize_host_noc_write(soc_desc, this->virtual_worker_cores_.at(chip_id), this->virtual_eth_cores_.at(chip_id), {core.x, core.y}, addr, sz_in_bytes);

    }

    tt_cxy_pair umd_core = this->virtual_to_umd_coord_mapping_.at(core);
    this->driver_->write_to_device(mem_ptr, sz_in_bytes, umd_core, addr, "LARGE_WRITE_TLB");
    if (this->cluster_desc_->is_chip_remote(chip_id)) {
        this->driver_->wait_for_non_mmio_flush(chip_id);
    }
}

void Cluster::read_core(
    void *mem_ptr, uint32_t size_in_bytes, tt_cxy_pair core, uint64_t addr, bool small_access) const {
    int chip_id = core.chip;
    const metal_SocDescriptor &soc_desc = this->get_soc_desc(chip_id);

    if (tt::llrt::RunTimeOptions::get_instance().get_watcher_enabled()) {
        tt::watcher_sanitize_host_noc_read(soc_desc, this->virtual_worker_cores_.at(chip_id), this->virtual_eth_cores_.at(chip_id), {core.x, core.y}, addr, size_in_bytes);
    }

    tt_cxy_pair umd_core = this->virtual_to_umd_coord_mapping_.at(core);
    this->driver_->read_from_device(mem_ptr, umd_core, addr, size_in_bytes, "LARGE_READ_TLB");
}

void Cluster::read_core(
    std::vector<uint32_t> &data, uint32_t size_in_bytes, tt_cxy_pair core, uint64_t addr, bool small_access) const {
    data.resize(size_in_bytes / sizeof(uint32_t));
    read_core(data.data(), size_in_bytes, core, addr, small_access);
}

void Cluster::write_reg(const std::uint32_t *mem_ptr, tt_cxy_pair target, uint64_t addr) const {
    const unsigned int size_in_bytes = sizeof(uint32_t);
    int chip_id = target.chip;
    const metal_SocDescriptor &soc_desc = this->get_soc_desc(chip_id);

    if (tt::llrt::RunTimeOptions::get_instance().get_watcher_enabled()) {
        tt::watcher_sanitize_host_noc_write(soc_desc, this->virtual_worker_cores_.at(chip_id), this->virtual_eth_cores_.at(chip_id), {target.x, target.y}, addr, size_in_bytes);
    }
    tt_cxy_pair umd_target = this->virtual_to_umd_coord_mapping_.at(target);
    this->driver_->write_to_device(mem_ptr, size_in_bytes, umd_target, addr, "REG_TLB");
    if (this->cluster_desc_->is_chip_remote(chip_id)) {
        this->driver_->wait_for_non_mmio_flush(chip_id);
    }
}

void Cluster::read_reg(std::uint32_t *mem_ptr, tt_cxy_pair target, uint64_t addr) const {
    const unsigned int size_in_bytes = sizeof(uint32_t);
    int chip_id = target.chip;
    const metal_SocDescriptor &soc_desc = this->get_soc_desc(chip_id);

    if (tt::llrt::RunTimeOptions::get_instance().get_watcher_enabled()) {
        tt::watcher_sanitize_host_noc_read(soc_desc, this->virtual_worker_cores_.at(chip_id), this->virtual_eth_cores_.at(chip_id), {target.x, target.y}, addr, size_in_bytes);
    }
    tt_cxy_pair umd_target = this->virtual_to_umd_coord_mapping_.at(target);
    this->driver_->read_from_device(mem_ptr, umd_target, addr, size_in_bytes, "REG_TLB");
}

void Cluster::write_sysmem(
    const void *vec, uint32_t size_in_bytes, uint64_t addr, chip_id_t src_device_id, uint16_t channel) const {
    TT_ASSERT(this->cluster_desc_->is_chip_mmio_capable(src_device_id));
    this->driver_->write_to_sysmem(vec, size_in_bytes, addr, channel & HOST_MEM_CHANNELS_MASK, src_device_id);
}

void Cluster::read_sysmem(
    void *vec, uint32_t size_in_bytes, uint64_t addr, chip_id_t src_device_id, uint16_t channel) const {
    TT_ASSERT(this->cluster_desc_->is_chip_mmio_capable(src_device_id));
    this->driver_->read_from_sysmem(vec, addr, channel & HOST_MEM_CHANNELS_MASK, size_in_bytes, src_device_id);
}

void Cluster::verify_sw_fw_versions(
    int device_id, std::uint32_t sw_version, std::vector<std::uint32_t> &fw_versions) const {
    tt_version sw(sw_version), fw_first_eth_core(fw_versions.at(0));
    tt::log_info(
        tt::LogDevice,
        "Software version {}, Ethernet FW version {} (Device {})",
        sw.str(),
        fw_first_eth_core.str(),
        device_id);
    for (std::uint32_t &fw_version : fw_versions) {
        tt_version fw(fw_version);

        TT_FATAL(fw == fw_first_eth_core, "FW versions are not the same across different ethernet cores");
        TT_FATAL(sw.major == fw.major, "SW/FW major version number out of sync");
        TT_FATAL(sw.minor <= fw.minor, "SW version is newer than FW version");
    }
}

// DRAM barrier is used to implement host-to-device synchronization and should be used when all previous writes to DRAM
// need to be flushed This is needed because writes to device are not blocking unless strict TLB ordering is used
// (default ordering is posted) This barrier is intended to prevent races caused by out of order writes, specifically to
// ensure metadata and data to compute on are committed before launching kernels
void Cluster::dram_barrier(chip_id_t chip_id) const {
    std::unordered_set<uint32_t> dram_channels;
    for (uint32_t channel = 0; channel < this->get_soc_desc(chip_id).get_num_dram_channels(); channel++) {
        dram_channels.insert(channel);
    }
    this->driver_->dram_membar(chip_id, "LARGE_WRITE_TLB", dram_channels);
}

// L1 barrier is used to implement host-to-device synchronization and should be used when all previous writes to L1 need
// to be flushed This is needed because writes to device are not blocking unless strict TLB ordering is used (default
// ordering is posted) This barrier is intended to prevent races caused by out of order writes, specifically to ensure
// binaries, metadata, and data to compute on are committed before launching kernels
void Cluster::l1_barrier(chip_id_t chip_id) const {
    // Sets and resets L1 barrier of all tensix cores and ethernet cores
    this->driver_->l1_membar(chip_id, "LARGE_WRITE_TLB");
}

uint32_t Cluster::get_num_host_channels(chip_id_t device_id) const {
    bool mmio_capable = this->cluster_desc_->is_chip_mmio_capable(device_id);
    return mmio_capable ? this->driver_->get_num_host_channels(device_id) : 0;
}

uint32_t Cluster::get_host_channel_size(chip_id_t device_id, uint32_t channel) const {
    TT_ASSERT(this->cluster_desc_->is_chip_mmio_capable(device_id));
    return this->driver_->get_host_channel_size(device_id, channel & HOST_MEM_CHANNELS_MASK);
}

void *Cluster::host_dma_address(uint64_t offset, chip_id_t src_device_id, uint16_t channel) const {
    TT_ASSERT(this->cluster_desc_->is_chip_mmio_capable(src_device_id));
    return this->driver_->host_dma_address(offset, src_device_id, channel & HOST_MEM_CHANNELS_MASK);
}

uint64_t Cluster::get_pcie_base_addr_from_device(chip_id_t chip_id) const {
    return this->driver_->get_pcie_base_addr_from_device(chip_id);
}

std::unordered_map<chip_id_t, std::vector<CoreCoord>> Cluster::get_ethernet_cores_grouped_by_connected_chips(
    chip_id_t chip_id) const {
    const auto &soc_desc = get_soc_desc(chip_id);
    std::unordered_map<chip_id_t, std::vector<CoreCoord>> connected_chips;
    const auto &all_eth_connections = this->cluster_desc_->get_ethernet_connections();
    if (all_eth_connections.find(chip_id) == all_eth_connections.end()) {
        return {};
    }
    for (const auto &[eth_chan, connected_chip_chan] : all_eth_connections.at(chip_id)) {
        const auto &other_chip_id = std::get<0>(connected_chip_chan);
        if (connected_chips.find(other_chip_id) == connected_chips.end()) {
            std::vector<CoreCoord> active_ethernet_cores;

            for (const auto &channel_pair :
                 this->cluster_desc_->get_directly_connected_ethernet_channels_between_chips(chip_id, other_chip_id)) {
                ethernet_channel_t local_chip_chan = std::get<0>(channel_pair);
                active_ethernet_cores.emplace_back(
                    get_soc_desc(chip_id).chan_to_logical_eth_core_map.at(local_chip_chan));
            }
            connected_chips.insert({other_chip_id, active_ethernet_cores});
        } else {
            continue;
        }
    }
    return connected_chips;
}
#define MAX_TUNNEL_DEPTH 4
void Cluster::set_tunnels_from_mmio_device() {
    for (const auto &[mmio_chip_id, physical_chip_id] : this->cluster_desc_->get_chips_with_mmio()) {
        std::vector<std::vector<chip_id_t>> tunnels_from_mmio = {};
        const auto &all_eth_connections = this->cluster_desc_->get_ethernet_connections();
        TT_ASSERT(this->cluster_desc_->is_chip_mmio_capable(mmio_chip_id));

        if (all_eth_connections.find(mmio_chip_id) == all_eth_connections.end()) {
            this->tunnels_from_mmio_device.insert({mmio_chip_id, {}});
            continue;
        }

        std::set<chip_id_t> device_ids = get_devices_controlled_by_mmio_device(mmio_chip_id);
        device_ids.erase(mmio_chip_id);

        if (device_ids.size() == 0) {
            this->tunnels_from_mmio_device.insert({mmio_chip_id, {}});
            continue;
        }

        for (const auto &[eth_chan, connected_chip_chan] : all_eth_connections.at(mmio_chip_id)) {
            const auto &other_chip_id = std::get<0>(connected_chip_chan);
            if (device_ids.find(other_chip_id) != device_ids.end()) {
                // mmio chip is connected to a remote chip in its mmio group.
                // erase from the pool so multiple ethenret connections to same remote device do not
                // pollute the counts.
                device_ids.erase(other_chip_id);
                std::vector<chip_id_t> first_stop = {other_chip_id};
                auto it = std::find(tunnels_from_mmio.begin(), tunnels_from_mmio.end(), first_stop);
                TT_ASSERT(
                    it == tunnels_from_mmio.end(),
                    "Duplicate first tunnel stop found when finding FD2 Tunnel devices.");
                tunnels_from_mmio.push_back(first_stop);
            }
        }

        log_debug(
            tt::LogMetal,
            "Found {} FD Tunnels originating from MMIO Device {}",
            tunnels_from_mmio.size(),
            mmio_chip_id);

        device_ids = get_devices_controlled_by_mmio_device(mmio_chip_id);
        device_ids.erase(mmio_chip_id);

        for (auto &tunnel : tunnels_from_mmio) {
            TT_ASSERT(tunnel.size() == 1, "Tunnel depth must be 1 when it has only 1 stop in it.");
            device_ids.erase(tunnel[0]);
        }

        bool tunneled_device_hit;
        for (auto it = device_ids.begin(); it != device_ids.end();) {
            tunneled_device_hit = false;
            for (auto &dev_vec : tunnels_from_mmio) {
                for (const auto &[eth_chan, connected_chip_chan] : all_eth_connections.at(dev_vec.back())) {
                    const auto &other_chip_id = std::get<0>(connected_chip_chan);
                    auto id_iter = device_ids.find(other_chip_id);
                    if (id_iter != device_ids.end()) {
                        it = device_ids.erase(id_iter);
                        dev_vec.push_back(other_chip_id);
                        tunneled_device_hit = true;
                        break;
                    }
                }
            }
            TT_FATAL(
                tunneled_device_hit || (it == device_ids.end()),
                "Detected ethernet connections did not match expected device connectivity, try re-running "
                "tt-topology.");
        }

        TT_ASSERT(tunnels_from_mmio.size() != 0, "Must have at least 1 tunnel from MMIO Device.");
        uint32_t tunnel_depth = tunnels_from_mmio[0].size();
        log_debug(tt::LogMetal, "Each FD Tunnel is {} deep.", tunnel_depth);

        for (auto &dev_vec : tunnels_from_mmio) {
            TT_ASSERT(
                dev_vec.size() == tunnel_depth,
                "All tunnels from mmio device must have same depth. Found {}. Expected {}.",
                dev_vec.size(),
                tunnel_depth);
            // Now that all remotete chips have been added to respective tunnels,
            // add mmio device at start of each of the tunnels.
            if (dev_vec.size() > MAX_TUNNEL_DEPTH) {
                dev_vec.resize(dev_vec.size() - (dev_vec.size() - MAX_TUNNEL_DEPTH));
            }
            dev_vec.insert(dev_vec.begin(), mmio_chip_id);
        }
        this->tunnels_from_mmio_device.insert({mmio_chip_id, tunnels_from_mmio});
    }
}

// Ethernet cluster api
void Cluster::initialize_ethernet_sockets() {
    for (const auto &chip_id : this->cluster_desc_->get_all_chips()) {
        if (this->ethernet_sockets_.find(chip_id) == this->ethernet_sockets_.end()) {
            this->ethernet_sockets_.insert({chip_id, {}});
        }
        for (const auto &[connected_chip_id, eth_cores] :
             this->get_ethernet_cores_grouped_by_connected_chips(chip_id)) {
            if (this->ethernet_sockets_.at(chip_id).find(connected_chip_id) ==
                this->ethernet_sockets_.at(chip_id).end()) {
                this->ethernet_sockets_.at(chip_id).insert({connected_chip_id, {}});
            }
            if (this->ethernet_sockets_.find(connected_chip_id) == this->ethernet_sockets_.end()) {
                this->ethernet_sockets_.insert({connected_chip_id, {}});
            }
            if (this->ethernet_sockets_.at(connected_chip_id).find(chip_id) ==
                this->ethernet_sockets_.at(connected_chip_id).end()) {
                this->ethernet_sockets_.at(connected_chip_id).insert({chip_id, {}});
            } else {
                continue;
            }
            for (const auto &eth_core : eth_cores) {
                if (this->device_eth_routing_info_.at(chip_id).at(eth_core) == EthRouterMode::IDLE) {
                    this->ethernet_sockets_.at(chip_id).at(connected_chip_id).emplace_back(eth_core);
                    this->ethernet_sockets_.at(connected_chip_id)
                        .at(chip_id)
                        .emplace_back(
                            std::get<1>(this->get_connected_ethernet_core(std::make_tuple(chip_id, eth_core))));
                }
            }
        }
    }
}

void Cluster::reserve_ethernet_cores_for_tunneling() {
    const char *TT_METAL_SLOW_DISPATCH_MODE = std::getenv("TT_METAL_SLOW_DISPATCH_MODE");
    const uint32_t routing_info_addr = eth_l1_mem::address_map::ERISC_APP_ROUTING_INFO_BASE;
    for (const auto &[assoc_mmio_device, devices] : this->devices_grouped_by_assoc_mmio_device_) {
        for (const auto &chip_id : devices) {
            if (this->device_eth_routing_info_.find(chip_id) == this->device_eth_routing_info_.end()) {
                this->device_eth_routing_info_.insert({chip_id, {}});
            }
        }
        std::map<std::tuple<chip_id_t, chip_id_t>, bool> reserved_chip_connections = {};
        for (const auto &chip_id : devices) {
            if (TT_METAL_SLOW_DISPATCH_MODE == nullptr) {
                for (const auto &[connected_chip_id, active_eth_cores] :
                     this->get_ethernet_cores_grouped_by_connected_chips(chip_id)) {
                    for (const auto &eth_core : active_eth_cores) {
                        const auto connected_eth_core =
                            std::get<1>(this->get_connected_ethernet_core(std::make_tuple(chip_id, eth_core)));
                        if (this->device_eth_routing_info_.at(chip_id).find(eth_core) ==
                            this->device_eth_routing_info_.at(chip_id).end()) {
                            if (devices.find(connected_chip_id) != devices.end() &&
                                reserved_chip_connections.find(std::make_tuple(chip_id, connected_chip_id)) ==
                                    reserved_chip_connections.end() &&
                                this->cluster_desc_->get_ethernet_link_distance(chip_id, assoc_mmio_device) !=
                                    this->cluster_desc_->get_ethernet_link_distance(
                                        connected_chip_id, assoc_mmio_device)) {
                                // only setup fd tunneling for devices grouped with same mmio device and if no bi dir
                                // tunnel found between the two chips and if link distance between both chips to mmio
                                // chip is not the same
                                log_debug(
                                    LogDevice,
                                    "Reserving {} for tunneling",
                                    tt_cxy_pair(chip_id, ethernet_core_from_logical_core(chip_id, eth_core)).str());
                                log_debug(
                                    LogDevice,
                                    "Reserving {} for tunneling",
                                    tt_cxy_pair(
                                        connected_chip_id,
                                        ethernet_core_from_logical_core(connected_chip_id, connected_eth_core))
                                        .str());
                                this->device_eth_routing_info_.at(chip_id).insert(
                                    {eth_core, EthRouterMode::BI_DIR_TUNNELING});
                                this->device_eth_routing_info_.at(connected_chip_id)
                                    .insert({connected_eth_core, EthRouterMode::BI_DIR_TUNNELING});
                                reserved_chip_connections.insert({std::make_tuple(chip_id, connected_chip_id), true});
                                reserved_chip_connections.insert({std::make_tuple(connected_chip_id, chip_id), true});
                            } else {
                                this->device_eth_routing_info_.at(chip_id).insert({eth_core, EthRouterMode::IDLE});
                            }
                        }
                    }
                }
            } else {
                // Slow dispatch mode
                for (const auto &[connected_chip_id, active_eth_cores] :
                     this->get_ethernet_cores_grouped_by_connected_chips(chip_id)) {
                    for (const auto &eth_core : active_eth_cores) {
                        this->device_eth_routing_info_.at(chip_id).insert({eth_core, EthRouterMode::IDLE});
                    }
                }
            }
        }
    }
}

std::unordered_set<chip_id_t> Cluster::get_ethernet_connected_device_ids(chip_id_t chip_id) const {
    std::unordered_set<chip_id_t> device_ids;
    const auto &connected_chips = this->get_ethernet_cores_grouped_by_connected_chips(chip_id);
    for (const auto &[other_chip_id, eth_cores] : connected_chips) {
        for (const auto &eth_core : eth_cores) {
            if (this->device_eth_routing_info_.at(chip_id).at(eth_core) == EthRouterMode::IDLE) {
                device_ids.insert(other_chip_id);
            }
        }
    }
    return device_ids;
}

std::unordered_set<CoreCoord> Cluster::get_active_ethernet_cores(
    chip_id_t chip_id, bool skip_reserved_tunnel_cores) const {
    std::unordered_set<CoreCoord> active_ethernet_cores;
    const auto &connected_chips = this->get_ethernet_cores_grouped_by_connected_chips(chip_id);
    for (const auto &[other_chip_id, eth_cores] : connected_chips) {
        for (const auto &eth_core : eth_cores) {
            if (this->device_eth_routing_info_.at(chip_id).at(eth_core) == EthRouterMode::BI_DIR_TUNNELING and
                skip_reserved_tunnel_cores) {
                continue;
            }
            active_ethernet_cores.insert(eth_core);
        }
    }
    return active_ethernet_cores;
}

std::unordered_set<CoreCoord> Cluster::get_inactive_ethernet_cores(chip_id_t chip_id) const {
    std::unordered_set<CoreCoord> active_ethernet_cores = this->get_active_ethernet_cores(chip_id);
    std::unordered_set<CoreCoord> inactive_ethernet_cores;
    std::unordered_set<int> channels_to_skip = {};
    // UMD routing FW uses these cores for base routing
    // channel 15 is used by syseng tools.
    // TODO (abhullar): For BH single-chip bringup we assume all ethernet cores are inactive. Update this with (#9823)
    if (this->is_galaxy_cluster()) {
        // TODO: This may need to change, if we need additional eth cores for dispatch on Galaxy
        channels_to_skip = {0, 1, 2, 3, 15};
    }
    else if (this->arch_ == tt::ARCH::WORMHOLE_B0) {
        channels_to_skip = {8, 9, 15};
    }
    for (const auto &[eth_core, chan] : get_soc_desc(chip_id).logical_eth_core_to_chan_map) {
        if (this->cluster_desc_->is_chip_mmio_capable(chip_id) and (channels_to_skip.find(chan) != channels_to_skip.end())) {
            continue;
        }
        if (active_ethernet_cores.find(eth_core) == active_ethernet_cores.end()) {
            inactive_ethernet_cores.insert(eth_core);
        }
    }
    return inactive_ethernet_cores;
}

std::tuple<chip_id_t, CoreCoord> Cluster::get_connected_ethernet_core(std::tuple<chip_id_t, CoreCoord> eth_core) const {
    const auto &soc_desc = get_soc_desc(std::get<0>(eth_core));
    ethernet_channel_t eth_chan = soc_desc.logical_eth_core_to_chan_map.at(std::get<1>(eth_core));
    TT_ASSERT(
        (this->cluster_desc_->ethernet_core_has_active_ethernet_link(std::get<0>(eth_core), eth_chan)),
        "Logical eth core {} is not an active eth core on chip {}.",
        std::get<1>(eth_core).str(),
        std::get<0>(eth_core));
    auto connected_eth_core =
        this->cluster_desc_->get_chip_and_channel_of_remote_ethernet_core(std::get<0>(eth_core), eth_chan);
    return std::make_tuple(
        std::get<0>(connected_eth_core), soc_desc.chan_to_logical_eth_core_map.at(std::get<1>(connected_eth_core)));
}

std::vector<CoreCoord> Cluster::get_ethernet_sockets(chip_id_t local_chip, chip_id_t remote_chip) const {
    const auto &local_ethernet_sockets = this->ethernet_sockets_.at(local_chip);
    TT_FATAL(
        local_ethernet_sockets.find(remote_chip) != local_ethernet_sockets.end(),
        "Device {} is not connected to Device {}",
        local_chip,
        remote_chip);
    return local_ethernet_sockets.at(remote_chip);
}

CoreCoord Cluster::ethernet_core_from_logical_core(chip_id_t chip_id, const CoreCoord &logical_core) const {
    const metal_SocDescriptor &soc_desc = get_soc_desc(chip_id);
    return soc_desc.get_physical_ethernet_core_from_logical(logical_core);
}

tt_cxy_pair Cluster::get_eth_core_for_dispatch_core(
    tt_cxy_pair logical_dispatch_core, EthRouterMode mode, chip_id_t connected_chip_id) const {
    const auto &local_chip_id = logical_dispatch_core.chip;
    for (const auto &[eth_core, router_mode] : this->device_eth_routing_info_.at(local_chip_id)) {
        // Check for connected chip id since one chip can be bi directional tunneling to multiple chips
        const auto connected_tunnel_chip_id =
            std::get<0>(this->get_connected_ethernet_core(std::make_tuple(local_chip_id, eth_core)));
        if (router_mode == mode and connected_tunnel_chip_id == connected_chip_id) {
            return tt_cxy_pair(local_chip_id, eth_core);
        }
    }
    TT_ASSERT(false, "Cluster does not contain requested eth routing core");
    return {};
}

std::tuple<tt_cxy_pair, tt_cxy_pair> Cluster::get_eth_tunnel_core(
    chip_id_t upstream_chip_id, chip_id_t downstream_chip_id, EthRouterMode mode) const {
    for (const auto &[eth_core, router_mode] : this->device_eth_routing_info_.at(downstream_chip_id)) {

      // Check for connected chip id since one chip can be bi directional tunneling to multiple chips
        const auto [tunnel_chip_id, tunnel_eth_core] = this->get_connected_ethernet_core(std::make_tuple(downstream_chip_id, eth_core));
        if (router_mode == mode and tunnel_chip_id == upstream_chip_id) {
            return std::make_tuple(tt_cxy_pair(tunnel_chip_id, tunnel_eth_core), tt_cxy_pair(downstream_chip_id, eth_core));
        }
    }
    TT_ASSERT(false, "Cluster does not contain requested eth routing core");
    return {};
}

// TODO: ALLAN Can change to write one bit
void Cluster::set_internal_routing_info_for_ethernet_cores(bool enable_internal_routing, const std::vector<chip_id_t> &target_mmio_devices) const {
    log_debug(tt::LogDevice, "Set internal routing bit {}", enable_internal_routing);
    const uint32_t routing_info_addr = eth_l1_mem::address_map::ERISC_APP_ROUTING_INFO_BASE;
    // TODO: initialize devices if user does not
    // Must initialize remote chips first, then mmio chips since once mmio chips are doing fd routing
    // we do not always context switch to base FW
    std::vector<chip_id_t> non_mmio_devices;
    std::vector<chip_id_t> mmio_devices = target_mmio_devices;
    if (mmio_devices.size() == 0) {
        for (const auto &[assoc_mmio_device, devices] : this->devices_grouped_by_assoc_mmio_device_) {
            mmio_devices.emplace_back(assoc_mmio_device);
        }
    }
    for (const auto &mmio_chip_id : mmio_devices) {
        for (const auto &chip_id : this->devices_grouped_by_assoc_mmio_device_.at(mmio_chip_id)) {
            non_mmio_devices.emplace_back(chip_id);
        }
    }

    if (enable_internal_routing) {
        const routing_info_t routing_info_enabled = {
            .routing_enabled = 1,
            .src_sent_valid_cmd = 0,
            .dst_acked_valid_cmd = 0,
        };
        for (const auto &chip_id : non_mmio_devices) {
            for (const auto &[eth_core, routing_info] : this->device_eth_routing_info_.at(chip_id)) {
                tt_cxy_pair virtual_eth_core(chip_id, get_virtual_coordinate_from_logical_coordinates(chip_id, eth_core, CoreType::ETH));
                // Enable internal ethernet routing for non-mmio devices
                write_core(
                    (void *)&routing_info_enabled, sizeof(routing_info_t), virtual_eth_core, routing_info_addr, false);
            }
        }
        for (const auto &chip_id : mmio_devices) {
            for (const auto &[eth_core, routing_info] : this->device_eth_routing_info_.at(chip_id)) {
                tt_cxy_pair virtual_eth_core(chip_id, get_virtual_coordinate_from_logical_coordinates(chip_id, eth_core, CoreType::ETH));
                // Enable internal ethernet routing for mmio devices
                write_core(
                    (void *)&routing_info_enabled, sizeof(routing_info_t), virtual_eth_core, routing_info_addr, false);
            }
        }
    } else {
        const routing_info_t routing_info_disabled = {
            .routing_enabled = 0,
            .src_sent_valid_cmd = 0,
            .dst_acked_valid_cmd = 0,
        };
        for (const auto &chip_id : mmio_devices) {
            for (const auto &[eth_core, routing_info] : this->device_eth_routing_info_.at(chip_id)) {
                tt_cxy_pair virtual_eth_core(chip_id, get_virtual_coordinate_from_logical_coordinates(chip_id, eth_core, CoreType::ETH));
                // Disable internal ethernet routing for mmio devices
                write_core(
                    (void *)&routing_info_disabled, sizeof(routing_info_t), virtual_eth_core, routing_info_addr, false);
            }
        }
        for (const auto &chip_id : non_mmio_devices) {
            for (const auto &[eth_core, routing_info] : this->device_eth_routing_info_.at(chip_id)) {
                tt_cxy_pair virtual_eth_core(chip_id, get_virtual_coordinate_from_logical_coordinates(chip_id, eth_core, CoreType::ETH));
                // Disable internal ethernet routing for non-mmio devices
                write_core(
                    (void *)&routing_info_disabled, sizeof(routing_info_t), virtual_eth_core, routing_info_addr, false);
            }
        }
    }
}

uint32_t Cluster::get_mmio_device_max_tunnel_depth(chip_id_t mmio_device) const {
    // Assume that tunnel depth for multiple tunnels are the same
    TT_ASSERT(
        (this->get_associated_mmio_device(mmio_device) == mmio_device), "Called mmio device api on non-mmio device");
    uint32_t depth = 0;
    for (const auto &[assoc_mmio_device, devices] : this->devices_grouped_by_assoc_mmio_device_) {
        for (const auto &chip_id : devices) {
            if (chip_id == assoc_mmio_device) {
                continue;
            }
            depth =
                std::max(depth, uint32_t(this->cluster_desc_->get_ethernet_link_distance(chip_id, assoc_mmio_device)));
        }
    }
    return depth;
}

uint32_t Cluster::get_mmio_device_tunnel_count(chip_id_t mmio_device) const {
    TT_ASSERT(
        (this->get_associated_mmio_device(mmio_device) == mmio_device), "Called mmio device api on non-mmio device");
    const auto &chip_eth_core_modes = this->device_eth_routing_info_.at(mmio_device);
    uint32_t tunnel_count = std::count_if(chip_eth_core_modes.begin(), chip_eth_core_modes.end(), [](const auto &e) {
        return e.second == EthRouterMode::BI_DIR_TUNNELING;
    });
    return tunnel_count;
}

uint32_t Cluster::get_device_tunnel_depth(chip_id_t chip_id) const {
    chip_id_t mmio_device_id = this->get_associated_mmio_device(chip_id);
    return (mmio_device_id == chip_id) ? 0 : this->cluster_desc_->get_ethernet_link_distance(chip_id, mmio_device_id);
}

}  // namespace tt

std::ostream &operator<<(std::ostream &os, tt_target_dram const &dram) {
    os << "Target DRAM chip = " << std::get<0>(dram) << ", chan = " << std::get<1>(dram)
       << ", subchan = " << std::get<2>(dram);
    return os;
}
