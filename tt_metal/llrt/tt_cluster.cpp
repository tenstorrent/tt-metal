// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "tt_cluster.hpp"

#include <core_coord.hpp>
#include <dev_msgs.h>
#include <logger.hpp>
#include <metal_soc_descriptor.h>
#include <rtoptions.hpp>
#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <initializer_list>
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

#include "control_plane.hpp"
#include "fabric_host_interface.h"
#include "fabric_types.hpp"
#include "fmt/base.h"
#include "get_platform_architecture.hpp"
#include "hal_types.hpp"
#include "llrt/hal.hpp"
#include "sanitize_noc_host.hpp"
#include "tracy/Tracy.hpp"
#include "tt_metal/llrt/tlb_config.hpp"
#include <umd/device/cluster.h>
#include <umd/device/hugepage.h>
#include <umd/device/tt_cluster_descriptor.h>
#include <umd/device/tt_simulation_device.h>
#include <umd/device/tt_xy_pair.h>
#include <umd/device/types/arch.h>
#include <umd/device/types/cluster_descriptor_types.h>
#include <umd/device/types/cluster_types.h>
#include <umd/device/types/xy_pair.h>

static constexpr uint32_t HOST_MEM_CHANNELS = 4;
static constexpr uint32_t HOST_MEM_CHANNELS_MASK = HOST_MEM_CHANNELS - 1;

namespace {

inline std::string get_soc_description_file(
    const tt::ARCH& arch, tt::TargetDevice target_device, const std::string& output_dir = "") {
    // Ability to skip this runtime opt, since trimmed SOC desc limits which DRAM channels are available.
    std::string tt_metal_home;
    if (getenv("TT_METAL_HOME")) {
        tt_metal_home = getenv("TT_METAL_HOME");
    } else {
        tt_metal_home = "./";
    }
    if (tt_metal_home.back() != '/') {
        tt_metal_home += "/";
    }
    if (target_device == tt::TargetDevice::Simulator) {
        switch (arch) {
            case tt::ARCH::Invalid: throw std::runtime_error("Invalid arch not supported");
            case tt::ARCH::GRAYSKULL: throw std::runtime_error("GRAYSKULL arch not supported");
            case tt::ARCH::WORMHOLE_B0: return tt_metal_home + "tt_metal/soc_descriptors/wormhole_b0_versim.yaml";
            case tt::ARCH::BLACKHOLE:
                return tt_metal_home + "tt_metal/soc_descriptors/blackhole_simulation_1x2_arch.yaml";
            default: throw std::runtime_error("Unsupported device arch");
        };
    } else {
        switch (arch) {
            case tt::ARCH::Invalid:
                throw std::runtime_error(
                    "Invalid arch not supported");  // will be overwritten in tt_global_state constructor
            case tt::ARCH::GRAYSKULL: return tt_metal_home + "tt_metal/soc_descriptors/grayskull_120_arch.yaml";
            case tt::ARCH::WORMHOLE_B0: return tt_metal_home + "tt_metal/soc_descriptors/wormhole_b0_80_arch.yaml";
            case tt::ARCH::BLACKHOLE: return tt_metal_home + "tt_metal/soc_descriptors/blackhole_140_arch.yaml";
            default: throw std::runtime_error("Unsupported device arch");
        };
    }
    return "";
}
}  // namespace
namespace tt {

Cluster::Cluster() {
    ZoneScoped;
    log_info(tt::LogDevice, "Opening user mode device driver");

    this->detect_arch_and_target();

    if (arch_ != ARCH::GRAYSKULL) {
        routing_info_addr_ = tt::tt_metal::hal_ref.get_dev_addr(
            tt::tt_metal::HalProgrammableCoreType::ACTIVE_ETH, tt::tt_metal::HalL1MemAddrType::APP_ROUTING_INFO);
    }

    this->initialize_device_drivers();

    this->disable_ethernet_cores_with_retrain();

    this->reserve_ethernet_cores_for_tunneling();

    this->initialize_ethernet_sockets();

    this->set_tunnels_from_mmio_device();

    this->assert_risc_reset();
}

void Cluster::detect_arch_and_target() {
    this->target_type_ = (llrt::RunTimeOptions::get_instance().get_simulator_enabled()) ? TargetDevice::Simulator
                                                                                        : TargetDevice::Silicon;

    this->arch_ = tt_metal::get_platform_architecture();

    TT_FATAL(
        this->target_type_ == TargetDevice::Silicon or this->target_type_ == TargetDevice::Simulator,
        "Target type={} is not supported",
        this->target_type_);
}

// TODO: remove this when we deprecate TG
bool Cluster::is_galaxy_cluster() const { return this->cluster_type_ == ClusterType::TG; }

ClusterType Cluster::get_cluster_type() const { return this->cluster_type_; }

tt_metal::FabricConfig Cluster::get_fabric_config() const { return this->fabric_config_; }

BoardType Cluster::get_board_type(chip_id_t chip_id) const {
  return this->cluster_desc_->get_board_type(chip_id);
}

void Cluster::generate_cluster_descriptor() {
    // Cluster descriptor yaml not available for Blackhole bring up
    if (this->target_type_ == TargetDevice::Simulator) {
        // Passing simulator reported physical devices as logical devices.
        this->mock_cluster_desc_ptr_ =
            tt_ClusterDescriptor::create_mock_cluster(tt_SimulationDevice::detect_available_device_ids(), this->arch_);
        this->cluster_desc_ = this->mock_cluster_desc_ptr_.get();
    } else {
        this->cluster_desc_ = this->driver_->get_cluster_description();
        for (const auto &chip_id : this->cluster_desc_->get_all_chips()) {
            if (this->cluster_desc_->get_board_type(chip_id) == BoardType::GALAXY) {
                this->cluster_type_ = ClusterType::TG;
                break;
            }
        }
        TT_ASSERT(this->cluster_desc_->get_all_chips().size() > 0, "No chips detected in the cluster");
        const auto board_type = this->cluster_desc_->get_board_type(*this->cluster_desc_->get_all_chips().begin());
        bool all_same_board = true;
        for (const auto& chip_id : this->cluster_desc_->get_all_chips()) {
            if (this->cluster_desc_->get_board_type(chip_id) != board_type) {
                all_same_board = false;
                break;
            }
        }

        if (all_same_board) {
            if (board_type == BoardType::N300) {
                if (this->cluster_desc_->get_all_chips().size() == 2) {
                    this->cluster_type_ = ClusterType::N300;
                } else if (this->cluster_desc_->get_all_chips().size() == 8) {
                    this->cluster_type_ = ClusterType::T3K;
                }
            } else if (board_type == BoardType::N150) {
                if (this->cluster_desc_->get_all_chips().size() == 1) {
                    this->cluster_type_ = ClusterType::N150;
                }
            } else if (board_type == BoardType::P100) {
                if (this->cluster_desc_->get_all_chips().size() == 1) {
                    this->cluster_type_ = ClusterType::P100;
                }
            } else if (board_type == BoardType::P150) {
                if (this->cluster_desc_->get_all_chips().size() == 1) {
                    this->cluster_type_ = ClusterType::P150;
                } else if (this->cluster_desc_->get_all_chips().size() == 2) {
                    this->cluster_type_ = ClusterType::P150_X2;
                } else if (this->cluster_desc_->get_all_chips().size() == 4) {
                    this->cluster_type_ = ClusterType::P150_X4;
                }
            } else if (board_type == BoardType::UBB) {
                this->cluster_type_ = ClusterType::GALAXY;
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
    if (this->cluster_type_ == ClusterType::TG) {
        // TODO: don't think this check is correct, we want to have total num hugepages == num chips even for Galaxy
        TT_FATAL(
            this->arch_ == tt::ARCH::BLACKHOLE or total_num_hugepages >= this->cluster_desc_->get_all_chips().size()/4,
            "Machine setup error: Insufficient number of hugepages available, expected >= {} for {} devices but have {}. "
            "Increase number of hugepages!",
            this->cluster_desc_->get_all_chips().size()/4,
            this->cluster_desc_->get_all_chips().size(),
            total_num_hugepages);
    } else if (this->target_type_ != TargetDevice::Simulator) {
        // TODO (abhullar): ignore hugepage set up for BH bringup
        TT_FATAL(
            this->arch_ == tt::ARCH::BLACKHOLE or total_num_hugepages >= this->cluster_desc_->get_all_chips().size(),
            "Machine setup error: Insufficient number of hugepages available, expected one per device ({}) but have {}. "
            "Increase number of hugepages!",
            this->cluster_desc_->get_all_chips().size(),
            total_num_hugepages);
    }

    if (this->arch_ == tt::ARCH::WORMHOLE_B0 and not this->is_galaxy_cluster()) {
        // Give UMD Limited access to eth cores 8 and 9 for Non-Galaxy Wormhole Clusters
        for (const auto& [mmio_device_id, _] : this->cluster_desc_->get_chips_with_mmio()) {
            driver_->configure_active_ethernet_cores_for_mmio_device(mmio_device_id, {});
        }
    }
}

void Cluster::initialize_device_drivers() {
    this->open_driver();
    this->generate_cluster_descriptor();
    this->get_metal_desc_from_tt_desc();

    for (const auto &[mmio_device_id, controlled_devices] : this->devices_grouped_by_assoc_mmio_device_) {
        this->assign_mem_channels_to_devices(mmio_device_id, controlled_devices);
    }

    tt_device_params default_params;
    this->start_driver(default_params);
    this->generate_virtual_to_umd_coord_mapping();
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

void Cluster::get_metal_desc_from_tt_desc() {
    for (const auto& id : this->driver_->get_target_device_ids()) {
        this->sdesc_per_chip_.emplace(
            id, metal_SocDescriptor(this->driver_->get_soc_descriptor(id), this->cluster_desc_->get_board_type(id)));
    }
}

const std::unordered_map<CoreCoord, int32_t>& Cluster::get_virtual_routing_to_profiler_flat_id(chip_id_t chip_id) const {
    return this->virtual_routing_to_profiler_flat_id_.at(this->get_board_type(chip_id));
}

void Cluster::open_driver(const bool &skip_driver_allocs) {
    std::unique_ptr<tt_device> device_driver;
    if (this->target_type_ == TargetDevice::Silicon) {
        const std::string sdesc_path = get_soc_description_file(this->arch_, this->target_type_);
        // umd::Cluster::detect_available_device_ids only lists MMIO device ids, since we need remote chip ids
        // generate the cluster desc and pull chip ids from there
        auto temp_cluster_desc = tt::umd::Cluster::create_cluster_descriptor();
        std::unordered_set<chip_id_t> all_chips = temp_cluster_desc->get_all_chips();
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

        // Adding this check is a workaround for current UMD bug that only uses this getter to populate private metadata
        // that is later expected to be populated by unrelated APIs
        // TT_FATAL(device_driver->get_target_mmio_device_ids().size() == 1, "Only one target mmio device id allowed.");
    } else if (this->target_type_ == TargetDevice::Simulator) {
        auto simulator_directory = llrt::RunTimeOptions::get_instance().get_simulator_path();
        device_driver = std::make_unique<tt_SimulationDevice>(simulator_directory);
    }

    barrier_address_params barrier_params;
    barrier_params.tensix_l1_barrier_base =
        tt_metal::hal_ref.get_dev_addr(tt_metal::HalProgrammableCoreType::TENSIX, tt_metal::HalL1MemAddrType::BARRIER);
    barrier_params.dram_barrier_base = tt_metal::hal_ref.get_dev_addr(tt_metal::HalDramMemAddrType::DRAM_BARRIER);

    if (tt_metal::hal_ref.get_arch() != tt::ARCH::GRAYSKULL) {
        barrier_params.eth_l1_barrier_base = tt_metal::hal_ref.get_dev_addr(
            tt_metal::HalProgrammableCoreType::ACTIVE_ETH, tt_metal::HalL1MemAddrType::BARRIER);
    }
    device_driver->set_barrier_address_params(barrier_params);

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
    this->ethernet_sockets_.clear();
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

size_t Cluster::number_of_user_devices() const {
    if (this->cluster_type_ == ClusterType::TG) {
        const auto& chips = this->cluster_desc_->get_all_chips();
        return std::count_if(chips.begin(), chips.end(), [&](const auto& id) {
            return this->cluster_desc_->get_board_type(id) == BoardType::GALAXY;
        });
    } else {
        return this->cluster_desc_->get_number_of_chips();
    }
}

std::unordered_set<chip_id_t> Cluster::user_exposed_chip_ids() const {
    if (this->cluster_type_ == ClusterType::TG) {
        std::unordered_set<chip_id_t> galaxy_boards;
        const auto& chips = this->cluster_desc_->get_all_chips();
        for (const auto& id : chips) {
            if (this->cluster_desc_->get_board_type(id) == BoardType::GALAXY) {
                galaxy_boards.insert(id);
            }
        }
        return galaxy_boards;
    } else {
        return this->cluster_desc_->get_all_chips();
    }
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
    for (auto chip_id : this->cluster_desc_->get_all_chips()) {
        this->virtual_worker_cores_[chip_id] = {};
        for (const tt::umd::CoreCoord& core :
             get_soc_desc(chip_id).get_cores(CoreType::TENSIX, CoordSystem::TRANSLATED)) {
            this->virtual_worker_cores_[chip_id].insert({core.x, core.y});
        }
        this->virtual_eth_cores_[chip_id] = {};
        for (const tt::umd::CoreCoord& core : get_soc_desc(chip_id).get_cores(CoreType::ETH, CoordSystem::TRANSLATED)) {
            this->virtual_eth_cores_[chip_id].insert({core.x, core.y});
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
            this->virtual_routing_to_profiler_flat_id_.at(board_type)
                .insert(
                    {this->get_virtual_coordinate_from_physical_coordinates(chip_id, core_to_profiler_id.first),
                     core_to_profiler_id.second});
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

CoreCoord Cluster::get_virtual_coordinate_from_logical_coordinates(
    chip_id_t chip_id, CoreCoord logical_coord, const CoreType& core_type) const {
    // Keeping the old behavior, although UMD does define translation for other cores as well.
    if (core_type != CoreType::WORKER && core_type != CoreType::DRAM && core_type != CoreType::ETH) {
        TT_THROW("Undefined conversion for core type.");
    }

    auto& soc_desc = this->get_soc_desc(chip_id);
    if (core_type == CoreType::DRAM) {
        return soc_desc.get_physical_dram_core_from_logical(logical_coord);
    }

    // TBD: Remove when all WORKER are rewritten to TENSIX
    CoreType core_type_to_use = core_type;
    if (core_type_to_use == CoreType::WORKER) {
        core_type_to_use = CoreType::TENSIX;
    }

    tt::umd::CoreCoord translated_coord =
        soc_desc.translate_coord_to({logical_coord, core_type_to_use, CoordSystem::LOGICAL}, CoordSystem::TRANSLATED);
    return {translated_coord.x, translated_coord.y};
}

tt_cxy_pair Cluster::get_virtual_coordinate_from_logical_coordinates(tt_cxy_pair logical_coordinate, const CoreType& core_type) const {
    auto xy_virtual_coord = this->get_virtual_coordinate_from_logical_coordinates(logical_coordinate.chip, CoreCoord(logical_coordinate.x, logical_coordinate.y), core_type);
    return tt_cxy_pair(logical_coordinate.chip, xy_virtual_coord);
}
CoreCoord Cluster::get_virtual_coordinate_from_physical_coordinates(chip_id_t chip_id, CoreCoord physical_coord) const {
    auto& soc_desc = this->get_soc_desc(chip_id);
    tt::umd::CoreCoord translated_coord =
        soc_desc.translate_coord_to(physical_coord, CoordSystem::PHYSICAL, CoordSystem::TRANSLATED);
    return {translated_coord.x, translated_coord.y};
}

CoreCoord Cluster::get_physical_coordinate_from_logical_coordinates(
    chip_id_t chip_id, CoreCoord logical_coord, const CoreType& core_type, bool no_warn) const {
    if (!no_warn) {
        log_warning(
            tt::LogDevice,
            "Conversion requested to Physical Coordinates. Please note that Physical Coordinates are not expected to "
            "be used in tt-metal APIs.");
    }
    auto& soc_desc = this->get_soc_desc(chip_id);
    return soc_desc.get_physical_core_from_logical_core(logical_coord, core_type);
}

CoreCoord Cluster::get_logical_ethernet_core_from_virtual(chip_id_t chip, CoreCoord core) const {
    const metal_SocDescriptor& soc_desc = this->get_soc_desc(chip);
    tt::umd::CoreCoord logical_core =
        get_soc_desc(chip).translate_coord_to(core, CoordSystem::TRANSLATED, CoordSystem::LOGICAL);
    return {logical_core.x, logical_core.y};
}

const std::unordered_map<int, int> Cluster::get_worker_logical_to_virtual_x(chip_id_t chip_id) const {
    std::unordered_map<int, int> worker_logical_to_virtual_x;
    const auto& soc_desc = this->get_soc_desc(chip_id);
    for (const tt::umd::CoreCoord& logical_core : soc_desc.get_cores(CoreType::TENSIX, CoordSystem::LOGICAL)) {
        tt::umd::CoreCoord translated_core = soc_desc.translate_coord_to(logical_core, CoordSystem::TRANSLATED);
        worker_logical_to_virtual_x[logical_core.x] = translated_core.x;
    }
    return worker_logical_to_virtual_x;
}

const std::unordered_map<int, int> Cluster::get_worker_logical_to_virtual_y(chip_id_t chip_id) const {
    std::unordered_map<int, int> worker_logical_to_virtual_y;
    const auto& soc_desc = this->get_soc_desc(chip_id);
    for (const tt::umd::CoreCoord& logical_core : soc_desc.get_cores(CoreType::TENSIX, CoordSystem::LOGICAL)) {
        tt::umd::CoreCoord translated_core = soc_desc.translate_coord_to(logical_core, CoordSystem::TRANSLATED);
        worker_logical_to_virtual_y[logical_core.y] = translated_core.y;
    }
    return worker_logical_to_virtual_y;
}

int Cluster::get_device_aiclk(const chip_id_t& chip_id) const {
    if (this->device_to_mmio_device_.find(chip_id) != this->device_to_mmio_device_.end()) {
        // get_clocks returns MMIO device ID -> clock frequency
        // There is one driver per MMIO device, so we use that to index returned map
        chip_id_t mmio_device_id = this->device_to_mmio_device_.at(chip_id);
        return this->driver_->get_clocks().at(mmio_device_id);
    }
    TT_THROW("Cannot get frequency for device {} that is not initialized!", chip_id);
    return 0;
}

void Cluster::deassert_risc_reset_at_core(const tt_cxy_pair& core, const TensixSoftResetOptions& soft_resets) const {
    const metal_SocDescriptor &soc_desc = this->get_soc_desc(core.chip);
    tt::umd::CoreCoord core_coord = soc_desc.get_coord_at(core, CoordSystem::TRANSLATED);
    this->driver_->deassert_risc_reset_at_core(core.chip, core_coord, soft_resets);
}

void Cluster::assert_risc_reset_at_core(const tt_cxy_pair& core, const TensixSoftResetOptions& soft_resets) const {
    const metal_SocDescriptor &soc_desc = this->get_soc_desc(core.chip);
    tt::umd::CoreCoord core_coord = soc_desc.get_coord_at(core, CoordSystem::TRANSLATED);
    this->driver_->assert_risc_reset_at_core(core.chip, core_coord, soft_resets);
}

void Cluster::write_dram_vec(std::vector<uint32_t> &vec, tt_target_dram dram, uint64_t addr, bool small_access) const {
    int chip_id, d_view, d_subchannel;
    std::tie(chip_id, d_view, d_subchannel) = dram;
    const metal_SocDescriptor &desc_to_use = get_soc_desc(chip_id);
    TT_FATAL(
        d_view < desc_to_use.get_num_dram_views(),
        "Bounds-Error -- dram_view={} is outside of num_dram_views={}",
        d_view,
        desc_to_use.get_num_dram_views());
    int d_chan = desc_to_use.get_channel_for_dram_view(d_view);
    TT_ASSERT(
        d_subchannel < desc_to_use.get_dram_cores().at(d_chan).size(),
        "Trying to address dram sub channel that doesnt exist in the device descriptor");
    tt::umd::CoreCoord dram_core_coord =
        desc_to_use.get_dram_core_for_channel(d_chan, d_subchannel, CoordSystem::VIRTUAL);
    tt_cxy_pair dram_core = tt_cxy_pair(chip_id, dram_core_coord.x, dram_core_coord.y);
    size_t offset = desc_to_use.get_address_offset(d_view);
    write_core(vec.data(), vec.size() * sizeof(uint32_t), dram_core, addr + offset, small_access);
}

void Cluster::read_dram_vec(
    std::vector<uint32_t> &vec, uint32_t sz_in_bytes, tt_target_dram dram, uint64_t addr, bool small_access) const {
    int chip_id, d_view, d_subchannel;
    std::tie(chip_id, d_view, d_subchannel) = dram;
    const metal_SocDescriptor &desc_to_use = get_soc_desc(chip_id);
    TT_FATAL(
        d_view < desc_to_use.get_num_dram_views(),
        "Bounds-Error -- dram_view={} is outside of num_dram_views={}",
        d_view,
        desc_to_use.get_num_dram_views());
    int d_chan = desc_to_use.get_channel_for_dram_view(d_view);
    TT_ASSERT(
        d_subchannel < desc_to_use.get_dram_cores().at(d_chan).size(),
        "Trying to address dram sub channel that doesnt exist in the device descriptor");
    tt::umd::CoreCoord dram_core_coord =
        desc_to_use.get_dram_core_for_channel(d_chan, d_subchannel, CoordSystem::VIRTUAL);
    tt_cxy_pair dram_core = tt_cxy_pair(chip_id, dram_core_coord.x, dram_core_coord.y);
    size_t offset = desc_to_use.get_address_offset(d_view);
    read_core(vec, sz_in_bytes, dram_core, addr + offset, small_access);
}

void Cluster::write_core(
    const void *mem_ptr, uint32_t sz_in_bytes, tt_cxy_pair core, uint64_t addr, bool small_access) const {
    chip_id_t chip_id = core.chip;
    const metal_SocDescriptor &soc_desc = this->get_soc_desc(chip_id);
    if (tt::llrt::RunTimeOptions::get_instance().get_watcher_enabled()) {
        tt::watcher_sanitize_host_noc_write(soc_desc, this->virtual_worker_cores_.at(chip_id), this->virtual_eth_cores_.at(chip_id), {core.x, core.y}, addr, sz_in_bytes);

    }
    tt::umd::CoreCoord core_coord = soc_desc.get_coord_at(core, CoordSystem::TRANSLATED);

    this->driver_->write_to_device(mem_ptr, sz_in_bytes, core.chip, core_coord, addr, "LARGE_WRITE_TLB");
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
    tt::umd::CoreCoord core_coord = soc_desc.get_coord_at(core, CoordSystem::TRANSLATED);

    this->driver_->read_from_device(mem_ptr, core.chip, core_coord, addr, size_in_bytes, "LARGE_READ_TLB");
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
    tt::umd::CoreCoord target_coord = soc_desc.get_coord_at(target, CoordSystem::TRANSLATED);
    this->driver_->write_to_device(mem_ptr, size_in_bytes, target.chip, target_coord, addr, "REG_TLB");
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
    tt::umd::CoreCoord target_coord = soc_desc.get_coord_at(target, CoordSystem::TRANSLATED);
    this->driver_->read_from_device(mem_ptr, target.chip, target_coord, addr, size_in_bytes, "REG_TLB");
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
                    get_soc_desc(chip_id).get_eth_core_for_channel(local_chip_chan, CoordSystem::LOGICAL));
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

void Cluster::disable_ethernet_cores_with_retrain() {
    std::vector<uint32_t> read_vec;
    const auto& chips = this->cluster_desc_->get_all_chips();
    for (const auto& chip_id : chips) {
        if (this->frequent_retrain_cores_.find(chip_id) == this->frequent_retrain_cores_.end()) {
            this->frequent_retrain_cores_.insert({chip_id, {}});
        }
        const auto& connected_chips = this->get_ethernet_cores_grouped_by_connected_chips(chip_id);
        for (const auto& [other_chip_id, eth_cores] : connected_chips) {
            for (const auto& eth_core : eth_cores) {
                if (llrt::RunTimeOptions::get_instance().get_skip_eth_cores_with_retrain() and
                    this->cluster_desc_->get_board_type(chip_id) == BoardType::UBB) {
                    tt_cxy_pair virtual_eth_core(
                        chip_id, get_virtual_coordinate_from_logical_coordinates(chip_id, eth_core, CoreType::ETH));
                    auto retrain_count_addr = tt::tt_metal::hal_ref.get_dev_addr(
                        tt::tt_metal::HalProgrammableCoreType::ACTIVE_ETH,
                        tt::tt_metal::HalL1MemAddrType::RETRAIN_COUNT);
                    this->read_core(read_vec, sizeof(uint32_t), virtual_eth_core, retrain_count_addr);
                    if (read_vec[0] != 0) {
                        log_warning(
                            LogDevice,
                            "Disabling active eth core {} due to retraining (count={})",
                            virtual_eth_core.str(),
                            read_vec[0]);
                        this->frequent_retrain_cores_[chip_id].insert(eth_core);
                    }
                }
            }
        }
    }
}

void Cluster::reserve_ethernet_cores_for_tunneling() {
    const char *TT_METAL_SLOW_DISPATCH_MODE = std::getenv("TT_METAL_SLOW_DISPATCH_MODE");
    for (const auto &[assoc_mmio_device, devices] : this->devices_grouped_by_assoc_mmio_device_) {
        for (const auto &chip_id : devices) {
            if (this->device_eth_routing_info_.find(chip_id) == this->device_eth_routing_info_.end()) {
                this->device_eth_routing_info_.insert({chip_id, {}});
            }
        }
        std::map<std::tuple<chip_id_t, chip_id_t>, bool> reserved_chip_connections = {};
        for (const auto &chip_id : devices) {
            if (TT_METAL_SLOW_DISPATCH_MODE == nullptr and arch_ == ARCH::WORMHOLE_B0) {
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
            if (this->device_eth_routing_info_.at(chip_id).at(eth_core) != EthRouterMode::BI_DIR_TUNNELING) {
                device_ids.insert(other_chip_id);
            }
        }
    }
    return device_ids;
}

std::unordered_set<CoreCoord> Cluster::get_active_ethernet_cores(
    chip_id_t chip_id, bool skip_reserved_tunnel_cores) const {
    std::unordered_set<CoreCoord> active_ethernet_cores;
    if (arch_ == ARCH::BLACKHOLE) {
        // Can't just use `get_ethernet_cores_grouped_by_connected_chips` because there are some active ethernet cores
        // without links. Only risc1 on these cores is available for Metal and should not be classified as idle
        // to ensure that Metal does not try to program both riscs.
        const auto& soc_desc = get_soc_desc(chip_id);
        std::set<uint32_t> logical_active_eth_channels = cluster_desc_->get_active_eth_channels(chip_id);
        for (auto logical_active_eth_channel : logical_active_eth_channels) {
            tt::umd::CoreCoord logical_active_eth =
                soc_desc.get_eth_core_for_channel(logical_active_eth_channel, CoordSystem::LOGICAL);
            active_ethernet_cores.insert(CoreCoord(logical_active_eth.x, logical_active_eth.y));
        }

    } else {
        const auto& connected_chips = this->get_ethernet_cores_grouped_by_connected_chips(chip_id);
        for (const auto& [other_chip_id, eth_cores] : connected_chips) {
            for (const auto& eth_core : eth_cores) {
                if (this->device_eth_routing_info_.at(chip_id).at(eth_core) == EthRouterMode::BI_DIR_TUNNELING and
                    skip_reserved_tunnel_cores) {
                    continue;
                }
                if (this->frequent_retrain_cores_.at(chip_id).find(eth_core) !=
                    this->frequent_retrain_cores_.at(chip_id).end()) {
                    continue;
                }

                active_ethernet_cores.insert(eth_core);
            }
        }
    }
    return active_ethernet_cores;
}

tt::tt_fabric::ControlPlane* Cluster::get_control_plane() {
    if (control_plane_.get() == nullptr) {
        this->initialize_control_plane();
    }
    return control_plane_.get();
}

void Cluster::initialize_fabric_config(tt_metal::FabricConfig fabric_config) {
    this->fabric_config_ = fabric_config;
    if (fabric_config != tt_metal::FabricConfig::DISABLED) {
        this->reserve_ethernet_cores_for_fabric_routers();
    } else {
        this->release_ethernet_cores_for_fabric_routers();
    }
    this->get_control_plane()->configure_routing_tables_for_fabric_ethernet_channels();
}

void Cluster::reserve_ethernet_cores_for_fabric_routers() {
    for (const auto& [chip_id, eth_cores] : this->device_eth_routing_info_) {
        for (const auto& [eth_core, mode] : eth_cores) {
            if (mode == EthRouterMode::IDLE) {
                this->device_eth_routing_info_[chip_id][eth_core] = EthRouterMode::FABRIC_ROUTER;
            }
        }
    }
    // Update sockets to reflect fabric routing
    this->ethernet_sockets_.clear();
}

void Cluster::release_ethernet_cores_for_fabric_routers() {
    for (const auto& [chip_id, eth_cores] : this->device_eth_routing_info_) {
        for (const auto& [eth_core, mode] : eth_cores) {
            if (mode == EthRouterMode::FABRIC_ROUTER) {
                this->device_eth_routing_info_[chip_id][eth_core] = EthRouterMode::IDLE;
            }
        }
    }
    // TODO: We should just cache restore
    this->initialize_ethernet_sockets();
}

std::set<tt_fabric::chan_id_t> Cluster::get_fabric_ethernet_channels(chip_id_t chip_id) const {
    std::set<tt_fabric::chan_id_t> fabric_ethernet_channels;
    const auto& connected_chips = this->get_ethernet_cores_grouped_by_connected_chips(chip_id);
    for (const auto& [other_chip_id, eth_cores] : connected_chips) {
        for (const auto& eth_core : eth_cores) {
            if (this->device_eth_routing_info_.at(chip_id).at(eth_core) == EthRouterMode::FABRIC_ROUTER) {
                fabric_ethernet_channels.insert(this->get_soc_desc(chip_id).logical_eth_core_to_chan_map.at(eth_core));
            }
        }
    }
    return fabric_ethernet_channels;
}

std::unordered_set<CoreCoord> Cluster::get_inactive_ethernet_cores(chip_id_t chip_id) const {
    std::unordered_set<CoreCoord> active_ethernet_cores = this->get_active_ethernet_cores(chip_id);
    std::unordered_set<CoreCoord> inactive_ethernet_cores;
    std::unordered_set<int> channels_to_skip = {};
    // UMD routing FW uses these cores for base routing
    // channel 15 is used by syseng tools.
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

bool Cluster::is_ethernet_link_up(chip_id_t chip_id, const CoreCoord& logical_core) const {
    const auto& soc_desc = get_soc_desc(chip_id);
    ethernet_channel_t eth_chan = soc_desc.logical_eth_core_to_chan_map.at(logical_core);
    return this->cluster_desc_->ethernet_core_has_active_ethernet_link(chip_id, eth_chan);
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
        std::get<0>(connected_eth_core),
        soc_desc.get_eth_core_for_channel(std::get<1>(connected_eth_core), CoordSystem::LOGICAL));
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

CoreCoord Cluster::get_virtual_eth_core_from_channel(chip_id_t chip_id, int channel) const {
    tt::umd::CoreCoord logical_coord =
        this->get_soc_desc(chip_id).get_eth_core_for_channel(channel, CoordSystem::LOGICAL);
    return this->get_virtual_coordinate_from_logical_coordinates(
        chip_id, {logical_coord.x, logical_coord.y}, CoreType::ETH);
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
    // TODO: initialize devices if user does not
    // Must initialize remote chips first, then mmio chips since once mmio chips are doing fd routing
    // we do not always context switch to base FW
    std::vector<chip_id_t> non_mmio_devices;
    std::vector<chip_id_t> mmio_devices = target_mmio_devices;
    if (mmio_devices.size() == 0) {
        mmio_devices.reserve(this->number_of_pci_devices());
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
                    (void *)&routing_info_enabled, sizeof(routing_info_t), virtual_eth_core, routing_info_addr_, false);
            }
        }
        for (const auto &chip_id : mmio_devices) {
            for (const auto &[eth_core, routing_info] : this->device_eth_routing_info_.at(chip_id)) {
                tt_cxy_pair virtual_eth_core(chip_id, get_virtual_coordinate_from_logical_coordinates(chip_id, eth_core, CoreType::ETH));
                // Enable internal ethernet routing for mmio devices
                write_core(
                    (void *)&routing_info_enabled, sizeof(routing_info_t), virtual_eth_core, routing_info_addr_, false);
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
                    (void *)&routing_info_disabled, sizeof(routing_info_t), virtual_eth_core, routing_info_addr_, false);
            }
        }
        for (const auto &chip_id : non_mmio_devices) {
            for (const auto &[eth_core, routing_info] : this->device_eth_routing_info_.at(chip_id)) {
                tt_cxy_pair virtual_eth_core(chip_id, get_virtual_coordinate_from_logical_coordinates(chip_id, eth_core, CoreType::ETH));
                // Disable internal ethernet routing for non-mmio devices
                write_core(
                    (void *)&routing_info_disabled, sizeof(routing_info_t), virtual_eth_core, routing_info_addr_, false);
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

void Cluster::initialize_control_plane() {
    // Default mode, auto select mesh graph descriptor. In future, we can add a way for user to specify custom
    // descriptors
    std::string mesh_graph_descriptor;
    switch (this->cluster_type_) {
        case tt::ClusterType::N150: mesh_graph_descriptor = "n150_mesh_graph_descriptor.yaml"; break;
        case tt::ClusterType::N300: mesh_graph_descriptor = "n300_mesh_graph_descriptor.yaml"; break;
        case tt::ClusterType::T3K: mesh_graph_descriptor = "t3k_mesh_graph_descriptor.yaml"; break;
        case tt::ClusterType::GALAXY: mesh_graph_descriptor = "quanta_galaxy_mesh_graph_descriptor.yaml"; break;
        case tt::ClusterType::TG: mesh_graph_descriptor = "tg_mesh_graph_descriptor.yaml"; break;
        case tt::ClusterType::P100: mesh_graph_descriptor = "p100_mesh_graph_descriptor.yaml"; break;
        case tt::ClusterType::P150: mesh_graph_descriptor = "p150_mesh_graph_descriptor.yaml"; break;
        case tt::ClusterType::P150_X2: mesh_graph_descriptor = "p150_x2_mesh_graph_descriptor.yaml"; break;
        case tt::ClusterType::P150_X4: mesh_graph_descriptor = "p150_x4_mesh_graph_descriptor.yaml"; break;
        default: TT_THROW("Unknown cluster type"); // TODO: we could expose this as a custom mesh graph option
    }
    const std::filesystem::path mesh_graph_desc_path =
        std::filesystem::path(tt::llrt::RunTimeOptions::get_instance().get_root_dir()) /
        "tt_metal/fabric/mesh_graph_descriptors" / mesh_graph_descriptor;

    control_plane_ = std::make_unique<tt::tt_fabric::ControlPlane>(mesh_graph_desc_path.string());
}

}  // namespace tt

std::ostream &operator<<(std::ostream &os, tt_target_dram const &dram) {
    os << "Target DRAM chip = " << std::get<0>(dram) << ", chan = " << std::get<1>(dram);
    return os;
}
