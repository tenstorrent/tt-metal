// SPDX-FileCopyrightText: © 2023 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "tt_cluster.hpp"
#include "llrt/rtoptions.hpp"

#include <core_coord.hpp>
#include <tt-logger/tt-logger.hpp>
#include "llrt/metal_soc_descriptor.hpp"
#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <tuple>  // for get
#include <unordered_map>
#include <unordered_set>
#include <utility>

#include <tt-metalium/experimental/fabric/control_plane.hpp>
#include <tt-metalium/experimental/fabric/fabric_types.hpp>
#include "common/executor.hpp"
#include "get_platform_architecture.hpp"
#include "hal_types.hpp"
#include "impl/context/metal_context.hpp"
#include "llrt/hal.hpp"
#include "sanitize_noc_host.hpp"
#include "tracy/Tracy.hpp"
#include "tt_metal/llrt/tlb_config.hpp"
#include "tunnels_from_mmio_device.hpp"
#include "umd/device/utils/semver.hpp"
#include <umd/device/cluster.hpp>
#include <umd/device/cluster_descriptor.hpp>
#include <umd/device/simulation/simulation_chip.hpp>
#include <umd/device/pcie/pci_device.hpp>
#include <umd/device/types/arch.hpp>
#include <umd/device/types/cluster_descriptor_types.hpp>
#include <umd/device/types/cluster_types.hpp>
#include <umd/device/types/xy_pair.hpp>
#include <unistd.h>

static constexpr uint32_t HOST_MEM_CHANNELS = 4;
static constexpr uint32_t HOST_MEM_CHANNELS_MASK = HOST_MEM_CHANNELS - 1;

namespace {

inline std::string get_soc_description_file(
    const tt::ARCH& arch, tt::TargetDevice target_device, const tt::llrt::RunTimeOptions& rtoptions) {
    if (target_device == tt::TargetDevice::Simulator) {
        return tt::umd::SimulationChip::get_soc_descriptor_path_from_simulator_path(rtoptions.get_simulator_path());
    }
    std::string path = rtoptions.get_root_dir();
    if (path.back() != '/') {
        path.push_back('/');
    }
    path += "tt_metal/soc_descriptors/";
    const char* file = nullptr;
    switch (arch) {
        case tt::ARCH::WORMHOLE_B0: file = "wormhole_b0_80_arch.yaml"; break;
        case tt::ARCH::BLACKHOLE: file = "blackhole_140_arch.yaml"; break;
        case tt::ARCH::QUASAR:  // Quasar is currently only supported for simulation
        default: throw std::runtime_error("Unsupported device arch");
    }
    path += file;
    return path;
}

std::unique_ptr<tt::umd::ClusterDescriptor> get_mock_cluster_desc(const tt::llrt::RunTimeOptions& rtoptions) {
    TT_FATAL(rtoptions.get_mock_enabled(), "Mock cluster descriptor not enabled");
    std::unique_ptr<tt::umd::ClusterDescriptor> mock_cluster_desc =
        tt::umd::ClusterDescriptor::create_from_yaml(rtoptions.get_mock_cluster_desc_path());
    TT_FATAL(
        mock_cluster_desc != nullptr,
        "Failed to load mock cluster descriptor from {}",
        rtoptions.get_mock_cluster_desc_path());
    return mock_cluster_desc;
}

}  // namespace
namespace tt {

tt::tt_metal::ClusterType Cluster::get_cluster_type_from_cluster_desc(
    const llrt::RunTimeOptions& rtoptions, const umd::ClusterDescriptor* cluster_desc) {
    if (rtoptions.get_simulator_enabled() && !rtoptions.get_mock_enabled()) {
        auto soc_desc =
            tt::umd::SimulationChip::get_soc_descriptor_path_from_simulator_path(rtoptions.get_simulator_path());
        auto arch = tt::umd::SocDescriptor::get_arch_from_soc_descriptor_path(soc_desc);
        if (arch == tt::ARCH::WORMHOLE_B0) {
            return tt::tt_metal::ClusterType::SIMULATOR_WORMHOLE_B0;
        }
        if (arch == tt::ARCH::BLACKHOLE) {
            return tt::tt_metal::ClusterType::SIMULATOR_BLACKHOLE;
        }
        if (arch == tt::ARCH::QUASAR) {
            return tt::tt_metal::ClusterType::SIMULATOR_QUASAR;
        }
        return tt::tt_metal::ClusterType::INVALID;
    }

    std::unique_ptr<umd::ClusterDescriptor> temp_cluster_desc = nullptr;
    if (cluster_desc == nullptr) {
        temp_cluster_desc = rtoptions.get_mock_enabled() ? get_mock_cluster_desc(rtoptions)
                                                         : tt::umd::Cluster::create_cluster_descriptor();
        cluster_desc = temp_cluster_desc.get();
    }
    tt::tt_metal::ClusterType cluster_type = tt::tt_metal::ClusterType::INVALID;
    for (const auto& chip_id : cluster_desc->get_all_chips()) {
        if (cluster_desc->get_board_type(chip_id) == BoardType::GALAXY) {
            cluster_type = tt::tt_metal::ClusterType::TG;
            break;
        }
    }
    const auto num_chips = cluster_desc->get_all_chips().size();
    TT_FATAL(num_chips > 0, "No chips detected in the cluster");
    const auto board_type = cluster_desc->get_board_type(*cluster_desc->get_all_chips().begin());
    bool all_same_board = true;
    for (const auto& chip_id : cluster_desc->get_all_chips()) {
        if (cluster_desc->get_board_type(chip_id) != board_type) {
            all_same_board = false;
            break;
        }
    }

    if (all_same_board) {
        if (board_type == BoardType::N300) {
            if (num_chips == 8) {
                cluster_type = tt::tt_metal::ClusterType::T3K;
                // Basic check to determine if the cluster is a T3K cluster
                // MMIO chips should have 3 connections to other chips, remote chips should have 2 connections to other
                // chips
                for (const auto& [chip_id, connections] : cluster_desc->get_ethernet_connections()) {
                    std::unordered_set<ChipId> remote_chips;
                    for (const auto& [channel, remote_chip_and_channel] : connections) {
                        remote_chips.insert(std::get<0>(remote_chip_and_channel));
                    }
                    if (cluster_desc->is_chip_mmio_capable(chip_id)) {
                        if (remote_chips.size() != 3) {
                            cluster_type = tt::tt_metal::ClusterType::N300;
                            break;
                        }
                    } else {
                        if (remote_chips.size() != 2) {
                            cluster_type = tt::tt_metal::ClusterType::N300;
                            break;
                        }
                    }
                }
            } else if (num_chips == 4) {
                cluster_type = tt::tt_metal::ClusterType::N300_2x2;

                // Expect every chip to have exactly two remote connections
                for (const auto& [chip_id, connections] : cluster_desc->get_ethernet_connections()) {
                    std::unordered_set<ChipId> remote_chips;
                    for (const auto& [channel, remote_chip_and_channel] : connections) {
                        remote_chips.insert(std::get<0>(remote_chip_and_channel));
                    }
                    if (remote_chips.size() != 2) {
                        cluster_type = tt::tt_metal::ClusterType::N300;
                        break;
                    }
                }
            } else {
                cluster_type = tt::tt_metal::ClusterType::N300;
            }
        } else if (board_type == BoardType::N150) {
            cluster_type = tt::tt_metal::ClusterType::N150;
        } else if (board_type == BoardType::P100) {
            if (num_chips == 1) {
                cluster_type = tt::tt_metal::ClusterType::P100;
            } else {
                log_warning(tt::LogDevice, "Using CUSTOM cluster type for P100 board with {} chips", num_chips);
                cluster_type = tt::tt_metal::ClusterType::CUSTOM;
            }
        } else if (board_type == BoardType::P150) {
            if (num_chips == 1) {
                cluster_type = tt::tt_metal::ClusterType::P150;
            } else if (num_chips == 2) {
                cluster_type = tt::tt_metal::ClusterType::P150_X2;
            } else if (num_chips == 4) {
                cluster_type = tt::tt_metal::ClusterType::P150_X4;
            } else if (num_chips == 8) {
                cluster_type = tt::tt_metal::ClusterType::P150_X8;
            } else {
                log_warning(tt::LogDevice, "Using CUSTOM cluster type for P150 board with {} chips", num_chips);
                cluster_type = tt::tt_metal::ClusterType::CUSTOM;
            }
        } else if (board_type == BoardType::P300) {
            // PCIe is enabled to both chips on the P300 board
            if (num_chips == 2) {
                cluster_type = tt::tt_metal::ClusterType::P300;
            } else if (num_chips == 4) {
                cluster_type = tt::tt_metal::ClusterType::P300_X2;
            } else {
                log_warning(tt::LogDevice, "Using CUSTOM cluster type for P300 board with {} chips", num_chips);
                cluster_type = tt::tt_metal::ClusterType::CUSTOM;
            }
        } else if (board_type == BoardType::UBB) {
            cluster_type = tt::tt_metal::ClusterType::GALAXY;
        } else if (board_type == BoardType::UBB_BLACKHOLE) {
            cluster_type = tt::tt_metal::ClusterType::BLACKHOLE_GALAXY;
        }
    }
    return cluster_type;
}

bool Cluster::is_base_routing_fw_enabled(tt::tt_metal::ClusterType cluster_type) {
    // Ideally we should get the routing enabled/disabled from a config in L1
    return (
        cluster_type == tt::tt_metal::ClusterType::INVALID || cluster_type == tt::tt_metal::ClusterType::N150 ||
        cluster_type == tt::tt_metal::ClusterType::N300 || cluster_type == tt::tt_metal::ClusterType::T3K ||
        cluster_type == tt::tt_metal::ClusterType::N300_2x2 || cluster_type == tt::tt_metal::ClusterType::TG);
}

bool Cluster::is_iommu_enabled() const { return this->iommu_enabled_; }

bool Cluster::is_noc_mapping_enabled() const { return this->noc_mapping_enabled_; }

Cluster::Cluster(llrt::RunTimeOptions& rtoptions, const tt_metal::Hal& hal) : rtoptions_(rtoptions), hal_(hal) {
    ZoneScoped;
    log_info(tt::LogDevice, "Opening user mode device driver");

    this->detect_arch_and_target();

    routing_info_addr_ = hal_.get_dev_addr(
        tt::tt_metal::HalProgrammableCoreType::ACTIVE_ETH, tt::tt_metal::HalL1MemAddrType::APP_ROUTING_INFO);

    this->initialize_device_drivers();

    this->disable_ethernet_cores_with_retrain();

    this->initialize_ethernet_cores_router_mode();

    this->initialize_ethernet_sockets();

    TT_FATAL(this->driver_, "UMD cluster object must be initialized and available");
    this->tunnels_from_mmio_device = llrt::discover_tunnels_from_mmio_device(*this->driver_);

    if (this->target_type_ != tt::TargetDevice::Mock) {
        this->assert_risc_reset();
    }
}

void Cluster::detect_arch_and_target() {
    this->target_type_ = rtoptions_.get_target_device();

    this->arch_ = tt_metal::get_platform_architecture(rtoptions_);

    if (this->target_type_ == tt::TargetDevice::Mock) {
        log_warning(tt::LogDevice, "Currently using mock cluster descriptor, all device driver calls will be mocked");
    }

    TT_FATAL(
        this->target_type_ == tt::TargetDevice::Silicon || this->target_type_ == tt::TargetDevice::Simulator ||
            this->target_type_ == tt::TargetDevice::Mock,
        "Target type={} is not supported",
        this->target_type_);
}

// TODO: remove this when we deprecate TG
bool Cluster::is_galaxy_cluster() const { return this->cluster_type_ == tt::tt_metal::ClusterType::TG; }

bool Cluster::is_ubb_galaxy() const {
    return this->cluster_type_ == tt::tt_metal::ClusterType::BLACKHOLE_GALAXY ||
           this->cluster_type_ == tt::tt_metal::ClusterType::GALAXY;
}

tt::tt_metal::ClusterType Cluster::get_cluster_type() const { return this->cluster_type_; }

BoardType Cluster::get_board_type(ChipId chip_id) const { return this->cluster_desc_->get_board_type(chip_id); }

bool Cluster::is_base_routing_fw_enabled() const { return Cluster::is_base_routing_fw_enabled(this->cluster_type_); }

void Cluster::generate_cluster_descriptor() {
    this->cluster_desc_ = this->driver_->get_cluster_description();
    this->cluster_type_ = Cluster::get_cluster_type_from_cluster_desc(this->rtoptions_, this->cluster_desc_);
    if (this->cluster_type_ == tt::tt_metal::ClusterType::CUSTOM) {
        TT_FATAL(
            this->rtoptions_.is_custom_fabric_mesh_graph_desc_path_specified(),
            "Custom fabric mesh graph descriptor path must be specified for CUSTOM cluster type");
    }
    if (this->target_type_ == TargetDevice::Simulator || this->target_type_ == TargetDevice::Mock) {
        return;
    }

    if (this->arch_ == tt::ARCH::BLACKHOLE) {
        TT_FATAL(
            this->cluster_desc_->get_noc_translation_table_en().at(0),
            "Running Metal on Blackhole requires FW >= 80.18.0.0");
    }
}

void Cluster::validate_harvesting_masks() const {
    // Metal expects all chips to have same number of harvested cores for a given core type
    std::optional<HarvestingMasks> harvesting_mask_tracker = std::nullopt;
    for (const auto device_id : this->user_exposed_chip_ids()) {
        HarvestingMasks masks = sdesc_per_chip_.at(device_id).harvesting_masks;
        if (!harvesting_mask_tracker.has_value()) {
            harvesting_mask_tracker = masks;
        } else {
            TT_FATAL(
                std::popcount(masks.tensix_harvesting_mask) ==
                    std::popcount(harvesting_mask_tracker->tensix_harvesting_mask),
                "Number of harvested Tensix mismatch across devices");
            TT_FATAL(
                std::popcount(masks.dram_harvesting_mask) ==
                    std::popcount(harvesting_mask_tracker->dram_harvesting_mask),
                "Number of harvested Dram mismatch across devices");
            TT_FATAL(
                std::popcount(masks.eth_harvesting_mask) == std::popcount(harvesting_mask_tracker->eth_harvesting_mask),
                "Number of harvested Eth mismatch across devices");
            TT_FATAL(
                std::popcount(masks.pcie_harvesting_mask) ==
                    std::popcount(harvesting_mask_tracker->pcie_harvesting_mask),
                "Number of harvested Pcie mismatch across devices");
        }
    }
}

void Cluster::initialize_device_drivers() {
    this->open_driver();
    this->generate_cluster_descriptor();
    this->get_metal_desc_from_tt_desc();
    this->validate_harvesting_masks();

    for (const auto& [mmio_device_id, controlled_devices] : this->cluster_desc_->get_chips_grouped_by_closest_mmio()) {
        this->assign_mem_channels_to_devices(mmio_device_id, controlled_devices);
    }

    umd::DeviceParams default_params;
    this->start_driver(default_params);
    this->generate_virtual_to_umd_coord_mapping();
    this->generate_virtual_to_profiler_flat_id_mapping();

    // Cache IOMMU status (expensive to query repeatedly)
    this->iommu_enabled_ = false;
    this->noc_mapping_enabled_ = false;
    if (this->target_type_ == tt::TargetDevice::Silicon) {
        const auto& mmio_ids = this->driver_->get_target_mmio_device_ids();
        if (!mmio_ids.empty()) {
            ChipId mmio_id = *mmio_ids.begin();
            auto pci = this->driver_->get_chip(mmio_id)->get_tt_device()->get_pci_device();
            if (pci) {
                this->iommu_enabled_ = pci->is_iommu_enabled();
                this->noc_mapping_enabled_ = tt::umd::PCIDevice::is_mapping_buffer_to_noc_supported();
            }
        }
    }
}

void Cluster::assert_risc_reset() { this->driver_->assert_risc_reset(); }

void Cluster::assign_mem_channels_to_devices(
    ChipId mmio_device_id, const std::unordered_set<ChipId>& controlled_device_ids) {
    // g_MAX_HOST_MEM_CHANNELS (4) is defined in tt::umd::Cluster and denotes the max number of host memory channels per
    // MMIO device Metal currently assigns 1 channel per device. See https://github.com/tenstorrent/tt-metal/issues/4087
    // One WH gateway should have 8 remote deivces in its control group.
    TT_ASSERT(controlled_device_ids.size() <= 9, "Unable to assign each device to its own host memory channel!");
    uint16_t channel = 0;
    this->device_to_host_mem_channel_[mmio_device_id] = channel++;
    for (const ChipId& device_id : controlled_device_ids) {
        if (device_id == mmio_device_id) {
            continue;
        }
        this->device_to_host_mem_channel_[device_id] = channel++;
        if ((channel + 1) % 4 == 0) {
            channel++;
        }
    }
}

void Cluster::get_metal_desc_from_tt_desc() {
    for (const auto& id : this->driver_->get_target_device_ids()) {
        this->sdesc_per_chip_.emplace(
            id, metal_SocDescriptor(this->driver_->get_soc_descriptor(id), this->cluster_desc_->get_board_type(id)));
    }
}

const std::unordered_map<CoreCoord, int32_t>& Cluster::get_virtual_routing_to_profiler_flat_id(ChipId chip_id) const {
    return this->virtual_routing_to_profiler_flat_id_.at(this->get_board_type(chip_id));
}

void Cluster::open_driver(const bool& /*skip_driver_allocs*/) {
    std::unique_ptr<tt::umd::Cluster> device_driver;
    std::string sdesc_path = get_soc_description_file(this->arch_, this->target_type_, rtoptions_);
    if (this->target_type_ == TargetDevice::Silicon) {
        // This is the target/desired number of mem channels per arch/device.
        // Silicon driver will attempt to open this many hugepages as channels per mmio chip,
        // and assert if workload uses more than available.
        auto temp_cluster_desc = tt::umd::Cluster::create_cluster_descriptor();
        auto grouped_chips = temp_cluster_desc->get_chips_grouped_by_closest_mmio();
        uint32_t max_chips_per_mmio = 0;
        for (const auto& [mmio_device_id, chips] : grouped_chips) {
            max_chips_per_mmio = std::max(max_chips_per_mmio, static_cast<uint32_t>(chips.size()));
        }
        device_driver = std::make_unique<tt::umd::Cluster>(tt::umd::ClusterOptions{
            .num_host_mem_ch_per_mmio_device = std::min(HOST_MEM_CHANNELS, max_chips_per_mmio),
            .sdesc_path = sdesc_path,
        });
    } else if (this->target_type_ == TargetDevice::Simulator) {
        std::unique_ptr<umd::ClusterDescriptor> mock_cluster_desc;
        if (rtoptions_.get_mock_enabled()) {
            mock_cluster_desc = get_mock_cluster_desc(rtoptions_);
            device_driver = std::make_unique<tt::umd::Cluster>(tt::umd::ClusterOptions{
                .chip_type = tt::umd::ChipType::SIMULATION,
                .sdesc_path = sdesc_path,
                .cluster_descriptor = mock_cluster_desc.get(),
                .simulator_directory = rtoptions_.get_simulator_path(),
            });
        } else {
            device_driver = std::make_unique<tt::umd::Cluster>(tt::umd::ClusterOptions{
                .chip_type = tt::umd::ChipType::SIMULATION,
                .target_devices = {0},
                .simulator_directory = rtoptions_.get_simulator_path(),
            });
        }
    } else if (this->target_type_ == TargetDevice::Mock) {
        // If a cluster descriptor was not provided via constructor, and mock is enabled via rtoptions,
        // load it from the YAML path and pass it into UMD for mock initialization.
        auto mock_cluster_desc = get_mock_cluster_desc(rtoptions_);

        device_driver = std::make_unique<tt::umd::Cluster>(tt::umd::ClusterOptions{
            .chip_type = tt::umd::ChipType::MOCK,
            .sdesc_path = sdesc_path,
            .cluster_descriptor = mock_cluster_desc.get(),
        });
    }

    umd::BarrierAddressParams barrier_params;
    barrier_params.tensix_l1_barrier_base =
        hal_.get_dev_addr(tt_metal::HalProgrammableCoreType::TENSIX, tt_metal::HalL1MemAddrType::BARRIER);
    barrier_params.dram_barrier_base = hal_.get_dev_addr(tt_metal::HalDramMemAddrType::BARRIER);

    barrier_params.eth_l1_barrier_base =
        hal_.get_dev_addr(tt_metal::HalProgrammableCoreType::ACTIVE_ETH, tt_metal::HalL1MemAddrType::BARRIER);
    device_driver->set_barrier_address_params(barrier_params);

    this->driver_ = std::move(device_driver);
}

void Cluster::start_driver(umd::DeviceParams& device_params) const {
    device_params.init_device = true;

    TT_FATAL(!this->sdesc_per_chip_.empty(), "Descriptor must be loaded. Try open_driver()");

    // May block waiting for other processes to release the device.
    this->driver_->start_device(device_params);

    if (this->target_type_ == TargetDevice::Silicon && device_params.init_device) {
        // Configure TLBs on all MMIO devices in parallel
        std::vector<std::shared_future<void>> futures;
        const auto& mmio_device_ids = driver_->get_target_mmio_device_ids();
        futures.reserve(mmio_device_ids.size());

        for (const auto& mmio_device_id : mmio_device_ids) {
            futures.emplace_back(tt_metal::detail::async([this, mmio_device_id]() {
                ll_api::configure_static_tlbs(
                    this->arch_, mmio_device_id, this->get_soc_desc(mmio_device_id), *this->driver_);
            }));
        }

        // Wait for all TLB configurations to complete
        for (auto& future : futures) {
            future.get();
        }
    }
}

Cluster::~Cluster() {
    log_info(tt::LogDevice, "Closing user mode device drivers");
    this->driver_->close_device();

    this->sdesc_per_chip_.clear();
    this->device_to_host_mem_channel_.clear();
    this->device_eth_routing_info_.clear();
    this->tunnels_from_mmio_device.clear();
    this->ethernet_sockets_.clear();
}

std::unordered_map<ChipId, EthCoord> Cluster::get_user_chip_ethernet_coordinates() const {
    auto user_chip_ethernet_coordinates = this->cluster_desc_->get_chip_locations();
    if (this->is_galaxy_cluster()) {
        std::erase_if(user_chip_ethernet_coordinates, [this](const auto& entry) {
            return this->cluster_desc_->get_board_type(entry.first) != BoardType::GALAXY;
        });
    }
    return user_chip_ethernet_coordinates;
}

std::unordered_map<ChipId, EthCoord> Cluster::get_all_chip_ethernet_coordinates() const {
    return this->cluster_desc_->get_chip_locations();
}

ChipId Cluster::get_physical_chip_id_from_eth_coord(const EthCoord& eth_coord) const {
    for (const auto& [physical_chip_id, coord] : this->get_all_chip_ethernet_coordinates()) {
        if (coord == eth_coord) {
            return physical_chip_id;
        }
    }
    TT_FATAL(false, "Physical chip id not found for eth coord");
    return 0;
}

size_t Cluster::number_of_user_devices() const {
    if (this->cluster_type_ == tt::tt_metal::ClusterType::TG) {
        const auto& chips = this->driver_->get_target_device_ids();
        return std::count_if(chips.begin(), chips.end(), [&](const auto& id) {
            return this->cluster_desc_->get_board_type(id) == BoardType::GALAXY;
        });
    }
    return this->driver_->get_target_device_ids().size();
}

std::set<ChipId> Cluster::user_exposed_chip_ids() const {
    if (this->cluster_type_ == tt::tt_metal::ClusterType::TG) {
        std::set<ChipId> galaxy_boards;
        const auto& chips = this->driver_->get_target_device_ids();
        for (const auto& id : chips) {
            if (this->cluster_desc_->get_board_type(id) == BoardType::GALAXY) {
                galaxy_boards.insert(id);
            }
        }
        return galaxy_boards;
    }
    return this->driver_->get_target_device_ids();
}

const metal_SocDescriptor& Cluster::get_soc_desc(ChipId chip) const {
    if (!this->sdesc_per_chip_.contains(chip)) {
        TT_THROW(
            "Cannot access soc descriptor for {} before device driver is initialized! Call "
            "initialize_device_driver({}) first",
            chip,
            chip);
    }
    return this->sdesc_per_chip_.at(chip);
}

void Cluster::generate_virtual_to_umd_coord_mapping() {
    for (auto chip_id : this->driver_->get_target_device_ids()) {
        this->virtual_worker_cores_[chip_id] = {};
        for (const tt::umd::CoreCoord& core :
             get_soc_desc(chip_id).get_cores(CoreType::TENSIX, CoordSystem::TRANSLATED)) {
            this->virtual_worker_cores_[chip_id].insert({core.x, core.y});
        }
        this->virtual_eth_cores_[chip_id] = {};
        for (const tt::umd::CoreCoord& core : get_soc_desc(chip_id).get_cores(CoreType::ETH, CoordSystem::TRANSLATED)) {
            this->virtual_eth_cores_[chip_id].insert({core.x, core.y});
        }
        this->virtual_pcie_cores_[chip_id] = {};
        this->virtual_dram_cores_[chip_id] = {};
        if (this->arch_ == ARCH::BLACKHOLE) {
            for (const tt::umd::CoreCoord& core :
                 get_soc_desc(chip_id).get_cores(CoreType::PCIE, CoordSystem::TRANSLATED)) {
                this->virtual_pcie_cores_[chip_id].insert({core.x, core.y});
            }

            for (uint32_t noc = 0; noc < hal_.get_num_nocs(); noc++) {
                for (auto dram_channel = 0; dram_channel < this->get_soc_desc(chip_id).get_num_dram_views();
                     dram_channel++) {
                    auto worker_dram_ep =
                        this->get_soc_desc(chip_id).get_preferred_worker_core_for_dram_view(dram_channel, noc);
                    auto eth_dram_ep =
                        this->get_soc_desc(chip_id).get_preferred_eth_core_for_dram_view(dram_channel, noc);
                    this->virtual_dram_cores_[chip_id].insert({worker_dram_ep.x, worker_dram_ep.y});
                    if (worker_dram_ep != eth_dram_ep) {
                        this->virtual_dram_cores_[chip_id].insert({eth_dram_ep.x, eth_dram_ep.y});
                    }
                }
            }
        }
    }
}

void Cluster::generate_virtual_to_profiler_flat_id_mapping() {
#if defined(TRACY_ENABLE)
    for (auto chip_id : this->driver_->get_target_device_ids()) {
        auto board_type = this->get_board_type(chip_id);
        if (this->virtual_routing_to_profiler_flat_id_.contains(board_type)) {
            continue;
        }
        this->virtual_routing_to_profiler_flat_id_.insert({board_type, {}});
        const auto& soc_desc = this->get_soc_desc(chip_id);
        for (const auto& core_to_profiler_id : soc_desc.physical_routing_to_profiler_flat_id) {
            this->virtual_routing_to_profiler_flat_id_.at(board_type)
                .insert(
                    {this->get_virtual_coordinate_from_physical_coordinates(chip_id, core_to_profiler_id.first),
                     core_to_profiler_id.second});
        }
    }
#endif
}

bool Cluster::is_worker_core(const CoreCoord& core, ChipId chip_id) const {
    return this->virtual_worker_cores_.at(chip_id).contains(core);
}

bool Cluster::is_ethernet_core(const CoreCoord& core, ChipId chip_id) const {
    return this->virtual_eth_cores_.contains(chip_id) and this->virtual_eth_cores_.at(chip_id).contains(core);
}

const std::unordered_set<CoreCoord>& Cluster::get_virtual_worker_cores(ChipId chip_id) const {
    return this->virtual_worker_cores_.at(chip_id);
}

const std::unordered_set<CoreCoord>& Cluster::get_virtual_eth_cores(ChipId chip_id) const {
    return this->virtual_eth_cores_.at(chip_id);
}

CoreCoord Cluster::get_virtual_coordinate_from_logical_coordinates(
    ChipId chip_id, CoreCoord logical_coord, const CoreType& core_type) const {
    // TBD: Remove when all WORKER are rewritten to TENSIX
    CoreType core_type_to_use = core_type;
    if (core_type_to_use == CoreType::WORKER) {
        core_type_to_use = CoreType::TENSIX;
    }

    // Keeping the old behavior, although UMD does define translation for other cores as well.
    if (core_type_to_use != CoreType::TENSIX && core_type != CoreType::DRAM && core_type != CoreType::ETH) {
        TT_THROW("Undefined conversion for core type.");
    }

    const auto& soc_desc = this->get_soc_desc(chip_id);
    if (core_type == CoreType::DRAM) {
        return soc_desc.get_physical_dram_core_from_logical(logical_coord);
    }

    tt::umd::CoreCoord translated_coord =
        soc_desc.translate_coord_to({logical_coord, core_type_to_use, CoordSystem::LOGICAL}, CoordSystem::TRANSLATED);
    return {translated_coord.x, translated_coord.y};
}

tt_cxy_pair Cluster::get_virtual_coordinate_from_logical_coordinates(
    tt_cxy_pair logical_coordinate, const CoreType& core_type) const {
    auto xy_virtual_coord = this->get_virtual_coordinate_from_logical_coordinates(
        logical_coordinate.chip, CoreCoord(logical_coordinate.x, logical_coordinate.y), core_type);
    return tt_cxy_pair(logical_coordinate.chip, xy_virtual_coord);
}
CoreCoord Cluster::get_virtual_coordinate_from_physical_coordinates(ChipId chip_id, CoreCoord physical_coord) const {
    const auto& soc_desc = this->get_soc_desc(chip_id);
    tt::umd::CoreCoord translated_coord =
        soc_desc.translate_coord_to(physical_coord, CoordSystem::NOC0, CoordSystem::TRANSLATED);
    return {translated_coord.x, translated_coord.y};
}

CoreCoord Cluster::get_physical_coordinate_from_logical_coordinates(
    ChipId chip_id, CoreCoord logical_coord, const CoreType& core_type, bool no_warn) const {
    if (!no_warn) {
        log_warning(
            tt::LogDevice,
            "Conversion requested to Physical Coordinates. Please note that Physical Coordinates are not expected to "
            "be used in tt-metal APIs.");
    }
    const auto& soc_desc = this->get_soc_desc(chip_id);
    return soc_desc.get_physical_core_from_logical_core(logical_coord, core_type);
}

CoreCoord Cluster::get_logical_ethernet_core_from_virtual(ChipId chip, CoreCoord core) const {
    tt::umd::CoreCoord logical_core =
        get_soc_desc(chip).translate_coord_to(core, CoordSystem::TRANSLATED, CoordSystem::LOGICAL);
    return {logical_core.x, logical_core.y};
}

std::unordered_map<int, int> Cluster::get_worker_logical_to_virtual_x(ChipId chip_id) const {
    std::unordered_map<int, int> worker_logical_to_virtual_x;
    const auto& soc_desc = this->get_soc_desc(chip_id);
    for (const tt::umd::CoreCoord& logical_core : soc_desc.get_cores(CoreType::TENSIX, CoordSystem::LOGICAL)) {
        tt::umd::CoreCoord translated_core = soc_desc.translate_coord_to(logical_core, CoordSystem::TRANSLATED);
        worker_logical_to_virtual_x[logical_core.x] = translated_core.x;
    }
    return worker_logical_to_virtual_x;
}

std::unordered_map<int, int> Cluster::get_worker_logical_to_virtual_y(ChipId chip_id) const {
    std::unordered_map<int, int> worker_logical_to_virtual_y;
    const auto& soc_desc = this->get_soc_desc(chip_id);
    for (const tt::umd::CoreCoord& logical_core : soc_desc.get_cores(CoreType::TENSIX, CoordSystem::LOGICAL)) {
        tt::umd::CoreCoord translated_core = soc_desc.translate_coord_to(logical_core, CoordSystem::TRANSLATED);
        worker_logical_to_virtual_y[logical_core.y] = translated_core.y;
    }
    return worker_logical_to_virtual_y;
}

int Cluster::get_device_aiclk(const ChipId& chip_id) const { return this->driver_->get_chip(chip_id)->get_clock(); }

uint16_t Cluster::get_bus_id(ChipId chip) const { return this->cluster_desc_->get_bus_id(chip); }

std::optional<int> Cluster::get_physical_slot(ChipId chip) const {
    if (this->target_type_ != tt::TargetDevice::Silicon) {
        log_warning(tt::LogDevice, "get_physical_slot is not supported for non-silicon devices");
        return std::nullopt;
    }
    return this->driver_->get_chip(chip)->get_tt_device()->get_pci_device()->get_device_info().physical_slot;
}

void Cluster::deassert_risc_reset_at_core(
    const tt_cxy_pair& core, const tt::umd::RiscType& soft_resets, bool staggered_start) const {
    const metal_SocDescriptor& soc_desc = this->get_soc_desc(core.chip);
    tt::umd::CoreCoord core_coord = soc_desc.get_coord_at(core, CoordSystem::TRANSLATED);
    this->driver_->deassert_risc_reset(core.chip, core_coord, soft_resets, staggered_start);
}

void Cluster::assert_risc_reset_at_core(const tt_cxy_pair& core, const tt::umd::RiscType& soft_resets) const {
    const metal_SocDescriptor& soc_desc = this->get_soc_desc(core.chip);
    tt::umd::CoreCoord core_coord = soc_desc.get_coord_at(core, CoordSystem::TRANSLATED);
    this->driver_->assert_risc_reset(core.chip, core_coord, soft_resets);
}

void Cluster::write_dram_vec(
    const void* mem_ptr, uint32_t sz_in_bytes, ChipId device_id, int dram_view, uint64_t addr) const {
    const metal_SocDescriptor& desc_to_use = get_soc_desc(device_id);
    TT_FATAL(
        dram_view < desc_to_use.get_num_dram_views(),
        "Bounds-Error -- dram_view={} is outside of num_dram_views={}",
        dram_view,
        desc_to_use.get_num_dram_views());

    CoreCoord dram_core_coord = desc_to_use.get_preferred_worker_core_for_dram_view(dram_view, tt_metal::NOC::NOC_0);
    tt_cxy_pair dram_core = tt_cxy_pair(device_id, dram_core_coord.x, dram_core_coord.y);
    size_t offset = desc_to_use.get_address_offset(dram_view);
    write_core(mem_ptr, sz_in_bytes, tt_cxy_pair(device_id, dram_core.x, dram_core.y), addr + offset);
}

void Cluster::read_dram_vec(void* mem_ptr, uint32_t sz_in_bytes, ChipId device_id, int dram_view, uint64_t addr) const {
    const metal_SocDescriptor& desc_to_use = get_soc_desc(device_id);
    TT_FATAL(
        dram_view < desc_to_use.get_num_dram_views(),
        "Bounds-Error -- dram_view={} is outside of num_dram_views={}",
        dram_view,
        desc_to_use.get_num_dram_views());

    CoreCoord dram_core_coord = desc_to_use.get_preferred_worker_core_for_dram_view(dram_view, tt_metal::NOC::NOC_0);
    tt_cxy_pair dram_core = tt_cxy_pair(device_id, dram_core_coord.x, dram_core_coord.y);
    size_t offset = desc_to_use.get_address_offset(dram_view);
    read_core(mem_ptr, sz_in_bytes, tt_cxy_pair(device_id, dram_core.x, dram_core.y), addr + offset);
}

bool Cluster::supports_dma_operations(ChipId chip_id, uint32_t sz_in_bytes) const {
    if (this->rtoptions_.get_disable_dma_ops()) {
        return false;
    }

    // Currently, DMA reads/writes hang for small sizes. As a safety measure, we disable DMA for small sizes.
    // TODO: Remove this once we have a proper fix for small DMA sizes.
    constexpr uint32_t min_dma_size_bytes = 32;

    // DMA reads and writes are only supported on WH. If/when DMA reads and writes are supported on BH, this should be
    // updated to support BH architectures as well. See https://github.com/tenstorrent/tt-metal/issues/22957
    return this->arch_ == tt::ARCH::WORMHOLE_B0 && this->cluster_desc_->is_chip_mmio_capable(chip_id) &&
           sz_in_bytes >= min_dma_size_bytes;
}

void Cluster::write_core(const void* mem_ptr, uint32_t sz_in_bytes, tt_cxy_pair core, uint64_t addr) const {
    const ChipId chip_id = core.chip;
    const metal_SocDescriptor& soc_desc = this->get_soc_desc(chip_id);
    if (rtoptions_.get_watcher_enabled()) {
        tt::watcher_sanitize_host_noc_write(
            soc_desc,
            this->virtual_worker_cores_.at(chip_id),
            this->virtual_eth_cores_.at(chip_id),
            this->virtual_pcie_cores_.at(chip_id),
            this->virtual_dram_cores_.at(chip_id),
            {core.x, core.y},
            addr,
            sz_in_bytes);
    }
    tt::umd::CoreCoord core_coord = soc_desc.get_coord_at(core, CoordSystem::TRANSLATED);

    if (this->supports_dma_operations(chip_id, sz_in_bytes)) {
        this->driver_->dma_write_to_device(mem_ptr, sz_in_bytes, core.chip, core_coord, addr);
    } else {
        this->driver_->write_to_device(mem_ptr, sz_in_bytes, core.chip, core_coord, addr);
    }

    if (this->cluster_desc_->is_chip_remote(chip_id)) {
        this->driver_->wait_for_non_mmio_flush(chip_id);
    }
}

void Cluster::read_core(void* mem_ptr, uint32_t size_in_bytes, tt_cxy_pair core, uint64_t addr) const {
    const ChipId chip_id = core.chip;
    const metal_SocDescriptor& soc_desc = this->get_soc_desc(chip_id);

    if (rtoptions_.get_watcher_enabled()) {
        tt::watcher_sanitize_host_noc_read(
            soc_desc,
            this->virtual_worker_cores_.at(chip_id),
            this->virtual_eth_cores_.at(chip_id),
            this->virtual_pcie_cores_.at(chip_id),
            this->virtual_dram_cores_.at(chip_id),
            {core.x, core.y},
            addr,
            size_in_bytes);
    }
    tt::umd::CoreCoord core_coord = soc_desc.get_coord_at(core, CoordSystem::TRANSLATED);

    if (this->supports_dma_operations(chip_id, size_in_bytes)) {
        this->driver_->dma_read_from_device(mem_ptr, size_in_bytes, core.chip, core_coord, addr);
    } else {
        this->driver_->read_from_device(mem_ptr, core.chip, core_coord, addr, size_in_bytes);
    }
}

void Cluster::write_core_immediate(const void* mem_ptr, uint32_t sz_in_bytes, tt_cxy_pair core, uint64_t addr) const {
    const ChipId chip_id = core.chip;
    const metal_SocDescriptor& soc_desc = this->get_soc_desc(chip_id);

    if (rtoptions_.get_watcher_enabled()) {
        tt::watcher_sanitize_host_noc_write(
            soc_desc,
            this->virtual_worker_cores_.at(chip_id),
            this->virtual_eth_cores_.at(chip_id),
            this->virtual_pcie_cores_.at(chip_id),
            this->virtual_dram_cores_.at(chip_id),
            {core.x, core.y},
            addr,
            sz_in_bytes);
    }

    tt::umd::CoreCoord core_coord = soc_desc.get_coord_at(core, CoordSystem::TRANSLATED);
    this->driver_->write_to_device_reg(mem_ptr, sz_in_bytes, core.chip, core_coord, addr);

    if (this->cluster_desc_->is_chip_remote(chip_id)) {
        this->driver_->wait_for_non_mmio_flush(chip_id);
    }
}

void Cluster::read_core(std::vector<uint32_t>& data, uint32_t size_in_bytes, tt_cxy_pair core, uint64_t addr) const {
    data.resize(size_in_bytes / sizeof(uint32_t));
    read_core(data.data(), size_in_bytes, core, addr);
}

void Cluster::write_reg(const std::uint32_t* mem_ptr, tt_cxy_pair target, uint64_t addr) const {
    const unsigned int size_in_bytes = sizeof(uint32_t);
    int chip_id = target.chip;
    const metal_SocDescriptor& soc_desc = this->get_soc_desc(chip_id);

    if (rtoptions_.get_watcher_enabled()) {
        tt::watcher_sanitize_host_noc_write(
            soc_desc,
            this->virtual_worker_cores_.at(chip_id),
            this->virtual_eth_cores_.at(chip_id),
            this->virtual_pcie_cores_.at(chip_id),
            this->virtual_dram_cores_.at(chip_id),
            {target.x, target.y},
            addr,
            size_in_bytes);
    }
    tt::umd::CoreCoord target_coord = soc_desc.get_coord_at(target, CoordSystem::TRANSLATED);
    this->driver_->write_to_device_reg(mem_ptr, size_in_bytes, target.chip, target_coord, addr);
    if (this->cluster_desc_->is_chip_remote(chip_id)) {
        this->driver_->wait_for_non_mmio_flush(chip_id);
    }
}

void Cluster::read_reg(std::uint32_t* mem_ptr, tt_cxy_pair target, uint64_t addr) const {
    const unsigned int size_in_bytes = sizeof(uint32_t);
    int chip_id = target.chip;
    const metal_SocDescriptor& soc_desc = this->get_soc_desc(chip_id);

    if (rtoptions_.get_watcher_enabled()) {
        tt::watcher_sanitize_host_noc_read(
            soc_desc,
            this->virtual_worker_cores_.at(chip_id),
            this->virtual_eth_cores_.at(chip_id),
            this->virtual_pcie_cores_.at(chip_id),
            this->virtual_dram_cores_.at(chip_id),
            {target.x, target.y},
            addr,
            size_in_bytes);
    }
    tt::umd::CoreCoord target_coord = soc_desc.get_coord_at(target, CoordSystem::TRANSLATED);
    this->driver_->read_from_device_reg(mem_ptr, target.chip, target_coord, addr, size_in_bytes);
}

void Cluster::noc_multicast_write(
    const void* mem_ptr, uint32_t sz_in_bytes, tt_cxy_pair core_start, tt_cxy_pair core_end, uint64_t addr) const {
    TT_FATAL(core_start.chip == core_end.chip, "core_start and core_end must be on the same chip");
    noc_multicast_write(
        mem_ptr,
        sz_in_bytes,
        core_start.chip,
        tt_xy_pair(core_start.x, core_start.y),
        tt_xy_pair(core_end.x, core_end.y),
        addr);
}

void Cluster::noc_multicast_write(
    const void* mem_ptr, uint32_t sz_in_bytes, ChipId chip_id, CoreCoord core_start, CoreCoord core_end, uint64_t addr)
    const {
    const metal_SocDescriptor& soc_desc = this->get_soc_desc(chip_id);

    if (rtoptions_.get_watcher_enabled()) {
        tt::watcher_sanitize_host_noc_multicast_write(
            soc_desc,
            this->virtual_worker_cores_.at(chip_id),
            {core_start.x, core_start.y},
            {core_end.x, core_end.y},
            addr,
            sz_in_bytes);
    }

    tt::umd::CoreCoord start_coord = soc_desc.get_coord_at(core_start, CoordSystem::TRANSLATED);
    tt::umd::CoreCoord end_coord = soc_desc.get_coord_at(core_end, CoordSystem::TRANSLATED);

    this->driver_->noc_multicast_write(const_cast<void*>(mem_ptr), sz_in_bytes, chip_id, start_coord, end_coord, addr);

    if (this->cluster_desc_->is_chip_remote(chip_id)) {
        this->driver_->wait_for_non_mmio_flush(chip_id);
    }
}

void Cluster::write_sysmem(
    const void* vec, uint32_t size_in_bytes, uint64_t addr, ChipId src_device_id, uint16_t channel) const {
    TT_ASSERT(this->cluster_desc_->is_chip_mmio_capable(src_device_id));
    this->driver_->write_to_sysmem(vec, size_in_bytes, addr, channel & HOST_MEM_CHANNELS_MASK, src_device_id);
}

void Cluster::read_sysmem(
    void* vec, uint32_t size_in_bytes, uint64_t addr, ChipId src_device_id, uint16_t channel) const {
    TT_ASSERT(this->cluster_desc_->is_chip_mmio_capable(src_device_id));
    this->driver_->read_from_sysmem(vec, addr, channel & HOST_MEM_CHANNELS_MASK, size_in_bytes, src_device_id);
}

std::unique_ptr<tt::umd::SysmemBuffer> Cluster::allocate_sysmem_buffer(
    ChipId device_id, size_t sysmem_buffer_size, bool map_to_noc) const {
    tt::umd::SysmemManager* sysmem_manager = this->driver_->get_chip(device_id)->get_sysmem_manager();
    if (!sysmem_manager) {
        TT_THROW("Failed to get SysmemManager for device {}", device_id);
    }
    return sysmem_manager->allocate_sysmem_buffer(sysmem_buffer_size, map_to_noc);
}

std::unique_ptr<tt::umd::SysmemBuffer> Cluster::map_sysmem_buffer(
    ChipId device_id, void* buffer, size_t sysmem_buffer_size, bool map_to_noc) const {
    tt::umd::SysmemManager* sysmem_manager = this->driver_->get_chip(device_id)->get_sysmem_manager();
    if (!sysmem_manager) {
        TT_THROW("Failed to get SysmemManager for device {}", device_id);
    }
    return sysmem_manager->map_sysmem_buffer(buffer, sysmem_buffer_size, map_to_noc);
}

void Cluster::verify_sw_fw_versions(
    int device_id, std::uint32_t sw_version, std::vector<std::uint32_t>& fw_versions) const {
    umd::semver_t sw(umd::semver_t::from_eth_fw_tag(sw_version)),
        fw_first_eth_core(umd::semver_t::from_eth_fw_tag(fw_versions.at(0)));
    log_info(
        tt::LogDevice,
        "Software version {}, Ethernet FW version {} (Device {})",
        sw.to_string(),
        fw_first_eth_core.to_string(),
        device_id);
    for (std::uint32_t& fw_version : fw_versions) {
        umd::semver_t fw(umd::semver_t::from_eth_fw_tag(fw_version));

        TT_FATAL(fw == fw_first_eth_core, "FW versions are not the same across different ethernet cores");
        TT_FATAL(sw.major == fw.major, "SW/FW major version number out of sync");
        TT_FATAL(sw.minor <= fw.minor, "SW version is newer than FW version");
    }
}

bool Cluster::verify_eth_fw_capability() const {
    // get_ethernet_fw_version is not supported in the simulation environment. assume it's correct!
    if (rtoptions_.get_simulator_enabled()) {
        return true;
    }
    const auto fw_version = this->driver_->get_ethernet_firmware_version();
    if (fw_version) {
        return hal_.verify_eth_fw_version(fw_version.value());
    }
    return true;
}

// DRAM barrier is used to implement host-to-device synchronization and should be used when all previous writes to DRAM
// need to be flushed This is needed because writes to device are not blocking unless strict TLB ordering is used
// (default ordering is posted) This barrier is intended to prevent races caused by out of order writes, specifically to
// ensure metadata and data to compute on are committed before launching kernels
void Cluster::dram_barrier(ChipId chip_id) const {
    std::unordered_set<uint32_t> dram_channels;
    for (uint32_t channel = 0; channel < this->get_soc_desc(chip_id).get_num_dram_channels(); channel++) {
        dram_channels.insert(channel);
    }
    this->driver_->dram_membar(chip_id, dram_channels);
}

// L1 barrier is used to implement host-to-device synchronization and should be used when all previous writes to L1 need
// to be flushed This is needed because writes to device are not blocking unless strict TLB ordering is used (default
// ordering is posted) This barrier is intended to prevent races caused by out of order writes, specifically to ensure
// binaries, metadata, and data to compute on are committed before launching kernels
void Cluster::l1_barrier(ChipId chip_id) const {
    // Sets and resets L1 barrier of all tensix cores and ethernet cores
    this->driver_->l1_membar(chip_id);
}

uint32_t Cluster::get_num_host_channels(ChipId device_id) const {
    bool mmio_capable = this->cluster_desc_->is_chip_mmio_capable(device_id);
    return mmio_capable ? this->driver_->get_num_host_channels(device_id) : 0;
}

uint32_t Cluster::get_host_channel_size(ChipId device_id, uint32_t channel) const {
    TT_ASSERT(this->cluster_desc_->is_chip_mmio_capable(device_id));
    return this->driver_->get_host_channel_size(device_id, channel & HOST_MEM_CHANNELS_MASK);
}

void* Cluster::host_dma_address(uint64_t offset, ChipId src_device_id, uint16_t channel) const {
    TT_ASSERT(this->cluster_desc_->is_chip_mmio_capable(src_device_id));
    return this->driver_->host_dma_address(offset, src_device_id, channel & HOST_MEM_CHANNELS_MASK);
}

uint64_t Cluster::get_pcie_base_addr_from_device(ChipId chip_id) const {
    return this->driver_->get_pcie_base_addr_from_device(chip_id);
}

const std::unordered_set<ChipId>& Cluster::get_devices_controlled_by_mmio_device(ChipId mmio_device_id) const {
    TT_FATAL(driver_, "UMD cluster object must be initialized and available");
    return llrt::get_devices_controlled_by_mmio_device(*driver_, mmio_device_id);
}

std::unordered_map<ChipId, std::vector<CoreCoord>> Cluster::get_ethernet_cores_grouped_by_connected_chips(
    ChipId chip_id) const {
    std::unordered_map<ChipId, std::vector<CoreCoord>> connected_chips;
    const auto& all_eth_connections = this->cluster_desc_->get_ethernet_connections();
    if (!all_eth_connections.contains(chip_id)) {
        return {};
    }
    for (const auto& [eth_chan, connected_chip_chan] : all_eth_connections.at(chip_id)) {
        const auto& other_chip_id = std::get<0>(connected_chip_chan);
        if (!connected_chips.contains(other_chip_id)) {
            std::vector<CoreCoord> active_ethernet_cores;

            for (const auto& channel_pair :
                 this->cluster_desc_->get_directly_connected_ethernet_channels_between_chips(chip_id, other_chip_id)) {
                EthernetChannel local_chip_chan = std::get<0>(channel_pair);
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

// Ethernet cluster api
void Cluster::initialize_ethernet_sockets() {
    for (const auto& chip_id : this->driver_->get_target_device_ids()) {
        if (!this->ethernet_sockets_.contains(chip_id)) {
            this->ethernet_sockets_.insert({chip_id, {}});
        }
        for (const auto& [connected_chip_id, eth_cores] :
             this->get_ethernet_cores_grouped_by_connected_chips(chip_id)) {
            if (!this->ethernet_sockets_.at(chip_id).contains(connected_chip_id)) {
                this->ethernet_sockets_.at(chip_id).insert({connected_chip_id, {}});
            }
            if (!this->ethernet_sockets_.contains(connected_chip_id)) {
                this->ethernet_sockets_.insert({connected_chip_id, {}});
            }
            if (!this->ethernet_sockets_.at(connected_chip_id).contains(chip_id)) {
                this->ethernet_sockets_.at(connected_chip_id).insert({chip_id, {}});
            } else {
                continue;
            }
            for (const auto& eth_core : eth_cores) {
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
    const auto& chips = this->driver_->get_target_device_ids();
    for (const auto& chip_id : chips) {
        if (!this->frequent_retrain_cores_.contains(chip_id)) {
            this->frequent_retrain_cores_.insert({chip_id, {}});
        }
        const auto& connected_chips = this->get_ethernet_cores_grouped_by_connected_chips(chip_id);
        for (const auto& [other_chip_id, eth_cores] : connected_chips) {
            for (const auto& eth_core : eth_cores) {
                if (rtoptions_.get_skip_eth_cores_with_retrain() and
                    this->cluster_desc_->get_board_type(chip_id) == BoardType::UBB) {
                    tt_cxy_pair virtual_eth_core(
                        chip_id, get_virtual_coordinate_from_logical_coordinates(chip_id, eth_core, CoreType::ETH));
                    auto retrain_count_addr = hal_.get_dev_addr(
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

void Cluster::initialize_ethernet_cores_router_mode() {
    for (const auto& [assoc_mmio_device, devices] : this->cluster_desc_->get_chips_grouped_by_closest_mmio()) {
        for (const auto& chip_id : devices) {
            if (!this->device_eth_routing_info_.contains(chip_id)) {
                this->device_eth_routing_info_.insert({chip_id, {}});
            }
        }
        std::map<std::tuple<ChipId, ChipId>, bool> reserved_chip_connections = {};
        for (const auto& chip_id : devices) {
            // Mark all remaining ethernet cores as idle to be used by fabric
            const auto& soc_desc = get_soc_desc(chip_id);
            for (const auto& eth_channel : cluster_desc_->get_active_eth_channels(chip_id)) {
                auto eth_core = soc_desc.get_eth_core_for_channel(eth_channel, CoordSystem::LOGICAL);
                // Chip ID is guaranteed to be present in device_eth_routing_info_, since it was populated above
                auto& routing_info = this->device_eth_routing_info_[chip_id];
                if (!routing_info.contains(eth_core)) {
                    routing_info.insert({eth_core, EthRouterMode::IDLE});
                }
            }
        }
    }
}

std::unordered_set<ChipId> Cluster::get_ethernet_connected_device_ids(ChipId chip_id) const {
    std::unordered_set<ChipId> device_ids;
    const auto& connected_chips = this->get_ethernet_cores_grouped_by_connected_chips(chip_id);
    for (const auto& [other_chip_id, eth_cores] : connected_chips) {
        for ([[maybe_unused]] const auto& eth_core : eth_cores) {
            device_ids.insert(other_chip_id);
        }
    }
    return device_ids;
}

void Cluster::configure_ethernet_cores_for_fabric_routers(
    tt_fabric::FabricConfig fabric_config, std::optional<uint8_t> num_routing_planes) {
    if (fabric_config != tt_fabric::FabricConfig::DISABLED) {
        TT_FATAL(num_routing_planes.has_value(), "num_routing_planes should be set for reserving cores for fabric");
        TT_FATAL(num_routing_planes.value() > 0, "Expected non-zero num_routing_planes for reserving cores for fabric");
        this->reserve_ethernet_cores_for_fabric_routers(num_routing_planes.value());
    } else {
        if (num_routing_planes.has_value()) {
            log_warning(
                tt::LogMetal,
                "Got num_routing_planes while releasing fabric cores, ignoring it and releasing all reserved cores");
        }
        this->release_ethernet_cores_for_fabric_routers();
    }
}

void Cluster::reserve_ethernet_cores_for_fabric_routers(uint8_t num_routing_planes) {
    if (num_routing_planes == std::numeric_limits<uint8_t>::max()) {
        // default behavior, reserve whatever cores are available
        for (const auto& [chip_id, eth_cores] : this->device_eth_routing_info_) {
            for (const auto& [eth_core, mode] : eth_cores) {
                if (mode == EthRouterMode::IDLE) {
                    this->device_eth_routing_info_[chip_id][eth_core] = EthRouterMode::FABRIC_ROUTER;
                }
            }
        }

        // Update sockets to reflect fabric routing
        this->ethernet_sockets_.clear();
        return;
    }

    std::set<std::pair<ChipId, ChipId>> pairs_done;
    // to reserve specified number of cores, ensure that the same are avaialble on connected chip id as well
    for (const auto& chip_id : this->driver_->get_target_device_ids()) {
        const auto& connected_chips_and_cores = this->get_ethernet_cores_grouped_by_connected_chips(chip_id);
        for (const auto& [connected_chip_id, cores] : connected_chips_and_cores) {
            if (pairs_done.contains(std::make_pair(chip_id, connected_chip_id))) {
                // the cores for this pair of chips are already allocated, skip
                continue;
            }

            const uint8_t num_cores_to_reserve = std::min(num_routing_planes, static_cast<uint8_t>(cores.size()));
            uint8_t num_reserved_cores = 0;
            for (const auto eth_core : cores) {
                if (num_reserved_cores == num_cores_to_reserve) {
                    pairs_done.insert(std::make_pair(chip_id, connected_chip_id));
                    pairs_done.insert(std::make_pair(connected_chip_id, chip_id));
                    break;
                }

                // Last link reserved for dispatch
                // Only need fabric routers in the same tunnel
                // TODO: https://github.com/tenstorrent/tt-metal/issues/24413
                const auto is_mmio_device = [&](int id) { return cluster_desc_->is_chip_mmio_capable(id); };
                const auto is_last_link = [&]() { return num_reserved_cores == num_cores_to_reserve - 1; };
                if (is_last_link() && is_mmio_device(chip_id) && is_mmio_device(connected_chip_id)) {
                    num_reserved_cores++;
                    break;
                }

                const auto connected_core =
                    std::get<1>(this->get_connected_ethernet_core(std::make_tuple(chip_id, eth_core)));
                if (this->device_eth_routing_info_.at(chip_id).at(eth_core) == EthRouterMode::FABRIC_ROUTER) {
                    // already reserved for fabric, potenially by the connected chip id
                    num_reserved_cores++;
                    continue;
                }

                if (this->device_eth_routing_info_[chip_id][eth_core] == EthRouterMode::IDLE &&
                    this->device_eth_routing_info_.at(connected_chip_id).at(connected_core) == EthRouterMode::IDLE) {
                    this->device_eth_routing_info_[chip_id][eth_core] = EthRouterMode::FABRIC_ROUTER;
                    this->device_eth_routing_info_[connected_chip_id][connected_core] = EthRouterMode::FABRIC_ROUTER;
                    num_reserved_cores++;
                }
            }

            TT_FATAL(
                num_reserved_cores == num_cores_to_reserve,
                "Unable to reserve {} routing planes b/w chip {} and {} for fabric, reserved only {}",
                num_cores_to_reserve,
                chip_id,
                connected_chip_id,
                num_reserved_cores);
        }
    }

    // re-init sockets to reflect fabric routing
    this->ethernet_sockets_.clear();
    this->initialize_ethernet_sockets();
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

std::set<tt_fabric::chan_id_t> Cluster::get_fabric_ethernet_channels(ChipId chip_id) const {
    std::set<tt_fabric::chan_id_t> fabric_ethernet_channels;
    const auto& control_plane = tt::tt_metal::MetalContext::instance().get_control_plane();
    const auto& active_eth_cores = control_plane.get_active_ethernet_cores(chip_id, false);
    for (const auto& eth_core : active_eth_cores) {
        if (!this->is_ethernet_link_up(chip_id, eth_core)) {
            continue;
        }
        if (this->device_eth_routing_info_.at(chip_id).at(eth_core) == EthRouterMode::FABRIC_ROUTER) {
            fabric_ethernet_channels.insert(this->get_soc_desc(chip_id).logical_eth_core_to_chan_map.at(eth_core));
        }
    }
    return fabric_ethernet_channels;
}

std::vector<CoreCoord> Cluster::get_fabric_ethernet_routers_between_src_and_dest(ChipId src_id, ChipId dst_id) const {
    std::vector<CoreCoord> fabric_ethernet_channels;
    const auto& connected_chips = this->get_ethernet_cores_grouped_by_connected_chips(src_id);
    TT_FATAL(connected_chips.contains(dst_id), "Dst Chip {} is not connected to Src Chip {}", dst_id, src_id);
    for (const auto& eth_core : connected_chips.at(dst_id)) {
        if (this->device_eth_routing_info_.at(src_id).at(eth_core) == EthRouterMode::FABRIC_ROUTER) {
            fabric_ethernet_channels.push_back(eth_core);
        }
    }
    return fabric_ethernet_channels;
}

bool Cluster::is_ethernet_link_up(ChipId chip_id, const CoreCoord& logical_core) const {
    const auto& soc_desc = get_soc_desc(chip_id);
    EthernetChannel eth_chan = soc_desc.logical_eth_core_to_chan_map.at(logical_core);
    return this->cluster_desc_->ethernet_core_has_active_ethernet_link(chip_id, eth_chan);
}

std::tuple<ChipId, CoreCoord> Cluster::get_connected_ethernet_core(std::tuple<ChipId, CoreCoord> eth_core) const {
    const auto& soc_desc = get_soc_desc(std::get<0>(eth_core));
    EthernetChannel eth_chan = soc_desc.logical_eth_core_to_chan_map.at(std::get<1>(eth_core));
    TT_FATAL(
        this->is_ethernet_link_up(std::get<0>(eth_core), std::get<1>(eth_core)),
        "Logical eth core {} is not an active eth core on chip {}.",
        std::get<1>(eth_core).str(),
        std::get<0>(eth_core));
    const auto& ethernet_connections_within_cluster = this->get_ethernet_connections();
    TT_FATAL(
        ethernet_connections_within_cluster.contains(std::get<0>(eth_core)) and
            ethernet_connections_within_cluster.at(std::get<0>(eth_core)).contains(eth_chan),
        "Chip {} logical eth core {} connects to a remote mmio device",
        std::get<0>(eth_core),
        std::get<1>(eth_core).str());
    auto connected_eth_core =
        this->cluster_desc_->get_chip_and_channel_of_remote_ethernet_core(std::get<0>(eth_core), eth_chan);
    return std::make_tuple(
        std::get<0>(connected_eth_core),
        soc_desc.get_eth_core_for_channel(std::get<1>(connected_eth_core), CoordSystem::LOGICAL));
}

// TODO: unify uint64_t with ChipUID
std::tuple<uint64_t, CoreCoord> Cluster::get_connected_ethernet_core_to_remote_mmio_device(
    std::tuple<ChipId, CoreCoord> eth_core) const {
    const auto& soc_desc = get_soc_desc(std::get<0>(eth_core));
    EthernetChannel eth_chan = soc_desc.logical_eth_core_to_chan_map.at(std::get<1>(eth_core));
    TT_FATAL(
        this->is_ethernet_link_up(std::get<0>(eth_core), std::get<1>(eth_core)),
        "Logical eth core {} is not an active eth core on chip {}.",
        std::get<1>(eth_core).str(),
        std::get<0>(eth_core));
    const auto& ethernet_connections_to_remote_cluster = this->get_ethernet_connections_to_remote_devices();
    const auto& local_chip_id = std::get<0>(eth_core);
    const auto& local_eth_core = std::get<1>(eth_core);
    TT_FATAL(
        ethernet_connections_to_remote_cluster.contains(local_chip_id) and
            ethernet_connections_to_remote_cluster.at(local_chip_id).contains(eth_chan),
        "Chip {} logical eth core {} connects to a local mmio device",
        local_chip_id,
        local_eth_core.str());

    const auto& connected_eth_core = ethernet_connections_to_remote_cluster.at(local_chip_id).at(eth_chan);
    return std::make_tuple(
        std::get<0>(connected_eth_core),
        soc_desc.get_eth_core_for_channel(std::get<1>(connected_eth_core), CoordSystem::LOGICAL));
}

std::vector<CoreCoord> Cluster::get_ethernet_sockets(ChipId local_chip, ChipId remote_chip) const {
    const auto& local_ethernet_sockets = this->ethernet_sockets_.at(local_chip);
    TT_FATAL(
        local_ethernet_sockets.contains(remote_chip),
        "Device {} is not connected to Device {}",
        local_chip,
        remote_chip);
    return local_ethernet_sockets.at(remote_chip);
}

CoreCoord Cluster::ethernet_core_from_logical_core(ChipId chip_id, const CoreCoord& logical_core) const {
    const metal_SocDescriptor& soc_desc = get_soc_desc(chip_id);
    return soc_desc.get_physical_ethernet_core_from_logical(logical_core);
}

CoreCoord Cluster::get_virtual_eth_core_from_channel(ChipId chip_id, int channel) const {
    tt::umd::CoreCoord logical_coord =
        this->get_soc_desc(chip_id).get_eth_core_for_channel(channel, CoordSystem::LOGICAL);
    return this->get_virtual_coordinate_from_logical_coordinates(
        chip_id, {logical_coord.x, logical_coord.y}, CoreType::ETH);
}

// TODO: ALLAN Can change to write one bit
void Cluster::set_internal_routing_info_for_ethernet_cores(
    bool enable_internal_routing, const std::vector<ChipId>& target_mmio_devices) const {
    log_debug(tt::LogDevice, "Set internal routing bit {}", enable_internal_routing);
    // TODO: initialize devices if user does not
    // Must initialize remote chips first, then mmio chips since once mmio chips are doing fd routing
    // we do not always context switch to base FW
    std::vector<ChipId> non_mmio_devices;
    std::vector<ChipId> mmio_devices = target_mmio_devices;
    if (mmio_devices.empty()) {
        mmio_devices.reserve(this->number_of_pci_devices());
        for (auto chip_id : this->driver_->get_target_mmio_device_ids()) {
            mmio_devices.emplace_back(chip_id);
        }
    }
    for (auto chip_id : this->driver_->get_target_remote_device_ids()) {
        non_mmio_devices.emplace_back(chip_id);
    }
    const auto& control_plane = tt::tt_metal::MetalContext::instance().get_control_plane();
    auto dev_msgs_factory = hal_.get_dev_msgs_factory(tt_metal::HalProgrammableCoreType::ACTIVE_ETH);
    if (enable_internal_routing) {
        auto routing_info_enabled = dev_msgs_factory.create<tt_metal::dev_msgs::routing_info_t>();
        routing_info_enabled.view().routing_enabled() = 1;
        routing_info_enabled.view().src_sent_valid_cmd() = 0;
        routing_info_enabled.view().dst_acked_valid_cmd() = 0;
        for (const auto& chip_id : non_mmio_devices) {
            for (const auto& eth_core : control_plane.get_active_ethernet_cores(chip_id, false)) {
                tt_cxy_pair virtual_eth_core(
                    chip_id, get_virtual_coordinate_from_logical_coordinates(chip_id, eth_core, CoreType::ETH));
                // Enable internal ethernet routing for non-mmio devices
                write_core(
                    routing_info_enabled.data(), routing_info_enabled.size(), virtual_eth_core, routing_info_addr_);
            }
        }
        for (const auto& chip_id : mmio_devices) {
            for (const auto& eth_core : control_plane.get_active_ethernet_cores(chip_id, false)) {
                tt_cxy_pair virtual_eth_core(
                    chip_id, get_virtual_coordinate_from_logical_coordinates(chip_id, eth_core, CoreType::ETH));
                // Enable internal ethernet routing for mmio devices
                write_core(
                    routing_info_enabled.data(), routing_info_enabled.size(), virtual_eth_core, routing_info_addr_);
            }
        }
    } else {
        auto routing_info_disabled = dev_msgs_factory.create<tt_metal::dev_msgs::routing_info_t>();
        routing_info_disabled.view().routing_enabled() = 0;
        routing_info_disabled.view().src_sent_valid_cmd() = 0;
        routing_info_disabled.view().dst_acked_valid_cmd() = 0;
        for (const auto& chip_id : mmio_devices) {
            for (const auto& eth_core : control_plane.get_active_ethernet_cores(chip_id, false)) {
                tt_cxy_pair virtual_eth_core(
                    chip_id, get_virtual_coordinate_from_logical_coordinates(chip_id, eth_core, CoreType::ETH));
                // Disable internal ethernet routing for mmio devices
                write_core(
                    routing_info_disabled.data(), routing_info_disabled.size(), virtual_eth_core, routing_info_addr_);
            }
        }
        for (const auto& chip_id : non_mmio_devices) {
            for (const auto& eth_core : control_plane.get_active_ethernet_cores(chip_id, false)) {
                tt_cxy_pair virtual_eth_core(
                    chip_id, get_virtual_coordinate_from_logical_coordinates(chip_id, eth_core, CoreType::ETH));
                // Disable internal ethernet routing for non-mmio devices
                write_core(
                    routing_info_disabled.data(), routing_info_disabled.size(), virtual_eth_core, routing_info_addr_);
            }
        }
    }
}

std::uint32_t Cluster::get_ubb_asic_id(ChipId physical_chip_id) const {
    auto unique_chip_id = this->get_unique_chip_ids().at(physical_chip_id);
    return ((unique_chip_id >> 56) & 0xFF);
}

bool Cluster::is_external_cable(ChipId physical_chip_id, CoreCoord eth_core) const {
    auto chan_id = this->get_soc_desc(physical_chip_id).logical_eth_core_to_chan_map.at(eth_core);
    bool is_external_cable = false;
    auto board_type = this->get_board_type(physical_chip_id);
    if (board_type == BoardType::UBB) {
        auto ubb_asic_id = get_ubb_asic_id(physical_chip_id);
        if (ubb_asic_id == 1) {
            // UBB 1 has external cables on channels 0-7
            is_external_cable = (chan_id >= 0 and chan_id <= 7);
        } else if (ubb_asic_id >= 2 and ubb_asic_id <= 4) {
            // UBB 2 to 4 has external cables on channels 0-3
            is_external_cable = (chan_id >= 0 and chan_id <= 3);
        } else if (ubb_asic_id == 5) {
            // UBB 5 has external cables on channels 4-7
            is_external_cable = (chan_id >= 4 and chan_id <= 7);
        }
    } else if (board_type == BoardType::N300) {
        // N300 has external cables on channels 8-9 on MMIO chips and channels 0-1 on non-MMIO chips
        auto mmio_device_id = this->get_associated_mmio_device(physical_chip_id);
        if (mmio_device_id == physical_chip_id) {
            is_external_cable = (chan_id != 8 and chan_id != 9);
        } else {
            is_external_cable = (chan_id != 0 and chan_id != 1);
        }
    } else if (board_type == BoardType::P150) {
        is_external_cable = (4 <= chan_id && chan_id <= 11);
    } else if (board_type == BoardType::P300) {
        auto asic_loc = this->get_cluster_desc()->get_asic_location(physical_chip_id);
        if (asic_loc == 1) {
            is_external_cable = (chan_id == 2 || chan_id == 3 || chan_id == 4 || chan_id == 6);
        } else if (asic_loc == 0) {
            is_external_cable = (chan_id == 4 || chan_id == 5 || chan_id == 7 || chan_id == 9);
        }
    }
    return is_external_cable;
}

uint32_t Cluster::get_alignment_requirements(ChipId chip_id, uint32_t size_in_bytes) const {
    if (this->supports_dma_operations(chip_id, size_in_bytes)) {
        return this->hal_.get_dma_alignment();
    }
    return 1;
}

}  // namespace tt

std::ostream& operator<<(std::ostream& os, const tt_target_dram& dram) {
    os << "Target DRAM chip = " << std::get<0>(dram) << ", chan = " << std::get<1>(dram);
    return os;
}
