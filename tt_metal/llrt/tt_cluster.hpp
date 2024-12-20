// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <chrono>
#include <functional>

#include "common/base.hpp"
#include "common/metal_soc_descriptor.h"
#include "common/test_common.hpp"
#include "common/tt_backend_api_types.hpp"
#include "umd/device/device_api_metal.h"
#include "umd/device/tt_cluster_descriptor.h"
#include "umd/device/tt_xy_pair.h"

#include "dev_msgs.h"

#include "llrt/hal.hpp"

static constexpr std::uint32_t SW_VERSION = 0x00020000;

using tt_target_dram = std::tuple<int, int, int>;

enum EthRouterMode : uint32_t {
    IDLE = 0,
    BI_DIR_TUNNELING = 1,
};

namespace tt {

class Cluster {
   public:
    Cluster &operator=(const Cluster &) = delete;
    Cluster &operator=(Cluster &&other) noexcept = delete;
    Cluster(const Cluster &) = delete;
    Cluster(Cluster &&other) noexcept = delete;

    static const Cluster &instance();

    // For TG Galaxy systems, mmio chips are gateway chips that are only used for dispatc, so user_devices are meant for
    // user facing host apis
    size_t number_of_user_devices() const {
        if (this->is_tg_cluster_) {
            const auto &chips = this->cluster_desc_->get_all_chips();
            return std::count_if(chips.begin(), chips.end(), [&](const auto &id) {
                return this->cluster_desc_->get_board_type(id) == BoardType::GALAXY;
            });
        } else {
            return this->cluster_desc_->get_number_of_chips();
        }
    }

    std::unordered_map<chip_id_t, eth_coord_t> get_user_chip_ethernet_coordinates() const;

    size_t number_of_devices() const { return this->cluster_desc_->get_number_of_chips(); }

    size_t number_of_pci_devices() const { return this->cluster_desc_->get_chips_with_mmio().size(); }

    ARCH arch() const { return this->arch_; }

    const metal_SocDescriptor &get_soc_desc(chip_id_t chip) const;
    CoreCoord get_virtual_coordinate_from_logical_coordinates(chip_id_t chip_id, CoreCoord logical_coord, const CoreType& core_type) const;
    CoreCoord get_virtual_coordinate_from_physical_coordinates(chip_id_t chip_id, CoreCoord physical_coord, const CoreType& core_type) const;
    tt_cxy_pair get_virtual_coordinate_from_logical_coordinates(tt_cxy_pair logical_coordinate, const CoreType& core_type) const;
    const std::unordered_set<CoreCoord>& get_virtual_worker_cores(chip_id_t chip_id) const;
    const std::unordered_set<CoreCoord>& get_virtual_eth_cores(chip_id_t chip_id) const;

    uint32_t get_harvested_rows(chip_id_t chip) const;
    uint32_t get_harvesting_mask(chip_id_t chip) const {
        return this->driver_->get_harvesting_masks_for_soc_descriptors().at(chip);
    }

    //! device driver and misc apis
    void verify_sw_fw_versions(int device_id, std::uint32_t sw_version, std::vector<std::uint32_t> &fw_versions) const;

    void deassert_risc_reset_at_core(const tt_cxy_pair &physical_chip_coord) const;
    void assert_risc_reset_at_core(const tt_cxy_pair &physical_chip_coord) const;

    void write_dram_vec(std::vector<uint32_t> &vec, tt_target_dram dram, uint64_t addr, bool small_access = false) const;
    void read_dram_vec(
        std::vector<uint32_t> &vec,
        uint32_t size_in_bytes,
        tt_target_dram dram,
        uint64_t addr,
        bool small_access = false) const;

    // Accepts physical noc coordinates
    void write_core(
        const void *mem_ptr, uint32_t sz_in_bytes, tt_cxy_pair core, uint64_t addr, bool small_access = false) const;
    void read_core(
        void *mem_ptr, uint32_t sz_in_bytes, tt_cxy_pair core, uint64_t addr, bool small_access = false) const;
    void read_core(
        std::vector<uint32_t> &data, uint32_t sz_in_bytes, tt_cxy_pair core, uint64_t addr, bool small_access = false) const;

    std::optional<std::tuple<uint32_t, uint32_t>> get_tlb_data(const tt_cxy_pair &target) const {
        tt::umd::Cluster *device = dynamic_cast<tt::umd::Cluster *>(driver_.get());
        tt_cxy_pair umd_target = this->virtual_to_umd_coord_mapping_.at(target);
        return device->get_tlb_data_from_target(umd_target);
    }

    std::function<void(uint32_t, uint32_t, const uint8_t *)> get_fast_pcie_static_tlb_write_callable(
        int chip_id) const {
        chip_id_t mmio_device_id = device_to_mmio_device_.at(chip_id);
        tt::umd::Cluster *device = dynamic_cast<tt::umd::Cluster *>(driver_.get());
        return device->get_fast_pcie_static_tlb_write_callable(mmio_device_id);
    }

    // Returns a writer object which holds a pointer to a static tlb
    // Allows for fast writes when targeting same device core by only doing the lookup once and avoiding repeated stack traversals
    tt::Writer get_static_tlb_writer(tt_cxy_pair target) const {
        tt::umd::Cluster *device = dynamic_cast<tt::umd::Cluster *>(driver_.get());
        tt_cxy_pair umd_target = this->virtual_to_umd_coord_mapping_.at(target);
        return device->get_static_tlb_writer(umd_target);
    }

    std::uint32_t get_numa_node_for_device(uint32_t device_id) const {
        uint32_t mmio_device_id = this->get_associated_mmio_device(device_id);
        tt::umd::Cluster *device = dynamic_cast<tt::umd::Cluster *>(driver_.get());
        return driver_->get_numa_node_for_pcie_device(mmio_device_id);
    }

    void write_reg(const std::uint32_t *mem_ptr, tt_cxy_pair target, uint64_t addr) const;
    void read_reg(std::uint32_t *mem_ptr, tt_cxy_pair target, uint64_t addr) const;

    void write_sysmem(
        const void *mem_ptr, uint32_t size_in_bytes, uint64_t addr, chip_id_t src_device_id, uint16_t channel) const;
    void read_sysmem(
        void *mem_ptr, uint32_t size_in_bytes, uint64_t addr, chip_id_t src_device_id, uint16_t channel) const;

    int get_device_aiclk(const chip_id_t &chip_id) const;

    void dram_barrier(chip_id_t chip_id) const;
    void l1_barrier(chip_id_t chip_id) const;

    uint32_t get_num_host_channels(chip_id_t device_id) const;
    uint32_t get_host_channel_size(chip_id_t device_id, uint32_t channel) const;
    // Returns address in host space
    void *host_dma_address(uint64_t offset, chip_id_t src_device_id, uint16_t channel) const;
    uint64_t get_pcie_base_addr_from_device(chip_id_t chip_id) const;

    // Ethernet cluster api
    // Returns set of device ids connected via ethernet
    std::unordered_set<chip_id_t> get_ethernet_connected_device_ids(chip_id_t chip_id) const;

    // Returns set of logical active ethernet coordinates on chip
    // If skip_reserved_tunnel_cores is true, will return cores that dispatch is not using,
    // intended for users to grab available eth cores for testing
    std::unordered_set<CoreCoord> get_active_ethernet_cores(
        chip_id_t chip_id, bool skip_reserved_tunnel_cores = false) const;

    // Returns set of logical inactive ethernet coordinates on chip
    std::unordered_set<CoreCoord> get_inactive_ethernet_cores(chip_id_t chip_id) const;

    // Returns connected ethernet core on the other chip
    std::tuple<chip_id_t, CoreCoord> get_connected_ethernet_core(std::tuple<chip_id_t, CoreCoord> eth_core) const;

    // Returns a ethernet sockets between local chip and remote chip
    // get_ethernet_sockets(a, b)[0] is connected to get_ethernet_sockets(b, a)[0]
    std::vector<CoreCoord> get_ethernet_sockets(chip_id_t local_chip, chip_id_t remote_chip) const;
    // Converts logical ethernet core coord to physical ethernet core coord
    CoreCoord ethernet_core_from_logical_core(chip_id_t chip_id, const CoreCoord &logical_core) const;

    // Bookkeeping for mmio device tunnels
    uint32_t get_mmio_device_max_tunnel_depth(chip_id_t mmio_device) const;
    uint32_t get_mmio_device_tunnel_count(chip_id_t mmio_device) const;
    uint32_t get_device_tunnel_depth(chip_id_t chip_id) const;

    // Dispatch core is managed by device, so this is an api for device to get the each eth core used in FD tunneling.
    // Returns logical eth core that communicates with specified dispatch core
    tt_cxy_pair get_eth_core_for_dispatch_core(
        tt_cxy_pair logical_dispatch_core, EthRouterMode mode, chip_id_t connected_chip_id) const;

    std::tuple<tt_cxy_pair, tt_cxy_pair> get_eth_tunnel_core(chip_id_t upstream_chip_id, chip_id_t downstream_chip_id, EthRouterMode mode) const;

    // Internal routing for SD and FD enables launching user ethernet kernels and FD tunneling for all devices in the
    // cluster. When using multiple devices in a cluster, this should be the flow:
    //       CreateDevice(0)
    //       CreateDevice(1)
    //       set_internal_routing_info_for_ethernet_cores(true);
    //       set_internal_routing_info_for_ethernet_cores(false);
    //       CloseDevice(0)
    //       CloseDevice(1)
    void set_internal_routing_info_for_ethernet_cores(bool enable_internal_routing, const std::vector<chip_id_t>& target_mmio_devices = {}) const;

    // Returns MMIO device ID (logical) that controls given `device_id`. If `device_id` is MMIO device it is returned.
    chip_id_t get_associated_mmio_device(chip_id_t device_id) const {
        return this->device_to_mmio_device_.at(device_id);
    }

    uint16_t get_assigned_channel_for_device(chip_id_t device_id) const {
        return this->device_to_host_mem_channel_.at(device_id);
    }

    // Returns collection of devices that are controlled by the specified MMIO device inclusive of the MMIO device
    const std::set<chip_id_t> &get_devices_controlled_by_mmio_device(chip_id_t mmio_device_id) const {
        TT_ASSERT(
            this->devices_grouped_by_assoc_mmio_device_.count(mmio_device_id),
            "Expected device {} to be an MMIO device!",
            mmio_device_id);
        return this->devices_grouped_by_assoc_mmio_device_.at(mmio_device_id);
    }

    // Returns vector of unique tunnels originating from mmio device.
    // Each vector entry is another vector of remote devices on that tunnel.
    std::vector<std::vector<chip_id_t>> get_tunnels_from_mmio_device(chip_id_t mmio_chip_id) const {
        return this->tunnels_from_mmio_device.at(mmio_chip_id);
    }

    // Returns whether we are running on Galaxy.
    bool is_galaxy_cluster() const;

    // Returns Wormhole chip board type.
    BoardType get_board_type(chip_id_t chip_id) const;

    bool is_worker_core(const CoreCoord &core, chip_id_t chip_id) const;
    bool is_ethernet_core(const CoreCoord &core, chip_id_t chip_id) const;
    CoreCoord get_logical_ethernet_core_from_virtual(chip_id_t chip, CoreCoord core) const;
    const std::unordered_map<int, int>& get_worker_logical_to_virtual_x(chip_id_t chip_id) const { return this->worker_logical_to_virtual_x_.at(this->get_board_type(chip_id)); };
    const std::unordered_map<int, int>& get_worker_logical_to_virtual_y(chip_id_t chip_id) const { return this->worker_logical_to_virtual_y_.at(this->get_board_type(chip_id)); };
    const std::unordered_map<CoreCoord, int32_t>& get_virtual_routing_to_profiler_flat_id(chip_id_t chip_id) const;
   private:
    Cluster();
    ~Cluster();

    void detect_arch_and_target();
    void generate_cluster_descriptor();
    void initialize_device_drivers();
    void assert_risc_reset();
    void assign_mem_channels_to_devices(chip_id_t mmio_device_id, const std::set<chip_id_t> &controlled_device_ids);
    void open_driver(
        const bool &skip_driver_allocs = false);
    void start_driver(tt_device_params &device_params) const;

    void get_metal_desc_from_tt_desc(
        const std::unordered_map<chip_id_t, tt_SocDescriptor> &input,
        const std::unordered_map<chip_id_t, uint32_t> &per_chip_id_harvesting_masks);
    void generate_virtual_to_umd_coord_mapping();
    void generate_logical_to_virtual_coord_mapping();
    void generate_virtual_to_profiler_flat_id_mapping();

    // Reserves ethernet cores in cluster for tunneling
    void reserve_ethernet_cores_for_tunneling();
    // Returns map of connected chip ids to active ethernet cores
    std::unordered_map<chip_id_t, std::vector<CoreCoord>> get_ethernet_cores_grouped_by_connected_chips(
        chip_id_t chip_id) const;
    void initialize_ethernet_sockets();

    // Set tunnels from mmio
    void set_tunnels_from_mmio_device();

    ARCH arch_;
    TargetDevice target_type_;

    // There is a single device driver for all connected chips. It might contain multiple MMIO devices/cards.
    std::unique_ptr<tt_device> driver_;

    // Need to hold reference to cluster descriptor to detect total number of devices available in cluster
    // UMD static APIs `detect_available_device_ids` and `detect_number_of_chips` only returns number of MMIO mapped
    // devices
    std::unique_ptr<tt_ClusterDescriptor> cluster_desc_;
    // There is an entry for every device that can be targeted (MMIO and remote)
    std::unordered_map<chip_id_t, metal_SocDescriptor> sdesc_per_chip_;

    // Collections of devices that are grouped based on the associated MMIO device. MMIO device is included in the
    // grouping
    std::unordered_map<chip_id_t, std::set<chip_id_t>> devices_grouped_by_assoc_mmio_device_;
    // Save mapping of device id to associated MMIO device id for fast lookup
    std::unordered_map<chip_id_t, chip_id_t> device_to_mmio_device_;
    // Data Structures Tracking Virtual Coordinates
    std::unordered_map<tt_cxy_pair, tt_cxy_pair> virtual_to_umd_coord_mapping_;
    std::unordered_map<chip_id_t, std::unordered_set<CoreCoord>> virtual_worker_cores_;
    std::unordered_map<chip_id_t, std::unordered_set<CoreCoord>> virtual_eth_cores_;
    std::unordered_map<BoardType, std::unordered_map<int, int>> worker_logical_to_virtual_x_;
    std::unordered_map<BoardType, std::unordered_map<int, int>> worker_logical_to_virtual_y_;
    std::unordered_map<BoardType, std::unordered_map<CoreCoord, CoreCoord>> eth_logical_to_virtual_;
    std::unordered_map<BoardType, std::unordered_map<CoreCoord, int32_t>> virtual_routing_to_profiler_flat_id_;
    // Flag to tell whether we are on a TG type of system.
    // If any device has to board type of GALAXY, we are on a TG cluster.
    bool is_tg_cluster_;

    // Tunnels setup in cluster
    std::map<chip_id_t, std::vector<std::vector<chip_id_t>>> tunnels_from_mmio_device = {};

    // Currently, each device is mapped to its own channel in host memory to enable fast dispatch
    // Channels are unique within a group of devices all controlled by a particular MMIO device
    // For example:
    //      Two N300 cards where MMIO device IDs are 0, 1 and R chips are 2, 3
    //      0 L controls 2 R and 1 L controls 3 R then, device_to_host_mem_channel_:
    //          0 -> 0
    //          2 -> 1
    //          1 -> 0
    //          3 -> 1
    std::unordered_map<chip_id_t, uint16_t> device_to_host_mem_channel_;

    // Mapping of each devices' ethernet routing mode
    std::unordered_map<chip_id_t, std::unordered_map<CoreCoord, EthRouterMode>> device_eth_routing_info_;

    std::unordered_map<chip_id_t, std::unordered_map<chip_id_t, std::vector<CoreCoord>>> ethernet_sockets_;
};

}  // namespace tt

std::ostream &operator<<(std::ostream &os, tt_target_dram const &dram);
