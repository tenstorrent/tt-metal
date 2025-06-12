// SPDX-FileCopyrightText: © 2023 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <tt-metalium/fabric_host_interface.h>
#include <tt-metalium/fabric_types.hpp>
#include <tt-metalium/metal_soc_descriptor.h>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <map>
#include <memory>
#include <optional>
#include <ostream>
#include <set>
#include <tuple>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "assert.hpp"
#include "core_coord.hpp"
#include <umd/device/cluster.h>
#include <umd/device/device_api_metal.h>
#include <umd/device/tt_cluster_descriptor.h>
#include <umd/device/tt_core_coordinates.h>
#include <umd/device/tt_io.hpp>
#include <umd/device/tt_silicon_driver_common.hpp>
#include <umd/device/tt_soc_descriptor.h>
#include <umd/device/tt_xy_pair.h>
#include <umd/device/types/cluster_descriptor_types.h>
#include <umd/device/types/harvesting.h>

namespace tt {
enum class ARCH;
namespace llrt {
class RunTimeOptions;
}
namespace tt_fabric {
class ControlPlane;
class GlobalControlPlane;
class FabricNodeId;
}
namespace tt_metal {
class Hal;
}
}  // namespace tt
struct tt_device_params;

static constexpr std::uint32_t SW_VERSION = 0x00020000;

using tt_target_dram = std::tuple<int, int, int>;

namespace tt {

/**
 * @brief Specifies the target devices on which the graph can be run.
 */
enum class TargetDevice : std::uint8_t {
    Silicon = 0,
    Simulator = 1,
    Invalid = 0xFF,
};

enum class ClusterType : std::uint8_t {
    INVALID = 0,
    N150 = 1,                    // Production N150
    N300 = 2,                    // Production N300
    T3K = 3,                     // Production T3K, built with 4 N300s
    GALAXY = 4,                  // Production Galaxy, all chips with mmio
    TG = 5,                      // Will be deprecated
    P100 = 6,                    // Blackhole single card, ethernet disabled
    P150 = 7,                    // Blackhole single card, ethernet enabled
    P150_X2 = 8,                 // 2 Blackhole single card, ethernet connected
    P150_X4 = 9,                 // 4 Blackhole single card, ethernet connected
    SIMULATOR_WORMHOLE_B0 = 10,  // Simulator Wormhole B0
    SIMULATOR_BLACKHOLE = 11,    // Simulator Blackhole
};

enum class EthRouterMode : uint32_t {
    IDLE = 0,
    BI_DIR_TUNNELING = 1,
    FABRIC_ROUTER = 2,
};

class Cluster {
public:
    // TODO: #21245: Remove these workaround APIs and instead refactor UMD component out of Cluster
    static ClusterType get_cluster_type_from_cluster_desc(
        const llrt::RunTimeOptions& rtoptions, const tt_ClusterDescriptor* cluster_desc = nullptr);
    static bool is_base_routing_fw_enabled(ClusterType cluster_type);
    Cluster& operator=(const Cluster&) = delete;
    Cluster& operator=(Cluster&& other) noexcept = delete;
    Cluster(const Cluster&) = delete;
    Cluster(Cluster&& other) noexcept = delete;

    Cluster(const llrt::RunTimeOptions& rtoptions, const tt_metal::Hal& hal);
    ~Cluster();

    // For TG Galaxy systems, mmio chips are gateway chips that are only used for dispatch, so user_devices are meant
    // for user facing host apis
    std::unordered_map<chip_id_t, eth_coord_t> get_user_chip_ethernet_coordinates() const;
    size_t number_of_user_devices() const;
    std::set<chip_id_t> user_exposed_chip_ids() const;

    size_t number_of_devices() const { return this->driver_->get_target_device_ids().size(); }

    std::set<chip_id_t> all_chip_ids() const { return this->driver_->get_target_device_ids(); };

    size_t number_of_pci_devices() const { return this->driver_->get_target_mmio_device_ids().size(); }

    // TODO: UMD will eventually consolidate ethernet coordinates and unique ids, we can remove the ethernet coord
    // getter after that change is in
    const std::unordered_map<chip_id_t, uint64_t>& get_unique_chip_ids() const {
        return this->cluster_desc_->get_chip_unique_ids();
    }
    std::unordered_map<chip_id_t, eth_coord_t> get_all_chip_ethernet_coordinates() const;

    chip_id_t get_physical_chip_id_from_eth_coord(const eth_coord_t& eth_coord) const;

    ARCH arch() const { return this->arch_; }

    const metal_SocDescriptor& get_soc_desc(chip_id_t chip) const;
    CoreCoord get_virtual_coordinate_from_logical_coordinates(
        chip_id_t chip_id, CoreCoord logical_coord, const CoreType& core_type) const;
    CoreCoord get_virtual_coordinate_from_physical_coordinates(chip_id_t chip_id, CoreCoord physical_coord) const;
    tt_cxy_pair get_virtual_coordinate_from_logical_coordinates(
        tt_cxy_pair logical_coordinate, const CoreType& core_type) const;
    CoreCoord get_physical_coordinate_from_logical_coordinates(
        chip_id_t chip_id, CoreCoord logical_coord, const CoreType& core_type, bool no_warn = false) const;
    const std::unordered_set<CoreCoord>& get_virtual_worker_cores(chip_id_t chip_id) const;
    const std::unordered_set<CoreCoord>& get_virtual_eth_cores(chip_id_t chip_id) const;

    uint32_t get_harvesting_mask(chip_id_t chip) const {
        return this->driver_->get_soc_descriptor(chip).harvesting_masks.tensix_harvesting_mask;
    }

    uint16_t get_bus_id(chip_id_t chip) const {
        return this->driver_->get_chip(chip)->get_tt_device()->get_pci_device()->get_device_info().pci_bus;
    }

    //! device driver and misc apis
    void verify_sw_fw_versions(int device_id, std::uint32_t sw_version, std::vector<std::uint32_t>& fw_versions) const;

    void deassert_risc_reset_at_core(
        const tt_cxy_pair& physical_chip_coord,
        const TensixSoftResetOptions& soft_resets = TENSIX_DEASSERT_SOFT_RESET) const;
    void assert_risc_reset_at_core(
        const tt_cxy_pair& physical_chip_coord,
        const TensixSoftResetOptions& soft_resets = TENSIX_ASSERT_SOFT_RESET) const;

    void write_dram_vec(
        const void* mem_ptr, uint32_t sz_in_bytes, chip_id_t device_id, int dram_view, uint64_t addr) const;
    void read_dram_vec(void* mem_ptr, uint32_t size_in_bytes, chip_id_t device_id, int dram_view, uint64_t addr) const;

    // Accepts physical noc coordinates
    void write_core(const void* mem_ptr, uint32_t sz_in_bytes, tt_cxy_pair core, uint64_t addr) const;
    void read_core(void* mem_ptr, uint32_t sz_in_bytes, tt_cxy_pair core, uint64_t addr) const;
    void read_core(std::vector<uint32_t>& data, uint32_t sz_in_bytes, tt_cxy_pair core, uint64_t addr) const;

    std::optional<std::tuple<uint32_t, uint32_t>> get_tlb_data(const tt_cxy_pair& target) const {
        tt::umd::CoreCoord target_coord = get_soc_desc(target.chip).get_coord_at(target, CoordSystem::TRANSLATED);
        auto tlb_configuration = driver_->get_tlb_configuration(target.chip, target_coord);
        return std::tuple((uint32_t)tlb_configuration.tlb_offset, (uint32_t)tlb_configuration.size);
    }

    std::function<void(uint32_t, uint32_t, const uint8_t*)> get_fast_pcie_static_tlb_write_callable(int chip_id) const {
        chip_id_t mmio_device_id = this->cluster_desc_->get_closest_mmio_capable_chip(chip_id);
        return driver_->get_fast_pcie_static_tlb_write_callable(mmio_device_id);
    }

    // Returns a writer object which holds a pointer to a static tlb
    // Allows for fast writes when targeting same device core by only doing the lookup once and avoiding repeated stack
    // traversals
    tt::Writer get_static_tlb_writer(tt_cxy_pair target) const {
        tt::umd::CoreCoord target_coord = get_soc_desc(target.chip).get_coord_at(target, CoordSystem::TRANSLATED);
        return driver_->get_static_tlb_writer(target.chip, target_coord);
    }

    std::uint32_t get_numa_node_for_device(uint32_t device_id) const {
        uint32_t mmio_device_id = this->get_associated_mmio_device(device_id);
        return driver_->get_numa_node_for_pcie_device(mmio_device_id);
    }

    void write_reg(const std::uint32_t* mem_ptr, tt_cxy_pair target, uint64_t addr) const;
    void read_reg(std::uint32_t* mem_ptr, tt_cxy_pair target, uint64_t addr) const;

    void write_sysmem(
        const void* mem_ptr, uint32_t size_in_bytes, uint64_t addr, chip_id_t src_device_id, uint16_t channel) const;
    void read_sysmem(
        void* mem_ptr, uint32_t size_in_bytes, uint64_t addr, chip_id_t src_device_id, uint16_t channel) const;

    int get_device_aiclk(const chip_id_t& chip_id) const;

    void dram_barrier(chip_id_t chip_id) const;
    void l1_barrier(chip_id_t chip_id) const;

    uint32_t get_num_host_channels(chip_id_t device_id) const;
    uint32_t get_host_channel_size(chip_id_t device_id, uint32_t channel) const;
    // Returns address in host space
    void* host_dma_address(uint64_t offset, chip_id_t src_device_id, uint16_t channel) const;
    uint64_t get_pcie_base_addr_from_device(chip_id_t chip_id) const;

    // Ethernet cluster api
    // Returns set of device ids connected via ethernet
    std::unordered_set<chip_id_t> get_ethernet_connected_device_ids(chip_id_t chip_id) const;

    // Returns set of logical active ethernet coordinates on chip
    // If skip_reserved_tunnel_cores is true, will return cores that dispatch is not using,
    // intended for users to grab available eth cores for testing
    // `skip_reserved_tunnel_cores` is ignored on BH because there are no ethernet cores used for Fast Dispatch
    // tunneling
    std::unordered_set<CoreCoord> get_active_ethernet_cores(
        chip_id_t chip_id, bool skip_reserved_tunnel_cores = false) const;

    // Returns set of logical inactive ethernet coordinates on chip
    std::unordered_set<CoreCoord> get_inactive_ethernet_cores(chip_id_t chip_id) const;

    // Returns whether `logical_core` has an eth link to a core on a connected chip
    // Cores that connect to another cluster will show up as connected
    bool is_ethernet_link_up(chip_id_t chip_id, const CoreCoord& logical_core) const;

    // Returns connected ethernet core on the other chip
    // If the core is connected to a device not accessible through this Cluster, it will assert
    std::tuple<chip_id_t, CoreCoord> get_connected_ethernet_core(std::tuple<chip_id_t, CoreCoord> eth_core) const;

    // Returns connected ethernet core on the other chip that is not managed by this Cluster
    std::tuple<uint64_t, CoreCoord> get_connected_ethernet_core_to_remote_mmio_device(
        std::tuple<chip_id_t, CoreCoord> eth_core) const;

    // Returns a ethernet sockets between local chip and remote chip
    // get_ethernet_sockets(a, b)[0] is connected to get_ethernet_sockets(b, a)[0]
    std::vector<CoreCoord> get_ethernet_sockets(chip_id_t local_chip, chip_id_t remote_chip) const;
    // Converts logical ethernet core coord to physical ethernet core coord
    CoreCoord ethernet_core_from_logical_core(chip_id_t chip_id, const CoreCoord& logical_core) const;

    // Returns virtual eth coord from channel
    CoreCoord get_virtual_eth_core_from_channel(chip_id_t chip_id, int channel) const;

    // Bookkeeping for mmio device tunnels
    uint32_t get_mmio_device_max_tunnel_depth(chip_id_t mmio_device) const;
    uint32_t get_mmio_device_tunnel_count(chip_id_t mmio_device) const;
    uint32_t get_device_tunnel_depth(chip_id_t chip_id) const;

    // Dispatch core is managed by device, so this is an api for device to get the each eth core used in FD tunneling.
    // Returns logical eth core that communicates with specified dispatch core
    tt_cxy_pair get_eth_core_for_dispatch_core(
        tt_cxy_pair logical_dispatch_core, EthRouterMode mode, chip_id_t connected_chip_id) const;

    std::tuple<tt_cxy_pair, tt_cxy_pair> get_eth_tunnel_core(
        chip_id_t upstream_chip_id, chip_id_t downstream_chip_id, EthRouterMode mode) const;

    // Internal routing for SD and FD enables launching user ethernet kernels and FD tunneling for all devices in the
    // cluster. When using multiple devices in a cluster, this should be the flow:
    //       CreateDevice(0)
    //       CreateDevice(1)
    //       set_internal_routing_info_for_ethernet_cores(true);
    //       set_internal_routing_info_for_ethernet_cores(false);
    //       CloseDevice(0)
    //       CloseDevice(1)
    void set_internal_routing_info_for_ethernet_cores(
        bool enable_internal_routing, const std::vector<chip_id_t>& target_mmio_devices = {}) const;

    std::unordered_map<chip_id_t, std::unordered_map<ethernet_channel_t, std::tuple<chip_id_t, ethernet_channel_t>>>
    get_ethernet_connections() const {
        return this->cluster_desc_->get_ethernet_connections();
    }

    // TODO: unify uint64_t with ChipUID
    std::unordered_map<chip_id_t, std::unordered_map<ethernet_channel_t, std::tuple<uint64_t, ethernet_channel_t>>>
    get_ethernet_connections_to_remote_mmio_devices() const {
        return this->cluster_desc_->get_ethernet_connections_to_remote_mmio_devices();
    }

    // Returns MMIO device ID (logical) that controls given `device_id`. If `device_id` is MMIO device it is returned.
    chip_id_t get_associated_mmio_device(chip_id_t device_id) const {
        return this->cluster_desc_->get_closest_mmio_capable_chip(device_id);
    }

    uint16_t get_assigned_channel_for_device(chip_id_t device_id) const {
        return this->device_to_host_mem_channel_.at(device_id);
    }

    // Returns collection of devices that are controlled by the specified MMIO device inclusive of the MMIO device
    const std::unordered_set<chip_id_t>& get_devices_controlled_by_mmio_device(chip_id_t mmio_device_id) const {
        TT_ASSERT(
            this->cluster_desc_->get_chips_grouped_by_closest_mmio().count(mmio_device_id),
            "Expected device {} to be an MMIO device!",
            mmio_device_id);
        return this->cluster_desc_->get_chips_grouped_by_closest_mmio().at(mmio_device_id);
    }

    // Returns map of connected chip ids to active ethernet cores
    std::unordered_map<chip_id_t, std::vector<CoreCoord>> get_ethernet_cores_grouped_by_connected_chips(
        chip_id_t chip_id) const;

    // Returns vector of unique tunnels originating from mmio device.
    // Each vector entry is another vector of remote devices on that tunnel.
    std::vector<std::vector<chip_id_t>> get_tunnels_from_mmio_device(chip_id_t mmio_chip_id) const {
        return this->tunnels_from_mmio_device.at(mmio_chip_id);
    }

    // Configures ethernet cores for fabric routers depending on whether fabric is enabled
    void configure_ethernet_cores_for_fabric_routers(
        tt_metal::FabricConfig fabric_config, std::optional<uint8_t> num_routing_planes = std::nullopt);

    // Returns whether we are running on Galaxy.
    bool is_galaxy_cluster() const;

    // Returns Wormhole chip board type.
    BoardType get_board_type(chip_id_t chip_id) const;

    ClusterType get_cluster_type() const;

    bool is_base_routing_fw_enabled() const;

    // Get all fabric ethernet cores
    std::set<tt_fabric::chan_id_t> get_fabric_ethernet_channels(chip_id_t chip_id) const;

    // Get fabric ethernet cores connecting src to dst
    std::vector<CoreCoord> get_fabric_ethernet_routers_between_src_and_dest(chip_id_t src_id, chip_id_t dst_id) const;

    bool is_worker_core(const CoreCoord& core, chip_id_t chip_id) const;
    bool is_ethernet_core(const CoreCoord& core, chip_id_t chip_id) const;
    CoreCoord get_logical_ethernet_core_from_virtual(chip_id_t chip, CoreCoord core) const;

    // These two functions should be removed in favor of direct translation.
    std::unordered_map<int, int> get_worker_logical_to_virtual_x(chip_id_t chip_id) const;
    std::unordered_map<int, int> get_worker_logical_to_virtual_y(chip_id_t chip_id) const;

    const std::unordered_map<CoreCoord, int32_t>& get_virtual_routing_to_profiler_flat_id(chip_id_t chip_id) const;

    std::uint32_t get_ubb_asic_id(chip_id_t physical_chip_id) const;

    // TODO: move to separate system descriptor class
    // return enum for connection type, Internal, QSFP, Other, Unknown
    bool is_external_cable(chip_id_t physical_chip_id, CoreCoord eth_core) const;

private:
    void detect_arch_and_target();
    void generate_cluster_descriptor();
    void initialize_device_drivers();
    void assert_risc_reset();
    void assign_mem_channels_to_devices(
        chip_id_t mmio_device_id, const std::unordered_set<chip_id_t>& controlled_device_ids);
    void open_driver(const bool& skip_driver_allocs = false);
    void start_driver(tt_device_params& device_params) const;
    void validate_harvesting_masks() const;

    void get_metal_desc_from_tt_desc();
    void generate_virtual_to_umd_coord_mapping();
    void generate_virtual_to_profiler_flat_id_mapping();

    // Reserves ethernet cores in cluster for tunneling
    void reserve_ethernet_cores_for_tunneling();

    void initialize_ethernet_sockets();

    // Disable ethernet cores that retrain
    // This should be removed when we handle retraining or dropped links in control plane properly
    void disable_ethernet_cores_with_retrain();


    // Set tunnels from mmio
    void set_tunnels_from_mmio_device();

    bool supports_dma_operations(chip_id_t chip_id, uint32_t sz_in_bytes) const;

    ARCH arch_;
    TargetDevice target_type_;

    // There is a single device driver for all connected chips. It might contain multiple MMIO devices/cards.
    std::unique_ptr<tt::umd::Cluster> driver_;

    // Need to hold reference to cluster descriptor to detect total number of devices available in cluster
    // UMD static APIs `detect_available_device_ids` and `detect_number_of_chips` only returns number of MMIO mapped
    // devices
    tt_ClusterDescriptor* cluster_desc_ = nullptr;
    // In case of mock cluster descriptor, the tt_cluster holds the ownership of the created object;
    // This is obviously a design issue. This should go away once the design is fixed.
    std::unique_ptr<tt_ClusterDescriptor> mock_cluster_desc_ptr_;
    // There is an entry for every device that can be targeted (MMIO and remote)
    std::unordered_map<chip_id_t, metal_SocDescriptor> sdesc_per_chip_;

    // Data Structures Tracking Virtual Coordinates
    std::unordered_map<tt_cxy_pair, tt_cxy_pair> virtual_to_umd_coord_mapping_;
    std::unordered_map<chip_id_t, std::unordered_set<CoreCoord>> virtual_worker_cores_;
    std::unordered_map<chip_id_t, std::unordered_set<CoreCoord>> virtual_eth_cores_;
    std::unordered_map<chip_id_t, std::unordered_set<CoreCoord>> virtual_dram_cores_;
    std::unordered_map<chip_id_t, std::unordered_set<CoreCoord>> virtual_pcie_cores_;
    std::unordered_map<BoardType, std::unordered_map<CoreCoord, int32_t>> virtual_routing_to_profiler_flat_id_;
    std::unordered_map<chip_id_t, std::unordered_set<CoreCoord>> frequent_retrain_cores_;
    // Flag to tell whether we are on a TG type of system.
    // If any device has to board type of GALAXY, we are on a TG cluster.
    ClusterType cluster_type_ = ClusterType::INVALID;

    // Reserves specified number of ethernet cores for fabric routers
    void reserve_ethernet_cores_for_fabric_routers(uint8_t num_routing_planes);

    // Releases all reserved ethernet cores for fabric routers
    void release_ethernet_cores_for_fabric_routers();

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

    uint32_t routing_info_addr_ = 0;

    // Cluster depends on RunTimeOptions and Hal to set up, but they're all initialized/accessed by MetalContext, so
    // keep a local reference for init.
    const llrt::RunTimeOptions& rtoptions_;
    const tt_metal::Hal& hal_;
};

}  // namespace tt

std::ostream& operator<<(std::ostream& os, const tt_target_dram& dram);
