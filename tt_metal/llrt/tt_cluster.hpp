// SPDX-FileCopyrightText: © 2023 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "hostdevcommon/fabric_common.h"
#include <tt-metalium/experimental/fabric/fabric_types.hpp>
#include <tt-metalium/metal_soc_descriptor.h>
#include <tt-metalium/cluster.hpp>
#include "llrt/rtoptions.hpp"
#include "llrt/tt_target_device.hpp"
#include <cstddef>
#include <cstdint>
#include <map>
#include <memory>
#include <optional>
#include <ostream>
#include <set>
#include <tuple>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include <tt_stl/assert.hpp>
#include "core_coord.hpp"
#include <umd/device/cluster.hpp>
#include <umd/device/driver_atomics.hpp>
#include <umd/device/cluster_descriptor.hpp>
#include <umd/device/types/core_coordinates.hpp>
#include <umd/device/tt_io.hpp>
#include <umd/device/soc_descriptor.hpp>
#include <umd/device/types/xy_pair.hpp>
#include <umd/device/types/cluster_descriptor_types.hpp>
#include <umd/device/types/cluster_types.hpp>

namespace tt {
namespace llrt {
class RunTimeOptions;
}
namespace tt_fabric {
class ControlPlane;
class FabricNodeId;
}  // namespace tt_fabric
namespace tt_metal {
class Hal;
}
}  // namespace tt

static constexpr std::uint32_t SW_VERSION = 0x00020000;

using tt_target_dram = std::tuple<int, int, int>;

namespace tt {

enum class EthRouterMode : uint32_t {
    IDLE = 0,
    FABRIC_ROUTER = 1,
};

class Cluster {
public:
    // TODO: #21245: Remove these workaround APIs and instead refactor UMD component out of Cluster
    static tt::tt_metal::ClusterType get_cluster_type_from_cluster_desc(
        const llrt::RunTimeOptions& rtoptions, const umd::ClusterDescriptor* cluster_desc = nullptr);
    static bool is_base_routing_fw_enabled(tt::tt_metal::ClusterType cluster_type);
    Cluster& operator=(const Cluster&) = delete;
    Cluster& operator=(Cluster&& other) noexcept = delete;
    Cluster(const Cluster&) = delete;
    Cluster(Cluster&& other) noexcept = delete;

    Cluster(llrt::RunTimeOptions& rtoptions, const tt_metal::Hal& hal);
    ~Cluster();

    // For TG Galaxy systems, mmio chips are gateway chips that are only used for dispatch, so user_devices are meant
    // for user facing host apis
    std::unordered_map<ChipId, EthCoord> get_user_chip_ethernet_coordinates() const;
    size_t number_of_user_devices() const;
    std::set<ChipId> user_exposed_chip_ids() const;

    size_t number_of_devices() const { return this->driver_->get_target_device_ids().size(); }

    std::set<ChipId> all_chip_ids() const { return this->driver_->get_target_device_ids(); };

    std::set<ChipId> mmio_chip_ids() const { return this->driver_->get_target_mmio_device_ids(); }

    size_t number_of_pci_devices() const { return this->driver_->get_target_mmio_device_ids().size(); }

    std::set<ChipId> all_pci_chip_ids() const { return this->driver_->get_target_mmio_device_ids(); }

    umd::ClusterDescriptor* get_cluster_desc() const {
        TT_FATAL(this->cluster_desc_ != nullptr, "Cluster descriptor is not initialized.");
        return this->cluster_desc_;
    }

    const std::unique_ptr<tt::umd::Cluster>& get_driver() const {
        TT_FATAL(driver_ != nullptr, "UMD driver is not initialized.");
        return driver_;
    }

    // TODO: UMD will eventually consolidate ethernet coordinates and unique ids, we can remove the ethernet coord
    // getter after that change is in
    const std::unordered_map<ChipId, uint64_t>& get_unique_chip_ids() const {
        return this->cluster_desc_->get_chip_unique_ids();
    }

    // Returns map of logical chip ID to PCIe device ID
    const std::unordered_map<ChipId, ChipId>& get_chips_with_mmio() const {
        return this->cluster_desc_->get_chips_with_mmio();
    }

    std::unordered_map<ChipId, EthCoord> get_all_chip_ethernet_coordinates() const;

    ChipId get_physical_chip_id_from_eth_coord(const EthCoord& eth_coord) const;

    ARCH arch() const { return this->arch_; }

    const metal_SocDescriptor& get_soc_desc(ChipId chip) const;
    CoreCoord get_virtual_coordinate_from_logical_coordinates(
        ChipId chip_id, CoreCoord logical_coord, const CoreType& core_type) const;
    CoreCoord get_virtual_coordinate_from_physical_coordinates(ChipId chip_id, CoreCoord physical_coord) const;
    tt_cxy_pair get_virtual_coordinate_from_logical_coordinates(
        tt_cxy_pair logical_coordinate, const CoreType& core_type) const;
    CoreCoord get_physical_coordinate_from_logical_coordinates(
        ChipId chip_id, CoreCoord logical_coord, const CoreType& core_type, bool no_warn = false) const;
    const std::unordered_set<CoreCoord>& get_virtual_worker_cores(ChipId chip_id) const;
    const std::unordered_set<CoreCoord>& get_virtual_eth_cores(ChipId chip_id) const;

    uint32_t get_harvesting_mask(ChipId chip) const {
        return this->driver_->get_soc_descriptor(chip).harvesting_masks.tensix_harvesting_mask;
    }

    uint16_t get_bus_id(ChipId chip) const;

    std::optional<int> get_physical_slot(ChipId chip) const;

    //! device driver and misc apis
    void verify_sw_fw_versions(int device_id, std::uint32_t sw_version, std::vector<std::uint32_t>& fw_versions) const;
    bool verify_eth_fw_capability() const;

    void deassert_risc_reset_at_core(
        const tt_cxy_pair& core, const tt::umd::RiscType& soft_resets, bool staggered_start = true) const;
    void assert_risc_reset_at_core(const tt_cxy_pair& core, const tt::umd::RiscType& soft_resets) const;

    void write_dram_vec(
        const void* mem_ptr, uint32_t sz_in_bytes, ChipId device_id, int dram_view, uint64_t addr) const;
    void read_dram_vec(void* mem_ptr, uint32_t sz_in_bytes, ChipId device_id, int dram_view, uint64_t addr) const;

    // Write to core. Accepts physical noc coordinates
    void write_core(const void* mem_ptr, uint32_t sz_in_bytes, tt_cxy_pair core, uint64_t addr) const;

    // Access physical noc coordinates. Does write without effects of write combining
    void write_core_immediate(const void* mem_ptr, uint32_t sz_in_bytes, tt_cxy_pair core, uint64_t addr) const;

    // Write to core without effects of write combining
    template <typename DType>
    void write_core_immediate(
        ChipId device_id, const CoreCoord& core, const std::span<DType>& hex_vec, uint64_t addr) const {
        write_core_immediate(hex_vec.data(), hex_vec.size() * sizeof(DType), tt_cxy_pair(device_id, core), addr);
    }

    // Write to core without effects of write combining
    template <typename DType>
    void write_core_immediate(
        ChipId device_id, const CoreCoord& core, const std::vector<DType>& hex_vec, uint64_t addr) const {
        write_core_immediate(hex_vec.data(), hex_vec.size() * sizeof(DType), tt_cxy_pair(device_id, core), addr);
    }

    // Write span to core
    template <typename DType>
    void write_core(ChipId device_id, const CoreCoord& core, const std::span<DType>& hex_vec, uint64_t addr) const {
        write_core(hex_vec.data(), hex_vec.size() * sizeof(DType), tt_cxy_pair(device_id, core), addr);
    }

    // Write vector to core
    template <typename DType>
    void write_core(ChipId device_id, const CoreCoord& core, const std::vector<DType>& hex_vec, uint64_t addr) const {
        write_core(hex_vec.data(), hex_vec.size() * sizeof(DType), tt_cxy_pair(device_id, core), addr);
    }

    void read_core(void* mem_ptr, uint32_t size_in_bytes, tt_cxy_pair core, uint64_t addr) const;

    void read_core(std::vector<uint32_t>& data, uint32_t size_in_bytes, tt_cxy_pair core, uint64_t addr) const;

    template <typename DType = uint32_t>
    [[nodiscard]] std::vector<DType> read_core(ChipId chip, const CoreCoord& core, uint64_t addr, uint32_t size) const {
        std::vector<DType> read_hex_vec;
        read_core(read_hex_vec, size, tt_cxy_pair(chip, core), addr);
        return read_hex_vec;
    }

    // NOC multicast write wrappers
    void noc_multicast_write(
        const void* mem_ptr, uint32_t sz_in_bytes, tt_cxy_pair core_start, tt_cxy_pair core_end, uint64_t addr) const;
    void noc_multicast_write(
        const void* mem_ptr,
        uint32_t sz_in_bytes,
        ChipId chip_id,
        CoreCoord core_start,
        CoreCoord core_end,
        uint64_t addr) const;

    std::optional<std::tuple<uint32_t, uint32_t>> get_tlb_data(const tt_cxy_pair& target) const {
        tt::umd::CoreCoord target_coord = get_soc_desc(target.chip).get_coord_at(target, CoordSystem::TRANSLATED);
        auto tlb_configuration = driver_->get_tlb_configuration(target.chip, target_coord);
        return std::tuple((uint32_t)tlb_configuration.tlb_offset, (uint32_t)tlb_configuration.size);
    }

    // Returns a writer object which holds a pointer to a static tlb
    // Allows for fast writes when targeting same device core by only doing the lookup once and avoiding repeated stack
    // traversals
    umd::Writer get_static_tlb_writer(tt_cxy_pair target) const {
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
        const void* vec, uint32_t size_in_bytes, uint64_t addr, ChipId src_device_id, uint16_t channel) const;
    void read_sysmem(void* vec, uint32_t size_in_bytes, uint64_t addr, ChipId src_device_id, uint16_t channel) const;

    // System memory buffer allocation methods
    std::unique_ptr<tt::umd::SysmemBuffer> allocate_sysmem_buffer(
        ChipId device_id, size_t sysmem_buffer_size, bool map_to_noc = false) const;
    std::unique_ptr<tt::umd::SysmemBuffer> map_sysmem_buffer(
        ChipId device_id, void* buffer, size_t sysmem_buffer_size, bool map_to_noc = false) const;

    int get_device_aiclk(const ChipId& chip_id) const;

    void dram_barrier(ChipId chip_id) const;
    void l1_barrier(ChipId chip_id) const;

    uint32_t get_num_host_channels(ChipId device_id) const;
    uint32_t get_host_channel_size(ChipId device_id, uint32_t channel) const;
    // Returns address in host space
    void* host_dma_address(uint64_t offset, ChipId src_device_id, uint16_t channel) const;
    uint64_t get_pcie_base_addr_from_device(ChipId chip_id) const;

    // Ethernet cluster api
    // Returns set of device ids connected via ethernet
    std::unordered_set<ChipId> get_ethernet_connected_device_ids(ChipId chip_id) const;

    // Returns whether `logical_core` has an eth link to a core on a connected chip
    // Cores that connect to another cluster will show up as connected
    bool is_ethernet_link_up(ChipId chip_id, const CoreCoord& logical_core) const;

    // Returns connected ethernet core on the other chip
    // If the core is connected to a device not accessible through this Cluster, it will assert
    std::tuple<ChipId, CoreCoord> get_connected_ethernet_core(std::tuple<ChipId, CoreCoord> eth_core) const;

    // Returns connected ethernet core on the other chip that is not managed by this Cluster
    std::tuple<uint64_t, CoreCoord> get_connected_ethernet_core_to_remote_mmio_device(
        std::tuple<ChipId, CoreCoord> eth_core) const;

    // Returns a ethernet sockets between local chip and remote chip
    // get_ethernet_sockets(a, b)[0] is connected to get_ethernet_sockets(b, a)[0]
    std::vector<CoreCoord> get_ethernet_sockets(ChipId local_chip, ChipId remote_chip) const;
    // Converts logical ethernet core coord to physical ethernet core coord
    CoreCoord ethernet_core_from_logical_core(ChipId chip_id, const CoreCoord& logical_core) const;

    // Returns virtual eth coord from channel
    CoreCoord get_virtual_eth_core_from_channel(ChipId chip_id, int channel) const;

    // Internal routing for SD and FD enables launching user ethernet kernels and FD tunneling for all devices in the
    // cluster. When using multiple devices in a cluster, this should be the flow:
    //       CreateDevice(0)
    //       CreateDevice(1)
    //       set_internal_routing_info_for_ethernet_cores(true);
    //       set_internal_routing_info_for_ethernet_cores(false);
    //       CloseDevice(0)
    //       CloseDevice(1)
    void set_internal_routing_info_for_ethernet_cores(
        bool enable_internal_routing, const std::vector<ChipId>& target_mmio_devices = {}) const;

    const std::unordered_map<ChipId, std::unordered_map<EthernetChannel, std::tuple<ChipId, EthernetChannel>>>&
    get_ethernet_connections() const {
        return this->cluster_desc_->get_ethernet_connections();
    }

    // TODO: unify uint64_t with ChipUID
    const std::unordered_map<ChipId, std::unordered_map<EthernetChannel, std::tuple<uint64_t, EthernetChannel>>>&
    get_ethernet_connections_to_remote_devices() const {
        return this->cluster_desc_->get_ethernet_connections_to_remote_devices();
    }

    // Returns MMIO device ID (logical) that controls given `device_id`. If `device_id` is MMIO device it is returned.
    ChipId get_associated_mmio_device(ChipId device_id) const {
        return this->cluster_desc_->get_closest_mmio_capable_chip(device_id);
    }

    uint16_t get_assigned_channel_for_device(ChipId device_id) const {
        return this->device_to_host_mem_channel_.at(device_id);
    }

    // Returns collection of devices that are controlled by the specified MMIO device inclusive of the MMIO device
    const std::unordered_set<ChipId>& get_devices_controlled_by_mmio_device(ChipId mmio_device_id) const;

    // Returns map of connected chip ids to active ethernet cores
    std::unordered_map<ChipId, std::vector<CoreCoord>> get_ethernet_cores_grouped_by_connected_chips(
        ChipId chip_id) const;

    // Returns vector of unique tunnels originating from mmio device.
    // Each vector entry is another vector of remote devices on that tunnel.
    std::vector<std::vector<ChipId>> get_tunnels_from_mmio_device(ChipId mmio_chip_id) const {
        return this->tunnels_from_mmio_device.at(mmio_chip_id);
    }

    // Configures ethernet cores for fabric routers depending on whether fabric is enabled
    void configure_ethernet_cores_for_fabric_routers(
        tt_fabric::FabricConfig fabric_config, std::optional<uint8_t> num_routing_planes = std::nullopt);

    void initialize_fabric_config(
        tt_fabric::FabricConfig fabric_config, tt_fabric::FabricReliabilityMode reliability_mode);

    // Returns whether we are running on Legacy Galaxy.
    bool is_galaxy_cluster() const;

    // Returns whether we are running on UBB Galaxy.
    bool is_ubb_galaxy() const;

    // Returns Wormhole chip board type.
    BoardType get_board_type(ChipId chip_id) const;

    // Returns whether IOMMU is enabled on the system (cached at init time)
    bool is_iommu_enabled() const;
    // Returns whether NOC mapping is enabled on the system (cached at init time)
    bool is_noc_mapping_enabled() const;

    tt::tt_metal::ClusterType get_cluster_type() const;

    tt::TargetDevice get_target_device_type() const { return this->target_type_; }

    bool is_base_routing_fw_enabled() const;

    // Get all fabric ethernet cores
    std::set<tt_fabric::chan_id_t> get_fabric_ethernet_channels(ChipId chip_id) const;

    // Get fabric ethernet cores connecting src to dst
    std::vector<CoreCoord> get_fabric_ethernet_routers_between_src_and_dest(ChipId src_id, ChipId dst_id) const;

    bool is_worker_core(const CoreCoord& core, ChipId chip_id) const;
    bool is_ethernet_core(const CoreCoord& core, ChipId chip_id) const;
    CoreCoord get_logical_ethernet_core_from_virtual(ChipId chip, CoreCoord core) const;

    // These two functions should be removed in favor of direct translation.
    std::unordered_map<int, int> get_worker_logical_to_virtual_x(ChipId chip_id) const;
    std::unordered_map<int, int> get_worker_logical_to_virtual_y(ChipId chip_id) const;

    const std::unordered_map<CoreCoord, int32_t>& get_virtual_routing_to_profiler_flat_id(ChipId chip_id) const;

    std::uint32_t get_ubb_asic_id(ChipId physical_chip_id) const;

    // TODO: move to separate system descriptor class
    // return enum for connection type, Internal, QSFP, Other, Unknown
    bool is_external_cable(ChipId physical_chip_id, CoreCoord eth_core) const;

    uint32_t get_alignment_requirements(ChipId chip_id, uint32_t size_in_bytes) const;

    const std::unordered_set<CoreCoord>& get_eth_cores_with_frequent_retraining(ChipId chip_id) const {
        return this->frequent_retrain_cores_.at(chip_id);
    }

    const std::unordered_map<CoreCoord, EthRouterMode>& get_eth_routing_info(ChipId chip_id) const {
        return this->device_eth_routing_info_.at(chip_id);
    }

private:
    void detect_arch_and_target();
    void generate_cluster_descriptor();
    void initialize_device_drivers();
    void assert_risc_reset();
    void assign_mem_channels_to_devices(ChipId mmio_device_id, const std::unordered_set<ChipId>& controlled_device_ids);
    void open_driver(const bool& skip_driver_allocs = false);
    void start_driver(umd::DeviceParams& device_params) const;
    void validate_harvesting_masks() const;

    void get_metal_desc_from_tt_desc();
    void generate_virtual_to_umd_coord_mapping();
    void generate_virtual_to_profiler_flat_id_mapping();

    // Reserves ethernet cores in cluster for tunneling
    void initialize_ethernet_cores_router_mode();

    void initialize_ethernet_sockets();

    // Disable ethernet cores that retrain
    // This should be removed when we handle retraining or dropped links in control plane properly
    void disable_ethernet_cores_with_retrain();

    // Set tunnels from mmio
    void set_tunnels_from_mmio_device();

    bool supports_dma_operations(ChipId chip_id, uint32_t sz_in_bytes) const;

    ARCH arch_{tt::ARCH::Invalid};
    TargetDevice target_type_{0};

    // There is a single device driver for all connected chips. It might contain multiple MMIO devices/cards.
    std::unique_ptr<tt::umd::Cluster> driver_;

    // Cached system IOMMU status to avoid slow queries at MeshDevice construction
    bool iommu_enabled_ = false;
    // Cached system NOC mapping status to avoid slow queries at MeshDevice construction
    bool noc_mapping_enabled_ = false;

    // Need to hold reference to cluster descriptor to detect total number of devices available in cluster
    // UMD static APIs `detect_available_device_ids` and `detect_number_of_chips` only returns number of MMIO mapped
    // devices
    umd::ClusterDescriptor* cluster_desc_ = nullptr;

    // There is an entry for every device that can be targeted (MMIO and remote)
    std::unordered_map<ChipId, metal_SocDescriptor> sdesc_per_chip_;

    // Data Structures Tracking Virtual Coordinates
    std::unordered_map<tt_cxy_pair, tt_cxy_pair> virtual_to_umd_coord_mapping_;
    std::unordered_map<ChipId, std::unordered_set<CoreCoord>> virtual_worker_cores_;
    std::unordered_map<ChipId, std::unordered_set<CoreCoord>> virtual_eth_cores_;
    std::unordered_map<ChipId, std::unordered_set<CoreCoord>> virtual_dram_cores_;
    std::unordered_map<ChipId, std::unordered_set<CoreCoord>> virtual_pcie_cores_;
    std::unordered_map<BoardType, std::unordered_map<CoreCoord, int32_t>> virtual_routing_to_profiler_flat_id_;
    std::unordered_map<ChipId, std::unordered_set<CoreCoord>> frequent_retrain_cores_;
    // Flag to tell whether we are on a TG type of system.
    // If any device has to board type of GALAXY, we are on a TG cluster.
    tt::tt_metal::ClusterType cluster_type_ = tt::tt_metal::ClusterType::INVALID;

    // Reserves specified number of ethernet cores for fabric routers
    void reserve_ethernet_cores_for_fabric_routers(uint8_t num_routing_planes);

    // Releases all reserved ethernet cores for fabric routers
    void release_ethernet_cores_for_fabric_routers();

    // Tunnels setup in cluster
    std::map<ChipId, std::vector<std::vector<ChipId>>> tunnels_from_mmio_device;

    // Currently, each device is mapped to its own channel in host memory to enable fast dispatch
    // Channels are unique within a group of devices all controlled by a particular MMIO device
    // For example:
    //      Two N300 cards where MMIO device IDs are 0, 1 and R chips are 2, 3
    //      0 L controls 2 R and 1 L controls 3 R then, device_to_host_mem_channel_:
    //          0 -> 0
    //          2 -> 1
    //          1 -> 0
    //          3 -> 1
    std::unordered_map<ChipId, uint16_t> device_to_host_mem_channel_;

    // Mapping of each devices' ethernet routing mode
    std::unordered_map<ChipId, std::unordered_map<CoreCoord, EthRouterMode>> device_eth_routing_info_;

    std::unordered_map<ChipId, std::unordered_map<ChipId, std::vector<CoreCoord>>> ethernet_sockets_;

    uint32_t routing_info_addr_ = 0;

    // Cluster depends on RunTimeOptions and Hal to set up, but they're all initialized/accessed by MetalContext, so
    // keep a local reference for init.
    const llrt::RunTimeOptions& rtoptions_;
    const tt_metal::Hal& hal_;
};

}  // namespace tt

std::ostream& operator<<(std::ostream& os, const tt_target_dram& dram);
