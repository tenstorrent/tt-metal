// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include <memory>
#include <stdexcept>
#include <iostream>
#include <typeinfo>

#include "umd/device/cluster.h"
#include "umd/device/tt_cluster_descriptor.h"
#include "llrt/rtoptions.hpp"
#include <tt-metalium/control_plane.hpp>
#include "impl/context/metal_context.hpp"

namespace tt::tt_fabric {

namespace mock_cluster_tests {

class ClusterFixture : public ::testing::Test {
protected:
    std::unique_ptr<tt_ClusterDescriptor> cluster_desc;

    void SetUp() override { printf("ClusterFixture SetUp\n"); }

    void TearDown() override { printf("ClusterFixture TearDown\n"); }
};

class MockCluster : public tt::ClusterBase {
public:
    MockCluster(tt_ClusterDescriptor* cluster_desc) : tt::ClusterBase() {
        this->cluster_desc_ = cluster_desc;

        this->driver_ = std::make_unique<tt::umd::Cluster>(
            tt::umd::ClusterOptions{.chip_type = tt::umd::ChipType::MOCK, .cluster_descriptor = cluster_desc});

        this->cluster_type_ = tt::Cluster::get_cluster_type_from_cluster_desc(this->rtoptions_, this->cluster_desc_);
    }

    ~MockCluster() {
        printf("MockCluster destructor\n");
    }

    size_t number_of_devices() const override { return this->driver_->get_target_device_ids().size(); }

    std::set<chip_id_t> all_chip_ids() const override { return this->driver_->get_target_device_ids(); };

    std::set<chip_id_t> mmio_chip_ids() const override { return this->driver_->get_target_mmio_device_ids(); }

    size_t number_of_pci_devices() const override { return this->driver_->get_target_mmio_device_ids().size(); }

    std::set<chip_id_t> all_pci_chip_ids() const override { return this->driver_->get_target_mmio_device_ids(); }

    tt_ClusterDescriptor* get_cluster_desc() const override {
        TT_FATAL(this->cluster_desc_ != nullptr, "Cluster descriptor is not initialized.");
        return this->cluster_desc_;
    }


    // For TG Galaxy systems, mmio chips are gateway chips that are only used for dispatch, so user_devices are meant
    // for user facing host apis
    std::unordered_map<chip_id_t, eth_coord_t> get_user_chip_ethernet_coordinates() const override {
        printf("Not implemented\n");
        return {};
    }
    size_t number_of_user_devices() const override {
        printf("Not implemented\n");
        return 0;
    }
    std::set<chip_id_t> user_exposed_chip_ids() const override {
        if (this->cluster_type_ == tt::tt_metal::ClusterType::TG) {
            std::set<chip_id_t> galaxy_boards;
            const auto& chips = this->driver_->get_target_device_ids();
            for (const auto& id : chips) {
                if (this->cluster_desc_->get_board_type(id) == BoardType::GALAXY) {
                    galaxy_boards.insert(id);
                }
            }
            return galaxy_boards;
        } else {
            return this->driver_->get_target_device_ids();
        }
    }

    // TODO: UMD will eventually consolidate ethernet coordinates and unique ids, we can remove the ethernet coord
    // getter after that change is in
    const std::unordered_map<chip_id_t, uint64_t>& get_unique_chip_ids() const override {
        return this->cluster_desc_->get_chip_unique_ids();
    }
    std::unordered_map<chip_id_t, eth_coord_t> get_all_chip_ethernet_coordinates() const override {
        return this->cluster_desc_->get_chip_locations();
    }

    chip_id_t get_physical_chip_id_from_eth_coord(const eth_coord_t& eth_coord) const override {
        for (const auto& [physical_chip_id, coord] : this->get_all_chip_ethernet_coordinates()) {
            if (coord == eth_coord) {
                return physical_chip_id;
            }
        }
        TT_FATAL(false, "Physical chip id not found for eth coord");
        return 0;
    }

    ARCH arch() const override { 
        printf("Not implemented\n");
        return tt::ARCH::WORMHOLE_B0; 
    }

    const metal_SocDescriptor& get_soc_desc(chip_id_t chip) const override {
        if (this->sdesc_per_chip_.find(chip) == this->sdesc_per_chip_.end()) {
            TT_THROW(
                "Cannot access soc descriptor for {} before device driver is initialized! Call "
                "initialize_device_driver({}) first",
                chip,
                chip);
        }
        return this->sdesc_per_chip_.at(chip);
    }
    CoreCoord get_virtual_coordinate_from_logical_coordinates(
        chip_id_t chip_id, CoreCoord logical_coord, const CoreType& core_type) const override {
        printf("Not implemented\n");
        return CoreCoord{0, 0};
    }
    CoreCoord get_virtual_coordinate_from_physical_coordinates(chip_id_t chip_id, CoreCoord physical_coord) const override {
        printf("Not implemented\n");
        return CoreCoord{0, 0};
    }
    tt_cxy_pair get_virtual_coordinate_from_logical_coordinates(
        tt_cxy_pair logical_coordinate, const CoreType& core_type) const override {
        printf("Not implemented\n");
        return tt_cxy_pair{0, 0, 0};
    }
    CoreCoord get_physical_coordinate_from_logical_coordinates(
        chip_id_t chip_id, CoreCoord logical_coord, const CoreType& core_type, bool no_warn = false) const override {
        printf("Not implemented\n");
        return CoreCoord{0, 0};
    }
    const std::unordered_set<CoreCoord>& get_virtual_worker_cores(chip_id_t chip_id) const override {
        printf("Not implemented\n");
        static std::unordered_set<CoreCoord> dummy_set;
        return dummy_set;
    }
    const std::unordered_set<CoreCoord>& get_virtual_eth_cores(chip_id_t chip_id) const override {
        printf("Not implemented\n");
        static std::unordered_set<CoreCoord> dummy_set;
        return dummy_set;
    }

    uint32_t get_harvesting_mask(chip_id_t chip) const override {
        return this->driver_->get_soc_descriptor(chip).harvesting_masks.tensix_harvesting_mask;
    }

    uint16_t get_bus_id(chip_id_t chip) const override {
        return this->driver_->get_chip(chip)->get_tt_device()->get_pci_device()->get_device_info().pci_bus;
    }

    //! device driver and misc apis
    void verify_sw_fw_versions(int device_id, std::uint32_t sw_version, std::vector<std::uint32_t>& fw_versions) const override {
        printf("Not implemented\n");
    }

    void deassert_risc_reset_at_core(
        const tt_cxy_pair& physical_chip_coord,
        const TensixSoftResetOptions& soft_resets = TENSIX_DEASSERT_SOFT_RESET) const override {
        printf("Not implemented\n");
    }
    void assert_risc_reset_at_core(
        const tt_cxy_pair& physical_chip_coord,
        const TensixSoftResetOptions& soft_resets = TENSIX_ASSERT_SOFT_RESET) const override {
        printf("Not implemented\n");
    }

    void write_dram_vec(
        const void* mem_ptr, uint32_t sz_in_bytes, chip_id_t device_id, int dram_view, uint64_t addr) const override {
        printf("Not implemented\n");
    }
    void read_dram_vec(void* mem_ptr, uint32_t size_in_bytes, chip_id_t device_id, int dram_view, uint64_t addr) const override {
        printf("Not implemented\n");
    }

    // Accepts physical noc coordinates
    void write_core(const void* mem_ptr, uint32_t sz_in_bytes, tt_cxy_pair core, uint64_t addr) const override {
        printf("Not implemented\n");
    }
    void read_core(void* mem_ptr, uint32_t sz_in_bytes, tt_cxy_pair core, uint64_t addr) const override {
        printf("Not implemented\n");
    }
    void read_core(std::vector<uint32_t>& data, uint32_t sz_in_bytes, tt_cxy_pair core, uint64_t addr) const override {
        printf("Not implemented\n");
    }

    std::optional<std::tuple<uint32_t, uint32_t>> get_tlb_data(const tt_cxy_pair& target) const override {
        printf("Not implemented\n");
        return std::nullopt;
    }

    std::function<void(uint32_t, uint32_t, const uint8_t*)> get_fast_pcie_static_tlb_write_callable(int chip_id) const override {
        printf("Not implemented\n");
        return nullptr;
    }

    // Returns a writer object which holds a pointer to a static tlb
    // Allows for fast writes when targeting same device core by only doing the lookup once and avoiding repeated stack
    // traversals
    tt::Writer get_static_tlb_writer(tt_cxy_pair target) const override {
        printf("Not implemented\n");
        tt::umd::CoreCoord target_coord = get_soc_desc(target.chip).get_coord_at(target, CoordSystem::TRANSLATED);
        return this->driver_->get_static_tlb_writer(target.chip, target_coord);
    }

    std::uint32_t get_numa_node_for_device(uint32_t device_id) const override {
        printf("Not implemented\n");
        return 0;
    }

    void write_reg(const std::uint32_t* mem_ptr, tt_cxy_pair target, uint64_t addr) const override {
        printf("Not implemented\n");
    }
    void read_reg(std::uint32_t* mem_ptr, tt_cxy_pair target, uint64_t addr) const override {
        printf("Not implemented\n");
    }

    void write_sysmem(
        const void* mem_ptr, uint32_t size_in_bytes, uint64_t addr, chip_id_t src_device_id, uint16_t channel) const override {
        printf("Not implemented\n");
    }
    void read_sysmem(
        void* mem_ptr, uint32_t size_in_bytes, uint64_t addr, chip_id_t src_device_id, uint16_t channel) const override {
        printf("Not implemented\n");
    }

    int get_device_aiclk(const chip_id_t& chip_id) const override {
        printf("Not implemented\n");
        return 0;
    }

    void dram_barrier(chip_id_t chip_id) const override {
        printf("Not implemented\n");
    }
    void l1_barrier(chip_id_t chip_id) const override {
        printf("Not implemented\n");
    }

    uint32_t get_num_host_channels(chip_id_t device_id) const override {
        printf("Not implemented\n");
        return 0;
    }
    uint32_t get_host_channel_size(chip_id_t device_id, uint32_t channel) const override {
        printf("Not implemented\n");
        return 0;
    }
    // Returns address in host space
    void* host_dma_address(uint64_t offset, chip_id_t src_device_id, uint16_t channel) const override {
        printf("Not implemented\n");
        return nullptr;
    }
    uint64_t get_pcie_base_addr_from_device(chip_id_t chip_id) const override {
        printf("Not implemented\n");
        return 0;
    }

    // Ethernet cluster api
    // Returns set of device ids connected via ethernet
    std::unordered_set<chip_id_t> get_ethernet_connected_device_ids(chip_id_t chip_id) const override {
        printf("Not implemented\n");
        return {};
    }

    // Returns whether `logical_core` has an eth link to a core on a connected chip
    // Cores that connect to another cluster will show up as connected
    bool is_ethernet_link_up(chip_id_t chip_id, const CoreCoord& logical_core) const override {
        printf("Not implemented\n");
        return false;
    }

    // Returns connected ethernet core on the other chip
    // If the core is connected to a device not accessible through this Cluster, it will assert
    std::tuple<chip_id_t, CoreCoord> get_connected_ethernet_core(std::tuple<chip_id_t, CoreCoord> eth_core) const override {
        printf("Not implemented\n");
        return std::make_tuple(0, CoreCoord{0, 0});
    }

    // Returns connected ethernet core on the other chip that is not managed by this Cluster
    std::tuple<uint64_t, CoreCoord> get_connected_ethernet_core_to_remote_mmio_device(
        std::tuple<chip_id_t, CoreCoord> eth_core) const override {
        printf("Not implemented\n");
        return std::make_tuple(0, CoreCoord{0, 0});
    }

    // Returns a ethernet sockets between local chip and remote chip
    // get_ethernet_sockets(a, b)[0] is connected to get_ethernet_sockets(b, a)[0]
    std::vector<CoreCoord> get_ethernet_sockets(chip_id_t local_chip, chip_id_t remote_chip) const override {
        printf("Not implemented\n");
        return {};
    }
    // Converts logical ethernet core coord to physical ethernet core coord
    CoreCoord ethernet_core_from_logical_core(chip_id_t chip_id, const CoreCoord& logical_core) const override {
        printf("Not implemented\n");
        return CoreCoord{0, 0};
    }

    // Returns virtual eth coord from channel
    CoreCoord get_virtual_eth_core_from_channel(chip_id_t chip_id, int channel) const override {
        printf("Not implemented\n");
        return CoreCoord{0, 0};
    }

    // Internal routing for SD and FD enables launching user ethernet kernels and FD tunneling for all devices in the
    // cluster. When using multiple devices in a cluster, this should be the flow:
    //       CreateDevice(0)
    //       CreateDevice(1)
    //       set_internal_routing_info_for_ethernet_cores(true);
    //       set_internal_routing_info_for_ethernet_cores(false);
    //       CloseDevice(0)
    //       CloseDevice(1)
    void set_internal_routing_info_for_ethernet_cores(
        bool enable_internal_routing, const std::vector<chip_id_t>& target_mmio_devices = {}) const override {
        printf("Not implemented\n");
    }

    const std::
        unordered_map<chip_id_t, std::unordered_map<ethernet_channel_t, std::tuple<chip_id_t, ethernet_channel_t>>>&
        get_ethernet_connections() const override {
        return this->cluster_desc_->get_ethernet_connections();
    }

    // TODO: unify uint64_t with ChipUID
    const std::
        unordered_map<chip_id_t, std::unordered_map<ethernet_channel_t, std::tuple<uint64_t, ethernet_channel_t>>>&
        get_ethernet_connections_to_remote_devices() const override {
        return this->cluster_desc_->get_ethernet_connections_to_remote_devices();
    }

    // Returns MMIO device ID (logical) that controls given `device_id`. If `device_id` is MMIO device it is returned.
    chip_id_t get_associated_mmio_device(chip_id_t device_id) const override {
        return this->cluster_desc_->get_closest_mmio_capable_chip(device_id);
    }

    uint16_t get_assigned_channel_for_device(chip_id_t device_id) const override {
        printf("Not implemented\n");
        return 0;
    }

    // Returns collection of devices that are controlled by the specified MMIO device inclusive of the MMIO device
    const std::unordered_set<chip_id_t>& get_devices_controlled_by_mmio_device(chip_id_t mmio_device_id) const override {
        TT_ASSERT(
            this->cluster_desc_->get_chips_grouped_by_closest_mmio().count(mmio_device_id),
            "Expected device {} to be an MMIO device!",
            mmio_device_id);
        return this->cluster_desc_->get_chips_grouped_by_closest_mmio().at(mmio_device_id);
    }

    // Returns map of connected chip ids to active ethernet cores
    std::unordered_map<chip_id_t, std::vector<CoreCoord>> get_ethernet_cores_grouped_by_connected_chips(
        chip_id_t chip_id) const override {
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

    // Returns vector of unique tunnels originating from mmio device.
    // Each vector entry is another vector of remote devices on that tunnel.
    std::vector<std::vector<chip_id_t>> get_tunnels_from_mmio_device(chip_id_t mmio_chip_id) const override {
        printf("Not implemented\n");
        return {};
    }

    // Configures ethernet cores for fabric routers depending on whether fabric is enabled
    void configure_ethernet_cores_for_fabric_routers(
        tt_fabric::FabricConfig fabric_config, std::optional<uint8_t> num_routing_planes = std::nullopt) override {
        printf("Not implemented\n");
    }

    // Returns whether we are running on Galaxy.
    bool is_galaxy_cluster() const override {
        printf("Not implemented\n");
        return false;
    }

    // Returns Wormhole chip board type.
    BoardType get_board_type(chip_id_t chip_id) const override {
  return this->cluster_desc_->get_board_type(chip_id);
    }

    tt::tt_metal::ClusterType get_cluster_type() const override {
        printf("Not implemented\n");
        return tt::tt_metal::ClusterType::N300;
    }

    bool is_base_routing_fw_enabled() const override {
        printf("Not implemented\n");
        return false;
    }

    // Get all fabric ethernet cores
    std::set<tt_fabric::chan_id_t> get_fabric_ethernet_channels(chip_id_t chip_id) const override {
        printf("Not implemented\n");
        return {};
    }

    // Get fabric ethernet cores connecting src to dst
    std::vector<CoreCoord> get_fabric_ethernet_routers_between_src_and_dest(chip_id_t src_id, chip_id_t dst_id) const override {
        printf("Not implemented\n");
        return {};
    }

    bool is_worker_core(const CoreCoord& core, chip_id_t chip_id) const override {
        printf("Not implemented\n");
        return false;
    }
    bool is_ethernet_core(const CoreCoord& core, chip_id_t chip_id) const override {
        printf("Not implemented\n");
        return false;
    }
    CoreCoord get_logical_ethernet_core_from_virtual(chip_id_t chip, CoreCoord core) const override {
        printf("Not implemented\n");
        return CoreCoord{0, 0};
    }

    // These two functions should be removed in favor of direct translation.
    std::unordered_map<int, int> get_worker_logical_to_virtual_x(chip_id_t chip_id) const override {
        printf("Not implemented\n");
        return {};
    }
    std::unordered_map<int, int> get_worker_logical_to_virtual_y(chip_id_t chip_id) const override {
        printf("Not implemented\n");
        return {};
    }

    const std::unordered_map<CoreCoord, int32_t>& get_virtual_routing_to_profiler_flat_id(chip_id_t chip_id) const override {
        printf("Not implemented\n");
        static std::unordered_map<CoreCoord, int32_t> dummy_map;
        return dummy_map;
    }

    std::uint32_t get_ubb_asic_id(chip_id_t physical_chip_id) const override {
        printf("Not implemented\n");
        return 0;
    }

    // TODO: move to separate system descriptor class
    // return enum for connection type, Internal, QSFP, Other, Unknown
    bool is_external_cable(chip_id_t physical_chip_id, CoreCoord eth_core) const override {
        printf("Not implemented\n");
        return false;
    }

    const std::unordered_set<CoreCoord>& get_eth_cores_with_frequent_retraining(chip_id_t chip_id) const override {
        printf("Not implemented\n");
        static std::unordered_set<CoreCoord> dummy_set;
        return dummy_set;
    }

    const std::unordered_map<CoreCoord, EthRouterMode>& get_eth_routing_info(chip_id_t chip_id) const override {
        printf("Not implemented\n");
        static std::unordered_map<CoreCoord, EthRouterMode> dummy_map;
        return dummy_map;
    }

private:
    std::unique_ptr<tt::umd::Cluster> driver_;
    tt_ClusterDescriptor* cluster_desc_ = nullptr;
    tt::tt_metal::ClusterType cluster_type_ = tt::tt_metal::ClusterType::INVALID;
    tt::llrt::RunTimeOptions rtoptions_;
    std::unordered_map<chip_id_t, metal_SocDescriptor> sdesc_per_chip_;
};

TEST_F(ClusterFixture, TestCustomCluster) {
    std::unique_ptr<tt_ClusterDescriptor> cluster_desc =
        tt_ClusterDescriptor::create_from_yaml("./t3k_cluster_desc.yaml");

   tt::tt_metal::MetalContext::set_default_cluster(std::make_unique<MockCluster>(cluster_desc.get()));

    tt::tt_metal::MetalContext::instance();

    const std::filesystem::path mesh_graph_desc_path =
        std::filesystem::path(tt::tt_metal::MetalContext::instance().rtoptions().get_root_dir()) /
        "tests/tt_metal/tt_fabric/custom_mesh_descriptors/t3k_2x2_mesh_graph_descriptor.yaml";

    auto control_plane = std::make_unique<tt::tt_fabric::ControlPlane>(mesh_graph_desc_path.string());
}

}  // namespace mock_cluster_tests
}  // namespace tt::tt_fabric
