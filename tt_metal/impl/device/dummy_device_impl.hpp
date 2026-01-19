// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <memory>
#include <set>
#include <unordered_set>
#include <vector>

#include <tt-metalium/device.hpp>
#include <tt-metalium/program_cache.hpp>
#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/sub_device_types.hpp>
#include <hostdevcommon/common_values.hpp>
#include <tt-metalium/hal_types.hpp>

namespace tt::tt_metal {

class Allocator;
class AllocatorImpl;
class SystemMemoryManager;
namespace distributed {
class MeshDevice;
}

// A dummy device implementation for testing and development
// This device does not interact with hardware and provides default implementations
class DummyDevice : public IDevice {
public:
    DummyDevice(ChipId device_id = 0, tt::ARCH arch = tt::ARCH::Invalid);
    ~DummyDevice() override;

    DummyDevice(const DummyDevice& other) = delete;
    DummyDevice& operator=(const DummyDevice& other) = delete;

    DummyDevice(DummyDevice&& other) noexcept = default;
    DummyDevice& operator=(DummyDevice&& other) noexcept = default;

    // IDevice interface implementation
    tt::ARCH arch() const override;

    ChipId id() const override;
    ChipId build_id() const override;

    uint8_t num_hw_cqs() const override;

    bool is_initialized() const override;

    int num_dram_channels() const override;
    uint32_t l1_size_per_core() const override;
    uint32_t dram_size_per_channel() const override;
    CoreCoord grid_size() const override;
    CoreCoord logical_grid_size() const override;
    CoreCoord dram_grid_size() const override;

    CoreCoord virtual_noc0_coordinate(uint8_t noc_index, CoreCoord coord) const override;

    std::vector<CoreCoord> worker_cores_from_logical_cores(const std::vector<CoreCoord>& logical_cores) const override;
    std::vector<CoreCoord> ethernet_cores_from_logical_cores(
        const std::vector<CoreCoord>& logical_cores) const override;
    std::vector<CoreCoord> get_optimal_dram_bank_to_logical_worker_assignment(NOC noc) override;

    CoreCoord virtual_core_from_logical_core(const CoreCoord& logical_coord, const CoreType& core_type) const override;
    CoreCoord worker_core_from_logical_core(const CoreCoord& logical_core) const override;
    CoreCoord ethernet_core_from_logical_core(const CoreCoord& logical_core) const override;
    CoreCoord logical_core_from_ethernet_core(const CoreCoord& ethernet_core) const override;

    std::unordered_set<CoreCoord> get_active_ethernet_cores(bool skip_reserved_tunnel_cores = false) const override;
    std::unordered_set<CoreCoord> get_inactive_ethernet_cores() const override;
    bool is_active_ethernet_core(CoreCoord logical_core, bool skip_reserved_tunnel_cores = false) const override;
    std::tuple<ChipId, CoreCoord> get_connected_ethernet_core(CoreCoord eth_core) const override;
    std::vector<CoreCoord> get_ethernet_sockets(ChipId connected_chip_id) const override;
    bool is_inactive_ethernet_core(CoreCoord logical_core) const override;

    CoreCoord compute_with_storage_grid_size() const override;

    CoreRangeSet worker_cores(HalProgrammableCoreType core_type, SubDeviceId sub_device_id) const override;
    uint32_t num_worker_cores(HalProgrammableCoreType core_type, SubDeviceId sub_device_id) const override;

    const std::unique_ptr<Allocator>& allocator() const override;
    const std::unique_ptr<Allocator>& allocator(SubDeviceId sub_device_id) const override;
    const std::unique_ptr<AllocatorImpl>& allocator_impl() const override;
    const std::unique_ptr<AllocatorImpl>& allocator_impl(SubDeviceId sub_device_id) const override;

    CoreCoord logical_core_from_dram_channel(uint32_t dram_channel) const override;
    uint32_t dram_channel_from_logical_core(const CoreCoord& logical_core) const override;
    uint32_t dram_channel_from_virtual_core(const CoreCoord& virtual_core) const override;

    std::optional<DeviceAddr> lowest_occupied_compute_l1_address() const override;
    std::optional<DeviceAddr> lowest_occupied_compute_l1_address(
        tt::stl::Span<const SubDeviceId> sub_device_ids) const override;

    const std::set<CoreCoord>& ethernet_cores() const override;
    const std::set<CoreCoord>& storage_only_cores() const override;

    uint32_t get_noc_unicast_encoding(uint8_t noc_index, const CoreCoord& core) const override;
    uint32_t get_noc_multicast_encoding(uint8_t noc_index, const CoreRange& cores) const override;

    SystemMemoryManager& sysmem_manager() override;

    uint32_t get_trace_buffers_size() const override;
    void set_trace_buffers_size(uint32_t size) override;

    bool initialize(
        uint8_t num_hw_cqs,
        size_t l1_small_size,
        size_t trace_region_size,
        size_t worker_l1_size,
        tt::stl::Span<const std::uint32_t> l1_bank_remap = {},
        bool minimal = false) override;
    void init_command_queue_host() override;
    void init_command_queue_device() override;

    bool compile_fabric() override;
    void configure_fabric() override;
    void init_fabric() override;
    bool close() override;

    void enable_program_cache() override;
    void clear_program_cache() override;
    void disable_and_clear_program_cache() override;
    program_cache::detail::ProgramCache& get_program_cache() override;
    std::size_t num_program_cache_entries() override;

    HalProgrammableCoreType get_programmable_core_type(CoreCoord virtual_core) const override;
    HalMemType get_mem_type_of_core(CoreCoord virtual_core) const override;

    bool has_noc_mcast_txns(SubDeviceId sub_device_id) const override;
    uint8_t num_noc_unicast_txns(SubDeviceId sub_device_id) const override;
    uint8_t noc_data_start_index(SubDeviceId sub_device_id, bool unicast_data = true) const override;

    SubDeviceManagerId get_active_sub_device_manager_id() const override;
    SubDeviceManagerId get_default_sub_device_manager_id() const override;
    SubDeviceManagerId create_sub_device_manager(
        tt::stl::Span<const SubDevice> sub_devices, DeviceAddr local_l1_size) override;
    SubDeviceManagerId create_sub_device_manager(
        std::initializer_list<SubDevice> sub_devices, DeviceAddr local_l1_size) override;
    void remove_sub_device_manager(SubDeviceManagerId sub_device_manager_id) override;
    void load_sub_device_manager(SubDeviceManagerId sub_device_manager_id) override;
    void clear_loaded_sub_device_manager() override;
    CoreCoord virtual_program_dispatch_core(uint8_t cq_id) const override;
    const std::vector<SubDeviceId>& get_sub_device_ids() const override;
    const std::vector<SubDeviceId>& get_sub_device_stall_group() const override;
    void set_sub_device_stall_group(tt::stl::Span<const SubDeviceId> sub_device_ids) override;
    void reset_sub_device_stall_group() override;
    uint32_t num_sub_devices() const override;
    uint32_t num_virtual_eth_cores(SubDeviceId sub_device_id) override;

    bool is_mmio_capable() const override;

    std::shared_ptr<distributed::MeshDevice> get_mesh_device() override;

private:
    ChipId device_id_;
    tt::ARCH arch_;
    bool initialized_;
    uint8_t num_hw_cqs_;
    uint32_t trace_buffers_size_;

    std::set<CoreCoord> ethernet_cores_;
    std::set<CoreCoord> storage_only_cores_;
    std::vector<CoreCoord> optimal_dram_bank_to_logical_worker_assignment_;

    program_cache::detail::ProgramCache program_cache_;

    // Dummy objects for methods that return references
    // These will throw if accessed, as they cannot be meaningfully implemented without hardware
    mutable std::unique_ptr<Allocator> dummy_allocator_;
    mutable std::unique_ptr<AllocatorImpl> dummy_allocator_impl_;
    std::unique_ptr<SystemMemoryManager> dummy_sysmem_manager_;

    std::vector<SubDeviceId> sub_device_ids_;
    std::vector<SubDeviceId> sub_device_stall_group_;
};

}  // namespace tt::tt_metal
