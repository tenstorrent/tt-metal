// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "dummy_device_impl.hpp"

#include <tt_stl/assert.hpp>
#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/hal_types.hpp>
#include <tt-metalium/sub_device_types.hpp>
#include <hostdevcommon/common_values.hpp>

namespace tt::tt_metal {

DummyDevice::DummyDevice(ChipId device_id, tt::ARCH arch) :
    device_id_(device_id), arch_(arch), initialized_(false), num_hw_cqs_(1), trace_buffers_size_(0) {}

DummyDevice::~DummyDevice() = default;

tt::ARCH DummyDevice::arch() const { return arch_; }

ChipId DummyDevice::id() const { return device_id_; }

ChipId DummyDevice::build_id() const { return device_id_; }

uint8_t DummyDevice::num_hw_cqs() const { return num_hw_cqs_; }

bool DummyDevice::is_initialized() const { return initialized_; }

int DummyDevice::num_dram_channels() const { return 0; }

uint32_t DummyDevice::l1_size_per_core() const { return 0; }

uint32_t DummyDevice::dram_size_per_channel() const { return 0; }

CoreCoord DummyDevice::grid_size() const { return CoreCoord{0, 0}; }

CoreCoord DummyDevice::logical_grid_size() const { return CoreCoord{0, 0}; }

CoreCoord DummyDevice::dram_grid_size() const { return CoreCoord{0, 0}; }

CoreCoord DummyDevice::virtual_noc0_coordinate(uint8_t noc_index, CoreCoord coord) const {
    (void)noc_index;
    return coord;
}

std::vector<CoreCoord> DummyDevice::worker_cores_from_logical_cores(const std::vector<CoreCoord>& logical_cores) const {
    return logical_cores;
}

std::vector<CoreCoord> DummyDevice::ethernet_cores_from_logical_cores(
    const std::vector<CoreCoord>& logical_cores) const {
    return logical_cores;
}

std::vector<CoreCoord> DummyDevice::get_optimal_dram_bank_to_logical_worker_assignment(NOC noc) {
    (void)noc;
    return optimal_dram_bank_to_logical_worker_assignment_;
}

CoreCoord DummyDevice::virtual_core_from_logical_core(const CoreCoord& logical_coord, const CoreType& core_type) const {
    (void)core_type;
    return logical_coord;
}

CoreCoord DummyDevice::worker_core_from_logical_core(const CoreCoord& logical_core) const { return logical_core; }

CoreCoord DummyDevice::ethernet_core_from_logical_core(const CoreCoord& logical_core) const { return logical_core; }

CoreCoord DummyDevice::logical_core_from_ethernet_core(const CoreCoord& ethernet_core) const { return ethernet_core; }

std::unordered_set<CoreCoord> DummyDevice::get_active_ethernet_cores(bool skip_reserved_tunnel_cores) const {
    (void)skip_reserved_tunnel_cores;
    return std::unordered_set<CoreCoord>();
}

std::unordered_set<CoreCoord> DummyDevice::get_inactive_ethernet_cores() const {
    return std::unordered_set<CoreCoord>();
}

bool DummyDevice::is_active_ethernet_core(CoreCoord logical_core, bool skip_reserved_tunnel_cores) const {
    (void)logical_core;
    (void)skip_reserved_tunnel_cores;
    return false;
}

std::tuple<ChipId, CoreCoord> DummyDevice::get_connected_ethernet_core(CoreCoord eth_core) const {
    (void)eth_core;
    return std::make_tuple(0, CoreCoord{0, 0});
}

std::vector<CoreCoord> DummyDevice::get_ethernet_sockets(ChipId connected_chip_id) const {
    (void)connected_chip_id;
    return std::vector<CoreCoord>();
}

bool DummyDevice::is_inactive_ethernet_core(CoreCoord logical_core) const {
    (void)logical_core;
    return true;
}

// TODO(p1-0tr): figure out a sensible default for dummy device
CoreCoord DummyDevice::compute_with_storage_grid_size() const { return CoreCoord{2, 2}; }

CoreRangeSet DummyDevice::worker_cores(HalProgrammableCoreType core_type, SubDeviceId sub_device_id) const {
    (void)core_type;
    (void)sub_device_id;
    return CoreRangeSet();
}

uint32_t DummyDevice::num_worker_cores(HalProgrammableCoreType core_type, SubDeviceId sub_device_id) const {
    (void)core_type;
    (void)sub_device_id;
    return 0;
}

const std::unique_ptr<Allocator>& DummyDevice::allocator() const {
    TT_THROW("allocator() not supported for DummyDevice");
}

const std::unique_ptr<Allocator>& DummyDevice::allocator(SubDeviceId sub_device_id) const {
    (void)sub_device_id;
    TT_THROW("allocator(SubDeviceId) not supported for DummyDevice");
}

const std::unique_ptr<AllocatorImpl>& DummyDevice::allocator_impl() const {
    TT_THROW("allocator_impl() not supported for DummyDevice");
}

const std::unique_ptr<AllocatorImpl>& DummyDevice::allocator_impl(SubDeviceId sub_device_id) const {
    (void)sub_device_id;
    TT_THROW("allocator_impl(SubDeviceId) not supported for DummyDevice");
}

CoreCoord DummyDevice::logical_core_from_dram_channel(uint32_t dram_channel) const {
    (void)dram_channel;
    return CoreCoord{0, 0};
}

uint32_t DummyDevice::dram_channel_from_logical_core(const CoreCoord& logical_core) const {
    (void)logical_core;
    return 0;
}

uint32_t DummyDevice::dram_channel_from_virtual_core(const CoreCoord& virtual_core) const {
    (void)virtual_core;
    return 0;
}

std::optional<DeviceAddr> DummyDevice::lowest_occupied_compute_l1_address() const { return std::nullopt; }

std::optional<DeviceAddr> DummyDevice::lowest_occupied_compute_l1_address(
    tt::stl::Span<const SubDeviceId> sub_device_ids) const {
    (void)sub_device_ids;
    return std::nullopt;
}

const std::set<CoreCoord>& DummyDevice::ethernet_cores() const { return ethernet_cores_; }

const std::set<CoreCoord>& DummyDevice::storage_only_cores() const { return storage_only_cores_; }

uint32_t DummyDevice::get_noc_unicast_encoding(uint8_t noc_index, const CoreCoord& core) const {
    (void)noc_index;
    (void)core;
    return 0;
}

uint32_t DummyDevice::get_noc_multicast_encoding(uint8_t noc_index, const CoreRange& cores) const {
    (void)noc_index;
    (void)cores;
    return 0;
}

SystemMemoryManager& DummyDevice::sysmem_manager() { TT_THROW("sysmem_manager() not supported for DummyDevice"); }

uint32_t DummyDevice::get_trace_buffers_size() const { return trace_buffers_size_; }

void DummyDevice::set_trace_buffers_size(uint32_t size) { trace_buffers_size_ = size; }

bool DummyDevice::initialize(
    uint8_t num_hw_cqs,
    size_t l1_small_size,
    size_t trace_region_size,
    size_t worker_l1_size,
    tt::stl::Span<const std::uint32_t> l1_bank_remap,
    bool minimal) {
    (void)l1_small_size;
    (void)trace_region_size;
    (void)worker_l1_size;
    (void)l1_bank_remap;
    (void)minimal;
    num_hw_cqs_ = num_hw_cqs;
    initialized_ = true;
    return true;
}

void DummyDevice::init_command_queue_host() {
    // No-op for dummy device
}

void DummyDevice::init_command_queue_device() {
    // No-op for dummy device
}

bool DummyDevice::compile_fabric() { return true; }

void DummyDevice::configure_fabric() {
    // No-op for dummy device
}

void DummyDevice::init_fabric() {
    // No-op for dummy device
}

bool DummyDevice::close() {
    initialized_ = false;
    return true;
}

void DummyDevice::enable_program_cache() { program_cache_.enable(); }

void DummyDevice::clear_program_cache() { program_cache_.clear(); }

void DummyDevice::disable_and_clear_program_cache() {
    program_cache_.disable();
    program_cache_.clear();
}

program_cache::detail::ProgramCache& DummyDevice::get_program_cache() { return program_cache_; }

std::size_t DummyDevice::num_program_cache_entries() { return program_cache_.num_entries(); }

HalProgrammableCoreType DummyDevice::get_programmable_core_type(CoreCoord virtual_core) const {
    (void)virtual_core;
    return HalProgrammableCoreType::TENSIX;
}

HalMemType DummyDevice::get_mem_type_of_core(CoreCoord virtual_core) const {
    (void)virtual_core;
    return HalMemType::L1;
}

bool DummyDevice::has_noc_mcast_txns(SubDeviceId sub_device_id) const {
    (void)sub_device_id;
    return false;
}

uint8_t DummyDevice::num_noc_unicast_txns(SubDeviceId sub_device_id) const {
    (void)sub_device_id;
    return 0;
}

uint8_t DummyDevice::noc_data_start_index(SubDeviceId sub_device_id, bool unicast_data) const {
    (void)sub_device_id;
    (void)unicast_data;
    return 0;
}

SubDeviceManagerId DummyDevice::get_active_sub_device_manager_id() const { return SubDeviceManagerId{0}; }

SubDeviceManagerId DummyDevice::get_default_sub_device_manager_id() const { return SubDeviceManagerId{0}; }

SubDeviceManagerId DummyDevice::create_sub_device_manager(
    tt::stl::Span<const SubDevice> sub_devices, DeviceAddr local_l1_size) {
    (void)sub_devices;
    (void)local_l1_size;
    return SubDeviceManagerId{0};
}

SubDeviceManagerId DummyDevice::create_sub_device_manager(
    std::initializer_list<SubDevice> sub_devices, DeviceAddr local_l1_size) {
    (void)sub_devices;
    (void)local_l1_size;
    return SubDeviceManagerId{0};
}

void DummyDevice::remove_sub_device_manager(SubDeviceManagerId sub_device_manager_id) {
    (void)sub_device_manager_id;
    // No-op for dummy device
}

void DummyDevice::load_sub_device_manager(SubDeviceManagerId sub_device_manager_id) {
    (void)sub_device_manager_id;
    // No-op for dummy device
}

void DummyDevice::clear_loaded_sub_device_manager() {
    // No-op for dummy device
}

CoreCoord DummyDevice::virtual_program_dispatch_core(uint8_t cq_id) const {
    (void)cq_id;
    return CoreCoord{0, 0};
}

const std::vector<SubDeviceId>& DummyDevice::get_sub_device_ids() const { return sub_device_ids_; }

const std::vector<SubDeviceId>& DummyDevice::get_sub_device_stall_group() const { return sub_device_stall_group_; }

void DummyDevice::set_sub_device_stall_group(tt::stl::Span<const SubDeviceId> sub_device_ids) {
    sub_device_stall_group_.assign(sub_device_ids.begin(), sub_device_ids.end());
}

void DummyDevice::reset_sub_device_stall_group() { sub_device_stall_group_.clear(); }

uint32_t DummyDevice::num_sub_devices() const { return static_cast<uint32_t>(sub_device_ids_.size()); }

uint32_t DummyDevice::num_virtual_eth_cores(SubDeviceId sub_device_id) {
    (void)sub_device_id;
    return 0;
}

bool DummyDevice::is_mmio_capable() const { return false; }

std::shared_ptr<distributed::MeshDevice> DummyDevice::get_mesh_device() { return nullptr; }

}  // namespace tt::tt_metal
