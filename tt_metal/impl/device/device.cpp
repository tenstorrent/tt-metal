// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "device_impl.hpp"

#include <core_descriptor.hpp>
#include <host_api.hpp>
#include <initializer_list>
#include <sub_device.hpp>
#include <sub_device_types.hpp>
#include "impl/sub_device/sub_device_impl.hpp"
#include <tt-metalium/program_cache.hpp>
#include <tt-metalium/hal.hpp>
#include <tt_align.hpp>
#include <tt_metal.hpp>
#include <tt_stl/span.hpp>
#include <algorithm>
#include <cstdint>
#include <cstring>
#include <memory>
#include <optional>
#include <set>
#include <tuple>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "allocator.hpp"
#include <tt_stl/assert.hpp>
#include "dispatch/command_queue.hpp"
#include "dispatch/command_queue_common.hpp"
#include "common/core_assignment.hpp"
#include "program/program_impl.hpp"
#include "core_coord.hpp"
#include "device.hpp"
#include "dispatch/dispatch_settings.hpp"
#include "dispatch/dispatch_query_manager.hpp"
#include "hal_types.hpp"
#include "impl/context/metal_context.hpp"
#include "lightmetal/lightmetal_capture.hpp"
#include "llrt.hpp"
#include <tt-logger/tt-logger.hpp>
#include "llrt/metal_soc_descriptor.hpp"
#include "tt-metalium/program.hpp"
#include <tt_stl/strong_type.hpp>
#include "dispatch/system_memory_manager.hpp"
#include "tracy/Tracy.hpp"
#include "tt_metal/impl/allocator/l1_banking_allocator.hpp"
#include "tt_metal/impl/dispatch/hardware_command_queue.hpp"
#include "tt_metal/impl/dispatch/topology.hpp"
#include "tt_metal/impl/sub_device/sub_device_manager.hpp"
#include "tt_metal/fabric/fabric_init.hpp"
#include <tt-metalium/experimental/fabric/control_plane.hpp>
#include <umd/device/coordinates/coordinate_manager.hpp>
#include <umd/device/types/core_coordinates.hpp>
#include <umd/device/types/xy_pair.hpp>
#include <impl/debug/watcher_server.hpp>
#include <impl/dispatch/dispatch_mem_map.hpp>

namespace tt::tt_metal {

uint64_t IDevice::get_dev_addr(CoreCoord virtual_core, HalL1MemAddrType addr_type) const {
    return MetalContext::instance().hal().get_dev_addr(this->get_programmable_core_type(virtual_core), addr_type);
}

uint64_t IDevice::get_dev_size(CoreCoord virtual_core, HalL1MemAddrType addr_type) const {
    return MetalContext::instance().hal().get_dev_size(this->get_programmable_core_type(virtual_core), addr_type);
}

void IDevice::set_program_cache_misses_allowed(bool allowed) {
    this->get_program_cache().set_cache_misses_allowed(allowed);
}

Device::Device(Device&& other) noexcept = default;
Device& Device::operator=(Device&& other) noexcept = default;

Device::Device(
    ChipId device_id,
    const uint8_t num_hw_cqs,
    size_t l1_small_size,
    size_t trace_region_size,
    tt::stl::Span<const std::uint32_t> l1_bank_remap,
    bool minimal,
    uint32_t /*worker_thread_core*/,
    uint32_t completion_queue_reader_core,
    size_t worker_l1_size) :
    id_(device_id), completion_queue_reader_core_(completion_queue_reader_core) {
    ZoneScoped;
    this->initialize(num_hw_cqs, l1_small_size, trace_region_size, worker_l1_size, l1_bank_remap, minimal);
}

std::unordered_set<CoreCoord> Device::get_active_ethernet_cores(bool skip_reserved_tunnel_cores) const {
    return tt::tt_metal::MetalContext::instance().get_control_plane().get_active_ethernet_cores(
        this->id_, skip_reserved_tunnel_cores);
}

bool Device::is_active_ethernet_core(CoreCoord logical_core, bool skip_reserved_tunnel_cores) const {
    auto active_ethernet_cores = this->get_active_ethernet_cores(skip_reserved_tunnel_cores);
    return active_ethernet_cores.contains(logical_core);
}

std::unordered_set<CoreCoord> Device::get_inactive_ethernet_cores() const {
    return tt::tt_metal::MetalContext::instance().get_control_plane().get_inactive_ethernet_cores(this->id_);
}

bool Device::is_inactive_ethernet_core(CoreCoord logical_core) const {
    auto inactive_ethernet_cores =
        tt::tt_metal::MetalContext::instance().get_control_plane().get_inactive_ethernet_cores(this->id_);
    return inactive_ethernet_cores.contains(logical_core);
}

uint32_t Device::num_virtual_eth_cores(SubDeviceId sub_device_id) {
    return this->num_worker_cores(HalProgrammableCoreType::ACTIVE_ETH, sub_device_id);
}

std::tuple<ChipId, CoreCoord> Device::get_connected_ethernet_core(CoreCoord eth_core) const {
    return tt::tt_metal::MetalContext::instance().get_cluster().get_connected_ethernet_core(
        std::make_tuple(this->id_, eth_core));
}

std::vector<CoreCoord> Device::get_ethernet_sockets(ChipId connected_chip_id) const {
    return tt::tt_metal::MetalContext::instance().get_cluster().get_ethernet_sockets(this->id_, connected_chip_id);
}

bool Device::is_mmio_capable() const {
    return tt::tt_metal::MetalContext::instance().get_cluster().get_associated_mmio_device(this->id_) == this->id_;
}

CoreRangeSet Device::worker_cores(HalProgrammableCoreType /*core_type*/, SubDeviceId /*sub_device_id*/) const {
    TT_FATAL(false, "worker_cores is deprecated for device");
    return CoreRangeSet{};
}

uint32_t Device::num_worker_cores(HalProgrammableCoreType /*core_type*/, SubDeviceId /*sub_device_id*/) const {
    TT_FATAL(false, "num_worker_cores is deprecated for device");
    return 0U;
}

std::unique_ptr<AllocatorImpl> Device::initialize_allocator(
    size_t l1_small_size,
    size_t trace_region_size,
    size_t worker_l1_unreserved_start,
    tt::stl::Span<const std::uint32_t> l1_bank_remap) {
    ZoneScoped;
    const metal_SocDescriptor& soc_desc = tt::tt_metal::MetalContext::instance().get_cluster().get_soc_desc(this->id_);
    auto config = L1BankingAllocator::generate_config(
        this->id(),
        this->num_hw_cqs(),
        l1_small_size,
        trace_region_size,
        worker_l1_unreserved_start,
        {l1_bank_remap.begin(), l1_bank_remap.end()});

    for (const tt::umd::CoreCoord& core : soc_desc.get_cores(CoreType::ETH, CoordSystem::LOGICAL)) {
        this->ethernet_cores_.insert({core.x, core.y});
    }

    // L1 Banking Allocator creates 1 bank per DRAM core and splits up L1 such that there are power 2 num L1 banks
    // This is the only allocator scheme supported because kernel APIs assume num L1 banks are power of 2
    return std::make_unique<L1BankingAllocator>(config);
}

// Writes issue and completion queue pointers to device and in sysmem and loads fast dispatch program onto dispatch
// cores
void Device::configure_command_queue_programs() {
    ChipId device_id = this->id();
    ChipId mmio_device_id = tt::tt_metal::MetalContext::instance().get_cluster().get_associated_mmio_device(device_id);

    std::vector<uint32_t> zero = {0x0};  // Reset state in case L1 Clear is disabled.
    std::vector<uint32_t> pointers;
    uint32_t cq_size = this->sysmem_manager().get_cq_size();
    TT_ASSERT(this->command_queue_programs_.size() == 1);

    Program& command_queue_program = *this->command_queue_programs_[0];
    uint8_t num_hw_cqs = this->num_hw_cqs();

    // Reset host-side command queue pointers for all channels controlled by this mmio device
    if (this->is_mmio_capable()) {
        for (ChipId serviced_device_id :
             tt::tt_metal::MetalContext::instance().get_cluster().get_devices_controlled_by_mmio_device(device_id)) {
            uint16_t channel = tt::tt_metal::MetalContext::instance().get_cluster().get_assigned_channel_for_device(
                serviced_device_id);
            uint32_t host_issue_q_rd_ptr = MetalContext::instance().dispatch_mem_map().get_host_command_queue_addr(
                CommandQueueHostAddrType::ISSUE_Q_RD);
            uint32_t host_issue_q_wr_ptr = MetalContext::instance().dispatch_mem_map().get_host_command_queue_addr(
                CommandQueueHostAddrType::ISSUE_Q_WR);
            uint32_t host_completion_q_wr_ptr = MetalContext::instance().dispatch_mem_map().get_host_command_queue_addr(
                CommandQueueHostAddrType::COMPLETION_Q_WR);
            uint32_t host_completion_q_rd_ptr = MetalContext::instance().dispatch_mem_map().get_host_command_queue_addr(
                CommandQueueHostAddrType::COMPLETION_Q_RD);
            uint32_t cq_start = MetalContext::instance().dispatch_mem_map().get_host_command_queue_addr(
                CommandQueueHostAddrType::UNRESERVED);
            pointers.resize(cq_start / sizeof(uint32_t));
            for (uint8_t cq_id = 0; cq_id < num_hw_cqs; cq_id++) {
                // Reset the host manager's pointer for this command queue
                this->sysmem_manager_->reset(cq_id);

                pointers[host_issue_q_rd_ptr / sizeof(uint32_t)] =
                    (cq_start + get_absolute_cq_offset(channel, cq_id, cq_size)) >> 4;
                pointers[host_issue_q_wr_ptr / sizeof(uint32_t)] =
                    (cq_start + get_absolute_cq_offset(channel, cq_id, cq_size)) >> 4;
                pointers[host_completion_q_wr_ptr / sizeof(uint32_t)] =
                    (cq_start + this->sysmem_manager_->get_issue_queue_size(cq_id) +
                     get_absolute_cq_offset(channel, cq_id, cq_size)) >>
                    4;
                pointers[host_completion_q_rd_ptr / sizeof(uint32_t)] =
                    (cq_start + this->sysmem_manager_->get_issue_queue_size(cq_id) +
                     get_absolute_cq_offset(channel, cq_id, cq_size)) >>
                    4;

                tt::tt_metal::MetalContext::instance().get_cluster().write_sysmem(
                    pointers.data(),
                    pointers.size() * sizeof(uint32_t),
                    get_absolute_cq_offset(channel, cq_id, cq_size),
                    mmio_device_id,
                    get_umd_channel(channel));
            }
        }
    }

    // Write device-side cq pointers
    configure_dispatch_cores(this);

    // Run the cq program
    command_queue_program.impl().finalize_offsets(this);
    detail::ConfigureDeviceWithProgram(this, command_queue_program, true);
    tt::tt_metal::MetalContext::instance().get_cluster().l1_barrier(this->id());
}

void Device::init_command_queue_host() {
    // SystemMemoryManager now has internal stubs for mock devices
    sysmem_manager_ = std::make_unique<SystemMemoryManager>(this->id_, this->num_hw_cqs());

    // For mock devices, skip HWCommandQueue creation (they don't need real command queues)
    if (tt::tt_metal::MetalContext::instance().get_cluster().get_target_device_type() == tt::TargetDevice::Mock) {
        return;
    }

    auto cq_shared_state = std::make_shared<CQSharedState>();
    cq_shared_state->sub_device_cq_owner.resize(1);
    command_queues_.reserve(num_hw_cqs());
    for (size_t cq_id = 0; cq_id < num_hw_cqs(); cq_id++) {
        command_queues_.push_back(
            std::make_unique<HWCommandQueue>(
                this, cq_shared_state, cq_id, k_dispatch_downstream_noc, completion_queue_reader_core_));
    }
}

void Device::init_command_queue_device() {
    this->command_queue_programs_.push_back(get_compiled_cq_program(this));
    TT_ASSERT(this->command_queue_programs_.size() == 1);
    this->configure_command_queue_programs();
    Program& command_queue_program = *this->command_queue_programs_[0];

    // Write 0 to all workers launch message read pointer. Need to do this since dispatch cores are written new on each
    // Device init. TODO: remove this once dispatch init moves to one-shot.
    auto reset_launch_message_rd_ptr = [&](const CoreCoord& logical_core, const CoreType& core_type) {
        CoreCoord virtual_core = MetalContext::instance().get_cluster().get_virtual_coordinate_from_logical_coordinates(
            id_, logical_core, core_type);
        auto programmable_core_type = get_programmable_core_type(virtual_core);
        uint64_t launch_msg_buffer_read_ptr_addr = MetalContext::instance().hal().get_dev_addr(
            programmable_core_type, HalL1MemAddrType::LAUNCH_MSG_BUFFER_RD_PTR);
        uint32_t zero = 0;
        MetalContext::instance().get_cluster().write_core(
            &zero, sizeof(uint32_t), tt_cxy_pair(id_, virtual_core), launch_msg_buffer_read_ptr_addr);
    };
    auto reset_go_message_index = [&](const CoreCoord& logical_core, const CoreType& core_type) {
        CoreCoord virtual_core = MetalContext::instance().get_cluster().get_virtual_coordinate_from_logical_coordinates(
            id_, logical_core, core_type);
        auto programmable_core_type = get_programmable_core_type(virtual_core);
        uint32_t go_message_addr =
            MetalContext::instance().hal().get_dev_addr(programmable_core_type, HalL1MemAddrType::GO_MSG);
        uint32_t zero = 0;
        MetalContext::instance().get_cluster().write_core(
            &zero, sizeof(uint32_t), tt_cxy_pair(id_, virtual_core), go_message_addr);
        tt::tt_metal::MetalContext::instance().get_cluster().l1_barrier(id_);
        uint32_t go_message_index_addr =
            MetalContext::instance().hal().get_dev_addr(programmable_core_type, HalL1MemAddrType::GO_MSG_INDEX);
        MetalContext::instance().get_cluster().write_core(
            &zero, sizeof(uint32_t), tt_cxy_pair(id_, virtual_core), go_message_index_addr);
    };
    std::optional<std::unique_lock<std::mutex>> watcher_lock;
    if (tt::tt_metal::MetalContext::instance().rtoptions().get_watcher_enabled()) {
        watcher_lock = MetalContext::instance().watcher_server()->get_lock();
    }
    for (uint32_t y = 0; y < logical_grid_size().y; y++) {
        for (uint32_t x = 0; x < logical_grid_size().x; x++) {
            CoreCoord logical_core(x, y);
            reset_launch_message_rd_ptr(logical_core, CoreType::WORKER);
            reset_go_message_index(logical_core, CoreType::WORKER);
        }
    }
    for (const auto& logical_core : this->get_active_ethernet_cores()) {
        if (!has_flag(MetalContext::instance().get_fabric_manager(), tt_fabric::FabricManagerMode::INIT_FABRIC)) {
            continue;
        }
        reset_launch_message_rd_ptr(logical_core, CoreType::ETH);
    }
    for (const auto& logical_core : this->get_inactive_ethernet_cores()) {
        if (!has_flag(MetalContext::instance().get_fabric_manager(), tt_fabric::FabricManagerMode::INIT_FABRIC)) {
            continue;
        }
        reset_launch_message_rd_ptr(logical_core, CoreType::ETH);
    }
    if (watcher_lock) {
        watcher_lock.value().unlock();
    }

    std::vector<std::vector<CoreCoord>> logical_cores = command_queue_program.impl().logical_cores();
    const auto& hal = MetalContext::instance().hal();
    for (uint32_t index = 0; index < hal.get_programmable_core_type_count(); index++) {
        const auto& logical_dispatch_cores = logical_cores[index];
        CoreType core_type = hal.get_core_type(index);
        for (const CoreCoord& logical_dispatch_core : logical_dispatch_cores) {
            const auto* kernel = command_queue_program.impl().kernels_on_core(logical_dispatch_core, index);
            dev_msgs::launch_msg_t msg = kernel->launch_msg;  // copy
            dev_msgs::go_msg_t::ConstView go_msg = kernel->go_msg.view();
            CoreCoord virtual_core = this->virtual_core_from_logical_core(logical_dispatch_core, core_type);
            tt::llrt::write_launch_msg_to_core(
                this->id(),
                virtual_core,
                msg.view(),
                go_msg,
                this->get_dev_addr(virtual_core, HalL1MemAddrType::LAUNCH));
        }
    }

    // Precompute NOC data for go signals and set on dispatch command queues
    const auto& active_eth_cores = get_active_ethernet_cores(true);
    std::vector<CoreRange> active_eth_core_ranges;
    active_eth_core_ranges.reserve(active_eth_cores.size());
    for (const auto& core : active_eth_cores) {
        active_eth_core_ranges.emplace_back(core, core);
    }

    const NOC noc_index = MetalContext::instance().get_dispatch_query_manager().go_signal_noc();
    uint32_t idx = 0U;
    vector_aligned<uint32_t> noc_mcast_unicast_data;
    for (uint32_t i = 0U; i < num_sub_devices(); ++i) {
        for (const auto& core_range : active_eth_core_ranges) {
            noc_mcast_unicast_data.resize(idx + core_range.size());
            for (const auto& core : core_range) {
                const auto virtual_core = virtual_core_from_logical_core(core, CoreType::ETH);
                noc_mcast_unicast_data[idx++] = get_noc_unicast_encoding(noc_index, virtual_core);
            }
        }
    }

    // Set num_worker_sems and go_signal_noc_data on dispatch for the default sub device config
    for (auto& hw_cq : this->command_queues_) {
        hw_cq->set_go_signal_noc_data_and_dispatch_sems(num_sub_devices(), noc_mcast_unicast_data);
    }
}

bool Device::compile_fabric() {
    fabric_program_ = tt::tt_fabric::create_and_compile_fabric_program(this);
    return fabric_program_ != nullptr;
}

void Device::configure_fabric() {
    if (fabric_program_ == nullptr) {
        return;
    }

    tt::tt_fabric::configure_fabric_cores(this);

    fabric_program_->impl().finalize_offsets(this);

    detail::WriteRuntimeArgsToDevice(this, *fabric_program_, using_fast_dispatch_);
    detail::ConfigureDeviceWithProgram(this, *fabric_program_, using_fast_dispatch_);

    // Note: the l1_barrier below is needed to be sure writes to cores that
    // don't get the GO mailbox have all landed
    tt::tt_metal::MetalContext::instance().get_cluster().l1_barrier(this->id());
    std::vector<std::vector<CoreCoord>> logical_cores_used_in_program = fabric_program_->impl().logical_cores();
    const auto& hal = MetalContext::instance().hal();
    for (uint32_t programmable_core_type_index = 0; programmable_core_type_index < logical_cores_used_in_program.size();
         programmable_core_type_index++) {
        CoreType core_type = hal.get_core_type(programmable_core_type_index);
        for (const auto& logical_core : logical_cores_used_in_program[programmable_core_type_index]) {
            auto* kg = fabric_program_->impl().kernels_on_core(logical_core, programmable_core_type_index);
            dev_msgs::launch_msg_t::View msg = kg->launch_msg.view();
            dev_msgs::go_msg_t::ConstView go_msg = kg->go_msg.view();
            msg.kernel_config().host_assigned_id() = fabric_program_->get_runtime_id();

            auto physical_core = this->virtual_core_from_logical_core(logical_core, core_type);
            tt::llrt::write_launch_msg_to_core(
                this->id(), physical_core, msg, go_msg, this->get_dev_addr(physical_core, HalL1MemAddrType::LAUNCH));
        }
    }
    log_info(tt::LogMetal, "Fabric initialized on Device {}", this->id_);
}

// backward compatibility
void Device::init_fabric() {
    this->compile_fabric();
    this->configure_fabric();
}

bool Device::initialize(
    const uint8_t num_hw_cqs,
    size_t l1_small_size,
    size_t trace_region_size,
    size_t worker_l1_size,
    tt::stl::Span<const std::uint32_t> l1_bank_remap,
    bool minimal) {
    ZoneScoped;
    // Every initialization call should enable program cache
    this->program_cache_.enable();
    log_debug(
        tt::LogMetal,
        "Initializing device {}. Program cache is {}enabled",
        this->id_,
        this->program_cache_.is_enabled() ? "" : "NOT ");
    log_debug(tt::LogMetal, "Running with {} cqs ", num_hw_cqs);
    TT_FATAL(
        num_hw_cqs > 0 and num_hw_cqs <= dispatch_core_manager::MAX_NUM_HW_CQS,
        "num_hw_cqs can be between 1 and {}",
        dispatch_core_manager::MAX_NUM_HW_CQS);
    using_fast_dispatch_ = MetalContext::instance().rtoptions().get_fast_dispatch();
    num_hw_cqs_ = num_hw_cqs;
    const auto& hal = MetalContext::instance().hal();
    if (worker_l1_size == 0) {
        worker_l1_size = hal.get_dev_size(HalProgrammableCoreType::TENSIX, HalL1MemAddrType::DEFAULT_UNRESERVED);
    }

    size_t max_worker_l1_size = hal.get_dev_addr(HalProgrammableCoreType::TENSIX, HalL1MemAddrType::BASE) +
                                hal.get_dev_size(HalProgrammableCoreType::TENSIX, HalL1MemAddrType::BASE) -
                                hal.get_dev_addr(HalProgrammableCoreType::TENSIX, HalL1MemAddrType::KERNEL_CONFIG);

    TT_FATAL(
        worker_l1_size <= max_worker_l1_size,
        "Worker L1 size {} is larger than max size {}",
        worker_l1_size,
        max_worker_l1_size);
    log_debug(tt::LogMetal, "Worker L1 size: {} Max: {}", worker_l1_size, max_worker_l1_size);

    const uint32_t max_alignment = std::max(hal.get_alignment(HalMemType::DRAM), hal.get_alignment(HalMemType::L1));
    const uint32_t worker_l1_unreserved_start = tt::align(
        hal.get_dev_addr(HalProgrammableCoreType::TENSIX, HalL1MemAddrType::BASE) +
            hal.get_dev_size(HalProgrammableCoreType::TENSIX, HalL1MemAddrType::BASE) - worker_l1_size,
        max_alignment);
    default_allocator_ =
        initialize_allocator(l1_small_size, trace_region_size, worker_l1_unreserved_start, l1_bank_remap);

    // For minimal setup, don't initialize FW, watcher, dprint. They won't work if we're attaching to a hung chip.
    if (minimal) {
        return true;
    }

    this->initialized_ = true;

    return true;
}

bool Device::close() {
    log_trace(tt::LogMetal, "Closing device {}", this->id_);
    if (not this->initialized_) {
        TT_THROW("Cannot close device {} that has not been initialized!", this->id_);
    }

    this->disable_and_clear_program_cache();
    this->set_program_cache_misses_allowed(true);

    default_allocator_.reset();

    this->ethernet_cores_.clear();
    this->command_queue_programs_.clear();
    this->command_queues_.clear();
    this->sysmem_manager_.reset();
    this->initialized_ = false;

    return true;
}

Device::~Device() {
    log_debug(tt::LogMetal, "Device {} destructor", this->id_);
    if (this->initialized_) {
        this->close();
    }
}

tt::ARCH Device::arch() const { return tt::tt_metal::MetalContext::instance().get_cluster().arch(); }

int Device::num_dram_channels() const {
    return tt::tt_metal::MetalContext::instance().get_cluster().get_soc_desc(id_).get_num_dram_views();
}

uint32_t Device::l1_size_per_core() const {
    return tt::tt_metal::MetalContext::instance().get_cluster().get_soc_desc(id_).worker_l1_size;
}
uint32_t Device::dram_size_per_channel() const {
    return tt::tt_metal::MetalContext::instance().get_cluster().get_soc_desc(id_).dram_view_size;
}

CoreCoord Device::grid_size() const {
    return tt::tt_metal::MetalContext::instance().get_cluster().get_soc_desc(id_).grid_size;
}

CoreCoord Device::logical_grid_size() const {
    return tt::tt_metal::MetalContext::instance().get_cluster().get_soc_desc(id_).get_grid_size(CoreType::TENSIX);
}

CoreCoord Device::dram_grid_size() const {
    return tt::tt_metal::MetalContext::instance().get_cluster().get_soc_desc(id_).get_dram_grid_size();
}

CoreCoord Device::compute_with_storage_grid_size() const {
    const auto& dispatch_core_config = MetalContext::instance().get_dispatch_core_manager().get_dispatch_core_config();
    return tt::get_compute_grid_size(id_, num_hw_cqs_, dispatch_core_config);
}

CoreCoord Device::virtual_noc0_coordinate(uint8_t noc_index, CoreCoord coord) const {
    if (coord.x >= this->grid_size().x || coord.y >= this->grid_size().y || this->arch() == ARCH::BLACKHOLE) {
        // Coordinate already in virtual space: NOC0 and NOC1 are the same
        return coord;
    }
    const auto& grid_size = this->grid_size();
    // Coordinate in Physical NOC0 Space. Convert to Virtual.
    coord = this->virtual_core_from_physical_core(coord);
    // Derive virtual coord in noc_index space.
    CoreCoord virtual_coord = {
        MetalContext::instance().hal().noc_coordinate(noc_index, grid_size.x, coord.x),
        MetalContext::instance().hal().noc_coordinate(noc_index, grid_size.y, coord.y)};
    return virtual_coord;
}

CoreCoord Device::physical_worker_core_from_logical_core(const CoreCoord& logical_core) const {
    const metal_SocDescriptor& soc_desc = tt::tt_metal::MetalContext::instance().get_cluster().get_soc_desc(this->id_);
    return soc_desc.get_physical_tensix_core_from_logical(logical_core);
}

std::vector<CoreCoord> Device::worker_cores_from_logical_cores(const std::vector<CoreCoord>& logical_cores) const {
    std::vector<CoreCoord> worker_cores(logical_cores.size());
    for (std::size_t idx = 0; idx < logical_cores.size(); idx++) {
        worker_cores[idx] = this->worker_core_from_logical_core(logical_cores[idx]);
    }

    return worker_cores;
}

std::vector<CoreCoord> Device::ethernet_cores_from_logical_cores(const std::vector<CoreCoord>& logical_cores) const {
    std::vector<CoreCoord> eth_cores(logical_cores.size());
    for (std::size_t idx = 0; idx < logical_cores.size(); idx++) {
        eth_cores[idx] = this->ethernet_core_from_logical_core(logical_cores[idx]);
    }
    return eth_cores;
}

CoreCoord Device::virtual_core_from_logical_core(const CoreCoord& logical_coord, const CoreType& core_type) const {
    return tt::tt_metal::MetalContext::instance().get_cluster().get_virtual_coordinate_from_logical_coordinates(
        this->id_, logical_coord, core_type);
}

CoreCoord Device::virtual_core_from_physical_core(const CoreCoord& physical_coord) const {
    return tt::tt_metal::MetalContext::instance().get_cluster().get_virtual_coordinate_from_physical_coordinates(
        this->id_, physical_coord);
}

CoreCoord Device::worker_core_from_logical_core(const CoreCoord& logical_core) const {
    return this->virtual_core_from_logical_core(logical_core, CoreType::WORKER);
}

CoreCoord Device::ethernet_core_from_logical_core(const CoreCoord& logical_core) const {
    return this->virtual_core_from_logical_core(logical_core, CoreType::ETH);
}

CoreCoord Device::logical_core_from_ethernet_core(const CoreCoord& ethernet_core) const {
    return tt::tt_metal::MetalContext::instance().get_cluster().get_logical_ethernet_core_from_virtual(
        this->id(), ethernet_core);
}

uint32_t Device::get_noc_unicast_encoding(uint8_t noc_index, const CoreCoord& core) const {
    auto virtual_noc_coord = this->virtual_noc0_coordinate(noc_index, core);
    return tt::tt_metal::MetalContext::instance().hal().noc_xy_encoding(virtual_noc_coord.x, virtual_noc_coord.y);
}

uint32_t Device::get_noc_multicast_encoding(uint8_t noc_index, const CoreRange& cores) const {
    auto virtual_noc_start = this->virtual_noc0_coordinate(noc_index, cores.start_coord);
    auto virtual_noc_end = this->virtual_noc0_coordinate(noc_index, cores.end_coord);

    // NOC 1 mcasts from bottom left to top right, so we need to reverse the coords
    if (noc_index == 0) {
        return tt::tt_metal::MetalContext::instance().hal().noc_multicast_encoding(
            virtual_noc_start.x, virtual_noc_start.y, virtual_noc_end.x, virtual_noc_end.y);
    }
    return tt::tt_metal::MetalContext::instance().hal().noc_multicast_encoding(
        virtual_noc_end.x, virtual_noc_end.y, virtual_noc_start.x, virtual_noc_start.y);
}

const std::unique_ptr<AllocatorImpl>& Device::allocator_impl() const { return default_allocator_; }

const std::unique_ptr<Allocator>& Device::allocator() const {
    const auto& allocator{this->allocator_impl()};
    return allocator->view();
}

const std::unique_ptr<AllocatorImpl>& Device::allocator_impl(SubDeviceId /*sub_device_id*/) const {
    return default_allocator_;
}

const std::unique_ptr<Allocator>& Device::allocator(SubDeviceId sub_device_id) const {
    const auto& allocator{this->allocator_impl(sub_device_id)};
    return allocator->view();
}

uint32_t Device::num_sub_devices() const { return 1U; }

CoreCoord Device::dram_core_from_dram_channel(uint32_t dram_channel, NOC noc) const {
    return tt::tt_metal::MetalContext::instance()
        .get_cluster()
        .get_soc_desc(id_)
        .get_preferred_worker_core_for_dram_view(dram_channel, noc);
}

CoreCoord Device::logical_core_from_dram_channel(uint32_t dram_channel) const {
    return tt::tt_metal::MetalContext::instance().get_cluster().get_soc_desc(id_).get_logical_core_for_dram_view(
        dram_channel);
}

uint32_t Device::dram_channel_from_logical_core(const CoreCoord& logical_core) const {
    return tt::tt_metal::MetalContext::instance().get_cluster().get_soc_desc(id_).get_dram_channel_from_logical_core(
        logical_core);
}

uint32_t Device::dram_channel_from_virtual_core(const CoreCoord& virtual_core) const {
    const metal_SocDescriptor& soc_desc = tt::tt_metal::MetalContext::instance().get_cluster().get_soc_desc(this->id_);
    uint32_t num_nocs = MetalContext::instance().hal().get_num_nocs();
    for (uint32_t noc = 0; noc < num_nocs; noc++) {
        for (uint32_t channel = 0; channel < this->num_dram_channels(); ++channel) {
            if (soc_desc.get_preferred_worker_core_for_dram_view(channel, noc) == virtual_core) {
                return channel;
            }
        }
    }
    TT_THROW("Virtual core {} is not a DRAM core", virtual_core.str());
}

std::optional<DeviceAddr> Device::lowest_occupied_compute_l1_address() const { return std::nullopt; }

std::optional<DeviceAddr> Device::lowest_occupied_compute_l1_address(
    tt::stl::Span<const SubDeviceId> /*sub_device_ids*/) const {
    return std::nullopt;
}

CommandQueue& Device::command_queue(std::optional<uint8_t> cq_id) {
    detail::DispatchStateCheck(using_fast_dispatch_);
    if (!using_fast_dispatch_) {
        return *(CommandQueue*)(IDevice*)this;
    }
    auto actual_cq_id = cq_id.value_or(GetCurrentCommandQueueIdForThread());
    TT_FATAL(actual_cq_id < command_queues_.size(), "cq_id {} is out of range", actual_cq_id);
    TT_FATAL(this->is_initialized(), "Device has not been initialized, did you forget to call InitializeDevice?");
    return *command_queues_[actual_cq_id];
}

SystemMemoryManager& Device::sysmem_manager() {
    // SystemMemoryManager handles mock devices internally with stubs
    // For mock devices, ensure lazy initialization if not already done
    if (!sysmem_manager_) {
        sysmem_manager_ = std::make_unique<SystemMemoryManager>(this->id_, 1);
    }
    return *sysmem_manager_;
}

void Device::enable_program_cache() {
    log_info(tt::LogMetal, "Enabling program cache on device {}", this->id_);
    program_cache_.enable();
}
void Device::clear_program_cache() {
    log_info(tt::LogMetal, "Clearing program cache on device {}", this->id_);
    program_cache_.clear();
}

void Device::disable_and_clear_program_cache() {
    log_trace(tt::LogMetal, "Disabling and clearing program cache on device {}", this->id_);
    if (this->program_cache_.is_enabled()) {
        program_cache_.disable();
    }
    program_cache_.clear();
}
std::size_t Device::num_program_cache_entries() { return program_cache_.num_entries(); }

// NOLINTNEXTLINE(readability-make-member-function-const)
void Device::mark_allocations_unsafe() { this->allocator_impl()->mark_allocations_unsafe(); }

// NOLINTNEXTLINE(readability-make-member-function-const)
void Device::mark_allocations_safe() { this->allocator_impl()->mark_allocations_safe(); }

bool Device::has_noc_mcast_txns(SubDeviceId /*sub_device_id*/) const {
    TT_FATAL(false, "has_noc_mcast_txns is deprecated for device");
    return false;
}

uint8_t Device::num_noc_unicast_txns(SubDeviceId /*sub_device_id*/) const {
    TT_FATAL(false, "num_noc_unicast_txns is deprecated for device");
    return 0U;
}

uint8_t Device::noc_data_start_index(SubDeviceId /*sub_device_id*/, bool unicast_data) const {
    if (unicast_data) {
        TT_FATAL(false, "noc_data_start_index is deprecated for unicast mode for device");
    }
    return 0U;
}

CoreCoord Device::virtual_program_dispatch_core(uint8_t cq_id) const {
    if (cq_id >= this->command_queues_.size() || !this->command_queues_[cq_id]) {
        return CoreCoord{0, 0};  // Return default for mock devices
    }
    return this->command_queues_[cq_id]->virtual_enqueue_program_dispatch_core();
}

SubDeviceManagerId Device::get_active_sub_device_manager_id() const { return SubDeviceManagerId{0U}; }

SubDeviceManagerId Device::get_default_sub_device_manager_id() const { return SubDeviceManagerId{0U}; }

SubDeviceManagerId Device::create_sub_device_manager(
    std::initializer_list<SubDevice> /*sub_devices*/, DeviceAddr /*local_l1_size*/) {
    TT_FATAL(false, "create_sub_device_manager is deprecated for device");
    return SubDeviceManagerId{0U};
}

SubDeviceManagerId Device::create_sub_device_manager(
    tt::stl::Span<const SubDevice> /*sub_devices*/, DeviceAddr /*local_l1_size*/) {
    TT_FATAL(false, "create_sub_device_manager is deprecated for device");
    return SubDeviceManagerId{0U};
}

void Device::load_sub_device_manager(SubDeviceManagerId /*sub_device_manager_id*/) {
    TT_FATAL(false, "load_sub_device_manager is deprecated for device");
}

void Device::clear_loaded_sub_device_manager() {
    TT_FATAL(false, "clear_loaded_sub_device_manager is deprecated for device");
}

void Device::remove_sub_device_manager(SubDeviceManagerId /*sub_device_manager_id*/) {
    TT_FATAL(false, "remove_sub_device_manager is deprecated for device");
}

const std::vector<SubDeviceId>& Device::get_sub_device_ids() const {
    static std::vector<SubDeviceId> ids;
    TT_FATAL(false, "get_sub_device_ids is deprecated for device");
    return ids;
}

const std::vector<SubDeviceId>& Device::get_sub_device_stall_group() const {
    static std::vector<SubDeviceId> ids;
    TT_FATAL(false, "get_sub_device_stall_group is deprecated for device");
    return ids;
}

void Device::set_sub_device_stall_group(tt::stl::Span<const SubDeviceId> /*sub_device_ids*/) {
    TT_FATAL(false, "set_sub_device_stall_group is deprecated for device");
}

void Device::reset_sub_device_stall_group() {
    TT_FATAL(false, "reset_sub_device_stall_group is deprecated for device");
}

std::vector<CoreCoord> Device::get_optimal_dram_bank_to_logical_worker_assignment(NOC noc) {
    // Top level function that users (ex: Op Writers) can use to assign Tensix Worker cores
    // as DRAM readers or writers. Returns logical coordinates of optimally placed workers.
    // This function queries Physical Coordinates (only exposed directly to the Device class)
    // and passes them to logic in core_assignment.cpp to derive the most optimal core placement
    // based on architecture specific logic and Physical Grid configuration.
    if (this->optimal_dram_bank_to_logical_worker_assignment_.empty()) {
        uint32_t full_grid_size_x = this->grid_size().x;
        uint32_t full_grid_size_y = this->grid_size().y;

        auto compute_with_storage_grid_size = this->compute_with_storage_grid_size();
        uint32_t num_cores_x = compute_with_storage_grid_size.x;
        uint32_t num_cores_y = compute_with_storage_grid_size.y;
        // Get physical coordinates of DRAM Controller NOC end-points
        uint32_t num_dram_banks = this->num_dram_channels();

        const auto& hal = MetalContext::instance().hal();
        bool noc_translation_enabled = true;
        if (tt::tt_metal::MetalContext::instance().get_cluster().get_target_device_type() != tt::TargetDevice::Mock) {
            noc_translation_enabled = tt::tt_metal::MetalContext::instance()
                                          .get_cluster()
                                          .get_cluster_desc()
                                          ->get_noc_translation_table_en()
                                          .at(this->id());
        }
        bool dram_is_virtualized =
            noc_translation_enabled && (hal.get_virtualized_core_types().contains(dev_msgs::AddressableCoreType::DRAM));
        const metal_SocDescriptor& soc_d =
            tt::tt_metal::MetalContext::instance().get_cluster().get_soc_desc(this->id());
        std::vector<CoreCoord> dram_phy_coords;
        for (int i = 0; i < num_dram_banks; ++i) {
            auto dram_core = this->dram_core_from_dram_channel(i, noc);
            if (dram_is_virtualized) {
                tt::umd::CoreCoord umd_dram_coord = soc_d.translate_coord_to(
                    tt_xy_pair(dram_core.x, dram_core.y), CoordSystem::TRANSLATED, CoordSystem::NOC0);
                dram_core = CoreCoord(umd_dram_coord.x, umd_dram_coord.y);
            }
            dram_phy_coords.push_back(dram_core);
        }
        // Get all logical cores in the worker grid
        std::vector<CoreCoord> all_worker_cores_logical;
        for (int i = 0; i < num_cores_x; ++i) {
            for (int j = 0; j < num_cores_y; ++j) {
                all_worker_cores_logical.push_back(CoreCoord(i, j));
            }
        }
        // Get the physical rows and cols  (y, x) in the worker grid
        std::vector<uint32_t> worker_phy_y = std::vector<uint32_t>(num_cores_y);
        for (int i = 0; i < num_cores_y; ++i) {
            auto core_phy = this->physical_worker_core_from_logical_core(CoreCoord(0, i));
            worker_phy_y.at(i) = core_phy.y;
        }
        std::vector<uint32_t> worker_phy_x = std::vector<uint32_t>(num_cores_x);
        for (int i = 0; i < num_cores_x; ++i) {
            auto core_phy = this->physical_worker_core_from_logical_core(CoreCoord(i, 0));
            worker_phy_x.push_back(core_phy.x);
        }
        // Get optimal placement of worker cores interfacing with DRAM Controllers in physical coordinate space
        auto physical_worker_cores = get_optimal_dram_to_physical_worker_assignment(
            this->arch(), dram_phy_coords, full_grid_size_x, full_grid_size_y, worker_phy_x, worker_phy_y);

        const metal_SocDescriptor& soc_desc =
            tt::tt_metal::MetalContext::instance().get_cluster().get_soc_desc(this->id_);
        // Convert to physical worker coordinates to logical. This gets returned to the user.
        for (auto physical_worker_core : physical_worker_cores) {
            tt::umd::CoreCoord logical_coord_translated =
                soc_desc.translate_coord_to(physical_worker_core, CoordSystem::NOC0, CoordSystem::LOGICAL);
            this->optimal_dram_bank_to_logical_worker_assignment_.push_back(
                CoreCoord(logical_coord_translated.x, logical_coord_translated.y));
            TT_ASSERT(
                logical_coord_translated.core_type == CoreType::TENSIX,
                "Worker dram interface core {} should be a Tensix core, algorithm to place DRAM interfacing workers is "
                "invalid",
                logical_coord_translated.str());
        }
    }
    return this->optimal_dram_bank_to_logical_worker_assignment_;
}

HalProgrammableCoreType Device::get_programmable_core_type(CoreCoord virtual_core) const {
    if (!tt::tt_metal::MetalContext::instance().get_cluster().is_ethernet_core(virtual_core, this->id_)) {
        return HalProgrammableCoreType::TENSIX;
    }

    // Eth pcores have a different address, but only active ones.
    CoreCoord logical_core = this->logical_core_from_ethernet_core(virtual_core);
    if (this->is_active_ethernet_core(logical_core)) {
        return HalProgrammableCoreType::ACTIVE_ETH;
    }

    return HalProgrammableCoreType::IDLE_ETH;
}

HalMemType Device::get_mem_type_of_core(CoreCoord virtual_core) const {
    if (!tt::tt_metal::MetalContext::instance().get_cluster().is_ethernet_core(virtual_core, this->id_) &&
        !tt::tt_metal::MetalContext::instance().get_cluster().is_worker_core(virtual_core, this->id_)) {
        return HalMemType::DRAM;
    }
    return HalMemType::L1;
}

std::shared_ptr<distributed::MeshDevice> Device::get_mesh_device() { return mesh_device.lock(); }

}  // namespace tt::tt_metal
