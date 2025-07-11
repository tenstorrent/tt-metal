// SPDX-FileCopyrightText: © 2023 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "device_impl.hpp"

#include <core_descriptor.hpp>
#include "dev_msgs.h"
#include <device_pool.hpp>
#include <host_api.hpp>
#include <limits>
#include <magic_enum/magic_enum.hpp>
#include <persistent_kernel_cache.hpp>
#include <sub_device.hpp>
#include <sub_device_types.hpp>
#include <tt-metalium/program_cache.hpp>
#include <tt-metalium/hal.hpp>
#include <tt_align.hpp>
#include <tt_metal.hpp>
#include <tt_stl/span.hpp>
#include <algorithm>
#include <array>
#include <cstdint>
#include <cstring>
#include <functional>
#include <memory>
#include <optional>
#include <set>
#include <stdexcept>
#include <string>
#include <string_view>
#include <tuple>
#include <type_traits>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <variant>
#include <vector>

#include "allocator.hpp"
#include "allocator_types.hpp"
#include "assert.hpp"
#include "buffer_types.hpp"
#include "command_queue.hpp"
#include "dispatch/command_queue_common.hpp"
#include "common/core_assignment.hpp"
#include "program/program_impl.hpp"
#include "core_coord.hpp"
#include "device.hpp"
#include "impl/context/metal_context.hpp"
#include "trace/trace.hpp"
#include "dispatch_core_common.hpp"
#include "dispatch/dispatch_settings.hpp"
#include "hal_types.hpp"
#include "jit_build/build.hpp"
#include "dispatch/launch_message_ring_buffer_state.hpp"
#include "lightmetal/lightmetal_capture.hpp"
#include "llrt.hpp"
#include <tt-logger/tt-logger.hpp>
#include "metal_soc_descriptor.h"
#include "multi_producer_single_consumer_queue.hpp"
#include "profiler_types.hpp"
#include "tt-metalium/program.hpp"
#include <tt_stl/strong_type.hpp>
#include "dispatch/system_memory_manager.hpp"
#include "trace/trace_buffer.hpp"
#include "tracy/Tracy.hpp"
#include "tt_memory.h"
#include "tt_metal/impl/allocator/l1_banking_allocator.hpp"
#include "tt_metal/impl/dispatch/hardware_command_queue.hpp"
#include "tt_metal/impl/dispatch/topology.hpp"
#include "tt_metal/impl/sub_device/sub_device_manager.hpp"
#include "sub_device/sub_device_manager_tracker.hpp"
#include "tt_metal/jit_build/build_env_manager.hpp"
#include "tt_metal/tools/profiler/tt_metal_tracy.hpp"
#include <tt-metalium/control_plane.hpp>
#include <umd/device/coordinate_manager.h>
#include <umd/device/tt_core_coordinates.h>
#include <umd/device/tt_silicon_driver_common.hpp>
#include <umd/device/tt_xy_pair.h>
#include <umd/device/types/xy_pair.h>

namespace tt {
enum class ARCH;

namespace tt_metal {

uint64_t IDevice::get_dev_addr(CoreCoord virtual_core, HalL1MemAddrType addr_type) const {
    return MetalContext::instance().hal().get_dev_addr(this->get_programmable_core_type(virtual_core), addr_type);
}

uint64_t IDevice::get_dev_size(CoreCoord virtual_core, HalL1MemAddrType addr_type) const {
    return MetalContext::instance().hal().get_dev_size(this->get_programmable_core_type(virtual_core), addr_type);
}

void IDevice::set_program_cache_misses_allowed(bool allowed) {
    this->get_program_cache().set_cache_misses_allowed(allowed);
}

Device::Device(Device&& other) = default;
Device& Device::operator=(Device&& other) = default;

Device::Device(
    chip_id_t device_id,
    const uint8_t num_hw_cqs,
    size_t l1_small_size,
    size_t trace_region_size,
    tt::stl::Span<const std::uint32_t> l1_bank_remap,
    bool minimal,
    uint32_t worker_thread_core,
    uint32_t completion_queue_reader_core,
    size_t worker_l1_size) :
    id_(device_id),
    worker_thread_core_(worker_thread_core),
    completion_queue_reader_core_(completion_queue_reader_core) {
    ZoneScoped;
    this->initialize(num_hw_cqs, l1_small_size, trace_region_size, worker_l1_size, l1_bank_remap, minimal);
}

std::unordered_set<CoreCoord> Device::get_active_ethernet_cores(bool skip_reserved_tunnel_cores) const {
    return tt::tt_metal::MetalContext::instance().get_control_plane().get_active_ethernet_cores(
        this->id_, skip_reserved_tunnel_cores);
}

bool Device::is_active_ethernet_core(CoreCoord logical_core, bool skip_reserved_tunnel_cores) const {
    auto active_ethernet_cores = this->get_active_ethernet_cores(skip_reserved_tunnel_cores);
    return active_ethernet_cores.find(logical_core) != active_ethernet_cores.end();
}

std::unordered_set<CoreCoord> Device::get_inactive_ethernet_cores() const {
    return tt::tt_metal::MetalContext::instance().get_control_plane().get_inactive_ethernet_cores(this->id_);
}

bool Device::is_inactive_ethernet_core(CoreCoord logical_core) const {
    auto inactive_ethernet_cores =
        tt::tt_metal::MetalContext::instance().get_control_plane().get_inactive_ethernet_cores(this->id_);
    return inactive_ethernet_cores.find(logical_core) != inactive_ethernet_cores.end();
}

uint32_t Device::num_virtual_eth_cores(SubDeviceId sub_device_id) {
    return this->num_worker_cores(HalProgrammableCoreType::ACTIVE_ETH, sub_device_id);
}

std::tuple<chip_id_t, CoreCoord> Device::get_connected_ethernet_core(CoreCoord eth_core) const {
    return tt::tt_metal::MetalContext::instance().get_cluster().get_connected_ethernet_core(
        std::make_tuple(this->id_, eth_core));
}

std::vector<CoreCoord> Device::get_ethernet_sockets(chip_id_t connected_chip_id) const {
    return tt::tt_metal::MetalContext::instance().get_cluster().get_ethernet_sockets(this->id_, connected_chip_id);
}

bool Device::is_mmio_capable() const {
    return tt::tt_metal::MetalContext::instance().get_cluster().get_associated_mmio_device(this->id_) == this->id_;
}

CoreRangeSet Device::worker_cores(HalProgrammableCoreType core_type, SubDeviceId sub_device_id) const {
    return sub_device_manager_tracker_->get_active_sub_device_manager()->sub_device(sub_device_id).cores(core_type);
}

uint32_t Device::num_worker_cores(HalProgrammableCoreType core_type, SubDeviceId sub_device_id) const {
    return sub_device_manager_tracker_->get_active_sub_device_manager()->sub_device(sub_device_id).num_cores(core_type);
}

void Device::initialize_default_sub_device_state(
    size_t l1_small_size,
    size_t trace_region_size,
    size_t worker_l1_unreserved_start,
    tt::stl::Span<const std::uint32_t> l1_bank_remap) {
    // Create the default sub-device manager representing the entire chip
    const auto& compute_grid_size = this->compute_with_storage_grid_size();
    const auto& active_eth_cores = this->get_active_ethernet_cores(true);
    std::vector<CoreRange> active_eth_core_ranges;
    active_eth_core_ranges.reserve(active_eth_cores.size());
    for (const auto& core : active_eth_cores) {
        active_eth_core_ranges.emplace_back(core, core);
    }

    auto sub_devices = {SubDevice(std::array{
        CoreRangeSet(CoreRange({0, 0}, {compute_grid_size.x - 1, compute_grid_size.y - 1})),
        CoreRangeSet(std::move(active_eth_core_ranges))})};

    sub_device_manager_tracker_ = std::make_unique<SubDeviceManagerTracker>(
        this,
        this->initialize_allocator(l1_small_size, trace_region_size, worker_l1_unreserved_start, l1_bank_remap),
        sub_devices);
}

std::unique_ptr<Allocator> Device::initialize_allocator(
    size_t l1_small_size,
    size_t trace_region_size,
    size_t worker_l1_unreserved_start,
    tt::stl::Span<const std::uint32_t> l1_bank_remap) {
    ZoneScoped;
    const metal_SocDescriptor& soc_desc = tt::tt_metal::MetalContext::instance().get_cluster().get_soc_desc(this->id_);
    const auto& dispatch_core_config = MetalContext::instance().get_dispatch_core_manager().get_dispatch_core_config();
    auto config = L1BankingAllocator::generate_config(
        this->id(),
        this->num_hw_cqs(),
        l1_small_size,
        trace_region_size,
        worker_l1_unreserved_start,
        {l1_bank_remap.begin(), l1_bank_remap.end()});

    for (const CoreCoord& core : tt::get_logical_compute_cores(id_, num_hw_cqs_, dispatch_core_config)) {
        this->compute_cores_.insert(core);
    }
    for (const CoreCoord& core : tt::get_logical_storage_cores(id_, num_hw_cqs_, dispatch_core_config)) {
        this->storage_only_cores_.insert(core);
    }
    for (const tt::umd::CoreCoord& core : soc_desc.get_cores(CoreType::ETH, CoordSystem::LOGICAL)) {
        this->ethernet_cores_.insert({core.x, core.y});
    }

    // L1_BANKING scheme creates 1 bank per DRAM core and splits up L1 such that there are power 2 num L1 banks
    // This is the only allocator scheme supported because kernel APIs assume num L1 banks are power of 2
    TT_ASSERT(this->allocator_scheme_ == MemoryAllocator::L1_BANKING);
    return std::make_unique<L1BankingAllocator>(config);
}

// Writes issue and completion queue pointers to device and in sysmem and loads fast dispatch program onto dispatch cores
void Device::configure_command_queue_programs() {
    chip_id_t device_id = this->id();
    chip_id_t mmio_device_id =
        tt::tt_metal::MetalContext::instance().get_cluster().get_associated_mmio_device(device_id);

    std::vector<uint32_t> zero = {0x0}; // Reset state in case L1 Clear is disabled.
    std::vector<uint32_t> pointers;
    uint32_t cq_size = this->sysmem_manager().get_cq_size();
    TT_ASSERT(this->command_queue_programs_.size() == 1);

    Program& command_queue_program = *this->command_queue_programs_[0];
    uint8_t num_hw_cqs = this->num_hw_cqs();

    // Reset host-side command queue pointers for all channels controlled by this mmio device
    if (this->is_mmio_capable()) {
        for (chip_id_t serviced_device_id :
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
            pointers.resize(cq_start/sizeof(uint32_t));
            for (uint8_t cq_id = 0; cq_id < num_hw_cqs; cq_id++) {
                // Reset the host manager's pointer for this command queue
                this->sysmem_manager_->reset(cq_id);

                pointers[host_issue_q_rd_ptr / sizeof(uint32_t)] = (cq_start + get_absolute_cq_offset(channel, cq_id, cq_size)) >> 4;
                pointers[host_issue_q_wr_ptr / sizeof(uint32_t)] = (cq_start + get_absolute_cq_offset(channel, cq_id, cq_size)) >> 4;
                pointers[host_completion_q_wr_ptr / sizeof(uint32_t)] = (cq_start + this->sysmem_manager_->get_issue_queue_size(cq_id) + get_absolute_cq_offset(channel, cq_id, cq_size)) >> 4;
                pointers[host_completion_q_rd_ptr / sizeof(uint32_t)] = (cq_start + this->sysmem_manager_->get_issue_queue_size(cq_id) + get_absolute_cq_offset(channel, cq_id, cq_size)) >> 4;

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
    command_queue_program.finalize_offsets(this);
    detail::ConfigureDeviceWithProgram(this, command_queue_program, true);
    tt::tt_metal::MetalContext::instance().get_cluster().l1_barrier(this->id());
}

void Device::init_command_queue_host() {
    using_fast_dispatch_ = true;
    sysmem_manager_ = std::make_unique<SystemMemoryManager>(this->id_, this->num_hw_cqs());

    auto cq_shared_state = std::make_shared<CQSharedState>();
    cq_shared_state->sub_device_cq_owner.resize(1);
    command_queues_.reserve(num_hw_cqs());
    for (size_t cq_id = 0; cq_id < num_hw_cqs(); cq_id++) {
        command_queues_.push_back(std::make_unique<HWCommandQueue>(
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
    const auto& storage_only_cores = tt::get_logical_storage_cores(
        id_, num_hw_cqs_, MetalContext::instance().get_dispatch_core_manager().get_dispatch_core_config());
    auto storage_only_cores_set = std::unordered_set<CoreCoord>(storage_only_cores.begin(), storage_only_cores.end());
    std::optional<std::unique_lock<std::mutex>> watcher_lock;
    if (tt::tt_metal::MetalContext::instance().rtoptions().get_watcher_enabled()) {
        watcher_lock = MetalContext::instance().watcher_server()->get_lock();
    }
    for (uint32_t y = 0; y < logical_grid_size().y; y++) {
        for (uint32_t x = 0; x < logical_grid_size().x; x++) {
            CoreCoord logical_core(x, y);
            if (!storage_only_cores_set.count(logical_core)) {
                reset_launch_message_rd_ptr(logical_core, CoreType::WORKER);
                reset_go_message_index(logical_core, CoreType::WORKER);
            }
        }
    }
    for (const auto& logical_core : this->get_active_ethernet_cores()) {
        reset_launch_message_rd_ptr(logical_core, CoreType::ETH);
    }
    for (const auto& logical_core : this->get_inactive_ethernet_cores()) {
        reset_launch_message_rd_ptr(logical_core, CoreType::ETH);
    }
    if (watcher_lock) {
        watcher_lock.value().unlock();
    }

    // TODO: should get a const ref
    std::vector<std::vector<CoreCoord>>logical_cores = command_queue_program.logical_cores();
    const auto& hal = MetalContext::instance().hal();
    for (uint32_t index = 0; index < hal.get_programmable_core_type_count(); index++) {
        const auto& logical_dispatch_cores = logical_cores[index];
        CoreType core_type = hal.get_core_type(index);
        for (const CoreCoord &logical_dispatch_core : logical_dispatch_cores) {
            launch_msg_t msg = command_queue_program.impl().kernels_on_core(logical_dispatch_core, index)->launch_msg;
            go_msg_t go_msg = command_queue_program.impl().kernels_on_core(logical_dispatch_core, index)->go_msg;
            CoreCoord virtual_core = this->virtual_core_from_logical_core(logical_dispatch_core, core_type);
            tt::llrt::write_launch_msg_to_core(this->id(), virtual_core, &msg, &go_msg, this->get_dev_addr(virtual_core, HalL1MemAddrType::LAUNCH));
        }
    }
    // Set num_worker_sems and go_signal_noc_data on dispatch for the default sub device config
    for (auto& hw_cq : this->command_queues_) {
        hw_cq->set_go_signal_noc_data_and_dispatch_sems(
            sub_device_manager_tracker_->get_active_sub_device_manager()->num_sub_devices(),
            sub_device_manager_tracker_->get_active_sub_device_manager()->noc_mcast_unicast_data());
    }
}

bool Device::compile_fabric() {
    fabric_program_ = create_and_compile_fabric_program(this);
    return fabric_program_ != nullptr;
}

void Device::configure_fabric() {
    if (fabric_program_ == nullptr) {
        return;
    }

    configure_fabric_cores(this);

    fabric_program_->finalize_offsets(this);

    detail::WriteRuntimeArgsToDevice(this, *fabric_program_, this->using_fast_dispatch());
    detail::ConfigureDeviceWithProgram(this, *fabric_program_, this->using_fast_dispatch());

    // Note: the l1_barrier below is needed to be sure writes to cores that
    // don't get the GO mailbox (eg, storage cores) have all landed
    tt::tt_metal::MetalContext::instance().get_cluster().l1_barrier(this->id());
    std::vector<std::vector<CoreCoord>> logical_cores_used_in_program = fabric_program_->logical_cores();
    const auto& hal = MetalContext::instance().hal();
    for (uint32_t programmable_core_type_index = 0; programmable_core_type_index < logical_cores_used_in_program.size();
         programmable_core_type_index++) {
        CoreType core_type = hal.get_core_type(programmable_core_type_index);
        for (const auto& logical_core : logical_cores_used_in_program[programmable_core_type_index]) {
            launch_msg_t* msg =
                &fabric_program_->impl().kernels_on_core(logical_core, programmable_core_type_index)->launch_msg;
            go_msg_t* go_msg =
                &fabric_program_->impl().kernels_on_core(logical_core, programmable_core_type_index)->go_msg;
            msg->kernel_config.host_assigned_id = fabric_program_->get_runtime_id();

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
    TT_FATAL(num_hw_cqs > 0 and num_hw_cqs <= dispatch_core_manager::MAX_NUM_HW_CQS, "num_hw_cqs can be between 1 and {}", dispatch_core_manager::MAX_NUM_HW_CQS);
    this->using_fast_dispatch_ = false;
    // Trying to preserve logic that was in device_pool.cpp
    // However, I honestly don't understand it
    this->num_hw_cqs_ = num_hw_cqs;
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
    std::uint32_t max_alignment = std::max(hal.get_alignment(HalMemType::DRAM), hal.get_alignment(HalMemType::L1));
    uint32_t worker_l1_unreserved_start = tt::align(
        hal.get_dev_addr(HalProgrammableCoreType::TENSIX, HalL1MemAddrType::BASE) +
            hal.get_dev_size(HalProgrammableCoreType::TENSIX, HalL1MemAddrType::BASE) - worker_l1_size,
        max_alignment);
    this->initialize_default_sub_device_state(
        l1_small_size, trace_region_size, worker_l1_unreserved_start, l1_bank_remap);

    // For minimal setup, don't initialize FW, watcher, dprint. They won't work if we're attaching to a hung chip.
    if (minimal)
        return true;

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

    sub_device_manager_tracker_.reset(nullptr);

    this->compute_cores_.clear();
    this->storage_only_cores_.clear();
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
    } else {
        const auto& grid_size = this->grid_size();
        // Coordinate in Physical NOC0 Space. Convert to Virtual.
        coord = this->virtual_core_from_physical_core(coord);
        // Derive virtual coord in noc_index space.
        CoreCoord virtual_coord = {
            MetalContext::instance().hal().noc_coordinate(noc_index, grid_size.x, coord.x),
            MetalContext::instance().hal().noc_coordinate(noc_index, grid_size.y, coord.y)};
        return virtual_coord;
    }
}

CoreCoord Device::physical_worker_core_from_logical_core(const CoreCoord &logical_core) const {
    const metal_SocDescriptor& soc_desc = tt::tt_metal::MetalContext::instance().get_cluster().get_soc_desc(this->id_);
    return soc_desc.get_physical_tensix_core_from_logical(logical_core);
}

std::vector<CoreCoord> Device::worker_cores_from_logical_cores(const std::vector<CoreCoord> &logical_cores) const {
    std::vector<CoreCoord> worker_cores(logical_cores.size());
    for (std::size_t idx = 0; idx < logical_cores.size(); idx++)
        worker_cores[idx] = this->worker_core_from_logical_core(logical_cores[idx]);

    return worker_cores;
}

std::vector<CoreCoord> Device::ethernet_cores_from_logical_cores(const std::vector<CoreCoord> &logical_cores) const {
    std::vector<CoreCoord> eth_cores(logical_cores.size());
    for (std::size_t idx = 0; idx < logical_cores.size(); idx++) {
        eth_cores[idx] = this->ethernet_core_from_logical_core(logical_cores[idx]);
    }
    return eth_cores;
}

CoreCoord Device::virtual_core_from_logical_core(const CoreCoord &logical_coord, const CoreType& core_type) const {
    return tt::tt_metal::MetalContext::instance().get_cluster().get_virtual_coordinate_from_logical_coordinates(
        this->id_, logical_coord, core_type);
}

CoreCoord Device::virtual_core_from_physical_core(const CoreCoord& physical_coord) const {
    return tt::tt_metal::MetalContext::instance().get_cluster().get_virtual_coordinate_from_physical_coordinates(
        this->id_, physical_coord);
}

CoreCoord Device::worker_core_from_logical_core(const CoreCoord &logical_core) const {
    return this->virtual_core_from_logical_core(logical_core, CoreType::WORKER);
}

CoreCoord Device::ethernet_core_from_logical_core(const CoreCoord &logical_core) const {
    return this->virtual_core_from_logical_core(logical_core, CoreType::ETH);
}

CoreCoord Device::logical_core_from_ethernet_core(const CoreCoord &ethernet_core) const {
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
    } else {
        return tt::tt_metal::MetalContext::instance().hal().noc_multicast_encoding(
            virtual_noc_end.x, virtual_noc_end.y, virtual_noc_start.x, virtual_noc_start.y);
    }
}

const std::unique_ptr<Allocator>& Device::allocator() const {
    return sub_device_manager_tracker_->get_default_sub_device_manager()->allocator(SubDeviceId{0});
}

const std::unique_ptr<Allocator>& Device::allocator(SubDeviceId sub_device_id) const {
    return sub_device_manager_tracker_->get_active_sub_device_manager()->allocator(sub_device_id);
}

uint32_t Device::num_sub_devices() const {
    return sub_device_manager_tracker_->get_active_sub_device_manager()->num_sub_devices();
}

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

std::optional<DeviceAddr> Device::lowest_occupied_compute_l1_address() const {
    return sub_device_manager_tracker_->lowest_occupied_compute_l1_address();
}

std::optional<DeviceAddr> Device::lowest_occupied_compute_l1_address(tt::stl::Span<const SubDeviceId> sub_device_ids) const {
    return sub_device_manager_tracker_->lowest_occupied_compute_l1_address(sub_device_ids);
}

CommandQueue& Device::command_queue(size_t cq_id) {
    detail::DispatchStateCheck(using_fast_dispatch_);
    TT_FATAL(cq_id < command_queues_.size(), "cq_id {} is out of range", cq_id);
    TT_FATAL(this->is_initialized(), "Device has not been initialized, did you forget to call InitializeDevice?");
    return *command_queues_[cq_id];
}

bool Device::using_slow_dispatch() const {
    return !using_fast_dispatch();
}

bool Device::using_fast_dispatch() const {
    return using_fast_dispatch_;
}

void Device::begin_trace(const uint8_t cq_id, const uint32_t tid) {
    ZoneScoped;

    TracyTTMetalBeginTrace(this->id(), tid);
    TT_FATAL(
        !this->command_queues_[cq_id]->tid().has_value(),
        "CQ {} is already being used for tracing tid {}",
        (uint32_t)cq_id,
        tid);
    this->mark_allocations_safe();
    // Create an empty trace buffer here. This will get initialized in end_trace
    auto* active_sub_device_manager = sub_device_manager_tracker_->get_active_sub_device_manager();
    TT_FATAL(
        active_sub_device_manager->get_trace(tid) == nullptr,
        "Trace already exists for tid {} on device {}'s active sub-device manager {}",
        tid,
        this->id_,
        active_sub_device_manager->id());
    auto& trace_buffer = active_sub_device_manager->create_trace(tid);
    this->command_queues_[cq_id]->record_begin(tid, trace_buffer->desc);
}

void Device::end_trace(const uint8_t cq_id, const uint32_t tid) {
    ZoneScoped;
    TracyTTMetalEndTrace(this->id(), tid);
    TT_FATAL(
        this->command_queues_[cq_id]->tid() == tid, "CQ {} is not being used for tracing tid {}", (uint32_t)cq_id, tid);
    auto* active_sub_device_manager = sub_device_manager_tracker_->get_active_sub_device_manager();
    auto trace_buffer = active_sub_device_manager->get_trace(tid);
    TT_FATAL(
        trace_buffer != nullptr,
        "Trace instance {} must exist on device {}'s active sub-device manager {}",
        tid,
        this->id_,
        active_sub_device_manager->id());
    this->command_queues_[cq_id]->record_end();

    // Capture Trace if light metal trace capturing is enabled.
    auto& lm_capture_ctx = LightMetalCaptureContext::get();
    if (lm_capture_ctx.is_tracing()) {
        lm_capture_ctx.capture_trace_descriptor(*trace_buffer->desc, tid);
    }

    Trace::initialize_buffer(this->command_queue(cq_id), trace_buffer);
    this->mark_allocations_unsafe();
}

// Load the TraceDescriptor for a given trace_id to the device. A combination of logic from begin/end_trace.
void Device::load_trace(const uint8_t cq_id, const uint32_t trace_id, const TraceDescriptor& trace_desc) {
    this->mark_allocations_safe();

    auto* active_sub_device_manager = sub_device_manager_tracker_->get_active_sub_device_manager();
    TT_FATAL(
        active_sub_device_manager->get_trace(trace_id) == nullptr,
        "Trace already exists for trace_id {} on device {}'s active sub-device manager {}",
        trace_id,
        this->id_,
        active_sub_device_manager->id());

    auto& trace_buffer = active_sub_device_manager->create_trace(trace_id);
    *trace_buffer->desc = trace_desc;
    Trace::initialize_buffer(this->command_queue(cq_id), trace_buffer);
    this->mark_allocations_unsafe();
}

void Device::replay_trace(
    const uint8_t cq_id, const uint32_t tid, const bool block_on_device, const bool block_on_worker_thread) {
    // If blocking, ensure that worker thread blocks until trace is completed
    ZoneScoped;
    TracyTTMetalReplayTrace(this->id(), tid);
    constexpr bool check = false;
    auto* active_sub_device_manager = sub_device_manager_tracker_->get_active_sub_device_manager();
    const auto& trace_buffer = active_sub_device_manager->get_trace(tid);
    TT_FATAL(
        trace_buffer != nullptr,
        "Trace instance {} must exist on device {}'s active sub-device manager {}",
        tid,
        this->id_,
        active_sub_device_manager->id());
    if constexpr (check) {
        trace_buffer->validate();
    }
    EnqueueTrace(this->command_queue(cq_id), tid, block_on_device);
}

void Device::release_trace(const uint32_t tid) {
    ZoneScoped;
    TracyTTMetalReleaseTrace(this->id(), tid);

    sub_device_manager_tracker_->get_active_sub_device_manager()->release_trace(tid);

    // Only enable allocations once all captured traces are released
    if (this->trace_buffers_size_ == 0) {
        this->mark_allocations_safe();
    }
}

std::shared_ptr<TraceBuffer> Device::get_trace(uint32_t tid) {
    return sub_device_manager_tracker_->get_active_sub_device_manager()->get_trace(tid);
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

void Device::mark_allocations_unsafe() { this->allocator()->mark_allocations_unsafe(); }

void Device::mark_allocations_safe() { this->allocator()->mark_allocations_safe(); }

bool Device::has_noc_mcast_txns(SubDeviceId sub_device_id) const {
    return sub_device_manager_tracker_->get_active_sub_device_manager()->has_noc_mcast_txns(sub_device_id);
}

uint8_t Device::num_noc_unicast_txns(SubDeviceId sub_device_id) const {
    return sub_device_manager_tracker_->get_active_sub_device_manager()->num_noc_unicast_txns(sub_device_id);
}

uint8_t Device::noc_data_start_index(SubDeviceId sub_device_id, bool unicast_data) const {
    if (unicast_data) {
        return sub_device_manager_tracker_->get_active_sub_device_manager()->noc_unicast_data_start_index(
            sub_device_id);
    } else {
        return 0;
    }
}

CoreCoord Device::virtual_program_dispatch_core(uint8_t cq_id) const {
    return this->command_queues_[cq_id]->virtual_enqueue_program_dispatch_core();
}

SubDeviceManagerId Device::get_active_sub_device_manager_id() const {
    return sub_device_manager_tracker_->get_active_sub_device_manager()->id();
}

SubDeviceManagerId Device::get_default_sub_device_manager_id() const {
    return sub_device_manager_tracker_->get_default_sub_device_manager()->id();
}

SubDeviceManagerId Device::create_sub_device_manager(tt::stl::Span<const SubDevice> sub_devices, DeviceAddr local_l1_size) {
    return sub_device_manager_tracker_->create_sub_device_manager(sub_devices, local_l1_size);
}

void Device::load_sub_device_manager(SubDeviceManagerId sub_device_manager_id) {
    sub_device_manager_tracker_->load_sub_device_manager(sub_device_manager_id);
}

void Device::clear_loaded_sub_device_manager() { sub_device_manager_tracker_->clear_loaded_sub_device_manager(); }

void Device::remove_sub_device_manager(SubDeviceManagerId sub_device_manager_id) {
    sub_device_manager_tracker_->remove_sub_device_manager(sub_device_manager_id);
}

const std::vector<SubDeviceId> &Device::get_sub_device_ids() const {
    return sub_device_manager_tracker_->get_active_sub_device_manager()->get_sub_device_ids();
}

const std::vector<SubDeviceId> &Device::get_sub_device_stall_group() const {
    return sub_device_manager_tracker_->get_active_sub_device_manager()->get_sub_device_stall_group();
}

void Device::set_sub_device_stall_group(tt::stl::Span<const SubDeviceId> sub_device_ids) {
    sub_device_manager_tracker_->get_active_sub_device_manager()->set_sub_device_stall_group(sub_device_ids);
}

void Device::reset_sub_device_stall_group() {
    sub_device_manager_tracker_->get_active_sub_device_manager()->reset_sub_device_stall_group();
}

std::vector<CoreCoord> Device::get_optimal_dram_bank_to_logical_worker_assignment(NOC noc) {
    // Top level function that users (ex: Op Writers) can use to assign Tensix Worker cores
    // as DRAM readers or writers. Returns logical coordinates of optimally placed workers.
    // This function queries Physical Coordinates (only exposed directly to the Device class)
    // and passes them to logic in core_assignment.cpp to derive the most optimal core placement
    // based on architecture specific logic and Physical Grid configuration.
    if (not this->optimal_dram_bank_to_logical_worker_assignment_.size()) {
        uint32_t full_grid_size_x = this->grid_size().x;
        uint32_t full_grid_size_y = this->grid_size().y;

        auto compute_with_storage_grid_size = this->compute_with_storage_grid_size();
        uint32_t num_cores_x = compute_with_storage_grid_size.x;
        uint32_t num_cores_y = compute_with_storage_grid_size.y;
        // Get physical coordinates of DRAM Controller NOC end-points
        uint32_t num_dram_banks = this->num_dram_channels();

        const auto& hal = MetalContext::instance().hal();
        bool dram_is_virtualized =
            hal.get_virtualized_core_types().find(AddressableCoreType::DRAM) != hal.get_virtualized_core_types().end();
        const metal_SocDescriptor& soc_d =
            tt::tt_metal::MetalContext::instance().get_cluster().get_soc_desc(this->id());
        std::vector<CoreCoord> dram_phy_coords;
        for (int i = 0; i < num_dram_banks; ++i) {
            auto dram_core = this->dram_core_from_dram_channel(i, noc);
            if (dram_is_virtualized) {
                tt::umd::CoreCoord umd_dram_coord = soc_d.translate_coord_to(
                    tt_xy_pair(dram_core.x, dram_core.y), CoordSystem::TRANSLATED, CoordSystem::PHYSICAL);
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
        auto physical_worker_cores = get_optimal_dram_to_physical_worker_assignment(this->arch(), dram_phy_coords, full_grid_size_x, full_grid_size_y, worker_phy_x, worker_phy_y);

        const metal_SocDescriptor& soc_desc =
            tt::tt_metal::MetalContext::instance().get_cluster().get_soc_desc(this->id_);
        // Convert to physical worker coordinates to logical. This gets returned to the user.
        for (auto physical_worker_core : physical_worker_cores) {
            tt::umd::CoreCoord logical_coord_translated =
                soc_desc.translate_coord_to(physical_worker_core, CoordSystem::PHYSICAL, CoordSystem::LOGICAL);
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
    } else {
        return HalMemType::L1;
    }
}

std::shared_ptr<distributed::MeshDevice> Device::get_mesh_device() { return mesh_device.lock(); }

}  // namespace tt_metal

}  // namespace tt
