// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <tt_stl/fmt.hpp>
#include "context/context_types.hpp"
#include "context/metal_env_accessor.hpp"
#include "device_impl.hpp"

#include <core_descriptor.hpp>
#include <host_api.hpp>
#include <chrono>
#include <initializer_list>
#include <thread>
#include <tt_stl/tt_pause.hpp>
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
#include <cstdlib>
#include <cstring>
#include <memory>
#include <map>
#include <optional>
#include <set>
#include <tuple>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "allocator.hpp"
#include "common/env_lib.hpp"
#include <tt_stl/assert.hpp>
#include "dispatch/command_queue_common.hpp"
#include "common/core_assignment.hpp"
#include "program/program_impl.hpp"
#include "memory_tracking/memory_stats_shm.hpp"
#include "memory_tracking/shm_tracking_processor.hpp"
#include <tt-metalium/graph_tracking.hpp>
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
#include "fabric/fabric_context.hpp"
#include "fabric/fabric_builder_context.hpp"
#include "fabric/fabric_tensix_builder.hpp"
#include "fabric/fabric_tensix_builder_impl.hpp"
#include "fabric/fabric_edm_packet_header.hpp"
#include "fabric/fabric_host_utils.hpp"
#include <umd/device/coordinates/coordinate_manager.hpp>
#include <umd/device/types/core_coordinates.hpp>
#include <umd/device/types/xy_pair.hpp>
#include <impl/debug/watcher_server.hpp>
#include <impl/dispatch/dispatch_mem_map.hpp>

namespace tt::tt_metal {

void IDevice::set_program_cache_misses_allowed(bool allowed) {
    this->get_program_cache().set_cache_misses_allowed(allowed);
}

Device::Device(
    MetalEnv* env,
    MetalContext* context,
    ChipId device_id,
    const uint8_t num_hw_cqs,
    size_t l1_small_size,
    size_t trace_region_size,
    tt::stl::Span<const std::uint32_t> l1_bank_remap,
    bool minimal,
    uint32_t /*worker_thread_core*/,
    uint32_t /*completion_queue_reader_core*/,
    size_t worker_l1_size) :
    context_(context), env_(env), id_(device_id) {
    ZoneScoped;
    TT_FATAL(env != nullptr, "env is nullptr");
    TT_FATAL(context != nullptr, "context is nullptr");
    this->initialize(num_hw_cqs, l1_small_size, trace_region_size, worker_l1_size, l1_bank_remap, minimal);
}

std::unordered_set<CoreCoord> Device::get_active_ethernet_cores(bool skip_reserved_tunnel_cores) const {
    return MetalEnvAccessor(*env_).impl().get_control_plane().get_active_ethernet_cores(
        this->id_, skip_reserved_tunnel_cores);
}

bool Device::is_active_ethernet_core(CoreCoord logical_core, bool skip_reserved_tunnel_cores) const {
    auto active_ethernet_cores = this->get_active_ethernet_cores(skip_reserved_tunnel_cores);
    return active_ethernet_cores.contains(logical_core);
}

std::unordered_set<CoreCoord> Device::get_inactive_ethernet_cores() const {
    return MetalEnvAccessor(*env_).impl().get_control_plane().get_inactive_ethernet_cores(this->id_);
}

bool Device::is_inactive_ethernet_core(CoreCoord logical_core) const {
    auto inactive_ethernet_cores =
        MetalEnvAccessor(*env_).impl().get_control_plane().get_inactive_ethernet_cores(this->id_);
    return inactive_ethernet_cores.contains(logical_core);
}

uint32_t Device::num_virtual_eth_cores(SubDeviceId sub_device_id) {
    return this->num_worker_cores(HalProgrammableCoreType::ACTIVE_ETH, sub_device_id);
}

std::tuple<ChipId, CoreCoord> Device::get_connected_ethernet_core(CoreCoord eth_core) const {
    return MetalEnvAccessor(*env_).impl().get_cluster().get_connected_ethernet_core(
        std::make_tuple(this->id_, eth_core));
}

std::vector<CoreCoord> Device::get_ethernet_sockets(ChipId connected_chip_id) const {
    return MetalEnvAccessor(*env_).impl().get_cluster().get_ethernet_sockets(this->id_, connected_chip_id);
}

bool Device::is_mmio_capable() const {
    return MetalEnvAccessor(*env_).impl().get_cluster().get_associated_mmio_device(this->id_) == this->id_;
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
    const metal_SocDescriptor& soc_desc = MetalEnvAccessor(*env_).impl().get_cluster().get_soc_desc(this->id_);
    auto& dispatch_core_manager = context_->get_dispatch_core_manager();
    auto config = L1BankingAllocator::generate_config(
        dispatch_core_manager,
        MetalEnvAccessor(*env_).impl(),
        this->id(),
        this->num_hw_cqs(),
        l1_small_size,
        trace_region_size,
        worker_l1_unreserved_start,
        {l1_bank_remap.begin(), l1_bank_remap.end()});
    config.allocator_mode =
        context_->rtoptions().get_allocator_mode_hybrid() ? AllocatorMode::HYBRID : AllocatorMode::LOCKSTEP;

    for (const tt::umd::CoreCoord& core : soc_desc.get_cores(CoreType::ETH, CoordSystem::LOGICAL)) {
        this->ethernet_cores_.insert({core.x, core.y});
    }

    // L1 Banking Allocator creates 1 bank per DRAM core and splits up L1 such that there are power 2 num L1 banks
    // This is the only allocator scheme supported because kernel APIs assume num L1 banks are power of 2
    return std::make_unique<L1BankingAllocator>(config);
}

// Writes issue and completion queue pointers to device and in sysmem and loads fast dispatch program onto dispatch
// cores
void Device::configure_command_queue_programs(DispatchTopology* dispatch_topology) {
    ChipId device_id = this->id();
    ChipId mmio_device_id = MetalEnvAccessor(*env_).impl().get_cluster().get_associated_mmio_device(device_id);

    std::vector<uint32_t> zero = {0x0};  // Reset state in case L1 Clear is disabled.
    std::vector<uint32_t> pointers;
    uint32_t cq_size = this->sysmem_manager().get_cq_size();
    TT_ASSERT(this->command_queue_programs_.size() == 1);

    Program& command_queue_program = *this->command_queue_programs_[0];
    uint8_t num_hw_cqs = this->num_hw_cqs();

    // Reset host-side command queue pointers for all channels controlled by this mmio device
    if (this->is_mmio_capable()) {
        for (ChipId serviced_device_id :
             MetalEnvAccessor(*env_).impl().get_cluster().get_devices_controlled_by_mmio_device(device_id)) {
            uint16_t channel =
                MetalEnvAccessor(*env_).impl().get_cluster().get_assigned_channel_for_device(serviced_device_id);
            uint32_t host_issue_q_rd_ptr =
                context_->dispatch_mem_map().get_host_command_queue_addr(CommandQueueHostAddrType::ISSUE_Q_RD);
            uint32_t host_issue_q_wr_ptr =
                context_->dispatch_mem_map().get_host_command_queue_addr(CommandQueueHostAddrType::ISSUE_Q_WR);
            uint32_t host_completion_q_wr_ptr =
                context_->dispatch_mem_map().get_host_command_queue_addr(CommandQueueHostAddrType::COMPLETION_Q_WR);
            uint32_t host_completion_q_rd_ptr =
                context_->dispatch_mem_map().get_host_command_queue_addr(CommandQueueHostAddrType::COMPLETION_Q_RD);
            uint32_t cq_start =
                context_->dispatch_mem_map().get_host_command_queue_addr(CommandQueueHostAddrType::UNRESERVED);
            pointers.resize(cq_start / sizeof(uint32_t));
            for (uint8_t cq_id = 0; cq_id < num_hw_cqs; cq_id++) {
                // Reset the host manager's pointer for this command queue
                this->sysmem_manager_->reset(cq_id);

                const uint32_t cq_offset =
                    this->sysmem_manager_->is_dram_backed()
                        ? get_absolute_cq_offset(
                              channel, cq_id, cq_size, this->sysmem_manager_->get_dram_region_base_addr())
                        : get_absolute_cq_offset(channel, cq_id, cq_size);

                pointers[host_issue_q_rd_ptr / sizeof(uint32_t)] = (cq_start + cq_offset) >> 4;
                pointers[host_issue_q_wr_ptr / sizeof(uint32_t)] = (cq_start + cq_offset) >> 4;
                pointers[host_completion_q_wr_ptr / sizeof(uint32_t)] =
                    (cq_start + this->sysmem_manager_->get_issue_queue_size(cq_id) + cq_offset) >> 4;
                pointers[host_completion_q_rd_ptr / sizeof(uint32_t)] =
                    (cq_start + this->sysmem_manager_->get_issue_queue_size(cq_id) + cq_offset) >> 4;

                if (this->sysmem_manager_->is_dram_backed()) {
                    MetalEnvAccessor(*env_).impl().get_cluster().write_dram_vec(
                        pointers.data(), pointers.size() * sizeof(uint32_t), this->id(), 0, cq_offset);
                } else {
                    MetalEnvAccessor(*env_).impl().get_cluster().write_sysmem(
                        pointers.data(),
                        pointers.size() * sizeof(uint32_t),
                        cq_offset,
                        mmio_device_id,
                        get_umd_channel(channel));
                }
            }
        }
    }

    // Write device-side cq pointers
    TT_ASSERT(
        dispatch_topology != nullptr,
        "Dispatch topology required for configure_command_queue_programs (fast dispatch)");
    dispatch_topology->configure_dispatch_cores(this);

    // Run the cq program
    command_queue_program.impl().finalize_offsets(this);
    detail::ConfigureDeviceWithProgram(this, command_queue_program, true);
    MetalEnvAccessor(*env_).impl().get_cluster().l1_barrier(this->id());
}

void Device::init_command_queue_host() {
    // SystemMemoryManager now has internal stubs for mock/emulated devices
    sysmem_manager_ = std::make_unique<SystemMemoryManager>(context_->get_context_id(), this->id_, this->num_hw_cqs());

    // For mock/emulated devices, skip HWCommandQueue creation (they don't need real command queues)
    if (MetalEnvAccessor(*env_).impl().get_cluster().is_mock_or_emulated()) {
        return;
    }

    command_queues_.reserve(num_hw_cqs());
    for (size_t cq_id = 0; cq_id < num_hw_cqs(); cq_id++) {
        command_queues_.push_back(std::make_unique<HWCommandQueue>(this, cq_id, k_dispatch_downstream_noc));
    }
}

void Device::init_command_queue_device_with_topology(DispatchTopology* topo) {
    TT_ASSERT(topo != nullptr, "Dispatch topology required for init_command_queue_device_with_topology");
    this->command_queue_programs_.push_back(topo->get_compiled_cq_program(this));
    TT_ASSERT(this->command_queue_programs_.size() == 1);
    this->configure_command_queue_programs(topo);
    Program& command_queue_program = *this->command_queue_programs_[0];
    MetalEnvImpl& env_impl = MetalEnvAccessor(*env_).impl();
    auto& cluster = env_impl.get_cluster();
    const auto& hal = env_impl.get_hal();

    // Write 0 to all workers launch message read pointer. Need to do this since dispatch cores are written new on each
    // Device init. TODO: remove this once dispatch init moves to one-shot.
    auto reset_launch_message_rd_ptr_virtual = [&](const CoreCoord& virtual_core,
                                                   HalProgrammableCoreType programmable_core_type) {
        uint64_t launch_msg_buffer_read_ptr_addr =
            hal.get_dev_noc_addr(programmable_core_type, HalL1MemAddrType::LAUNCH_MSG_BUFFER_RD_PTR);
        uint32_t zero = 0;
        cluster.write_core(&zero, sizeof(uint32_t), tt_cxy_pair(id_, virtual_core), launch_msg_buffer_read_ptr_addr);
    };
    auto reset_launch_message_rd_ptr = [&](const CoreCoord& logical_core, const CoreType& core_type) {
        CoreCoord virtual_core = cluster.get_virtual_coordinate_from_logical_coordinates(id_, logical_core, core_type);
        auto programmable_core_type = get_programmable_core_type(virtual_core);
        reset_launch_message_rd_ptr_virtual(virtual_core, programmable_core_type);
    };
    auto reset_go_message_index = [&](const CoreCoord& logical_core, const CoreType& core_type) {
        CoreCoord virtual_core = cluster.get_virtual_coordinate_from_logical_coordinates(id_, logical_core, core_type);
        auto programmable_core_type = get_programmable_core_type(virtual_core);
        uint64_t go_message_addr = hal.get_dev_noc_addr(programmable_core_type, HalL1MemAddrType::GO_MSG);
        uint32_t zero = 0;
        cluster.write_core(&zero, sizeof(uint32_t), tt_cxy_pair(id_, virtual_core), go_message_addr);
        cluster.l1_barrier(id_);
        uint64_t go_message_index_addr = hal.get_dev_noc_addr(programmable_core_type, HalL1MemAddrType::GO_MSG_INDEX);
        cluster.write_core(&zero, sizeof(uint32_t), tt_cxy_pair(id_, virtual_core), go_message_index_addr);
    };
    std::optional<std::unique_lock<std::mutex>> watcher_lock;
    if (MetalEnvAccessor(*env_).impl().get_rtoptions().get_watcher_enabled()) {
        watcher_lock = context_->watcher_server()->get_lock();
    }
    for (uint32_t y = 0; y < logical_grid_size().y; y++) {
        for (uint32_t x = 0; x < logical_grid_size().x; x++) {
            CoreCoord logical_core(x, y);
            reset_launch_message_rd_ptr(logical_core, CoreType::WORKER);
            reset_go_message_index(logical_core, CoreType::WORKER);
        }
    }
    for (const auto& logical_core : this->get_active_ethernet_cores()) {
        if (!has_flag(env_impl.get_fabric_manager(), tt_fabric::FabricManagerMode::INIT_FABRIC)) {
            continue;
        }
        reset_launch_message_rd_ptr(logical_core, CoreType::ETH);
    }
    for (const auto& logical_core : this->get_inactive_ethernet_cores()) {
        if (!has_flag(env_impl.get_fabric_manager(), tt_fabric::FabricManagerMode::INIT_FABRIC)) {
            continue;
        }
        reset_launch_message_rd_ptr(logical_core, CoreType::ETH);
    }
    if (hal.has_programmable_core_type(HalProgrammableCoreType::DRAM)) {
        for (const auto& dram_core : cluster.get_soc_desc(id_).get_cores(CoreType::DRAM, CoordSystem::TRANSLATED)) {
            reset_launch_message_rd_ptr_virtual({dram_core.x, dram_core.y}, HalProgrammableCoreType::DRAM);
        }
    }
    if (watcher_lock) {
        watcher_lock.value().unlock();
    }

    std::vector<std::vector<CoreCoord>> logical_cores = command_queue_program.impl().logical_cores();
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
                hal.get_dev_addr(this->get_programmable_core_type(virtual_core), HalL1MemAddrType::LAUNCH));
        }
    }

    // Precompute NOC data for go signals and set on dispatch command queues
    const auto& active_eth_cores = get_active_ethernet_cores(true);
    std::vector<CoreRange> active_eth_core_ranges;
    active_eth_core_ranges.reserve(active_eth_cores.size());
    for (const auto& core : active_eth_cores) {
        active_eth_core_ranges.emplace_back(core, core);
    }

    const NOC noc_index = context_->get_dispatch_query_manager().go_signal_noc();
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

void Device::init_command_queue_device() { TT_FATAL(false, "Call init_command_queue_device_with_topology instead"); }

bool Device::compile_fabric() {
    fabric_program_ = tt::tt_fabric::create_and_compile_fabric_program(this);
    return fabric_program_ != nullptr;
}

void Device::configure_fabric(
    const std::unordered_set<uint32_t>& pre_dead_channels,
    const std::unordered_set<uint32_t>& skip_soft_reset_channels) {
    if (fabric_program_ == nullptr) {
        return;
    }

    // Reset relay-broken flag: configure_fabric() initialises fresh fabric firmware on all
    // channels, including the MMIO device's relay ERISCs.  After this call the UMD relay path
    // is valid again, so fabric_relay_path_broken_ must be cleared so that the next quiesce
    // cycle does not falsely short-circuit Phase 2.5, Phase 3, and Phase 5 for non-MMIO
    // devices.  (#42429 — flag is set in quiesce ENTRY snapshot / Phase 5 catch block).
    fabric_relay_path_broken_ = false;

    // FIX P2 (#42429): Persist pre_dead_channels so quiesce Phase 5 can skip them.
    // These channels have no fabric firmware loaded; Phase 5's READY_FOR_TRAFFIC check must
    // not throw for them (they will never reach that state).
    fabric_pre_dead_channels_ = pre_dead_channels;

    // Returns FabricCoresHealth describing per-channel reset results.
    // newly_dead_channels: channels that NEWLY failed soft reset in this call (not pre-known).
    // If only pre-known dead channels exist, we can continue in degraded mode with a warning;
    // if newly-discovered dead channels appear, we must TT_THROW (all subsequent L1 writes to
    // this device would route through the dead ETH path and hang indefinitely).  See #42429.
    //
    // pre_dead_channels: channels already confirmed dead by terminate_stale_erisc_routers().
    // Passed through so configure_fabric_cores() can skip assert_risc_reset_at_core() for them
    // and avoid the indefinite hang that occurs on non-MMIO device channels in T3K (#42429).
    // skip_soft_reset_channels: channels with base-UMD relay firmware (0x49706550).
    // FIX M (#42429): soft reset would halt the relay BRISC and cascade → hang.
    const auto health = tt::tt_fabric::configure_fabric_cores(this, pre_dead_channels, skip_soft_reset_channels);
    if (!health.all_channels_healthy) {
        if (!health.newly_dead_channels.empty()) {
            // Truly unexpected new dead channels: ALL L1 writes to this device now route through
            // the dead ETH path and will hang.  Hard-throw immediately.
            TT_THROW(
                "configure_fabric: Device {} has {} newly-dead ETH channel(s) (soft reset timed out "
                "for channels not in pre_dead_channels). Cannot write fabric firmware — all L1 writes "
                "would hang on the dead ethernet path. Hardware requires a reset to recover.",
                this->id_,
                health.newly_dead_channels.size());
        }
        // All dead channels were pre-confirmed by terminate_stale_erisc_routers(); continue in
        // degraded mode.  Firmware will not be loaded on those channels but the remaining
        // channels are healthy and writes to them are safe.
        log_warning(
            tt::LogMetal,
            "configure_fabric: Device {} running in degraded mode — {} pre-confirmed dead ETH "
            "channel(s) skipped. Fabric firmware will not be loaded on those channels.",
            this->id_,
            pre_dead_channels.size());
    }

    // FIX C (#42429): Build the set of all dead ETH channels (pre-known + any newly found)
    // so we can skip write_launch_msg_to_core for dead ETH cores.  Attempting to write launch
    // messages to an ERISC core whose ETH relay is broken causes the NOC write to route through
    // the dead channel and hang (same failure mode as the l1_barrier above).
    const auto& all_dead_channels = pre_dead_channels.empty() ? health.newly_dead_channels
                                                                : pre_dead_channels;
    // Look up SOC descriptor once for the ETH-core→channel reverse mapping.
    MetalEnvImpl& env_impl = MetalEnvAccessor(*env_).impl();
    const auto& soc_desc_for_dead = env_impl.get_cluster().get_soc_desc(this->id_);

    fabric_program_->impl().finalize_offsets(this);

    // FIX D2 (#42429): Build the set of dead ETH logical cores so WriteRuntimeArgsToDevice
    // and ConfigureDeviceWithProgram can skip them.  Both functions write to L1 through the
    // ETH relay on non-MMIO devices; writing to a dead relay channel hangs indefinitely.
    // FIX C (write_launch_msg_to_core) already guards the launch-message writes below — these
    // two calls need the same treatment.
    std::unordered_set<CoreCoord> dead_eth_logical_cores;
    if (!all_dead_channels.empty()) {
        const auto& logical_cores_in_prog = fabric_program_->impl().logical_cores();
        const auto& hal_for_dead = env_impl.get_hal();
        for (uint32_t pct_idx = 0; pct_idx < logical_cores_in_prog.size(); pct_idx++) {
            if (hal_for_dead.get_core_type(pct_idx) != CoreType::ETH) {
                continue;
            }
            for (const auto& lc : logical_cores_in_prog[pct_idx]) {
                try {
                    auto eth_chan = soc_desc_for_dead.get_eth_channel_for_core(
                        tt::umd::CoreCoord(lc.x, lc.y, CoreType::ETH, CoordSystem::LOGICAL),
                        CoordSystem::LOGICAL);
                    if (all_dead_channels.count(eth_chan)) {
                        dead_eth_logical_cores.insert(lc);
                    }
                } catch (...) {
                    // Cannot resolve channel — conservatively skip this core.
                    dead_eth_logical_cores.insert(lc);
                }
            }
        }
        if (!dead_eth_logical_cores.empty()) {
            log_warning(
                tt::LogMetal,
                "configure_fabric: Device {} skipping WriteRuntimeArgsToDevice and "
                "ConfigureDeviceWithProgram for {} dead ETH logical core(s) to avoid relay hang",
                this->id_,
                dead_eth_logical_cores.size());
        }
    }

    detail::WriteRuntimeArgsToDevice(this, *fabric_program_, using_fast_dispatch_, dead_eth_logical_cores);
    detail::ConfigureDeviceWithProgram(this, *fabric_program_, using_fast_dispatch_, dead_eth_logical_cores);

    // Only issue l1_barrier if we have no dead ETH channels on this device.  l1_barrier
    // calls driver_->l1_membar which internally calls wait_for_non_mmio_flush — on a non-MMIO
    // device with stuck relay commands that call blocks until the relay drains, which can hang
    // indefinitely if the relay ERISC is dead.
    if (all_dead_channels.empty()) {
        env_impl.get_cluster().l1_barrier(this->id());
    } else {
        log_warning(
            tt::LogMetal,
            "configure_fabric: Device {} skipping l1_barrier (dead ETH channels present — "
            "l1_membar/wait_for_non_mmio_flush would hang on dead relay path)",
            this->id_);
    }
    std::vector<std::vector<CoreCoord>> logical_cores_used_in_program = fabric_program_->impl().logical_cores();
    const auto& hal = env_impl.get_hal();
    for (uint32_t programmable_core_type_index = 0; programmable_core_type_index < logical_cores_used_in_program.size();
         programmable_core_type_index++) {
        CoreType core_type = hal.get_core_type(programmable_core_type_index);
        for (const auto& logical_core : logical_cores_used_in_program[programmable_core_type_index]) {
            // FIX C: skip write_launch_msg_to_core for ETH cores whose channel is dead.
            // write_launch_msg_to_core calls tt_cluster::write_core → WriteToDevice → UMD
            // write_to_device.  On a non-MMIO device, this routes through the ETH relay;
            // if the relay ERISC is dead the write hangs indefinitely (no per-write timeout).
            if (core_type == CoreType::ETH && !all_dead_channels.empty()) {
                // Get the ETH channel for this logical core and skip if it is dead.
                try {
                    auto eth_chan = soc_desc_for_dead.get_eth_channel_for_core(
                        tt::umd::CoreCoord(logical_core.x, logical_core.y, CoreType::ETH, CoordSystem::LOGICAL),
                        CoordSystem::LOGICAL);
                    if (all_dead_channels.count(eth_chan)) {
                        log_debug(
                            tt::LogMetal,
                            "configure_fabric: Device {} skipping write_launch_msg_to_core for "
                            "dead ETH core ({},{}) channel {}",
                            this->id_,
                            logical_core.x,
                            logical_core.y,
                            eth_chan);
                        continue;
                    }
                } catch (const std::exception& e) {
                    // If we can't resolve the channel, skip the write conservatively.
                    log_warning(
                        tt::LogMetal,
                        "configure_fabric: Device {} cannot resolve ETH channel for logical core "
                        "({},{}) — skipping write_launch_msg_to_core to avoid potential hang: {}",
                        this->id_,
                        logical_core.x,
                        logical_core.y,
                        e.what());
                    continue;
                }
            }

            auto* kg = fabric_program_->impl().kernels_on_core(logical_core, programmable_core_type_index);
            dev_msgs::launch_msg_t::View msg = kg->launch_msg.view();
            dev_msgs::go_msg_t::ConstView go_msg = kg->go_msg.view();
            msg.kernel_config().host_assigned_id() = fabric_program_->get_runtime_id();

            auto physical_core = this->virtual_core_from_logical_core(logical_core, core_type);
            tt::llrt::write_launch_msg_to_core(
                this->id(),
                physical_core,
                msg,
                go_msg,
                hal.get_dev_addr(this->get_programmable_core_type(physical_core), HalL1MemAddrType::LAUNCH));
        }
    }
    log_info(tt::LogMetal, "Fabric initialized on Device {}", this->id_);
}

void Device::quiesce_and_restart_fabric_workers() {
    // Diagnostic: env toggle lets CI / repro runs skip this restart path entirely to isolate
    // whether the Tensix MUX restart is the cause of a post-quiesce hang. When set, we return
    // before any fabric MUX termination. See plan Experiment B.
    if (const char* env = std::getenv("TT_METAL_DISABLE_QUIESCE_FABRIC_RESTART");
        env != nullptr && env[0] != '\0' && env[0] != '0') {
        log_info(
            tt::LogMetal,
            "quiesce_and_restart_fabric_workers: Device {} early-return: "
            "TT_METAL_DISABLE_QUIESCE_FABRIC_RESTART={} (restart path disabled)",
            this->id(),
            env);
        return;
    }

    auto fabric_config = MetalContext::instance().get_fabric_config();
    if (!tt_fabric::is_tt_fabric_config(fabric_config)) {
        log_info(
            tt::LogMetal,
            "quiesce_and_restart_fabric_workers: Device {} early-return at guard L426: "
            "!is_tt_fabric_config(fabric_config={})",
            this->id(),
            static_cast<uint32_t>(fabric_config));
        return;
    }

    auto tensix_config_mode = MetalContext::instance().get_fabric_tensix_config();
    const bool has_tensix_mux = (tensix_config_mode != tt::tt_fabric::FabricTensixConfig::DISABLED);
    if (!has_tensix_mux) {
        log_info(
            tt::LogMetal,
            "quiesce_and_restart_fabric_workers: Device {} FabricTensixConfig::DISABLED — "
            "no Tensix MUX cores; running ERISC-only path (Phase 2.5 + Phase 3 ETH cores only). "
            "Phases 1/2/4 (Tensix MUX) are skipped.",
            this->id());
        // Do NOT return: Phase 2.5 (ERISC termination) is still required to drain in-flight ETH
        // packets before Phase 3 overwrites ERISC L1. Without it, orphaned packets corrupt the
        // next iteration's BRISC .text at 0x8220, causing hangs in AllGather teardown races.
    }

    const auto& control_plane = MetalContext::instance().get_control_plane();
    const auto& fabric_context = control_plane.get_fabric_context();
    const auto& builder_ctx = fabric_context.get_builder_context();

    if (builder_ctx.get_num_fabric_initialized_routers(this->id()) == 0) {
        log_info(
            tt::LogMetal,
            "quiesce_and_restart_fabric_workers: Device {} early-return at guard L439: "
            "get_num_fabric_initialized_routers == 0",
            this->id());
        return;
    }

    const auto fabric_node_id = control_plane.get_fabric_node_id_from_physical_chip_id(this->id());
    const auto& active_channels = control_plane.get_active_fabric_eth_channels(fabric_node_id);

    MetalEnvImpl& env_impl = MetalEnvAccessor(*env_).impl();

    // Diagnostic entry snapshot (#42429): read every active ERISC channel's edm_status_address
    // BEFORE any phase runs.  This lets us distinguish "prior test left channels in bad state"
    // vs "failing test caused the problem" — per Neil's request.
    {
        const auto router_sync_addr_diag =
            builder_ctx.get_fabric_router_sync_address_and_status().first;
        const auto& soc_desc_entry = env_impl.get_cluster().get_soc_desc(this->id());
        log_info(
            tt::LogMetal,
            "quiesce_and_restart_fabric_workers: Device {} ENTRY snapshot: "
            "{} active ERISC channel(s), edm_status_addr=0x{:08x}",
            this->id(),
            active_channels.size(),
            router_sync_addr_diag);

        // Snapshot deadline: if the relay path is already known broken, skip reads
        // entirely (all would throw 5s timeouts or hang).  Otherwise bound the total
        // snapshot time to kSnapshotDeadlineMs so a single hanging read cannot block
        // here indefinitely — the ENTRY snapshot is diagnostic-only and must never
        // gate the quiesce path.
        const auto snapshot_start = std::chrono::steady_clock::now();
        // Allow up to one UMD relay timeout worth of reads (6 s) before bailing.
        // In practice healthy reads take <1 ms each; this only fires when the relay
        // path is broken and reads start accumulating 5 s timeouts.
        constexpr int64_t kSnapshotDeadlineMs = 6000;

        if (fabric_relay_path_broken_) {
            log_warning(
                tt::LogMetal,
                "quiesce_and_restart_fabric_workers: Device {} ENTRY snapshot: "
                "relay path known broken (fabric_relay_path_broken_) — skipping all "
                "relay reads to prevent 5s-per-channel timeout accumulation.",
                this->id());
        } else {
        for (const auto& [eth_chan_id, direction] : active_channels) {
            // Per-read deadline: bail once we've spent kSnapshotDeadlineMs on snapshot
            // reads.  Prevents a single hanging chan (relay ERISC alive but peering with
            // fabric-firmware peer that never responds) from blocking indefinitely.
            const auto snapshot_elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
                                              std::chrono::steady_clock::now() - snapshot_start)
                                              .count();
            if (snapshot_elapsed > kSnapshotDeadlineMs) {
                // FIX S (#42429): Snapshot deadline exceeded means the UMD relay path to this
                // non-MMIO device is broken (relay reads accumulate 5s timeouts per channel).
                // Set fabric_relay_path_broken_ so Phase 2.5 and Phase 3 skip all relay
                // reads/writes — preventing the ~650s hang where some channels throw 5s
                // timeouts but others hang indefinitely without exception (relay ERISC alive
                // but peering firmware unresponsive to UMD protocol).
                fabric_relay_path_broken_ = true;
                log_warning(
                    tt::LogMetal,
                    "quiesce_and_restart_fabric_workers: Device {} ENTRY snapshot: "
                    "deadline ({}ms) exceeded after {}ms — relay path broken, setting "
                    "fabric_relay_path_broken_=true to skip Phase 2.5/3 relay reads.",
                    this->id(),
                    kSnapshotDeadlineMs,
                    snapshot_elapsed);
                break;
            }

            const auto eth_lc_diag = soc_desc_entry.get_eth_core_for_channel(
                eth_chan_id, CoordSystem::LOGICAL);
            std::vector<uint32_t> diag_buf(1, 0U);
            // Wrap in try/catch: non-MMIO reads route through the MMIO device's ERISC relay.
            // If the MMIO device was quiesced earlier in Pass 1, its ERISCs may not yet be
            // READY_FOR_TRAFFIC and UMD will timeout.  The ENTRY snapshot is diagnostic only —
            // log a warning but do NOT abort the quiesce.
            try {
                detail::ReadFromDeviceL1(
                    this, eth_lc_diag, router_sync_addr_diag, 4, diag_buf, CoreType::ETH);
            } catch (const std::exception& ex) {
                log_warning(
                    tt::LogMetal,
                    "quiesce_and_restart_fabric_workers: Device {} ENTRY snapshot: "
                    "eth_chan {} L1 read failed (relay not ready?): {}",
                    this->id(),
                    eth_chan_id,
                    ex.what());
                diag_buf[0] = 0xDEADBEEF;
            }
            log_info(
                tt::LogMetal,
                "quiesce_and_restart_fabric_workers: Device {} ENTRY snapshot: "
                "eth_chan {} (logical={}) edm_status=0x{:08x}",
                this->id(),
                eth_chan_id,
                eth_lc_diag.str(),
                diag_buf[0]);
        }
        }  // end else (!fabric_relay_path_broken_)
    }

    // Phase 1: Send IMMEDIATELY_TERMINATE to each MUX worker core
    // Skipped in ETH-only fabric mode (FabricTensixConfig::DISABLED) — no Tensix MUX workers exist.
    if (has_tensix_mux) {
        const auto& tensix_config = builder_ctx.get_tensix_config();
        for (const auto& [eth_chan_id, direction] : active_channels) {
            auto core_id = tensix_config.get_core_id_for_channel(this->id(), eth_chan_id);
            auto [term_addr, term_signal] = tensix_config.get_termination_address_and_signal(core_id);
            std::vector<uint32_t> term_buf(1, term_signal);
            auto mux_core = tensix_config.get_core_for_channel(this->id(), eth_chan_id);

            // Diagnostic: log pre-terminate status so we can confirm the MUX was READY_FOR_TRAFFIC
            // (not already dead/stuck) before we send the termination signal.
            auto config = tensix_config.get_config(core_id);
            uint32_t status_addr_p1 = static_cast<uint32_t>(config->get_status_address());
            std::vector<uint32_t> pre_status(1, 0);
            detail::ReadFromDeviceL1(this, mux_core, status_addr_p1, 4, pre_status, CoreType::WORKER);
            log_info(
                tt::LogMetal,
                "quiesce_and_restart_fabric_workers: Device {} eth_chan {} Phase 1: pre-terminate status=0x{:08x}",
                this->id(),
                eth_chan_id,
                pre_status[0]);

            detail::WriteToDeviceL1(this, mux_core, term_addr, term_buf, CoreType::WORKER);
        }

        env_impl.get_cluster().l1_barrier(this->id());
    }

    // Phase 2: Poll each MUX core until its status is TERMINATED.
    //
    // On timeout we DO NOT throw: this function is called from MeshDevice::quiesce_internal
    // between iterations; throwing mid-quiesce leaves the device half-torn-down and surfaces
    // as a confusing exception from user-facing `quiesce_devices()`.  Instead, mirror the ETH
    // router path in MetalEnvImpl::teardown_fabric_config and force-reset the stuck Tensix
    // MUX core so the next bring-up starts from a clean state. The caller continues to
    // re-configure and re-launch the fabric workers in Phase 3.
    // Skipped in ETH-only fabric mode (FabricTensixConfig::DISABLED) — no Tensix MUX workers exist.
    if (has_tensix_mux) {
        const auto& tensix_config = builder_ctx.get_tensix_config();
        for (const auto& [eth_chan_id, direction] : active_channels) {
            auto core_id = tensix_config.get_core_id_for_channel(this->id(), eth_chan_id);
            auto config = tensix_config.get_config(core_id);
            uint32_t status_addr = static_cast<uint32_t>(config->get_status_address());
            auto mux_core = tensix_config.get_core_for_channel(this->id(), eth_chan_id);

            std::vector<uint32_t> status_buf(1, 0);
            const auto start = std::chrono::steady_clock::now();
            constexpr uint32_t timeout_ms = 5000;
            constexpr uint32_t kSpinsBetweenSleeps = 64;
            uint32_t spin_counter = 0;
            bool terminated = false;
            while (true) {
                detail::ReadFromDeviceL1(this, mux_core, status_addr, 4, status_buf, CoreType::WORKER);
                if (status_buf[0] == static_cast<uint32_t>(tt::tt_fabric::EDMStatus::TERMINATED)) {
                    terminated = true;
                    break;
                }
                const auto elapsed =
                    std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - start).count();
                if (elapsed > timeout_ms) {
                    break;
                }
                if (++spin_counter >= kSpinsBetweenSleeps) {
                    spin_counter = 0;
                    std::this_thread::sleep_for(std::chrono::microseconds(100));
                } else {
                    ttsl::pause();
                }
            }

            log_info(
                tt::LogMetal,
                "quiesce_and_restart_fabric_workers: Device {} eth_chan {} Phase 2: {} (status=0x{:08x})",
                this->id(),
                eth_chan_id,
                terminated ? "TERMINATED cleanly" : "TIMEOUT",
                status_buf[0]);

            if (!terminated) {
                log_warning(
                    tt::LogMetal,
                    "quiesce_and_restart_fabric_workers: Timeout waiting for fabric MUX TERMINATED on "
                    "Device {} eth_chan {} (status=0x{:08x}), force-resetting Tensix MUX to prevent "
                    "stale NOC traffic into worker L1",
                    this->id(),
                    eth_chan_id,
                    status_buf[0]);
            }

            // Always halt the MUX BRISC before Phase 3 overwrites its L1, regardless of whether
            // termination was clean or timed out.
            //
            // The CCL MUX kernel writes TERMINATED *before* close_finish() completes (close_finish
            // spins on worker_teardown_addr waiting for the ERISC ACK).  So even when Phase 2 sees
            // a clean TERMINATED, the BRISC is still running close_finish().  If we proceed directly
            // to Phase 3's ConfigureDeviceWithProgram, the still-running BRISC executes whatever
            // instructions now reside in its overwritten L1, generating invalid NOC traffic — including
            // writes to ARC_RESET_SCRATCH_ADDR (0x880030060) — that corrupt ERISC or ARC state.
            //
            // Halting the core here makes Phase 3's L1 overwrite safe.  The BRISC will be deasserted
            // after write_launch_msg_to_core in Phase 3, which starts the new CCL MUX kernel from its
            // reset vector with clean state.
            try {
                const auto virtual_mux_coord = env_impl.get_cluster().get_virtual_coordinate_from_logical_coordinates(
                    this->id(), mux_core, CoreType::WORKER);
                env_impl.get_cluster().assert_risc_reset_at_core(
                    tt_cxy_pair(this->id(), virtual_mux_coord), tt::umd::RiscType::ALL);
            } catch (const std::exception& e) {
                log_warning(
                    tt::LogMetal,
                    "quiesce_and_restart_fabric_workers: assert_risc_reset_at_core failed on Device {} "
                    "eth_chan {}: {}",
                    this->id(),
                    eth_chan_id,
                    e.what());
            }
        }
    }

    // Phase 2.5: Terminate ERISC fabric routers before re-loading firmware.
    //
    // The Tensix MUX BRISC was halted in Phase 2, but the ERISC routers were not
    // explicitly told to stop.  Phase 3 calls configure_fabric_cores() which
    // unconditionally overwrites every active ERISC's L1 with fresh firmware.  If an
    // ERISC is still executing (e.g. draining the ETH TXQ after a send) when its L1 is
    // overwritten, it continues running with corrupted program state and generates invalid
    // NOC traffic — including writes to ARC scratch (0x880030060) — that corrupts device
    // state and causes the next iteration's AllGather to hang.
    //
    // We send the TERMINATE signal directly to each active ERISC channel and wait for
    // it to write EDMStatus::TERMINATED before proceeding.  On timeout we log a warning
    // and continue — configure_fabric_cores() will still safely overwrite L1 and boot
    // fresh firmware.  We do NOT assert_risc_reset_at_core on WH: resetting an ERISC
    // tears down the ETH PHY link and breaks non-MMIO L1 access for the whole mesh.
    // FIX R (#42429): Skip Phase 2.5 entirely for non-MMIO devices when relay path is broken.
    //
    // Phase 2.5 L1 reads go through the UMD non-MMIO relay.  When fabric_relay_path_broken_
    // is true the relay ERISC on the MMIO device runs fabric firmware, not relay firmware.
    // For some channels this causes the UMD relay read to throw a 5s timeout exception
    // (caught below and handled via `continue`), but for others (notably eth_chan 7 which
    // was already in TERMINATED state) the relay read hangs *indefinitely* with no exception —
    // producing the 665s CI kill gap observed in run #24920724360.
    //
    // Phase 3 already short-circuits for this same condition (see guard below), so TERMINATE
    // is redundant: configure_fabric_cores() in Phase 3 overwrites ERISC L1 regardless of
    // ERISC state.  Skipping Phase 2.5 relay reads here is safe and eliminates the hang.
    if (fabric_relay_path_broken_ && !this->is_mmio_capable()) {
        log_warning(
            tt::LogMetal,
            "quiesce_and_restart_fabric_workers: Device {} Phase 2.5: relay path known broken "
            "(fabric_relay_path_broken_) and device is non-MMIO — skipping all Phase 2.5 relay "
            "reads/writes to prevent indefinite UMD relay hang.  Phase 3 will also be skipped.",
            this->id());
    } else {
    {
        const auto [erisc_term_addr, erisc_term_signal] =
            builder_ctx.get_fabric_router_termination_address_and_signal();
        const auto router_sync_addr =
            builder_ctx.get_fabric_router_sync_address_and_status().first;
        constexpr uint32_t terminated_val = static_cast<uint32_t>(tt::tt_fabric::EDMStatus::TERMINATED);
        // 2000ms: extended from 150ms to observe actual termination latency.
        // ERISCs typically respond in <1ms; a longer window lets us see whether a slow
        // or unresponsive ERISC eventually self-terminates, which guides whether we need
        // a hardware reset vs. just waiting longer. Fast cases still exit the loop
        // immediately on the first successful read — no performance regression.
        constexpr uint32_t erisc_timeout_ms = 2000;
        // Log current status every 200ms if ERISC has not yet terminated.
        constexpr uint32_t kTermIntermediateLogMs = 200;
        constexpr uint32_t kSpinsBetweenSleeps = 64;

        std::vector<uint32_t> term_buf(1, static_cast<uint32_t>(erisc_term_signal));

        for (const auto& [eth_chan_id, direction] : active_channels) {
            const auto eth_logical_core = env_impl.get_cluster()
                                              .get_soc_desc(this->id())
                                              .get_eth_core_for_channel(eth_chan_id, CoordSystem::LOGICAL);

            std::vector<uint32_t> status_buf(1, 0);
            try {
                detail::ReadFromDeviceL1(this, eth_logical_core, router_sync_addr, 4, status_buf, CoreType::ETH);
            } catch (const std::exception& e) {
                // Non-MMIO relay read failed (e.g. UMD relay ERISC not running relay firmware
                // after Phase 3 loaded fabric firmware on the MMIO device's channels).  We cannot
                // safely send TERMINATE because WriteToDeviceL1 via the relay can hang
                // indefinitely with no write timeout.  Skip this channel — configure_fabric_cores()
                // in Phase 3 overwrites L1 regardless of ERISC state.
                log_warning(
                    tt::LogMetal,
                    "quiesce_and_restart_fabric_workers: Device {} eth_chan {} Phase 2.5: "
                    "L1 read failed (relay not ready?) — skipping TERMINATE: {}",
                    this->id(),
                    eth_chan_id,
                    e.what());
                continue;
            }

            if (status_buf[0] == 0 || status_buf[0] == terminated_val) {
                log_info(
                    tt::LogMetal,
                    "quiesce_and_restart_fabric_workers: Device {} eth_chan {} Phase 2.5: "
                    "ERISC already clean (status=0x{:08x}), skipping",
                    this->id(),
                    eth_chan_id,
                    status_buf[0]);
                continue;
            }

            log_info(
                tt::LogMetal,
                "quiesce_and_restart_fabric_workers: Device {} eth_chan {} Phase 2.5: "
                "ERISC active (status=0x{:08x}), sending TERMINATE",
                this->id(),
                eth_chan_id,
                status_buf[0]);

            detail::WriteToDeviceL1(this, eth_logical_core, erisc_term_addr, term_buf, CoreType::ETH);

            const auto start = std::chrono::steady_clock::now();
            uint32_t spin_counter = 0;
            bool terminated = false;
            int64_t last_term_log_ms = -1;
            while (true) {
                try {
                    detail::ReadFromDeviceL1(
                        this, eth_logical_core, router_sync_addr, 4, status_buf, CoreType::ETH);
                } catch (const std::exception& e) {
                    log_warning(
                        tt::LogMetal,
                        "quiesce_and_restart_fabric_workers: Device {} eth_chan {} Phase 2.5: "
                        "polling read failed — treating as timed out: {}",
                        this->id(),
                        eth_chan_id,
                        e.what());
                    break;
                }
                if (status_buf[0] == terminated_val) {
                    terminated = true;
                    break;
                }
                const auto elapsed =
                    std::chrono::duration_cast<std::chrono::milliseconds>(
                        std::chrono::steady_clock::now() - start)
                        .count();
                // Log every kTermIntermediateLogMs so we can observe actual termination latency
                // in CI logs and determine whether timeout thresholds need adjustment.
                if (elapsed / kTermIntermediateLogMs >
                    last_term_log_ms / static_cast<int64_t>(kTermIntermediateLogMs)) {
                    last_term_log_ms = elapsed;
                    log_info(
                        tt::LogMetal,
                        "quiesce_and_restart_fabric_workers: Device {} eth_chan {} Phase 2.5: "
                        "still waiting for TERMINATED after {}ms — current status=0x{:08x}",
                        this->id(),
                        eth_chan_id,
                        elapsed,
                        status_buf[0]);
                }
                if (elapsed > erisc_timeout_ms) {
                    break;
                }
                if (++spin_counter >= kSpinsBetweenSleeps) {
                    spin_counter = 0;
                    std::this_thread::sleep_for(std::chrono::microseconds(100));
                } else {
                    ttsl::pause();
                }
            }

            const auto p25_elapsed =
                std::chrono::duration_cast<std::chrono::milliseconds>(
                    std::chrono::steady_clock::now() - start)
                    .count();
            if (terminated) {
                log_info(
                    tt::LogMetal,
                    "quiesce_and_restart_fabric_workers: Device {} eth_chan {} Phase 2.5: "
                    "ERISC TERMINATED in {}ms",
                    this->id(),
                    eth_chan_id,
                    p25_elapsed);
            } else {
                // Do NOT assert_risc_reset_at_core on WH ERISCs: resetting tears down the
                // ETH PHY link, breaking non-MMIO L1 access for the entire mesh.
                // configure_fabric_cores() in Phase 3 will safely overwrite L1.
                log_warning(
                    tt::LogMetal,
                    "quiesce_and_restart_fabric_workers: Device {} eth_chan {} Phase 2.5: "
                    "ERISC did not terminate within {}ms (status=0x{:08x}) — continuing "
                    "without reset (WH: resetting ERISC tears down ETH PHY link)",
                    this->id(),
                    eth_chan_id,
                    erisc_timeout_ms,
                    status_buf[0]);
            }
        }
    }
    }  // end else (Phase 2.5 — relay path not broken)

    // Phase 3: Re-configure and re-launch the fabric workers
    // Reset termination signals, clear channel state, and re-send launch messages
    // for WORKER cores in the fabric program.
    if (fabric_program_ == nullptr) {
        log_info(
            tt::LogMetal,
            "quiesce_and_restart_fabric_workers: Device {} early-return at guard L564: "
            "fabric_program_ == nullptr (Phase 1/2 ran but Phase 3/4 skipped)",
            this->id());
        return;
    }

    log_info(tt::LogMetal, "quiesce_and_restart_fabric_workers: Device {} entering Phase 3 (re-configure + re-launch)", this->id());

    // FIX Q (#42429): Skip Phase 3 entirely for non-MMIO devices when the relay path is known broken.
    //
    // All Phase 3 operations (configure_fabric_cores L1 writes, WriteRuntimeArgsToDevice,
    // ConfigureDeviceWithProgram, l1_barrier, write_launch_msg_to_core) route through the
    // UMD non-MMIO relay for non-MMIO devices.  When fabric_relay_path_broken_ is true, the
    // relay ERISC on the MMIO device is running fabric firmware (not UMD relay firmware), so
    // relay WRITES have no timeout and hang indefinitely — unlike relay reads which throw a
    // 5s UMD timeout.  This causes the entire test process to hang until the CI job-level
    // 660s timeout fires.
    //
    // When fabric_relay_path_broken_ is true, Phase 5 (wait_for_fabric_workers_ready) is also
    // skipped, so there is no handshake to complete for this device.  Skipping Phase 3 is
    // safe: the device state is already degraded, and the relay cannot be used until the
    // next TT-Metal session initializes the fabric from scratch.
    if (fabric_relay_path_broken_ && !this->is_mmio_capable()) {
        log_warning(
            tt::LogMetal,
            "quiesce_and_restart_fabric_workers: Device {} Phase 3: relay path known broken "
            "(fabric_relay_path_broken_) and device is non-MMIO — skipping all relay writes "
            "(configure_fabric_cores, WriteRuntimeArgs, l1_barrier, write_launch_msg) to prevent "
            "indefinite relay hang.  Phase 5 will also be skipped for this device.",
            this->id());
        log_info(
            tt::LogMetal,
            "quiesce_and_restart_fabric_workers: Device {} Phase 3 skipped — "
            "relay path broken; configure_fabric_cores/WriteRuntimeArgs/write_launch_msg "
            "omitted. wait_for_fabric_workers_ready() will also skip Phase 5 for this device.",
            this->id());
        return;
    }

    // FIX N (#42429): For non-MMIO devices, skip soft reset in configure_fabric_cores.
    //
    // assert_risc_reset on a non-MMIO ETH channel resets the entire ETH core including
    // the NOC router.  The subsequent deassert_risc_reset write must be delivered via
    // the UMD relay — through that same NOC which is now in reset — causing
    // RemoteCommunicationLegacyFirmware::wait_for_non_mmio_flush to time out and
    // permanently damaging the relay endpoint, making ALL subsequent non-MMIO writes fail.
    //
    // After Phase 2.5 the ETH channels are in TERMINATED state.  TERMINATED firmware
    // sits in a halt loop that polls for launch messages.  write_launch_msg_to_core
    // (issued explicitly below, after L1 is loaded) is sufficient to restart the fabric
    // router without soft reset.
    // FIX M confirmed this mechanism works: base-UMD → fabric firmware transition via
    // write_launch_msg_to_core alone succeeded in CI run #24830129500.
    tt::tt_fabric::FabricCoresHealth quiesce_health;
    {
        std::unordered_set<uint32_t> quiesce_skip_reset_chans;
        if (!this->is_mmio_capable()) {
            for (const auto& [chan, unused_direction] : active_channels) {
                quiesce_skip_reset_chans.insert(chan);
            }
        }
        quiesce_health = tt::tt_fabric::configure_fabric_cores(this, {}, quiesce_skip_reset_chans);
    }
    // Log configure_fabric_cores health — newly_dead_channels here means configure itself
    // killed channels (e.g. assert_risc_reset_at_core timed out).  These channels will be
    // skipped in write_launch_msg_to_core below and won't reach READY_FOR_TRAFFIC.
    log_info(
        tt::LogMetal,
        "quiesce_and_restart_fabric_workers: Device {} Phase 3: configure_fabric_cores complete — "
        "newly_dead={}",
        this->id(),
        quiesce_health.newly_dead_channels.size());
    if (!quiesce_health.newly_dead_channels.empty()) {
        std::string newly_dead_str;
        for (auto ch : quiesce_health.newly_dead_channels) {
            newly_dead_str += fmt::format("{} ", ch);
        }
        log_warning(
            tt::LogMetal,
            "quiesce_and_restart_fabric_workers: Device {} Phase 3: configure_fabric_cores newly-dead "
            "channels: [{}] — write_launch_msg will be skipped for these",
            this->id(),
            newly_dead_str);
    }
    detail::WriteRuntimeArgsToDevice(this, *fabric_program_, using_fast_dispatch_);
    detail::ConfigureDeviceWithProgram(this, *fabric_program_, using_fast_dispatch_);

    env_impl.get_cluster().l1_barrier(this->id());

    std::vector<std::vector<CoreCoord>> logical_cores_used = fabric_program_->impl().logical_cores();
    const auto& hal = env_impl.get_hal();
    const auto& soc_desc_q = env_impl.get_cluster().get_soc_desc(this->id());

    // Re-launch worker cores from the fabric program.
    // WORKER (Tensix MUX) cores were halted in Phase 2; deassert BRISC reset after writing
    // the launch message so the new kernel starts executing.
    for (uint32_t pct_idx = 0; pct_idx < logical_cores_used.size(); pct_idx++) {
        CoreType core_type = hal.get_core_type(pct_idx);
        if (core_type != CoreType::WORKER) {
            continue;
        }
        for (const auto& logical_core : logical_cores_used[pct_idx]) {
            auto* kg = fabric_program_->impl().kernels_on_core(logical_core, pct_idx);
            dev_msgs::launch_msg_t::View msg = kg->launch_msg.view();
            dev_msgs::go_msg_t::ConstView go_msg = kg->go_msg.view();
            msg.kernel_config().host_assigned_id() = fabric_program_->get_runtime_id();

            auto physical_core = this->virtual_core_from_logical_core(logical_core, core_type);
            tt::llrt::write_launch_msg_to_core(
                this->id(),
                physical_core,
                msg,
                go_msg,
                hal.get_dev_addr(this->get_programmable_core_type(physical_core), HalL1MemAddrType::LAUNCH));

            // Deassert BRISC reset so the new CCL MUX kernel executes.  The BRISC was halted at
            // the end of Phase 2 (after seeing TERMINATED) to prevent the old kernel's
            // close_finish() from executing garbled instructions while Phase 3 overwrites its L1.
            // Now that the new binary is loaded and the launch message is written, start the core.
            // We deassert only BRISC (not NCRISC/TRISC) since the CCL MUX kernel runs on BRISC.
            try {
                env_impl.get_cluster().deassert_risc_reset_at_core(
                    tt_cxy_pair(this->id(), physical_core), tt::umd::RiscType::BRISC);
            } catch (const std::exception& e) {
                log_warning(
                    tt::LogMetal,
                    "quiesce_and_restart_fabric_workers: deassert_risc_reset_at_core failed on Device {} "
                    "core ({},{}): {}",
                    this->id(),
                    physical_core.x,
                    physical_core.y,
                    e.what());
            }
        }
    }

    // FIX O (#42429): Re-launch ETH ERISC cores from the fabric program.
    //
    // configure_fabric_cores() only performs soft reset (or skips it for non-MMIO) and
    // clears L1.  It does NOT send write_launch_msg_to_core to ETH cores — that is done
    // explicitly here and in configure_fabric().
    //
    // After Phase 2.5, ERISC channels are in TERMINATED state with BRISC polling for a
    // launch message.  Without this loop, no launch message is ever sent, BRISC stays in
    // its poll loop, status address stays at 0x00000000, and Phase 5 throws:
    //   "Fabric health check failed — 4 ERISC channel(s) not at READY_FOR_TRAFFIC (0x00000000)"
    //
    // ETH cores must NOT have deassert_risc_reset_at_core called here — their BRISC was not
    // halted in Phase 2 (resetting ERISC tears down the ETH PHY link on WH).
    for (uint32_t pct_idx = 0; pct_idx < logical_cores_used.size(); pct_idx++) {
        CoreType core_type = hal.get_core_type(pct_idx);
        if (core_type != CoreType::ETH) {
            continue;
        }
        for (const auto& logical_core : logical_cores_used[pct_idx]) {
            // Skip channels that newly died during configure_fabric_cores() — writing launch
            // messages through a dead ETH relay hangs indefinitely.
            if (!quiesce_health.newly_dead_channels.empty()) {
                try {
                    auto eth_chan = soc_desc_q.get_eth_channel_for_core(
                        tt::umd::CoreCoord(logical_core.x, logical_core.y, CoreType::ETH, CoordSystem::LOGICAL),
                        CoordSystem::LOGICAL);
                    if (quiesce_health.newly_dead_channels.count(eth_chan)) {
                        log_warning(
                            tt::LogMetal,
                            "quiesce_and_restart_fabric_workers: Device {} skipping "
                            "write_launch_msg_to_core for dead ETH core ({},{}) channel {}",
                            this->id(),
                            logical_core.x,
                            logical_core.y,
                            eth_chan);
                        continue;
                    }
                } catch (...) {
                    log_warning(
                        tt::LogMetal,
                        "quiesce_and_restart_fabric_workers: Device {} cannot resolve ETH channel "
                        "for logical core ({},{}) — skipping write_launch_msg_to_core",
                        this->id(),
                        logical_core.x,
                        logical_core.y);
                    continue;
                }
            }

            auto* kg = fabric_program_->impl().kernels_on_core(logical_core, pct_idx);
            dev_msgs::launch_msg_t::View msg = kg->launch_msg.view();
            dev_msgs::go_msg_t::ConstView go_msg = kg->go_msg.view();
            msg.kernel_config().host_assigned_id() = fabric_program_->get_runtime_id();

            auto physical_core = this->virtual_core_from_logical_core(logical_core, core_type);

            // Pre-launch status read: confirm ERISC is in TERMINATED (0xA4B4C4D4) or 0x0
            // before we send the launch message.  Any other value means Phase 2.5 didn't
            // terminate the ERISC cleanly and we may be overwriting a live core.
            {
                const auto [erisc_sync_addr_pre, unused_pre] =
                    builder_ctx.get_fabric_router_sync_address_and_status();
                std::vector<uint32_t> pre_launch_buf(1, 0U);
                try {
                    detail::ReadFromDeviceL1(
                        this, logical_core, erisc_sync_addr_pre, 4, pre_launch_buf, CoreType::ETH);
                } catch (...) {
                    pre_launch_buf[0] = 0xDEADBEEF;
                }
                log_info(
                    tt::LogMetal,
                    "quiesce_and_restart_fabric_workers: Device {} Phase 3: "
                    "write_launch_msg_to_core ETH logical ({},{}) pre_status=0x{:08x}",
                    this->id(),
                    logical_core.x,
                    logical_core.y,
                    pre_launch_buf[0]);
            }

            tt::llrt::write_launch_msg_to_core(
                this->id(),
                physical_core,
                msg,
                go_msg,
                hal.get_dev_addr(this->get_programmable_core_type(physical_core), HalL1MemAddrType::LAUNCH));

            log_info(
                tt::LogMetal,
                "quiesce_and_restart_fabric_workers: Device {} Phase 3: "
                "write_launch_msg_to_core ETH logical ({},{}) done",
                this->id(),
                logical_core.x,
                logical_core.y);
        }
    }

    // FIX P REMOVED (#42429): The per-device MAGIC injection (FIX P) was removed because it
    // caused a regression when BOTH MMIO and non-MMIO devices have
    // get_num_fabric_initialized_routers > 0 (i.e. both run through quiesce).  FIX P
    // completed the non-MMIO receiver handshake BEFORE the MMIO device's sender ERISCs were
    // relaunched.  When the MMIO device's Phase 3 then launched its senders, they wrote MAGIC
    // to the non-MMIO device — but it was already at READY_FOR_TRAFFIC and no longer in the
    // receiver loop.  The MMIO senders waited forever for REMOTE_HANDSHAKE_COMPLETE.
    //
    // The fix is structural: the mesh-level caller now runs Phase 3 (relaunch) on ALL
    // devices first, then runs wait_for_fabric_workers_ready() on all devices.  This
    // ensures both sender and receiver ERISCs are running before the host polls for
    // handshake completion.  The natural sender-receiver handshake completes without
    // host intervention, and wait_for_fabric_workers_ready() replicates the
    // wait_for_fabric_router_sync() pattern (poll LOCAL_HANDSHAKE_COMPLETE on master,
    // write READY_FOR_TRAFFIC).

    log_info(
        tt::LogMetal,
        "quiesce_and_restart_fabric_workers: Device {} Phase 3 complete — "
        "all cores relaunched. Handshake completion deferred to wait_for_fabric_workers_ready().",
        this->id());
    log_info(
        tt::LogMetal,
        "quiesce_and_restart_fabric_workers: Device {} SUMMARY — "
        "relay_path_broken={} mmio={} "
        "Phase1=done Phase2=done Phase2.5={} Phase3={} "
        "(wait_for_fabric_workers_ready() handles Phase4+Phase5)",
        this->id(),
        fabric_relay_path_broken_,
        this->is_mmio_capable(),
        (fabric_relay_path_broken_ && !this->is_mmio_capable()) ? "skipped(relay_broken)" : "done",
        (fabric_relay_path_broken_ && !this->is_mmio_capable()) ? "skipped(relay_broken)" : "done");
}

void Device::wait_for_fabric_workers_ready() {
    // This method is the second pass of the two-pass quiesce restart.
    // quiesce_and_restart_fabric_workers() (first pass) terminates and relaunches ALL cores
    // on ALL devices.  This method then waits for readiness — matching the initial startup
    // pattern where configure_fabric() runs on all devices, then wait_for_fabric_router_sync()
    // polls them in tunnel order (farthest-first).
    //
    // The caller (MeshDevice::wait_for_fabric_workers_ready_for_quiesce) must invoke this on
    // all devices AFTER all devices have completed quiesce_and_restart_fabric_workers().

    auto fabric_config = MetalContext::instance().get_fabric_config();
    if (!tt_fabric::is_tt_fabric_config(fabric_config)) {
        return;
    }

    const auto& control_plane = MetalContext::instance().get_control_plane();
    const auto& fabric_context = control_plane.get_fabric_context();
    const auto& builder_ctx = fabric_context.get_builder_context();

    if (builder_ctx.get_num_fabric_initialized_routers(this->id()) == 0) {
        return;
    }

    // FIX I2 (#42429): This MMIO device's master ETH channel connects to a dead-relay peer.
    // Firmware was loaded on this device but the peer will never complete the handshake —
    // the peer's ETH relay is broken.  Waiting here would always time out (10s) then throw.
    // This mirrors the init-time FIX I skip in wait_for_fabric_router_sync() and
    // verify_all_fabric_channels_healthy().
    if (fabric_is_mmio_dead_peer_device_) {
        log_warning(
            tt::LogMetal,
            "wait_for_fabric_workers_ready: Device {} is an MMIO dead-peer device — "
            "skipping Phase 5 handshake poll + health check (master ETH peer is dead-relay, "
            "firmware loaded but peer handshake will never complete). (#42429 FIX I2)",
            this->id());
        return;
    }

    // Skip Phase 5 entirely when the relay path to this device is known broken.
    // fabric_relay_path_broken_ is set on the first quiesce where Phase 5's relay
    // read threw (UMD relay timeout or relay ERISC on the MMIO device running fabric
    // firmware).  GTest TearDown triggers a second quiesce — without this guard, that
    // second call would repeat the same failing/hanging relay reads.
    if (fabric_relay_path_broken_) {
        log_warning(
            tt::LogMetal,
            "wait_for_fabric_workers_ready: Device {} relay path is known broken — "
            "skipping Phase 5 + Phase 5b entirely (relay read would hang or throw 5s "
            "timeout; fabric_relay_path_broken_ set on prior quiesce call).",
            this->id());
        return;
    }

    const auto fabric_node_id = control_plane.get_fabric_node_id_from_physical_chip_id(this->id());
    const auto& active_channels = control_plane.get_active_fabric_eth_channels(fabric_node_id);

    MetalEnvImpl& env_impl = MetalEnvAccessor(*env_).impl();

    auto tensix_config_mode = MetalContext::instance().get_fabric_tensix_config();
    const bool has_tensix_mux = (tensix_config_mode != tt::tt_fabric::FabricTensixConfig::DISABLED);

    // Phase 4: Wait for each MUX core to reach READY_FOR_TRAFFIC before returning.
    //
    // Without this wait, the next dispatch op can arrive while the MUX is still in its
    // startup path (waiting for ERISC, opening the EDM connection, etc.).  The dispatch
    // relay kernel calls wait_for_fabric_endpoint_ready(mux). That wait is bounded, but
    // proceeding after it times out still leaves dispatch connected to an endpoint that
    // never opened, which later looks like an opaque CQ or AllGather hang.
    //
    // Use a bounded poll, then force-reset and throw on timeout. Continuing after the
    // reset would leave the MUX halted without a relaunch in this phase; failing here
    // preserves the first useful diagnostic instead of letting the next operation hang.
    // Skipped in ETH-only fabric mode (FabricTensixConfig::DISABLED) — no Tensix MUX workers exist.
    if (has_tensix_mux) {
        const auto& tensix_config = builder_ctx.get_tensix_config();
        for (const auto& [eth_chan_id, direction] : active_channels) {
            auto core_id = tensix_config.get_core_id_for_channel(this->id(), eth_chan_id);
            auto config = tensix_config.get_config(core_id);
            uint32_t status_addr = static_cast<uint32_t>(config->get_status_address());
            auto mux_core = tensix_config.get_core_for_channel(this->id(), eth_chan_id);

            std::vector<uint32_t> status_buf(1, 0);
            const auto start = std::chrono::steady_clock::now();
            constexpr uint32_t timeout_ms = 5000;
            constexpr uint32_t kSpinsBetweenSleeps = 64;
            uint32_t spin_counter = 0;
            bool ready = false;
            while (true) {
                detail::ReadFromDeviceL1(this, mux_core, status_addr, 4, status_buf, CoreType::WORKER);
                if (status_buf[0] == static_cast<uint32_t>(tt::tt_fabric::EDMStatus::READY_FOR_TRAFFIC)) {
                    ready = true;
                    break;
                }
                const auto elapsed =
                    std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - start).count();
                if (elapsed > timeout_ms) {
                    break;
                }
                if (++spin_counter >= kSpinsBetweenSleeps) {
                    spin_counter = 0;
                    std::this_thread::sleep_for(std::chrono::microseconds(100));
                } else {
                    ttsl::pause();
                }
            }

            const auto p4_elapsed =
                std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - start).count();
            log_info(
                tt::LogMetal,
                "quiesce_and_restart_fabric_workers: Device {} eth_chan {} Phase 4: {} in {}ms (status=0x{:08x})",
                this->id(),
                eth_chan_id,
                ready ? "READY_FOR_TRAFFIC" : "TIMEOUT",
                p4_elapsed,
                status_buf[0]);

            if (!ready) {
                log_warning(
                    tt::LogMetal,
                    "wait_for_fabric_workers_ready: Timeout waiting for fabric MUX READY_FOR_TRAFFIC on "
                    "Device {} eth_chan {} (status=0x{:08x}), force-resetting Tensix MUX core before failing",
                    this->id(),
                    eth_chan_id,
                    status_buf[0]);
                try {
                    const auto virtual_mux_coord = env_impl.get_cluster().get_virtual_coordinate_from_logical_coordinates(
                        this->id(), mux_core, CoreType::WORKER);
                    env_impl.get_cluster().assert_risc_reset_at_core(
                        tt_cxy_pair(this->id(), virtual_mux_coord), tt::umd::RiscType::ALL);
                } catch (const std::exception& e) {
                    log_warning(
                        tt::LogMetal,
                        "quiesce_and_restart_fabric_workers: assert_risc_reset_at_core failed on Device {} "
                        "eth_chan {}: {}",
                        this->id(),
                        eth_chan_id,
                        e.what());
                }
                TT_THROW(
                    "Fabric MUX did not reach READY_FOR_TRAFFIC after quiesce restart on Device {} eth_chan {} "
                    "(status=0x{:08x}, waited {}ms). The MUX was reset and the restart is aborted so the next "
                    "dispatch/AllGather does not connect to a non-ready fabric endpoint.",
                    this->id(),
                    eth_chan_id,
                    status_buf[0],
                    timeout_ms);
            }
        }
    }

    // Phase 5: Wait for ERISC handshake completion — replicates wait_for_fabric_router_sync().
    //
    // At this point, ALL devices have completed Phase 3 (the mesh-level caller ensures this).
    // Both sender and receiver ERISCs are running and will naturally complete the EDM
    // handshake (sender writes MAGIC to receiver's local_value via eth_send_packet, receiver
    // responds by copying its scratch to sender's local_value).
    //
    // We poll the master ERISC channel for LOCAL_HANDSHAKE_COMPLETE, then write
    // READY_FOR_TRAFFIC — exactly what wait_for_fabric_router_sync() does at initial startup.
    // After that, a final per-channel health check confirms all channels are healthy.
    {
        const auto [router_sync_addr, sync_status] =
            builder_ctx.get_fabric_router_sync_address_and_status();
        constexpr uint32_t expected_ready =
            static_cast<uint32_t>(tt::tt_fabric::EDMStatus::READY_FOR_TRAFFIC);
        // 10s matches the fabric router sync timeout used at initial startup.
        constexpr uint32_t kSyncTimeoutMs = 10000;
        constexpr uint32_t kSpinLimit = 64U;
        // Log intermediate status every 2.5s so we can see whether the ERISC is stuck at
        // STARTED (0xA0B0C0D0), 0x0 (never launched), or something unexpected.
        constexpr uint32_t kIntermediateLogIntervalMs = 2500;

        const auto master_chan = builder_ctx.get_fabric_master_router_chan(this->id());
        const auto& soc_desc_p5 = env_impl.get_cluster().get_soc_desc(this->id());
        const auto master_logical_core =
            soc_desc_p5.get_eth_core_for_channel(master_chan, CoordSystem::LOGICAL);

        // Track whether Phase 5 relay read threw so Phase 5b can skip reads that would hang.
        // When the Phase 5 master-channel read throws, it means the UMD relay path to this
        // device is broken (fabric firmware was loaded on the relay ERISC after Phase 3).
        // In that state, Phase 5b reads would either throw 5-second timeouts (accumulating
        // seconds per channel) or hang indefinitely (relay ERISC alive but peer fabric
        // firmware never responds to UMD relay protocol).  Skip Phase 5b entirely instead.
        bool phase5_relay_read_threw = false;

        // Skip the handshake poll if the master channel is pre-known dead.
        const bool master_is_dead = fabric_pre_dead_channels_.count(master_chan) > 0;
        if (!master_is_dead) {
            log_info(
                tt::LogMetal,
                "wait_for_fabric_workers_ready: Device {} Phase 5: polling master chan {} for "
                "LOCAL_HANDSHAKE_COMPLETE (expected=0x{:08x})",
                this->id(),
                master_chan,
                sync_status);
            std::vector<uint32_t> sync_buf(1, 0U);
            const auto sync_start = std::chrono::steady_clock::now();
            uint32_t sync_spin = 0U;
            int64_t last_log_ms = -1;
            while (true) {
                try {
                    detail::ReadFromDeviceL1(
                        this, master_logical_core, router_sync_addr, 4, sync_buf, CoreType::ETH);
                } catch (const std::exception& e) {
                    phase5_relay_read_threw = true;
                    // Persist across quiesce calls: TearDown will trigger a second quiesce,
                    // and without this flag both wait_for_fabric_workers_ready() (Phase 5)
                    // and quiesce_and_restart_fabric_workers() (ENTRY snapshot) would
                    // attempt relay reads to a device whose path is now broken — each read
                    // either accumulates a 5s UMD timeout or hangs indefinitely.
                    fabric_relay_path_broken_ = true;
                    log_warning(
                        tt::LogMetal,
                        "wait_for_fabric_workers_ready: Device {} Phase 5: read failed on master "
                        "chan {} — {}.  Treating as timeout.",
                        this->id(),
                        master_chan,
                        e.what());
                    break;
                }
                if (sync_buf[0] == sync_status) {
                    break;
                }
                // Also accept READY_FOR_TRAFFIC — this can happen if a peer device's
                // wait_for_fabric_workers_ready already wrote READY_FOR_TRAFFIC to its master
                // and the master propagated it here.
                if (sync_buf[0] == expected_ready) {
                    break;
                }
                const auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
                                         std::chrono::steady_clock::now() - sync_start)
                                         .count();
                // Periodic intermediate log every kIntermediateLogIntervalMs — tells us whether
                // ERISC is stuck at STARTED (launched but no peer), at 0x0 (launch msg never
                // arrived), or at some unexpected value.
                if (elapsed / kIntermediateLogIntervalMs > last_log_ms / static_cast<int64_t>(kIntermediateLogIntervalMs)) {
                    last_log_ms = elapsed;
                    log_info(
                        tt::LogMetal,
                        "wait_for_fabric_workers_ready: Device {} Phase 5: still waiting for "
                        "LOCAL_HANDSHAKE_COMPLETE on master chan {} after {}ms — "
                        "current status=0x{:08x} (expected=0x{:08x})",
                        this->id(),
                        master_chan,
                        elapsed,
                        sync_buf[0],
                        sync_status);
                }
                if (elapsed > kSyncTimeoutMs) {
                    log_warning(
                        tt::LogMetal,
                        "wait_for_fabric_workers_ready: Device {} Phase 5: timeout ({}ms) waiting "
                        "for LOCAL_HANDSHAKE_COMPLETE on master chan {} (status=0x{:08x}, "
                        "expected=0x{:08x}). Continuing to health check.",
                        this->id(),
                        kSyncTimeoutMs,
                        master_chan,
                        sync_buf[0],
                        sync_status);
                    break;
                }
                if (++sync_spin >= kSpinLimit) {
                    sync_spin = 0U;
                    std::this_thread::sleep_for(std::chrono::microseconds(100));
                }
            }

            // Write READY_FOR_TRAFFIC to master channel — master firmware distributes to all
            // subordinates via notify_subordinate_routers(), matching wait_for_fabric_router_sync().
            if (sync_buf[0] == sync_status || sync_buf[0] == expected_ready) {
                auto ready_sig = builder_ctx.get_fabric_router_ready_address_and_signal();
                if (ready_sig) {
                    std::vector<uint32_t> ready_buf(1, static_cast<uint32_t>(ready_sig->second));
                    detail::WriteToDeviceL1(
                        this, master_logical_core, ready_sig->first, ready_buf, CoreType::ETH);
                }
                log_info(
                    tt::LogMetal,
                    "wait_for_fabric_workers_ready: Device {} Phase 5: master chan {} sync "
                    "complete (0x{:08x}), READY_FOR_TRAFFIC written",
                    this->id(),
                    master_chan,
                    sync_buf[0]);
            }
        } else {
            log_warning(
                tt::LogMetal,
                "wait_for_fabric_workers_ready: Device {} Phase 5: master chan {} is pre-known "
                "dead — skipping handshake poll",
                this->id(),
                master_chan);
        }

        // Phase 5b: Post-ready ERISC health check.
        //
        // After the handshake poll + READY_FOR_TRAFFIC write, verify that ALL active ERISC
        // channels have reached READY_FOR_TRAFFIC.  Extended to 2000ms (was 500ms) to observe
        // how long propagation from master to subordinate channels actually takes and whether
        // 500ms was cutting off channels that would have become healthy shortly after.
        //
        // Skip Phase 5b entirely if Phase 5 reads threw (relay path broken after Phase 3).
        // In that case, channel reads in Phase 5b either accumulate 5-second timeouts per
        // channel or hang indefinitely (relay ERISC still alive but peer fabric firmware never
        // responds to UMD relay protocol) — both outcomes block the quiesce path.
        if (phase5_relay_read_threw) {
            // Phase 5 relay read failed — the UMD relay path to this non-MMIO device is
            // broken.  Proceeding to AllGather on fabric that cannot be verified would risk
            // silent data corruption or a harder-to-diagnose hang downstream.  Abort here
            // with a clear error so the test fails fast with a useful message rather than
            // hanging indefinitely or corrupting data silently.
            //
            // NOTE: If Phase 5's own read hangs (rather than throws — possible when the relay
            // ERISC is alive but forwards to a peer running fabric firmware), we never reach
            // this throw.  That scenario requires either a non-blocking UMD probe API or a
            // thread-based timeout on ReadFromDeviceL1, which is not yet implemented.
            TT_THROW(
                "wait_for_fabric_workers_ready: Device {}: Phase 5 relay read failed — "
                "UMD relay path to this non-MMIO device is broken (relay ERISC on the MMIO "
                "device now runs fabric firmware, so subsequent reads via that relay either "
                "throw a 5s UMD timeout or hang indefinitely).  "
                "Cannot verify fabric health.  Aborting to prevent AllGather proceeding on "
                "degraded fabric.  If this fires consistently, check whether "
                "terminate_stale_erisc_routers skipped soft-resetting a channel that should "
                "have been reset before Phase 3 firmware load.",
                this->id());
        } else {
        constexpr uint32_t kHealthCheckTimeoutMs = 2000;
        // Log unhealthy channels every 200ms for observability.
        constexpr uint32_t kHCIntermediateLogMs = 200;

        struct UnhealthyChannel {
            uint32_t eth_chan_id;
            uint32_t actual_status;
        };

        struct ChanToCheck {
            uint32_t eth_chan_id;
            CoreCoord eth_logical_core;
        };
        std::vector<ChanToCheck> chans;
        for (const auto& [eth_chan_id, direction] : active_channels) {
            const auto eth_logical_core =
                soc_desc_p5.get_eth_core_for_channel(eth_chan_id, CoordSystem::LOGICAL);
            chans.push_back({eth_chan_id, eth_logical_core});
        }

        // Poll all channels until healthy or timeout.
        std::vector<size_t> pending;
        pending.reserve(chans.size());
        for (size_t i = 0; i < chans.size(); i++) {
            pending.push_back(i);
        }
        const size_t total_chans = chans.size();
        size_t chans_checked = 0;  // channels for which a read was attempted (vs deadline-skipped)

        std::vector<UnhealthyChannel> unhealthy;
        const auto hc_start = std::chrono::steady_clock::now();
        uint32_t hc_spin = 0U;
        int64_t last_hc_log_ms = -1;
        while (!pending.empty()) {
            std::vector<size_t> still_pending;
            std::vector<std::pair<uint32_t, uint32_t>> still_pending_statuses;  // (chan, status)
            for (size_t idx : pending) {
                const auto& ch = chans[idx];
                std::vector<uint32_t> status_buf(1, 0);

                // Per-read deadline guard: if the Phase 5b budget has already been
                // consumed before this read begins, mark the channel unhealthy and skip
                // the read entirely.  This prevents accumulating kHealthCheckTimeoutMs
                // per remaining channel when a prior read in the same round took the full
                // UMD relay timeout (~5 s each), which would otherwise block for
                // (N_remaining_channels × 5 s) before the outer timeout fires.
                //
                // NOTE: this guard does not protect against a read that *starts* within
                // budget but then hangs indefinitely (e.g., relay ERISC alive but
                // forwarding to a peer running fabric firmware).  That scenario requires
                // either a non-blocking UMD probe API or a thread-based read timeout,
                // neither of which is implemented here.  The primary defence against that
                // hang is the phase5_relay_read_threw guard above (Phase 5b is skipped
                // entirely when Phase 5's own read threw).
                const auto pre_read_elapsed =
                    std::chrono::duration_cast<std::chrono::milliseconds>(
                        std::chrono::steady_clock::now() - hc_start)
                        .count();
                if (pre_read_elapsed > kHealthCheckTimeoutMs) {
                    log_warning(
                        tt::LogMetal,
                        "wait_for_fabric_workers_ready: Device {} Phase 5b: deadline "
                        "exceeded ({}ms) before reading chan {} — treating as "
                        "not-READY_FOR_TRAFFIC without attempting read "
                        "({}/{} channels checked so far)",
                        this->id(),
                        pre_read_elapsed,
                        ch.eth_chan_id,
                        chans_checked,
                        total_chans);
                    still_pending.push_back(idx);
                    still_pending_statuses.push_back({ch.eth_chan_id, 0xdeadbeef});
                    continue;
                }
                chans_checked++;

                try {
                    detail::ReadFromDeviceL1(
                        this, ch.eth_logical_core, router_sync_addr, 4, status_buf, CoreType::ETH);
                } catch (const std::exception& e) {
                    // Non-MMIO relay read timed out — treat this channel as not-ready.
                    // Without this catch, the exception propagates uncaught through
                    // quiesce_devices(), causing GTest TearDown to call quiesce again on
                    // already-degraded hardware, which accumulates 5s timeouts and hangs.
                    log_warning(
                        tt::LogMetal,
                        "wait_for_fabric_workers_ready: Device {} Phase 5b: read failed on "
                        "chan {} — treating as not-READY_FOR_TRAFFIC: {}",
                        this->id(),
                        ch.eth_chan_id,
                        e.what());
                    status_buf[0] = 0xdeadbeef;
                }
                if (status_buf[0] != expected_ready) {
                    still_pending.push_back(idx);
                    still_pending_statuses.push_back({ch.eth_chan_id, status_buf[0]});
                }
            }
            pending = std::move(still_pending);
            if (pending.empty()) {
                break;
            }
            const auto hc_elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
                                        std::chrono::steady_clock::now() - hc_start)
                                        .count();
            // Log every kHCIntermediateLogMs to observe actual time-to-healthy
            if (hc_elapsed / kHCIntermediateLogMs >
                last_hc_log_ms / static_cast<int64_t>(kHCIntermediateLogMs)) {
                last_hc_log_ms = hc_elapsed;
                std::string pending_str;
                for (const auto& [cid, st] : still_pending_statuses) {
                    pending_str += fmt::format(" chan{}=0x{:08x}", cid, st);
                }
                log_info(
                    tt::LogMetal,
                    "wait_for_fabric_workers_ready: Device {} Phase 5b: {}ms elapsed, "
                    "{} channel(s) still not READY_FOR_TRAFFIC:{}",
                    this->id(),
                    hc_elapsed,
                    still_pending_statuses.size(),
                    pending_str);
            }
            if (hc_elapsed > kHealthCheckTimeoutMs) {
                // Final diagnostic read — best-effort snapshot of each remaining channel.
                // Apply the same per-read deadline guard: skip the read and record
                // 0xdeadbeef if another channel's read already consumed the budget.
                for (size_t idx : pending) {
                    const auto& ch = chans[idx];
                    std::vector<uint32_t> status_buf(1, 0);
                    const auto diag_elapsed =
                        std::chrono::duration_cast<std::chrono::milliseconds>(
                            std::chrono::steady_clock::now() - hc_start)
                            .count();
                    constexpr int64_t kDiagBudgetMs = kHealthCheckTimeoutMs + 6000;
                    if (diag_elapsed > kDiagBudgetMs) {
                        log_warning(
                            tt::LogMetal,
                            "wait_for_fabric_workers_ready: Device {} Phase 5b: final "
                            "diagnostic deadline ({}ms) exceeded before reading chan {} — "
                            "recording as 0xdeadbeef without read",
                            this->id(),
                            kDiagBudgetMs,
                            ch.eth_chan_id);
                        status_buf[0] = 0xdeadbeef;
                    } else {
                        try {
                            detail::ReadFromDeviceL1(
                                this,
                                ch.eth_logical_core,
                                router_sync_addr,
                                4,
                                status_buf,
                                CoreType::ETH);
                        } catch (const std::exception& e) {
                            log_warning(
                                tt::LogMetal,
                                "wait_for_fabric_workers_ready: Device {} Phase 5b: final "
                                "diagnostic read failed on chan {} — recording as 0xdeadbeef: {}",
                                this->id(),
                                ch.eth_chan_id,
                                e.what());
                            status_buf[0] = 0xdeadbeef;
                        }
                    }
                    unhealthy.push_back({ch.eth_chan_id, status_buf[0]});
                }
                break;
            }
            if (++hc_spin >= kSpinLimit) {
                hc_spin = 0U;
                std::this_thread::sleep_for(std::chrono::microseconds(100));
            }
        }

        if (!unhealthy.empty()) {
            // FIX P2 (#42429): Separate pre-known-dead channels from genuinely new failures.
            // Pre-dead channels were identified during configure_fabric() (probe timed out or
            // corrupt state) and never received firmware — they will never reach READY_FOR_TRAFFIC.
            // Only channels that WERE loaded with firmware but failed to become healthy are errors.
            std::vector<UnhealthyChannel> truly_unhealthy;
            std::vector<UnhealthyChannel> pre_dead_unhealthy;
            for (const auto& u : unhealthy) {
                if (fabric_pre_dead_channels_.count(u.eth_chan_id)) {
                    pre_dead_unhealthy.push_back(u);
                } else {
                    truly_unhealthy.push_back(u);
                }
            }
            if (!pre_dead_unhealthy.empty()) {
                std::string dead_details;
                for (const auto& u : pre_dead_unhealthy) {
                    dead_details += fmt::format(
                        "  dev={} chan={} status=0x{:08x}\n", this->id(), u.eth_chan_id, u.actual_status);
                }
                log_warning(
                    tt::LogMetal,
                    "wait_for_fabric_workers_ready: Device {} Phase 5: {} pre-known-dead ERISC "
                    "channel(s) not at READY_FOR_TRAFFIC (0x{:08x}) — expected, no firmware was "
                    "loaded for these channels (#42429):\n{}",
                    this->id(),
                    pre_dead_unhealthy.size(),
                    expected_ready,
                    dead_details);
            }
            if (!truly_unhealthy.empty()) {
                std::string details;
                for (const auto& u : truly_unhealthy) {
                    details += fmt::format(
                        "  dev={} chan={} status=0x{:08x}\n", this->id(), u.eth_chan_id, u.actual_status);
                }
                // Interpret the observed status before throwing so the reader knows
                // whether this is a timing issue or genuine hardware/firmware failure.
                // STARTED (0xa0b0c0d0) = launched but handshake not complete → timing issue
                // 0x0 = launch message never arrived → firmware loading failure
                // 0x49705180 = L1 corrupt (prior crash left garbage) → needs tt-smi -r
                // Other values = unexpected firmware state
                TT_THROW(
                    "Fabric health check failed after quiesce restart on Device {} — "
                    "{} ERISC channel(s) not at READY_FOR_TRAFFIC (0x{:08x}) after {}ms. "
                    "Check status values: STARTED=0xa0b0c0d0 (handshake incomplete, timing issue); "
                    "0x0 (launch msg lost); 0x49705180 (L1 corrupt — run tt-smi -r); "
                    "other = unexpected firmware state.\n{}",
                    this->id(),
                    truly_unhealthy.size(),
                    expected_ready,
                    kHealthCheckTimeoutMs,
                    details);
            }
        }

        log_info(
            tt::LogMetal,
            "wait_for_fabric_workers_ready: Device {} Phase 5: {} ERISC channels healthy, "
            "{} pre-known dead (no firmware, skipped)",
            this->id(),
            chans.size() - fabric_pre_dead_channels_.size(),
            fabric_pre_dead_channels_.size());
        }  // end else (phase5_relay_read_threw == false)
    }

    log_info(tt::LogMetal, "Fabric workers ready on Device {}", this->id_);
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
    MetalEnvImpl& env_impl = MetalEnvAccessor(*env_).impl();
    using_fast_dispatch_ = env_impl.get_rtoptions().get_fast_dispatch();
    num_hw_cqs_ = num_hw_cqs;
    const auto& hal = env_impl.get_hal();
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

    // Create shared memory stats provider (enabled by default, disable with TT_METAL_SHM_TRACKING_DISABLED=1)
    if (!MetalContext::instance().rtoptions().get_shm_tracking_disabled()) {
        // Use UMD's chip_unique_ids for globally unique chip identification.
        // This ID is computed by topology discovery from hardware-reported board_id and asic_location,
        // and is consistent across all board types (P300, N300, UBB Wormhole, UBB Blackhole, etc.).
        uint64_t asic_id = 0;

        try {
            const auto& cluster = tt::tt_metal::MetalContext::instance().get_cluster();
            auto* cluster_desc = cluster.get_cluster_desc();

            if (cluster_desc) {
                const auto& unique_ids = cluster_desc->get_chip_unique_ids();
                auto it = unique_ids.find(this->id_);
                if (it != unique_ids.end()) {
                    asic_id = it->second;
                    log_debug(
                        tt::LogMetal,
                        "Device {}: using UMD chip_unique_id=0x{:x} for SHM tracking",
                        this->id_,
                        asic_id);
                } else {
                    asic_id = this->id_;
                    log_warning(
                        tt::LogMetal, "Device {} not found in chip_unique_ids, using device_id as asic_id", this->id_);
                }
            } else {
                asic_id = this->id_;
                log_warning(
                    tt::LogMetal,
                    "ClusterDescriptor not available for device {}, using device_id as asic_id",
                    this->id_);
            }
        } catch (const std::exception& e) {
            log_warning(
                tt::LogMetal,
                "Error getting asic_id for device {}: {}. Using device_id as fallback.",
                this->id_,
                e.what());
            asic_id = this->id_;
        }

        shm_stats_provider_ = std::make_unique<SharedMemoryStatsProvider>(asic_id, this->id_);
        log_debug(tt::LogMetal, "Shared memory tracking enabled for device {}, asic_id=0x{:x}", this->id_, asic_id);

        // Register ShmTrackingProcessor globally once (when first device with SHM is created)
        static bool shm_processor_registered = false;
        if (!shm_processor_registered) {
            tt::tt_metal::GraphTracker::instance().push_processor(
                std::make_shared<tt::tt_metal::ShmTrackingProcessor>());
            log_debug(tt::LogMetal, "ShmTrackingProcessor registered with GraphTracker");
            shm_processor_registered = true;
        }
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

    // Clean up shared memory stats provider
    this->shm_stats_provider_.reset();

    this->initialized_ = false;

    return true;
}

Device::~Device() {
    log_debug(tt::LogMetal, "Device {} destructor", this->id_);
    if (this->initialized_) {
        this->close();
    }
}

tt::ARCH Device::arch() const { return MetalEnvAccessor(*env_).impl().get_cluster().arch(); }

int Device::num_dram_channels() const {
    return MetalEnvAccessor(*env_).impl().get_cluster().get_soc_desc(id_).get_num_dram_views();
}

uint32_t Device::l1_size_per_core() const {
    return MetalEnvAccessor(*env_).impl().get_cluster().get_soc_desc(id_).worker_l1_size;
}
uint32_t Device::dram_size_per_channel() const {
    return MetalEnvAccessor(*env_).impl().get_cluster().get_soc_desc(id_).dram_view_size;
}

int Device::get_clock_rate_mhz() const { return MetalEnvAccessor(*env_).impl().get_cluster().get_device_aiclk(id_); }

CoreCoord Device::grid_size() const { return MetalEnvAccessor(*env_).impl().get_cluster().get_soc_desc(id_).grid_size; }

CoreCoord Device::logical_grid_size() const {
    return MetalEnvAccessor(*env_).impl().get_cluster().get_soc_desc(id_).get_grid_size(CoreType::TENSIX);
}

CoreCoord Device::dram_grid_size() const {
    return MetalEnvAccessor(*env_).impl().get_cluster().get_soc_desc(id_).get_dram_grid_size();
}

CoreCoord Device::compute_with_storage_grid_size() const {
    const auto& dispatch_core_config = context_->get_dispatch_core_manager().get_dispatch_core_config();
    return tt::get_compute_grid_size(MetalEnvAccessor(*env_).impl(), id_, num_hw_cqs_, dispatch_core_config);
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
    const Hal& hal = MetalEnvAccessor(*env_).impl().get_hal();
    CoreCoord virtual_coord = {
        hal.noc_coordinate(noc_index, grid_size.x, coord.x), hal.noc_coordinate(noc_index, grid_size.y, coord.y)};
    return virtual_coord;
}

CoreCoord Device::physical_worker_core_from_logical_core(const CoreCoord& logical_core) const {
    const metal_SocDescriptor& soc_desc = MetalEnvAccessor(*env_).impl().get_cluster().get_soc_desc(this->id_);
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
    return MetalEnvAccessor(*env_).impl().get_cluster().get_virtual_coordinate_from_logical_coordinates(
        this->id_, logical_coord, core_type);
}

CoreCoord Device::virtual_core_from_physical_core(const CoreCoord& physical_coord) const {
    return MetalEnvAccessor(*env_).impl().get_cluster().get_virtual_coordinate_from_physical_coordinates(
        this->id_, physical_coord);
}

CoreCoord Device::worker_core_from_logical_core(const CoreCoord& logical_core) const {
    return this->virtual_core_from_logical_core(logical_core, CoreType::WORKER);
}

CoreCoord Device::ethernet_core_from_logical_core(const CoreCoord& logical_core) const {
    return this->virtual_core_from_logical_core(logical_core, CoreType::ETH);
}

CoreCoord Device::logical_core_from_ethernet_core(const CoreCoord& ethernet_core) const {
    return MetalEnvAccessor(*env_).impl().get_cluster().get_logical_ethernet_core_from_virtual(
        this->id(), ethernet_core);
}

uint32_t Device::get_noc_unicast_encoding(uint8_t noc_index, const CoreCoord& core) const {
    auto virtual_noc_coord = this->virtual_noc0_coordinate(noc_index, core);
    return MetalEnvAccessor(*env_).impl().get_hal().noc_xy_encoding(virtual_noc_coord.x, virtual_noc_coord.y);
}

uint32_t Device::get_noc_multicast_encoding(uint8_t noc_index, const CoreRange& cores) const {
    auto virtual_noc_start = this->virtual_noc0_coordinate(noc_index, cores.start_coord);
    auto virtual_noc_end = this->virtual_noc0_coordinate(noc_index, cores.end_coord);
    const auto& hal = MetalEnvAccessor(*env_).impl().get_hal();
    // NOC 1 mcasts from bottom left to top right, so we need to reverse the coords
    if (noc_index == 0) {
        return hal.noc_multicast_encoding(
            virtual_noc_start.x, virtual_noc_start.y, virtual_noc_end.x, virtual_noc_end.y);
    }
    return hal.noc_multicast_encoding(virtual_noc_end.x, virtual_noc_end.y, virtual_noc_start.x, virtual_noc_start.y);
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
    return MetalEnvAccessor(*env_).impl().get_cluster().get_soc_desc(id_).get_preferred_worker_core_for_dram_view(
        dram_channel, noc);
}

CoreCoord Device::logical_core_from_dram_channel(uint32_t dram_channel) const {
    return MetalEnvAccessor(*env_).impl().get_cluster().get_soc_desc(id_).get_logical_core_for_dram_view(dram_channel);
}

uint32_t Device::dram_channel_from_logical_core(const CoreCoord& logical_core) const {
    return MetalEnvAccessor(*env_).impl().get_cluster().get_soc_desc(id_).get_dram_channel_from_logical_core(
        logical_core);
}

uint32_t Device::dram_channel_from_virtual_core(const CoreCoord& virtual_core) const {
    const metal_SocDescriptor& soc_desc = MetalEnvAccessor(*env_).impl().get_cluster().get_soc_desc(this->id_);
    uint32_t num_nocs = MetalEnvAccessor(*env_).impl().get_hal().get_num_nocs();
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
    return default_allocator_->get_lowest_occupied_l1_address(0);
}

std::optional<DeviceAddr> Device::lowest_occupied_compute_l1_address(
    tt::stl::Span<const SubDeviceId> /*sub_device_ids*/) const {
    return default_allocator_->get_lowest_occupied_l1_address(0);
}

HWCommandQueue& Device::command_queue(std::optional<uint8_t> cq_id) {
    detail::DispatchStateCheck(using_fast_dispatch_);
    TT_FATAL(using_fast_dispatch_, "Fast dispatch must be enabled to use command_queue");
    auto actual_cq_id = cq_id.value_or(GetCurrentCommandQueueIdForThread());
    TT_FATAL(actual_cq_id < command_queues_.size(), "cq_id {} is out of range", actual_cq_id);
    TT_FATAL(this->is_initialized(), "Device has not been initialized, did you forget to call InitializeDevice?");
    return *command_queues_[actual_cq_id];
}

SystemMemoryManager& Device::sysmem_manager() {
    // SystemMemoryManager handles mock devices internally with stubs
    // For mock devices, ensure lazy initialization if not already done
    if (!sysmem_manager_) {
        sysmem_manager_ = std::make_unique<SystemMemoryManager>(context_->get_context_id(), this->id_, 1);
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

        const auto& hal = MetalEnvAccessor(*env_).impl().get_hal();
        bool noc_translation_enabled = true;
        if (!MetalEnvAccessor(*env_).impl().get_cluster().is_mock_or_emulated()) {
            noc_translation_enabled =
                MetalEnvAccessor(*env_).impl().get_cluster().get_cluster_desc()->get_noc_translation_table_en().at(
                    this->id());
        }
        bool dram_is_virtualized =
            noc_translation_enabled && (hal.get_virtualized_core_types().contains(dev_msgs::AddressableCoreType::DRAM));
        const metal_SocDescriptor& soc_d = MetalEnvAccessor(*env_).impl().get_cluster().get_soc_desc(this->id());
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

        const metal_SocDescriptor& soc_desc = MetalEnvAccessor(*env_).impl().get_cluster().get_soc_desc(this->id_);
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
    if (!MetalEnvAccessor(*env_).impl().get_cluster().is_ethernet_core(virtual_core, this->id_)) {
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
    if (!MetalEnvAccessor(*env_).impl().get_cluster().is_ethernet_core(virtual_core, this->id_) &&
        !MetalEnvAccessor(*env_).impl().get_cluster().is_worker_core(virtual_core, this->id_)) {
        return HalMemType::DRAM;
    }
    return HalMemType::L1;
}

std::shared_ptr<distributed::MeshDevice> Device::get_mesh_device() { return mesh_device.lock(); }

// Program tracking for accurate CB memory reporting
void Device::register_program(detail::ProgramImpl* program) {
    std::lock_guard<std::mutex> lock(active_programs_mutex_);
    active_programs_.insert(program);
}

void Device::unregister_program(detail::ProgramImpl* program) {
    std::lock_guard<std::mutex> lock(active_programs_mutex_);
    active_programs_.erase(program);
}

uint64_t Device::get_total_cb_allocated() const {
    std::lock_guard<std::mutex> lock(active_programs_mutex_);

    // For PHYSICAL CB tracking accounting for address reuse:
    // Collect L1 regions per core and merge overlapping addresses
    // This handles cached/traced programs that share the same physical L1 addresses on the same core

    std::map<CoreCoord, std::vector<std::pair<uint64_t, uint64_t>>> device_regions_per_core;

    for (const auto* program : active_programs_) {
        size_t num_devices = program->get_num_cb_devices();

        // Get L1 regions per core for this program on this device
        auto program_regions = program->get_cb_l1_regions_per_core(this->id(), num_devices);

        // Merge into device-wide map
        for (const auto& [core, regions] : program_regions) {
            auto& core_regions = device_regions_per_core[core];
            core_regions.insert(core_regions.end(), regions.begin(), regions.end());
        }
    }

    // Merge overlapping regions per core to get actual physical usage
    uint64_t total_physical = 0;

    for (auto& [core, regions] : device_regions_per_core) {
        if (regions.empty()) {
            continue;
        }

        // Sort by start address
        std::sort(regions.begin(), regions.end());

        // Merge overlapping ranges
        std::vector<std::pair<uint64_t, uint64_t>> merged;
        merged.push_back(regions[0]);

        for (size_t i = 1; i < regions.size(); i++) {
            auto& last = merged.back();
            const auto& current = regions[i];

            if (current.first <= last.second) {
                // Overlapping - merge
                last.second = std::max(last.second, current.second);
            } else {
                // Non-overlapping - add new region
                merged.push_back(current);
            }
        }

        // Sum merged regions for this core
        for (const auto& [start, end] : merged) {
            total_physical += (end - start);
        }
    }

    return total_physical;
}

}  // namespace tt::tt_metal
