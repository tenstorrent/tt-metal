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
#include "device/edm_status_utils.hpp"
#include <umd/device/coordinates/coordinate_manager.hpp>
#include <umd/device/types/core_coordinates.hpp>
#include <umd/device/types/xy_pair.hpp>
#include <impl/debug/watcher_server.hpp>
#include <impl/dispatch/dispatch_mem_map.hpp>
#include "hostdev/fabric_boot_fence.h"  // FIX S8/S9 (#42429): boot fence + session ID constants

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

    // Reset host-side CQ manager state for this device.
    // Non-MMIO devices: host-side hugepage pointers are owned by the MMIO device
    // (reset below), but sysmem_manager_ tracks in-flight counts, quiesce flags,
    // and dev_fences locally. Reset unconditionally so re-init after firmware
    // reload cannot inherit stale prefetch_q_dev_fences or cq_to_quiesced.
    // Skip if relay is broken — reset() reads HW fence via relay which would hang.
    if (!this->fabric_relay_path_broken_.load()) {
        for (uint8_t cq_id = 0; cq_id < num_hw_cqs; cq_id++) {
            this->sysmem_manager_->reset(cq_id);
        }
    }

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
                /* send_go= */ true);  // FIX SENDGO: was hal.get_dev_addr (uint64_t→bool implicit conversion)
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

// Maps a raw EDMStatus uint32 to its enum name for log readability.
// EthDiagSentinel values (deadline-skipped, read-exception, etc.) are also named.
// Defined once here and shared by configure_fabric(), quiesce_and_restart_fabric_workers(),
// phase5b_erisc_health_check(), and wait_for_fabric_workers_ready() to avoid duplication.
// edm_status_str(), edm_status_name(), and is_known_edm_status() are defined in
// device/edm_status_utils.hpp (shared with fabric_firmware_initializer.cpp).

bool Device::compile_fabric() {
    fabric_program_ = tt::tt_fabric::create_and_compile_fabric_program(this);
    return fabric_program_ != nullptr;
}

void Device::configure_fabric(
    const std::unordered_set<uint32_t>& pre_dead_channels,
    const std::unordered_set<uint32_t>& skip_soft_reset_channels,
    const std::unordered_set<uint32_t>& external_umd_channels,
    const std::unordered_set<uint32_t>& mmio_base_umd_channels) {
    // FIX IJ/KL code that used mmio_base_umd_channels is disabled under
    // #ifdef FIXIJ_REDUNDANT_AFTER_FIX_MM (made redundant by FIX MM).  The
    // parameter is kept in the signature so callers need no churn when/if the
    // ifdef block is removed.  Suppress the -Wunused-parameter warning.
    (void)mmio_base_umd_channels;
    if (fabric_program_ == nullptr) {
        return;
    }

    // GAP-R5 (#42429): Total elapsed timer for configure_fabric — post-mortem can't tell how
    // long fabric init took without this.  Logged at function exit.
    const auto configure_fabric_start = std::chrono::steady_clock::now();

    // FIX S9 (#42429): Bump monotonic session ID.  Firmware uses this to reject stale data
    // from a previous session.  SESSION_ID_INVALID(0) is never used — the counter starts at 1
    // and only increments forward.  Wrap at UINT32_MAX is astronomically unlikely.
    fabric_session_id_++;

    // Reset relay-broken flag: configure_fabric() initialises fresh fabric firmware on all
    // channels, including the MMIO device's relay ERISCs.  After this call the UMD relay path
    // is valid again, so fabric_relay_path_broken_ must be cleared so that the next quiesce
    // cycle does not falsely short-circuit Phase 2.5, Phase 3, and Phase 5 for non-MMIO
    // devices.  (#42429 — flag is set in quiesce ENTRY snapshot / Phase 5 catch block).
    if (fabric_relay_path_broken_.load()) {
        log_info(
            tt::LogMetal,
            "configure_fabric: Device {} relay-broken flag reset by configure_fabric — "
            "UMD relay path restored (fresh firmware loaded on MMIO relay ERISCs)",
            this->id_);
    }
    fabric_relay_path_broken_ = false;
    // FIX AM (#42429): Clear channels-not-ready flag — configure_fabric() relaunches fresh
    // fabric firmware on all channels, so after this call all channels will go through the
    // full handshake again in the next quiesce cycle.
    fabric_channels_not_ready_for_traffic_ = false;
    // FIX TK (#42429): Clear ring-sync-timed-out flag — a fresh configure_fabric() means we are
    // starting a new init cycle and any previous ring sync result no longer applies.
    fabric_ring_sync_timed_out_ = false;
    // FIX RZ (#42429): Clear stale-base-UMD flag unconditionally at configure_fabric() start;
    // re-set below if skip_soft_reset_channels is non-empty on a non-MMIO device.
    fabric_stale_base_umd_channels_ = false;
    // FIX RZ3 (#42429): Clear the persistent companion flag at session start.
    // This is the ONLY place it is cleared — FIX RZ2 does not clear it.
    fabric_base_umd_fixm_init_ = false;
    // Clear accumulated soft-reset failures from the prior quiesce cycle.  A channel that was
    // force-reset and recovered should not be permanently excluded from Phase 5 health checks
    // in subsequent cycles — the set is repopulated by this configure_fabric() call below.
    fabric_pre_dead_channels_.clear();
    // FIX EXT (#42429): Clear and repopulate external_umd_channels each configure_fabric() call.
    fabric_external_umd_channels_.clear();

    // FIX P2 (#42429): Persist pre_dead_channels so quiesce Phase 5 can skip them.
    // Updated below after configure_fabric_cores() — FIX RR may recover some channels.
    // FIX EXT (#42429): Persist external_umd_channels so phase5b_erisc_health_check treats them
    // as expected-non-responding (pre_dead_unhealthy) rather than truly_unhealthy.
    fabric_external_umd_channels_ = external_umd_channels;

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
    // FIX EXT (#42429): external_umd_channels are also passed as skip_soft_reset_channels so
    // configure_fabric_cores() applies FIX TG2 partial L1 clear (preserves 0x49706550 sentinel)
    // and skips soft-reset for them — merged union used for configure_fabric_cores call only.
    std::unordered_set<uint32_t> skip_soft_reset_with_ext = skip_soft_reset_channels;
    skip_soft_reset_with_ext.insert(external_umd_channels.begin(), external_umd_channels.end());
    const auto health = tt::tt_fabric::configure_fabric_cores(this, pre_dead_channels, skip_soft_reset_with_ext);

    // FIX RR (#42429): Compute the effective dead set: pre_dead_channels that were NOT
    // recovered by FIX RR or deferred to FIX SA (Strategy B).
    // Recovered channels got their L1 cleared and are ready for firmware load.
    // FIX SA-B deferred channels are halted and will get firmware via FIX SA — not dead.
    std::unordered_set<uint32_t> effective_pre_dead;
    for (const auto& ch : pre_dead_channels) {
        if (!health.recovered_channels.count(ch) && !health.deferred_deassert_channels.count(ch)) {
            effective_pre_dead.insert(ch);
        }
    }

    // FIX P2 (#42429): Persist effective (post-RR) pre_dead set so Phase 5 skips channels
    // that truly have no firmware loaded.  Recovered channels are NOT in this set.
    fabric_pre_dead_channels_ = effective_pre_dead;

    // FIX RS (#42429): Log recovered channels so analyze_fabric_hang_log.sh can track them.
    // FIX RR recovery is silent at configure_fabric() level without this entry.
    if (!health.recovered_channels.empty()) {
        log_info(
            tt::LogMetal,
            "configure_fabric: Device {} FIX RS — {} channel(s) recovered by FIX RR "
            "propagated back: effective_pre_dead={} (was {}). Firmware WILL load on recovered channels.",
            this->id_,
            health.recovered_channels.size(),
            effective_pre_dead.size(),
            pre_dead_channels.size());
    }

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
            "channel(s) skipped. Fabric firmware will not be loaded on those channels. "
            "({} were recovered by FIX RR and will get firmware.)",
            this->id_,
            effective_pre_dead.size(),
            health.recovered_channels.size());
    }

    // FIX C (#42429): Build the set of all dead ETH channels (effective dead + any newly found)
    // so we can skip write_launch_msg_to_core for dead ETH cores.  Attempting to write launch
    // messages to an ERISC core whose ETH relay is broken causes the NOC write to route through
    // the dead channel and hang (same failure mode as the l1_barrier above).
    // FIX EXT (#42429): also treat external_umd_channels as "no firmware loaded" so
    // write_launch_msg_to_core, WriteRuntimeArgsToDevice, and ConfigureDeviceWithProgram
    // all skip them.  External channels preserve live relay BRISC (0x49706550) but must NOT
    // have FABRIC_1D loaded on them — their out-of-mesh peer can never complete the handshake.
    std::unordered_set<uint32_t> all_dead_channels_storage;
    const std::unordered_set<uint32_t>* all_dead_channels_ptr;
    if (!external_umd_channels.empty()) {
        // Merge base dead + external into a new set so write_launch_msg_to_core skips both.
        const auto& base_dead = effective_pre_dead.empty() ? health.newly_dead_channels : effective_pre_dead;
        all_dead_channels_storage = base_dead;
        all_dead_channels_storage.insert(external_umd_channels.begin(), external_umd_channels.end());
        all_dead_channels_ptr = &all_dead_channels_storage;
    } else {
        all_dead_channels_storage = effective_pre_dead.empty() ? health.newly_dead_channels : effective_pre_dead;
        all_dead_channels_ptr = &all_dead_channels_storage;
    }
    // FIX SA-B (#42429): Remove deferred_deassert channels from dead sets.
    // Pre-dead MMIO channels merged into FIX SA (Strategy B) are no longer truly dead —
    // they just need firmware loaded while halted, then deassert via FIX SA.
    // Ensure ConfigureDeviceWithProgram writes firmware to these channels.
    if (!health.deferred_deassert_channels.empty()) {
        for (const auto& deferred_ch : health.deferred_deassert_channels) {
            all_dead_channels_storage.erase(deferred_ch);
        }
    }
    const auto& all_dead_channels = *all_dead_channels_ptr;
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
                        tt::umd::CoreCoord(lc.x, lc.y, CoreType::ETH, CoordSystem::LOGICAL), CoordSystem::LOGICAL);
                    if (all_dead_channels.count(eth_chan)) {
                        dead_eth_logical_cores.insert(lc);
                    }
                } catch (...) {
                    // Cannot resolve channel — conservatively skip this core.
                    log_warning(
                        tt::LogMetal,
                        "compile_fabric: Device {} cannot resolve ETH channel for logical core ({},{}) "
                        "— unknown exception; core added to dead_eth_logical_cores to avoid relay hang",
                        this->id_,
                        lc.x,
                        lc.y);
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

    // =========================================================================
    // FIX SA (#42429): Strategy A — Deferred Deassert.
    // For MMIO channels where configure_fabric_cores() asserted ERISC reset (FIX S9)
    // but deferred the deassert:  ERISC is still halted.  All L1 writes above
    // (L1 clear, WriteRuntimeArgsToDevice, ConfigureDeviceWithProgram) completed
    // while ERISC was in reset — zero race with base-UMD execution.
    //
    // Now perform the post-L1-write sequence for each deferred channel:
    //   1. Write fw_launch_addr=1 while halted (FIX MM equivalent)
    //   2. Write launch_msg while halted (send_go=false — no go_msg yet)
    //   3. Write handshake_bypass=1 while halted (STRATEGY7 / FIX S7-MOVE)
    //   4. Deassert ERISC reset
    //   5. FIX S8: Write BOOT_FENCE_READY to L1 (replaces old FIX DW+DU+PQ)
    //   6. (removed — was FIX DU, replaced by S8 boot fence)
    //   7. Write go_msg (RUN_MSG_GO) — ERISC is now in base-UMD polling loop
    // =========================================================================
    if (!health.deferred_deassert_channels.empty()) {
        auto& cluster_sa = env_impl.get_cluster();
        const auto& hal_sa = env_impl.get_hal();
        const auto& soc_desc_sa = cluster_sa.get_soc_desc(this->id_);
        const auto& control_plane_sa = MetalContext::instance().get_control_plane();
        const auto& fabric_context_sa = control_plane_sa.get_fabric_context();
        const auto& builder_ctx_sa = fabric_context_sa.get_builder_context();
        const auto& router_config_sa = builder_ctx_sa.get_fabric_router_config();

        // HAL info for fw_launch_addr (FIX MM equivalent).
        const auto aeth_idx_sa = hal_sa.get_programmable_core_type_index(
            HalProgrammableCoreType::ACTIVE_ETH);
        const uint32_t fw_launch_addr_sa = hal_sa.get_jit_build_config(aeth_idx_sa, 0, 0).fw_launch_addr;
        const uint32_t fw_launch_val_sa = hal_sa.get_jit_build_config(aeth_idx_sa, 0, 0).fw_launch_addr_value;

        // STRATEGY7 handshake_bypass address.
        const uint32_t handshake_bypass_offset_sa = 32;  // offsetof(handshake_info_t, handshake_bypass)
        const uint32_t handshake_bypass_l1_addr_sa =
            static_cast<uint32_t>(router_config_sa.handshake_addr) + handshake_bypass_offset_sa;

        // FIX S8/S9 (#42429): Compute AERISC_FABRIC_SCRATCH base from HAL.
        // scratch_base = FABRIC_TELEMETRY_BASE + FABRIC_TELEMETRY_SIZE - POSTCODES_SIZE(4)
        // This formula matches dev_mem_map.h for both WH and BH.
        const uint32_t scratch_base_sa = static_cast<uint32_t>(
            hal_sa.get_dev_addr(HalProgrammableCoreType::ACTIVE_ETH, HalL1MemAddrType::FABRIC_TELEMETRY) +
            hal_sa.get_dev_size(HalProgrammableCoreType::ACTIVE_ETH, HalL1MemAddrType::FABRIC_TELEMETRY) -
            4u /* MEM_AERISC_FABRIC_POSTCODES_SIZE */);
        const uint32_t boot_fence_addr_sa = scratch_base_sa + BOOT_FENCE_OFFSET;
        const uint32_t session_id_addr_sa = scratch_base_sa + SESSION_ID_OFFSET;

        // Build logical_core→channel mapping for the write_launch_msg_to_core calls.
        // We need to find the program's kernel group for each deferred channel's logical core.
        const auto logical_cores_sa = fabric_program_->impl().logical_cores();

        log_info(
            tt::LogMetal,
            "FIX SA (#42429): Strategy A — Device {} processing {} deferred deassert channel(s) "
            "after ConfigureDeviceWithProgram",
            this->id_, health.deferred_deassert_channels.size());

        // FIX SA-S (#42429): Stagger deasserts across MMIO devices by 100ms per chip_id.
        // When multiple MMIO devices (e.g., 4 N300s in T3K) all deassert simultaneously,
        // their ETH links try to train at the same time → cross-chip link training loops →
        // ALL channels timeout in FIX BH (now SA-A).  Staggering by chip_id breaks the
        // simultaneous link training deadlock: Device 0 deasserts first, Device 1 100ms later, etc.
        // Only MMIO devices have deferred deassert channels so this only affects MMIO paths.
        {
            constexpr uint32_t kStaggerPerChipMs = 100;
            const uint32_t stagger_ms = this->id_ * kStaggerPerChipMs;
            if (stagger_ms > 0) {
                log_info(
                    tt::LogMetal,
                    "FIX SA-S (#42429): Device {} staggering deassert by {}ms (chip_id * {}ms) "
                    "to prevent simultaneous cross-chip ETH link training.",
                    this->id_, stagger_ms, kStaggerPerChipMs);
                auto stagger_start = std::chrono::steady_clock::now();
                std::this_thread::sleep_for(std::chrono::milliseconds(stagger_ms));
                auto stagger_actual = std::chrono::duration_cast<std::chrono::milliseconds>(
                    std::chrono::steady_clock::now() - stagger_start).count();
                log_info(
                    tt::LogMetal,
                    "FIX SA-S (#42429): Device {} stagger complete — requested {}ms, actual {}ms.",
                    this->id_, stagger_ms, stagger_actual);
            }
        }

        for (const auto& deferred_chan : health.deferred_deassert_channels) {
            if (all_dead_channels.count(deferred_chan)) {
                continue;  // Became dead during configure_fabric_cores — skip.
            }

            // GAP 12 / FIX OP SA (#42429): Time the entire Strategy A per-channel sequence
            // (steps 1-7).  Mirrors FIX OP on the normal FIX S9 path and FIX OP RR on the
            // recovery path.  Without this, CI logs show no timing data for Strategy A —
            // impossible to tell if one channel took 50ms vs 5000ms.
            auto fix_op_sa_start = std::chrono::steady_clock::now();

            auto virtual_core_sa = cluster_sa.get_virtual_eth_core_from_channel(this->id_, deferred_chan);
            tt_cxy_pair core_loc_sa(this->id_, virtual_core_sa);
            auto logical_core_sa = soc_desc_sa.get_eth_core_for_channel(deferred_chan, CoordSystem::LOGICAL);

            // Step 1: Write fw_launch_addr=1 while halted (FIX MM equivalent).
            // PCIe writes to halted ERISC L1 work — proven by FIX EG.
            cluster_sa.write_core_immediate(
                this->id_, virtual_core_sa, std::vector<uint32_t>{fw_launch_val_sa}, fw_launch_addr_sa);
            // Readback verify (FIX GI pattern).
            std::vector<uint32_t> mm_verify_sa(1, 0);
            cluster_sa.read_core(mm_verify_sa, sizeof(uint32_t), core_loc_sa,
                static_cast<uint64_t>(fw_launch_addr_sa));
            if (mm_verify_sa[0] != fw_launch_val_sa) {
                log_warning(
                    tt::LogMetal,
                    "FIX SA (#42429): FIX MM fw_launch_addr readback MISMATCH — wrote 0x{:08X} "
                    "to 0x{:08X} on Device {} chan={} but read back 0x{:08X}",
                    fw_launch_val_sa, fw_launch_addr_sa, this->id_, deferred_chan, mm_verify_sa[0]);
            } else {
                log_info(
                    tt::LogMetal,
                    "FIX SA (#42429): FIX MM restored fw_launch_addr=0x{:08X} on Device {} chan={} "
                    "while halted (readback verified)",
                    fw_launch_addr_sa, this->id_, deferred_chan);
            }

            // Step 2: Write launch_msg while halted (send_go=false).
            // Find the kernel group for this logical core to get launch_msg/go_msg.
            // We store the KernelGroup* so step 7 can get go_msg.view() without
            // default-constructing ConstView (BaseStructView has no default ctor).
            bool launch_msg_written = false;
            KernelGroup* kg_sa_ptr = nullptr;
            for (uint32_t pct_sa = 0; pct_sa < logical_cores_sa.size(); pct_sa++) {
                if (hal_sa.get_core_type(pct_sa) != CoreType::ETH) continue;
                for (const auto& lc : logical_cores_sa[pct_sa]) {
                    if (lc.x == logical_core_sa.x && lc.y == logical_core_sa.y) {
                        kg_sa_ptr = fabric_program_->impl().kernels_on_core(lc, pct_sa);
                        dev_msgs::launch_msg_t::View msg_sa = kg_sa_ptr->launch_msg.view();
                        auto go_msg_sa = kg_sa_ptr->go_msg.view();
                        msg_sa.kernel_config().host_assigned_id() = fabric_program_->get_runtime_id();

                        auto physical_core_sa = this->virtual_core_from_logical_core(lc, CoreType::ETH);
                        tt::llrt::write_launch_msg_to_core(
                            this->id(),
                            physical_core_sa,
                            msg_sa,
                            go_msg_sa,
                            /* send_go= */ false);  // FIX SA: launch_msg only, go_msg after deassert

                        log_info(
                            tt::LogMetal,
                            "FIX SA (#42429): wrote launch_msg (send_go=false) for Device {} chan={} "
                            "logical ({},{}) while ERISC halted",
                            this->id_, deferred_chan, lc.x, lc.y);
                        launch_msg_written = true;
                        break;
                    }
                }
                if (launch_msg_written) break;
            }
            if (!launch_msg_written) {
                log_warning(
                    tt::LogMetal,
                    "FIX SA (#42429): Device {} chan={} — could not find kernel group for "
                    "logical core ({},{}).  Skipping deferred deassert for this channel.",
                    this->id_, deferred_chan, logical_core_sa.x, logical_core_sa.y);
                continue;
            }

            // Step 3: Write handshake_bypass=1 while halted (STRATEGY7 / FIX S7-MOVE).
            // This is AFTER ConfigureDeviceWithProgram, so BSS zeroing cannot overwrite it.
            {
                std::vector<uint32_t> bypass_buf_sa = {1};
                detail::WriteToDeviceL1(
                    this, CoreCoord(logical_core_sa.x, logical_core_sa.y),
                    handshake_bypass_l1_addr_sa, bypass_buf_sa, CoreType::ETH);
                // FIX NO readback verify.
                std::vector<uint32_t> bypass_verify_sa(1, 0xFFFFFFFF);
                detail::ReadFromDeviceL1(
                    this, CoreCoord(logical_core_sa.x, logical_core_sa.y),
                    handshake_bypass_l1_addr_sa, sizeof(uint32_t), bypass_verify_sa, CoreType::ETH);
                if (bypass_verify_sa[0] != 1) {
                    log_warning(
                        tt::LogMetal,
                        "FIX SA (#42429): STRATEGY7 handshake_bypass readback MISMATCH — wrote 1 "
                        "to L1[0x{:08X}] on Device {} chan={} but read back 0x{:08X} [FIX S7-MOVE]",
                        handshake_bypass_l1_addr_sa, this->id_, deferred_chan, bypass_verify_sa[0]);
                } else {
                    log_info(
                        tt::LogMetal,
                        "FIX SA (#42429): STRATEGY7 handshake_bypass=1 at L1[0x{:08X}] for "
                        "Device {} chan={} while halted — readback verified [FIX S7-MOVE]",
                        handshake_bypass_l1_addr_sa, this->id_, deferred_chan);
                }
            }

            // Step 4: Deassert ERISC reset.
            cluster_sa.deassert_risc_reset_at_core(core_loc_sa, tt::umd::RiscType::ERISC0);
            log_info(
                tt::LogMetal,
                "FIX SA (#42429): deasserted ERISC0 on Device {} chan={} — "
                "all L1 writes complete (fw_launch_addr + launch_msg + handshake_bypass)",
                this->id_, deferred_chan);

            // Step 4b: FIX SA-A (#42429) Firmware-Side Ready Gate — wait for ERISC to
            // signal that init is complete before writing boot fence token.
            // ERISC writes FW_READY_VALUE to FW_READY_OFFSET after flag_disable +
            // go_messages setup in active_erisc.cc.  Host polls until it appears or
            // FW_READY_TIMEOUT_MS expires.  If timeout, channel is marked dead (ERISC
            // stuck in ROM or crashed) — no boot fence or go_msg is written.
            // Constants defined in fabric_boot_fence.h (included by both host and firmware).
            {
                const uint32_t fw_ready_addr_sa = scratch_base_sa + FW_READY_OFFSET;
                constexpr uint32_t kFwReadyPollMs = 5;
                uint32_t fw_ready_elapsed = 0;
                bool fw_ready_ok = false;
                while (fw_ready_elapsed < FW_READY_TIMEOUT_MS) {
                    std::vector<uint32_t> rb(1, 0);
                    cluster_sa.read_core(rb, sizeof(uint32_t), core_loc_sa,
                        static_cast<uint64_t>(fw_ready_addr_sa));
                    if (rb[0] == FW_READY_VALUE) {
                        fw_ready_ok = true;
                        break;
                    }
                    std::this_thread::sleep_for(std::chrono::milliseconds(kFwReadyPollMs));
                    fw_ready_elapsed += kFwReadyPollMs;
                }
                if (!fw_ready_ok) {
                    // FIX QQ (#42429): Read the actual fw_ready value at timeout for diagnostics.
                    // Without this, post-mortem can't tell if ERISC wrote a partial/wrong value
                    // vs. the slot being still zeroed (ERISC never reached init).
                    uint32_t fw_ready_actual = 0;
                    try {
                        std::vector<uint32_t> rb_timeout(1, 0);
                        cluster_sa.read_core(rb_timeout, sizeof(uint32_t), core_loc_sa,
                            static_cast<uint64_t>(fw_ready_addr_sa));
                        fw_ready_actual = rb_timeout[0];
                    } catch (...) {
                        fw_ready_actual = 0xDEADDEAD;  // PCIe read failed
                    }
                    log_warning(
                        tt::LogMetal,
                        "FIX SA-A (#42429): Device {} chan={} ERISC did not signal FW_READY within {}ms. "
                        "actual fw_ready=0x{:08X} (expected 0x{:08X}). "
                        "Channel may be stuck in ROM or link training. Marking dead — skipping boot fence + go_msg. "
                        "FIX QQ (#42429): timeout snapshot.",
                        this->id_, deferred_chan, FW_READY_TIMEOUT_MS,
                        fw_ready_actual, static_cast<uint32_t>(FW_READY_VALUE));
                    // Mark as dead so downstream code knows this channel failed.
                    all_dead_channels_storage.insert(deferred_chan);
                    continue;  // skip boot fence + go_msg for this channel
                }
                log_info(
                    tt::LogMetal,
                    "FIX SA-A (#42429): Device {} chan={} ERISC signaled FW_READY after {}ms — "
                    "init confirmed, proceeding to boot fence.",
                    this->id_, deferred_chan, fw_ready_elapsed);
            }

            // Step 5: FIX S8 (#42429) Boot Fence — write BOOT_FENCE_READY to L1.
            // Replaces FIX DW (50ms sleep) + FIX DU (ROM postcode poll) + FIX PQ (two-phase guard).
            // Firmware (active_erisc.cc) polls boot_fence after ROM boot completes and before
            // entering the go_messages dispatch loop.  By writing the token here (after deassert,
            // before go_msg), we provide an explicit host→firmware synchronization barrier.
            // The old FIX DW+DU+PQ sequence was timing-based and fragile; S8 is deterministic.
            {
                cluster_sa.write_core_immediate(
                    this->id_, virtual_core_sa,
                    std::vector<uint32_t>{BOOT_FENCE_READY_VALUE},
                    boot_fence_addr_sa);
                // Readback verify (follows FIX GI pattern).
                std::vector<uint32_t> bf_verify_sa(1, 0);
                cluster_sa.read_core(bf_verify_sa, sizeof(uint32_t), core_loc_sa,
                    static_cast<uint64_t>(boot_fence_addr_sa));
                if (bf_verify_sa[0] != BOOT_FENCE_READY_VALUE) {
                    log_warning(
                        tt::LogMetal,
                        "FIX S8 (#42429): boot_fence readback MISMATCH — wrote 0x{:08X} "
                        "to L1[0x{:08X}] on Device {} chan={} but read back 0x{:08X}",
                        BOOT_FENCE_READY_VALUE, boot_fence_addr_sa,
                        this->id_, deferred_chan, bf_verify_sa[0]);
                } else {
                    log_info(
                        tt::LogMetal,
                        "FIX S8 (#42429): boot_fence=0x{:08X} at L1[0x{:08X}] for "
                        "Device {} chan={} — firmware will proceed past boot fence poll",
                        BOOT_FENCE_READY_VALUE, boot_fence_addr_sa,
                        this->id_, deferred_chan);
                }
            }

            // Step 5b: FIX S9 (#42429) Session ID — write monotonic session_id to L1.
            // Firmware reads this at boot and checks it against go_msg / launch_msg before
            // acting.  Stale session data is silently rejected.  The session_id is a simple
            // incrementing counter managed by the Device object (this->fabric_session_id_).
            {
                const uint32_t sid = this->fabric_session_id_;
                cluster_sa.write_core_immediate(
                    this->id_, virtual_core_sa,
                    std::vector<uint32_t>{sid},
                    session_id_addr_sa);
                // Readback verify (follows FIX GI pattern).
                std::vector<uint32_t> sid_verify_sa(1, 0);
                cluster_sa.read_core(sid_verify_sa, sizeof(uint32_t), core_loc_sa,
                    static_cast<uint64_t>(session_id_addr_sa));
                if (sid_verify_sa[0] != sid) {
                    log_warning(
                        tt::LogMetal,
                        "FIX S9 (#42429): session_id readback MISMATCH — wrote 0x{:08X} "
                        "to L1[0x{:08X}] on Device {} chan={} but read back 0x{:08X}",
                        sid, session_id_addr_sa,
                        this->id_, deferred_chan, sid_verify_sa[0]);
                } else {
                    log_info(
                        tt::LogMetal,
                        "FIX S9 (#42429): session_id=0x{:08X} at L1[0x{:08X}] for "
                        "Device {} chan={} — firmware will use this session tag",
                        sid, session_id_addr_sa,
                        this->id_, deferred_chan);
                }
            }

            // Steps 5-6 (old FIX DW + FIX DU + FIX PQ) removed by FIX S8.
            // The boot fence above provides deterministic host→firmware sync, replacing:
            //   - FIX DW: 50ms sleep after deassert
            //   - FIX DU: poll edm_status until != ROM postcode (5s timeout)
            //   - FIX PQ: two-phase guard against pre-cleared edm_status

            // Step 7: Write go_msg (RUN_MSG_GO) — ERISC is now in base-UMD polling loop.
            // GAP 14 / FIX SA-GV (#42429): Readback verify go_msg first word after write.
            // Steps 1 and 3 have readback verify, but go_msg did not.  If go_msg write
            // silently fails (PCIe error, wrong address), ERISC stays in base-UMD loop
            // forever — no fabric firmware launches, FIX EF times out.
            {
                auto go_msg_view_sa = kg_sa_ptr->go_msg.view();
                auto physical_core_go = this->virtual_core_from_logical_core(
                    CoreCoord(logical_core_sa.x, logical_core_sa.y), CoreType::ETH);
                uint64_t go_addr_sa = hal_sa.get_dev_noc_addr(
                    HalProgrammableCoreType::ACTIVE_ETH, HalL1MemAddrType::GO_MSG);
                cluster_sa.write_core_immediate(
                    go_msg_view_sa.data(), go_msg_view_sa.size(),
                    {static_cast<size_t>(this->id_), physical_core_go}, go_addr_sa);
                // FIX SA-GV: readback verify first word of go_msg.
                uint32_t expected_go_word = 0;
                if (go_msg_view_sa.size() >= sizeof(uint32_t)) {
                    std::memcpy(&expected_go_word, go_msg_view_sa.data(), sizeof(uint32_t));
                    std::vector<uint32_t> go_verify_sa(1, 0xFFFFFFFF);
                    cluster_sa.read_core(go_verify_sa, sizeof(uint32_t), core_loc_sa, go_addr_sa);
                    if (go_verify_sa[0] != expected_go_word) {
                        log_warning(
                            tt::LogMetal,
                            "FIX SA (#42429): FIX SA-GV go_msg readback MISMATCH — wrote 0x{:08X} "
                            "to go_addr=0x{:08X} on Device {} chan={} but read back 0x{:08X}",
                            expected_go_word, static_cast<uint32_t>(go_addr_sa),
                            this->id_, deferred_chan, go_verify_sa[0]);
                    } else {
                        log_info(
                            tt::LogMetal,
                            "FIX SA (#42429): wrote go_msg (RUN_MSG_GO) for Device {} chan={} — "
                            "readback verified (first word=0x{:08X}). "
                            "ERISC should now launch fabric firmware. [FIX SA-GV]",
                            this->id_, deferred_chan, expected_go_word);
                    }
                } else {
                    log_info(
                        tt::LogMetal,
                        "FIX SA (#42429): wrote go_msg (RUN_MSG_GO) for Device {} chan={} — "
                        "ERISC should now launch fabric firmware (go_msg too small for readback verify)",
                        this->id_, deferred_chan);
                }
            }

            // GAP 12 / FIX OP SA (#42429): Log elapsed time for this channel's full
            // Strategy A sequence (steps 1-7).  Mirrors FIX OP on the normal FIX S9 path.
            {
                auto fix_op_sa_end = std::chrono::steady_clock::now();
                auto fix_op_sa_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                    fix_op_sa_end - fix_op_sa_start).count();
                log_info(
                    tt::LogMetal,
                    "FIX OP SA (#42429): Strategy A per-channel — Device {} chan={} "
                    "total sequence took {}ms (steps 1-7: MM+launch_msg+bypass+deassert+S8_boot_fence+go_msg)",
                    this->id_, deferred_chan, fix_op_sa_ms);
            }
        }

        // GAP-R11 (#42429): Count SA-A failures (FW_READY timeouts) for summary.
        // all_dead_channels_storage was modified in-place by SA-A timeout marking.
        // SA-A failures = current dead size - dead size before SA loop.
        // We can approximate: SA-A failures are channels in deferred_deassert that
        // ended up in all_dead_channels_storage.
        uint32_t sa_a_timeout_count = 0;
        for (const auto& dc : health.deferred_deassert_channels) {
            if (all_dead_channels_storage.count(dc)) {
                sa_a_timeout_count++;
            }
        }
        log_info(
            tt::LogMetal,
            "FIX SA (#42429): Strategy A complete for Device {} — {} deferred channel(s) "
            "processed, {} SA-A FW_READY timeouts, session_id=0x{:08X}",
            this->id_, health.deferred_deassert_channels.size(),
            sa_a_timeout_count, this->fabric_session_id_);
    }
    // =========================================================================
    // End FIX SA (Strategy A)
    // =========================================================================

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

    // HOST PRE-LAUNCH CANARY (#42429 race-condition fix):
    // Write kHostPreLaunchCanary to router_sync_address for each ETH channel BEFORE sending the
    // launch message.  This lets terminate_stale_erisc_routers() in the NEXT session distinguish:
    //   0x49706550  → base-UMD relay, launch message never sent     (leave alone)
    //   0xDEADB07E  → host wrote canary but ERISC never started     (soft-reset needed)
    //   0xA0A0A0A0  → ERISC entered kernel_main but crashed early   (soft-reset needed)
    //   EDMStatus   → valid fabric firmware state
    // Channels in skip_soft_reset_channels carry live UMD relay firmware — must NOT be disturbed.
    const auto& control_plane_cf = MetalContext::instance().get_control_plane();
    const auto& fabric_context_cf = control_plane_cf.get_fabric_context();
    const auto& builder_ctx_cf = fabric_context_cf.get_builder_context();
    const auto router_sync_address = builder_ctx_cf.get_fabric_router_sync_address_and_status().first;
    static constexpr uint32_t kHostPreLaunchCanary = static_cast<uint32_t>(EthDiagSentinel::HOST_PRE_LAUNCH_CANARY);
    std::vector<uint32_t> canary_buf{kHostPreLaunchCanary};

    for (uint32_t programmable_core_type_index = 0; programmable_core_type_index < logical_cores_used_in_program.size();
         programmable_core_type_index++) {
        CoreType core_type = hal.get_core_type(programmable_core_type_index);
        for (const auto& logical_core : logical_cores_used_in_program[programmable_core_type_index]) {
            // FIX DY: Hoist ETH channel resolution to outer loop scope so downstream FIX C,
            // the canary write, and the FIX DY postcode-verification poll all share a single
            // resolution instead of repeating the soc_desc lookup three times per core.
            static constexpr uint32_t kUnresolvedChan = std::numeric_limits<uint32_t>::max();
            uint32_t hoisted_eth_chan = kUnresolvedChan;
            bool is_fixm_chan = false;       // non-MMIO and in skip_soft_reset_channels (FIX M)
            bool is_skip_reset_chan = false;  // skip canary: in skip_soft_reset OR external
            if (core_type == CoreType::ETH) {
                try {
                    hoisted_eth_chan = soc_desc_for_dead.get_eth_channel_for_core(
                        tt::umd::CoreCoord(logical_core.x, logical_core.y, CoreType::ETH, CoordSystem::LOGICAL),
                        CoordSystem::LOGICAL);
                    is_fixm_chan =
                        !this->is_mmio_capable() && skip_soft_reset_channels.count(hoisted_eth_chan) > 0;
                    is_skip_reset_chan =
                        skip_soft_reset_channels.count(hoisted_eth_chan) > 0 ||
                        external_umd_channels.count(hoisted_eth_chan) > 0;
                } catch (...) {
                    // Cannot resolve channel.  Treat conservatively: skip canary write (same as
                    // before), and do NOT attempt FIX DY poll (unknown channel = no safe read).
                    is_skip_reset_chan = true;
                }
            }

            // FIX C: skip write_launch_msg_to_core for ETH cores whose channel is dead.
            // write_launch_msg_to_core calls tt_cluster::write_core → WriteToDevice → UMD
            // write_to_device.  On a non-MMIO device, this routes through the ETH relay;
            // if the relay ERISC is dead the write hangs indefinitely (no per-write timeout).
            if (core_type == CoreType::ETH && !all_dead_channels.empty()) {
                if (hoisted_eth_chan == kUnresolvedChan) {
                    // Could not resolve channel earlier — skip conservatively.
                    log_warning(
                        tt::LogMetal,
                        "configure_fabric: Device {} cannot resolve ETH channel for logical core "
                        "({},{}) — skipping write_launch_msg_to_core to avoid potential hang",
                        this->id_,
                        logical_core.x,
                        logical_core.y);
                    continue;
                }
                if (all_dead_channels.count(hoisted_eth_chan)) {
                    log_debug(
                        tt::LogMetal,
                        "configure_fabric: Device {} skipping write_launch_msg_to_core for "
                        "dead ETH core ({},{}) channel {}",
                        this->id_,
                        logical_core.x,
                        logical_core.y,
                        hoisted_eth_chan);
                    continue;
                }
            }

            // FIX SA (#42429): Skip deferred-deassert channels — launch_msg + go_msg already
            // written in the Strategy A block above (after ConfigureDeviceWithProgram).
            if (core_type == CoreType::ETH && hoisted_eth_chan != kUnresolvedChan &&
                health.deferred_deassert_channels.count(hoisted_eth_chan)) {
                log_debug(
                    tt::LogMetal,
                    "FIX SA (#42429): Device {} skipping write_launch_msg_to_core for "
                    "deferred-deassert chan={} logical ({},{}) — already handled in Strategy A",
                    this->id_, hoisted_eth_chan, logical_core.x, logical_core.y);
                continue;
            }

            // Write host-side pre-launch canary so terminate_stale_erisc_routers() in the next
            // session can distinguish "UMD relay never launched" (0x49706550) from "launch sent
            // but ERISC crashed before writing 0xA0A0A0A0 firmware canary" (0xDEADB07E).
            // Skip channels in skip_soft_reset_channels (live UMD relay — must NOT disturb).
            if (core_type == CoreType::ETH && !is_skip_reset_chan) {
                try {
                    detail::WriteToDeviceL1(this, logical_core, router_sync_address, canary_buf, CoreType::ETH);
                } catch (const std::exception& e) {
                    log_warning(
                        tt::LogMetal,
                        "configure_fabric: Device {} core ({},{}) host-canary write failed: {}",
                        this->id_,
                        logical_core.x,
                        logical_core.y,
                        e.what());
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
                /* send_go= */ true);  // FIX SENDGO: was hal.get_dev_addr (uint64_t→bool implicit conversion)

            // FIX IJ/KL (#42429): DISABLED — now redundant after FIX MM.
            // FIX MM unconditionally restores fw_launch_addr in configure_fabric_cores()
            // for ALL surviving MMIO channels after L1 clear. This eliminates the
            // whack-a-mole problem where FIX IJ/KL conditions were incomplete and missed
            // some channel paths (26 DEADB07E occurrences in CI run 26022229604).
#ifdef FIXIJ_REDUNDANT_AFTER_FIX_MM
            if (core_type == CoreType::ETH && this->is_mmio_capable() &&
                hoisted_eth_chan != kUnresolvedChan &&
                (skip_soft_reset_channels.count(hoisted_eth_chan) ||
                 mmio_base_umd_channels.count(hoisted_eth_chan))) {
                const bool via_ee_path = !skip_soft_reset_channels.count(hoisted_eth_chan) &&
                                          mmio_base_umd_channels.count(hoisted_eth_chan);
                if (via_ee_path) {
                    log_debug(
                        tt::LogMetal,
                        "FIX IJ EE (#42429): restoring fw_launch_addr for MMIO base-UMD "
                        "chan={} on Device {} (FIX KL path — not in skip_soft_reset_channels).",
                        hoisted_eth_chan,
                        this->id_);
                }
                const auto& hal_ij = env_impl.get_hal();
                const auto aeth_idx_ij = hal_ij.get_programmable_core_type_index(
                    HalProgrammableCoreType::ACTIVE_ETH);
                const auto& jit_cfg_ij = hal_ij.get_jit_build_config(aeth_idx_ij, 0, 0);
                auto& cluster_ij = env_impl.get_cluster();
                auto virtual_core_ij = cluster_ij.get_virtual_eth_core_from_channel(this->id(), hoisted_eth_chan);
                // FIX MN (#42429): Pre-restore snapshot of fw_launch_addr (initial path).
                std::vector<uint32_t> ij_pre(1, 0xFFFFFFFF);
                cluster_ij.read_core(ij_pre, sizeof(uint32_t),
                    tt_cxy_pair(this->id(), virtual_core_ij),
                    static_cast<uint64_t>(jit_cfg_ij.fw_launch_addr));
                log_info(
                    tt::LogMetal,
                    "FIX MN (#42429): FIX IJ initial pre-restore — Device {} chan={} "
                    "fw_launch_addr=0x{:08X} pre_val=0x{:08X} (expect 0 from FIX EG)",
                    this->id(), hoisted_eth_chan,
                    jit_cfg_ij.fw_launch_addr, ij_pre[0]);
                cluster_ij.write_core(
                    &jit_cfg_ij.fw_launch_addr_value,
                    sizeof(uint32_t),
                    tt_cxy_pair(this->id(), virtual_core_ij),
                    jit_cfg_ij.fw_launch_addr);
                std::vector<uint32_t> ij_verify(1, 0);
                cluster_ij.read_core(ij_verify, sizeof(uint32_t),
                    tt_cxy_pair(this->id(), virtual_core_ij),
                    static_cast<uint64_t>(jit_cfg_ij.fw_launch_addr));
                if (ij_verify[0] != jit_cfg_ij.fw_launch_addr_value) {
                    log_warning(
                        tt::LogMetal,
                        "FIX IJ (#42429): fw_launch_addr readback MISMATCH — wrote "
                        "0x{:08X} to 0x{:08X} on Device {} chan={} but read back 0x{:08X}. "
                        "Base-UMD may stay at 0xDEADB07E.",
                        jit_cfg_ij.fw_launch_addr_value,
                        jit_cfg_ij.fw_launch_addr,
                        this->id_,
                        hoisted_eth_chan,
                        ij_verify[0]);
                } else {
                    log_info(
                        tt::LogMetal,
                        "FIX IJ (#42429): restored fw_launch_addr_value=0x{:08X} at "
                        "fw_launch_addr=0x{:08X} for Device {} chan={} after firmware load "
                        "(readback verified). Base-UMD will now launch fabric firmware.",
                        jit_cfg_ij.fw_launch_addr_value,
                        jit_cfg_ij.fw_launch_addr,
                        this->id_,
                        hoisted_eth_chan);
                }
            }
#endif  // FIXIJ_REDUNDANT_AFTER_FIX_MM

            // FIX DY: For FIX M channels (non-MMIO ETH, skipped soft-reset), verify that the
            // ERISC actually transitioned away from 0x49706550 (base-UMD sentinel).
            //
            // Root cause: write_launch_msg_to_core uses write_core_immediate (fire-and-forget,
            // no relay-flush ACK).  If the MMIO relay drops the packet — e.g. due to congestion
            // from a concurrent FIX S9 soft-reset on a nearby MMIO device — the ERISC silently
            // stays at 0x49706550 ("zombie" state).  Subsequent dispatch init calls write_core
            // (with timeout) on those zombie ERISCs and hits the 5-second hang × N devices that
            // we observed in CI run 25979253612.
            //
            // Fix: poll router_sync_address for != 0x49706550, max 500 ms at 5 ms intervals.
            // The MMIO relay is still alive here (Pass 1, before FIX S9 resets it), so the read
            // goes through the working relay path.  On timeout, retry write_launch_msg_to_core
            // once and re-poll for another 500 ms.  Log a warning if still stuck after retry.
            if (is_fixm_chan) {
                static constexpr uint32_t kBaseUmdSentinel = 0x49706550u;
                static constexpr int kFIX_DY_PollIntervalMs = 5;
                static constexpr int kFIX_DY_PollMaxMs = 500;
                auto fixdy_poll = [&](bool is_retry) -> bool {
                    const int max_iters = kFIX_DY_PollMaxMs / kFIX_DY_PollIntervalMs;
                    for (int iter = 0; iter < max_iters; ++iter) {
                        std::vector<uint32_t> status_buf(1, kBaseUmdSentinel);
                        try {
                            detail::ReadFromDeviceL1(
                                this, logical_core, router_sync_address, 4, status_buf, CoreType::ETH);
                        } catch (const std::exception& e) {
                            log_warning(
                                tt::LogMetal,
                                "FIX DY: Device {} chan {} read failed{}: {}",
                                this->id_,
                                hoisted_eth_chan,
                                is_retry ? " (retry)" : "",
                                e.what());
                            return false;
                        }
                        if (status_buf[0] != kBaseUmdSentinel) {
                            log_info(
                                tt::LogMetal,
                                "FIX DY: Device {} chan {} transitioned from 0x49706550 → 0x{:08x} "
                                "after {}ms{}",
                                this->id_,
                                hoisted_eth_chan,
                                status_buf[0],
                                iter * kFIX_DY_PollIntervalMs,
                                is_retry ? " (retry)" : "");
                            return true;
                        }
                        std::this_thread::sleep_for(std::chrono::milliseconds(kFIX_DY_PollIntervalMs));
                    }
                    return false;
                };

                if (!fixdy_poll(/*is_retry=*/false)) {
                    // Timeout on first attempt — relay may have dropped the packet.  Retry once.
                    log_warning(
                        tt::LogMetal,
                        "FIX DY: Device {} chan {} still at 0x49706550 after {}ms — "
                        "retrying write_launch_msg_to_core (relay packet may have been dropped)",
                        this->id_,
                        hoisted_eth_chan,
                        kFIX_DY_PollMaxMs);
                    tt::llrt::write_launch_msg_to_core(
                        this->id(),
                        physical_core,
                        msg,
                        go_msg,
                        /* send_go= */ true);  // FIX SENDGO: was hal.get_dev_addr (uint64_t→bool implicit conversion)
                    if (!fixdy_poll(/*is_retry=*/true)) {
                        log_warning(
                            tt::LogMetal,
                            "FIX DY: Device {} chan {} still at 0x49706550 after retry — "
                            "channel is zombie; downstream write_core may hang.  "
                            "Setting fabric_relay_path_broken_=true.",
                            this->id_,
                            hoisted_eth_chan);
                        fabric_relay_path_broken_.store(true);
                    }
                }
            } else if (
                this->is_mmio_capable() && core_type == CoreType::ETH &&
                hoisted_eth_chan != kUnresolvedChan && !is_skip_reset_chan) {
                // FIX EF (#42429): For MMIO ETH channels (D0-D3 relay ERISCs), poll after
                // write_launch_msg_to_core until the channel exits HOST_PRE_LAUNCH_CANARY
                // (0xDEADB07E) before proceeding.
                //
                // Root cause of missing sync: write_launch_msg_to_core is fire-and-forget.
                // On MMIO devices the write goes direct via PCIe, so there is no relay-drop
                // risk — but there IS a startup-time gap between the host sending the launch
                // message and the ERISC actually executing kernel_main and overwriting
                // router_sync_address.  If D4-D7 routing-table writes reach the D0-D3 relay
                // ERISC before it has finished initialising, those writes may be silently
                // dropped or processed out of order, leaving D4-D7 permanently stuck at
                // REMOTE_HANDSHAKE_COMPLETE (FIX NX) or causing FIX DT-1 dispatch timeouts
                // on physical cores 23-17 / 19-17.
                //
                // This is the MMIO analogue of FIX DY (which protects non-MMIO FIX M
                // channels from the same fire-and-forget race).  FIX DY polls for exit from
                // 0x49706550; FIX EF polls for exit from 0xDEADB07E.
                //
                // Poll parameters: 3000 ms max, 5 ms interval.  MMIO reads are direct PCIe
                // and typically complete in < 1 ms, so the poll adds negligible overhead on
                // the healthy path.  3 s gives ample margin for ERISC startup (typically
                // < 50 ms) while bounding the worst-case stall.
                //
                // BLOCKING WAIT (FIX EF upgrade): This poll is now a hard gate.  Routing
                // table writes from D4-D7 configure_fabric() pass through the D0-D3 relay
                // ERISCs.  If a D0-D3 relay ERISC is still at 0xDEADB07E when those writes
                // arrive, they are silently dropped — leaving D4-D7 permanently stuck at
                // REMOTE_HANDSHAKE_COMPLETE (FIX NX) or producing FIX DT-1 dispatch timeouts.
                // On timeout: set fabric_relay_path_broken_=true so that compile_and_configure
                // fabric FIX SB2 propagates the broken flag to dependent non-MMIO devices,
                // preventing 5 s-per-channel relay-read timeouts from accumulating downstream.
                static constexpr uint32_t kPreLaunchCanary =
                    static_cast<uint32_t>(EthDiagSentinel::HOST_PRE_LAUNCH_CANARY);
                static constexpr int kFIX_EF_PollIntervalMs = 5;
                static constexpr int kFIX_EF_PollMaxMs = 3000;
                const int max_iters = kFIX_EF_PollMaxMs / kFIX_EF_PollIntervalMs;
                bool exited_canary = false;
                for (int iter = 0; iter < max_iters; ++iter) {
                    std::vector<uint32_t> status_buf(1, kPreLaunchCanary);
                    try {
                        detail::ReadFromDeviceL1(
                            this, logical_core, router_sync_address, 4, status_buf, CoreType::ETH);
                    } catch (const std::exception& e) {
                        log_warning(
                            tt::LogMetal,
                            "FIX EF: Device {} chan {} read failed: {}",
                            this->id_,
                            hoisted_eth_chan,
                            e.what());
                        break;
                    }
                    if (status_buf[0] != kPreLaunchCanary) {
                        log_info(
                            tt::LogMetal,
                            "FIX EF: Device {} chan {} (MMIO) exited 0xDEADB07E → 0x{:08x} "
                            "after {}ms — relay ERISC ready. (#42429)",
                            this->id_,
                            hoisted_eth_chan,
                            status_buf[0],
                            iter * kFIX_EF_PollIntervalMs);
                        exited_canary = true;
                        break;
                    }
                    std::this_thread::sleep_for(std::chrono::milliseconds(kFIX_EF_PollIntervalMs));
                }
                if (!exited_canary) {
                    // BLOCKING WAIT TIMED OUT: relay ERISC did not start within 3 s.
                    // Set fabric_relay_path_broken_ so compile_and_configure_fabric FIX SB2
                    // propagates the broken state to non-MMIO devices, preventing them from
                    // issuing routing-table writes through a relay that never became ready.

                    // FIX NP (#42429): Read actual edm_status at timeout for accurate diagnostics.
                    // The poll assumes 0xDEADB07E but ERISC may have transitioned to a DIFFERENT
                    // bad state (e.g., 0x00000000, 0xA0A0A0A0). Without this read, the log says
                    // "still at 0xDEADB07E" which may be misleading.
                    uint32_t actual_status_at_timeout = kPreLaunchCanary;  // default assumption
                    try {
                        std::vector<uint32_t> timeout_buf(1, kPreLaunchCanary);
                        detail::ReadFromDeviceL1(
                            this, logical_core, router_sync_address, 4, timeout_buf, CoreType::ETH);
                        actual_status_at_timeout = timeout_buf[0];
                    } catch (...) {
                        // Best-effort — PCIe read may fail if device is in a bad state.
                    }
                    fabric_relay_path_broken_.store(true);
                    log_warning(
                        tt::LogMetal,
                        "FIX EF: Device {} chan {} (MMIO) did not exit canary after {}ms — "
                        "actual edm_status=0x{:08x} (expected 0xDEADB07E). "
                        "Setting fabric_relay_path_broken_=true "
                        "to block D4-D7 routing writes through dead relay. "
                        "FIX NP (#42429): timeout snapshot.",
                        this->id_,
                        hoisted_eth_chan,
                        kFIX_EF_PollMaxMs,
                        actual_status_at_timeout);
                }
            }
        }
    }
    // FIX RZ (#42429): If this non-MMIO device had base-UMD relay channels that required
    // FIX M's skip-soft-reset / launch_msg transition, mark the device as having stale
    // base-UMD channels.  The Python test's is_fabric_degraded() check uses this flag to
    // skip AllGather operations that would hang on devices whose channels were transitioned
    // via launch_msg but cannot handle AllGather traffic reliably.
    if (!skip_soft_reset_channels.empty() && !this->is_mmio_capable()) {
        fabric_stale_base_umd_channels_ = true;
        // FIX RZ3 (#42429): Also set the persistent companion — ring-sync passing later
        // (FIX RZ2) must NOT clear this.  is_fabric_degraded() and GAP-A/GAP-C will check
        // it to block AllGather dispatch for the entire session.
        fabric_base_umd_fixm_init_ = true;
        log_warning(
            tt::LogMetal,
            "configure_fabric: Device {} (non-MMIO) has {} base-UMD channel(s) transitioned "
            "via launch_msg (FIX M).  Setting fabric_stale_base_umd_channels_=true and "
            "fabric_base_umd_fixm_init_=true — AllGather on this cluster may hang.  "
            "FIX RZ skips the Python stress test; FIX RZ3 persists the guard past ring-sync. (#42429)",
            this->id_,
            skip_soft_reset_channels.size());
    }
    // Exit summary: on the healthy path (no dead channels, no relay-soft-reset skips) emit
    // a single compact line.  Only log the verbose detail block when something is non-trivial
    // so CI logs are not flooded with 8+ lines per device per iteration on every healthy run.
    const bool relay_broken_now = fabric_relay_path_broken_.load();
    if (relay_broken_now || !skip_soft_reset_channels.empty() || !pre_dead_channels.empty()) {
        log_info(
            tt::LogMetal,
            "configure_fabric: Device {} complete — relay_broken={} mmio={} "
            "pre_dead_channels={} skip_soft_reset_channels={} newly_dead={} rr_recovered={}",
            this->id_,
            relay_broken_now,
            this->is_mmio_capable(),
            effective_pre_dead.size(),
            skip_soft_reset_channels.size(),
            health.newly_dead_channels.size(),
            health.recovered_channels.size());
    }
    // GAP-R5 (#42429): Log total elapsed for configure_fabric.
    {
        auto configure_fabric_elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::steady_clock::now() - configure_fabric_start).count();
        log_info(
            tt::LogMetal,
            "configure_fabric: Device {} complete — total elapsed {}ms (session_id=0x{:08X})",
            this->id_,
            configure_fabric_elapsed,
            fabric_session_id_);
    }
    log_info(tt::LogMetal, "Fabric initialized on Device {}", this->id_);
}

void Device::quiesce_and_restart_fabric_workers(bool defer_eth_launch) {
    // GAP-R6 (#42429): Total elapsed timer for quiesce_and_restart — can take minutes with
    // MUX terminate + ERISC terminate + re-launch phases.  Logged at every exit point.
    const auto quiesce_restart_start = std::chrono::steady_clock::now();

#ifdef STRATEGY9_SESSION_ID
    // FIX V11-QS89 (#42429): Bump session ID for quiesce restart — parity with
    // configure_fabric(). Without this, force-reset channels that go through the
    // S8/S9 boot fence sequence in the quiesce path would reuse the stale session_id
    // from the previous configure_fabric() call.
    fabric_session_id_++;
    log_info(
        tt::LogMetal,
        "FIX V11-QS89 (#42429): quiesce_and_restart_fabric_workers Device {} bumped "
        "fabric_session_id_ to 0x{:08X}",
        this->id(), fabric_session_id_);
#endif  // STRATEGY9_SESSION_ID

    // Diagnostic: env toggle lets CI / repro runs skip this restart path entirely to isolate
    // whether the Tensix MUX restart is the cause of a post-quiesce hang. When set, we return
    // before any fabric MUX termination. See plan Experiment B.
    if (const char* env = std::getenv("TT_METAL_DISABLE_QUIESCE_FABRIC_RESTART");
        env != nullptr && env[0] != '\0' && env[0] != '0') {
        log_info(
            tt::LogMetal,
            "quiesce_and_restart_fabric_workers: Device {} early-return: "
            "TT_METAL_DISABLE_QUIESCE_FABRIC_RESTART={} (restart path disabled) — elapsed {}ms",
            this->id(),
            env,
            std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::steady_clock::now() - quiesce_restart_start).count());
        return;
    }

    auto fabric_config = MetalContext::instance().get_fabric_config();
    if (!tt_fabric::is_tt_fabric_config(fabric_config)) {
        log_info(
            tt::LogMetal,
            "quiesce_and_restart_fabric_workers: Device {} early-return at guard L426: "
            "!is_tt_fabric_config(fabric_config={}) — elapsed {}ms",
            this->id(),
            static_cast<uint32_t>(fabric_config),
            std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::steady_clock::now() - quiesce_restart_start).count());
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
            "get_num_fabric_initialized_routers == 0 — elapsed {}ms",
            this->id(),
            std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::steady_clock::now() - quiesce_restart_start).count());
        return;
    }

    const auto fabric_node_id = control_plane.get_fabric_node_id_from_physical_chip_id(this->id());
    const auto& active_channels = control_plane.get_active_fabric_eth_channels(fabric_node_id);

    MetalEnvImpl& env_impl = MetalEnvAccessor(*env_).impl();

    // GAP-R14 (#42429): Log session_id unconditionally at quiesce entry.
    // The FIX V11-QS89 bump only logs under #ifdef STRATEGY9_SESSION_ID.
    // Without this, post-mortem has no session context when the ifdef is disabled.
    log_info(
        tt::LogMetal,
        "GAP-R14 (#42429): quiesce_and_restart_fabric_workers: Device {} session_id=0x{:08X} "
        "relay_broken={} mmio={} channels_not_ready={} stale_base_umd={}",
        this->id(),
        fabric_session_id_,
        fabric_relay_path_broken_.load(),
        this->is_mmio_capable(),
        fabric_channels_not_ready_for_traffic_,
        fabric_stale_base_umd_channels_);

    // Diagnostic entry snapshot (#42429): read every active ERISC channel's edm_status_address
    // BEFORE any phase runs.  This lets us distinguish "prior test left channels in bad state"
    // vs "failing test caused the problem" — per Neil's request.
    {
        const auto router_sync_addr_diag = builder_ctx.get_fabric_router_sync_address_and_status().first;
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

                const auto eth_lc_diag = soc_desc_entry.get_eth_core_for_channel(eth_chan_id, CoordSystem::LOGICAL);
                std::vector<uint32_t> diag_buf(1, 0U);
                // Wrap in try/catch: non-MMIO reads route through the MMIO device's ERISC relay.
                // If the MMIO device was quiesced earlier in Pass 1, its ERISCs may not yet be
                // READY_FOR_TRAFFIC and UMD will timeout.  The ENTRY snapshot is diagnostic only —
                // log a warning but do NOT abort the quiesce.
                try {
                    detail::ReadFromDeviceL1(this, eth_lc_diag, router_sync_addr_diag, 4, diag_buf, CoreType::ETH);
                } catch (const std::exception& ex) {
                    log_warning(
                        tt::LogMetal,
                        "quiesce_and_restart_fabric_workers: Device {} ENTRY snapshot: "
                        "eth_chan {} L1 read failed (relay not ready?): {}",
                        this->id(),
                        eth_chan_id,
                        ex.what());
                    diag_buf[0] = static_cast<uint32_t>(EthDiagSentinel::READ_EXCEPTION);
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

    // FIX-4: Track MUX logical cores whose assert_risc_reset_at_core failed in Phase 2.
    // Phase 3 will skip write_launch_msg_to_core for these cores — writing a launch message
    // to a core whose BRISC may still be running is an unsafe L1 overwrite.
    std::unordered_set<CoreCoord> mux_reset_failed_cores;

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
                    std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - start)
                        .count();
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
                // Escalate to ERROR: a failed force-reset means the MUX BRISC may still be
                // running when Phase 3 overwrites its L1, which can corrupt the MUX kernel
                // and cause an opaque hang on the next dispatch or AllGather operation.
                log_error(
                    tt::LogMetal,
                    "quiesce_and_restart_fabric_workers: assert_risc_reset_at_core FAILED on Device {} "
                    "eth_chan {} — Phase 3 L1 overwrite is unsafe, MUX may still be running: {}",
                    this->id(),
                    eth_chan_id,
                    e.what());
                // Record the failed core so Phase 3 can skip write_launch_msg_to_core for it.
                mux_reset_failed_cores.insert(mux_core);
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
    // and force-reset the ERISC (assert_risc_reset_at_core) to guarantee it is halted
    // before Phase 3 overwrites its L1 — preventing mid-packet-send data corruption.
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
    // FIX AI-2 (#42429): Clear Phase 2.5 force-reset tracking from any prior quiesce cycle.
    // Populated below when Phase 2.5 force-halts unresponsive ERISCs; consumed by Phase 3
    // (inline) or launch_eth_cores_for_quiesce() (deferred) to deassert those RISCs after
    // writing the launch message.
    pending_phase25_force_reset_chans_.clear();
    quiesce_relay_transitioned_ = false;  // FIX DV: clear per-cycle relay-transitioned flag

    if (fabric_relay_path_broken_ && !this->is_mmio_capable()) {
        log_warning(
            tt::LogMetal,
            "quiesce_and_restart_fabric_workers: Device {} Phase 2.5: relay path known broken "
            "(fabric_relay_path_broken_) and device is non-MMIO — skipping all {} Phase 2.5 relay "
            "reads/writes to prevent indefinite UMD relay hang.  Phase 3 will also be skipped.",
            this->id(),
            active_channels.size());
    } else {
        {
            const auto [erisc_term_addr, erisc_term_signal] =
                builder_ctx.get_fabric_router_termination_address_and_signal();
            const auto router_sync_addr = builder_ctx.get_fabric_router_sync_address_and_status().first;
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
                // FIX PY (#42429): If relay_path_broken_ was set by a prior channel in this
                // same Phase 2.5 loop iteration (not just the outer guard above), skip remaining
                // channels on this non-MMIO device immediately.  Without this, after the FIRST
                // channel fails and spends 3×retry×5s≈21s setting fabric_relay_path_broken_=true,
                // the REMAINING channels each also spin through the full 21s retry cycle even
                // though we already know the relay is dead.  Skipping them is safe — Phase 3's
                // FIX Q guard already skips configure_fabric_cores() when relay_path_broken_=true.
                if (fabric_relay_path_broken_ && !this->is_mmio_capable()) {
                    log_warning(
                        tt::LogMetal,
                        "quiesce_and_restart_fabric_workers: Device {} eth_chan {} Phase 2.5: "
                        "relay already marked broken by prior channel — skipping (FIX PY #42429).",
                        this->id(),
                        eth_chan_id);
                    continue;
                }
                const auto eth_logical_core = env_impl.get_cluster()
                                                  .get_soc_desc(this->id())
                                                  .get_eth_core_for_channel(eth_chan_id, CoordSystem::LOGICAL);

                std::vector<uint32_t> status_buf(1, 0);
                // FIX PG (#42429): For non-MMIO devices the L1 read goes through the UMD relay.
                // After an AllGather completes, in-flight relay traffic can cause a transient
                // 5-second relay timeout.  Retry up to 2 times with a 3-second sleep to give
                // the relay time to drain before declaring the path broken (FIX AN).
                // MMIO devices have no relay path, so no retry is needed (max_retries=0).
                const int pg_max_retries = this->is_mmio_capable() ? 0 : 2;
                bool pg_read_ok = false;
                for (int pg_attempt = 0; pg_attempt <= pg_max_retries; ++pg_attempt) {
                    try {
                        detail::ReadFromDeviceL1(
                            this, eth_logical_core, router_sync_addr, 4, status_buf, CoreType::ETH);
                        pg_read_ok = true;
                        break;
                    } catch (const std::exception& e) {
                        if (pg_attempt < pg_max_retries) {
                            log_warning(
                                tt::LogMetal,
                                "quiesce_and_restart_fabric_workers: Device {} eth_chan {} Phase 2.5: "
                                "relay read attempt {}/{} failed, retrying in 3s (FIX PG #42429): {}",
                                this->id(),
                                eth_chan_id,
                                pg_attempt + 1,
                                pg_max_retries + 1,
                                e.what());
                            std::this_thread::sleep_for(std::chrono::seconds(3));
                        } else {
                            // L1 read failed in Phase 2.5.  Two distinct failure modes:
                            // (a) Non-MMIO: UMD relay ERISC is not running relay firmware (Phase 3
                            //     loaded fabric firmware on the MMIO device's relay channels, or
                            //     relay was force-reset).  WriteToDeviceL1 via relay would hang.
                            // (b) MMIO: channel is out-of-mesh (e.g. chan 14/15 not in SOC descriptor
                            //     for a partial-mesh N300 peer) — get_eth_core_for_channel throws.
                            // In both cases: set fabric_relay_path_broken_ = true.  Phase 5 has an
                            // unconditional fabric_relay_path_broken_ skip that protects BOTH MMIO
                            // and non-MMIO devices from reading channels whose ETH state is
                            // indeterminate after a failed Phase 2.5 read.
                            //
                            // FIX AN (#42429): Set fabric_relay_path_broken_ = true unconditionally
                            // so that Phase 3's FIX Q guard skips configure_fabric_cores() and
                            // Phase 5 is skipped via the relay-broken early-return for this device.
                            fabric_relay_path_broken_ = true;
                            log_warning(
                                tt::LogMetal,
                                "quiesce_and_restart_fabric_workers: Device {} eth_chan {} Phase 2.5: "
                                "L1 read failed after {} attempt(s) — setting "
                                "fabric_relay_path_broken_=true (FIX AN/PG #42429): {}",
                                this->id(),
                                eth_chan_id,
                                pg_attempt + 1,
                                e.what());
                        }
                    }
                }
                if (!pg_read_ok) {
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
                        std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - start)
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
                    std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - start)
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
                    log_warning(
                        tt::LogMetal,
                        "quiesce_and_restart_fabric_workers: Device {} (mmio={}) eth_chan {} Phase 2.5: "
                        "ERISC did not terminate within budget {}ms (actual elapsed {}ms, status=0x{:08x} {}) — "
                        "force-resetting ERISC to guarantee it is halted before Phase 3 overwrites L1",
                        this->id(),
                        this->is_mmio_capable(),
                        eth_chan_id,
                        erisc_timeout_ms,
                        p25_elapsed,
                        status_buf[0],
                        edm_status_str(status_buf[0]));
                    // R1: Force-reset the unresponsive ERISC before Phase 3 writes new firmware to its L1.
                    // Without this, Phase 3's configure_fabric_cores() may overwrite L1 while the ERISC
                    // is mid-packet-send, causing data corruption (the same issue this branch fixes).
                    try {
                        const auto eth_virtual_core = virtual_core_from_logical_core(eth_logical_core, CoreType::ETH);
                        env_impl.get_cluster().assert_risc_reset_at_core(
                            tt_cxy_pair(this->id(), eth_virtual_core), tt::umd::RiscType::ALL);
                        pending_phase25_force_reset_chans_.insert(eth_chan_id);
                        log_warning(
                            tt::LogMetal,
                            "quiesce_and_restart_fabric_workers: Device {} eth_chan {} Phase 2.5: "
                            "force-reset applied — ERISC halted before Phase 3 L1 overwrite "
                            "(tracked for Phase 3 deassert)",
                            this->id(),
                            eth_chan_id);
                    } catch (const std::exception& e) {
                        log_warning(
                            tt::LogMetal,
                            "quiesce_and_restart_fabric_workers: Device {} eth_chan {} Phase 2.5: "
                            "force-reset failed — Phase 3 L1 overwrite may be unsafe: {}",
                            this->id(),
                            eth_chan_id,
                            e.what());
                    }
                }
            }
        }
    }  // end else (Phase 2.5 — relay path not broken)

    // GAP-R8 (#42429): Phase 2.5 force-reset summary — how many channels needed force-halt
    // vs how many were active.  High force_reset_count relative to active_channels indicates
    // ERISCs are routinely failing to self-terminate within the 2000ms budget.
    log_info(
        tt::LogMetal,
        "quiesce_and_restart_fabric_workers: Device {} Phase 2.5 summary: "
        "force_reset_count={}/{} active_channels, relay_path_broken={}",
        this->id(),
        pending_phase25_force_reset_chans_.size(),
        active_channels.size(),
        fabric_relay_path_broken_.load());

    // Phase 3: Re-configure and re-launch the fabric workers
    // Reset termination signals, clear channel state, and re-send launch messages
    // for WORKER cores in the fabric program.
    if (fabric_program_ == nullptr) {
        log_info(
            tt::LogMetal,
            "quiesce_and_restart_fabric_workers: Device {} early-return at guard L564: "
            "fabric_program_ == nullptr (Phase 1/2 ran but Phase 3/4 skipped) — elapsed {}ms",
            this->id(),
            std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::steady_clock::now() - quiesce_restart_start).count());
        return;
    }

    log_info(
        tt::LogMetal,
        "quiesce_and_restart_fabric_workers: Device {} entering Phase 3 (re-configure + re-launch)",
        this->id());

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
            "omitted. wait_for_fabric_workers_ready() will also skip Phase 5 for this device. "
            "— elapsed {}ms",
            this->id(),
            std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::steady_clock::now() - quiesce_restart_start).count());
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
    //
    // FIX AD (#42429): Extend FIX N to MMIO devices during quiesce.
    //
    // MMIO ETH channels that act as UMD dispatch tunnel relay senders (connecting MMIO
    // to non-MMIO peers via the card-internal ETH link) must NOT be soft-reset during
    // quiesce Phase 3.  When an MMIO device's ETH channel is asserted into reset via
    // assert_risc_reset_at_core, the UMD ETH relay path to the non-MMIO peer dies
    // immediately — the ETH NOC and relay sender ERISC are both halted.  Any subsequent
    // non-MMIO Phase 5 relay read then times out after 5 s (UMD ETH relay timeout),
    // setting fabric_relay_path_broken_ and cascading into a permanently-broken second
    // quiesce.
    //
    // After Phase 2.5, MMIO ETH channels are in TERMINATED state (fabric router kernel
    // exited, BRISC polling for next launch msg).  In TERMINATED state the ETH core is
    // NOT halted — only the kernel exited.  write_launch_msg_to_core alone (no soft
    // reset) is therefore sufficient to restart the fabric router on MMIO channels,
    // and preserves the ETH relay path for non-MMIO peers throughout the quiesce.
    //
    // All active channels (MMIO and non-MMIO) now skip the soft reset in quiesce Phase 3.
    tt::tt_fabric::FabricCoresHealth quiesce_health;
    {
        std::unordered_set<uint32_t> quiesce_skip_reset_chans;
        // Skip soft reset for all channels (MMIO and non-MMIO) — see FIX N and FIX AD above.
        for (const auto& [chan, unused_direction] : active_channels) {
            quiesce_skip_reset_chans.insert(chan);
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
            // FIX-4: Skip write_launch_msg_to_core if Phase 2 assert_risc_reset_at_core failed
            // for this MUX WORKER core.  The BRISC may still be running; issuing a launch message
            // would overwrite its L1 state with a new kernel binary header, which can generate
            // invalid NOC traffic and corrupt device state.
            if (mux_reset_failed_cores.count(logical_core)) {
                log_error(
                    tt::LogMetal,
                    "quiesce_and_restart_fabric_workers: Phase 3: skipping write_launch_msg_to_core for "
                    "MUX WORKER core ({},{}) — Phase 2 assert_risc_reset_at_core failed, core may still "
                    "be running",
                    logical_core.x,
                    logical_core.y);
                continue;
            }

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
                /* send_go= */ true);  // FIX SENDGO: was hal.get_dev_addr (uint64_t→bool implicit conversion)

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

    // FIX AE (#42429): Optionally defer ETH write_launch_msg to allow the mesh-level caller
    // (quiesce_internal) to sequence MMIO ETH launch before non-MMIO ETH launch.
    //
    // Simultaneous ETH handshake deadlock: when two ETH peer channels both start within a
    // narrow window (~6ms), both initiate the handshake simultaneously, neither responds to
    // the other's initiation, and both remain stuck at STARTED (0xa0b0c0d0) indefinitely.
    //
    // This is observed when Device 4 (non-MMIO, slow relay ~200ms/channel) and Device 5
    // (non-MMIO, fast relay <1ms/channel via idle MMIO partner) are processed sequentially:
    // Device 4's last write_launch_msg completes at T, Device 5's corresponding peer channel
    // write_launch_msg completes at T+6ms — too close for the handshake protocol to avoid
    // simultaneous initiation.
    //
    // Fix: mesh-level caller defers ETH launch (defer_eth_launch=true), then sequences:
    //   1. MMIO devices ETH launch (fast, direct PCIe) — MMIO peers running first
    //   2. Non-MMIO devices ETH launch (slow, via relay, sequential) — relay serialization
    //      creates ~200ms gap between successive non-MMIO device channel starts, breaking
    //      the simultaneous initiation window.
    if (defer_eth_launch) {
        pending_quiesce_newly_dead_eth_chans_ = quiesce_health.newly_dead_channels;
        pending_eth_launch_ = true;
        log_info(
            tt::LogMetal,
            "quiesce_and_restart_fabric_workers: Device {} Phase 3: ETH write_launch_msg deferred "
            "(defer_eth_launch=true) — call launch_eth_cores_for_quiesce() to complete.",
            this->id());
        log_info(
            tt::LogMetal,
            "quiesce_and_restart_fabric_workers: Device {} Phase 3 complete (ETH deferred) — "
            "WORKER cores relaunched. ETH ERISC launch pending launch_eth_cores_for_quiesce(). "
            "— elapsed {}ms",
            this->id(),
            std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::steady_clock::now() - quiesce_restart_start).count());
        return;
    }
    // ETH write_launch_msg executes inline (defer_eth_launch=false, the default).
    pending_eth_launch_ = false;

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
    // FIX AR (#42429): Pass 0 — deassert Phase-2.5 force-reset channels BEFORE writing
    // launch messages, then sleep briefly to let base UMD firmware complete .bss init.
    // See launch_eth_cores_for_quiesce for the detailed root-cause explanation.
    // ETH cores not in p25 force-reset must NOT have deassert called — their BRISC was
    // never halted in Phase 2 (resetting ERISC tears down the ETH PHY link on WH).
    if (!pending_phase25_force_reset_chans_.empty()) {
        // FIX AS (#42429): track which logical cores were successfully deasserted so we can
        // poll each one individually rather than sleeping a fixed 50ms.
        std::vector<CoreCoord> deasserted_lcs_inline;
        for (uint32_t p0_idx = 0; p0_idx < logical_cores_used.size(); p0_idx++) {
            if (hal.get_core_type(p0_idx) != CoreType::ETH) {
                continue;
            }
            for (const auto& lc0 : logical_cores_used[p0_idx]) {
                try {
                    auto eth_chan_0 = soc_desc_q.get_eth_channel_for_core(
                        tt::umd::CoreCoord(lc0.x, lc0.y, CoreType::ETH, CoordSystem::LOGICAL), CoordSystem::LOGICAL);
                    if (!pending_phase25_force_reset_chans_.count(eth_chan_0)) {
                        continue;
                    }
                    auto phys_core_0 = this->virtual_core_from_logical_core(lc0, CoreType::ETH);
                    env_impl.get_cluster().deassert_risc_reset_at_core(
                        tt_cxy_pair(this->id(), phys_core_0), tt::umd::RiscType::ALL);
                    log_info(
                        tt::LogMetal,
                        "quiesce_and_restart_fabric_workers: Device {} Pass-0 (FIX AR): "
                        "deassert_risc_reset(ALL) for Phase-2.5-halted ETH logical ({},{}) "
                        "channel {} — ERISC now booting into base UMD firmware",
                        this->id(),
                        lc0.x,
                        lc0.y,
                        eth_chan_0);
                    // FIX PD (GAP-50): clear ERISC dispatch fw_launch_addr after Phase 2.5
                    // force-reset. HW reset does NOT zero L1. If dispatch firmware was running,
                    // fw_launch_addr retains its non-zero value → 500ms cascade on next open.
                    // (140 occurrences/run observed in t3k_ttnn_tests, MMIO devices 0-3 chans
                    //  6-9,14. FIX PC was in the wrong path; this quiesce path is the source.)
                    try {
                        const auto& hal_pd = env_impl.get_hal();
                        const auto aeth_idx =
                            hal_pd.get_programmable_core_type_index(HalProgrammableCoreType::ACTIVE_ETH);
                        const uint32_t fw_launch_addr_pd = hal_pd.get_jit_build_config(aeth_idx, 0, 0).fw_launch_addr;
                        env_impl.get_cluster().write_core_immediate(
                            this->id(), phys_core_0, std::vector<uint32_t>{0}, fw_launch_addr_pd);
                    } catch (const std::exception& ex_pd) {
                        // Best-effort: non-MMIO dead relay may throw — FIX PA in reset_cores()
                        // handles the one-time fallback.
                        // GAP-B (#42429): Log warning for MMIO devices where PCIe writes should
                        // never fail. Silent failure masks HAL/hardware bugs → 500ms cascade
                        // persists with no diagnostic.
                        if (this->is_mmio_capable()) {
                            log_warning(
                                tt::LogMetal,
                                "FIX PD (GAP-B): MMIO device {} fw_launch_addr clear FAILED "
                                "(PCIe write should not fail): {}",
                                this->id(),
                                ex_pd.what());
                        }
                    } catch (...) {
                        if (this->is_mmio_capable()) {
                            log_warning(
                                tt::LogMetal,
                                "FIX PD (GAP-B): MMIO device {} fw_launch_addr clear FAILED "
                                "(non-std exception on PCIe write)",
                                this->id());
                        }
                    }
                    deasserted_lcs_inline.push_back(lc0);
                } catch (const std::exception& e) {
                    log_warning(
                        tt::LogMetal,
                        "quiesce_and_restart_fabric_workers: Device {} Pass-0 (FIX AR): "
                        "deassert_risc_reset(ALL) failed for ETH logical ({},{}): {}",
                        this->id(),
                        lc0.x,
                        lc0.y,
                        e.what());
                } catch (...) {
                    log_warning(
                        tt::LogMetal,
                        "quiesce_and_restart_fabric_workers: Device {} Pass-0 (FIX AR): "
                        "deassert_risc_reset(ALL) failed (unknown) for ETH logical ({},{})",
                        this->id(),
                        lc0.x,
                        lc0.y);
                }
            }
        }
        // FIX V11-QS89 (#42429): Per-channel FW_READY poll + S7/S8/S9 boot sequence for
        // force-reset channels.  Replaces the old UMD canary poll (FIX AS) which was polling
        // for 0x49706550 — a value the ERISC only writes AFTER passing the S8 boot fence.
        // With S8 active, ERISC blocks at the boot fence poll, so the UMD canary never appears.
        // Fix: poll FW_READY (written by ERISC before boot fence), then write S7+S8+S9 to
        // unblock the ERISC, then confirm with a brief UMD canary poll.
        {
            const auto& hal_qs89 = env_impl.get_hal();
            const auto& cluster_qs89 = env_impl.get_cluster();
            const auto& soc_desc_qs89 = cluster_qs89.get_soc_desc(this->id_);
            const auto& router_config_qs89 = builder_ctx.get_fabric_router_config();

            // Compute addresses (mirrors SA init path, lines 650-663).
            const uint32_t handshake_bypass_offset_qs89 = 32;
            [[maybe_unused]] const uint32_t handshake_bypass_l1_addr_qs89 =
                static_cast<uint32_t>(router_config_qs89.handshake_addr) + handshake_bypass_offset_qs89;
            const uint32_t scratch_base_qs89 = static_cast<uint32_t>(
                hal_qs89.get_dev_addr(HalProgrammableCoreType::ACTIVE_ETH, HalL1MemAddrType::FABRIC_TELEMETRY) +
                hal_qs89.get_dev_size(HalProgrammableCoreType::ACTIVE_ETH, HalL1MemAddrType::FABRIC_TELEMETRY) -
                4u /* MEM_AERISC_FABRIC_POSTCODES_SIZE */);
            [[maybe_unused]] const uint32_t fw_ready_addr_qs89 = scratch_base_qs89 + FW_READY_OFFSET;
            [[maybe_unused]] const uint32_t boot_fence_addr_qs89 = scratch_base_qs89 + BOOT_FENCE_OFFSET;
            [[maybe_unused]] const uint32_t session_id_addr_qs89 = scratch_base_qs89 + SESSION_ID_OFFSET;

            [[maybe_unused]] constexpr uint32_t kFwReadyPollMs_qs89 = 5;
            const auto erisc_sync_addr_p0 = builder_ctx.get_fabric_router_sync_address_and_status().first;
            const uint32_t umd_relay_canary_p0 =
                static_cast<uint32_t>(tt::tt_metal::EthDiagSentinel::BASE_UMD_FIRMWARE_SENTINEL);
            constexpr uint32_t terminated_val_p0 = static_cast<uint32_t>(tt::tt_fabric::EDMStatus::TERMINATED);

            for (const auto& lc_qs : deasserted_lcs_inline) {
                uint32_t eth_chan_qs = 0;
                try {
                    eth_chan_qs = soc_desc_qs89.get_eth_channel_for_core(
                        tt::umd::CoreCoord(lc_qs.x, lc_qs.y, CoreType::ETH, CoordSystem::LOGICAL),
                        CoordSystem::LOGICAL);
                } catch (...) {
                    log_warning(
                        tt::LogMetal,
                        "FIX V11-QS89 (#42429): Device {} cannot resolve channel for ETH logical ({},{}) — "
                        "skipping S7/S8/S9",
                        this->id(), lc_qs.x, lc_qs.y);
                    continue;
                }

                auto phys_core_qs = this->virtual_core_from_logical_core(lc_qs, CoreType::ETH);
                tt_cxy_pair core_loc_qs(this->id(), phys_core_qs);

#ifdef STRATEGY8_BOOT_FENCE
                // Step 1: FW_READY poll — wait for ERISC to signal init complete before
                // writing boot fence token.  Mirrors FIX SA-A in configure_fabric().
                uint32_t fw_ready_elapsed_qs = 0;
                bool fw_ready_ok_qs = false;
                while (fw_ready_elapsed_qs < FW_READY_TIMEOUT_MS) {
                    std::vector<uint32_t> rb_qs(1, 0);
                    try {
                        cluster_qs89.read_core(rb_qs, sizeof(uint32_t), core_loc_qs,
                            static_cast<uint64_t>(fw_ready_addr_qs89));
                    } catch (...) {
                        rb_qs[0] = 0;
                    }
                    if (rb_qs[0] == FW_READY_VALUE) {
                        fw_ready_ok_qs = true;
                        break;
                    }
                    std::this_thread::sleep_for(std::chrono::milliseconds(kFwReadyPollMs_qs89));
                    fw_ready_elapsed_qs += kFwReadyPollMs_qs89;
                }
                if (!fw_ready_ok_qs) {
                    // FIX QQ-V (#42429): Read actual fw_ready value at timeout for diagnostics —
                    // mirrors FIX QQ on the FIX SA-A path.  Without this, post-mortem can't
                    // distinguish dormant (0xD0DEAD09) vs stuck-in-ROM (0x00000000) vs PCIe fail.
                    uint32_t fw_ready_actual_qs = 0;
                    try {
                        std::vector<uint32_t> rb_timeout_qs(1, 0);
                        cluster_qs89.read_core(rb_timeout_qs, sizeof(uint32_t), core_loc_qs,
                            static_cast<uint64_t>(fw_ready_addr_qs89));
                        fw_ready_actual_qs = rb_timeout_qs[0];
                    } catch (...) {
                        fw_ready_actual_qs = 0xDEADDEAD;  // PCIe read failed
                    }
                    log_warning(
                        tt::LogMetal,
                        "FIX V11-QS89 (#42429): Device {} chan={} ETH logical ({},{}) ERISC did not "
                        "signal FW_READY within {}ms — actual fw_ready=0x{:08X} (expected 0x{:08X}). "
                        "Marking dead, skipping S7/S8/S9. FIX QQ-V (#42429): timeout snapshot.",
                        this->id(), eth_chan_qs, lc_qs.x, lc_qs.y, FW_READY_TIMEOUT_MS,
                        fw_ready_actual_qs, static_cast<uint32_t>(FW_READY_VALUE));
                    pending_quiesce_newly_dead_eth_chans_.insert(eth_chan_qs);
                    continue;
                }
                log_info(
                    tt::LogMetal,
                    "FIX V11-QS89 (#42429): Device {} chan={} ETH logical ({},{}) FW_READY after {}ms.",
                    this->id(), eth_chan_qs, lc_qs.x, lc_qs.y, fw_ready_elapsed_qs);
#else
                // Without S8, fall back to a brief sleep to let .bss init complete.
                std::this_thread::sleep_for(std::chrono::milliseconds(50));
#endif  // STRATEGY8_BOOT_FENCE

#ifdef STRATEGY7_HANDSHAKE_BYPASS
                // Step 2: FIX V11-QS7 — write handshake_bypass=1.
                // GAP-R10 (#42429): readback verify — mirrors FIX NO pattern on SA path.
                {
                    std::vector<uint32_t> bypass_qs = {1};
                    detail::WriteToDeviceL1(
                        this, CoreCoord(lc_qs.x, lc_qs.y),
                        handshake_bypass_l1_addr_qs89, bypass_qs, CoreType::ETH);
                    std::vector<uint32_t> bypass_rb_qs(1, 0xFFFFFFFF);
                    try {
                        detail::ReadFromDeviceL1(
                            this, CoreCoord(lc_qs.x, lc_qs.y),
                            handshake_bypass_l1_addr_qs89, sizeof(uint32_t), bypass_rb_qs, CoreType::ETH);
                    } catch (...) {
                        bypass_rb_qs[0] = 0xDEADDEAD;
                    }
                    if (bypass_rb_qs[0] != 1) {
                        log_warning(
                            tt::LogMetal,
                            "GAP-R10 (#42429): V11-QS7 handshake_bypass readback MISMATCH — "
                            "wrote 1 to L1[0x{:08X}] Device {} chan={} but read 0x{:08X}",
                            handshake_bypass_l1_addr_qs89, this->id(), eth_chan_qs, bypass_rb_qs[0]);
                    } else {
                        log_info(
                            tt::LogMetal,
                            "FIX V11-QS7 (#42429): handshake_bypass=1 at L1[0x{:08X}] for "
                            "Device {} chan={} (force-reset, inline quiesce path) — readback verified [GAP-R10]",
                            handshake_bypass_l1_addr_qs89, this->id(), eth_chan_qs);
                    }
                }
#endif  // STRATEGY7_HANDSHAKE_BYPASS

#ifdef STRATEGY8_BOOT_FENCE
                // Step 3: FIX V11-QS89 — write BOOT_FENCE_READY to unblock ERISC.
                // GAP-R10 (#42429): readback verify — mirrors FIX S8 on SA path.
                {
                    std::vector<uint32_t> bf_qs = {BOOT_FENCE_READY_VALUE};
                    detail::WriteToDeviceL1(
                        this, CoreCoord(lc_qs.x, lc_qs.y),
                        boot_fence_addr_qs89, bf_qs, CoreType::ETH);
                    std::vector<uint32_t> bf_rb_qs(1, 0);
                    try {
                        detail::ReadFromDeviceL1(
                            this, CoreCoord(lc_qs.x, lc_qs.y),
                            boot_fence_addr_qs89, sizeof(uint32_t), bf_rb_qs, CoreType::ETH);
                    } catch (...) {
                        bf_rb_qs[0] = 0xDEADDEAD;
                    }
                    if (bf_rb_qs[0] != BOOT_FENCE_READY_VALUE) {
                        log_warning(
                            tt::LogMetal,
                            "GAP-R10 (#42429): V11-QS89 boot_fence readback MISMATCH — "
                            "wrote 0x{:08X} to L1[0x{:08X}] Device {} chan={} but read 0x{:08X}",
                            BOOT_FENCE_READY_VALUE, boot_fence_addr_qs89, this->id(), eth_chan_qs, bf_rb_qs[0]);
                    } else {
                        log_info(
                            tt::LogMetal,
                            "FIX V11-QS89 (#42429): boot_fence=0x{:08X} at L1[0x{:08X}] for "
                            "Device {} chan={} (force-reset, inline quiesce path) — readback verified [GAP-R10]",
                            BOOT_FENCE_READY_VALUE, boot_fence_addr_qs89, this->id(), eth_chan_qs);
                    }
                }
#endif  // STRATEGY8_BOOT_FENCE

#ifdef STRATEGY9_SESSION_ID
                // Step 4: FIX V11-QS89 — write session_id.
                // GAP-R10 (#42429): readback verify — mirrors FIX S9 on SA path.
                {
                    const uint32_t sid_qs = this->fabric_session_id_;
                    std::vector<uint32_t> sid_buf_qs = {sid_qs};
                    detail::WriteToDeviceL1(
                        this, CoreCoord(lc_qs.x, lc_qs.y),
                        session_id_addr_qs89, sid_buf_qs, CoreType::ETH);
                    std::vector<uint32_t> sid_rb_qs(1, 0);
                    try {
                        detail::ReadFromDeviceL1(
                            this, CoreCoord(lc_qs.x, lc_qs.y),
                            session_id_addr_qs89, sizeof(uint32_t), sid_rb_qs, CoreType::ETH);
                    } catch (...) {
                        sid_rb_qs[0] = 0xDEADDEAD;
                    }
                    if (sid_rb_qs[0] != sid_qs) {
                        log_warning(
                            tt::LogMetal,
                            "GAP-R10 (#42429): V11-QS89 session_id readback MISMATCH — "
                            "wrote 0x{:08X} to L1[0x{:08X}] Device {} chan={} but read 0x{:08X}",
                            sid_qs, session_id_addr_qs89, this->id(), eth_chan_qs, sid_rb_qs[0]);
                    } else {
                        log_info(
                            tt::LogMetal,
                            "FIX V11-QS89 (#42429): session_id=0x{:08X} at L1[0x{:08X}] for "
                            "Device {} chan={} (force-reset, inline quiesce path) — readback verified [GAP-R10]",
                            sid_qs, session_id_addr_qs89, this->id(), eth_chan_qs);
                    }
                }
#endif  // STRATEGY9_SESSION_ID

                // Step 5: Brief UMD canary poll as readiness confirmation.
                // After boot fence write, ERISC passes S8, enters dispatch loop, writes canary.
                {
                    constexpr uint32_t kCanaryPollMs_qs = 5;
                    constexpr uint32_t kCanaryPollTimeout_qs = 500;
                    uint32_t canary_waited_qs = 0;
                    bool canary_ok_qs = false;
                    while (canary_waited_qs < kCanaryPollTimeout_qs) {
                        std::vector<uint32_t> cpoll(1, 0U);
                        try {
                            detail::ReadFromDeviceL1(this, lc_qs, erisc_sync_addr_p0, 4, cpoll, CoreType::ETH);
                        } catch (...) {
                            cpoll[0] = 0U;
                        }
                        if (cpoll[0] == umd_relay_canary_p0 || cpoll[0] == terminated_val_p0) {
                            canary_ok_qs = true;
                            break;
                        }
                        std::this_thread::sleep_for(std::chrono::milliseconds(kCanaryPollMs_qs));
                        canary_waited_qs += kCanaryPollMs_qs;
                    }
                    if (canary_ok_qs) {
                        log_info(
                            tt::LogMetal,
                            "FIX V11-QS89 (#42429): Device {} chan={} ETH logical ({},{}) UMD canary "
                            "confirmed after {}ms (inline quiesce path).",
                            this->id(), eth_chan_qs, lc_qs.x, lc_qs.y, canary_waited_qs);
                    } else {
                        log_warning(
                            tt::LogMetal,
                            "FIX V11-QS89 (#42429): Device {} chan={} ETH logical ({},{}) UMD canary "
                            "NOT seen after {}ms — marking dead (inline quiesce path).",
                            this->id(), eth_chan_qs, lc_qs.x, lc_qs.y, canary_waited_qs);
                        pending_quiesce_newly_dead_eth_chans_.insert(eth_chan_qs);
                    }
                }
            }
        }
        log_info(
            tt::LogMetal,
            "quiesce_and_restart_fabric_workers: Device {} Pass-0 (FIX AR+AS+V11-QS89) complete — "
            "{} channel(s) deasserted with S7/S8/S9 boot sequence",
            this->id(),
            deasserted_lcs_inline.size());
    }

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

            // GAP-R9 (#42429): Per-channel elapsed timer for Phase 3 ETH relaunch.
            // Without this, post-mortem cannot tell if one channel's launch sequence
            // took 5ms vs 5000ms — critical for diagnosing relay-timeout cascades.
            auto gap_r9_launch_start = std::chrono::steady_clock::now();

            auto* kg = fabric_program_->impl().kernels_on_core(logical_core, pct_idx);
            dev_msgs::launch_msg_t::View msg = kg->launch_msg.view();
            dev_msgs::go_msg_t::ConstView go_msg = kg->go_msg.view();
            msg.kernel_config().host_assigned_id() = fabric_program_->get_runtime_id();

            auto physical_core = this->virtual_core_from_logical_core(logical_core, core_type);

            // Pre-launch status read: confirm ERISC is in TERMINATED (0xA4B4C4D4) or 0x0
            // before we send the launch message.  Any other value means Phase 2.5 didn't
            // terminate the ERISC cleanly and we may be overwriting a live core.
            // FIX-3 (#42429): gate launch on ERISC being in a quiesced state.
            // FIX AR (#42429): after Pass-0 deassert + 50ms boot sleep, force-reset channels
            // will show 0x49706550 (UMD relay firmware canary) — allow that for them.
            {
                const auto [erisc_sync_addr_pre, unused_pre] = builder_ctx.get_fabric_router_sync_address_and_status();
                std::vector<uint32_t> pre_launch_buf(1, 0U);
                try {
                    detail::ReadFromDeviceL1(this, logical_core, erisc_sync_addr_pre, 4, pre_launch_buf, CoreType::ETH);
                } catch (const std::exception& e) {
                    pre_launch_buf[0] = static_cast<uint32_t>(EthDiagSentinel::READ_EXCEPTION);
                    log_warning(
                        tt::LogMetal,
                        "quiesce_and_restart_fabric_workers: Device {} Phase 3 pre-launch status "
                        "read threw for ETH logical ({},{}) — proceeding with 0xDEADBEEF: {}",
                        this->id(),
                        logical_core.x,
                        logical_core.y,
                        e.what());
                } catch (...) {
                    pre_launch_buf[0] = static_cast<uint32_t>(EthDiagSentinel::READ_EXCEPTION);
                    log_warning(
                        tt::LogMetal,
                        "quiesce_and_restart_fabric_workers: Device {} Phase 3 pre-launch status "
                        "read threw unknown exception for ETH logical ({},{}) — proceeding with 0xDEADBEEF",
                        this->id(),
                        logical_core.x,
                        logical_core.y);
                }
                log_info(
                    tt::LogMetal,
                    "quiesce_and_restart_fabric_workers: Device {} Phase 3: "
                    "write_launch_msg_to_core ETH logical ({},{}) pre_status=0x{:08x}",
                    this->id(),
                    logical_core.x,
                    logical_core.y,
                    pre_launch_buf[0]);

                constexpr uint32_t terminated_val = static_cast<uint32_t>(tt::tt_fabric::EDMStatus::TERMINATED);
                const uint32_t umd_relay_canary =
                    static_cast<uint32_t>(tt::tt_metal::EthDiagSentinel::BASE_UMD_FIRMWARE_SENTINEL);
                bool is_force_reset_chan_inline = false;
                if (!pending_phase25_force_reset_chans_.empty()) {
                    try {
                        auto eth_chan_chk = soc_desc_q.get_eth_channel_for_core(
                            tt::umd::CoreCoord(logical_core.x, logical_core.y, CoreType::ETH, CoordSystem::LOGICAL),
                            CoordSystem::LOGICAL);
                        is_force_reset_chan_inline = pending_phase25_force_reset_chans_.count(eth_chan_chk) > 0;
                    } catch (...) {
                        log_debug(
                            tt::LogMetal,
                            "Device {} Phase 3 get_eth_channel_for_core at ETH logical ({},{}) threw non-std exception "
                            "— assuming not force-reset chan",
                            this->id(),
                            logical_core.x,
                            logical_core.y);
                    }
                }
                // FIX AS (#42429): force-reset channels must show UMD canary or TERMINATED —
                // NOT 0x0, which means .bss init hasn't completed and edm_status would get
                // zeroed after we write the launch message (the root cause of the race).
                const bool status_ok_inline = (pre_launch_buf[0] == terminated_val) ||
                                              (!is_force_reset_chan_inline && pre_launch_buf[0] == 0x0) ||
                                              (is_force_reset_chan_inline && pre_launch_buf[0] == umd_relay_canary);
                if (!status_ok_inline) {
                    log_warning(
                        tt::LogMetal,
                        "quiesce_and_restart_fabric_workers: Device {} Phase 3: ETH logical ({},{}) "
                        "pre_status=0x{:08x} ({}) — ERISC not quiesced, skipping "
                        "write_launch_msg_to_core to prevent firmware-init stall. "
                        "Marking channel as dead.  (FIX-3: #42429)",
                        this->id(),
                        logical_core.x,
                        logical_core.y,
                        pre_launch_buf[0],
                        edm_status_str(pre_launch_buf[0]));
                    try {
                        auto eth_chan_dead = soc_desc_q.get_eth_channel_for_core(
                            tt::umd::CoreCoord(logical_core.x, logical_core.y, CoreType::ETH, CoordSystem::LOGICAL),
                            CoordSystem::LOGICAL);
                        pending_quiesce_newly_dead_eth_chans_.insert(eth_chan_dead);
                    } catch (...) {
                        log_debug(
                            tt::LogMetal,
                            "Device {} Phase 3 dead-channel resolution at ETH logical ({},{}) threw non-std exception "
                            "— cannot mark dead",
                            this->id(),
                            logical_core.x,
                            logical_core.y);
                    }
                    continue;
                }
            }

#ifdef STRATEGY7_HANDSHAKE_BYPASS
            // FIX V11-QS7 (#42429): Write handshake_bypass=1 BEFORE launch message for all
            // non-force-reset ETH channels.  ConfigureDeviceWithProgram loaded firmware BSS
            // (zeroing handshake_bypass).  Without this, the fabric kernel starts with
            // bypass=0 and attempts the full ETH handshake on every quiesce cycle.
            // Force-reset channels already got S7 in the Pass-0 block above.
            {
                const auto& router_config_qs7 = builder_ctx.get_fabric_router_config();
                const uint32_t hs_bypass_addr_qs7 =
                    static_cast<uint32_t>(router_config_qs7.handshake_addr) + 32;
                std::vector<uint32_t> bypass_qs7 = {1};
                detail::WriteToDeviceL1(
                    this, logical_core, hs_bypass_addr_qs7, bypass_qs7, CoreType::ETH);
                log_info(
                    tt::LogMetal,
                    "FIX V11-QS7 (#42429): handshake_bypass=1 at L1[0x{:08X}] for "
                    "Device {} ETH logical ({},{}) (inline quiesce Phase 3)",
                    hs_bypass_addr_qs7, this->id(), logical_core.x, logical_core.y);
            }
#endif  // STRATEGY7_HANDSHAKE_BYPASS

            tt::llrt::write_launch_msg_to_core(
                this->id(),
                physical_core,
                msg,
                go_msg,
                /* send_go= */ true);  // FIX SENDGO: was hal.get_dev_addr (uint64_t→bool implicit conversion)

            // GAP-R9 (#42429): Log per-channel elapsed time for Phase 3 ETH relaunch.
            {
                auto gap_r9_launch_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                    std::chrono::steady_clock::now() - gap_r9_launch_start).count();
                log_info(
                    tt::LogMetal,
                    "GAP-R9 (#42429): Device {} Phase 3 ETH launch — logical ({},{}) "
                    "pre_read+gate+launch took {}ms (inline quiesce path)",
                    this->id(),
                    logical_core.x,
                    logical_core.y,
                    gap_r9_launch_ms);
            }

            // FIX IJ quiesce (#42429): DISABLED — now redundant after FIX MM.
            // FIX MM in configure_fabric_cores() unconditionally restores fw_launch_addr
            // for ALL surviving MMIO channels after L1 clear (including quiesce path).
#ifdef FIXIJ_REDUNDANT_AFTER_FIX_MM
            if (this->is_mmio_capable()) {
                static constexpr uint32_t kUnresolvedChan = std::numeric_limits<uint32_t>::max();
                uint32_t hoisted_eth_chan_q = kUnresolvedChan;
                try {
                    hoisted_eth_chan_q = soc_desc_q.get_eth_channel_for_core(
                        tt::umd::CoreCoord(logical_core.x, logical_core.y, CoreType::ETH, CoordSystem::LOGICAL),
                        CoordSystem::LOGICAL);
                } catch (...) {
                    log_warning(
                        tt::LogMetal,
                        "FIX IJ quiesce (#42429): Device {} cannot resolve ETH channel for "
                        "logical ({},{}) — skipping fw_launch_addr restore",
                        this->id(),
                        logical_core.x,
                        logical_core.y);
                }
                if (hoisted_eth_chan_q != kUnresolvedChan) {
                    const auto& hal_ij_q = env_impl.get_hal();
                    const auto aeth_idx_ij_q = hal_ij_q.get_programmable_core_type_index(
                        HalProgrammableCoreType::ACTIVE_ETH);
                    const auto& jit_cfg_ij_q = hal_ij_q.get_jit_build_config(aeth_idx_ij_q, 0, 0);
                    auto& cluster_ij_q = env_impl.get_cluster();
                    auto virtual_core_ij_q =
                        cluster_ij_q.get_virtual_eth_core_from_channel(this->id(), hoisted_eth_chan_q);
                    std::vector<uint32_t> ij_q_pre(1, 0xFFFFFFFF);
                    cluster_ij_q.read_core(ij_q_pre, sizeof(uint32_t),
                        tt_cxy_pair(this->id(), virtual_core_ij_q),
                        static_cast<uint64_t>(jit_cfg_ij_q.fw_launch_addr));
                    log_info(
                        tt::LogMetal,
                        "FIX MN (#42429): FIX IJ quiesce pre-restore — Device {} chan={} "
                        "fw_launch_addr=0x{:08X} pre_val=0x{:08X} (expect 0 from FIX EG)",
                        this->id(), hoisted_eth_chan_q,
                        jit_cfg_ij_q.fw_launch_addr, ij_q_pre[0]);
                    cluster_ij_q.write_core(
                        &jit_cfg_ij_q.fw_launch_addr_value,
                        sizeof(uint32_t),
                        tt_cxy_pair(this->id(), virtual_core_ij_q),
                        jit_cfg_ij_q.fw_launch_addr);
                    std::vector<uint32_t> ij_q_verify(1, 0);
                    cluster_ij_q.read_core(ij_q_verify, sizeof(uint32_t),
                        tt_cxy_pair(this->id(), virtual_core_ij_q),
                        static_cast<uint64_t>(jit_cfg_ij_q.fw_launch_addr));
                    if (ij_q_verify[0] != jit_cfg_ij_q.fw_launch_addr_value) {
                        log_warning(
                            tt::LogMetal,
                            "FIX IJ quiesce (#42429): fw_launch_addr readback MISMATCH — wrote "
                            "0x{:08X} to 0x{:08X} on Device {} chan={} but read back 0x{:08X}. "
                            "Base-UMD may stay at 0xDEADB07E.",
                            jit_cfg_ij_q.fw_launch_addr_value,
                            jit_cfg_ij_q.fw_launch_addr,
                            this->id_,
                            hoisted_eth_chan_q,
                            ij_q_verify[0]);
                    } else {
                        log_info(
                            tt::LogMetal,
                            "FIX IJ quiesce (#42429): restored fw_launch_addr_value=0x{:08X} at "
                            "fw_launch_addr=0x{:08X} for Device {} chan={} after quiesce firmware "
                            "load (readback verified). Base-UMD will now launch fabric firmware.",
                            jit_cfg_ij_q.fw_launch_addr_value,
                            jit_cfg_ij_q.fw_launch_addr,
                            this->id_,
                            hoisted_eth_chan_q);
                    }
                }
            }
#endif  // FIXIJ_REDUNDANT_AFTER_FIX_MM
        }
    }
    // FIX AR (#42429): clear force-reset set after inline ETH launch pass — channels are
    // now either launched or skipped; no further deassert needed.
    pending_phase25_force_reset_chans_.clear();

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
    // GAP-R6 (#42429): Log total elapsed at normal exit.
    {
        auto quiesce_restart_elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::steady_clock::now() - quiesce_restart_start).count();
        log_info(
            tt::LogMetal,
            "quiesce_and_restart_fabric_workers: Device {} SUMMARY — "
            "relay_path_broken={} mmio={} "
            "Phase1=done Phase2=done Phase2.5={} Phase3={} "
            "total_elapsed={}ms "
            "(wait_for_fabric_workers_ready() handles Phase4+Phase5)",
            this->id(),
            fabric_relay_path_broken_.load(),
            this->is_mmio_capable(),
            (fabric_relay_path_broken_.load() && !this->is_mmio_capable()) ? "skipped(relay_broken)" : "done",
            (fabric_relay_path_broken_.load() && !this->is_mmio_capable()) ? "skipped(relay_broken)" : "done",
            quiesce_restart_elapsed);
    }
}

// FIX AE (#42429): Deferred ETH write_launch_msg for quiesce.
//
// This method is the second half of Phase 3 when called with defer_eth_launch=true.
// It sends write_launch_msg_to_core for all ETH cores in the fabric program, using
// the newly_dead_channels set stored during configure_fabric_cores() to skip dead channels.
//
// The mesh-level caller (quiesce_internal) invokes this in order:
//   1. MMIO devices launch first (fast, direct PCIe, ~1ms/channel).
//   2. Non-MMIO devices launch sequentially (slow, via UMD relay, ~200ms/channel).
//
// This ordering ensures MMIO peer ERISCs are running before non-MMIO initiates handshake,
// and creates ~200ms timing asymmetry between successive non-MMIO peer channel starts,
// preventing simultaneous ETH handshake initiation deadlock (STARTED → STARTED deadlock).
void Device::launch_eth_cores_for_quiesce() {
    if (!pending_eth_launch_) {
        log_info(
            tt::LogMetal,
            "launch_eth_cores_for_quiesce: Device {} — no pending ETH launch (Phase 3 was "
            "skipped or defer_eth_launch was not used). No-op.",
            this->id());
        return;
    }
    pending_eth_launch_ = false;

    // Mirror the Phase 3 relay-broken guard: if relay is broken for non-MMIO, skip.
    if (fabric_relay_path_broken_ && !this->is_mmio_capable()) {
        log_warning(
            tt::LogMetal,
            "launch_eth_cores_for_quiesce: Device {} — relay path broken, non-MMIO: "
            "skipping ETH write_launch_msg to prevent indefinite relay hang.",
            this->id());
        pending_quiesce_newly_dead_eth_chans_.clear();
        return;
    }

    if (!fabric_program_) {
        log_warning(
            tt::LogMetal,
            "launch_eth_cores_for_quiesce: Device {} — fabric_program_ is null, cannot launch ETH cores.",
            this->id());
        pending_quiesce_newly_dead_eth_chans_.clear();
        return;
    }

    const auto& control_plane = MetalContext::instance().get_control_plane();
    const auto& fabric_context = control_plane.get_fabric_context();
    const auto& builder_ctx = fabric_context.get_builder_context();
    MetalEnvImpl& env_impl = MetalEnvAccessor(*env_).impl();
    const auto& hal = env_impl.get_hal();
    const auto& soc_desc_q = env_impl.get_cluster().get_soc_desc(this->id());

    std::vector<std::vector<CoreCoord>> logical_cores_used = fabric_program_->impl().logical_cores();
    // Take a local copy of the dead channels set so we can clear the member early.
    const auto newly_dead = std::move(pending_quiesce_newly_dead_eth_chans_);
    pending_quiesce_newly_dead_eth_chans_.clear();
    // FIX AR (#42429): Take local copy of Phase 2.5 force-reset channels for Pass-0 deassert.
    const auto p25_force_reset = std::move(pending_phase25_force_reset_chans_);
    pending_phase25_force_reset_chans_.clear();

    log_info(
        tt::LogMetal,
        "launch_eth_cores_for_quiesce: Device {} — launching ETH ERISC cores "
        "(mmio={}, newly_dead_count={}, p25_force_reset_count={}).",
        this->id(),
        this->is_mmio_capable(),
        newly_dead.size(),
        p25_force_reset.size());

    // FIX AR (#42429): Deassert Phase-2.5 force-reset channels BEFORE writing launch
    // messages, then sleep briefly to let base UMD firmware complete .bss init.
    //
    // Root cause of the "status=0x0 in Phase 5" second-quiesce failure:
    //   Phase 2.5 force-halts stuck ERISCs via assert_risc_reset_at_core(ALL).
    //   FIX AI-2 wrote the launch message first (while ERISC was halted), then
    //   deasserted.  On hardware reset the base UMD firmware re-runs its C-runtime
    //   startup, which zeroes .bss — including the HalL1MemAddrType::LAUNCH mailbox
    //   area — before entering the polling loop.  The launch message is cleared
    //   before the ERISC ever reads it.  The ERISC enters the poll loop, finds
    //   nothing, stays in base firmware, and edm_status remains 0x0 forever.
    //   Phase 5 times out and sets fabric_relay_path_broken_=true.
    //
    // Fix: deassert all force-reset channels in Pass 0, sleep to let .bss init and
    // polling start, then write launch messages in Pass 1.  At that point the ERISC
    // is already in the polling loop (exactly like TERMINATED channels) and picks up
    // the message immediately.
    if (!p25_force_reset.empty()) {
        // Collect the logical cores that were successfully deasserted so we can poll them.
        std::vector<CoreCoord> deasserted_lcs;
        for (uint32_t p0_idx = 0; p0_idx < logical_cores_used.size(); p0_idx++) {
            if (hal.get_core_type(p0_idx) != CoreType::ETH) {
                continue;
            }
            for (const auto& lc0 : logical_cores_used[p0_idx]) {
                try {
                    auto eth_chan_0 = soc_desc_q.get_eth_channel_for_core(
                        tt::umd::CoreCoord(lc0.x, lc0.y, CoreType::ETH, CoordSystem::LOGICAL), CoordSystem::LOGICAL);
                    if (!p25_force_reset.count(eth_chan_0)) {
                        continue;
                    }
                    auto phys_core_0 = this->virtual_core_from_logical_core(lc0, CoreType::ETH);
                    env_impl.get_cluster().deassert_risc_reset_at_core(
                        tt_cxy_pair(this->id(), phys_core_0), tt::umd::RiscType::ALL);
                    log_info(
                        tt::LogMetal,
                        "launch_eth_cores_for_quiesce: Device {} Pass-0 (FIX AR): "
                        "deassert_risc_reset(ALL) for Phase-2.5-halted ETH logical ({},{}) "
                        "channel {} — ERISC now booting into base UMD firmware",
                        this->id(),
                        lc0.x,
                        lc0.y,
                        eth_chan_0);
                    // FIX PD (GAP-50): Mirror of the quiesce_and_restart_fabric_workers() FIX PD.
                    // This path is taken when defer_eth_launch=true (launch_eth_cores_for_quiesce).
                    try {
                        const auto& hal_pd = env_impl.get_hal();
                        const auto aeth_idx =
                            hal_pd.get_programmable_core_type_index(HalProgrammableCoreType::ACTIVE_ETH);
                        const uint32_t fw_launch_addr_pd = hal_pd.get_jit_build_config(aeth_idx, 0, 0).fw_launch_addr;
                        env_impl.get_cluster().write_core_immediate(
                            this->id(), phys_core_0, std::vector<uint32_t>{0}, fw_launch_addr_pd);
                    } catch (const std::exception& ex_pd) {
                        // Best-effort: non-MMIO dead relay may throw.
                        // GAP-B (#42429): Log warning for MMIO devices — PCIe writes should not fail.
                        if (this->is_mmio_capable()) {
                            log_warning(
                                tt::LogMetal,
                                "FIX PD (GAP-B): MMIO device {} fw_launch_addr clear FAILED "
                                "(launch_eth_cores_for_quiesce path, PCIe write should not fail): {}",
                                this->id(),
                                ex_pd.what());
                        }
                    } catch (...) {
                        if (this->is_mmio_capable()) {
                            log_warning(
                                tt::LogMetal,
                                "FIX PD (GAP-B): MMIO device {} fw_launch_addr clear FAILED "
                                "(launch_eth_cores_for_quiesce path, non-std exception)",
                                this->id());
                        }
                    }
                    deasserted_lcs.push_back(lc0);
                } catch (const std::exception& e) {
                    log_warning(
                        tt::LogMetal,
                        "launch_eth_cores_for_quiesce: Device {} Pass-0 (FIX AR): "
                        "deassert_risc_reset(ALL) failed for ETH logical ({},{}): {}",
                        this->id(),
                        lc0.x,
                        lc0.y,
                        e.what());
                } catch (...) {
                    log_warning(
                        tt::LogMetal,
                        "launch_eth_cores_for_quiesce: Device {} Pass-0 (FIX AR): "
                        "deassert_risc_reset(ALL) failed (unknown) for ETH logical ({},{})",
                        this->id(),
                        lc0.x,
                        lc0.y);
                }
            }
        }
        // FIX V11-QS89 (#42429): Per-channel FW_READY poll + S7/S8/S9 boot sequence for
        // deferred force-reset channels.  Mirrors the inline quiesce path replacement above.
        {
            const auto& hal_dqs89 = env_impl.get_hal();
            const auto& cluster_dqs89 = env_impl.get_cluster();
            const auto& soc_desc_dqs89 = cluster_dqs89.get_soc_desc(this->id_);
            const auto& router_config_dqs89 = builder_ctx.get_fabric_router_config();

            const uint32_t handshake_bypass_offset_dqs89 = 32;
            [[maybe_unused]] const uint32_t handshake_bypass_l1_addr_dqs89 =
                static_cast<uint32_t>(router_config_dqs89.handshake_addr) + handshake_bypass_offset_dqs89;
            const uint32_t scratch_base_dqs89 = static_cast<uint32_t>(
                hal_dqs89.get_dev_addr(HalProgrammableCoreType::ACTIVE_ETH, HalL1MemAddrType::FABRIC_TELEMETRY) +
                hal_dqs89.get_dev_size(HalProgrammableCoreType::ACTIVE_ETH, HalL1MemAddrType::FABRIC_TELEMETRY) -
                4u);
            [[maybe_unused]] const uint32_t fw_ready_addr_dqs89 = scratch_base_dqs89 + FW_READY_OFFSET;
            [[maybe_unused]] const uint32_t boot_fence_addr_dqs89 = scratch_base_dqs89 + BOOT_FENCE_OFFSET;
            [[maybe_unused]] const uint32_t session_id_addr_dqs89 = scratch_base_dqs89 + SESSION_ID_OFFSET;

            [[maybe_unused]] constexpr uint32_t kFwReadyPollMs_dqs89 = 5;
            const auto erisc_sync_addr_poll = builder_ctx.get_fabric_router_sync_address_and_status().first;
            const uint32_t umd_relay_canary_poll =
                static_cast<uint32_t>(tt::tt_metal::EthDiagSentinel::BASE_UMD_FIRMWARE_SENTINEL);
            constexpr uint32_t terminated_val_poll = static_cast<uint32_t>(tt::tt_fabric::EDMStatus::TERMINATED);

            for (const auto& lc_dqs : deasserted_lcs) {
                uint32_t eth_chan_dqs = 0;
                try {
                    eth_chan_dqs = soc_desc_dqs89.get_eth_channel_for_core(
                        tt::umd::CoreCoord(lc_dqs.x, lc_dqs.y, CoreType::ETH, CoordSystem::LOGICAL),
                        CoordSystem::LOGICAL);
                } catch (...) {
                    log_warning(
                        tt::LogMetal,
                        "FIX V11-QS89 (#42429): Device {} cannot resolve channel for ETH logical ({},{}) — "
                        "skipping S7/S8/S9 (deferred path)",
                        this->id(), lc_dqs.x, lc_dqs.y);
                    continue;
                }

                auto phys_core_dqs = this->virtual_core_from_logical_core(lc_dqs, CoreType::ETH);
                tt_cxy_pair core_loc_dqs(this->id(), phys_core_dqs);

#ifdef STRATEGY8_BOOT_FENCE
                // Step 1: FW_READY poll.
                uint32_t fw_ready_elapsed_dqs = 0;
                bool fw_ready_ok_dqs = false;
                while (fw_ready_elapsed_dqs < FW_READY_TIMEOUT_MS) {
                    std::vector<uint32_t> rb_dqs(1, 0);
                    try {
                        cluster_dqs89.read_core(rb_dqs, sizeof(uint32_t), core_loc_dqs,
                            static_cast<uint64_t>(fw_ready_addr_dqs89));
                    } catch (...) {
                        rb_dqs[0] = 0;
                    }
                    if (rb_dqs[0] == FW_READY_VALUE) {
                        fw_ready_ok_dqs = true;
                        break;
                    }
                    std::this_thread::sleep_for(std::chrono::milliseconds(kFwReadyPollMs_dqs89));
                    fw_ready_elapsed_dqs += kFwReadyPollMs_dqs89;
                }
                if (!fw_ready_ok_dqs) {
                    // FIX QQ-V (#42429): Read actual fw_ready value at timeout (deferred path).
                    uint32_t fw_ready_actual_dqs = 0;
                    try {
                        std::vector<uint32_t> rb_timeout_dqs(1, 0);
                        cluster_dqs89.read_core(rb_timeout_dqs, sizeof(uint32_t), core_loc_dqs,
                            static_cast<uint64_t>(fw_ready_addr_dqs89));
                        fw_ready_actual_dqs = rb_timeout_dqs[0];
                    } catch (...) {
                        fw_ready_actual_dqs = 0xDEADDEAD;  // PCIe read failed
                    }
                    log_warning(
                        tt::LogMetal,
                        "FIX V11-QS89 (#42429): Device {} chan={} ETH logical ({},{}) ERISC did not "
                        "signal FW_READY within {}ms — actual fw_ready=0x{:08X} (expected 0x{:08X}). "
                        "Marking dead (deferred path). FIX QQ-V (#42429): timeout snapshot.",
                        this->id(), eth_chan_dqs, lc_dqs.x, lc_dqs.y, FW_READY_TIMEOUT_MS,
                        fw_ready_actual_dqs, static_cast<uint32_t>(FW_READY_VALUE));
                    pending_quiesce_newly_dead_eth_chans_.insert(eth_chan_dqs);
                    continue;
                }
                log_info(
                    tt::LogMetal,
                    "FIX V11-QS89 (#42429): Device {} chan={} ETH logical ({},{}) FW_READY after {}ms "
                    "(deferred path).",
                    this->id(), eth_chan_dqs, lc_dqs.x, lc_dqs.y, fw_ready_elapsed_dqs);
#else
                std::this_thread::sleep_for(std::chrono::milliseconds(50));
#endif  // STRATEGY8_BOOT_FENCE

#ifdef STRATEGY7_HANDSHAKE_BYPASS
                // Step 2: FIX V11-QS7 — write handshake_bypass=1.
                {
                    std::vector<uint32_t> bypass_dqs = {1};
                    detail::WriteToDeviceL1(
                        this, CoreCoord(lc_dqs.x, lc_dqs.y),
                        handshake_bypass_l1_addr_dqs89, bypass_dqs, CoreType::ETH);
                    log_info(
                        tt::LogMetal,
                        "FIX V11-QS7 (#42429): handshake_bypass=1 at L1[0x{:08X}] for "
                        "Device {} chan={} (force-reset, deferred quiesce path)",
                        handshake_bypass_l1_addr_dqs89, this->id(), eth_chan_dqs);
                }
#endif  // STRATEGY7_HANDSHAKE_BYPASS

#ifdef STRATEGY8_BOOT_FENCE
                // Step 3: Write BOOT_FENCE_READY.
                {
                    std::vector<uint32_t> bf_dqs = {BOOT_FENCE_READY_VALUE};
                    detail::WriteToDeviceL1(
                        this, CoreCoord(lc_dqs.x, lc_dqs.y),
                        boot_fence_addr_dqs89, bf_dqs, CoreType::ETH);
                    log_info(
                        tt::LogMetal,
                        "FIX V11-QS89 (#42429): boot_fence=0x{:08X} at L1[0x{:08X}] for "
                        "Device {} chan={} (force-reset, deferred quiesce path)",
                        BOOT_FENCE_READY_VALUE, boot_fence_addr_dqs89, this->id(), eth_chan_dqs);
                }
#endif  // STRATEGY8_BOOT_FENCE

#ifdef STRATEGY9_SESSION_ID
                // Step 4: Write session_id.
                {
                    const uint32_t sid_dqs = this->fabric_session_id_;
                    std::vector<uint32_t> sid_buf_dqs = {sid_dqs};
                    detail::WriteToDeviceL1(
                        this, CoreCoord(lc_dqs.x, lc_dqs.y),
                        session_id_addr_dqs89, sid_buf_dqs, CoreType::ETH);
                    log_info(
                        tt::LogMetal,
                        "FIX V11-QS89 (#42429): session_id=0x{:08X} at L1[0x{:08X}] for "
                        "Device {} chan={} (force-reset, deferred quiesce path)",
                        sid_dqs, session_id_addr_dqs89, this->id(), eth_chan_dqs);
                }
#endif  // STRATEGY9_SESSION_ID

                // Step 5: Brief UMD canary poll as readiness confirmation.
                {
                    constexpr uint32_t kCanaryPollMs_dqs = 5;
                    constexpr uint32_t kCanaryPollTimeout_dqs = 500;
                    uint32_t canary_waited_dqs = 0;
                    bool canary_ok_dqs = false;
                    while (canary_waited_dqs < kCanaryPollTimeout_dqs) {
                        std::vector<uint32_t> cpoll_d(1, 0U);
                        try {
                            detail::ReadFromDeviceL1(this, lc_dqs, erisc_sync_addr_poll, 4, cpoll_d, CoreType::ETH);
                        } catch (...) {
                            cpoll_d[0] = 0U;
                        }
                        if (cpoll_d[0] == umd_relay_canary_poll || cpoll_d[0] == terminated_val_poll) {
                            canary_ok_dqs = true;
                            break;
                        }
                        std::this_thread::sleep_for(std::chrono::milliseconds(kCanaryPollMs_dqs));
                        canary_waited_dqs += kCanaryPollMs_dqs;
                    }
                    if (canary_ok_dqs) {
                        log_info(
                            tt::LogMetal,
                            "FIX V11-QS89 (#42429): Device {} chan={} ETH logical ({},{}) UMD canary "
                            "confirmed after {}ms (deferred quiesce path).",
                            this->id(), eth_chan_dqs, lc_dqs.x, lc_dqs.y, canary_waited_dqs);
                    } else {
                        log_warning(
                            tt::LogMetal,
                            "FIX V11-QS89 (#42429): Device {} chan={} ETH logical ({},{}) UMD canary "
                            "NOT seen after {}ms — marking dead (deferred quiesce path).",
                            this->id(), eth_chan_dqs, lc_dqs.x, lc_dqs.y, canary_waited_dqs);
                        pending_quiesce_newly_dead_eth_chans_.insert(eth_chan_dqs);
                    }
                }
            }
        }

        log_info(
            tt::LogMetal,
            "launch_eth_cores_for_quiesce: Device {} Pass-0 (FIX AR+AS+V11-QS89) complete — "
            "{} force-reset channel(s) deasserted with S7/S8/S9 boot sequence",
            this->id(),
            p25_force_reset.size());
    }

    for (uint32_t pct_idx = 0; pct_idx < logical_cores_used.size(); pct_idx++) {
        CoreType core_type = hal.get_core_type(pct_idx);
        if (core_type != CoreType::ETH) {
            continue;
        }
        for (const auto& logical_core : logical_cores_used[pct_idx]) {
            // Skip channels that died during configure_fabric_cores() — writing launch
            // messages through a dead ETH relay hangs indefinitely.
            if (!newly_dead.empty()) {
                try {
                    auto eth_chan = soc_desc_q.get_eth_channel_for_core(
                        tt::umd::CoreCoord(logical_core.x, logical_core.y, CoreType::ETH, CoordSystem::LOGICAL),
                        CoordSystem::LOGICAL);
                    if (newly_dead.count(eth_chan)) {
                        log_warning(
                            tt::LogMetal,
                            "launch_eth_cores_for_quiesce: Device {} skipping "
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
                        "launch_eth_cores_for_quiesce: Device {} cannot resolve ETH channel "
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

            // Pre-launch status read: confirm ERISC is in TERMINATED (0xA4B4C4D4) or 0x0.
            // FIX-3 (#42429): If the ERISC is live (relay firmware or any non-quiesced state),
            // sending write_launch_msg_to_core while BRISC is still executing the old relay
            // firmware's shutdown path stalls fabric ERISC init (never writes EDMStatus::STARTED).
            // Skip the channel and mark it newly-dead so Phase 5 / subsequent quiesce do not relay
            // through it.
            {
                const auto [erisc_sync_addr_pre, unused_pre] = builder_ctx.get_fabric_router_sync_address_and_status();
                std::vector<uint32_t> pre_launch_buf(1, 0U);
                try {
                    detail::ReadFromDeviceL1(this, logical_core, erisc_sync_addr_pre, 4, pre_launch_buf, CoreType::ETH);
                } catch (const std::exception& e) {
                    pre_launch_buf[0] = static_cast<uint32_t>(EthDiagSentinel::READ_EXCEPTION);
                    log_warning(
                        tt::LogMetal,
                        "launch_eth_cores_for_quiesce: Device {} Phase 3 pre-launch status "
                        "read threw for ETH logical ({},{}) — proceeding with 0xDEADBEEF: {}",
                        this->id(),
                        logical_core.x,
                        logical_core.y,
                        e.what());
                } catch (...) {
                    pre_launch_buf[0] = static_cast<uint32_t>(EthDiagSentinel::READ_EXCEPTION);
                    log_warning(
                        tt::LogMetal,
                        "launch_eth_cores_for_quiesce: Device {} Phase 3 pre-launch status "
                        "read threw unknown for ETH logical ({},{})",
                        this->id(),
                        logical_core.x,
                        logical_core.y);
                }
                log_info(
                    tt::LogMetal,
                    "launch_eth_cores_for_quiesce: Device {} Phase 3: "
                    "write_launch_msg_to_core ETH logical ({},{}) pre_status=0x{:08x}",
                    this->id(),
                    logical_core.x,
                    logical_core.y,
                    pre_launch_buf[0]);

                // FIX-3: gate launch on ERISC being in a quiesced state.
                constexpr uint32_t terminated_val = static_cast<uint32_t>(tt::tt_fabric::EDMStatus::TERMINATED);
                // FIX AR (#42429): after Pass-0 deassert + 50ms boot sleep, force-reset channels
                // will have completed base UMD .bss init and show 0x49706550 (UMD relay firmware
                // canary) at edm_status_address.  This is an expected quiesced state for those
                // channels — allow it through so we don't incorrectly mark them dead.
                const uint32_t umd_relay_canary =
                    static_cast<uint32_t>(tt::tt_metal::EthDiagSentinel::BASE_UMD_FIRMWARE_SENTINEL);
                bool is_force_reset_chan = false;
                if (!p25_force_reset.empty()) {
                    try {
                        auto eth_chan_check = soc_desc_q.get_eth_channel_for_core(
                            tt::umd::CoreCoord(logical_core.x, logical_core.y, CoreType::ETH, CoordSystem::LOGICAL),
                            CoordSystem::LOGICAL);
                        is_force_reset_chan = p25_force_reset.count(eth_chan_check) > 0;
                    } catch (...) {
                        log_debug(
                            tt::LogMetal,
                            "Device {} Phase 5 get_eth_channel_for_core at ETH logical ({},{}) threw non-std exception "
                            "— assuming not force-reset chan",
                            this->id(),
                            logical_core.x,
                            logical_core.y);
                    }
                }
                // FIX AS (#42429): force-reset channels must show UMD canary or TERMINATED —
                // NOT 0x0, which means .bss init hasn't completed and edm_status would be
                // zeroed after the launch message write (root cause of the race condition).
                const bool status_ok = (pre_launch_buf[0] == terminated_val) ||
                                       (!is_force_reset_chan && pre_launch_buf[0] == 0x0) ||
                                       (is_force_reset_chan && pre_launch_buf[0] == umd_relay_canary);
                if (!status_ok) {
                    log_warning(
                        tt::LogMetal,
                        "launch_eth_cores_for_quiesce: Device {} Phase 3: ETH logical ({},{}) "
                        "pre_status=0x{:08x} ({}) — ERISC not quiesced, skipping "
                        "write_launch_msg_to_core to prevent firmware-init stall. "
                        "Marking channel as dead.  (FIX-3: #42429)",
                        this->id(),
                        logical_core.x,
                        logical_core.y,
                        pre_launch_buf[0],
                        edm_status_str(pre_launch_buf[0]));
                    // Resolve the channel number and mark it dead so Phase 5 / subsequent
                    // quiesce relay ops skip it.
                    try {
                        auto eth_chan = soc_desc_q.get_eth_channel_for_core(
                            tt::umd::CoreCoord(logical_core.x, logical_core.y, CoreType::ETH, CoordSystem::LOGICAL),
                            CoordSystem::LOGICAL);
                        pending_quiesce_newly_dead_eth_chans_.insert(eth_chan);
                    } catch (...) {
                        log_debug(
                            tt::LogMetal,
                            "Device {} Phase 5 dead-channel resolution at ETH logical ({},{}) threw non-std exception "
                            "— cannot mark dead",
                            this->id(),
                            logical_core.x,
                            logical_core.y);
                    }
                    continue;
                }
            }

#ifdef STRATEGY7_HANDSHAKE_BYPASS
            // FIX V11-QS7 (#42429): Write handshake_bypass=1 BEFORE launch message for all
            // non-force-reset ETH channels.  Force-reset channels got S7 in Pass-0 above.
            {
                const auto& router_config_dqs7 = builder_ctx.get_fabric_router_config();
                const uint32_t hs_bypass_addr_dqs7 =
                    static_cast<uint32_t>(router_config_dqs7.handshake_addr) + 32;
                std::vector<uint32_t> bypass_dqs7 = {1};
                detail::WriteToDeviceL1(
                    this, logical_core, hs_bypass_addr_dqs7, bypass_dqs7, CoreType::ETH);
                log_info(
                    tt::LogMetal,
                    "FIX V11-QS7 (#42429): handshake_bypass=1 at L1[0x{:08X}] for "
                    "Device {} ETH logical ({},{}) (deferred quiesce Phase 3)",
                    hs_bypass_addr_dqs7, this->id(), logical_core.x, logical_core.y);
            }
#endif  // STRATEGY7_HANDSHAKE_BYPASS

            tt::llrt::write_launch_msg_to_core(
                this->id(),
                physical_core,
                msg,
                go_msg,
                /* send_go= */ true);  // FIX SENDGO: was hal.get_dev_addr (uint64_t→bool implicit conversion)

            log_info(
                tt::LogMetal,
                "launch_eth_cores_for_quiesce: Device {} Phase 3: "
                "write_launch_msg_to_core ETH logical ({},{}) done",
                this->id(),
                logical_core.x,
                logical_core.y);

            // FIX IJ deferred-quiesce (#42429): DISABLED — now redundant after FIX MM.
            // FIX MM in configure_fabric_cores() unconditionally restores fw_launch_addr
            // for ALL surviving MMIO channels after L1 clear (including deferred-quiesce path).
#ifdef FIXIJ_REDUNDANT_AFTER_FIX_MM
            if (this->is_mmio_capable()) {
                static constexpr uint32_t kUnresolvedChan = std::numeric_limits<uint32_t>::max();
                uint32_t hoisted_eth_chan_dq = kUnresolvedChan;
                try {
                    hoisted_eth_chan_dq = soc_desc_q.get_eth_channel_for_core(
                        tt::umd::CoreCoord(logical_core.x, logical_core.y, CoreType::ETH, CoordSystem::LOGICAL),
                        CoordSystem::LOGICAL);
                } catch (...) {
                    log_warning(
                        tt::LogMetal,
                        "FIX IJ deferred-quiesce (#42429): Device {} cannot resolve ETH channel for "
                        "logical ({},{}) — skipping fw_launch_addr restore",
                        this->id(),
                        logical_core.x,
                        logical_core.y);
                }
                if (hoisted_eth_chan_dq != kUnresolvedChan) {
                    const auto& hal_ij_dq = env_impl.get_hal();
                    const auto aeth_idx_ij_dq = hal_ij_dq.get_programmable_core_type_index(
                        HalProgrammableCoreType::ACTIVE_ETH);
                    const auto& jit_cfg_ij_dq = hal_ij_dq.get_jit_build_config(aeth_idx_ij_dq, 0, 0);
                    auto& cluster_ij_dq = env_impl.get_cluster();
                    auto virtual_core_ij_dq =
                        cluster_ij_dq.get_virtual_eth_core_from_channel(this->id(), hoisted_eth_chan_dq);
                    std::vector<uint32_t> ij_dq_pre(1, 0xFFFFFFFF);
                    cluster_ij_dq.read_core(ij_dq_pre, sizeof(uint32_t),
                        tt_cxy_pair(this->id(), virtual_core_ij_dq),
                        static_cast<uint64_t>(jit_cfg_ij_dq.fw_launch_addr));
                    log_info(
                        tt::LogMetal,
                        "FIX MN (#42429): FIX IJ deferred-quiesce pre-restore — Device {} chan={} "
                        "fw_launch_addr=0x{:08X} pre_val=0x{:08X} (expect 0 from FIX EG)",
                        this->id(), hoisted_eth_chan_dq,
                        jit_cfg_ij_dq.fw_launch_addr, ij_dq_pre[0]);
                    cluster_ij_dq.write_core(
                        &jit_cfg_ij_dq.fw_launch_addr_value,
                        sizeof(uint32_t),
                        tt_cxy_pair(this->id(), virtual_core_ij_dq),
                        jit_cfg_ij_dq.fw_launch_addr);
                    std::vector<uint32_t> ij_dq_verify(1, 0);
                    cluster_ij_dq.read_core(ij_dq_verify, sizeof(uint32_t),
                        tt_cxy_pair(this->id(), virtual_core_ij_dq),
                        static_cast<uint64_t>(jit_cfg_ij_dq.fw_launch_addr));
                    if (ij_dq_verify[0] != jit_cfg_ij_dq.fw_launch_addr_value) {
                        log_warning(
                            tt::LogMetal,
                            "FIX IJ deferred-quiesce (#42429): fw_launch_addr readback MISMATCH — wrote "
                            "0x{:08X} to 0x{:08X} on Device {} chan={} but read back 0x{:08X}. "
                            "Base-UMD may stay at 0xDEADB07E.",
                            jit_cfg_ij_dq.fw_launch_addr_value,
                            jit_cfg_ij_dq.fw_launch_addr,
                            this->id_,
                            hoisted_eth_chan_dq,
                            ij_dq_verify[0]);
                    } else {
                        log_info(
                            tt::LogMetal,
                            "FIX IJ deferred-quiesce (#42429): restored fw_launch_addr_value=0x{:08X} at "
                            "fw_launch_addr=0x{:08X} for Device {} chan={} after deferred quiesce firmware "
                            "load (readback verified). Base-UMD will now launch fabric firmware.",
                            jit_cfg_ij_dq.fw_launch_addr_value,
                            jit_cfg_ij_dq.fw_launch_addr,
                            this->id_,
                            hoisted_eth_chan_dq);
                    }
                }
            }
#endif  // FIXIJ_REDUNDANT_AFTER_FIX_MM
        }
    }

    // FIX QR (#42429): FIX EF quiesce analogue — poll MMIO relay ERISCs after FIX IJ
    // restore to confirm they exited 0xDEADB07E before returning.
    //
    // Race window: quiesce_internal() Pass 1b calls launch_eth_cores_for_quiesce() on
    // MMIO devices, then Pass 1c immediately launches non-MMIO devices.  Non-MMIO
    // write_launch_msg_to_core routes through the MMIO relay ERISCs.  If an MMIO relay
    // ERISC is still at 0xDEADB07E when Pass 1c starts, the non-MMIO launch message
    // is silently dropped → non-MMIO ERISC never starts → ring-sync timeout.
    //
    // This is the quiesce analogue of the initial configure_fabric() FIX EF poll.
    // The poll is only needed for MMIO devices (relay ERISCs that serve non-MMIO peers).
    if (this->is_mmio_capable()) {
        static constexpr uint32_t kPreLaunchCanary_QR =
            static_cast<uint32_t>(EthDiagSentinel::HOST_PRE_LAUNCH_CANARY);
        static constexpr int kFIX_QR_PollIntervalMs = 5;
        static constexpr int kFIX_QR_PollMaxMs = 3000;
        const auto [qr_sync_addr, unused_qr] = builder_ctx.get_fabric_router_sync_address_and_status();
        const auto& env_qr = MetalContext::instance();
        auto& cluster_qr = env_qr.get_cluster();
        const auto fabric_node_qr = control_plane.get_fabric_node_id_from_physical_chip_id(this->id());
        const auto router_chans_qr = control_plane.get_active_fabric_eth_channels(fabric_node_qr);

        for (const auto& [qr_chan, _] : router_chans_qr) {
            if (newly_dead.count(qr_chan)) continue;
            auto qr_virtual = cluster_qr.get_virtual_eth_core_from_channel(this->id(), qr_chan);
            const int max_iters = kFIX_QR_PollMaxMs / kFIX_QR_PollIntervalMs;
            bool exited = false;
            for (int iter = 0; iter < max_iters; ++iter) {
                std::vector<uint32_t> qr_buf(1, kPreLaunchCanary_QR);
                try {
                    cluster_qr.read_core(qr_buf, sizeof(uint32_t),
                        tt_cxy_pair(this->id(), qr_virtual),
                        static_cast<uint64_t>(qr_sync_addr));
                } catch (const std::exception& e) {
                    log_warning(tt::LogMetal,
                        "FIX QR: Device {} chan {} quiesce poll read failed: {}",
                        this->id(), qr_chan, e.what());
                    break;
                }
                if (qr_buf[0] != kPreLaunchCanary_QR) {
                    log_info(tt::LogMetal,
                        "FIX QR: Device {} chan {} (MMIO) exited 0xDEADB07E → 0x{:08x} "
                        "after {}ms in quiesce — relay ERISC ready for Pass 1c. (#42429)",
                        this->id(), qr_chan, qr_buf[0],
                        iter * kFIX_QR_PollIntervalMs);
                    exited = true;
                    break;
                }
                std::this_thread::sleep_for(std::chrono::milliseconds(kFIX_QR_PollIntervalMs));
            }
            if (!exited) {
                // GAP 6 (#42429): Escalate FIX QR timeout to set fabric_relay_path_broken_,
                // matching FIX EF behaviour.  Without this, non-MMIO Pass 1c launch messages
                // are silently dropped through the dead relay with no recovery path — the
                // non-MMIO ERISC never starts and ring-sync times out 120s later.
                // With the flag set, FIX SB2 propagates the broken state to non-MMIO devices,
                // which then skip their relay reads and avoid the 120s hang.

                // FIX NP (#42429): Read actual edm_status at timeout for accurate diagnostics.
                // Mirrors FIX NP in FIX EF path — the poll assumes 0xDEADB07E but ERISC may
                // have transitioned to a different bad state.
                uint32_t actual_qr_status = kPreLaunchCanary_QR;  // default assumption
                try {
                    std::vector<uint32_t> timeout_qr_buf(1, kPreLaunchCanary_QR);
                    cluster_qr.read_core(timeout_qr_buf, sizeof(uint32_t),
                        tt_cxy_pair(this->id(), qr_virtual),
                        static_cast<uint64_t>(qr_sync_addr));
                    actual_qr_status = timeout_qr_buf[0];
                } catch (...) {
                    // Best-effort read — may fail if relay is truly dead.
                }
                fabric_relay_path_broken_.store(true);
                log_warning(tt::LogMetal,
                    "FIX QR: Device {} chan {} (MMIO) did not exit canary after {}ms in quiesce — "
                    "actual edm_status=0x{:08x} (expected 0xDEADB07E). "
                    "Setting fabric_relay_path_broken_=true to block "
                    "non-MMIO Pass 1c launch through dead relay. "
                    "FIX NP (#42429): timeout snapshot.",
                    this->id(), qr_chan, kFIX_QR_PollMaxMs, actual_qr_status);
            }
        }
    }

    log_info(
        tt::LogMetal, "launch_eth_cores_for_quiesce: Device {} complete — all ETH ERISC cores launched.", this->id());
}

// FIX AF (#42429): Poll this device's ETH channels until all show a non-zero
// edm_status (STARTED or beyond), or until timeout_ms elapses.  Callers in
// mesh_device.cpp's Pass 1c use this to guarantee that SENDER ERISCs on the
// just-launched non-MMIO device have written EDMStatus::STARTED — meaning they
// have exited early firmware init and are in (or entering) the handshake loop —
// before the next non-MMIO device is launched.  Without this barrier, both
// non-MMIO devices can enter their handshake simultaneously, causing the
// SENDER↔SENDER deadlock at STARTED that FIX AE attempted (but failed) to fix.
void Device::wait_for_eth_cores_launched(uint32_t timeout_ms) {
    // Skip for MMIO devices (their ETH channels are directly PCIe-accessible and
    // always fast; no ordering concern with non-MMIO peers in Pass 1c).
    // Skip for non-MMIO devices with a broken relay path (launch was skipped too).
    if (this->is_mmio_capable() || fabric_relay_path_broken_.load()) {
        log_info(
            tt::LogMetal,
            "wait_for_eth_cores_launched: Device {} — skipping (mmio={}, relay_broken={}).",
            this->id(),
            this->is_mmio_capable(),
            static_cast<bool>(fabric_relay_path_broken_));
        return;
    }
    if (!fabric_program_) {
        log_warning(
            tt::LogMetal,
            "wait_for_eth_cores_launched: Device {} — fabric_program_ is null, nothing to poll.",
            this->id());
        return;
    }

    const auto& control_plane = MetalContext::instance().get_control_plane();
    const auto& fabric_context = control_plane.get_fabric_context();
    const auto& builder_ctx = fabric_context.get_builder_context();
    MetalEnvImpl& env_impl = MetalEnvAccessor(*env_).impl();
    const auto& hal = env_impl.get_hal();

    const auto [erisc_sync_addr, unused_expected] = builder_ctx.get_fabric_router_sync_address_and_status();

    // Collect all ETH logical cores from the fabric program.
    struct ChanInfo {
        CoreCoord logical_core;
    };
    std::vector<ChanInfo> eth_cores;
    std::vector<std::vector<CoreCoord>> logical_cores_used = fabric_program_->impl().logical_cores();
    for (uint32_t pct_idx = 0; pct_idx < logical_cores_used.size(); pct_idx++) {
        if (hal.get_core_type(pct_idx) != CoreType::ETH) {
            continue;
        }
        for (const auto& lc : logical_cores_used[pct_idx]) {
            eth_cores.push_back({lc});
        }
    }
    if (eth_cores.empty()) {
        log_info(
            tt::LogMetal, "wait_for_eth_cores_launched: Device {} — no ETH cores in fabric program, done.", this->id());
        return;
    }

    log_info(
        tt::LogMetal,
        "wait_for_eth_cores_launched: Device {} — polling {} ETH channel(s) for STARTED "
        "(timeout={}ms, edm_status_addr=0x{:08x}).",
        this->id(),
        eth_cores.size(),
        timeout_ms,
        erisc_sync_addr);

    // pending[i] = index into eth_cores that hasn't shown non-zero status yet.
    std::vector<size_t> pending;
    pending.reserve(eth_cores.size());
    for (size_t i = 0; i < eth_cores.size(); i++) {
        pending.push_back(i);
    }

    const auto start = std::chrono::steady_clock::now();
    int64_t last_log_ms = -1;
    constexpr uint32_t kLogIntervalMs = 100;
    constexpr uint32_t kSpinsBetweenSleeps = 64;
    uint32_t launch_spin = 0U;

    while (!pending.empty()) {
        const auto elapsed_ms =
            std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - start).count();
        if (elapsed_ms > static_cast<int64_t>(timeout_ms)) {
            log_warning(
                tt::LogMetal,
                "wait_for_eth_cores_launched: Device {} — timeout ({}ms) reached with {} "
                "channel(s) still at 0x0 (not yet STARTED). Proceeding anyway.",
                this->id(),
                elapsed_ms,
                pending.size());
            break;
        }

        std::vector<size_t> still_pending;
        for (size_t idx : pending) {
            std::vector<uint32_t> buf(1, 0U);
            try {
                detail::ReadFromDeviceL1(this, eth_cores[idx].logical_core, erisc_sync_addr, 4, buf, CoreType::ETH);
            } catch (const std::exception& e) {
                log_warning(
                    tt::LogMetal,
                    "wait_for_eth_cores_launched: Device {} ETH logical ({},{}) read threw: {}",
                    this->id(),
                    eth_cores[idx].logical_core.x,
                    eth_cores[idx].logical_core.y,
                    e.what());
                // Treat read failure as "not yet started" and move on.
                still_pending.push_back(idx);
                continue;
            }
            if (buf[0] != 0U) {
                log_info(
                    tt::LogMetal,
                    "wait_for_eth_cores_launched: Device {} ETH logical ({},{}) reached "
                    "edm_status=0x{:08x} ({}) after {}ms.",
                    this->id(),
                    eth_cores[idx].logical_core.x,
                    eth_cores[idx].logical_core.y,
                    buf[0],
                    edm_status_str(buf[0]),
                    elapsed_ms);
            } else {
                still_pending.push_back(idx);
            }
        }
        pending = std::move(still_pending);

        if (!pending.empty()) {
            const auto now_ms =
                std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - start).count();
            if (now_ms / kLogIntervalMs > last_log_ms / static_cast<int64_t>(kLogIntervalMs)) {
                last_log_ms = now_ms;
                log_info(
                    tt::LogMetal,
                    "wait_for_eth_cores_launched: Device {} — {}ms elapsed, "
                    "{}/{} channel(s) still at 0x0 (not yet STARTED).",
                    this->id(),
                    now_ms,
                    pending.size(),
                    eth_cores.size());
            }
            if (++launch_spin >= kSpinsBetweenSleeps) {
                launch_spin = 0U;
                std::this_thread::sleep_for(std::chrono::microseconds(100));
            } else {
                ttsl::pause();
            }
        }
    }

    const auto total_ms =
        std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - start).count();
    log_info(
        tt::LogMetal,
        "wait_for_eth_cores_launched: Device {} — done in {}ms ({}/{} channels confirmed STARTED).",
        this->id(),
        total_ms,
        eth_cores.size() - pending.size(),
        eth_cores.size());
}

bool Device::phase5b_erisc_health_check(
    const std::set<std::pair<tt::tt_fabric::chan_id_t, tt::tt_fabric::eth_chan_directions>>& active_channels,
    const metal_SocDescriptor& soc_desc_p5,
    uint32_t router_sync_addr,
    uint32_t expected_ready) {
    // FIX DO (#42429): When FIX M skipped soft-reset on base-UMD relay channels and called
    // write_launch_msg_to_core, all relay channels die simultaneously.  Each subsequent L1 read
    // via a dead relay hangs 1-5s.  With 6 channels and 2000ms budget the deadline is exhausted
    // before all channels are checked → Phase 5b deadline exceeded → dispatch teardown failure.
    // Extend to 30000ms (30s) so even 6 hanging reads at 5s each are covered.  The per-read
    // deadline guard at line 2554 still prevents accumulating unlimited time per remaining channel.
    //
    // FIX DQ (#42429): When stale base-UMD channels are present (fabric_stale_base_umd_channels_
    // is true), FIX BO already extends the Phase 5 ring-sync deadline from 10000ms to 120000ms.
    // Phase 5b needs the same treatment: base-UMD channel transition via launch_msg can take >30s
    // (confirmed in cycle 16: Phase 5b deadline exceeded at 30001ms on devices 1/9, 5/6, 3/15),
    // triggering global teardown → FIX BH cascade → all 24 MMIO channels dead.
    // Extend to 120000ms when stale base-UMD channels present (matching the 12× FIX BO pattern).
    const uint32_t kHealthCheckTimeoutMs = this->is_fabric_stale_base_umd_channels() ? 120000 : 30000;
    // Log unhealthy channels every 200ms for observability.
    constexpr uint32_t kHCIntermediateLogMs = 200;
    constexpr uint32_t kSpinLimit = 64U;

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
        // FIX EXT (#42429): skip external channels — firmware not loaded, expected at 0x49706550.
        // Treat them as pre-dead (non-participating) rather than truly_unhealthy.
        if (fabric_external_umd_channels_.count(eth_chan_id) > 0) {
            log_info(
                tt::LogMetal,
                "phase5b_erisc_health_check: Device {} chan={} is external ETH (no in-cluster "
                "peer, firmware not loaded) — skipping health check. (FIX EXT #42429)",
                this->id(),
                eth_chan_id);
            continue;
        }
        const auto eth_logical_core = soc_desc_p5.get_eth_core_for_channel(eth_chan_id, CoordSystem::LOGICAL);
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
            // hang is the phase5_relay_read_threw guard in the caller (Phase 5b is
            // skipped entirely when Phase 5's own read threw).
            const auto pre_read_elapsed =
                std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - hc_start)
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
                // 0xDEAD5B5B: Phase 5b per-iteration deadline exceeded — read was skipped.
                still_pending_statuses.push_back(
                    {ch.eth_chan_id, static_cast<uint32_t>(EthDiagSentinel::PHASE5B_DEADLINE_SKIPPED)});
                continue;
            }
            chans_checked++;

            try {
                detail::ReadFromDeviceL1(this, ch.eth_logical_core, router_sync_addr, 4, status_buf, CoreType::ETH);
            } catch (const std::exception& e) {
                // Non-MMIO relay read timed out — treat this channel as not-ready.
                // Without this catch, the exception propagates uncaught through
                // quiesce_devices(), causing GTest TearDown to call quiesce again on
                // already-degraded hardware, which accumulates 5s timeouts and hangs.
                log_warning(
                    tt::LogMetal,
                    "wait_for_fabric_workers_ready: Device {} (mmio={}) Phase 5b: read failed on "
                    "chan {} — treating as not-READY_FOR_TRAFFIC: {}",
                    this->id(),
                    this->is_mmio_capable(),
                    ch.eth_chan_id,
                    e.what());
                // 0xDEADECE7: Phase 5b relay read threw an exception.
                status_buf[0] = static_cast<uint32_t>(EthDiagSentinel::PHASE5B_READ_EXCEPTION);
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
        const auto hc_elapsed =
            std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - hc_start).count();
        // Log every kHCIntermediateLogMs to observe actual time-to-healthy
        if (hc_elapsed / kHCIntermediateLogMs > last_hc_log_ms / static_cast<int64_t>(kHCIntermediateLogMs)) {
            last_hc_log_ms = hc_elapsed;
            std::string pending_str;
            for (const auto& [cid, st] : still_pending_statuses) {
                pending_str += fmt::format(" chan{}=0x{:08x}({})", cid, st, edm_status_str(st));
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
            // Apply the same per-read deadline guard: skip the read and record a
            // distinct sentinel if the diagnostic budget is exhausted.
            for (size_t idx : pending) {
                const auto& ch = chans[idx];
                std::vector<uint32_t> status_buf(1, 0);
                const auto diag_elapsed =
                    std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - hc_start)
                        .count();
                const int64_t kDiagBudgetMs = kHealthCheckTimeoutMs + 6000;
                if (diag_elapsed > kDiagBudgetMs) {
                    log_warning(
                        tt::LogMetal,
                        "wait_for_fabric_workers_ready: Device {} Phase 5b: final "
                        "diagnostic deadline ({}ms) exceeded before reading chan {} — "
                        "recording as 0xDEAD5B5B without read",
                        this->id(),
                        kDiagBudgetMs,
                        ch.eth_chan_id);
                    // 0xDEAD5B5B: Phase 5b deadline exceeded — read skipped.
                    status_buf[0] = static_cast<uint32_t>(EthDiagSentinel::PHASE5B_DEADLINE_SKIPPED);
                } else {
                    try {
                        detail::ReadFromDeviceL1(
                            this, ch.eth_logical_core, router_sync_addr, 4, status_buf, CoreType::ETH);
                    } catch (const std::exception& e) {
                        log_warning(
                            tt::LogMetal,
                            "wait_for_fabric_workers_ready: Device {} Phase 5b: final "
                            "diagnostic read failed on chan {} — recording as 0xDEADECE7: {}",
                            this->id(),
                            ch.eth_chan_id,
                            e.what());
                        // 0xDEADECE7: Phase 5b relay read threw an exception.
                        status_buf[0] = static_cast<uint32_t>(EthDiagSentinel::PHASE5B_READ_EXCEPTION);
                    }
                }
                unhealthy.push_back({ch.eth_chan_id, status_buf[0]});
            }
            break;
        }
        if (++hc_spin >= kSpinLimit) {
            hc_spin = 0U;
            std::this_thread::sleep_for(std::chrono::microseconds(100));
        } else {
            ttsl::pause();
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
                    "  dev={} chan={} status=0x{:08x}({})\n",
                    this->id(),
                    u.eth_chan_id,
                    u.actual_status,
                    edm_status_str(u.actual_status));
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
                    "  dev={} chan={} status=0x{:08x}({})\n",
                    this->id(),
                    u.eth_chan_id,
                    u.actual_status,
                    edm_status_str(u.actual_status));
            }
            // FIX V/W (#42429): If ALL truly-unhealthy channels are at 0x0,
            // 0xDEAD5B5B (deadline-skipped), or 0xDEADECE7 (read exception), the
            // device's ETH channels never booted fabric firmware after Phase 3.
            // For non-MMIO devices this means the UMD relay path is broken (FIX V).
            // For MMIO devices this means a cascade from a peer device's relay failure
            // caused the firmware never to respond (FIX W).  In both cases TT_THROW
            // is wrong: it triggers a teardown second quiesce that hits
            // rescue_stuck_dispatch_cores via the still-broken UMD relay on non-MMIO
            // devices, accumulating 5-second timeouts per stream for 8+ minutes.
            // Instead: log_error + return cleanly.
            const bool all_dead =
                std::all_of(truly_unhealthy.begin(), truly_unhealthy.end(), [](const UnhealthyChannel& u) {
                    return u.actual_status == 0x0 ||
                           u.actual_status == static_cast<uint32_t>(EthDiagSentinel::PHASE5B_DEADLINE_SKIPPED) ||
                           u.actual_status == static_cast<uint32_t>(EthDiagSentinel::PHASE5B_READ_EXCEPTION);
                });
            if (all_dead) {
                if (!this->is_mmio_capable()) {
                    // Non-MMIO: set flag so Phase 2.5/3/5 relay ops are skipped in
                    // subsequent quiesce (reads through UMD relay would hang).
                    fabric_relay_path_broken_ = true;
                }
                log_error(
                    tt::LogMetal,
                    "wait_for_fabric_workers_ready: Device {} Phase 5b: all {} truly-unhealthy "
                    "channel(s) stuck at 0x0/0xDEAD5B5B/0xDEADECE7 (mmio={}) — "
                    "ETH firmware did not boot after Phase 3, probable cascade from peer "
                    "device relay failure.  Returning cleanly to prevent teardown "
                    "rescue_stuck_dispatch_cores cascade.  (FIX W: #42429)\n{}",
                    this->id(),
                    truly_unhealthy.size(),
                    this->is_mmio_capable(),
                    details);
                return true;  // early exit — caller should return
            }
            // FIX AK (#42429): partial-mesh quiesce — channels stuck below READY_FOR_TRAFFIC
            // because their peers are non-mesh devices that didn't participate in this quiesce
            // cycle.  In a 1x4 (or any sub-8) mesh on T3K, some ETH channels of mesh-edge
            // devices connect to chips outside the mesh.  Those out-of-mesh peers run base-UMD
            // firmware during quiesce and never respond to the EDM handshake.  As a result the
            // local ERISC reaches STARTED or REMOTE_HANDSHAKE_COMPLETE but cannot advance to
            // READY_FOR_TRAFFIC.  These are non-fatal: the test already completed, teardown
            // is running, and Phase 2.5 in the next quiesce will TERMINATE these channels.
            // We must NOT set fabric_relay_path_broken_ here — the relay IS functional, it is
            // the PEER device that did not respond.
            using EDMSt = tt::tt_fabric::EDMStatus;
            const bool all_handshake_incomplete =
                !all_dead && std::all_of(truly_unhealthy.begin(), truly_unhealthy.end(), [](const UnhealthyChannel& u) {
                    return u.actual_status == 0x0 ||
                           u.actual_status == static_cast<uint32_t>(EthDiagSentinel::PHASE5B_DEADLINE_SKIPPED) ||
                           u.actual_status == static_cast<uint32_t>(EthDiagSentinel::PHASE5B_READ_EXCEPTION) ||
                           u.actual_status == static_cast<uint32_t>(EDMSt::STARTED) ||
                           u.actual_status == static_cast<uint32_t>(EDMSt::REMOTE_HANDSHAKE_COMPLETE) ||
                           u.actual_status == static_cast<uint32_t>(EDMSt::LOCAL_HANDSHAKE_COMPLETE);
                });
            if (all_handshake_incomplete) {
                log_warning(
                    tt::LogMetal,
                    "wait_for_fabric_workers_ready: Device {} Phase 5b: all {} truly-unhealthy "
                    "channel(s) stuck at or below LOCAL_HANDSHAKE_COMPLETE — "
                    "peer device(s) not in quiesce set (partial-mesh teardown).  "
                    "Non-fatal: Phase 2.5 will TERMINATE these in the next quiesce.  "
                    "(FIX AK: #42429)\n{}",
                    this->id(),
                    truly_unhealthy.size(),
                    details);
                // FIX AM (#42429): Record that ETH channels are not at READY_FOR_TRAFFIC so
                // callers (e.g. tests) can distinguish this state from relay-path-broken and
                // skip AllGather operations that require full fabric readiness.
                fabric_channels_not_ready_for_traffic_ = true;
                return true;  // early exit — caller should return
            }
            // Truly unexpected states (L1 corrupt, init postcodes, garbage) — throw.
            // STARTED (0xa0b0c0d0) = launched but handshake not complete → timing issue
            // 0x0 = launch message never arrived → firmware loading failure
            // 0x49705180 = L1 corrupt (prior crash left garbage) → needs tt-smi -r
            // Other values = unexpected firmware state
            // FIX AK-2 (#42429): Non-MMIO devices (remote peers in a partial mesh) may see
            // unexpected channel states during async teardown without being at fault.
            // Return cleanly rather than throwing — the MMIO host is the authoritative
            // fabric controller and will detect true failures.
            if (!is_mmio_capable()) {
                log_warning(
                    LogDevice,
                    "Device {}: Phase 5b fabric health: {} truly-unhealthy channel(s) with unexpected "
                    "status (non-MMIO device in partial mesh) — returning non-fatal",
                    id_,
                    truly_unhealthy.size());
                fabric_channels_not_ready_for_traffic_ = true;
                return true;
            }
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
    return false;  // success — caller should continue to success log
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

    // FIX RZ4 (#42429): FIX AK/AM (phase5b_erisc_health_check) sets
    // fabric_channels_not_ready_for_traffic_=true when channels are stuck at
    // REMOTE_HANDSHAKE_COMPLETE or fail reads (0xDEADECE7) during a quiesce Phase 5b.
    // That flag persists into the NEXT quiesce call, where FIX QV (below) would skip
    // Phase 4+5 entirely on the FRESH Phase 3 relaunch — channels never get a chance to
    // reach READY_FOR_TRAFFIC, and FIX AK fires again next quiesce → cascade of GTEST_SKIP.
    //
    // FIX RZ4: if the flag is set AND the master ETH router channel is NOT permanently dead
    // (not in fabric_pre_dead_channels_ or fabric_external_umd_channels_), the flag was set
    // transiently.  Phase 2.5 already TERMINATED those stuck channels; Phase 3 relaunched
    // them fresh.  Allow Phase 4+5 to run to verify the relaunch; clear the flag only AFTER
    // Phase 5b confirms all channels reached READY_FOR_TRAFFIC (see clear point below).
    //
    // Exception (FIX QV case): master chan IS permanently dead → keep the flag and let FIX QV
    // skip Phase 4 (Phase 4 would TT_THROW after 5000ms timeout per channel on a dead-master
    // MMIO device — every test TearDown would cost +5s and mark the test FAILED not SKIPPED).
    bool rz4_transient_channels_not_ready = false;
    if (fabric_channels_not_ready_for_traffic_) {
        const auto master_chan_rz4 = builder_ctx.get_fabric_master_router_chan(this->id());
        const bool master_permanently_dead =
            fabric_pre_dead_channels_.count(master_chan_rz4) > 0 ||
            fabric_external_umd_channels_.count(master_chan_rz4) > 0;
        if (!master_permanently_dead) {
            log_info(
                tt::LogMetal,
                "wait_for_fabric_workers_ready: Device {} FIX RZ4 — transient "
                "fabric_channels_not_ready_for_traffic_ detected (set by FIX AK/AM in prior "
                "quiesce cycle; master chan {} is not permanently dead); proceeding with Phase 4+5 "
                "to verify fresh Phase 3 relaunch — flag cleared only after channels confirmed "
                "READY_FOR_TRAFFIC. (#42429)",
                this->id(),
                master_chan_rz4);
            rz4_transient_channels_not_ready = true;
        }
    }

    // FIX QV (#42429): MMIO devices whose master ERISC router channel is pre-dead have
    // fabric_channels_not_ready_for_traffic_=true set by FIX QU after configure_fabric().
    // Phase 3 loaded Tensix MUX firmware on these channels (configure_fabric_cores runs
    // before the flag is set), but the MUX firmware immediately writes TERMINATED because
    // its associated ERISC router channel is dead.  Phase 4 polls for READY_FOR_TRAFFIC and
    // always times out (5000ms) on each such channel — each test TearDown costs +5s and
    // ultimately throws, marking the test FAILED instead of SKIPPED.
    //
    // Skip Phase 4 (and Phase 5) for such devices: the MUX won't carry traffic
    // (test guards skip ops when this flag is set via FIX QS), so the poll is unnecessary.
    // This mirrors the Phase 4+5 skip for fabric_relay_path_broken_ on non-MMIO devices.
    //
    // Exception: rz4_transient_channels_not_ready=true means the flag was set transiently
    // (FIX AK/AM) and the master chan is alive — Phase 4+5 must run to verify the relaunch.
    if (fabric_channels_not_ready_for_traffic_ && !rz4_transient_channels_not_ready) {
        log_warning(
            tt::LogMetal,
            "wait_for_fabric_workers_ready: Device {} has channels not ready for traffic "
            "(fabric_channels_not_ready_for_traffic_) — skipping Phase 4 MUX poll + Phase 5 "
            "handshake (MMIO dead-master-chan device; MUX will be TERMINATED, not READY_FOR_TRAFFIC). "
            "(#42429 FIX QV)",
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
            constexpr int64_t kP4IntermediateLogMs = 2000;
            uint32_t spin_counter = 0;
            int64_t last_p4_log_ms = 0;
            bool ready = false;
            while (true) {
                detail::ReadFromDeviceL1(this, mux_core, status_addr, 4, status_buf, CoreType::WORKER);
                if (status_buf[0] == static_cast<uint32_t>(tt::tt_fabric::EDMStatus::READY_FOR_TRAFFIC)) {
                    ready = true;
                    break;
                }
                const auto elapsed =
                    std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - start)
                        .count();
                // Log every kP4IntermediateLogMs so CI logs show progress while waiting.
                if (elapsed / kP4IntermediateLogMs > last_p4_log_ms / kP4IntermediateLogMs) {
                    last_p4_log_ms = elapsed;
                    log_info(
                        tt::LogMetal,
                        "wait_for_fabric_workers_ready: Device {} eth_chan {} Phase 4: still waiting for "
                        "MUX READY_FOR_TRAFFIC ({}ms elapsed, status=0x{:08x} {})",
                        this->id(),
                        eth_chan_id,
                        elapsed,
                        status_buf[0],
                        edm_status_str(status_buf[0]));
                }
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
                    const auto virtual_mux_coord =
                        env_impl.get_cluster().get_virtual_coordinate_from_logical_coordinates(
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
        // FIX DZ3 (#42429): Pass use_fix_m_nonce=true for FIX M devices so the host uses nonce=0,
        // matching the firmware's session_nonce_effective=0 on the FIX M path.
        const auto [router_sync_addr, sync_status] =
            builder_ctx.get_fabric_router_sync_address_and_status(this->is_fabric_stale_base_umd_channels());
        constexpr uint32_t expected_ready =
            static_cast<uint32_t>(tt::tt_fabric::EDMStatus::READY_FOR_TRAFFIC);
        // FIX BO (#42429): When stale base-UMD channels are present in the cluster (set by
        // MeshDeviceImpl::wait_for_fabric_workers_ready_for_quiesce on ALL devices when any
        // non-MMIO device has fabric_stale_base_umd_channels_=true), the ring handshake takes
        // longer than 10s — matching the initial startup case that FIX TH3 addresses.
        // Extend to 120s (12x) to match FIX TH3's get_fabric_router_sync_timeout_ms() multiplier.
        const uint32_t kSyncTimeoutMs = this->is_fabric_stale_base_umd_channels() ? 120000 : 10000;
        if (this->is_fabric_stale_base_umd_channels()) {
            log_info(
                tt::LogMetal,
                "wait_for_fabric_workers_ready: Device {} FIX BO — stale base-UMD channels "
                "detected, extending Phase 5 kSyncTimeoutMs from 10000ms to 120000ms. (#42429)",
                this->id());
        }
        // FIX AL (#42429): When the ERISC is running (status=STARTED, not 0x0) but has not
        // completed the ETH handshake after kStartedTimeoutMs, the master channel's peer is
        // likely not responding — most commonly because the peer is an out-of-mesh device
        // that was never included in this quiesce set and remains in base-UMD mode.  In the
        // STARTED case we already know firmware booted (so the relay path is not broken), and
        // Phase 5b's FIX AK health-check will handle the non-fatal partial-mesh diagnosis.
        // FIX AO (#42429): reduced from 10000ms → 1000ms.  A channel at STARTED means firmware
        // booted and is waiting for the ETH handshake — which completes in <1ms when both peers
        // are alive.  If still STARTED after 1s, the peer is not responding (out-of-mesh device
        // in base-UMD mode, or peer's fabric firmware never launched).  The previous 10s timeout
        // wasted 40s+ per quiesce cycle on a T3K when 4 devices had out-of-mesh master channels.
        // 1000ms is generous (100x the normal handshake time) while eliminating the 10x waste.
        constexpr uint32_t kStartedTimeoutMs = 1000;
        constexpr uint32_t kSpinLimit = 64U;
        // Log intermediate status every 500ms so we can see whether the ERISC is stuck at
        // STARTED (0xA0B0C0D0), 0x0 (never launched), or something unexpected.
        // FIX AO: reduced from 2500ms to match the shorter kStartedTimeoutMs (1000ms).
        constexpr uint32_t kIntermediateLogIntervalMs = 500;

        const auto master_chan = builder_ctx.get_fabric_master_router_chan(this->id());
        const auto& soc_desc_p5 = env_impl.get_cluster().get_soc_desc(this->id());
        const auto master_logical_core = soc_desc_p5.get_eth_core_for_channel(master_chan, CoordSystem::LOGICAL);

        // Track whether Phase 5 relay read threw so Phase 5b can skip reads that would hang.
        // When the Phase 5 master-channel read throws, it means the UMD relay path to this
        // device is broken (fabric firmware was loaded on the relay ERISC after Phase 3).
        // In that state, Phase 5b reads would either throw 5-second timeouts (accumulating
        // seconds per channel) or hang indefinitely (relay ERISC alive but peer fabric
        // firmware never responds to UMD relay protocol).  Skip Phase 5b entirely instead.
        //
        // Safety note on non-MMIO relay reads in Phase 5:
        // For non-MMIO devices, ReadFromDeviceL1 routes through the UMD ETH relay on the MMIO
        // device.  The function-entry guard (fabric_relay_path_broken_, ~line 1765) prevents
        // re-entry after the relay path is known broken.  However, on the FIRST quiesce call,
        // fabric_relay_path_broken_ is false and this read executes.  If the relay ERISC on
        // the MMIO device is alive but running fabric firmware (not UMD relay protocol), the
        // read HANGS rather than throws — hanging bypasses the catch block below and blocks
        // indefinitely until TT_METAL_OPERATION_TIMEOUT_SECONDS fires at the process level.
        //
        // Defensive guard: if fabric_relay_path_broken_ has been set between the function-entry
        // check and here (e.g. set concurrently from another device's quiesce path), bail before
        // entering the relay read.  For the first quiesce, log that we are entering the relay
        // read path so CI logs clearly show where a potential hang originates.
        if (!this->is_mmio_capable() && fabric_relay_path_broken_) {
            // fabric_relay_path_broken_ set between function-entry check and here.
            // Return rather than enter a relay read that would hang or accumulate 5s UMD timeout.
            log_warning(
                tt::LogMetal,
                "wait_for_fabric_workers_ready: Device {} (non-MMIO) Phase 5: "
                "fabric_relay_path_broken_ set before relay read — returning early to avoid "
                "blocking relay read.  (R3: defensive guard)",
                this->id());
            return;
        }
        if (!this->is_mmio_capable()) {
            // Informational: non-MMIO Phase 5 poll goes through UMD ETH relay.
            // If the relay ERISC is alive but running fabric firmware, this poll loop may
            // hang until TT_METAL_OPERATION_TIMEOUT_SECONDS fires rather than throwing.
            // The catch block handles throws and sets fabric_relay_path_broken_; a hang
            // cannot be caught without a thread-based timeout.  See FIX U/V comments above.
            log_info(
                tt::LogMetal,
                "wait_for_fabric_workers_ready: Device {} (non-MMIO) Phase 5: "
                "entering relay read path via UMD ETH relay on MMIO device. "
                "If this hangs, the relay ERISC is running fabric firmware and not "
                "responding to UMD relay protocol — process timeout will fire.",
                this->id());
        }
        bool phase5_relay_read_threw = false;

        // Skip the handshake poll if the master channel is pre-known dead (FIX AN) or
        // was newly-dead via FIX AS Pass-0 timeout (FIX AT).
        //
        // FIX AT (#42429): FIX AS marks master-chan newly-dead in pending_quiesce_newly_dead_eth_chans_
        // when the UMD canary poll (500ms) fires before the BRISC writes 0x49706550.  Without FIX AT,
        // Phase 5 polls master chan for kSyncTimeoutMs (10s) and reads 0x0 throughout — no firmware
        // was launched, so the status never advances.  FIX-1 then sets fabric_relay_path_broken_=true
        // and Phase 5b is skipped.  FIX AT short-circuits that 10s wait:
        //   1. Detects master chan in pending_quiesce_newly_dead_eth_chans_ (FIX AS outcome).
        //   2. Sets fabric_relay_path_broken_=true immediately (so Phase 5b is skipped via FIX U/V).
        //   3. Skips the 10s Phase 5 poll.
        // Per-device savings: kSyncTimeoutMs (10s).  T3K 2×4 with 2 MMIO devices affected: 20s/cycle.
        const bool master_newly_dead_fixas = pending_quiesce_newly_dead_eth_chans_.count(master_chan) > 0;
        if (master_newly_dead_fixas) {
            fabric_relay_path_broken_ = true;
            log_warning(
                tt::LogMetal,
                "wait_for_fabric_workers_ready: Device {} Phase 5: master chan {} was FIX AS "
                "Pass-0 timeout'd (status=0x0, no firmware loaded) — skipping {}ms handshake "
                "poll + setting fabric_relay_path_broken_=true to skip Phase 5b. (FIX AT: #42429)",
                this->id(),
                master_chan,
                kSyncTimeoutMs);
        }
        // FIX EXT (#42429): also treat external master_chan as dead — firmware not loaded on it,
        // so the handshake poll would spin until FIX AL STARTED early-exit (kStartedTimeoutMs)
        // and then set fabric_channels_not_ready_for_traffic_=true via FIX AM.  Skip cleanly.
        const bool master_is_external = fabric_external_umd_channels_.count(master_chan) > 0;
        if (master_is_external) {
            log_info(
                tt::LogMetal,
                "wait_for_fabric_workers_ready: Device {} Phase 5: master chan={} is external ETH "
                "(no in-cluster peer, firmware not loaded) — skipping handshake poll. "
                "(FIX EXT #42429)",
                this->id(),
                master_chan);
        }
        const bool master_is_dead =
            fabric_pre_dead_channels_.count(master_chan) > 0 || master_newly_dead_fixas || master_is_external;
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
                    detail::ReadFromDeviceL1(this, master_logical_core, router_sync_addr, 4, sync_buf, CoreType::ETH);
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
                const auto elapsed =
                    std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - sync_start)
                        .count();
                // Periodic intermediate log every kIntermediateLogIntervalMs — tells us whether
                // ERISC is stuck at STARTED (launched but no peer), at 0x0 (launch msg never
                // arrived), or at some unexpected value.
                if (elapsed / kIntermediateLogIntervalMs >
                    last_log_ms / static_cast<int64_t>(kIntermediateLogIntervalMs)) {
                    last_log_ms = elapsed;
                    log_info(
                        tt::LogMetal,
                        "wait_for_fabric_workers_ready: Device {} Phase 5: still waiting for "
                        "LOCAL_HANDSHAKE_COMPLETE on master chan {} after {}ms — "
                        "current status=0x{:08x} ({}) (expected=0x{:08x})",
                        this->id(),
                        master_chan,
                        elapsed,
                        sync_buf[0],
                        edm_status_str(sync_buf[0]),
                        sync_status);
                }
                // FIX AL (#42429): STARTED early-exit — ERISC firmware is running but peer
                // is not responding.  This is the partial-mesh scenario where the master
                // channel connects to a device outside the quiesce set.  Phase 5b (FIX AK)
                // will diagnose this as non-fatal.  No need to wait the full 10s.
                if (sync_buf[0] == static_cast<uint32_t>(tt::tt_fabric::EDMStatus::STARTED) &&
                    elapsed > kStartedTimeoutMs) {
                    log_warning(
                        tt::LogMetal,
                        "wait_for_fabric_workers_ready: Device {} Phase 5: STARTED early-exit "
                        "after {}ms — master chan {} ERISC is running but peer is not responding "
                        "(likely out-of-mesh device). Deferring to Phase 5b health check. "
                        "(FIX AL: #42429)",
                        this->id(),
                        elapsed,
                        master_chan);
                    break;
                }
                // FIX BU (#42429): REMOTE_HANDSHAKE_COMPLETE early-exit — local ERISC received
                // the remote handshake but cannot advance to LOCAL_HANDSHAKE_COMPLETE.  This
                // happens when the channel was transitioned via launch_msg (FIX M) from base-UMD
                // relay firmware (0x49706550): the fabric firmware initialises, exchanges the
                // remote-handshake packet with its peer, but the peer's ETH link is already
                // shutting down so LOCAL_HANDSHAKE_COMPLETE is never reached.  2000ms is generous
                // — a healthy handshake completes in <1ms.  Phase 5b (FIX AK) will declare this
                // non-fatal as a partial-mesh teardown scenario.
                constexpr uint32_t kRemoteHandshakeCompleteTimeoutMs = 2000;
                if (sync_buf[0] == static_cast<uint32_t>(tt::tt_fabric::EDMStatus::REMOTE_HANDSHAKE_COMPLETE) &&
                    elapsed > kRemoteHandshakeCompleteTimeoutMs) {
                    log_warning(
                        tt::LogMetal,
                        "wait_for_fabric_workers_ready: Device {} Phase 5: REMOTE_HANDSHAKE_COMPLETE "
                        "early-exit after {}ms — master chan {} local ERISC stuck (launch_msg "
                        "transition?). Deferring to Phase 5b. (FIX BU: #42429)",
                        this->id(),
                        elapsed,
                        master_chan);
                    break;
                }
                if (elapsed > kSyncTimeoutMs) {
                    // FIX V (#42429): On a non-MMIO device, status=0x0 after 10s means the
                    // device's ETH channels never booted fabric firmware — the relay path to
                    // this device is broken.  Set fabric_relay_path_broken_ so that Phase 5b
                    // (health check reads) and any subsequent quiesce relay ops are skipped,
                    // exactly as done for Phase 5 relay read exceptions (FIX U) and ENTRY
                    // snapshot deadline exceeded (FIX S/T).
                    if (sync_buf[0] == 0x0) {
                        // FIX-1 (#42429): status=0x0 after 10s means ETH firmware never booted
                        // on this device (MMIO or non-MMIO).  Set fabric_relay_path_broken_ to
                        // skip Phase 5b health-check reads (which hit channels 14/15 on MMIO T3K
                        // devices — invalid TRANSLATED coordinates → crash) and all relay ops in
                        // subsequent quiesce.  Previously this only applied to !is_mmio_capable();
                        // extending to MMIO closes the crash window for MMIO devices as well.
                        fabric_relay_path_broken_ = true;
                        log_warning(
                            tt::LogMetal,
                            "wait_for_fabric_workers_ready: Device {} (mmio={}) Phase 5: timeout ({}ms) "
                            "waiting for LOCAL_HANDSHAKE_COMPLETE on master chan {} — status "
                            "still 0x0.  Setting fabric_relay_path_broken_=true "
                            "to skip Phase 5b health check and relay ops in subsequent quiesce.  (FIX-1: #42429)",
                            this->id(),
                            this->is_mmio_capable(),
                            kSyncTimeoutMs,
                            master_chan);
                    } else {
                        log_warning(
                            tt::LogMetal,
                            "wait_for_fabric_workers_ready: Device {} Phase 5: timeout ({}ms) waiting "
                            "for LOCAL_HANDSHAKE_COMPLETE on master chan {} (status=0x{:08x} {}, "
                            "expected=0x{:08x}). Continuing to health check.",
                            this->id(),
                            kSyncTimeoutMs,
                            master_chan,
                            sync_buf[0],
                            edm_status_str(sync_buf[0]),
                            sync_status);
                    }
                    break;
                }
                if (++sync_spin >= kSpinLimit) {
                    sync_spin = 0U;
                    std::this_thread::sleep_for(std::chrono::microseconds(100));
                } else {
                    ttsl::pause();
                }
            }

            // Write READY_FOR_TRAFFIC to master channel — master firmware distributes to all
            // subordinates via notify_subordinate_routers(), matching wait_for_fabric_router_sync().
            if (sync_buf[0] == sync_status || sync_buf[0] == expected_ready) {
                auto ready_sig = builder_ctx.get_fabric_router_ready_address_and_signal();
                if (ready_sig) {
                    std::vector<uint32_t> ready_buf(1, static_cast<uint32_t>(ready_sig->second));
                    // Phase 5 already confirmed LOCAL_HANDSHAKE_COMPLETE — this write is
                    // best-effort.  For non-MMIO devices it goes through the UMD ETH relay; if
                    // the relay is in a degraded state this could throw.  Catch and log rather
                    // than rethrowing: the successful Phase 5 handshake result must not be masked
                    // by a best-effort write failure.
                    try {
                        detail::WriteToDeviceL1(this, master_logical_core, ready_sig->first, ready_buf, CoreType::ETH);
                    } catch (const std::exception& e) {
                        log_warning(
                            tt::LogMetal,
                            "wait_for_fabric_workers_ready: Device {} Phase 5: READY_FOR_TRAFFIC "
                            "write to master chan {} failed (best-effort, ignoring): {}",
                            this->id(),
                            master_chan,
                            e.what());
                    } catch (...) {
                        log_warning(
                            tt::LogMetal,
                            "wait_for_fabric_workers_ready: Device {} Phase 5: READY_FOR_TRAFFIC "
                            "write to master chan {} failed with unknown exception (best-effort, ignoring)",
                            this->id(),
                            master_chan);
                    }
                }
                log_info(
                    tt::LogMetal,
                    "wait_for_fabric_workers_ready: Device {} Phase 5: master chan {} sync "
                    "complete (0x{:08x}), READY_FOR_TRAFFIC written",
                    this->id(),
                    master_chan,
                    sync_buf[0]);
            }

            // FIX AM (#42429): STARTED early-exit shortcut.
            //
            // When FIX AL fires (master chan stuck at STARTED after kStartedTimeoutMs — out-of-mesh
            // peer not responding), the READY_FOR_TRAFFIC write above was skipped (guard condition
            // false: sync_buf[0]==STARTED, not LOCAL_HANDSHAKE_COMPLETE or READY_FOR_TRAFFIC).
            //
            // Consequently, ALL subordinate ERISCs are stuck at REMOTE_HANDSHAKE_COMPLETE waiting
            // for master to call notify_subordinate_routers() — which requires master to first
            // complete the ETH handshake (impossible for an out-of-mesh channel).  Running Phase 5b
            // in this state only burns 2001ms per device confirming what we already know.  FIX AK
            // (in phase5b_erisc_health_check) would then fire anyway and set the same flag.
            //
            // Skip Phase 5b immediately: set fabric_channels_not_ready_for_traffic_=true and return.
            // This saves ~2s per device (e.g., 4 devices × 2001ms ≈ 8s quiesce overhead eliminated).
            if (sync_buf[0] == static_cast<uint32_t>(tt::tt_fabric::EDMStatus::STARTED)) {
                fabric_channels_not_ready_for_traffic_ = true;
                log_warning(
                    tt::LogMetal,
                    "wait_for_fabric_workers_ready: Device {} Phase 5: master chan {} still at STARTED "
                    "after early-exit — subordinate channels cannot advance past REMOTE_HANDSHAKE_COMPLETE "
                    "without master completing ETH handshake. Setting "
                    "fabric_channels_not_ready_for_traffic_=true and skipping Phase 5b. (FIX AM: #42429)",
                    this->id(),
                    master_chan);
                return;
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
        // Skip Phase 5b entirely if Phase 5 relay reads threw OR if fabric_relay_path_broken_
        // was set by the Phase 5 timeout (FIX V — non-MMIO device, status=0x0 after 10s).
        // In either case, channel reads in Phase 5b either accumulate 5-second timeouts per
        // channel or hang indefinitely (relay ERISC still alive but peer fabric firmware never
        // responds to UMD relay protocol) — both outcomes block the quiesce path.
        if (phase5_relay_read_threw || fabric_relay_path_broken_) {
            // FIX U / FIX V: Phase 5 relay read failed or timed out with 0x0 on non-MMIO
            // device — UMD relay path is broken.  fabric_relay_path_broken_ is already set
            // (catch block or timeout block above), so subsequent quiesce calls will skip
            // all relay ops.
            //
            // Previously we TT_THROWed here, but that caused a cascading failure:
            //   1. TT_THROW triggers a teardown second quiesce
            //   2. Second quiesce runs Phase 3 for MMIO devices again, re-flashing channels
            //      that are mid-boot from the first Phase 3 (status 0x0 = launch msg lost)
            //   3. MMIO device Phase 5b sees all channels at 0x0 and also TT_THROWs
            //
            // Returning cleanly avoids that cascade.  The AllGather may still succeed:
            // Device N's own ETH channels likely booted fabric firmware correctly in Phase 3;
            // we simply cannot verify it via UMD relay because that relay path has been
            // transitioned to fabric firmware.  If the AllGather hangs, TT_METAL_OPERATION_TIMEOUT
            // will fire and the test will fail with a clear timeout rather than a cascade.
            //
            // NOTE: If Phase 5's read hangs rather than throws (relay ERISC alive but peer
            // fabric firmware ignores UMD relay protocol), we never reach this point.  That
            // scenario requires a non-blocking UMD probe or thread-based timeout.
            // The R3 defensive guard above this loop logs when a non-MMIO device enters the
            // relay read path and short-circuits if fabric_relay_path_broken_ is set between
            // the function-entry check and the read — but it cannot prevent a hang when
            // fabric_relay_path_broken_ is still false (first quiesce, relay not yet known broken).
            log_warning(
                tt::LogMetal,
                "wait_for_fabric_workers_ready: Device {} Phase 5: relay path broken "
                "(phase5_relay_read_threw={} fabric_relay_path_broken_={}) — "
                "skipping Phase 5b to prevent cascading second quiesce from re-flashing "
                "MMIO relay channels mid-boot.  (FIX U/V: #42429)",
                this->id(),
                phase5_relay_read_threw,
                fabric_relay_path_broken_.load());
            return;
        } else {
            // Phase 5b: Per-channel ERISC health check.
            // Extracted to Device::phase5b_erisc_health_check() for readability.
            if (phase5b_erisc_health_check(active_channels, soc_desc_p5, router_sync_addr, expected_ready)) {
                return;
            }
            // FIX RZ4 clear point (#42429): Phase 5b completed without detecting unhealthy
            // channels — all active channels are confirmed at READY_FOR_TRAFFIC.  If we entered
            // Phase 4+5 with a transient flag (rz4_transient_channels_not_ready=true, set by
            // FIX AK/AM in the prior quiesce cycle), clear it now.  From this point the device
            // is fully healthy; the next quiesce will no longer see a stale flag and will not
            // cascade into GTEST_SKIP via FIX QV.
            if (rz4_transient_channels_not_ready) {
                log_info(
                    tt::LogMetal,
                    "wait_for_fabric_workers_ready: Device {} FIX RZ4 — all channels confirmed "
                    "READY_FOR_TRAFFIC after Phase 4+5; clearing transient "
                    "fabric_channels_not_ready_for_traffic_. (#42429)",
                    this->id());
                fabric_channels_not_ready_for_traffic_ = false;
            }
            // FIX DK-1 (#42429): Post-quiesce re-check.
            // FIX M sets fabric_stale_base_umd_channels_ when a channel already had
            // base-UMD relay firmware and was re-launched via launch_msg instead of soft
            // reset.  FIX RZ2 (in configure()) clears it after ring-sync + health check,
            // but only on the configure path.  After quiesce_devices() Phase 3 re-initializes
            // channels via launch_msg, Phase 4+5 (here) independently confirms all channels
            // are at READY_FOR_TRAFFIC.  If Phase 5b just passed — meaning every active ETH
            // channel is confirmed healthy — the stale flag is no longer accurate and can be
            // cleared here.  Without this clear, FIX QW skips all AllGather tests for the
            // remainder of the session even though the channels are actually ready, producing
            // spurious GTEST_SKIP outcomes that mask real failures.
            if (fabric_stale_base_umd_channels_.load()) {
                log_info(
                    tt::LogMetal,
                    "wait_for_fabric_workers_ready: Device {} FIX DK-1 — all channels confirmed "
                    "READY_FOR_TRAFFIC after Phase 4+5 post-quiesce; clearing stale "
                    "fabric_stale_base_umd_channels_ (was set by FIX M). (#42429)",
                    this->id());
                fabric_stale_base_umd_channels_ = false;
            }
        }  // end else (!phase5_relay_read_threw && !fabric_relay_path_broken_)
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
