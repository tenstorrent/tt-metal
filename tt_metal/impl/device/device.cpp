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

Device::Device(Device&& other) noexcept = default;
Device& Device::operator=(Device&& other) noexcept = default;

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
    // SystemMemoryManager now has internal stubs for mock devices
    sysmem_manager_ = std::make_unique<SystemMemoryManager>(context_->get_context_id(), this->id_, this->num_hw_cqs());

    // For mock devices, skip HWCommandQueue creation (they don't need real command queues)
    if (MetalEnvAccessor(*env_).impl().get_cluster().get_target_device_type() == tt::TargetDevice::Mock) {
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

void Device::configure_fabric() {
    if (fabric_program_ == nullptr) {
        return;
    }

    // Returns false if any channel had a timed-out soft reset (remote chip unreachable).
    // In that case the dead channels will also hang on read-barrier operations, so we skip
    // l1_barrier below to avoid a 10-minute freeze.  See #42429.
    const bool fabric_cores_healthy = tt::tt_fabric::configure_fabric_cores(this);

    fabric_program_->impl().finalize_offsets(this);

    detail::WriteRuntimeArgsToDevice(this, *fabric_program_, using_fast_dispatch_);
    detail::ConfigureDeviceWithProgram(this, *fabric_program_, using_fast_dispatch_);

    // Note: the l1_barrier below is needed to be sure writes to cores that
    // don't get the GO mailbox have all landed.
    // Skip for degraded devices (some ETH channels timed out in configure_fabric_cores):
    // l1_barrier reads back from every ETH core to confirm writes landed, and those reads
    // will hang indefinitely on dead channels — same root cause as the timed-out soft resets.
    MetalEnvImpl& env_impl = MetalEnvAccessor(*env_).impl();
    if (fabric_cores_healthy) {
        env_impl.get_cluster().l1_barrier(this->id());
    } else {
        log_warning(
            tt::LogMetal,
            "configure_fabric: Skipping l1_barrier for Device {} — some ETH channels had "
            "soft reset failures (dead remote chip?). Fabric may not start on those channels.",
            this->id_);
    }
    std::vector<std::vector<CoreCoord>> logical_cores_used_in_program = fabric_program_->impl().logical_cores();
    const auto& hal = env_impl.get_hal();
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
    {
        const auto [erisc_term_addr, erisc_term_signal] =
            builder_ctx.get_fabric_router_termination_address_and_signal();
        const auto router_sync_addr =
            builder_ctx.get_fabric_router_sync_address_and_status().first;
        constexpr uint32_t terminated_val = static_cast<uint32_t>(tt::tt_fabric::EDMStatus::TERMINATED);
        // 150ms gives cooperative ERISC shutdowns more margin (was 100ms).
        // Still well below the 500ms that added ~9s total quiesce latency.
        constexpr uint32_t erisc_timeout_ms = 150;
        constexpr uint32_t kSpinsBetweenSleeps = 64;

        std::vector<uint32_t> term_buf(1, static_cast<uint32_t>(erisc_term_signal));

        for (const auto& [eth_chan_id, direction] : active_channels) {
            const auto eth_logical_core = env_impl.get_cluster()
                                              .get_soc_desc(this->id())
                                              .get_eth_core_for_channel(eth_chan_id, CoordSystem::LOGICAL);

            std::vector<uint32_t> status_buf(1, 0);
            detail::ReadFromDeviceL1(this, eth_logical_core, router_sync_addr, 4, status_buf, CoreType::ETH);

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
            while (true) {
                detail::ReadFromDeviceL1(
                    this, eth_logical_core, router_sync_addr, 4, status_buf, CoreType::ETH);
                if (status_buf[0] == terminated_val) {
                    terminated = true;
                    break;
                }
                const auto elapsed =
                    std::chrono::duration_cast<std::chrono::milliseconds>(
                        std::chrono::steady_clock::now() - start)
                        .count();
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

    tt::tt_fabric::configure_fabric_cores(this);
    detail::WriteRuntimeArgsToDevice(this, *fabric_program_, using_fast_dispatch_);
    detail::ConfigureDeviceWithProgram(this, *fabric_program_, using_fast_dispatch_);

    env_impl.get_cluster().l1_barrier(this->id());

    // Re-launch only worker cores from the fabric program
    std::vector<std::vector<CoreCoord>> logical_cores_used = fabric_program_->impl().logical_cores();
    const auto& hal = env_impl.get_hal();
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

    // Phase 4: Wait for each MUX core to reach READY_FOR_TRAFFIC before returning.
    //
    // Without this wait, the next dispatch op can arrive while the MUX is still in its
    // startup path (waiting for ERISC, opening the EDM connection, etc.).  The dispatch
    // relay kernel calls wait_for_fabric_endpoint_ready(mux) — an unbounded spin on
    // device — and hangs if the MUX hasn't written READY_FOR_TRAFFIC yet.
    //
    // Use the same bounded-poll + force-reset pattern as Phase 2 so a stuck MUX can
    // never hang the host indefinitely.
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
                    "quiesce_and_restart_fabric_workers: Timeout waiting for fabric MUX READY_FOR_TRAFFIC on "
                    "Device {} eth_chan {} (status=0x{:08x}), force-resetting Tensix MUX core",
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
            }
        }
    }

    // Phase 5: Post-restart ERISC health check.
    //
    // After Phase 3 re-launched firmware and Phase 4 confirmed Tensix MUX readiness, verify
    // that ALL active ERISC channels on this device have reached READY_FOR_TRAFFIC.  If a
    // prior process crash left BRISC-halted ERISC cores with corrupt L1, Phase 3's firmware
    // re-launch may not have recovered them — and the next dispatch op (e.g. all_gather) will
    // hang in completion_queue_wait_front with no useful diagnostic.
    //
    // This check catches that failure mode and throws immediately with a clear message,
    // preventing the downstream hang.  We retry a few times with short delays to tolerate
    // channels that are a few ms away from ready.
    {
        const auto [router_sync_addr, sync_status] =
            builder_ctx.get_fabric_router_sync_address_and_status();
        constexpr uint32_t expected_status =
            static_cast<uint32_t>(tt::tt_fabric::EDMStatus::READY_FOR_TRAFFIC);
        constexpr uint32_t kHealthCheckRetries = 3;
        constexpr uint32_t kHealthCheckRetryDelayMs = 10;

        struct UnhealthyChannel {
            uint32_t eth_chan_id;
            uint32_t actual_status;
        };
        std::vector<UnhealthyChannel> unhealthy;

        // Build list of (eth_chan_id, eth_logical_core) pairs to check.
        struct ChanToCheck {
            uint32_t eth_chan_id;
            CoreCoord eth_logical_core;
        };
        std::vector<ChanToCheck> chans;
        for (const auto& [eth_chan_id, direction] : active_channels) {
            const auto eth_logical_core = env_impl.get_cluster()
                                              .get_soc_desc(this->id())
                                              .get_eth_core_for_channel(eth_chan_id, CoordSystem::LOGICAL);
            chans.push_back({eth_chan_id, eth_logical_core});
        }

        // Retry loop: on each attempt, only re-check channels that haven't passed yet.
        std::vector<size_t> pending;
        pending.reserve(chans.size());
        for (size_t i = 0; i < chans.size(); i++) {
            pending.push_back(i);
        }

        for (uint32_t attempt = 0; attempt < kHealthCheckRetries && !pending.empty(); attempt++) {
            if (attempt > 0) {
                std::this_thread::sleep_for(std::chrono::milliseconds(kHealthCheckRetryDelayMs));
            }
            std::vector<size_t> still_pending;
            for (size_t idx : pending) {
                const auto& ch = chans[idx];
                std::vector<uint32_t> status_buf(1, 0);
                detail::ReadFromDeviceL1(
                    this, ch.eth_logical_core, router_sync_addr, 4, status_buf, CoreType::ETH);
                if (status_buf[0] != expected_status) {
                    still_pending.push_back(idx);
                    if (attempt == kHealthCheckRetries - 1) {
                        unhealthy.push_back({ch.eth_chan_id, status_buf[0]});
                    }
                }
            }
            pending = std::move(still_pending);
        }

        if (!unhealthy.empty()) {
            std::string details;
            for (const auto& u : unhealthy) {
                details += fmt::format(
                    "  dev={} chan={} status=0x{:08x}\n", this->id(), u.eth_chan_id, u.actual_status);
            }
            TT_THROW(
                "Fabric health check failed after quiesce restart on Device {} — "
                "{} ERISC channel(s) not at READY_FOR_TRAFFIC (0x{:08x}). "
                "Possible corrupt ERISC state from prior process crash (#42429). "
                "Run tt-smi -r to reset chips.\n{}",
                this->id(),
                unhealthy.size(),
                expected_status,
                details);
        }

        log_info(
            tt::LogMetal,
            "quiesce_and_restart_fabric_workers: Device {} Phase 5: all {} ERISC channels healthy",
            this->id(),
            chans.size());
    }

    log_info(tt::LogMetal, "Fabric MUX workers restarted on Device {}", this->id_);
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
        if (MetalEnvAccessor(*env_).impl().get_cluster().get_target_device_type() != tt::TargetDevice::Mock) {
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

}  // namespace tt::tt_metal
