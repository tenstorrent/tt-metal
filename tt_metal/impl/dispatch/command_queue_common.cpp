// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "command_queue_common.hpp"
#include "allocator/allocator.hpp"
#include "device/device_manager.hpp"
#include "dispatch/system_memory_manager.hpp"
#include "dispatch_settings.hpp"

#include "impl/context/metal_context.hpp"

#include <umd/device/types/core_coordinates.hpp>
#include <llrt/tt_cluster.hpp>
#include <llrt/hal.hpp>
#include <impl/dispatch/dispatch_mem_map.hpp>
#include <impl/dispatch/dispatch_core_manager.hpp>
#include <experimental/fabric/control_plane.hpp>
#include <experimental/fabric/fabric_types.hpp>
#include <optional>

namespace tt::tt_metal {

uint32_t get_relative_cq_offset(uint8_t cq_id, uint32_t cq_size) { return cq_id * cq_size; }

uint16_t get_umd_channel(uint16_t channel) { return channel & 0x3; }

uint32_t get_absolute_cq_offset(uint16_t channel, uint8_t cq_id, uint32_t cq_size, uint32_t base) {
    return base + (DispatchSettings::MAX_HUGEPAGE_SIZE * get_umd_channel(channel)) +
           ((channel >> 2) * DispatchSettings::MAX_DEV_CHANNEL_SIZE) + get_relative_cq_offset(cq_id, cq_size);
}

// Device-written pointers live at an offset within the hugepage that includes the device channel offset, while
// host-written pointers do not
template <bool include_device_channel_offset>
uint32_t read_cq_host_ptr(
    const SystemMemoryManager& sysmem_manager,
    ChipId chip_id,
    uint8_t cq_id,
    uint32_t cq_size,
    uint32_t host_addr_offset) {
    uint32_t recv;
    if (sysmem_manager.is_dram_backed()) {
        const uint32_t dram_channel = MetalContext::instance()
                                          .device_manager()
                                          ->get_active_device(chip_id)
                                          ->allocator_impl()
                                          ->get_dram_channel_from_bank_id(sysmem_manager.get_dram_region_bank_id());
        MetalContext::instance().get_cluster().read_dram_vec(
            &recv,
            sizeof(uint32_t),
            chip_id,
            dram_channel,
            sysmem_manager.get_dram_region_base_addr() + get_relative_cq_offset(cq_id, cq_size) + host_addr_offset);
    } else {
        ChipId mmio_device_id = MetalContext::instance().get_cluster().get_associated_mmio_device(chip_id);
        uint16_t channel = MetalContext::instance().get_cluster().get_assigned_channel_for_device(chip_id);
        uint32_t sysmem_offset = host_addr_offset + get_relative_cq_offset(cq_id, cq_size);
        if constexpr (include_device_channel_offset) {
            sysmem_offset += (channel >> 2) * DispatchSettings::MAX_DEV_CHANNEL_SIZE;
        }
        MetalContext::instance().get_cluster().read_sysmem(
            &recv, sizeof(uint32_t), sysmem_offset, mmio_device_id, channel);
    }
    return recv;
}

template <bool addr_16B>
uint32_t get_cq_issue_rd_ptr(ChipId chip_id, uint8_t cq_id, uint32_t cq_size) {
    uint32_t issue_q_rd_ptr =
        MetalContext::instance().dispatch_mem_map().get_host_command_queue_addr(CommandQueueHostAddrType::ISSUE_Q_RD);
    const SystemMemoryManager& sysmem_manager =
        MetalContext::instance().device_manager()->get_active_device(chip_id)->sysmem_manager();
    uint32_t recv = read_cq_host_ptr<true>(sysmem_manager, chip_id, cq_id, cq_size, issue_q_rd_ptr);
    if constexpr (!addr_16B) {
        return recv << 4;
    }
    return recv;
}

template uint32_t get_cq_issue_rd_ptr<true>(ChipId chip_id, uint8_t cq_id, uint32_t cq_size);
template uint32_t get_cq_issue_rd_ptr<false>(ChipId chip_id, uint8_t cq_id, uint32_t cq_size);

template <bool addr_16B>
uint32_t get_cq_issue_wr_ptr(ChipId chip_id, uint8_t cq_id, uint32_t cq_size) {
    uint32_t issue_q_wr_ptr =
        MetalContext::instance().dispatch_mem_map().get_host_command_queue_addr(CommandQueueHostAddrType::ISSUE_Q_WR);
    const SystemMemoryManager& sysmem_manager =
        MetalContext::instance().device_manager()->get_active_device(chip_id)->sysmem_manager();
    uint32_t recv = read_cq_host_ptr<false>(sysmem_manager, chip_id, cq_id, cq_size, issue_q_wr_ptr);
    if constexpr (!addr_16B) {
        return recv << 4;
    }
    return recv;
}

template uint32_t get_cq_issue_wr_ptr<true>(ChipId chip_id, uint8_t cq_id, uint32_t cq_size);
template uint32_t get_cq_issue_wr_ptr<false>(ChipId chip_id, uint8_t cq_id, uint32_t cq_size);

template <bool addr_16B>
uint32_t get_cq_completion_wr_ptr(ChipId chip_id, uint8_t cq_id, uint32_t cq_size) {
    uint32_t completion_q_wr_ptr = MetalContext::instance().dispatch_mem_map().get_host_command_queue_addr(
        CommandQueueHostAddrType::COMPLETION_Q_WR);
    const SystemMemoryManager& sysmem_manager =
        MetalContext::instance().device_manager()->get_active_device(chip_id)->sysmem_manager();
    uint32_t recv = read_cq_host_ptr<true>(sysmem_manager, chip_id, cq_id, cq_size, completion_q_wr_ptr);
    if constexpr (!addr_16B) {
        return recv << 4;
    }
    return recv;
}

template uint32_t get_cq_completion_wr_ptr<true>(ChipId chip_id, uint8_t cq_id, uint32_t cq_size);
template uint32_t get_cq_completion_wr_ptr<false>(ChipId chip_id, uint8_t cq_id, uint32_t cq_size);

template <bool addr_16B>
uint32_t get_cq_completion_rd_ptr(ChipId chip_id, uint8_t cq_id, uint32_t cq_size) {
    uint32_t completion_q_rd_ptr = MetalContext::instance().dispatch_mem_map().get_host_command_queue_addr(
        CommandQueueHostAddrType::COMPLETION_Q_RD);
    const SystemMemoryManager& sysmem_manager =
        MetalContext::instance().device_manager()->get_active_device(chip_id)->sysmem_manager();
    uint32_t recv = read_cq_host_ptr<false>(sysmem_manager, chip_id, cq_id, cq_size, completion_q_rd_ptr);
    if constexpr (!addr_16B) {
        return recv << 4;
    }
    return recv;
}

template uint32_t get_cq_completion_rd_ptr<true>(ChipId chip_id, uint8_t cq_id, uint32_t cq_size);
template uint32_t get_cq_completion_rd_ptr<false>(ChipId chip_id, uint8_t cq_id, uint32_t cq_size);

uint32_t get_cq_dispatch_progress(ChipId chip_id, uint8_t cq_id) {
    uint32_t progress = 0;

    // Get the dispatcher core for this command queue
    // For remote chips: read from DISPATCH_D (on the remote chip where work actually happens)
    // For local chips: read from DISPATCH_HD (combined dispatcher on local chip)
    uint16_t channel = tt::tt_metal::MetalContext::instance().get_cluster().get_assigned_channel_for_device(chip_id);
    auto& dispatch_core_manager = MetalContext::instance().get_dispatch_core_manager();

    const tt_cxy_pair& dispatcher_core_logical =
        dispatch_core_manager.is_dispatcher_d_core_allocated(chip_id, channel, cq_id)
            ? dispatch_core_manager.dispatcher_d_core(chip_id, channel, cq_id)
            : dispatch_core_manager.dispatcher_core(chip_id, channel, cq_id);

    // Get the L1 address where dispatch progress counter is stored
    CoreType dispatch_core_type = dispatch_core_manager.get_dispatch_core_type();
    uint32_t dev_dispatch_progress_ptr =
        MetalContext::instance()
            .dispatch_mem_map(dispatch_core_type)
            .get_device_command_queue_addr(CommandQueueDeviceAddrType::DISPATCH_PROGRESS);

    // read_core expects TRANSLATED (virtual) coordinates
    // dispatcher_core_manager stores logical coordinates, so convert LOGICAL -> TRANSLATED (virtual)
    tt_cxy_pair dispatcher_core_virtual =
        MetalContext::instance().get_cluster().get_virtual_coordinate_from_logical_coordinates(
            dispatcher_core_logical, dispatch_core_type);

    tt::tt_metal::MetalContext::instance().get_cluster().read_core(
        &progress, sizeof(uint32_t), dispatcher_core_virtual, dev_dispatch_progress_ptr);

    return progress;
}

uint32_t calculate_expected_workers_to_finish(
    const tt::tt_metal::IDevice* device,
    const SubDeviceId& sub_device_id,
    tt::tt_metal::HalProgrammableCoreType core_type) {
    // Sub Device manager state must be correct (from device init)
    // If core type is active ethernet, it does not include fabric routers which were created using slow dispatch
    // Not managed by fast dispatch
    const auto num_workers = device->num_worker_cores(core_type, sub_device_id);
    return num_workers;
}

// FIX LT9-PROGRESS (#42429 Approach B): Read the always-on fabric packet-progress counter from
// every active fabric ERISC on every device.  The counter lives at FABRIC_KERNEL_HEARTBEAT_ADDR+4
// (written by fabric_erisc_router.cpp whenever tx_progress||rx_progress is true in the main loop).
// Returns an XOR of all per-ERISC counter values; any change between two calls means at least one
// ERISC is processing packets — the host timeout clock should reset.
//
// Returns 0 immediately when:
//  - Fabric is disabled (FabricConfig::DISABLED)
//  - DeviceManager is not initialized
//  - The read throws (e.g. relay broken on non-MMIO device) — we eat the exception to keep the
//    timeout path non-fatal, at the cost of possibly not seeing ERISC progress for that device.
uint32_t get_fabric_erisc_progress() {
    auto& ctx = MetalContext::instance();
    if (!ctx.is_device_manager_initialized()) {
        return 0;
    }

    auto& control_plane = ctx.get_control_plane();
    if (control_plane.get_fabric_config() == tt_fabric::FabricConfig::DISABLED) {
        return 0;
    }

    // Heartbeat address is arch-dependent; packet counter lives at +4.
    const uint32_t heartbeat_addr = ctx.hal().get_eth_fw_mailbox_val(FWMailboxMsg::HEARTBEAT);
    const uint32_t packet_counter_addr = heartbeat_addr + 4;

    auto& cluster = ctx.get_cluster();
    uint32_t combined = 0;

    for (ChipId chip_id : ctx.device_manager()->get_all_active_device_ids()) {
        // get_active_fabric_eth_channels returns {chan_id, direction} pairs for channels
        // that are running fabric router kernels.
        const auto fabric_node_id = [&]() -> std::optional<tt_fabric::FabricNodeId> {
            try {
                return control_plane.get_fabric_node_id_from_physical_chip_id(chip_id);
            } catch (...) {
                return std::nullopt;
            }
        }();
        if (!fabric_node_id) {
            continue;  // chip not in fabric cluster
        }

        for (const auto& [eth_chan_id, _direction] :
             control_plane.get_active_fabric_eth_channels(*fabric_node_id)) {
            try {
                const CoreCoord eth_logical_core =
                    cluster.get_soc_desc(chip_id).get_eth_core_for_channel(eth_chan_id, CoordSystem::LOGICAL);
                const tt_cxy_pair eth_logical_pair(chip_id, eth_logical_core.x, eth_logical_core.y);
                const tt_cxy_pair eth_virtual =
                    cluster.get_virtual_coordinate_from_logical_coordinates(eth_logical_pair, CoreType::ETH);
                uint32_t counter_val = 0;
                cluster.read_core(&counter_val, sizeof(uint32_t), eth_virtual, packet_counter_addr);
                combined ^= counter_val;
            } catch (...) {
                // Non-MMIO relay broken or core unreachable — skip silently.
            }
        }
    }

    return combined;
}

// FIX LT9-PROGRESS (#42429 Approach C): Diagnostic dump of all active ERISC state when a dispatch
// timeout fires.  Logs the heartbeat, packet counter, and EDMStatus for every active fabric ERISC.
// This runs synchronously inside the on_timeout lambda — it is best-effort and never throws.
// The goal is to give ops engineers a post-mortem snapshot of which ERISC(s) are truly stuck.
void dump_fabric_erisc_state() {
    auto& ctx = MetalContext::instance();
    if (!ctx.is_device_manager_initialized()) {
        return;
    }

    auto& control_plane = ctx.get_control_plane();
    if (control_plane.get_fabric_config() == tt_fabric::FabricConfig::DISABLED) {
        return;
    }

    const uint32_t heartbeat_addr = ctx.hal().get_eth_fw_mailbox_val(FWMailboxMsg::HEARTBEAT);
    const uint32_t packet_counter_addr = heartbeat_addr + 4;

    auto& cluster = ctx.get_cluster();

    log_warning(tt::LogMetal, "=== ERISC Fabric State Dump (timeout diagnostic) ===");

    for (ChipId chip_id : ctx.device_manager()->get_all_active_device_ids()) {
        const auto fabric_node_id = [&]() -> std::optional<tt_fabric::FabricNodeId> {
            try {
                return control_plane.get_fabric_node_id_from_physical_chip_id(chip_id);
            } catch (...) {
                return std::nullopt;
            }
        }();
        if (!fabric_node_id) {
            continue;
        }

        auto active_channels = control_plane.get_active_fabric_eth_channels(*fabric_node_id);
        if (active_channels.empty()) {
            continue;
        }

        for (const auto& [eth_chan_id, direction] : active_channels) {
            try {
                const CoreCoord eth_logical_core =
                    cluster.get_soc_desc(chip_id).get_eth_core_for_channel(eth_chan_id, CoordSystem::LOGICAL);
                const tt_cxy_pair eth_logical_pair(chip_id, eth_logical_core.x, eth_logical_core.y);
                const tt_cxy_pair eth_virtual =
                    cluster.get_virtual_coordinate_from_logical_coordinates(eth_logical_pair, CoreType::ETH);

                uint32_t heartbeat_val = 0;
                uint32_t packet_counter_val = 0;

                cluster.read_core(&heartbeat_val, sizeof(uint32_t), eth_virtual, heartbeat_addr);
                cluster.read_core(&packet_counter_val, sizeof(uint32_t), eth_virtual, packet_counter_addr);

                // heartbeat format: 0xDCBAxxxx where xxxx is the 16-bit loop counter
                const bool main_loop_alive = (heartbeat_val >> 16) == 0xDCBA;
                log_warning(
                    tt::LogMetal,
                    "  dev={} chan={} eth=({},{})  heartbeat=0x{:08X} ({})  packets_processed={}",
                    chip_id,
                    eth_chan_id,
                    eth_logical_core.x,
                    eth_logical_core.y,
                    heartbeat_val,
                    main_loop_alive ? "ALIVE" : "STALLED",
                    packet_counter_val);
            } catch (const std::exception& e) {
                log_warning(
                    tt::LogMetal,
                    "  dev={} chan={}  read FAILED: {}",
                    chip_id,
                    eth_chan_id,
                    e.what());
            } catch (...) {
                log_warning(tt::LogMetal, "  dev={} chan={}  read FAILED (unknown exception)", chip_id, eth_chan_id);
            }
        }
    }

    log_warning(tt::LogMetal, "=== End ERISC Fabric State Dump ===");
}

}  // namespace tt::tt_metal
