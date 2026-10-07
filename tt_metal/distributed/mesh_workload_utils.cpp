// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <tt_stl/fmt.hpp>
#include "device.hpp"
#include "mesh_device.hpp"
#include "distributed/mesh_device_impl.hpp"
#include "impl/context/metal_context.hpp"
#include "impl/context/metal_env_impl.hpp"
#include "dispatch/kernels/cq_commands.hpp"
#include "hal_types.hpp"
#include "llrt/hal.hpp"
#include <cstdint>
#include <tt_stl/strong_type.hpp>
#include "dispatch/system_memory_manager.hpp"
#include "tt_align.hpp"
#include "tt_metal/distributed/mesh_workload_utils.hpp"
#include "tt_metal/impl/dispatch/device_command.hpp"
#include "tt_metal/impl/dispatch/device_command_calculator.hpp"

#include <umd/device/types/core_coordinates.hpp>
#include <impl/dispatch/dispatch_query_manager.hpp>
#include <impl/dispatch/dispatch_mem_map.hpp>

namespace tt::tt_metal::distributed {

namespace {

struct GoSignalSequenceConfig {
    uint8_t cq_id;
    MeshDevice* mesh_device;
    SubDeviceId sub_device_id;
    uint32_t expected_num_workers_completed;
    CoreCoord dispatch_core;
    bool send_mcast;
    bool send_unicasts;
    const program_dispatch::ProgramDispatchMetadata& dispatch_metadata;
    std::optional<uint32_t> config_ring_sync_count;
};

uint32_t go_signal_sequence_size(const GoSignalSequenceConfig& config) {
    auto& mesh_impl = config.mesh_device->impl();
    auto& metal_ctx = mesh_impl.metal_context();
    DeviceCommandCalculator calculator(metal_ctx);
    if (config.config_ring_sync_count.has_value()) {
        calculator.add_dispatch_wait();
    }
    if (metal_ctx.get_dispatch_query_manager().dispatch_s_enabled()) {
        calculator.add_notify_dispatch_s_go_signal_cmd();
    }
    calculator.add_dispatch_go_signal_mcast();

    const uint32_t pcie_alignment = mesh_impl.metal_env().get_hal().get_alignment(HalMemType::HOST);
    return calculator.write_offset_bytes() +
           (config.dispatch_metadata.prefetcher_cache_info.is_cached
                ? 0
                : tt::align(static_cast<uint32_t>(sizeof(CQPrefetchCmd)), pcie_alignment));
}

template <bool HugepageWrite>
void populate_go_signal_sequence(DeviceCommand<HugepageWrite>& commands, const GoSignalSequenceConfig& config) {
    auto& mesh_impl = config.mesh_device->impl();
    auto& metal_ctx = mesh_impl.metal_context();
    const auto& hal = mesh_impl.metal_env().get_hal();

    if (!config.dispatch_metadata.prefetcher_cache_info.is_cached) {
        commands.add_prefetch_set_ringbuffer_offset(
            config.dispatch_metadata.prefetcher_cache_info.offset +
                config.dispatch_metadata.prefetcher_cache_info.mesh_max_program_kernels_sizeB,
            true);
    }

    if (config.config_ring_sync_count.has_value()) {
        commands.add_dispatch_wait(
            CQ_DISPATCH_CMD_WAIT_FLAG_WAIT_STREAM,
            0,
            metal_ctx.dispatch_mem_map().get_dispatch_stream_index(*config.sub_device_id),
            *config.config_ring_sync_count,
            config.cq_id);
    }

    const uint8_t sub_device_index = *config.sub_device_id;
    const uint32_t go_message = hal.make_go_msg_u32(
        dev_msgs::RUN_MSG_GO,
        config.dispatch_core.x,
        config.dispatch_core.y,
        metal_ctx.dispatch_mem_map().get_dispatch_message_update_offset(sub_device_index) +
            metal_ctx.dispatch_mem_map().get_completion_counter_offset(config.cq_id));

    // When running with dispatch_s enabled:
    //   - dispatch_d must notify dispatch_s that a go signal can be sent.
    //   - dispatch_s then multicasts the go signal to all workers.
    // When running without dispatch_s:
    //   - dispatch_d sends the go signal to all workers.
    // No dispatch_d barrier is needed before the notification or go signal because
    // this sequence is not preceded by NOC transactions for program configuration data.
    DispatcherSelect dispatcher = DispatcherSelect::DISPATCH_MASTER;
    if (metal_ctx.get_dispatch_query_manager().dispatch_s_enabled()) {
        // Each bit selects the dispatch_s semaphore for one sub-device; this sequence targets only sub_device_index.
        commands.add_notify_dispatch_s_go_signal_cmd(0, static_cast<uint16_t>(1U << sub_device_index));
        dispatcher = DispatcherSelect::DISPATCH_SUBORDINATE;
    }
    commands.add_dispatch_go_signal_mcast(
        config.expected_num_workers_completed,
        go_message,
        metal_ctx.dispatch_mem_map().get_dispatch_stream_index(sub_device_index),
        (config.send_mcast && config.mesh_device->impl().has_noc_mcast_txns(config.sub_device_id))
            ? *config.sub_device_id
            : CQ_DISPATCH_CMD_GO_NO_MULTICAST_OFFSET,
        config.send_unicasts ? config.mesh_device->impl().num_virtual_eth_cores(config.sub_device_id) : 0,
        config.mesh_device->impl().noc_data_start_index(config.sub_device_id, config.send_unicasts),
        dispatcher);

    TT_ASSERT(commands.size_bytes() == commands.write_offset_bytes());
}

}  // namespace

HostMemDeviceCommand build_go_signal_sequence(
    uint8_t cq_id,
    MeshDevice* mesh_device,
    SubDeviceId sub_device_id,
    uint32_t expected_num_workers_completed,
    CoreCoord dispatch_core,
    bool send_mcast,
    bool send_unicasts,
    const program_dispatch::ProgramDispatchMetadata& dispatch_metadata,
    std::optional<uint32_t> config_ring_sync_count) {
    const GoSignalSequenceConfig config{
        .cq_id = cq_id,
        .mesh_device = mesh_device,
        .sub_device_id = sub_device_id,
        .expected_num_workers_completed = expected_num_workers_completed,
        .dispatch_core = dispatch_core,
        .send_mcast = send_mcast,
        .send_unicasts = send_unicasts,
        .dispatch_metadata = dispatch_metadata,
        .config_ring_sync_count = config_ring_sync_count};
    HostMemDeviceCommand commands(mesh_device->impl().metal_context(), go_signal_sequence_size(config));
    populate_go_signal_sequence(commands, config);
    return commands;
}

// Write the dispatch sequence for a device not running a program.
// In the MeshWorkload context, a go signal must be sent to each device when
// a workload is dispatched, in order to maintain consistent global state.
void write_go_signal_sequence(
    uint8_t cq_id,
    MeshDevice* mesh_device,
    SubDeviceId sub_device_id,
    SystemMemoryManager& sysmem_manager,
    uint32_t expected_num_workers_completed,
    CoreCoord dispatch_core,
    bool send_mcast,
    bool send_unicasts,
    const program_dispatch::ProgramDispatchMetadata& dispatch_md,
    std::optional<uint32_t> config_ring_sync_count) {
    const GoSignalSequenceConfig config{
        .cq_id = cq_id,
        .mesh_device = mesh_device,
        .sub_device_id = sub_device_id,
        .expected_num_workers_completed = expected_num_workers_completed,
        .dispatch_core = dispatch_core,
        .send_mcast = send_mcast,
        .send_unicasts = send_unicasts,
        .dispatch_metadata = dispatch_md,
        .config_ring_sync_count = config_ring_sync_count};
    const uint32_t cmd_sequence_sizeB = go_signal_sequence_size(config);
    void* cmd_region = sysmem_manager.issue_queue_reserve(cmd_sequence_sizeB, cq_id);
    HugepageDeviceCommand go_signal_cmd_sequence(mesh_device->impl().metal_context(), cmd_region, cmd_sequence_sizeB);
    populate_go_signal_sequence(go_signal_cmd_sequence, config);

    sysmem_manager.issue_queue_push_back(cmd_sequence_sizeB, cq_id);

    sysmem_manager.fetch_queue_reserve_back(cq_id);
    sysmem_manager.fetch_queue_write(cmd_sequence_sizeB, cq_id);
}

void write_rt_profiler_flush(
    uint8_t cq_id, SubDeviceId sub_device_id, SystemMemoryManager& sysmem_manager, uint32_t wait_count) {
    MetalContext& metal_ctx = MetalContext::instance(sysmem_manager.get_context_id());
    DeviceCommandCalculator calculator(metal_ctx);
    calculator.add_dispatch_rt_profiler_flush();
    uint32_t cmd_sequence_sizeB = calculator.write_offset_bytes();

    void* cmd_region = sysmem_manager.issue_queue_reserve(cmd_sequence_sizeB, cq_id);

    HugepageDeviceCommand flush_cmd_sequence(metal_ctx, cmd_region, cmd_sequence_sizeB);
    const uint32_t wait_stream = metal_ctx.dispatch_mem_map().get_dispatch_stream_index(*sub_device_id);
    flush_cmd_sequence.add_dispatch_rt_profiler_flush(wait_count, wait_stream);

    TT_ASSERT(flush_cmd_sequence.size_bytes() == flush_cmd_sequence.write_offset_bytes());

    sysmem_manager.issue_queue_push_back(cmd_sequence_sizeB, cq_id);
    sysmem_manager.fetch_queue_reserve_back(cq_id);
    sysmem_manager.fetch_queue_write(cmd_sequence_sizeB, cq_id);
}
}  // namespace tt::tt_metal::distributed
