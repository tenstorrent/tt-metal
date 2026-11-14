// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "command_queue_common.hpp"
#include "allocator/allocator.hpp"
#include "device/device_manager.hpp"
#include "dispatch_settings.hpp"

#include "impl/context/metal_context.hpp"
#include "impl/context/metal_env_impl.hpp"

#include <tt-umd/types/core_coordinates.hpp>
#include <llrt/tt_cluster.hpp>
#include <impl/dispatch/dispatch_mem_map.hpp>
#include <impl/dispatch/dispatch_core_manager.hpp>

namespace tt::tt_metal {

uint32_t get_relative_cq_offset(uint8_t cq_id, uint32_t cq_size) { return cq_id * cq_size; }

uint16_t get_umd_channel(uint16_t channel) { return channel & 0x3; }

uint32_t get_absolute_cq_offset(uint16_t channel, uint8_t cq_id, uint32_t cq_size, uint32_t base) {
    if (base != 0) {
        return base + get_relative_cq_offset(cq_id, cq_size);
    }
    return base + (DispatchSettings::MAX_HUGEPAGE_SIZE * get_umd_channel(channel)) +
           ((channel >> 2) * DispatchSettings::MAX_DEV_CHANNEL_SIZE) + get_relative_cq_offset(cq_id, cq_size);
}

uint32_t get_cq_dispatch_progress(MetalContext& ctx, ChipId chip_id, uint8_t cq_id) {
    uint32_t progress = 0;

    MetalContext& metal_ctx = ctx;
    // Get the dispatcher core for this command queue
    // For remote chips: read from DISPATCH_D (on the remote chip where work actually happens)
    // For local chips: read from DISPATCH_HD (combined dispatcher on local chip)
    uint16_t channel = metal_ctx.get_cluster().get_assigned_channel_for_device(chip_id);
    auto& dispatch_core_manager = metal_ctx.get_dispatch_core_manager();

    const tt_cxy_pair& dispatcher_core_logical =
        dispatch_core_manager.is_dispatcher_d_core_allocated(chip_id, channel, cq_id)
            ? dispatch_core_manager.dispatcher_d_core(chip_id, channel, cq_id)
            : dispatch_core_manager.dispatcher_core(chip_id, channel, cq_id);

    CoreType dispatch_core_type = dispatch_core_manager.get_dispatch_core_type();

    // Get the L1 address where dispatch progress counter is stored
    uint32_t dev_dispatch_progress_ptr = metal_ctx.dispatch_mem_map().get_device_command_queue_addr(
        CommandQueueDeviceAddrType::DISPATCH_PROGRESS, cq_id);

    // read_core expects TRANSLATED (virtual) coordinates
    // dispatcher_core_manager stores logical coordinates, so convert LOGICAL -> TRANSLATED (virtual)
    tt_cxy_pair dispatcher_core_virtual = metal_ctx.get_cluster().get_virtual_coordinate_from_logical_coordinates(
        dispatcher_core_logical, dispatch_core_type);

    metal_ctx.get_cluster().read_core(&progress, sizeof(uint32_t), dispatcher_core_virtual, dev_dispatch_progress_ptr);

    return progress;
}

bool is_cq_shared(CommandQueueDeviceAddrType addr_type) {
    return addr_type == CommandQueueDeviceAddrType::WORKER_COMPLETION_SEMAPHORES ||
           addr_type == CommandQueueDeviceAddrType::COMPLETION_Q0_LAST_EVENT ||
           addr_type == CommandQueueDeviceAddrType::COMPLETION_Q1_LAST_EVENT;
}

}  // namespace tt::tt_metal
