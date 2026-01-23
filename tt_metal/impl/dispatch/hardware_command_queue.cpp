// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "hardware_command_queue.hpp"

#include <device.hpp>
#include "dispatch_settings.hpp"
#include "impl/context/metal_context.hpp"
#include "system_memory_manager.hpp"
#include "program/dispatch.hpp"
#include <tracy/Tracy.hpp>
#include <tt_stl/assert.hpp>
#include <tt-logger/tt-logger.hpp>
#include <umd/device/types/xy_pair.hpp>
#include "llrt/tt_cluster.hpp"
#include "dispatch_query_manager.hpp"
#include <impl/dispatch/dispatch_query_manager.hpp>

namespace tt::tt_metal {
enum NOC : uint8_t;
}  // namespace tt::tt_metal

namespace tt::tt_metal {

HWCommandQueue::HWCommandQueue(IDevice* device, uint32_t id, NOC /*noc_index*/) :
    id_(id), manager_(device->sysmem_manager()), device_(device) {
    ZoneScopedN("CommandQueue_constructor");

    uint16_t channel =
        tt::tt_metal::MetalContext::instance().get_cluster().get_assigned_channel_for_device(device_->id());

    CoreCoord enqueue_program_dispatch_core;
    CoreType core_type = MetalContext::instance().get_dispatch_core_manager().get_dispatch_core_type();
    if (this->device_->num_hw_cqs() == 1 or core_type == CoreType::WORKER) {
        // dispatch_s exists with this configuration. Workers write to dispatch_s
        enqueue_program_dispatch_core =
            MetalContext::instance().get_dispatch_core_manager().dispatcher_s_core(device_->id(), channel, id);
    } else {
        if (device_->is_mmio_capable()) {
            enqueue_program_dispatch_core =
                MetalContext::instance().get_dispatch_core_manager().dispatcher_core(device_->id(), channel, id);
        } else {
            enqueue_program_dispatch_core =
                MetalContext::instance().get_dispatch_core_manager().dispatcher_d_core(device_->id(), channel, id);
        }
    }
    this->virtual_enqueue_program_dispatch_core_ =
        device_->virtual_core_from_logical_core(enqueue_program_dispatch_core, core_type);
}

uint32_t HWCommandQueue::id() const { return this->id_; }

SystemMemoryManager& HWCommandQueue::sysmem_manager() { return this->manager_; }

void HWCommandQueue::set_go_signal_noc_data_and_dispatch_sems(
    uint32_t num_dispatch_sems, const vector_aligned<uint32_t>& noc_mcast_unicast_data) {
    program_dispatch::set_num_worker_sems_on_dispatch(device_, this->manager_, id_, num_dispatch_sems);
    program_dispatch::set_go_signal_noc_data_on_dispatch(device_, noc_mcast_unicast_data, this->manager_, id_);
}

IDevice* HWCommandQueue::device() { return this->device_; }

const CoreCoord& HWCommandQueue::virtual_enqueue_program_dispatch_core() const {
    return this->virtual_enqueue_program_dispatch_core_;
}

void HWCommandQueue::terminate() {
    ZoneScopedN("HWCommandQueue_terminate");
    TT_FATAL(!this->manager_.get_bypass_mode(), "Terminate cannot be used with tracing");
    log_debug(tt::LogDispatch, "Terminating dispatch kernels for command queue {}", this->id_);
    // CQ_PREFETCH_CMD_RELAY_INLINE + CQ_DISPATCH_CMD_TERMINATE
    // CQ_PREFETCH_CMD_TERMINATE
    uint32_t cmd_sequence_sizeB = MetalContext::instance().hal().get_alignment(HalMemType::HOST);

    // dispatch and prefetch terminate commands each needs to be a separate fetch queue entry
    void* cmd_region = this->manager_.issue_queue_reserve(cmd_sequence_sizeB, this->id_);
    HugepageDeviceCommand dispatch_d_command_sequence(cmd_region, cmd_sequence_sizeB);
    dispatch_d_command_sequence.add_dispatch_terminate(DispatcherSelect::DISPATCH_MASTER);
    this->manager_.issue_queue_push_back(cmd_sequence_sizeB, this->id_);
    this->manager_.fetch_queue_reserve_back(this->id_);
    this->manager_.fetch_queue_write(cmd_sequence_sizeB, this->id_);
    if (MetalContext::instance().get_dispatch_query_manager().dispatch_s_enabled()) {
        // Terminate dispatch_s if enabled
        cmd_region = this->manager_.issue_queue_reserve(cmd_sequence_sizeB, this->id_);
        HugepageDeviceCommand dispatch_s_command_sequence(cmd_region, cmd_sequence_sizeB);
        dispatch_s_command_sequence.add_dispatch_terminate(DispatcherSelect::DISPATCH_SUBORDINATE);
        this->manager_.issue_queue_push_back(cmd_sequence_sizeB, this->id_);
        this->manager_.fetch_queue_reserve_back(this->id_);
        this->manager_.fetch_queue_write(cmd_sequence_sizeB, this->id_);
    }
    cmd_region = this->manager_.issue_queue_reserve(cmd_sequence_sizeB, this->id_);
    HugepageDeviceCommand prefetch_command_sequence(cmd_region, cmd_sequence_sizeB);
    prefetch_command_sequence.add_prefetch_terminate();
    this->manager_.issue_queue_push_back(cmd_sequence_sizeB, this->id_);
    this->manager_.fetch_queue_reserve_back(this->id_);
    this->manager_.fetch_queue_write(cmd_sequence_sizeB, this->id_);
}

}  // namespace tt::tt_metal
