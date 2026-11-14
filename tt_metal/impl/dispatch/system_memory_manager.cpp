// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <tt_stl/fmt.hpp>
#include "impl/context/metal_context.hpp"
#include <internal/l1_ecc_halt.hpp>
#include "impl/context/metal_env_impl.hpp"
#include "system_memory_manager.hpp"
#include <tt-metalium/math.hpp>
#include <algorithm>
#include <array>
#include <atomic>
#include <cstring>
#include <chrono>
#include <cstdlib>
#include <thread>
#include <string>

#include <tt_stl/assert.hpp>
#include "core_coord.hpp"
#include "dispatch_settings.hpp"
#include "hal_types.hpp"
#include "memcpy.hpp"
#include "allocator/allocator.hpp"
#include "command_queue_common.hpp"
#include "system_memory_cq_interface.hpp"
#include <tt-logger/tt-logger.hpp>
#include <tt-umd/types/cluster_descriptor_types.hpp>
#include <tt-umd/types/xy_pair.hpp>
#include "tt_metal/tools/profiler/tracy_debug_zones.hpp"
#include <tt-umd/types/core_coordinates.hpp>
#include <impl/dispatch/dispatch_core_manager.hpp>
#include "impl/dispatch/kernels/cq_prefetch.hpp"
#include <llrt/tt_cluster.hpp>
#include <impl/dispatch/dispatch_mem_map.hpp>
#include <impl/device/device_manager.hpp>

namespace tt::tt_metal {

namespace {

bool wrap_ge(uint32_t a, uint32_t b) {
    // Signed Diff uses 2's Complement to handle wrap
    // Works as long as a and b are 2^31 apart
    int32_t diff = a - b;
    return diff >= 0;
}

}  // namespace

SystemMemoryManager::SystemMemoryManager(MetalEnvImpl& env, MetalContext& ctx, ChipId device_id, uint8_t num_hw_cqs) :
    env_(env),
    ctx_(ctx),
    device_id(device_id),
    backing_(CqBacking::create(env, ctx, device_id, num_hw_cqs)),
    cq_interfaces(backing_->make_cq_interfaces(num_hw_cqs)),
    cq_events_(std::make_unique<CqEventProgress[]>(num_hw_cqs)) {}

void SystemMemoryManager::set_last_issued_event(const uint8_t cq_id, const uint32_t event_id) {
    this->cq_events_[cq_id].last_issued.store(event_id, std::memory_order_relaxed);
}

uint32_t SystemMemoryManager::get_last_issued_event(const uint8_t cq_id) const {
    return this->cq_events_[cq_id].last_issued.load(std::memory_order_relaxed);
}

void SystemMemoryManager::set_last_completed_event(const uint8_t cq_id, const uint32_t event_id) {
    std::atomic<uint32_t>& last_completed = this->cq_events_[cq_id].last_completed;
    TT_ASSERT(
        wrap_ge(event_id, last_completed.load(std::memory_order_relaxed)),
        "Event ID is expected to increase. Wrapping not supported for sync. Completed event {} but last recorded "
        "completed event is {}, manager {}",
        event_id,
        last_completed.load(std::memory_order_relaxed),
        fmt::ptr(this));
    // Pairs with the acquire in get_last_completed_event, publishing the completion reader's earlier writes.
    last_completed.store(event_id, std::memory_order_release);
}

uint32_t SystemMemoryManager::get_last_completed_event(const uint8_t cq_id) const {
    return this->cq_events_[cq_id].last_completed.load(std::memory_order_acquire);
}

void SystemMemoryManager::reset_events(const uint8_t cq_id, const uint32_t event_id) {
    this->cq_events_[cq_id].last_issued.store(event_id, std::memory_order_relaxed);
    this->cq_events_[cq_id].last_completed.store(event_id, std::memory_order_release);
}

void SystemMemoryManager::reset(const uint8_t cq_id) {
    SystemMemoryCQInterface& cq_interface = this->cq_interfaces[cq_id];
    cq_interface.issue_fifo_wr_ptr = (cq_interface.cq_start + cq_interface.offset) >> 4;  // In 16B words
    cq_interface.issue_fifo_wr_toggle = false;
    cq_interface.completion_fifo_rd_ptr = cq_interface.issue_fifo_limit;
    cq_interface.completion_fifo_rd_toggle = false;
}

uint32_t SystemMemoryManager::get_issue_queue_size(const uint8_t cq_id) const {
    return this->cq_interfaces[cq_id].issue_fifo_size << 4;
}

uint32_t SystemMemoryManager::get_issue_queue_limit(const uint8_t cq_id) const {
    return this->cq_interfaces[cq_id].issue_fifo_limit << 4;
}

uint32_t SystemMemoryManager::get_completion_queue_size(const uint8_t cq_id) const {
    return this->cq_interfaces[cq_id].completion_fifo_size << 4;
}

uint32_t SystemMemoryManager::get_completion_queue_limit(const uint8_t cq_id) const {
    return this->cq_interfaces[cq_id].completion_fifo_limit << 4;
}

uint32_t SystemMemoryManager::get_issue_queue_write_ptr(const uint8_t cq_id) const {
    return this->cq_interfaces[cq_id].issue_fifo_wr_ptr << 4;
}

uint32_t SystemMemoryManager::get_completion_queue_read_ptr(const uint8_t cq_id) const {
    return this->cq_interfaces[cq_id].completion_fifo_rd_ptr << 4;
}

void* SystemMemoryManager::get_completion_queue_ptr(uint8_t cq_id) const {
    // The completion queue follows the issue queue in contiguous memory.
    backing_->sync_completion_queue(cq_id, this->cq_interfaces[cq_id]);
    return backing_->host_ptr(this->get_issue_queue_limit(cq_id));
}

uint32_t SystemMemoryManager::get_completion_queue_read_toggle(const uint8_t cq_id) const {
    return this->cq_interfaces[cq_id].completion_fifo_rd_toggle;
}

uint32_t SystemMemoryManager::get_cq_size() const { return backing_->cq_size(); }

ChipId SystemMemoryManager::get_device_id() const { return this->device_id; }

std::vector<SystemMemoryCQInterface>& SystemMemoryManager::get_cq_interfaces() { return this->cq_interfaces; }

void* SystemMemoryManager::issue_queue_reserve(uint32_t cmd_size_B, const uint8_t cq_id) {
    TT_ASSERT(
        cmd_size_B <= this->get_issue_queue_size(cq_id),
        "Command of {} B does not fit in issue queue {} of {} B",
        cmd_size_B,
        cq_id,
        this->get_issue_queue_size(cq_id));
    uint32_t issue_q_write_ptr = this->get_issue_queue_write_ptr(cq_id);

    const uint32_t command_issue_limit = this->get_issue_queue_limit(cq_id);
    if (issue_q_write_ptr + align(cmd_size_B, backing_->issue_alignment()) > command_issue_limit) {
        this->wrap_issue_queue_wr_ptr(cq_id);
        issue_q_write_ptr = this->get_issue_queue_write_ptr(cq_id);
    }

    return backing_->host_ptr(issue_q_write_ptr);
}

void SystemMemoryManager::cq_write(const void* data, uint32_t size_in_bytes, uint32_t write_ptr) {
    backing_->copy_to_issue_queue(backing_->host_ptr(write_ptr), data, size_in_bytes);
}

// TODO: RENAME issue_queue_stride ?
void SystemMemoryManager::issue_queue_push_back(uint32_t push_size_B, const uint8_t cq_id) {
    const uint32_t push_size_16B = align(push_size_B, backing_->issue_alignment()) >> 4;

    SystemMemoryCQInterface& cq_interface = this->cq_interfaces[cq_id];

    // Capture before advancing: issue_fifo_wr_ptr points to the slot just written; after the advance below it points to
    // the next slot.
    const uint32_t wr_ptr_bytes = cq_interface.issue_fifo_wr_ptr << 4;

    if (cq_interface.issue_fifo_wr_ptr + push_size_16B >= cq_interface.issue_fifo_limit) {
        cq_interface.issue_fifo_wr_ptr = (cq_interface.cq_start + cq_interface.offset) >> 4;  // In 16B words
        cq_interface.issue_fifo_wr_toggle = not cq_interface.issue_fifo_wr_toggle;            // Flip the toggle
    } else {
        cq_interface.issue_fifo_wr_ptr += push_size_16B;
    }

    backing_->issue_queue_pushed(cq_id, wr_ptr_bytes, push_size_16B << 4, cq_interface.issue_fifo_wr_ptr);
}

void SystemMemoryManager::send_completion_queue_read_ptr(const uint8_t cq_id) const {
    const SystemMemoryCQInterface& cq_interface = this->cq_interfaces[cq_id];
    const uint32_t read_ptr_and_toggle =
        cq_interface.completion_fifo_rd_ptr | (cq_interface.completion_fifo_rd_toggle << 31);
    backing_->send_completion_queue_read_ptr(cq_id, read_ptr_and_toggle);
}

void SystemMemoryManager::fetch_queue_reserve_back(const uint8_t cq_id) { backing_->fetch_queue_reserve_back(cq_id); }

uint32_t SystemMemoryManager::completion_queue_wait_front(
    const uint8_t cq_id, std::atomic<bool>& exit_condition) const {
    return backing_->completion_queue_wait_front(cq_id, this->cq_interfaces[cq_id], exit_condition);
}

void SystemMemoryManager::wrap_issue_queue_wr_ptr(const uint8_t cq_id) {
    SystemMemoryCQInterface& cq_interface = this->cq_interfaces[cq_id];
    cq_interface.issue_fifo_wr_ptr = (cq_interface.cq_start + cq_interface.offset) >> 4;
    cq_interface.issue_fifo_wr_toggle = not cq_interface.issue_fifo_wr_toggle;
}

void SystemMemoryManager::completion_queue_pop_front(uint32_t num_pages_read, const uint8_t cq_id) {
    uint32_t data_read_B = num_pages_read * DispatchSettings::TRANSFER_PAGE_SIZE;
    uint32_t data_read_16B = data_read_B >> 4;

    SystemMemoryCQInterface& cq_interface = this->cq_interfaces[cq_id];
    cq_interface.completion_fifo_rd_ptr += data_read_16B;
    if (cq_interface.completion_fifo_rd_ptr >= cq_interface.completion_fifo_limit) {
        cq_interface.completion_fifo_rd_ptr = cq_interface.issue_fifo_limit;
        cq_interface.completion_fifo_rd_toggle = not cq_interface.completion_fifo_rd_toggle;
    }

    // Notify dispatch core
    this->send_completion_queue_read_ptr(cq_id);
}

void SystemMemoryManager::fetch_queue_write(uint32_t command_size_B, const uint8_t cq_id, bool stall_prefetcher) {
    backing_->fetch_queue_write(command_size_B, cq_id, stall_prefetcher);
}

uint32_t SystemMemoryManager::get_dram_region_base_addr() const {
    TT_FATAL(this->is_dram_backed(), "CQs are not DRAM backed");
    return env_.get_hal().get_dev_addr(HalDramMemAddrType::DRAM_BACKED_COMMAND_QUEUES);
}

uint32_t SystemMemoryManager::get_dram_region_bank_id() const {
    TT_FATAL(this->is_dram_backed(), "CQs are not DRAM backed");
    return 0;
}

}  // namespace tt::tt_metal
