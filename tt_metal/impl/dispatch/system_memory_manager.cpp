// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "impl/context/metal_context.hpp"
#include "system_memory_manager.hpp"
#include <tt-metalium/tt_align.hpp>
#include <algorithm>
#include <atomic>
#include <cstdlib>
#include <optional>
#include <string>
#include <tuple>

#include "assert.hpp"
#include "core_coord.hpp"
#include "dispatch_settings.hpp"
#include "hal.hpp"
#include "hal_types.hpp"
#include "memcpy.hpp"
#include "command_queue_common.hpp"
#include "system_memory_cq_interface.hpp"
// #include <umd/device/driver_atomics.h> - Should be included as it is used here, but the file is missing include
// guards
#include <umd/device/tt_io.hpp>
#include <umd/device/tt_xy_pair.h>
#include <umd/device/types/cluster_descriptor_types.h>
#include <umd/device/types/xy_pair.h>
#include "utils.hpp"

enum class CoreType;

namespace tt::tt_metal {

SystemMemoryManager::SystemMemoryManager(chip_id_t device_id, uint8_t num_hw_cqs) :
    device_id(device_id),
    num_hw_cqs(num_hw_cqs),
    fast_write_callable(
        tt::tt_metal::MetalContext::instance().get_cluster().get_fast_pcie_static_tlb_write_callable(device_id)),
    bypass_enable(false),
    bypass_buffer_write_offset(0) {
    this->completion_byte_addrs.resize(num_hw_cqs);
    this->prefetcher_cores.resize(num_hw_cqs);
    this->prefetch_q_writers.reserve(num_hw_cqs);
    this->prefetch_q_dev_ptrs.resize(num_hw_cqs);
    this->prefetch_q_dev_fences.resize(num_hw_cqs);

    // Split hugepage into however many pieces as there are CQs
    chip_id_t mmio_device_id =
        tt::tt_metal::MetalContext::instance().get_cluster().get_associated_mmio_device(device_id);
    uint16_t channel = tt::tt_metal::MetalContext::instance().get_cluster().get_assigned_channel_for_device(device_id);
    char* hugepage_start =
        (char*)tt::tt_metal::MetalContext::instance().get_cluster().host_dma_address(0, mmio_device_id, channel);
    hugepage_start += (channel >> 2) * DispatchSettings::MAX_DEV_CHANNEL_SIZE;
    this->cq_sysmem_start = hugepage_start;

    // TODO(abhullar): Remove env var and expose sizing at the API level
    char* cq_size_override_env = std::getenv("TT_METAL_CQ_SIZE_OVERRIDE");
    if (cq_size_override_env != nullptr) {
        uint32_t cq_size_override = std::stoi(string(cq_size_override_env));
        this->cq_size = cq_size_override;
    } else {
        this->cq_size =
            tt::tt_metal::MetalContext::instance().get_cluster().get_host_channel_size(mmio_device_id, channel) /
            num_hw_cqs;
        if (tt::tt_metal::MetalContext::instance().get_cluster().is_galaxy_cluster()) {
            // We put 4 galaxy devices per huge page since number of hugepages available is less than number of
            // devices.
            this->cq_size = this->cq_size / DispatchSettings::DEVICES_PER_UMD_CHANNEL;
        }
    }
    this->channel_offset = DispatchSettings::MAX_HUGEPAGE_SIZE * get_umd_channel(channel) +
                           (channel >> 2) * DispatchSettings::MAX_DEV_CHANNEL_SIZE;

    CoreType core_type = tt::tt_metal::MetalContext::instance().get_dispatch_core_manager().get_dispatch_core_type();
    uint32_t completion_q_rd_ptr = MetalContext::instance().dispatch_mem_map().get_device_command_queue_addr(
        CommandQueueDeviceAddrType::COMPLETION_Q_RD);
    uint32_t prefetch_q_base = MetalContext::instance().dispatch_mem_map().get_device_command_queue_addr(
        CommandQueueDeviceAddrType::UNRESERVED);
    uint32_t cq_start =
        MetalContext::instance().dispatch_mem_map().get_host_command_queue_addr(CommandQueueHostAddrType::UNRESERVED);
    for (uint8_t cq_id = 0; cq_id < num_hw_cqs; cq_id++) {
        tt_cxy_pair prefetcher_core =
            tt::tt_metal::MetalContext::instance().get_dispatch_core_manager().prefetcher_core(
                device_id, channel, cq_id);
        auto prefetcher_virtual =
            tt::tt_metal::MetalContext::instance().get_cluster().get_virtual_coordinate_from_logical_coordinates(
                prefetcher_core.chip, CoreCoord(prefetcher_core.x, prefetcher_core.y), core_type);
        this->prefetcher_cores[cq_id] = tt_cxy_pair(prefetcher_core.chip, prefetcher_virtual.x, prefetcher_virtual.y);
        this->prefetch_q_writers.emplace_back(
            tt::tt_metal::MetalContext::instance().get_cluster().get_static_tlb_writer(this->prefetcher_cores[cq_id]));

        tt_cxy_pair completion_queue_writer_core =
            tt::tt_metal::MetalContext::instance().get_dispatch_core_manager().completion_queue_writer_core(
                device_id, channel, cq_id);
        auto completion_queue_writer_virtual =
            tt::tt_metal::MetalContext::instance().get_cluster().get_virtual_coordinate_from_logical_coordinates(
                completion_queue_writer_core.chip,
                CoreCoord(completion_queue_writer_core.x, completion_queue_writer_core.y),
                core_type);

        const std::tuple<uint32_t, uint32_t> completion_interface_tlb_data = tt::tt_metal::MetalContext::instance()
                                                                                 .get_cluster()
                                                                                 .get_tlb_data(tt_cxy_pair(
                                                                                     completion_queue_writer_core.chip,
                                                                                     completion_queue_writer_virtual.x,
                                                                                     completion_queue_writer_virtual.y))
                                                                                 .value();
        auto [completion_tlb_offset, completion_tlb_size] = completion_interface_tlb_data;
        this->completion_byte_addrs[cq_id] = completion_tlb_offset + completion_q_rd_ptr % completion_tlb_size;

        this->cq_interfaces.push_back(SystemMemoryCQInterface(channel, cq_id, this->cq_size, cq_start));
        // Prefetch queue acts as the sync mechanism to ensure that issue queue has space to write, so issue queue
        // must be as large as the max amount of space the prefetch queue can specify Plus 1 to handle wrapping Plus
        // 1 to allow us to start writing to issue queue before we reserve space in the prefetch queue
        TT_FATAL(
            MetalContext::instance().dispatch_mem_map().max_prefetch_command_size() *
                    (MetalContext::instance().dispatch_mem_map().prefetch_q_entries() + 2) <=
                this->get_issue_queue_size(cq_id),
            "Issue queue for cq_id {} has size of {} which is too small",
            cq_id,
            this->get_issue_queue_size(cq_id));
        this->cq_to_event.push_back(0);
        this->cq_to_last_completed_event.push_back(0);
        this->prefetch_q_dev_ptrs[cq_id] = prefetch_q_base;
        this->prefetch_q_dev_fences[cq_id] =
            prefetch_q_base + MetalContext::instance().dispatch_mem_map().prefetch_q_entries() *
                                  sizeof(DispatchSettings::prefetch_q_entry_type);
    }
    std::vector<std::mutex> temp_mutexes(num_hw_cqs);
    cq_to_event_locks.swap(temp_mutexes);
}

uint32_t SystemMemoryManager::get_next_event(const uint8_t cq_id) {
    cq_to_event_locks[cq_id].lock();
    uint32_t next_event = ++this->cq_to_event[cq_id];  // Event ids start at 1
    cq_to_event_locks[cq_id].unlock();
    return next_event;
}

void SystemMemoryManager::reset_event_id(const uint8_t cq_id) {
    cq_to_event_locks[cq_id].lock();
    this->cq_to_event[cq_id] = 0;
    cq_to_event_locks[cq_id].unlock();
}

void SystemMemoryManager::increment_event_id(const uint8_t cq_id, const uint32_t val) {
    cq_to_event_locks[cq_id].lock();
    this->cq_to_event[cq_id] += val;
    cq_to_event_locks[cq_id].unlock();
}

void SystemMemoryManager::set_last_completed_event(const uint8_t cq_id, const uint32_t event_id) {
    TT_ASSERT(
        event_id >= this->cq_to_last_completed_event[cq_id],
        "Event ID is expected to increase. Wrapping not supported for sync. Completed event {} but last recorded "
        "completed event is {}",
        event_id,
        this->cq_to_last_completed_event[cq_id]);
    cq_to_event_locks[cq_id].lock();
    this->cq_to_last_completed_event[cq_id] = event_id;
    cq_to_event_locks[cq_id].unlock();
}

uint32_t SystemMemoryManager::get_last_completed_event(const uint8_t cq_id) {
    cq_to_event_locks[cq_id].lock();
    uint32_t last_completed_event = this->cq_to_last_completed_event[cq_id];
    cq_to_event_locks[cq_id].unlock();
    return last_completed_event;
}

void SystemMemoryManager::reset(const uint8_t cq_id) {
    SystemMemoryCQInterface& cq_interface = this->cq_interfaces[cq_id];
    cq_interface.issue_fifo_wr_ptr = (cq_interface.cq_start + cq_interface.offset) >> 4;  // In 16B words
    cq_interface.issue_fifo_wr_toggle = false;
    cq_interface.completion_fifo_rd_ptr = cq_interface.issue_fifo_limit;
    cq_interface.completion_fifo_rd_toggle = false;
}

void SystemMemoryManager::set_issue_queue_size(const uint8_t cq_id, const uint32_t issue_queue_size) {
    SystemMemoryCQInterface& cq_interface = this->cq_interfaces[cq_id];
    cq_interface.issue_fifo_size = (issue_queue_size >> 4);
    cq_interface.issue_fifo_limit = (cq_interface.cq_start + cq_interface.offset + issue_queue_size) >> 4;
}

void SystemMemoryManager::set_bypass_mode(const bool enable, const bool clear) {
    this->bypass_enable = enable;
    if (clear) {
        this->bypass_buffer.clear();
        this->bypass_buffer_write_offset = 0;
    }
}

bool SystemMemoryManager::get_bypass_mode() { return this->bypass_enable; }

std::vector<uint32_t>& SystemMemoryManager::get_bypass_data() { return this->bypass_buffer; }

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
    if (this->bypass_enable) {
        return this->bypass_buffer_write_offset;
    } else {
        return this->cq_interfaces[cq_id].issue_fifo_wr_ptr << 4;
    }
}

uint32_t SystemMemoryManager::get_completion_queue_read_ptr(const uint8_t cq_id) const {
    return this->cq_interfaces[cq_id].completion_fifo_rd_ptr << 4;
}

uint32_t SystemMemoryManager::get_completion_queue_read_toggle(const uint8_t cq_id) const {
    return this->cq_interfaces[cq_id].completion_fifo_rd_toggle;
}

uint32_t SystemMemoryManager::get_cq_size() const { return this->cq_size; }

chip_id_t SystemMemoryManager::get_device_id() const { return this->device_id; }

std::vector<SystemMemoryCQInterface>& SystemMemoryManager::get_cq_interfaces() { return this->cq_interfaces; }

void* SystemMemoryManager::issue_queue_reserve(uint32_t cmd_size_B, const uint8_t cq_id) {
    if (this->bypass_enable) {
        uint32_t curr_size = this->bypass_buffer.size();
        uint32_t new_size = curr_size + (cmd_size_B / sizeof(uint32_t));
        this->bypass_buffer.resize(new_size);
        return (void*)((char*)this->bypass_buffer.data() + this->bypass_buffer_write_offset);
    }

    uint32_t issue_q_write_ptr = this->get_issue_queue_write_ptr(cq_id);

    const uint32_t command_issue_limit = this->get_issue_queue_limit(cq_id);
    if (issue_q_write_ptr +
            align(
                cmd_size_B,
                tt::tt_metal::MetalContext::instance().hal().get_alignment(tt::tt_metal::HalMemType::HOST)) >
        command_issue_limit) {
        this->wrap_issue_queue_wr_ptr(cq_id);
        issue_q_write_ptr = this->get_issue_queue_write_ptr(cq_id);
    }

    // Currently read / write pointers on host and device assumes contiguous ranges for each channel
    // Device needs absolute offset of a hugepage to access the region of sysmem that holds a particular command
    // queue
    //  but on host, we access a region of sysmem using addresses relative to a particular channel
    //  this->cq_sysmem_start gives start of hugepage for a given channel
    //  since all rd/wr pointers include channel offset from address 0 to match device side pointers
    //  so channel offset needs to be subtracted to get address relative to channel
    // TODO: Reconsider offset sysmem offset calculations based on
    // https://github.com/tenstorrent/tt-metal/issues/4757
    void* issue_q_region = this->cq_sysmem_start + (issue_q_write_ptr - this->channel_offset);

    return issue_q_region;
}

void SystemMemoryManager::cq_write(const void* data, uint32_t size_in_bytes, uint32_t write_ptr) {
    // Currently read / write pointers on host and device assumes contiguous ranges for each channel
    // Device needs absolute offset of a hugepage to access the region of sysmem that holds a particular command
    // queue
    //  but on host, we access a region of sysmem using addresses relative to a particular channel
    //  this->cq_sysmem_start gives start of hugepage for a given channel
    //  since all rd/wr pointers include channel offset from address 0 to match device side pointers
    //  so channel offset needs to be subtracted to get address relative to channel
    // TODO: Reconsider offset sysmem offset calculations based on
    // https://github.com/tenstorrent/tt-metal/issues/4757
    void* user_scratchspace = this->cq_sysmem_start + (write_ptr - this->channel_offset);

    if (this->bypass_enable) {
        std::copy((uint8_t*)data, (uint8_t*)data + size_in_bytes, (uint8_t*)this->bypass_buffer.data() + write_ptr);
    } else {
        memcpy_to_device(user_scratchspace, data, size_in_bytes);
    }
}

// TODO: RENAME issue_queue_stride ?
void SystemMemoryManager::issue_queue_push_back(uint32_t push_size_B, const uint8_t cq_id) {
    if (this->bypass_enable) {
        this->bypass_buffer_write_offset += push_size_B;
        return;
    }

    // All data needs to be PCIE_ALIGNMENT aligned
    uint32_t push_size_16B =
        align(
            push_size_B, tt::tt_metal::MetalContext::instance().hal().get_alignment(tt::tt_metal::HalMemType::HOST)) >>
        4;

    SystemMemoryCQInterface& cq_interface = this->cq_interfaces[cq_id];
    uint32_t issue_q_wr_ptr =
        MetalContext::instance().dispatch_mem_map().get_host_command_queue_addr(CommandQueueHostAddrType::ISSUE_Q_WR);

    if (cq_interface.issue_fifo_wr_ptr + push_size_16B >= cq_interface.issue_fifo_limit) {
        cq_interface.issue_fifo_wr_ptr = (cq_interface.cq_start + cq_interface.offset) >> 4;  // In 16B words
        cq_interface.issue_fifo_wr_toggle = not cq_interface.issue_fifo_wr_toggle;            // Flip the toggle
    } else {
        cq_interface.issue_fifo_wr_ptr += push_size_16B;
    }

    // Also store this data in hugepages, so if a hang happens we can see what was written by host.
    chip_id_t mmio_device_id =
        tt::tt_metal::MetalContext::instance().get_cluster().get_associated_mmio_device(this->device_id);
    uint16_t channel =
        tt::tt_metal::MetalContext::instance().get_cluster().get_assigned_channel_for_device(this->device_id);
    tt::tt_metal::MetalContext::instance().get_cluster().write_sysmem(
        &cq_interface.issue_fifo_wr_ptr,
        sizeof(uint32_t),
        issue_q_wr_ptr + get_relative_cq_offset(cq_id, this->cq_size),
        mmio_device_id,
        channel);
}

void SystemMemoryManager::send_completion_queue_read_ptr(const uint8_t cq_id) const {
    const SystemMemoryCQInterface& cq_interface = this->cq_interfaces[cq_id];

    uint32_t read_ptr_and_toggle = cq_interface.completion_fifo_rd_ptr | (cq_interface.completion_fifo_rd_toggle << 31);
    this->fast_write_callable(this->completion_byte_addrs[cq_id], 4, (uint8_t*)&read_ptr_and_toggle);

    // Also store this data in hugepages in case we hang and can't get it from the device.
    chip_id_t mmio_device_id =
        tt::tt_metal::MetalContext::instance().get_cluster().get_associated_mmio_device(this->device_id);
    uint16_t channel =
        tt::tt_metal::MetalContext::instance().get_cluster().get_assigned_channel_for_device(this->device_id);
    uint32_t completion_q_rd_ptr = MetalContext::instance().dispatch_mem_map().get_host_command_queue_addr(
        CommandQueueHostAddrType::COMPLETION_Q_RD);
    tt::tt_metal::MetalContext::instance().get_cluster().write_sysmem(
        &read_ptr_and_toggle,
        sizeof(uint32_t),
        completion_q_rd_ptr + get_relative_cq_offset(cq_id, this->cq_size),
        mmio_device_id,
        channel);
}

void SystemMemoryManager::fetch_queue_reserve_back(const uint8_t cq_id) {
    if (this->bypass_enable) {
        return;
    }

    const uint32_t prefetch_q_rd_ptr = MetalContext::instance().dispatch_mem_map().get_device_command_queue_addr(
        CommandQueueDeviceAddrType::PREFETCH_Q_RD);

    // Helper to wait for fetch queue space, if needed
    uint32_t fence;
    auto wait_for_fetch_q_space = [&]() {
        // Loop until space frees up
        while (this->prefetch_q_dev_ptrs[cq_id] == this->prefetch_q_dev_fences[cq_id]) {
            tt::tt_metal::MetalContext::instance().get_cluster().read_core(
                &fence, sizeof(uint32_t), this->prefetcher_cores[cq_id], prefetch_q_rd_ptr);
            this->prefetch_q_dev_fences[cq_id] = fence;
        }
    };

    wait_for_fetch_q_space();

    // Wrap FetchQ if possible
    uint32_t prefetch_q_base = MetalContext::instance().dispatch_mem_map().get_device_command_queue_addr(
        CommandQueueDeviceAddrType::UNRESERVED);
    uint32_t prefetch_q_limit = prefetch_q_base + MetalContext::instance().dispatch_mem_map().prefetch_q_entries() *
                                                      sizeof(DispatchSettings::prefetch_q_entry_type);
    if (this->prefetch_q_dev_ptrs[cq_id] == prefetch_q_limit) {
        this->prefetch_q_dev_ptrs[cq_id] = prefetch_q_base;
        wait_for_fetch_q_space();
    }
}

uint32_t SystemMemoryManager::completion_queue_wait_front(
    const uint8_t cq_id, std::atomic<bool>& exit_condition) const {
    uint32_t write_ptr_and_toggle;
    uint32_t write_ptr;
    uint32_t write_toggle;
    const SystemMemoryCQInterface& cq_interface = this->cq_interfaces[cq_id];

    do {
        write_ptr_and_toggle = get_cq_completion_wr_ptr<true>(this->device_id, cq_id, this->cq_size);
        write_ptr = write_ptr_and_toggle & 0x7fffffff;
        write_toggle = write_ptr_and_toggle >> 31;
    } while (cq_interface.completion_fifo_rd_ptr == write_ptr and
             cq_interface.completion_fifo_rd_toggle == write_toggle and not exit_condition.load());
    return write_ptr_and_toggle;
}

void SystemMemoryManager::wrap_issue_queue_wr_ptr(const uint8_t cq_id) {
    if (this->bypass_enable) {
        return;
    }
    SystemMemoryCQInterface& cq_interface = this->cq_interfaces[cq_id];
    cq_interface.issue_fifo_wr_ptr = (cq_interface.cq_start + cq_interface.offset) >> 4;
    cq_interface.issue_fifo_wr_toggle = not cq_interface.issue_fifo_wr_toggle;
}

void SystemMemoryManager::wrap_completion_queue_rd_ptr(const uint8_t cq_id) {
    SystemMemoryCQInterface& cq_interface = this->cq_interfaces[cq_id];
    cq_interface.completion_fifo_rd_ptr = cq_interface.issue_fifo_limit;
    cq_interface.completion_fifo_rd_toggle = not cq_interface.completion_fifo_rd_toggle;
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
    uint32_t max_command_size_B = MetalContext::instance().dispatch_mem_map().max_prefetch_command_size();
    TT_ASSERT(
        command_size_B <= max_command_size_B,
        "Generated prefetcher command of size {} B exceeds max command size {} B",
        command_size_B,
        max_command_size_B);
    TT_ASSERT(
        (command_size_B >> DispatchSettings::PREFETCH_Q_LOG_MINSIZE) < 0xFFFF, "FetchQ command too large to represent");
    if (this->bypass_enable) {
        return;
    }
    tt_driver_atomics::sfence();
    DispatchSettings::prefetch_q_entry_type command_size_16B =
        command_size_B >> DispatchSettings::PREFETCH_Q_LOG_MINSIZE;

    // stall_prefetcher is used for enqueuing traces, as replaying a trace will hijack the cmd_data_q
    // so prefetcher fetches multiple cmds that include the trace cmd, they will be corrupted by trace pulling data
    // from DRAM stall flag prevents pulling prefetch q entries that occur after the stall entry Stall flag for
    // prefetcher is MSB of FetchQ entry.
    if (stall_prefetcher) {
        command_size_16B |= (1 << ((sizeof(DispatchSettings::prefetch_q_entry_type) * 8) - 1));
    }
    this->prefetch_q_writers[cq_id].write(this->prefetch_q_dev_ptrs[cq_id], command_size_16B);
    this->prefetch_q_dev_ptrs[cq_id] += sizeof(DispatchSettings::prefetch_q_entry_type);
}

}  // namespace tt::tt_metal
