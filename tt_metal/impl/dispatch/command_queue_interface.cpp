// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "command_queue_interface.hpp"

#include "tt_cluster.hpp"

namespace tt::tt_metal {

template <bool addr_16B>
uint32_t get_cq_issue_rd_ptr(chip_id_t chip_id, uint8_t cq_id, uint32_t cq_size) {
    uint32_t recv;
    chip_id_t mmio_device_id = tt::Cluster::instance().get_associated_mmio_device(chip_id);
    uint16_t channel = tt::Cluster::instance().get_assigned_channel_for_device(chip_id);
    uint32_t channel_offset = (channel >> 2) * tt::tt_metal::DispatchSettings::MAX_DEV_CHANNEL_SIZE;
    CoreType core_type = tt::tt_metal::dispatch_core_manager::instance().get_dispatch_core_type(chip_id);
    uint32_t issue_q_rd_ptr =
        DispatchMemMap::get(core_type).get_host_command_queue_addr(CommandQueueHostAddrType::ISSUE_Q_RD);
    tt::Cluster::instance().read_sysmem(
        &recv,
        sizeof(uint32_t),
        issue_q_rd_ptr + channel_offset + get_relative_cq_offset(cq_id, cq_size),
        mmio_device_id,
        channel);
    if constexpr (!addr_16B) {
        return recv << 4;
    }
    return recv;
}

template uint32_t get_cq_issue_rd_ptr<true>(chip_id_t chip_id, uint8_t cq_id, uint32_t cq_size);
template uint32_t get_cq_issue_rd_ptr<false>(chip_id_t chip_id, uint8_t cq_id, uint32_t cq_size);

template <bool addr_16B>
uint32_t get_cq_issue_wr_ptr(chip_id_t chip_id, uint8_t cq_id, uint32_t cq_size) {
    uint32_t recv;
    chip_id_t mmio_device_id = tt::Cluster::instance().get_associated_mmio_device(chip_id);
    uint16_t channel = tt::Cluster::instance().get_assigned_channel_for_device(chip_id);
    CoreType core_type = tt::tt_metal::dispatch_core_manager::instance().get_dispatch_core_type(chip_id);
    uint32_t issue_q_wr_ptr =
        DispatchMemMap::get(core_type).get_host_command_queue_addr(CommandQueueHostAddrType::ISSUE_Q_WR);
    tt::Cluster::instance().read_sysmem(
        &recv, sizeof(uint32_t), issue_q_wr_ptr + get_relative_cq_offset(cq_id, cq_size), mmio_device_id, channel);
    if constexpr (!addr_16B) {
        return recv << 4;
    }
    return recv;
}

template uint32_t get_cq_issue_wr_ptr<true>(chip_id_t chip_id, uint8_t cq_id, uint32_t cq_size);
template uint32_t get_cq_issue_wr_ptr<false>(chip_id_t chip_id, uint8_t cq_id, uint32_t cq_size);

template <bool addr_16B>
uint32_t get_cq_completion_wr_ptr(chip_id_t chip_id, uint8_t cq_id, uint32_t cq_size) {
    uint32_t recv;
    chip_id_t mmio_device_id = tt::Cluster::instance().get_associated_mmio_device(chip_id);
    uint16_t channel = tt::Cluster::instance().get_assigned_channel_for_device(chip_id);
    uint32_t channel_offset = (channel >> 2) * tt::tt_metal::DispatchSettings::MAX_DEV_CHANNEL_SIZE;
    CoreType core_type = tt::tt_metal::dispatch_core_manager::instance().get_dispatch_core_type(chip_id);
    uint32_t completion_q_wr_ptr =
        DispatchMemMap::get(core_type).get_host_command_queue_addr(CommandQueueHostAddrType::COMPLETION_Q_WR);
    tt::Cluster::instance().read_sysmem(
        &recv,
        sizeof(uint32_t),
        completion_q_wr_ptr + channel_offset + get_relative_cq_offset(cq_id, cq_size),
        mmio_device_id,
        channel);
    if constexpr (!addr_16B) {
        return recv << 4;
    }
    return recv;
}

template uint32_t get_cq_completion_wr_ptr<true>(chip_id_t chip_id, uint8_t cq_id, uint32_t cq_size);
template uint32_t get_cq_completion_wr_ptr<false>(chip_id_t chip_id, uint8_t cq_id, uint32_t cq_size);

template <bool addr_16B>
inline uint32_t get_cq_completion_rd_ptr(chip_id_t chip_id, uint8_t cq_id, uint32_t cq_size) {
    uint32_t recv;
    chip_id_t mmio_device_id = tt::Cluster::instance().get_associated_mmio_device(chip_id);
    uint16_t channel = tt::Cluster::instance().get_assigned_channel_for_device(chip_id);
    CoreType core_type = tt::tt_metal::dispatch_core_manager::instance().get_dispatch_core_type(chip_id);
    uint32_t completion_q_rd_ptr =
        DispatchMemMap::get(core_type).get_host_command_queue_addr(CommandQueueHostAddrType::COMPLETION_Q_RD);
    tt::Cluster::instance().read_sysmem(
        &recv, sizeof(uint32_t), completion_q_rd_ptr + get_relative_cq_offset(cq_id, cq_size), mmio_device_id, channel);
    if constexpr (!addr_16B) {
        return recv << 4;
    }
    return recv;
}

template uint32_t get_cq_completion_rd_ptr<true>(chip_id_t chip_id, uint8_t cq_id, uint32_t cq_size);
template uint32_t get_cq_completion_rd_ptr<false>(chip_id_t chip_id, uint8_t cq_id, uint32_t cq_size);

SystemMemoryManager::SystemMemoryManager(chip_id_t device_id, uint8_t num_hw_cqs) :
    device_id(device_id),
    num_hw_cqs(num_hw_cqs),
    fast_write_callable(tt::Cluster::instance().get_fast_pcie_static_tlb_write_callable(device_id)),
    bypass_enable(false),
    bypass_buffer_write_offset(0) {
    this->completion_byte_addrs.resize(num_hw_cqs);
    this->prefetcher_cores.resize(num_hw_cqs);
    this->prefetch_q_writers.reserve(num_hw_cqs);
    this->prefetch_q_dev_ptrs.resize(num_hw_cqs);
    this->prefetch_q_dev_fences.resize(num_hw_cqs);

    // Split hugepage into however many pieces as there are CQs
    chip_id_t mmio_device_id = tt::Cluster::instance().get_associated_mmio_device(device_id);
    uint16_t channel = tt::Cluster::instance().get_assigned_channel_for_device(device_id);
    char* hugepage_start = (char*)tt::Cluster::instance().host_dma_address(0, mmio_device_id, channel);
    hugepage_start += (channel >> 2) * DispatchSettings::MAX_DEV_CHANNEL_SIZE;
    this->cq_sysmem_start = hugepage_start;

    // TODO(abhullar): Remove env var and expose sizing at the API level
    char* cq_size_override_env = std::getenv("TT_METAL_CQ_SIZE_OVERRIDE");
    if (cq_size_override_env != nullptr) {
        uint32_t cq_size_override = std::stoi(string(cq_size_override_env));
        this->cq_size = cq_size_override;
    } else {
        this->cq_size = tt::Cluster::instance().get_host_channel_size(mmio_device_id, channel) / num_hw_cqs;
        if (tt::Cluster::instance().is_galaxy_cluster()) {
            // We put 4 galaxy devices per huge page since number of hugepages available is less than number of
            // devices.
            this->cq_size = this->cq_size / DispatchSettings::DEVICES_PER_UMD_CHANNEL;
        }
    }
    this->channel_offset = DispatchSettings::MAX_HUGEPAGE_SIZE * get_umd_channel(channel) +
                           (channel >> 2) * DispatchSettings::MAX_DEV_CHANNEL_SIZE;

    CoreType core_type = tt::tt_metal::dispatch_core_manager::instance().get_dispatch_core_type(device_id);
    uint32_t completion_q_rd_ptr =
        DispatchMemMap::get(core_type).get_device_command_queue_addr(CommandQueueDeviceAddrType::COMPLETION_Q_RD);
    uint32_t prefetch_q_base =
        DispatchMemMap::get(core_type).get_device_command_queue_addr(CommandQueueDeviceAddrType::UNRESERVED);
    uint32_t cq_start =
        DispatchMemMap::get(core_type).get_host_command_queue_addr(CommandQueueHostAddrType::UNRESERVED);
    for (uint8_t cq_id = 0; cq_id < num_hw_cqs; cq_id++) {
        tt_cxy_pair prefetcher_core =
            tt::tt_metal::dispatch_core_manager::instance().prefetcher_core(device_id, channel, cq_id);
        auto prefetcher_virtual = tt::Cluster::instance().get_virtual_coordinate_from_logical_coordinates(
            prefetcher_core.chip, CoreCoord(prefetcher_core.x, prefetcher_core.y), core_type);
        this->prefetcher_cores[cq_id] = tt_cxy_pair(prefetcher_core.chip, prefetcher_virtual.x, prefetcher_virtual.y);
        this->prefetch_q_writers.emplace_back(
            tt::Cluster::instance().get_static_tlb_writer(this->prefetcher_cores[cq_id]));

        tt_cxy_pair completion_queue_writer_core =
            tt::tt_metal::dispatch_core_manager::instance().completion_queue_writer_core(device_id, channel, cq_id);
        auto completion_queue_writer_virtual = tt::Cluster::instance().get_virtual_coordinate_from_logical_coordinates(
            completion_queue_writer_core.chip,
            CoreCoord(completion_queue_writer_core.x, completion_queue_writer_core.y),
            core_type);

        const std::tuple<uint32_t, uint32_t> completion_interface_tlb_data = tt::Cluster::instance()
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
            DispatchMemMap::get(core_type, num_hw_cqs).max_prefetch_command_size() *
                    (DispatchMemMap::get(core_type, num_hw_cqs).prefetch_q_entries() + 2) <=
                this->get_issue_queue_size(cq_id),
            "Issue queue for cq_id {} has size of {} which is too small",
            cq_id,
            this->get_issue_queue_size(cq_id));
        this->cq_to_event.push_back(0);
        this->cq_to_last_completed_event.push_back(0);
        this->prefetch_q_dev_ptrs[cq_id] = prefetch_q_base;
        this->prefetch_q_dev_fences[cq_id] =
            prefetch_q_base + DispatchMemMap::get(core_type, num_hw_cqs).prefetch_q_entries() *
                                  sizeof(DispatchSettings::prefetch_q_entry_type);
    }
    std::vector<std::mutex> temp_mutexes(num_hw_cqs);
    cq_to_event_locks.swap(temp_mutexes);
}

// TODO: RENAME issue_queue_stride ?
void SystemMemoryManager::issue_queue_push_back(uint32_t push_size_B, const uint8_t cq_id) {
    if (this->bypass_enable) {
        this->bypass_buffer_write_offset += push_size_B;
        return;
    }

    // All data needs to be PCIE_ALIGNMENT aligned
    uint32_t push_size_16B = align(push_size_B, tt::tt_metal::hal.get_alignment(tt::tt_metal::HalMemType::HOST)) >> 4;

    SystemMemoryCQInterface& cq_interface = this->cq_interfaces[cq_id];
    CoreType core_type = tt::tt_metal::dispatch_core_manager::instance().get_dispatch_core_type(this->device_id);
    uint32_t issue_q_wr_ptr =
        DispatchMemMap::get(core_type).get_host_command_queue_addr(CommandQueueHostAddrType::ISSUE_Q_WR);

    if (cq_interface.issue_fifo_wr_ptr + push_size_16B >= cq_interface.issue_fifo_limit) {
        cq_interface.issue_fifo_wr_ptr = (cq_interface.cq_start + cq_interface.offset) >> 4;  // In 16B words
        cq_interface.issue_fifo_wr_toggle = not cq_interface.issue_fifo_wr_toggle;            // Flip the toggle
    } else {
        cq_interface.issue_fifo_wr_ptr += push_size_16B;
    }

    // Also store this data in hugepages, so if a hang happens we can see what was written by host.
    chip_id_t mmio_device_id = tt::Cluster::instance().get_associated_mmio_device(this->device_id);
    uint16_t channel = tt::Cluster::instance().get_assigned_channel_for_device(this->device_id);
    tt::Cluster::instance().write_sysmem(
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
    chip_id_t mmio_device_id = tt::Cluster::instance().get_associated_mmio_device(this->device_id);
    uint16_t channel = tt::Cluster::instance().get_assigned_channel_for_device(this->device_id);
    CoreType core_type = tt::tt_metal::dispatch_core_manager::instance().get_dispatch_core_type(this->device_id);
    uint32_t completion_q_rd_ptr =
        DispatchMemMap::get(core_type).get_host_command_queue_addr(CommandQueueHostAddrType::COMPLETION_Q_RD);
    tt::Cluster::instance().write_sysmem(
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

    CoreType core_type = tt::tt_metal::dispatch_core_manager::instance().get_dispatch_core_type(device_id);
    const uint32_t prefetch_q_rd_ptr =
        DispatchMemMap::get(core_type).get_device_command_queue_addr(CommandQueueDeviceAddrType::PREFETCH_Q_RD);

    // Helper to wait for fetch queue space, if needed
    uint32_t fence;
    auto wait_for_fetch_q_space = [&]() {
        // Loop until space frees up
        while (this->prefetch_q_dev_ptrs[cq_id] == this->prefetch_q_dev_fences[cq_id]) {
            tt::Cluster::instance().read_core(
                &fence, sizeof(uint32_t), this->prefetcher_cores[cq_id], prefetch_q_rd_ptr);
            this->prefetch_q_dev_fences[cq_id] = fence;
        }
    };

    wait_for_fetch_q_space();

    // Wrap FetchQ if possible
    uint32_t prefetch_q_base =
        DispatchMemMap::get(core_type).get_device_command_queue_addr(CommandQueueDeviceAddrType::UNRESERVED);
    uint32_t prefetch_q_limit = prefetch_q_base + DispatchMemMap::get(core_type, num_hw_cqs).prefetch_q_entries() *
                                                      sizeof(DispatchSettings::prefetch_q_entry_type);
    if (this->prefetch_q_dev_ptrs[cq_id] == prefetch_q_limit) {
        this->prefetch_q_dev_ptrs[cq_id] = prefetch_q_base;
        wait_for_fetch_q_space();
    }
}

}  // namespace tt::tt_metal
