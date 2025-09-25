// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "dispatch.hpp"
#include <cstdint>
#include "dispatch/device_command.hpp"
#include "dispatch/device_command_calculator.hpp"
#include "dispatch/system_memory_manager.hpp"

namespace tt {
namespace tt_metal {

uint32_t calculate_max_prefetch_data_size_bytes(const CoreType& dispatch_core_type) {
    return tt::tt_metal::MetalContext::instance().dispatch_mem_map().max_prefetch_command_size() -
           (tt::tt_metal::MetalContext::instance().hal().get_alignment(HalMemType::HOST) *
            2);  // * 2 to account for issue
}

namespace device_dispatch {

struct CoreWriteDispatchParams : public CoreDispatchParams {
    const void* src = nullptr;
};

void validate_core_read_write_bounds(
    IDevice* device, const CoreCoord& virtual_core, DeviceAddr address, uint32_t size_bytes) {
    const HalMemType mem_type = device->get_mem_type_of_core(virtual_core);
    if (mem_type == HalMemType::L1) {
        const DeviceAddr l1_base_address = device->get_dev_addr(virtual_core, HalL1MemAddrType::BASE);
        const DeviceAddr l1_size = device->get_dev_size(virtual_core, HalL1MemAddrType::BASE);

        TT_FATAL(address >= l1_base_address, "Region in L1 is out of bounds");
        TT_FATAL(address + size_bytes <= l1_base_address + l1_size, "Region in L1 is out of bounds");
    } else {
        TT_ASSERT(mem_type == HalMemType::DRAM);

        auto& soc_desc = tt::tt_metal::MetalContext::instance().get_cluster().get_soc_desc(device->id());
        const uint32_t dram_channel = device->dram_channel_from_virtual_core(virtual_core);
        const DeviceAddr dram_base_address = soc_desc.get_address_offset(dram_channel);

        const DeviceAddr dram_channel_size = device->dram_size_per_channel();

        TT_FATAL(address >= dram_base_address, "Region in DRAM is out of bounds");
        TT_FATAL(address + size_bytes <= dram_base_address + dram_channel_size, "Region in DRAM is out of bounds");
    }
}

DeviceAddr add_bank_offset_to_address(IDevice* device, const CoreCoord& virtual_core, DeviceAddr address) {
    const HalMemType mem_type = device->get_mem_type_of_core(virtual_core);
    if (mem_type == HalMemType::DRAM) {
        auto& soc_desc = tt::tt_metal::MetalContext::instance().get_cluster().get_soc_desc(device->id());
        const uint32_t dram_channel = device->dram_channel_from_virtual_core(virtual_core);
        address += soc_desc.get_address_offset(dram_channel);
    }
    return address;
}

void issue_core_write_command_sequence(const CoreWriteDispatchParams& dispatch_params) {
    const uint32_t num_worker_counters = dispatch_params.sub_device_ids.size();

    DeviceCommandCalculator calculator;
    for (uint32_t i = 0; i < num_worker_counters; ++i) {
        calculator.add_dispatch_wait();
    }
    calculator.add_dispatch_write_linear<true, true>(dispatch_params.size_bytes);

    const uint32_t cmd_sequence_sizeB = calculator.write_offset_bytes();

    SystemMemoryManager& sysmem_manager = dispatch_params.device->sysmem_manager();
    void* cmd_region = sysmem_manager.issue_queue_reserve(cmd_sequence_sizeB, dispatch_params.cq_id);
    HugepageDeviceCommand command_sequence(cmd_region, cmd_sequence_sizeB);

    for (uint32_t i = 0; i < num_worker_counters; ++i) {
        const uint8_t offset_index = *dispatch_params.sub_device_ids[i];
        command_sequence.add_dispatch_wait(
            CQ_DISPATCH_CMD_WAIT_FLAG_WAIT_STREAM,
            0,
            tt::tt_metal::MetalContext::instance().dispatch_mem_map().get_dispatch_stream_index(offset_index),
            dispatch_params.expected_num_workers_completed[offset_index]);
    }

    command_sequence.add_dispatch_write_linear<true, true>(
        0,
        dispatch_params.device->get_noc_unicast_encoding(k_dispatch_downstream_noc, dispatch_params.virtual_core),
        dispatch_params.address,
        dispatch_params.size_bytes,
        (uint8_t*)dispatch_params.src);

    sysmem_manager.issue_queue_push_back(cmd_sequence_sizeB, dispatch_params.cq_id);
    sysmem_manager.fetch_queue_reserve_back(dispatch_params.cq_id);
    sysmem_manager.fetch_queue_write(cmd_sequence_sizeB, dispatch_params.cq_id);
}

void write_to_core(
    IDevice* device,
    const CoreCoord& virtual_core,
    const void* src,
    DeviceAddr address,
    uint32_t size_bytes,
    uint32_t cq_id,
    tt::stl::Span<const uint32_t> expected_num_workers_completed,
    tt::stl::Span<const SubDeviceId> sub_device_ids) {
    validate_core_read_write_bounds(device, virtual_core, address, size_bytes);

    while (size_bytes > 0) {
        const CoreType dispatch_core_type =
            MetalContext::instance().get_dispatch_core_manager().get_dispatch_core_type();
        const uint32_t size_bytes_to_write =
            std::min(size_bytes, calculate_max_prefetch_data_size_bytes(dispatch_core_type));

        CoreWriteDispatchParams dispatch_params{
            {virtual_core,
             address,
             size_bytes_to_write,
             device,
             cq_id,
             dispatch_core_type,
             expected_num_workers_completed,
             sub_device_ids},
            src};
        issue_core_write_command_sequence(dispatch_params);

        size_bytes -= size_bytes_to_write;
        address += size_bytes_to_write;
        src = (uint8_t*)src + size_bytes_to_write;
    }
}

void issue_core_read_command_sequence(const CoreReadDispatchParams& dispatch_params) {
    const uint32_t num_worker_counters = dispatch_params.sub_device_ids.size();
    DeviceCommandCalculator calculator;
    for (uint32_t i = 0; i < num_worker_counters - 1; ++i) {
        calculator.add_dispatch_wait();
    }
    calculator.add_dispatch_wait_with_prefetch_stall();
    calculator.add_dispatch_write_linear_host();
    calculator.add_prefetch_relay_linear();
    const uint32_t cmd_sequence_sizeB = calculator.write_offset_bytes();

    SystemMemoryManager& sysmem_manager = dispatch_params.device->sysmem_manager();
    void* cmd_region = sysmem_manager.issue_queue_reserve(cmd_sequence_sizeB, dispatch_params.cq_id);
    HugepageDeviceCommand command_sequence(cmd_region, cmd_sequence_sizeB);

    // We only need the write barrier + prefetch stall for the last wait cmd
    const uint32_t last_index = num_worker_counters - 1;
    for (uint32_t i = 0; i < last_index; ++i) {
        const uint8_t offset_index = *dispatch_params.sub_device_ids[i];
        command_sequence.add_dispatch_wait(
            CQ_DISPATCH_CMD_WAIT_FLAG_WAIT_STREAM,
            0,
            tt::tt_metal::MetalContext::instance().dispatch_mem_map().get_dispatch_stream_index(offset_index),
            dispatch_params.expected_num_workers_completed[offset_index]);
    }
    const uint8_t offset_index = *dispatch_params.sub_device_ids[last_index];
    command_sequence.add_dispatch_wait_with_prefetch_stall(
        CQ_DISPATCH_CMD_WAIT_FLAG_WAIT_STREAM | CQ_DISPATCH_CMD_WAIT_FLAG_BARRIER,
        0,
        tt::tt_metal::MetalContext::instance().dispatch_mem_map().get_dispatch_stream_index(offset_index),
        dispatch_params.expected_num_workers_completed[offset_index]);

    command_sequence.add_dispatch_write_host(false, dispatch_params.size_bytes, false);

    command_sequence.add_prefetch_relay_linear(
        dispatch_params.device->get_noc_unicast_encoding(k_dispatch_downstream_noc, dispatch_params.virtual_core),
        dispatch_params.size_bytes,
        dispatch_params.address);

    sysmem_manager.issue_queue_push_back(cmd_sequence_sizeB, dispatch_params.cq_id);
    sysmem_manager.fetch_queue_reserve_back(dispatch_params.cq_id);
    sysmem_manager.fetch_queue_write(cmd_sequence_sizeB, dispatch_params.cq_id);
}

void read_core_data_from_completion_queue(
    const ReadCoreDataDescriptor& read_descriptor,
    chip_id_t mmio_device_id,
    uint16_t channel,
    uint8_t cq_id,
    SystemMemoryManager& sysmem_manager,
    std::atomic<bool>& exit_condition) {
    uint32_t completion_queue_read_offset = sizeof(CQDispatchCmd);
    const uint32_t num_bytes_to_read = read_descriptor.size_bytes;
    uint32_t num_bytes_read = 0;
    while (num_bytes_read < num_bytes_to_read) {
        const uint32_t completion_queue_write_ptr_and_toggle =
            sysmem_manager.completion_queue_wait_front(cq_id, exit_condition);

        if (exit_condition) {
            break;
        }

        const uint32_t completion_q_write_ptr = (completion_queue_write_ptr_and_toggle & 0x7fffffff) << 4;
        const uint32_t completion_q_write_toggle = completion_queue_write_ptr_and_toggle >> (31);
        const uint32_t completion_q_read_ptr = sysmem_manager.get_completion_queue_read_ptr(cq_id);
        const uint32_t completion_q_read_toggle = sysmem_manager.get_completion_queue_read_toggle(cq_id);

        uint32_t num_bytes_available_in_completion_queue;
        if (completion_q_write_ptr > completion_q_read_ptr and completion_q_write_toggle == completion_q_read_toggle) {
            num_bytes_available_in_completion_queue = completion_q_write_ptr - completion_q_read_ptr;
        } else {
            // Completion queue write pointer on device wrapped but read pointer is lagging behind.
            //  In this case read up until the end of the completion queue first
            num_bytes_available_in_completion_queue =
                sysmem_manager.get_completion_queue_limit(cq_id) - completion_q_read_ptr;
        }

        const uint32_t num_bytes_to_copy = std::min(
            num_bytes_to_read - num_bytes_read, num_bytes_available_in_completion_queue - completion_queue_read_offset);

        tt::tt_metal::MetalContext::instance().get_cluster().read_sysmem(
            (char*)(uint64_t(read_descriptor.dst) + num_bytes_read),
            num_bytes_to_copy,
            completion_q_read_ptr + completion_queue_read_offset,
            mmio_device_id,
            channel);

        num_bytes_read += num_bytes_to_copy;
        const uint32_t num_pages_read =
            div_up(num_bytes_to_copy + completion_queue_read_offset, DispatchSettings::TRANSFER_PAGE_SIZE);
        sysmem_manager.completion_queue_pop_front(num_pages_read, cq_id);
        completion_queue_read_offset = 0;
    }
}

}  // namespace device_dispatch
}  // namespace tt_metal
}  // namespace tt
