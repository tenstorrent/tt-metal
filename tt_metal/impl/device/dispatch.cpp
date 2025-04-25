// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "dispatch.hpp"
#include <cstdint>
#include "dispatch/device_command.hpp"
#include "dispatch/device_command_calculator.hpp"

namespace tt {
namespace tt_metal {

uint32_t calculate_max_prefetch_data_size_bytes(const CoreType& dispatch_core_type) {
    return tt::tt_metal::MetalContext::instance().dispatch_mem_map().max_prefetch_command_size() -
           (tt::tt_metal::MetalContext::instance().hal().get_alignment(HalMemType::HOST) *
            2);  // * 2 to account for issue
}

namespace device_dispatch {

void issue_l1_write_command_sequence(const L1WriteDispatchParams& dispatch_params) {
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

void issue_l1_read_command_sequence(const L1ReadDispatchParams& dispatch_params) {
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

void read_l1_data_from_completion_queue(
    const ReadL1DataDescriptor& read_descriptor,
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
