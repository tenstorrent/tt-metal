// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "buffer_dispatch_utils.hpp"
#include "tt_metal/impl/dispatch/device_command.hpp"
#include "tt_metal/impl/device/device.hpp"

namespace tt::tt_metal {
namespace buffer_utils {

BufferDispatchConstants generate_buffer_dispatch_constants(
    const SystemMemoryManager& sysmem_manager, CoreType dispatch_core_type, uint32_t cq_id) {
    BufferDispatchConstants buf_dispatch_constants;

    buf_dispatch_constants.max_prefetch_cmd_size = sysmem_manager.get_issue_queue_limit(cq_id);
    buf_dispatch_constants.max_prefetch_cmd_size =
        dispatch_constants::get(dispatch_core_type).max_prefetch_command_size();
    buf_dispatch_constants.max_data_sizeB = buf_dispatch_constants.max_prefetch_cmd_size -
                                            (hal.get_alignment(HalMemType::HOST) * 2);  // * 2 to account for issue

    return buf_dispatch_constants;
}

ShardedBufferDispatchParams initialize_sharded_buf_dispatch_params(
    Buffer& buffer,
    uint32_t cq_id,
    std::array<uint32_t, dispatch_constants::DISPATCH_MESSAGE_ENTRIES> expected_num_workers_completed) {
    ShardedBufferDispatchParams dispatch_params;
    dispatch_params.width_split = buffer.shard_spec().shape_in_pages()[1] != buffer.shard_spec().tensor2d_shape[1];
    dispatch_params.buffer_page_mapping = (dispatch_params.width_split) ? buffer.get_buffer_page_mapping() : nullptr;
    dispatch_params.num_total_pages = buffer.num_pages();
    dispatch_params.max_pages_per_shard = buffer.shard_spec().size();
    dispatch_params.device = buffer.device();
    dispatch_params.cq_id = cq_id;
    dispatch_params.expected_num_workers_completed = expected_num_workers_completed;
    dispatch_params.dst_page_index = 0;
    dispatch_params.padded_page_size = buffer.aligned_page_size();
    return dispatch_params;
}

// InterleavedBufferDispatchParams generate_interleaved_buf_dispatch_params() {

// }

std::vector<CoreCoord> get_cores_for_sharded_buffer(
    const ShardedBufferDispatchParams& dispatch_params, Buffer& buffer) {
    return dispatch_params.width_split ? dispatch_params.buffer_page_mapping->all_cores_
                                       : corerange_to_cores(
                                             buffer.shard_spec().grid(),
                                             buffer.num_cores(),
                                             buffer.shard_spec().orientation() == ShardOrientation::ROW_MAJOR);
}

void populate_sharded_buffer_write_dispatch_cmds(
    const void* src,
    HugepageDeviceCommand& command_sequence,
    Buffer& buffer,
    ShardedBufferDispatchParams& dispatch_params) {
    uint32_t data_size_bytes = dispatch_params.pages_to_write * dispatch_params.padded_page_size;
    auto noc_index = dispatch_downstream_noc;
    const CoreCoord virtual_core =
        buffer.device()->virtual_core_from_logical_core(dispatch_params.core, buffer.core_type());
    command_sequence.add_dispatch_write_linear(
        0,
        buffer.device()->get_noc_unicast_encoding(noc_index, virtual_core),
        dispatch_params.address,
        data_size_bytes);

    if (dispatch_params.buffer_page_mapping) {
        const auto& page_mapping = *(dispatch_params.buffer_page_mapping);
        uint8_t* dst = command_sequence.reserve_space<uint8_t*, true>(data_size_bytes);
        // TODO: Expose getter for cmd_write_offsetB?
        uint32_t dst_offset = dst - (uint8_t*)command_sequence.data();
        for (uint32_t dev_page = dispatch_params.dst_page_index;
             dev_page < dispatch_params.dst_page_index + dispatch_params.pages_to_write;
             ++dev_page) {
            auto& host_page = page_mapping.dev_page_to_host_page_mapping_[dev_page];
            if (host_page.has_value()) {
                command_sequence.update_cmd_sequence(
                    dst_offset, (char*)(src) + host_page.value() * buffer.page_size(), buffer.page_size());
            }
            dst_offset += dispatch_params.padded_page_size;
        }
    } else {
        if (buffer.page_size() != dispatch_params.padded_page_size and buffer.page_size() != buffer.size()) {
            uint32_t unpadded_src_offset = dispatch_params.dst_page_index * buffer.page_size();
            for (uint32_t i = 0; i < dispatch_params.pages_to_write; ++i) {
                command_sequence.add_data(
                    (char*)src + unpadded_src_offset, buffer.page_size(), dispatch_params.padded_page_size);
                unpadded_src_offset += buffer.page_size();
            }
        } else {
            uint32_t unpadded_src_offset = dispatch_params.dst_page_index * buffer.page_size();
            command_sequence.add_data((char*)src + unpadded_src_offset, data_size_bytes, data_size_bytes);
        }
    }
}

void issue_buffer_dispatch_command_sequence(
    const void* src,
    Buffer& buffer,
    ShardedBufferDispatchParams& dispatch_params,
    tt::stl::Span<const SubDeviceId> sub_device_ids) {
    uint32_t num_worker_counters = sub_device_ids.size();
    uint32_t data_size_bytes = dispatch_params.pages_to_write * dispatch_params.padded_page_size;
    uint32_t pcie_alignment = hal.get_alignment(HalMemType::HOST);
    uint32_t cmd_sequence_sizeB = align(
        sizeof(CQPrefetchCmd) +      // CQ_PREFETCH_CMD_RELAY_INLINE
            sizeof(CQDispatchCmd) +  // CQ_DISPATCH_CMD_WRITE_PAGED or CQ_DISPATCH_CMD_WRITE_LINEAR
            data_size_bytes,
        pcie_alignment);
    if (dispatch_params.issue_wait) {
        cmd_sequence_sizeB += hal.get_alignment(HalMemType::HOST) *
                              num_worker_counters;  // CQ_PREFETCH_CMD_RELAY_INLINE + CQ_DISPATCH_CMD_WAIT
    }
    SystemMemoryManager& sysmem_manager = dispatch_params.device->sysmem_manager();
    void* cmd_region = sysmem_manager.issue_queue_reserve(cmd_sequence_sizeB, dispatch_params.cq_id);

    HugepageDeviceCommand command_sequence(cmd_region, cmd_sequence_sizeB);

    if (dispatch_params.issue_wait) {
        uint32_t dispatch_message_base_addr =
            dispatch_constants::get(CoreType::WORKER)
                .get_device_command_queue_addr(CommandQueueDeviceAddrType::DISPATCH_MESSAGE);
        for (const auto& sub_device_id : sub_device_ids) {
            auto offset_index = sub_device_id.to_index();
            uint32_t dispatch_message_addr =
                dispatch_message_base_addr +
                dispatch_constants::get(CoreType::WORKER).get_dispatch_message_offset(offset_index);
            command_sequence.add_dispatch_wait(
                false, dispatch_message_addr, dispatch_params.expected_num_workers_completed[offset_index]);
        }
    }
    populate_sharded_buffer_write_dispatch_cmds(src, command_sequence, buffer, dispatch_params);

    sysmem_manager.issue_queue_push_back(cmd_sequence_sizeB, dispatch_params.cq_id);
    sysmem_manager.fetch_queue_reserve_back(dispatch_params.cq_id);
    sysmem_manager.fetch_queue_write(cmd_sequence_sizeB, dispatch_params.cq_id);
}

void write_sharded_buffer_to_core(
    const void* src,
    uint32_t core_id,
    Buffer& buffer,
    ShardedBufferDispatchParams& dispatch_params,
    BufferDispatchConstants& buf_dispatch_constants,
    tt::stl::Span<const SubDeviceId> sub_device_ids,
    const std::vector<CoreCoord>& cores) {
    // Skip writing the padded pages along the bottom
    // Currently since writing sharded tensors uses write_linear, we write the padded pages on width
    // Alternative write each page row into separate commands, or have a strided linear write
    SystemMemoryManager& sysmem_manager = dispatch_params.device->sysmem_manager();
    uint32_t num_pages;
    if (dispatch_params.width_split) {
        num_pages = dispatch_params.buffer_page_mapping->core_shard_shape_[core_id][0] *
                    buffer.shard_spec().shape_in_pages()[1];
        if (num_pages == 0) {
            return;
        }
        dispatch_params.dst_page_index = dispatch_params.buffer_page_mapping->host_page_to_dev_page_mapping_
                                             [dispatch_params.buffer_page_mapping->core_host_page_indices_[core_id][0]];
    } else {
        num_pages = std::min(dispatch_params.num_total_pages, dispatch_params.max_pages_per_shard);
        dispatch_params.num_total_pages -= num_pages;
    }
    uint32_t curr_page_idx_in_shard = 0;
    uint32_t bank_base_address = buffer.address();
    if (buffer.is_dram()) {
        bank_base_address += buffer.device()->bank_offset(
            BufferType::DRAM, buffer.device()->dram_channel_from_logical_core(cores[core_id]));
    }

    uint32_t pages_to_write = 0;
    uint32_t address = 0;
    bool issue_wait = false;
    dispatch_params.core = cores[core_id];

    auto update_dispatch_params = [&]() {
        dispatch_params.address = address;
        dispatch_params.pages_to_write = pages_to_write;
        dispatch_params.issue_wait = issue_wait;
    };

    while (num_pages != 0) {
        // data appended after CQ_PREFETCH_CMD_RELAY_INLINE + CQ_DISPATCH_CMD_WRITE_PAGED
        uint32_t data_offset_bytes = (sizeof(CQPrefetchCmd) + sizeof(CQDispatchCmd));
        issue_wait = dispatch_params.dst_page_index == 0;  // only stall for the first write of the buffer
        if (issue_wait) {
            // commands prefixed with CQ_PREFETCH_CMD_RELAY_INLINE + CQ_DISPATCH_CMD_WAIT
            data_offset_bytes *= 2;
        }
        uint32_t space_available_bytes = std::min(
            buf_dispatch_constants.issue_queue_cmd_limit -
                sysmem_manager.get_issue_queue_write_ptr(dispatch_params.cq_id),
            buf_dispatch_constants.max_prefetch_cmd_size);
        int32_t num_pages_available =
            (int32_t(space_available_bytes) - int32_t(data_offset_bytes)) / int32_t(dispatch_params.padded_page_size);

        if (num_pages_available <= 0) {
            sysmem_manager.wrap_issue_queue_wr_ptr(dispatch_params.cq_id);
            continue;
        }

        pages_to_write = std::min(num_pages, (uint32_t)num_pages_available);
        address = bank_base_address + curr_page_idx_in_shard * dispatch_params.padded_page_size;

        tt::log_debug(tt::LogDispatch, "EnqueueWriteBuffer for channel {}", dispatch_params.cq_id);

        update_dispatch_params();
        issue_buffer_dispatch_command_sequence(src, buffer, dispatch_params, sub_device_ids);
        curr_page_idx_in_shard += pages_to_write;
        num_pages -= pages_to_write;
        dispatch_params.dst_page_index += pages_to_write;
    }
}

}  // namespace buffer_utils

}  // namespace tt::tt_metal
