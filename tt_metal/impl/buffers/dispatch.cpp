// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <device_command.hpp>
#include <device.hpp>
#include "dispatch.hpp"

namespace tt::tt_metal {
namespace buffer_dispatch {

// Dispatch constants required for writing buffer data
struct BufferDispatchConstants {
    uint32_t issue_queue_cmd_limit;
    uint32_t max_prefetch_cmd_size;
    uint32_t max_data_sizeB;
};

// Dipatch parameters computed during runtime. These are used
// to assemble dispatch commands and compute src + dst offsets
// required to write buffer data.
struct BufferDispatchParams {
    tt::stl::Span<const uint32_t> expected_num_workers_completed;
    uint32_t address;
    uint32_t dst_page_index;
    uint32_t page_size_to_write;
    uint32_t total_pages_to_write;
    uint32_t pages_per_txn;
    bool issue_wait;
    IDevice* device;
    uint32_t cq_id;
};

// Parameters specific to interleaved buffers
struct InterleavedBufferDispatchParams : BufferDispatchParams {
    uint32_t write_partial_pages;
    uint32_t padded_buffer_size;
    uint32_t max_num_pages_to_write;
    uint32_t initial_src_addr_offset;
};

// Parameters specific to sharded buffers
struct ShardedBufferDispatchParams : BufferDispatchParams {
    bool width_split;
    std::shared_ptr<const BufferPageMapping> buffer_page_mapping;
    uint32_t max_pages_per_shard;
    CoreCoord core;
};

// Generate dispatch constants
BufferDispatchConstants generate_buffer_dispatch_constants(
    const SystemMemoryManager& sysmem_manager, CoreType dispatch_core_type, uint32_t cq_id) {
    BufferDispatchConstants buf_dispatch_constants;

    buf_dispatch_constants.issue_queue_cmd_limit = sysmem_manager.get_issue_queue_limit(cq_id);
    buf_dispatch_constants.max_prefetch_cmd_size =
        dispatch_constants::get(dispatch_core_type).max_prefetch_command_size();
    buf_dispatch_constants.max_data_sizeB = buf_dispatch_constants.max_prefetch_cmd_size -
                                            (hal.get_alignment(HalMemType::HOST) * 2);  // * 2 to account for issue

    return buf_dispatch_constants;
}

// Initialize Dispatch Parameters - reused across write txns
ShardedBufferDispatchParams initialize_sharded_buf_dispatch_params(
    Buffer& buffer,
    uint32_t cq_id,
    tt::stl::Span<const uint32_t> expected_num_workers_completed,
    const BufferDispatchConstants& buf_dispatch_constants) {
    ShardedBufferDispatchParams dispatch_params;
    dispatch_params.width_split = buffer.shard_spec().shape_in_pages()[1] != buffer.shard_spec().tensor2d_shape[1];
    dispatch_params.buffer_page_mapping = (dispatch_params.width_split) ? buffer.get_buffer_page_mapping() : nullptr;
    dispatch_params.total_pages_to_write = buffer.num_pages();
    dispatch_params.max_pages_per_shard = buffer.shard_spec().size();
    dispatch_params.page_size_to_write = buffer.aligned_page_size();
    dispatch_params.dst_page_index = 0;
    dispatch_params.device = buffer.device();
    dispatch_params.cq_id = cq_id;
    dispatch_params.expected_num_workers_completed = expected_num_workers_completed;

    TT_FATAL(
        buf_dispatch_constants.max_data_sizeB >= dispatch_params.page_size_to_write,
        "Writing padded page size > {} is currently unsupported for sharded tensors.",
        buf_dispatch_constants.max_data_sizeB);
    return dispatch_params;
}

InterleavedBufferDispatchParams initialize_interleaved_buf_dispatch_params(
    Buffer& buffer,
    const BufferDispatchConstants& buf_dispatch_constants,
    uint32_t cq_id,
    tt::stl::Span<const uint32_t> expected_num_workers_completed,
    const BufferRegion& region) {
    if (buffer.is_valid_partial_region(region)) {
        TT_FATAL(
            region.offset % buffer.page_size() == 0,
            "Offset {} must be divisible by the buffer page size {}.",
            region.offset,
            buffer.page_size());
        TT_FATAL(
            region.size % buffer.page_size() == 0,
            "Size {} must be divisible by the buffer page size {}.",
            region.size,
            buffer.page_size());
        TT_FATAL(
            (region.size + region.offset) <= buffer.size(),
            "(Size + offset) {} must be <= the buffer size {}.",
            region.size + region.offset,
            buffer.size());
    }

    InterleavedBufferDispatchParams dispatch_params;
    dispatch_params.dst_page_index = region.offset / buffer.page_size();
    uint32_t num_pages = region.size / buffer.page_size();

    uint32_t padded_page_size = buffer.aligned_page_size();
    dispatch_params.total_pages_to_write = num_pages;
    dispatch_params.write_partial_pages = padded_page_size > buf_dispatch_constants.max_data_sizeB;
    dispatch_params.page_size_to_write = padded_page_size;
    dispatch_params.padded_buffer_size = num_pages * padded_page_size;

    if (dispatch_params.write_partial_pages) {
        TT_FATAL(num_pages == 1, "TODO: add support for multi-paged buffer with page size > 64KB");
        uint32_t partial_size = dispatch_constants::BASE_PARTIAL_PAGE_SIZE;
        uint32_t pcie_alignment = hal.get_alignment(HalMemType::HOST);
        while (dispatch_params.padded_buffer_size % partial_size != 0) {
            partial_size += pcie_alignment;
        }
        dispatch_params.page_size_to_write = partial_size;
        dispatch_params.total_pages_to_write = dispatch_params.padded_buffer_size / dispatch_params.page_size_to_write;
    }
    const uint32_t num_banks = buffer.device()->num_banks(buffer.buffer_type());
    const uint32_t num_pages_round_robined = num_pages / num_banks;
    const uint32_t num_banks_with_residual_pages = num_pages % num_banks;
    const uint32_t num_partial_pages_per_page = padded_page_size / dispatch_params.page_size_to_write;
    const uint32_t num_partials_round_robined = num_partial_pages_per_page * num_pages_round_robined;

    dispatch_params.max_num_pages_to_write =
        (dispatch_params.write_partial_pages)
            ? (num_pages_round_robined > 0 ? (num_banks * num_partials_round_robined) : num_banks_with_residual_pages)
            : dispatch_params.total_pages_to_write;
    dispatch_params.address = buffer.address();
    dispatch_params.device = buffer.device();
    dispatch_params.cq_id = cq_id;
    dispatch_params.expected_num_workers_completed = expected_num_workers_completed;
    return dispatch_params;
}

// Populate/Assemble dispatch commands for writing buffer data
void populate_interleaved_buffer_write_dispatch_cmds(
    const void* src,
    HugepageDeviceCommand& command_sequence,
    Buffer& buffer,
    InterleavedBufferDispatchParams& dispatch_params) {
    uint8_t is_dram = uint8_t(buffer.is_dram());
    TT_ASSERT(
        dispatch_params.dst_page_index <= 0xFFFF,
        "Page offset needs to fit within range of uint16_t, bank_base_address was computed incorrectly!");
    uint16_t start_page = uint16_t(dispatch_params.dst_page_index & 0xFFFF);
    bool flush_prefetch = true;
    command_sequence.add_dispatch_write_paged(
        flush_prefetch,
        is_dram,
        start_page,
        dispatch_params.address,
        dispatch_params.page_size_to_write,
        dispatch_params.pages_per_txn);

    uint32_t data_size_bytes = dispatch_params.pages_per_txn * dispatch_params.page_size_to_write;
    uint32_t full_page_size = buffer.aligned_page_size();  // dispatch_params.page_size_to_write could be a partial
                                                           // page if buffer page size > MAX_PREFETCH_CMD_SIZE
    bool write_partial_pages = dispatch_params.page_size_to_write < full_page_size;
    uint32_t buffer_addr_offset = dispatch_params.address - buffer.address();
    const uint32_t num_banks = buffer.device()->num_banks(buffer.buffer_type());

    // TODO: Consolidate
    if (write_partial_pages) {
        uint32_t padding = full_page_size - buffer.page_size();
        uint32_t src_address_offset = dispatch_params.initial_src_addr_offset;
        for (uint32_t sysmem_address_offset = 0; sysmem_address_offset < data_size_bytes;
             sysmem_address_offset += dispatch_params.page_size_to_write) {
            uint32_t page_size_to_copy = dispatch_params.page_size_to_write;
            if (src_address_offset + dispatch_params.page_size_to_write > buffer.page_size()) {
                // last partial page being copied from unpadded src buffer
                page_size_to_copy -= padding;
            }
            command_sequence.add_data(
                (char*)src + src_address_offset, page_size_to_copy, dispatch_params.page_size_to_write);
            src_address_offset += page_size_to_copy;
        }
    } else {
        uint32_t unpadded_src_offset =
            (((buffer_addr_offset / dispatch_params.page_size_to_write) * num_banks) + dispatch_params.dst_page_index) *
            buffer.page_size();
        if (buffer.page_size() % buffer.alignment() != 0 and buffer.page_size() != buffer.size()) {
            // If page size is not aligned, we cannot do a contiguous write
            uint32_t src_address_offset = dispatch_params.initial_src_addr_offset;
            for (uint32_t sysmem_address_offset = 0; sysmem_address_offset < data_size_bytes;
                 sysmem_address_offset += dispatch_params.page_size_to_write) {
                command_sequence.add_data(
                    (char*)src + src_address_offset, buffer.page_size(), dispatch_params.page_size_to_write);
                src_address_offset += buffer.page_size();
            }
        } else {
            command_sequence.add_data(
                (char*)src + dispatch_params.initial_src_addr_offset, data_size_bytes, data_size_bytes);
        }
    }
}

void populate_sharded_buffer_write_dispatch_cmds(
    const void* src,
    HugepageDeviceCommand& command_sequence,
    Buffer& buffer,
    ShardedBufferDispatchParams& dispatch_params) {
    uint32_t data_size_bytes = dispatch_params.pages_per_txn * dispatch_params.page_size_to_write;
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
             dev_page < dispatch_params.dst_page_index + dispatch_params.pages_per_txn;
             ++dev_page) {
            auto& host_page = page_mapping.dev_page_to_host_page_mapping_[dev_page];
            if (host_page.has_value()) {
                command_sequence.update_cmd_sequence(
                    dst_offset, (char*)(src) + host_page.value() * buffer.page_size(), buffer.page_size());
            }
            dst_offset += dispatch_params.page_size_to_write;
        }
    } else {
        if (buffer.page_size() != dispatch_params.page_size_to_write and buffer.page_size() != buffer.size()) {
            uint32_t unpadded_src_offset = dispatch_params.dst_page_index * buffer.page_size();
            for (uint32_t i = 0; i < dispatch_params.pages_per_txn; ++i) {
                command_sequence.add_data(
                    (char*)src + unpadded_src_offset, buffer.page_size(), dispatch_params.page_size_to_write);
                unpadded_src_offset += buffer.page_size();
            }
        } else {
            uint32_t unpadded_src_offset = dispatch_params.dst_page_index * buffer.page_size();
            command_sequence.add_data((char*)src + unpadded_src_offset, data_size_bytes, data_size_bytes);
        }
    }
}

// Issue dispatch commands for writing buffer data
template <typename T>
void issue_buffer_dispatch_command_sequence(
    const void* src,
    Buffer& buffer,
    T& dispatch_params,
    tt::stl::Span<const SubDeviceId> sub_device_ids,
    CoreType dispatch_core_type) {
    uint32_t num_worker_counters = sub_device_ids.size();
    uint32_t data_size_bytes = dispatch_params.pages_per_txn * dispatch_params.page_size_to_write;
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
            dispatch_constants::get(dispatch_core_type)
                .get_device_command_queue_addr(CommandQueueDeviceAddrType::DISPATCH_MESSAGE);
        for (const auto& sub_device_id : sub_device_ids) {
            auto offset_index = sub_device_id.to_index();
            uint32_t dispatch_message_addr =
                dispatch_message_base_addr +
                dispatch_constants::get(dispatch_core_type).get_dispatch_message_offset(offset_index);
            command_sequence.add_dispatch_wait(
                false, dispatch_message_addr, dispatch_params.expected_num_workers_completed[offset_index]);
        }
    }
    if constexpr (std::is_same_v<T, ShardedBufferDispatchParams>) {
        populate_sharded_buffer_write_dispatch_cmds(src, command_sequence, buffer, dispatch_params);
    } else {
        populate_interleaved_buffer_write_dispatch_cmds(src, command_sequence, buffer, dispatch_params);
    }

    sysmem_manager.issue_queue_push_back(cmd_sequence_sizeB, dispatch_params.cq_id);
    sysmem_manager.fetch_queue_reserve_back(dispatch_params.cq_id);
    sysmem_manager.fetch_queue_write(cmd_sequence_sizeB, dispatch_params.cq_id);
}

// Top level helper functions to write buffer data
void write_interleaved_buffer_to_device(
    const void* src,
    InterleavedBufferDispatchParams& dispatch_params,
    Buffer& buffer,
    const BufferDispatchConstants& buf_dispatch_constants,
    tt::stl::Span<const SubDeviceId> sub_device_ids,
    CoreType dispatch_core_type) {
    SystemMemoryManager& sysmem_manager = dispatch_params.device->sysmem_manager();
    uint32_t data_offsetB = hal.get_alignment(HalMemType::HOST);  // data appended after CQ_PREFETCH_CMD_RELAY_INLINE
                                                                  // + CQ_DISPATCH_CMD_WRITE_PAGED
    const uint32_t orig_dst_page_index = dispatch_params.dst_page_index;
    uint32_t total_num_pages_written = 0;
    while (dispatch_params.total_pages_to_write > 0) {
        dispatch_params.issue_wait =
            (dispatch_params.dst_page_index == orig_dst_page_index and
             dispatch_params.address == buffer.address());  // only stall for the first write of the buffer
        if (dispatch_params.issue_wait) {
            data_offsetB *= 2;  // commands prefixed with CQ_PREFETCH_CMD_RELAY_INLINE + CQ_DISPATCH_CMD_WAIT
        }

        uint32_t space_availableB = std::min(
            buf_dispatch_constants.issue_queue_cmd_limit -
                sysmem_manager.get_issue_queue_write_ptr(dispatch_params.cq_id),
            buf_dispatch_constants.max_prefetch_cmd_size);
        int32_t num_pages_available =
            (int32_t(space_availableB) - int32_t(data_offsetB)) / int32_t(dispatch_params.page_size_to_write);

        if (num_pages_available <= 0) {
            sysmem_manager.wrap_issue_queue_wr_ptr(dispatch_params.cq_id);
            continue;
        }

        dispatch_params.pages_per_txn = std::min(
            std::min((uint32_t)num_pages_available, dispatch_params.max_num_pages_to_write),
            dispatch_params.total_pages_to_write);

        // Page offset in CQ_DISPATCH_CMD_WRITE_PAGED is uint16_t
        // To handle larger page offsets move bank base address up and update page offset to be relative to the new
        // bank address
        if (dispatch_params.dst_page_index > 0xFFFF or
            (dispatch_params.pages_per_txn == dispatch_params.max_num_pages_to_write and
             dispatch_params.write_partial_pages)) {
            uint32_t num_banks = buffer.device()->num_banks(buffer.buffer_type());
            uint32_t num_banks_to_use =
                dispatch_params.write_partial_pages ? dispatch_params.max_num_pages_to_write : num_banks;
            uint32_t residual = dispatch_params.dst_page_index % num_banks_to_use;
            uint32_t num_pages_written_per_bank = dispatch_params.dst_page_index / num_banks_to_use;
            dispatch_params.address += num_pages_written_per_bank * dispatch_params.page_size_to_write;
            dispatch_params.dst_page_index = residual;
        }
        dispatch_params.initial_src_addr_offset = dispatch_params.write_partial_pages
                                                      ? dispatch_params.address - buffer.address()
                                                      : total_num_pages_written * buffer.page_size();

        tt::log_debug(tt::LogDispatch, "EnqueueWriteBuffer for command queue {}", dispatch_params.cq_id);

        issue_buffer_dispatch_command_sequence(src, buffer, dispatch_params, sub_device_ids, dispatch_core_type);
        total_num_pages_written += dispatch_params.pages_per_txn;
        dispatch_params.total_pages_to_write -= dispatch_params.pages_per_txn;
        dispatch_params.dst_page_index += dispatch_params.pages_per_txn;
    }
}

std::vector<CoreCoord> get_cores_for_sharded_buffer(
    const ShardedBufferDispatchParams& dispatch_params, Buffer& buffer) {
    return dispatch_params.width_split ? dispatch_params.buffer_page_mapping->all_cores_
                                       : corerange_to_cores(
                                             buffer.shard_spec().grid(),
                                             buffer.num_cores(),
                                             buffer.shard_spec().orientation() == ShardOrientation::ROW_MAJOR);
}

void write_sharded_buffer_to_core(
    const void* src,
    uint32_t core_id,
    Buffer& buffer,
    ShardedBufferDispatchParams& dispatch_params,
    const BufferDispatchConstants& buf_dispatch_constants,
    tt::stl::Span<const SubDeviceId> sub_device_ids,
    const CoreCoord core,
    CoreType dispatch_core_type) {
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
        num_pages = std::min(dispatch_params.total_pages_to_write, dispatch_params.max_pages_per_shard);
        dispatch_params.total_pages_to_write -= num_pages;
    }
    uint32_t curr_page_idx_in_shard = 0;
    uint32_t bank_base_address = buffer.address();
    if (buffer.is_dram()) {
        bank_base_address +=
            buffer.device()->bank_offset(BufferType::DRAM, buffer.device()->dram_channel_from_logical_core(core));
    }

    while (num_pages != 0) {
        // data appended after CQ_PREFETCH_CMD_RELAY_INLINE + CQ_DISPATCH_CMD_WRITE_PAGED
        uint32_t data_offset_bytes = (sizeof(CQPrefetchCmd) + sizeof(CQDispatchCmd));
        dispatch_params.issue_wait =
            dispatch_params.dst_page_index == 0;  // only stall for the first write of the buffer
        if (dispatch_params.issue_wait) {
            // commands prefixed with CQ_PREFETCH_CMD_RELAY_INLINE + CQ_DISPATCH_CMD_WAIT
            data_offset_bytes *= 2;
        }
        uint32_t space_available_bytes = std::min(
            buf_dispatch_constants.issue_queue_cmd_limit -
                sysmem_manager.get_issue_queue_write_ptr(dispatch_params.cq_id),
            buf_dispatch_constants.max_prefetch_cmd_size);
        int32_t num_pages_available =
            (int32_t(space_available_bytes) - int32_t(data_offset_bytes)) / int32_t(dispatch_params.page_size_to_write);

        if (num_pages_available <= 0) {
            sysmem_manager.wrap_issue_queue_wr_ptr(dispatch_params.cq_id);
            continue;
        }

        dispatch_params.pages_per_txn = std::min(num_pages, (uint32_t)num_pages_available);
        dispatch_params.address = bank_base_address + curr_page_idx_in_shard * dispatch_params.page_size_to_write;
        dispatch_params.core = core;

        tt::log_debug(tt::LogDispatch, "EnqueueWriteBuffer for channel {}", dispatch_params.cq_id);

        issue_buffer_dispatch_command_sequence(src, buffer, dispatch_params, sub_device_ids, dispatch_core_type);
        curr_page_idx_in_shard += dispatch_params.pages_per_txn;
        num_pages -= dispatch_params.pages_per_txn;
        dispatch_params.dst_page_index += dispatch_params.pages_per_txn;
    }
}

// Main API to write buffer data
void write_to_device_buffer(
    const void* src,
    Buffer& buffer,
    const BufferRegion& region,
    uint32_t cq_id,
    tt::stl::Span<const uint32_t> expected_num_workers_completed,
    CoreType dispatch_core_type,
    tt::stl::Span<const SubDeviceId> sub_device_ids) {
    SystemMemoryManager& sysmem_manager = buffer.device()->sysmem_manager();
    const BufferDispatchConstants buf_dispatch_constants =
        generate_buffer_dispatch_constants(sysmem_manager, dispatch_core_type, cq_id);

    if (is_sharded(buffer.buffer_layout())) {
        ShardedBufferDispatchParams dispatch_params = initialize_sharded_buf_dispatch_params(
            buffer, cq_id, expected_num_workers_completed, buf_dispatch_constants);
        const auto cores = get_cores_for_sharded_buffer(dispatch_params, buffer);
        // Since we read core by core we are reading the device pages sequentially
        for (uint32_t core_id = 0; core_id < buffer.num_cores(); ++core_id) {
            write_sharded_buffer_to_core(
                src,
                core_id,
                buffer,
                dispatch_params,
                buf_dispatch_constants,
                sub_device_ids,
                cores[core_id],
                dispatch_core_type);
        }
    } else {
        InterleavedBufferDispatchParams dispatch_params = initialize_interleaved_buf_dispatch_params(
            buffer, buf_dispatch_constants, cq_id, expected_num_workers_completed, region);
        write_interleaved_buffer_to_device(
            src, dispatch_params, buffer, buf_dispatch_constants, sub_device_ids, dispatch_core_type);
    }
}

template void issue_buffer_dispatch_command_sequence<InterleavedBufferDispatchParams>(
    const void*, Buffer&, InterleavedBufferDispatchParams&, tt::stl::Span<const SubDeviceId>, CoreType);
template void issue_buffer_dispatch_command_sequence<ShardedBufferDispatchParams>(
    const void*, Buffer&, ShardedBufferDispatchParams&, tt::stl::Span<const SubDeviceId>, CoreType);
}  // namespace buffer_dispatch

}  // namespace tt::tt_metal
