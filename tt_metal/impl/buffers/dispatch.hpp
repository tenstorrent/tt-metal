// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <command_queue_interface.hpp>
#include <sub_device_types.hpp>
#include <hardware_command_queue.hpp>  // Need this for ReadBufferDesriptor -> this should be moved to a separate header
#include "buffer.hpp"

namespace tt::tt_metal {

// Contains helper functions to interface with buffers on device
namespace buffer_dispatch {

struct BufferReadDispatchParams {
    tt::stl::Span<const uint32_t> expected_num_workers_completed;
    uint32_t cq_id = 0;
    IDevice* device = nullptr;
    uint32_t padded_page_size = 0;
    uint32_t src_page_index = 0;
    uint32_t unpadded_dst_offset = 0;
    uint32_t pages_per_txn = 0;
    uint32_t address = 0;
};

struct ShardedBufferReadDispatchParams : BufferReadDispatchParams {
    bool width_split = false;
    uint32_t initial_pages_skipped = 0;
    uint32_t starting_src_host_page_index = 0;
    std::shared_ptr<const BufferPageMapping> buffer_page_mapping = nullptr;
    uint32_t total_pages_to_read = 0;
    uint32_t total_pages_read = 0;
    uint32_t max_pages_per_shard = 0;
    CoreCoord core;
};

void write_to_device_buffer(
    const void* src,
    Buffer& buffer,
    const BufferRegion& region,
    uint32_t cq_id,
    tt::stl::Span<const uint32_t> expected_num_workers_completed,
    CoreType dispatch_core_type,
    tt::stl::Span<const SubDeviceId> sub_device_ids);

ShardedBufferReadDispatchParams initialize_sharded_buf_read_dispatch_params(
    Buffer& buffer,
    uint32_t cq_id,
    tt::stl::Span<const uint32_t> expected_num_workers_completed,
    const BufferRegion& region);

BufferReadDispatchParams initialize_interleaved_buf_read_dispatch_params(
    Buffer& buffer,
    uint32_t cq_id,
    tt::stl::Span<const uint32_t> expected_num_workers_completed,
    const BufferRegion& region);

void copy_sharded_buffer_from_core_to_completion_queue(
    uint32_t core_id,
    Buffer& buffer,
    ShardedBufferReadDispatchParams& dispatch_params,
    tt::stl::Span<const SubDeviceId> sub_device_ids,
    const CoreCoord core,
    CoreType dispatch_core_type);

void copy_interleaved_buffer_to_completion_queue(
    BufferReadDispatchParams& dispatch_params,
    Buffer& buffer,
    tt::stl::Span<const SubDeviceId> sub_device_ids,
    CoreType dispatch_core_type);

void copy_completion_queue_data_into_user_space(
    const detail::ReadBufferDescriptor& read_buffer_descriptor,
    chip_id_t mmio_device_id,
    uint16_t channel,
    uint32_t cq_id,
    SystemMemoryManager& sysmem_manager,
    volatile bool& exit_condition);

std::vector<CoreCoord> get_cores_for_sharded_buffer(
    bool width_split, const std::shared_ptr<const BufferPageMapping>& buffer_page_mapping, Buffer& buffer);

// Selects all sub-devices in the sub device stall group if none are specified
tt::stl::Span<const SubDeviceId> select_sub_device_ids(
    IDevice* device, tt::stl::Span<const SubDeviceId> sub_device_ids);

std::shared_ptr<::tt::tt_metal::detail::CompletionReaderVariant> generate_sharded_buffer_read_descriptor(
    void* dst, ShardedBufferReadDispatchParams& dispatch_params, Buffer& buffer);
std::shared_ptr<::tt::tt_metal::detail::CompletionReaderVariant> generate_interleaved_buffer_read_descriptor(
    void* dst, BufferReadDispatchParams& dispatch_params, Buffer& buffer);

}  // namespace buffer_dispatch

}  // namespace tt::tt_metal
