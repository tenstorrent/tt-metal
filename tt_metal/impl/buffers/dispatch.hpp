// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <command_queue_interface.hpp>
#include <sub_device_types.hpp>
#include <command_queue.hpp>  // Need this for ReadBufferDesriptor -> this should be moved to a separate header
#include "buffer.hpp"

namespace tt::tt_metal {

// Contains helper functions to interface with buffers on device
namespace buffer_dispatch {

struct ShardedBufferReadDispatchParams {
    tt::stl::Span<const uint32_t> expected_num_workers_completed;
    uint32_t cq_id;
    IDevice* device;
    uint32_t padded_page_size;
    uint32_t src_page_index;
    uint32_t unpadded_dst_offset;
    bool width_split;
    std::shared_ptr<const BufferPageMapping> buffer_page_mapping;
    uint32_t num_total_pages;
    uint32_t max_pages_per_shard;
    CoreCoord core;
    uint32_t address;
    uint32_t num_pages_to_read;
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
    Buffer& buffer, uint32_t cq_id, tt::stl::Span<const uint32_t> expected_num_workers_completed);

void read_sharded_buffer_from_core(
    uint32_t core_id,
    Buffer& buffer,
    ShardedBufferReadDispatchParams& dispatch_params,
    tt::stl::Span<const SubDeviceId> sub_device_ids,
    const CoreCoord core,
    CoreType dispatch_core_type);

std::shared_ptr<::tt::tt_metal::detail::CompletionReaderVariant> generate_read_buffer_descriptor(
    void* dst, ShardedBufferReadDispatchParams& dispatch_params, Buffer& buffer);
}  // namespace buffer_dispatch

}  // namespace tt::tt_metal
