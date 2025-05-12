// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <command_queue.hpp>
#include <command_queue_interface.hpp>
#include <stdint.h>
#include <sub_device_types.hpp>
#include <atomic>
#include <memory>
#include <variant>
#include <vector>

#include "buffer.hpp"
#include "core_coord.hpp"
#include <tt_stl/span.hpp>
#include "dispatch/system_memory_manager.hpp"

enum class CoreType;
namespace tt {
namespace tt_metal {
class IDevice;
enum class TensorMemoryLayout;
}  // namespace tt_metal
}  // namespace tt

namespace tt::tt_metal {

// Used so the host knows how to properly copy data into user space from the completion queue (in hugepages)
struct ReadBufferDescriptor {
    TensorMemoryLayout buffer_layout;
    uint32_t page_size;
    uint32_t padded_page_size;
    std::shared_ptr<const BufferPageMapping> buffer_page_mapping;
    void* dst;
    uint32_t dst_offset;
    uint32_t num_pages_read;
    uint32_t cur_dev_page_id;
    uint32_t starting_host_page_id;

    ReadBufferDescriptor(
        TensorMemoryLayout buffer_layout,
        uint32_t page_size,
        uint32_t padded_page_size,
        void* dst,
        uint32_t dst_offset,
        uint32_t num_pages_read,
        uint32_t cur_dev_page_id,
        uint32_t starting_host_page_id = 0,
        const std::shared_ptr<const BufferPageMapping>& buffer_page_mapping = nullptr) :
        buffer_layout(buffer_layout),
        page_size(page_size),
        padded_page_size(padded_page_size),
        buffer_page_mapping(buffer_page_mapping),
        dst(dst),
        dst_offset(dst_offset),
        num_pages_read(num_pages_read),
        cur_dev_page_id(cur_dev_page_id),
        starting_host_page_id(starting_host_page_id) {}
};

using CompletionReaderVariant =
    std::variant<std::monostate, ReadBufferDescriptor, ReadEventDescriptor, ReadCoreDataDescriptor>;

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
    uint32_t total_pages_to_read = 0;
    uint32_t total_pages_read = 0;
    uint32_t num_banks = 0;

    virtual ~BufferReadDispatchParams() = default;

    virtual void update_params_to_be_within_bounds(const Buffer& buffer) {
        const uint32_t num_pages_per_bank = this->src_page_index / this->num_banks;
        this->address += num_pages_per_bank * this->padded_page_size;
        this->src_page_index = this->src_page_index % this->num_banks;
    }

    virtual void calculate_num_pages_for_read_transaction() { this->pages_per_txn = this->total_pages_to_read; }

    virtual void update_params_after_read_transaction() {
        this->total_pages_to_read -= this->pages_per_txn;
        this->total_pages_read += this->pages_per_txn;
        this->src_page_index += this->pages_per_txn;
    }
};

struct PartialPageSpec {
    uint32_t partial_page_size = 0;
    uint32_t num_partial_pages_per_full_page = 0;
};

struct BufferReadLargePageDispatchParams : BufferReadDispatchParams {
    PartialPageSpec partial_page_spec;
};

using BufferReadDispatchParamsVariant = std::variant<BufferReadDispatchParams, BufferReadLargePageDispatchParams>;

struct ShardedBufferReadDispatchParams : BufferReadDispatchParams {
    bool width_split = false;
    uint32_t initial_pages_skipped = 0;
    uint32_t starting_src_host_page_index = 0;
    std::shared_ptr<const BufferPageMapping> buffer_page_mapping = nullptr;
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

BufferReadDispatchParamsVariant initialize_interleaved_buf_read_dispatch_params(
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
    const ReadBufferDescriptor& read_buffer_descriptor,
    chip_id_t mmio_device_id,
    uint16_t channel,
    uint32_t cq_id,
    SystemMemoryManager& sysmem_manager,
    std::atomic<bool>& exit_condition);

std::vector<CoreCoord> get_cores_for_sharded_buffer(
    bool width_split, const std::shared_ptr<const BufferPageMapping>& buffer_page_mapping, Buffer& buffer);

// Selects all sub-devices in the sub device stall group if none are specified
tt::stl::Span<const SubDeviceId> select_sub_device_ids(
    IDevice* device, tt::stl::Span<const SubDeviceId> sub_device_ids);

std::shared_ptr<::tt::tt_metal::CompletionReaderVariant> generate_sharded_buffer_read_descriptor(
    void* dst, ShardedBufferReadDispatchParams& dispatch_params, Buffer& buffer);
std::shared_ptr<::tt::tt_metal::CompletionReaderVariant> generate_interleaved_buffer_read_descriptor(
    void* dst, BufferReadDispatchParams* dispatch_params, Buffer& buffer);

bool are_pages_larger_than_max_prefetch_cmd_size(const Buffer& buffer);

PartialPageSpec calculate_partial_page_spec(const Buffer& buffer);
}  // namespace buffer_dispatch

}  // namespace tt::tt_metal
