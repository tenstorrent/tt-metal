
// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0
#include "mesh_trace.hpp"

#include <boost/move/utility_core.hpp>
#include <mesh_command_queue.hpp>
#include <mesh_coord.hpp>
#include <stdint.h>
#include <tt-metalium/allocator.hpp>
#include <algorithm>
#include <atomic>
#include <cstddef>
#include <functional>
#include <iterator>
#include <memory>
#include <optional>
#include <unordered_map>
#include <utility>
#include <vector>

#include "allocator_types.hpp"
#include "assert.hpp"
#include "buffer.hpp"
#include "buffer_types.hpp"
#include "device.hpp"
#include "hal.hpp"
#include "hal_types.hpp"
#include "math.hpp"
#include "mesh_buffer.hpp"
#include "mesh_device.hpp"
#include "mesh_trace_id.hpp"
#include "dispatch/system_memory_manager.hpp"
#include "trace/trace_buffer.hpp"
#include "tt_metal/impl/dispatch/device_command.hpp"
#include "tt_metal/impl/trace/dispatch.hpp"

namespace tt::tt_metal::distributed {

MeshTraceId MeshTrace::next_id() {
    static std::atomic<uint32_t> global_trace_id{0};
    return MeshTraceId(global_trace_id++);
}

std::shared_ptr<MeshTraceBuffer> MeshTrace::create_empty_mesh_trace_buffer() {
    return std::make_shared<MeshTraceBuffer>(std::make_shared<MeshTraceDescriptor>(), nullptr);
}

void MeshTrace::populate_mesh_buffer(MeshCommandQueue& mesh_cq, std::shared_ptr<MeshTraceBuffer>& trace_buffer) {
    auto mesh_device = mesh_cq.device();
    uint64_t unpadded_size = trace_buffer->desc->total_trace_size;
    size_t page_size = trace_dispatch::compute_interleaved_trace_buf_page_size(
        unpadded_size, mesh_cq.device()->allocator()->get_num_banks(BufferType::DRAM));
    size_t padded_size = round_up(unpadded_size, page_size);

    const auto current_trace_buffers_size = mesh_cq.device()->get_trace_buffers_size();
    mesh_cq.device()->set_trace_buffers_size(current_trace_buffers_size + padded_size);
    auto trace_region_size = mesh_cq.device()->allocator()->get_config().trace_region_size;
    TT_FATAL(
        mesh_cq.device()->get_trace_buffers_size() <= trace_region_size,
        "Creating trace buffers of size {}B on MeshDevice {}, but only {}B is allocated for trace region.",
        mesh_cq.device()->get_trace_buffers_size(),
        mesh_cq.device()->id(),
        trace_region_size);

    DeviceLocalBufferConfig device_local_trace_buf_config = {
        .page_size = page_size,
        .buffer_type = BufferType::TRACE,
        .buffer_layout = TensorMemoryLayout::INTERLEAVED,
    };

    ReplicatedBufferConfig global_trace_buf_config = {
        .size = padded_size,
    };

    trace_buffer->mesh_buffer =
        MeshBuffer::create(global_trace_buf_config, device_local_trace_buf_config, mesh_cq.device());

    std::unordered_map<MeshCoordinateRange, uint32_t> write_offset_per_device_range = {};
    for (auto& mesh_trace_data : trace_buffer->desc->ordered_trace_data) {
        auto& device_range = mesh_trace_data.device_range;
        if (write_offset_per_device_range.find(device_range) == write_offset_per_device_range.end()) {
            write_offset_per_device_range.insert({device_range, 0});
        }
        std::vector<uint32_t> write_data = mesh_trace_data.data;
        auto unpadded_data_size = write_data.size() * sizeof(uint32_t);
        auto padded_data_size = round_up(unpadded_data_size, page_size);
        size_t numel_padding = (padded_data_size - unpadded_data_size) / sizeof(uint32_t);
        if (numel_padding > 0) {
            write_data.resize(write_data.size() + numel_padding, 0);
        }
        auto write_region =
            BufferRegion(write_offset_per_device_range.at(device_range), write_data.size() * sizeof(uint32_t));
        mesh_cq.enqueue_write_shard_to_sub_grid(
            *(trace_buffer->mesh_buffer), write_data.data(), device_range, true, write_region);
        write_offset_per_device_range.at(device_range) += mesh_trace_data.data.size() * sizeof(uint32_t);
    }
}

}  // namespace tt::tt_metal::distributed
