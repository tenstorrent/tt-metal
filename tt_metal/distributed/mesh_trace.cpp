
// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
#include "mesh_trace.hpp"

#include <mesh_command_queue.hpp>
#include <mesh_coord.hpp>
#include <cstdint>
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

#include <tt_stl/assert.hpp>
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
#include "impl/allocator/allocator.hpp"
#include <distributed/mesh_device_impl.hpp>

namespace tt::tt_metal::distributed {

MeshTraceId MeshTrace::next_id() {
    static std::atomic<uint32_t> global_trace_id{0};
    return MeshTraceId(global_trace_id++);
}

void MeshTraceDescriptor::assemble_dispatch_commands(
    MeshDevice* mesh_device, const std::vector<MeshTraceStagingMetadata>& mesh_trace_md) {
    auto& trace_data = this->ordered_trace_data;
    // Track the trace region size per device range
    // The size of the allocated trace buffer will be the max
    // across all regions
    std::unordered_map<MeshCoordinateRange, uint32_t> trace_sizes;

    for (const auto& trace_md : mesh_trace_md) {
        const auto& sysmem_mgr_coord = trace_md.sysmem_manager_coord;
        auto& sysmem_manager = mesh_device->impl().get_device(sysmem_mgr_coord)->sysmem_manager();
        auto trace_data_word_offset = trace_md.offset / sizeof(uint32_t);
        auto trace_data_size_words = trace_md.size / sizeof(uint32_t);
        auto& bypass_data = sysmem_manager.get_bypass_data();
        bool intersection_found = false;

        std::vector<MeshTraceData> intermed_trace_data = {};
        std::vector<uint32_t> program_cmds_vector(
            std::make_move_iterator(bypass_data.begin() + trace_data_word_offset),
            std::make_move_iterator(bypass_data.begin() + trace_data_word_offset + trace_data_size_words));
        std::vector<MeshCoordinateRange> device_ranges_to_invalidate;
        for (auto& program : trace_data) {
            if (program.device_range.intersects(trace_md.device_range)) {
                // The current program intersects with a program that was previously
                // placed on the Mesh.
                intersection_found = true;
                auto intersection = *program.device_range.intersection(trace_md.device_range);
                if (intersection == program.device_range) {
                    // Intersection matches the originally placed program.
                    program.data.insert(
                        program.data.end(),
                        std::make_move_iterator(program_cmds_vector.begin()),
                        std::make_move_iterator(program_cmds_vector.end()));
                    // Update size of intersection
                    trace_sizes[intersection] += trace_md.size;
                } else {
                    // Intersection is a subset of the originally placed program.
                    auto complement = subtract(program.device_range, intersection);
                    for (const auto& complement_range : complement.ranges()) {
                        intermed_trace_data.push_back(MeshTraceData{complement_range, program.data});
                        // Trace region size of complement doesn't change
                        trace_sizes[complement_range] = trace_sizes[program.device_range];
                    }
                    intermed_trace_data.push_back(MeshTraceData{intersection, program.data});
                    // Update size of intersection
                    trace_sizes[intersection] = trace_sizes[program.device_range] + trace_md.size;
                    auto& intersection_data = intermed_trace_data.back().data;
                    intersection_data.insert(
                        intersection_data.end(),
                        std::make_move_iterator(program_cmds_vector.begin()),
                        std::make_move_iterator(program_cmds_vector.end()));
                    device_ranges_to_invalidate.push_back(program.device_range);
                }
            }
        }
        if (!intermed_trace_data.empty()) {
            // Invalidate programs with partial intersections with current programs.
            for (auto& program : trace_data) {
                if (std::find(
                        device_ranges_to_invalidate.begin(), device_ranges_to_invalidate.end(), program.device_range) ==
                    device_ranges_to_invalidate.end()) {
                    intermed_trace_data.push_back(std::move(program));
                } else {
                    // Clear trace size of invalid range
                    trace_sizes.erase(trace_sizes.find(program.device_range));
                }
            }
            trace_data = intermed_trace_data;
        }
        if (not intersection_found) {
            // Intersection not found, place program on Mesh.
            trace_data.push_back(MeshTraceData{trace_md.device_range, std::move(program_cmds_vector)});
            trace_sizes[trace_md.device_range] = trace_md.size;
        }
    }
    uint32_t max_trace_size = 0;
    for (const auto& size_per_range : trace_sizes) {
        max_trace_size = std::max(size_per_range.second, max_trace_size);
    }
    this->total_trace_size = max_trace_size;
    MeshCoordinateRange bcast_device_range(mesh_device->shape());
    std::vector<uint32_t> exec_buf_end = {};

    DeviceCommand command_sequence(MetalContext::instance().hal().get_alignment(HalMemType::HOST));
    command_sequence.add_prefetch_exec_buf_end();

    exec_buf_end.reserve(command_sequence.size_bytes() / sizeof(uint32_t));
    for (int i = 0; i < command_sequence.size_bytes() / sizeof(uint32_t); i++) {
        exec_buf_end.push_back(((uint32_t*)command_sequence.data())[i]);
    }

    for (auto& program : trace_data) {
        if (program.device_range.intersects(bcast_device_range)) {
            program.data.insert(program.data.end(), exec_buf_end.begin(), exec_buf_end.end());
        }
    }
    this->total_trace_size += command_sequence.size_bytes();

    this->sub_device_ids.reserve(this->descriptors.size());
    for (const auto& [id, _] : this->descriptors) {
        this->sub_device_ids.push_back(id);
    }
}

std::shared_ptr<MeshTraceBuffer> MeshTrace::create_empty_mesh_trace_buffer() {
    return std::make_shared<MeshTraceBuffer>(std::make_shared<MeshTraceDescriptor>(), nullptr);
}

void MeshTrace::populate_mesh_buffer(MeshCommandQueue& mesh_cq, std::shared_ptr<MeshTraceBuffer>& trace_buffer) {
    uint64_t unpadded_size = trace_buffer->desc->total_trace_size;
    size_t page_size = trace_dispatch::compute_interleaved_trace_buf_page_size(
        unpadded_size, mesh_cq.device()->allocator()->get_num_banks(BufferType::DRAM));
    size_t padded_size = round_up(unpadded_size, page_size);

    const auto current_trace_buffers_size = mesh_cq.device()->get_trace_buffers_size();
    mesh_cq.device()->set_trace_buffers_size(current_trace_buffers_size + padded_size);
    auto trace_region_size = mesh_cq.device()->allocator_impl()->get_config().trace_region_size;
    TT_FATAL(
        mesh_cq.device()->get_trace_buffers_size() <= trace_region_size,
        "Creating trace buffers of size {}B on MeshDevice {}, but only {}B is allocated for trace region.",
        mesh_cq.device()->get_trace_buffers_size(),
        mesh_cq.device()->id(),
        trace_region_size);

    DeviceLocalBufferConfig device_local_trace_buf_config = {
        .page_size = page_size,
        .buffer_type = BufferType::TRACE,
    };

    ReplicatedBufferConfig global_trace_buf_config = {
        .size = padded_size,
    };

    trace_buffer->mesh_buffer =
        MeshBuffer::create(global_trace_buf_config, device_local_trace_buf_config, mesh_cq.device());

    std::unordered_map<MeshCoordinateRange, uint32_t> write_offset_per_device_range = {};
    for (auto& mesh_trace_data : trace_buffer->desc->ordered_trace_data) {
        auto& device_range = mesh_trace_data.device_range;
        if (!write_offset_per_device_range.contains(device_range)) {
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

MeshTraceBuffer::~MeshTraceBuffer() {
    if (this->mesh_buffer && this->mesh_buffer->is_allocated() && this->mesh_buffer->device()) {
        auto current_trace_buffers_size = this->mesh_buffer->device()->get_trace_buffers_size();
        this->mesh_buffer->device()->set_trace_buffers_size(current_trace_buffers_size - this->mesh_buffer->size());
    }
}

}  // namespace tt::tt_metal::distributed
