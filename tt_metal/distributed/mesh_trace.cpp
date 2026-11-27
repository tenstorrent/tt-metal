
// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
#include "mesh_trace.hpp"

#include <mesh_command_queue.hpp>
#include <mesh_coord.hpp>
#include <tt-metalium/shard_data_transfer.hpp>
#include <cstdint>
#include <tt-metalium/allocator.hpp>
#include <algorithm>
#include <atomic>
#include <cstddef>
#include <cstdlib>
#include <memory>
#include <optional>
#include <unordered_map>
#include <vector>

#include <tt_stl/assert.hpp>
#include <tt-metalium/experimental/allocation_context.hpp>
#include "buffer.hpp"
#include "buffer_types.hpp"
#include "device.hpp"
#include <tt-metalium/hal_types.hpp>
#include "math.hpp"
#include "mesh_buffer.hpp"
#include "mesh_device.hpp"
#include "mesh_trace_id.hpp"
#include "tt_metal/impl/trace/dispatch.hpp"
#include "impl/allocator/allocator.hpp"
#include <distributed/mesh_device_impl.hpp>

namespace tt::tt_metal::distributed {

namespace {

// TT_METAL_TRACE_WRITE_SERIAL=1 restores the old trace write: one blocking write per device range, in order. Kept
// for A/B measurement on one build.
bool trace_write_serial() {
    static const bool serial = [] {
        const char* v = std::getenv("TT_METAL_TRACE_WRITE_SERIAL");
        return v != nullptr && v[0] != '\0' && v[0] != '0';
    }();
    return serial;
}

void populate_mesh_buffer_serial(
    MeshCommandQueue& mesh_cq, std::shared_ptr<MeshTraceBuffer>& trace_buffer, size_t page_size) {
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
        if (mesh_trace_data.device_patches.empty()) {
            mesh_cq.enqueue_write_shard_to_sub_grid(
                *(trace_buffer->mesh_buffer), write_data.data(), device_range, true, write_region);
        } else {
            // Per-device runtime args: the stream goes to the whole range once; then each device gets only the pages
            // that hold its own words, patched.
            mesh_cq.enqueue_write_shard_to_sub_grid(
                *(trace_buffer->mesh_buffer), write_data.data(), device_range, true, write_region);
            const size_t page_words = page_size / sizeof(uint32_t);
            std::vector<uint32_t> pages;
            std::vector<uint32_t> run;
            for (const auto& coord : device_range) {
                if (!mesh_cq.device()->impl().is_local(coord)) {
                    continue;
                }
                pages.clear();
                for (const auto& patch : mesh_trace_data.device_patches) {
                    if (patch.device != coord || patch.words.empty()) {
                        continue;
                    }
                    TT_FATAL(
                        patch.byte_offset % sizeof(uint32_t) == 0 &&
                            patch.byte_offset + patch.words.size() * sizeof(uint32_t) <=
                                mesh_trace_data.data.size() * sizeof(uint32_t),
                        "Per-device runtime args trace patch out of bounds");
                    const uint32_t first = patch.byte_offset / page_size;
                    const uint32_t last = (patch.byte_offset + patch.words.size() * sizeof(uint32_t) - 1) / page_size;
                    for (uint32_t page = first; page <= last; ++page) {
                        pages.push_back(page);
                    }
                }
                std::sort(pages.begin(), pages.end());
                pages.erase(std::unique(pages.begin(), pages.end()), pages.end());
                // Each run of consecutive pages is one write of the device's patched words.
                for (size_t a = 0; a < pages.size();) {
                    size_t b = a + 1;
                    while (b < pages.size() && pages[b] == pages[b - 1] + 1) {
                        ++b;
                    }
                    const size_t begin_word = pages[a] * page_words;
                    const size_t end_word = (pages[b - 1] + 1) * page_words;
                    run.assign(write_data.begin() + begin_word, write_data.begin() + end_word);
                    for (const auto& patch : mesh_trace_data.device_patches) {
                        if (patch.device != coord) {
                            continue;
                        }
                        const size_t patch_word = patch.byte_offset / sizeof(uint32_t);
                        for (size_t w = 0; w < patch.words.size(); ++w) {
                            const size_t word = patch_word + w;
                            if (word >= begin_word && word < end_word) {
                                run[word - begin_word] = patch.words[w];
                            }
                        }
                    }
                    mesh_cq.enqueue_write_shard_to_sub_grid(
                        *(trace_buffer->mesh_buffer),
                        run.data(),
                        MeshCoordinateRange(coord),
                        true,
                        BufferRegion(
                            write_offset_per_device_range.at(device_range) + begin_word * sizeof(uint32_t),
                            run.size() * sizeof(uint32_t)));
                    a = b;
                }
            }
        }
        write_offset_per_device_range.at(device_range) += mesh_trace_data.data.size() * sizeof(uint32_t);
    }
}

}  // namespace

MeshTraceId MeshTrace::next_id() {
    static std::atomic<uint32_t> global_trace_id{0};
    return MeshTraceId(global_trace_id++);
}

std::shared_ptr<MeshTraceBuffer> MeshTrace::create_empty_mesh_trace_buffer() {
    return std::make_shared<MeshTraceBuffer>(std::make_shared<MeshTraceDescriptor>(), nullptr);
}

void MeshTrace::populate_mesh_buffer(
    MeshCommandQueue& mesh_cq,
    std::shared_ptr<MeshTraceBuffer>& trace_buffer,
    DeviceAddr dram_allocation_high_water_mark,
    DeviceAddr dram_deletion_high_water_mark,
    DeviceAddr max_live_trace_high_water_mark) {
    uint64_t unpadded_size = trace_buffer->desc->total_trace_size;
    size_t page_size = trace_dispatch::compute_interleaved_trace_buf_page_size(
        unpadded_size, mesh_cq.device()->allocator()->get_num_banks(BufferType::DRAM));
    size_t padded_size = round_up(unpadded_size, page_size);

    const auto current_trace_buffers_size = mesh_cq.device()->get_trace_buffers_size();
    mesh_cq.device()->set_trace_buffers_size(current_trace_buffers_size + padded_size);
    auto trace_region_size = mesh_cq.device()->allocator_impl()->get_config().trace_region_size;

    // When trace_region_size is 0, use dynamic allocation mode (allocate in DRAM, top-down)
    BufferType buffer_type = BufferType::TRACE;
    std::optional<bool> bottom_up = std::nullopt;

    if (trace_region_size == 0) {
        // Dynamic allocation mode: allocate trace buffer in DRAM (top-down)
        buffer_type = BufferType::DRAM;
        bottom_up = false;  // Top-down allocation
    } else {
        // Traditional mode: use dedicated trace region. trace_region_size is the TOTAL trace budget across all
        // DRAM banks (see AllocatorImpl::init_one_bank_per_channel), matching get_trace_buffers_size(), which
        // tracks the total interleaved size across all banks.
        TT_FATAL(
            mesh_cq.device()->get_trace_buffers_size() <= trace_region_size,
            "Creating trace buffers of size {}B on MeshDevice {}, but only {}B is allocated for trace region.",
            mesh_cq.device()->get_trace_buffers_size(),
            mesh_cq.device()->id(),
            trace_region_size);
    }

    DeviceLocalBufferConfig device_local_trace_buf_config = {
        .page_size = page_size,
        .buffer_type = buffer_type,
        .bottom_up = bottom_up,
    };

    ReplicatedBufferConfig global_trace_buf_config = {
        .size = padded_size,
    };

    // Trace storage is safe from every live trace: reserved BufferType::TRACE storage is disjoint from ordinary DRAM,
    // and dynamic top-down BufferType::DRAM storage is checked against every live trace's high-water mark below. The
    // "trace_storage" allocation context is explicitly recognized by the trace allocation tracker, which uses it to
    // exclude both forms from unsafe-allocation accounting (and from the tracking-disabled warning).
    {
        auto trace_storage_context = tt::tt_metal::make_allocation_context_guard("trace_storage");
        trace_buffer->mesh_buffer =
            MeshBuffer::create(global_trace_buf_config, device_local_trace_buf_config, mesh_cq.device());
    }

    // In dynamic allocation mode, validate that the new trace buffer is outside the replay footprint of every live
    // trace.
    trace_buffer->dram_high_water_mark = std::max(dram_allocation_high_water_mark, dram_deletion_high_water_mark);
    const DeviceAddr effective_high_water_mark = std::max({trace_buffer->dram_high_water_mark, max_live_trace_high_water_mark});
    if (trace_region_size == 0 && effective_high_water_mark > 0) {
        DeviceAddr trace_buffer_address = trace_buffer->mesh_buffer->address();
        if (trace_buffer_address < effective_high_water_mark) {
            // Determine which high water mark caused the overlap for a more specific error message
            bool allocation_overlap =
                dram_allocation_high_water_mark > 0 && trace_buffer_address < dram_allocation_high_water_mark;
            bool deletion_overlap =
                dram_deletion_high_water_mark > 0 && trace_buffer_address < dram_deletion_high_water_mark;

            if (allocation_overlap && deletion_overlap) {
                TT_FATAL(
                    false,
                    "Trace buffer at address {} overlaps with DRAM activity during trace capture. "
                    "Allocation high water mark: {}, Deletion high water mark: {}. "
                    "Reduce allocations during trace capture or set a non-zero trace_region_size.",
                    trace_buffer_address,
                    dram_allocation_high_water_mark,
                    dram_deletion_high_water_mark);
            } else if (deletion_overlap) {
                TT_FATAL(
                    false,
                    "Trace buffer at address {} overlaps with buffers deallocated during trace capture "
                    "(deletion high water mark: {}). "
                    "Avoid deallocating DRAM buffers during trace capture or set a non-zero trace_region_size.",
                    trace_buffer_address,
                    dram_deletion_high_water_mark);
            } else if (allocation_overlap) {
                TT_FATAL(
                    false,
                    "Trace buffer at address {} overlaps with buffers allocated during trace capture "
                    "(allocation high water mark: {}). "
                    "Reduce allocations during trace capture or set a non-zero trace_region_size.",
                    trace_buffer_address,
                    dram_allocation_high_water_mark);
            } else {
                TT_FATAL(
                    false,
                    "Trace buffer at address {} overlaps with DRAM activity captured by another live trace "
                    "(maximum live trace high water mark: {}). "
                    "Release the existing trace before capturing this trace or set a non-zero trace_region_size.",
                    trace_buffer_address,
                    max_live_trace_high_water_mark);
            }
        }
    }

    if (trace_write_serial()) {
        populate_mesh_buffer_serial(mesh_cq, trace_buffer, page_size);
        return;
    }

    // All the trace's writes go out as non-blocking shard writes with one finish at the end. Each device's writes keep
    // their order (the dispatch pool runs one device's writes in index order), and the devices are written in parallel.
    // A write's host data is copied into the issue queue before the call returns, so the buffers below only need to
    // live until their batch is issued.
    std::vector<ShardDataTransfer> transfers;
    std::vector<std::vector<uint32_t>> held;  // Tail pages and patched runs; moving a vector keeps its data pointer.
    size_t held_bytes = 0;
    constexpr size_t max_held_bytes = size_t{1} << 30;
    auto flush = [&]() {
        if (!transfers.empty()) {
            mesh_cq.enqueue_write_shards(trace_buffer->mesh_buffer, transfers, false);
        }
        transfers.clear();
        held.clear();
        held_bytes = 0;
    };
    auto add_write = [&](const MeshCoordinate& coord, const uint32_t* data, uint64_t offset, size_t bytes) {
        transfers.push_back(
            ShardDataTransfer(coord).host_data(const_cast<uint32_t*>(data)).region(BufferRegion(offset, bytes)));
    };
    auto hold = [&](std::vector<uint32_t>&& words) -> const uint32_t* {
        held_bytes += words.size() * sizeof(uint32_t);
        held.push_back(std::move(words));
        return held.back().data();
    };

    const size_t page_words = page_size / sizeof(uint32_t);
    std::unordered_map<MeshCoordinateRange, uint32_t> write_offset_per_device_range = {};
    std::vector<uint32_t> pages;
    for (auto& mesh_trace_data : trace_buffer->desc->ordered_trace_data) {
        auto& device_range = mesh_trace_data.device_range;
        if (!write_offset_per_device_range.contains(device_range)) {
            write_offset_per_device_range.insert({device_range, 0});
        }
        const uint64_t offset = write_offset_per_device_range.at(device_range);
        const std::vector<uint32_t>& data = mesh_trace_data.data;
        // Same bytes as a copy padded with zeros to whole pages: the whole pages straight from the trace data, the
        // last part page from a small zero-padded buffer.
        const size_t full_words = (data.size() / page_words) * page_words;
        const uint32_t* tail = nullptr;
        if (full_words < data.size()) {
            std::vector<uint32_t> tail_page(page_words, 0);
            std::copy(data.begin() + full_words, data.end(), tail_page.begin());
            tail = hold(std::move(tail_page));
        }
        std::vector<MeshCoordinate> local_coords;
        for (const auto& coord : device_range) {
            if (mesh_cq.device()->impl().is_local(coord)) {
                local_coords.push_back(coord);
            }
        }
        for (const auto& coord : local_coords) {
            if (full_words > 0) {
                add_write(coord, data.data(), offset, full_words * sizeof(uint32_t));
            }
            if (tail != nullptr) {
                add_write(coord, tail, offset + full_words * sizeof(uint32_t), page_size);
            }
        }
        // Per-device runtime args: after the stream, each device gets only the pages that hold its own words, patched.
        if (!mesh_trace_data.device_patches.empty()) {
            for (const auto& coord : local_coords) {
                pages.clear();
                for (const auto& patch : mesh_trace_data.device_patches) {
                    if (patch.device != coord || patch.words.empty()) {
                        continue;
                    }
                    TT_FATAL(
                        patch.byte_offset % sizeof(uint32_t) == 0 &&
                            patch.byte_offset + patch.words.size() * sizeof(uint32_t) <=
                                data.size() * sizeof(uint32_t),
                        "Per-device runtime args trace patch out of bounds");
                    const uint32_t first = patch.byte_offset / page_size;
                    const uint32_t last = (patch.byte_offset + patch.words.size() * sizeof(uint32_t) - 1) / page_size;
                    for (uint32_t page = first; page <= last; ++page) {
                        pages.push_back(page);
                    }
                }
                std::sort(pages.begin(), pages.end());
                pages.erase(std::unique(pages.begin(), pages.end()), pages.end());
                // Each run of consecutive pages is one write of the device's patched words.
                for (size_t a = 0; a < pages.size();) {
                    size_t b = a + 1;
                    while (b < pages.size() && pages[b] == pages[b - 1] + 1) {
                        ++b;
                    }
                    const size_t begin_word = pages[a] * page_words;
                    const size_t end_word = (pages[b - 1] + 1) * page_words;
                    std::vector<uint32_t> run(end_word - begin_word, 0);
                    const size_t copy_end = std::min(end_word, data.size());
                    if (copy_end > begin_word) {
                        std::copy(data.begin() + begin_word, data.begin() + copy_end, run.begin());
                    }
                    for (const auto& patch : mesh_trace_data.device_patches) {
                        if (patch.device != coord) {
                            continue;
                        }
                        const size_t patch_word = patch.byte_offset / sizeof(uint32_t);
                        for (size_t w = 0; w < patch.words.size(); ++w) {
                            const size_t word = patch_word + w;
                            if (word >= begin_word && word < end_word) {
                                run[word - begin_word] = patch.words[w];
                            }
                        }
                    }
                    const size_t run_bytes = run.size() * sizeof(uint32_t);
                    add_write(coord, hold(std::move(run)), offset + begin_word * sizeof(uint32_t), run_bytes);
                    a = b;
                }
            }
        }
        write_offset_per_device_range.at(device_range) += data.size() * sizeof(uint32_t);
        if (held_bytes >= max_held_bytes) {
            flush();
        }
    }
    flush();
    mesh_cq.finish();
}

MeshTraceBuffer::~MeshTraceBuffer() {
    if (this->mesh_buffer && this->mesh_buffer->is_allocated() && this->mesh_buffer->device()) {
        auto current_trace_buffers_size = this->mesh_buffer->device()->get_trace_buffers_size();
        this->mesh_buffer->device()->set_trace_buffers_size(current_trace_buffers_size - this->mesh_buffer->size());
    }
}

}  // namespace tt::tt_metal::distributed
