// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// tt-lite PoC: Load a .ttb trace binary and replay it on hardware.
// This program:
//  1. Opens a MeshDevice (same config as capture)
//  2. Reads the .ttb file containing the trace command stream + buffer metadata
//  3. Allocates IO buffers at the same addresses as capture time
//  4. Registers the trace data with the device and populates DRAM
//  5. Writes input data, replays the trace, reads output

#include <cassert>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include <tt-metalium/bfloat16.hpp>
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/mesh_coord.hpp>
#include "tt_metal/distributed/mesh_trace.hpp"

#include "trace_binary.h"

using namespace tt;
using namespace tt::tt_metal;
using namespace tt::tt_metal::distributed;

static void print_usage(const char* prog) {
    std::cerr << "Usage: " << prog << " <model.ttb> [options]\n"
              << "Options:\n"
              << "  --input  NAME=FILE   Write binary file to input buffer NAME\n"
              << "  --output NAME=FILE   Read output buffer NAME to binary file\n"
              << "  --trace-region-size N  Override trace region size (bytes)\n"
              << "  --l1-small-size N      Override l1_small_size\n"
              << std::endl;
}

static int find_io_buffer(const tt::lite::TraceBinary& ttb, const std::string& name) {
    for (size_t i = 0; i < ttb.io_buffer_names.size(); i++) {
        if (ttb.io_buffer_names[i] == name) return static_cast<int>(i);
    }
    return -1;
}

static std::vector<uint8_t> read_file(const std::string& path) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f.is_open()) return {};
    auto sz = f.tellg();
    f.seekg(0);
    std::vector<uint8_t> data(sz);
    f.read(reinterpret_cast<char*>(data.data()), sz);
    return data;
}

static bool write_file(const std::string& path, const void* data, size_t size) {
    std::ofstream f(path, std::ios::binary);
    if (!f.is_open()) return false;
    f.write(reinterpret_cast<const char*>(data), size);
    return f.good();
}

int main(int argc, char** argv) {
    if (argc < 2) {
        print_usage(argv[0]);
        return 1;
    }
    std::string input_path = argv[1];

    // Parse options
    struct IOSpec {
        std::string buf_name;
        std::string file_path;
    };
    std::vector<IOSpec> input_specs, output_specs;
    uint32_t trace_region_override = 0;
    uint32_t l1_small_override = 0;

    for (int i = 2; i < argc; i++) {
        std::string arg = argv[i];
        if ((arg == "--input" || arg == "--output") && i + 1 < argc) {
            std::string spec = argv[++i];
            auto eq = spec.find('=');
            if (eq == std::string::npos) {
                std::cerr << "Invalid spec (expected NAME=FILE): " << spec << std::endl;
                return 1;
            }
            IOSpec s{spec.substr(0, eq), spec.substr(eq + 1)};
            if (arg == "--input")
                input_specs.push_back(s);
            else
                output_specs.push_back(s);
        } else if (arg == "--trace-region-size" && i + 1 < argc) {
            trace_region_override = std::stoul(argv[++i]);
        } else if (arg == "--l1-small-size" && i + 1 < argc) {
            l1_small_override = std::stoul(argv[++i]);
        } else if (arg == "--help" || arg == "-h") {
            print_usage(argv[0]);
            return 0;
        }
    }

    // 1. Load .ttb
    tt::lite::TraceBinary ttb;
    if (!tt::lite::read_trace_binary(ttb, input_path)) {
        std::cerr << "Failed to read trace binary: " << input_path << std::endl;
        return 1;
    }
    std::cout << "Loaded trace binary: " << input_path << std::endl;
    std::cout << "  Worker descriptors: " << ttb.worker_descs.size() << std::endl;
    std::cout << "  Trace streams: " << ttb.trace_streams.size() << std::endl;
    for (size_t i = 0; i < ttb.trace_streams.size(); i++) {
        std::cout << "    Stream " << i << ": " << ttb.trace_streams[i].size() * 4 << " bytes" << std::endl;
    }
    std::cout << "  Persistent buffers: " << ttb.persistent_buffers.size() << std::endl;
    std::cout << "  IO buffers: " << ttb.io_buffers.size() << std::endl;
    for (size_t i = 0; i < ttb.io_buffers.size(); i++) {
        std::cout << "    " << ttb.io_buffer_names[i] << " addr=0x" << std::hex << ttb.io_buffers[i].address
                  << " size=" << std::dec << ttb.io_buffers[i].size << " page_size=" << ttb.io_buffers[i].page_size
                  << std::endl;
    }

    // 2. Compute trace region size from .ttb data
    uint32_t trace_data_bytes = 0;
    for (auto& stream : ttb.trace_streams) {
        trace_data_bytes = std::max(trace_data_bytes, static_cast<uint32_t>(stream.size() * sizeof(uint32_t)));
    }
    uint32_t trace_region_size = trace_region_override;
    if (trace_region_size == 0) {
        trace_region_size = trace_data_bytes * 2;
        trace_region_size = std::max(trace_region_size, uint32_t(2u << 20));
        trace_region_size = ((trace_region_size + 0xFFFFF) / 0x100000) * 0x100000;
    }
    uint32_t l1_small_size = l1_small_override ? l1_small_override : 0;

    std::cout << "  Trace data: " << trace_data_bytes << " bytes" << std::endl;
    std::cout << "  trace_region_size: " << trace_region_size << std::endl;
    std::cout << "  l1_small_size: " << l1_small_size << std::endl;

    // 3. Open device
    auto mesh_shape = MeshShape{1, 1};
    auto mesh_device = MeshDevice::create(
        MeshDeviceConfig(mesh_shape),
        l1_small_size,
        trace_region_size);

    auto& cq = mesh_device->mesh_command_queue();

    // 4. Merge all DRAM buffers and allocate in address order to match capture-time layout
    //    L1 buffers are managed by the trace and not pre-allocated here.
    struct BufferEntry {
        uint64_t address;
        uint64_t size;
        uint32_t page_size;
        bool is_persistent;
        size_t orig_index;
    };
    std::vector<BufferEntry> all_entries;
    for (size_t i = 0; i < ttb.persistent_buffers.size(); i++) {
        auto& bp = ttb.persistent_buffers[i];
        if (bp.buffer_type != 0) continue;  // skip non-DRAM
        all_entries.push_back({bp.address, bp.size, bp.page_size, true, i});
    }
    for (size_t i = 0; i < ttb.io_buffers.size(); i++) {
        auto& bp = ttb.io_buffers[i];
        if (bp.buffer_type != 0) {
            std::cout << "IO buffer '" << ttb.io_buffer_names[i] << "' is L1 (trace-managed), skipping allocation"
                      << std::endl;
            continue;
        }
        all_entries.push_back({bp.address, bp.size, bp.page_size, false, i});
    }
    std::sort(all_entries.begin(), all_entries.end(), [](const BufferEntry& a, const BufferEntry& b) {
        return a.address < b.address;
    });

    std::vector<std::shared_ptr<MeshBuffer>> persistent_bufs(ttb.persistent_buffers.size());
    std::vector<std::shared_ptr<MeshBuffer>> io_bufs(ttb.io_buffers.size());
    bool address_mismatch = false;

    for (auto& entry : all_entries) {
        ReplicatedBufferConfig global_cfg{.size = entry.size};
        DeviceLocalBufferConfig local_cfg{
            .page_size = entry.page_size, .buffer_type = BufferType::DRAM, .bottom_up = true};
        auto buf = MeshBuffer::create(global_cfg, local_cfg, mesh_device.get());

        if (buf->address() != entry.address) {
            if (entry.is_persistent) {
                std::cerr << "WARNING: Persistent buffer " << entry.orig_index << " address mismatch: "
                          << "expected 0x" << std::hex << entry.address << " got 0x" << buf->address() << std::dec
                          << std::endl;
            } else {
                std::cerr << "WARNING: IO buffer '" << ttb.io_buffer_names[entry.orig_index]
                          << "' address mismatch: expected 0x" << std::hex << entry.address << " got 0x"
                          << buf->address() << std::dec << std::endl;
            }
            address_mismatch = true;
        } else if (!entry.is_persistent) {
            std::cout << "Buffer '" << ttb.io_buffer_names[entry.orig_index] << "' allocated at expected address 0x"
                      << std::hex << buf->address() << std::dec << std::endl;
        }

        if (entry.is_persistent) {
            persistent_bufs[entry.orig_index] = buf;
        } else {
            io_bufs[entry.orig_index] = buf;
        }
    }

    // Write persistent buffer data to device
    for (size_t i = 0; i < ttb.persistent_buffers.size(); i++) {
        if (!persistent_bufs[i]) continue;
        if (i < ttb.persistent_buffer_data.size() && !ttb.persistent_buffer_data[i].empty()) {
            auto& data = ttb.persistent_buffer_data[i];
            size_t u32_count = (data.size() + 3) / 4;
            std::vector<uint32_t> u32_data(u32_count, 0);
            std::memcpy(u32_data.data(), data.data(), data.size());
            EnqueueWriteMeshBuffer(cq, persistent_bufs[i], u32_data, true);
        }
    }

    if (!ttb.persistent_buffers.empty()) {
        std::cout << "Loaded " << persistent_bufs.size() << " persistent buffers" << std::endl;
    }

    if (address_mismatch) {
        std::cerr << "WARNING: Address mismatches detected. Trace replay may produce incorrect results." << std::endl;
    }

    // 6. Register trace
    std::cout << "Registering trace with device..." << std::endl;
    auto trace_id = BeginTraceCapture(mesh_device.get(), 0);
    mesh_device->end_mesh_trace(0, trace_id);

    auto trace_buffer = mesh_device->get_mesh_trace(trace_id);
    auto& desc = *trace_buffer->desc;

    desc.descriptors.clear();
    desc.sub_device_ids.clear();
    for (auto& wd : ttb.worker_descs) {
        SubDeviceId id(wd.sub_device_id);
        desc.descriptors[id] = TraceWorkerDescriptor{
            .num_completion_worker_cores = wd.num_completion_worker_cores,
            .num_traced_programs_needing_go_signal_multicast = wd.num_mcast_programs,
            .num_traced_programs_needing_go_signal_unicast = wd.num_unicast_programs,
        };
        desc.sub_device_ids.push_back(id);
    }

    desc.ordered_trace_data.clear();
    auto local_range = mesh_device->get_view().get_local_mesh_coord_range();
    uint32_t max_size = 0;
    for (auto& stream : ttb.trace_streams) {
        desc.ordered_trace_data.push_back(MeshTraceData{local_range, stream});
        max_size = std::max(max_size, static_cast<uint32_t>(stream.size()));
    }
    desc.total_trace_size = max_size * sizeof(uint32_t);

    MeshTrace::populate_mesh_buffer(cq, trace_buffer);
    std::cout << "Trace loaded into DRAM at 0x" << std::hex << trace_buffer->mesh_buffer->address() << std::dec
              << std::endl;

    // 7. Write input data
    for (auto& spec : input_specs) {
        int idx = find_io_buffer(ttb, spec.buf_name);
        if (idx < 0) {
            std::cerr << "Input buffer '" << spec.buf_name << "' not found in .ttb" << std::endl;
            continue;
        }
        if (!io_bufs[idx]) {
            std::cerr << "Input buffer '" << spec.buf_name << "' is L1 (not writable from host)" << std::endl;
            continue;
        }
        auto file_data = read_file(spec.file_path);
        if (file_data.empty()) {
            std::cerr << "Failed to read input file: " << spec.file_path << std::endl;
            continue;
        }
        size_t buf_size = ttb.io_buffers[idx].size;
        size_t copy_size = std::min(file_data.size(), buf_size);
        std::vector<uint32_t> u32_data(buf_size / 4, 0);
        std::memcpy(u32_data.data(), file_data.data(), copy_size);
        EnqueueWriteMeshBuffer(cq, io_bufs[idx], u32_data, true);
        std::cout << "Wrote " << file_data.size() << " bytes to buffer '" << spec.buf_name << "'" << std::endl;
    }

    // 8. Replay trace
    std::cout << "Replaying trace..." << std::endl;
    mesh_device->replay_mesh_trace(0, trace_id, /*blocking=*/true);
    std::cout << "Trace replay complete." << std::endl;

    // 9. Read output data
    for (auto& spec : output_specs) {
        int idx = find_io_buffer(ttb, spec.buf_name);
        if (idx < 0) {
            std::cerr << "Output buffer '" << spec.buf_name << "' not found in .ttb" << std::endl;
            continue;
        }
        if (!io_bufs[idx]) {
            std::cerr << "Output buffer '" << spec.buf_name << "' is L1 (not readable from host)" << std::endl;
            continue;
        }
        size_t buf_size = ttb.io_buffers[idx].size;
        std::vector<uint32_t> result(buf_size / 4, 0);
        EnqueueReadMeshBuffer(cq, result, io_bufs[idx], true);
        if (write_file(spec.file_path, result.data(), buf_size)) {
            std::cout << "Wrote " << buf_size << " bytes from buffer '" << spec.buf_name << "' to " << spec.file_path
                      << std::endl;
        } else {
            std::cerr << "Failed to write output file: " << spec.file_path << std::endl;
        }
    }

    // 10. Summary: print first few values of each DRAM IO buffer for inspection
    for (size_t i = 0; i < ttb.io_buffers.size(); i++) {
        if (!io_bufs[i]) {
            std::cout << "  " << ttb.io_buffer_names[i] << " [L1, trace-managed]" << std::endl;
            continue;
        }
        size_t buf_size = ttb.io_buffers[i].size;
        std::vector<uint32_t> data(buf_size / 4, 0);
        EnqueueReadMeshBuffer(cq, data, io_bufs[i], true);
        bfloat16* bf16 = reinterpret_cast<bfloat16*>(data.data());
        size_t num_values = buf_size / 2;
        size_t preview = std::min(num_values, size_t(8));
        std::cout << "  " << ttb.io_buffer_names[i] << " [" << num_values << " values]: ";
        for (size_t j = 0; j < preview; j++) {
            std::cout << static_cast<float>(bf16[j]) << " ";
        }
        if (num_values > preview) std::cout << "...";
        std::cout << std::endl;
    }

    std::cout << "Done." << std::endl;

    mesh_device->release_mesh_trace(trace_id);
    mesh_device->close();
    return 0;
}
