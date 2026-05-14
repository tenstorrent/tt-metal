// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

namespace tt::lite {

static constexpr uint32_t TTB_MAGIC = 0x54544230;  // "TTB0"
static constexpr uint32_t TTB_VERSION = 0;

struct TraceWorkerDesc {
    uint8_t sub_device_id;
    uint32_t num_completion_worker_cores;
    uint32_t num_mcast_programs;
    uint32_t num_unicast_programs;
};

struct BufferPlacement {
    uint64_t address;
    uint64_t size;
    uint32_t page_size;
    uint8_t buffer_type;  // 0=DRAM, 1=L1 (v1+)
};

struct TraceBinaryHeader {
    uint32_t magic;
    uint32_t version;
    uint32_t num_worker_descs;
    uint32_t num_trace_streams;
    uint32_t num_persistent_buffers;
    uint32_t num_io_buffers;
};

struct TraceBinary {
    TraceBinaryHeader header;
    std::vector<TraceWorkerDesc> worker_descs;
    std::vector<std::vector<uint32_t>> trace_streams;
    std::vector<BufferPlacement> persistent_buffers;
    std::vector<std::vector<uint8_t>> persistent_buffer_data;
    std::vector<BufferPlacement> io_buffers;
    std::vector<std::string> io_buffer_names;

    uint64_t trace_buf_address;
    uint32_t trace_buf_page_size;
    uint32_t trace_buf_num_pages;
};

inline bool write_trace_binary(const TraceBinary& bin, const std::string& path) {
    std::ofstream out(path, std::ios::binary);
    if (!out.is_open()) return false;

    out.write(reinterpret_cast<const char*>(&bin.header), sizeof(TraceBinaryHeader));

    for (auto& wd : bin.worker_descs) {
        out.write(reinterpret_cast<const char*>(&wd.sub_device_id), 1);
        out.write(reinterpret_cast<const char*>(&wd.num_completion_worker_cores), 4);
        out.write(reinterpret_cast<const char*>(&wd.num_mcast_programs), 4);
        out.write(reinterpret_cast<const char*>(&wd.num_unicast_programs), 4);
    }

    for (auto& stream : bin.trace_streams) {
        uint32_t len = stream.size();
        out.write(reinterpret_cast<const char*>(&len), 4);
        out.write(reinterpret_cast<const char*>(stream.data()), len * sizeof(uint32_t));
    }

    for (uint32_t i = 0; i < bin.persistent_buffers.size(); i++) {
        auto& bp = bin.persistent_buffers[i];
        out.write(reinterpret_cast<const char*>(&bp.address), 8);
        out.write(reinterpret_cast<const char*>(&bp.size), 8);
        out.write(reinterpret_cast<const char*>(&bp.page_size), 4);
        out.write(reinterpret_cast<const char*>(&bp.buffer_type), 1);
        uint64_t data_size = bin.persistent_buffer_data[i].size();
        out.write(reinterpret_cast<const char*>(&data_size), 8);
        out.write(reinterpret_cast<const char*>(bin.persistent_buffer_data[i].data()), data_size);
    }

    for (uint32_t i = 0; i < bin.io_buffers.size(); i++) {
        auto& bp = bin.io_buffers[i];
        out.write(reinterpret_cast<const char*>(&bp.address), 8);
        out.write(reinterpret_cast<const char*>(&bp.size), 8);
        out.write(reinterpret_cast<const char*>(&bp.page_size), 4);
        out.write(reinterpret_cast<const char*>(&bp.buffer_type), 1);
        uint32_t name_len = bin.io_buffer_names[i].size();
        out.write(reinterpret_cast<const char*>(&name_len), 4);
        out.write(bin.io_buffer_names[i].data(), name_len);
    }

    out.write(reinterpret_cast<const char*>(&bin.trace_buf_address), 8);
    out.write(reinterpret_cast<const char*>(&bin.trace_buf_page_size), 4);
    out.write(reinterpret_cast<const char*>(&bin.trace_buf_num_pages), 4);

    return out.good();
}

inline bool read_trace_binary(TraceBinary& bin, const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in.is_open()) return false;

    in.read(reinterpret_cast<char*>(&bin.header), sizeof(TraceBinaryHeader));
    if (bin.header.magic != TTB_MAGIC) return false;

    bin.worker_descs.resize(bin.header.num_worker_descs);
    for (auto& wd : bin.worker_descs) {
        in.read(reinterpret_cast<char*>(&wd.sub_device_id), 1);
        in.read(reinterpret_cast<char*>(&wd.num_completion_worker_cores), 4);
        in.read(reinterpret_cast<char*>(&wd.num_mcast_programs), 4);
        in.read(reinterpret_cast<char*>(&wd.num_unicast_programs), 4);
    }

    bin.trace_streams.resize(bin.header.num_trace_streams);
    for (auto& stream : bin.trace_streams) {
        uint32_t len;
        in.read(reinterpret_cast<char*>(&len), 4);
        stream.resize(len);
        in.read(reinterpret_cast<char*>(stream.data()), len * sizeof(uint32_t));
    }

    bin.persistent_buffers.resize(bin.header.num_persistent_buffers);
    bin.persistent_buffer_data.resize(bin.header.num_persistent_buffers);
    for (uint32_t i = 0; i < bin.header.num_persistent_buffers; i++) {
        auto& bp = bin.persistent_buffers[i];
        in.read(reinterpret_cast<char*>(&bp.address), 8);
        in.read(reinterpret_cast<char*>(&bp.size), 8);
        in.read(reinterpret_cast<char*>(&bp.page_size), 4);
        in.read(reinterpret_cast<char*>(&bp.buffer_type), 1);
        uint64_t data_size;
        in.read(reinterpret_cast<char*>(&data_size), 8);
        bin.persistent_buffer_data[i].resize(data_size);
        in.read(reinterpret_cast<char*>(bin.persistent_buffer_data[i].data()), data_size);
    }

    bin.io_buffers.resize(bin.header.num_io_buffers);
    bin.io_buffer_names.resize(bin.header.num_io_buffers);
    for (uint32_t i = 0; i < bin.header.num_io_buffers; i++) {
        auto& bp = bin.io_buffers[i];
        in.read(reinterpret_cast<char*>(&bp.address), 8);
        in.read(reinterpret_cast<char*>(&bp.size), 8);
        in.read(reinterpret_cast<char*>(&bp.page_size), 4);
        in.read(reinterpret_cast<char*>(&bp.buffer_type), 1);
        uint32_t name_len;
        in.read(reinterpret_cast<char*>(&name_len), 4);
        bin.io_buffer_names[i].resize(name_len);
        in.read(bin.io_buffer_names[i].data(), name_len);
    }

    in.read(reinterpret_cast<char*>(&bin.trace_buf_address), 8);
    in.read(reinterpret_cast<char*>(&bin.trace_buf_page_size), 4);
    in.read(reinterpret_cast<char*>(&bin.trace_buf_num_pages), 4);

    return in.good();
}

}  // namespace tt::lite
