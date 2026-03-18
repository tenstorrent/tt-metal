// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "tt_metal/distributed/hd_socket_descriptor.hpp"
#include "tt_metal/distributed/named_shm.hpp"
#include "hd_socket_descriptor_generated.h"

#include <tt_stl/assert.hpp>
#include <tt-metalium/distributed.hpp>
#include "impl/context/metal_context.hpp"

#include <chrono>
#include <filesystem>
#include <fstream>
#include <thread>

namespace tt::tt_metal::distributed {

void HDSocketDescriptor::populate_from_owner(
    const std::string& type,
    const NamedShm& shm,
    uint32_t fifo_size_arg,
    uint32_t config_buffer_address_arg,
    MeshDevice* mesh_device,
    const MeshCoreCoord& core) {
    socket_type = type;
    shm_name = shm.name();
    shm_size = shm.size();
    data_offset = 0;
    fifo_size = fifo_size_arg;
    config_buffer_address = config_buffer_address_arg;
    device_id = static_cast<uint32_t>(mesh_device->get_device(core.device_coord)->id());
    core_x = core.core_coord.x;
    core_y = core.core_coord.y;
    auto vc = mesh_device->worker_core_from_logical_core(core.core_coord);
    virtual_core_x = vc.x;
    virtual_core_y = vc.y;
    pcie_alignment = MetalContext::instance().hal().get_alignment(HalMemType::HOST);
}

void HDSocketDescriptor::write_to_file(const std::string& path) const {
    flatbuffers::FlatBufferBuilder builder(512);
    auto fb_socket_type = builder.CreateString(socket_type);
    auto fb_shm_name = builder.CreateString(shm_name);

    auto fb_desc = flatbuffer::CreateHDSocketDescriptor(
        builder,
        fb_socket_type,
        fb_shm_name,
        shm_size,
        data_offset,
        bytes_acked_offset,
        bytes_sent_offset,
        fifo_size,
        h2d_mode,
        config_buffer_address,
        aligned_data_buf_start,
        device_id,
        core_x,
        core_y,
        virtual_core_x,
        virtual_core_y,
        pcie_alignment,
        bytes_acked_device_offset);
    builder.Finish(fb_desc);

    std::ofstream ofs(path, std::ios::binary);
    TT_FATAL(ofs.is_open(), "Failed to open descriptor file for writing: {}", path);
    ofs.write(reinterpret_cast<const char*>(builder.GetBufferPointer()), builder.GetSize());
    ofs.close();
    TT_FATAL(!ofs.fail(), "Failed to write descriptor file: {}", path);
}

HDSocketDescriptor HDSocketDescriptor::read_from_file(const std::string& path) {
    std::ifstream ifs(path, std::ios::binary | std::ios::ate);
    TT_FATAL(ifs.is_open(), "Failed to open descriptor file for reading: {}", path);

    auto size = ifs.tellg();
    ifs.seekg(0, std::ios::beg);
    std::vector<uint8_t> buf(size);
    ifs.read(reinterpret_cast<char*>(buf.data()), size);
    TT_FATAL(!ifs.fail(), "Failed to read descriptor file: {}", path);

    auto* fb = flatbuffer::GetHDSocketDescriptor(buf.data());
    TT_FATAL(fb, "Failed to parse flatbuffer descriptor from: {}", path);

    HDSocketDescriptor desc;
    desc.socket_type = fb->socket_type() ? fb->socket_type()->str() : "";
    desc.shm_name = fb->shm_name() ? fb->shm_name()->str() : "";
    desc.shm_size = fb->shm_size();
    desc.data_offset = fb->data_offset();
    desc.bytes_acked_offset = fb->bytes_acked_offset();
    desc.bytes_sent_offset = fb->bytes_sent_offset();
    desc.fifo_size = fb->fifo_size();
    desc.h2d_mode = fb->h2d_mode();
    desc.config_buffer_address = fb->config_buffer_address();
    desc.aligned_data_buf_start = fb->aligned_data_buf_start();
    desc.device_id = fb->device_id();
    desc.core_x = fb->core_x();
    desc.core_y = fb->core_y();
    desc.virtual_core_x = fb->virtual_core_x();
    desc.virtual_core_y = fb->virtual_core_y();
    desc.pcie_alignment = fb->pcie_alignment();
    desc.bytes_acked_device_offset = fb->bytes_acked_device_offset();

    return desc;
}

HDSocketDescriptor HDSocketDescriptor::wait_and_read(
    const std::string& descriptor_path, const std::string& expected_type, uint32_t timeout_ms) {
    auto start_time = std::chrono::high_resolution_clock::now();
    while (!std::filesystem::exists(descriptor_path)) {
        auto elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                              std::chrono::high_resolution_clock::now() - start_time)
                              .count();
        if (elapsed_ms > timeout_ms) {
            TT_THROW("Timeout waiting for descriptor file to be created: {}", descriptor_path);
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    auto desc = read_from_file(descriptor_path);
    TT_FATAL(
        desc.socket_type == expected_type,
        "Descriptor type mismatch: expected '{}', got '{}'",
        expected_type,
        desc.socket_type);
    return desc;
}

}  // namespace tt::tt_metal::distributed
