// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "tt_metal/distributed/socket_descriptor.hpp"

#include <tt_stl/assert.hpp>
#include <nlohmann/json.hpp>

#include <fstream>

namespace tt::tt_metal::distributed {

void SocketDescriptor::write_to_file(const std::string& path) const {
    nlohmann::json j;
    j["socket_type"] = socket_type;
    j["shm_name"] = shm_name;
    j["shm_size"] = shm_size;
    j["data_offset"] = data_offset;
    j["bytes_acked_offset"] = bytes_acked_offset;
    j["bytes_sent_offset"] = bytes_sent_offset;
    j["fifo_size"] = fifo_size;
    j["h2d_mode"] = h2d_mode;
    j["config_buffer_address"] = config_buffer_address;
    j["aligned_data_buf_start"] = aligned_data_buf_start;
    j["device_id"] = device_id;
    j["core_x"] = core_x;
    j["core_y"] = core_y;
    j["virtual_core_x"] = virtual_core_x;
    j["virtual_core_y"] = virtual_core_y;
    j["pcie_alignment"] = pcie_alignment;
    j["bytes_acked_device_offset"] = bytes_acked_device_offset;

    std::ofstream ofs(path);
    TT_FATAL(ofs.is_open(), "Failed to open descriptor file for writing: {}", path);
    ofs << j.dump(2);
    ofs.close();
    TT_FATAL(!ofs.fail(), "Failed to write descriptor file: {}", path);
}

SocketDescriptor SocketDescriptor::read_from_file(const std::string& path) {
    std::ifstream ifs(path);
    TT_FATAL(ifs.is_open(), "Failed to open descriptor file for reading: {}", path);

    nlohmann::json j;
    ifs >> j;
    TT_FATAL(!ifs.fail(), "Failed to parse descriptor JSON from: {}", path);

    SocketDescriptor desc;
    desc.socket_type = j.at("socket_type").get<std::string>();
    desc.shm_name = j.at("shm_name").get<std::string>();
    desc.shm_size = j.at("shm_size").get<uint64_t>();
    desc.data_offset = j.at("data_offset").get<uint32_t>();
    desc.bytes_acked_offset = j.at("bytes_acked_offset").get<uint32_t>();
    desc.bytes_sent_offset = j.at("bytes_sent_offset").get<uint32_t>();
    desc.fifo_size = j.at("fifo_size").get<uint32_t>();
    desc.h2d_mode = j.at("h2d_mode").get<uint32_t>();
    desc.config_buffer_address = j.at("config_buffer_address").get<uint32_t>();
    desc.aligned_data_buf_start = j.at("aligned_data_buf_start").get<uint32_t>();
    desc.device_id = j.at("device_id").get<uint32_t>();
    desc.core_x = j.at("core_x").get<uint32_t>();
    desc.core_y = j.at("core_y").get<uint32_t>();
    desc.virtual_core_x = j.value("virtual_core_x", uint32_t{0});
    desc.virtual_core_y = j.value("virtual_core_y", uint32_t{0});
    desc.pcie_alignment = j.value("pcie_alignment", uint32_t{0});
    desc.bytes_acked_device_offset = j.value("bytes_acked_device_offset", uint32_t{0});

    return desc;
}

}  // namespace tt::tt_metal::distributed
