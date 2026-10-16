// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <tt-metalium/experimental/sockets/hd_socket_descriptor.hpp>
#include <tt-metalium/experimental/sockets/named_shm.hpp>
#include <tt-metalium/experimental/sockets/pcie_core_location.hpp>
#include <internal/descriptor_file.hpp>
#include "hd_socket_descriptor_serialization.hpp"

#include <tt_stl/assert.hpp>
#include <tt-metalium/distributed.hpp>
#include "distributed/mesh_device_impl.hpp"
#include "impl/context/metal_env_impl.hpp"
#include <umd/device/pcie/pci_device.hpp>

#include <vector>

namespace tt::tt_metal::distributed {

PCIeCoreLocation PCIeCoreLocation::resolve(MeshDevice& mesh_device, const MeshCoreCoord& core) {
    const auto logical_device_id = mesh_device.get_device(core.device_coord)->id();
    const auto physical_device_num = tt::umd::PCIDevice::get_pci_device_id(static_cast<int>(logical_device_id));
    TT_FATAL(
        physical_device_num.has_value(),
        "Failed to resolve physical PCIe device for logical chip id {} (mesh coord {}).",
        logical_device_id,
        core.device_coord);
    return PCIeCoreLocation{
        .pcie_device_id = static_cast<uint32_t>(physical_device_num.value()),
        .core = core,
        .virtual_core = mesh_device.worker_core_from_logical_core(core.core_coord),
    };
}

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
    location = PCIeCoreLocation::resolve(*mesh_device, core);
    pcie_alignment = mesh_device->impl().metal_env().get_hal().get_alignment(HalMemType::HOST);
}

flatbuffers::Offset<flatbuffer::HDSocketDescriptor> to_flatbuffer(
    flatbuffers::FlatBufferBuilder& builder, const HDSocketDescriptor& desc) {
    auto fb_socket_type = builder.CreateString(desc.socket_type);
    auto fb_shm_name = builder.CreateString(desc.shm_name);
    const auto& mesh_coord = desc.location.core.device_coord.coords();
    auto fb_mesh_coord = builder.CreateVector(std::vector<uint32_t>(mesh_coord.begin(), mesh_coord.end()));
    return flatbuffer::CreateHDSocketDescriptor(
        builder,
        fb_socket_type,
        fb_shm_name,
        desc.shm_size,
        desc.data_offset,
        desc.bytes_acked_offset,
        desc.bytes_sent_offset,
        desc.fifo_size,
        desc.h2d_mode,
        desc.config_buffer_address,
        desc.aligned_data_buf_start,
        desc.location.pcie_device_id,
        static_cast<uint32_t>(desc.location.core.core_coord.x),
        static_cast<uint32_t>(desc.location.core.core_coord.y),
        static_cast<uint32_t>(desc.location.virtual_core.x),
        static_cast<uint32_t>(desc.location.virtual_core.y),
        desc.pcie_alignment,
        desc.bytes_acked_device_offset,
        desc.connector_state_offset,
        fb_mesh_coord);
}

HDSocketDescriptor from_flatbuffer(const flatbuffer::HDSocketDescriptor& fb) {
    HDSocketDescriptor desc;
    desc.socket_type = fb.socket_type() ? fb.socket_type()->str() : "";
    desc.shm_name = fb.shm_name() ? fb.shm_name()->str() : "";
    desc.shm_size = fb.shm_size();
    desc.data_offset = fb.data_offset();
    desc.bytes_acked_offset = fb.bytes_acked_offset();
    desc.bytes_sent_offset = fb.bytes_sent_offset();
    desc.fifo_size = fb.fifo_size();
    desc.h2d_mode = fb.h2d_mode();
    desc.config_buffer_address = fb.config_buffer_address();
    desc.aligned_data_buf_start = fb.aligned_data_buf_start();
    // Descriptors written before mesh_coord was added leave it empty; those were always at (0, 0).
    const auto* mc = fb.mesh_coord();
    const MeshCoordinate device_coord = (mc == nullptr || mc->size() == 0)
                                            ? MeshCoordinate(0, 0)
                                            : MeshCoordinate(ttsl::Span<const uint32_t>(mc->data(), mc->size()));
    desc.location = PCIeCoreLocation{
        .pcie_device_id = fb.device_id(),
        .core = MeshCoreCoord(device_coord, CoreCoord(fb.core_x(), fb.core_y())),
        .virtual_core = CoreCoord(fb.virtual_core_x(), fb.virtual_core_y()),
    };
    desc.pcie_alignment = fb.pcie_alignment();
    desc.bytes_acked_device_offset = fb.bytes_acked_device_offset();
    desc.connector_state_offset = fb.connector_state_offset();
    return desc;
}

void HDSocketDescriptor::write_to_file(const std::string& path) const {
    flatbuffers::FlatBufferBuilder builder(512);
    builder.Finish(to_flatbuffer(builder, *this));
    internal::write_file_atomically(path, ttsl::Span<const uint8_t>(builder.GetBufferPointer(), builder.GetSize()));
}

HDSocketDescriptor HDSocketDescriptor::read_from_file(const std::string& path) {
    const auto bytes = internal::read_file(path);
    const auto* fb = flatbuffer::GetHDSocketDescriptor(bytes.data());
    TT_FATAL(fb, "Failed to parse flatbuffer descriptor from: {}", path);
    return from_flatbuffer(*fb);
}

HDSocketDescriptor HDSocketDescriptor::wait_and_read(
    const std::string& descriptor_path, const std::string& expected_type, uint32_t timeout_ms) {
    const auto bytes = internal::wait_and_read_file(descriptor_path, timeout_ms);
    const auto* fb = flatbuffer::GetHDSocketDescriptor(bytes.data());
    TT_FATAL(fb, "Failed to parse flatbuffer descriptor from: {}", descriptor_path);
    auto desc = from_flatbuffer(*fb);
    TT_FATAL(
        desc.socket_type == expected_type,
        "Descriptor type mismatch: expected '{}', got '{}'",
        expected_type,
        desc.socket_type);
    return desc;
}

}  // namespace tt::tt_metal::distributed
