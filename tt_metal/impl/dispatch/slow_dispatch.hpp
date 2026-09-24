// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <span>
#include <vector>

#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/mesh_buffer.hpp>
#include <tt-metalium/mesh_coord.hpp>
#include <tt-metalium/mesh_device.hpp>
#include <tt-metalium/tt_metal.hpp>

namespace tt::tt_metal::slow_dispatch {

// MeshDevice-aware L1 / DRAM channel / MeshBuffer access for unit meshes.
//
// WriteToDeviceL1/ReadFromDeviceL1 and DRAM channel APIs index the cluster by IDevice::id(),
// which for MeshDevice is a logical mesh id — not a chip id. These helpers require a unit mesh
// (num_devices == 1) and forward to its physical sub-device.

inline IDevice* physical_device_from_unit_mesh(distributed::MeshDevice& unit_mesh) {
    TT_FATAL(
        unit_mesh.num_devices() == 1, "Expected a unit MeshDevice (num_devices == 1), got {}", unit_mesh.num_devices());
    return unit_mesh.get_devices().at(0);
}

// MeshBuffer host↔device transfer for a unit mesh (coord (0, 0)). Prefer these over
// detail::WriteToBuffer/ReadFromBuffer on MeshBuffer::get_reference_buffer().
template <typename DType>
inline void WriteToBuffer(const distributed::MeshBuffer& unit_mesh_buffer, const std::vector<DType>& host_buffer) {
    (void)physical_device_from_unit_mesh(*unit_mesh_buffer.device());
    detail::WriteToBuffer(*unit_mesh_buffer.get_reference_buffer(), host_buffer);
}

template <typename DType>
inline void ReadFromBuffer(const distributed::MeshBuffer& unit_mesh_buffer, std::vector<DType>& host_buffer) {
    (void)physical_device_from_unit_mesh(*unit_mesh_buffer.device());
    detail::ReadFromBuffer(*unit_mesh_buffer.get_reference_buffer(), host_buffer);
}

inline bool WriteToL1(
    distributed::MeshDevice& unit_mesh,
    CoreCoord logical_core,
    uint32_t address,
    std::vector<uint32_t>& host_buffer,
    CoreType core_type = CoreType::WORKER) {
    return detail::WriteToDeviceL1(
        physical_device_from_unit_mesh(unit_mesh), logical_core, address, host_buffer, core_type);
}

inline bool WriteToL1(
    distributed::MeshDevice& unit_mesh,
    CoreCoord logical_core,
    uint32_t address,
    std::span<const uint8_t> host_buffer,
    CoreType core_type = CoreType::WORKER) {
    return detail::WriteToDeviceL1(
        physical_device_from_unit_mesh(unit_mesh), logical_core, address, host_buffer, core_type);
}

inline bool ReadFromL1(
    distributed::MeshDevice& unit_mesh,
    CoreCoord logical_core,
    uint32_t address,
    uint32_t size,
    std::vector<uint32_t>& host_buffer,
    CoreType core_type = CoreType::WORKER) {
    return detail::ReadFromDeviceL1(
        physical_device_from_unit_mesh(unit_mesh), logical_core, address, size, host_buffer, core_type);
}

inline bool ReadFromL1(
    distributed::MeshDevice& unit_mesh,
    CoreCoord logical_core,
    uint32_t address,
    std::span<uint8_t> host_buffer,
    CoreType core_type = CoreType::WORKER) {
    return detail::ReadFromDeviceL1(
        physical_device_from_unit_mesh(unit_mesh), logical_core, address, host_buffer, core_type);
}

inline bool WriteToDRAMChannel(
    distributed::MeshDevice& unit_mesh, int dram_channel, uint32_t address, std::vector<uint32_t>& host_buffer) {
    return detail::WriteToDeviceDRAMChannel(
        physical_device_from_unit_mesh(unit_mesh), dram_channel, address, host_buffer);
}

inline bool WriteToDRAMChannel(
    distributed::MeshDevice& unit_mesh, int dram_channel, uint32_t address, std::span<const uint8_t> host_buffer) {
    return detail::WriteToDeviceDRAMChannel(
        physical_device_from_unit_mesh(unit_mesh), dram_channel, address, host_buffer);
}

inline bool ReadFromDRAMChannel(
    distributed::MeshDevice& unit_mesh,
    int dram_channel,
    uint32_t address,
    uint32_t size,
    std::vector<uint32_t>& host_buffer) {
    return detail::ReadFromDeviceDRAMChannel(
        physical_device_from_unit_mesh(unit_mesh), dram_channel, address, size, host_buffer);
}

inline bool ReadFromDRAMChannel(
    distributed::MeshDevice& unit_mesh, int dram_channel, uint32_t address, std::span<uint8_t> host_buffer) {
    return detail::ReadFromDeviceDRAMChannel(
        physical_device_from_unit_mesh(unit_mesh), dram_channel, address, host_buffer);
}

}  // namespace tt::tt_metal::slow_dispatch
