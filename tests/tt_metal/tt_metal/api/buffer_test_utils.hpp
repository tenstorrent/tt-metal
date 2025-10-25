// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <stdint.h>
#include <vector>
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-logger/tt-logger.hpp>
#include <tt-metalium/tt_metal.hpp>

namespace tt::test::buffer::detail {
inline void writeL1Backdoor(
    tt::tt_metal::IDevice* device, CoreCoord coord, uint32_t address, std::vector<uint32_t>& data) {
    log_info(tt::LogTest, "{} -- coord={} address={}", __FUNCTION__, coord.str(), address);
    tt_metal::detail::WriteToDeviceL1(device, coord, address, data);
}
inline void writeL1Backdoor(
    const std::shared_ptr<tt::tt_metal::distributed::MeshDevice>& mesh_device,
    CoreCoord coord,
    uint32_t address,
    std::vector<uint32_t>& data) {
    log_info(tt::LogTest, "{} -- coord={} address={}", __FUNCTION__, coord.str(), address);
    tt_metal::detail::WriteToDeviceL1(mesh_device->get_devices()[0], coord, address, data);
}
inline void writeL1Backdoor(
    const std::shared_ptr<tt::tt_metal::distributed::MeshDevice>& mesh_device,
    CoreCoord coord,
    uint32_t address,
    std::span<const uint8_t> data) {
    log_info(tt::LogTest, "{} -- coord={} address={}", __FUNCTION__, coord.str(), address);
    tt_metal::detail::WriteToDeviceL1(mesh_device->get_devices()[0], coord, address, data);
}
inline void readL1Backdoor(
    tt::tt_metal::IDevice* device, CoreCoord coord, uint32_t address, uint32_t byte_size, std::vector<uint32_t>& data) {
    log_info(tt::LogTest, "{} -- coord={} address={} byte_size={}", __FUNCTION__, coord.str(), address, byte_size);
    tt_metal::detail::ReadFromDeviceL1(device, coord, address, byte_size, data);
}
inline void readL1Backdoor(
    const std::shared_ptr<tt::tt_metal::distributed::MeshDevice>& mesh_device,
    CoreCoord coord,
    uint32_t address,
    uint32_t byte_size,
    std::vector<uint32_t>& data) {
    log_info(tt::LogTest, "{} -- coord={} address={} byte_size={}", __FUNCTION__, coord.str(), address, byte_size);
    tt_metal::detail::ReadFromDeviceL1(mesh_device->get_devices()[0], coord, address, byte_size, data);
}
inline void readL1Backdoor(
    const std::shared_ptr<tt::tt_metal::distributed::MeshDevice>& mesh_device,
    CoreCoord coord,
    uint32_t address,
    std::span<uint8_t> data) {
    log_info(tt::LogTest, "{} -- coord={} address={} byte_size={}", __FUNCTION__, coord.str(), address, data.size());
    tt_metal::detail::ReadFromDeviceL1(mesh_device->get_devices()[0], coord, address, data);
}
inline void writeDramBackdoor(
    tt::tt_metal::IDevice* device, uint32_t channel, uint32_t address, std::vector<uint32_t>& data) {
    log_info(tt::LogTest, "{} -- channel={} address={}", __FUNCTION__, channel, address);
    tt_metal::detail::WriteToDeviceDRAMChannel(device, channel, address, data);
}
inline void writeDramBackdoor(
    const std::shared_ptr<tt::tt_metal::distributed::MeshDevice>& mesh_device,
    uint32_t channel,
    uint32_t address,
    std::vector<uint32_t>& data) {
    log_info(tt::LogTest, "{} -- channel={} address={}", __FUNCTION__, channel, address);
    tt_metal::detail::WriteToDeviceDRAMChannel(mesh_device->get_devices()[0], channel, address, data);
}
inline void writeDramBackdoor(
    const std::shared_ptr<tt::tt_metal::distributed::MeshDevice>& mesh_device,
    uint32_t channel,
    uint32_t address,
    std::span<const uint8_t> data) {
    log_info(tt::LogTest, "{} -- channel={} address={}", __FUNCTION__, channel, address);
    tt_metal::detail::WriteToDeviceDRAMChannel(mesh_device->get_devices()[0], channel, address, data);
}
inline void readDramBackdoor(
    tt::tt_metal::IDevice* device, uint32_t channel, uint32_t address, uint32_t byte_size, std::vector<uint32_t>& data) {
    log_info(tt::LogTest, "{} -- channel={} address={} byte_size={}", __FUNCTION__, channel, address, byte_size);
    tt_metal::detail::ReadFromDeviceDRAMChannel(device, channel, address, byte_size, data);
}
inline void readDramBackdoor(
    const std::shared_ptr<tt::tt_metal::distributed::MeshDevice>& mesh_device,
    uint32_t channel,
    uint32_t address,
    uint32_t byte_size,
    std::vector<uint32_t>& data) {
    log_info(tt::LogTest, "{} -- channel={} address={} byte_size={}", __FUNCTION__, channel, address, byte_size);
    tt_metal::detail::ReadFromDeviceDRAMChannel(mesh_device->get_devices()[0], channel, address, byte_size, data);
}
inline void readDramBackdoor(
    const std::shared_ptr<tt::tt_metal::distributed::MeshDevice>& mesh_device,
    uint32_t channel,
    uint32_t address,
    std::span<uint8_t> data) {
    log_info(tt::LogTest, "{} -- channel={} address={} byte_size={}", __FUNCTION__, channel, address, data.size());
    tt_metal::detail::ReadFromDeviceDRAMChannel(mesh_device->get_devices()[0], channel, address, data);
}
}  // namespace tt::test::buffer::detail
