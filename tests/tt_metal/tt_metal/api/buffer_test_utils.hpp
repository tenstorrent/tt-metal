// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <stdint.h>
#include <vector>

#include <tt-metalium/host_api.hpp>
#include <tt-logger/tt-logger.hpp>
#include <tt-metalium/tt_metal.hpp>

namespace tt::test::buffer::detail {
inline void writeL1Backdoor(
    tt::tt_metal::IDevice* device, CoreCoord coord, uint32_t address, std::vector<uint32_t>& data) {
    log_info(tt::LogTest, "{} -- coord={} address={}", __FUNCTION__, coord.str(), address);
    tt_metal::detail::WriteToDeviceL1(device, coord, address, data);
}
inline void readL1Backdoor(
    tt::tt_metal::IDevice* device, CoreCoord coord, uint32_t address, uint32_t byte_size, std::vector<uint32_t>& data) {
    log_info(tt::LogTest, "{} -- coord={} address={} byte_size={}", __FUNCTION__, coord.str(), address, byte_size);
    tt_metal::detail::ReadFromDeviceL1(device, coord, address, byte_size, data);
}
inline void writeDramBackdoor(
    tt::tt_metal::IDevice* device, uint32_t channel, uint32_t address, std::vector<uint32_t>& data) {
    log_info(tt::LogTest, "{} -- channel={} address={}", __FUNCTION__, channel, address);
    tt_metal::detail::WriteToDeviceDRAMChannel(device, channel, address, data);
}
inline void readDramBackdoor(
    tt::tt_metal::IDevice* device, uint32_t channel, uint32_t address, uint32_t byte_size, std::vector<uint32_t>& data) {
    log_info(tt::LogTest, "{} -- channel={} address={} byte_size={}", __FUNCTION__, channel, address, byte_size);
    tt_metal::detail::ReadFromDeviceDRAMChannel(device, channel, address, byte_size, data);
}
}  // namespace tt::test::buffer::detail
