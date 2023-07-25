#pragma once

#include "tt_metal/host_api.hpp"

namespace tt::test::buffer::detail {
void writeL1Backdoor(Device* device, CoreCoord coord, uint32_t address, std::vector<uint32_t>& data);
void readL1Backdoor(Device* device, CoreCoord coord, uint32_t address, uint32_t byte_size, std::vector<uint32_t>& data);
void writeDramBackdoor(Device* device, uint32_t channel, uint32_t address, std::vector<uint32_t>& data);
void readDramBackdoor(
    Device* device, uint32_t channel, uint32_t address, uint32_t byte_size, std::vector<uint32_t>& data);
}  // namespace tt::test::buffer::detail
