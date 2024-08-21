// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "tt_metal/host_api.hpp"

namespace tt::test::buffer::detail {
void writeL1Backdoor(tt::tt_metal::Device* device, CoreCoord coord, uint32_t address, std::vector<uint32_t>& data);
void readL1Backdoor(tt::tt_metal::Device* device, CoreCoord coord, uint32_t address, uint32_t byte_size, std::vector<uint32_t>& data);
void writeDramBackdoor(tt::tt_metal::Device* device, uint32_t channel, uint32_t address, std::vector<uint32_t>& data);
void readDramBackdoor(
    tt::tt_metal::Device* device, uint32_t channel, uint32_t address, uint32_t byte_size, std::vector<uint32_t>& data);
}  // namespace tt::test::buffer::detail
