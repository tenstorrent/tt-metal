// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "tt_metal/experimental/hal.hpp"
#include "tt_metal/llrt/hal.hpp"

using tt::tt_metal::HalL1MemAddrType;
using tt::tt_metal::HalMemType;
using tt::tt_metal::HalProgrammableCoreType;
using tt::tt_metal::HalSingleton;

namespace tt::tt_metal::experimental::hal {

uint32_t get_l1_size() {
    return HalSingleton::getInstance().get_dev_size(HalProgrammableCoreType::TENSIX, HalL1MemAddrType::BASE);
}

uint32_t get_dram_alignment() { return HalSingleton::getInstance().get_alignment(HalMemType::DRAM); }

uint32_t get_l1_alignment() { return HalSingleton::getInstance().get_alignment(HalMemType::L1); }

uint32_t get_pcie_alignment() { return HalSingleton::getInstance().get_alignment(HalMemType::HOST); }

}  // namespace tt::tt_metal::experimental::hal
