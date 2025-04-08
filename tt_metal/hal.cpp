// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <hal.hpp>
#include <tt_backend_api_types.hpp>
#include <umd/device/types/arch.h>
#include <cstdint>
#include <string>

#include "hal_types.hpp"
#include "llrt/hal.hpp"

using tt::tt_metal::HalL1MemAddrType;
using tt::tt_metal::HalMemType;
using tt::tt_metal::HalProgrammableCoreType;
using tt::tt_metal::HalSingleton;

namespace tt::tt_metal::hal {

tt::ARCH get_arch() { return HalSingleton::getInstance().get_arch(); }

std::string get_arch_name() {
    auto arch_enum = HalSingleton::getInstance().get_arch();
    return tt::get_string_lowercase(arch_enum);
}

uint32_t get_l1_size() {
    return HalSingleton::getInstance().get_dev_size(HalProgrammableCoreType::TENSIX, HalL1MemAddrType::BASE);
}

uint32_t get_dram_alignment() { return HalSingleton::getInstance().get_alignment(HalMemType::DRAM); }

uint32_t get_l1_alignment() { return HalSingleton::getInstance().get_alignment(HalMemType::L1); }

uint32_t get_pcie_alignment() { return HalSingleton::getInstance().get_alignment(HalMemType::HOST); }

uint32_t get_erisc_l1_unreserved_base() {
    auto& hal_ref = HalSingleton::getInstance();
    if (hal_ref.get_arch() != tt::ARCH::GRAYSKULL) {
        return hal_ref.get_dev_addr(HalProgrammableCoreType::ACTIVE_ETH, HalL1MemAddrType::UNRESERVED);
    }
    return 0;
}

uint32_t get_erisc_l1_unreserved_size() {
    auto& hal_ref = HalSingleton::getInstance();
    if (hal_ref.get_arch() != tt::ARCH::GRAYSKULL) {
        return hal_ref.get_dev_size(HalProgrammableCoreType::ACTIVE_ETH, HalL1MemAddrType::UNRESERVED);
    }
    return 0;
}

uint32_t get_tensix_l1_unreserved_base() {
    auto& hal_ref = HalSingleton::getInstance();
    if (hal_ref.get_arch() != tt::ARCH::GRAYSKULL) {
        return hal_ref.get_dev_addr(HalProgrammableCoreType::TENSIX, HalL1MemAddrType::UNRESERVED);
    }
    return 0;
}

uint32_t get_tensix_l1_unreserved_size() {
    auto& hal_ref = HalSingleton::getInstance();
    if (hal_ref.get_arch() != tt::ARCH::GRAYSKULL) {
        return hal_ref.get_dev_size(HalProgrammableCoreType::TENSIX, HalL1MemAddrType::UNRESERVED);
    }
    return 0;
}

float get_eps() { return HalSingleton::getInstance().get_eps(); }

float get_nan() { return HalSingleton::getInstance().get_nan(); }

float get_inf() { return HalSingleton::getInstance().get_inf(); }

}  // namespace tt::tt_metal::hal
