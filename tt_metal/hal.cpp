// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <hal.hpp>
#include <tt_backend_api_types.hpp>
#include <umd/device/types/arch.h>
#include <cstdint>
#include <string>

#include "hal_types.hpp"
#include "impl/context/metal_context.hpp"

using tt::tt_metal::HalL1MemAddrType;
using tt::tt_metal::HalMemType;
using tt::tt_metal::HalProgrammableCoreType;

namespace tt::tt_metal::hal {

tt::ARCH get_arch() { return tt::tt_metal::MetalContext::instance().hal().get_arch(); }

std::string get_arch_name() {
    auto arch_enum = tt::tt_metal::MetalContext::instance().hal().get_arch();
    return tt::get_string_lowercase(arch_enum);
}

uint32_t get_l1_size() {
    return tt::tt_metal::MetalContext::instance().hal().get_dev_size(
        HalProgrammableCoreType::TENSIX, HalL1MemAddrType::BASE);
}

uint32_t get_dram_alignment() { return tt::tt_metal::MetalContext::instance().hal().get_alignment(HalMemType::DRAM); }

uint32_t get_l1_alignment() { return tt::tt_metal::MetalContext::instance().hal().get_alignment(HalMemType::L1); }

uint32_t get_pcie_alignment() { return tt::tt_metal::MetalContext::instance().hal().get_alignment(HalMemType::HOST); }

uint32_t get_erisc_l1_unreserved_base() {
    auto& hal_ref = tt::tt_metal::MetalContext::instance().hal();
    return hal_ref.get_dev_addr(HalProgrammableCoreType::ACTIVE_ETH, HalL1MemAddrType::UNRESERVED);
}

uint32_t get_erisc_l1_unreserved_size() {
    auto& hal_ref = tt::tt_metal::MetalContext::instance().hal();
    return hal_ref.get_dev_size(HalProgrammableCoreType::ACTIVE_ETH, HalL1MemAddrType::UNRESERVED);
}

uint32_t get_max_worker_l1_unreserved_size() {
    auto& hal_ref = tt::tt_metal::MetalContext::instance().hal();
    size_t l1_end = hal_ref.get_dev_addr(HalProgrammableCoreType::TENSIX, HalL1MemAddrType::BASE) +
                    hal_ref.get_dev_size(HalProgrammableCoreType::TENSIX, HalL1MemAddrType::BASE);
    return l1_end - hal_ref.get_dev_addr(HalProgrammableCoreType::TENSIX, HalL1MemAddrType::KERNEL_CONFIG);
}

float get_eps() { return tt::tt_metal::MetalContext::instance().hal().get_eps(); }

float get_nan() { return tt::tt_metal::MetalContext::instance().hal().get_nan(); }

float get_inf() { return tt::tt_metal::MetalContext::instance().hal().get_inf(); }

}  // namespace tt::tt_metal::hal
