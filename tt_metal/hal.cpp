// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <hal.hpp>
#include <experimental/hal.hpp>
#include <tt-metalium/experimental/context/metalium_env.hpp>
#include <tt-metalium/tt_backend_api_types.hpp>
#include "common/tt_backend_api_types.hpp"
#include <umd/device/types/arch.hpp>
#include <cstdint>
#include <string>

#include "hal_types.hpp"
#include "impl/context/metalium_env_accessor.hpp"
#include "impl/context/metal_context.hpp"

// NOLINTBEGIN(misc-unused-using-decls)
using tt::tt_metal::HalL1MemAddrType;
using tt::tt_metal::HalMemType;
using tt::tt_metal::HalProgrammableCoreType;
// NOLINTEND(misc-unused-using-decls)

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
    const auto& hal_ref = tt::tt_metal::MetalContext::instance().hal();
    return hal_ref.get_dev_addr(HalProgrammableCoreType::ACTIVE_ETH, HalL1MemAddrType::UNRESERVED);
}

uint32_t get_erisc_l1_unreserved_size() {
    const auto& hal_ref = tt::tt_metal::MetalContext::instance().hal();
    return hal_ref.get_dev_size(HalProgrammableCoreType::ACTIVE_ETH, HalL1MemAddrType::UNRESERVED);
}

uint32_t get_max_worker_l1_unreserved_size() {
    const auto& hal_ref = tt::tt_metal::MetalContext::instance().hal();
    size_t l1_end = hal_ref.get_dev_addr(HalProgrammableCoreType::TENSIX, HalL1MemAddrType::BASE) +
                    hal_ref.get_dev_size(HalProgrammableCoreType::TENSIX, HalL1MemAddrType::BASE);
    return l1_end - hal_ref.get_dev_addr(HalProgrammableCoreType::TENSIX, HalL1MemAddrType::KERNEL_CONFIG);
}

float get_eps() { return tt::tt_metal::MetalContext::instance().hal().get_eps(); }

float get_nan() { return tt::tt_metal::MetalContext::instance().hal().get_nan(); }

float get_inf() { return tt::tt_metal::MetalContext::instance().hal().get_inf(); }

uint32_t get_arch_num_circular_buffers() {
    return tt::tt_metal::MetalContext::instance().hal().get_arch_num_circular_buffers();
}

}  // namespace tt::tt_metal::hal

namespace tt::tt_metal::experimental::hal {

tt::ARCH get_arch(const MetaliumEnv& env) { return MetaliumEnvAccessor(env).get_hal().get_arch(); }

std::string get_arch_name(const MetaliumEnv& env) {
    auto arch_enum = MetaliumEnvAccessor(env).get_hal().get_arch();
    return tt::get_string_lowercase(arch_enum);
}

uint32_t get_l1_size(const MetaliumEnv& env) {
    return MetaliumEnvAccessor(env).get_hal().get_dev_size(HalProgrammableCoreType::TENSIX, HalL1MemAddrType::BASE);
}

uint32_t get_dram_alignment(const MetaliumEnv& env) {
    return MetaliumEnvAccessor(env).get_hal().get_alignment(HalMemType::DRAM);
}

uint32_t get_l1_alignment(const MetaliumEnv& env) {
    return MetaliumEnvAccessor(env).get_hal().get_alignment(HalMemType::L1);
}

uint32_t get_pcie_alignment(const MetaliumEnv& env) {
    return MetaliumEnvAccessor(env).get_hal().get_alignment(HalMemType::HOST);
}

uint32_t get_erisc_l1_unreserved_base(const MetaliumEnv& env) {
    const auto& hal_ref = MetaliumEnvAccessor(env).get_hal();
    return hal_ref.get_dev_addr(HalProgrammableCoreType::ACTIVE_ETH, HalL1MemAddrType::UNRESERVED);
}

uint32_t get_erisc_l1_unreserved_size(const MetaliumEnv& env) {
    const auto& hal_ref = MetaliumEnvAccessor(env).get_hal();
    return hal_ref.get_dev_size(HalProgrammableCoreType::ACTIVE_ETH, HalL1MemAddrType::UNRESERVED);
}

uint32_t get_max_worker_l1_unreserved_size(const MetaliumEnv& env) {
    const auto& hal_ref = MetaliumEnvAccessor(env).get_hal();
    size_t l1_end = hal_ref.get_dev_addr(HalProgrammableCoreType::TENSIX, HalL1MemAddrType::BASE) +
                    hal_ref.get_dev_size(HalProgrammableCoreType::TENSIX, HalL1MemAddrType::BASE);
    return l1_end - hal_ref.get_dev_addr(HalProgrammableCoreType::TENSIX, HalL1MemAddrType::KERNEL_CONFIG);
}

float get_eps(const MetaliumEnv& env) { return MetaliumEnvAccessor(env).get_hal().get_eps(); }

float get_nan(const MetaliumEnv& env) { return MetaliumEnvAccessor(env).get_hal().get_nan(); }

float get_inf(const MetaliumEnv& env) { return MetaliumEnvAccessor(env).get_hal().get_inf(); }

uint32_t get_arch_num_circular_buffers(const MetaliumEnv& env) {
    return MetaliumEnvAccessor(env).get_hal().get_arch_num_circular_buffers();
}

}  // namespace tt::tt_metal::experimental::hal
