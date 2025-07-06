// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <gtest/gtest.h>
#include <map>
#include <optional>
#include <utility>
#include <variant>
#include <vector>
#include <random>

#include <tt-metalium/allocator.hpp>
#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/data_types.hpp>
#include <tt-metalium/device_pool.hpp>
#include <tt-metalium/device.hpp>
#include <tt-metalium/hal.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/tt_metal.hpp>
#include "tt_metal/jit_build/build_env_manager.hpp"
#include "impl/kernels/kernel_impl.hpp"
#include "tests/tt_metal/tt_metal/tunneling/kernels/lite_fabric.h"

// namespace tt::tt_metal {
namespace tunneling {

struct TestConfig {
    bool init_all_eth_cores = false;
    bool init_handshake_only = true;
};

struct MmmioAndEthDeviceDesc {
    tt::tt_metal::IDevice* mmio_device = nullptr;
    tt::tt_metal::IDevice* eth_device = nullptr;
    std::optional<CoreCoord> mmio_eth = std::nullopt;
    std::optional<CoreCoord> eth_to_init = std::nullopt;
};

inline void get_mmio_device_and_eth_device_to_init(
    const std::vector<tt::tt_metal::IDevice*>& devices, MmmioAndEthDeviceDesc& desc) {
    for (auto device : devices) {
        if (device->id() ==
            tt::tt_metal::MetalContext::instance().get_cluster().get_associated_mmio_device(device->id())) {
            desc.mmio_device = device;
            // whichever chip this is connected to will be considered the remote chip
            for (const auto& active_eth : device->get_active_ethernet_cores()) {
                if (tt::tt_metal::MetalContext::instance().get_cluster().is_ethernet_link_up(
                        desc.mmio_device->id(), active_eth)) {
                    desc.mmio_eth = active_eth;
                    auto connected_chip_eth =
                        tt::tt_metal::MetalContext::instance().get_cluster().get_connected_ethernet_core(
                            {desc.mmio_device->id(), active_eth});
                    auto remote_device_id = std::get<0>(connected_chip_eth);
                    desc.eth_to_init = std::get<1>(connected_chip_eth);
                    for (auto potential_remote_device : devices) {
                        if (potential_remote_device->id() == remote_device_id) {
                            desc.eth_device = potential_remote_device;
                            break;
                        }
                    }
                    break;
                }
            }
            if (desc.eth_device != nullptr and desc.eth_to_init.has_value()) {
                break;
            }
        }
    }

    if (desc.mmio_device == nullptr || desc.eth_device == nullptr || !desc.mmio_eth.has_value() ||
        !desc.eth_to_init.has_value()) {
        GTEST_SKIP() << "Skipping test, could not find connected devices to act as mmio and eth connected device";
    }

    std::cout << "MMIO Device: " << desc.mmio_device->id() << ", Remote Device: " << desc.eth_device->id() << std::endl;
    std::cout << "MMIO Chip Eth: " << desc.mmio_eth->str() << ", Remote Chip Eth: " << desc.eth_to_init->str()
              << std::endl;
}

inline tt::tt_metal::Program create_eth_init_program(const MmmioAndEthDeviceDesc& desc, const TestConfig& config) {
    // Create a program on the MMIO device with the kernel that is responsible for loading itself onto the remote eth.
    // This kernel will stall until it receives a signal from the remote eth core.
    // Remote eth core will complete the handshake only after all ethernets on its chip have been initialized.
    tt::tt_metal::Program mmio_program = tt::tt_metal::Program();

    std::unordered_map<CoreCoord, tt::tt_metal::KernelHandle> mmio_eth_to_kernel;

    const std::string kernel_file = config.init_handshake_only
                                        ? "tests/tt_metal/tt_metal/tunneling/kernels/lite_fabric_handshake.cpp"
                                        : "tests/tt_metal/tt_metal/tunneling/kernels/lite_fabric.cpp";

    uint32_t eth_chans_mask = 0;
    for (const auto& core : desc.mmio_device->get_active_ethernet_cores()) {
        if (!config.init_all_eth_cores && core != desc.mmio_eth.value()) {
            continue;  // Skip other eth cores if we are initializing only one
        }
        if (!tt::tt_metal::MetalContext::instance().get_cluster().is_ethernet_link_up(desc.mmio_device->id(), core)) {
            continue;
        }

        auto connected_chip_eth = tt::tt_metal::MetalContext::instance().get_cluster().get_connected_ethernet_core(
            {desc.mmio_device->id(), core});
        auto remote_device_id = std::get<0>(connected_chip_eth);
        std::cout << "Eth core is " << core.str() << " remote device " << remote_device_id << std::endl;
        auto kernel_handle = tt::tt_metal::CreateKernel(
            mmio_program, kernel_file, core, tt::tt_metal::EthernetConfig{.noc = tt::tt_metal::NOC::NOC_0});
        mmio_eth_to_kernel[core] = kernel_handle;
        eth_chans_mask += 0x1 << (uint32_t)core.y;
    }
    uint32_t num_local_eths = mmio_eth_to_kernel.size();

    // Compile the program because we need to write the binary into mmio eth core so it can send it over
    tt::tt_metal::detail::CompileProgram(desc.mmio_device, mmio_program);

    // Extract the binary and write it to the mmio eth core
    const auto& kernels =
        mmio_program.get_kernels(static_cast<uint32_t>(tt::tt_metal::HalProgrammableCoreType::ACTIVE_ETH));
    auto eth_kernel = kernels.at(mmio_eth_to_kernel.at(desc.mmio_eth.value()));

    const ll_api::memory& binary_mem = *tt::tt_metal::KernelImpl::from(*eth_kernel)
                                            .binaries(tt::tt_metal::BuildEnvManager::get_instance()
                                                          .get_device_build_env(desc.mmio_device->build_id())
                                                          .build_key)[0];

    auto num_spans = binary_mem.num_spans();
    uint32_t erisc_core_type = tt::tt_metal::MetalContext::instance().hal().get_programmable_core_type_index(
        tt::tt_metal::HalProgrammableCoreType::ACTIVE_ETH);
    uint32_t processor_class_idx = magic_enum::enum_integer(tt::tt_metal::HalProcessorClassType::DM);
    int processor_type_idx =
        magic_enum::enum_integer(std::get<tt::tt_metal::EthernetConfig>(eth_kernel->config()).processor);

    TT_FATAL(
        binary_mem.num_spans() == 1,
        "Expected 1 binary span for lite fabric handshake kernel, got {}",
        binary_mem.num_spans());

    uint64_t local_init_addr = tt::tt_metal::MetalContext::instance()
                                   .hal()
                                   .get_jit_build_config(erisc_core_type, processor_class_idx, processor_type_idx)
                                   .local_init_addr;
    uint32_t dst_binary_address;
    uint32_t binary_size_bytes;
    binary_mem.process_spans([&](std::vector<uint32_t>::const_iterator mem_ptr, uint64_t addr, uint32_t len_words) {
        uint32_t relo_addr = tt::tt_metal::MetalContext::instance().hal().relocate_dev_addr(addr, local_init_addr);
        dst_binary_address = relo_addr;
        binary_size_bytes = len_words * sizeof(uint32_t);
    });

    binary_size_bytes = (binary_size_bytes + 15) & ~0xF;  // Round up to the nearest 16 bytes
    std::cout << "dst_binary_address: " << dst_binary_address << " binary_size_bytes " << binary_size_bytes
              << std::endl;

    auto primary_eth_core = desc.mmio_device->ethernet_core_from_logical_core(desc.mmio_eth.value());

    std::vector<uint32_t> lite_fabric_rtargs(256, 0);
    lite_fabric_rtargs[0] = tt::tt_metal::MetalContext::instance().hal().get_dev_addr(
        tt::tt_metal::HalProgrammableCoreType::ACTIVE_ETH, tt::tt_metal::HalL1MemAddrType::UNRESERVED);

    for (const auto& [core, kernel_handle] : mmio_eth_to_kernel) {
        CoreCoord virtual_eth_core = desc.mmio_device->ethernet_core_from_logical_core(core);
        LiteFabricInitState initial_state = (core == desc.mmio_eth.value())
                                                ? LiteFabricInitState::MMIO_ETH_INIT_NEIGHBOUR
                                                : LiteFabricInitState::LOCAL_HANDSHAKE;
        tunneling::lite_fabric_config_t lite_fabric_config;
        lite_fabric_config.binary_address = dst_binary_address;
        lite_fabric_config.binary_size_bytes = binary_size_bytes;
        lite_fabric_config.eth_chans_mask = eth_chans_mask;
        lite_fabric_config.num_local_eths = num_local_eths;
        lite_fabric_config.primary_eth_core_x = (uint32_t)primary_eth_core.x;
        lite_fabric_config.primary_eth_core_y = (uint32_t)primary_eth_core.y;
        lite_fabric_config.init_state = static_cast<LiteFabricInitState>(initial_state);
        lite_fabric_config.multi_eth_cores_setup = config.init_all_eth_cores;

        tt::tt_metal::MetalContext::instance().get_cluster().write_core(
            (void*)&lite_fabric_config,
            sizeof(lite_fabric_config_t),
            tt_cxy_pair(desc.mmio_device->id(), virtual_eth_core),
            lite_fabric_rtargs[0]);

        tt::tt_metal::SetRuntimeArgs(mmio_program, kernel_handle, core, lite_fabric_rtargs);
    }

    return mmio_program;
}

}  // namespace tunneling
// }  // namespace tt::tt_metal
