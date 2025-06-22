// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

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

namespace tt::tt_metal {
namespace tunneling {

struct MmmioAndEthDeviceDesc {
    IDevice* mmio_device = nullptr;
    IDevice* eth_device = nullptr;
    std::optional<CoreCoord> mmio_eth = std::nullopt;
    std::optional<CoreCoord> eth_to_init = std::nullopt;
};

inline void get_mmio_device_and_eth_device_to_init(const std::vector<IDevice*>& devices, MmmioAndEthDeviceDesc& desc) {
    for (auto device : devices) {
        if (device->id() == MetalContext::instance().get_cluster().get_associated_mmio_device(device->id())) {
            desc.mmio_device = device;
            // whichever chip this is connected to will be considered the remote chip
            for (const auto& active_eth : device->get_active_ethernet_cores()) {
                if (MetalContext::instance().get_cluster().is_ethernet_link_up(desc.mmio_device->id(), active_eth)) {
                    desc.mmio_eth = active_eth;
                    auto connected_chip_eth = MetalContext::instance().get_cluster().get_connected_ethernet_core(
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

inline Program create_eth_init_program(const MmmioAndEthDeviceDesc& desc, bool init_all_eth_cores) {
    // Create a program on the MMIO device with the kernel that is responsible for loading itself onto the remote eth.
    // This kernel will stall until it receives a signal from the remote eth core.
    // Remote eth core will complete the handshake only after all ethernets on its chip have been initialized.
    tt_metal::Program mmio_program = tt_metal::Program();

    std::unordered_map<CoreCoord, KernelHandle> mmio_eth_to_kernel;

    uint32_t eth_chans_mask = 0;
    for (const auto& core : desc.mmio_device->get_active_ethernet_cores()) {
        if (!init_all_eth_cores && core != desc.mmio_eth.value()) {
            continue;  // Skip other eth cores if we are initializing only one
        }
        if (!MetalContext::instance().get_cluster().is_ethernet_link_up(desc.mmio_device->id(), core)) {
            continue;
        }

        auto connected_chip_eth =
            MetalContext::instance().get_cluster().get_connected_ethernet_core({desc.mmio_device->id(), core});
        auto remote_device_id = std::get<0>(connected_chip_eth);
        std::cout << "Eth core is " << core.str() << " remote device " << remote_device_id << std::endl;
        auto kernel_handle = tt_metal::CreateKernel(
            mmio_program,
            "tests/tt_metal/tt_metal/tunneling/kernels/lite_fabric_handshake.cpp",
            core,
            tt_metal::EthernetConfig{.noc = tt_metal::NOC::NOC_0});
        mmio_eth_to_kernel[core] = kernel_handle;
        eth_chans_mask += 0x1 << (uint32_t)core.y;
    }
    uint32_t num_local_eths = mmio_eth_to_kernel.size();

    // Compile the program because we need to write the binary into mmio eth core so it can send it over
    tt_metal::detail::CompileProgram(desc.mmio_device, mmio_program);

    // Extract the binary and write it to the mmio eth core
    const auto& kernels = mmio_program.get_kernels(static_cast<uint32_t>(HalProgrammableCoreType::ACTIVE_ETH));
    auto eth_kernel = kernels.at(mmio_eth_to_kernel.at(desc.mmio_eth.value()));

    const ll_api::memory& binary_mem =
        *tt_metal::KernelImpl::from(*eth_kernel)
             .binaries(BuildEnvManager::get_instance().get_device_build_env(desc.mmio_device->build_id()).build_key)[0];

    auto num_spans = binary_mem.num_spans();
    uint32_t erisc_core_type =
        MetalContext::instance().hal().get_programmable_core_type_index(HalProgrammableCoreType::ACTIVE_ETH);
    uint32_t processor_class_idx = magic_enum::enum_integer(HalProcessorClassType::DM);
    int processor_type_idx = magic_enum::enum_integer(std::get<EthernetConfig>(eth_kernel->config()).processor);

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

    std::cout << "dst_binary_address: " << dst_binary_address << " binary_size_bytes " << binary_size_bytes
              << std::endl;

    auto primary_eth_core = desc.mmio_device->ethernet_core_from_logical_core(desc.mmio_eth.value());

    std::vector<uint32_t> lite_fabric_metadata(256, 0);
    for (const auto& [core, kernel_handle] : mmio_eth_to_kernel) {
        uint32_t initial_state = (core == desc.mmio_eth.value()) ? 0 : 1;
        lite_fabric_metadata[0] = initial_state;
        lite_fabric_metadata[1] = dst_binary_address;
        lite_fabric_metadata[2] = binary_size_bytes;
        lite_fabric_metadata[3] = init_all_eth_cores;  // test only
        lite_fabric_metadata[4] = (uint32_t)primary_eth_core.x;
        lite_fabric_metadata[5] = (uint32_t)primary_eth_core.y;
        lite_fabric_metadata[6] = eth_chans_mask;
        lite_fabric_metadata[7] = num_local_eths;

        tt_metal::SetRuntimeArgs(mmio_program, kernel_handle, core, lite_fabric_metadata);
    }

    return mmio_program;
}

}  // namespace tunneling
}  // namespace tt::tt_metal
