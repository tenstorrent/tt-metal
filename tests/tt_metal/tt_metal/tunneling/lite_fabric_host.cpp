// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <utility>
#include <memory>

#include "host_api.hpp"
#include "tt_metal.hpp"
#include "fabric_lite.hpp"

#include "kernel_types.hpp"
#include "llrt.hpp"
#include "program.hpp"
#include "rtoptions.hpp"
#include "tt_align.hpp"
#include "tt_cluster.hpp"
#include "assert.hpp"
#include "context/metal_context.hpp"
#include "hal.hpp"
#include "hal_types.hpp"
#include "jit_build/build_env_manager.hpp"
#include "impl/kernels/kernel_impl.hpp"
#include "core_coord.hpp"
#include "data_types.hpp"
#include "lite_fabric_host.hpp"
#include <tt-metalium/control_plane.hpp>

namespace lite_fabric {

uint32_t GetEthChannelMask(chip_id_t device_id) {
    auto& cp = tt::tt_metal::MetalContext::instance().get_control_plane();

    uint32_t mask = 0;
    for (const auto& core : cp.get_active_ethernet_cores(device_id)) {
        mask |= 0x1 << core.y;
    }

    return mask;
}

SystemDescriptor GetSystemDescriptor2Devices(
    const auto& devices, chip_id_t mmio_device_id, chip_id_t connected_device_id) {
    auto& cluster = tt::tt_metal::MetalContext::instance().get_cluster();

    SystemDescriptor desc;

    // Get the eth mask for each device
    desc.enabled_eth_channels[mmio_device_id] = GetEthChannelMask(mmio_device_id);
    desc.enabled_eth_channels[connected_device_id] = GetEthChannelMask(connected_device_id);

    for (const auto& mmio_eth_core : cluster.get_ethernet_sockets(mmio_device_id, connected_device_id)) {
        const auto& [other_device, other_core] = cluster.get_connected_ethernet_core({mmio_device_id, mmio_eth_core});
        TT_FATAL(
            other_device == connected_device_id,
            "Error in ethernet core descriptors. Expected other device id to be {} but got {}",
            connected_device_id,
            other_device);
        desc.tunnels_from_mmio.push_back(TunnelDescriptor{
            .mmio_id = mmio_device_id,
            .mmio_core =
                cluster.get_virtual_coordinate_from_logical_coordinates(mmio_device_id, mmio_eth_core, CoreType::ETH),
            .connected_id = other_device,
            .connected_core =
                cluster.get_virtual_coordinate_from_logical_coordinates(other_device, other_core, CoreType::ETH),
        });
        log_info(
            tt::LogMetal, "Add tunnel from {} {} to {} {}", mmio_device_id, mmio_eth_core, other_device, other_core);
    }

    return desc;
}

uint32_t GetLocalInitAddr(std::shared_ptr<tt::tt_metal::Kernel> kernel) {
    auto erisc_core_dx = tt::tt_metal::MetalContext::instance().hal().get_programmable_core_type_index(
        kernel->get_kernel_programmable_core_type());
    auto processor_class_idx = magic_enum::enum_integer(tt::tt_metal::HalProcessorClassType::DM);
    auto processor_type_idx =
        magic_enum::enum_integer(std::get<tt::tt_metal::EthernetConfig>(kernel->config()).processor);

    return tt::tt_metal::MetalContext::instance()
        .hal()
        .get_jit_build_config(erisc_core_dx, processor_class_idx, processor_type_idx)
        .local_init_addr;
}

// Compile Lite Fabric and return the binary. The device is needed due to JIT Build requirements
std::pair<const ll_api::memory&, uint32_t> CompileLiteFabric(
    tt::tt_metal::IDevice* device, const CoreCoord& logical_core) {
    const std::string k_LiteFabricPath = "tests/tt_metal/tt_metal/tunneling/tunnel.cpp";
    // Opening the device is needed because jit build system requirements
    log_info(tt::LogMetal, "Building LiteFabric. Reference device {} {}", device->id(), logical_core);
    auto pgm = std::make_unique<tt::tt_metal::Program>();
    std::map<std::string, std::string> defines;
    defines["COMPILE_FOR_LITE_FABRIC"] = "1";
    auto kernel = tt::tt_metal::CreateKernel(
        *pgm,
        k_LiteFabricPath,
        logical_core,
        tt::tt_metal::EthernetConfig{.eth_mode = tt::tt_metal::SENDER, .compile_args = {}, .defines = defines});

    pgm->compile(device, true);

    auto& build_env = tt::tt_metal::BuildEnvManager::get_instance().get_device_build_env(device->build_id());

    const auto& kernels = pgm->get_kernels(static_cast<uint32_t>(tt::tt_metal::HalProgrammableCoreType::ACTIVE_ETH));
    auto eth_kernel = kernels.at(kernel);

    const ll_api::memory& bin = *tt::tt_metal::KernelImpl::from(*eth_kernel).binaries(build_env.build_key)[0];

    TT_FATAL(bin.num_spans() == 1, "Expected 1 binary span for lite fabric kernel, got {}", bin.num_spans());

    auto local_init = GetLocalInitAddr(eth_kernel);

    return {bin, local_init};
}

void SetResetState(std::shared_ptr<tt::Cluster> cluster, tt_cxy_pair virtual_core, bool assert_reset) {
    // We run on DM1. Don't touch DM0. It is running base firmware
    TensixSoftResetOptions reset_val = TENSIX_ASSERT_SOFT_RESET;
    if (assert_reset) {
        reset_val = reset_val & static_cast<TensixSoftResetOptions>(
                                    ~std::underlying_type<TensixSoftResetOptions>::type(TensixSoftResetOptions::BRISC));
        cluster->assert_risc_reset_at_core(virtual_core, reset_val);
    } else {
        reset_val = TENSIX_DEASSERT_SOFT_RESET &
                    static_cast<TensixSoftResetOptions>(
                        ~std::underlying_type<TensixSoftResetOptions>::type(TensixSoftResetOptions::TRISC0));
        cluster->deassert_risc_reset_at_core(virtual_core, reset_val);
    }
}

void SetPC(std::shared_ptr<tt::Cluster> cluster, tt_cxy_pair virtual_core, uint32_t pc_addr, uint32_t pc_val) {
    cluster->write_core((void*)&pc_val, sizeof(uint32_t), virtual_core, pc_addr);
}

void LaunchLiteFabric() {
    constexpr chip_id_t k_MMIODeviceId = 0;
    constexpr chip_id_t k_OtherDeviceId = 1;
    auto devices = tt::tt_metal::detail::CreateDevices({k_MMIODeviceId, k_OtherDeviceId});
    auto mmio_device = devices[0];

    SystemDescriptor desc = GetSystemDescriptor2Devices(devices, k_MMIODeviceId, k_OtherDeviceId);
    TT_FATAL(desc.tunnels_from_mmio.size(), "No ethernet cores found from device {}", k_MMIODeviceId);
    auto [bin, local_init_addr] = CompileLiteFabric(mmio_device, desc.tunnels_from_mmio.front().mmio_core);

    uint32_t text_start = bin.get_text_addr();
    uint32_t bin_size = tt::align(bin.get_text_size() + 1, 16);
    uint32_t lite_fabric_config_addr = tt::tt_metal::MetalContext::instance().hal().get_dev_addr(
        tt::tt_metal::HalProgrammableCoreType::ACTIVE_ETH, tt::tt_metal::HalL1MemAddrType::FABRIC_LITE_CONFIG);

    // Dont need the device anymore
    tt::tt_metal::detail::CloseDevices(devices);

    auto rtoptions = tt::llrt::RunTimeOptions();
    auto hal = tt::tt_metal::Hal(tt::ARCH::BLACKHOLE, false);
    auto cluster = std::make_shared<tt::Cluster>(rtoptions, hal);

    log_info(
        tt::LogMetal,
        "Launching Lite Fabric Binary Text Addr: {:#x}, Size: {:#x}, Config: {:#x}, Local Init: {:#x}",
        bin.get_text_addr(),
        bin_size,
        lite_fabric_config_addr,
        local_init_addr);

    // 1. Initialize configuration struct on MMIO cores
    LiteFabricConfig config;
    config.binary_addr = text_start;
    config.binary_size = bin_size;
    config.initial_state = InitState::ETH_HANDSHAKE_LOCAL;
    config.current_state = InitState::ETH_HANDSHAKE_LOCAL;

    for (auto& tunnel : desc.tunnels_from_mmio) {
        // It should already be in reset because metal is not running (devices were closed)
        log_info(tt::LogMetal, "Assert reset on {}", tunnel.mmio_cxy().str());
        SetResetState(cluster, tunnel.mmio_cxy(), true);

        cluster->write_core((void*)&config, sizeof(LiteFabricConfig), tunnel.mmio_cxy(), lite_fabric_config_addr);
    }

    cluster->l1_barrier(k_MMIODeviceId);
    log_info(tt::LogMetal, "Configuration written to MMIO cores");

    for (auto& tunnel : desc.tunnels_from_mmio) {
        auto cxy = tt_cxy_pair{tunnel.mmio_id, tunnel.mmio_core};
        SetPC(cluster, cxy, 0xFFB00000 | 0x14008, text_start);

        bin.process_spans([&](std::vector<uint32_t>::const_iterator mem_ptr, uint64_t addr, uint32_t len_words) {
            uint64_t relo_addr = hal.relocate_dev_addr(addr, local_init_addr);

            cluster->write_core((void*)&config, sizeof(LiteFabricConfig), cxy, text_start);
        });
    }

    cluster->l1_barrier(k_MMIODeviceId);

    log_info(tt::LogMetal, "Binary written to MMIO cores");
    for (auto& tunnel : desc.tunnels_from_mmio) {
        SetResetState(cluster, tunnel.mmio_cxy(), false);
    }

    while (true) {
        __asm__ volatile("nop");
    }
}

}  // namespace lite_fabric
