// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <utility>
#include <memory>

#include "host_api.hpp"
#include "tt_metal.hpp"
#include "fabric_lite.hpp"

#include "kernel_types.hpp"
#include "program.hpp"
#include "rtoptions.hpp"
#include "tt_cluster.hpp"
#include "assert.hpp"
#include "context/metal_context.hpp"
#include "hal_types.hpp"
#include "jit_build/build_env_manager.hpp"
#include "impl/kernels/kernel_impl.hpp"
#include "core_coord.hpp"
#include "data_types.hpp"
#include "lite_fabric_host.hpp"
#include <umd/device/types/xy_pair.h>
#include <tt-metalium/control_plane.hpp>

namespace {
uint32_t GetStateAddress() {
    uint32_t state_addr =
        tt::tt_metal::MetalContext::instance().hal().get_dev_addr(
            tt::tt_metal::HalProgrammableCoreType::ACTIVE_ETH, tt::tt_metal::HalL1MemAddrType::FABRIC_LITE_CONFIG) +
        offsetof(lite_fabric::LiteFabricMemoryMap, config) + offsetof(lite_fabric::LiteFabricConfig, current_state);
    return state_addr;
}
}  // namespace

namespace lite_fabric {

using chip_id_t = tt::umd::chip_id_t;

uint32_t GetEthChannelMask(chip_id_t device_id) {
    auto& cp = tt::tt_metal::MetalContext::instance().get_control_plane();

    uint32_t mask = 0;
    for (const auto& core : cp.get_active_ethernet_cores(device_id)) {
        mask |= 0x1 << core.y;
    }

    return mask;
}

SystemDescriptor GetSystemDescriptor2Devices(
    const std::map<chip_id_t, tt::tt_metal::IDevice*>& devices,
    chip_id_t mmio_device_id,
    chip_id_t connected_device_id) {
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
            .mmio_core_virtual =
                cluster.get_virtual_coordinate_from_logical_coordinates(mmio_device_id, mmio_eth_core, CoreType::ETH),
            .mmio_core_logical = mmio_eth_core,
            .connected_id = other_device,
            .connected_core_virtual =
                cluster.get_virtual_coordinate_from_logical_coordinates(other_device, other_core, CoreType::ETH),
            .connected_core_logical = other_core,
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

std::pair<const ll_api::memory&, uint32_t> GetBinaryMetadata(
    uint32_t build_id, tt::tt_metal::Program& pgm, tt::tt_metal::KernelHandle kernel_handle) {
    auto& build_env = tt::tt_metal::BuildEnvManager::get_instance().get_device_build_env(build_id);

    const auto& kernels = pgm.get_kernels(static_cast<uint32_t>(tt::tt_metal::HalProgrammableCoreType::ACTIVE_ETH));
    auto eth_kernel = kernels.at(kernel_handle);

    const ll_api::memory& bin = *tt::tt_metal::KernelImpl::from(*eth_kernel).binaries(build_env.build_key)[0];

    TT_FATAL(bin.num_spans() == 1, "Expected 1 binary span for lite fabric kernel, got {}", bin.num_spans());

    auto local_init = GetLocalInitAddr(eth_kernel);

    return {bin, local_init};
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

    return GetBinaryMetadata(device->build_id(), *pgm, kernel);
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

void wait_for_state(tt::Cluster& cluster, tt_cxy_pair virtual_core, lite_fabric::InitState state) {
    uint32_t state_addr = GetStateAddress();
    std::vector<uint32_t> readback{static_cast<uint32_t>(lite_fabric::InitState::UNKNOWN)};
    while (static_cast<lite_fabric::InitState>(readback[0]) != state) {
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
        cluster.read_core(readback, sizeof(uint32_t), virtual_core, state_addr);
    }
}

std::unique_ptr<tt::tt_metal::Program> LaunchLiteFabricWithMetal(
    std::map<chip_id_t, tt::tt_metal::IDevice*>& devices, const SystemDescriptor& desc) {
    log_info(tt::LogMetal, "Eth Mask = {:0b}", desc.enabled_eth_channels.at(0));

    lite_fabric::LiteFabricConfig config{};
    config.is_primary = true;
    config.is_mmio = true;
    config.initial_state = lite_fabric::InitState::ETH_INIT_NEIGHBOUR;
    config.current_state = lite_fabric::InitState::ETH_INIT_NEIGHBOUR;
    config.binary_addr = 0;
    config.binary_size = 0;
    config.eth_chans_mask = desc.enabled_eth_channels.at(0);

    // Compile kernels
    auto pgm = std::make_unique<tt::tt_metal::Program>();
    for (auto tunnel_1x : desc.tunnels_from_mmio) {
        log_info(
            tt::LogMetal,
            "Host to initialize Device {} {} (virtual={})",
            0,
            tunnel_1x.mmio_core_logical,
            tunnel_1x.mmio_core_virtual);
        tunnel_1x.mmio_kernel = tt::tt_metal::CreateKernel(
            *pgm,
            "tests/tt_metal/tt_metal/tunneling/lite_fabric.cpp",
            tunnel_1x.mmio_core_logical,
            tt::tt_metal::EthernetConfig{
                .eth_mode = tt::tt_metal::SENDER,
                .compile_args = {},
                .defines =
                    {
                        {"METAL_LAUNCH", "1"},
                        {"COMPILE_FOR_LITE_FABRIC", "1"},
                    },
            });
    }
    pgm->compile(devices[0]);

    // Write configuration struct
    auto& cluster = tt::tt_metal::MetalContext::instance().get_cluster();
    auto config_addr = tt::tt_metal::MetalContext::instance().hal().get_dev_addr(
        tt::tt_metal::HalProgrammableCoreType::ACTIVE_ETH, tt::tt_metal::HalL1MemAddrType::FABRIC_LITE_CONFIG);
    for (auto tunnel_1x : desc.tunnels_from_mmio) {
        // These should be the same for all cores
        auto [bin, local_init] = lite_fabric::GetBinaryMetadata(devices[0]->build_id(), *pgm, tunnel_1x.mmio_kernel);
        bin.process_spans([&](std::vector<uint32_t>::const_iterator mem_ptr, uint64_t addr, uint32_t len_words) {
            uint32_t relo_addr = tt::tt_metal::MetalContext::instance().hal().relocate_dev_addr(addr, local_init);
            config.binary_addr = relo_addr;
            config.binary_size = (len_words * sizeof(uint32_t));
            config.binary_size = (config.binary_size + 15) & ~0xF;
        });

        log_info(
            tt::LogMetal,
            "{} text={:#x}, size={:#x}, local_init={:#x}, config={:#x}",
            tunnel_1x.mmio_core_logical,
            config.binary_addr,
            config.binary_size,
            local_init,
            config_addr);
        cluster.write_core(
            (void*)&config, sizeof(lite_fabric::LiteFabricConfig), tunnel_1x.mmio_cxy_virtual(), config_addr);
    }
    cluster.l1_barrier(0);

    tt::tt_metal::detail::LaunchProgram(devices[0], *pgm, false);

    for (auto tunnel_1x : desc.tunnels_from_mmio) {
        lite_fabric::wait_for_state(cluster, tunnel_1x.mmio_cxy_virtual(), lite_fabric::InitState::READY);
    }

    return pgm;
}

void TerminateLiteFabric(tt::Cluster& cluster, const SystemDescriptor& desc) {
    uint32_t routing_enabled_address =
        tt::tt_metal::MetalContext::instance().hal().get_dev_addr(
            tt::tt_metal::HalProgrammableCoreType::ACTIVE_ETH, tt::tt_metal::HalL1MemAddrType::FABRIC_LITE_CONFIG) +
        offsetof(lite_fabric::LiteFabricMemoryMap, config) + offsetof(lite_fabric::LiteFabricConfig, routing_enabled);
    uint32_t enabled = 0;
    for (const auto& tunnel_1x : desc.tunnels_from_mmio) {
        log_info(
            tt::LogMetal,
            "Host to terminate Device {} {} (virtual={})",
            0,
            tunnel_1x.mmio_core_logical,
            tunnel_1x.mmio_core_virtual);
        cluster.write_core((void*)&enabled, sizeof(uint32_t), tunnel_1x.mmio_cxy_virtual(), routing_enabled_address);
    }
    cluster.l1_barrier(0);
}

}  // namespace lite_fabric
