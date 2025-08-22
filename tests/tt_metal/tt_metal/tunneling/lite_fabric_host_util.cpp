// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "lite_fabric_host_util.hpp"
#include <enchantum/entries.hpp>
#include <tt-logger/tt-logger.hpp>
#include <fstream>
#include "build.hpp"
#include "lite_fabric.hpp"
#include "lite_fabric_memory_defs.h"

namespace {
uint32_t GetStateAddressMetal() {
    uint32_t state_addr =
        tt::tt_metal::MetalContext::instance().hal().get_dev_addr(
            tt::tt_metal::HalProgrammableCoreType::ACTIVE_ETH, tt::tt_metal::HalL1MemAddrType::FABRIC_LITE_CONFIG) +
        offsetof(lite_fabric::LiteFabricMemoryMap, config) + offsetof(lite_fabric::LiteFabricConfig, current_state);
    return state_addr;
}
uint32_t GetStateAddress() { return LITE_FABRIC_CONFIG_START + offsetof(lite_fabric::LiteFabricConfig, current_state); }

lite_fabric::LiteFabricConfig GetInitLiteFabricConfig(const lite_fabric::SystemDescriptor& desc) {
    lite_fabric::LiteFabricConfig config{};
    config.is_primary = true;
    config.is_mmio = true;
    config.initial_state = lite_fabric::InitState::ETH_INIT_NEIGHBOUR;
    config.current_state = lite_fabric::InitState::ETH_INIT_NEIGHBOUR;
    config.binary_addr = LITE_FABRIC_TEXT_START;
    config.binary_size = (LITE_FABRIC_TEXT_SIZE + 15) & ~0xF;  // Align to 16 bytes;
    config.eth_chans_mask = desc.enabled_eth_channels.at(0);
    config.routing_enabled = true;
    return config;
}

}  // namespace

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
    tt::Cluster& cluster, chip_id_t mmio_device_id, chip_id_t connected_device_id) {
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

void SetResetState(tt::Cluster& cluster, tt_cxy_pair virtual_core, bool assert_reset) {
    // We run on DM1. Don't touch DM0. It is running base firmware
    TensixSoftResetOptions reset_val = TENSIX_ASSERT_SOFT_RESET;
    if (assert_reset) {
        reset_val = reset_val & static_cast<TensixSoftResetOptions>(
                                    ~std::underlying_type<TensixSoftResetOptions>::type(TensixSoftResetOptions::BRISC));
        cluster.assert_risc_reset_at_core(virtual_core, reset_val);
    } else {
        reset_val = TENSIX_DEASSERT_SOFT_RESET &
                    static_cast<TensixSoftResetOptions>(
                        ~std::underlying_type<TensixSoftResetOptions>::type(TensixSoftResetOptions::TRISC0));
        cluster.deassert_risc_reset_at_core(virtual_core, reset_val);
    }
}

void SetResetState(tt::Cluster& cluster, const SystemDescriptor& desc, bool assert_reset) {
    for (auto tunnel_1x : desc.tunnels_from_mmio) {
        SetResetState(cluster, tunnel_1x.mmio_cxy_virtual(), assert_reset);
    }
}

void SetPC(tt::Cluster& cluster, tt_cxy_pair virtual_core, uint32_t pc_addr, uint32_t pc_val) {
    cluster.write_core((void*)&pc_val, sizeof(uint32_t), virtual_core, pc_addr);
}

void SetPC(tt::Cluster& cluster, const SystemDescriptor& desc, uint32_t pc_addr, uint32_t pc_val) {
    for (auto tunnel_1x : desc.tunnels_from_mmio) {
        SetPC(cluster, tunnel_1x.mmio_cxy_virtual(), pc_addr, pc_val);
    }
}

void WaitForState(tt::Cluster& cluster, tt_cxy_pair virtual_core, uint32_t addr, lite_fabric::InitState state) {
    std::vector<uint32_t> readback{static_cast<uint32_t>(lite_fabric::InitState::UNKNOWN)};
    while (static_cast<lite_fabric::InitState>(readback[0]) != state) {
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
        cluster.read_core(readback, sizeof(uint32_t), virtual_core, addr);
    }
}

void WaitForState(tt::Cluster& cluster, const SystemDescriptor& desc, uint32_t addr, lite_fabric::InitState state) {
    for (auto tunnel_1x : desc.tunnels_from_mmio) {
        WaitForState(cluster, tunnel_1x.mmio_cxy_virtual(), addr, state);
    }
}

void LaunchLiteFabric(
    tt::Cluster& cluster,
    const tt::tt_metal::Hal& hal,
    const SystemDescriptor& desc,
    const std::filesystem::path& bin_path) {
    constexpr uint32_t k_FirmwareStart = LITE_FABRIC_TEXT_START;
    constexpr uint32_t k_PcResetAddress = LITE_FABRIC_RESET_PC;

    lite_fabric::LiteFabricConfig config = GetInitLiteFabricConfig(desc);
    auto config_addr = lite_fabric::LiteFabricMemoryMap::get_address();

    for (const auto& tunnel_1x : desc.tunnels_from_mmio) {
        lite_fabric::SetResetState(cluster, tunnel_1x.mmio_cxy_virtual(), true);
        lite_fabric::SetPC(cluster, tunnel_1x.mmio_cxy_virtual(), k_PcResetAddress, k_FirmwareStart);

        std::ifstream bin_file(bin_path, std::ios::binary);
        if (!bin_file) {
            throw std::runtime_error(fmt::format("Failed to open binary file: {}", bin_path));
        }

        // Get file size
        bin_file.seekg(0, std::ios::end);
        size_t bin_size = bin_file.tellg();
        bin_file.seekg(0, std::ios::beg);

        // Read entire binary into memory
        std::vector<uint8_t> binary_data(bin_size);
        bin_file.read(reinterpret_cast<char*>(binary_data.data()), bin_size);
        bin_file.close();

        log_debug(tt::LogMetal, "Loaded flat binary {} size {} B", bin_path, bin_size);

        // Set up configuration
        cluster.write_core(
            (void*)&config, sizeof(lite_fabric::LiteFabricConfig), tunnel_1x.mmio_cxy_virtual(), config_addr);

        // Write entire binary as single operation to device memory
        log_debug(tt::LogMetal, "Writing flat binary to {:#x} size {} B", LITE_FABRIC_TEXT_START, bin_size);
        cluster.write_core(binary_data.data(), bin_size, tunnel_1x.mmio_cxy_virtual(), LITE_FABRIC_TEXT_START);

        log_debug(
            tt::LogMetal,
            "Wrote lite fabric. Core: {}, Config: {:#x}, Binary: {:#x}, Size: {} B",
            tunnel_1x.mmio_core_logical,
            config_addr,
            config.binary_addr,
            config.binary_size);
    }

    cluster.l1_barrier(0);

    for (auto tunnel_1x : desc.tunnels_from_mmio) {
        lite_fabric::SetResetState(cluster, tunnel_1x.mmio_cxy_virtual(), false);
    }

    cluster.l1_barrier(0);
    // Wait for ready
    for (auto tunnel_1x : desc.tunnels_from_mmio) {
        lite_fabric::WaitForState(
            cluster, tunnel_1x.mmio_cxy_virtual(), GetStateAddress(), lite_fabric::InitState::READY);
        log_info(
            tt::LogMetal,
            "Lite Fabric {} (virtual={}) is ready",
            tunnel_1x.mmio_core_logical.str(),
            tunnel_1x.mmio_core_virtual.str());
    }
}

void LaunchLiteFabric(tt::Cluster& cluster, const tt::tt_metal::Hal& hal, const SystemDescriptor& desc) {
    auto home_directory = std::filesystem::path(std::getenv("TT_METAL_HOME"));
    auto output_directory = home_directory / "lite_fabric";
    // Throw exception if any of these return non zero
    if (lite_fabric::CompileLiteFabric(cluster, home_directory, output_directory)) {
        throw std::runtime_error("Failed to compile lite fabric");
    }
    if (lite_fabric::LinkLiteFabric(home_directory, output_directory, output_directory / "lite_fabric.elf")) {
        throw std::runtime_error("Failed to link lite fabric");
    }

    std::filesystem::path bin_path{output_directory / "lite_fabric.bin"};

    lite_fabric::LaunchLiteFabric(cluster, hal, desc, bin_path);
}

template <typename HOST_INTERFACE>
void ResumeLiteFabric(
    tt::Cluster& cluster, const tt::tt_metal::Hal& hal, const SystemDescriptor& desc, HOST_INTERFACE& host_interface) {
    constexpr uint32_t k_FirmwareStart = LITE_FABRIC_TEXT_START;
    constexpr uint32_t k_PcResetAddress = LITE_FABRIC_RESET_PC;

    // NOTE: If we didn't want to reinit back to firmware start we could set the PC to the service routing, and update
    // the host interface pointers by reading from the device.
    lite_fabric::LiteFabricConfig config = GetInitLiteFabricConfig(desc);
    auto config_addr = lite_fabric::LiteFabricMemoryMap::get_address();

    host_interface.init();
    for (const auto& tunnel_1x : desc.tunnels_from_mmio) {
        lite_fabric::SetResetState(cluster, tunnel_1x.mmio_cxy_virtual(), true);
        lite_fabric::SetPC(cluster, tunnel_1x.mmio_cxy_virtual(), k_PcResetAddress, k_FirmwareStart);
        cluster.write_core(
            (void*)&config, sizeof(lite_fabric::LiteFabricConfig), tunnel_1x.mmio_cxy_virtual(), config_addr);
    }

    for (auto tunnel_1x : desc.tunnels_from_mmio) {
        lite_fabric::SetResetState(cluster, tunnel_1x.mmio_cxy_virtual(), false);
    }

    cluster.l1_barrier(0);
    // Wait for ready
    for (auto tunnel_1x : desc.tunnels_from_mmio) {
        lite_fabric::WaitForState(
            cluster, tunnel_1x.mmio_cxy_virtual(), GetStateAddress(), lite_fabric::InitState::READY);
        log_info(
            tt::LogMetal,
            "Lite Fabric {} (virtual={}) is ready",
            tunnel_1x.mmio_core_logical.str(),
            tunnel_1x.mmio_core_virtual.str());
    }
}

void TerminateLiteFabric(tt::Cluster& cluster, const SystemDescriptor& desc) {
    uint32_t routing_enabled_address = LITE_FABRIC_CONFIG_START + offsetof(lite_fabric::LiteFabricMemoryMap, config) +
                                       offsetof(lite_fabric::LiteFabricConfig, routing_enabled);
    uint32_t enabled = 0;
    for (const auto& tunnel_1x : desc.tunnels_from_mmio) {
        log_debug(
            tt::LogMetal,
            "Host to terminate Device {} {} (virtual={})",
            0,
            tunnel_1x.mmio_core_logical,
            tunnel_1x.mmio_core_virtual);
        cluster.write_core((void*)&enabled, sizeof(uint32_t), tunnel_1x.mmio_cxy_virtual(), routing_enabled_address);
    }
    cluster.l1_barrier(0);
    SetResetState(cluster, desc, true);
}

}  // namespace lite_fabric
