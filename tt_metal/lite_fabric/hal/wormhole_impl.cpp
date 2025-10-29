// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "wormhole_impl.hpp"
#include <cstddef>
#include "hw/inc/wormhole/lf_dev_mem_map.hpp"
#include "tt_metal/impl/context/metal_context.hpp"

namespace {

constexpr uint32_t GetStateAddress() {
    return LITE_FABRIC_CONFIG_START + offsetof(lite_fabric::FabricLiteMemoryMap, config) +
           offsetof(lite_fabric::FabricLiteConfig, current_state);
}

constexpr uint32_t GetConfigAddress() {
    return LITE_FABRIC_CONFIG_START + offsetof(lite_fabric::FabricLiteMemoryMap, config);
}

lite_fabric::FabricLiteConfig GetInitFabricLiteConfig(const lite_fabric::SystemDescriptor& desc) {
    lite_fabric::FabricLiteConfig config{};
    config.is_primary = true;
    config.is_mmio = true;
    config.initial_state = lite_fabric::InitState::ETH_INIT_NEIGHBOUR;
    config.current_state = lite_fabric::InitState::ETH_INIT_NEIGHBOUR;
    config.binary_addr = LITE_FABRIC_TEXT_START;
    config.binary_size = (LITE_FABRIC_TEXT_SIZE + 15) & ~0xF;  // Align to 16 bytes;
    config.eth_chans_mask = desc.enabled_eth_channels.at(0);
    config.routing_enabled = lite_fabric::RoutingEnabledState::ENABLED;
    return config;
}

void base_firmware_lite_fabric_enable(tt_cxy_pair virtual_core, bool enable) {
    constexpr uint32_t lite_fabric_routing_enable = 0x18000;
    const uint32_t routing_enable = enable ? 1 : 0;
    auto& cluster = tt::tt_metal::MetalContext::instance().get_cluster();
    cluster.write_core((void*)&routing_enable, sizeof(uint32_t), virtual_core, lite_fabric_routing_enable);
}

}  // namespace

namespace lite_fabric {

void WormholeLiteFabricHal::set_reset_state(tt_cxy_pair virtual_core, bool assert_reset) {}

void WormholeLiteFabricHal::set_pc(tt_cxy_pair virtual_core, uint32_t pc_val) {}

tt::umd::tt_version WormholeLiteFabricHal::get_binary_version() {
    return tt::umd::tt_version{0, 0, 0};
}

void WormholeLiteFabricHal::launch(const std::filesystem::path& bin_path) {
    auto& cluster = tt::tt_metal::MetalContext::instance().get_cluster();

    lite_fabric::FabricLiteConfig config = GetInitFabricLiteConfig(system_descriptor_);
    auto config_addr = GetConfigAddress();

    for (const auto& tunnel_1x : system_descriptor_.tunnels_from_mmio) {
        // Write config and binary to ethernet core
        std::ifstream bin_file(bin_path, std::ios::binary);
        if (!bin_file) {
            throw std::runtime_error(fmt::format("Failed to open binary file: {}", bin_path));
        }

        bin_file.seekg(0, std::ios::end);
        size_t bin_size = bin_file.tellg();
        bin_file.seekg(0, std::ios::beg);

        std::vector<uint8_t> binary_data(bin_size);
        bin_file.read(reinterpret_cast<char*>(binary_data.data()), bin_size);
        bin_file.close();

        log_debug(tt::LogMetal, "Loaded flat binary {} size {} B", bin_path, bin_size);

        // Set up configuration
        cluster.write_core(
            (void*)&config, sizeof(lite_fabric::FabricLiteConfig), tunnel_1x.mmio_cxy_virtual(), config_addr);

        // Write entire binary as single operation to device memory
        log_debug(tt::LogMetal, "Writing flat binary to {:#x} size {} B", LITE_FABRIC_TEXT_START, bin_size);
        cluster.write_core(binary_data.data(), bin_size, tunnel_1x.mmio_cxy_virtual(), LITE_FABRIC_TEXT_START);

        log_info(
            tt::LogMetal,
            "Wrote lite fabric. Core: {}, Config: {:#x}, Binary: {:#x}, Size: {} B. Initial config state {}",
            tunnel_1x.mmio_core_logical,
            config_addr,
            static_cast<uint32_t>(config.binary_addr),
            static_cast<uint32_t>(config.binary_size),
            static_cast<uint32_t>(config.initial_state));

        wait_for_state(tunnel_1x.mmio_cxy_virtual(), lite_fabric::InitState::ETH_INIT_NEIGHBOUR);

        // Get base firmware to execute the lite fabric binary by setting the lite fabric routing bit
        base_firmware_lite_fabric_enable(tunnel_1x.mmio_cxy_virtual(), true);

        // Wait for Ready state
        wait_for_state(tunnel_1x.mmio_cxy_virtual(), lite_fabric::InitState::READY);
    }
}

void WormholeLiteFabricHal::terminate() {}

void WormholeLiteFabricHal::wait_for_state(tt_cxy_pair virtual_core, lite_fabric::InitState state) {
    auto& cluster = tt::tt_metal::MetalContext::instance().get_cluster();
    std::vector<uint32_t> readback{static_cast<uint32_t>(lite_fabric::InitState::UNKNOWN)};
    while (static_cast<lite_fabric::InitState>(readback[0]) != state) {
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
        log_info(tt::LogMetal, "{}", static_cast<lite_fabric::InitState>(readback[0]));
        cluster.read_core(readback, sizeof(uint32_t), virtual_core, GetStateAddress());
    }
    log_info(tt::LogMetal, "{}", static_cast<lite_fabric::InitState>(readback[0]));
}

std::vector<std::filesystem::path> WormholeLiteFabricHal::build_srcs(const std::filesystem::path& root_dir) {
    return {};
}

std::vector<std::filesystem::path> WormholeLiteFabricHal::build_includes(const std::filesystem::path& root_dir) {
    return {
        root_dir / "tt_metal/hw/inc/tt-1xx/wormhole",
        root_dir / "tt_metal/hw/inc/tt-1xx/wormhole/wormhole_b0_defines",
        root_dir / "tt_metal/hw/inc/tt-1xx/wormhole/noc",
        root_dir / "tt_metal/hw/ckernels/wormhole/metal/common",
        root_dir / "tt_metal/hw/ckernels/wormhole/metal/llk_io",
        root_dir / "tt_metal/third_party/tt_llk/tt_llk_wormhole_b0/common/inc",
        root_dir / "tt_metal/third_party/tt_llk/tt_llk_wormhole_b0/llk_lib",
        root_dir / "tt_metal/lite_fabric/hw/inc",
        root_dir / "tt_metal/lite_fabric/hw/inc/wormhole",
    };
}

std::vector<std::string> WormholeLiteFabricHal::build_defines() {
    return {
        "ARCH_WORMHOLE",
        "TENSIX_FIRMWARE",
        "LOCAL_MEM_EN=0",
        "COMPILE_FOR_ERISC",
        "ERISC",
        "RISC_B0_HW",
        "FW_BUILD",
        "NOC_INDEX=0",
        "DISPATCH_MESSAGE_ADDR=0",
        "COMPILE_FOR_LITE_FABRIC=1",
        "ROUTING_FW_ENABLED",
        "NUM_DRAM_BANKS=1",
        "NUM_L1_BANKS=1",
        "LOG_BASE_2_OF_NUM_DRAM_BANKS=0",
        "LOG_BASE_2_OF_NUM_L1_BANKS=0",
        "PCIE_NOC_X=0",
        "PCIE_NOC_Y=0",
        "PROCESSOR_INDEX=0",
    };
}

std::vector<std::filesystem::path> WormholeLiteFabricHal::build_linker(const std::filesystem::path& root_dir) {
    return {
        root_dir / "runtime/hw/lib/blackhole/substitutes.o",
    };
}

std::optional<std::filesystem::path> WormholeLiteFabricHal::build_asm_startup(const std::filesystem::path& root_dir) {
    return root_dir / "tt_metal/lite_fabric/toolchain/lite_fabric-crt0-bare.S";
}

}  // namespace lite_fabric
