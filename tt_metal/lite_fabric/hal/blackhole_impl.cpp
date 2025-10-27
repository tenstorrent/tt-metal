// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "blackhole_impl.hpp"
#include "hw/inc/host_interface.hpp"
#include "tt_metal/lite_fabric/hw/inc/lf_dev_mem_map.hpp"
#include "tt_metal/impl/context/metal_context.hpp"

namespace {

uint32_t GetStateAddress() {
    return LITE_FABRIC_CONFIG_START + offsetof(lite_fabric::FabricLiteMemoryMap, config) +
           offsetof(lite_fabric::FabricLiteConfig, current_state);
}

uint32_t GetConfigAddress() { return LITE_FABRIC_CONFIG_START + offsetof(lite_fabric::FabricLiteMemoryMap, config); }

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

}  // namespace

namespace lite_fabric {

void BlackholeLiteFabricHal::set_reset_state(tt_cxy_pair virtual_core, bool assert_reset) {
    // We run on ERISC1. Don't touch ERISC0. It is running base firmware
    auto& cluster = tt::tt_metal::MetalContext::instance().get_cluster();
    if (assert_reset) {
        // Assert all cores except ERISC0.
        tt::umd::RiscType reset_val = tt::umd::RiscType::ALL_TENSIX & ~tt::umd::RiscType::ERISC0;
        cluster.assert_risc_reset_at_core(virtual_core, reset_val);
    } else {
        // Deassert only ERISC1.
        tt::umd::RiscType reset_val = tt::umd::RiscType::ERISC1;
        cluster.deassert_risc_reset_at_core(virtual_core, reset_val);
    }
}

void BlackholeLiteFabricHal::set_pc(tt_cxy_pair virtual_core, uint32_t pc_val) {
    auto& cluster = tt::tt_metal::MetalContext::instance().get_cluster();
    cluster.write_core(reinterpret_cast<void*>(&pc_val), sizeof(uint32_t), virtual_core, LITE_FABRIC_RESET_PC);
}

tt::umd::tt_version BlackholeLiteFabricHal::get_binary_version() {
    return tt::umd::tt_version{0, 0, 0};
}

void BlackholeLiteFabricHal::launch(const std::filesystem::path& bin_path) {
    constexpr uint32_t k_FirmwareStart = LITE_FABRIC_TEXT_START;

    auto& cluster = tt::tt_metal::MetalContext::instance().get_cluster();

    lite_fabric::FabricLiteConfig config = GetInitFabricLiteConfig(system_descriptor_);
    auto config_addr = GetConfigAddress();

    for (const auto& tunnel_1x : system_descriptor_.tunnels_from_mmio) {
        set_reset_state(tunnel_1x.mmio_cxy_virtual(), true);
        set_pc(tunnel_1x.mmio_cxy_virtual(), k_FirmwareStart);

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
    }

    cluster.l1_barrier(0);

    for (auto tunnel_1x : system_descriptor_.tunnels_from_mmio) {
        set_reset_state(tunnel_1x.mmio_cxy_virtual(), false);
    }

    // Wait for ready
    for (auto tunnel_1x : system_descriptor_.tunnels_from_mmio) {
        wait_for_state(tunnel_1x.mmio_cxy_virtual(), lite_fabric::InitState::READY);
        log_info(
            tt::LogMetal,
            "Lite Fabric {} (virtual={}) is ready",
            tunnel_1x.mmio_core_logical.str(),
            tunnel_1x.mmio_core_virtual.str());
    }
}

void BlackholeLiteFabricHal::terminate() {
    uint32_t routing_enabled_address = LITE_FABRIC_CONFIG_START + offsetof(lite_fabric::FabricLiteMemoryMap, config) +
                                       offsetof(lite_fabric::FabricLiteConfig, routing_enabled);
    uint32_t enabled = 0;
    auto& cluster = tt::tt_metal::MetalContext::instance().get_cluster();

    // On blackhole we run on a second erisc which can be put into reset
    for (const auto& tunnel_1x : system_descriptor_.tunnels_from_mmio) {
        log_info(
            tt::LogMetal,
            "Host to terminate lite fabric on device {} {} (virtual={})",
            0,
            tunnel_1x.mmio_core_logical,
            tunnel_1x.mmio_core_virtual);
        cluster.write_core((void*)&enabled, sizeof(uint32_t), tunnel_1x.mmio_cxy_virtual(), routing_enabled_address);
    }
    cluster.l1_barrier(0);

    LiteFabricHal::set_reset_state(true);
}

void BlackholeLiteFabricHal::wait_for_state(tt_cxy_pair virtual_core, lite_fabric::InitState state) {
    auto& cluster = tt::tt_metal::MetalContext::instance().get_cluster();
    std::vector<uint32_t> readback{static_cast<uint32_t>(lite_fabric::InitState::UNKNOWN)};
    while (static_cast<lite_fabric::InitState>(readback[0]) != state) {
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
        cluster.read_core(readback, sizeof(uint32_t), virtual_core, GetStateAddress());
    }
}

std::vector<std::filesystem::path> BlackholeLiteFabricHal::build_includes(const std::filesystem::path& root_dir) {
    return {
        root_dir,
        root_dir.parent_path(),
        root_dir / "tt_metal",
        root_dir / "tt_metal/include",
        root_dir / "tt_metal/hw/inc",
        root_dir / "tt_metal/hw/inc/ethernet",
        root_dir / "tt_metal/hostdevcommon/api",
        root_dir / "tt_metal/hw/inc/debug",
        root_dir / "tt_metal/hw/inc/tt-1xx/",
        root_dir / "tt_metal/hw/inc/tt-1xx/blackhole",
        root_dir / "tt_metal/hw/inc/tt-1xx/blackhole/blackhole_defines",
        root_dir / "tt_metal/hw/inc/tt-1xx/blackhole/noc",
        root_dir / "tt_metal/hw/ckernels/blackhole/metal/common",
        root_dir / "tt_metal/hw/ckernels/blackhole/metal/llk_io",
        root_dir / "tt_metal/third_party/tt_llk/tt_llk_blackhole/common/inc",
        root_dir / "tt_metal/api/",
        root_dir / "tt_metal/api/tt-metalium/",
        root_dir / "tt_metal/third_party/tt_llk/tt_llk_blackhole/llk_lib",
        root_dir / "tt_metal/lite_fabric/hw/inc"};  // For memory configuration headers
}

std::vector<std::string> BlackholeLiteFabricHal::build_defines() {
    return {
        "ARCH_BLACKHOLE",
        "TENSIX_FIRMWARE",
        "LOCAL_MEM_EN=0",
        "COMPILE_FOR_ERISC",  // This is needed to enable the ethernet APIs
        "ERISC",
        "RISC_B0_HW",
        "FW_BUILD",
        "NOC_INDEX=0",
        "DISPATCH_MESSAGE_ADDR=0",
        "COMPILE_FOR_LITE_FABRIC=1",
        "ROUTING_FW_ENABLED",
        // This is needed to get things to compile
        "NUM_DRAM_BANKS=1",
        "NUM_L1_BANKS=1",
        "LOG_BASE_2_OF_NUM_DRAM_BANKS=0",
        "LOG_BASE_2_OF_NUM_L1_BANKS=0",
        // We do not access the PCIe cores
        "PCIE_NOC_X=0",
        "PCIE_NOC_Y=0",
        // Lite Fabric is intended to run on risc1
        "PROCESSOR_INDEX=1",
    };
}

std::vector<std::filesystem::path> BlackholeLiteFabricHal::build_linker(const std::filesystem::path& root_dir) {
    return {
        root_dir / "runtime/hw/lib/blackhole/tmu-crt0.o",
        root_dir / "runtime/hw/lib/blackhole/substitutes.o",
    };
}

}  // namespace lite_fabric
