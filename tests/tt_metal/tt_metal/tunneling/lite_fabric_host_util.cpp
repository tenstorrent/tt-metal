// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "lite_fabric_host_util.hpp"
#include <tt-logger/tt-logger.hpp>
#include "blackhole/dev_mem_map.h"
#include "build.hpp"
#include "lite_fabric.hpp"
#include "tt_memory.h"

namespace {
uint32_t GetStateAddressMetal() {
    uint32_t state_addr =
        tt::tt_metal::MetalContext::instance().hal().get_dev_addr(
            tt::tt_metal::HalProgrammableCoreType::ACTIVE_ETH, tt::tt_metal::HalL1MemAddrType::FABRIC_LITE_CONFIG) +
        offsetof(lite_fabric::LiteFabricMemoryMap, config) + offsetof(lite_fabric::LiteFabricConfig, current_state);
    return state_addr;
}
uint32_t GetStateAddress() {
    return MEM_LITE_FABRIC_CONFIG_BASE + offsetof(lite_fabric::LiteFabricConfig, current_state);
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

SystemDescriptor GetSystemDescriptor2Devices(chip_id_t mmio_device_id, chip_id_t connected_device_id) {
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

    return {std::move(bin), local_init};
}

std::pair<const ll_api::memory&, uint32_t> GetBinaryMetadata(std::string path, const tt::tt_metal::Hal& hal) {
    const ll_api::memory& bin = ll_api::memory(path, ll_api::memory::Loading::DISCRETE);

    auto local_init = hal.get_dev_addr(
        tt::tt_metal::HalProgrammableCoreType::ACTIVE_ETH, tt::tt_metal::HalL1MemAddrType::LOCAL_L1_INIT_SCRATCH);

    return {std::move(bin), local_init};
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
    config.routing_enabled = true;

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
    auto config_addr = lite_fabric::LiteFabricMemoryMap::get_address();
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
            "{} text={:#x}, size={:#x}, local_init={:#x}, config={:#x}, routing_enabled={}",
            tunnel_1x.mmio_core_logical,
            config.binary_addr,
            config.binary_size,
            local_init,
            config_addr,
            config.routing_enabled);
        cluster.write_core(
            (void*)&config, sizeof(lite_fabric::LiteFabricConfig), tunnel_1x.mmio_cxy_virtual(), config_addr);
    }
    cluster.l1_barrier(0);

    tt::tt_metal::detail::LaunchProgram(devices[0], *pgm, false);

    for (auto tunnel_1x : desc.tunnels_from_mmio) {
        lite_fabric::WaitForState(
            cluster, tunnel_1x.mmio_cxy_virtual(), GetStateAddressMetal(), lite_fabric::InitState::READY);
        log_info(
            tt::LogMetal,
            "Lite Fabric {} (virtual={}) is ready",
            tunnel_1x.mmio_core_logical.str(),
            tunnel_1x.mmio_core_virtual.str());
    }

    return pgm;
}

void LaunchLiteFabric(
    tt::Cluster& cluster,
    const tt::tt_metal::Hal& hal,
    const SystemDescriptor& desc,
    const std::filesystem::path& elf_path) {
    constexpr uint32_t k_FirmwareStart = MEM_LITE_FABRIC_FIRMWARE_BASE;
    constexpr uint32_t k_PcResetAddress = MEM_LITE_FABRIC_RESET_PC;

    lite_fabric::LiteFabricConfig config{};
    config.is_primary = true;
    config.is_mmio = true;
    config.initial_state = lite_fabric::InitState::ETH_INIT_NEIGHBOUR;
    config.current_state = lite_fabric::InitState::ETH_INIT_NEIGHBOUR;
    config.binary_addr = 0;
    config.binary_size = 0;
    config.eth_chans_mask = desc.enabled_eth_channels.at(0);
    config.routing_enabled = true;

    // Need an abstraction layer for Lite Fabric
    auto config_addr = MEM_LITE_FABRIC_CONFIG_BASE;

    for (const auto& tunnel_1x : desc.tunnels_from_mmio) {
        lite_fabric::SetResetState(cluster, tunnel_1x.mmio_cxy_virtual(), true);
        lite_fabric::SetPC(cluster, tunnel_1x.mmio_cxy_virtual(), k_PcResetAddress, k_FirmwareStart);

        const ll_api::memory& bin = ll_api::memory(elf_path.string(), ll_api::memory::Loading::DISCRETE);

        auto local_init = MEM_LITE_FABRIC_INIT_LOCAL_L1_BASE_SCRATCH;

        bin.process_spans([&](std::vector<uint32_t>::const_iterator mem_ptr, uint64_t addr, uint32_t len_words) {
            // Move data from private memory into L1 to be copied into private memory during kernel init
            uint32_t relo_addr = hal.relocate_dev_addr(addr, local_init);
            if (relo_addr != addr) {
                // Local memory relocated to L1 for copying in kernel init
                cluster.write_core(&*mem_ptr, len_words * sizeof(uint32_t), tunnel_1x.mmio_cxy_virtual(), relo_addr);
                log_info(
                    tt::LogMetal,
                    "Writing local memory to {:#x} -> reloc {:#x} size {} B",
                    addr,
                    relo_addr,
                    len_words * sizeof(uint32_t));
            } else {
                config.binary_addr = relo_addr;
                config.binary_size = len_words * sizeof(uint32_t);
                config.binary_size = (config.binary_size + 15) & ~0xF;

                cluster.write_core(
                    (void*)&config, sizeof(lite_fabric::LiteFabricConfig), tunnel_1x.mmio_cxy_virtual(), config_addr);

                log_info(
                    tt::LogMetal,
                    "Writing binary to {:#x} -> reloc {:#x} size {} B",
                    addr,
                    relo_addr,
                    len_words * sizeof(uint32_t));
                cluster.write_core(&*mem_ptr, len_words * sizeof(uint32_t), tunnel_1x.mmio_cxy_virtual(), relo_addr);
            }
        });

        log_info(
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
            "Lite Fabric {} {} (virtual={}) is ready",
            tunnel_1x.mmio_core_logical,
            tunnel_1x.mmio_core_virtual.y,
            tunnel_1x.mmio_core_virtual.x);
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

    std::filesystem::path elf_path{output_directory / "lite_fabric.elf"};

    lite_fabric::LaunchLiteFabric(cluster, hal, desc, elf_path);
}

void TerminateLiteFabricWithMetal(tt::Cluster& cluster, const SystemDescriptor& desc) {
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

void TerminateLiteFabric(tt::Cluster& cluster, const SystemDescriptor& desc) {
    uint32_t routing_enabled_address = MEM_LITE_FABRIC_CONFIG_BASE +
                                       offsetof(lite_fabric::LiteFabricMemoryMap, config) +
                                       offsetof(lite_fabric::LiteFabricConfig, routing_enabled);
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
