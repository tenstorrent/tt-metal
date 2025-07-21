// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>
#include <umd/device/types/arch.h>

#include <memory>
#include <tt-metalium/device.hpp>
#include <tt-metalium/device_pool.hpp>
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/erisc_datamover_builder.hpp>
#include "hal_types.hpp"
#include "tests/tt_metal/test_utils/env_vars.hpp"
#include "host_api.hpp"
#include "impl/context/metal_context.hpp"
#include "impl/lite_fabric.hpp"
#include "impl/kernels/kernel_impl.hpp"
#include "tt_align.hpp"
#include "tt_metal/jit_build/build_env_manager.hpp"

class LiteFabricTestFixture : public ::testing::Test {
private:
    inline static tt::ARCH arch_ = tt::ARCH::Invalid;
    inline static int numberOfDevices_ = -1;

protected:
    std::map<int, std::shared_ptr<tt::tt_metal::distributed::MeshDevice>> reserved_devices_;

    static void SetUpTestSuite() {
        arch_ = tt::get_arch_from_string(tt::test_utils::get_umd_arch_name());
        numberOfDevices_ = tt::tt_metal::GetNumAvailableDevices();
    }

    void SetUp() override {
        if (arch_ != tt::ARCH::BLACKHOLE) {
            GTEST_SKIP() << "Only supported on Blackhole";
        }
        if (numberOfDevices_ != 2) {
            GTEST_SKIP() << "Need 2 devices to run this test";
        }
        if (std::getenv("TT_METAL_SLOW_DISPATCH_MODE") == nullptr) {
            GTEST_SKIP() << "Need to set TT_METAL_SLOW_DISPATCH_MODE to run this test";
        }
        if (std::getenv("TT_METAL_CLEAR_L1") == nullptr) {
            GTEST_SKIP() << "Need to set TT_METAL_CLEAR_L1 to run this test";
        }
        std::vector<int> devices_to_open(numberOfDevices_);
        std::iota(devices_to_open.begin(), devices_to_open.end(), 0);
        reserved_devices_ = tt::tt_metal::distributed::MeshDevice::create_unit_meshes(devices_to_open);
    }

    void TearDown() override {
    }
};

TEST_F(LiteFabricTestFixture, TestLiteFabricInit) {
    auto& hal = tt::tt_metal::MetalContext::instance().hal();
    auto& control_plane = tt::tt_metal::MetalContext::instance().get_control_plane();
    const auto source_node_id = control_plane.get_fabric_node_id_from_physical_chip_id(0);
    const auto dest_node_id = control_plane.get_fabric_node_id_from_physical_chip_id(1);
    log_info(tt::LogTest, "Source node id: {}, Dest node id: {}", source_node_id, dest_node_id);

    auto mmio_device = reserved_devices_[0]->get_devices()[0];
    auto eth_device = reserved_devices_[1]->get_devices()[0];

    // Create program that will run on the MMIO device (the local primary core)
    // This will clone itself to other active ethernet cores on the same device.
    // It will also clone itself to one other ethernet core on the ethernet device, which will do the
    // same to the ethernet cores on the ethernet device.
    uint32_t expected_eth_chan_count = 0;
    CoreCoord primary_mmio_logical_eth_core{0,0};
    for (const auto& core : mmio_device->get_active_ethernet_cores()) {
        if (!tt::tt_metal::MetalContext::instance().get_cluster().is_ethernet_link_up(mmio_device->id(), core)) {
            continue;
        }
        expected_eth_chan_count++;
        primary_mmio_logical_eth_core = core;
        const auto [connected_chip_id, connected_chip_eth_core] = tt::tt_metal::MetalContext::instance().get_cluster().get_connected_ethernet_core({mmio_device->id(), core});
        log_info(tt::LogTest, "Chip {} Active Ethernet core {} connects to chip {} core {}", mmio_device->id(), core.str(), connected_chip_id, connected_chip_eth_core.str());
    }

    static constexpr std::size_t edm_buffer_size =
    tt::tt_fabric::FabricEriscDatamoverBuilder::default_packet_payload_size_bytes + sizeof(tt::tt_fabric::PacketHeader);
    constexpr uint32_t risc_id = 0;

    // Compile program
    auto pgm = std::make_unique<tt::tt_metal::Program>();
    tt::tt_fabric::FabricEriscDatamoverBuilder edm_builder = tt::tt_fabric::FabricEriscDatamoverBuilder::build(
        mmio_device,
        *pgm,
        primary_mmio_logical_eth_core,
        source_node_id,
        dest_node_id,
        tt::tt_fabric::FabricEriscDatamoverConfig(edm_buffer_size),
        false, /* build_in_worker_connection_mode */
        false /* is_dateline */);
        
    const auto ct_args = edm_builder.get_compile_time_args(risc_id);
    auto kernel_handle = tt::tt_metal::CreateKernel(
        *pgm,
        "tt_metal/fabric/impl/kernels/edm_fabric/fabric_erisc_datamover.cpp",
        primary_mmio_logical_eth_core,
        tt::tt_metal::EthernetConfig{
            .noc = edm_builder.config.risc_configs[risc_id].get_configured_noc(),
            .processor = static_cast<tt::tt_metal::DataMovementProcessor>(risc_id),
            .compile_args = ct_args,
            .defines = {},
            .opt_level = tt::tt_metal::KernelBuildOptLevel::O3});
    tt::tt_metal::detail::CompileProgram(mmio_device, *pgm);

    // Obtain misc metadata needed to generate the launch and go message because we are running on
    // top of metal firwmare
    const auto config_addr = hal.get_dev_addr(tt::tt_metal::HalProgrammableCoreType::ACTIVE_ETH, tt::tt_metal::HalL1MemAddrType::ETH_LITE_FABRIC_CONFIG);
    const auto config_size = hal.get_dev_size(tt::tt_metal::HalProgrammableCoreType::ACTIVE_ETH, tt::tt_metal::HalL1MemAddrType::ETH_LITE_FABRIC_CONFIG);
    ASSERT_GE(config_size, sizeof(LiteFabricConfig));

    const auto& kernel =
        pgm->get_kernels(static_cast<uint32_t>(tt::tt_metal::HalProgrammableCoreType::ACTIVE_ETH)).at(kernel_handle);
    const ll_api::memory& kernel_binary = *tt::tt_metal::KernelImpl::from(*kernel)
        .binaries(tt::tt_metal::BuildEnvManager::get_instance().get_device_build_env(mmio_device->build_id()).build_key)[0];

    uint32_t dst_binary_address;
    uint32_t binary_size_bytes;
    EXPECT_EQ(kernel_binary.num_spans(), 1) << "Need a contiguous binary span for the lite fabric kernel";
    kernel_binary.process_spans([&](std::vector<uint32_t>::const_iterator mem_ptr, uint64_t addr, uint32_t len_words) {
        const auto erisc_core_type = hal.get_programmable_core_type_index(
            tt::tt_metal::HalProgrammableCoreType::ACTIVE_ETH);
        constexpr auto processor_class_idx = magic_enum::enum_integer(tt::tt_metal::HalProcessorClassType::DM);
        const auto alignment = hal.get_alignment(tt::tt_metal::HalMemType::L1);
        const auto processor_type_idx =
        magic_enum::enum_integer(std::get<tt::tt_metal::EthernetConfig>(kernel->config()).processor);
        const auto local_init_addr = hal.get_jit_build_config(erisc_core_type, processor_class_idx, processor_type_idx).local_init_addr;
        const auto relo_addr = hal.relocate_dev_addr(addr, local_init_addr);
        dst_binary_address = relo_addr;
        binary_size_bytes = tt::align(len_words * sizeof(uint32_t), alignment);
    });
    log_info(tt::LogTest, "Kernel binary address: {}, size: {}", dst_binary_address, binary_size_bytes);

    // Write lite fabric config to the primary ethernet core
    LiteFabricConfig config;
}