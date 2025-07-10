// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <chrono>
#include <stdint.h>
#include <tt-metalium/control_plane.hpp>
#include <tt-metalium/device_pool.hpp>
#include <tt-metalium/erisc_datamover_builder.hpp>
#include <tt-metalium/fabric_host_interface.h>
#include <tt-metalium/fabric_edm_packet_header.hpp>
#include <array>
#include <cstddef>

#include "device_fixture.hpp"
#include "utils.hpp"
#include "test_lite_fabric_utils.hpp"
#include <tt-metalium/kernel_types.hpp>
#include <tt-metalium/mesh_coord.hpp>
#include <tt-metalium/mesh_graph.hpp>
#include <tt_stl/span.hpp>

namespace tt::tt_metal {
namespace tunneling {

TEST_F(DeviceFixture, MmioEthCoreRunLiteFabricWritesSingleEthCore) {
    if (arch_ == tt::ARCH::WORMHOLE_B0) {
        GTEST_SKIP() << "Skipping test for Wormhole B0, as it does not support tunneling yet";
    }
    if (devices_.size() != 2) {
        GTEST_SKIP() << "Only expect to be initializing 1 eth device per MMIO chip. Test should ";
    }

    ::tunneling::MmmioAndEthDeviceDesc desc;
    get_mmio_device_and_eth_device_to_init(devices_, desc);

    auto mmio_program = create_eth_init_program(
        desc, ::tunneling::TestConfig{.init_all_eth_cores = false, .init_handshake_only = false});

    auto virtual_eth_core = desc.mmio_device->ethernet_core_from_logical_core(desc.mmio_eth.value());
    std::cout << "virtual_eth_core " << virtual_eth_core.str() << std::endl;

    std::cout << "Number of active ethernet cores on mmio device: "
              << desc.mmio_device->get_active_ethernet_cores().size() << std::endl;

    tt::tt_metal::MetalContext::instance().get_cluster().l1_barrier(
        desc.mmio_device->id());  // don't need launch program should do

    tt_metal::detail::LaunchProgram(desc.mmio_device, mmio_program, /*don't wait until this finishes*/ false);

    // in the meantime send some fabric writes to neighbouring eth core tensix
    // for each send check if there is space in the txq buffer and then push data

    // send the termination signal to the eth core
    uint32_t lite_fabric_config_addr = tt::tt_metal::MetalContext::instance().hal().get_dev_addr(
        tt::tt_metal::HalProgrammableCoreType::ACTIVE_ETH, tt::tt_metal::HalL1MemAddrType::UNRESERVED);
    uint32_t termination_signal_addr =
        lite_fabric_config_addr + offsetof(::tunneling::lite_fabric_config_t, termination_signal);
    uint32_t termination_val = 1;
    tt::tt_metal::MetalContext::instance().get_cluster().write_core(
        &termination_val,
        sizeof(uint32_t),
        tt_cxy_pair(desc.mmio_device->id(), virtual_eth_core),
        termination_signal_addr);
    tt::tt_metal::MetalContext::instance().get_cluster().l1_barrier(desc.mmio_device->id());

    tt_metal::detail::WaitProgramDone(desc.mmio_device, mmio_program);

    // uint32_t free_slots_addr = 0xFFB40000 + (17 * 0x1000) + (297 << 2);
    // uint32_t free_slots_val;
    // tt::tt_metal::MetalContext::instance().get_cluster().read_core(
    //     &free_slots_val, sizeof(uint32_t), tt_cxy_pair(desc.mmio_device->id(), virtual_eth_core), free_slots_addr);
    // std::cout << "Freee slots value: " << free_slots_val
    //            << " addr " << free_slots_addr << std::endl;
}

// mmio eth core ->

}  // namespace tunneling
}  // namespace tt::tt_metal
