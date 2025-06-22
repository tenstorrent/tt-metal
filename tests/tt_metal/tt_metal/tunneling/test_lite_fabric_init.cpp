// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <chrono>
#include <gtest/gtest.h>
#include <stdint.h>
#include <tt-metalium/control_plane.hpp>
#include <tt-metalium/device_pool.hpp>
#include <tt-metalium/erisc_datamover_builder.hpp>
#include <tt-metalium/fabric_host_interface.h>
#include <array>
#include <cstddef>
#include <map>
#include <optional>
#include <utility>
#include <variant>
#include <vector>
#include <random>

#include <tt-metalium/allocator.hpp>
#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/data_types.hpp>
#include <tt-metalium/device.hpp>
#include "device_fixture.hpp"
#include "utils.hpp"
#include "test_lite_fabric_utils.hpp"
#include <tt-metalium/hal.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/kernel_types.hpp>
#include <tt-metalium/mesh_coord.hpp>
#include <tt-metalium/mesh_graph.hpp>
#include <tt_stl/span.hpp>
#include <tt-metalium/tt_metal.hpp>

// Tests in this file are to verify the initialization / handshake sequence for setting up Lite Fabric kernel
// on remote chips. The tests don't require remote chips, if all chips have PCIe, remote chips are spoofed.

namespace tt::tt_metal {
namespace tunneling {

TEST_F(DeviceFixture, MmioEthCoreInitSingleEthCore) {
    if (arch_ == tt::ARCH::WORMHOLE_B0) {
        GTEST_SKIP() << "Skipping test for Wormhole B0, as it does not support tunneling yet";
    }
    // if (devices_.size() != 2) {
    //     GTEST_SKIP() << "Only expect to be initializing 1 eth device per MMIO chip. Test should ";
    // }

    MmmioAndEthDeviceDesc desc;
    get_mmio_device_and_eth_device_to_init(devices_, desc);

    auto mmio_program = create_eth_init_program(desc, false);

    auto virtual_eth_core = desc.mmio_device->ethernet_core_from_logical_core(desc.mmio_eth.value());
    std::cout << "virtual_eth_core " << virtual_eth_core.str() << std::endl;

    std::cout << "Number of active ethernet cores on mmio device: "
              << desc.mmio_device->get_active_ethernet_cores().size() << std::endl;

    tt::tt_metal::MetalContext::instance().get_cluster().l1_barrier(
        desc.mmio_device->id());  // don't need launch program should do
    tt_metal::detail::LaunchProgram(desc.mmio_device, mmio_program);
}

TEST_F(DeviceFixture, MmioEthCoreInitAllEthCores) {
    if (arch_ == tt::ARCH::WORMHOLE_B0) {
        GTEST_SKIP() << "Skipping test for Wormhole B0, as it does not support tunneling yet";
    }
    if (devices_.size() != 2) {
        GTEST_SKIP() << "Only expect to be initializing 1 eth device per MMIO chip. Test should ";
    }

    MmmioAndEthDeviceDesc desc;
    get_mmio_device_and_eth_device_to_init(devices_, desc);

    auto mmio_program = create_eth_init_program(desc, true);

    auto virtual_eth_core = desc.mmio_device->ethernet_core_from_logical_core(desc.mmio_eth.value());
    std::cout << "virtual_eth_core " << virtual_eth_core.str() << std::endl;

    std::cout << "Number of active ethernet cores on mmio device: "
              << desc.mmio_device->get_active_ethernet_cores().size() << std::endl;

    tt::tt_metal::MetalContext::instance().get_cluster().l1_barrier(
        desc.mmio_device->id());  // don't need launch program should do
    tt_metal::detail::LaunchProgram(desc.mmio_device, mmio_program);
}

}  // namespace tunneling
}  // namespace tt::tt_metal
