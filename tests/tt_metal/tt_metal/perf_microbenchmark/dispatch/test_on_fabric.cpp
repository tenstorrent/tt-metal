// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <magic_enum/magic_enum.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/tt_metal.hpp>
#include <tt-metalium/device_pool.hpp>
#include <tt-metalium/logger.hpp>
#include <vector>
#include "command_queue_interface.hpp"
#include "core_coord.hpp"
#include "hostdevcommon/common_values.hpp"
#include "tt_cluster.hpp"

auto get_all_device_ids() {
    auto& cluster = tt::Cluster::instance();
    tt::log_info("Cluster type {}", magic_enum::enum_name(cluster.get_cluster_type()));
    std::vector<int> device_ids;
    for (int i = 0; i < cluster.number_of_devices(); ++i) {
        device_ids.push_back(i);
    }
    return device_ids;
}

std::pair<uint32_t, uint32_t> get_fabric_router_core_and_channel_xy(IDevice* src, IDevice* dst) {
    // Not implemented yet. Just give any core with Fabric on it -- which is any active eth
    auto soc_desc = tt::Cluster::instance().get_soc_desc(src->id());
    CoreCoord phy_coord;
    uint32_t channel;
    for (const auto& router_logical_core : src->get_active_ethernet_cores(true)) {
        phy_coord = src->virtual_core_from_logical_core(router_logical_core, CoreType::ETH);
        channel = soc_desc.logical_eth_core_to_chan_map.at(router_logical_core);
        tt::log_info(
            tt::LogTest,
            "Device {} Fabric Router Phy = {} Channel = {}",
            src->id(),
            router_logical_core.str(),
            channel);
        break;
    }
    return {tt::tt_metal::hal.noc_xy_encoding(phy_coord.x, phy_coord.y), channel};
}

int main(int argc, char** argv) {
    // Force some options
    setenv("TT_METAL_SLOW_DISPATCH_MODE", "true", 1);
    setenv("TT_METAL_CLEAR_L1", "1", 1);

    // 1 - Configure Fabric to initialize
    tt::tt_metal::detail::InitializeFabricSetting(tt::tt_metal::detail::FabricSetting::FABRIC);

    auto idle_eth_size =
        HalSingleton::getInstance().get_dev_size(HalProgrammableCoreType::IDLE_ETH, HalL1MemAddrType::UNRESERVED);
    auto tensix_size =
        HalSingleton::getInstance().get_dev_size(HalProgrammableCoreType::TENSIX, HalL1MemAddrType::UNRESERVED);
    std::cout << "ERISC L1 Unreserved = " << idle_eth_size << "\n";
    std::cout << "TENSIX L1 Unreserved = " << tensix_size << "\n";

    // 2 - Create Devices
    // All devices need to be enabled for Fabric
    auto devices =
        tt::tt_metal::detail::CreateDevices(get_all_device_ids(), 1, DEFAULT_L1_SMALL_SIZE, DEFAULT_TRACE_REGION_SIZE);

    auto control_plane = tt::DevicePool::instance().get_control_plane();

    // 3 - Make a kernel on the MMIO and remote device talk to each other
    static const std::string k_DummyKernelSrc =
        "tests/tt_metal/tt_metal/perf_microbenchmark/dispatch/kernels/dummy_kernel.cpp";

    CoreCoord core{0, 0};
    auto pgm_0 = tt::tt_metal::CreateProgram();
    auto pgm_1 = tt::tt_metal::CreateProgram();
    // Mesh info for the destination device
    auto [mesh_id_0, logical_dev_id_0] = control_plane->get_mesh_chip_id_from_physical_chip_id(devices[0]->id());
    auto [mesh_id_1, logical_dev_id_1] = control_plane->get_mesh_chip_id_from_physical_chip_id(devices[1]->id());
    tt::log_info(tt::LogTest, "Device 0 Mesh ID = {} Logical Device ID = {}", mesh_id_0, logical_dev_id_0);
    tt::log_info(tt::LogTest, "Device 1 Mesh ID = {} Logical Device ID = {}", mesh_id_1, logical_dev_id_1);

    // Find the fabric router XY
    auto [dev_0_router, dev_0_chan] = get_fabric_router_core_and_channel_xy(devices[0], devices[1]);
    auto [dev_1_router, dev_1_chan] = get_fabric_router_core_and_channel_xy(devices[1], devices[0]);

    // CB Base
    auto& mem_map = DispatchMemMap::get(CoreType::WORKER, 1);
    auto cb_base = mem_map.dispatch_buffer_base();
    auto fabric_interface_addr = mem_map.get_device_command_queue_addr(CommandQueueDeviceAddrType::FABRIC_STATE);

    // Upstream (device 0)
    tt::tt_metal::CreateKernel(
        pgm_0,
        k_DummyKernelSrc,
        core,
        tt::tt_metal::DataMovementConfig{
            .processor = tt::tt_metal::DataMovementProcessor::RISCV_0,
            .noc = tt::tt_metal::NOC::RISCV_0_default,
            .compile_args =
                {
                    false,
                    mesh_id_1,
                    logical_dev_id_1,
                    dev_0_router,
                    dev_0_chan,
                    cb_base,
                    fabric_interface_addr,
                },
        });

    // Downstream (device 1)
    tt::tt_metal::CreateKernel(
        pgm_1,
        k_DummyKernelSrc,
        core,
        tt::tt_metal::DataMovementConfig{
            .processor = tt::tt_metal::DataMovementProcessor::RISCV_0,
            .noc = tt::tt_metal::NOC::RISCV_0_default,
            .compile_args =
                {
                    true,
                    mesh_id_0,
                    logical_dev_id_0,
                    dev_1_router,
                    dev_1_chan,
                    cb_base,
                    fabric_interface_addr,
                },
        });

    tt::tt_metal::detail::LaunchProgram(devices[0], pgm_0);
    tt::tt_metal::detail::LaunchProgram(devices[1], pgm_1);
    tt::tt_metal::detail::WaitProgramDone(devices[0], pgm_0);
    tt::tt_metal::detail::WaitProgramDone(devices[1], pgm_1);

    // Teardown
    tt::tt_metal::detail::CloseDevices(devices);
    return 0;
}
