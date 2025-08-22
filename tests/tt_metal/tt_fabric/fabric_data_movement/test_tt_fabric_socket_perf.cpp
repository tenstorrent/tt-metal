// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <fmt/base.h>
#include <gtest/gtest.h>
#include <stdint.h>
#include <tt-metalium/device_pool.hpp>
#include "hostdevcommon/fabric_common.h"
#include <algorithm>
#include <map>
#include <memory>
#include <numeric>
#include <set>
#include <string>
#include <unordered_map>
#include <utility>
#include <variant>
#include <vector>

#include <tt-metalium/buffer.hpp>
#include <tt-metalium/buffer_types.hpp>
#include <tt-metalium/circular_buffer_config.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/control_plane.hpp>
#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/data_types.hpp>
#include <tt-metalium/device.hpp>
#include "fabric_fixture.hpp"
#include "t3k_mesh_descriptor_chip_mappings.hpp"
#include "utils.hpp"
#include <tt-metalium/hal.hpp>
#include <tt-metalium/host_api.hpp>
#include "hostdevcommon/kernel_structs.h"
#include <tt-metalium/kernel_types.hpp>
#include <tt-logger/tt-logger.hpp>
#include <tt-metalium/mesh_graph.hpp>
#include <tt-metalium/program.hpp>
#include <tt_stl/span.hpp>
#include "impl/context/metal_context.hpp"
#include "test_common.hpp"
#include <tt-metalium/tt_backend_api_types.hpp>
#include <tt-metalium/tt_metal.hpp>
#include "umd/device/types/xy_pair.h"
#include <tt-metalium/utils.hpp>
#include "tests/tt_metal/tt_fabric/common/utils.hpp"
#include <tt-metalium/device_pool.hpp>
#include <tt-metalium/buffer.hpp>
#include <tt-metalium/buffer_types.hpp>
#include <tt-metalium/global_semaphore.hpp>
using tt::DevicePool;

namespace tt::tt_fabric {
namespace fabric_router_tests {

void run_unicast_test_bw_chips(
    BaseFabricFixture* fixture,
    chip_id_t src_physical_device_id,
    chip_id_t dst_physical_device_id,
    uint32_t num_hops,
    bool use_dram_dst = false);

TEST_F(Fabric2DFixture, UnicastRaw_Skeleton) { RunTestUnicastRaw(this); }

TEST_F(Fabric2DFixture, UnicastConn_Skeleton) { RunTestUnicastConnAPI(this, /*num_connections=*/1); }

TEST_F(Fabric2DFixture, UnicastConn_Timed_Skeleton) {
    auto t0 = std::chrono::high_resolution_clock::now();
    RunTestUnicastConnAPI(this, /*num_connections=*/1);
    auto t1 = std::chrono::high_resolution_clock::now();

    double sec = std::chrono::duration<double>(t1 - t0).count();
    std::cout << "[UnicastConn_Timed_Skeleton] wall_time_s=" << sec << "\n";
}

struct PerfParams {
    uint32_t mesh_id = 0;       // mesh to use
    chip_id_t src_chip = 0;     // logical chip id in that mesh
    chip_id_t dst_chip = 1;     // logical chip id in that mesh
    uint32_t num_hops = 1;      // 1 = direct neighbor, >1 = farther away
    bool use_dram_dst = false;  // false -> land in L1 on dst; true -> land in DRAM
    uint32_t tensor_bytes = 1024 * 1024;
    uint32_t page_size = 4096;
    CoreCoord sender_core = {0, 0};
    CoreCoord receiver_core = {0, 0};
};

static inline tt::tt_metal::IDevice* find_device_by_id(chip_id_t phys_id) {
    auto devices = DevicePool::instance().get_all_active_devices();
    for (auto* d : devices) {
        if (d->id() == phys_id) {
            return d;
        }
    }
    return nullptr;
}

static inline void RunUnicastConnWithParams(BaseFabricFixture* fixture, const PerfParams& p) {
    const auto& cp = tt::tt_metal::MetalContext::instance().get_control_plane();

    tt::tt_fabric::FabricNodeId src{tt::tt_fabric::MeshId{p.mesh_id}, p.src_chip};
    tt::tt_fabric::FabricNodeId dst{tt::tt_fabric::MeshId{p.mesh_id}, p.dst_chip};

    chip_id_t src_phys = cp.get_physical_chip_id_from_fabric_node_id(src);
    chip_id_t dst_phys = cp.get_physical_chip_id_from_fabric_node_id(dst);

    // Get IDevice*
    auto* src_dev = find_device_by_id(src_phys);
    auto* dst_dev = find_device_by_id(dst_phys);
    ASSERT_NE(src_dev, nullptr);
    ASSERT_NE(dst_dev, nullptr);

    CoreCoord tx_xy = src_dev->worker_core_from_logical_core(p.sender_core);
    CoreCoord rx_xy = dst_dev->worker_core_from_logical_core(p.receiver_core);
    (void)tx_xy;
    (void)rx_xy;  // silence unused for now (sender comes next)

    // Allocate simple flat buffers (you control size via p.tensor_bytes)
    tt::tt_metal::BufferConfig src_cfg{
        .device = src_dev,
        .size = p.tensor_bytes,
        .page_size = p.page_size,
        .buffer_type = tt::tt_metal::BufferType::DRAM  // or L1 if it fits
    };
    tt::tt_metal::BufferConfig dst_cfg{
        .device = dst_dev,
        .size = p.tensor_bytes,
        .page_size = p.page_size,
        .buffer_type = p.use_dram_dst ? tt::tt_metal::BufferType::DRAM : tt::tt_metal::BufferType::L1};

    auto src_buf = tt::tt_metal::CreateBuffer(src_cfg);
    auto dst_buf = tt::tt_metal::CreateBuffer(dst_cfg);

    std::cout << "[alloc] src_phys=" << src_phys << " dst_phys=" << dst_phys << " bytes=" << p.tensor_bytes
              << std::endl;

    // run_unicast_test_bw_chips(fixture, src_phys, dst_phys, p.num_hops, p.use_dram_dst);

    // ---------- Build a tiny receiver-only program (expected_value = 0 so it returns immediately) ----------
    tt::tt_metal::Program receiver_prog = tt::tt_metal::CreateProgram();

    // create a global semaphore on the dst device
    auto gsem = tt::tt_metal::CreateGlobalSemaphore(
        dst_dev,
        dst_dev->worker_cores(tt::tt_metal::HalProgrammableCoreType::TENSIX, tt::tt_metal::SubDeviceId{0}),
        /*initial=*/0,
        tt::tt_metal::BufferType::L1);

    const CoreCoord receiver_core = p.receiver_core;
    auto rx_wait_k = tt::tt_metal::CreateKernel(
        receiver_prog,
        "tests/tt_metal/tt_fabric/fabric_data_movement/kernels/fabric_receiver_for_perf.cpp",
        receiver_core,
        tt::tt_metal::DataMovementConfig{
            .processor = tt::tt_metal::DataMovementProcessor::RISCV_0, .noc = tt::tt_metal::NOC::RISCV_0_default});

    // expected_value = 0  -> returns immediately (no sender needed yet)
    tt::tt_metal::SetRuntimeArgs(receiver_prog, rx_wait_k, receiver_core, {gsem.address(), 0u});

    // Run it
    fixture->RunProgramNonblocking(dst_dev, receiver_prog);
    fixture->WaitForSingleProgramDone(dst_dev, receiver_prog);
}

TEST_F(Fabric2DFixture, UnicastConn_CodeControlled) {
    PerfParams p;
    p.mesh_id = 0;
    p.src_chip = 0;
    p.dst_chip = 1;
    p.num_hops = 1;          // e.g., 1 for neighbor
    p.use_dram_dst = true;   // set true to land in DRAM on dst
    p.tensor_bytes = 4 * 1024 * 1024;

    RunUnicastConnWithParams(this, p);
}

}  // namespace fabric_router_tests
}  // namespace tt::tt_fabric
