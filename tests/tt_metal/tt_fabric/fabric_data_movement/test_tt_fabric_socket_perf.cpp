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
#include <fstream>
#include <filesystem>
#include <iomanip>

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
#include <chrono>
using tt::DevicePool;

namespace tt::tt_fabric {
namespace fabric_router_tests {

struct PerfPoint {
    uint64_t bytes;  // p.tensor_bytes
    double sec;
    double ms;
    double gbps;
};

struct PerfParams {
    uint32_t mesh_id = 0;       // mesh to use
    chip_id_t src_chip = 0;     // logical chip id in that mesh
    chip_id_t dst_chip = 1;     // logical chip id in that mesh
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

static inline PerfPoint RunUnicastConnWithParams(BaseFabricFixture* fixture, const PerfParams& p) {
    const auto& cp = tt::tt_metal::MetalContext::instance().get_control_plane();

    tt::tt_fabric::FabricNodeId src{tt::tt_fabric::MeshId{p.mesh_id}, p.src_chip};
    tt::tt_fabric::FabricNodeId dst{tt::tt_fabric::MeshId{p.mesh_id}, p.dst_chip};

    chip_id_t src_phys = cp.get_physical_chip_id_from_fabric_node_id(src);
    chip_id_t dst_phys = cp.get_physical_chip_id_from_fabric_node_id(dst);

    // Get IDevice*
    auto* src_dev = find_device_by_id(src_phys);
    auto* dst_dev = find_device_by_id(dst_phys);
    if (!src_dev || !dst_dev) {
        ADD_FAILURE() << "Failed to find devices: src=" << src_phys << " dst=" << dst_phys;
        return PerfPoint{};
    }

    CoreCoord tx_xy = src_dev->worker_core_from_logical_core(p.sender_core);
    CoreCoord rx_xy = dst_dev->worker_core_from_logical_core(p.receiver_core);

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

    // ---------- Build a receiver progrma ----------
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

    // expected_value = 1  -> return after sender bumps the sem once
    tt::tt_metal::SetRuntimeArgs(receiver_prog, rx_wait_k, receiver_core, {gsem.address(), 1u});

    // ---------------- Sender program: READER (RISCV_0) + WRITER (RISCV_1) ----------------
    tt::tt_metal::Program sender_prog = tt::tt_metal::CreateProgram();

    // Compute how many pages we need to send for the requested tensor size.
    const uint32_t NUM_PAGES = (p.tensor_bytes + p.page_size - 1) / p.page_size;
    const uint32_t CB_ID = tt::CBIndex::c_0;
    auto cb_cfg = tt::tt_metal::CircularBufferConfig(2 * p.page_size, {{CB_ID, tt::DataFormat::Float16}})
                      .set_page_size(CB_ID, p.page_size);
    (void)tt::tt_metal::CreateCircularBuffer(sender_prog, p.sender_core, cb_cfg);

    std::cout << "[host] src_buf=0x" << std::hex << src_buf->address() << " dst_buf=0x" << dst_buf->address()
              << " sem_addr=0x" << gsem.address() << std::dec << "\n";

    std::cout << "[host] launching RX: core=(" << receiver_core.x << "," << receiver_core.y << ") expect=1\n";
    std::cout << "[host] launching TX: pages=" << NUM_PAGES << " page_size=" << p.page_size
              << " dst_is_dram=" << (p.use_dram_dst ? 1 : 0) << "\n";

    // READER kernel (DRAM->CB or L1->CB). We read from src_buf (DRAM).
    auto reader_k = tt::tt_metal::CreateKernel(
        sender_prog,
        "tests/tt_metal/tt_fabric/fabric_data_movement/kernels/fabric_sender_reader_for_perf.cpp",
        p.sender_core,
        tt::tt_metal::DataMovementConfig{
            .processor = tt::tt_metal::DataMovementProcessor::RISCV_0,
            .noc = tt::tt_metal::NOC::RISCV_0_default,
            .compile_args = {1u /*SRC_IS_DRAM*/, NUM_PAGES, p.page_size}});
    tt::tt_metal::SetRuntimeArgs(sender_prog, reader_k, p.sender_core, {(uint32_t)src_buf->address()});

    // WRITER kernel (CB->Fabric->dst + final sem INC)
    auto writer_k = tt::tt_metal::CreateKernel(
        sender_prog,
        "tests/tt_metal/tt_fabric/fabric_data_movement/kernels/fabric_sender_writer_for_perf.cpp",
        p.sender_core,
        tt::tt_metal::DataMovementConfig{
            .processor = tt::tt_metal::DataMovementProcessor::RISCV_1,
            .noc = tt::tt_metal::NOC::RISCV_1_default,
            .compile_args = {NUM_PAGES, p.page_size}});

    // Writer runtime args
    std::vector<uint32_t> writer_rt = {
        (uint32_t)dst_buf->address(),        // 0: dst_base (receiver L1 offset)
        (uint32_t)(p.use_dram_dst ? 1 : 0),  // 1: dst_is_dram (we set false above)
        (uint32_t)p.mesh_id,                 // 2: dst_mesh_id (logical)
        (uint32_t)p.dst_chip,                // 3: dst_dev_id  (logical)
        (uint32_t)rx_xy.x,                   // 4: receiver_noc_x
        (uint32_t)rx_xy.y,                   // 5: receiver_noc_y
        (uint32_t)gsem.address()             // 6: receiver L1 semaphore addr
    };

    auto dir_opt = get_eth_forwarding_direction(src, dst);

    auto dir_to_str = [](eth_chan_directions d) {
        switch (d) {
            case eth_chan_directions::NORTH: return "NORTH";
            case eth_chan_directions::SOUTH: return "SOUTH";
            case eth_chan_directions::EAST: return "EAST";
            case eth_chan_directions::WEST: return "WEST";
            default: return "UNKNOWN";
        }
    };

    if (dir_opt.has_value()) {
        std::cout << "[host] CP forwarding dir = " << dir_to_str(*dir_opt) << "\n";
    } else {
        std::cout << "[host] CP forwarding dir = <none>\n";
    }

    std::cout << "[host] logical src(mesh=" << p.mesh_id << ", dev=" << p.src_chip << ") -> dst(mesh=" << p.mesh_id
              << ", dev=" << p.dst_chip << ")\n";

    std::cout << "[host] physical src=" << src_phys << " -> dst=" << dst_phys << "\n";

    // Append fabric connection RT args so WorkerToFabricEdmSender can open the link
    auto links = tt::tt_fabric::get_forwarding_link_indices(src, dst);
    if (links.empty()) {
        ADD_FAILURE() << "No forwarding links from src(mesh=" << p.mesh_id << ",dev=" << p.src_chip
                      << ") to dst(mesh=" << p.mesh_id << ",dev=" << p.dst_chip << ")";
        return PerfPoint{};
    }

    uint32_t link_idx = links[0];
    std::cout << "[host] forwarding links:";
    for (auto li : links) {
        std::cout << " " << li;
    }
    std::cout << " (using " << link_idx << ")\n";

    tt::tt_fabric::append_fabric_connection_rt_args(
        src, dst, /*link_index=*/link_idx, sender_prog, p.sender_core, writer_rt);

    tt::tt_metal::SetRuntimeArgs(sender_prog, writer_k, p.sender_core, writer_rt);

    // ---------------- Run order: receiver first (waits), then sender ----------------
    fixture->RunProgramNonblocking(dst_dev, receiver_prog);

    auto t0 = std::chrono::steady_clock::now();
    fixture->RunProgramNonblocking(src_dev, sender_prog);
    fixture->WaitForSingleProgramDone(dst_dev, receiver_prog);
    auto t1 = std::chrono::steady_clock::now();

    fixture->WaitForSingleProgramDone(src_dev, sender_prog);

    // Compute E2E metrics
    const double e2e_sec = std::chrono::duration<double>(t1 - t0).count();
    const uint64_t bytes = static_cast<uint64_t>(p.tensor_bytes);
    const double gb = static_cast<double>(bytes) / 1e9;
    const double gbps = (e2e_sec > 0.0) ? (gb / e2e_sec) : 0.0;
    const double ms = e2e_sec * 1000.0;

    std::cout << "[perf] E2E: bytes=" << static_cast<uint64_t>(bytes) << " time=" << e2e_sec << " s"
              << " (" << ms << " ms)"
              << " throughput=" << gbps << " GB/s\n";

    return PerfPoint{
        .bytes = bytes,
        .sec = e2e_sec,
        .ms = ms,
        .gbps = gbps,
    };
}

TEST_F(Fabric2DFixture, UnicastConn_CodeControlled) {
    PerfParams p;
    p.mesh_id = 0;
    p.src_chip = 0;
    p.dst_chip = 1;
    p.use_dram_dst = false;
    p.page_size = 2048;
    p.tensor_bytes = 100 * p.page_size;

    RunUnicastConnWithParams(this, p);
}

TEST_F(Fabric2DFixture, UnicastConn_SweepTensorSize) {
    PerfParams base;
    base.mesh_id = 0;
    base.src_chip = 0;
    base.dst_chip = 1;
    base.use_dram_dst = false;
    base.page_size = 2048;
    base.sender_core = {0, 0};
    base.receiver_core = {0, 0};

    std::vector<uint32_t> sizes_bytes = {
        1 * base.page_size,
        2 * base.page_size,
        4 * base.page_size,
        8 * base.page_size,
        16 * base.page_size,
        32 * base.page_size,
        64 * base.page_size,
        128 * base.page_size,
        256 * base.page_size};

    std::vector<PerfPoint> results;
    results.reserve(sizes_bytes.size());

    for (auto sz : sizes_bytes) {
        PerfParams p = base;
        p.tensor_bytes = sz;

        auto r = RunUnicastConnWithParams(this, p);
        results.push_back(r);
    }

    std::filesystem::create_directories("artifacts");
    const auto now = std::chrono::system_clock::now();
    const auto ts = std::chrono::duration_cast<std::chrono::seconds>(now.time_since_epoch()).count();
    const std::string csv_name = fmt::format(
        "artifacts/unicast_sweep_m{}_s{}_d{}_p{}_t{}.csv",
        base.mesh_id,
        base.src_chip,
        base.dst_chip,
        base.page_size,
        ts);
    std::ofstream ofs(csv_name);
    ofs << std::fixed << std::setprecision(6);
    ofs << "bytes,ms,gbps\n";
    for (const auto& r : results) {
        ofs << r.bytes << "," << r.ms << "," << r.gbps << "\n";
    }
    ofs.close();
    std::cout << "[perf] wrote " << results.size() << " points -> " << csv_name << "\n";
}

}  // namespace fabric_router_tests
}  // namespace tt::tt_fabric
