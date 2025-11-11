// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>
#include <fmt/format.h>
#include <algorithm>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <vector>

#include <tt-metalium/tt_metal.hpp>
#include "tests/tt_metal/tt_metal/common/multi_device_fixture.hpp"
#include "tests/tt_metal/tt_fabric/common/utils.hpp"
#include "tests/tt_metal/tt_fabric/fabric_data_movement/addrgen_write/test_common.hpp"
#include "tt_metal/tt_fabric/benchmark/collectives/common/perf_helpers.hpp"
#include "tt_metal/fabric/fabric_context.hpp"
#include <tt-metalium/global_semaphore.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/mesh_device.hpp>
#include <tt-metalium/mesh_device_view.hpp>

namespace tt::tt_fabric::test {

// Import needed types
using tt::tt_fabric::test::AddrgenApiVariant;
using tt::tt_fabric::test::AddrgenTestParams;

// ---------- helpers (validation / utilities) ----------

namespace {

// Lookup device by physical chip ID
inline tt::tt_metal::IDevice* find_device_by_id(ChipId phys_id) {
    auto devices = tt::DevicePool::instance().get_all_active_devices();
    for (auto* d : devices) {
        if (d->id() == phys_id) {
            return d;
        }
    }
    return nullptr;
}

// Validate workload
inline bool validate_workload_or_fail(const AddrgenTestParams& p) {
    if ((p.tensor_bytes % 4) != 0) {
        ADD_FAILURE() << "tensor_bytes must be a multiple of 4 (word-aligned) for verification.";
        return false;
    }
    return true;
}

// Generate deterministic TX pattern.
inline std::vector<uint32_t> make_tx_pattern(size_t n_words) {
    std::vector<uint32_t> tx(n_words);
    for (size_t i = 0; i < n_words; ++i) {
        tx[i] = 0xA5A50000u + static_cast<uint32_t>(i);
    }
    return tx;
}

// Validate RX payload equals TX payload.
inline void verify_payload_words(const std::vector<uint32_t>& rx, const std::vector<uint32_t>& tx) {
    if (rx.size() != tx.size()) {
        ADD_FAILURE() << "RX size mismatch: got " << rx.size() << " words, expected " << tx.size();
        return;
    }
    for (size_t i = 0; i < rx.size(); ++i) {
        if (rx[i] != tx[i]) {
            ADD_FAILURE() << "Data mismatch at word " << i << " (got 0x" << std::hex << rx[i] << ", exp 0x" << tx[i]
                          << std::dec << ")";
            return;
        }
    }
    // OK -> no failure emitted
}

}  // anonymous namespace

// ----------------------------------- program -----------------------------------
void run_multicast_write_test(tt::tt_metal::MeshDeviceFixtureBase* fixture, const AddrgenTestParams& p) {
    const auto& cp = tt::tt_metal::MetalContext::instance().get_control_plane();
    namespace Dist = tt::tt_metal::distributed;

    // Check if fabric is 2D and create defines map
    const auto& fabric_context = cp.get_fabric_context();
    const bool is_2d_fabric = fabric_context.is_2D_routing_enabled();
    std::map<std::string, std::string> defines = {};
    if (is_2d_fabric) {
        defines["FABRIC_2D"] = "1";
    }

    // src/dst nodes
    tt::tt_fabric::FabricNodeId src{tt::tt_fabric::MeshId{p.mesh_id}, p.src_chip};
    tt::tt_fabric::FabricNodeId dst{tt::tt_fabric::MeshId{p.mesh_id}, p.dst_chip};

    ChipId src_phys = cp.get_physical_chip_id_from_fabric_node_id(src);
    ChipId dst_phys = cp.get_physical_chip_id_from_fabric_node_id(dst);

    tt::tt_metal::IDevice* src_dev = find_device_by_id(src_phys);
    tt::tt_metal::IDevice* dst_dev = find_device_by_id(dst_phys);
    if (!src_dev || !dst_dev) {
        ADD_FAILURE() << "Failed to find devices: src=" << src_phys << " dst=" << dst_phys;
        return;
    }

    if (!validate_workload_or_fail(p)) {
        return;
    }

    // For multicast: build 2x2 destination rectangle starting at dst_chip
    // dst_chip is top-left (e.g. 0:0), then 0:1, 1:0, 1:1
    const uint32_t rows = p.mesh_rows;
    const uint32_t cols = p.mesh_cols;

    std::vector<Dist::MeshCoordinate> dst_coords;
    for (uint32_t r = 0; r < rows; ++r) {
        for (uint32_t c = 0; c < cols; ++c) {
            dst_coords.push_back(Dist::MeshCoordinate(r, c));
        }
    }

    // --- Mesh device + coords ---
    auto mesh = fixture->get_mesh_device();
    auto view = mesh->get_view();
    auto coord_of_phys = [&](ChipId phys) -> Dist::MeshCoordinate {
        for (const auto& c : Dist::MeshCoordinateRange(view.shape())) {
            if (view.get_device(c)->id() == phys) {
                return c;
            }
        }
        TT_FATAL(false, "Physical chip {} is not part of this MeshDevice", phys);
        return Dist::MeshCoordinate(0);
    };
    Dist::MeshCoordinate src_coord = coord_of_phys(src_phys);

    // --- IO buffers & initialization ---
    Dist::DeviceLocalBufferConfig src_local{.page_size = p.page_size, .buffer_type = tt::tt_metal::BufferType::DRAM};
    Dist::DeviceLocalBufferConfig dst_local{
        .page_size = p.page_size,
        .buffer_type = p.use_dram_dst ? tt::tt_metal::BufferType::DRAM : tt::tt_metal::BufferType::L1};
    Dist::ReplicatedBufferConfig rcfg{.size = p.tensor_bytes};
    auto src_buf = Dist::MeshBuffer::create(rcfg, src_local, mesh.get());
    auto dst_buf = Dist::MeshBuffer::create(rcfg, dst_local, mesh.get());

    const size_t n_words = p.tensor_bytes / 4;
    auto tx = make_tx_pattern(n_words);
    std::vector<uint32_t> zeros(n_words, 0u);

    // Mesh CQ
    auto& mcq = mesh->mesh_command_queue();
    // Initialize src shard
    Dist::WriteShard(mcq, src_buf, tx, src_coord, /*blocking=*/true);
    // Initialize all dst shards to zero
    for (auto mc : dst_coords) {
        Dist::WriteShard(mcq, dst_buf, zeros, mc, /*blocking=*/true);
    }

    // ---------------------------- PROGRAM FACTORY ----------------------------
    /*
Multicast addrgen write test — top-level flow:
Sends data from src_chip (e.g. 0:2) to a 2x2 rectangle starting at dst_chip (e.g. 0:0).
Uses the fabric_multicast_noc_unicast_write overload with TensorAccessor.
*/

    // Global semaphore on all receivers
    tt::tt_metal::CoreRangeSet rx_core_set(tt::tt_metal::CoreRange(p.receiver_core, p.receiver_core));
    static std::optional<tt::tt_metal::GlobalSemaphore> gsem;
    if (!gsem) {
        gsem = tt::tt_metal::CreateGlobalSemaphore(mesh.get(), rx_core_set, /*initial_value=*/0);
    }

    const std::string KDIR = "tests/tt_metal/tt_fabric/fabric_data_movement/addrgen_write/kernels/multicast/";

    tt::tt_metal::CoreCoord receiver_core = p.receiver_core;

    // --- Prepare receiver kernel parameters (will create programs later) ---
    auto dst_buffer_addr = dst_buf->address();
    auto rx_noc = src_dev->worker_core_from_logical_core(receiver_core);

    uint32_t pages = p.tensor_bytes / p.page_size;
    uint32_t sem_l1_addr = gsem->address();

    // --- create sender program (reader + writer) ---
    tt::tt_metal::Program sender_prog = tt::tt_metal::CreateProgram();
    tt::tt_metal::CoreCoord sender_core = p.sender_core;

    // CB for staging
    constexpr uint32_t cb_id = tt::CBIndex::c_0;
    constexpr uint32_t cb_num_pages = 8;
    tt::tt_metal::CircularBufferConfig cb_config =
        tt::tt_metal::CircularBufferConfig(cb_num_pages * p.page_size, {{cb_id, tt::DataFormat::Float16}})
            .set_page_size(cb_id, p.page_size);
    tt::tt_metal::CreateCircularBuffer(sender_prog, sender_core, cb_config);

    // Reader kernel (DRAM → CB)
    std::vector<uint32_t> reader_cta;
    tt::tt_metal::TensorAccessorArgs(*src_buf).append_to(reader_cta);
    reader_cta.push_back(1u);  // SRC_IS_DRAM
    reader_cta.push_back(pages);
    reader_cta.push_back(p.page_size);

    auto reader_kernel_id = tt::tt_metal::CreateKernel(
        sender_prog,
        KDIR + "multicast_tx_reader_to_cb_addrgen.cpp",
        sender_core,
        tt::tt_metal::DataMovementConfig{
            .processor = tt::tt_metal::DataMovementProcessor::RISCV_0,
            .noc = tt::tt_metal::NOC::RISCV_0_default,
            .compile_args = reader_cta,
            .defines = defines});
    auto src_buffer_addr = src_buf->address();
    tt::tt_metal::SetRuntimeArgs(sender_prog, reader_kernel_id, sender_core, {(uint32_t)src_buffer_addr});

    // Writer kernel (CB → fabric multicast)
    // Compute 2x2 bounding box and hop counts
    uint16_t x_min = dst_coords[0][1];  // col
    uint16_t x_max = x_min;
    uint16_t y_min = dst_coords[0][0];  // row
    uint16_t y_max = y_min;
    for (auto mc : dst_coords) {
        if (mc[1] < x_min) {
            x_min = mc[1];  // col
        }
        if (mc[1] > x_max) {
            x_max = mc[1];  // col
        }
        if (mc[0] < y_min) {
            y_min = mc[0];  // row
        }
        if (mc[0] > y_max) {
            y_max = mc[0];  // row
        }
    }

    // src is at src_coord, compute hops
    uint16_t e_hops = (src_coord[1] < x_max) ? (x_max - src_coord[1]) : 0;  // col
    uint16_t w_hops = (src_coord[1] > x_min) ? (src_coord[1] - x_min) : 0;  // col
    uint16_t n_hops = (src_coord[0] > y_min) ? (src_coord[0] - y_min) : 0;  // row
    uint16_t s_hops = (src_coord[0] < y_max) ? (y_max - src_coord[0]) : 0;  // row

    // Build direction bitmask: W=0x1, E=0x2, N=0x4, S=0x8
    uint32_t dir_mask = 0;
    if (w_hops > 0) {
        dir_mask |= 0x1u;
    }
    if (e_hops > 0) {
        dir_mask |= 0x2u;
    }
    if (n_hops > 0) {
        dir_mask |= 0x4u;
    }
    if (s_hops > 0) {
        dir_mask |= 0x8u;
    }

    // Pick representative coordinates for each direction to get link indices
    std::vector<uint32_t> writer_rt_args = {dst_buffer_addr, rx_noc.x, rx_noc.y, sem_l1_addr, dir_mask};

    // Helper to append fabric connection args for a given direction
    auto append_fabric_connection = [&](ChipId dest_phys) {
        tt::tt_fabric::FabricNodeId dest_node{tt::tt_fabric::MeshId{p.mesh_id}, dest_phys};
        auto links = tt::tt_fabric::get_forwarding_link_indices(
            tt::tt_fabric::FabricNodeId{tt::tt_fabric::MeshId{p.mesh_id}, p.src_chip}, dest_node);
        if (links.empty()) {
            ADD_FAILURE() << "No forwarding link from src to dest " << dest_phys;
            return;
        }
        uint32_t link_idx = links[0];
        tt::tt_fabric::append_fabric_connection_rt_args(
            src, dest_node, link_idx, sender_prog, sender_core, writer_rt_args);
    };

    // WEST: leftmost chip in src's row
    if (w_hops > 0) {
        auto west_coord = Dist::MeshCoordinate(src_coord[0], x_min);  // row, col
        ChipId west_phys = view.get_device(west_coord)->id();
        append_fabric_connection(west_phys);
    }

    // EAST: rightmost chip in src's row
    if (e_hops > 0) {
        auto east_coord = Dist::MeshCoordinate(src_coord[0], x_max);  // row, col
        ChipId east_phys = view.get_device(east_coord)->id();
        append_fabric_connection(east_phys);
    }

    // NORTH: topmost chip in src's col
    if (n_hops > 0) {
        auto north_coord = Dist::MeshCoordinate(y_min, src_coord[1]);  // row, col
        ChipId north_phys = view.get_device(north_coord)->id();
        append_fabric_connection(north_phys);
    }

    // SOUTH: bottommost chip in src's col
    if (s_hops > 0) {
        auto south_coord = Dist::MeshCoordinate(y_max, src_coord[1]);  // row, col
        ChipId south_phys = view.get_device(south_coord)->id();
        append_fabric_connection(south_phys);
    }

    // Append hop counts
    writer_rt_args.push_back(e_hops);
    writer_rt_args.push_back(w_hops);
    writer_rt_args.push_back(n_hops);
    writer_rt_args.push_back(s_hops);

    std::vector<uint32_t> writer_cta;
    tt::tt_metal::TensorAccessorArgs(*dst_buf).append_to(writer_cta);
    writer_cta.push_back(pages);
    writer_cta.push_back(p.page_size);

    auto writer_kernel_id = tt::tt_metal::CreateKernel(
        sender_prog,
        KDIR + "multicast_tx_writer_cb_to_dst_addrgen.cpp",
        sender_core,
        tt::tt_metal::DataMovementConfig{
            .processor = tt::tt_metal::DataMovementProcessor::RISCV_1,
            .noc = tt::tt_metal::NOC::RISCV_1_default,
            .compile_args = writer_cta,
            .defines = defines});
    tt::tt_metal::SetRuntimeArgs(sender_prog, writer_kernel_id, sender_core, writer_rt_args);

    // --- Enqueue both workloads ---
    Dist::MeshWorkload receiver_workload;
    Dist::MeshWorkload sender_workload;

    // Create and add separate receiver program for each destination device
    for (auto mc : dst_coords) {
        tt::tt_metal::Program recv_prog = tt::tt_metal::CreateProgram();
        auto rx_kernel_id = tt::tt_metal::CreateKernel(
            recv_prog,
            KDIR + "multicast_rx_addrgen.cpp",
            receiver_core,
            tt::tt_metal::DataMovementConfig{
                .processor = tt::tt_metal::DataMovementProcessor::RISCV_0,
                .noc = tt::tt_metal::NOC::RISCV_0_default,
                .defines = defines});
        tt::tt_metal::SetRuntimeArgs(recv_prog, rx_kernel_id, receiver_core, {sem_l1_addr, pages});
        receiver_workload.add_program(Dist::MeshCoordinateRange(mc), std::move(recv_prog));
    }
    sender_workload.add_program(Dist::MeshCoordinateRange(src_coord), std::move(sender_prog));

    // Enqueue
    Dist::EnqueueMeshWorkload(mcq, receiver_workload, /*blocking=*/false);
    Dist::EnqueueMeshWorkload(mcq, sender_workload, /*blocking=*/true);

    // --- Verify data correctness on all dst devices ---
    for (auto mc : dst_coords) {
        std::vector<uint32_t> rx_data(n_words, 0u);
        Dist::ReadShard(mcq, rx_data, dst_buf, mc, /*blocking=*/true);
        verify_payload_words(rx_data, tx);
    }
}

}  // namespace tt::tt_fabric::test
