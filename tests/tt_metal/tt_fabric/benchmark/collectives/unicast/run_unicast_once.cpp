// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>
#include <fmt/format.h>
#include <algorithm>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <vector>

#include <tt-metalium/tt_metal.hpp>
#include "fabric_fixture.hpp"
#include "tests/tt_metal/tt_fabric/common/utils.hpp"
#include "tt_metal/tt_fabric/benchmark/collectives/common/perf_helpers.hpp"
#include <tt-metalium/global_semaphore.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>

namespace tt::tt_fabric::bench {
PerfPoint run_unicast_once(HelpersFixture* fixture, const PerfParams& p) {
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

    tt::tt_metal::CoreCoord tx_xy = src_dev->worker_core_from_logical_core(p.sender_core);
    tt::tt_metal::CoreCoord rx_xy = dst_dev->worker_core_from_logical_core(p.receiver_core);

    // Allocate simple flat buffers
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

    // ---- Initialize source buffer with a deterministic pattern; clear dest ----
    auto& cq_src = src_dev->command_queue();
    auto& cq_dst = dst_dev->command_queue();

    if ((p.tensor_bytes % 4) != 0) {
        ADD_FAILURE() << "tensor_bytes must be a multiple of 4 for word-wise verification";
        return PerfPoint{};
    }
    const size_t n_words = p.tensor_bytes / 4;

    std::vector<uint32_t> tx(n_words);
    for (size_t i = 0; i < n_words; ++i) {
        // simple deterministic pattern
        tx[i] = 0xA5A50000u + static_cast<uint32_t>(i);
    }
    // Blocking writes so data is resident before kernels run
    tt::tt_metal::EnqueueWriteBuffer(cq_src, *src_buf, tx, /*blocking=*/true);

    // clear dst so we can detect partial/corrupt writes
    std::vector<uint32_t> zeros(n_words, 0u);
    tt::tt_metal::EnqueueWriteBuffer(cq_dst, *dst_buf, zeros, /*blocking=*/true);

    std::cout << "[alloc] src_phys=" << src_phys << " dst_phys=" << dst_phys << " bytes=" << p.tensor_bytes
              << std::endl;

    // ---------- Build a receiver program ----------
    tt::tt_metal::Program receiver_prog = tt::tt_metal::CreateProgram();

    // create a global semaphore on the dst device
    auto gsem = tt::tt_metal::CreateGlobalSemaphore(
        dst_dev,
        dst_dev->worker_cores(tt::tt_metal::HalProgrammableCoreType::TENSIX, tt::tt_metal::SubDeviceId{0}),
        /*initial_value=*/0,
        tt::tt_metal::BufferType::L1);

    const tt::tt_metal::CoreCoord receiver_core = p.receiver_core;

    constexpr const char* KDIR = "tests/tt_metal/tt_fabric/benchmark/collectives/unicast/kernels/";
    auto rx_wait_k = tt::tt_metal::CreateKernel(
        receiver_prog,
        std::string(KDIR) + "unicast_rx.cpp",
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
    std::vector<uint32_t> reader_cta;
    tt::tt_metal::TensorAccessorArgs(*src_buf).append_to(reader_cta);
    reader_cta.push_back(1u /*SRC_IS_DRAM*/);
    reader_cta.push_back(NUM_PAGES);
    reader_cta.push_back(p.page_size);

    auto reader_k = tt::tt_metal::CreateKernel(
        sender_prog,
        std::string(KDIR) + "unicast_tx_reader_to_cb.cpp",
        p.sender_core,
        tt::tt_metal::DataMovementConfig{
            .processor = tt::tt_metal::DataMovementProcessor::RISCV_0,
            .noc = tt::tt_metal::NOC::RISCV_0_default,
            .compile_args = reader_cta});
    tt::tt_metal::SetRuntimeArgs(sender_prog, reader_k, p.sender_core, {(uint32_t)src_buf->address()});

    // WRITER kernel (CB->Fabric->dst + final sem INC)
    std::vector<uint32_t> writer_cta;
    tt::tt_metal::TensorAccessorArgs(*dst_buf).append_to(writer_cta);
    writer_cta.push_back(NUM_PAGES);  // == TOTAL_PAGES in kernel
    writer_cta.push_back(p.page_size);

    auto writer_k = tt::tt_metal::CreateKernel(
        sender_prog,
        std::string(KDIR) + "unicast_tx_writer_cb_to_dst.cpp",
        p.sender_core,
        tt::tt_metal::DataMovementConfig{
            .processor = tt::tt_metal::DataMovementProcessor::RISCV_1,
            .noc = tt::tt_metal::NOC::RISCV_1_default,
            .compile_args = writer_cta});

    // Writer runtime args
    std::vector<uint32_t> writer_rt = {
        (uint32_t)dst_buf->address(),  // 0: dst_base (receiver L1 offset)
        (uint32_t)p.mesh_id,           // 1: dst_mesh_id (logical)
        (uint32_t)p.dst_chip,          // 2: dst_dev_id  (logical)
        (uint32_t)rx_xy.x,             // 3: receiver_noc_x
        (uint32_t)rx_xy.y,             // 4: receiver_noc_y
        (uint32_t)gsem.address()       // 5: receiver L1 semaphore addr
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
        src, dst, /*link_idx=*/link_idx, sender_prog, p.sender_core, writer_rt);

    tt::tt_metal::SetRuntimeArgs(sender_prog, writer_k, p.sender_core, writer_rt);

    // ---------------- Run order: receiver first (waits), then sender ----------------
    fixture->RunProgramNonblocking(dst_dev, receiver_prog);

    auto t0 = std::chrono::steady_clock::now();
    fixture->RunProgramNonblocking(src_dev, sender_prog);
    fixture->WaitForSingleProgramDone(dst_dev, receiver_prog);
    auto t1 = std::chrono::steady_clock::now();

    fixture->WaitForSingleProgramDone(src_dev, sender_prog);

    // ---- Read back destination buffer and verify ----
    tt::tt_metal::Finish(cq_dst);
    std::vector<uint32_t> rx;
    tt::tt_metal::EnqueueReadBuffer(cq_dst, *dst_buf, rx, /*blocking=*/true);

    if (rx.size() != tx.size()) {
        ADD_FAILURE() << "RX size mismatch: got " << rx.size() << " words, expected " << tx.size();
    } else {
        // Compare content
        size_t first_bad = rx.size();
        for (size_t i = 0; i < rx.size(); ++i) {
            if (rx[i] != tx[i]) {
                first_bad = i;
                break;
            }
        }
        if (first_bad != rx.size()) {
            ADD_FAILURE() << "Data mismatch at word " << first_bad << " (got 0x" << std::hex << rx[first_bad]
                          << ", exp 0x" << tx[first_bad] << std::dec << ")";
        } else {
            std::cout << "[verify] payload OK (" << rx.size() * 4 << " bytes)\n";
        }
    }

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
}  // namespace tt::tt_fabric::bench
