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
#include "tests/tt_metal/tt_metal/common/multi_device_fixture.hpp"
#include "tests/tt_metal/tt_fabric/common/utils.hpp"
#include "tt_metal/tt_fabric/benchmark/collectives/common/perf_helpers.hpp"
#include <tt-metalium/global_semaphore.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/mesh_device.hpp>
#include <tt-metalium/mesh_device_view.hpp>
#include <distributed/mesh_device_view_impl.hpp>

namespace tt::tt_fabric::bench {

// ---------- helpers (validation / utilities) ----------

namespace {

// Validate workload
inline bool validate_workload_or_fail(const PerfParams& p) {
    if ((p.tensor_bytes % 4) != 0) {
        ADD_FAILURE() << "tensor_bytes must be a multiple of 4 (word-aligned) for verification.";
        return false;
    }
    return true;
}

// Resolve forwarding link and fail early if none found.
inline bool pick_forwarding_link_or_fail(
    const tt::tt_fabric::FabricNodeId& /*src*/,
    const tt::tt_fabric::FabricNodeId& /*dst*/,
    uint32_t& out_link_idx,
    const PerfParams& p) {
    auto links = tt::tt_fabric::get_forwarding_link_indices(
        tt::tt_fabric::FabricNodeId{tt::tt_fabric::MeshId{p.mesh_id}, p.src_chip},
        tt::tt_fabric::FabricNodeId{tt::tt_fabric::MeshId{p.mesh_id}, p.dst_chip});

    if (links.empty()) {
        ADD_FAILURE() << "No forwarding links from src(mesh=" << p.mesh_id << ",dev=" << p.src_chip
                      << ") to dst(mesh=" << p.mesh_id << ",dev=" << p.dst_chip << ")";
        return false;
    }
    out_link_idx = links[0];
    return true;
}

// Device lookup and basic existence check.
inline bool lookup_devices_or_fail(
    ChipId src_phys, ChipId dst_phys, tt::tt_metal::IDevice*& src_dev, tt::tt_metal::IDevice*& dst_dev) {
    src_dev = tt::tt_metal::detail::GetActiveDevice(src_phys);
    dst_dev = tt::tt_metal::detail::GetActiveDevice(dst_phys);
    if (!src_dev || !dst_dev) {
        ADD_FAILURE() << "Failed to find devices: src=" << src_phys << " dst=" << dst_phys;
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
PerfPoint run_unicast_once(HelpersFixture* fixture, const PerfParams& p) {
    const auto& cp = tt::tt_metal::MetalContext::instance().get_control_plane();
    namespace Dist = tt::tt_metal::distributed;

    tt::tt_fabric::FabricNodeId src{tt::tt_fabric::MeshId{p.mesh_id}, p.src_chip};
    tt::tt_fabric::FabricNodeId dst{tt::tt_fabric::MeshId{p.mesh_id}, p.dst_chip};

    ChipId src_phys = cp.get_physical_chip_id_from_fabric_node_id(src);
    ChipId dst_phys = cp.get_physical_chip_id_from_fabric_node_id(dst);

    tt::tt_metal::IDevice* src_dev = nullptr;
    tt::tt_metal::IDevice* dst_dev = nullptr;
    if (!lookup_devices_or_fail(src_phys, dst_phys, src_dev, dst_dev)) {
        return PerfPoint{};
    }

    if (!validate_workload_or_fail(p)) {
        return PerfPoint{};
    }

    tt::tt_metal::CoreCoord rx_xy = dst_dev->worker_core_from_logical_core(p.receiver_core);

    // --- Mesh device + coords for per-shard IO ---
    auto mesh = fixture->get_mesh_device();
    auto view = mesh->get_view();
    auto coord_of_phys = [&](ChipId phys) -> Dist::MeshCoordinate {
        for (const auto& c : Dist::MeshCoordinateRange(view.shape())) {
            if (view.impl().get_device(c)->id() == phys) {
                return c;
            }
        }
        TT_FATAL(false, "Physical chip {} is not part of this MeshDevice", phys);
        return Dist::MeshCoordinate(0);
    };
    Dist::MeshCoordinate src_coord = coord_of_phys(src_phys);
    Dist::MeshCoordinate dst_coord = coord_of_phys(dst_phys);

    // --- IO buffers & initialization (MeshBuffer style) ---
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

    // Mesh CQ (needed for shard I/O and later trace)
    auto& mcq = mesh->mesh_command_queue();
    // Initialize shards on specific src/dst devices (pass CQ, use vectors)
    Dist::WriteShard(mcq, src_buf, tx, src_coord, /*blocking=*/true);
    Dist::WriteShard(mcq, dst_buf, zeros, dst_coord, /*blocking=*/true);

    // ---------------------------- PROGRAM FACTORY ----------------------------
    /*
Unicast bench — top-level flow:

┌────────────────────────────┐                               ┌────────────────────────────┐
│ Device SRC (chip p.src)    │                               │ Device DST (chip p.dst)    │
│                            │                               │                            │
│  DRAM src_buf ──► Reader   │ pages →  L1 CB (c_0)  ──►     │  L1/DRAM dst_buf           │
│                 (RISCV_0)  │            ▲          │       │        ▲                   │
│                            │            │          │       │        │                   │
│       Writer (RISCV_1) ────┴────────────┴──────────┼──────►│  Receiver wait kernel      │
│       + fabric send adapter|            payload+hdr│       │  (RISCV_0) on GLOBAL sem   │
│                            │                       │       │                            │
│ after last page: send      │                       │       │ fabric delivers all data   │
│ atomic_inc to dst.sem ─────┼───────────────────────┼──-───►│ then sem++ → receiver exit │
└────────────────────────────┘      (Fabric link)            └────────────────────────────┘

Flow:
1) Reader DMA-batches DRAM → CB. 2) Writer drains CB, sends packets over fabric.
3) Writer finally sends a semaphore INC to DST. Fabric orders this after payloads.
4) Receiver sees sem++ and returns. Host verifies bytes.

Notes:
- We use a GLOBAL semaphore on DST so a different core can observe the signal.
- Route setup uses current 2D API. This will change soon. The 1D reference shape is in linear/api.h.
*/

    // Global semaphore so a remote chip can signal it.
    // Fabric guarantees payload is visible before the bump is seen.
    tt::tt_metal::Program receiver_prog = tt::tt_metal::CreateProgram();
    static std::optional<tt::tt_metal::GlobalSemaphore> gsemA;
    static std::optional<tt::tt_metal::GlobalSemaphore> gsemB;
    // Create the semaphore on the specific receiver logical core of the *mesh*.
    tt::tt_metal::CoreRangeSet rx_core_set(tt::tt_metal::CoreRange(p.receiver_core, p.receiver_core));
    if (!gsemA) {
        gsemA = tt::tt_metal::CreateGlobalSemaphore(
            mesh.get(),
            rx_core_set,
            /*initial_value=*/0);
    }
    if (!gsemB) {
        gsemB = tt::tt_metal::CreateGlobalSemaphore(
            mesh.get(),
            rx_core_set,
            /*initial_value=*/0);
    }

    static uint32_t sem_sel = 0;
    auto& gsem = (sem_sel++ & 1) ? *gsemB : *gsemA;

    const tt::tt_metal::CoreCoord receiver_core = p.receiver_core;
    constexpr const char* KDIR = "tests/tt_metal/tt_fabric/benchmark/collectives/unicast/kernels/";

    auto rx_wait_k = tt::tt_metal::CreateKernel(
        receiver_prog,
        std::string(KDIR) + "unicast_rx.cpp",
        receiver_core,
        tt::tt_metal::DataMovementConfig{
            .processor = tt::tt_metal::DataMovementProcessor::RISCV_0, .noc = tt::tt_metal::NOC::RISCV_0_default});
    tt::tt_metal::SetRuntimeArgs(receiver_prog, rx_wait_k, receiver_core, {gsem.address(), 1u});

    // Sender program: READER (RISCV_0) + WRITER (RISCV_1)
    tt::tt_metal::Program sender_prog = tt::tt_metal::CreateProgram();

    const uint32_t NUM_PAGES = (p.tensor_bytes + p.page_size - 1) / p.page_size;
    const uint32_t CB_ID = tt::CBIndex::c_0;
    // CB holds 8 pages total so the reader can fill 4 while the writer drains 4.
    auto cb_cfg = tt::tt_metal::CircularBufferConfig(8 * p.page_size, {{CB_ID, tt::DataFormat::Float16}})
                      .set_page_size(CB_ID, p.page_size);
    (void)tt::tt_metal::CreateCircularBuffer(sender_prog, p.sender_core, cb_cfg);

    // Reader kernel (DRAM->CB)
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

    // Writer kernel (CB->Fabric->dst + final sem INC)
    std::vector<uint32_t> writer_cta;
    tt::tt_metal::TensorAccessorArgs(*dst_buf).append_to(writer_cta);
    writer_cta.push_back(NUM_PAGES);
    writer_cta.push_back(p.page_size);

    auto writer_k = tt::tt_metal::CreateKernel(
        sender_prog,
        std::string(KDIR) + "unicast_tx_writer_cb_to_dst.cpp",
        p.sender_core,
        tt::tt_metal::DataMovementConfig{
            .processor = tt::tt_metal::DataMovementProcessor::RISCV_1,
            .noc = tt::tt_metal::NOC::RISCV_1_default,
            .compile_args = writer_cta});

    // Resolve forwarding and append fabric connection args
    uint32_t link_idx = 0;
    if (!pick_forwarding_link_or_fail(src, dst, link_idx, p)) {
        return PerfPoint{};
    }

    std::vector<uint32_t> writer_rt = {
        (uint32_t)dst_buf->address(),  // 0: dst_base (receiver L1 offset)
        (uint32_t)p.mesh_id,           // 1: dst_mesh_id (logical)
        (uint32_t)p.dst_chip,          // 2: dst_dev_id  (logical)
        (uint32_t)rx_xy.x,             // 3: receiver_noc_x
        (uint32_t)rx_xy.y,             // 4: receiver_noc_y
        (uint32_t)gsem.address()       // 5: receiver L1 semaphore addr
    };

    // Pack the fabric-connection runtime args for the writer kernel.
    // This establishes the send path (routing/link identifiers) for fabric traffic.
    // The device kernel must unpack these in the same order via build_from_args(...).
    tt::tt_fabric::append_fabric_connection_rt_args(
        src, dst, /*link_idx=*/link_idx, sender_prog, p.sender_core, writer_rt);
    tt::tt_metal::SetRuntimeArgs(sender_prog, writer_k, p.sender_core, writer_rt);
    // -------------------------- end PROGRAM FACTORY --------------------------

    // --- Mesh trace capture & replay ---
    Dist::MeshWorkload sender_workload;
    Dist::MeshWorkload receiver_workload;
    sender_workload.add_program(Dist::MeshCoordinateRange(src_coord), std::move(sender_prog));
    receiver_workload.add_program(Dist::MeshCoordinateRange(dst_coord), std::move(receiver_prog));

    // 1) Warm-up each workload (receiver first so it's ready, then sender)
    Dist::EnqueueMeshWorkload(mcq, receiver_workload, /*blocking=*/false);
    Dist::EnqueueMeshWorkload(mcq, sender_workload, /*blocking=*/true);
    // 2) Capture p.trace_iters enqueues back-to-back
    auto trace_id = Dist::BeginTraceCapture(mesh.get(), mcq.id());
    for (uint32_t i = 0; i < p.trace_iters; ++i) {
        Dist::EnqueueMeshWorkload(mcq, receiver_workload, /*blocking=*/false);
        Dist::EnqueueMeshWorkload(mcq, sender_workload, /*blocking=*/false);
    }
    mesh->end_mesh_trace(mcq.id(), trace_id);
    // 3) Replay measured section
    auto t0 = std::chrono::steady_clock::now();
    mesh->replay_mesh_trace(mcq.id(), trace_id, /*blocking=*/false);
    Dist::Finish(mcq);
    auto t1 = std::chrono::steady_clock::now();
    mesh->release_mesh_trace(trace_id);

    // Read back (single shard) and verify
    std::vector<uint32_t> rx(n_words, 0u);
    Dist::ReadShard(mcq, rx, dst_buf, dst_coord, /*blocking=*/true);
    verify_payload_words(rx, tx);

    // Perf point
    const double e2e_sec_total = std::chrono::duration<double>(t1 - t0).count();
    const double e2e_sec = (p.trace_iters > 0) ? (e2e_sec_total / static_cast<double>(p.trace_iters)) : 0.0;
    const uint64_t bytes = static_cast<uint64_t>(p.tensor_bytes);
    const double GB = static_cast<double>(bytes) / 1e9;          // gigabytes
    const double GB_s = (e2e_sec > 0.0) ? (GB / e2e_sec) : 0.0;  // GB per second
    const double ms = e2e_sec * 1000.0;

    return PerfPoint{
        .bytes = bytes,
        .sec = e2e_sec,
        .ms = ms,
        .GB_s = GB_s,
    };
}

}  // namespace tt::tt_fabric::bench
