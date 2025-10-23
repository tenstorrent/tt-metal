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

// // Resolve forwarding link and fail early if none found.
// inline bool pick_forwarding_link_or_fail(
//     const tt::tt_fabric::FabricNodeId& /*src*/,
//     const tt::tt_fabric::FabricNodeId& /*dst*/,
//     uint32_t& out_link_idx,
//     const PerfParams& p) {
//     auto links = tt::tt_fabric::get_forwarding_link_indices(
//         tt::tt_fabric::FabricNodeId{tt::tt_fabric::MeshId{p.mesh_id}, p.src_chip},
//         tt::tt_fabric::FabricNodeId{tt::tt_fabric::MeshId{p.mesh_id}, p.dst_chip});

//     if (links.empty()) {
//         ADD_FAILURE() << "No forwarding links from src(mesh=" << p.mesh_id << ",dev=" << p.src_chip
//                       << ") to dst(mesh=" << p.mesh_id << ",dev=" << p.dst_chip << ")";
//         return false;
//     }
//     out_link_idx = links[0];
//     return true;
// }

// Device lookup and basic existence check.
inline bool lookup_devices_or_fail(
    chip_id_t src_phys, chip_id_t dst_phys, tt::tt_metal::IDevice*& src_dev, tt::tt_metal::IDevice*& dst_dev) {
    src_dev = find_device_by_id(src_phys);
    dst_dev = find_device_by_id(dst_phys);
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
PerfPoint run_all_gather_once(HelpersFixture* fixture, const PerfParams& p) {
    const auto& cp = tt::tt_metal::MetalContext::instance().get_control_plane();

    tt::tt_fabric::FabricNodeId src{tt::tt_fabric::MeshId{p.mesh_id}, p.src_chip};
    // representative destination to seed the forwarding link
    tt::tt_fabric::FabricNodeId dst{tt::tt_fabric::MeshId{p.mesh_id}, p.dst_chip};

    chip_id_t src_phys = cp.get_physical_chip_id_from_fabric_node_id(src);
    chip_id_t dst_phys = cp.get_physical_chip_id_from_fabric_node_id(dst);

    tt::tt_metal::IDevice* src_dev = nullptr;
    tt::tt_metal::IDevice* dst_dev = nullptr;
    if (!lookup_devices_or_fail(src_phys, dst_phys, src_dev, dst_dev)) {
        return PerfPoint{};
    }

    if (!validate_workload_or_fail(p)) {
        return PerfPoint{};
    }

    tt::tt_metal::CoreCoord rx_xy = dst_dev->worker_core_from_logical_core(p.receiver_core);

    // --- IO buffers & initialization ---
    namespace Dist = tt::tt_metal::distributed;

    // Mesh + view + coords (we’ll need these before I/O)
    auto mesh = fixture->get_mesh_device();
    auto view = mesh->get_view();
    auto coord_of_phys = [&](chip_id_t phys) -> Dist::MeshCoordinate {
        for (const auto& c : Dist::MeshCoordinateRange(view.shape())) {
            if (view.get_device(c)->id() == phys) {
                return c;
            }
        }
        TT_FATAL(false, "Physical chip {} is not part of this MeshDevice", phys);
        return Dist::MeshCoordinate(0);
    };
    Dist::MeshCoordinate src_coord = coord_of_phys(src_phys);
    Dist::MeshCoordinate dst_coord = coord_of_phys(dst_phys);

    // MeshBuffer-based IO
    Dist::DeviceLocalBufferConfig src_local{.page_size = p.page_size, .buffer_type = tt::tt_metal::BufferType::DRAM};
    Dist::DeviceLocalBufferConfig dst_local{.page_size = p.page_size, .buffer_type = tt::tt_metal::BufferType::DRAM};
    Dist::ReplicatedBufferConfig rcfg{.size = p.tensor_bytes};
    auto src_buf = Dist::MeshBuffer::create(rcfg, src_local, mesh.get());
    auto dst_buf = Dist::MeshBuffer::create(rcfg, dst_local, mesh.get());

    const size_t n_words = p.tensor_bytes / 4;
    auto tx = make_tx_pattern(n_words);
    std::vector<uint32_t> zeros(n_words, 0u);

    // Blocking writes so data is resident before kernels run
    auto& mcq = mesh->mesh_command_queue();
    Dist::WriteShard(mcq, src_buf, tx, src_coord, /*blocking=*/true);
    Dist::WriteShard(mcq, dst_buf, zeros, dst_coord, /*blocking=*/true);
    // === Build the multicast receiver set from a rectangular sub-mesh (0,0) .. (rows-1, cols-1) ===
    std::vector<Dist::MeshCoordinate> dst_coords;
    const auto shape = view.shape();  // [rows, cols]
    const uint32_t M = (p.mesh_rows ? p.mesh_rows : (uint32_t)shape[0]);
    const uint32_t N = (p.mesh_cols ? p.mesh_cols : (uint32_t)shape[1]);
    if (M == 0 || N == 0 || M > (uint32_t)shape[0] || N > (uint32_t)shape[1]) {
        ADD_FAILURE() << "Invalid --mesh-rows/--mesh-cols for physical mesh shape (" << shape[0] << "x" << shape[1]
                      << ")";
        return PerfPoint{};
    }
    for (uint32_t r = 0; r < M; ++r) {
        for (uint32_t c = 0; c < N; ++c) {
            Dist::MeshCoordinate mc{(int)r, (int)c};
            auto dev = view.get_device(mc);
            if (!dev) {
                continue;
            }
            if (dev->id() == src_phys) {
                continue;  // exclude sender chip from fabric RX
            }
            dst_coords.push_back(mc);
        }
    }
    if (dst_coords.empty()) {
        ADD_FAILURE() << "Receiver set is empty (rectangle excludes all but sender).";
        return PerfPoint{};
    }
    for (auto c : dst_coords) {
        Dist::WriteShard(mcq, dst_buf, zeros, c, /*blocking=*/true);
    }
    Dist::WriteShard(mcq, dst_buf, zeros, src_coord, /*blocking=*/true);

    // ---------------------------- PROGRAM FACTORY ----------------------------
    /*
All gather bench — top-level flow:

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
    // Build RX programs: one per receiver chip
    std::vector<tt::tt_metal::Program> receiver_progs;
    receiver_progs.reserve(dst_coords.size());

    // One semaphore at a single logical core → same L1 offset on every chip in the MeshDevice
    static std::optional<tt::tt_metal::GlobalSemaphore> gsem_done;
    if (!gsem_done) {
        tt::tt_metal::CoreRangeSet rx_core_one(tt::tt_metal::CoreRange(p.receiver_core, p.receiver_core));
        gsem_done = tt::tt_metal::CreateGlobalSemaphore(mesh.get(), rx_core_one, /*initial_value=*/0);
    }

    constexpr const char* KDIR = "tests/tt_metal/tt_fabric/benchmark/collectives/all_gather/kernels/";

    for (size_t i = 0; i < dst_coords.size(); ++i) {
        receiver_progs.emplace_back(tt::tt_metal::CreateProgram());
        auto rx_wait_k = tt::tt_metal::CreateKernel(
            receiver_progs.back(),
            std::string(KDIR) + "all_gather_rx.cpp",
            p.receiver_core,
            tt::tt_metal::DataMovementConfig{
                .processor = tt::tt_metal::DataMovementProcessor::RISCV_0, .noc = tt::tt_metal::NOC::RISCV_0_default});
        tt::tt_metal::SetRuntimeArgs(receiver_progs.back(), rx_wait_k, p.receiver_core, {gsem_done->address(), 1u});
    }

    // Ensure the same logical worker maps to the same physical XY across all receiver chips
    for (auto mc : dst_coords) {
        auto dev_i = view.get_device(mc);
        auto xy_i = dev_i->worker_core_from_logical_core(p.receiver_core);
        if (xy_i != rx_xy) {
            ADD_FAILURE() << "Receiver worker XY mismatch across chips; need identical mapping of logical core "
                          << "(" << p.receiver_core.x << "," << p.receiver_core.y << ") on all chips.";
            return PerfPoint{};
        }
    }

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
        std::string(KDIR) + "all_gather_tx_reader_to_cb.cpp",
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
        std::string(KDIR) + "all_gather_tx_writer_cb_to_dst.cpp",
        p.sender_core,
        tt::tt_metal::DataMovementConfig{
            .processor = tt::tt_metal::DataMovementProcessor::RISCV_1,
            .noc = tt::tt_metal::NOC::RISCV_1_default,
            .compile_args = writer_cta});

    // Writer kernel RT args (base): dst_base, rx_x, rx_y, sem_l1
    std::vector<uint32_t> writer_rt = {
        (uint32_t)dst_buf->address(), (uint32_t)rx_xy.x, (uint32_t)rx_xy.y, (uint32_t)gsem_done->address()};

    tt::tt_metal::Program local_prog = tt::tt_metal::CreateProgram();

    // Recreate the same CB on the same core for this program
    {
        const uint32_t CB_ID = tt::CBIndex::c_0;
        auto cb_cfg_local = tt::tt_metal::CircularBufferConfig(8 * p.page_size, {{CB_ID, tt::DataFormat::Float16}})
                                .set_page_size(CB_ID, p.page_size);
        (void)tt::tt_metal::CreateCircularBuffer(local_prog, p.sender_core, cb_cfg_local);
    }

    // Reader (DRAM src_buf -> CB): same kernel/args as the sender's reader
    std::vector<uint32_t> local_reader_cta;
    tt::tt_metal::TensorAccessorArgs(*src_buf).append_to(local_reader_cta);
    local_reader_cta.push_back(1u /*SRC_IS_DRAM*/);
    local_reader_cta.push_back(NUM_PAGES);
    local_reader_cta.push_back(p.page_size);

    auto local_reader_k = tt::tt_metal::CreateKernel(
        local_prog,
        std::string(KDIR) + "all_gather_tx_reader_to_cb.cpp",
        p.sender_core,
        tt::tt_metal::DataMovementConfig{
            .processor = tt::tt_metal::DataMovementProcessor::RISCV_0,
            .noc = tt::tt_metal::NOC::RISCV_0_default,
            .compile_args = local_reader_cta});
    tt::tt_metal::SetRuntimeArgs(local_prog, local_reader_k, p.sender_core, {(uint32_t)src_buf->address()});

    // Local writer (CB -> dst_buf on THIS chip via NoC)
    std::vector<uint32_t> local_writer_cta;
    tt::tt_metal::TensorAccessorArgs(*dst_buf).append_to(local_writer_cta);
    local_writer_cta.push_back(NUM_PAGES);
    local_writer_cta.push_back(p.page_size);

    auto local_writer_k = tt::tt_metal::CreateKernel(
        local_prog,
        std::string(KDIR) + "all_gather_local_writer_cb_to_self.cpp",
        p.sender_core,
        tt::tt_metal::DataMovementConfig{
            .processor = tt::tt_metal::DataMovementProcessor::RISCV_1,
            .noc = tt::tt_metal::NOC::RISCV_0_default,
            .compile_args = local_writer_cta});
    tt::tt_metal::SetRuntimeArgs(local_prog, local_writer_k, p.sender_core, {(uint32_t)dst_buf->address()});

    // Wrap as a workload on the *source* device
    auto local_workload = Dist::MeshWorkload();
    local_workload.add_program(Dist::MeshCoordinateRange(src_coord), std::move(local_prog));

    // -------------------------- end PROGRAM FACTORY --------------------------

    // Phase A hops: bounding box of all receivers relative to sender
    uint16_t e_hops = 0, w_hops = 0, n_hops = 0, s_hops = 0;
    int src_r = src_coord[0], src_c = src_coord[1];
    int min_r = src_r, max_r = src_r, min_c = src_c, max_c = src_c;
    for (auto mc : dst_coords) {
        min_r = std::min(min_r, (int)mc[0]);
        max_r = std::max(max_r, (int)mc[0]);
        min_c = std::min(min_c, (int)mc[1]);
        max_c = std::max(max_c, (int)mc[1]);
    }
    if (max_c > src_c) {
        e_hops = (uint16_t)(max_c - src_c);
    }
    if (min_c < src_c) {
        w_hops = (uint16_t)(src_c - min_c);
    }
    if (max_r > src_r) {
        s_hops = (uint16_t)(max_r - src_r);
    }
    if (min_r < src_r) {
        n_hops = (uint16_t)(src_r - min_r);
    }

    // === Per-direction fabric connections (W,E,N,S) ===
    // For each active direction, choose a representative receiver coordinate in that direction
    // and compute a forwarding link from the source to that representative.
    // Map MeshDeviceView coord -> IDevice -> physical chip id -> FabricNodeId via control-plane.
    auto coord_to_fabric_id = [&](Dist::MeshCoordinate mc) -> tt::tt_fabric::FabricNodeId {
        auto dev = view.get_device(mc);
        TT_FATAL(dev != nullptr, "No device at mesh coord ({}, {})", (int)mc[0], (int)mc[1]);
        chip_id_t phys = dev->id();  // physical chip id
        // control-plane helper to convert physical chip id to FabricNodeId
        return cp.get_fabric_node_id_from_physical_chip_id(phys);
    };
    auto src_fn = tt::tt_fabric::FabricNodeId{tt::tt_fabric::MeshId{p.mesh_id}, p.src_chip};

    auto pick_link = [&](Dist::MeshCoordinate mc, uint32_t& out_link_idx) {
        auto dst_fn = coord_to_fabric_id(mc);
        auto links = tt::tt_fabric::get_forwarding_link_indices(src_fn, dst_fn);
        if (links.empty()) {
            ADD_FAILURE() << "No forwarding link from src(mesh=" << p.mesh_id << ",dev=" << p.src_chip
                          << ") to representative at (" << mc[0] << "," << mc[1] << ")";
            return false;
        }
        out_link_idx = links[0];
        return true;
    };

    // Representatives at the edge of the receiver rectangle
    Dist::MeshCoordinate rep_e = src_coord;
    if (e_hops) {
        rep_e[1] = max_c;
    }
    Dist::MeshCoordinate rep_w = src_coord;
    if (w_hops) {
        rep_w[1] = min_c;
    }
    Dist::MeshCoordinate rep_n = src_coord;
    if (n_hops) {
        rep_n[0] = min_r;
    }
    Dist::MeshCoordinate rep_s = src_coord;
    if (s_hops) {
        rep_s[0] = max_r;
    }

    uint32_t link_idx_w = 0, link_idx_e = 0, link_idx_n = 0, link_idx_s = 0;
    if (w_hops && !pick_link(rep_w, link_idx_w)) {
        return PerfPoint{};
    }
    if (e_hops && !pick_link(rep_e, link_idx_e)) {
        return PerfPoint{};
    }
    if (n_hops && !pick_link(rep_n, link_idx_n)) {
        return PerfPoint{};
    }
    if (s_hops && !pick_link(rep_s, link_idx_s)) {
        return PerfPoint{};
    }

    // Direction bitmask encoded into RT args so the kernel knows how many connections to parse.
    // bit0=W, bit1=E, bit2=N, bit3=S
    const uint32_t dir_mask = (w_hops ? 1u : 0u) | (e_hops ? 2u : 0u) | (n_hops ? 4u : 0u) | (s_hops ? 8u : 0u);
    writer_rt.push_back(dir_mask);

    // Append the fabric connection blocks in fixed order: W, E, N, S (only if active)
    if (w_hops) {
        tt::tt_fabric::append_fabric_connection_rt_args(
            src_fn, coord_to_fabric_id(rep_w), link_idx_w, sender_prog, p.sender_core, writer_rt);
    }
    if (e_hops) {
        tt::tt_fabric::append_fabric_connection_rt_args(
            src_fn, coord_to_fabric_id(rep_e), link_idx_e, sender_prog, p.sender_core, writer_rt);
    }
    if (n_hops) {
        tt::tt_fabric::append_fabric_connection_rt_args(
            src_fn, coord_to_fabric_id(rep_n), link_idx_n, sender_prog, p.sender_core, writer_rt);
    }
    if (s_hops) {
        tt::tt_fabric::append_fabric_connection_rt_args(
            src_fn, coord_to_fabric_id(rep_s), link_idx_s, sender_prog, p.sender_core, writer_rt);
    }

    // Append hops AFTER fabric-connection args so the kernel’s parsing isn’t disturbed.
    writer_rt.push_back((uint32_t)e_hops);
    writer_rt.push_back((uint32_t)w_hops);
    writer_rt.push_back((uint32_t)n_hops);
    writer_rt.push_back((uint32_t)s_hops);

    tt::tt_metal::SetRuntimeArgs(sender_prog, writer_k, p.sender_core, writer_rt);

    // Build two workloads so the RX wait kernel is always running before TX starts.
    auto sender_workload = Dist::MeshWorkload();
    auto receiver_workload = Dist::MeshWorkload();

    sender_workload.add_program(Dist::MeshCoordinateRange(src_coord), std::move(sender_prog));
    for (size_t i = 0; i < dst_coords.size(); ++i) {
        receiver_workload.add_program(Dist::MeshCoordinateRange(dst_coords[i]), std::move(receiver_progs[i]));
    }

    // 1) Warm-up outside capture: launch RX first so it's waiting, then TX
    Dist::EnqueueMeshWorkload(mcq, receiver_workload, /*blocking=*/false);
    Dist::EnqueueMeshWorkload(mcq, sender_workload, /*blocking=*/true);
    // 2) Capture p.trace_iters enqueues back-to-back
    auto trace_id = Dist::BeginTraceCapture(mesh.get(), mcq.id());
    for (uint32_t i = 0; i < p.trace_iters; ++i) {
        // Maintain RX→TX order inside the capture as well
        Dist::EnqueueMeshWorkload(mcq, receiver_workload, /*blocking=*/false);
        Dist::EnqueueMeshWorkload(mcq, sender_workload, /*blocking=*/false);
    }
    Dist::EndTraceCapture(mesh.get(), mcq.id(), trace_id);
    // 3) Replay measured section
    auto t0 = std::chrono::steady_clock::now();
    Dist::ReplayTrace(mesh.get(), mcq.id(), trace_id, /*blocking=*/false);
    Dist::Finish(mcq);
    auto t1 = std::chrono::steady_clock::now();
    Dist::ReleaseTrace(mesh.get(), trace_id);

    Dist::EnqueueMeshWorkload(mcq, local_workload, /*blocking=*/true);

    // Read back and verify
    std::vector<uint32_t> rx_self(n_words, 0u);
    for (auto mc : dst_coords) {
        std::vector<uint32_t> rx(n_words, 0u);
        Dist::ReadShard(mcq, rx, dst_buf, mc, /*blocking=*/true);
        verify_payload_words(rx, tx);
    }
    Dist::ReadShard(mcq, rx_self, dst_buf, src_coord, /*blocking=*/true);
    verify_payload_words(rx_self, tx);

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

// --- Shim so perf_helpers can drive this binary without knowing about all_gather.
namespace tt::tt_fabric::bench {
PerfPoint run_unicast_once(HelpersFixture* fixture, const PerfParams& p) { return run_all_gather_once(fixture, p); }
}  // namespace tt::tt_fabric::bench
