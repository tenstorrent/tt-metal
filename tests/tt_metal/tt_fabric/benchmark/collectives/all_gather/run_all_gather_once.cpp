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
#include <iostream>

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
//     const tt::tt_fabric::FabricNodeId& src,
//     const tt::tt_fabric::FabricNodeId& dst,
//     uint32_t& out_link_idx,
//     const PerfParams& p) {
//     auto links = tt::tt_fabric::get_forwarding_link_indices(src, dst);

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
    tt::tt_metal::CoreCoord tx_xy = src_dev->worker_core_from_logical_core(p.sender_core);

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

    // Writer kernel RT args (short): dst_base, rx_x, rx_y, sem_l1
    std::vector<uint32_t> writer_rt = {
        (uint32_t)dst_buf->address(), (uint32_t)rx_xy.x, (uint32_t)rx_xy.y, (uint32_t)gsem_done->address()};

    std::cerr << "[host] rx_xy=(" << (int)rx_xy.x << "," << (int)rx_xy.y << ") sem_l1=0x" << std::hex
              << gsem_done->address() << std::dec << "\n";

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

    // ---- Choose representative neighbors to discover per-leg link indices (AFTER hop extents) ----
    auto in_mesh = [&](int r, int c) -> bool {
        return 0 <= r && r < (int)shape[0] && 0 <= c && c < (int)shape[1] && (view.get_device({r, c}) != nullptr);
    };

    auto dev_id_at = [&](int r, int c) -> uint16_t {
        auto* d = view.get_device({r, c});
        TT_FATAL(d != nullptr, "No device at mesh coord [{},{}]", r, c);
        return static_cast<uint16_t>(d->id());
    };
    auto pick_adjacent = [&](bool use, int r, int c, const char* dir) -> uint16_t {
        if (!use) {
            return 0;
        }
        TT_FATAL(in_mesh(r, c), "Start-node out of mesh for {} leg at [{},{}]", dir, r, c);
        uint16_t dev = dev_id_at(r, c);
        TT_FATAL(
            dev != static_cast<uint16_t>(src_phys),
            "Start-node for {} leg equals source dev ({}). This will hang.",
            dir,
            dev);
        return dev;
    };

    auto fabric_id_of = [&](Dist::MeshCoordinate mc) -> tt::tt_fabric::FabricNodeId {
        auto* d = view.get_device(mc);
        TT_FATAL(d != nullptr, "No device at mesh coord [{},{}]", (int)mc[0], (int)mc[1]);
        chip_id_t phys = d->id();  // on this rig, physical id == fabric chip id
        return tt::tt_fabric::FabricNodeId{tt::tt_fabric::MeshId{p.mesh_id}, phys};
    };

    const bool want_W = (w_hops > 0);
    const bool want_E = (e_hops > 0);
    const bool want_N = (n_hops > 0);
    const bool want_S = (s_hops > 0);

    // Debug: basic topology snapshot
    std::cerr << "[host] mesh=" << shape[0] << "x" << shape[1] << " src_phys=" << src_phys << " dst_phys=" << dst_phys
              << " src_coord=(" << src_coord[0] << "," << src_coord[1] << ")"
              << " dst_coord=(" << dst_coord[0] << "," << dst_coord[1] << ")"
              << " rx_xy=(" << (int)rx_xy.x << "," << (int)rx_xy.y << ")"
              << " tx_xy=(" << (int)tx_xy.x << "," << (int)tx_xy.y << ")\n";

    auto src_fid = tt::tt_fabric::FabricNodeId{tt::tt_fabric::MeshId{p.mesh_id}, src_phys};
    std::optional<tt::tt_fabric::FabricNodeId> dstW, dstE, dstN, dstS;
    constexpr uint32_t ROUTING_PLANE_ID = 0;  // single plane in this run

    if (want_W) {
        TT_FATAL(in_mesh(src_r, src_c - 1), "No west neighbor for source.");
        dstW = fabric_id_of({src_r, src_c - 1});
    }

    if (want_E) {
        TT_FATAL(in_mesh(src_r, src_c + 1), "No east neighbor for source.");
        dstE = fabric_id_of({src_r, src_c + 1});
    }

    if (want_N) {
        TT_FATAL(in_mesh(src_r - 1, src_c), "No north neighbor for source.");
        dstN = fabric_id_of({src_r - 1, src_c});
    }

    if (want_S) {
        TT_FATAL(in_mesh(src_r + 1, src_c), "No south neighbor for source.");
        dstS = fabric_id_of({src_r + 1, src_c});
    }

    // Fallback for unused legs: duplicate any available leg so kernel arg layout stays fixed (W,E,N,S).
    const int have_any = (want_W ? 1 : 0) + (want_E ? 1 : 0) + (want_N ? 1 : 0) + (want_S ? 1 : 0);
    TT_FATAL(have_any > 0, "No active multicast legs detected.");
    auto pick_any_dst = [&]() -> tt::tt_fabric::FabricNodeId {
        if (want_E) {
            return *dstE;
        }
        if (want_W) {
            return *dstW;
        }
        if (want_N) {
            return *dstN;
        }
        return *dstS;
    };
    tt::tt_fabric::FabricNodeId any_dst = pick_any_dst();

    // Append FOUR connections in fixed order: W, E, N, S (so kernel can parse uniformly).
    tt::tt_fabric::append_fabric_connection_rt_args(
        src_fid, (want_W ? *dstW : any_dst), ROUTING_PLANE_ID, sender_prog, p.sender_core, writer_rt);
    tt::tt_fabric::append_fabric_connection_rt_args(
        src_fid, (want_E ? *dstE : any_dst), ROUTING_PLANE_ID, sender_prog, p.sender_core, writer_rt);
    tt::tt_fabric::append_fabric_connection_rt_args(
        src_fid, (want_N ? *dstN : any_dst), ROUTING_PLANE_ID, sender_prog, p.sender_core, writer_rt);
    tt::tt_fabric::append_fabric_connection_rt_args(
        src_fid, (want_S ? *dstS : any_dst), ROUTING_PLANE_ID, sender_prog, p.sender_core, writer_rt);

    uint16_t dev_W = pick_adjacent(want_W, src_coord[0], src_coord[1] - 1, "W");
    uint16_t dev_E = pick_adjacent(want_E, src_coord[0], src_coord[1] + 1, "E");
    uint16_t dev_N = pick_adjacent(want_N, src_coord[0] - 1, src_coord[1], "N");
    uint16_t dev_S = pick_adjacent(want_S, src_coord[0] + 1, src_coord[1], "S");
    auto mesh_or_zero = [&](bool use) -> uint16_t { return use ? static_cast<uint16_t>(p.mesh_id) : 0; };
    uint16_t mesh_W = mesh_or_zero(want_W);
    uint16_t mesh_E = mesh_or_zero(want_E);
    uint16_t mesh_N = mesh_or_zero(want_N);
    uint16_t mesh_S = mesh_or_zero(want_S);

    // Push (dev,mesh) pairs for W,E,N,S — kernel expects this order
    writer_rt.push_back((uint32_t)dev_W);
    writer_rt.push_back((uint32_t)mesh_W);
    writer_rt.push_back((uint32_t)dev_E);
    writer_rt.push_back((uint32_t)mesh_E);
    writer_rt.push_back((uint32_t)dev_N);
    writer_rt.push_back((uint32_t)mesh_N);
    writer_rt.push_back((uint32_t)dev_S);
    writer_rt.push_back((uint32_t)mesh_S);

    std::cerr << "[host] start_nodes (mesh,dev) W/E/N/S=(" << mesh_W << "," << dev_W << ")/(" << mesh_E << "," << dev_E
              << ")/(" << mesh_N << "," << dev_N << ")/(" << mesh_S << "," << dev_S << ")\n";

    // Pack hop extents and leg mask after connections + first-hop ids.
    writer_rt.push_back((uint32_t)e_hops);
    writer_rt.push_back((uint32_t)w_hops);
    writer_rt.push_back((uint32_t)n_hops);
    writer_rt.push_back((uint32_t)s_hops);
    uint32_t leg_mask = (want_W ? 1u : 0u) | (want_E ? 2u : 0u) | (want_N ? 4u : 0u) | (want_S ? 8u : 0u);
    writer_rt.push_back(leg_mask);
    std::cerr << "[host] hops E/W/N/S=" << e_hops << "/" << w_hops << "/" << n_hops << "/" << s_hops
              << " leg_mask=" << leg_mask << " receivers=" << dst_coords.size() << "\n";
    tt::tt_metal::SetRuntimeArgs(sender_prog, writer_k, p.sender_core, writer_rt);

    // Build two workloads so the RX wait kernel is always running before TX starts.
    auto sender_workload = Dist::MeshWorkload();
    auto receiver_workload = Dist::MeshWorkload();

    sender_workload.add_program(Dist::MeshCoordinateRange(src_coord), std::move(sender_prog));
    for (size_t i = 0; i < dst_coords.size(); ++i) {
        receiver_workload.add_program(Dist::MeshCoordinateRange(dst_coords[i]), std::move(receiver_progs[i]));
    }

    // 1) Warm-up outside capture: launch RX first so it's waiting, then TX
    std::cerr << "[host] enqueue RX (warmup)\n";
    Dist::EnqueueMeshWorkload(mcq, receiver_workload, /*blocking=*/false);
    std::cerr << "[host] enqueue TX (warmup)\n";
    Dist::EnqueueMeshWorkload(mcq, sender_workload, /*blocking=*/false);
    std::cerr << "[host] finish warmup\n";
    Dist::Finish(mcq);  // if we hang here, TX/RX warmup didn’t complete
    std::cerr << "[host] warmup finished\n";
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
