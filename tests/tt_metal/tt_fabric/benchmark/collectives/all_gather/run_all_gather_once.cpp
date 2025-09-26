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

    // Phase 2: N-rank all-gather — run on the whole mesh view

    namespace Dist = tt::tt_metal::distributed;
    auto mesh = fixture->get_mesh_device();
    auto view = mesh->get_view();
    const uint32_t num_ranks = static_cast<uint32_t>(view.shape()[0] * view.shape()[1]);

    if (!validate_workload_or_fail(p)) {
        return PerfPoint{};
    }

    // Per-device objects
    constexpr const char* KDIR = "tests/tt_metal/tt_fabric/benchmark/collectives/all_gather/kernels/";
    struct RankCtx {
        tt::tt_metal::IDevice* dev{};
        std::shared_ptr<tt::tt_metal::Buffer> src_buf;
        std::shared_ptr<tt::tt_metal::Buffer> dst_buf;  // size S * num_ranks
        tt::tt_metal::CommandQueue* cq{};
        tt::tt_metal::Program rx_prog;
        tt::tt_metal::Program tx_prog;
        tt::tt_metal::KernelHandle rx_k{};
        tt::tt_metal::KernelHandle reader_k{};
        tt::tt_metal::KernelHandle writer_k{};
        std::optional<tt::tt_metal::GlobalSemaphore> gsem;
        tt::tt_metal::CoreCoord tx_xy{};
        tt::tt_metal::CoreCoord rx_xy{};
        std::optional<tt::tt_fabric::FabricNodeId> node;
        chip_id_t phys{};
    };
    std::vector<RankCtx> ranks;
    ranks.reserve(num_ranks);

    auto coord_of_phys = [&](chip_id_t phys) -> Dist::MeshCoordinate {
        for (const auto& c : Dist::MeshCoordinateRange(view.shape())) {
            if (view.get_device(c)->id() == phys) {
                return c;
            }
        }
        TT_FATAL(false, "Physical chip {} is not part of this MeshDevice", phys);
        return Dist::MeshCoordinate(0);
    };

    // Build per-rank context, buffers, patterns, receiver(s)
    const uint32_t S = p.tensor_bytes;
    const uint32_t NUM_PAGES = (S + p.page_size - 1) / p.page_size;
    const size_t n_words = S / 4;
    std::vector<std::vector<uint32_t>> tx_patterns;
    tx_patterns.reserve(num_ranks);

    // Enumerate ranks in mesh order to define rank indices
    for (const auto& c : Dist::MeshCoordinateRange(view.shape())) {
        RankCtx r{};
        r.dev = view.get_device(c);
        r.cq = &r.dev->command_queue();
        r.phys = r.dev->id();
        const std::uint32_t rank_idx = static_cast<std::uint32_t>(ranks.size());
        r.node = tt::tt_fabric::FabricNodeId{tt::tt_fabric::MeshId{p.mesh_id}, rank_idx};
        r.tx_xy = r.dev->worker_core_from_logical_core(p.sender_core);
        r.rx_xy = r.dev->worker_core_from_logical_core(p.receiver_core);

        // Buffers: src=S, dst=S*num_ranks
        tt::tt_metal::BufferConfig src_cfg{
            .device = r.dev, .size = S, .page_size = p.page_size, .buffer_type = tt::tt_metal::BufferType::DRAM};
        tt::tt_metal::BufferConfig dst_cfg{
            .device = r.dev,
            .size = S * num_ranks,
            .page_size = p.page_size,
            .buffer_type = p.use_dram_dst ? tt::tt_metal::BufferType::DRAM : tt::tt_metal::BufferType::L1};
        r.src_buf = tt::tt_metal::CreateBuffer(src_cfg);
        r.dst_buf = tt::tt_metal::CreateBuffer(dst_cfg);

        // Unique TX pattern per rank (rank id = ranks.size())
        std::vector<uint32_t> tx(n_words);
        const uint32_t tag = 0xA5A50000u + static_cast<uint32_t>(ranks.size()) * 0x101;  // rank-stamped
        for (size_t i = 0; i < n_words; ++i) {
            tx[i] = tag + static_cast<uint32_t>(i);
        }
        tx_patterns.push_back(tx);
        std::vector<uint32_t> zeros(n_words, 0u);
        tt::tt_metal::EnqueueWriteBuffer(*r.cq, *r.src_buf, tx_patterns.back(), /*blocking=*/true);
        // Zero the whole concat buffer once
        std::vector<uint32_t> zeros_full((static_cast<size_t>(S) * num_ranks) / 4, 0u);
        tt::tt_metal::EnqueueWriteBuffer(*r.cq, *r.dst_buf, zeros_full, /*blocking=*/true);

        // One receiver program per device; expected = num_ranks
        r.rx_prog = tt::tt_metal::CreateProgram();
        r.gsem = tt::tt_metal::CreateGlobalSemaphore(
            r.dev,
            r.dev->worker_cores(tt::tt_metal::HalProgrammableCoreType::TENSIX, tt::tt_metal::SubDeviceId{0}),
            /*initial_value=*/0,
            tt::tt_metal::BufferType::L1);

        r.rx_k = tt::tt_metal::CreateKernel(
            r.rx_prog,
            std::string(KDIR) + "all_gather_rx.cpp",
            p.receiver_core,
            tt::tt_metal::DataMovementConfig{
                .processor = tt::tt_metal::DataMovementProcessor::RISCV_0, .noc = tt::tt_metal::NOC::RISCV_0_default});
        tt::tt_metal::SetRuntimeArgs(r.rx_prog, r.rx_k, p.receiver_core, {r.gsem->address(), num_ranks});
        ranks.push_back(std::move(r));
    }

    // ---------------------------- PROGRAM FACTORY ----------------------------

    // Build sender programs on every rank
    for (uint32_t rank = 0; rank < num_ranks; ++rank) {
        auto& R = ranks[rank];
        R.tx_prog = tt::tt_metal::CreateProgram();

        const uint32_t CB_ID = tt::CBIndex::c_0;
        auto cb_cfg = tt::tt_metal::CircularBufferConfig(8 * p.page_size, {{CB_ID, tt::DataFormat::Float16}})
                          .set_page_size(CB_ID, p.page_size);
        (void)tt::tt_metal::CreateCircularBuffer(R.tx_prog, p.sender_core, cb_cfg);

        // Reader (DRAM -> CB)
        std::vector<uint32_t> reader_cta;
        tt::tt_metal::TensorAccessorArgs(*R.src_buf).append_to(reader_cta);
        reader_cta.push_back(1u /*SRC_IS_DRAM*/);
        reader_cta.push_back(NUM_PAGES);
        reader_cta.push_back(p.page_size);
        R.reader_k = tt::tt_metal::CreateKernel(
            R.tx_prog,
            std::string(KDIR) + "all_gather_tx_reader_to_cb.cpp",
            p.sender_core,
            tt::tt_metal::DataMovementConfig{
                .processor = tt::tt_metal::DataMovementProcessor::RISCV_0,
                .noc = tt::tt_metal::NOC::RISCV_0_default,
                .compile_args = reader_cta});
        tt::tt_metal::SetRuntimeArgs(R.tx_prog, R.reader_k, p.sender_core, {(uint32_t)R.src_buf->address()});

        // Writer (CB -> Fabric -> every dst rank at the same slot)
        std::vector<uint32_t> writer_cta;
        tt::tt_metal::TensorAccessorArgs(*R.dst_buf).append_to(writer_cta);  // accessor for page geometry
        writer_cta.push_back(NUM_PAGES);
        writer_cta.push_back(p.page_size);
        R.writer_k = tt::tt_metal::CreateKernel(
            R.tx_prog,
            std::string(KDIR) + "all_gather_tx_writer_cb_to_dst.cpp",
            p.sender_core,
            tt::tt_metal::DataMovementConfig{
                .processor = tt::tt_metal::DataMovementProcessor::RISCV_1,
                .noc = tt::tt_metal::NOC::RISCV_1_default,
                .compile_args = writer_cta});

        // Build writer RT args:
        //   dst_base      = start of concat buffer (all ranks identical geometry)
        //   dst_mesh/dev  = this rank's logical ids (unused for mcast route but kept)
        //   rank_offset   = rank * S
        std::vector<uint32_t> writer_rt = {
            (uint32_t)ranks[rank].dst_buf->address(),  // 0 dst_base (concat start)
            (uint32_t)p.mesh_id,                       // 1 dst_mesh_id (unused in mcast temp API)
            0u,                                        // 2 dst_dev_id (unused here)
            (uint32_t)(rank * S)                       // 3 rank_offset_bytes
        };

        // Route/link for payload (same as Phase A helper)
        // Pick any valid forwarding link from this sender to any other device (API needs one).
        // We use the first link between (this rank) and itself as placeholder; HW uses mcast route set in kernel.
        uint32_t link_idx = 0;
        const auto src_node = ranks[rank].node.value();
        const auto dst_node = ranks[(rank + 1) % num_ranks].node.value();  // ensure src != dst
        tt::tt_fabric::append_fabric_connection_rt_args(
            src_node, dst_node, link_idx, R.tx_prog, p.sender_core, writer_rt);

        // 2D hops as before (keep simple: east/west only OK; full 2D route will be wired by fabric mcast API)
        uint16_t e_hops = 0, w_hops = 0, n_hops = 0, s_hops = 0;
        writer_rt.push_back((uint32_t)e_hops);
        writer_rt.push_back((uint32_t)w_hops);
        writer_rt.push_back((uint32_t)n_hops);
        writer_rt.push_back((uint32_t)s_hops);

        // Completion fan-out: num_ranks then triples (rx_x, rx_y, sem_l1) for every receiver
        writer_rt.push_back(num_ranks);
        for (uint32_t r = 0; r < num_ranks; ++r) {
            writer_rt.push_back((uint32_t)ranks[r].rx_xy.x);
            writer_rt.push_back((uint32_t)ranks[r].rx_xy.y);
            writer_rt.push_back((uint32_t)ranks[r].gsem->address());
        }
        tt::tt_metal::SetRuntimeArgs(R.tx_prog, R.writer_k, p.sender_core, writer_rt);
    }

    // Make two workloads: one RX-only, one TX-only (no overlapping ranges within each)
    auto mesh_workload_rx = Dist::CreateMeshWorkload();
    auto mesh_workload_tx = Dist::CreateMeshWorkload();
    for (auto& R : ranks) {
        Dist::AddProgramToMeshWorkload(
            mesh_workload_rx, std::move(R.rx_prog), Dist::MeshCoordinateRange(coord_of_phys(R.phys)));
    }
    for (auto& R : ranks) {
        Dist::AddProgramToMeshWorkload(
            mesh_workload_tx, std::move(R.tx_prog), Dist::MeshCoordinateRange(coord_of_phys(R.phys)));
    }

    auto& mcq = mesh->mesh_command_queue();
    // 1) Warm-up outside capture: arm receivers, then senders
    Dist::EnqueueMeshWorkload(mcq, mesh_workload_rx, /*blocking=*/false);
    Dist::EnqueueMeshWorkload(mcq, mesh_workload_tx, /*blocking=*/true);
    // 2) Capture p.trace_iters enqueues back-to-back
    auto trace_id = Dist::BeginTraceCapture(mesh.get(), mcq.id());
    for (uint32_t i = 0; i < p.trace_iters; ++i) {
        // capture both enqueues back-to-back
        Dist::EnqueueMeshWorkload(mcq, mesh_workload_rx, /*blocking=*/false);
        Dist::EnqueueMeshWorkload(mcq, mesh_workload_tx, /*blocking=*/false);
    }
    Dist::EndTraceCapture(mesh.get(), mcq.id(), trace_id);
    // 3) Replay measured section
    auto t0 = std::chrono::steady_clock::now();
    Dist::ReplayTrace(mesh.get(), mcq.id(), trace_id, /*blocking=*/false);
    Dist::Finish(mcq);
    auto t1 = std::chrono::steady_clock::now();
    Dist::ReleaseTrace(mesh.get(), trace_id);

    // Read back and verify concat layout on every rank: [rank0 | rank1 | ...]
    for (uint32_t r = 0; r < num_ranks; ++r) {
        auto& R = ranks[r];
        tt::tt_metal::Finish(*R.cq);
        std::vector<uint32_t> full;
        tt::tt_metal::EnqueueReadBuffer(*R.cq, *R.dst_buf, full, /*blocking=*/true);
        for (uint32_t k = 0; k < num_ranks; ++k) {
            const uint32_t off_words = (k * S) / 4;
            std::vector<uint32_t> chunk(full.begin() + off_words, full.begin() + off_words + n_words);
            // compare with rank-k pattern
            verify_payload_words(chunk, tx_patterns[k]);
        }
    }

    // Perf point
    const double e2e_sec_total = std::chrono::duration<double>(t1 - t0).count();
    const double e2e_sec = (p.trace_iters > 0) ? (e2e_sec_total / static_cast<double>(p.trace_iters)) : 0.0;
    // Aggregate payload per rank per iteration = S sent once and received by all.
    const uint64_t bytes = static_cast<uint64_t>(S) * num_ranks;
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
