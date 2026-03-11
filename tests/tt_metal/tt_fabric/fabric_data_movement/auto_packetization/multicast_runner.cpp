// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Host-side test runner for raw-size multicast auto-packetization tests.
// Dispatches the multicast_tx_writer_raw device kernel which sends a large
// payload via the auto-packetizing fabric_multicast_noc_unicast_write wrapper
// to all chips in a rectangular sub-mesh, then validates that all data
// arrives correctly at each destination.

#include <gtest/gtest.h>
#include <algorithm>
#include <vector>

#include <tt-metalium/tt_metal.hpp>
#include "tests/tt_metal/tt_metal/common/multi_device_fixture.hpp"
#include "tests/tt_metal/tt_fabric/common/utils.hpp"
#include "tests/tt_metal/tt_fabric/fabric_data_movement/auto_packetization/test_common.hpp"
#include "tt_metal/fabric/fabric_context.hpp"
#include <tt-metalium/global_semaphore.hpp>
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/mesh_device.hpp>
#include <tt-metalium/mesh_device_view.hpp>
#include <distributed/mesh_device_view_impl.hpp>

namespace tt::tt_fabric::test {

namespace {

// Validate workload parameters
inline bool validate_workload_or_fail(const RawTestParams& p) {
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
            ADD_FAILURE() << "Data mismatch at word " << i << " (got 0x" << std::hex << rx[i]
                          << ", exp 0x" << tx[i] << std::dec << ")";
            return;
        }
    }
}

}  // anonymous namespace

// ----------------------------------- program -----------------------------------
void run_raw_multicast_write_test(tt::tt_metal::MeshDeviceFixtureBase* fixture, const RawTestParams& p) {
    const auto& cp = tt::tt_metal::MetalContext::instance().get_control_plane();
    namespace Dist = tt::tt_metal::distributed;

    // Check if fabric is 2D and create defines map
    const auto& fabric_context = cp.get_fabric_context();
    const bool is_2d_fabric = fabric_context.is_2D_routing_enabled();
    std::map<std::string, std::string> defines = {};
    if (is_2d_fabric) {
        defines["FABRIC_2D"] = "1";
    }

    tt::tt_fabric::FabricNodeId src{tt::tt_fabric::MeshId{p.mesh_id}, p.src_chip};
    tt::tt_fabric::FabricNodeId dst{tt::tt_fabric::MeshId{p.mesh_id}, p.dst_chip};

    ChipId src_phys = cp.get_physical_chip_id_from_fabric_node_id(src);
    ChipId dst_phys = cp.get_physical_chip_id_from_fabric_node_id(dst);

    tt::tt_metal::IDevice* src_dev = tt::tt_metal::detail::GetActiveDevice(src_phys);
    tt::tt_metal::IDevice* dst_dev = tt::tt_metal::detail::GetActiveDevice(dst_phys);
    if (!src_dev || !dst_dev) {
        ADD_FAILURE() << "Failed to find devices: src=" << src_phys << " dst=" << dst_phys;
        return;
    }

    if (!validate_workload_or_fail(p)) {
        return;
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

    // --- IO buffers ---
    Dist::DeviceLocalBufferConfig src_local{
        .page_size = p.tensor_bytes, .buffer_type = tt::tt_metal::BufferType::DRAM};
    Dist::DeviceLocalBufferConfig dst_local{
        .page_size = p.tensor_bytes, .buffer_type = tt::tt_metal::BufferType::DRAM};
    Dist::ReplicatedBufferConfig rcfg{.size = p.tensor_bytes};
    auto src_buf = Dist::MeshBuffer::create(rcfg, src_local, mesh.get());
    auto dst_buf = Dist::MeshBuffer::create(rcfg, dst_local, mesh.get());

    const size_t n_words = p.tensor_bytes / 4;
    auto tx = make_tx_pattern(n_words);
    std::vector<uint32_t> zeros(n_words, 0u);

    auto& mcq = mesh->mesh_command_queue();
    Dist::WriteShard(mcq, src_buf, tx, src_coord, /*blocking=*/true);
    Dist::WriteShard(mcq, dst_buf, zeros, dst_coord, /*blocking=*/true);

    // === Build the multicast receiver set from a rectangular sub-mesh ===
    std::vector<Dist::MeshCoordinate> dst_coords;
    const auto shape = view.shape();
    const uint32_t M = (p.mesh_rows ? p.mesh_rows : (uint32_t)shape[0]);
    const uint32_t N = (p.mesh_cols ? p.mesh_cols : (uint32_t)shape[1]);
    if (M == 0 || N == 0 || M > (uint32_t)shape[0] || N > (uint32_t)shape[1]) {
        ADD_FAILURE() << "Invalid mesh_rows/mesh_cols for physical mesh shape ("
                      << shape[0] << "x" << shape[1] << ")";
        return;
    }
    for (uint32_t r = 0; r < M; ++r) {
        for (uint32_t c = 0; c < N; ++c) {
            Dist::MeshCoordinate mc{(int)r, (int)c};
            auto* dev = view.impl().get_device(mc);
            if (!dev) {
                continue;
            }
            if (dev->id() == src_phys) {
                continue;  // exclude sender chip
            }
            dst_coords.push_back(mc);
        }
    }
    if (dst_coords.empty()) {
        ADD_FAILURE() << "Receiver set is empty (rectangle excludes all but sender).";
        return;
    }
    for (const auto& c : dst_coords) {
        Dist::WriteShard(mcq, dst_buf, zeros, c, /*blocking=*/true);
    }
    Dist::WriteShard(mcq, dst_buf, zeros, src_coord, /*blocking=*/true);

    // --- Receiver programs: one per destination chip ---
    std::vector<tt::tt_metal::Program> receiver_progs;
    receiver_progs.reserve(dst_coords.size());

    static std::optional<tt::tt_metal::GlobalSemaphore> gsem_done;
    if (!gsem_done) {
        tt::tt_metal::CoreRangeSet rx_core_one(tt::tt_metal::CoreRange(p.receiver_core, p.receiver_core));
        gsem_done = tt::tt_metal::CreateGlobalSemaphore(mesh.get(), rx_core_one, /*initial_value=*/0);
    }

    const std::string RX_KDIR = "tests/tt_metal/tt_fabric/fabric_data_movement/addrgen_write/kernels/";

    for (size_t i = 0; i < dst_coords.size(); ++i) {
        receiver_progs.emplace_back(tt::tt_metal::CreateProgram());
        auto rx_wait_k = tt::tt_metal::CreateKernel(
            receiver_progs.back(),
            RX_KDIR + "rx_addrgen.cpp",
            p.receiver_core,
            tt::tt_metal::DataMovementConfig{
                .processor = tt::tt_metal::DataMovementProcessor::RISCV_0,
                .noc = tt::tt_metal::NOC::RISCV_0_default,
                .defines = defines});
        // Each receiver waits for 1 atomic inc (sent per-direction after data)
        tt::tt_metal::SetRuntimeArgs(
            receiver_progs.back(), rx_wait_k, p.receiver_core, {gsem_done->address(), 1u});
    }

    // Ensure the same logical worker maps to the same physical XY across all receiver chips
    for (const auto& mc : dst_coords) {
        auto* dev_i = view.impl().get_device(mc);
        auto xy_i = dev_i->worker_core_from_logical_core(p.receiver_core);
        if (xy_i != rx_xy) {
            ADD_FAILURE() << "Receiver worker XY mismatch across chips";
            return;
        }
    }

    // --- Sender program ---
    tt::tt_metal::Program sender_prog = tt::tt_metal::CreateProgram();

    const std::string TX_KDIR =
        "tests/tt_metal/tt_fabric/fabric_data_movement/auto_packetization/kernels/";

    auto writer_k = tt::tt_metal::CreateKernel(
        sender_prog,
        TX_KDIR + "multicast_tx_writer_raw.cpp",
        p.sender_core,
        tt::tt_metal::DataMovementConfig{
            .processor = tt::tt_metal::DataMovementProcessor::RISCV_1,
            .noc = tt::tt_metal::NOC::RISCV_1_default,
            .defines = defines});

    // Multicast hop counts: bounding box of all receivers relative to sender
    uint16_t e_hops = 0, w_hops = 0, n_hops = 0, s_hops = 0;
    int src_r = src_coord[0], src_c = src_coord[1];
    int min_r = src_r, max_r = src_r, min_c = src_c, max_c = src_c;
    for (auto mc : dst_coords) {
        min_r = std::min(min_r, (int)mc[0]);
        max_r = std::max(max_r, (int)mc[0]);
        min_c = std::min(min_c, (int)mc[1]);
        max_c = std::max(max_c, (int)mc[1]);
    }
    if (max_c > src_c) { e_hops = (uint16_t)(max_c - src_c); }
    if (min_c < src_c) { w_hops = (uint16_t)(src_c - min_c); }
    if (max_r > src_r) { s_hops = (uint16_t)(max_r - src_r); }
    if (min_r < src_r) { n_hops = (uint16_t)(src_r - min_r); }

    // Direction bitmask
    const uint32_t dir_mask =
        (w_hops ? 1u : 0u) | (e_hops ? 2u : 0u) | (n_hops ? 4u : 0u) | (s_hops ? 8u : 0u);

    // Writer runtime args: pass address components (device kernel computes NOC addr)
    std::vector<uint32_t> writer_rt = {
        (uint32_t)src_buf->address(),                 // 0: src_l1_addr
        (uint32_t)p.tensor_bytes,                     // 1: total_size
        (uint32_t)dst_buf->address(),                 // 2: dst_base_addr
        (uint32_t)rx_xy.x,                            // 3: rx_noc_x
        (uint32_t)rx_xy.y,                            // 4: rx_noc_y
        (uint32_t)gsem_done->address(),               // 5: sem_l1_addr
        dir_mask,                                     // 6: dir_mask
    };

    // --- Per-direction fabric connections (W, E, N, S) ---
    auto coord_to_fabric_id = [&](Dist::MeshCoordinate mc) -> tt::tt_fabric::FabricNodeId {
        auto* dev = view.impl().get_device(mc);
        TT_FATAL(dev != nullptr, "No device at mesh coord ({}, {})", (int)mc[0], (int)mc[1]);
        ChipId phys = dev->id();
        return cp.get_fabric_node_id_from_physical_chip_id(phys);
    };
    auto src_fn = tt::tt_fabric::FabricNodeId{tt::tt_fabric::MeshId{p.mesh_id}, p.src_chip};

    // Representatives at the edge of the receiver rectangle
    Dist::MeshCoordinate rep_e = src_coord;
    if (e_hops) { rep_e[1] = max_c; }
    Dist::MeshCoordinate rep_w = src_coord;
    if (w_hops) { rep_w[1] = min_c; }
    Dist::MeshCoordinate rep_n = src_coord;
    if (n_hops) { rep_n[0] = min_r; }
    Dist::MeshCoordinate rep_s = src_coord;
    if (s_hops) { rep_s[0] = max_r; }

    auto pick_link = [&](Dist::MeshCoordinate mc, uint32_t& out_link_idx) {
        auto dst_fn = coord_to_fabric_id(std::move(mc));
        auto links = tt::tt_fabric::get_forwarding_link_indices(src_fn, dst_fn);
        if (links.empty()) {
            ADD_FAILURE() << "No forwarding link from src to representative";
            return false;
        }
        out_link_idx = links[0];
        return true;
    };

    uint32_t link_idx_w = 0, link_idx_e = 0, link_idx_n = 0, link_idx_s = 0;
    if (w_hops && !pick_link(rep_w, link_idx_w)) { return; }
    if (e_hops && !pick_link(rep_e, link_idx_e)) { return; }
    if (n_hops && !pick_link(rep_n, link_idx_n)) { return; }
    if (s_hops && !pick_link(rep_s, link_idx_s)) { return; }

    // Append fabric connection blocks in fixed order: W, E, N, S
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

    // Append hop counts
    writer_rt.push_back((uint32_t)e_hops);
    writer_rt.push_back((uint32_t)w_hops);
    writer_rt.push_back((uint32_t)n_hops);
    writer_rt.push_back((uint32_t)s_hops);

    tt::tt_metal::SetRuntimeArgs(sender_prog, writer_k, p.sender_core, writer_rt);

    // --- Build workloads ---
    auto sender_workload = Dist::MeshWorkload();
    auto receiver_workload = Dist::MeshWorkload();

    sender_workload.add_program(Dist::MeshCoordinateRange(src_coord), std::move(sender_prog));
    for (size_t i = 0; i < dst_coords.size(); ++i) {
        receiver_workload.add_program(
            Dist::MeshCoordinateRange(dst_coords[i]), std::move(receiver_progs[i]));
    }

    // Execute: receiver first (so it's ready), then sender
    Dist::EnqueueMeshWorkload(mcq, receiver_workload, /*blocking=*/false);
    Dist::EnqueueMeshWorkload(mcq, sender_workload, /*blocking=*/true);

    // --- Verify all destinations ---
    for (const auto& mc : dst_coords) {
        std::vector<uint32_t> rx(n_words, 0u);
        Dist::ReadShard(mcq, rx, dst_buf, mc, /*blocking=*/true);
        verify_payload_words(rx, tx);
    }
}

}  // namespace tt::tt_fabric::test
