// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Host-side test runner for raw-size multicast auto-packetization silicon tests.
// Uses BaseFabricFixture (per-chip MeshDevice) pattern -- NOT MeshDeviceFixtureBase.
// Dispatches the appropriate multicast device kernel based on AutoPacketFamily,
// writes test data directly to L1, and validates byte-for-byte correctness
// on ALL receiver chips.

#include <gtest/gtest.h>
#include <algorithm>
#include <vector>

#include <tt-metalium/tt_metal.hpp>
#include <tt-metalium/global_semaphore.hpp>
#include <tt-metalium/experimental/fabric/fabric.hpp>
#include <tt-metalium/mesh_device.hpp>

#include "tests/tt_metal/tt_fabric/common/fabric_fixture.hpp"
#include "tt_metal/fabric/fabric_context.hpp"
#include <tt-metalium/experimental/fabric/mesh_graph.hpp>
#include "tests/tt_metal/tt_fabric/fabric_data_movement/auto_packetization/test_common.hpp"

namespace tt::tt_fabric::test {

using fabric_router_tests::BaseFabricFixture;

namespace {

// Generate deterministic TX pattern: 0xA5A50000 + i
inline std::vector<uint32_t> make_tx_pattern(size_t n_words) {
    std::vector<uint32_t> tx(n_words);
    for (size_t i = 0; i < n_words; ++i) {
        tx[i] = 0xA5A50000u + static_cast<uint32_t>(i);
    }
    return tx;
}

// Validate RX payload equals TX payload word-by-word.
inline void verify_payload_words(
    const std::vector<uint32_t>& rx,
    const std::vector<uint32_t>& tx,
    size_t word_offset = 0,
    size_t n_words = 0) {
    size_t count = (n_words > 0) ? n_words : tx.size();
    for (size_t i = 0; i < count; ++i) {
        if (rx[i + word_offset] != tx[i]) {
            ADD_FAILURE() << "Data mismatch at word " << i << " (offset " << word_offset
                          << "): got 0x" << std::hex << rx[i + word_offset]
                          << ", exp 0x" << tx[i] << std::dec;
            return;
        }
    }
}

}  // anonymous namespace

// Main multicast runner: dispatches the correct device kernel for the given family
// and verifies data correctness on silicon across all receiver chips.
void run_raw_multicast_write_test(BaseFabricFixture* fixture, const RawTestParams& p) {
    const auto& cp = tt::tt_metal::MetalContext::instance().get_control_plane();

    // Check if fabric is 2D
    const auto& fabric_context = cp.get_fabric_context();
    const bool is_2d_fabric = fabric_context.is_2D_routing_enabled();
    std::map<std::string, std::string> defines;
    if (is_2d_fabric) {
        defines["FABRIC_2D"] = "1";
    }

    tt::tt_fabric::FabricNodeId src{tt::tt_fabric::MeshId{p.mesh_id}, p.src_chip};

    ChipId src_phys = cp.get_physical_chip_id_from_fabric_node_id(src);

    // Get per-chip MeshDevice objects from BaseFabricFixture
    auto src_mesh = fixture->get_device(src_phys);
    auto* src_dev = src_mesh->get_devices()[0];

    // Collect all receiver devices (all devices except source)
    const auto& all_devices = fixture->get_devices();
    struct ReceiverInfo {
        ChipId phys_id;
        std::shared_ptr<tt::tt_metal::distributed::MeshDevice> mesh;
        tt::tt_metal::IDevice* dev;
        tt::tt_fabric::FabricNodeId fabric_node;
    };
    std::vector<ReceiverInfo> receivers;
    for (const auto& dmesh : all_devices) {
        auto* dev = dmesh->get_devices()[0];
        if (dev->id() == src_phys) {
            continue;
        }
        auto fn = cp.get_fabric_node_id_from_physical_chip_id(dev->id());
        receivers.push_back(ReceiverInfo{dev->id(), dmesh, dev, fn});
    }
    if (receivers.empty()) {
        ADD_FAILURE() << "No receiver devices found (need at least 2 devices for multicast)";
        return;
    }

    // Resolve receiver NOC coordinates (use first receiver as reference; verify all match)
    tt::tt_metal::CoreCoord rx_xy = receivers[0].dev->worker_core_from_logical_core(p.receiver_core);
    for (const auto& r : receivers) {
        auto xy_i = r.dev->worker_core_from_logical_core(p.receiver_core);
        if (xy_i != rx_xy) {
            ADD_FAILURE() << "Receiver worker XY mismatch across chips";
            return;
        }
    }

    // Get L1 memory layout from worker mem map
    const auto topology = fabric_context.get_fabric_topology();
    auto src_mem_map = BaseFabricFixture::generate_worker_mem_map(src_mesh, topology);

    const uint32_t src_l1_addr = src_mem_map.source_l1_buffer_address;
    // Use dst mem map from first receiver (all should have same layout)
    auto dst_mem_map = BaseFabricFixture::generate_worker_mem_map(receivers[0].mesh, topology);
    const uint32_t dst_l1_addr = dst_mem_map.target_address;

    // Validate payload fits in L1 data space
    constexpr uint32_t DATA_SPACE_BYTES = 851968;
    if (p.tensor_bytes > DATA_SPACE_BYTES) {
        ADD_FAILURE() << "Payload " << p.tensor_bytes << " exceeds L1 data space " << DATA_SPACE_BYTES;
        return;
    }
    if ((p.tensor_bytes % 4) != 0) {
        ADD_FAILURE() << "tensor_bytes must be a multiple of 4 for word-aligned verification.";
        return;
    }

    const size_t n_words = p.tensor_bytes / 4;
    const bool is_scatter = family_is_scatter(p.family);
    uint32_t scatter_half_bytes = 0;
    uint32_t scatter_offset = 0;
    if (is_scatter) {
        scatter_half_bytes = p.tensor_bytes / 2;
        scatter_offset = scatter_half_bytes;
    }

    // Generate test data pattern
    auto tx = make_tx_pattern(n_words);
    std::vector<uint32_t> zeros(n_words, 0u);

    // Write source data to sender L1
    tt::tt_metal::detail::WriteToDeviceL1(
        src_dev, p.sender_core, src_l1_addr, tx, CoreType::WORKER);

    // Zero destination L1 on ALL receivers
    for (const auto& r : receivers) {
        tt::tt_metal::detail::WriteToDeviceL1(
            r.dev, p.receiver_core, dst_l1_addr, zeros, CoreType::WORKER);
        if (is_scatter) {
            tt::tt_metal::detail::WriteToDeviceL1(
                r.dev, p.receiver_core, dst_l1_addr + scatter_offset, zeros, CoreType::WORKER);
        }
    }

    // --- Global semaphore for completion signaling on each receiver ---
    const std::string RX_KDIR = "tests/tt_metal/tt_fabric/fabric_data_movement/addrgen_write/kernels/";
    std::vector<tt::tt_metal::Program> receiver_progs;
    std::vector<tt::tt_metal::GlobalSemaphore> receiver_sems;
    receiver_progs.reserve(receivers.size());
    receiver_sems.reserve(receivers.size());

    for (const auto& r : receivers) {
        tt::tt_metal::CoreRangeSet rx_core_set(
            tt::tt_metal::CoreRange(p.receiver_core, p.receiver_core));
        receiver_sems.push_back(tt::tt_metal::CreateGlobalSemaphore(
            r.mesh.get(), rx_core_set, /*initial_value=*/0));

        receiver_progs.emplace_back(tt::tt_metal::CreateProgram());
        auto rx_wait_k = tt::tt_metal::CreateKernel(
            receiver_progs.back(),
            RX_KDIR + "rx_addrgen.cpp",
            p.receiver_core,
            tt::tt_metal::DataMovementConfig{
                .processor = tt::tt_metal::DataMovementProcessor::RISCV_0,
                .noc = tt::tt_metal::NOC::RISCV_0_default,
                .defines = defines});

        // Fused and non-fused both use sem_wait_value=1 (fused fires once on final chunk;
        // non-fused gets one separate atomic_inc per direction, but each receiver is reached
        // by exactly one direction in the bounding-box multicast tree)
        const uint32_t sem_wait_value = 1u;
        tt::tt_metal::SetRuntimeArgs(
            receiver_progs.back(), rx_wait_k, p.receiver_core,
            {receiver_sems.back().address(), sem_wait_value});
    }

    // --- Sender program ---
    tt::tt_metal::Program sender_prog = tt::tt_metal::CreateProgram();

    std::string kernel_path = family_kernel_path(p.family);
    auto writer_k = tt::tt_metal::CreateKernel(
        sender_prog,
        kernel_path,
        p.sender_core,
        tt::tt_metal::DataMovementConfig{
            .processor = tt::tt_metal::DataMovementProcessor::RISCV_1,
            .noc = tt::tt_metal::NOC::RISCV_1_default,
            .defines = defines});

    // Use the first receiver's semaphore address for the sender RT args.
    // All receivers have the same semaphore L1 address (same core + same allocation).
    const uint32_t sem_l1_addr = receiver_sems[0].address();

    // Build runtime args -- differs between 2D and 1D modes
    std::vector<uint32_t> writer_rt;

    if (is_2d_fabric) {
        // ===== 2D Mesh Mode =====
        // Compute bounding-box hop counts for all receivers relative to sender.
        const auto& mesh_graph = cp.get_mesh_graph();
        auto src_mesh_coord = mesh_graph.chip_to_coordinate(
            tt::tt_fabric::MeshId{p.mesh_id}, p.src_chip);
        int src_r = src_mesh_coord[0], src_c = src_mesh_coord[1];
        int min_r = src_r, max_r = src_r, min_c = src_c, max_c = src_c;

        for (const auto& r : receivers) {
            auto coord = mesh_graph.chip_to_coordinate(r.fabric_node.mesh_id, r.fabric_node.chip_id);
            min_r = std::min(min_r, (int)coord[0]);
            max_r = std::max(max_r, (int)coord[0]);
            min_c = std::min(min_c, (int)coord[1]);
            max_c = std::max(max_c, (int)coord[1]);
        }

        uint16_t e_hops = 0, w_hops = 0, n_hops = 0, s_hops = 0;
        if (max_c > src_c) { e_hops = (uint16_t)(max_c - src_c); }
        if (min_c < src_c) { w_hops = (uint16_t)(src_c - min_c); }
        if (max_r > src_r) { s_hops = (uint16_t)(max_r - src_r); }
        if (min_r < src_r) { n_hops = (uint16_t)(src_r - min_r); }

        // Direction bitmask: bit0=W, bit1=E, bit2=N, bit3=S
        const uint32_t dir_mask =
            (w_hops ? 1u : 0u) | (e_hops ? 2u : 0u) | (n_hops ? 4u : 0u) | (s_hops ? 8u : 0u);

        // Common prefix
        writer_rt = {src_l1_addr, p.tensor_bytes, dst_l1_addr, rx_xy.x, rx_xy.y, sem_l1_addr, dir_mask};
        if (is_scatter) {
            writer_rt.push_back(scatter_offset);
        }

        // Per-direction fabric connections (W, E, N, S)
        auto find_receiver_at = [&](int row, int col) -> tt::tt_fabric::FabricNodeId {
            for (const auto& r : receivers) {
                auto coord = mesh_graph.chip_to_coordinate(r.fabric_node.mesh_id, r.fabric_node.chip_id);
                if ((int)coord[0] == row && (int)coord[1] == col) {
                    return r.fabric_node;
                }
            }
            return receivers[0].fabric_node;
        };

        auto pick_link = [&](tt::tt_fabric::FabricNodeId dst_fn, uint32_t& out_link_idx) {
            auto links = tt::tt_fabric::get_forwarding_link_indices(src, dst_fn);
            if (links.empty()) {
                ADD_FAILURE() << "No forwarding link from src to representative";
                return false;
            }
            out_link_idx = links[0];
            return true;
        };

        uint32_t link_idx_w = 0, link_idx_e = 0, link_idx_n = 0, link_idx_s = 0;
        tt::tt_fabric::FabricNodeId rep_w = src, rep_e = src, rep_n = src, rep_s = src;
        if (w_hops) { rep_w = find_receiver_at(src_r, min_c); if (!pick_link(rep_w, link_idx_w)) return; }
        if (e_hops) { rep_e = find_receiver_at(src_r, max_c); if (!pick_link(rep_e, link_idx_e)) return; }
        if (n_hops) { rep_n = find_receiver_at(min_r, src_c); if (!pick_link(rep_n, link_idx_n)) return; }
        if (s_hops) { rep_s = find_receiver_at(max_r, src_c); if (!pick_link(rep_s, link_idx_s)) return; }

        if (w_hops) { tt::tt_fabric::append_fabric_connection_rt_args(src, rep_w, link_idx_w, sender_prog, p.sender_core, writer_rt); }
        if (e_hops) { tt::tt_fabric::append_fabric_connection_rt_args(src, rep_e, link_idx_e, sender_prog, p.sender_core, writer_rt); }
        if (n_hops) { tt::tt_fabric::append_fabric_connection_rt_args(src, rep_n, link_idx_n, sender_prog, p.sender_core, writer_rt); }
        if (s_hops) { tt::tt_fabric::append_fabric_connection_rt_args(src, rep_s, link_idx_s, sender_prog, p.sender_core, writer_rt); }

        writer_rt.push_back((uint32_t)e_hops);
        writer_rt.push_back((uint32_t)w_hops);
        writer_rt.push_back((uint32_t)n_hops);
        writer_rt.push_back((uint32_t)s_hops);
    } else {
        // ===== 1D Linear Mode =====
        // In linear topology, multicast uses start_distance and range.
        // Source multicasts to all receivers in one direction.
        // start_distance = 1 (first hop neighbor), range = number of receivers.
        const uint8_t start_distance = 1;
        const uint8_t range = static_cast<uint8_t>(receivers.size());

        writer_rt = {src_l1_addr, p.tensor_bytes, dst_l1_addr, rx_xy.x, rx_xy.y, sem_l1_addr};
        writer_rt.push_back(static_cast<uint32_t>(start_distance));
        writer_rt.push_back(static_cast<uint32_t>(range));
        if (is_scatter) {
            writer_rt.push_back(scatter_offset);
        }

        // Find a direct neighbor of src to use as the fabric connection target.
        // In 1D, append_fabric_connection_rt_args requires the dst to be a direct neighbor.
        tt::tt_fabric::FabricNodeId neighbor_node = receivers[0].fabric_node;
        bool found_neighbor = false;
        for (const auto& direction : FabricContext::routing_directions) {
            auto neighbors = cp.get_chip_neighbors(src, direction);
            auto mesh_neighbors = neighbors.find(src.mesh_id);
            if (mesh_neighbors != neighbors.end() && !mesh_neighbors->second.empty()) {
                // Use the first direct neighbor found as the fabric connection target
                neighbor_node = tt::tt_fabric::FabricNodeId{src.mesh_id, mesh_neighbors->second[0]};
                found_neighbor = true;
                break;
            }
        }
        if (!found_neighbor) {
            ADD_FAILURE() << "No direct neighbor found for src in 1D mode";
            return;
        }

        auto forwarding_links = tt::tt_fabric::get_forwarding_link_indices(src, neighbor_node);
        if (forwarding_links.empty()) {
            ADD_FAILURE() << "No forwarding links from src to direct neighbor";
            return;
        }
        tt::tt_fabric::append_fabric_connection_rt_args(
            src, neighbor_node, forwarding_links[0], sender_prog, p.sender_core, writer_rt);
    }

    tt::tt_metal::SetRuntimeArgs(sender_prog, writer_k, p.sender_core, writer_rt);

    // --- Execute: all receivers first (so they're ready), then sender ---
    for (size_t i = 0; i < receivers.size(); ++i) {
        fixture->RunProgramNonblocking(receivers[i].mesh, receiver_progs[i]);
    }
    fixture->RunProgramNonblocking(src_mesh, sender_prog);
    fixture->WaitForSingleProgramDone(src_mesh, sender_prog);
    for (size_t i = 0; i < receivers.size(); ++i) {
        fixture->WaitForSingleProgramDone(receivers[i].mesh, receiver_progs[i]);
    }

    // --- Read back and verify on ALL receivers ---
    for (const auto& r : receivers) {
        if (is_scatter) {
            size_t half_words = scatter_half_bytes / 4;

            std::vector<uint32_t> rx_half0(half_words, 0u);
            tt::tt_metal::detail::ReadFromDeviceL1(
                r.dev, p.receiver_core, dst_l1_addr, scatter_half_bytes, rx_half0, CoreType::WORKER);

            std::vector<uint32_t> rx_half1(half_words, 0u);
            tt::tt_metal::detail::ReadFromDeviceL1(
                r.dev, p.receiver_core, dst_l1_addr + scatter_offset, scatter_half_bytes, rx_half1, CoreType::WORKER);

            std::vector<uint32_t> tx_half0(tx.begin(), tx.begin() + half_words);
            std::vector<uint32_t> tx_half1(tx.begin() + half_words, tx.end());

            verify_payload_words(rx_half0, tx_half0);
            verify_payload_words(rx_half1, tx_half1);
        } else {
            std::vector<uint32_t> rx(n_words, 0u);
            tt::tt_metal::detail::ReadFromDeviceL1(
                r.dev, p.receiver_core, dst_l1_addr, p.tensor_bytes, rx, CoreType::WORKER);
            verify_payload_words(rx, tx);
        }
    }
}

}  // namespace tt::tt_fabric::test
