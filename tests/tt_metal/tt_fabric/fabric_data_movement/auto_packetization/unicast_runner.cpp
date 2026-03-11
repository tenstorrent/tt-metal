// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Host-side test runner for raw-size unicast auto-packetization silicon tests.
// Uses BaseFabricFixture (per-chip MeshDevice) pattern -- NOT MeshDeviceFixtureBase.
// Dispatches the appropriate unicast device kernel based on AutoPacketFamily,
// writes test data directly to L1, and validates byte-for-byte correctness.

#include <gtest/gtest.h>
#include <algorithm>
#include <vector>

#include <tt-metalium/tt_metal.hpp>
#include <tt-metalium/global_semaphore.hpp>
#include <tt-metalium/experimental/fabric/fabric.hpp>

#include "tests/tt_metal/tt_fabric/common/fabric_fixture.hpp"
#include "tt_metal/fabric/fabric_context.hpp"
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

// Main unicast runner: dispatches the correct device kernel for the given family
// and verifies data correctness on silicon.
void run_raw_unicast_write_test(BaseFabricFixture* fixture, const RawTestParams& p) {
    const auto& cp = tt::tt_metal::MetalContext::instance().get_control_plane();

    // Check if fabric is 2D
    const auto& fabric_context = cp.get_fabric_context();
    const bool is_2d_fabric = fabric_context.is_2D_routing_enabled();
    std::map<std::string, std::string> defines;
    if (is_2d_fabric) {
        defines["FABRIC_2D"] = "1";
    }

    tt::tt_fabric::FabricNodeId src{tt::tt_fabric::MeshId{p.mesh_id}, p.src_chip};
    tt::tt_fabric::FabricNodeId dst{tt::tt_fabric::MeshId{p.mesh_id}, p.dst_chip};

    ChipId src_phys = cp.get_physical_chip_id_from_fabric_node_id(src);
    ChipId dst_phys = cp.get_physical_chip_id_from_fabric_node_id(dst);

    // Get per-chip MeshDevice objects from BaseFabricFixture
    auto src_mesh = fixture->get_device(src_phys);
    auto dst_mesh = fixture->get_device(dst_phys);
    auto* src_dev = src_mesh->get_devices()[0];
    auto* dst_dev = dst_mesh->get_devices()[0];

    // Resolve receiver NOC coordinates
    tt::tt_metal::CoreCoord rx_xy = dst_dev->worker_core_from_logical_core(p.receiver_core);

    // Get L1 memory layout from worker mem map
    const auto topology = fabric_context.get_fabric_topology();
    auto src_mem_map = BaseFabricFixture::generate_worker_mem_map(src_mesh, topology);
    auto dst_mem_map = BaseFabricFixture::generate_worker_mem_map(dst_mesh, topology);

    const uint32_t src_l1_addr = src_mem_map.source_l1_buffer_address;
    const uint32_t dst_l1_addr = dst_mem_map.target_address;

    // Validate payload fits in L1 data space (851968 bytes available)
    constexpr uint32_t DATA_SPACE_BYTES = 851968;
    if (p.tensor_bytes > DATA_SPACE_BYTES) {
        ADD_FAILURE() << "Payload " << p.tensor_bytes << " exceeds L1 data space " << DATA_SPACE_BYTES;
        return;
    }

    // Validate word alignment
    if ((p.tensor_bytes % 4) != 0) {
        ADD_FAILURE() << "tensor_bytes must be a multiple of 4 for word-aligned verification.";
        return;
    }

    const size_t n_words = p.tensor_bytes / 4;
    const bool is_scatter = family_is_scatter(p.family);

    // For scatter: payload goes to two addresses, each getting half
    uint32_t scatter_half_bytes = 0;
    uint32_t scatter_offset = 0;
    if (is_scatter) {
        scatter_half_bytes = p.tensor_bytes / 2;
        // Place second scatter chunk after the first in destination L1
        scatter_offset = scatter_half_bytes;
    }

    // Generate test data pattern
    auto tx = make_tx_pattern(n_words);
    std::vector<uint32_t> zeros(n_words, 0u);

    // Write source data directly to sender L1
    tt::tt_metal::detail::WriteToDeviceL1(
        src_dev, p.sender_core, src_l1_addr, tx, CoreType::WORKER);

    // Zero destination L1 buffer
    tt::tt_metal::detail::WriteToDeviceL1(
        dst_dev, p.receiver_core, dst_l1_addr, zeros, CoreType::WORKER);

    // If scatter, also zero the second half region
    if (is_scatter) {
        tt::tt_metal::detail::WriteToDeviceL1(
            dst_dev, p.receiver_core, dst_l1_addr + scatter_offset, zeros, CoreType::WORKER);
    }

    // --- Global semaphore for completion signaling ---
    tt::tt_metal::CoreRangeSet rx_core_set(
        tt::tt_metal::CoreRange(p.receiver_core, p.receiver_core));
    auto gsem = tt::tt_metal::CreateGlobalSemaphore(
        dst_mesh.get(), rx_core_set, /*initial_value=*/0);

    // --- Receiver program: wait for semaphore bump ---
    tt::tt_metal::Program receiver_prog = tt::tt_metal::CreateProgram();
    const std::string RX_KDIR = "tests/tt_metal/tt_fabric/fabric_data_movement/addrgen_write/kernels/";
    auto rx_wait_k = tt::tt_metal::CreateKernel(
        receiver_prog,
        RX_KDIR + "rx_addrgen.cpp",
        p.receiver_core,
        tt::tt_metal::DataMovementConfig{
            .processor = tt::tt_metal::DataMovementProcessor::RISCV_0,
            .noc = tt::tt_metal::NOC::RISCV_0_default,
            .defines = defines});

    // Fused families fire atomic_inc once on final chunk. Non-fused get separate atomic_inc.
    const uint32_t sem_wait_value = 1u;
    tt::tt_metal::SetRuntimeArgs(
        receiver_prog, rx_wait_k, p.receiver_core, {gsem.address(), sem_wait_value});

    // --- Sender program: single writer kernel on RISCV_1 ---
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

    // Build runtime args -- layout depends on family and fabric mode (2D vs 1D)
    std::vector<uint32_t> writer_rt;

    // Common prefix: src_l1_addr, total_size, dst_base_addr
    writer_rt.push_back(src_l1_addr);
    writer_rt.push_back(p.tensor_bytes);
    writer_rt.push_back(dst_l1_addr);

    if (is_2d_fabric) {
        // Mesh mode: dst_mesh_id, dst_dev_id before NOC coords
        writer_rt.push_back(p.mesh_id);
        writer_rt.push_back(static_cast<uint32_t>(p.dst_chip));
    }

    writer_rt.push_back(rx_xy.x);
    writer_rt.push_back(rx_xy.y);
    writer_rt.push_back(gsem.address());

    if (!is_2d_fabric) {
        // Linear mode: num_hops after sem_addr (before fabric connection args)
        // For linear unicast, num_hops = 1 (single-hop between adjacent devices)
        writer_rt.push_back(1u);
    }

    if (is_scatter) {
        writer_rt.push_back(scatter_offset);
    }

    // Append fabric connection runtime args
    auto forwarding_links = tt::tt_fabric::get_forwarding_link_indices(src, dst);
    if (forwarding_links.empty()) {
        ADD_FAILURE() << "No forwarding links from src to dst";
        return;
    }
    uint32_t link_idx = forwarding_links[0];
    tt::tt_fabric::append_fabric_connection_rt_args(
        src, dst, link_idx, sender_prog, p.sender_core, writer_rt);

    tt::tt_metal::SetRuntimeArgs(sender_prog, writer_k, p.sender_core, writer_rt);

    // --- Execute: receiver first (so it's ready), then sender ---
    fixture->RunProgramNonblocking(dst_mesh, receiver_prog);
    fixture->RunProgramNonblocking(src_mesh, sender_prog);
    fixture->WaitForSingleProgramDone(src_mesh, sender_prog);
    fixture->WaitForSingleProgramDone(dst_mesh, receiver_prog);

    // --- Read back and verify ---
    if (is_scatter) {
        // Scatter: verify two halves at dst_l1_addr and dst_l1_addr + scatter_offset
        size_t half_words = scatter_half_bytes / 4;

        std::vector<uint32_t> rx_half0(half_words, 0u);
        tt::tt_metal::detail::ReadFromDeviceL1(
            dst_dev, p.receiver_core, dst_l1_addr, scatter_half_bytes, rx_half0, CoreType::WORKER);

        std::vector<uint32_t> rx_half1(half_words, 0u);
        tt::tt_metal::detail::ReadFromDeviceL1(
            dst_dev, p.receiver_core, dst_l1_addr + scatter_offset, scatter_half_bytes, rx_half1, CoreType::WORKER);

        // First half of TX goes to addr0, second half to addr1
        std::vector<uint32_t> tx_half0(tx.begin(), tx.begin() + half_words);
        std::vector<uint32_t> tx_half1(tx.begin() + half_words, tx.end());

        verify_payload_words(rx_half0, tx_half0);
        verify_payload_words(rx_half1, tx_half1);
    } else {
        // Non-scatter: verify full payload at dst_l1_addr
        std::vector<uint32_t> rx(n_words, 0u);
        tt::tt_metal::detail::ReadFromDeviceL1(
            dst_dev, p.receiver_core, dst_l1_addr, p.tensor_bytes, rx, CoreType::WORKER);

        verify_payload_words(rx, tx);
    }
}

}  // namespace tt::tt_fabric::test
