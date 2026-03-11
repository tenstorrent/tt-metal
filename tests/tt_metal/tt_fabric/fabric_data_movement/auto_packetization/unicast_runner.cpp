// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Host-side test runner for raw-size unicast auto-packetization tests.
// Dispatches the unicast_tx_writer_raw device kernel which sends a large
// payload via the auto-packetizing fabric_unicast_noc_unicast_write wrapper,
// then validates that all data arrives correctly at the destination.

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

// Resolve forwarding link and fail early if none found.
inline bool pick_forwarding_link_or_fail(
    const tt::tt_fabric::FabricNodeId& src,
    const tt::tt_fabric::FabricNodeId& dst,
    uint32_t& out_link_idx) {
    auto links = tt::tt_fabric::get_forwarding_link_indices(src, dst);
    if (links.empty()) {
        ADD_FAILURE() << "No forwarding links from src to dst";
        return false;
    }
    out_link_idx = links[0];
    return true;
}

// Device lookup and basic existence check.
inline bool lookup_devices_or_fail(
    ChipId src_phys, ChipId dst_phys,
    tt::tt_metal::IDevice*& src_dev, tt::tt_metal::IDevice*& dst_dev) {
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
            ADD_FAILURE() << "Data mismatch at word " << i << " (got 0x" << std::hex << rx[i]
                          << ", exp 0x" << tx[i] << std::dec << ")";
            return;
        }
    }
}

}  // anonymous namespace

// ----------------------------------- program -----------------------------------
void run_raw_unicast_write_test(tt::tt_metal::MeshDeviceFixtureBase* fixture, const RawTestParams& p) {
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

    tt::tt_metal::IDevice* src_dev = nullptr;
    tt::tt_metal::IDevice* dst_dev = nullptr;
    if (!lookup_devices_or_fail(src_phys, dst_phys, src_dev, dst_dev)) {
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

    // --- IO buffers & initialization (MeshBuffer style) ---
    // Source buffer in DRAM (sender reads from it into L1 before fabric send)
    Dist::DeviceLocalBufferConfig src_local{
        .page_size = p.tensor_bytes, .buffer_type = tt::tt_metal::BufferType::DRAM};
    // Destination buffer where fabric delivers data
    Dist::DeviceLocalBufferConfig dst_local{
        .page_size = p.tensor_bytes,
        .buffer_type = p.use_dram_dst ? tt::tt_metal::BufferType::DRAM : tt::tt_metal::BufferType::L1};
    Dist::ReplicatedBufferConfig rcfg{.size = p.tensor_bytes};
    auto src_buf = Dist::MeshBuffer::create(rcfg, src_local, mesh.get());
    auto dst_buf = Dist::MeshBuffer::create(rcfg, dst_local, mesh.get());

    const size_t n_words = p.tensor_bytes / 4;
    auto tx = make_tx_pattern(n_words);
    std::vector<uint32_t> zeros(n_words, 0u);

    auto& mcq = mesh->mesh_command_queue();
    Dist::WriteShard(mcq, src_buf, tx, src_coord, /*blocking=*/true);
    Dist::WriteShard(mcq, dst_buf, zeros, dst_coord, /*blocking=*/true);

    // --- Global semaphore for completion signaling ---
    tt::tt_metal::Program receiver_prog = tt::tt_metal::CreateProgram();
    tt::tt_metal::CoreRangeSet rx_core_set(tt::tt_metal::CoreRange(p.receiver_core, p.receiver_core));
    static std::optional<tt::tt_metal::GlobalSemaphore> gsem;
    if (!gsem) {
        gsem = tt::tt_metal::CreateGlobalSemaphore(mesh.get(), rx_core_set, /*initial_value=*/0);
    }

    // --- Receiver kernel: waits for semaphore bump, then exits ---
    const std::string RX_KDIR = "tests/tt_metal/tt_fabric/fabric_data_movement/addrgen_write/kernels/";
    auto rx_wait_k = tt::tt_metal::CreateKernel(
        receiver_prog,
        RX_KDIR + "rx_addrgen.cpp",
        p.receiver_core,
        tt::tt_metal::DataMovementConfig{
            .processor = tt::tt_metal::DataMovementProcessor::RISCV_0,
            .noc = tt::tt_metal::NOC::RISCV_0_default,
            .defines = defines});

    // Receiver waits for 1 atomic inc (sent after all data)
    tt::tt_metal::SetRuntimeArgs(receiver_prog, rx_wait_k, p.receiver_core, {gsem->address(), 1u});

    // --- Sender program: single kernel on RISCV_1 ---
    tt::tt_metal::Program sender_prog = tt::tt_metal::CreateProgram();

    const std::string TX_KDIR =
        "tests/tt_metal/tt_fabric/fabric_data_movement/auto_packetization/kernels/";

    auto writer_k = tt::tt_metal::CreateKernel(
        sender_prog,
        TX_KDIR + "unicast_tx_writer_raw.cpp",
        p.sender_core,
        tt::tt_metal::DataMovementConfig{
            .processor = tt::tt_metal::DataMovementProcessor::RISCV_1,
            .noc = tt::tt_metal::NOC::RISCV_1_default,
            .defines = defines});

    // Writer runtime args: pass address components (device kernel computes NOC addr)
    std::vector<uint32_t> writer_rt = {
        (uint32_t)src_buf->address(),                 // 0: src_l1_addr
        (uint32_t)p.tensor_bytes,                     // 1: total_size
        (uint32_t)dst_buf->address(),                 // 2: dst_base_addr
        (uint32_t)p.mesh_id,                          // 3: dst_mesh_id
        (uint32_t)p.dst_chip,                         // 4: dst_dev_id
        (uint32_t)rx_xy.x,                            // 5: rx_noc_x
        (uint32_t)rx_xy.y,                            // 6: rx_noc_y
        (uint32_t)gsem->address(),                    // 7: sem_l1_addr
    };

    // Append fabric connection runtime args
    uint32_t link_idx = 0;
    if (!pick_forwarding_link_or_fail(src, dst, link_idx)) {
        return;
    }
    tt::tt_fabric::append_fabric_connection_rt_args(
        src, dst, link_idx, sender_prog, p.sender_core, writer_rt);

    tt::tt_metal::SetRuntimeArgs(sender_prog, writer_k, p.sender_core, writer_rt);

    // --- Execute ---
    Dist::MeshWorkload receiver_workload;
    Dist::MeshWorkload sender_workload;
    receiver_workload.add_program(Dist::MeshCoordinateRange(dst_coord), std::move(receiver_prog));
    sender_workload.add_program(Dist::MeshCoordinateRange(src_coord), std::move(sender_prog));

    // Receiver first (so it's ready), then sender
    Dist::EnqueueMeshWorkload(mcq, receiver_workload, /*blocking=*/false);
    Dist::EnqueueMeshWorkload(mcq, sender_workload, /*blocking=*/true);

    // --- Verify ---
    std::vector<uint32_t> rx(n_words, 0u);
    Dist::ReadShard(mcq, rx, dst_buf, dst_coord, /*blocking=*/true);
    verify_payload_words(rx, tx);
}

}  // namespace tt::tt_fabric::test
