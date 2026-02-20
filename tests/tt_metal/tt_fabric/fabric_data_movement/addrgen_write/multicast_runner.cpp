// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>
#include <fmt/format.h>
#include <algorithm>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <utility>
#include <vector>

#include <tt-metalium/tt_metal.hpp>
#include "tests/tt_metal/tt_metal/common/multi_device_fixture.hpp"
#include "tests/tt_metal/tt_fabric/common/utils.hpp"
#include "tests/tt_metal/tt_fabric/fabric_data_movement/addrgen_write/test_common.hpp"
#include "tests/tt_metal/tt_fabric/fabric_data_movement/addrgen_write/kernels/kernel_common.hpp"
#include "tt_metal/fabric/fabric_context.hpp"
#include <tt-metalium/global_semaphore.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/mesh_device.hpp>
#include <tt-metalium/mesh_device_view.hpp>
#include <distributed/mesh_device_view_impl.hpp>

namespace tt::tt_fabric::test {

// Import needed types
using tt::tt_fabric::test::AddrgenTestParams;

// ---------- helpers (validation / utilities) ----------

namespace {

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
}

}  // anonymous namespace

// ----------------------------------- program -----------------------------------
void run_multicast_write_test(tt::tt_metal::MeshDeviceFixtureBase* fixture, const AddrgenTestParams& p) {
    const auto& cp = tt::tt_metal::MetalContext::instance().get_control_plane();

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

    // --- IO buffers & initialization ---
    namespace Dist = tt::tt_metal::distributed;

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

    // === Build the multicast receiver set from a rectangular sub-mesh ===
    std::vector<Dist::MeshCoordinate> dst_coords;
    const auto shape = view.shape();  // [rows, cols]
    const uint32_t M = (p.mesh_rows ? p.mesh_rows : (uint32_t)shape[0]);
    const uint32_t N = (p.mesh_cols ? p.mesh_cols : (uint32_t)shape[1]);
    if (M == 0 || N == 0 || M > (uint32_t)shape[0] || N > (uint32_t)shape[1]) {
        ADD_FAILURE() << "Invalid mesh_rows/mesh_cols for physical mesh shape (" << shape[0] << "x" << shape[1] << ")";
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
                continue;  // exclude sender chip from fabric RX
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

    // Build RX programs: one per receiver chip
    std::vector<tt::tt_metal::Program> receiver_progs;
    receiver_progs.reserve(dst_coords.size());

    // One semaphore at a single logical core → same L1 offset on every chip in the MeshDevice
    static std::optional<tt::tt_metal::GlobalSemaphore> gsem_done;
    if (!gsem_done) {
        tt::tt_metal::CoreRangeSet rx_core_one(tt::tt_metal::CoreRange(p.receiver_core, p.receiver_core));
        gsem_done = tt::tt_metal::CreateGlobalSemaphore(mesh.get(), rx_core_one, /*initial_value=*/0);
    }

    constexpr const char* KDIR = "tests/tt_metal/tt_fabric/fabric_data_movement/addrgen_write/kernels/";

    // Helper to map API variant to OPERATION_TYPE and API_VARIANT compile-time parameters
    auto get_operation_and_api_variant = [](AddrgenApiVariant variant) -> std::pair<OperationType, ApiVariant> {
        // Returns OperationType and ApiVariant enums
        switch (variant) {
            case AddrgenApiVariant::MulticastWrite: return {OperationType::BasicWrite, ApiVariant::Basic};
            case AddrgenApiVariant::MulticastWriteWithState: return {OperationType::BasicWrite, ApiVariant::WithState};
            case AddrgenApiVariant::MulticastWriteSetState: return {OperationType::BasicWrite, ApiVariant::SetState};
            case AddrgenApiVariant::MulticastFusedAtomicIncWrite:
                return {OperationType::FusedAtomicInc, ApiVariant::Basic};
            case AddrgenApiVariant::MulticastFusedAtomicIncWriteWithState:
                return {OperationType::FusedAtomicInc, ApiVariant::WithState};
            case AddrgenApiVariant::MulticastFusedAtomicIncWriteSetState:
                return {OperationType::FusedAtomicInc, ApiVariant::SetState};
            case AddrgenApiVariant::MulticastScatterWrite: return {OperationType::Scatter, ApiVariant::Basic};
            case AddrgenApiVariant::MulticastScatterWriteWithState:
                return {OperationType::Scatter, ApiVariant::WithState};
            case AddrgenApiVariant::MulticastScatterWriteSetState:
                return {OperationType::Scatter, ApiVariant::SetState};
            default: TT_FATAL(false, "Unknown API variant"); return {OperationType::BasicWrite, ApiVariant::Basic};
        }
    };

    auto [operation_type, api_variant] = get_operation_and_api_variant(p.api_variant);

    const bool is_fused_atomic_inc = (operation_type == OperationType::FusedAtomicInc);

    // Move NUM_PAGES calculation before receiver setup
    const uint32_t NUM_PAGES = (p.tensor_bytes + p.page_size - 1) / p.page_size;

    // Calculate aligned page sizes for source (DRAM) and destination (DRAM or L1)
    const auto& hal = tt::tt_metal::MetalContext::instance().hal();
    uint32_t src_alignment = hal.get_alignment(tt::tt_metal::HalMemType::DRAM);  // Source is always DRAM
    uint32_t dst_alignment = p.use_dram_dst ? hal.get_alignment(tt::tt_metal::HalMemType::DRAM)
                                            : hal.get_alignment(tt::tt_metal::HalMemType::L1);

    // Round up to alignment boundary
    uint32_t src_aligned_page_size = ((p.page_size + src_alignment - 1) / src_alignment) * src_alignment;
    uint32_t dst_aligned_page_size = ((p.page_size + dst_alignment - 1) / dst_alignment) * dst_alignment;

    for (size_t i = 0; i < dst_coords.size(); ++i) {
        receiver_progs.emplace_back(tt::tt_metal::CreateProgram());
        auto rx_wait_k = tt::tt_metal::CreateKernel(
            receiver_progs.back(),
            std::string(KDIR) + "rx_addrgen.cpp",  // Unified receiver kernel
            p.receiver_core,
            tt::tt_metal::DataMovementConfig{
                .processor = tt::tt_metal::DataMovementProcessor::RISCV_0,
                .noc = tt::tt_metal::NOC::RISCV_0_default,
                .defines = defines});
        const uint32_t sem_wait_value = is_fused_atomic_inc ? NUM_PAGES : 1u;
        tt::tt_metal::SetRuntimeArgs(
            receiver_progs.back(), rx_wait_k, p.receiver_core, {gsem_done->address(), sem_wait_value});
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

    // Sender program: READER (RISCV_0) + WRITER (RISCV_1)
    tt::tt_metal::Program sender_prog = tt::tt_metal::CreateProgram();
    const uint32_t CB_ID = tt::CBIndex::c_0;
    // CB holds 8 pages total so the reader can fill 4 while the writer drains 4.
    // Use source aligned page size (reader reads from DRAM buffer with DRAM alignment)
    auto cb_cfg = tt::tt_metal::CircularBufferConfig(8 * src_aligned_page_size, {{CB_ID, tt::DataFormat::Float16}})
                      .set_page_size(CB_ID, src_aligned_page_size);
    (void)tt::tt_metal::CreateCircularBuffer(sender_prog, p.sender_core, cb_cfg);

    // Reader kernel (DRAM->CB) - now uses unified kernel with OPERATION_TYPE compile-time arg
    std::vector<uint32_t> reader_cta;
    tt::tt_metal::TensorAccessorArgs(*src_buf).append_to(reader_cta);
    reader_cta.push_back(static_cast<uint32_t>(operation_type));  // OPERATION_TYPE
    reader_cta.push_back(1u);              // SRC_IS_DRAM
    reader_cta.push_back(NUM_PAGES);
    reader_cta.push_back(p.page_size);     // Raw page size (actual data size to transfer)
    reader_cta.push_back(src_aligned_page_size);  // Aligned page size (source buffer spacing)

    auto reader_k = tt::tt_metal::CreateKernel(
        sender_prog,
        std::string(KDIR) + "tx_reader_to_cb_addrgen.cpp",  // Unified reader kernel
        p.sender_core,
        tt::tt_metal::DataMovementConfig{
            .processor = tt::tt_metal::DataMovementProcessor::RISCV_0,
            .noc = tt::tt_metal::NOC::RISCV_0_default,
            .compile_args = reader_cta,
            .defines = defines});
    tt::tt_metal::SetRuntimeArgs(sender_prog, reader_k, p.sender_core, {(uint32_t)src_buf->address()});

    // Writer kernel (CB->Fabric->dst + final sem INC) - now uses unified kernel with compile-time args
    std::vector<uint32_t> writer_cta;
    tt::tt_metal::TensorAccessorArgs(*dst_buf).append_to(writer_cta);
    writer_cta.push_back(static_cast<uint32_t>(operation_type));  // OPERATION_TYPE
    writer_cta.push_back(static_cast<uint32_t>(api_variant));     // API_VARIANT
    writer_cta.push_back(NUM_PAGES);       // TOTAL_PAGES
    writer_cta.push_back(p.page_size);     // Raw page size (actual data size to transfer)
    writer_cta.push_back(dst_aligned_page_size);  // Aligned page size (dest buffer addressing)
    writer_cta.push_back(src_aligned_page_size);  // Source aligned page size (CB stride for scatter)

    auto writer_k = tt::tt_metal::CreateKernel(
        sender_prog,
        std::string(KDIR) + "multicast_tx_writer_addrgen.cpp",  // Unified multicast writer kernel
        p.sender_core,
        tt::tt_metal::DataMovementConfig{
            .processor = tt::tt_metal::DataMovementProcessor::RISCV_1,
            .noc = tt::tt_metal::NOC::RISCV_1_default,
            .compile_args = writer_cta,
            .defines = defines});

    // Writer kernel RT args (base): dst_base, rx_x, rx_y, sem_l1
    std::vector<uint32_t> writer_rt = {
        (uint32_t)dst_buf->address(), (uint32_t)rx_xy.x, (uint32_t)rx_xy.y, (uint32_t)gsem_done->address()};

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
    auto coord_to_fabric_id = [&](Dist::MeshCoordinate mc) -> tt::tt_fabric::FabricNodeId {
        auto* dev = view.impl().get_device(mc);
        TT_FATAL(dev != nullptr, "No device at mesh coord ({}, {})", (int)mc[0], (int)mc[1]);
        ChipId phys = dev->id();
        return cp.get_fabric_node_id_from_physical_chip_id(phys);
    };
    auto src_fn = tt::tt_fabric::FabricNodeId{tt::tt_fabric::MeshId{p.mesh_id}, p.src_chip};

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
        return;
    }
    if (e_hops && !pick_link(rep_e, link_idx_e)) {
        return;
    }
    if (n_hops && !pick_link(rep_n, link_idx_n)) {
        return;
    }
    if (s_hops && !pick_link(rep_s, link_idx_s)) {
        return;
    }

    // Direction bitmask
    const uint32_t dir_mask = (w_hops ? 1u : 0u) | (e_hops ? 2u : 0u) | (n_hops ? 4u : 0u) | (s_hops ? 8u : 0u);
    writer_rt.push_back(dir_mask);

    // Append the fabric connection blocks in fixed order: W, E, N, S
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

    // Append hops
    writer_rt.push_back((uint32_t)e_hops);
    writer_rt.push_back((uint32_t)w_hops);
    writer_rt.push_back((uint32_t)n_hops);
    writer_rt.push_back((uint32_t)s_hops);

    tt::tt_metal::SetRuntimeArgs(sender_prog, writer_k, p.sender_core, writer_rt);

    // Build workloads
    auto sender_workload = Dist::MeshWorkload();
    auto receiver_workload = Dist::MeshWorkload();

    sender_workload.add_program(Dist::MeshCoordinateRange(src_coord), std::move(sender_prog));
    for (size_t i = 0; i < dst_coords.size(); ++i) {
        receiver_workload.add_program(Dist::MeshCoordinateRange(dst_coords[i]), std::move(receiver_progs[i]));
    }

    // Execute
    Dist::EnqueueMeshWorkload(mcq, receiver_workload, /*blocking=*/false);
    Dist::EnqueueMeshWorkload(mcq, sender_workload, /*blocking=*/true);

    // Read back and verify
    for (const auto& mc : dst_coords) {
        std::vector<uint32_t> rx(n_words, 0u);
        Dist::ReadShard(mcq, rx, dst_buf, mc, /*blocking=*/true);
        verify_payload_words(rx, tx);
    }
}

}  // namespace tt::tt_fabric::test
