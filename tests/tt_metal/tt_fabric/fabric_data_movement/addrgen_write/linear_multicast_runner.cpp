// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>
#include <fmt/format.h>
#include <algorithm>
#include <filesystem>
#include <fstream>
#include <iomanip>
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

// Map AddrgenApiVariant to kernel OperationType for linear multicast
inline OperationType get_linear_multicast_operation_type(AddrgenApiVariant variant) {
    switch (variant) {
        case AddrgenApiVariant::LinearMulticastWrite:
        case AddrgenApiVariant::LinearMulticastWriteWithState:
        case AddrgenApiVariant::LinearMulticastWriteSetState: return OperationType::BasicWrite;
        case AddrgenApiVariant::LinearMulticastScatterWrite:
        case AddrgenApiVariant::LinearMulticastScatterWriteWithState:
        case AddrgenApiVariant::LinearMulticastScatterWriteSetState: return OperationType::Scatter;
        case AddrgenApiVariant::LinearMulticastFusedAtomicIncWrite:
        case AddrgenApiVariant::LinearMulticastFusedAtomicIncWriteWithState:
        case AddrgenApiVariant::LinearMulticastFusedAtomicIncWriteSetState: return OperationType::FusedAtomicInc;
        default: return OperationType::BasicWrite;
    }
}

// Map AddrgenApiVariant to kernel ApiVariant for linear multicast
inline ApiVariant get_linear_multicast_api_variant(AddrgenApiVariant variant) {
    switch (variant) {
        case AddrgenApiVariant::LinearMulticastWrite:
        case AddrgenApiVariant::LinearMulticastScatterWrite:
        case AddrgenApiVariant::LinearMulticastFusedAtomicIncWrite: return ApiVariant::Basic;
        case AddrgenApiVariant::LinearMulticastWriteWithState:
        case AddrgenApiVariant::LinearMulticastScatterWriteWithState:
        case AddrgenApiVariant::LinearMulticastFusedAtomicIncWriteWithState: return ApiVariant::WithState;
        case AddrgenApiVariant::LinearMulticastWriteSetState:
        case AddrgenApiVariant::LinearMulticastScatterWriteSetState:
        case AddrgenApiVariant::LinearMulticastFusedAtomicIncWriteSetState: return ApiVariant::SetState;
        default: return ApiVariant::Basic;
    }
}

}  // anonymous namespace

// ----------------------------------- program -----------------------------------
void run_linear_multicast_write_test(tt::tt_metal::MeshDeviceFixtureBase* fixture, const AddrgenTestParams& p) {
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

    // Determine operation type
    OperationType operation_type = get_linear_multicast_operation_type(p.api_variant);
    bool is_scatter = (operation_type == OperationType::Scatter);
    bool is_fused = (operation_type == OperationType::FusedAtomicInc);

    // --- IO buffers & initialization ---
    namespace Dist = tt::tt_metal::distributed;

    auto mesh = fixture->get_mesh_device();
    auto view = mesh->get_view();
    auto coord_of_phys = [&](ChipId phys) -> Dist::MeshCoordinate {
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
    Dist::DeviceLocalBufferConfig dst_local{
        .page_size = p.page_size,
        .buffer_type = p.use_dram_dst ? tt::tt_metal::BufferType::DRAM : tt::tt_metal::BufferType::L1};
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

    // === Build the multicast receiver set for linear (1D) topology ===
    // For linear multicast, we send from src_chip to multiple chips in a linear chain
    // Use mesh_rows to determine how many chips to multicast to (default: 2 chips)
    std::vector<Dist::MeshCoordinate> dst_coords;
    const auto shape = view.shape();  // [rows, cols]

    // For linear multicast, assume we're multicasting along a row or column
    // Default: multicast to 2 chips starting from dst_chip
    uint32_t num_receivers = (p.mesh_rows > 0) ? p.mesh_rows : 2;

    // Find chips in linear order starting from dst_chip
    // For simplicity, assume chips are numbered sequentially in the mesh
    ChipId current_chip = p.dst_chip;
    for (uint32_t i = 0; i < num_receivers; ++i) {
        tt::tt_fabric::FabricNodeId dst_node{tt::tt_fabric::MeshId{p.mesh_id}, current_chip};
        ChipId dst_phys_i = cp.get_physical_chip_id_from_fabric_node_id(dst_node);
        Dist::MeshCoordinate dst_coord_i = coord_of_phys(dst_phys_i);

        // Skip sender chip
        if (dst_phys_i != src_phys) {
            dst_coords.push_back(dst_coord_i);
            Dist::WriteShard(mcq, dst_buf, zeros, dst_coord_i, /*blocking=*/true);
        }

        current_chip++;
    }

    if (dst_coords.empty()) {
        ADD_FAILURE() << "Receiver set is empty.";
        return;
    }

    // Build RX programs: one per receiver chip
    std::vector<tt::tt_metal::Program> receiver_progs;
    receiver_progs.reserve(dst_coords.size());

    // One semaphore at a single logical core → same L1 offset on every chip in the MeshDevice
    static std::optional<tt::tt_metal::GlobalSemaphore> gsem_linear_mcast;
    if (!gsem_linear_mcast) {
        tt::tt_metal::CoreRangeSet rx_core_one(tt::tt_metal::CoreRange(p.receiver_core, p.receiver_core));
        gsem_linear_mcast = tt::tt_metal::CreateGlobalSemaphore(mesh.get(), rx_core_one, /*initial_value=*/0);
    }
    // Reset semaphore value before each test
    gsem_linear_mcast->reset_semaphore_value(0);

    constexpr const char* KDIR = "tests/tt_metal/tt_fabric/fabric_data_movement/addrgen_write/kernels/";

    const uint32_t NUM_PAGES = (p.tensor_bytes + p.page_size - 1) / p.page_size;

    // Calculate aligned page sizes
    const auto& hal = tt::tt_metal::MetalContext::instance().hal();
    uint32_t src_alignment = hal.get_alignment(tt::tt_metal::HalMemType::DRAM);
    uint32_t dst_alignment = p.use_dram_dst ? hal.get_alignment(tt::tt_metal::HalMemType::DRAM)
                                            : hal.get_alignment(tt::tt_metal::HalMemType::L1);
    uint32_t src_aligned_page_size = ((p.page_size + src_alignment - 1) / src_alignment) * src_alignment;
    uint32_t dst_aligned_page_size = ((p.page_size + dst_alignment - 1) / dst_alignment) * dst_alignment;

    // Compute expected wait count for receiver:
    // - BasicWrite/Scatter: 1 (single atomic inc after all writes)
    // - FusedAtomicInc: NUM_PAGES (atomic inc fused with each write)
    uint32_t rx_wait_count = is_fused ? NUM_PAGES : 1;

    for (size_t i = 0; i < dst_coords.size(); ++i) {
        receiver_progs.emplace_back(tt::tt_metal::CreateProgram());
        auto rx_wait_k = tt::tt_metal::CreateKernel(
            receiver_progs.back(),
            std::string(KDIR) + "rx_addrgen.cpp",
            p.receiver_core,
            tt::tt_metal::DataMovementConfig{
                .processor = tt::tt_metal::DataMovementProcessor::RISCV_0,
                .noc = tt::tt_metal::NOC::RISCV_0_default,
                .defines = defines});
        tt::tt_metal::SetRuntimeArgs(
            receiver_progs.back(), rx_wait_k, p.receiver_core, {gsem_linear_mcast->address(), rx_wait_count});
    }

    // Ensure the same logical worker maps to the same physical XY across all receiver chips
    for (const auto& mc : dst_coords) {
        auto* dev_i = view.get_device(mc);
        auto xy_i = dev_i->worker_core_from_logical_core(p.receiver_core);
        if (xy_i != rx_xy) {
            ADD_FAILURE() << "Receiver worker XY mismatch across chips";
            return;
        }
    }

    // Sender program: READER (RISCV_0) + WRITER (RISCV_1)
    tt::tt_metal::Program sender_prog = tt::tt_metal::CreateProgram();
    const uint32_t CB_ID = tt::CBIndex::c_0;

    // For scatter operations, we need 2 pages per CB slot (read in pairs)
    uint32_t cb_page_size = is_scatter ? src_aligned_page_size * 2 : src_aligned_page_size;
    uint32_t num_cb_pages = is_scatter ? 4 : 8;  // Fewer but larger slots for scatter

    auto cb_cfg = tt::tt_metal::CircularBufferConfig(num_cb_pages * cb_page_size, {{CB_ID, tt::DataFormat::Float16}})
                      .set_page_size(CB_ID, cb_page_size);
    (void)tt::tt_metal::CreateCircularBuffer(sender_prog, p.sender_core, cb_cfg);

    // Reader kernel (DRAM->CB)
    std::vector<uint32_t> reader_cta;
    tt::tt_metal::TensorAccessorArgs(*src_buf).append_to(reader_cta);
    reader_cta.push_back(static_cast<uint32_t>(operation_type));  // OPERATION_TYPE
    reader_cta.push_back(1u);                                     // SRC_IS_DRAM
    reader_cta.push_back(NUM_PAGES);
    reader_cta.push_back(p.page_size);            // Raw page size
    reader_cta.push_back(src_aligned_page_size);  // Aligned page size

    auto reader_k = tt::tt_metal::CreateKernel(
        sender_prog,
        std::string(KDIR) + "tx_reader_to_cb_addrgen.cpp",
        p.sender_core,
        tt::tt_metal::DataMovementConfig{
            .processor = tt::tt_metal::DataMovementProcessor::RISCV_0,
            .noc = tt::tt_metal::NOC::RISCV_0_default,
            .compile_args = reader_cta,
            .defines = defines});
    tt::tt_metal::SetRuntimeArgs(sender_prog, reader_k, p.sender_core, {(uint32_t)src_buf->address()});

    // Map API variant for linear multicast
    auto api_variant = get_linear_multicast_api_variant(p.api_variant);

    // Writer kernel (linear multicast addrgen)
    std::vector<uint32_t> writer_cta;
    tt::tt_metal::TensorAccessorArgs(*dst_buf).append_to(writer_cta);
    writer_cta.push_back(static_cast<uint32_t>(operation_type));  // OPERATION_TYPE
    writer_cta.push_back(static_cast<uint32_t>(api_variant));     // API_VARIANT
    writer_cta.push_back(NUM_PAGES);                              // TOTAL_PAGES
    writer_cta.push_back(p.page_size);                            // Raw page size
    writer_cta.push_back(dst_aligned_page_size);                  // Aligned page size (for dst addr calc)
    writer_cta.push_back(src_aligned_page_size);                  // SRC_ALIGNED_PAGE_SIZE (for scatter CB stride)

    auto writer_k = tt::tt_metal::CreateKernel(
        sender_prog,
        std::string(KDIR) + "linear_multicast_tx_writer_addrgen.cpp",
        p.sender_core,
        tt::tt_metal::DataMovementConfig{
            .processor = tt::tt_metal::DataMovementProcessor::RISCV_1,
            .noc = tt::tt_metal::NOC::RISCV_1_default,
            .compile_args = writer_cta,
            .defines = defines});

    // Multicast parameters: start_distance=1 (skip sender), range=num_receivers
    uint8_t start_distance = 1;
    uint8_t range = static_cast<uint8_t>(num_receivers);

    std::vector<uint32_t> writer_rt = {
        (uint32_t)dst_buf->address(),            // 0: dst_base
        (uint32_t)rx_xy.x,                       // 1: rx_noc_x
        (uint32_t)rx_xy.y,                       // 2: rx_noc_y
        (uint32_t)gsem_linear_mcast->address(),  // 3: sem_l1_addr
        (uint32_t)start_distance,                // 4: start_distance
        (uint32_t)range                          // 5: range
    };

    // Get forwarding link for fabric connection (to first destination)
    auto forwarding_links = tt::tt_fabric::get_forwarding_link_indices(src, dst);
    if (forwarding_links.empty()) {
        ADD_FAILURE() << "No forwarding links from src to dst";
        return;
    }
    uint32_t link_idx = forwarding_links[0];

    // Pack fabric connection runtime args
    tt::tt_fabric::append_fabric_connection_rt_args(src, dst, link_idx, sender_prog, p.sender_core, writer_rt);

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
