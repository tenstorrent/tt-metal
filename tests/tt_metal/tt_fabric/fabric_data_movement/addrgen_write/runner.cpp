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
#include "tests/tt_metal/tt_fabric/fabric_data_movement/runner_common.hpp"
#include "tt_metal/tt_fabric/benchmark/collectives/common/perf_helpers.hpp"
#include "tt_metal/fabric/fabric_context.hpp"
#include <tt-metalium/global_semaphore.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/mesh_device.hpp>
#include <tt-metalium/mesh_device_view.hpp>
#include <distributed/mesh_device_view_impl.hpp>

namespace tt::tt_fabric::test {

// ---------- helpers (validation / utilities) ----------

namespace {

// Resolve forwarding link and fail early if none found.
inline bool pick_forwarding_link_or_fail(
    const tt::tt_fabric::FabricNodeId& /*src*/,
    const tt::tt_fabric::FabricNodeId& /*dst*/,
    uint32_t& out_link_idx,
    const AddrgenTestParams& p) {
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

// Convert FabricTestVariant fields to kernel-level enums
inline OperationType to_kernel_operation_type(WriteOp op) {
    switch (op) {
        case WriteOp::Write: return OperationType::BasicWrite;
        case WriteOp::Scatter: return OperationType::Scatter;
        case WriteOp::FusedAtomicInc: return OperationType::FusedAtomicInc;
        default: return OperationType::BasicWrite;
    }
}

inline ApiVariant to_kernel_api_variant(StateMode state, ConnectionMode conn) {
    if (conn == ConnectionMode::ConnMgr) {
        switch (state) {
            case StateMode::Stateless: return ApiVariant::ConnMgrBasic;
            case StateMode::WithState: return ApiVariant::ConnMgrWithState;
            case StateMode::SetState: return ApiVariant::ConnMgrSetState;
        }
    }
    switch (state) {
        case StateMode::Stateless: return ApiVariant::Basic;
        case StateMode::WithState: return ApiVariant::WithState;
        case StateMode::SetState: return ApiVariant::SetState;
    }
    return ApiVariant::Basic;
}

}  // anonymous namespace

// ----------------------------------- unicast program -----------------------------------
void run_unicast_write_test(tt::tt_metal::MeshDeviceFixtureBase* fixture, const AddrgenTestParams& p) {
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

    if (!validate_word_alignment_or_fail(p.tensor_bytes)) {
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

    // Check if this is a connection manager variant (needed for second destination setup)
    const bool is_conn_mgr_variant_from_params = p.variant.is_conn_mgr();

    // For connection manager variants: pick a second destination from the mesh
    // (any device that isn't src or dst)
    ChipId dst2_phys = 0;
    tt::tt_metal::IDevice* dst2_dev = nullptr;
    Dist::MeshCoordinate dst2_coord{0, 0};
    tt::tt_metal::CoreCoord rx2_xy = rx_xy;
    if (is_conn_mgr_variant_from_params) {
        for (const auto& c : Dist::MeshCoordinateRange(view.shape())) {
            auto* dev = view.impl().get_device(c);
            if (!dev || dev->id() == src_phys || dev->id() == dst_phys) {
                continue;
            }
            dst2_phys = dev->id();
            dst2_dev = dev;
            dst2_coord = c;
            rx2_xy = dev->worker_core_from_logical_core(p.receiver_core);
            break;
        }
    }

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
    // For connection manager variants: initialize second destination buffer
    if (is_conn_mgr_variant_from_params && dst2_dev) {
        Dist::WriteShard(mcq, dst_buf, zeros, dst2_coord, /*blocking=*/true);
    }

    // ---------------------------- PROGRAM FACTORY ----------------------------
    /*
Unicast addrgen write test -- top-level flow:

+----------------------------+                               +----------------------------+
| Device SRC (chip p.src)    |                               | Device DST (chip p.dst)    |
|                            |                               |                            |
|  DRAM src_buf --> Reader   | pages ->  L1 CB (c_0)  -->     |  L1/DRAM dst_buf           |
|                 (RISCV_0)  |            ^          |       |        ^                   |
|                            |            |          |       |        |                   |
|       Writer (RISCV_1) ----+------------+----------+------>|  Receiver wait kernel      |
|       + fabric send adapter|            payload+hdr|       |  (RISCV_0) on GLOBAL sem   |
|       + ADDRGEN OVERLOAD   |                       |       |                            |
|                            |                       |       |                            |
| after last page: send      |                       |       | fabric delivers all data   |
| atomic_inc to dst.sem -----+--------------------->-+------>| then sem++ -> receiver exit |
+----------------------------+      (Fabric link)            +----------------------------+

Flow:
1) Reader DMA-batches DRAM -> CB. 2) Writer drains CB, sends packets over fabric using addrgen overload.
3) Writer finally sends a semaphore INC to DST. Fabric orders this after payloads.
4) Receiver sees sem++ and returns. Host verifies bytes.

Notes:
- Uses fabric_unicast_noc_unicast_write overload with TensorAccessor for cleaner code.
- We use a GLOBAL semaphore on DST so a different core can observe the signal.
- Route setup uses current 2D API. This will change soon. The 1D reference shape is in linear/api.h.
*/

    // Global semaphore so a remote chip can signal it.
    // Fabric guarantees payload is visible before the bump is seen.
    tt::tt_metal::Program receiver_prog = tt::tt_metal::CreateProgram();
    // Create the semaphore on the specific receiver logical core of the *mesh*.
    tt::tt_metal::CoreRangeSet rx_core_set(tt::tt_metal::CoreRange(p.receiver_core, p.receiver_core));
    static std::optional<tt::tt_metal::GlobalSemaphore> gsem;
    if (!gsem) {
        gsem = tt::tt_metal::CreateGlobalSemaphore(
            mesh.get(),
            rx_core_set,
            /*initial_value=*/0);
    }

    const tt::tt_metal::CoreCoord receiver_core = p.receiver_core;

    // Determine if this is a fused atomic inc variant
    const bool is_fused_atomic_inc = p.variant.is_fused();

    const std::string KDIR = "tests/tt_metal/tt_fabric/fabric_data_movement/addrgen_write/kernels/";

    auto operation_type = to_kernel_operation_type(p.variant.op);
    auto api_variant = to_kernel_api_variant(p.variant.state, p.variant.conn);

    // All receivers use the unified kernel now
    const std::string receiver_kernel_name = "rx_addrgen.cpp";

    auto rx_wait_k = tt::tt_metal::CreateKernel(
        receiver_prog,
        KDIR + receiver_kernel_name,
        receiver_core,
        tt::tt_metal::DataMovementConfig{
            .processor = tt::tt_metal::DataMovementProcessor::RISCV_0,
            .noc = tt::tt_metal::NOC::RISCV_0_default,
            .defines = defines});

    const uint32_t NUM_PAGES = (p.tensor_bytes + p.page_size - 1) / p.page_size;

    // Calculate aligned page sizes for source (DRAM) and destination (DRAM or L1)
    const auto& hal = tt::tt_metal::MetalContext::instance().hal();
    uint32_t src_alignment = hal.get_alignment(tt::tt_metal::HalMemType::DRAM);  // Source is always DRAM
    uint32_t dst_alignment = p.use_dram_dst ? hal.get_alignment(tt::tt_metal::HalMemType::DRAM)
                                            : hal.get_alignment(tt::tt_metal::HalMemType::L1);

    // Round up to alignment boundary
    uint32_t src_aligned_page_size = ((p.page_size + src_alignment - 1) / src_alignment) * src_alignment;
    uint32_t dst_aligned_page_size = ((p.page_size + dst_alignment - 1) / dst_alignment) * dst_alignment;

    // For fused atomic inc, each write increments the semaphore, so receiver waits for NUM_PAGES
    // For regular unicast, a single atomic inc is sent after all writes, so receiver waits for 1
    const uint32_t sem_wait_value = is_fused_atomic_inc ? NUM_PAGES : 1u;
    tt::tt_metal::SetRuntimeArgs(receiver_prog, rx_wait_k, receiver_core, {gsem->address(), sem_wait_value});

    // For connection manager variants: set up second receiver program
    tt::tt_metal::Program receiver_prog2 = tt::tt_metal::CreateProgram();
    if (is_conn_mgr_variant_from_params && dst2_dev) {
        auto rx_wait_k2 = tt::tt_metal::CreateKernel(
            receiver_prog2,
            KDIR + receiver_kernel_name,
            p.receiver_core,
            tt::tt_metal::DataMovementConfig{
                .processor = tt::tt_metal::DataMovementProcessor::RISCV_0,
                .noc = tt::tt_metal::NOC::RISCV_0_default,
                .defines = defines});
        tt::tt_metal::SetRuntimeArgs(receiver_prog2, rx_wait_k2, p.receiver_core, {gsem->address(), sem_wait_value});
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
        KDIR + "tx_reader_to_cb_addrgen.cpp",  // Unified reader kernel
        p.sender_core,
        tt::tt_metal::DataMovementConfig{
            .processor = tt::tt_metal::DataMovementProcessor::RISCV_0,
            .noc = tt::tt_metal::NOC::RISCV_0_default,
            .compile_args = reader_cta,
            .defines = defines});
    tt::tt_metal::SetRuntimeArgs(sender_prog, reader_k, p.sender_core, {(uint32_t)src_buf->address()});

    // Writer kernel (CB->Fabric->dst + final sem INC) - select kernel based on connection manager variant
    std::vector<uint32_t> writer_cta;
    tt::tt_metal::TensorAccessorArgs(*dst_buf).append_to(writer_cta);
    writer_cta.push_back(static_cast<uint32_t>(operation_type));  // OPERATION_TYPE
    writer_cta.push_back(static_cast<uint32_t>(api_variant));     // API_VARIANT
    writer_cta.push_back(NUM_PAGES);       // TOTAL_PAGES
    writer_cta.push_back(p.page_size);     // Raw page size (actual data size to transfer)
    writer_cta.push_back(dst_aligned_page_size);  // Aligned page size (dest buffer addressing)
    writer_cta.push_back(src_aligned_page_size);  // Source aligned page size (CB stride for scatter)

    // Select kernel based on connection manager variant
    const std::string writer_kernel_name =
        p.variant.is_conn_mgr() ? "unicast_tx_writer_addrgen_conn_mgr.cpp" : "unicast_tx_writer_addrgen.cpp";

    auto writer_k = tt::tt_metal::CreateKernel(
        sender_prog,
        KDIR + writer_kernel_name,
        p.sender_core,
        tt::tt_metal::DataMovementConfig{
            .processor = tt::tt_metal::DataMovementProcessor::RISCV_1,
            .noc = tt::tt_metal::NOC::RISCV_1_default,
            .compile_args = writer_cta,
            .defines = defines});

    std::vector<uint32_t> writer_rt = {
        (uint32_t)dst_buf->address(),  // 0: dst_base (receiver L1 offset)
        (uint32_t)p.mesh_id,           // 1: dst_mesh_id (logical)
        (uint32_t)p.dst_chip,          // 2: dst_dev_id  (logical)
        (uint32_t)rx_xy.x,             // 3: receiver_noc_x
        (uint32_t)rx_xy.y,             // 4: receiver_noc_y
        (uint32_t)gsem->address()      // 5: receiver L1 semaphore addr
    };

    if (p.variant.is_conn_mgr()) {
        // Connection manager variant: use routing plane connection manager
        // Add num_connections (2 for dual destination test, fallback to 1 if dst2_dev is null)
        uint32_t num_connections = (dst2_dev != nullptr) ? 2u : 1u;
        writer_rt.push_back(num_connections);

        // Use append_routing_plane_connection_manager_rt_args for route setup
        auto dst2_fn = cp.get_fabric_node_id_from_physical_chip_id(dst2_phys);
        std::vector<tt::tt_fabric::FabricNodeId> dst_nodes = (dst2_dev != nullptr)
                                                                 ? std::vector<tt::tt_fabric::FabricNodeId>{dst, dst2_fn}
                                                                 : std::vector<tt::tt_fabric::FabricNodeId>{dst};
        std::vector<uint32_t> connection_link_indices = {};  // Empty means auto-select
        tt::tt_fabric::append_routing_plane_connection_manager_rt_args(
            src,
            dst_nodes,
            connection_link_indices,
            sender_prog,
            writer_k,
            p.sender_core,
            writer_rt,
            tt::tt_fabric::FabricApiType::Mesh,
            CoreType::WORKER);
    } else {
        // Basic variant: use regular fabric connection args
        uint32_t link_idx = 0;
        if (!pick_forwarding_link_or_fail(src, dst, link_idx, p)) {
            return;
        }
        // Pack the fabric-connection runtime args for the writer kernel.
        // This establishes the send path (routing/link identifiers) for fabric traffic.
        // The device kernel must unpack these in the same order via build_from_args(...).
        tt::tt_fabric::append_fabric_connection_rt_args(
            src, dst, /*link_idx=*/link_idx, sender_prog, p.sender_core, writer_rt);
    }
    tt::tt_metal::SetRuntimeArgs(sender_prog, writer_k, p.sender_core, writer_rt);
    // -------------------------- end PROGRAM FACTORY --------------------------

    // --- Simple single-run execution ---
    Dist::MeshWorkload receiver_workload;
    Dist::MeshWorkload sender_workload;
    receiver_workload.add_program(Dist::MeshCoordinateRange(dst_coord), std::move(receiver_prog));
    // For connection manager variants: add second receiver to workload
    if (is_conn_mgr_variant_from_params && dst2_dev) {
        receiver_workload.add_program(Dist::MeshCoordinateRange(dst2_coord), std::move(receiver_prog2));
    }
    sender_workload.add_program(Dist::MeshCoordinateRange(src_coord), std::move(sender_prog));

    // Run once: receiver first (so it's ready), then sender
    Dist::EnqueueMeshWorkload(mcq, receiver_workload, /*blocking=*/false);
    Dist::EnqueueMeshWorkload(mcq, sender_workload, /*blocking=*/true);

    // Read back (single shard) and verify
    std::vector<uint32_t> rx(n_words, 0u);
    Dist::ReadShard(mcq, rx, dst_buf, dst_coord, /*blocking=*/true);
    verify_payload_words(rx, tx);

    // For connection manager variants: verify second destination
    if (is_conn_mgr_variant_from_params && dst2_dev) {
        std::vector<uint32_t> rx2(n_words, 0u);
        Dist::ReadShard(mcq, rx2, dst_buf, dst2_coord, /*blocking=*/true);
        verify_payload_words(rx2, tx);
    }
}

// ----------------------------------- multicast program -----------------------------------
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

    if (!validate_word_alignment_or_fail(p.tensor_bytes)) {
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

    // One semaphore at a single logical core -> same L1 offset on every chip in the MeshDevice
    static std::optional<tt::tt_metal::GlobalSemaphore> gsem_done;
    if (!gsem_done) {
        tt::tt_metal::CoreRangeSet rx_core_one(tt::tt_metal::CoreRange(p.receiver_core, p.receiver_core));
        gsem_done = tt::tt_metal::CreateGlobalSemaphore(mesh.get(), rx_core_one, /*initial_value=*/0);
    }

    constexpr const char* KDIR = "tests/tt_metal/tt_fabric/fabric_data_movement/addrgen_write/kernels/";

    auto operation_type = to_kernel_operation_type(p.variant.op);
    auto api_variant = to_kernel_api_variant(p.variant.state, p.variant.conn);

    const bool is_fused_atomic_inc = p.variant.is_fused();
    const bool is_conn_mgr_variant = p.variant.is_conn_mgr();

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

    // Writer kernel (CB->Fabric->dst + final sem INC) - select kernel based on connection manager variant and operation
    // type
    std::vector<uint32_t> writer_cta;
    tt::tt_metal::TensorAccessorArgs(*dst_buf).append_to(writer_cta);

    // Both conn_mgr and non-conn_mgr use unified kernels with same CT arg layout
    writer_cta.push_back(static_cast<uint32_t>(operation_type));  // OPERATION_TYPE
    writer_cta.push_back(static_cast<uint32_t>(api_variant));     // API_VARIANT
    writer_cta.push_back(NUM_PAGES);                              // TOTAL_PAGES
    writer_cta.push_back(p.page_size);                            // Raw page size (actual data size to transfer)
    writer_cta.push_back(dst_aligned_page_size);                  // Aligned page size (dest buffer addressing)
    writer_cta.push_back(src_aligned_page_size);                  // Source aligned page size (CB stride for scatter)

    const std::string writer_kernel_name =
        is_conn_mgr_variant ? "multicast_tx_writer_addrgen_conn_mgr.cpp" : "multicast_tx_writer_addrgen.cpp";

    auto writer_k = tt::tt_metal::CreateKernel(
        sender_prog,
        std::string(KDIR) + writer_kernel_name,
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

    // Direction bitmask (same for both connection manager and non-connection manager variants)
    const uint32_t dir_mask = (w_hops ? 1u : 0u) | (e_hops ? 2u : 0u) | (n_hops ? 4u : 0u) | (s_hops ? 8u : 0u);
    writer_rt.push_back(dir_mask);

    // Per-direction representatives and hops, ordered W, E, N, S
    const Dist::MeshCoordinate reps[4] = {rep_w, rep_e, rep_n, rep_s};
    const uint16_t dir_hops[4] = {w_hops, e_hops, n_hops, s_hops};

    if (is_conn_mgr_variant) {
        auto append_conn_mgr_for_dir = [&](uint16_t hops, const Dist::MeshCoordinate& rep) {
            if (!hops) {
                return;
            }
            tt::tt_fabric::append_routing_plane_connection_manager_rt_args(
                src,
                {coord_to_fabric_id(rep)},
                {},
                sender_prog,
                writer_k,
                p.sender_core,
                writer_rt,
                tt::tt_fabric::FabricApiType::Mesh,
                CoreType::WORKER);
        };
        for (uint32_t d = 0; d < 4; ++d) {
            append_conn_mgr_for_dir(dir_hops[d], reps[d]);
        }
    } else {
        auto pick_link = [&](const Dist::MeshCoordinate& mc, uint32_t& out_link_idx) {
            auto dst_fn = coord_to_fabric_id(mc);
            auto links = tt::tt_fabric::get_forwarding_link_indices(src_fn, dst_fn);
            if (links.empty()) {
                ADD_FAILURE() << "No forwarding link from src to representative";
                return false;
            }
            out_link_idx = links[0];
            return true;
        };

        uint32_t link_idx[4] = {};
        for (uint32_t d = 0; d < 4; ++d) {
            if (dir_hops[d] && !pick_link(reps[d], link_idx[d])) {
                return;
            }
        }
        for (uint32_t d = 0; d < 4; ++d) {
            if (dir_hops[d]) {
                tt::tt_fabric::append_fabric_connection_rt_args(
                    src_fn, coord_to_fabric_id(reps[d]), link_idx[d], sender_prog, p.sender_core, writer_rt);
            }
        }
    }

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

// ----------------------------------- Linear (1D) unicast program -----------------------------------
void run_linear_unicast_write_test(tt::tt_metal::MeshDeviceFixtureBase* fixture, const AddrgenTestParams& p) {
    const auto& cp = tt::tt_metal::MetalContext::instance().get_control_plane();
    namespace Dist = tt::tt_metal::distributed;

    // Linear API test does NOT use FABRIC_2D define
    std::map<std::string, std::string> defines = {};

    tt::tt_fabric::FabricNodeId src{tt::tt_fabric::MeshId{p.mesh_id}, p.src_chip};
    tt::tt_fabric::FabricNodeId dst{tt::tt_fabric::MeshId{p.mesh_id}, p.dst_chip};

    ChipId src_phys = cp.get_physical_chip_id_from_fabric_node_id(src);
    ChipId dst_phys = cp.get_physical_chip_id_from_fabric_node_id(dst);

    tt::tt_metal::IDevice* src_dev = nullptr;
    tt::tt_metal::IDevice* dst_dev = nullptr;
    if (!lookup_devices_or_fail(src_phys, dst_phys, src_dev, dst_dev)) {
        return;
    }

    if (!validate_word_alignment_or_fail(p.tensor_bytes)) {
        return;
    }

    tt::tt_metal::CoreCoord rx_xy = dst_dev->worker_core_from_logical_core(p.receiver_core);

    // --- Mesh device + coords for per-shard IO ---
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

    // --- Global semaphore ---
    tt::tt_metal::Program receiver_prog = tt::tt_metal::CreateProgram();
    tt::tt_metal::CoreRangeSet rx_core_set(tt::tt_metal::CoreRange(p.receiver_core, p.receiver_core));
    static std::optional<tt::tt_metal::GlobalSemaphore> gsem_linear;
    if (!gsem_linear) {
        gsem_linear = tt::tt_metal::CreateGlobalSemaphore(mesh.get(), rx_core_set, /*initial_value=*/0);
    }

    const std::string KDIR = "tests/tt_metal/tt_fabric/fabric_data_movement/addrgen_write/kernels/";

    // Receiver kernel
    auto rx_wait_k = tt::tt_metal::CreateKernel(
        receiver_prog,
        KDIR + "rx_addrgen.cpp",
        p.receiver_core,
        tt::tt_metal::DataMovementConfig{
            .processor = tt::tt_metal::DataMovementProcessor::RISCV_0,
            .noc = tt::tt_metal::NOC::RISCV_0_default,
            .defines = defines});

    const uint32_t NUM_PAGES = (p.tensor_bytes + p.page_size - 1) / p.page_size;

    // Calculate aligned page sizes
    const auto& hal = tt::tt_metal::MetalContext::instance().hal();
    uint32_t src_alignment = hal.get_alignment(tt::tt_metal::HalMemType::DRAM);
    uint32_t dst_alignment = p.use_dram_dst ? hal.get_alignment(tt::tt_metal::HalMemType::DRAM)
                                            : hal.get_alignment(tt::tt_metal::HalMemType::L1);
    uint32_t src_aligned_page_size = ((p.page_size + src_alignment - 1) / src_alignment) * src_alignment;
    uint32_t dst_aligned_page_size = ((p.page_size + dst_alignment - 1) / dst_alignment) * dst_alignment;

    // Determine if this is a fused atomic inc variant
    const bool is_fused_atomic_inc = p.variant.is_fused();

    // For fused atomic inc, each write increments the semaphore, so receiver waits for NUM_PAGES
    // For regular unicast, a single atomic inc is sent after all writes, so receiver waits for 1
    const uint32_t sem_wait_value = is_fused_atomic_inc ? NUM_PAGES : 1u;
    tt::tt_metal::SetRuntimeArgs(receiver_prog, rx_wait_k, p.receiver_core, {gsem_linear->address(), sem_wait_value});

    // Sender program: READER (RISCV_0) + WRITER (RISCV_1)
    tt::tt_metal::Program sender_prog = tt::tt_metal::CreateProgram();
    const uint32_t CB_ID = tt::CBIndex::c_0;
    auto cb_cfg = tt::tt_metal::CircularBufferConfig(8 * src_aligned_page_size, {{CB_ID, tt::DataFormat::Float16}})
                      .set_page_size(CB_ID, src_aligned_page_size);
    (void)tt::tt_metal::CreateCircularBuffer(sender_prog, p.sender_core, cb_cfg);

    auto operation_type = to_kernel_operation_type(p.variant.op);
    auto api_variant = to_kernel_api_variant(p.variant.state, p.variant.conn);

    // Reader kernel (DRAM->CB) - uses unified kernel with OPERATION_TYPE compile-time arg
    std::vector<uint32_t> reader_cta;
    tt::tt_metal::TensorAccessorArgs(*src_buf).append_to(reader_cta);
    reader_cta.push_back(static_cast<uint32_t>(operation_type));  // OPERATION_TYPE
    reader_cta.push_back(1u);                                     // SRC_IS_DRAM
    reader_cta.push_back(NUM_PAGES);
    reader_cta.push_back(p.page_size);            // Raw page size (actual data size to transfer)
    reader_cta.push_back(src_aligned_page_size);  // Aligned page size (source buffer spacing)

    auto reader_k = tt::tt_metal::CreateKernel(
        sender_prog,
        KDIR + "tx_reader_to_cb_addrgen.cpp",  // Unified reader kernel
        p.sender_core,
        tt::tt_metal::DataMovementConfig{
            .processor = tt::tt_metal::DataMovementProcessor::RISCV_0,
            .noc = tt::tt_metal::NOC::RISCV_0_default,
            .compile_args = reader_cta,
            .defines = defines});
    tt::tt_metal::SetRuntimeArgs(sender_prog, reader_k, p.sender_core, {(uint32_t)src_buf->address()});

    // Writer kernel (linear addrgen)
    std::vector<uint32_t> writer_cta;
    tt::tt_metal::TensorAccessorArgs(*dst_buf).append_to(writer_cta);
    writer_cta.push_back(static_cast<uint32_t>(operation_type));  // OPERATION_TYPE
    writer_cta.push_back(static_cast<uint32_t>(api_variant));     // API_VARIANT
    writer_cta.push_back(NUM_PAGES);                              // TOTAL_PAGES
    writer_cta.push_back(p.page_size);                            // Raw page size (actual data size to transfer)
    writer_cta.push_back(dst_aligned_page_size);                  // Aligned page size (dest buffer addressing)
    writer_cta.push_back(src_aligned_page_size);                  // Source aligned page size (CB stride for scatter)

    // Check if this is a connection manager variant
    const bool is_linear_conn_mgr = p.variant.is_conn_mgr();

    // Select kernel based on connection manager variant
    const std::string writer_kernel_name =
        is_linear_conn_mgr ? "linear_unicast_tx_writer_addrgen_conn_mgr.cpp" : "linear_unicast_tx_writer_addrgen.cpp";

    auto writer_k = tt::tt_metal::CreateKernel(
        sender_prog,
        KDIR + writer_kernel_name,
        p.sender_core,
        tt::tt_metal::DataMovementConfig{
            .processor = tt::tt_metal::DataMovementProcessor::RISCV_1,
            .noc = tt::tt_metal::NOC::RISCV_1_default,
            .compile_args = writer_cta,
            .defines = defines});

    // Get forwarding link for fabric connection
    auto forwarding_links = tt::tt_fabric::get_forwarding_link_indices(src, dst);
    if (forwarding_links.empty()) {
        ADD_FAILURE() << "No forwarding links from src to dst";
        return;
    }
    uint32_t link_idx = forwarding_links[0];

    // For linear fabric, num_hops is a test parameter (like other linear tests)
    // Hardcoded to 1 for simple single-hop test between adjacent chips
    uint32_t num_hops = 1;

    std::vector<uint32_t> writer_rt = {
        (uint32_t)dst_buf->address(),     // 0: dst_base
        (uint32_t)rx_xy.x,                // 1: receiver_noc_x
        (uint32_t)rx_xy.y,                // 2: receiver_noc_y
        (uint32_t)gsem_linear->address()  // 3: receiver L1 semaphore addr
    };

    if (is_linear_conn_mgr) {
        // Connection manager variant: pack num_connections and num_hops array
        uint32_t num_connections = 1;          // Single destination for now
        writer_rt.push_back(num_connections);  // 4: num_connections
        writer_rt.push_back(num_hops);         // 5: num_hops[0]

        // Use routing plane connection manager args
        std::vector<tt::tt_fabric::FabricNodeId> dst_nodes = {dst};
        std::vector<uint32_t> connection_link_indices = {};  // Empty means auto-select
        tt::tt_fabric::append_routing_plane_connection_manager_rt_args(
            src,
            dst_nodes,
            connection_link_indices,
            sender_prog,
            writer_k,
            p.sender_core,
            writer_rt,
            tt::tt_fabric::FabricApiType::Linear,
            CoreType::WORKER);
    } else {
        // Basic variant: use regular fabric connection args
        writer_rt.push_back(num_hops);  // 4: num_hops for linear unicast
        tt::tt_fabric::append_fabric_connection_rt_args(src, dst, link_idx, sender_prog, p.sender_core, writer_rt);
    }

    tt::tt_metal::SetRuntimeArgs(sender_prog, writer_k, p.sender_core, writer_rt);

    // --- Execution ---
    Dist::MeshWorkload receiver_workload;
    Dist::MeshWorkload sender_workload;
    receiver_workload.add_program(Dist::MeshCoordinateRange(dst_coord), std::move(receiver_prog));
    sender_workload.add_program(Dist::MeshCoordinateRange(src_coord), std::move(sender_prog));

    Dist::EnqueueMeshWorkload(mcq, receiver_workload, /*blocking=*/false);
    Dist::EnqueueMeshWorkload(mcq, sender_workload, /*blocking=*/true);

    // Read back and verify
    std::vector<uint32_t> rx(n_words, 0u);
    Dist::ReadShard(mcq, rx, dst_buf, dst_coord, /*blocking=*/true);
    verify_payload_words(rx, tx);
}

// ----------------------------------- Linear (1D) multicast program -----------------------------------
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

    if (!validate_word_alignment_or_fail(p.tensor_bytes)) {
        return;
    }

    tt::tt_metal::CoreCoord rx_xy = dst_dev->worker_core_from_logical_core(p.receiver_core);

    // Determine operation type
    auto operation_type = to_kernel_operation_type(p.variant.op);
    bool is_scatter = p.variant.is_scatter();
    bool is_fused = p.variant.is_fused();

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

    // One semaphore at a single logical core -> same L1 offset on every chip in the MeshDevice
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
    auto api_variant = to_kernel_api_variant(p.variant.state, p.variant.conn);

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
