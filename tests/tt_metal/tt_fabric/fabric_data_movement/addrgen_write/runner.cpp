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
#include "tt_metal/tt_fabric/benchmark/collectives/common/perf_helpers.hpp"
#include "tt_metal/fabric/fabric_context.hpp"
#include <tt-metalium/global_semaphore.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/mesh_device.hpp>
#include <tt-metalium/mesh_device_view.hpp>

namespace tt::tt_fabric::test {

// Import needed types from bench namespace
using tt::tt_fabric::bench::AddrgenApiVariant;
using tt::tt_fabric::bench::HelpersFixture;
using tt::tt_fabric::bench::PerfParams;

// ---------- helpers (validation / utilities) ----------

namespace {

// Lookup device by physical chip ID
inline tt::tt_metal::IDevice* find_device_by_id(ChipId phys_id) {
    auto devices = tt::DevicePool::instance().get_all_active_devices();
    for (auto* d : devices) {
        if (d->id() == phys_id) {
            return d;
        }
    }
    return nullptr;
}

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
    ChipId src_phys, ChipId dst_phys, tt::tt_metal::IDevice*& src_dev, tt::tt_metal::IDevice*& dst_dev) {
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
void run_addrgen_write_test(HelpersFixture* fixture, const PerfParams& p) {
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

    // ---------------------------- PROGRAM FACTORY ----------------------------
    /*
Unicast addrgen write test — top-level flow:

┌────────────────────────────┐                               ┌────────────────────────────┐
│ Device SRC (chip p.src)    │                               │ Device DST (chip p.dst)    │
│                            │                               │                            │
│  DRAM src_buf ──► Reader   │ pages →  L1 CB (c_0)  ──►     │  L1/DRAM dst_buf           │
│                 (RISCV_0)  │            ▲          │       │        ▲                   │
│                            │            │          │       │        │                   │
│       Writer (RISCV_1) ────┴────────────┴──────────┼──────►│  Receiver wait kernel      │
│       + fabric send adapter|            payload+hdr│       │  (RISCV_0) on GLOBAL sem   │
│       + ADDRGEN OVERLOAD   │                       │       │                            │
│                            │                       │       │                            │
│ after last page: send      │                       │       │ fabric delivers all data   │
│ atomic_inc to dst.sem ─────┼───────────────────────┼──-───►│ then sem++ → receiver exit │
└────────────────────────────┘      (Fabric link)            └────────────────────────────┘

Flow:
1) Reader DMA-batches DRAM → CB. 2) Writer drains CB, sends packets over fabric using addrgen overload.
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

    // Determine kernel directory based on API variant
    const bool is_fused_atomic_inc =
        (p.api_variant == AddrgenApiVariant::FusedAtomicIncWrite ||
         p.api_variant == AddrgenApiVariant::FusedAtomicIncWriteWithState ||
         p.api_variant == AddrgenApiVariant::FusedAtomicIncWriteSetState);
    const bool is_scatter =
        (p.api_variant == AddrgenApiVariant::ScatterWrite ||
         p.api_variant == AddrgenApiVariant::ScatterWriteWithState ||
         p.api_variant == AddrgenApiVariant::ScatterWriteSetState);

    std::string KDIR;
    if (is_fused_atomic_inc) {
        KDIR = "tests/tt_metal/tt_fabric/fabric_data_movement/addrgen_write/kernels/fused_atomic_inc/";
    } else if (is_scatter) {
        KDIR = "tests/tt_metal/tt_fabric/fabric_data_movement/addrgen_write/kernels/scatter/";
    } else {
        KDIR = "tests/tt_metal/tt_fabric/fabric_data_movement/addrgen_write/kernels/unicast/";
    }

    // Helper to select writer kernel based on API variant
    auto get_writer_kernel_path = [&KDIR](AddrgenApiVariant variant) -> std::string {
        switch (variant) {
            case AddrgenApiVariant::UnicastWrite: return KDIR + "unicast_tx_writer_cb_to_dst_addrgen.cpp";
            case AddrgenApiVariant::UnicastWriteWithState: return KDIR + "unicast_tx_writer_with_state_addrgen.cpp";
            case AddrgenApiVariant::UnicastWriteSetState: return KDIR + "unicast_tx_writer_set_state_addrgen.cpp";
            case AddrgenApiVariant::ScatterWrite: return KDIR + "scatter_tx_writer_addrgen.cpp";
            case AddrgenApiVariant::ScatterWriteWithState: return KDIR + "scatter_tx_writer_with_state_addrgen.cpp";
            case AddrgenApiVariant::ScatterWriteSetState: return KDIR + "scatter_tx_writer_set_state_addrgen.cpp";
            case AddrgenApiVariant::FusedAtomicIncWrite: return KDIR + "fused_atomic_inc_tx_writer_addrgen.cpp";
            case AddrgenApiVariant::FusedAtomicIncWriteWithState:
                return KDIR + "fused_atomic_inc_tx_writer_with_state_addrgen.cpp";
            case AddrgenApiVariant::FusedAtomicIncWriteSetState:
                return KDIR + "fused_atomic_inc_tx_writer_set_state_addrgen.cpp";
            default: TT_FATAL(false, "Unknown API variant"); return "";
        }
    };

    std::string receiver_kernel_name;
    if (is_fused_atomic_inc) {
        receiver_kernel_name = "fused_atomic_inc_rx_addrgen.cpp";
    } else if (is_scatter) {
        receiver_kernel_name = "scatter_rx_addrgen.cpp";
    } else {
        receiver_kernel_name = "unicast_rx_addrgen.cpp";
    }

    auto rx_wait_k = tt::tt_metal::CreateKernel(
        receiver_prog,
        KDIR + receiver_kernel_name,
        receiver_core,
        tt::tt_metal::DataMovementConfig{
            .processor = tt::tt_metal::DataMovementProcessor::RISCV_0,
            .noc = tt::tt_metal::NOC::RISCV_0_default,
            .defines = defines});

    const uint32_t NUM_PAGES = (p.tensor_bytes + p.page_size - 1) / p.page_size;

    // For fused atomic inc, each write increments the semaphore, so receiver waits for NUM_PAGES
    // For regular unicast, a single atomic inc is sent after all writes, so receiver waits for 1
    const uint32_t sem_wait_value = is_fused_atomic_inc ? NUM_PAGES : 1u;
    tt::tt_metal::SetRuntimeArgs(receiver_prog, rx_wait_k, receiver_core, {gsem->address(), sem_wait_value});

    // Sender program: READER (RISCV_0) + WRITER (RISCV_1)
    tt::tt_metal::Program sender_prog = tt::tt_metal::CreateProgram();
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

    std::string reader_kernel_name;
    if (is_fused_atomic_inc) {
        reader_kernel_name = "fused_atomic_inc_tx_reader_to_cb_addrgen.cpp";
    } else if (is_scatter) {
        reader_kernel_name = "scatter_tx_reader_to_cb_addrgen.cpp";
    } else {
        reader_kernel_name = "unicast_tx_reader_to_cb_addrgen.cpp";
    }

    auto reader_k = tt::tt_metal::CreateKernel(
        sender_prog,
        KDIR + reader_kernel_name,
        p.sender_core,
        tt::tt_metal::DataMovementConfig{
            .processor = tt::tt_metal::DataMovementProcessor::RISCV_0,
            .noc = tt::tt_metal::NOC::RISCV_0_default,
            .compile_args = reader_cta,
            .defines = defines});
    tt::tt_metal::SetRuntimeArgs(sender_prog, reader_k, p.sender_core, {(uint32_t)src_buf->address()});

    // Writer kernel (CB->Fabric->dst + final sem INC)
    std::vector<uint32_t> writer_cta;
    tt::tt_metal::TensorAccessorArgs(*dst_buf).append_to(writer_cta);
    writer_cta.push_back(NUM_PAGES);
    writer_cta.push_back(p.page_size);

    auto writer_k = tt::tt_metal::CreateKernel(
        sender_prog,
        get_writer_kernel_path(p.api_variant),  // Use helper to select kernel
        p.sender_core,
        tt::tt_metal::DataMovementConfig{
            .processor = tt::tt_metal::DataMovementProcessor::RISCV_1,
            .noc = tt::tt_metal::NOC::RISCV_1_default,
            .compile_args = writer_cta,
            .defines = defines});

    // Resolve forwarding and append fabric connection args
    uint32_t link_idx = 0;
    if (!pick_forwarding_link_or_fail(src, dst, link_idx, p)) {
        return;
    }

    std::vector<uint32_t> writer_rt = {
        (uint32_t)dst_buf->address(),  // 0: dst_base (receiver L1 offset)
        (uint32_t)p.mesh_id,           // 1: dst_mesh_id (logical)
        (uint32_t)p.dst_chip,          // 2: dst_dev_id  (logical)
        (uint32_t)rx_xy.x,             // 3: receiver_noc_x
        (uint32_t)rx_xy.y,             // 4: receiver_noc_y
        (uint32_t)gsem->address()      // 5: receiver L1 semaphore addr
    };

    // Pack the fabric-connection runtime args for the writer kernel.
    // This establishes the send path (routing/link identifiers) for fabric traffic.
    // The device kernel must unpack these in the same order via build_from_args(...).
    tt::tt_fabric::append_fabric_connection_rt_args(
        src, dst, /*link_idx=*/link_idx, sender_prog, p.sender_core, writer_rt);
    tt::tt_metal::SetRuntimeArgs(sender_prog, writer_k, p.sender_core, writer_rt);
    // -------------------------- end PROGRAM FACTORY --------------------------

    // --- Simple single-run execution ---
    Dist::MeshWorkload receiver_workload;
    Dist::MeshWorkload sender_workload;
    receiver_workload.add_program(Dist::MeshCoordinateRange(dst_coord), std::move(receiver_prog));
    sender_workload.add_program(Dist::MeshCoordinateRange(src_coord), std::move(sender_prog));

    // Run once: receiver first (so it's ready), then sender
    Dist::EnqueueMeshWorkload(mcq, receiver_workload, /*blocking=*/false);
    Dist::EnqueueMeshWorkload(mcq, sender_workload, /*blocking=*/true);

    // Read back (single shard) and verify
    std::vector<uint32_t> rx(n_words, 0u);
    Dist::ReadShard(mcq, rx, dst_buf, dst_coord, /*blocking=*/true);
    verify_payload_words(rx, tx);
}

}  // namespace tt::tt_fabric::test
