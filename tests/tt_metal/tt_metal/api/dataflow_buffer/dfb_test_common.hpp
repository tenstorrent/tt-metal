// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <map>
#include <memory>
#include <numeric>
#include <optional>
#include <string>
#include <tuple>
#include <utility>
#include <vector>
#include <gtest/gtest.h>
#include <tt-metalium/buffer.hpp>
#include <tt-metalium/buffer_types.hpp>
#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/device.hpp>
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/kernel_types.hpp>
#include <tt-metalium/program.hpp>
#include <tt-metalium/tt_metal.hpp>
#include <tt-logger/tt-logger.hpp>
#include <tt-metalium/experimental/metal2_host_api/program.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_spec.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_run_args.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include "device_fixture.hpp"
#include "tt_metal/impl/dispatch/slow_dispatch.hpp"
#include "tt_metal/test_utils/stimulus.hpp"
#include "tt_metal/hw/inc/internal/tt-2xx/dataflow_buffer/dataflow_buffer_config.h"
#include "impl/dataflow_buffer/dataflow_buffer.hpp"
#include "impl/program/program_impl.hpp"
#include "impl/kernels/kernel.hpp"
#include <tt-metalium/tensor/mesh_tensor.hpp>
#include <tt-metalium/experimental/distributed_tensor/topology/tensor_topology.hpp>
#include <tt-metalium/tensor/spec/tensor_spec.hpp>
#include <tt-metalium/tensor/spec/layout/tensor_layout.hpp>
#include <tt-metalium/tensor/spec/layout/page_config.hpp>
#include <algorithm>
#include <chrono>
#include <filesystem>
#include <functional>
#include <thread>
#include <tt-metalium/bfloat16.hpp>
#include "impl/data_format/bfloat16_utils.hpp"
#include "tt_metal/impl/dataflow_buffer/dataflow_buffer_impl.hpp"
#include "impl/context/metal_context.hpp"
#include <llrt/tt_cluster.hpp>

namespace tt::tt_metal {

// Host readback of NEO-local tile-counter mirror fields used by isolation tests.
// NEO mirror: tiles_available@0x8, space_available@0xC, capacity@0x10.
struct LiveTcSnapshot {
    std::vector<uint32_t> capacity;
    std::vector<uint32_t> tiles_available;
    std::vector<uint32_t> space_available;
};

inline LiveTcSnapshot read_live_tcs(
    distributed::MeshDevice& unit_mesh, const CoreCoord& logical_core, uint32_t neo_id) {
    const auto& hal = MetalContext::instance().hal();
    const uint32_t base = hal.get_neo_tile_counters_base_addr() + neo_id * hal.get_neo_tile_counters_stride();
    const uint32_t size = hal.get_neo_tile_counters_size();
    const uint32_t cap_offset = hal.get_neo_tile_counters_buffer_capacity_offset();
    const uint32_t tiles_available_offset = hal.get_neo_tile_counters_tiles_available_offset();
    // Same offset as NEO_REGS_*_SPACE_AVAILABLE.
    constexpr uint32_t space_available_offset = 0x0000000Cu;
    const uint32_t num_counters = size != 0 ? static_cast<uint32_t>(::dfb::NUM_TILE_COUNTERS_PER_TENSIX) : 0u;
    const CoreCoord virtual_core = unit_mesh.worker_core_from_logical_core(logical_core);

    LiveTcSnapshot snap;
    snap.capacity.reserve(num_counters);
    snap.tiles_available.reserve(num_counters);
    snap.space_available.reserve(num_counters);
    auto& cluster = MetalContext::instance().get_cluster();
    const auto device_id = unit_mesh.get_device_ids()[0];
    for (uint32_t i = 0; i < num_counters; i++) {
        const uint32_t tc_base = base + i * size;
        snap.capacity.push_back(cluster.read_core(device_id, virtual_core, tc_base + cap_offset, sizeof(uint32_t))[0]);
        snap.tiles_available.push_back(
            cluster.read_core(device_id, virtual_core, tc_base + tiles_available_offset, sizeof(uint32_t))[0]);
        snap.space_available.push_back(
            cluster.read_core(device_id, virtual_core, tc_base + space_available_offset, sizeof(uint32_t))[0]);
    }
    return snap;
}

namespace m2 = experimental;

// ---- endpoint kind enums (legacy + Metal 2.0) ----
enum class DFBPorCType : uint8_t { DM, TENSIX };
enum class M2PorCType : uint8_t { DM, TENSIX };

// ---- parameterized fixtures (legacy + Metal 2.0) ----
class DFBImplicitSyncParamFixture : public UnitMeshFixture, public ::testing::WithParamInterface<bool> {};
class DFBImplicitSyncParamFixture_2_0 : public UnitMeshFixture, public ::testing::WithParamInterface<bool> {};

// ---- shared kernel / tensor factory helpers (Metal 2.0) ----
// Default dtype UINT32 keeps the legacy two-argument call sites (entry_size, total_entries)
// byte-identical: Shape{num_pages, page_size_bytes/4} == the old Shape{total_entries, entry_size/4}.
inline TensorSpec make_flat_dram_tensor_spec(
    uint32_t page_size_bytes, uint32_t num_pages, DataType dtype = DataType::UINT32) {
    auto page_config = PageConfig(Layout::ROW_MAJOR);
    auto memory_config = MemoryConfig{TensorMemoryLayout::INTERLEAVED, BufferType::DRAM};
    auto tensor_layout = TensorLayout(dtype, page_config, memory_config);
    // Page size in elements
    const uint32_t elem_size = dtype == DataType::UINT32 ? 4u : 2u;  // UINT32 or BFLOAT16
    const uint32_t elements_per_page = page_size_bytes / elem_size;
    return TensorSpec(Shape{num_pages, elements_per_page}, tensor_layout);
}

template <typename T>
inline void m2_writeshard_barrier_uint32(
    distributed::MeshDevice& unit_mesh, const MeshTensor& in_tensor, const std::vector<T>& input) {
    if (unit_mesh.arch() != ARCH::QUASAR) {
        return;
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(500));
    std::vector<T> rdback;
    slow_dispatch::ReadFromBuffer(in_tensor.mesh_buffer(), rdback);
    tt_driver_atomics::mfence();
    ASSERT_EQ(rdback, input) << "M2: WriteShard did not complete before LaunchProgram (Quasar emu #38042)";
}

inline m2::KernelSpec make_dm_kernel(
    const m2::KernelSpecName& unique_id,
    const std::string& source_path,
    uint8_t num_threads = 1,
    std::vector<m2::DFBSpecName> disable_implicit_sync_for = {}) {
    return m2::KernelSpec{
        .unique_id = unique_id,
        .source = std::filesystem::path{source_path},
        .num_threads = num_threads,
        .hw_config =
            m2::DataMovementHardwareConfig{
                .config_2xx =
                    m2::DataMovementHardwareConfig::DataMovement2XXConfig{
                        .disable_dfb_implicit_sync_for = std::move(disable_implicit_sync_for),
                    },
            },
    };
}

inline m2::KernelSpec make_compute_kernel(
    const m2::KernelSpecName& unique_id, const std::string& source_path, uint8_t num_threads = 1) {
    return m2::KernelSpec{
        .unique_id = unique_id,
        .source = std::filesystem::path{source_path},
        .num_threads = num_threads,
        .hw_config = m2::ComputeHardwareConfig{},
    };
}

inline void disable_implicit_sync_for(m2::KernelSpec& kernel, m2::DFBSpecName dfb_name) {
    auto& dm_cfg = std::get<m2::DataMovementHardwareConfig>(kernel.hw_config);
    if (!dm_cfg.config_2xx) {
        dm_cfg.config_2xx = m2::DataMovementHardwareConfig::DataMovement2XXConfig{};
    }
    auto& config_2xx = *dm_cfg.config_2xx;
    config_2xx.disable_dfb_implicit_sync_for.push_back(std::move(dfb_name));
}

inline void maybe_disable_implicit_sync(m2::KernelSpec& kernel, bool implicit_sync, m2::DFBSpecName dfb_name) {
    if (!implicit_sync) {
        disable_implicit_sync_for(kernel, std::move(dfb_name));
    }
}

inline m2::KernelSpec make_dm_dfb_producer(
    const m2::KernelSpecName& unique_id,
    const m2::DFBSpecName& dfb,
    const m2::TensorParamName& tensor,
    uint32_t num_entries_per_producer,
    bool implicit_sync,
    m2::DFBAccessPattern pap = m2::DFBAccessPattern::STRIDED,
    uint8_t num_threads = 1) {
    auto kernel =
        make_dm_kernel(unique_id, "tests/tt_metal/tt_metal/test_kernels/dataflow/dfb_producer_2_0.cpp", num_threads);
    kernel.dfb_bindings = {
        {.dfb_spec_name = dfb,
         .accessor_name = "out",
         .endpoint_type = m2::DFBEndpointType::PRODUCER,
         .access_pattern = pap}};
    kernel.tensor_bindings = {{.tensor_parameter_name = tensor, .accessor_name = "src_tensor"}};
    kernel.compile_time_args = {
        {"num_entries_per_producer", num_entries_per_producer}, {"implicit_sync", implicit_sync ? 1u : 0u}};
    kernel.runtime_arg_schema = {.runtime_arg_names = {"chunk_offset", "entries_per_core"}};
    return kernel;
}

inline m2::KernelSpec make_dm_dfb_consumer(
    const m2::KernelSpecName& unique_id,
    const m2::DFBSpecName& dfb,
    const m2::TensorParamName& tensor,
    uint32_t num_entries_per_consumer,
    bool blocked_consumer,
    bool implicit_sync,
    m2::DFBAccessPattern cap = m2::DFBAccessPattern::STRIDED,
    uint8_t num_threads = 1) {
    auto kernel =
        make_dm_kernel(unique_id, "tests/tt_metal/tt_metal/test_kernels/dataflow/dfb_consumer_2_0.cpp", num_threads);
    kernel.dfb_bindings = {
        {.dfb_spec_name = dfb,
         .accessor_name = "in",
         .endpoint_type = m2::DFBEndpointType::CONSUMER,
         .access_pattern = cap}};
    kernel.tensor_bindings = {{.tensor_parameter_name = tensor, .accessor_name = "dst_tensor"}};
    kernel.compile_time_args = {
        {"num_entries_per_consumer", num_entries_per_consumer},
        {"blocked_consumer", blocked_consumer ? 1u : 0u},
        {"implicit_sync", implicit_sync ? 1u : 0u}};
    kernel.runtime_arg_schema = {.runtime_arg_names = {"chunk_offset", "entries_per_core"}};
    return kernel;
}

// Which oracle verifies a DM-consumer run.
//   ORDERED   -- output must equal a positionally-derived expectation. Requires knowing the
//                slot->page mapping, so it is only usable where that mapping is settled.
//   MULTISET  -- output must be a permutation of the expected multiset: with a host-prefilled
//                ring whose slots the producer never rewrites, slot s always holds input[s], and
//                every posted credit is consumed exactly once, so each input page must appear
//                exactly entries_per_core/num_entries times somewhere in the output. Derived from
//                the DFB contract alone and INDEPENDENT of the interleave, which is what makes it
//                usable on shapes whose mapping is not yet established.
enum class M2Oracle { ORDERED, MULTISET };

struct M2SingleDFBParams {
    M2PorCType producer_type;
    M2PorCType consumer_type;
    uint32_t num_producers;
    uint32_t num_consumers;
    m2::DFBAccessPattern pap = m2::DFBAccessPattern::STRIDED;
    m2::DFBAccessPattern cap = m2::DFBAccessPattern::STRIDED;
    bool implicit_sync = false;
    uint32_t entry_size = 1024;
    uint32_t num_entries = 16;
    std::optional<uint32_t> num_entries_in_buffer = std::nullopt;  // override for ring pressure
    M2Oracle oracle = M2Oracle::ORDERED;
    uint32_t block_size = 0;                                       // BLOCKED only: tiles per block (0 for STRIDED/ALL)
};

inline uint32_t default_num_entries(uint32_t num_p, uint32_t num_c) {
    const uint32_t m = (num_p / std::gcd(num_p, num_c)) * num_c;
    return ((16u + m - 1u) / m) * m;
}

// ---- Tensix-consumer digest verification ----
// A Tensix consumer has no DRAM output, so dfb_t6_consumer_2_0.cpp reports an FNV-1a
// digest of every entry it drains into an L1 scratch region. These two helpers are the
// host side of that contract: the same hash, and the region's placement/sizing.
constexpr uint32_t k_dfb_digest_sentinel = 0xDEADBEEFu;

inline uint32_t fnv1a_page_digest(const std::vector<uint32_t>& words, uint32_t page_id, uint32_t words_per_entry) {
    uint32_t digest = 2166136261u;  // FNV-1a offset basis
    for (uint32_t w = 0; w < words_per_entry; ++w) {
        digest = (digest ^ words[page_id * words_per_entry + w]) * 16777619u;  // FNV-1a prime
    }
    return digest;
}

// Top of L1, below anything the allocator hands out (a single-DFB program's ring sits at
// the allocator base). Canonical placement for host-seeded scratch the device writes back
// (digests, read_tile_value results, extent probes, multi-touch results).
inline uint32_t top_of_l1_scratch_addr(distributed::MeshDevice& mesh_device, uint32_t bytes) {
    const uint32_t alignment = mesh_device.allocator()->get_alignment(BufferType::L1);
    const uint32_t aligned = (bytes + alignment - 1u) / alignment * alignment;
    return static_cast<uint32_t>(mesh_device.l1_size_per_core()) - aligned;
}

// Host-side size of the Tensix-consumer digest region. Layout is
// [consumer_idx][drain_index], one uint32_t each — the same indexing
// dfb_t6_consumer_2_0.cpp uses: result_l1_addr + get_my_thread_id() *
// num_entries_per_consumer * sizeof(uint32_t). `num_entries_per_consumer` here
// must be the compile-time arg compiled into that kernel.
inline uint32_t dfb_tensix_digest_region_bytes(uint32_t num_consumers, uint32_t num_entries_per_consumer) {
    return num_consumers * num_entries_per_consumer * static_cast<uint32_t>(sizeof(uint32_t));
}

// ---- shared skip macros + ring-size helper (used by base + overrides) ----
#define DFB_SKIP_IF_UNSUPPORTED(num_p, num_c)                                                  \
    if (this->device().arch() != ARCH::QUASAR && (GetParam() || (num_p) > 1 || (num_c) > 1)) { \
        GTEST_SKIP();                                                                          \
    }

// ---- mismatch diagnostics ----
inline const char* access_pattern_name(m2::DFBAccessPattern pattern) {
    switch (pattern) {
        case m2::DFBAccessPattern::BLOCKED: return "BLOCKED";
        case m2::DFBAccessPattern::ALL: return "ALL";
        default: return "STRIDED";
    }
}

inline const char* endpoint_name(M2PorCType type) { return type == M2PorCType::DM ? "DM" : "Tensix"; }

// On a data mismatch, report which input page (ring slot 0..num_entries-1) actually landed at each of
// the first 16 output tiles, so the real slot -> consumer mapping can be read off the log.
inline void log_output_provenance(
    const std::vector<uint32_t>& input,
    const std::vector<uint32_t>& output,
    uint32_t wpe,
    uint32_t num_entries,
    const std::string& label) {
    const uint32_t output_tiles = static_cast<uint32_t>(output.size() / wpe);
    for (uint32_t t = 0; t < std::min<uint32_t>(output_tiles, 16); ++t) {
        int match = -1;
        for (uint32_t src = 0; src < num_entries; ++src) {
            if (std::equal(input.begin() + src * wpe, input.begin() + (src + 1) * wpe, output.begin() + t * wpe)) {
                match = static_cast<int>(src);
                break;
            }
        }
        log_info(
            tt::LogTest,
            "  {} output tile {} <- {}",
            label,
            t,
            match >= 0 ? ("input page " + std::to_string(match)) : std::string("UNKNOWN"));
    }
}

// ---- single-DFB program driver ----

inline void run_single_dfb_program_2_0(distributed::MeshDevice& mesh_device, const M2SingleDFBParams& p) {
    // The DFB 2.0 host/device path is arch-abstracted: on WH/BH a DFB has no tile-counter
    // registers so it lowers to a 4-word circular-buffer config, and the _2_0 kernels' explicit
    // path is arch-agnostic (only the implicit async_read/write<TXN_ID> path is #ifdef ARCH_QUASAR).
    // So the simple 1x1 explicit-sync cases run on WH/BH too; only implicit-sync and multi-core
    // are Quasar-only (mirrors the legacy DFB_SKIP_IF_UNSUPPORTED gate).
    // BLOCKED is Quasar-only: the device-side block support is #ifdef ARCH_QUASAR.
    if (mesh_device.arch() != ARCH::QUASAR &&
        (p.implicit_sync || p.num_producers > 1 || p.num_consumers > 1 || p.pap == m2::DFBAccessPattern::BLOCKED ||
         p.cap == m2::DFBAccessPattern::BLOCKED)) {
        GTEST_SKIP() << "M2 non-Quasar: only 1x1 explicit-sync non-BLOCKED DFB runs on WH/BH "
                        "(implicit-sync, multi-core and BLOCKED are Quasar-only)";
    }
    // Tensix→Tensix is unsupported (legacy parity).
    if (p.producer_type == M2PorCType::TENSIX && p.consumer_type == M2PorCType::TENSIX) {
        GTEST_SKIP() << "Tensix→Tensix unsupported (no NoC transfer)";
    }
    // An ALL (broadcast) DM consumer under implicit sync deadlocks: the broadcast
    // credit path is not driven for a DM consumer regardless of producer -- DM→DM
    // has no DM↔DM remapper, and a Tensix producer cannot post the DM consumer's
    // implicit credits (the ISR poster is DM-only). The explicit-sync variant is
    // fine. Legacy skipped DM→DM ALL implicit via DFB_SKIP_DM_DM_ALL_IMPLICIT_SYNC;
    // the DM-consumer ALL case (DM→DM and Tensix→DM) needs the same gate or the
    // per-config DFB_TEST_2_0 path hangs.
    if (p.consumer_type == M2PorCType::DM && p.cap == m2::DFBAccessPattern::ALL && p.implicit_sync) {
        GTEST_SKIP() << "ALL DM consumer with implicit_sync not supported (legacy parity)";
    }

    const m2::NodeCoord node{0, 0};
    const uint32_t entries_per_core = p.num_entries_in_buffer.value_or(p.num_entries);
    const bool is_all = (p.cap == m2::DFBAccessPattern::ALL);
    const bool producer_blocked = (p.pap == m2::DFBAccessPattern::BLOCKED);
    const bool consumer_blocked = (p.cap == m2::DFBAccessPattern::BLOCKED);
    // A Tensix BLOCKED producer feeding STRIDED consumers uses the generic share-loop producer (its share
    // is the block), which also exercises get_produce_share() on a BLOCKED Tensix side; the other BLOCKED
    // producer cases use the explicit per-block producer kernel.
    const bool blocked_to_strided = producer_blocked && (p.cap == m2::DFBAccessPattern::STRIDED);

    const m2::DFBSpecName DFB{"dfb"};
    const m2::KernelSpecName PRODUCER{"producer"};
    const m2::KernelSpecName CONSUMER{"consumer"};
    const m2::TensorParamName IN_TENSOR{"in_tensor"};
    const m2::TensorParamName OUT_TENSOR{"out_tensor"};

    const auto tensor_spec = make_flat_dram_tensor_spec(p.entry_size, entries_per_core, DataType::UINT32);
    // Only allocate (and bind) a DRAM tensor on the side that has a DM kernel.
    // Tensix producer reads from host-prefilled L1; Tensix consumer doesn't write DRAM.
    std::optional<MeshTensor> in_tensor;
    std::optional<MeshTensor> out_tensor;
    if (p.producer_type == M2PorCType::DM) {
        in_tensor = MeshTensor::allocate_on_device(mesh_device, tensor_spec);
    }
    if (p.consumer_type == M2PorCType::DM) {
        out_tensor = MeshTensor::allocate_on_device(mesh_device, tensor_spec);
    }

    m2::DataflowBufferSpec dfb_spec{
        .unique_id = DFB,
        .entry_size = p.entry_size,
        .num_entries = p.num_entries,
        .data_format_metadata = tt::DataFormat::Float16_b,
    };

    const uint32_t num_entries_per_producer = (entries_per_core + p.num_producers - 1) / p.num_producers;
    const uint32_t num_entries_per_consumer =
        is_all ? entries_per_core : (entries_per_core + p.num_consumers - 1) / p.num_consumers;

    // Producer kernel
    m2::KernelSpec producer;
    if (p.producer_type == M2PorCType::DM) {
        // One producer kernel per side kind: a BLOCKED producer moves whole blocks (the
        // interface's split_tc shares the credits when its consumers are STRIDED); everyone
        // else moves single entries.
        const char* producer_src = producer_blocked
                                       ? "tests/tt_metal/tt_metal/test_kernels/dataflow/dfb_blocked_producer.cpp"
                                       : "tests/tt_metal/tt_metal/test_kernels/dataflow/dfb_producer_2_0.cpp";
        producer = make_dm_kernel(PRODUCER, producer_src, p.num_producers);
        producer.tensor_bindings = {{.tensor_parameter_name = IN_TENSOR, .accessor_name = "src_tensor"}};
        producer.runtime_arg_schema = {.runtime_arg_names = {"chunk_offset", "entries_per_core"}};
    } else {
        // Tensix producer: num_threads must match num_producers so total credits
        // posted = num_producers * num_entries_per_producer = entries_per_core.
        // BLOCKED posts credits block_size-at-a-time (host pre-fills the L1 ring either way).
        producer = make_compute_kernel(
            PRODUCER,
            // (BLOCKED→STRIDED takes the share-loop producer; see blocked_to_strided above.)
            (producer_blocked && !blocked_to_strided)
                ? "tests/tt_metal/tt_metal/test_kernels/compute/dfb_t6_blocked_producer.cpp"
                : "tests/tt_metal/tt_metal/test_kernels/compute/dfb_t6_producer_2_0.cpp",
            static_cast<uint8_t>(p.num_producers));
    }
    producer.dfb_bindings = {
        {.dfb_spec_name = DFB,
         .accessor_name = "out",
         .endpoint_type = m2::DFBEndpointType::PRODUCER,
         .access_pattern = p.pap,
         .block_size = producer_blocked ? p.block_size : 0u}};
    // BLOCKED uses dedicated kernels with a block_size CTA, in both sync modes.
    if (producer_blocked) {
        producer.compile_time_args = {
            {"num_entries_per_producer", num_entries_per_producer},
            {"block_size", p.block_size},
            {"implicit_sync", p.implicit_sync ? 1u : 0u}};
    } else {
        producer.compile_time_args = {
            {"num_entries_per_producer", num_entries_per_producer}, {"implicit_sync", p.implicit_sync ? 1u : 0u}};
    }

    // Consumer kernel
    m2::KernelSpec consumer;
    if (p.consumer_type == M2PorCType::DM) {
        consumer = make_dm_kernel(
            CONSUMER,
            consumer_blocked ? "tests/tt_metal/tt_metal/test_kernels/dataflow/dfb_blocked_consumer.cpp"
                             : "tests/tt_metal/tt_metal/test_kernels/dataflow/dfb_consumer_2_0.cpp",
            p.num_consumers);
        consumer.tensor_bindings = {{.tensor_parameter_name = OUT_TENSOR, .accessor_name = "dst_tensor"}};
        // The legacy "blocked_consumer" CTA below is the ALL-pattern contiguous flag, not BLOCKED.
        if (consumer_blocked) {
            consumer.compile_time_args = {
                {"num_entries_per_consumer", num_entries_per_consumer},
                {"block_size", p.block_size},
                {"implicit_sync", p.implicit_sync ? 1u : 0u}};
        } else {
            consumer.compile_time_args = {
                {"num_entries_per_consumer", num_entries_per_consumer},
                {"blocked_consumer", is_all ? 1u : 0u},
                {"implicit_sync", p.implicit_sync ? 1u : 0u}};
        }
        consumer.runtime_arg_schema = {.runtime_arg_names = {"chunk_offset", "entries_per_core"}};
    } else {  // Tensix consumer
        consumer = make_compute_kernel(
            CONSUMER,
            "tests/tt_metal/tt_metal/test_kernels/compute/dfb_t6_consumer_2_0.cpp",
            static_cast<uint8_t>(p.num_consumers));
        // The kernel derives its share (get_consume_share) from the ring; no block_size CTA.
        consumer.compile_time_args = {{"num_entries_per_consumer", num_entries_per_consumer}};
        consumer.runtime_arg_schema = {.runtime_arg_names = {"result_l1_addr"}};
    }
    consumer.dfb_bindings = {
        {.dfb_spec_name = DFB,
         .accessor_name = "in",
         .endpoint_type = m2::DFBEndpointType::CONSUMER,
         .access_pattern = p.cap,
         .block_size = consumer_blocked ? p.block_size : 0u}};

    // Config is arch-specific (the _2_0 kernels are the same either way). On WH/BH a DFB lowers to
    // a circular buffer and ValidateProgramSpec requires config_1xx on DM kernels, so mirror the
    // legacy driver -- DM producer -> RISCV_0, DM consumer -> RISCV_1/NOC_1, Tensix -> default
    // ComputeHardwareConfig. The make_*_kernel helpers set no config_1xx; override on WH/BH here
    // (only 1x1 explicit-sync cases reach WH/BH per the skip gate above).
    if (mesh_device.arch() == ARCH::QUASAR) {
        // Gen2 implicit-sync opt-out (#45160): only DM endpoints carry the per-kernel flag; for
        // ImplicitSyncFalse it keeps the host from programming implicit ISR/txn metadata over the
        // kernels' explicit credit-flow path. Tensix endpoints have no DM side.
        if (p.producer_type == M2PorCType::DM) {
            maybe_disable_implicit_sync(producer, p.implicit_sync, DFB);
        }
        if (p.consumer_type == M2PorCType::DM) {
            maybe_disable_implicit_sync(consumer, p.implicit_sync, DFB);
        }
    } else {
        // WH/BH: config_1xx pins (WH/BH has no implicit sync, so no disable knob needed).
        if (p.producer_type == M2PorCType::DM) {
            producer.hw_config = m2::DataMovementHardwareConfig{
                .config_1xx =
                    m2::DataMovementHardwareConfig::DataMovement1XXConfig{
                        .processor = tt::tt_metal::DataMovementProcessor::RISCV_0,
                    },
            };
        } else {
            producer.hw_config = m2::ComputeHardwareConfig{};
        }
        if (p.consumer_type == M2PorCType::DM) {
            consumer.hw_config = m2::DataMovementHardwareConfig{
                .config_1xx =
                    m2::DataMovementHardwareConfig::DataMovement1XXConfig{
                        .processor = tt::tt_metal::DataMovementProcessor::RISCV_1,
                        .noc = tt::tt_metal::NOC::NOC_1,
                    },
            };
        } else {
            consumer.hw_config = m2::ComputeHardwareConfig{};
        }
    }

    m2::WorkUnitSpec wu{.name = "wu", .kernels = {PRODUCER, CONSUMER}, .target_nodes = node};

    std::vector<m2::TensorParameter> tensor_params;
    if (in_tensor) {
        tensor_params.push_back({.unique_id = IN_TENSOR, .spec = in_tensor->tensor_spec()});
    }
    if (out_tensor) {
        tensor_params.push_back({.unique_id = OUT_TENSOR, .spec = out_tensor->tensor_spec()});
    }

    m2::ProgramSpec spec{
        .name = "single_dfb_2_0",
        .kernels = {producer, consumer},
        .dataflow_buffers = {dfb_spec},
        .tensor_parameters = tensor_params,
        .work_units = {wu},
    };

    Program program = m2::MakeProgramFromSpec(mesh_device, spec);

    m2::ProgramRunArgs params;
    if (p.producer_type == M2PorCType::DM) {
        params.kernel_run_args.push_back({
            .kernel = PRODUCER,
            .runtime_arg_values = experimental::MakeRuntimeArgsForSingleNode(
                node, {{"chunk_offset", 0u}, {"entries_per_core", entries_per_core}}),
        });
    } else {
        params.kernel_run_args.push_back({.kernel = PRODUCER});
    }
    // Tensix consumer: hand it the L1 region it reports per-entry digests into (see the
    // verification block after LaunchProgram). Size it from the CTA compiled into the
    // kernel so the host region and dfb_t6_consumer_2_0.cpp indexing cannot drift.
    uint32_t digest_region_bytes = 0;
    if (p.consumer_type == M2PorCType::TENSIX) {
        const auto cta_num_entries_per_consumer = consumer.compile_time_args.get("num_entries_per_consumer");
        ASSERT_TRUE(cta_num_entries_per_consumer.has_value())
            << "Tensix consumer kernel must compile with num_entries_per_consumer";
        ASSERT_EQ(*cta_num_entries_per_consumer, num_entries_per_consumer)
            << "digest region must be sized from the same num_entries_per_consumer CTA the kernel compiles with";
        digest_region_bytes = dfb_tensix_digest_region_bytes(p.num_consumers, *cta_num_entries_per_consumer);
    }
    const uint32_t digest_l1_addr =
        p.consumer_type == M2PorCType::TENSIX ? top_of_l1_scratch_addr(mesh_device, digest_region_bytes) : 0u;
    if (p.consumer_type == M2PorCType::DM) {
        params.kernel_run_args.push_back({
            .kernel = CONSUMER,
            .runtime_arg_values = experimental::MakeRuntimeArgsForSingleNode(
                node, {{"chunk_offset", 0u}, {"entries_per_core", entries_per_core}}),
        });
    } else {
        params.kernel_run_args.push_back({
            .kernel = CONSUMER,
            .runtime_arg_values =
                experimental::MakeRuntimeArgsForSingleNode(node, {{"result_l1_addr", digest_l1_addr}}),
        });
    }
    if (in_tensor) {
        params.tensor_args.insert({IN_TENSOR, std::cref(*in_tensor)});
    }
    if (out_tensor) {
        params.tensor_args.insert({OUT_TENSOR, std::cref(*out_tensor)});
    }
    m2::SetProgramRunArgs(program, params);

    // Stimulus
    const uint32_t total_words = p.entry_size * entries_per_core / sizeof(uint32_t);
    auto input = tt::test_utils::generate_uniform_random_vector<uint32_t>(0, 1000000, total_words);
    if (in_tensor) {
        slow_dispatch::WriteToBuffer(in_tensor->mesh_buffer(), input);
        m2_writeshard_barrier_uint32(mesh_device, *in_tensor, input);
    }

    // For Tensix producer: host-prefill the DFB L1 ring with the input data so the
    // producer kernel (which only posts credits) has something for the consumer to read.
    //
    // The physical ring layout depends on stride_in_entries, which the finalize derives
    // from the consumer access pattern:
    //   STRIDED: stride = num_producers -> interleaved (slot = e*P + p), which is exactly
    //            linear page order, so an input[0..ring) copy is correct.
    //   ALL:     stride = 1 -> each producer owns a contiguous block (slot = p*E + e). The
    //            ALL consumer round-robins across the P blocks (drains slot (k%P)*E + k/P for
    //            the k-th entry), so producer p's e-th entry (input page e*P + p) must sit at
    //            slot p*E + e for the drained order to reconstruct the identity output. A
    //            linear copy only works for a single producer; with P>1 it drains a P-way
    //            transpose of the input.
    //   BLOCKED: the ring is prefilled flat (ring[s]=input[s]), which every BLOCKED golden assumes, so
    //            the ALL transpose is gated off for a BLOCKED producer below.
    if (p.producer_type == M2PorCType::TENSIX) {
        const uint32_t dfb_l1_addr =
            static_cast<uint32_t>(mesh_device.allocator()->get_base_allocator_addr(HalMemType::L1));
        const uint32_t wpe = p.entry_size / sizeof(uint32_t);
        const uint32_t ring_words = p.num_entries * wpe;
        std::vector<uint32_t> slice(ring_words, 0u);
        for (uint32_t prod = 0; prod < p.num_producers; ++prod) {
            for (uint32_t e = 0; e < num_entries_per_producer; ++e) {
                const uint32_t page_id = e * p.num_producers + prod;
                if (page_id >= entries_per_core) {
                    break;
                }
                const uint32_t dst_slot = (is_all && !producer_blocked) ? (prod * num_entries_per_producer + e)
                                                                        : (e * p.num_producers + prod);
                // Ring-pressure: stop once the physical ring is full; later pages alias
                // back onto already-filled slots (the producer cycles them).
                if (dst_slot >= p.num_entries) {
                    break;
                }
                std::copy(
                    input.begin() + page_id * wpe, input.begin() + (page_id + 1) * wpe, slice.begin() + dst_slot * wpe);
            }
        }
        slow_dispatch::WriteToL1(mesh_device, CoreCoord(0, 0), dfb_l1_addr, slice);
    }

    // Seed the digest region so a slot the consumer never reached reads back as the
    // sentinel rather than as stale data from a previous test in the same binary.
    if (p.consumer_type == M2PorCType::TENSIX) {
        std::vector<uint32_t> sentinel(digest_region_bytes / sizeof(uint32_t), k_dfb_digest_sentinel);
        slow_dispatch::WriteToL1(mesh_device, CoreCoord(0, 0), digest_l1_addr, sentinel);
    }

    LaunchProgram(mesh_device, std::move(program));

    // Verify (DM consumer only — Tensix consumer doesn't write DRAM).
    if (p.consumer_type == M2PorCType::DM) {
        std::vector<uint32_t> output;
        slow_dispatch::ReadFromBuffer(out_tensor->mesh_buffer(), output);
        const uint32_t wpe = p.entry_size / sizeof(uint32_t);
        const std::string label = fmt::format(
            "{}→{} {}→{}",
            endpoint_name(p.producer_type),
            endpoint_name(p.consumer_type),
            access_pattern_name(p.pap),
            access_pattern_name(p.cap));
        if (p.oracle == M2Oracle::MULTISET) {
            // Mapping-independent check -- see M2Oracle. Each ring slot must be delivered exactly
            // reps times across the whole output, in any order. A consumer sub-stream that receives
            // no valid data shows up as output pages matching no input slot.
            const uint32_t reps = entries_per_core / p.num_entries;
            // Both preconditions of the oracle, asserted rather than assumed. It only holds for a
            // host-prefilled ring (TENSIX producer) whose slots the producer never rewrites, and only
            // when the stream is a whole number of ring-fills.
            ASSERT_EQ(p.producer_type, M2PorCType::TENSIX) << "MULTISET assumes a host-prefilled ring";
            ASSERT_EQ(entries_per_core % p.num_entries, 0u) << "MULTISET oracle needs whole ring-fills";
            ASSERT_EQ(output.size(), input.size());

            std::vector<int> match_of(entries_per_core, -1);
            std::vector<uint32_t> delivered(p.num_entries, 0u);
            uint32_t unmatched = 0;
            for (uint32_t t = 0; t < entries_per_core; ++t) {
                for (uint32_t src = 0; src < p.num_entries; ++src) {
                    if (std::equal(
                            input.begin() + src * wpe, input.begin() + (src + 1) * wpe, output.begin() + t * wpe)) {
                        match_of[t] = static_cast<int>(src);
                        break;
                    }
                }
                if (match_of[t] < 0) {
                    ++unmatched;
                } else {
                    ++delivered[match_of[t]];
                }
            }

            // Attribute failures to consumer sub-streams the way the STRIDED split assigns them, so
            // the "exactly num_consumers-of-N serviced" signature is visible rather than inferred.
            if (unmatched != 0) {
                std::vector<uint32_t> bad_per_residue(p.num_consumers, 0u);
                for (uint32_t t = 0; t < entries_per_core; ++t) {
                    if (match_of[t] < 0) {
                        ++bad_per_residue[t % p.num_consumers];
                    }
                }
                for (uint32_t c = 0; c < p.num_consumers; ++c) {
                    log_info(
                        tt::LogTest,
                        "  consumer residue {}: {} of {} output entries match no ring slot",
                        c,
                        bad_per_residue[c],
                        entries_per_core / p.num_consumers);
                }
            }

            EXPECT_EQ(unmatched, 0u) << "M2 MULTISET: " << unmatched << " of " << entries_per_core
                                     << " output entries match no ring slot";
            for (uint32_t src = 0; src < p.num_entries; ++src) {
                EXPECT_EQ(delivered[src], reps) << "M2 MULTISET: ring slot " << src << " delivered " << delivered[src]
                                                << " times, expected " << reps;
            }
            return;
        }
        // For Tensix→DM ring-pressure with STRIDED, each consumer reads ring slot
        // (c % num_entries), so expected output is the corresponding input slice.
        if (p.producer_type == M2PorCType::TENSIX && entries_per_core > p.num_entries &&
            p.cap == m2::DFBAccessPattern::STRIDED) {
            std::vector<uint32_t> expected(input.size(), 0u);
            // Metal 2.0 STRIDED consumer slot allocation differs from legacy:
            // - Legacy: consumer c reads only slot c (formula (p % num_c) % num_entries)
            // - M2: consumer c reads slots {c, c+num_c, c+2*num_c, ...} interleaved
            //   across the ring. Diagnostic re-derived this formula by mapping
            //   output tile → input page (see TensixDMTest1xDFB_RingPressure_2Sx4S_2_0).
            // The resulting expected: output[p] = input[p % num_entries] (assumes
            // num_consumers divides num_entries cleanly, which is the case for the
            // 2Sx4S variant with 16-entry ring).
            for (uint32_t i = 0; i < entries_per_core; ++i) {
                const uint32_t ring_slot = i % p.num_entries;
                std::copy(
                    input.begin() + ring_slot * wpe, input.begin() + (ring_slot + 1) * wpe, expected.begin() + i * wpe);
            }
            // Diagnostic: identify which input page actually landed at each
            // output page. If the formula is off, this dump tells us the true
            // ring-slot → consumer mapping under Metal 2.0 so we can correct it.
            if (expected != output) {
                auto mm = std::mismatch(expected.begin(), expected.end(), output.begin());
                size_t first_diff = mm.first - expected.begin();
                if (first_diff < expected.size()) {
                    const size_t bad_tile = first_diff / wpe;
                    log_info(
                        tt::LogTest,
                        "M2 Tensix→DM ring-pressure: first mismatch at tile {} word {}. "
                        "expected=0x{:x} output=0x{:x}. Searching which input page produced this output:",
                        bad_tile,
                        first_diff % wpe,
                        expected[first_diff],
                        output[first_diff]);
                    log_output_provenance(input, output, wpe, p.num_entries, label);
                }
            }
            EXPECT_EQ(expected, output) << "M2 Tensix→DM ring-pressure mismatch";
        } else {
            // Every other shape is an in-order round trip -- a DM producer writes ring slots in page
            // order, a Tensix producer's ring is host-prefilled flat, and every consumer pattern
            // writes back the page ids that undo its drain order -- so the output must equal the input.
            if (input != output) {
                log_output_provenance(input, output, wpe, p.num_entries, label);
            }
            EXPECT_EQ(input, output) << "M2 " << label << " identity mismatch";
        }
    }
    // DM→Tensix: the Tensix consumer writes no DRAM, so it reports an FNV-1a digest of each
    // entry it drained into L1 (see dfb_t6_consumer_2_0.cpp). Verifying it needs no new golden
    // machinery -- the delivery mapping is the one the DM consumer's identity check above
    // already relies on, read off that kernel's page_id:
    //   STRIDED: the k-th entry handed to consumer c is input page k*num_consumers + c
    //            (dfb_consumer_2_0.cpp writes it to exactly that page and we expect identity).
    //            This holds for a BLOCKED producer too: a DM BLOCKED producer fills the ring flat
    //            in page order (dfb_blocked_producer.cpp), and a STRIDED consumer of a BLOCKED
    //            ring drains its share of each block at stride num_consumers, so its running
    //            k-th tile is ring position k*num_consumers + c (the DM→DM BLOCKED→STRIDED
    //            identity check above relies on the same fact).
    //   ALL:     every consumer sees the whole stream in order, so the k-th entry is input
    //            page k (blocked_consumer writes page_id = tile_id, again with identity
    //            expected). ALL is the *simpler* case, not a harder one.
    //   BLOCKED: consumer c takes whole blocks c, c+C, c+2C, ... of a flat ring, so its k-th
    //            tile is page ((k / bs) * C + c) * bs + k % bs -- the same mapping
    //            run_a1_fanout_blocked_pipeline's golden uses for its BLOCKED Tensix consumers.
    // This checks the payload bytes and the per-consumer delivery order of every entry, which
    // is what the DM-consumer path gets from its DRAM readback.
    if (p.consumer_type == M2PorCType::TENSIX) {
        // Under STRIDED/BLOCKED each consumer drains num_entries_per_consumer entries
        // unconditionally (unlike the DM consumer, which breaks once page_id runs past the
        // tensor), so an indivisible split would mean waiting on entries no producer sends.
        // default_num_entries returns a multiple of lcm(P,C), so every config in the sweep divides.
        if (!is_all) {
            ASSERT_EQ(entries_per_core % p.num_consumers, 0u)
                << "M2 DM→Tensix STRIDED/BLOCKED: entries_per_core must divide across consumers";
        }
        if (consumer_blocked) {
            ASSERT_GT(p.block_size, 0u) << "M2 DM→Tensix BLOCKED consumer needs block_size";
            ASSERT_EQ(num_entries_per_consumer % p.block_size, 0u)
                << "M2 DM→Tensix BLOCKED: each consumer must drain whole blocks";
        }
        std::vector<uint32_t> digests;
        slow_dispatch::ReadFromL1(mesh_device, CoreCoord(0, 0), digest_l1_addr, digest_region_bytes, digests);
        const uint32_t wpe = p.entry_size / sizeof(uint32_t);
        for (uint32_t c = 0; c < p.num_consumers; ++c) {
            for (uint32_t k = 0; k < num_entries_per_consumer; ++k) {
                uint32_t page_id = 0;
                if (is_all) {
                    page_id = k;
                } else if (consumer_blocked) {
                    page_id = ((k / p.block_size) * p.num_consumers + c) * p.block_size + (k % p.block_size);
                } else {
                    page_id = k * p.num_consumers + c;
                }
                ASSERT_LT(page_id, entries_per_core);
                EXPECT_EQ(digests[c * num_entries_per_consumer + k], fnv1a_page_digest(input, page_id, wpe))
                    << "M2 DM→Tensix digest mismatch: consumer " << c << " drain index " << k
                    << " should have been input page " << page_id;
            }
        }
    }
}

inline void run_a1_blocked_pipeline(
    distributed::MeshDevice& mesh_device,
    m2::DFBAccessPattern cap_in,
    uint32_t P,
    uint32_t block_size,
    uint32_t num_entries,
    bool implicit = false) {
    if (mesh_device.arch() != ARCH::QUASAR) {
        GTEST_SKIP() << "M2 path is Quasar-only (Gen2Config)";
    }
    constexpr uint32_t entry_size = 2 * 32 * 32;  // bf16 tile = 2048 B
    const m2::NodeCoord node{0, 0};

    const auto tensor_spec = make_flat_dram_tensor_spec(entry_size, num_entries, DataType::BFLOAT16);
    auto in_tensor = MeshTensor::allocate_on_device(mesh_device, tensor_spec);
    auto out_tensor = MeshTensor::allocate_on_device(mesh_device, tensor_spec);

    const m2::DFBSpecName DFB_IN{"dfb_in"};
    const m2::DFBSpecName DFB_OUT{"dfb_out"};
    const m2::KernelSpecName PRODUCER{"producer"};
    const m2::KernelSpecName CONSUMER{"consumer"};
    const m2::KernelSpecName COMPUTE{"compute"};
    const m2::TensorParamName IN_TENSOR{"in_tensor"};
    const m2::TensorParamName OUT_TENSOR{"out_tensor"};

    m2::DataflowBufferSpec dfb_in{
        .unique_id = DFB_IN,
        .entry_size = entry_size,
        .num_entries = num_entries,
        .data_format_metadata = tt::DataFormat::Float16_b,
    };
    m2::DataflowBufferSpec dfb_out{
        .unique_id = DFB_OUT,
        .entry_size = entry_size,
        .num_entries = num_entries,
        .data_format_metadata = tt::DataFormat::Float16_b,
    };

    // Front half: P DM BLOCKED producers → DFB_IN (consumer is the Tensix below, pattern cap_in).
    auto producer = make_dm_kernel(
        PRODUCER, "tests/tt_metal/tt_metal/test_kernels/dataflow/dfb_blocked_producer.cpp", static_cast<uint8_t>(P));
    producer.dfb_bindings = {
        {.dfb_spec_name = DFB_IN,
         .accessor_name = "out",
         .endpoint_type = m2::DFBEndpointType::PRODUCER,
         .access_pattern = m2::DFBAccessPattern::BLOCKED,
         .block_size = block_size}};
    producer.tensor_bindings = {{.tensor_parameter_name = IN_TENSOR, .accessor_name = "src_tensor"}};
    producer.compile_time_args = {
        {"num_entries_per_producer", num_entries / P},
        {"block_size", block_size},
        {"implicit_sync", implicit ? 1u : 0u}};
    producer.runtime_arg_schema = {.runtime_arg_names = {"chunk_offset", "entries_per_core"}};

    // Middle: single Tensix thread consumes DFB_IN (cap_in) and copies through to DFB_OUT (STRIDED).
    // dfb_eltwise_copy is pattern-agnostic (waits/pops its input share, copies tile by tile, pushes
    // one output tile each).
    auto compute =
        make_compute_kernel(COMPUTE, "tests/tt_metal/tt_metal/test_kernels/compute/dfb_eltwise_copy_2_0.cpp");
    compute.dfb_bindings = {
        {.dfb_spec_name = DFB_IN,
         .accessor_name = "in",
         .endpoint_type = m2::DFBEndpointType::CONSUMER,
         .access_pattern = cap_in,
         .block_size = (cap_in == m2::DFBAccessPattern::BLOCKED) ? block_size : 0u},
        {.dfb_spec_name = DFB_OUT,
         .accessor_name = "out",
         .endpoint_type = m2::DFBEndpointType::PRODUCER,
         .access_pattern = m2::DFBAccessPattern::STRIDED},
    };
    compute.compile_time_args = {{"per_core_tile_cnt", num_entries}};

    // Back half (identity pass-through): DFB_OUT → 1 DM STRIDED consumer → DRAM.
    auto consumer = make_dm_kernel(CONSUMER, "tests/tt_metal/tt_metal/test_kernels/dataflow/dfb_consumer_2_0.cpp");
    consumer.dfb_bindings = {
        {.dfb_spec_name = DFB_OUT,
         .accessor_name = "in",
         .endpoint_type = m2::DFBEndpointType::CONSUMER,
         .access_pattern = m2::DFBAccessPattern::STRIDED}};
    consumer.tensor_bindings = {{.tensor_parameter_name = OUT_TENSOR, .accessor_name = "dst_tensor"}};
    consumer.compile_time_args = {
        {"num_entries_per_consumer", num_entries}, {"blocked_consumer", 0u}, {"implicit_sync", implicit ? 1u : 0u}};
    consumer.runtime_arg_schema = {.runtime_arg_names = {"chunk_offset", "entries_per_core"}};

    // Explicit sync disables the implicit-sync ISR/txn metadata per DM endpoint; for implicit, leave it on.
    if (!implicit) {
        disable_implicit_sync_for(producer, DFB_IN);
        disable_implicit_sync_for(consumer, DFB_OUT);
    }

    m2::WorkUnitSpec wu{.name = "wu", .kernels = {PRODUCER, CONSUMER, COMPUTE}, .target_nodes = node};
    m2::ProgramSpec spec{
        .name = "a1_blocked_2_0",
        .kernels = {producer, consumer, compute},
        .dataflow_buffers = {dfb_in, dfb_out},
        .tensor_parameters =
            {
                {.unique_id = IN_TENSOR, .spec = in_tensor.tensor_spec()},
                {.unique_id = OUT_TENSOR, .spec = out_tensor.tensor_spec()},
            },
        .work_units = {wu},
    };
    Program program = m2::MakeProgramFromSpec(mesh_device, spec);

    m2::ProgramRunArgs params;
    params.kernel_run_args = {
        m2::ProgramRunArgs::KernelRunArgs{
            .kernel = PRODUCER,
            .runtime_arg_values = experimental::MakeRuntimeArgsForSingleNode(
                node, {{"chunk_offset", 0u}, {"entries_per_core", num_entries}}),
        },
        m2::ProgramRunArgs::KernelRunArgs{
            .kernel = CONSUMER,
            .runtime_arg_values = experimental::MakeRuntimeArgsForSingleNode(
                node, {{"chunk_offset", 0u}, {"entries_per_core", num_entries}}),
        },
        m2::ProgramRunArgs::KernelRunArgs{.kernel = COMPUTE},
    };
    params.tensor_args = {{IN_TENSOR, std::cref(in_tensor)}, {OUT_TENSOR, std::cref(out_tensor)}};
    m2::SetProgramRunArgs(program, params);

    const uint32_t total_bytes = entry_size * num_entries;
    auto input = create_random_vector_of_bfloat16(total_bytes, 2.0f, 0xA1B1);
    slow_dispatch::WriteToBuffer(in_tensor.mesh_buffer(), input);
    m2_writeshard_barrier_uint32(mesh_device, in_tensor, input);

    LaunchProgram(mesh_device, std::move(program));

    std::vector<uint32_t> output;
    slow_dispatch::ReadFromBuffer(out_tensor.mesh_buffer(), output);

    // DRAM_out[k] is the Tensix's k-th consumed tile from DFB_IN, since the back half is a FIFO
    // pass-through.
    const uint32_t wpe = entry_size / sizeof(uint32_t);
    const std::string label = fmt::format("A1 DM→Trisc BLOCKED→{}", access_pattern_name(cap_in));
    // Every BLOCKED-producer pattern here is an in-order round trip, so the output must equal the input.
    if (input != output) {
        log_output_provenance(input, output, wpe, num_entries, label);
    }
    EXPECT_EQ(input, output) << label << " data mismatch (P=" << P << ")";
}

// DM -> Tensix(copy, P threads) -> DM pipeline whose OUTPUT ring is BLOCKED on the Tensix side: the P packers
// really pack whole blocks (share = block_size, in-order placement) and C DM consumers drain the ring with
// the ALL pattern, each writing every page (identical data, so the shared output tensor holds one copy).
// This is the only Tensix-producer BLOCKED coverage whose data path is real: the DFB_TRISC_BLOCKED_* tests use a
// credit-only producer over a host-prefilled ring, so a wrong pack cursor is invisible to them.
// Golden: compute thread p is STRIDED consumer p of the input ring (tiles p, p+P, ...). It packs its i-th tile
// into its j = i/bs -th block at offset m = i%bs; that block is global block g = j*P + p, at ring position
// g*bs + m, and the ALL consumers drain the ring in order, so output[g*bs + m] = input[p + (j*bs + m)*P]
// (identity when P == 1).
inline void run_tensix_blocked_out_pipeline(
    distributed::MeshDevice& mesh_device, uint32_t P, uint32_t C, uint32_t block_size, uint32_t num_entries) {
    if (mesh_device.arch() != ARCH::QUASAR) {
        GTEST_SKIP() << "M2 path is Quasar-only (Gen2Config)";
    }
    TT_FATAL(num_entries % (P * block_size) == 0, "num_entries must be a whole number of blocks per producer");
    constexpr uint32_t entry_size = 2 * 32 * 32;  // bf16 tile = 2048 B
    const m2::NodeCoord node{0, 0};

    const auto tensor_spec = make_flat_dram_tensor_spec(entry_size, num_entries, DataType::BFLOAT16);
    auto in_tensor = MeshTensor::allocate_on_device(mesh_device, tensor_spec);
    auto out_tensor = MeshTensor::allocate_on_device(mesh_device, tensor_spec);

    const m2::DFBSpecName DFB_IN{"dfb_in"};
    const m2::DFBSpecName DFB_OUT{"dfb_out"};
    const m2::KernelSpecName PRODUCER{"producer"};
    const m2::KernelSpecName CONSUMER{"consumer"};
    const m2::KernelSpecName COMPUTE{"compute"};
    const m2::TensorParamName IN_TENSOR{"in_tensor"};
    const m2::TensorParamName OUT_TENSOR{"out_tensor"};

    m2::DataflowBufferSpec dfb_in{
        .unique_id = DFB_IN,
        .entry_size = entry_size,
        .num_entries = num_entries,
        .data_format_metadata = tt::DataFormat::Float16_b};
    m2::DataflowBufferSpec dfb_out{
        .unique_id = DFB_OUT,
        .entry_size = entry_size,
        .num_entries = num_entries,
        .data_format_metadata = tt::DataFormat::Float16_b};

    // One STRIDED DM producer feeds the P compute threads (STRIDED consumers of the input ring).
    auto producer = make_dm_kernel(
        PRODUCER, "tests/tt_metal/tt_metal/test_kernels/dataflow/dfb_producer_2_0.cpp", /*num_threads=*/1);
    producer.dfb_bindings = {
        {.dfb_spec_name = DFB_IN,
         .accessor_name = "out",
         .endpoint_type = m2::DFBEndpointType::PRODUCER,
         .access_pattern = m2::DFBAccessPattern::STRIDED}};
    producer.tensor_bindings = {{.tensor_parameter_name = IN_TENSOR, .accessor_name = "src_tensor"}};
    producer.compile_time_args = {{"num_entries_per_producer", num_entries}, {"implicit_sync", 0u}};
    producer.runtime_arg_schema = {.runtime_arg_names = {"chunk_offset", "entries_per_core"}};

    // P compute threads: STRIDED consumers of dfb_in, BLOCKED producers of dfb_out (share = block_size).
    auto compute = make_compute_kernel(
        COMPUTE, "tests/tt_metal/tt_metal/test_kernels/compute/dfb_eltwise_copy_2_0.cpp", static_cast<uint8_t>(P));
    compute.dfb_bindings = {
        {.dfb_spec_name = DFB_IN,
         .accessor_name = "in",
         .endpoint_type = m2::DFBEndpointType::CONSUMER,
         .access_pattern = m2::DFBAccessPattern::STRIDED},
        {.dfb_spec_name = DFB_OUT,
         .accessor_name = "out",
         .endpoint_type = m2::DFBEndpointType::PRODUCER,
         .access_pattern = m2::DFBAccessPattern::BLOCKED,
         .block_size = block_size},
    };
    compute.compile_time_args = {{"per_core_tile_cnt", num_entries / P}};

    // C DM consumers with the ALL pattern: each drains every entry and writes every page.
    auto consumer = make_dm_kernel(
        CONSUMER, "tests/tt_metal/tt_metal/test_kernels/dataflow/dfb_consumer_2_0.cpp", static_cast<uint8_t>(C));
    consumer.dfb_bindings = {
        {.dfb_spec_name = DFB_OUT,
         .accessor_name = "in",
         .endpoint_type = m2::DFBEndpointType::CONSUMER,
         .access_pattern = m2::DFBAccessPattern::ALL}};
    consumer.tensor_bindings = {{.tensor_parameter_name = OUT_TENSOR, .accessor_name = "dst_tensor"}};
    // blocked_consumer is the ALL-pattern "every consumer drains every entry, pages contiguous" flag.
    consumer.compile_time_args = {
        {"num_entries_per_consumer", num_entries}, {"blocked_consumer", 1u}, {"implicit_sync", 0u}};
    consumer.runtime_arg_schema = {.runtime_arg_names = {"chunk_offset", "entries_per_core"}};

    disable_implicit_sync_for(producer, DFB_IN);
    disable_implicit_sync_for(consumer, DFB_OUT);

    m2::WorkUnitSpec wu{.name = "wu", .kernels = {PRODUCER, CONSUMER, COMPUTE}, .target_nodes = node};
    m2::ProgramSpec spec{
        .name = "tensix_blocked_out_2_0",
        .kernels = {producer, consumer, compute},
        .dataflow_buffers = {dfb_in, dfb_out},
        .tensor_parameters =
            {{.unique_id = IN_TENSOR, .spec = in_tensor.tensor_spec()},
             {.unique_id = OUT_TENSOR, .spec = out_tensor.tensor_spec()}},
        .work_units = {wu},
    };
    Program program = m2::MakeProgramFromSpec(mesh_device, spec);

    m2::ProgramRunArgs params;
    params.kernel_run_args = {
        m2::ProgramRunArgs::KernelRunArgs{
            .kernel = PRODUCER,
            .runtime_arg_values = experimental::MakeRuntimeArgsForSingleNode(
                node, {{"chunk_offset", 0u}, {"entries_per_core", num_entries}})},
        m2::ProgramRunArgs::KernelRunArgs{
            .kernel = CONSUMER,
            .runtime_arg_values = experimental::MakeRuntimeArgsForSingleNode(
                node, {{"chunk_offset", 0u}, {"entries_per_core", num_entries}})},
        m2::ProgramRunArgs::KernelRunArgs{.kernel = COMPUTE},
    };
    params.tensor_args = {{IN_TENSOR, std::cref(in_tensor)}, {OUT_TENSOR, std::cref(out_tensor)}};
    m2::SetProgramRunArgs(program, params);

    const uint32_t total_bytes = entry_size * num_entries;
    auto input = create_random_vector_of_bfloat16(total_bytes, 2.0f, 0xB10C);
    slow_dispatch::WriteToBuffer(in_tensor.mesh_buffer(), input);
    m2_writeshard_barrier_uint32(mesh_device, in_tensor, input);

    LaunchProgram(mesh_device, std::move(program));

    std::vector<uint32_t> output;
    slow_dispatch::ReadFromBuffer(out_tensor.mesh_buffer(), output);

    const uint32_t wpe = entry_size / sizeof(uint32_t);
    const uint32_t bs = block_size;
    std::vector<uint32_t> expected(input.size(), 0u);
    for (uint32_t r = 0; r < num_entries; ++r) {
        const uint32_t g = r / bs, m = r % bs;
        const uint32_t p = g % P, j = g / P;
        const uint32_t src = p + (j * bs + m) * P;
        std::copy(input.begin() + src * wpe, input.begin() + (src + 1) * wpe, expected.begin() + r * wpe);
    }
    if (expected != output) {
        log_output_provenance(
            input, output, wpe, num_entries, fmt::format("Tensix BLOCKED->ALL P={} C={} bs={}", P, C, bs));
    }
    EXPECT_EQ(expected, output) << "Tensix BLOCKED producer -> DM ALL consumers data mismatch (P=" << P << ", C=" << C
                                << ", bs=" << bs << ")";
}

inline void run_a1_fanout_blocked_pipeline(
    distributed::MeshDevice& mesh_device,
    uint32_t C,
    uint32_t block_size,
    uint32_t num_entries,
    bool implicit = false,
    m2::DFBAccessPattern cap_in = m2::DFBAccessPattern::BLOCKED,
    m2::DFBAccessPattern pap_in = m2::DFBAccessPattern::BLOCKED) {
    if (mesh_device.arch() != ARCH::QUASAR) {
        GTEST_SKIP() << "M2 path is Quasar-only (Gen2Config)";
    }
    // C>1 Tensix consumers race on multi-thread coherence unless the watcher is polling.
    if (!MetalContext::instance().rtoptions().get_watcher_enabled()) {
        GTEST_SKIP() << "A1 fan-out needs the watcher (TT_METAL_WATCHER=1): multi-thread coherence race";
    }
    constexpr uint32_t entry_size = 2 * 32 * 32;  // bf16 tile = 2048 B
    const m2::NodeCoord node{0, 0};

    const auto tensor_spec = make_flat_dram_tensor_spec(entry_size, num_entries, DataType::BFLOAT16);
    auto in_tensor = MeshTensor::allocate_on_device(mesh_device, tensor_spec);
    auto out_tensor = MeshTensor::allocate_on_device(mesh_device, tensor_spec);

    const m2::DFBSpecName DFB_IN{"dfb_in"};
    const m2::DFBSpecName DFB_OUT{"dfb_out"};
    const m2::KernelSpecName PRODUCER{"producer"};
    const m2::KernelSpecName CONSUMER{"consumer"};
    const m2::KernelSpecName COMPUTE{"compute"};
    const m2::TensorParamName IN_TENSOR{"in_tensor"};
    const m2::TensorParamName OUT_TENSOR{"out_tensor"};

    m2::DataflowBufferSpec dfb_in{
        .unique_id = DFB_IN,
        .entry_size = entry_size,
        .num_entries = num_entries,
        .data_format_metadata = tt::DataFormat::Float16_b};
    m2::DataflowBufferSpec dfb_out{
        .unique_id = DFB_OUT,
        .entry_size = entry_size,
        .num_entries = num_entries,
        .data_format_metadata = tt::DataFormat::Float16_b};

    const bool prod_blocked = pap_in == m2::DFBAccessPattern::BLOCKED;
    auto producer = make_dm_kernel(
        PRODUCER,
        prod_blocked ? "tests/tt_metal/tt_metal/test_kernels/dataflow/dfb_blocked_producer.cpp"
                     : "tests/tt_metal/tt_metal/test_kernels/dataflow/dfb_producer_2_0.cpp",
        /*num_threads=*/1);
    producer.dfb_bindings = {
        {.dfb_spec_name = DFB_IN,
         .accessor_name = "out",
         .endpoint_type = m2::DFBEndpointType::PRODUCER,
         .access_pattern = pap_in,
         .block_size = prod_blocked ? block_size : 0u}};
    producer.tensor_bindings = {{.tensor_parameter_name = IN_TENSOR, .accessor_name = "src_tensor"}};
    if (prod_blocked) {
        producer.compile_time_args = {
            {"num_entries_per_producer", num_entries},
            {"block_size", block_size},
            {"implicit_sync", implicit ? 1u : 0u}};
    } else {
        producer.compile_time_args = {{"num_entries_per_producer", num_entries}, {"implicit_sync", implicit ? 1u : 0u}};
    }
    producer.runtime_arg_schema = {.runtime_arg_names = {"chunk_offset", "entries_per_core"}};

    auto compute = make_compute_kernel(
        COMPUTE, "tests/tt_metal/tt_metal/test_kernels/compute/dfb_eltwise_copy_2_0.cpp", static_cast<uint8_t>(C));
    compute.dfb_bindings = {
        {.dfb_spec_name = DFB_IN,
         .accessor_name = "in",
         .endpoint_type = m2::DFBEndpointType::CONSUMER,
         .access_pattern = cap_in,
         .block_size = (cap_in == m2::DFBAccessPattern::BLOCKED) ? block_size : 0u},
        {.dfb_spec_name = DFB_OUT,
         .accessor_name = "out",
         .endpoint_type = m2::DFBEndpointType::PRODUCER,
         .access_pattern = m2::DFBAccessPattern::STRIDED},
    };
    compute.compile_time_args = {{"per_core_tile_cnt", num_entries / C}};

    auto consumer = make_dm_kernel(
        CONSUMER, "tests/tt_metal/tt_metal/test_kernels/dataflow/dfb_consumer_2_0.cpp", /*num_threads=*/1);
    consumer.dfb_bindings = {
        {.dfb_spec_name = DFB_OUT,
         .accessor_name = "in",
         .endpoint_type = m2::DFBEndpointType::CONSUMER,
         .access_pattern = m2::DFBAccessPattern::STRIDED}};
    consumer.tensor_bindings = {{.tensor_parameter_name = OUT_TENSOR, .accessor_name = "dst_tensor"}};
    consumer.compile_time_args = {
        {"num_entries_per_consumer", num_entries}, {"blocked_consumer", 0u}, {"implicit_sync", implicit ? 1u : 0u}};
    consumer.runtime_arg_schema = {.runtime_arg_names = {"chunk_offset", "entries_per_core"}};

    // DFB_IN is BLOCKED with C>1 Tensix consumers, so the DM producer is the wider-fan-out side and its
    // implicit commit has to stay block-aware. DFB_OUT is STRIDED, where per-entry round-robin is right,
    // so its DM consumer can be implicit either way.
    if (!implicit) {
        disable_implicit_sync_for(producer, DFB_IN);
        disable_implicit_sync_for(consumer, DFB_OUT);
    }

    m2::WorkUnitSpec wu{.name = "wu", .kernels = {PRODUCER, CONSUMER, COMPUTE}, .target_nodes = node};
    m2::ProgramSpec spec{
        .name = "a1_sym_2_0",
        .kernels = {producer, consumer, compute},
        .dataflow_buffers = {dfb_in, dfb_out},
        .tensor_parameters =
            {{.unique_id = IN_TENSOR, .spec = in_tensor.tensor_spec()},
             {.unique_id = OUT_TENSOR, .spec = out_tensor.tensor_spec()}},
        .work_units = {wu},
    };
    Program program = m2::MakeProgramFromSpec(mesh_device, spec);

    m2::ProgramRunArgs params;
    params.kernel_run_args = {
        m2::ProgramRunArgs::KernelRunArgs{
            .kernel = PRODUCER,
            .runtime_arg_values = experimental::MakeRuntimeArgsForSingleNode(
                node, {{"chunk_offset", 0u}, {"entries_per_core", num_entries}})},
        m2::ProgramRunArgs::KernelRunArgs{
            .kernel = CONSUMER,
            .runtime_arg_values = experimental::MakeRuntimeArgsForSingleNode(
                node, {{"chunk_offset", 0u}, {"entries_per_core", num_entries}})},
        m2::ProgramRunArgs::KernelRunArgs{.kernel = COMPUTE},
    };
    params.tensor_args = {{IN_TENSOR, std::cref(in_tensor)}, {OUT_TENSOR, std::cref(out_tensor)}};
    m2::SetProgramRunArgs(program, params);

    const uint32_t total_bytes = entry_size * num_entries;
    auto input = create_random_vector_of_bfloat16(total_bytes, 2.0f, 0xA1C1);
    slow_dispatch::WriteToBuffer(in_tensor.mesh_buffer(), input);
    m2_writeshard_barrier_uint32(mesh_device, in_tensor, input);

    LaunchProgram(mesh_device, std::move(program));

    std::vector<uint32_t> output;
    slow_dispatch::ReadFromBuffer(out_tensor.mesh_buffer(), output);

    const uint32_t wpe = entry_size / sizeof(uint32_t);
    const uint32_t bs = block_size;
    std::vector<uint32_t> expected(input.size(), 0u);
    for (uint32_t r = 0; r < num_entries; ++r) {
        const uint32_t c = r % C;
        const uint32_t m = r / C;
        // Which input page Tensix consumer c consumed as its m-th tile. BLOCKED consumers take
        // whole blocks; STRIDED consumers take their within-block share in ring order, which
        // composes with the round-robin back half to the identity.
        const uint32_t src = (cap_in == m2::DFBAccessPattern::BLOCKED) ? ((c + (m / bs) * C) * bs + (m % bs)) : r;
        std::copy(input.begin() + src * wpe, input.begin() + (src + 1) * wpe, expected.begin() + r * wpe);
    }
    if (expected != output) {
        log_output_provenance(input, output, wpe, num_entries, fmt::format("A1-fanout BLOCKED C={}", C));
    }
    EXPECT_EQ(expected, output) << "A1-fanout multi-Tensix-consumer BLOCKED data mismatch (C=" << C << ")";
}

}  // namespace tt::tt_metal
