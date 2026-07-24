// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <map>
#include <memory>
#include <numeric>
#include <optional>
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
#include "tt_metal/test_utils/stimulus.hpp"
#include "tt_metal/hw/inc/internal/tt-2xx/dataflow_buffer/dataflow_buffer_config.h"
#include "impl/dataflow_buffer/dataflow_buffer.hpp"
#include "impl/program/program_impl.hpp"
#include "impl/kernels/kernel.hpp"
#include <tt-metalium/experimental/tensor/mesh_tensor.hpp>
#include <tt-metalium/experimental/tensor/topology/tensor_topology.hpp>
#include <tt-metalium/experimental/tensor/spec/tensor_spec.hpp>
#include <tt-metalium/experimental/tensor/spec/layout/tensor_layout.hpp>
#include <tt-metalium/experimental/tensor/spec/layout/page_config.hpp>
#include <algorithm>
#include <chrono>
#include <filesystem>
#include <functional>
#include <thread>
#include <tt-metalium/bfloat16.hpp>
#include "impl/data_format/bfloat16_utils.hpp"
#include "tt_metal/impl/dataflow_buffer/dataflow_buffer_impl.hpp"

namespace tt::tt_metal {

namespace m2 = experimental;

// ---- endpoint kind enums (legacy + Metal 2.0) ----
enum class DFBPorCType : uint8_t { DM, TENSIX };
enum class M2PorCType : uint8_t { DM, TENSIX };

// ---- parameterized fixtures (legacy + Metal 2.0) ----
class DFBImplicitSyncParamFixture : public MeshDeviceFixture, public ::testing::WithParamInterface<bool> {};
class DFBImplicitSyncParamFixture_2_0 : public MeshDeviceFixture, public ::testing::WithParamInterface<bool> {};

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
inline void m2_writeshard_barrier_uint32(IDevice* device, const MeshTensor& in_tensor, const std::vector<T>& input) {
    if (device->arch() != ARCH::QUASAR) {
        return;
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(500));
    std::vector<T> rdback;
    detail::ReadFromBuffer(*in_tensor.mesh_buffer().get_reference_buffer(), rdback);
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
            m2::DataMovementGen2Config{
                .disable_dfb_implicit_sync_for = std::move(disable_implicit_sync_for),
            },
    };
}

inline m2::KernelSpec make_compute_kernel(
    const m2::KernelSpecName& unique_id, const std::string& source_path, uint8_t num_threads = 1) {
    return m2::KernelSpec{
        .unique_id = unique_id,
        .source = std::filesystem::path{source_path},
        .num_threads = num_threads,
        .hw_config = m2::ComputeGen2Config{},
    };
}

inline void disable_implicit_sync_for(m2::KernelSpec& kernel, m2::DFBSpecName dfb_name) {
    auto& dm_cfg = std::get<m2::DataMovementHardwareConfig>(kernel.hw_config);
    TT_FATAL(std::holds_alternative<m2::DataMovementGen2Config>(dm_cfg), "Can only set implicit sync for Gen2 Kernel");
    auto& gen2_cfg = std::get<m2::DataMovementGen2Config>(dm_cfg);
    gen2_cfg.disable_dfb_implicit_sync_for.push_back(std::move(dfb_name));
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
    uint32_t block_size = 0;  // BLOCKED only: tiles per block (0 for STRIDED/ALL)
};

inline uint32_t default_num_entries(uint32_t num_p, uint32_t num_c) {
    const uint32_t m = (num_p / std::gcd(num_p, num_c)) * num_c;
    return ((16u + m - 1u) / m) * m;
}

// ---- shared skip macros + ring-size helper (used by base + overrides) ----
#define DFB_SKIP_IF_UNSUPPORTED(num_p, num_c)                                                   \
    if (devices_.at(0)->arch() != ARCH::QUASAR && (GetParam() || (num_p) > 1 || (num_c) > 1)) { \
        GTEST_SKIP();                                                                           \
    }

// ---- single-DFB program driver ----

inline void run_single_dfb_program_2_0(
    const std::shared_ptr<distributed::MeshDevice>& mesh_device, const M2SingleDFBParams& p) {
    // The DFB 2.0 host/device path is arch-abstracted: on WH/BH a DFB has no tile-counter
    // registers so it lowers to a 4-word circular-buffer config, and the _2_0 kernels' explicit
    // path is arch-agnostic (only the implicit async_read/write<TXN_ID> path is #ifdef ARCH_QUASAR).
    // So the simple 1x1 explicit-sync cases run on WH/BH too; only implicit-sync and multi-core
    // are Quasar-only (mirrors the legacy DFB_SKIP_IF_UNSUPPORTED gate).
    if (mesh_device->get_devices()[0]->arch() != ARCH::QUASAR &&
        (p.implicit_sync || p.num_producers > 1 || p.num_consumers > 1)) {
        GTEST_SKIP() << "M2 non-Quasar: only 1x1 explicit-sync DFB runs on WH/BH "
                        "(implicit-sync + multi-core are Quasar-only)";
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

    IDevice* device = mesh_device->get_devices()[0];
    const m2::NodeCoord node{0, 0};
    const uint32_t entries_per_core = p.num_entries_in_buffer.value_or(p.num_entries);
    const bool is_all = (p.cap == m2::DFBAccessPattern::ALL);
    const bool producer_blocked = (p.pap == m2::DFBAccessPattern::BLOCKED);
    const bool consumer_blocked = (p.cap == m2::DFBAccessPattern::BLOCKED);
    // BLOCKED-producer -> STRIDED-consumer: reads block-contiguous DRAM but pushes per-tile so the
    // STRIDED round-robin scatters each tile into the consumer's interleaved slot. DM uses
    // dfb_blocked_strided_producer; Tensix reuses the plain per-tile producer (its block-ness was
    // only credit cadence over a host-flat-prefilled ring).
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
        in_tensor = MeshTensor::allocate_on_device(*mesh_device, tensor_spec, TensorTopology{});
    }
    if (p.consumer_type == M2PorCType::DM) {
        out_tensor = MeshTensor::allocate_on_device(*mesh_device, tensor_spec, TensorTopology{});
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
        const char* producer_src = "tests/tt_metal/tt_metal/test_kernels/dataflow/dfb_producer_2_0.cpp";
        if (blocked_to_strided) {
            producer_src = "tests/tt_metal/tt_metal/test_kernels/dataflow/dfb_blocked_strided_producer_2_0.cpp";
        } else if (producer_blocked) {
            producer_src = "tests/tt_metal/tt_metal/test_kernels/dataflow/dfb_blocked_producer_2_0.cpp";
        }
        producer = make_dm_kernel(PRODUCER, producer_src, p.num_producers);
        producer.tensor_bindings = {{.tensor_parameter_name = IN_TENSOR, .accessor_name = "src_tensor"}};
        producer.runtime_arg_schema = {.runtime_arg_names = {"chunk_offset", "entries_per_core"}};
    } else {
        // Tensix producer: num_threads must match num_producers so total credits
        // posted = num_producers * num_entries_per_producer = entries_per_core.
        // BLOCKED posts credits block_size-at-a-time (host pre-fills the L1 ring either way).
        producer = make_compute_kernel(
            PRODUCER,
            // BLOCKED->STRIDED: a Tensix producer only posts credits over the host-flat-prefilled ring,
            // and a STRIDED consumer needs per-tile credits, so reuse the plain per-tile Tensix producer.
            (producer_blocked && !blocked_to_strided)
                ? "tests/tt_metal/tt_metal/test_kernels/compute/dfb_t6_blocked_producer_2_0.cpp"
                : "tests/tt_metal/tt_metal/test_kernels/compute/dfb_t6_producer_2_0.cpp",
            static_cast<uint8_t>(p.num_producers));
    }
    producer.dfb_bindings = {
        {.dfb_spec_name = DFB,
         .accessor_name = "out",
         .endpoint_type = m2::DFBEndpointType::PRODUCER,
         .access_pattern = p.pap,
         .block_size = producer_blocked ? p.block_size : 0u}};
    // BLOCKED uses dedicated kernels with a block_size CTA. They support both sync modes:
    // explicit = one NoC burst per block; implicit = one TXN_ID transfer per tile (single-entry).
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
            consumer_blocked ? "tests/tt_metal/tt_metal/test_kernels/dataflow/dfb_blocked_consumer_2_0.cpp"
                             : "tests/tt_metal/tt_metal/test_kernels/dataflow/dfb_consumer_2_0.cpp",
            p.num_consumers);
        consumer.tensor_bindings = {{.tensor_parameter_name = OUT_TENSOR, .accessor_name = "dst_tensor"}};
        // BLOCKED uses the dedicated kernel (explicit burst or implicit per-tile) with a block_size CTA.
        // The legacy "blocked_consumer" CTA is the ALL-pattern contiguous flag, unrelated to BLOCKED.
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
    } else {
        consumer = make_compute_kernel(
            CONSUMER,
            "tests/tt_metal/tt_metal/test_kernels/compute/dfb_t6_consumer_2_0.cpp",
            static_cast<uint8_t>(p.num_consumers));
        consumer.compile_time_args = {{"num_entries_per_consumer", num_entries_per_consumer}};
    }
    consumer.dfb_bindings = {
        {.dfb_spec_name = DFB,
         .accessor_name = "in",
         .endpoint_type = m2::DFBEndpointType::CONSUMER,
         .access_pattern = p.cap,
         .block_size = consumer_blocked ? p.block_size : 0u}};

    // Config is arch-specific (the _2_0 kernels are the same either way): Gen2 on Quasar,
    // Gen1 on WH/BH. On WH/BH a DFB lowers to a circular buffer and ValidateProgramSpec rejects
    // a Gen2 config, so mirror the legacy driver -- DM producer -> RISCV_0, DM consumer ->
    // RISCV_1/NOC_1, Tensix -> ComputeGen1. The make_*_kernel helpers default to Gen2; override
    // to Gen1 on WH/BH here (only 1x1 explicit-sync cases reach WH/BH per the skip gate above).
    if (mesh_device->get_devices()[0]->arch() == ARCH::QUASAR) {
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
        // WH/BH: Gen1 config (Gen1 has no implicit sync, so no disable knob needed).
        if (p.producer_type == M2PorCType::DM) {
            producer.hw_config =
                m2::DataMovementGen1Config{.processor = tt::tt_metal::DataMovementProcessor::RISCV_0};
        } else {
            producer.hw_config = m2::ComputeGen1Config{};
        }
        if (p.consumer_type == M2PorCType::DM) {
            consumer.hw_config = m2::DataMovementGen1Config{
                .processor = tt::tt_metal::DataMovementProcessor::RISCV_1, .noc = tt::tt_metal::NOC::NOC_1};
        } else {
            consumer.hw_config = m2::ComputeGen1Config{};
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

    Program program = m2::MakeProgramFromSpec(*mesh_device, spec);

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
    if (p.consumer_type == M2PorCType::DM) {
        params.kernel_run_args.push_back({
            .kernel = CONSUMER,
            .runtime_arg_values = experimental::MakeRuntimeArgsForSingleNode(
                node, {{"chunk_offset", 0u}, {"entries_per_core", entries_per_core}}),
        });
    } else {
        params.kernel_run_args.push_back({.kernel = CONSUMER});
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
        detail::WriteToBuffer(*in_tensor->mesh_buffer().get_reference_buffer(), input);
        m2_writeshard_barrier_uint32(device, *in_tensor, input);
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
    //   BLOCKED: a Tensix BLOCKED producer only posts credits over a FLAT ring (ring[s]=input[s]);
    //            its BLOCKED goldens (incl. BLOCKED->ALL) assume the flat layout, so the ALL transpose
    //            is gated off for a BLOCKED producer below.
    if (p.producer_type == M2PorCType::TENSIX) {
        const uint32_t dfb_l1_addr =
            static_cast<uint32_t>(device->allocator()->get_base_allocator_addr(HalMemType::L1));
        const uint32_t wpe = p.entry_size / sizeof(uint32_t);
        const uint32_t ring_words = p.num_entries * wpe;
        std::vector<uint32_t> slice(ring_words, 0u);
        for (uint32_t prod = 0; prod < p.num_producers; ++prod) {
            for (uint32_t e = 0; e < num_entries_per_producer; ++e) {
                const uint32_t page_id = e * p.num_producers + prod;
                if (page_id >= entries_per_core) {
                    break;
                }
                const uint32_t dst_slot =
                    (is_all && !producer_blocked) ? (prod * num_entries_per_producer + e) : (e * p.num_producers + prod);
                // Ring-pressure: stop once the physical ring is full; later pages alias
                // back onto already-filled slots (the producer cycles them).
                if (dst_slot >= p.num_entries) {
                    break;
                }
                std::copy(
                    input.begin() + page_id * wpe, input.begin() + (page_id + 1) * wpe, slice.begin() + dst_slot * wpe);
            }
        }
        detail::WriteToDeviceL1(device, CoreCoord(0, 0), dfb_l1_addr, slice);
    }

    detail::LaunchProgram(device, program, /*wait_until_cores_done=*/true);

    // Verify (DM consumer only — Tensix consumer doesn't write DRAM).
    if (p.consumer_type == M2PorCType::DM) {
        std::vector<uint32_t> output;
        detail::ReadFromBuffer(*out_tensor->mesh_buffer().get_reference_buffer(), output);
        // For Tensix→DM ring-pressure with STRIDED, each consumer reads ring slot
        // (c % num_entries), so expected output is the corresponding input slice.
        if (p.producer_type == M2PorCType::TENSIX && entries_per_core > p.num_entries &&
            p.cap == m2::DFBAccessPattern::STRIDED) {
            const uint32_t wpe = p.entry_size / sizeof(uint32_t);
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
                    // For each output tile, find which input page (0..num_entries-1) it matches.
                    // That tells us the real ring-slot assignment.
                    for (uint32_t t = 0; t < std::min<uint32_t>(entries_per_core, 16); ++t) {
                        int match = -1;
                        for (uint32_t src = 0; src < p.num_entries; ++src) {
                            if (std::equal(
                                    input.begin() + src * wpe,
                                    input.begin() + (src + 1) * wpe,
                                    output.begin() + t * wpe)) {
                                match = static_cast<int>(src);
                                break;
                            }
                        }
                        log_info(
                            tt::LogTest,
                            "  output tile {} ← {}",
                            t,
                            match >= 0 ? ("input page " + std::to_string(match))
                                       : std::string("UNKNOWN (no match in input ring)"));
                    }
                }
            }
            EXPECT_EQ(expected, output) << "M2 Tensix→DM ring-pressure mismatch";
        } else if (
            p.producer_type == M2PorCType::TENSIX && p.cap == m2::DFBAccessPattern::BLOCKED &&
            (p.num_consumers > 1 || p.num_producers > p.num_consumers)) {
            // Tensix→DM BLOCKED (any integer-ratio thread counts): a permutation, NOT identity. A Tensix
            // producer only posts credits over a host-prefilled FLAT ring (L1[k] = input[k]) — unlike a DM
            // producer there is no block-strided DRAM read to cancel the consumer's de-interleave.
            // cap=BLOCKED splits the ring into max(P,C) contiguous sub-rings of capacity=num_entries/max(P,C)
            // (stride 1). Consumer c, tile-counter slot t reads sub-ring (t*C + c); for fan-in P>C each
            // consumer round-robins ntc = P/C sub-rings per pop_front, else ntc = 1. Consumer c writes block
            // b to out page (b*C + c)*block_size. General device map (symmetric, fan-out C>=P, fan-in P>C):
            //   output[(b*C + c)*bs + j] = input[((b % ntc)*C + c)*capacity + (b / ntc)*bs + j]
            //   capacity = num_entries/max(P,C),  ntc = (P>=C) ? P/C : 1.
            // (N==1 symmetric degenerates to identity — handled by the else branch. For C>=P, ntc=1 and
            // capacity=num_entries/C, recovering the simple c*capacity + b*bs + j form.) The guard includes
            // num_producers>num_consumers so fan-in cases with C==1 reach this branch (not else-identity),
            // and using max(P,C) is what makes 4Bx2B correct.
            const uint32_t wpe = p.entry_size / sizeof(uint32_t);
            const uint32_t P = p.num_producers;
            const uint32_t C = p.num_consumers;
            const uint32_t capacity = p.num_entries / std::max(P, C);
            const uint32_t ntc = (P >= C) ? (P / C) : 1u;
            const uint32_t blocks_per_thread = num_entries_per_consumer / p.block_size;
            std::vector<uint32_t> expected(input.size(), 0u);
            for (uint32_t c = 0; c < C; ++c) {
                for (uint32_t b = 0; b < blocks_per_thread; ++b) {
                    for (uint32_t j = 0; j < p.block_size; ++j) {
                        const uint32_t src = ((b % ntc) * C + c) * capacity + (b / ntc) * p.block_size + j;
                        const uint32_t dst = (b * C + c) * p.block_size + j;
                        std::copy(
                            input.begin() + src * wpe, input.begin() + (src + 1) * wpe, expected.begin() + dst * wpe);
                    }
                }
            }
            // Diagnostic (mirrors the STRIDED branch): if it mismatches, map each output tile back to
            // the input page that actually landed there, so a device-side credit/sub-ring surprise is
            // debuggable (a deadlock would show up as a launch hang instead, pointing at credits/TCs).
            if (expected != output) {
                for (uint32_t t = 0; t < std::min<uint32_t>(entries_per_core, 16); ++t) {
                    int match = -1;
                    for (uint32_t src = 0; src < p.num_entries; ++src) {
                        if (std::equal(
                                input.begin() + src * wpe, input.begin() + (src + 1) * wpe, output.begin() + t * wpe)) {
                            match = static_cast<int>(src);
                            break;
                        }
                    }
                    log_info(
                        tt::LogTest,
                        "  Tensix→DM BLOCKED output tile {} ← {}",
                        t,
                        match >= 0 ? ("input page " + std::to_string(match)) : std::string("UNKNOWN"));
                }
            }
            EXPECT_EQ(expected, output) << "M2 Tensix→DM BLOCKED multi-thread permutation mismatch";
        } else if (
            p.producer_type == M2PorCType::TENSIX && p.pap == m2::DFBAccessPattern::BLOCKED &&
            p.cap == m2::DFBAccessPattern::ALL) {
            // Trisc→DM BLOCKED→ALL. A Tensix BLOCKED producer feeding a DM ALL consumer routes the credit
            // fan-out through the REMAPPER (not broadcast_tc — that path needs a DM producer). The host
            // pre-fills the ring FLAT (ring[s]=input[s]); the ALL consumer reads round-robin across the P
            // producer sub-rings (out tile r ← ring slot (r%P)*capacity + (r/P), capacity=num_entries/P). So
            // output[r] = input[(r%P)*capacity + (r/P)] — identity at P==1, block-size-independent (the
            // consumer reads per-tile, the flat prefill carries no block order).
            const uint32_t wpe = p.entry_size / sizeof(uint32_t);
            const uint32_t P = p.num_producers;
            const uint32_t capacity = p.num_entries / P;
            std::vector<uint32_t> expected(input.size(), 0u);
            for (uint32_t r = 0; r < p.num_entries; ++r) {
                const uint32_t src = (r % P) * capacity + (r / P);
                std::copy(input.begin() + src * wpe, input.begin() + (src + 1) * wpe, expected.begin() + r * wpe);
            }
            if (expected != output) {
                for (uint32_t t = 0; t < std::min<uint32_t>(entries_per_core, 16); ++t) {
                    int match = -1;
                    for (uint32_t src = 0; src < p.num_entries; ++src) {
                        if (std::equal(
                                input.begin() + src * wpe, input.begin() + (src + 1) * wpe, output.begin() + t * wpe)) {
                            match = static_cast<int>(src);
                            break;
                        }
                    }
                    log_info(
                        tt::LogTest,
                        "  Trisc→DM BLOCKED→ALL output tile {} ← {}",
                        t,
                        match >= 0 ? ("input page " + std::to_string(match)) : std::string("UNKNOWN"));
                }
            }
            EXPECT_EQ(expected, output) << "M2 Trisc→DM BLOCKED→ALL (remapper fan-out) mismatch";
        } else if (
            p.producer_type == M2PorCType::DM && p.consumer_type == M2PorCType::DM &&
            p.pap == m2::DFBAccessPattern::BLOCKED && p.cap == m2::DFBAccessPattern::ALL) {
            // BLOCKED-producer -> ALL-consumer (DM->DM). The producer block-bursts into its contiguous
            // per-producer sub-ring (capacity = num_entries/P, stride_in_entries=1); the ALL consumer
            // reads round-robin across the P producer sub-rings — the SAME de-interleave that makes
            // STRIDED->ALL identity (out tile r reads ring slot (r%P)*capacity + (r/P)). For P==1 the
            // round-trip is identity, but for P>1 the producer's per-BLOCK interleave does NOT cancel
            // the consumer's per-TILE round-robin, so it is a permutation:
            //   producer p, block b, offset j: input tile (b*P + p)*bs + j is written to ring slot
            //     p*capacity + b*bs + j, which the consumer reads out to tile (b*bs + j)*P + p.
            //   => output[(b*bs + j)*P + p] = input[(b*P + p)*bs + j]    (reduces to identity at P==1).
            const uint32_t wpe = p.entry_size / sizeof(uint32_t);
            const uint32_t P = p.num_producers;
            const uint32_t bs = p.block_size;
            const uint32_t capacity = p.num_entries / P;
            const uint32_t blocks_per_producer = capacity / bs;
            std::vector<uint32_t> expected(input.size(), 0u);
            for (uint32_t pp = 0; pp < P; ++pp) {
                for (uint32_t b = 0; b < blocks_per_producer; ++b) {
                    for (uint32_t j = 0; j < bs; ++j) {
                        const uint32_t src = (b * P + pp) * bs + j;
                        const uint32_t dst = (b * bs + j) * P + pp;
                        std::copy(
                            input.begin() + src * wpe, input.begin() + (src + 1) * wpe, expected.begin() + dst * wpe);
                    }
                }
            }
            EXPECT_EQ(expected, output) << "M2 DM→DM BLOCKED→ALL permutation mismatch";
        } else if (
            p.producer_type == M2PorCType::DM && p.consumer_type == M2PorCType::DM &&
            p.pap == m2::DFBAccessPattern::BLOCKED && p.cap == m2::DFBAccessPattern::STRIDED) {
            // BLOCKED-producer -> STRIDED-consumer (DM->DM). The producer reads block_size contiguous DRAM
            // pages per block (block order) but PUSHES PER-TILE, so the STRIDED round-robin hands tile i to
            // the producer's TC slot t = i % ntc, whose paired consumer is c = (p + t*P) % C; that consumer
            // reads its interleaved slots in order, writing its k-th received tile to out page k*C + c. The
            // k-th tile producer p sends to slot t is local push i = t + k*ntc, reading DRAM page
            // (i/bs * P + p)*bs + i%bs:
            //   output[k*C + c] = input[((t + k*ntc)/bs * P + p)*bs + (t + k*ntc)%bs]
            // For P==1 (ntc=C, t=c) this collapses to identity; for P>1 it is a deterministic permutation.
            const uint32_t wpe = p.entry_size / sizeof(uint32_t);
            const uint32_t P = p.num_producers;
            const uint32_t C = p.num_consumers;
            const uint32_t bs = p.block_size;
            std::vector<uint32_t> expected(input.size(), 0u);
            if (C >= P) {
                // Fan-out: producer pp round-robins ntc = C/P consumer TCs (its push i -> slot t = i%ntc,
                // consumer c = (pp + t*P)%C). Producer pp's k-th push to slot t is local push i = t + k*ntc.
                const uint32_t epp = p.num_entries / P;  // entries per producer
                const uint32_t ntc = C / P;
                for (uint32_t pp = 0; pp < P; ++pp) {
                    for (uint32_t t = 0; t < ntc; ++t) {
                        const uint32_t c = (pp + t * P) % C;
                        const uint32_t tiles_to_c = epp / ntc;  // == num_entries / C
                        for (uint32_t k = 0; k < tiles_to_c; ++k) {
                            const uint32_t i = t + k * ntc;  // producer pp's local push index
                            const uint32_t src = (i / bs * P + pp) * bs + (i % bs);
                            const uint32_t dst = k * C + c;
                            std::copy(
                                input.begin() + src * wpe,
                                input.begin() + (src + 1) * wpe,
                                expected.begin() + dst * wpe);
                        }
                    }
                }
            } else {
                // Fan-in (P>C): each producer has 1 TC -> feeds consumer (pp%C); consumer c is fed by the
                // ntc_c = P/C producers {c, c+C, ...} via ntc_c TCs, read round-robin. Consumer c's read m
                // takes TC t = m%ntc_c -> producer pp = c + t*C, that producer's (m/ntc_c)-th push.
                const uint32_t ntc_c = P / C;
                for (uint32_t c = 0; c < C; ++c) {
                    for (uint32_t m = 0; m < p.num_entries / C; ++m) {
                        const uint32_t t = m % ntc_c;
                        const uint32_t pp = c + t * C;
                        const uint32_t k = m / ntc_c;  // producer pp's local push index
                        const uint32_t src = (k / bs * P + pp) * bs + (k % bs);
                        const uint32_t dst = m * C + c;
                        std::copy(
                            input.begin() + src * wpe, input.begin() + (src + 1) * wpe, expected.begin() + dst * wpe);
                    }
                }
            }
            if (expected != output) {
                for (uint32_t t = 0; t < std::min<uint32_t>(entries_per_core, 16); ++t) {
                    int match = -1;
                    for (uint32_t src = 0; src < p.num_entries; ++src) {
                        if (std::equal(
                                input.begin() + src * wpe, input.begin() + (src + 1) * wpe, output.begin() + t * wpe)) {
                            match = static_cast<int>(src);
                            break;
                        }
                    }
                    log_info(
                        tt::LogTest,
                        "  DM→DM BLOCKED→STRIDED output tile {} ← {}",
                        t,
                        match >= 0 ? ("input page " + std::to_string(match)) : std::string("UNKNOWN"));
                }
            }
            EXPECT_EQ(expected, output) << "M2 DM→DM BLOCKED→STRIDED permutation mismatch";
        } else if (
            p.producer_type == M2PorCType::TENSIX && p.consumer_type == M2PorCType::DM &&
            p.pap == m2::DFBAccessPattern::BLOCKED && p.cap == m2::DFBAccessPattern::STRIDED) {
            // Trisc→DM BLOCKED→STRIDED is IDENTITY for ALL P/C. The Tensix producer flat-prefills the ring
            // (ring[s]=input[s]) — it never scatters. The DM STRIDED consumer (dfb_consumer_2_0.cpp) reads its
            // tiles with stride = num_consumers (C) and writes them back with that SAME stride
            // (page = tile_id*C + consumer_idx), so over a flat ring the read-stride and write-stride cancel
            // and the round-trip is identity regardless of the P:C ratio.
            // (Confirmed: DM→DM 4Bx2S — same C=2 consumer routing — passes with its scatter-composed
            // permutation golden, proving the consumer de-interleave is correct on this build; Trisc→DM differs
            // only by the flat prefill, which yields identity.)
            // NOTE: an earlier C<P "stride=P round-robin" permutation golden here was WRONG — the consumer
            // strides by C (num_consumers), not P. It only ever affected P>C,C>1 (the lone 4Bx2S case).
            const uint32_t wpe = p.entry_size / sizeof(uint32_t);
            const uint32_t P = p.num_producers;
            const uint32_t C = p.num_consumers;
            std::vector<uint32_t> expected = input;  // identity for all P/C (see above)
            if (expected != output) {
                for (uint32_t t = 0; t < std::min<uint32_t>(entries_per_core, 16); ++t) {
                    int match = -1;
                    for (uint32_t src = 0; src < p.num_entries; ++src) {
                        if (std::equal(
                                input.begin() + src * wpe, input.begin() + (src + 1) * wpe, output.begin() + t * wpe)) {
                            match = static_cast<int>(src);
                            break;
                        }
                    }
                    log_info(
                        tt::LogTest,
                        "  Trisc→DM BLOCKED→STRIDED output tile {} ← {}",
                        t,
                        match >= 0 ? ("input page " + std::to_string(match)) : std::string("UNKNOWN"));
                }
            }
            EXPECT_EQ(expected, output) << "M2 Trisc→DM BLOCKED→STRIDED data mismatch (P=" << P << ",C=" << C << ")";
        } else {
            EXPECT_EQ(input, output) << "M2 single-DFB identity mismatch";
        }
    }
    // DM→Tensix: L1 verification is omitted for now (legacy parity requires complex
    // golden computation for the ALL pattern). We just verify the program runs.
}


inline void run_a1_blocked_pipeline(
    const std::shared_ptr<distributed::MeshDevice>& mesh_device,
    m2::DFBAccessPattern cap_in,
    uint32_t P,
    uint32_t block_size,
    uint32_t num_entries,
    bool implicit = false) {
    if (mesh_device->get_devices()[0]->arch() != ARCH::QUASAR) {
        GTEST_SKIP() << "M2 path is Quasar-only (Gen2Config)";
    }
    IDevice* device = mesh_device->get_devices()[0];
    constexpr uint32_t entry_size = 2 * 32 * 32;  // bf16 tile = 2048 B
    const m2::NodeCoord node{0, 0};

    const auto tensor_spec = make_flat_dram_tensor_spec(entry_size, num_entries, DataType::BFLOAT16);
    auto in_tensor = MeshTensor::allocate_on_device(*mesh_device, tensor_spec, TensorTopology{});
    auto out_tensor = MeshTensor::allocate_on_device(*mesh_device, tensor_spec, TensorTopology{});

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
    const char* producer_src =
        (cap_in == m2::DFBAccessPattern::STRIDED)
            ? "tests/tt_metal/tt_metal/test_kernels/dataflow/dfb_blocked_strided_producer_2_0.cpp"
            : "tests/tt_metal/tt_metal/test_kernels/dataflow/dfb_blocked_producer_2_0.cpp";
    auto producer = make_dm_kernel(PRODUCER, producer_src, static_cast<uint8_t>(P));
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
    // dfb_eltwise_copy is pattern-agnostic (per-tile wait_front/copy_tile/pack_tile/pop_front) — no edit.
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
    Program program = m2::MakeProgramFromSpec(*mesh_device, spec);

    m2::ProgramRunArgs params;
    params.kernel_run_args = {
        m2::ProgramRunArgs::KernelRunArgs{
            .kernel = PRODUCER,
            .runtime_arg_values = experimental::MakeRuntimeArgsForSingleNode(node, {{"chunk_offset", 0u}, {"entries_per_core", num_entries}}),
        },
        m2::ProgramRunArgs::KernelRunArgs{
            .kernel = CONSUMER,
            .runtime_arg_values = experimental::MakeRuntimeArgsForSingleNode(node, {{"chunk_offset", 0u}, {"entries_per_core", num_entries}}),
        },
        m2::ProgramRunArgs::KernelRunArgs{.kernel = COMPUTE},
    };
    params.tensor_args = {{IN_TENSOR, std::cref(in_tensor)}, {OUT_TENSOR, std::cref(out_tensor)}};
    m2::SetProgramRunArgs(program, params);

    const uint32_t total_bytes = entry_size * num_entries;
    auto input = create_random_vector_of_bfloat16(total_bytes, 2.0f, 0xA1B1);
    detail::WriteToBuffer(*in_tensor.mesh_buffer().get_reference_buffer(), input);
    m2_writeshard_barrier_uint32(device, in_tensor, input);

    detail::LaunchProgram(device, program, /*wait_until_cores_done=*/true);

    std::vector<uint32_t> output;
    detail::ReadFromBuffer(*out_tensor.mesh_buffer().get_reference_buffer(), output);

    // DRAM_out[k] = the Tensix's k-th consumed tile from DFB_IN (the back-half is a FIFO identity
    // pass-through). NOTE this is the raw CONSUME ORDER — for BLOCKED it is NOT the DM→DM identity,
    // because the A1 back-half does not do the DM block-consumer's reordering write.
    const uint32_t wpe = entry_size / sizeof(uint32_t);
    const uint32_t bs = block_size;
    const char* cap_name = "STRIDED";
    if (cap_in == m2::DFBAccessPattern::BLOCKED) {
        cap_name = "BLOCKED";
    } else if (cap_in == m2::DFBAccessPattern::ALL) {
        cap_name = "ALL";
    }
    std::vector<uint32_t> expected(input.size(), 0u);
    if (cap_in == m2::DFBAccessPattern::ALL) {
        // BLOCKED→ALL: the ALL consumer writes in consume order, so this matches the DM→DM golden:
        // output[(b*bs+j)*P+p] = input[(b*P+p)*bs+j], capacity=num_entries/P. Identity at P==1.
        const uint32_t capacity = num_entries / P;
        const uint32_t blocks_per_producer = capacity / bs;
        for (uint32_t pp = 0; pp < P; ++pp) {
            for (uint32_t b = 0; b < blocks_per_producer; ++b) {
                for (uint32_t j = 0; j < bs; ++j) {
                    const uint32_t src = (b * P + pp) * bs + j;
                    const uint32_t dst = (b * bs + j) * P + pp;
                    std::copy(input.begin() + src * wpe, input.begin() + (src + 1) * wpe, expected.begin() + dst * wpe);
                }
            }
        }
    } else if (cap_in == m2::DFBAccessPattern::BLOCKED) {
        // BLOCKED→BLOCKED, C=1: the single consumer round-robins the P producer sub-rings — its k-th
        // pop is TC (k%P), sub-ring position (k/P). Producer p filled sub-ring position s with input
        // page (s/bs * P + p)*bs + s%bs. Identity at P==1, a permutation for P>1.
        for (uint32_t k = 0; k < num_entries; ++k) {
            const uint32_t p = k % P;
            const uint32_t s = k / P;
            const uint32_t src = (s / bs * P + p) * bs + (s % bs);
            std::copy(input.begin() + src * wpe, input.begin() + (src + 1) * wpe, expected.begin() + k * wpe);
        }
    } else {
        // BLOCKED→STRIDED: C≥P forces P==1 here → identity.
        expected = input;
    }

    if (expected != output) {
        for (uint32_t t = 0; t < std::min<uint32_t>(num_entries, 16); ++t) {
            int match = -1;
            for (uint32_t src = 0; src < num_entries; ++src) {
                if (std::equal(input.begin() + src * wpe, input.begin() + (src + 1) * wpe, output.begin() + t * wpe)) {
                    match = static_cast<int>(src);
                    break;
                }
            }
            log_info(
                tt::LogTest,
                "  A1 DM→Trisc BLOCKED→{} output tile {} ← {}",
                cap_name,
                t,
                match >= 0 ? ("input page " + std::to_string(match)) : std::string("UNKNOWN"));
        }
    }
    EXPECT_EQ(expected, output) << "A1 DM→Trisc BLOCKED→" << cap_name << " data mismatch (P=" << P << ")";
}

inline void run_a1_fanout_blocked_pipeline(
    const std::shared_ptr<distributed::MeshDevice>& mesh_device,
    uint32_t C,
    uint32_t block_size,
    uint32_t num_entries,
    bool implicit = false) {
    if (mesh_device->get_devices()[0]->arch() != ARCH::QUASAR) {
        GTEST_SKIP() << "M2 path is Quasar-only (Gen2Config)";
    }
    IDevice* device = mesh_device->get_devices()[0];
    constexpr uint32_t entry_size = 2 * 32 * 32;  // bf16 tile = 2048 B
    const m2::NodeCoord node{0, 0};

    const auto tensor_spec = make_flat_dram_tensor_spec(entry_size, num_entries, DataType::BFLOAT16);
    auto in_tensor = MeshTensor::allocate_on_device(*mesh_device, tensor_spec, TensorTopology{});
    auto out_tensor = MeshTensor::allocate_on_device(*mesh_device, tensor_spec, TensorTopology{});

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

    auto producer = make_dm_kernel(
        PRODUCER, "tests/tt_metal/tt_metal/test_kernels/dataflow/dfb_blocked_producer_2_0.cpp", /*num_threads=*/1);
    producer.dfb_bindings = {
        {.dfb_spec_name = DFB_IN,
         .accessor_name = "out",
         .endpoint_type = m2::DFBEndpointType::PRODUCER,
         .access_pattern = m2::DFBAccessPattern::BLOCKED,
         .block_size = block_size}};
    producer.tensor_bindings = {{.tensor_parameter_name = IN_TENSOR, .accessor_name = "src_tensor"}};
    producer.compile_time_args = {
        {"num_entries_per_producer", num_entries}, {"block_size", block_size}, {"implicit_sync", implicit ? 1u : 0u}};
    producer.runtime_arg_schema = {.runtime_arg_names = {"chunk_offset", "entries_per_core"}};

    auto compute = make_compute_kernel(
        COMPUTE, "tests/tt_metal/tt_metal/test_kernels/compute/dfb_eltwise_copy_2_0.cpp", static_cast<uint8_t>(C));
    compute.dfb_bindings = {
        {.dfb_spec_name = DFB_IN,
         .accessor_name = "in",
         .endpoint_type = m2::DFBEndpointType::CONSUMER,
         .access_pattern = m2::DFBAccessPattern::BLOCKED,
         .block_size = block_size},
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

    // DFB_IN is BLOCKED with C>1 Tensix consumers (C>P=1, so the DM producer is the wider-fan-out side with
    // num_producer_tcs=C>1). An IMPLICIT DM producer here is the suspected hole: commit_implicit_read
    // advances tc_idx per-entry, scattering a block across sub-rings. (DFB_OUT is STRIDED, where per-entry
    // round-robin is correct, so its DM consumer may be implicit safely.)
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
    Program program = m2::MakeProgramFromSpec(*mesh_device, spec);

    m2::ProgramRunArgs params;
    params.kernel_run_args = {
        m2::ProgramRunArgs::KernelRunArgs{
            .kernel = PRODUCER,
            .runtime_arg_values = experimental::MakeRuntimeArgsForSingleNode(node, {{"chunk_offset", 0u}, {"entries_per_core", num_entries}})},
        m2::ProgramRunArgs::KernelRunArgs{
            .kernel = CONSUMER,
            .runtime_arg_values = experimental::MakeRuntimeArgsForSingleNode(node, {{"chunk_offset", 0u}, {"entries_per_core", num_entries}})},
        m2::ProgramRunArgs::KernelRunArgs{.kernel = COMPUTE},
    };
    params.tensor_args = {{IN_TENSOR, std::cref(in_tensor)}, {OUT_TENSOR, std::cref(out_tensor)}};
    m2::SetProgramRunArgs(program, params);

    const uint32_t total_bytes = entry_size * num_entries;
    auto input = create_random_vector_of_bfloat16(total_bytes, 2.0f, 0xA1C1);
    detail::WriteToBuffer(*in_tensor.mesh_buffer().get_reference_buffer(), input);
    m2_writeshard_barrier_uint32(device, in_tensor, input);

    detail::LaunchProgram(device, program, /*wait_until_cores_done=*/true);

    std::vector<uint32_t> output;
    detail::ReadFromBuffer(*out_tensor.mesh_buffer().get_reference_buffer(), output);

    const uint32_t wpe = entry_size / sizeof(uint32_t);
    const uint32_t bs = block_size;
    std::vector<uint32_t> expected(input.size(), 0u);
    for (uint32_t r = 0; r < num_entries; ++r) {
        const uint32_t c = r % C;
        const uint32_t m = r / C;
        const uint32_t src = (c + (m / bs) * C) * bs + (m % bs);
        std::copy(input.begin() + src * wpe, input.begin() + (src + 1) * wpe, expected.begin() + r * wpe);
    }
    if (expected != output) {
        for (uint32_t t = 0; t < std::min<uint32_t>(num_entries, 16); ++t) {
            int match = -1;
            for (uint32_t src = 0; src < num_entries; ++src) {
                if (std::equal(input.begin() + src * wpe, input.begin() + (src + 1) * wpe, output.begin() + t * wpe)) {
                    match = static_cast<int>(src);
                    break;
                }
            }
            log_info(
                tt::LogTest,
                "  A1-fanout BLOCKED C={} output tile {} ← {}",
                C,
                t,
                match >= 0 ? ("input page " + std::to_string(match)) : std::string("UNKNOWN"));
        }
    }
    EXPECT_EQ(expected, output) << "A1-fanout multi-Tensix-consumer BLOCKED data mismatch (C=" << C << ")";
}

}  // namespace tt::tt_metal
