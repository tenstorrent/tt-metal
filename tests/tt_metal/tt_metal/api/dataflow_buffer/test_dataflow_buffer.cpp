// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include <map>
#include <memory>
#include <numeric>
#include <optional>
#include <tuple>
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
#include <tt-metalium/experimental/host_api.hpp>
#include <tt-metalium/experimental/metal2_host_api/program.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_spec.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_run_params.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>

#include "device_fixture.hpp"
#include "tt_metal/test_utils/stimulus.hpp"
#include "tt_metal/hw/inc/internal/tt-2xx/dataflow_buffer/dataflow_buffer_config.h"
#include <tt-metalium/experimental/dataflow_buffer/dataflow_buffer.hpp>
#include "impl/program/program_impl.hpp"
#include "impl/kernels/kernel.hpp"
#include <tt-metalium/experimental/tensor/mesh_tensor.hpp>
#include <tt-metalium/experimental/tensor/topology/tensor_topology.hpp>
#include <tt-metalium/experimental/tensor/spec/tensor_spec.hpp>
#include <tt-metalium/experimental/tensor/spec/layout/tensor_layout.hpp>
#include <tt-metalium/experimental/tensor/spec/layout/page_config.hpp>

namespace tt::tt_metal {

enum class DFBPorCType : uint8_t { DM, TENSIX };

class DFBImplicitSyncParamFixture : public MeshDeviceFixture, public ::testing::WithParamInterface<bool> {};

static std::string ImplicitSyncParamName(const ::testing::TestParamInfo<bool>& info) {
    return info.param ? "ImplicitSyncTrue" : "ImplicitSyncFalse";
}

// Templated launch + verify helper.
//
// Templated over the kind of "buffer handle" (MeshBuffer or MeshTensor) so the Quasar
// inline I/O path used by Metal 2.0 sites can share the launch + Quasar barrier dance
// with the legacy MeshBuffer path. `write_input` / `read_output` perform the actual
// host I/O on the handle using whatever underlying API matches the handle's type.
//
// expected_output, when set, is compared against the device output instead of input.
// This is used for Tensix→DM ring-pressure tests where the device cycles through fewer
// unique ring slots than entries_per_core, so output != input by design.
template <typename Handle, typename Write, typename Read>
void execute_program_and_verify_impl(
    const std::shared_ptr<distributed::MeshDevice>& mesh_device,
    Program& program,
    Handle& in_handle,
    Handle& out_handle,
    std::vector<uint32_t>& input,
    Write&& write_input,
    Read&& read_output,
    bool verify_output,
    std::optional<std::vector<uint32_t>> expected_output) {
    write_input(in_handle, input);

    if (mesh_device->get_devices()[0]->arch() == ARCH::QUASAR) {
        // TODO #38042: Need to wait for data to be written, the barrier needs to be uplifted for Quasar
        std::this_thread::sleep_for(std::chrono::milliseconds(500));

        std::vector<uint32_t> rdback_dram;
        read_output(in_handle, rdback_dram);

        tt_driver_atomics::mfence();

        EXPECT_EQ(rdback_dram, input);
    }

    // Execute using slow dispatch (DFBs not yet supported in MeshWorkload path)
    IDevice* device = mesh_device->get_devices()[0];
    detail::LaunchProgram(device, program, true /*wait_until_cores_done*/);

    std::vector<uint32_t> output;
    read_output(out_handle, output);

    if (verify_output) {
        const std::vector<uint32_t>& expected = expected_output ? *expected_output : input;
        if (expected != output) {
            log_info(tt::LogTest, "Printing expected");
            for (auto i : expected) {
                std::cout << i << " ";
            }
            std::cout << std::endl;
            log_info(tt::LogTest, "Printing output");
            for (auto i : output) {
                std::cout << i << " ";
            }
        }
        EXPECT_EQ(expected, output);
    }
}

// Legacy MeshBuffer path: I/O via WriteShard / ReadShard.
void execute_program_and_verify(
    const std::shared_ptr<distributed::MeshDevice>& mesh_device,
    Program& program,
    const std::shared_ptr<distributed::MeshBuffer>& in_buffer,
    const std::shared_ptr<distributed::MeshBuffer>& out_buffer,
    distributed::MeshCoordinate& zero_coord,
    std::vector<uint32_t>& input,
    bool verify_output = true,
    std::optional<std::vector<uint32_t>> expected_output = std::nullopt) {
    auto& cq = mesh_device->mesh_command_queue();
    auto write = [&](const std::shared_ptr<distributed::MeshBuffer>& buf, std::vector<uint32_t>& vec) {
        distributed::WriteShard(cq, buf, vec, zero_coord, true);
    };
    auto read = [&](const std::shared_ptr<distributed::MeshBuffer>& buf, std::vector<uint32_t>& vec) {
        distributed::ReadShard(cq, vec, buf, zero_coord, true);
    };
    execute_program_and_verify_impl(
        mesh_device, program, in_buffer, out_buffer, input, write, read, verify_output, expected_output);
}

// Metal 2.0 MeshTensor path: I/O through the bound tensor's underlying reference buffer.
void execute_program_and_verify(
    const std::shared_ptr<distributed::MeshDevice>& mesh_device,
    Program& program,
    MeshTensor& in_tensor,
    MeshTensor& out_tensor,
    std::vector<uint32_t>& input,
    bool verify_output = true,
    std::optional<std::vector<uint32_t>> expected_output = std::nullopt) {
    auto write = [](MeshTensor& t, std::vector<uint32_t>& vec) {
        detail::WriteToBuffer(*t.mesh_buffer().get_reference_buffer(), vec);
    };
    auto read = [](MeshTensor& t, std::vector<uint32_t>& vec) {
        detail::ReadFromBuffer(*t.mesh_buffer().get_reference_buffer(), vec);
    };
    execute_program_and_verify_impl(
        mesh_device, program, in_tensor, out_tensor, input, write, read, verify_output, expected_output);
}

// Build a TensorSpec describing a flat DRAM-interleaved buffer of `total_entries`
// pages, each `entry_size` bytes. Used so an existing buffer can be wrapped in a
// MeshTensor for binding via TensorParameter; underlying storage is unchanged.
inline tt::tt_metal::TensorSpec make_flat_dram_tensor_spec(uint32_t entry_size, uint32_t total_entries) {
    const uint32_t entry_size_words = entry_size / sizeof(uint32_t);
    auto page_config = tt::tt_metal::PageConfig(tt::tt_metal::Layout::ROW_MAJOR);
    auto memory_config =
        tt::tt_metal::MemoryConfig{tt::tt_metal::TensorMemoryLayout::INTERLEAVED, tt::tt_metal::BufferType::DRAM};
    auto tensor_layout = tt::tt_metal::TensorLayout(tt::tt_metal::DataType::UINT32, page_config, memory_config);
    return tt::tt_metal::TensorSpec(tt::tt_metal::Shape{total_entries, entry_size_words}, tensor_layout);
}

// Runs a single DFB program on one or more cores and verifies output == input.
//
// When core_range_set contains N > 1 cores the global DRAM buffers are sized
// N x entries_per_core x entry_size and each core receives a unique
// chunk_offset (= core_idx * entries_per_core) so it accesses a disjoint
// slice of the buffer.  Multi-core use requires DM producer and consumer.
void run_single_dfb_program(
    const std::shared_ptr<distributed::MeshDevice>& mesh_device,
    experimental::dfb::DataflowBufferConfig& dfb_config,
    DFBPorCType producer_type,
    DFBPorCType consumer_type,
    const CoreRangeSet& core_range_set = CoreRangeSet(CoreRange(CoreCoord(0, 0), CoreCoord(0, 0))),
    std::optional<uint32_t> num_entries_in_buffer = std::nullopt) {

    TT_FATAL(
        !(producer_type == DFBPorCType::TENSIX && consumer_type == DFBPorCType::TENSIX),
        "Both producer and consumer cannot be Tensix. At least one must be a DM kernel for NOC transfers.");
    TT_FATAL(
        core_range_set.num_cores() == 1 ||
            (producer_type == DFBPorCType::DM && consumer_type == DFBPorCType::DM),
        "Multi-core DFB programs only support DM producer and consumer.");

    const auto arch = mesh_device->get_devices()[0]->arch();
    const bool is_quasar = (arch == ARCH::QUASAR);

    if (!is_quasar) {
        // WH/BH DM: one BRISC (RISCV_0) as producer and one NCRISC (RISCV_1) as consumer.
        // Configs with num_producers > 1 or num_consumers > 1 require multi-threaded DM
        // which is not available on WH/BH.
        if (dfb_config.num_producers > 1 || dfb_config.num_consumers > 1) {
            GTEST_SKIP() << "WH/BH DFB supports only 1 DM producer (BRISC) and 1 DM consumer (NCRISC)";
        }
        // read_in / write_out are Quasar-only; the device-side kernel would fail to compile
        // if enable_implicit_sync=true is propagated as a compile-time arg.
        dfb_config.enable_implicit_sync = false;
    }

    auto zero_coord = distributed::MeshCoordinate(0, 0);

    const uint32_t num_cores = core_range_set.num_cores();
    const uint32_t entries_per_core = num_entries_in_buffer.has_value() ? num_entries_in_buffer.value() : dfb_config.num_entries;
    const uint32_t entry_size = dfb_config.entry_size;
    // page_size = entry_size makes every entry independently addressable by page_id.
    const uint32_t total_buffer_size = num_cores * entries_per_core * entry_size;
    const uint32_t total_entries = num_cores * entries_per_core;
    const bool is_all = (dfb_config.cap == dfb::AccessPattern::ALL);

    // Ceiling division so every producer gets a loop bound that covers the largest slice.
    // Producers whose page_id would exceed entries_per_core use the runtime bounds
    // check in the kernel to skip the out-of-range iteration.
    const uint32_t num_entries_per_producer =
        (entries_per_core + dfb_config.num_producers - 1) / dfb_config.num_producers;
    const uint32_t num_entries_per_consumer =
        is_all ? entries_per_core : (entries_per_core + dfb_config.num_consumers - 1) / dfb_config.num_consumers;

    // Build a per-core chunk-offset map (used for both runtime args and L1 pre-fill/verify).
    std::map<CoreCoord, uint32_t> core_to_chunk_offset;
    {
        uint32_t core_idx = 0;
        for (const CoreRange& cr : core_range_set.ranges()) {
            for (auto y = cr.start_coord.y; y <= cr.end_coord.y; y++) {
                for (auto x = cr.start_coord.x; x <= cr.end_coord.x; x++) {
                    core_to_chunk_offset[CoreCoord(x, y)] = core_idx++ * entries_per_core;
                }
            }
        }
    }

    Program program;
    // Quasar branch allocates input/output as MeshTensors (bound via TensorParameter);
    // legacy WH/BH branch allocates MeshBuffers directly.
    MeshTensor in_tensor;
    MeshTensor out_tensor;
    std::shared_ptr<distributed::MeshBuffer> in_buffer;
    std::shared_ptr<distributed::MeshBuffer> out_buffer;
    if (is_quasar) {
        constexpr const char* DFB_NAME = "dfb";
        constexpr const char* PRODUCER = "producer";
        constexpr const char* CONSUMER = "consumer";
        constexpr const char* IN_TENSOR = "in_tensor";
        constexpr const char* OUT_TENSOR = "out_tensor";

        // Only DM kernels bind to DRAM tensors; Tensix kernels operate purely on L1 DFB rings
        // (host pre-fills L1 for Tensix producers; verifies via L1 read for Tensix consumers).
        // Declaring an unbound TensorParameter triggers ProgramSpec validation failure.
        const bool need_in_tensor = (producer_type == DFBPorCType::DM);
        const bool need_out_tensor = (consumer_type == DFBPorCType::DM);

        const auto tensor_spec = make_flat_dram_tensor_spec(entry_size, total_entries);
        if (need_in_tensor) {
            in_tensor = MeshTensor::allocate_on_device(*mesh_device, tensor_spec, TensorTopology{});
            log_info(
                tt::LogTest,
                "In Tensor:  [address: {} B, size: {} B]",
                in_tensor.mesh_buffer().get_reference_buffer()->address(),
                in_tensor.mesh_buffer().get_reference_buffer()->size());
        }
        if (need_out_tensor) {
            out_tensor = MeshTensor::allocate_on_device(*mesh_device, tensor_spec, TensorTopology{});
            log_info(
                tt::LogTest,
                "Out Tensor: [address: {} B, size: {} B]",
                out_tensor.mesh_buffer().get_reference_buffer()->address(),
                out_tensor.mesh_buffer().get_reference_buffer()->size());
        }

        const auto consumer_pattern = is_all ? experimental::metal2_host_api::DFBAccessPattern::ALL
                                             : experimental::metal2_host_api::DFBAccessPattern::STRIDED;

        // enable_implicit_sync: true  -> default disable_implicit_sync = false (implicit sync ON)
        // enable_implicit_sync: false -> disable_implicit_sync = true
        experimental::metal2_host_api::DataflowBufferSpec dfb_spec{
            .unique_id = DFB_NAME,
            .entry_size = entry_size,
            .num_entries = dfb_config.num_entries,
            .data_format_metadata = dfb_config.data_format,
            .disable_implicit_sync = !dfb_config.enable_implicit_sync,
        };

        experimental::metal2_host_api::KernelSpec producer_spec;
        if (producer_type == DFBPorCType::DM) {
            producer_spec = experimental::metal2_host_api::KernelSpec{
                .unique_id = PRODUCER,
                .source =
                    experimental::metal2_host_api::KernelSpec::SourceFilePath{
                        "tests/tt_metal/tt_metal/test_kernels/dataflow/dfb_producer.cpp"},
                .num_threads = static_cast<uint8_t>(dfb_config.num_producers),
                .dfb_bindings = {{
                    .dfb_spec_name = DFB_NAME,
                    .local_accessor_name = "out",
                    .endpoint_type = experimental::metal2_host_api::KernelSpec::DFBEndpointType::PRODUCER,
                    .access_pattern = experimental::metal2_host_api::DFBAccessPattern::STRIDED,
                }},
                .tensor_bindings = {{
                    .tensor_parameter_name = IN_TENSOR,
                    .accessor_name = "src_tensor",  // kernel: ta::src_tensor
                }},
                .compile_time_arg_bindings =
                    {
                        {"num_entries_per_producer", num_entries_per_producer},
                        {"implicit_sync", static_cast<uint32_t>(dfb_config.enable_implicit_sync ? 1u : 0u)},
                        {"num_producers", dfb_config.num_producers},
                    },
                .runtime_arguments_schema = {.named_runtime_args = {"chunk_offset", "entries_per_core"}},
                .config_spec =
                    experimental::metal2_host_api::DataMovementConfiguration{
                        .gen2_data_movement_config =
                            experimental::metal2_host_api::DataMovementConfiguration::Gen2DataMovementConfig{}},
            };
        } else {
            producer_spec = experimental::metal2_host_api::KernelSpec{
                .unique_id = PRODUCER,
                .source =
                    experimental::metal2_host_api::KernelSpec::SourceFilePath{
                        "tests/tt_metal/tt_metal/test_kernels/compute/dfb_t6_producer.cpp"},
                .num_threads = static_cast<uint8_t>(dfb_config.num_producers),
                .dfb_bindings = {{
                    .dfb_spec_name = DFB_NAME,
                    .local_accessor_name = "out",
                    .endpoint_type = experimental::metal2_host_api::KernelSpec::DFBEndpointType::PRODUCER,
                    .access_pattern = experimental::metal2_host_api::DFBAccessPattern::STRIDED,
                }},
                .compile_time_arg_bindings = {{"num_entries_per_producer", num_entries_per_producer}},
                .config_spec = experimental::metal2_host_api::ComputeConfiguration{},
            };
        }

        experimental::metal2_host_api::KernelSpec consumer_spec;
        if (consumer_type == DFBPorCType::DM) {
            consumer_spec = experimental::metal2_host_api::KernelSpec{
                .unique_id = CONSUMER,
                .source =
                    experimental::metal2_host_api::KernelSpec::SourceFilePath{
                        "tests/tt_metal/tt_metal/test_kernels/dataflow/dfb_consumer.cpp"},
                .num_threads = static_cast<uint8_t>(dfb_config.num_consumers),
                .dfb_bindings = {{
                    .dfb_spec_name = DFB_NAME,
                    .local_accessor_name = "in",
                    .endpoint_type = experimental::metal2_host_api::KernelSpec::DFBEndpointType::CONSUMER,
                    .access_pattern = consumer_pattern,
                }},
                .tensor_bindings = {{
                    .tensor_parameter_name = OUT_TENSOR,
                    .accessor_name = "dst_tensor",  // kernel: ta::dst_tensor
                }},
                .compile_time_arg_bindings =
                    {
                        {"num_entries_per_consumer", num_entries_per_consumer},
                        {"blocked_consumer", static_cast<uint32_t>(is_all ? 1u : 0u)},
                        {"implicit_sync", static_cast<uint32_t>(dfb_config.enable_implicit_sync ? 1u : 0u)},
                        {"num_consumers", dfb_config.num_consumers},
                    },
                .runtime_arguments_schema = {.named_runtime_args = {"chunk_offset", "entries_per_core"}},
                .config_spec =
                    experimental::metal2_host_api::DataMovementConfiguration{
                        .gen2_data_movement_config =
                            experimental::metal2_host_api::DataMovementConfiguration::Gen2DataMovementConfig{}},
            };
        } else {
            consumer_spec = experimental::metal2_host_api::KernelSpec{
                .unique_id = CONSUMER,
                .source =
                    experimental::metal2_host_api::KernelSpec::SourceFilePath{
                        "tests/tt_metal/tt_metal/test_kernels/compute/dfb_t6_consumer.cpp"},
                .num_threads = static_cast<uint8_t>(dfb_config.num_consumers),
                .dfb_bindings = {{
                    .dfb_spec_name = DFB_NAME,
                    .local_accessor_name = "in",
                    .endpoint_type = experimental::metal2_host_api::KernelSpec::DFBEndpointType::CONSUMER,
                    .access_pattern = consumer_pattern,
                }},
                .compile_time_arg_bindings = {{"num_entries_per_consumer", num_entries_per_consumer}},
                .config_spec = experimental::metal2_host_api::ComputeConfiguration{},
            };
        }

        experimental::metal2_host_api::WorkUnitSpec wu{
            .unique_id = "main",
            .kernels = {PRODUCER, CONSUMER},
            .target_nodes = core_range_set,
        };

        std::vector<experimental::metal2_host_api::TensorParameter> tensor_parameters;
        if (need_in_tensor) {
            tensor_parameters.push_back({.unique_id = IN_TENSOR, .spec = in_tensor.tensor_spec()});
        }
        if (need_out_tensor) {
            tensor_parameters.push_back({.unique_id = OUT_TENSOR, .spec = out_tensor.tensor_spec()});
        }

        experimental::metal2_host_api::ProgramSpec spec{
            .program_id = "single_dfb",
            .kernels = {producer_spec, consumer_spec},
            .dataflow_buffers = {dfb_spec},
            .tensor_parameters = tensor_parameters,
            .work_units = {wu},
        };

        program = experimental::metal2_host_api::MakeProgramFromSpec(*mesh_device, spec);

        using NodeNamedRTAs = experimental::metal2_host_api::ProgramRunParams::KernelRunParams::NodeNamedRTAs;
        auto build_dm_named_rtas = [&]() {
            std::vector<NodeNamedRTAs> result;
            result.reserve(core_to_chunk_offset.size());
            for (const auto& [core, chunk_offset] : core_to_chunk_offset) {
                result.push_back(NodeNamedRTAs{
                    experimental::metal2_host_api::NodeCoord{core.x, core.y},
                    {{"chunk_offset", chunk_offset}, {"entries_per_core", entries_per_core}}});
            }
            return result;
        };

        experimental::metal2_host_api::ProgramRunParams run_params;
        experimental::metal2_host_api::ProgramRunParams::KernelRunParams producer_params{.kernel_spec_name = PRODUCER};
        if (producer_type == DFBPorCType::DM) {
            producer_params.named_runtime_args = build_dm_named_rtas();
        }
        experimental::metal2_host_api::ProgramRunParams::KernelRunParams consumer_params{.kernel_spec_name = CONSUMER};
        if (consumer_type == DFBPorCType::DM) {
            consumer_params.named_runtime_args = build_dm_named_rtas();
        }
        run_params.kernel_run_params = {producer_params, consumer_params};
        if (need_in_tensor) {
            run_params.tensor_args.push_back({.tensor_parameter_name = IN_TENSOR, .tensor = std::cref(in_tensor)});
        }
        if (need_out_tensor) {
            run_params.tensor_args.push_back({.tensor_parameter_name = OUT_TENSOR, .tensor = std::cref(out_tensor)});
        }
        experimental::metal2_host_api::SetProgramRunParameters(program, run_params);
    } else {
        distributed::DeviceLocalBufferConfig local_buffer_config{
            .page_size = entry_size, .buffer_type = BufferType::DRAM};
        distributed::ReplicatedBufferConfig buffer_config{.size = total_buffer_size};
        in_buffer = distributed::MeshBuffer::create(buffer_config, local_buffer_config, mesh_device.get());
        out_buffer = distributed::MeshBuffer::create(buffer_config, local_buffer_config, mesh_device.get());

        log_info(tt::LogTest, "In Buffer:  [address: {} B, size: {} B]", in_buffer->address(), in_buffer->size());
        log_info(tt::LogTest, "Out Buffer: [address: {} B, size: {} B]", out_buffer->address(), out_buffer->size());

        program = CreateProgram();

        std::vector<uint32_t> producer_cta = {
            (uint32_t)in_buffer->address(), num_entries_per_producer, (uint32_t)dfb_config.enable_implicit_sync};
        tt::tt_metal::TensorAccessorArgs(in_buffer).append_to(producer_cta);

        KernelHandle producer_kernel;
        if (producer_type == DFBPorCType::DM) {
            producer_kernel = CreateKernel(
                program,
                "tests/tt_metal/tt_metal/test_kernels/dataflow/dfb_producer.cpp",
                core_range_set,
                DataMovementConfig{.processor = DataMovementProcessor::RISCV_0, .compile_args = producer_cta});
        } else {
            producer_kernel = CreateKernel(
                program,
                "tests/tt_metal/tt_metal/test_kernels/compute/dfb_t6_producer.cpp",
                core_range_set,
                ComputeConfig{.compile_args = producer_cta});
        }

        std::vector<uint32_t> consumer_cta = {
            (uint32_t)out_buffer->address(),
            num_entries_per_consumer,
            (uint32_t)is_all,
            (uint32_t)dfb_config.enable_implicit_sync};
        tt::tt_metal::TensorAccessorArgs(out_buffer).append_to(consumer_cta);

        KernelHandle consumer_kernel;
        if (consumer_type == DFBPorCType::DM) {
            consumer_kernel = CreateKernel(
                program,
                "tests/tt_metal/tt_metal/test_kernels/dataflow/dfb_consumer.cpp",
                core_range_set,
                DataMovementConfig{.processor = DataMovementProcessor::RISCV_1, .compile_args = consumer_cta});
        } else {
            consumer_kernel = CreateKernel(
                program,
                "tests/tt_metal/tt_metal/test_kernels/compute/dfb_t6_consumer.cpp",
                core_range_set,
                ComputeConfig{.compile_args = consumer_cta});
        }

        auto logical_dfb_id = experimental::dfb::CreateDataflowBuffer(program, core_range_set, dfb_config);
        experimental::dfb::BindDataflowBufferToProducerConsumerKernels(
            program, logical_dfb_id, producer_kernel, consumer_kernel);

        auto dfb = program.impl().get_dataflow_buffer(logical_dfb_id);
        const uint32_t producer_mask = dfb->config.producer_risc_mask;
        const uint32_t consumer_mask = dfb->config.consumer_risc_mask;

        for (const auto& [core, chunk_offset] : core_to_chunk_offset) {
            SetRuntimeArgs(program, producer_kernel, core, {producer_mask, chunk_offset, entries_per_core});
            SetRuntimeArgs(
                program,
                consumer_kernel,
                core,
                {consumer_mask, (uint32_t)logical_dfb_id, chunk_offset, entries_per_core});
        }
    }

    // Generate input once; shared by tensor/buffer write, L1 pre-fill, and verification.
    auto input = tt::test_utils::generate_uniform_random_vector<uint32_t>(0, 100, total_buffer_size / sizeof(uint32_t));

    IDevice* device = mesh_device->get_devices()[0];

    // For Tensix → DM: pre-fill each core's DFB L1 with its input chunk so the
    // Tensix producer kernel can read from L1 while DM consumer drains to DRAM.
    //
    // Single-DFB programs always place the DFB at the L1 base allocator address
    // on every core where it's bound, so we use that directly here (instead of
    // introspecting dfb->groups[].l1_by_core, which is only populated after legacy
    // program compilation).
    //
    // IMPORTANT: the slice written to L1 must be exactly the physical ring size
    // (num_entries * entry_size). Writing more than the ring size would corrupt
    // L1 beyond the ring. For ring-pressure tests (entries_per_core > num_entries)
    // only the first num_entries slots are filled; the producer kernel cycles
    // through those same slots repeatedly.
    const uint32_t ring_total_bytes = dfb_config.num_entries * entry_size;
    const uint32_t ring_words = ring_total_bytes / sizeof(uint32_t);
    if (producer_type == DFBPorCType::TENSIX) {
        const uint32_t dfb_l1_addr =
            static_cast<uint32_t>(device->allocator()->get_base_allocator_addr(HalMemType::L1));
        for (const auto& [core, co] : core_to_chunk_offset) {
            const uint32_t wpe = entry_size / sizeof(uint32_t);
            std::vector<uint32_t> slice(ring_words, 0);
            for (uint32_t p = 0; p < dfb_config.num_producers; p++) {
                for (uint32_t e = 0; e < num_entries_per_producer; e++) {
                    const uint32_t page_id = co + e * dfb_config.num_producers + p;
                    if (page_id >= co + entries_per_core) {
                        break;
                    }
                    // Ring layout depends on stride_in_entries, which is set by the
                    // consumer access pattern:
                    //   STRIDED: stride = num_producers → interleaved (slot = e*P + p)
                    //   ALL: stride = 1 → TC-first   (slot = p*E + e)
                    const uint32_t dst_slot = (dfb_config.cap == dfb::AccessPattern::ALL)
                                                  ? (p * num_entries_per_producer + e)
                                                  : (e * dfb_config.num_producers + p);

                    // Stop once all physical ring slots are filled; for ring-pressure
                    // tests the remaining iterations would alias back to already-filled
                    // slots, so there is nothing new to write.
                    if (dst_slot >= dfb_config.num_entries) {
                        break;
                    }

                    std::copy(
                        input.begin() + page_id * wpe,
                        input.begin() + page_id * wpe + wpe,
                        slice.begin() + dst_slot * wpe);
                }
            }
            detail::WriteToDeviceL1(device, core, dfb_l1_addr, slice);
        }
    }

    // For Tensix → DM ring-pressure tests (entries_per_core > num_entries), the
    // Tensix producer cycles through the same num_entries ring slots indefinitely.
    // Each STRIDED consumer c always reads ring slot (c % num_entries), which was
    // pre-filled with input page c.  The expected out_buffer page p therefore
    // contains the data from ring slot (p % num_consumers) % num_entries, not
    // input[p].  Build the corrected expected vector so the verification is sound.
    std::optional<std::vector<uint32_t>> tensix_dm_expected;
    if (producer_type == DFBPorCType::TENSIX && consumer_type == DFBPorCType::DM &&
        entries_per_core > dfb_config.num_entries && dfb_config.cap == dfb::AccessPattern::STRIDED) {
        const uint32_t wpe = entry_size / sizeof(uint32_t);
        tensix_dm_expected.emplace(num_cores * entries_per_core * wpe, 0u);
        for (const auto& [core, co] : core_to_chunk_offset) {
            for (uint32_t p = 0; p < entries_per_core; p++) {
                // Consumer c = p % num_consumers always reads the ring slot it
                // was assigned (slot = c % num_entries), which holds input[co + c].
                const uint32_t ring_slot = (p % dfb_config.num_consumers) % dfb_config.num_entries;
                std::copy(
                    input.begin() + (co + ring_slot) * wpe,
                    input.begin() + (co + ring_slot + 1) * wpe,
                    tensix_dm_expected->begin() + (co + p) * wpe);
            }
        }
    }

    // Launch program; verify out_buffer/out_tensor only for DM → DM paths (Tensix
    // consumer does not write to DRAM, so out_buffer verification is skipped there).
    const bool verify_output = (consumer_type == DFBPorCType::DM);
    if (is_quasar) {
        // Tensor parameters are conditionally declared: only DM kernels carry tensor bindings.
        // Inline the write/read flow here so we can skip operations on unallocated tensors.
        if (producer_type == DFBPorCType::DM) {
            detail::WriteToBuffer(*in_tensor.mesh_buffer().get_reference_buffer(), input);
            std::this_thread::sleep_for(std::chrono::milliseconds(500));
            std::vector<uint32_t> rdback_dram;
            detail::ReadFromBuffer(*in_tensor.mesh_buffer().get_reference_buffer(), rdback_dram);
            tt_driver_atomics::mfence();
            EXPECT_EQ(rdback_dram, input);
        }

        detail::LaunchProgram(device, program, true /*wait_until_cores_done*/);

        if (verify_output) {
            std::vector<uint32_t> output;
            detail::ReadFromBuffer(*out_tensor.mesh_buffer().get_reference_buffer(), output);
            const std::vector<uint32_t>& expected = tensix_dm_expected ? *tensix_dm_expected : input;
            if (expected != output) {
                log_info(tt::LogTest, "Printing expected");
                for (auto i : expected) {
                    std::cout << i << " ";
                }
                std::cout << std::endl;
                log_info(tt::LogTest, "Printing output");
                for (auto i : output) {
                    std::cout << i << " ";
                }
            }
            EXPECT_EQ(expected, output);
        }
    } else {
        execute_program_and_verify(
            mesh_device, program, in_buffer, out_buffer, zero_coord, input, verify_output, tensix_dm_expected);
    }

    // For DM → Tensix: verify each core's DFB L1 against the expected input chunk.
    // Single-DFB programs place the DFB at the L1 base allocator address on every
    // bound core, so we iterate core_to_chunk_offset directly (works for both the
    // metal2 path and the legacy path).
    if (consumer_type == DFBPorCType::TENSIX) {
        const uint32_t dfb_l1_addr =
            static_cast<uint32_t>(device->allocator()->get_base_allocator_addr(HalMemType::L1));
        const uint32_t total_ring_words = ring_words;
        const uint32_t wpe_v = entry_size / sizeof(uint32_t);
        for (const auto& [core, co] : core_to_chunk_offset) {
            std::vector<uint32_t> l1_data;
            detail::ReadFromDeviceL1(device, core, dfb_l1_addr, ring_total_bytes, l1_data);
            // Physical ring holds dfb_config.num_entries entries; for ring-pressure
            // tests (entries_per_core > dfb_config.num_entries) the ring wraps and
            // only the last ring_capacity writes per producer survive in L1.
            std::vector<uint32_t> expected(total_ring_words, 0);
            if (dfb_config.cap == dfb::AccessPattern::ALL) {
                // ALL consumer: ring is TC-first (stride_in_entries=1).
                // Each producer p has ring_capacity consecutive ring slots.
                // After wrapping, only the last ring_capacity entries from each
                // producer survive: e in [num_entries_per_producer - ring_capacity, ...).
                const uint32_t ring_capacity = dfb_config.num_entries / dfb_config.num_producers;
                const uint32_t last_e_base = num_entries_per_producer - ring_capacity;
                for (uint32_t p = 0; p < dfb_config.num_producers; p++) {
                    for (uint32_t c = 0; c < ring_capacity; c++) {
                        const uint32_t ring_slot = p * ring_capacity + c;
                        const uint32_t e = last_e_base + c;
                        const uint32_t page_id = co + e * dfb_config.num_producers + p;
                        if (page_id >= co + entries_per_core) {
                            break;
                        }
                        std::copy(
                            input.begin() + page_id * wpe_v,
                            input.begin() + page_id * wpe_v + wpe_v,
                            expected.begin() + ring_slot * wpe_v);
                    }
                }
            } else {
                // STRIDED consumer: ring is interleaved, matching sequential input order.
                // For ring-pressure tests (entries_per_core > dfb_config.num_entries) only
                // the last dfb_config.num_entries entries survive in L1; copy that suffix.
                const uint32_t ring_start_page = co + entries_per_core - dfb_config.num_entries;
                std::copy(
                    input.begin() + ring_start_page * wpe_v,
                    input.begin() + ring_start_page * wpe_v + total_ring_words,
                    expected.begin());
            }
            if (expected != l1_data) {
                std::cout << "expected: ";
                for (const auto& e : expected) {
                    std::cout << e << " ";
                }
                std::cout << std::endl;
                std::cout << "l1_data: ";
                for (const auto& l : l1_data) {
                    std::cout << l << " ";
                }
                std::cout << std::endl;
            }
            EXPECT_EQ(expected, l1_data) << "DFB L1 mismatch on core (" << core.x << "," << core.y << ")";
        }
    }
}

// =====================================================================================
// Gap 7 harness – concurrent DFBs on the same core (TC allocator stress)
//
// Runs `num_dfbs` independent 1Sx1S DM→DM DFBs simultaneously on core (0,0).
//
// Thread assignment (Quasar has 8 DM threads total):
//   Producer threads : DM[0 .. num_dfbs-1]  (combined_producer_mask = low  num_dfbs bits)
//   Consumer threads : DM[num_dfbs .. 2*num_dfbs-1] (combined_consumer_mask = next num_dfbs bits)
//
// All num_dfbs DFBs are created in a single Program so their TCs are allocated
// simultaneously, stressing the TC allocator.  Each DFB is bound to the same
// multi-producer and multi-consumer kernel; each DM thread derives its DFB ID from
// its position in the combined mask (via mhartid) and owns a contiguous slice of
// the shared DRAM buffers.
//
// num_dfbs must satisfy: 2 * num_dfbs <= 8 (Quasar DM thread limit).
// dfb_config must use num_producers=1, num_consumers=1 (1Sx1S).
// =====================================================================================
void run_concurrent_dfbs_program(
    const std::shared_ptr<distributed::MeshDevice>& mesh_device,
    uint32_t num_dfbs,
    experimental::dfb::DataflowBufferConfig& dfb_config) {
    TT_FATAL(
        mesh_device->get_devices()[0]->arch() == ARCH::QUASAR,
        "Concurrent DFB tests require Quasar (multi-threaded DM)");
    TT_FATAL(
        dfb_config.num_producers == 1 && dfb_config.num_consumers == 1,
        "run_concurrent_dfbs_program requires 1Sx1S per DFB");
    TT_FATAL(2 * num_dfbs <= 6, "2*num_dfbs must fit in the 6 available Quasar DM threads (DM0 reserved for dispatch)");

    const CoreCoord core(0, 0);
    const CoreRangeSet core_range_set(CoreRange(core, core));

    const uint32_t entry_size      = dfb_config.entry_size;
    const uint32_t entries_per_dfb = dfb_config.num_entries;
    const uint32_t total_buf_size  = num_dfbs * entries_per_dfb * entry_size;
    const uint32_t total_entries = num_dfbs * entries_per_dfb;

    log_info(
        tt::LogTest,
        "Gap7: {} concurrent 1Sx1S DFBs, {} entries/DFB, total {}B in/out tensors",
        num_dfbs,
        entries_per_dfb,
        total_buf_size);

    // Shared in_tensor / out_tensor; per-DFB chunk_offset selects this DFB's slice.
    const auto tensor_spec = make_flat_dram_tensor_spec(entry_size, total_entries);
    auto in_tensor = MeshTensor::allocate_on_device(*mesh_device, tensor_spec, TensorTopology{});
    auto out_tensor = MeshTensor::allocate_on_device(*mesh_device, tensor_spec, TensorTopology{});

    constexpr const char* IN_TENSOR = "in_tensor";
    constexpr const char* OUT_TENSOR = "out_tensor";

    // Build N DataflowBufferSpec, 2N KernelSpec (1 producer + 1 consumer per DFB).
    // Each kernel instance binds to exactly its own DFB; this matches num_producers=1
    // / num_consumers=1 in the per-DFB config.  A shared kernel bound to N DFBs would
    // erroneously register all N threads as producers for every DFB.
    std::vector<experimental::metal2_host_api::DataflowBufferSpec> dfb_specs;
    std::vector<experimental::metal2_host_api::KernelSpec> kernel_specs;
    std::vector<std::string> kernel_names;
    dfb_specs.reserve(num_dfbs);
    kernel_specs.reserve(2 * num_dfbs);
    kernel_names.reserve(2 * num_dfbs);

    for (uint32_t i = 0; i < num_dfbs; ++i) {
        const uint32_t chunk_offset = i * entries_per_dfb;
        const std::string dfb_name = "dfb_" + std::to_string(i);
        const std::string producer_name = "producer_" + std::to_string(i);
        const std::string consumer_name = "consumer_" + std::to_string(i);

        dfb_specs.push_back(experimental::metal2_host_api::DataflowBufferSpec{
            .unique_id = dfb_name,
            .entry_size = entry_size,
            .num_entries = entries_per_dfb,
            .data_format_metadata = dfb_config.data_format,
            .disable_implicit_sync = !dfb_config.enable_implicit_sync,
        });

        kernel_specs.push_back(experimental::metal2_host_api::KernelSpec{
            .unique_id = producer_name,
            .source =
                experimental::metal2_host_api::KernelSpec::SourceFilePath{
                    "tests/tt_metal/tt_metal/test_kernels/dataflow/dfb_multi_producer.cpp"},
            .num_threads = 1,
            .dfb_bindings = {{
                .dfb_spec_name = dfb_name,
                .local_accessor_name = "out",
                .endpoint_type = experimental::metal2_host_api::KernelSpec::DFBEndpointType::PRODUCER,
                .access_pattern = experimental::metal2_host_api::DFBAccessPattern::STRIDED,
            }},
            .tensor_bindings = {{
                .tensor_parameter_name = IN_TENSOR,
                .accessor_name = "src_tensor",
            }},
            .compile_time_arg_bindings =
                {
                    {"num_entries_per_producer", entries_per_dfb},
                    {"implicit_sync", static_cast<uint32_t>(dfb_config.enable_implicit_sync ? 1u : 0u)},
                    {"chunk_offset", chunk_offset},
                },
            .config_spec =
                experimental::metal2_host_api::DataMovementConfiguration{
                    .gen2_data_movement_config =
                        experimental::metal2_host_api::DataMovementConfiguration::Gen2DataMovementConfig{}},
        });
        kernel_names.push_back(producer_name);

        kernel_specs.push_back(experimental::metal2_host_api::KernelSpec{
            .unique_id = consumer_name,
            .source =
                experimental::metal2_host_api::KernelSpec::SourceFilePath{
                    "tests/tt_metal/tt_metal/test_kernels/dataflow/dfb_multi_consumer.cpp"},
            .num_threads = 1,
            .dfb_bindings = {{
                .dfb_spec_name = dfb_name,
                .local_accessor_name = "in",
                .endpoint_type = experimental::metal2_host_api::KernelSpec::DFBEndpointType::CONSUMER,
                .access_pattern = experimental::metal2_host_api::DFBAccessPattern::STRIDED,
            }},
            .tensor_bindings = {{
                .tensor_parameter_name = OUT_TENSOR,
                .accessor_name = "dst_tensor",
            }},
            .compile_time_arg_bindings =
                {
                    {"num_entries_per_consumer", entries_per_dfb},
                    {"implicit_sync", static_cast<uint32_t>(dfb_config.enable_implicit_sync ? 1u : 0u)},
                    {"chunk_offset", chunk_offset},
                },
            .config_spec =
                experimental::metal2_host_api::DataMovementConfiguration{
                    .gen2_data_movement_config =
                        experimental::metal2_host_api::DataMovementConfiguration::Gen2DataMovementConfig{}},
        });
        kernel_names.push_back(consumer_name);
    }

    experimental::metal2_host_api::WorkUnitSpec wu{
        .unique_id = "main",
        .kernels = kernel_names,
        .target_nodes = core_range_set,
    };

    experimental::metal2_host_api::ProgramSpec spec{
        .program_id = "concurrent_dfbs",
        .kernels = kernel_specs,
        .dataflow_buffers = dfb_specs,
        .tensor_parameters =
            {
                {.unique_id = IN_TENSOR, .spec = in_tensor.tensor_spec()},
                {.unique_id = OUT_TENSOR, .spec = out_tensor.tensor_spec()},
            },
        .work_units = {wu},
    };

    Program program = experimental::metal2_host_api::MakeProgramFromSpec(*mesh_device, spec);

    // No per-instance runtime args needed (chunk_offset is a CTA).
    experimental::metal2_host_api::ProgramRunParams run_params;
    run_params.kernel_run_params.reserve(kernel_names.size());
    for (const auto& name : kernel_names) {
        run_params.kernel_run_params.push_back(
            experimental::metal2_host_api::ProgramRunParams::KernelRunParams{.kernel_spec_name = name});
    }
    run_params.tensor_args = {
        {.tensor_parameter_name = IN_TENSOR, .tensor = std::cref(in_tensor)},
        {.tensor_parameter_name = OUT_TENSOR, .tensor = std::cref(out_tensor)},
    };
    experimental::metal2_host_api::SetProgramRunParameters(program, run_params);

    auto input = tt::test_utils::generate_uniform_random_vector<uint32_t>(
        0, 100, total_buf_size / sizeof(uint32_t));
    execute_program_and_verify(mesh_device, program, in_tensor, out_tensor, input);
}

// =====================================================================================
// Tensix→DM concurrent DFBs
//
// Runs num_dfbs independent 1Sx1S Tensix→DM DFBs on core (0,0) with a single
// Neo thread looping through all DFBs sequentially and num_dfbs DM consumer threads
// running concurrently (each draining its own DFB the moment entries arrive).
//
// Using a sequential Tensix kernel (dfb_t6_seq_producer.cpp) avoids any dependency
// on Neo hartid values: the single Neo thread signals DFB_0 fully, waits for acks,
// then moves to DFB_1.  DM consumer threads start simultaneously but block on
// wait_front until their DFB has entries.
// =====================================================================================
void run_concurrent_tensix_dm_dfbs_program(
    const std::shared_ptr<distributed::MeshDevice>& mesh_device,
    uint32_t num_dfbs,
    experimental::dfb::DataflowBufferConfig& dfb_config) {
    TT_FATAL(
        mesh_device->get_devices()[0]->arch() == ARCH::QUASAR,
        "Concurrent Tensix→DM DFB tests require Quasar");
    TT_FATAL(
        dfb_config.num_producers == 1 && dfb_config.num_consumers == 1,
        "run_concurrent_tensix_dm_dfbs_program requires 1Sx1S per DFB");
    TT_FATAL(num_dfbs <= 6, "num_dfbs must fit in the 6 available Quasar DM threads (DM0 reserved for dispatch)");

    const CoreCoord core(0, 0);
    const CoreRangeSet core_range_set(CoreRange(core, core));
    IDevice* device = mesh_device->get_devices()[0];

    const uint32_t entry_size     = dfb_config.entry_size;
    const uint32_t entries_per_dfb = dfb_config.num_entries;
    const uint32_t wpe = entry_size / sizeof(uint32_t);

    // Separate DRAM out_tensor per DFB (DM consumers write here).
    std::vector<MeshTensor> out_tensors;
    out_tensors.reserve(num_dfbs);
    const auto out_tensor_spec = make_flat_dram_tensor_spec(entry_size, entries_per_dfb);
    for (uint32_t i = 0; i < num_dfbs; ++i) {
        out_tensors.push_back(MeshTensor::allocate_on_device(*mesh_device, out_tensor_spec, TensorTopology{}));
    }

    // Generate input – each DFB gets its own slice (entries_per_dfb entries).
    const uint32_t total_words = num_dfbs * entries_per_dfb * wpe;
    auto input = tt::test_utils::generate_uniform_random_vector<uint32_t>(0, 100, total_words);

    constexpr const char* PRODUCER = "producer";

    // Tensix sequential producer: 1 Neo thread, loops through num_dfbs DFBs.
    // The kernel uses #if NUM_DFBS >= I to gate per-DFB unrolled blocks (each
    // referencing dfb::dfb_<i>) so we MUST also pass NUM_DFBS as a compiler define
    // matching the number of DFB bindings declared here.
    experimental::metal2_host_api::KernelSpec producer_spec{
        .unique_id = PRODUCER,
        .source =
            experimental::metal2_host_api::KernelSpec::SourceFilePath{
                "tests/tt_metal/tt_metal/test_kernels/compute/dfb_t6_seq_producer.cpp"},
        .num_threads = 1,
        .compiler_options = {.defines = {{"NUM_DFBS", std::to_string(num_dfbs)}}},
        .compile_time_arg_bindings = {{"num_entries_per_producer", entries_per_dfb}},
        .config_spec = experimental::metal2_host_api::ComputeConfiguration{},
    };
    for (uint32_t i = 0; i < num_dfbs; ++i) {
        producer_spec.dfb_bindings.push_back(experimental::metal2_host_api::KernelSpec::DFBBinding{
            .dfb_spec_name = "dfb_" + std::to_string(i),
            .local_accessor_name = "dfb_" + std::to_string(i),
            .endpoint_type = experimental::metal2_host_api::KernelSpec::DFBEndpointType::PRODUCER,
            .access_pattern = experimental::metal2_host_api::DFBAccessPattern::STRIDED,
        });
    }

    // DM concurrent consumers: one 1-thread kernel instance per DFB, each binding
    // its own per-DFB OUT_TENSOR_<i>.
    std::vector<experimental::metal2_host_api::KernelSpec> kernel_specs;
    std::vector<std::string> kernel_names;
    std::vector<experimental::metal2_host_api::DataflowBufferSpec> dfb_specs;
    std::vector<experimental::metal2_host_api::TensorParameter> tensor_parameters;
    kernel_specs.reserve(1 + num_dfbs);
    kernel_names.reserve(1 + num_dfbs);
    dfb_specs.reserve(num_dfbs);
    tensor_parameters.reserve(num_dfbs);

    kernel_specs.push_back(producer_spec);
    kernel_names.push_back(PRODUCER);

    const uint32_t num_entries_per_consumer = entries_per_dfb;  // 1Sx1S: sole consumer gets all
    for (uint32_t i = 0; i < num_dfbs; ++i) {
        const std::string dfb_name = "dfb_" + std::to_string(i);
        const std::string consumer_name = "consumer_" + std::to_string(i);
        const std::string out_tensor_name = "out_tensor_" + std::to_string(i);

        dfb_specs.push_back(experimental::metal2_host_api::DataflowBufferSpec{
            .unique_id = dfb_name,
            .entry_size = entry_size,
            .num_entries = entries_per_dfb,
            .data_format_metadata = dfb_config.data_format,
            .disable_implicit_sync = !dfb_config.enable_implicit_sync,
        });

        tensor_parameters.push_back(experimental::metal2_host_api::TensorParameter{
            .unique_id = out_tensor_name,
            .spec = out_tensors[i].tensor_spec(),
        });

        kernel_specs.push_back(experimental::metal2_host_api::KernelSpec{
            .unique_id = consumer_name,
            .source =
                experimental::metal2_host_api::KernelSpec::SourceFilePath{
                    "tests/tt_metal/tt_metal/test_kernels/dataflow/dfb_multi_consumer_sep.cpp"},
            .num_threads = 1,
            .dfb_bindings = {{
                .dfb_spec_name = dfb_name,
                .local_accessor_name = "in",
                .endpoint_type = experimental::metal2_host_api::KernelSpec::DFBEndpointType::CONSUMER,
                .access_pattern = experimental::metal2_host_api::DFBAccessPattern::STRIDED,
            }},
            .tensor_bindings = {{
                .tensor_parameter_name = out_tensor_name,
                .accessor_name = "dst_tensor",
            }},
            .compile_time_arg_bindings =
                {
                    {"num_entries_per_consumer", num_entries_per_consumer},
                    {"implicit_sync", static_cast<uint32_t>(dfb_config.enable_implicit_sync ? 1u : 0u)},
                },
            .config_spec =
                experimental::metal2_host_api::DataMovementConfiguration{
                    .gen2_data_movement_config =
                        experimental::metal2_host_api::DataMovementConfiguration::Gen2DataMovementConfig{}},
        });
        kernel_names.push_back(consumer_name);
    }

    experimental::metal2_host_api::WorkUnitSpec wu{
        .unique_id = "main",
        .kernels = kernel_names,
        .target_nodes = core_range_set,
    };

    experimental::metal2_host_api::ProgramSpec spec{
        .program_id = "concurrent_tensix_dm_dfbs",
        .kernels = kernel_specs,
        .dataflow_buffers = dfb_specs,
        .tensor_parameters = tensor_parameters,
        .work_units = {wu},
    };

    Program program = experimental::metal2_host_api::MakeProgramFromSpec(*mesh_device, spec);

    experimental::metal2_host_api::ProgramRunParams run_params;
    run_params.kernel_run_params.reserve(kernel_names.size());
    for (const auto& name : kernel_names) {
        run_params.kernel_run_params.push_back(
            experimental::metal2_host_api::ProgramRunParams::KernelRunParams{.kernel_spec_name = name});
    }
    run_params.tensor_args.reserve(num_dfbs);
    for (uint32_t i = 0; i < num_dfbs; ++i) {
        run_params.tensor_args.push_back(
            {.tensor_parameter_name = "out_tensor_" + std::to_string(i), .tensor = std::cref(out_tensors[i])});
    }
    experimental::metal2_host_api::SetProgramRunParameters(program, run_params);

    // Pre-fill each DFB's L1 ring with its input slice.
    // DFBs are allocated consecutively starting at the base L1 allocator address;
    // each DFB occupies entries_per_dfb * entry_size bytes.
    const uint32_t l1_base = static_cast<uint32_t>(device->allocator()->get_base_allocator_addr(HalMemType::L1));
    const uint32_t ring_stride = entries_per_dfb * entry_size;  // bytes; same for all identical configs
    const uint32_t ring_words  = ring_stride / sizeof(uint32_t);

    for (uint32_t i = 0; i < num_dfbs; ++i) {
        const uint32_t dfb_l1_addr = l1_base + i * ring_stride;
        std::vector<uint32_t> slice(ring_words, 0u);
        for (uint32_t e = 0; e < entries_per_dfb; ++e) {
            const uint32_t src = (i * entries_per_dfb + e) * wpe;
            std::copy(input.begin() + src, input.begin() + src + wpe, slice.begin() + e * wpe);
        }
        detail::WriteToDeviceL1(device, core, dfb_l1_addr, slice);
    }

    // Launch program (slow dispatch; DFBs not yet supported in MeshWorkload path).
    detail::LaunchProgram(device, program, true /*wait_until_cores_done*/);

    // Verify: each out_tensor must match its DFB's L1 pre-fill slice.
    for (uint32_t i = 0; i < num_dfbs; ++i) {
        std::vector<uint32_t> output;
        detail::ReadFromBuffer(*out_tensors[i].mesh_buffer().get_reference_buffer(), output);

        std::vector<uint32_t> expected(entries_per_dfb * wpe);
        for (uint32_t e = 0; e < entries_per_dfb; ++e) {
            const uint32_t src = (i * entries_per_dfb + e) * wpe;
            std::copy(input.begin() + src, input.begin() + src + wpe, expected.begin() + e * wpe);
        }
        EXPECT_EQ(expected, output) << "TensixDM concurrent DFB " << i << " output mismatch";
    }
}

// =====================================================================================
// Gap 7 harness – sequential DM→DM DFBs
//
// N DM producer threads and N DM consumer threads cooperate sequentially through
// num_dfbs DFBs
//
// Producer threads: DM[0..num_producers-1]
// Consumer threads: DM[num_producers..num_producers+num_consumers-1]
// =====================================================================================
void run_sequential_dfbs_program(
    const std::shared_ptr<distributed::MeshDevice>& mesh_device,
    const std::vector<experimental::dfb::DataflowBufferConfig>& configs) {
    TT_FATAL(!configs.empty(), "configs must not be empty");
    TT_FATAL(
        mesh_device->get_devices()[0]->arch() == ARCH::QUASAR,
        "Sequential multi-DFB TC-exhaustion tests require Quasar");

    const uint32_t num_dfbs      = static_cast<uint32_t>(configs.size());
    const uint32_t num_producers = configs[0].num_producers;
    const uint32_t num_consumers = configs[0].num_consumers;
    const uint32_t num_entries   = configs[0].num_entries;
    const uint32_t entry_size    = configs[0].entry_size;

    for (const auto& c : configs) {
        TT_FATAL(c.num_producers == num_producers && c.num_consumers == num_consumers &&
                     c.num_entries == num_entries && c.entry_size == entry_size,
            "All DFB configs must share num_producers/num_consumers/num_entries/entry_size");
    }

    TT_FATAL(
        num_producers + num_consumers <= 6,
        "num_producers + num_consumers must fit in 6 available Quasar DM threads (DM0 reserved for dispatch)");
    // dfb_seq_producer / dfb_seq_consumer kernels unroll up to NUM_DFBS bindings
    // referencing dfb::dfb_<i> / ta::src_<i> / ta::dst_<i> for i in [0, NUM_DFBS).
    TT_FATAL(num_dfbs <= 6, "num_dfbs must be <= 6 (kernel binding-name limit dfb::dfb_0..5)");

    const CoreCoord core(0, 0);
    const CoreRangeSet core_range_set(CoreRange(core, core));
    IDevice* device = mesh_device->get_devices()[0];

    // Separate in_tensor and out_tensor per DFB.
    std::vector<MeshTensor> in_tensors, out_tensors;
    in_tensors.reserve(num_dfbs);
    out_tensors.reserve(num_dfbs);
    const auto tensor_spec = make_flat_dram_tensor_spec(entry_size, num_entries);
    for (uint32_t i = 0; i < num_dfbs; ++i) {
        in_tensors.push_back(MeshTensor::allocate_on_device(*mesh_device, tensor_spec, TensorTopology{}));
        out_tensors.push_back(MeshTensor::allocate_on_device(*mesh_device, tensor_spec, TensorTopology{}));
    }

    // Generate and write input data for each DFB's in_tensor.
    std::vector<std::vector<uint32_t>> inputs(num_dfbs);
    for (uint32_t i = 0; i < num_dfbs; ++i) {
        inputs[i] = tt::test_utils::generate_uniform_random_vector<uint32_t>(
            0, 100, num_entries * entry_size / sizeof(uint32_t));
        detail::WriteToBuffer(*in_tensors[i].mesh_buffer().get_reference_buffer(), inputs[i]);
    }

    // Read-back DRAM verify – mirrors the existing run_single_dfb_program Quasar path
    // safeguard against slow-dispatch write coalescing under the new APIs.
    std::this_thread::sleep_for(std::chrono::milliseconds(500));
    for (uint32_t i = 0; i < num_dfbs; ++i) {
        std::vector<uint32_t> rdback;
        detail::ReadFromBuffer(*in_tensors[i].mesh_buffer().get_reference_buffer(), rdback);
        tt_driver_atomics::mfence();
        EXPECT_EQ(rdback, inputs[i]) << "in_tensor[" << i << "] DRAM write verify failed";
    }

    const uint32_t num_entries_per_producer =
        (num_entries + num_producers - 1) / num_producers;  // ceiling; all configs share this

    constexpr const char* PRODUCER = "producer";
    constexpr const char* CONSUMER = "consumer";

    // Producer kernel: N DFB bindings (dfb::dfb_0..N-1) + N tensor bindings (ta::src_0..N-1).
    // NUM_DFBS compiler define gates the per-DFB unrolled blocks; it must equal the
    // number of dfb_bindings / tensor_bindings registered here.
    experimental::metal2_host_api::KernelSpec producer_spec{
        .unique_id = PRODUCER,
        .source =
            experimental::metal2_host_api::KernelSpec::SourceFilePath{
                "tests/tt_metal/tt_metal/test_kernels/dataflow/dfb_seq_producer.cpp"},
        .num_threads = num_producers,
        .compiler_options = {.defines = {{"NUM_DFBS", std::to_string(num_dfbs)}}},
        .compile_time_arg_bindings =
            {
                {"num_entries_per_producer", num_entries_per_producer},
                {"implicit_sync", static_cast<uint32_t>(configs[0].enable_implicit_sync ? 1u : 0u)},
                {"num_producers", num_producers},
            },
        .config_spec =
            experimental::metal2_host_api::DataMovementConfiguration{
                .gen2_data_movement_config =
                    experimental::metal2_host_api::DataMovementConfiguration::Gen2DataMovementConfig{}},
    };

    // Consumer kernel: same layout, plus per-DFB entries_per_consumer_<i> /
    // is_blocked_<i> CTAs (per-DFB cap is host-known so they collapse to CTAs).
    experimental::metal2_host_api::KernelSpec consumer_spec{
        .unique_id = CONSUMER,
        .source =
            experimental::metal2_host_api::KernelSpec::SourceFilePath{
                "tests/tt_metal/tt_metal/test_kernels/dataflow/dfb_seq_consumer.cpp"},
        .num_threads = num_consumers,
        .compiler_options = {.defines = {{"NUM_DFBS", std::to_string(num_dfbs)}}},
        .compile_time_arg_bindings =
            {
                {"implicit_sync", static_cast<uint32_t>(configs[0].enable_implicit_sync ? 1u : 0u)},
                {"num_consumers", num_consumers},
            },
        .config_spec =
            experimental::metal2_host_api::DataMovementConfiguration{
                .gen2_data_movement_config =
                    experimental::metal2_host_api::DataMovementConfiguration::Gen2DataMovementConfig{}},
    };

    // Build per-DFB DataflowBufferSpec, TensorParameter, and bindings (dfb_<i> /
    // src_<i> / dst_<i> / in_tensor_<i> / out_tensor_<i>).
    std::vector<experimental::metal2_host_api::DataflowBufferSpec> dfb_specs;
    std::vector<experimental::metal2_host_api::TensorParameter> tensor_parameters;
    dfb_specs.reserve(num_dfbs);
    tensor_parameters.reserve(2 * num_dfbs);

    for (uint32_t i = 0; i < num_dfbs; ++i) {
        const std::string idx = std::to_string(i);
        const std::string dfb_name = "dfb_" + idx;
        const std::string in_tensor_name = "in_tensor_" + idx;
        const std::string out_tensor_name = "out_tensor_" + idx;
        const bool is_all = (configs[i].cap == dfb::AccessPattern::ALL);
        const uint32_t epc = is_all ? num_entries : (num_entries + num_consumers - 1) / num_consumers;

        dfb_specs.push_back(experimental::metal2_host_api::DataflowBufferSpec{
            .unique_id = dfb_name,
            .entry_size = entry_size,
            .num_entries = num_entries,
            .data_format_metadata = configs[i].data_format,
            .disable_implicit_sync = !configs[i].enable_implicit_sync,
        });
        tensor_parameters.push_back(experimental::metal2_host_api::TensorParameter{
            .unique_id = in_tensor_name,
            .spec = in_tensors[i].tensor_spec(),
        });
        tensor_parameters.push_back(experimental::metal2_host_api::TensorParameter{
            .unique_id = out_tensor_name,
            .spec = out_tensors[i].tensor_spec(),
        });

        producer_spec.dfb_bindings.push_back(experimental::metal2_host_api::KernelSpec::DFBBinding{
            .dfb_spec_name = dfb_name,
            .local_accessor_name = "dfb_" + idx,
            .endpoint_type = experimental::metal2_host_api::KernelSpec::DFBEndpointType::PRODUCER,
            .access_pattern = experimental::metal2_host_api::DFBAccessPattern::STRIDED,
        });
        producer_spec.tensor_bindings.push_back(experimental::metal2_host_api::KernelSpec::TensorBinding{
            .tensor_parameter_name = in_tensor_name,
            .accessor_name = "src_" + idx,
        });

        consumer_spec.dfb_bindings.push_back(experimental::metal2_host_api::KernelSpec::DFBBinding{
            .dfb_spec_name = dfb_name,
            .local_accessor_name = "dfb_" + idx,
            .endpoint_type = experimental::metal2_host_api::KernelSpec::DFBEndpointType::CONSUMER,
            .access_pattern = is_all ? experimental::metal2_host_api::DFBAccessPattern::ALL
                                     : experimental::metal2_host_api::DFBAccessPattern::STRIDED,
        });
        consumer_spec.tensor_bindings.push_back(experimental::metal2_host_api::KernelSpec::TensorBinding{
            .tensor_parameter_name = out_tensor_name,
            .accessor_name = "dst_" + idx,
        });
        consumer_spec.compile_time_arg_bindings.push_back({"entries_per_consumer_" + idx, epc});
        consumer_spec.compile_time_arg_bindings.push_back(
            {"is_blocked_" + idx, static_cast<uint32_t>(is_all ? 1u : 0u)});
    }

    experimental::metal2_host_api::WorkUnitSpec wu{
        .unique_id = "main",
        .kernels = {PRODUCER, CONSUMER},
        .target_nodes = core_range_set,
    };

    experimental::metal2_host_api::ProgramSpec spec{
        .program_id = "sequential_dfbs",
        .kernels = {producer_spec, consumer_spec},
        .dataflow_buffers = dfb_specs,
        .tensor_parameters = tensor_parameters,
        .work_units = {wu},
    };

    Program program = experimental::metal2_host_api::MakeProgramFromSpec(*mesh_device, spec);

    experimental::metal2_host_api::ProgramRunParams run_params;
    run_params.kernel_run_params = {
        experimental::metal2_host_api::ProgramRunParams::KernelRunParams{.kernel_spec_name = PRODUCER},
        experimental::metal2_host_api::ProgramRunParams::KernelRunParams{.kernel_spec_name = CONSUMER},
    };
    run_params.tensor_args.reserve(2 * num_dfbs);
    for (uint32_t i = 0; i < num_dfbs; ++i) {
        const std::string idx = std::to_string(i);
        run_params.tensor_args.push_back(
            {.tensor_parameter_name = "in_tensor_" + idx, .tensor = std::cref(in_tensors[i])});
        run_params.tensor_args.push_back(
            {.tensor_parameter_name = "out_tensor_" + idx, .tensor = std::cref(out_tensors[i])});
    }
    experimental::metal2_host_api::SetProgramRunParameters(program, run_params);

    detail::LaunchProgram(device, program, true /*wait_until_cores_done*/);

    // Verify: out_tensor[i] should equal in_tensor[i] (strided and all alike).
    for (uint32_t i = 0; i < num_dfbs; ++i) {
        std::vector<uint32_t> output;
        detail::ReadFromBuffer(*out_tensors[i].mesh_buffer().get_reference_buffer(), output);
        EXPECT_EQ(inputs[i], output) << "Sequential DFB " << i << " output mismatch";
    }
}

void run_in_dfb_out_dfb_program(
    const std::shared_ptr<distributed::MeshDevice>& mesh_device,
    experimental::dfb::DataflowBufferConfig& dm2tensix_config,
    experimental::dfb::DataflowBufferConfig& tensix2dm_config) {
    TT_FATAL(
        dm2tensix_config.num_entries == tensix2dm_config.num_entries,
        "Num entries must be the same for in and out DFBs");
    TT_FATAL(
        dm2tensix_config.entry_size == tensix2dm_config.entry_size, "Entry size must be the same for in and out DFBs");
    TT_FATAL(mesh_device->get_devices()[0]->arch() == ARCH::QUASAR, "run_in_dfb_out_dfb_program is Quasar-only");

    const CoreCoord logical_core(0, 0);
    const CoreRangeSet core_range_set(CoreRange(logical_core, logical_core));

    const uint32_t entry_size = dm2tensix_config.entry_size;
    const uint32_t num_entries = dm2tensix_config.num_entries;
    const bool in_is_all = (dm2tensix_config.cap == dfb::AccessPattern::ALL);
    const bool out_is_all = (tensix2dm_config.cap == dfb::AccessPattern::ALL);

    const uint32_t num_entries_per_producer = num_entries / dm2tensix_config.num_producers;
    const uint32_t num_entries_per_unpacker = num_entries / dm2tensix_config.num_consumers;
    const uint32_t num_entries_per_packer = num_entries / tensix2dm_config.num_producers;
    TT_FATAL(
        num_entries_per_unpacker == num_entries_per_packer, "Num entries per unpacker and packer must be the same");
    const uint32_t num_entries_per_consumer = out_is_all ? num_entries : num_entries / tensix2dm_config.num_consumers;

    auto in_tensor = MeshTensor::allocate_on_device(
        *mesh_device, make_flat_dram_tensor_spec(entry_size, num_entries), TensorTopology{});
    auto out_tensor = MeshTensor::allocate_on_device(
        *mesh_device, make_flat_dram_tensor_spec(entry_size, num_entries), TensorTopology{});

    log_info(
        tt::LogTest,
        "In Tensor: [address: {} B, size: {} B]",
        in_tensor.mesh_buffer().get_reference_buffer()->address(),
        in_tensor.mesh_buffer().get_reference_buffer()->size());
    log_info(
        tt::LogTest,
        "Out Tensor: [address: {} B, size: {} B]",
        out_tensor.mesh_buffer().get_reference_buffer()->address(),
        out_tensor.mesh_buffer().get_reference_buffer()->size());

    constexpr const char* IN_DFB = "in_dfb";
    constexpr const char* OUT_DFB = "out_dfb";
    constexpr const char* PRODUCER = "producer";
    constexpr const char* COMPUTE = "compute";
    constexpr const char* CONSUMER = "consumer";
    constexpr const char* IN_TENSOR = "in_tensor";
    constexpr const char* OUT_TENSOR = "out_tensor";

    experimental::metal2_host_api::DataflowBufferSpec in_dfb_spec{
        .unique_id = IN_DFB,
        .entry_size = entry_size,
        .num_entries = num_entries,
        .data_format_metadata = dm2tensix_config.data_format,
        .disable_implicit_sync = !dm2tensix_config.enable_implicit_sync,
    };
    experimental::metal2_host_api::DataflowBufferSpec out_dfb_spec{
        .unique_id = OUT_DFB,
        .entry_size = entry_size,
        .num_entries = num_entries,
        .data_format_metadata = tensix2dm_config.data_format,
        .disable_implicit_sync = !tensix2dm_config.enable_implicit_sync,
    };

    experimental::metal2_host_api::KernelSpec producer_spec{
        .unique_id = PRODUCER,
        .source =
            experimental::metal2_host_api::KernelSpec::SourceFilePath{
                "tests/tt_metal/tt_metal/test_kernels/dataflow/dfb_producer.cpp"},
        .num_threads = static_cast<uint8_t>(dm2tensix_config.num_producers),
        .dfb_bindings = {{
            .dfb_spec_name = IN_DFB,
            .local_accessor_name = "out",
            .endpoint_type = experimental::metal2_host_api::KernelSpec::DFBEndpointType::PRODUCER,
            .access_pattern = experimental::metal2_host_api::DFBAccessPattern::STRIDED,
        }},
        .tensor_bindings = {{
            .tensor_parameter_name = IN_TENSOR,
            .accessor_name = "src_tensor",
        }},
        .compile_time_arg_bindings =
            {
                {"num_entries_per_producer", num_entries_per_producer},
                {"implicit_sync", static_cast<uint32_t>(dm2tensix_config.enable_implicit_sync ? 1u : 0u)},
                {"num_producers", dm2tensix_config.num_producers},
            },
        .runtime_arguments_schema = {.named_runtime_args = {"chunk_offset", "entries_per_core"}},
        .config_spec =
            experimental::metal2_host_api::DataMovementConfiguration{
                .gen2_data_movement_config =
                    experimental::metal2_host_api::DataMovementConfiguration::Gen2DataMovementConfig{}},
    };

    experimental::metal2_host_api::KernelSpec compute_spec{
        .unique_id = COMPUTE,
        .source =
            experimental::metal2_host_api::KernelSpec::SourceFilePath{
                "tests/tt_metal/tt_metal/test_kernels/compute/dfb_t6.cpp"},
        .num_threads = 1,
        .dfb_bindings =
            {
                {
                    .dfb_spec_name = IN_DFB,
                    .local_accessor_name = "in",
                    .endpoint_type = experimental::metal2_host_api::KernelSpec::DFBEndpointType::CONSUMER,
                    .access_pattern = in_is_all ? experimental::metal2_host_api::DFBAccessPattern::ALL
                                                : experimental::metal2_host_api::DFBAccessPattern::STRIDED,
                },
                {
                    .dfb_spec_name = OUT_DFB,
                    .local_accessor_name = "out",
                    .endpoint_type = experimental::metal2_host_api::KernelSpec::DFBEndpointType::PRODUCER,
                    .access_pattern = experimental::metal2_host_api::DFBAccessPattern::STRIDED,
                },
            },
        .compile_time_arg_bindings = {{"num_entries", num_entries_per_unpacker}},
        .config_spec = experimental::metal2_host_api::ComputeConfiguration{},
    };

    experimental::metal2_host_api::KernelSpec consumer_spec{
        .unique_id = CONSUMER,
        .source =
            experimental::metal2_host_api::KernelSpec::SourceFilePath{
                "tests/tt_metal/tt_metal/test_kernels/dataflow/dfb_consumer.cpp"},
        .num_threads = static_cast<uint8_t>(tensix2dm_config.num_consumers),
        .dfb_bindings = {{
            .dfb_spec_name = OUT_DFB,
            .local_accessor_name = "in",
            .endpoint_type = experimental::metal2_host_api::KernelSpec::DFBEndpointType::CONSUMER,
            .access_pattern = out_is_all ? experimental::metal2_host_api::DFBAccessPattern::ALL
                                         : experimental::metal2_host_api::DFBAccessPattern::STRIDED,
        }},
        .tensor_bindings = {{
            .tensor_parameter_name = OUT_TENSOR,
            .accessor_name = "dst_tensor",
        }},
        .compile_time_arg_bindings =
            {
                {"num_entries_per_consumer", num_entries_per_consumer},
                {"blocked_consumer", static_cast<uint32_t>(out_is_all ? 1u : 0u)},
                {"implicit_sync", static_cast<uint32_t>(tensix2dm_config.enable_implicit_sync ? 1u : 0u)},
                {"num_consumers", tensix2dm_config.num_consumers},
            },
        .runtime_arguments_schema = {.named_runtime_args = {"chunk_offset", "entries_per_core"}},
        .config_spec =
            experimental::metal2_host_api::DataMovementConfiguration{
                .gen2_data_movement_config =
                    experimental::metal2_host_api::DataMovementConfiguration::Gen2DataMovementConfig{}},
    };

    experimental::metal2_host_api::WorkUnitSpec wu{
        .unique_id = "main",
        .kernels = {PRODUCER, COMPUTE, CONSUMER},
        .target_nodes = core_range_set,
    };

    experimental::metal2_host_api::ProgramSpec spec{
        .program_id = "in_dfb_out_dfb",
        .kernels = {producer_spec, compute_spec, consumer_spec},
        .dataflow_buffers = {in_dfb_spec, out_dfb_spec},
        .tensor_parameters =
            {
                {.unique_id = IN_TENSOR, .spec = in_tensor.tensor_spec()},
                {.unique_id = OUT_TENSOR, .spec = out_tensor.tensor_spec()},
            },
        .work_units = {wu},
    };

    Program program = experimental::metal2_host_api::MakeProgramFromSpec(*mesh_device, spec);

    using NodeNamedRTAs = experimental::metal2_host_api::ProgramRunParams::KernelRunParams::NodeNamedRTAs;
    auto build_named_rtas = [&]() {
        return std::vector<NodeNamedRTAs>{NodeNamedRTAs{
            experimental::metal2_host_api::NodeCoord{logical_core.x, logical_core.y},
            {{"chunk_offset", 0u}, {"entries_per_core", num_entries}}}};
    };

    experimental::metal2_host_api::ProgramRunParams::KernelRunParams producer_params{.kernel_spec_name = PRODUCER};
    producer_params.named_runtime_args = build_named_rtas();
    experimental::metal2_host_api::ProgramRunParams::KernelRunParams compute_params{.kernel_spec_name = COMPUTE};
    experimental::metal2_host_api::ProgramRunParams::KernelRunParams consumer_params{.kernel_spec_name = CONSUMER};
    consumer_params.named_runtime_args = build_named_rtas();

    experimental::metal2_host_api::ProgramRunParams run_params;
    run_params.kernel_run_params = {producer_params, compute_params, consumer_params};
    run_params.tensor_args = {
        {.tensor_parameter_name = IN_TENSOR, .tensor = std::cref(in_tensor)},
        {.tensor_parameter_name = OUT_TENSOR, .tensor = std::cref(out_tensor)},
    };
    experimental::metal2_host_api::SetProgramRunParameters(program, run_params);

    auto input =
        tt::test_utils::generate_uniform_random_vector<uint32_t>(0, 100, num_entries * entry_size / sizeof(uint32_t));
    execute_program_and_verify(mesh_device, program, in_tensor, out_tensor, input);
}

// =====================================================================================
// Single-DFB test macros
//
// WH/BH supports only 1 DM producer (BRISC) + 1 DM consumer (NCRISC) and no implicit_sync,
// so 1x1 configurations may run there with implicit_sync=false; everything else is Quasar-only.
// =====================================================================================

#define DFB_SKIP_IF_UNSUPPORTED(num_p, num_c)                                                           \
    if (devices_.at(0)->arch() != ARCH::QUASAR && (GetParam() || (num_p) > 1 || (num_c) > 1)) {        \
        GTEST_SKIP();                                                                                   \
    }

// DM -> ALL DM is unsupported with implicit_sync today.
#define DFB_SKIP_DM_DM_ALL_IMPLICIT_SYNC                                                             \
    if (GetParam()) {                                                                                \
        GTEST_SKIP() << "Skipping DM to ALL DM with implicit sync until support is added";          \
    }

#define DFB_NO_EXTRA_SKIP ((void)0)

constexpr uint32_t dfb_default_num_entries(uint32_t num_p, uint32_t num_c) {
    const uint32_t m = (num_p / std::gcd(num_p, num_c)) * num_c;
    return ((16u + m - 1u) / m) * m;
}

// Single-DFB test on a single core with default ring derived from (num_p, num_c)
// and entry_size=1024.
//   prefix       e.g. DM, DMTensix, TensixDM
//   suffix       e.g. 3Sx1S, 6Sx2B
//   p_kind/c_kind   DM | TENSIX
//   num_p/num_c     1..6 (DM); 1..4 (TENSIX)
//   pap_kind/cap_kind   STRIDED | ALL
//   extra_skip      DFB_NO_EXTRA_SKIP or DFB_SKIP_DM_DM_ALL_IMPLICIT_SYNC
#define DFB_TEST(prefix, suffix, p_kind, c_kind, num_p, pap_kind, num_c, cap_kind, extra_skip)          \
    TEST_P(DFBImplicitSyncParamFixture, prefix##Test1xDFB##suffix) {                                    \
        DFB_SKIP_IF_UNSUPPORTED((num_p), (num_c));                                                      \
        extra_skip;                                                                                     \
        experimental::dfb::DataflowBufferConfig config{                                                 \
            .entry_size = 1024,                                                                         \
            .num_entries = dfb_default_num_entries((num_p), (num_c)),                                   \
            .num_producers = (num_p),                                                                   \
            .pap = dfb::AccessPattern::pap_kind,                                                        \
            .num_consumers = (num_c),                                                                   \
            .cap = dfb::AccessPattern::cap_kind,                                                        \
            .enable_implicit_sync = GetParam()};                                                        \
        run_single_dfb_program(                                                                         \
            this->devices_.at(0), config, DFBPorCType::p_kind, DFBPorCType::c_kind);                    \
    }

// Variant for DM->DM tests that pass an explicit num_entries_in_buffer (forces wraparound
// when the requested total exceeds the ring size).
#define DFB_TEST_BUF(prefix, suffix, p_kind, c_kind, num_p, pap_kind, num_c, cap_kind, extra_skip, n_buf) \
    TEST_P(DFBImplicitSyncParamFixture, prefix##Test1xDFB##suffix) {                                    \
        DFB_SKIP_IF_UNSUPPORTED((num_p), (num_c));                                                      \
        extra_skip;                                                                                     \
        experimental::dfb::DataflowBufferConfig config{                                                 \
            .entry_size = 1024,                                                                         \
            .num_entries = dfb_default_num_entries((num_p), (num_c)),                                   \
            .num_producers = (num_p), .pap = dfb::AccessPattern::pap_kind,                              \
            .num_consumers = (num_c), .cap = dfb::AccessPattern::cap_kind,                              \
            .enable_implicit_sync = GetParam()};                                                        \
        CoreRangeSet core_range_set(CoreRange(CoreCoord(0, 0), CoreCoord(0, 0)));                       \
        run_single_dfb_program(                                                                         \
            this->devices_.at(0), config, DFBPorCType::p_kind, DFBPorCType::c_kind,                     \
            core_range_set, (n_buf));                                                                   \
    }

// =====================================================================================
// Strided
// =====================================================================================

// 1x1 (DM->DM uses num_entries_in_buffer=18 to exercise wraparound)
DFB_TEST_BUF(DM,       1Sx1S, DM,     DM,     1, STRIDED, 1, STRIDED, DFB_NO_EXTRA_SKIP, 18)
DFB_TEST    (DMTensix, 1Sx1S, DM,     TENSIX, 1, STRIDED, 1, STRIDED, DFB_NO_EXTRA_SKIP)
DFB_TEST    (TensixDM, 1Sx1S, TENSIX, DM,     1, STRIDED, 1, STRIDED, DFB_NO_EXTRA_SKIP)

// TEST_F(MeshDeviceFixture, DMTensixDMTest2xDFB1Sx1S) {
//     if (devices_.at(0)->arch() != ARCH::QUASAR) {
//         GTEST_SKIP() << "Skipping DFB test for WH/BH until DFB is backported";
//     }
//     experimental::dfb::DataflowBufferConfig dm2tensix_config{
//         .entry_size = 1024,
//         .num_entries = 16,
//         .num_producers = 1,
//         .pap = dfb::AccessPattern::STRIDED,
//         .num_consumers = 1,
//         .cap = dfb::AccessPattern::STRIDED,
//         .enable_implicit_sync = false};

//     experimental::dfb::DataflowBufferConfig tensix2dm_config{
//         .entry_size = 1024,
//         .num_entries = 16,
//         .num_producers = 1,
//         .pap = dfb::AccessPattern::STRIDED,
//         .num_consumers = 1,
//             .cap = dfb::AccessPattern::STRIDED,
//         .enable_implicit_sync = false};

//     run_in_dfb_out_dfb_program(this->devices_.at(0), dm2tensix_config, tensix2dm_config);
// }

// TEST_F(MeshDeviceFixture, DMTensixDMTest1xDFB2Sx1S1xDFB1Sx2S) {
//     if (devices_.at(0)->arch() != ARCH::QUASAR) {
//         GTEST_SKIP() << "Skipping DFB test for WH/BH until DFB is backported";
//     }
//     experimental::dfb::DataflowBufferConfig dm2tensix_config{
//         .entry_size = 1024,
//         .num_entries = 16,
//         .num_producers = 2,
//         .pap = dfb::AccessPattern::STRIDED,
//         .num_consumers = 1,
//         .cap = dfb::AccessPattern::STRIDED,
//         .enable_implicit_sync = false};

//     experimental::dfb::DataflowBufferConfig tensix2dm_config{
//         .entry_size = 1024,
//         .num_entries = 16,
//         .num_producers = 1,
//         .pap = dfb::AccessPattern::STRIDED,
//         .num_consumers = 2,
//         .cap = dfb::AccessPattern::STRIDED,
//         .enable_implicit_sync = false};

//     run_in_dfb_out_dfb_program(this->devices_.at(0), dm2tensix_config, tensix2dm_config);
// }

// TEST_F(MeshDeviceFixture, DMTensixDMTest1xDFB4Sx1S1xDFB1Sx4S) {
//     if (devices_.at(0)->arch() != ARCH::QUASAR) {
//         GTEST_SKIP() << "Skipping DFB test for WH/BH until DFB is backported";
//     }
//     experimental::dfb::DataflowBufferConfig dm2tensix_config{
//         .entry_size = 1024,
//         .num_entries = 16,
//         .num_producers = 4,
//         .pap = dfb::AccessPattern::STRIDED,
//         .num_consumers = 1,
//         .cap = dfb::AccessPattern::STRIDED,
//         .enable_implicit_sync = false};

//     experimental::dfb::DataflowBufferConfig tensix2dm_config{
//         .entry_size = 1024,
//         .num_entries = 16,
//         .num_producers = 1,
//         .pap = dfb::AccessPattern::STRIDED,
//         .num_consumers = 4,
//         .cap = dfb::AccessPattern::STRIDED,
//         .enable_implicit_sync = false};

//     run_in_dfb_out_dfb_program(this->devices_.at(0), dm2tensix_config, tensix2dm_config);
// }

DFB_TEST    (DM,       1Sx4S, DM,     DM,     1, STRIDED, 4, STRIDED, DFB_NO_EXTRA_SKIP)
DFB_TEST    (DMTensix, 1Sx4S, DM,     TENSIX, 1, STRIDED, 4, STRIDED, DFB_NO_EXTRA_SKIP)
DFB_TEST    (TensixDM, 1Sx4S, TENSIX, DM,     1, STRIDED, 4, STRIDED, DFB_NO_EXTRA_SKIP)

DFB_TEST    (DM,       4Sx1S, DM,     DM,     4, STRIDED, 1, STRIDED, DFB_NO_EXTRA_SKIP)
DFB_TEST    (DMTensix, 4Sx1S, DM,     TENSIX, 4, STRIDED, 1, STRIDED, DFB_NO_EXTRA_SKIP)
DFB_TEST    (TensixDM, 4Sx1S, TENSIX, DM,     4, STRIDED, 1, STRIDED, DFB_NO_EXTRA_SKIP)

DFB_TEST_BUF(DM,       4Sx4S, DM,     DM,     4, STRIDED, 4, STRIDED, DFB_NO_EXTRA_SKIP, 29)
DFB_TEST    (DMTensix, 4Sx4S, DM,     TENSIX, 4, STRIDED, 4, STRIDED, DFB_NO_EXTRA_SKIP)
DFB_TEST    (TensixDM, 4Sx4S, TENSIX, DM,     4, STRIDED, 4, STRIDED, DFB_NO_EXTRA_SKIP)

DFB_TEST_BUF(DM,       2Sx4S, DM,     DM,     2, STRIDED, 4, STRIDED, DFB_NO_EXTRA_SKIP, 21)
DFB_TEST    (DMTensix, 2Sx4S, DM,     TENSIX, 2, STRIDED, 4, STRIDED, DFB_NO_EXTRA_SKIP)
DFB_TEST    (TensixDM, 2Sx4S, TENSIX, DM,     2, STRIDED, 4, STRIDED, DFB_NO_EXTRA_SKIP)

DFB_TEST    (DM,       4Sx2S, DM,     DM,     4, STRIDED, 2, STRIDED, DFB_NO_EXTRA_SKIP)
DFB_TEST    (DMTensix, 4Sx2S, DM,     TENSIX, 4, STRIDED, 2, STRIDED, DFB_NO_EXTRA_SKIP)
DFB_TEST    (TensixDM, 4Sx2S, TENSIX, DM,     4, STRIDED, 2, STRIDED, DFB_NO_EXTRA_SKIP)

// DM->DM strided: power-of-2 (1Sx2S, 2Sx1S, 2Sx2S)
DFB_TEST    (DM,       1Sx2S, DM,     DM,     1, STRIDED, 2, STRIDED, DFB_NO_EXTRA_SKIP)
DFB_TEST    (DM,       2Sx1S, DM,     DM,     2, STRIDED, 1, STRIDED, DFB_NO_EXTRA_SKIP)
DFB_TEST    (DM,       2Sx2S, DM,     DM,     2, STRIDED, 2, STRIDED, DFB_NO_EXTRA_SKIP)

// DM->DM strided: 3-DM consumer column
DFB_TEST    (DM,       1Sx3S, DM,     DM,     1, STRIDED, 3, STRIDED, DFB_NO_EXTRA_SKIP)
// DFB_TEST    (DM,       2Sx3S, DM,     DM,     2, STRIDED, 3, STRIDED, DFB_NO_EXTRA_SKIP) // needs ALL access pattern

DFB_TEST    (DM,       3Sx1S, DM,     DM,     3, STRIDED, 1, STRIDED, DFB_NO_EXTRA_SKIP)
// DFB_TEST    (DM,       3Sx2S, DM,     DM,     3, STRIDED, 2, STRIDED, DFB_NO_EXTRA_SKIP) // needs ALL access pattern
DFB_TEST    (DM,       3Sx3S, DM,     DM,     3, STRIDED, 3, STRIDED, DFB_NO_EXTRA_SKIP)

DFB_TEST    (DM,       1Sx5S, DM,     DM,     1, STRIDED, 5, STRIDED, DFB_NO_EXTRA_SKIP)
DFB_TEST    (DM,       5Sx1S, DM,     DM,     5, STRIDED, 1, STRIDED, DFB_NO_EXTRA_SKIP)

// DM->Tensix strided (Tensix consumers limited to {1,2,4})
// Power-of-2 gaps
DFB_TEST    (DMTensix, 1Sx2S, DM,     TENSIX, 1, STRIDED, 2, STRIDED, DFB_NO_EXTRA_SKIP)
DFB_TEST    (DMTensix, 2Sx1S, DM,     TENSIX, 2, STRIDED, 1, STRIDED, DFB_NO_EXTRA_SKIP)
// 3-DM producer
DFB_TEST    (DMTensix, 3Sx1S, DM,     TENSIX, 3, STRIDED, 1, STRIDED, DFB_NO_EXTRA_SKIP)
// DFB_TEST    (DMTensix, 3Sx2S, DM,     TENSIX, 3, STRIDED, 2, STRIDED, DFB_NO_EXTRA_SKIP) // needs BLOCKED access pattern
// DFB_TEST    (DMTensix, 3Sx4S, DM,     TENSIX, 3, STRIDED, 4, STRIDED, DFB_NO_EXTRA_SKIP) // needs BLOCKED access pattern
// 6-DM producer
DFB_TEST    (DMTensix, 6Sx1S, DM,     TENSIX, 6, STRIDED, 1, STRIDED, DFB_NO_EXTRA_SKIP)
DFB_TEST    (DMTensix, 6Sx2S, DM,     TENSIX, 6, STRIDED, 2, STRIDED, DFB_NO_EXTRA_SKIP)
// DFB_TEST    (DMTensix, 6Sx4S, DM,     TENSIX, 6, STRIDED, 4, STRIDED, DFB_NO_EXTRA_SKIP) // needs BLOCKED access pattern

// Tensix->DM strided (Tensix producers limited to {1,2,4})
// Power-of-2 gaps
DFB_TEST    (TensixDM, 2Sx1S, TENSIX, DM,     2, STRIDED, 1, STRIDED, DFB_NO_EXTRA_SKIP)
DFB_TEST    (TensixDM, 1Sx2S, TENSIX, DM,     1, STRIDED, 2, STRIDED, DFB_NO_EXTRA_SKIP)
// 3-DM consumer
DFB_TEST    (TensixDM, 1Sx3S, TENSIX, DM,     1, STRIDED, 3, STRIDED, DFB_NO_EXTRA_SKIP)
DFB_TEST    (TensixDM, 2Sx3S, TENSIX, DM,     2, STRIDED, 3, STRIDED, DFB_NO_EXTRA_SKIP)
// DFB_TEST    (TensixDM, 4Sx3S, TENSIX, DM,     4, STRIDED, 3, STRIDED, DFB_NO_EXTRA_SKIP) // needs BLOCKED access pattern
// 6-DM consumer
DFB_TEST    (TensixDM, 1Sx6S, TENSIX, DM,     1, STRIDED, 6, STRIDED, DFB_NO_EXTRA_SKIP)
DFB_TEST    (TensixDM, 2Sx6S, TENSIX, DM,     2, STRIDED, 6, STRIDED, DFB_NO_EXTRA_SKIP)
// DFB_TEST    (TensixDM, 4Sx6S, TENSIX, DM,     4, STRIDED, 6, STRIDED, DFB_NO_EXTRA_SKIP) // needs BLOCKED access pattern

// =====================================================================================
// ALL
// =====================================================================================

DFB_TEST    (DM,       1Sx4A, DM,     DM,     1, STRIDED, 4, ALL, DFB_SKIP_DM_DM_ALL_IMPLICIT_SYNC)
DFB_TEST    (DMTensix, 1Sx4A, DM,     TENSIX, 1, STRIDED, 4, ALL, DFB_NO_EXTRA_SKIP)
DFB_TEST    (TensixDM, 1Sx4A, TENSIX, DM,     1, STRIDED, 4, ALL, DFB_NO_EXTRA_SKIP)

DFB_TEST    (DM,       4Sx1A, DM,     DM,     4, STRIDED, 1, ALL, DFB_SKIP_DM_DM_ALL_IMPLICIT_SYNC)
DFB_TEST    (DMTensix, 4Sx1A, DM,     TENSIX, 4, STRIDED, 1, ALL, DFB_NO_EXTRA_SKIP)
DFB_TEST    (TensixDM, 4Sx1A, TENSIX, DM,     4, STRIDED, 1, ALL, DFB_NO_EXTRA_SKIP)

DFB_TEST    (DM,       4Sx4A, DM,     DM,     4, STRIDED, 4, ALL, DFB_SKIP_DM_DM_ALL_IMPLICIT_SYNC)
DFB_TEST    (DMTensix, 4Sx4A, DM,     TENSIX, 4, STRIDED, 4, ALL, DFB_NO_EXTRA_SKIP)
DFB_TEST    (TensixDM, 4Sx4A, TENSIX, DM,     4, STRIDED, 4, ALL, DFB_NO_EXTRA_SKIP)

DFB_TEST    (DM,       4Sx2A, DM,     DM,     4, STRIDED, 2, ALL, DFB_SKIP_DM_DM_ALL_IMPLICIT_SYNC)
DFB_TEST    (DMTensix, 4Sx2A, DM,     TENSIX, 4, STRIDED, 2, ALL, DFB_NO_EXTRA_SKIP)
DFB_TEST    (TensixDM, 4Sx2A, TENSIX, DM,     4, STRIDED, 2, ALL, DFB_NO_EXTRA_SKIP)

DFB_TEST    (DM,       2Sx4A, DM,     DM,     2, STRIDED, 4, ALL, DFB_SKIP_DM_DM_ALL_IMPLICIT_SYNC)
DFB_TEST    (DMTensix, 2Sx4A, DM,     TENSIX, 2, STRIDED, 4, ALL, DFB_NO_EXTRA_SKIP)
DFB_TEST    (TensixDM, 2Sx4A, TENSIX, DM,     2, STRIDED, 4, ALL, DFB_NO_EXTRA_SKIP)

// DM->DM ALL: 3-DM producer
DFB_TEST    (DM,       3Sx1A, DM,     DM,     3, STRIDED, 1, ALL, DFB_SKIP_DM_DM_ALL_IMPLICIT_SYNC)
DFB_TEST    (DM,       3Sx2A, DM,     DM,     3, STRIDED, 2, ALL, DFB_SKIP_DM_DM_ALL_IMPLICIT_SYNC)
DFB_TEST    (DM,       3Sx3A, DM,     DM,     3, STRIDED, 3, ALL, DFB_SKIP_DM_DM_ALL_IMPLICIT_SYNC)

// DM->DM ALL: 3-DM consumer
DFB_TEST    (DM,       1Sx3A, DM,     DM,     1, STRIDED, 3, ALL, DFB_SKIP_DM_DM_ALL_IMPLICIT_SYNC)
DFB_TEST    (DM,       2Sx3A, DM,     DM,     2, STRIDED, 3, ALL, DFB_SKIP_DM_DM_ALL_IMPLICIT_SYNC)

// DM->Tensix ALL (Tensix consumers limited to {1,2,4})
DFB_TEST    (DMTensix, 3Sx1A, DM,     TENSIX, 3, STRIDED, 1, ALL, DFB_NO_EXTRA_SKIP)
DFB_TEST    (DMTensix, 3Sx2A, DM,     TENSIX, 3, STRIDED, 2, ALL, DFB_NO_EXTRA_SKIP)
DFB_TEST    (DMTensix, 3Sx4A, DM,     TENSIX, 3, STRIDED, 4, ALL, DFB_NO_EXTRA_SKIP)
DFB_TEST    (DMTensix, 6Sx1A, DM,     TENSIX, 6, STRIDED, 1, ALL, DFB_NO_EXTRA_SKIP)
DFB_TEST    (DMTensix, 6Sx2A, DM,     TENSIX, 6, STRIDED, 2, ALL, DFB_NO_EXTRA_SKIP)
DFB_TEST    (DMTensix, 6Sx4A, DM,     TENSIX, 6, STRIDED, 4, ALL, DFB_NO_EXTRA_SKIP)

// Tensix->DM ALL (Tensix producers limited to {1,2,4}).
// DFB_TEST    (TensixDM, 1Sx3A, TENSIX, DM,     1, STRIDED, 3, ALL, DFB_NO_EXTRA_SKIP) // revisit the 3A Tensix consumer
// DFB_TEST    (TensixDM, 2Sx3A, TENSIX, DM,     2, STRIDED, 3, ALL, DFB_NO_EXTRA_SKIP)
// DFB_TEST    (TensixDM, 4Sx3A, TENSIX, DM,     4, STRIDED, 3, ALL, DFB_NO_EXTRA_SKIP)
// DFB_TEST    (TensixDM, 1Sx6A, TENSIX, DM,     1, STRIDED, 6, ALL, DFB_NO_EXTRA_SKIP) // revisit more than 4 ALL consumers
// DFB_TEST    (TensixDM, 2Sx6A, TENSIX, DM,     2, STRIDED, 6, ALL, DFB_NO_EXTRA_SKIP)
// DFB_TEST    (TensixDM, 4Sx6A, TENSIX, DM,     4, STRIDED, 6, ALL, DFB_NO_EXTRA_SKIP)

// 1 strided DM producer, 1 strided DM consumer, num_entries=4.
// Ring wraps 16x; baseline to confirm wraparound logic with minimum participants.
TEST_P(DFBImplicitSyncParamFixture, DMTest1xDFB_RingPressure_1Sx1S) {
    if (devices_.at(0)->arch() != ARCH::QUASAR) {
        GTEST_SKIP() << "Skipping DFB ring-pressure test for WH/BH until DFB is backported";
    }
    DFB_SKIP_IF_UNSUPPORTED(1, 1);
    experimental::dfb::DataflowBufferConfig config{
        .entry_size = 1024,
        .num_entries = 4,
        .num_producers = 1,
        .pap = dfb::AccessPattern::STRIDED,
        .num_consumers = 1,
        .cap = dfb::AccessPattern::STRIDED,
        .enable_implicit_sync = GetParam()};
    run_single_dfb_program(
        this->devices_.at(0), config, DFBPorCType::DM, DFBPorCType::DM,
        CoreRangeSet(CoreRange(CoreCoord(0, 0), CoreCoord(0, 0))), /*num_entries_in_buffer=*/64);
}

// 3 strided DM producers, 3 strided DM consumers, num_entries=3 -> capacity=1.
// Each producer stalls after every push; ring wraps 63x per producer.
TEST_P(DFBImplicitSyncParamFixture, DMTest1xDFB_RingPressure_3Sx3S) {
    if (devices_.at(0)->arch() != ARCH::QUASAR) {
        GTEST_SKIP() << "Skipping DFB ring-pressure test for WH/BH until DFB is backported";
    }
    DFB_SKIP_IF_UNSUPPORTED(3, 3);
    experimental::dfb::DataflowBufferConfig config{
        .entry_size = 1024,
        .num_entries = 3,
        .num_producers = 3,
        .pap = dfb::AccessPattern::STRIDED,
        .num_consumers = 3,
        .cap = dfb::AccessPattern::STRIDED,
        .enable_implicit_sync = GetParam()};
    run_single_dfb_program(
        this->devices_.at(0),
        config,
        DFBPorCType::DM,
        DFBPorCType::DM,
        CoreRangeSet(CoreRange(CoreCoord(0, 0), CoreCoord(0, 0))),
        /*num_entries_in_buffer=*/63);
}

// 4 DM producers (STRIDED), 4 Tensix consumers (ALL), num_entries=4 -> capacity=1.
// Every push stalls until all 4 ALL Tensix consumers ack; ring wraps 64x per producer.
// Exercises remapper fan-out on the DM->Tensix path under maximum ring pressure.
TEST_P(DFBImplicitSyncParamFixture, DMTensixTest1xDFB_RingPressure_4Sx4A) {
    if (devices_.at(0)->arch() != ARCH::QUASAR) {
        GTEST_SKIP() << "Skipping DFB ring-pressure test for WH/BH until DFB is backported";
    }
    DFB_SKIP_IF_UNSUPPORTED(4, 4);
    experimental::dfb::DataflowBufferConfig config{
        .entry_size = 1024,
        .num_entries = 4,  // tight ring: capacity = num_entries / num_producers = 1
        .num_producers = 4,
        .pap = dfb::AccessPattern::STRIDED,
        .num_consumers = 4,
        .cap = dfb::AccessPattern::ALL,
        .enable_implicit_sync = GetParam()};
    run_single_dfb_program(
        this->devices_.at(0), config, DFBPorCType::DM, DFBPorCType::TENSIX,
        CoreRangeSet(CoreRange(CoreCoord(0, 0), CoreCoord(0, 0))), /*num_entries_in_buffer=*/64);
}

// 2 Tensix producers (STRIDED), 4 DM consumers (STRIDED), num_entries=4 -> capacity=1.
// Ring wraps 64x per producer; exercises the Tensix->DM path with asymmetric P:C ratio
// under tight ring pressure (num_consumers > num_producers).
TEST_P(DFBImplicitSyncParamFixture, TensixDMTest1xDFB_RingPressure_2Sx4S) {
    if (devices_.at(0)->arch() != ARCH::QUASAR) {
        GTEST_SKIP() << "Skipping DFB ring-pressure test for WH/BH until DFB is backported";
    }
    DFB_SKIP_IF_UNSUPPORTED(2, 4);
    experimental::dfb::DataflowBufferConfig config{
        .entry_size = 1024,
        .num_entries = 4,  // tight ring: capacity = num_entries / max(num_p, num_c) = 1
        .num_producers = 2,
        .pap = dfb::AccessPattern::STRIDED,
        .num_consumers = 4,
        .cap = dfb::AccessPattern::STRIDED,
        .enable_implicit_sync = GetParam()};
    run_single_dfb_program(
        this->devices_.at(0), config, DFBPorCType::TENSIX, DFBPorCType::DM,
        CoreRangeSet(CoreRange(CoreCoord(0, 0), CoreCoord(0, 0))), /*num_entries_in_buffer=*/64);
}

TEST_P(DFBImplicitSyncParamFixture, MultiCoreDMTest2Core_1Sx1S) {
    if (devices_.at(0)->arch() != ARCH::QUASAR) {
        GTEST_SKIP() << "Skipping DFB test for WH/BH until DFB is backported";
    }
    experimental::dfb::DataflowBufferConfig config{
        .entry_size = 1024,
        .num_entries = 16,
        .num_producers = 1,
        .pap = dfb::AccessPattern::STRIDED,
        .num_consumers = 1,
        .cap = dfb::AccessPattern::STRIDED,
        .enable_implicit_sync = GetParam()};

    CoreRangeSet core_range_set(CoreRange(CoreCoord(0, 0), CoreCoord(1, 0)));
    run_single_dfb_program(this->devices_.at(0), config, DFBPorCType::DM, DFBPorCType::DM, core_range_set);
}

TEST_P(DFBImplicitSyncParamFixture, MultiCoreDMTest2Core_2Sx2S) {
    if (devices_.at(0)->arch() != ARCH::QUASAR) {
        GTEST_SKIP() << "Skipping DFB test for WH/BH until DFB is backported";
    }
    experimental::dfb::DataflowBufferConfig config{
        .entry_size = 1024,
        .num_entries = 16,
        .num_producers = 2,
        .pap = dfb::AccessPattern::STRIDED,
        .num_consumers = 2,
        .cap = dfb::AccessPattern::STRIDED,
        .enable_implicit_sync = GetParam()};

    CoreRangeSet core_range_set(CoreRange(CoreCoord(0, 0), CoreCoord(1, 0)));
    run_single_dfb_program(this->devices_.at(0), config, DFBPorCType::DM, DFBPorCType::DM, core_range_set);
}

TEST_P(DFBImplicitSyncParamFixture, MultiCoreDMTest2Core_1Sx4A) {
    if (devices_.at(0)->arch() != ARCH::QUASAR) {
        GTEST_SKIP() << "Skipping DFB test for WH/BH until DFB is backported";
    }
    experimental::dfb::DataflowBufferConfig config{
        .entry_size = 1024,
        .num_entries = 16,
        .num_producers = 1,
        .pap = dfb::AccessPattern::STRIDED,
        .num_consumers = 4,
        .cap = dfb::AccessPattern::ALL,
        .enable_implicit_sync = GetParam()};

    CoreRangeSet core_range_set(CoreRange(CoreCoord(0, 0), CoreCoord(1, 0)));
    run_single_dfb_program(this->devices_.at(0), config, DFBPorCType::DM, DFBPorCType::DM, core_range_set);
}

// =====================================================================================
// Concurrent DFBs on Same Core (TC Allocator Stress)
//
//
// TensixDMTest4xDFB_1Sx1S
//   4 concurrent 1Sx1S Tensix→DM DFBs.  Single Neo thread fills DFBs sequentially;
//   4 DM consumer threads drain each DFB as soon as entries arrive.
//   Harness: run_concurrent_tensix_dm_dfbs_program
//            + dfb_t6_seq_producer.cpp + dfb_multi_consumer_sep.cpp
//
// DMTest4xDFB_3Sx3S
//   4× 3Sx3S DFBs = 12 TCs total: stresses TC allocator across 6 DM threads.
//   DM threads 0..2 cooperate on each DFB in sequence as producers; DM threads 3..5 consume.
//   All 12 TCs allocated simultaneously in one Program.
//   Harness: run_sequential_dfbs_program + dfb_seq_producer.cpp + dfb_seq_consumer.cpp
//
// DMTest4xDFB_Mixed
//   2× 3Sx3S (strided) + 2× 3Sx3ALL (ALL) on the same core.
//   Mixed TC allocation (strided TCs + remapper-backed ALL TCs) across 6 DM threads.
//   Harness: run_sequential_dfbs_program + dfb_seq_producer.cpp + dfb_seq_consumer.cpp
// =====================================================================================

TEST_P(DFBImplicitSyncParamFixture, DMTest3xDFB_1Sx1S) {
    if (devices_.at(0)->arch() != ARCH::QUASAR) {
        GTEST_SKIP() << "Skipping: concurrent DFB test requires Quasar multi-threaded DM";
    }
    experimental::dfb::DataflowBufferConfig config{
        .entry_size    = 1024,
        .num_entries   = 16,
        .num_producers = 1,
        .pap           = dfb::AccessPattern::STRIDED,
        .num_consumers = 1,
        .cap           = dfb::AccessPattern::STRIDED,
        .enable_implicit_sync = GetParam()};
    run_concurrent_dfbs_program(this->devices_.at(0), /*num_dfbs=*/3, config);
}

TEST_P(DFBImplicitSyncParamFixture, TensixDMTest4xDFB_1Sx1S) {
    // 4 concurrent 1Sx1S DFBs: single Neo thread produces to all 4 sequentially while
    // 4 DM consumer threads drain their DFBs as soon as entries become available.
    // Uses dfb_t6_seq_producer.cpp + dfb_multi_consumer_sep.cpp.
    if (devices_.at(0)->arch() != ARCH::QUASAR) {
        GTEST_SKIP() << "Skipping: Tensix concurrent DFB test requires Quasar";
    }
    experimental::dfb::DataflowBufferConfig config{
        .entry_size    = 1024,
        .num_entries   = 16,
        .num_producers = 1,
        .pap           = dfb::AccessPattern::STRIDED,
        .num_consumers = 1,
        .cap           = dfb::AccessPattern::STRIDED,
        .enable_implicit_sync = GetParam()};
    run_concurrent_tensix_dm_dfbs_program(this->devices_.at(0), /*num_dfbs=*/4, config);
}

TEST_P(DFBImplicitSyncParamFixture, DMTest4xDFB_3Sx3S) {
    // 4 DFBs × 3Sx3S = 12 TCs: stresses TC allocator across 6 DM threads.
    // DM threads 0..2 cooperatively produce all 4 DFBs in sequence; DM threads 3..5
    // cooperatively consume all 4 DFBs in sequence.
    // Uses dfb_seq_producer.cpp + dfb_seq_consumer.cpp.
    if (devices_.at(0)->arch() != ARCH::QUASAR) {
        GTEST_SKIP() << "Skipping: sequential multi-DFB TC exhaustion test requires Quasar";
    }
    experimental::dfb::DataflowBufferConfig config{
        .entry_size    = 1024,
        .num_entries   = 16,
        .num_producers = 3,
        .pap           = dfb::AccessPattern::STRIDED,
        .num_consumers = 3,
        .cap           = dfb::AccessPattern::STRIDED,
        .enable_implicit_sync = GetParam()};
    run_sequential_dfbs_program(this->devices_.at(0), {config, config, config, config});
}

TEST_P(DFBImplicitSyncParamFixture, DMTest4xDFB_Mixed) {
    // 2× 3Sx3S (strided consumers) + 2× 3Sx3B (ALL consumers) on the same core.
    // Mixed TC allocation exercises both plain strided TCs and remapper-backed ALL TCs
    // simultaneously across 6 DM threads (DM threads 0..2 produce, 3..5 consume).
    // Uses dfb_seq_producer.cpp + dfb_seq_consumer.cpp with per-DFB is_all flags.
    if (devices_.at(0)->arch() != ARCH::QUASAR) {
        GTEST_SKIP() << "Skipping: mixed multi-DFB TC exhaustion test requires Quasar";
    }
    experimental::dfb::DataflowBufferConfig strided_cfg{
        .entry_size    = 1024,
        .num_entries   = 16,
        .num_producers = 3,
        .pap           = dfb::AccessPattern::STRIDED,
        .num_consumers = 3,
        .cap           = dfb::AccessPattern::STRIDED,
        .enable_implicit_sync = GetParam()};
    experimental::dfb::DataflowBufferConfig all_cfg{
        .entry_size    = 1024,
        .num_entries   = 16,
        .num_producers = 3,
        .pap           = dfb::AccessPattern::STRIDED,
        .num_consumers = 3,
        .cap           = dfb::AccessPattern::ALL,
        .enable_implicit_sync = GetParam()};
    run_sequential_dfbs_program(
        this->devices_.at(0), {strided_cfg, strided_cfg, all_cfg, all_cfg});
}

INSTANTIATE_TEST_SUITE_P(
    ImplicitSync,
    DFBImplicitSyncParamFixture,
    ::testing::Bool(),
    ImplicitSyncParamName);

// Runs an intra-tensix DFB program on one core.
static void run_intra_tensix_dfb_program(
    const std::shared_ptr<distributed::MeshDevice>& mesh_device,
    uint32_t entry_size,
    uint32_t num_entries,
    uint32_t num_threads) {
    IDevice* device = mesh_device->get_devices()[0];

    experimental::dfb::DataflowBufferConfig dfb_config{
        .entry_size = entry_size,
        .num_entries = num_entries,
        .num_producers = num_threads,
        .pap = dfb::AccessPattern::STRIDED,
        .num_consumers = num_threads,
        .cap = dfb::AccessPattern::STRIDED,
        .enable_implicit_sync = false,
        .tensix_scope = experimental::dfb::TensixScope::INTRA};

    CoreCoord logical_core = CoreCoord(0, 0);
    CoreRangeSet core_range_set(CoreRange(logical_core, logical_core));

    const uint32_t words_per_entry = entry_size / sizeof(uint32_t);

    TT_FATAL(
        num_entries % num_threads == 0,
        "num_entries ({}) must be divisible by num_threads ({}) for intra-tensix block partitioning",
        num_entries, num_threads);
    const uint32_t entries_per_neo = num_entries / num_threads;

    constexpr const char* INTRA_DFB = "intra_dfb";
    constexpr const char* COMPUTE = "compute";

    experimental::metal2_host_api::DataflowBufferSpec intra_dfb_spec{
        .unique_id = INTRA_DFB,
        .entry_size = entry_size,
        .num_entries = num_entries,
        .data_format_metadata = dfb_config.data_format,
        .disable_implicit_sync = !dfb_config.enable_implicit_sync,
    };

    // Self-looped: register both PRODUCER and CONSUMER bindings on the same kernel.
    // The kernel only references dfb::out; both bindings resolve to the same DFB.
    experimental::metal2_host_api::KernelSpec compute_spec{
        .unique_id = COMPUTE,
        .source =
            experimental::metal2_host_api::KernelSpec::SourceFilePath{
                "tests/tt_metal/tt_metal/test_kernels/compute/dfb_t6_intra.cpp"},
        .num_threads = static_cast<uint8_t>(num_threads),
        .dfb_bindings =
            {
                {
                    .dfb_spec_name = INTRA_DFB,
                    .local_accessor_name = "out",
                    .endpoint_type = experimental::metal2_host_api::KernelSpec::DFBEndpointType::PRODUCER,
                    .access_pattern = experimental::metal2_host_api::DFBAccessPattern::STRIDED,
                },
                {
                    .dfb_spec_name = INTRA_DFB,
                    .local_accessor_name = "in",
                    .endpoint_type = experimental::metal2_host_api::KernelSpec::DFBEndpointType::CONSUMER,
                    .access_pattern = experimental::metal2_host_api::DFBAccessPattern::STRIDED,
                },
            },
        .compile_time_arg_bindings =
            {
                {"entries_per_neo", entries_per_neo},
                {"words_per_entry", words_per_entry},
            },
        .config_spec = experimental::metal2_host_api::ComputeConfiguration{},
    };

    experimental::metal2_host_api::WorkUnitSpec wu{
        .unique_id = "main",
        .kernels = {COMPUTE},
        .target_nodes = core_range_set,
    };

    experimental::metal2_host_api::ProgramSpec spec{
        .program_id = "intra_tensix_dfb",
        .kernels = {compute_spec},
        .dataflow_buffers = {intra_dfb_spec},
        .work_units = {wu},
    };

    Program program = experimental::metal2_host_api::MakeProgramFromSpec(*mesh_device, spec);

    experimental::metal2_host_api::ProgramRunParams run_params;
    run_params.kernel_run_params = {{.kernel_spec_name = COMPUTE}};
    experimental::metal2_host_api::SetProgramRunParameters(program, run_params);

    const uint32_t total_size = num_entries * entry_size;
    auto input = tt::test_utils::generate_uniform_random_vector<uint32_t>(
        0, 100, total_size / sizeof(uint32_t));

    const uint32_t dfb_l1_addr =
        static_cast<uint32_t>(device->allocator()->get_base_allocator_addr(HalMemType::L1));

    detail::WriteToDeviceL1(device, logical_core, dfb_l1_addr, input);

    detail::LaunchProgram(device, program, true /*wait_until_cores_done*/);

    // Packer increments each word by 1, then unpacker increments it by 1 → +2 per word.
    // This holds for every Neo's ring independently, so the entire L1 region is input + 2.
    std::vector<uint32_t> expected(input.size());
    for (size_t i = 0; i < input.size(); i++) {
        expected[i] = input[i] + 2;
    }

    std::vector<uint32_t> l1_data;
    detail::ReadFromDeviceL1(device, logical_core, dfb_l1_addr, total_size, l1_data);
    EXPECT_EQ(expected, l1_data) << "Intra-tensix DFB L1 mismatch";
}

TEST_F(MeshDeviceFixture, TensixIntraTest1xDFB1Sx1S) {
    if (devices_.at(0)->arch() != ARCH::QUASAR) {
        GTEST_SKIP() << "Skipping intra-tensix DFB test for WH/BH until DFB is backported";
    }
    run_intra_tensix_dfb_program(this->devices_.at(0), /*entry_size=*/1024, /*num_entries=*/16, /*num_threads=*/1);
}

TEST_F(MeshDeviceFixture, TensixIntraTest1xDFB4Sx4S) {
    if (devices_.at(0)->arch() != ARCH::QUASAR) {
        GTEST_SKIP() << "Skipping intra-tensix DFB test for WH/BH until DFB is backported";
    }
    run_intra_tensix_dfb_program(this->devices_.at(0), /*entry_size=*/1024, /*num_entries=*/16, /*num_threads=*/4);
}

TEST_F(MeshDeviceFixture, TensixIntraAndRemapperTest_4Neo_DM1Sx4B) {
    if (devices_.at(0)->arch() != ARCH::QUASAR) {
        GTEST_SKIP() << "Skipping combined intra-tensix + remapper DFB test for WH/BH until DFB is backported";
    }

    IDevice* device = this->devices_.at(0)->get_devices()[0];
    CoreCoord logical_core(0, 0);
    CoreRangeSet core_range_set(CoreRange(logical_core, logical_core));

    constexpr uint32_t entry_size  = 1024;
    constexpr uint32_t num_entries = 16;
    constexpr uint32_t num_neos    = 4;
    const uint32_t words_per_entry = entry_size / sizeof(uint32_t);
    const uint32_t entries_per_neo = num_entries / num_neos;  // = 4

    // dfb(0): DM->Tensix, 1Sx4B with remapper, implicit sync enabled.
    experimental::dfb::DataflowBufferConfig remapper_dfb_config{
        .entry_size           = entry_size,
        .num_entries          = num_entries,
        .num_producers        = 1,
        .pap                  = dfb::AccessPattern::STRIDED,
        .num_consumers        = num_neos,
        .cap                  = dfb::AccessPattern::ALL,
        .enable_implicit_sync = true};

    // dfb(1): intra-tensix, 4 packer->unpacker pairs, hidden TCs.
    experimental::dfb::DataflowBufferConfig intra_dfb_config{
        .entry_size           = entry_size,
        .num_entries          = num_entries,
        .num_producers        = num_neos,
        .pap                  = dfb::AccessPattern::STRIDED,
        .num_consumers        = num_neos,
        .cap                  = dfb::AccessPattern::STRIDED,
        .enable_implicit_sync = false,
        .tensix_scope         = experimental::dfb::TensixScope::INTRA};

    auto in_tensor = MeshTensor::allocate_on_device(
        *this->devices_.at(0), make_flat_dram_tensor_spec(entry_size, num_entries), TensorTopology{});

    constexpr const char* REMAPPER_DFB = "remapper_dfb";
    constexpr const char* INTRA_DFB = "intra_dfb";
    constexpr const char* DM_PRODUCER = "dm_producer";
    constexpr const char* COMPUTE = "compute";
    constexpr const char* IN_TENSOR = "in_tensor";

    experimental::metal2_host_api::DataflowBufferSpec remapper_dfb_spec{
        .unique_id = REMAPPER_DFB,
        .entry_size = entry_size,
        .num_entries = num_entries,
        .data_format_metadata = remapper_dfb_config.data_format,
        .disable_implicit_sync = !remapper_dfb_config.enable_implicit_sync,
    };
    experimental::metal2_host_api::DataflowBufferSpec intra_dfb_spec{
        .unique_id = INTRA_DFB,
        .entry_size = entry_size,
        .num_entries = num_entries,
        .data_format_metadata = intra_dfb_config.data_format,
        .disable_implicit_sync = !intra_dfb_config.enable_implicit_sync,
    };

    experimental::metal2_host_api::KernelSpec dm_producer_spec{
        .unique_id = DM_PRODUCER,
        .source =
            experimental::metal2_host_api::KernelSpec::SourceFilePath{
                "tests/tt_metal/tt_metal/test_kernels/dataflow/dfb_producer.cpp"},
        .num_threads = 1,
        .dfb_bindings = {{
            .dfb_spec_name = REMAPPER_DFB,
            .local_accessor_name = "out",
            .endpoint_type = experimental::metal2_host_api::KernelSpec::DFBEndpointType::PRODUCER,
            .access_pattern = experimental::metal2_host_api::DFBAccessPattern::STRIDED,
        }},
        .tensor_bindings = {{
            .tensor_parameter_name = IN_TENSOR,
            .accessor_name = "src_tensor",
        }},
        .compile_time_arg_bindings =
            {
                {"num_entries_per_producer", num_entries},  // 1 producer owns all entries
                {"implicit_sync", 1u},
                {"num_producers", 1u},
            },
        .runtime_arguments_schema = {.named_runtime_args = {"chunk_offset", "entries_per_core"}},
        .config_spec =
            experimental::metal2_host_api::DataMovementConfiguration{
                .gen2_data_movement_config =
                    experimental::metal2_host_api::DataMovementConfiguration::Gen2DataMovementConfig{}},
    };

    // Combined compute kernel: BLOCKED consumer of remapper DFB ("remapper_in"),
    // self-looped on intra DFB (PRODUCER "intra_out" + CONSUMER "intra_in"; the
    // kernel only references dfb::intra_out).
    experimental::metal2_host_api::KernelSpec compute_spec{
        .unique_id = COMPUTE,
        .source =
            experimental::metal2_host_api::KernelSpec::SourceFilePath{
                "tests/tt_metal/tt_metal/test_kernels/compute/dfb_intra_and_consume_all.cpp"},
        .num_threads = static_cast<uint8_t>(num_neos),
        .dfb_bindings =
            {
                {
                    .dfb_spec_name = REMAPPER_DFB,
                    .local_accessor_name = "remapper_in",
                    .endpoint_type = experimental::metal2_host_api::KernelSpec::DFBEndpointType::CONSUMER,
                    .access_pattern = experimental::metal2_host_api::DFBAccessPattern::ALL,
                },
                {
                    .dfb_spec_name = INTRA_DFB,
                    .local_accessor_name = "intra_out",
                    .endpoint_type = experimental::metal2_host_api::KernelSpec::DFBEndpointType::PRODUCER,
                    .access_pattern = experimental::metal2_host_api::DFBAccessPattern::STRIDED,
                },
                {
                    .dfb_spec_name = INTRA_DFB,
                    .local_accessor_name = "intra_in",
                    .endpoint_type = experimental::metal2_host_api::KernelSpec::DFBEndpointType::CONSUMER,
                    .access_pattern = experimental::metal2_host_api::DFBAccessPattern::STRIDED,
                },
            },
        .compile_time_arg_bindings =
            {
                {"num_entries_consumer", num_entries},
                {"entries_per_neo", entries_per_neo},
                {"words_per_entry", words_per_entry},
            },
        .config_spec = experimental::metal2_host_api::ComputeConfiguration{},
    };

    experimental::metal2_host_api::WorkUnitSpec wu{
        .unique_id = "main",
        .kernels = {DM_PRODUCER, COMPUTE},
        .target_nodes = core_range_set,
    };

    experimental::metal2_host_api::ProgramSpec spec{
        .program_id = "intra_and_remapper",
        .kernels = {dm_producer_spec, compute_spec},
        .dataflow_buffers = {remapper_dfb_spec, intra_dfb_spec},
        .tensor_parameters = {{.unique_id = IN_TENSOR, .spec = in_tensor.tensor_spec()}},
        .work_units = {wu},
    };

    Program program = experimental::metal2_host_api::MakeProgramFromSpec(*this->devices_.at(0), spec);

    using NodeNamedRTAs = experimental::metal2_host_api::ProgramRunParams::KernelRunParams::NodeNamedRTAs;
    experimental::metal2_host_api::ProgramRunParams::KernelRunParams dm_producer_params{
        .kernel_spec_name = DM_PRODUCER};
    dm_producer_params.named_runtime_args = std::vector<NodeNamedRTAs>{NodeNamedRTAs{
        experimental::metal2_host_api::NodeCoord{logical_core.x, logical_core.y},
        {{"chunk_offset", 0u}, {"entries_per_core", num_entries}}}};
    experimental::metal2_host_api::ProgramRunParams::KernelRunParams compute_params{.kernel_spec_name = COMPUTE};

    experimental::metal2_host_api::ProgramRunParams run_params;
    run_params.kernel_run_params = {dm_producer_params, compute_params};
    run_params.tensor_args = {{.tensor_parameter_name = IN_TENSOR, .tensor = std::cref(in_tensor)}};
    experimental::metal2_host_api::SetProgramRunParameters(program, run_params);

    // L1 layout follows DFB creation order:
    //   [l1_base + 0              ] -> remapper DFB ring  (num_entries * entry_size bytes)
    //   [l1_base + remapper_size  ] -> intra DFB ring     (num_entries * entry_size bytes)
    const uint32_t l1_base           = static_cast<uint32_t>(device->allocator()->get_base_allocator_addr(HalMemType::L1));
    const uint32_t remapper_ring_size = num_entries * entry_size;
    const uint32_t intra_l1_addr     = l1_base + remapper_ring_size;

    // Pre-fill intra DFB's ring; kernel adds +2 per word.
    auto input_intra = tt::test_utils::generate_uniform_random_vector<uint32_t>(
        0, 100, num_entries * words_per_entry);
    detail::WriteToDeviceL1(device, logical_core, intra_l1_addr, input_intra);

    // Fill DRAM in_tensor; DM NOC-reads this into remapper DFB's ring.
    auto input_remapper = tt::test_utils::generate_uniform_random_vector<uint32_t>(
        0, 100, num_entries * words_per_entry);
    detail::WriteToBuffer(*in_tensor.mesh_buffer().get_reference_buffer(), input_remapper);

    std::this_thread::sleep_for(std::chrono::milliseconds(500));

    detail::LaunchProgram(device, program, true /*wait_until_cores_done*/);

    // Verify dfb(1): packer +1, unpacker +1 -> net +2 per word.
    {
        std::vector<uint32_t> expected(input_intra.size());
        for (size_t i = 0; i < input_intra.size(); i++) {
            expected[i] = input_intra[i] + 2;
        }
        std::vector<uint32_t> l1_data;
        detail::ReadFromDeviceL1(device, logical_core, intra_l1_addr, num_entries * entry_size, l1_data);
        EXPECT_EQ(expected, l1_data) << "Intra-tensix DFB L1 mismatch";
    }

    // Verify dfb(0): DM NOC-wrote input_remapper into L1; Tensix consumed but did not overwrite.
    {
        std::vector<uint32_t> l1_data;
        detail::ReadFromDeviceL1(device, logical_core, l1_base, num_entries * entry_size, l1_data);
        EXPECT_EQ(input_remapper, l1_data) << "DM->Tensix strided x all DFB L1 mismatch";
    }
}

}  // end namespace tt::tt_metal
