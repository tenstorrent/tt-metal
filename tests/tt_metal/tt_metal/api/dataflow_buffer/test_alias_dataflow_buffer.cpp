// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include <map>
#include <memory>
#include <numeric>
#include <optional>
#include <thread>
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
#include <tt-metalium/experimental/tensor/mesh_tensor.hpp>
#include <tt-metalium/experimental/tensor/topology/tensor_topology.hpp>
#include <tt-metalium/experimental/tensor/spec/tensor_spec.hpp>
#include <tt-metalium/experimental/tensor/spec/layout/tensor_layout.hpp>
#include <tt-metalium/experimental/tensor/spec/layout/page_config.hpp>

#include "device_fixture.hpp"
#include "tt_metal/test_utils/stimulus.hpp"
#include "impl/dataflow_buffer/dataflow_buffer_impl.hpp"
#include "impl/program/program_impl.hpp"

namespace tt::tt_metal {
namespace {

using namespace experimental;

constexpr const char* ALIAS_PRODUCER_KERNEL =
    "tests/tt_metal/tt_metal/test_kernels/dataflow/alias_dfb_producer.cpp";
constexpr const char* ALIAS_CONSUMER_KERNEL =
    "tests/tt_metal/tt_metal/test_kernels/dataflow/alias_dfb_consumer.cpp";

inline TensorSpec make_alias_dram_tensor_spec(uint32_t entry_size, uint32_t num_entries) {
    const uint32_t words = entry_size / sizeof(uint32_t);
    return TensorSpec(
        Shape{num_entries, words},
        TensorLayout(DataType::UINT32, PageConfig(Layout::ROW_MAJOR),
                     MemoryConfig{TensorMemoryLayout::INTERLEAVED, BufferType::DRAM}));
}

inline TensorSpec make_alias_l1_tensor_spec(uint32_t entry_size, uint32_t num_entries) {
    const uint32_t total_words = num_entries * entry_size / sizeof(uint32_t);
    return TensorSpec(
        Shape{1, total_words},
        TensorLayout(DataType::UINT32, PageConfig(Layout::ROW_MAJOR),
                     MemoryConfig{TensorMemoryLayout::INTERLEAVED, BufferType::L1}));
}

// Build a DataflowBufferConfig for direct-API tests (no kernel compilation).
// producer = DM RISC 0, consumer = DM RISC 1 (valid on WH/BH and Quasar).
inline dfb::DataflowBufferConfig make_1sx1s_config(uint32_t entry_size, uint32_t num_entries) {
    return dfb::DataflowBufferConfig{
        .entry_size = entry_size,
        .num_entries = num_entries,
        .producer_risc_mask = 0x1,
        .num_producers = 1,
        .pap = dfb::AccessPattern::STRIDED,
        .consumer_risc_mask = 0x2,
        .num_consumers = 1,
        .cap = dfb::AccessPattern::STRIDED,
        .enable_producer_implicit_sync = false,
        .enable_consumer_implicit_sync = false,
    };
}

struct AliasDFBProgramComponents {
    ProgramSpec       spec;
    MeshTensor        in_a;
    MeshTensor        in_b;
    MeshTensor        out_a;
    MeshTensor        out_b;
};

AliasDFBProgramComponents make_alias_dfb_program_spec(
    const std::shared_ptr<distributed::MeshDevice>& mesh_device,
    const NodeCoord& node,
    uint32_t entry_size_a,
    uint32_t num_entries_a,
    uint32_t entry_size_b,
    uint32_t num_entries_b,
    uint8_t  num_producers,
    uint8_t  num_consumers) {

    const uint32_t epp_a = (num_entries_a + num_producers - 1) / num_producers;
    const uint32_t epp_b = (num_entries_b + num_producers - 1) / num_producers;
    const uint32_t epc_a = (num_entries_a + num_consumers - 1) / num_consumers;
    const uint32_t epc_b = (num_entries_b + num_consumers - 1) / num_consumers;

    MeshTensor in_a  = MeshTensor::allocate_on_device(
        *mesh_device, make_alias_dram_tensor_spec(entry_size_a, num_entries_a), TensorTopology{});
    MeshTensor in_b  = MeshTensor::allocate_on_device(
        *mesh_device, make_alias_dram_tensor_spec(entry_size_b, num_entries_b), TensorTopology{});
    MeshTensor out_a = MeshTensor::allocate_on_device(
        *mesh_device, make_alias_dram_tensor_spec(entry_size_a, num_entries_a), TensorTopology{});
    MeshTensor out_b = MeshTensor::allocate_on_device(
        *mesh_device, make_alias_dram_tensor_spec(entry_size_b, num_entries_b), TensorTopology{});

    // DM kernel configs (Gen1 + Gen2 variants so the same spec runs everywhere).
    DataMovementHardwareConfig producer_cfg;
    DataMovementHardwareConfig consumer_cfg;
    if (mesh_device->arch() == ARCH::QUASAR) {
        producer_cfg = DataMovementGen2Config{.disable_dfb_implicit_sync_for_all = true};
        consumer_cfg = DataMovementGen2Config{.disable_dfb_implicit_sync_for_all = true};
    } else {
        producer_cfg = DataMovementGen1Config{.processor = DataMovementProcessor::RISCV_0};
        consumer_cfg = DataMovementGen1Config{.processor = DataMovementProcessor::RISCV_1, .noc = NOC::NOC_1};
    }

    DataflowBufferSpec dfb_a{
        .unique_id = experimental::DFBSpecName{"dfb_a"},
        .entry_size = entry_size_a,
        .num_entries = num_entries_a,
        .data_format_metadata = tt::DataFormat::Float16_b,
        .advanced_options = DFBAdvancedOptions{.alias_with = {experimental::DFBSpecName{"dfb_b"}}},
    };
    DataflowBufferSpec dfb_b{
        .unique_id = experimental::DFBSpecName{"dfb_b"},
        .entry_size = entry_size_b,
        .num_entries = num_entries_b,
        .data_format_metadata = tt::DataFormat::Float16_b,
        .advanced_options = DFBAdvancedOptions{.alias_with = {experimental::DFBSpecName{"dfb_a"}}},
    };

    KernelSpec producer{
        .unique_id = experimental::KernelSpecName{"producer"},
        .source = ALIAS_PRODUCER_KERNEL,
        .num_threads = num_producers,
        .dfb_bindings =
            {
                {.dfb_spec_name = experimental::DFBSpecName{"dfb_a"},
                 .accessor_name = "out_a",
                 .endpoint_type = DFBEndpointType::PRODUCER,
                 .access_pattern = DFBAccessPattern::STRIDED},
                {.dfb_spec_name = experimental::DFBSpecName{"dfb_b"},
                 .accessor_name = "out_b",
                 .endpoint_type = DFBEndpointType::PRODUCER,
                 .access_pattern = DFBAccessPattern::STRIDED},
            },
        .tensor_bindings =
            {
                {.tensor_parameter_name = experimental::TensorParamName{"in_tensor_a"}, .accessor_name = "src_a"},
                {.tensor_parameter_name = experimental::TensorParamName{"in_tensor_b"}, .accessor_name = "src_b"},
            },
        .compile_time_args =
            {
                {"num_entries_per_producer_a", epp_a},
                {"num_entries_per_producer_b", epp_b},
                {"num_producers", static_cast<uint32_t>(num_producers)},
            },
        .runtime_arg_schema =
            {.runtime_arg_names = {"chunk_offset_a", "chunk_offset_b", "entries_per_core_a", "entries_per_core_b"}},
        .hw_config = producer_cfg,
    };

    KernelSpec consumer{
        .unique_id = experimental::KernelSpecName{"consumer"},
        .source = ALIAS_CONSUMER_KERNEL,
        .num_threads = num_consumers,
        .dfb_bindings =
            {
                {.dfb_spec_name = experimental::DFBSpecName{"dfb_a"},
                 .accessor_name = "in_a",
                 .endpoint_type = DFBEndpointType::CONSUMER,
                 .access_pattern = DFBAccessPattern::STRIDED},
                {.dfb_spec_name = experimental::DFBSpecName{"dfb_b"},
                 .accessor_name = "in_b",
                 .endpoint_type = DFBEndpointType::CONSUMER,
                 .access_pattern = DFBAccessPattern::STRIDED},
            },
        .tensor_bindings =
            {
                {.tensor_parameter_name = experimental::TensorParamName{"out_tensor_a"}, .accessor_name = "dst_a"},
                {.tensor_parameter_name = experimental::TensorParamName{"out_tensor_b"}, .accessor_name = "dst_b"},
            },
        .compile_time_args =
            {
                {"num_entries_per_consumer_a", epc_a},
                {"num_entries_per_consumer_b", epc_b},
                {"num_consumers", static_cast<uint32_t>(num_consumers)},
            },
        .runtime_arg_schema =
            {.runtime_arg_names = {"chunk_offset_a", "chunk_offset_b", "entries_per_core_a", "entries_per_core_b"}},
        .hw_config = consumer_cfg,
    };

    ProgramSpec spec{
        .name = "alias_dfb",
        .kernels = {producer, consumer},
        .dataflow_buffers = {dfb_a, dfb_b},
        .tensor_parameters =
            {
                {.unique_id = experimental::TensorParamName{"in_tensor_a"}, .spec = in_a.tensor_spec()},
                {.unique_id = experimental::TensorParamName{"in_tensor_b"}, .spec = in_b.tensor_spec()},
                {.unique_id = experimental::TensorParamName{"out_tensor_a"}, .spec = out_a.tensor_spec()},
                {.unique_id = experimental::TensorParamName{"out_tensor_b"}, .spec = out_b.tensor_spec()},
            },
        .work_units = {WorkUnitSpec{
            .name = "wu",
            .kernels = {experimental::KernelSpecName{"producer"}, experimental::KernelSpecName{"consumer"}},
            .target_nodes = node,
        }},
    };

    return {std::move(spec), std::move(in_a), std::move(in_b), std::move(out_a), std::move(out_b)};
}

void run_alias_dfb_program(
    const std::shared_ptr<distributed::MeshDevice>& mesh_device,
    const NodeCoord& node,
    uint32_t entry_size_a,
    uint32_t num_entries_a,
    uint32_t entry_size_b,
    uint32_t num_entries_b,
    uint8_t  num_producers = 1,
    uint8_t  num_consumers = 1) {

    auto [spec, in_a, in_b, out_a, out_b] = make_alias_dfb_program_spec(
        mesh_device, node,
        entry_size_a, num_entries_a,
        entry_size_b, num_entries_b,
        num_producers, num_consumers);

    Program program = MakeProgramFromSpec(*mesh_device, spec);

    using RuntimeArgValues = decltype(ProgramRunArgs::KernelRunArgs::runtime_arg_values);
    auto rtas = [&](uint32_t epc_a, uint32_t epc_b) {
        return RuntimeArgValues{
            {node,
             {{"chunk_offset_a", 0u},
              {"chunk_offset_b", 0u},
              {"entries_per_core_a", epc_a},
              {"entries_per_core_b", epc_b}}}};
    };

    ProgramRunArgs run_params;
    run_params.kernel_run_args = {
        ProgramRunArgs::KernelRunArgs{
            .kernel = experimental::KernelSpecName{"producer"},
            .runtime_arg_values = rtas(num_entries_a, num_entries_b),
        },
        ProgramRunArgs::KernelRunArgs{
            .kernel = experimental::KernelSpecName{"consumer"},
            .runtime_arg_values = rtas(num_entries_a, num_entries_b),
        },
    };
    run_params.tensor_args = {
        {experimental::TensorParamName{"in_tensor_a"}, TensorArgument{in_a}},
        {experimental::TensorParamName{"in_tensor_b"}, TensorArgument{in_b}},
        {experimental::TensorParamName{"out_tensor_a"}, TensorArgument{out_a}},
        {experimental::TensorParamName{"out_tensor_b"}, TensorArgument{out_b}},
    };
    SetProgramRunArgs(program, run_params);

    // Generate random inputs.
    const uint32_t words_a = num_entries_a * entry_size_a / sizeof(uint32_t);
    const uint32_t words_b = num_entries_b * entry_size_b / sizeof(uint32_t);
    auto input_a = tt::test_utils::generate_uniform_random_vector<uint32_t>(0, 100, words_a);
    auto input_b = tt::test_utils::generate_uniform_random_vector<uint32_t>(0, 100, words_b);

    detail::WriteToBuffer(*in_a.mesh_buffer().get_reference_buffer(), input_a);
    detail::WriteToBuffer(*in_b.mesh_buffer().get_reference_buffer(), input_b);

    if (mesh_device->arch() == ARCH::QUASAR) {
        // TODO #38042: barrier for Quasar DRAM write visibility.
        std::this_thread::sleep_for(std::chrono::milliseconds(500));
    }

    IDevice* device = mesh_device->get_devices()[0];
    detail::LaunchProgram(device, program, /*wait_until_cores_done=*/true);

    std::vector<uint32_t> result_a, result_b;
    detail::ReadFromBuffer(*out_a.mesh_buffer().get_reference_buffer(), result_a);
    detail::ReadFromBuffer(*out_b.mesh_buffer().get_reference_buffer(), result_b);

    EXPECT_EQ(result_a, input_a)
        << "Phase A output mismatch: DFB_A data did not round-trip correctly";
    EXPECT_EQ(result_b, input_b)
        << "Phase B output mismatch: DFB_B (alias) data did not round-trip correctly";
}

struct AliasBorrowedDFBComponents {
    ProgramSpec  spec;
    MeshTensor   in_a;
    MeshTensor   in_b;
    MeshTensor   out_a;
    MeshTensor   out_b;
    MeshTensor   ring_tensor;  // L1 tensor that backs dfb
};

AliasBorrowedDFBComponents make_alias_borrowed_dfb_program_spec(
    const std::shared_ptr<distributed::MeshDevice>& mesh_device,
    const NodeCoord& node,
    uint32_t entry_size,
    uint32_t num_entries) {

    const uint32_t epp = num_entries;
    const uint32_t epc = num_entries;

    MeshTensor in_a  = MeshTensor::allocate_on_device(
        *mesh_device, make_alias_dram_tensor_spec(entry_size, num_entries), TensorTopology{});
    MeshTensor in_b  = MeshTensor::allocate_on_device(
        *mesh_device, make_alias_dram_tensor_spec(entry_size, num_entries), TensorTopology{});
    MeshTensor out_a = MeshTensor::allocate_on_device(
        *mesh_device, make_alias_dram_tensor_spec(entry_size, num_entries), TensorTopology{});
    MeshTensor out_b = MeshTensor::allocate_on_device(
        *mesh_device, make_alias_dram_tensor_spec(entry_size, num_entries), TensorTopology{});
    MeshTensor ring  = MeshTensor::allocate_on_device(
        *mesh_device, make_alias_l1_tensor_spec(entry_size, num_entries), TensorTopology{});

    DataMovementHardwareConfig producer_cfg;
    DataMovementHardwareConfig consumer_cfg;
    if (mesh_device->arch() == ARCH::QUASAR) {
        producer_cfg = DataMovementGen2Config{.disable_dfb_implicit_sync_for_all = true};
        consumer_cfg = DataMovementGen2Config{.disable_dfb_implicit_sync_for_all = true};
    } else {
        producer_cfg = DataMovementGen1Config{.processor = DataMovementProcessor::RISCV_0};
        consumer_cfg = DataMovementGen1Config{.processor = DataMovementProcessor::RISCV_1, .noc = NOC::NOC_1};
    }

    // dfb_borrowed: backed by ring_tensor (L1)
    DataflowBufferSpec dfb_borrowed{
        .unique_id = experimental::DFBSpecName{"dfb_borrowed"},
        .entry_size = entry_size,
        .num_entries = num_entries,
        .data_format_metadata = tt::DataFormat::Float16_b,
        .borrowed_from = experimental::TensorParamName{"ring_tensor"},
        .advanced_options = DFBAdvancedOptions{.alias_with = {experimental::DFBSpecName{"dfb_alias"}}},
    };
    DataflowBufferSpec dfb_alias_spec{
        .unique_id = experimental::DFBSpecName{"dfb_alias"},
        .entry_size = entry_size,
        .num_entries = num_entries,
        .data_format_metadata = tt::DataFormat::Float16_b,
        .borrowed_from = experimental::TensorParamName{"ring_tensor"},
        .advanced_options = DFBAdvancedOptions{.alias_with = {experimental::DFBSpecName{"dfb_borrowed"}}},
    };

    KernelSpec producer{
        .unique_id = experimental::KernelSpecName{"producer"},
        .source = ALIAS_PRODUCER_KERNEL,
        .num_threads = 1,
        .dfb_bindings =
            {
                {.dfb_spec_name = experimental::DFBSpecName{"dfb_borrowed"},
                 .accessor_name = "out_a",
                 .endpoint_type = DFBEndpointType::PRODUCER,
                 .access_pattern = DFBAccessPattern::STRIDED},
                {.dfb_spec_name = experimental::DFBSpecName{"dfb_alias"},
                 .accessor_name = "out_b",
                 .endpoint_type = DFBEndpointType::PRODUCER,
                 .access_pattern = DFBAccessPattern::STRIDED},
            },
        .tensor_bindings =
            {
                {.tensor_parameter_name = experimental::TensorParamName{"in_tensor_a"}, .accessor_name = "src_a"},
                {.tensor_parameter_name = experimental::TensorParamName{"in_tensor_b"}, .accessor_name = "src_b"},
                // ring_tensor must be bound to at least one kernel even though the
                // kernel doesn't access it directly (required by TensorParameter rules).
                {.tensor_parameter_name = experimental::TensorParamName{"ring_tensor"}, .accessor_name = "ring"},
            },
        .compile_time_args =
            {
                {"num_entries_per_producer_a", epp},
                {"num_entries_per_producer_b", epp},
                {"num_producers", 1u},
            },
        .runtime_arg_schema =
            {.runtime_arg_names = {"chunk_offset_a", "chunk_offset_b", "entries_per_core_a", "entries_per_core_b"}},
        .hw_config = producer_cfg,
    };

    KernelSpec consumer{
        .unique_id = experimental::KernelSpecName{"consumer"},
        .source = ALIAS_CONSUMER_KERNEL,
        .num_threads = 1,
        .dfb_bindings =
            {
                {.dfb_spec_name = experimental::DFBSpecName{"dfb_borrowed"},
                 .accessor_name = "in_a",
                 .endpoint_type = DFBEndpointType::CONSUMER,
                 .access_pattern = DFBAccessPattern::STRIDED},
                {.dfb_spec_name = experimental::DFBSpecName{"dfb_alias"},
                 .accessor_name = "in_b",
                 .endpoint_type = DFBEndpointType::CONSUMER,
                 .access_pattern = DFBAccessPattern::STRIDED},
            },
        .tensor_bindings =
            {
                {.tensor_parameter_name = experimental::TensorParamName{"out_tensor_a"}, .accessor_name = "dst_a"},
                {.tensor_parameter_name = experimental::TensorParamName{"out_tensor_b"}, .accessor_name = "dst_b"},
            },
        .compile_time_args =
            {
                {"num_entries_per_consumer_a", epc},
                {"num_entries_per_consumer_b", epc},
                {"num_consumers", 1u},
            },
        .runtime_arg_schema =
            {.runtime_arg_names = {"chunk_offset_a", "chunk_offset_b", "entries_per_core_a", "entries_per_core_b"}},
        .hw_config = consumer_cfg,
    };

    ProgramSpec spec{
        .name = "alias_borrowed_dfb",
        .kernels = {producer, consumer},
        .dataflow_buffers = {dfb_borrowed, dfb_alias_spec},
        .tensor_parameters =
            {
                {.unique_id = experimental::TensorParamName{"in_tensor_a"}, .spec = in_a.tensor_spec()},
                {.unique_id = experimental::TensorParamName{"in_tensor_b"}, .spec = in_b.tensor_spec()},
                {.unique_id = experimental::TensorParamName{"out_tensor_a"}, .spec = out_a.tensor_spec()},
                {.unique_id = experimental::TensorParamName{"out_tensor_b"}, .spec = out_b.tensor_spec()},
                {.unique_id = experimental::TensorParamName{"ring_tensor"}, .spec = ring.tensor_spec()},
            },
        .work_units = {WorkUnitSpec{
            .name = "wu",
            .kernels = {experimental::KernelSpecName{"producer"}, experimental::KernelSpecName{"consumer"}},
            .target_nodes = node,
        }},
    };

    return {std::move(spec), std::move(in_a), std::move(in_b), std::move(out_a), std::move(out_b), std::move(ring)};
}

TEST_F(MeshDeviceFixture, AliasDFBAddressEquality1Sx1S) {
    const NodeCoord node{0, 0};

    [[maybe_unused]] auto [spec, in_a, in_b, out_a, out_b] = make_alias_dfb_program_spec(
        devices_.at(0), node, 512, 8, 256, 16, 1, 1);

    Program program = MakeProgramFromSpec(*devices_.at(0), spec);

    IDevice* device = devices_.at(0)->get_devices()[0];
    detail::CompileProgram(device, program);
    program.impl().finalize_dataflow_buffer_configs();
    program.impl().allocate_dataflow_buffers(device);

    const uint32_t id_a = program.impl().get_dfb_handle("dfb_a");
    const uint32_t id_b = program.impl().get_dfb_handle("dfb_b");

    const uint32_t addr_a = program.impl().get_dataflow_buffer(id_a)->uniform_alloc_addr();
    const uint32_t addr_b = program.impl().get_dataflow_buffer(id_b)->uniform_alloc_addr();

    EXPECT_EQ(addr_a, addr_b) << "Aliased DFBs must share the same L1 base address";

    log_info(
        tt::LogTest,
        "AliasDFB_AddressEquality_1Sx1S: addr_a=0x{:x}  addr_b=0x{:x}",
        addr_a, addr_b);
}

TEST_F(MeshDeviceFixture, AliasDFBDataFlow1Sx1S) {
    run_alias_dfb_program(
        devices_.at(0), NodeCoord{0, 0},
        /*entry_size_a=*/512, /*num_entries_a=*/8,
        /*entry_size_b=*/256, /*num_entries_b=*/16);
}

TEST_F(MeshDeviceFixture, AliasDFBDataFlow2Sx4S) {
    if (devices_.at(0)->arch() != ARCH::QUASAR) {
        GTEST_SKIP() << "Multi-producer DFB requires Quasar TC hardware";
    }
    run_alias_dfb_program(
        devices_.at(0), NodeCoord{0, 0},
        /*entry_size_a=*/512, /*num_entries_a=*/8,
        /*entry_size_b=*/256, /*num_entries_b=*/16,
        /*num_producers=*/2, /*num_consumers=*/4);
}

TEST_F(MeshDeviceFixture, AliasDFBDataFlow4Sx2S) {
    if (devices_.at(0)->arch() != ARCH::QUASAR) {
        GTEST_SKIP() << "Multi-producer DFB requires Quasar TC hardware";
    }
    run_alias_dfb_program(
        devices_.at(0), NodeCoord{0, 0},
        /*entry_size_a=*/512, /*num_entries_a=*/8,
        /*entry_size_b=*/256, /*num_entries_b=*/16,
        /*num_producers=*/4, /*num_consumers=*/2);
}

TEST_F(MeshDeviceFixture, AliasDFBAllocSecondarySkipped) {
    IDevice* device = devices_.at(0)->get_devices()[0];
    const CoreCoord core{0, 0};

    const auto cfg_a = make_1sx1s_config(512, 8);
    const auto cfg_b = make_1sx1s_config(256, 16);
    const auto cfg_c = make_1sx1s_config(512, 4);

    Program program = CreateProgram();
    const uint32_t id_a = dfb::CreateDataflowBuffer(program, core, cfg_a);
    const uint32_t id_b = dfb::CreateDataflowBuffer(program, core, cfg_b);
    const uint32_t id_c = dfb::CreateDataflowBuffer(program, core, cfg_c);

    program.impl().set_dfb_alias(id_a, id_b);
    program.impl().finalize_dataflow_buffer_configs();
    program.impl().allocate_dataflow_buffers(device);

    const uint32_t addr_a = program.impl().get_dataflow_buffer(id_a)->uniform_alloc_addr();
    const uint32_t addr_b = program.impl().get_dataflow_buffer(id_b)->uniform_alloc_addr();
    const uint32_t addr_c = program.impl().get_dataflow_buffer(id_c)->uniform_alloc_addr();

    EXPECT_EQ(addr_a, addr_b) << "Aliased DFBs must share L1 address";

    // addr_c must follow the primary's footprint, not the secondary's.
    const uint32_t group_end = addr_a + cfg_a.entry_size * cfg_a.num_entries;
    EXPECT_GE(addr_c, group_end)
        << "Non-aliased DFB_C must start after the alias group";
    EXPECT_LT(addr_c, group_end + cfg_b.entry_size * cfg_b.num_entries)
        << "Allocator double-counted the secondary: addr_c is too far";
}

TEST_F(MeshDeviceFixture, AliasDFBAlloc3Way) {
    IDevice* device = devices_.at(0)->get_devices()[0];
    const CoreCoord core{0, 0};

    const auto cfg_a = make_1sx1s_config(512,  8);
    const auto cfg_b = make_1sx1s_config(256,  16);
    const auto cfg_c = make_1sx1s_config(1024, 4);

    Program program = CreateProgram();
    const uint32_t id_a = dfb::CreateDataflowBuffer(program, core, cfg_a);
    const uint32_t id_b = dfb::CreateDataflowBuffer(program, core, cfg_b);
    const uint32_t id_c = dfb::CreateDataflowBuffer(program, core, cfg_c);

    program.impl().set_dfb_alias(id_a, id_b);
    program.impl().set_dfb_alias(id_a, id_c);
    program.impl().finalize_dataflow_buffer_configs();
    program.impl().allocate_dataflow_buffers(device);

    const uint32_t addr_a = program.impl().get_dataflow_buffer(id_a)->uniform_alloc_addr();
    const uint32_t addr_b = program.impl().get_dataflow_buffer(id_b)->uniform_alloc_addr();
    const uint32_t addr_c = program.impl().get_dataflow_buffer(id_c)->uniform_alloc_addr();

    EXPECT_EQ(addr_a, addr_b) << "DFB_B must share DFB_A's address";
    EXPECT_EQ(addr_a, addr_c) << "DFB_C must share DFB_A's address";

    log_info(
        tt::LogTest,
        "AliasDFB_Alloc_3Way: addr_a=0x{:x}  addr_b=0x{:x}  addr_c=0x{:x}",
        addr_a, addr_b, addr_c);
}

TEST_F(MeshDeviceFixture, AliasDFBAgreedGroupResize) {
    IDevice* device = devices_.at(0)->get_devices()[0];
    const CoreCoord core{0, 0};

    // Two aliased DFBs starting at equal total size (4096 B), plus a trailing non-aliased DFB to
    // observe the alias group's L1 footprint.
    const auto cfg_a = make_1sx1s_config(512, 8);   // total 4096 (primary)
    const auto cfg_b = make_1sx1s_config(256, 16);  // total 4096 (secondary)
    const auto cfg_c = make_1sx1s_config(512, 4);   // total 2048 (trailing, non-aliased)

    Program program = CreateProgram();
    const uint32_t id_a = dfb::CreateDataflowBuffer(program, core, cfg_a);
    const uint32_t id_b = dfb::CreateDataflowBuffer(program, core, cfg_b);
    const uint32_t id_c = dfb::CreateDataflowBuffer(program, core, cfg_c);
    program.impl().set_dfb_alias(id_a, id_b);

    program.impl().finalize_dataflow_buffer_configs();
    program.impl().allocate_dataflow_buffers(device);

    const uint32_t addr_a0 = program.impl().get_dataflow_buffer(id_a)->uniform_alloc_addr();
    EXPECT_EQ(addr_a0, program.impl().get_dataflow_buffer(id_b)->uniform_alloc_addr())
        << "Aliased DFBs must share one L1 address";
    EXPECT_GE(program.impl().get_dataflow_buffer(id_c)->uniform_alloc_addr(), addr_a0 + 4096u)
        << "Trailing DFB_C must follow the 4096 B alias group";

    // Agreed group resize: both members -> new equal total 8192 B, via different views.
    std::vector<detail::ProgramImpl::DfbSizeOverride> overrides = {
        {.dfb_id = id_a, .entry_size = 512u, .num_entries = 16u},  // 8192
        {.dfb_id = id_b, .entry_size = 1024u, .num_entries = 8u},  // 8192
    };
    EXPECT_NO_THROW(program.impl().apply_dfb_size_overrides(overrides));
    program.impl().allocate_dataflow_buffers(device);

    auto dfb_a = program.impl().get_dataflow_buffer(id_a);
    auto dfb_b = program.impl().get_dataflow_buffer(id_b);
    EXPECT_EQ(dfb_a->total_size(), 8192u);
    EXPECT_EQ(dfb_b->total_size(), 8192u);
    const uint32_t addr_a1 = dfb_a->uniform_alloc_addr();
    EXPECT_EQ(addr_a1, dfb_b->uniform_alloc_addr())
        << "Aliased DFBs must still share one L1 address after an agreed resize";
    EXPECT_GE(program.impl().get_dataflow_buffer(id_c)->uniform_alloc_addr(), addr_a1 + 8192u)
        << "Trailing DFB_C must follow the resized (8192 B) alias group footprint";
}

TEST_F(MeshDeviceFixture, AliasDFBBorrowedMemoryAddressEquality) {
    const NodeCoord node{0, 0};
    constexpr uint32_t kEntrySize   = 512;
    constexpr uint32_t kNumEntries  = 8;

    auto [spec, in_a, in_b, out_a, out_b, ring] = make_alias_borrowed_dfb_program_spec(
        devices_.at(0), node, kEntrySize, kNumEntries);

    Program program = MakeProgramFromSpec(*devices_.at(0), spec);

    IDevice* device = devices_.at(0)->get_devices()[0];
    detail::CompileProgram(device, program);
    program.impl().finalize_dataflow_buffer_configs();
    program.impl().allocate_dataflow_buffers(device);

    using RuntimeArgValues = decltype(ProgramRunArgs::KernelRunArgs::runtime_arg_values);
    auto rtas = [&]() {
        return RuntimeArgValues{
            {node,
             {{"chunk_offset_a", 0u},
              {"chunk_offset_b", 0u},
              {"entries_per_core_a", kNumEntries},
              {"entries_per_core_b", kNumEntries}}}};
    };
    ProgramRunArgs run_params;
    run_params.kernel_run_args = {
        ProgramRunArgs::KernelRunArgs{
            .kernel = experimental::KernelSpecName{"producer"},
            .runtime_arg_values = rtas(),
        },
        ProgramRunArgs::KernelRunArgs{
            .kernel = experimental::KernelSpecName{"consumer"},
            .runtime_arg_values = rtas(),
        },
    };
    run_params.tensor_args = {
        {experimental::TensorParamName{"in_tensor_a"}, TensorArgument{in_a}},
        {experimental::TensorParamName{"in_tensor_b"}, TensorArgument{in_b}},
        {experimental::TensorParamName{"out_tensor_a"}, TensorArgument{out_a}},
        {experimental::TensorParamName{"out_tensor_b"}, TensorArgument{out_b}},
        {experimental::TensorParamName{"ring_tensor"}, TensorArgument{ring}},
    };
    SetProgramRunArgs(program, run_params);

    const uint32_t id_borrowed = program.impl().get_dfb_handle("dfb_borrowed");
    const uint32_t id_alias    = program.impl().get_dfb_handle("dfb_alias");

    const uint32_t addr_borrowed = program.impl().get_dataflow_buffer(id_borrowed)->uniform_alloc_addr();
    const uint32_t addr_alias    = program.impl().get_dataflow_buffer(id_alias)->uniform_alloc_addr();
    const uint32_t ring_addr     =
        static_cast<uint32_t>(ring.mesh_buffer().get_reference_buffer()->address());

    EXPECT_EQ(addr_borrowed, ring_addr)
        << "dfb_borrowed must resolve to the ring tensor's L1 address";
    EXPECT_EQ(addr_alias, ring_addr)
        << "dfb_alias must inherit the ring tensor's L1 address via alias propagation";

    log_info(
        tt::LogTest,
        "AliasDFB_BorrowedMemory_AddressEquality: addr_borrowed=0x{:x}  addr_alias=0x{:x}  ring=0x{:x}",
        addr_borrowed, addr_alias, ring_addr);
}

TEST_F(MeshDeviceFixture, AliasDFBBorrowedMemoryDataFlow1Sx1S) {
    const NodeCoord node{0, 0};
    constexpr uint32_t kEntrySize  = 512;
    constexpr uint32_t kNumEntries = 8;

    auto [spec, in_a, in_b, out_a, out_b, ring] = make_alias_borrowed_dfb_program_spec(
        devices_.at(0), node, kEntrySize, kNumEntries);

    Program program = MakeProgramFromSpec(*devices_.at(0), spec);

    using RuntimeArgValues = decltype(ProgramRunArgs::KernelRunArgs::runtime_arg_values);
    auto rtas = [&]() {
        return RuntimeArgValues{
            {node,
             {{"chunk_offset_a", 0u},
              {"chunk_offset_b", 0u},
              {"entries_per_core_a", kNumEntries},
              {"entries_per_core_b", kNumEntries}}}};
    };
    ProgramRunArgs run_params;
    run_params.kernel_run_args = {
        ProgramRunArgs::KernelRunArgs{
            .kernel = experimental::KernelSpecName{"producer"},
            .runtime_arg_values = rtas(),
        },
        ProgramRunArgs::KernelRunArgs{
            .kernel = experimental::KernelSpecName{"consumer"},
            .runtime_arg_values = rtas(),
        },
    };
    run_params.tensor_args = {
        {experimental::TensorParamName{"in_tensor_a"}, TensorArgument{in_a}},
        {experimental::TensorParamName{"in_tensor_b"}, TensorArgument{in_b}},
        {experimental::TensorParamName{"out_tensor_a"}, TensorArgument{out_a}},
        {experimental::TensorParamName{"out_tensor_b"}, TensorArgument{out_b}},
        {experimental::TensorParamName{"ring_tensor"}, TensorArgument{ring}},
    };
    SetProgramRunArgs(program, run_params);

    const uint32_t words = kNumEntries * kEntrySize / sizeof(uint32_t);
    auto input_a = tt::test_utils::generate_uniform_random_vector<uint32_t>(0, 100, words);
    auto input_b = tt::test_utils::generate_uniform_random_vector<uint32_t>(0, 100, words);

    detail::WriteToBuffer(*in_a.mesh_buffer().get_reference_buffer(), input_a);
    detail::WriteToBuffer(*in_b.mesh_buffer().get_reference_buffer(), input_b);

    if (devices_.at(0)->arch() == ARCH::QUASAR) {
        std::this_thread::sleep_for(std::chrono::milliseconds(500));
    }

    IDevice* device = devices_.at(0)->get_devices()[0];
    detail::LaunchProgram(device, program, /*wait_until_cores_done=*/true);

    std::vector<uint32_t> result_a, result_b;
    detail::ReadFromBuffer(*out_a.mesh_buffer().get_reference_buffer(), result_a);
    detail::ReadFromBuffer(*out_b.mesh_buffer().get_reference_buffer(), result_b);

    EXPECT_EQ(result_a, input_a)
        << "Phase A (dfb_borrowed) data did not round-trip correctly";
    EXPECT_EQ(result_b, input_b)
        << "Phase B (dfb_alias via borrowed L1) data did not round-trip correctly";
}

}  // namespace
}  // namespace tt::tt_metal
