// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// Tests for DFB device-slot packing across disjoint core sets.

#include <cstdint>
#include <optional>
#include <set>
#include <string>
#include <vector>

#include <gtest/gtest.h>
#include <tt-metalium/device.hpp>
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/tt_metal.hpp>
#include <tt-metalium/experimental/metal2_host_api/program.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_run_args.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_spec.hpp>
#include <tt-metalium/tensor/mesh_tensor.hpp>

#include "device_fixture.hpp"
#include "dfb_test_common.hpp"
#include "tt_metal/impl/dispatch/slow_dispatch.hpp"
#include "impl/dataflow_buffer/dataflow_buffer_impl.hpp"
#include "impl/program/program_impl.hpp"
#include "tt_metal/test_utils/stimulus.hpp"
#include "../metal2_host_api/test_helpers.hpp"

namespace tt::tt_metal {
namespace {

namespace m2 = experimental;
using experimental::test_helpers::MakeMinimalDFB;
using experimental::test_helpers::MakeMinimalWorkUnit;

void require_at_least_two_nodes(distributed::MeshDevice& mesh_device) {
    if (mesh_device.compute_with_storage_grid_size().x < 2) {
        GTEST_SKIP() << "Needs at least two worker nodes in a row (got "
                     << mesh_device.compute_with_storage_grid_size().x << "x"
                     << mesh_device.compute_with_storage_grid_size().y << ")";
    }
}

void apply_gen1_dm_configs(m2::KernelSpec& producer, m2::KernelSpec& consumer, distributed::MeshDevice& mesh_device) {
    if (mesh_device.arch() == ARCH::QUASAR) {
        return;
    }
    producer.hw_config = m2::DataMovementGen1Config{.processor = DataMovementProcessor::RISCV_0};
    consumer.hw_config = m2::DataMovementGen1Config{.processor = DataMovementProcessor::RISCV_1, .noc = NOC::NOC_1};
}

void expect_half_grid_slots_packed(Program& program, uint32_t num_dfbs_per_half, bool /*is_quasar*/) {
    for (uint32_t i = 0; i < num_dfbs_per_half * 2; ++i) {
        const std::string name = "dfb_" + std::to_string(i);
        const uint32_t id = program.impl().get_dfb_handle(name);
        const auto dfb = program.impl().get_dataflow_buffer(id);
        EXPECT_EQ(dfb->id, i);
        // Both Quasar and WH/BH pack device slots per core, so disjoint halves reuse 0..N-1.
        EXPECT_EQ(dfb->device_slot, i % num_dfbs_per_half) << name;
    }
}

void expect_no_slot_collision_on_core(Program& program, const CoreCoord& core) {
    std::set<uint32_t> slots;
    for (const auto& dfb : program.impl().dataflow_buffers_on_core(core)) {
        EXPECT_TRUE(slots.insert(dfb->device_slot).second)
            << "Core (" << core.x << "," << core.y << ") has two DFBs at device slot " << dfb->device_slot;
    }
}

m2::KernelSpec make_touch_producer(const std::string& name, uint32_t num_dfbs, distributed::MeshDevice& mesh_device) {
    const bool implicit_sync = mesh_device.arch() == ARCH::QUASAR;
    auto kernel = make_dm_kernel(
        m2::KernelSpecName{name}, "tests/tt_metal/tt_metal/test_kernels/dataflow/dfb_multi_touch_producer.cpp");
    kernel.compiler_options = {.defines = {{"TEST_NUM_DFBS", std::to_string(num_dfbs)}}};
    kernel.compile_time_args = {{"implicit_sync", implicit_sync ? 1u : 0u}};
    if (!implicit_sync) {
        kernel.hw_config = m2::DataMovementGen1Config{.processor = DataMovementProcessor::RISCV_0};
    }
    return kernel;
}

m2::KernelSpec make_touch_consumer(
    const std::string& name, uint32_t num_dfbs, uint32_t touched_magic, distributed::MeshDevice& mesh_device) {
    const bool implicit_sync = mesh_device.arch() == ARCH::QUASAR;
    auto kernel = make_dm_kernel(
        m2::KernelSpecName{name}, "tests/tt_metal/tt_metal/test_kernels/dataflow/dfb_multi_touch_consumer.cpp");
    kernel.compiler_options = {.defines = {{"TEST_NUM_DFBS", std::to_string(num_dfbs)}}};
    kernel.compile_time_args = {{"implicit_sync", implicit_sync ? 1u : 0u}, {"touched_magic", touched_magic}};
    kernel.runtime_arg_schema = {.runtime_arg_names = {"result_l1_addr"}};
    if (!implicit_sync) {
        kernel.hw_config = m2::DataMovementGen1Config{.processor = DataMovementProcessor::RISCV_1, .noc = NOC::NOC_1};
    }
    return kernel;
}

uint32_t touch_result_l1_addr(distributed::MeshDevice& mesh_device, uint32_t num_dfbs) {
    const uint32_t bytes = num_dfbs * 3 * sizeof(uint32_t);
    const uint32_t align = mesh_device.allocator()->get_alignment(BufferType::L1);
    const uint32_t aligned = (bytes + align - 1) / align * align;
    return static_cast<uint32_t>(mesh_device.l1_size_per_core()) - aligned;
}

void expect_touch_results(
    distributed::MeshDevice& mesh_device,
    const CoreCoord& core,
    uint32_t result_l1_addr,
    uint32_t num_dfbs,
    uint32_t expected_entry_size,
    uint32_t touched_magic,
    const std::vector<uint32_t>& expected_device_slots) {
    ASSERT_EQ(expected_device_slots.size(), num_dfbs);
    std::vector<uint32_t> got;
    slow_dispatch::ReadFromL1(mesh_device, core, result_l1_addr, num_dfbs * 3 * sizeof(uint32_t), got);
    ASSERT_EQ(got.size(), num_dfbs * 3);
    for (uint32_t i = 0; i < num_dfbs; ++i) {
        EXPECT_EQ(got[i * 3 + 0], expected_entry_size) << "DFB local " << i << " entry_size";
        EXPECT_EQ(got[i * 3 + 1], expected_device_slots[i]) << "DFB local " << i << " device slot (get_id)";
        EXPECT_EQ(got[i * 3 + 2], touched_magic | i) << "DFB local " << i << " touched magic";
    }
}

void bind_touch_dfbs(
    m2::KernelSpec& prod, m2::KernelSpec& cons, const std::string& dfb_spec_name, uint32_t local_index) {
    const std::string accessor = "dfb_" + std::to_string(local_index);
    const m2::DFBSpecName dfb{dfb_spec_name};
    prod.dfb_bindings.push_back(m2::DFBBinding{
        .dfb_spec_name = dfb,
        .accessor_name = accessor,
        .endpoint_type = m2::DFBEndpointType::PRODUCER,
    });
    cons.dfb_bindings.push_back(m2::DFBBinding{
        .dfb_spec_name = dfb,
        .accessor_name = accessor,
        .endpoint_type = m2::DFBEndpointType::CONSUMER,
    });
}

// Two WorkUnits on (0,0)/(1,0); each half owns `num_dfbs_per_half` DFBs and touches every one
// (credit handshake on Gen1; config probe under Quasar).
//
// Quasar TxnIdAllocator is program-wide over DFB pool [8, 31] (24 IDs); each implicit-sync
// DFB takes 4 IDs. Large programs (16+16) disable host implicit sync entirely and keep the
// kernel on the probe path so we never exhaust the pool or hang in finish()/ISR drain.
m2::ProgramSpec make_split_touch_spec(distributed::MeshDevice& mesh_device, uint32_t num_dfbs_per_half) {
    const m2::NodeCoord node_a{0, 0};
    const m2::NodeCoord node_b{1, 0};
    constexpr uint32_t touched_magic = 0xA11CED00u;

    auto prod_a = make_touch_producer("prod_a", num_dfbs_per_half, mesh_device);
    auto cons_a = make_touch_consumer("cons_a", num_dfbs_per_half, touched_magic, mesh_device);
    auto prod_b = make_touch_producer("prod_b", num_dfbs_per_half, mesh_device);
    auto cons_b = make_touch_consumer("cons_b", num_dfbs_per_half, touched_magic, mesh_device);

    m2::ProgramSpec spec;
    spec.name = "dfb_disjoint_split_touch";
    const bool is_quasar = mesh_device.arch() == ARCH::QUASAR;
    for (uint32_t i = 0; i < num_dfbs_per_half * 2; ++i) {
        const std::string name = "dfb_" + std::to_string(i);
        auto dfb = MakeMinimalDFB(name, /*entry_size=*/32, /*num_entries=*/2);
        dfb.data_format_metadata = tt::DataFormat::Float16_b;
        spec.dataflow_buffers.push_back(dfb);

        const uint32_t local = i % num_dfbs_per_half;
        auto& prod = i < num_dfbs_per_half ? prod_a : prod_b;
        auto& cons = i < num_dfbs_per_half ? cons_a : cons_b;
        bind_touch_dfbs(prod, cons, name, local);

        // 16+16 exhausts TxnIdAllocator if every DFB stays on implicit sync; probe path
        // does not need host ISR credits.
        if (is_quasar) {
            disable_implicit_sync_for(prod, m2::DFBSpecName{name});
            disable_implicit_sync_for(cons, m2::DFBSpecName{name});
        }
    }
    spec.kernels = {prod_a, cons_a, prod_b, cons_b};
    spec.work_units = {
        MakeMinimalWorkUnit("work_unit_a", node_a, {"prod_a", "cons_a"}),
        MakeMinimalWorkUnit("work_unit_b", node_b, {"prod_b", "cons_b"}),
    };
    return spec;
}

TEST_F(UnitMeshAnyDispatchFixture, HalfGrid3Plus3DFBsOnDevice) {
    require_at_least_two_nodes(this->device());
    const bool is_quasar = this->device().arch() == ARCH::QUASAR;

    constexpr uint32_t per_half = 3;
    constexpr uint32_t entry_size = 256;
    constexpr uint32_t num_entries = 4;
    const m2::NodeCoord node_a{0, 0};
    const m2::NodeCoord node_b{1, 0};

    const auto tensor_spec = make_flat_dram_tensor_spec(entry_size, num_entries, DataType::UINT32);
    std::vector<MeshTensor> inputs;
    std::vector<MeshTensor> outputs;
    inputs.reserve(per_half * 2);
    outputs.reserve(per_half * 2);
    for (uint32_t i = 0; i < per_half * 2; ++i) {
        inputs.push_back(MeshTensor::allocate_on_device(this->device(), tensor_spec));
        outputs.push_back(MeshTensor::allocate_on_device(this->device(), tensor_spec));
    }

    auto make_triple_prod = [&](const std::string& name) {
        auto kernel = make_dm_kernel(
            m2::KernelSpecName{name}, "tests/tt_metal/tt_metal/test_kernels/dataflow/dfb_triple_producer.cpp");
        kernel.compile_time_args = {{"num_entries_per_producer", num_entries}, {"implicit_sync", is_quasar ? 1u : 0u}};
        kernel.runtime_arg_schema = {.runtime_arg_names = {"chunk_offset", "entries_per_core"}};
        return kernel;
    };
    auto make_triple_cons = [&](const std::string& name) {
        auto kernel = make_dm_kernel(
            m2::KernelSpecName{name}, "tests/tt_metal/tt_metal/test_kernels/dataflow/dfb_triple_consumer.cpp");
        kernel.compile_time_args = {{"num_entries_per_consumer", num_entries}, {"implicit_sync", is_quasar ? 1u : 0u}};
        kernel.runtime_arg_schema = {.runtime_arg_names = {"chunk_offset", "entries_per_core"}};
        return kernel;
    };

    auto prod_a = make_triple_prod("prod_a");
    auto cons_a = make_triple_cons("cons_a");
    auto prod_b = make_triple_prod("prod_b");
    auto cons_b = make_triple_cons("cons_b");
    apply_gen1_dm_configs(prod_a, cons_a, this->device());
    apply_gen1_dm_configs(prod_b, cons_b, this->device());

    m2::ProgramSpec spec;
    spec.name = "half_grid_3plus3_dataflow";
    std::vector<m2::TensorParameter> tensor_params;
    for (uint32_t i = 0; i < per_half * 2; ++i) {
        const std::string dfb_name = "dfb_" + std::to_string(i);
        const uint32_t local = i % per_half;
        const bool left = i < per_half;
        auto& prod = left ? prod_a : prod_b;
        auto& cons = left ? cons_a : cons_b;

        spec.dataflow_buffers.push_back({
            .unique_id = m2::DFBSpecName{dfb_name},
            .entry_size = entry_size,
            .num_entries = num_entries,
            .data_format_metadata = tt::DataFormat::Float16_b,
        });
        const m2::DFBSpecName dfb{dfb_name};
        prod.dfb_bindings.push_back({
            .dfb_spec_name = dfb,
            .accessor_name = "out_" + std::to_string(local),
            .endpoint_type = m2::DFBEndpointType::PRODUCER,
        });
        cons.dfb_bindings.push_back({
            .dfb_spec_name = dfb,
            .accessor_name = "in_" + std::to_string(local),
            .endpoint_type = m2::DFBEndpointType::CONSUMER,
        });

        const std::string in_name = "in_" + std::to_string(i);
        const std::string out_name = "out_" + std::to_string(i);
        tensor_params.push_back({.unique_id = m2::TensorParamName{in_name}, .spec = inputs[i].tensor_spec()});
        tensor_params.push_back({.unique_id = m2::TensorParamName{out_name}, .spec = outputs[i].tensor_spec()});
        prod.tensor_bindings.push_back({
            .tensor_parameter_name = m2::TensorParamName{in_name},
            .accessor_name = "src_" + std::to_string(local),
        });
        cons.tensor_bindings.push_back({
            .tensor_parameter_name = m2::TensorParamName{out_name},
            .accessor_name = "dst_" + std::to_string(local),
        });
    }
    spec.kernels = {prod_a, cons_a, prod_b, cons_b};
    spec.tensor_parameters = std::move(tensor_params);
    spec.work_units = {
        {.name = "wu_a",
         .kernels = {m2::KernelSpecName{"prod_a"}, m2::KernelSpecName{"cons_a"}},
         .target_nodes = node_a},
        {.name = "wu_b",
         .kernels = {m2::KernelSpecName{"prod_b"}, m2::KernelSpecName{"cons_b"}},
         .target_nodes = node_b},
    };

    Program program = m2::MakeProgramFromSpec(this->device(), spec);
    expect_half_grid_slots_packed(program, per_half, is_quasar);
    expect_no_slot_collision_on_core(program, CoreCoord{0, 0});
    expect_no_slot_collision_on_core(program, CoreCoord{1, 0});

    auto rtas = [&](const m2::NodeCoord& node) {
        return m2::MakeRuntimeArgsForSingleNode(node, {{"chunk_offset", 0u}, {"entries_per_core", num_entries}});
    };
    m2::ProgramRunArgs params;
    params.kernel_run_args = {
        {.kernel = m2::KernelSpecName{"prod_a"}, .runtime_arg_values = rtas(node_a)},
        {.kernel = m2::KernelSpecName{"cons_a"}, .runtime_arg_values = rtas(node_a)},
        {.kernel = m2::KernelSpecName{"prod_b"}, .runtime_arg_values = rtas(node_b)},
        {.kernel = m2::KernelSpecName{"cons_b"}, .runtime_arg_values = rtas(node_b)},
    };
    const uint32_t words = num_entries * entry_size / sizeof(uint32_t);
    std::vector<std::vector<uint32_t>> expected(per_half * 2);
    for (uint32_t i = 0; i < per_half * 2; ++i) {
        expected[i] = tt::test_utils::generate_uniform_random_vector<uint32_t>(0, 1000000, words);
        slow_dispatch::WriteToBuffer(inputs[i].mesh_buffer(), expected[i]);
        m2_writeshard_barrier_uint32(this->device(), inputs[i], expected[i]);
        params.tensor_args.insert({m2::TensorParamName{"in_" + std::to_string(i)}, std::cref(inputs[i])});
        params.tensor_args.insert({m2::TensorParamName{"out_" + std::to_string(i)}, std::cref(outputs[i])});
    }
    m2::SetProgramRunArgs(program, params);

    LaunchProgram(this->device(), std::move(program), /*wait_until_cores_done=*/true);

    for (uint32_t i = 0; i < per_half * 2; ++i) {
        std::vector<uint32_t> got;
        slow_dispatch::ReadFromBuffer(outputs[i].mesh_buffer(), got);
        EXPECT_EQ(got, expected[i]) << "DFB " << i << " round-trip mismatch";
    }
}

// Stress: 16+16 DFBs. Credit-handshake over every buffer; host checks entry_size + device slot
// written back from each consumer.
TEST_F(UnitMeshAnyDispatchFixture, HalfGrid16Plus16DFBsOnDevice) {
    require_at_least_two_nodes(this->device());
    const bool is_quasar = this->device().arch() == ARCH::QUASAR;

    constexpr uint32_t per_half = 16;
    constexpr uint32_t touched_magic = 0xA11CED00u;
    m2::ProgramSpec spec = make_split_touch_spec(this->device(), per_half);
    auto device_range = distributed::MeshCoordinateRange(this->device().shape());
    distributed::MeshWorkload workload;
    workload.add_program(device_range, m2::MakeProgramFromSpec(this->device(), spec));
    Program& program = workload.get_programs().at(device_range);

    expect_half_grid_slots_packed(program, per_half, is_quasar);
    expect_no_slot_collision_on_core(program, CoreCoord{0, 0});
    expect_no_slot_collision_on_core(program, CoreCoord{1, 0});

    const uint32_t result_addr = touch_result_l1_addr(this->device(), per_half);
    auto cons_rtas = [&](const m2::NodeCoord& node) {
        return m2::MakeRuntimeArgsForSingleNode(node, {{"result_l1_addr", result_addr}});
    };
    m2::ProgramRunArgs params;
    params.kernel_run_args = {
        {.kernel = m2::KernelSpecName{"prod_a"}},
        {.kernel = m2::KernelSpecName{"cons_a"}, .runtime_arg_values = cons_rtas(m2::NodeCoord{0, 0})},
        {.kernel = m2::KernelSpecName{"prod_b"}},
        {.kernel = m2::KernelSpecName{"cons_b"}, .runtime_arg_values = cons_rtas(m2::NodeCoord{1, 0})},
    };
    m2::SetProgramRunArgs(program, params);
    distributed::EnqueueMeshWorkload(this->device().mesh_command_queue(), workload, /*blocking=*/true);

    std::vector<uint32_t> left_slots(per_half), right_slots(per_half);
    for (uint32_t i = 0; i < per_half; ++i) {
        left_slots[i] =
            program.impl().get_dataflow_buffer(program.impl().get_dfb_handle("dfb_" + std::to_string(i)))->device_slot;
        right_slots[i] = program.impl()
                             .get_dataflow_buffer(program.impl().get_dfb_handle("dfb_" + std::to_string(per_half + i)))
                             ->device_slot;
    }
    expect_touch_results(
        this->device(), CoreCoord{0, 0}, result_addr, per_half, /*expected_entry_size=*/32, touched_magic, left_slots);
    expect_touch_results(
        this->device(), CoreCoord{1, 0}, result_addr, per_half, /*expected_entry_size=*/32, touched_magic, right_slots);
}

// End-to-end identity across two disjoint halves: one DFB + producer/consumer pair per half.
TEST_F(UnitMeshAnyDispatchFixture, HalfGridOnDeviceDataflow1DFBEach) {
    require_at_least_two_nodes(this->device());
    const bool is_quasar = this->device().arch() == ARCH::QUASAR;

    constexpr uint32_t entry_size = 1024;
    constexpr uint32_t num_entries = 8;
    const m2::NodeCoord node_a{0, 0};
    const m2::NodeCoord node_b{1, 0};

    const auto tensor_spec = make_flat_dram_tensor_spec(entry_size, num_entries, DataType::UINT32);
    auto in_a = MeshTensor::allocate_on_device(this->device(), tensor_spec);
    auto out_a = MeshTensor::allocate_on_device(this->device(), tensor_spec);
    auto in_b = MeshTensor::allocate_on_device(this->device(), tensor_spec);
    auto out_b = MeshTensor::allocate_on_device(this->device(), tensor_spec);

    const m2::DFBSpecName dfb_a{"dfb_a"};
    const m2::DFBSpecName dfb_b{"dfb_b"};
    const m2::KernelSpecName prod_a{"producer_a"};
    const m2::KernelSpecName cons_a{"consumer_a"};
    const m2::KernelSpecName prod_b{"producer_b"};
    const m2::KernelSpecName cons_b{"consumer_b"};
    const m2::TensorParamName in_a_name{"in_a"};
    const m2::TensorParamName out_a_name{"out_a"};
    const m2::TensorParamName in_b_name{"in_b"};
    const m2::TensorParamName out_b_name{"out_b"};

    auto producer_a = make_dm_dfb_producer(prod_a, dfb_a, in_a_name, num_entries, is_quasar);
    auto consumer_a =
        make_dm_dfb_consumer(cons_a, dfb_a, out_a_name, num_entries, /*blocked_consumer=*/false, is_quasar);
    auto producer_b = make_dm_dfb_producer(prod_b, dfb_b, in_b_name, num_entries, is_quasar);
    auto consumer_b =
        make_dm_dfb_consumer(cons_b, dfb_b, out_b_name, num_entries, /*blocked_consumer=*/false, is_quasar);
    apply_gen1_dm_configs(producer_a, consumer_a, this->device());
    apply_gen1_dm_configs(producer_b, consumer_b, this->device());

    m2::ProgramSpec spec{
        .name = "half_grid_dataflow_1dfb",
        .kernels = {producer_a, consumer_a, producer_b, consumer_b},
        .dataflow_buffers =
            {
                {.unique_id = dfb_a,
                 .entry_size = entry_size,
                 .num_entries = num_entries,
                 .data_format_metadata = tt::DataFormat::Float16_b},
                {.unique_id = dfb_b,
                 .entry_size = entry_size,
                 .num_entries = num_entries,
                 .data_format_metadata = tt::DataFormat::Float16_b},
            },
        .tensor_parameters =
            {
                {.unique_id = in_a_name, .spec = in_a.tensor_spec()},
                {.unique_id = out_a_name, .spec = out_a.tensor_spec()},
                {.unique_id = in_b_name, .spec = in_b.tensor_spec()},
                {.unique_id = out_b_name, .spec = out_b.tensor_spec()},
            },
        .work_units =
            {
                {.name = "wu_a", .kernels = {prod_a, cons_a}, .target_nodes = node_a},
                {.name = "wu_b", .kernels = {prod_b, cons_b}, .target_nodes = node_b},
            },
    };

    Program program = m2::MakeProgramFromSpec(this->device(), spec);
    EXPECT_EQ(program.impl().get_dataflow_buffer(program.impl().get_dfb_handle("dfb_a"))->device_slot, 0u);
    EXPECT_EQ(program.impl().get_dataflow_buffer(program.impl().get_dfb_handle("dfb_b"))->device_slot, 0u);

    auto rtas = [&](const m2::NodeCoord& node) {
        return m2::MakeRuntimeArgsForSingleNode(node, {{"chunk_offset", 0u}, {"entries_per_core", num_entries}});
    };
    m2::ProgramRunArgs params;
    params.kernel_run_args = {
        {.kernel = prod_a, .runtime_arg_values = rtas(node_a)},
        {.kernel = cons_a, .runtime_arg_values = rtas(node_a)},
        {.kernel = prod_b, .runtime_arg_values = rtas(node_b)},
        {.kernel = cons_b, .runtime_arg_values = rtas(node_b)},
    };
    params.tensor_args = {
        {in_a_name, std::cref(in_a)},
        {out_a_name, std::cref(out_a)},
        {in_b_name, std::cref(in_b)},
        {out_b_name, std::cref(out_b)},
    };
    m2::SetProgramRunArgs(program, params);

    const uint32_t words = num_entries * entry_size / sizeof(uint32_t);
    auto input_a = tt::test_utils::generate_uniform_random_vector<uint32_t>(0, 1000000, words);
    auto input_b = tt::test_utils::generate_uniform_random_vector<uint32_t>(0, 1000000, words);
    slow_dispatch::WriteToBuffer(in_a.mesh_buffer(), input_a);
    slow_dispatch::WriteToBuffer(in_b.mesh_buffer(), input_b);
    m2_writeshard_barrier_uint32(this->device(), in_a, input_a);
    m2_writeshard_barrier_uint32(this->device(), in_b, input_b);

    LaunchProgram(this->device(), std::move(program), /*wait_until_cores_done=*/true);

    std::vector<uint32_t> result_a, result_b;
    slow_dispatch::ReadFromBuffer(out_a.mesh_buffer(), result_a);
    slow_dispatch::ReadFromBuffer(out_b.mesh_buffer(), result_b);
    EXPECT_EQ(result_a, input_a) << "left-half DFB round-trip mismatch";
    EXPECT_EQ(result_b, input_b) << "right-half DFB round-trip mismatch";
}

TEST_F(UnitMeshAnyDispatchFixture, CoordinatorWorkerSlotReuseOnDevice) {
    const CoreCoord grid = this->device().compute_with_storage_grid_size();
    if (grid.x < 3) {
        GTEST_SKIP() << "Needs at least coordinator + 2 workers (got " << grid.x << "x" << grid.y << ")";
    }

    const m2::NodeCoord coordinator{0, 0};
    const m2::NodeCoord worker_start{1, 0};
    const m2::NodeCoord worker_end{static_cast<size_t>(grid.x - 1), 0};

    // Coordinator: wide + coord-only (2). Workers: wide + 3 worker-only (4).
    constexpr uint32_t touched_magic = 0xA11CED00u;
    auto coord_prod = make_touch_producer("coord_prod", 2, this->device());
    auto coord_cons = make_touch_consumer("coord_cons", 2, touched_magic, this->device());
    auto worker_prod = make_touch_producer("worker_prod", 4, this->device());
    auto worker_cons = make_touch_consumer("worker_cons", 4, touched_magic, this->device());

    m2::ProgramSpec spec;
    spec.name = "dfb_sort_shape_touch";

    auto add_dfb = [&](const std::string& name) {
        auto dfb = MakeMinimalDFB(name, 32, 2);
        dfb.data_format_metadata = tt::DataFormat::Float16_b;
        spec.dataflow_buffers.push_back(dfb);
    };
    add_dfb("dfb_wide");
    add_dfb("dfb_worker_0");
    add_dfb("dfb_worker_1");
    add_dfb("dfb_worker_2");
    add_dfb("dfb_coord");

    bind_touch_dfbs(coord_prod, coord_cons, "dfb_wide", 0);
    bind_touch_dfbs(coord_prod, coord_cons, "dfb_coord", 1);
    bind_touch_dfbs(worker_prod, worker_cons, "dfb_wide", 0);
    bind_touch_dfbs(worker_prod, worker_cons, "dfb_worker_0", 1);
    bind_touch_dfbs(worker_prod, worker_cons, "dfb_worker_1", 2);
    bind_touch_dfbs(worker_prod, worker_cons, "dfb_worker_2", 3);

    spec.kernels = {coord_prod, coord_cons, worker_prod, worker_cons};
    spec.work_units = {
        MakeMinimalWorkUnit("wu_coord", coordinator, {"coord_prod", "coord_cons"}),
        MakeMinimalWorkUnit("wu_workers", m2::NodeRange{worker_start, worker_end}, {"worker_prod", "worker_cons"}),
    };

    Program program = m2::MakeProgramFromSpec(this->device(), spec);
    const auto& impl = program.impl();
    EXPECT_EQ(impl.get_dataflow_buffer(impl.get_dfb_handle("dfb_coord"))->id, 4u);
    EXPECT_EQ(impl.get_dataflow_buffer(impl.get_dfb_handle("dfb_wide"))->device_slot, 0u);
    EXPECT_EQ(impl.get_dataflow_buffer(impl.get_dfb_handle("dfb_worker_0"))->device_slot, 1u);
    EXPECT_EQ(impl.get_dataflow_buffer(impl.get_dfb_handle("dfb_worker_1"))->device_slot, 2u);
    EXPECT_EQ(impl.get_dataflow_buffer(impl.get_dfb_handle("dfb_worker_2"))->device_slot, 3u);
    EXPECT_EQ(impl.get_dataflow_buffer(impl.get_dfb_handle("dfb_coord"))->device_slot, 1u);
    expect_no_slot_collision_on_core(program, CoreCoord{0, 0});
    expect_no_slot_collision_on_core(program, CoreCoord{1, 0});

    m2::ProgramRunArgs params;
    const uint32_t coord_result_addr = touch_result_l1_addr(this->device(), 2);
    const uint32_t worker_result_addr = touch_result_l1_addr(this->device(), 4);
    m2::KernelRunArgs::RuntimeArgValues worker_cons_rtas;
    for (size_t x = worker_start.x; x <= worker_end.x; ++x) {
        m2::AddRuntimeArgsForNode(worker_cons_rtas, m2::NodeCoord{x, 0}, {{"result_l1_addr", worker_result_addr}});
    }
    params.kernel_run_args = {
        {.kernel = m2::KernelSpecName{"coord_prod"}},
        {.kernel = m2::KernelSpecName{"coord_cons"},
         .runtime_arg_values = m2::MakeRuntimeArgsForSingleNode(coordinator, {{"result_l1_addr", coord_result_addr}})},
        {.kernel = m2::KernelSpecName{"worker_prod"}},
        {.kernel = m2::KernelSpecName{"worker_cons"}, .runtime_arg_values = std::move(worker_cons_rtas)},
    };
    m2::SetProgramRunArgs(program, params);
    distributed::MeshWorkload coord_worker_workload;
    coord_worker_workload.add_program(distributed::MeshCoordinateRange(this->device().shape()), std::move(program));
    distributed::EnqueueMeshWorkload(this->device().mesh_command_queue(), coord_worker_workload, /*blocking=*/true);

    expect_touch_results(
        this->device(),
        CoreCoord{0, 0},
        coord_result_addr,
        2,
        /*expected_entry_size=*/32,
        touched_magic,
        {impl.get_dataflow_buffer(impl.get_dfb_handle("dfb_wide"))->device_slot,
         impl.get_dataflow_buffer(impl.get_dfb_handle("dfb_coord"))->device_slot});
    expect_touch_results(
        this->device(),
        CoreCoord{1, 0},
        worker_result_addr,
        4,
        /*expected_entry_size=*/32,
        touched_magic,
        {impl.get_dataflow_buffer(impl.get_dfb_handle("dfb_wide"))->device_slot,
         impl.get_dataflow_buffer(impl.get_dfb_handle("dfb_worker_0"))->device_slot,
         impl.get_dataflow_buffer(impl.get_dfb_handle("dfb_worker_1"))->device_slot,
         impl.get_dataflow_buffer(impl.get_dfb_handle("dfb_worker_2"))->device_slot});
}

// Same topology as CoordinatorWorkerSlotReuseOnDevice, but sparse DFBs are created first so the
// all-cores DFB lands at a high slot.
TEST_F(UnitMeshAnyDispatchFixture, CoordinatorWorkerGappedSlotsOnDevice) {
    const CoreCoord grid = this->device().compute_with_storage_grid_size();
    if (grid.x < 3) {
        GTEST_SKIP() << "Needs at least coordinator + 2 workers (got " << grid.x << "x" << grid.y << ")";
    }

    const m2::NodeCoord coordinator{0, 0};
    const m2::NodeCoord worker_start{1, 0};
    const m2::NodeCoord worker_end{static_cast<size_t>(grid.x - 1), 0};

    constexpr uint32_t touched_magic = 0xA11CED00u;
    auto coord_prod = make_touch_producer("coord_prod", 2, this->device());
    auto coord_cons = make_touch_consumer("coord_cons", 2, touched_magic, this->device());
    auto worker_prod = make_touch_producer("worker_prod", 4, this->device());
    auto worker_cons = make_touch_consumer("worker_cons", 4, touched_magic, this->device());

    m2::ProgramSpec spec;
    spec.name = "dfb_gapped_slot_touch";

    auto add_dfb = [&](const std::string& name) {
        auto dfb = MakeMinimalDFB(name, 32, 2);
        dfb.data_format_metadata = tt::DataFormat::Float16_b;
        spec.dataflow_buffers.push_back(dfb);
    };
    add_dfb("dfb_worker_0");
    add_dfb("dfb_worker_1");
    add_dfb("dfb_worker_2");
    add_dfb("dfb_coord");
    add_dfb("dfb_wide");

    bind_touch_dfbs(coord_prod, coord_cons, "dfb_wide", 0);
    bind_touch_dfbs(coord_prod, coord_cons, "dfb_coord", 1);
    bind_touch_dfbs(worker_prod, worker_cons, "dfb_wide", 0);
    bind_touch_dfbs(worker_prod, worker_cons, "dfb_worker_0", 1);
    bind_touch_dfbs(worker_prod, worker_cons, "dfb_worker_1", 2);
    bind_touch_dfbs(worker_prod, worker_cons, "dfb_worker_2", 3);

    spec.kernels = {coord_prod, coord_cons, worker_prod, worker_cons};
    spec.work_units = {
        MakeMinimalWorkUnit("wu_coord", coordinator, {"coord_prod", "coord_cons"}),
        MakeMinimalWorkUnit("wu_workers", m2::NodeRange{worker_start, worker_end}, {"worker_prod", "worker_cons"}),
    };

    Program program = m2::MakeProgramFromSpec(this->device(), spec);
    const auto& impl = program.impl();
    EXPECT_EQ(impl.get_dataflow_buffer(impl.get_dfb_handle("dfb_worker_0"))->device_slot, 0u);
    EXPECT_EQ(impl.get_dataflow_buffer(impl.get_dfb_handle("dfb_worker_1"))->device_slot, 1u);
    EXPECT_EQ(impl.get_dataflow_buffer(impl.get_dfb_handle("dfb_worker_2"))->device_slot, 2u);
    EXPECT_EQ(impl.get_dataflow_buffer(impl.get_dfb_handle("dfb_coord"))->device_slot, 0u);
    EXPECT_EQ(impl.get_dataflow_buffer(impl.get_dfb_handle("dfb_wide"))->device_slot, 3u);
    expect_no_slot_collision_on_core(program, CoreCoord{0, 0});
    expect_no_slot_collision_on_core(program, CoreCoord{1, 0});

    m2::ProgramRunArgs params;
    const uint32_t coord_result_addr = touch_result_l1_addr(this->device(), 2);
    const uint32_t worker_result_addr = touch_result_l1_addr(this->device(), 4);
    m2::KernelRunArgs::RuntimeArgValues worker_cons_rtas;
    for (size_t x = worker_start.x; x <= worker_end.x; ++x) {
        m2::AddRuntimeArgsForNode(worker_cons_rtas, m2::NodeCoord{x, 0}, {{"result_l1_addr", worker_result_addr}});
    }
    params.kernel_run_args = {
        {.kernel = m2::KernelSpecName{"coord_prod"}},
        {.kernel = m2::KernelSpecName{"coord_cons"},
         .runtime_arg_values = m2::MakeRuntimeArgsForSingleNode(coordinator, {{"result_l1_addr", coord_result_addr}})},
        {.kernel = m2::KernelSpecName{"worker_prod"}},
        {.kernel = m2::KernelSpecName{"worker_cons"}, .runtime_arg_values = std::move(worker_cons_rtas)},
    };
    m2::SetProgramRunArgs(program, params);
    distributed::MeshWorkload coord_worker_workload;
    coord_worker_workload.add_program(distributed::MeshCoordinateRange(this->device().shape()), std::move(program));
    distributed::EnqueueMeshWorkload(this->device().mesh_command_queue(), coord_worker_workload, /*blocking=*/true);

    // Coordinator: slots 0 (coord-only) and 3 (wide); 1 and 2 unused.
    expect_touch_results(
        this->device(),
        CoreCoord{0, 0},
        coord_result_addr,
        2,
        /*expected_entry_size=*/32,
        touched_magic,
        {impl.get_dataflow_buffer(impl.get_dfb_handle("dfb_wide"))->device_slot,
         impl.get_dataflow_buffer(impl.get_dfb_handle("dfb_coord"))->device_slot});
    expect_touch_results(
        this->device(),
        CoreCoord{1, 0},
        worker_result_addr,
        4,
        /*expected_entry_size=*/32,
        touched_magic,
        {impl.get_dataflow_buffer(impl.get_dfb_handle("dfb_wide"))->device_slot,
         impl.get_dataflow_buffer(impl.get_dfb_handle("dfb_worker_0"))->device_slot,
         impl.get_dataflow_buffer(impl.get_dfb_handle("dfb_worker_1"))->device_slot,
         impl.get_dataflow_buffer(impl.get_dfb_handle("dfb_worker_2"))->device_slot});
}

}  // namespace
}  // namespace tt::tt_metal
