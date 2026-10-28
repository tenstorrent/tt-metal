// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// ttnn::experimental::generic_op on a Metal 2.0 ProgramSpec and ProgramRunArgs, through the C++ API.
// tests/ttnn/unit_tests/operations/debug/test_generic_op_program_spec.py is the main suite.

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <cstdint>
#include <exception>
#include <string>
#include <utility>
#include <vector>

#include <tt-metalium/constants.hpp>
#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_run_args.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_spec.hpp>
#include <tt-metalium/work_split.hpp>

#include "ttnn/operations/core/data_movement_kernel/datamovement_kernel_config.hpp"
#include "ttnn/operations/eltwise/unary/unary.hpp"
#include "ttnn/operations/functions.hpp"
#include "ttnn/operations/generic/generic_op.hpp"
#include "ttnn/tensor/tensor.hpp"
#include "ttnn_test_fixtures.hpp"

namespace ttnn::operations::generic::test {
namespace {

namespace m2 = tt::tt_metal::experimental;
using ::testing::HasSubstr;
using ::testing::ThrowsMessage;

constexpr const char* kUnaryKernels = "ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels";

struct Program {
    m2::ProgramSpec spec;
    m2::ProgramRunArgs run_args;
};

// output = exp(input) with the tiles split over the cores, as in the design doc's example.
Program make_exp_program(const Tensor& input, const Tensor& output) {
    const uint32_t num_tiles = input.physical_volume() / tt::constants::TILE_HW;
    const auto [num_cores, cores, group_1, group_2, tiles_1, tiles_2] =
        tt::tt_metal::split_work_to_cores(input.device()->compute_with_storage_grid_size(), num_tiles);
    const uint32_t tile_bytes = tt::tile_size(tt::DataFormat::Float16_b);

    const m2::KernelSpec reader{
        .unique_id = m2::KernelSpecName{"reader"},
        .source = std::string(kUnaryKernels) + "/dataflow/reader_unary_interleaved_start_id_metal2.cpp",
        .dfb_bindings = {m2::ProducerOf(m2::DFBSpecName{"in"}, "in")},
        .tensor_bindings = {{.tensor_parameter_name = m2::TensorParamName{"input"}, .accessor_name = "src"}},
        .runtime_arg_schema = {.runtime_arg_names = {"num_pages", "start_id"}},
        .hw_config = ttnn::create_reader_datamovement_config(/*disable_dfb_implicit_sync_for_all=*/true),
    };
    const m2::KernelSpec writer{
        .unique_id = m2::KernelSpecName{"writer"},
        .source = std::string(kUnaryKernels) + "/dataflow/writer_unary_interleaved_start_id_metal2.cpp",
        .dfb_bindings = {m2::ConsumerOf(m2::DFBSpecName{"out"}, "out")},
        .tensor_bindings = {{.tensor_parameter_name = m2::TensorParamName{"output"}, .accessor_name = "dst"}},
        .runtime_arg_schema = {.runtime_arg_names = {"num_pages", "start_id"}},
        .hw_config = ttnn::create_writer_datamovement_config(/*disable_dfb_implicit_sync_for_all=*/true),
    };
    const m2::KernelSpec compute{
        .unique_id = m2::KernelSpecName{"compute"},
        .source = std::string(kUnaryKernels) + "/compute/eltwise_sfpu_metal2.cpp",
        .compiler_options =
            {.defines = {{"SFPU_OP_EXP_INCLUDE", "1"}, {"SFPU_OP_CHAIN_0", "exp_tile_init(); exp_tile(0);"}}},
        .dfb_bindings = {m2::ConsumerOf(m2::DFBSpecName{"in"}, "in"), m2::ProducerOf(m2::DFBSpecName{"out"}, "out")},
        .runtime_arg_schema = {.runtime_arg_names = {"num_tiles"}},
        .hw_config = m2::ComputeHardwareConfig{},
    };
    const auto dfb = [&](const char* name) {
        return m2::DataflowBufferSpec{
            .unique_id = m2::DFBSpecName{name},
            .entry_size = tile_bytes,
            .num_entries = 2,
            .data_format_metadata = tt::DataFormat::Float16_b};
    };
    Program program{
        .spec =
            {.name = "exp",
             .kernels = {reader, writer, compute},
             .dataflow_buffers = {dfb("in"), dfb("out")},
             .tensor_parameters =
                 {{.unique_id = m2::TensorParamName{"input"}, .spec = input.tensor_spec()},
                  {.unique_id = m2::TensorParamName{"output"}, .spec = output.tensor_spec()}},
             .work_units =
                 {{.name = "exp",
                   .kernels = {reader.unique_id, writer.unique_id, compute.unique_id},
                   .target_nodes = cores}}},
    };

    m2::KernelRunArgs reader_args{.kernel = reader.unique_id};
    m2::KernelRunArgs writer_args{.kernel = writer.unique_id};
    m2::KernelRunArgs compute_args{.kernel = compute.unique_id};
    uint32_t start_id = 0;
    for (const auto& [group, tiles] : {std::pair{group_1, tiles_1}, std::pair{group_2, tiles_2}}) {
        for (const auto& core : corerange_to_cores(group)) {
            m2::AddRuntimeArgsForNode(
                reader_args.runtime_arg_values, core, {{"num_pages", tiles}, {"start_id", start_id}});
            m2::AddRuntimeArgsForNode(
                writer_args.runtime_arg_values, core, {{"num_pages", tiles}, {"start_id", start_id}});
            m2::AddRuntimeArgsForNode(compute_args.runtime_arg_values, core, {{"num_tiles", tiles}});
            start_id += tiles;
        }
    }
    program.run_args = {
        .kernel_run_args = {reader_args, writer_args, compute_args},
        .tensor_args =
            {{m2::TensorParamName{"input"}, input.mesh_tensor()},
             {m2::TensorParamName{"output"}, output.mesh_tensor()}},
    };
    return program;
}

std::pair<Tensor, Tensor> make_tensors(tt::tt_metal::distributed::MeshDevice* device, uint32_t num_tiles) {
    const ttnn::Shape shape{1, num_tiles, tt::constants::TILE_HEIGHT, tt::constants::TILE_WIDTH};
    Tensor input = ttnn::random::random(shape, DataType::BFLOAT16).to_layout(Layout::TILE).to_device(device);
    Tensor output = ttnn::create_device_tensor(input.tensor_spec(), device);
    return {input, output};
}

m2::ProgramSpec make_hash_spec() {
    const tt::tt_metal::TensorSpec tensor_spec(
        ttnn::Shape({1, 1, tt::constants::TILE_HEIGHT, tt::constants::TILE_WIDTH}),
        TensorLayout(DataType::BFLOAT16, PageConfig(Layout::TILE), MemoryConfig{}));
    return m2::ProgramSpec{
        .name = "hash",
        .kernels = {m2::KernelSpec{
            .unique_id = m2::KernelSpecName{"kernel"},
            .source = "kernel.cpp",
            .compile_time_args = {{"block", 1}},
            .runtime_arg_schema = {.runtime_arg_names = {"x"}, .common_runtime_arg_names = {"y"}},
            .hw_config = ttnn::create_reader_datamovement_config(),
        }},
        .tensor_parameters = {{.unique_id = m2::TensorParamName{"input"}, .spec = tensor_spec}},
        .work_units = {{.name = "main", .kernels = {m2::KernelSpecName{"kernel"}}, .target_nodes = CoreCoord{0, 0}}},
    };
}

}  // namespace

// The rule test_generic_op_hashing.cpp checks for the ProgramDescriptor form: argument names and compile-time values
// change the hash. Runtime values live in the ProgramRunArgs, which the hash does not see.
TEST(GenericOpProgramSpecHash, ArgumentNamesAndCompileTimeValuesChangeHash) {
    const auto base = ttnn::experimental::compute_program_spec_hash(make_hash_spec());

    auto spec = make_hash_spec();
    spec.kernels[0].runtime_arg_schema.runtime_arg_names = {"other"};
    EXPECT_NE(ttnn::experimental::compute_program_spec_hash(spec), base);

    spec = make_hash_spec();
    spec.kernels[0].runtime_arg_schema.common_runtime_arg_names = {"other"};
    EXPECT_NE(ttnn::experimental::compute_program_spec_hash(spec), base);

    spec = make_hash_spec();
    spec.kernels[0].compile_time_args = {{"block", 2}};
    EXPECT_NE(ttnn::experimental::compute_program_spec_hash(spec), base);

    spec = make_hash_spec();
    spec.kernels[0].compile_time_args = {{"other", 1}};
    EXPECT_NE(ttnn::experimental::compute_program_spec_hash(spec), base);
}

TEST(GenericOpProgramSpecHash, NamesForMessagesKeepHash) {
    const auto base = ttnn::experimental::compute_program_spec_hash(make_hash_spec());
    auto spec = make_hash_spec();
    spec.name = "renamed";
    spec.work_units[0].name = "renamed";
    EXPECT_EQ(ttnn::experimental::compute_program_spec_hash(spec), base);
}

TEST_F(TTNNFixtureWithDevice, TestGenericOpProgramSpecReusesTheProgramWithNewTensors) {
    auto [input, output] = make_tensors(this->device_, 8);
    const auto program = make_exp_program(input, output);
    ttnn::experimental::generic_op({input, output}, program.spec, program.run_args);
    const auto entries = this->device_->num_program_cache_entries();

    auto [input_2, output_2] = make_tensors(this->device_, 8);
    const auto program_2 = make_exp_program(input_2, output_2);
    ttnn::experimental::generic_op({input_2, output_2}, program_2.spec, program_2.run_args);
    EXPECT_EQ(this->device_->num_program_cache_entries(), entries);

    EXPECT_TRUE(ttnn::allclose<bfloat16>(ttnn::exp(input).cpu(), output.cpu()));
    EXPECT_TRUE(ttnn::allclose<bfloat16>(ttnn::exp(input_2).cpu(), output_2.cpu()));
}

TEST_F(TTNNFixtureWithDevice, TestGenericOpMeshProgramSpec) {
    auto [input, output] = make_tensors(this->device_, 4);
    const auto program = make_exp_program(input, output);
    const ttnn::MeshCoordinateRange whole_mesh(this->device_->shape());
    const ttnn::experimental::MeshProgramSpec mesh_spec{.programs = {{whole_mesh, program.spec}}};
    const ttnn::experimental::MeshProgramRunArgs mesh_run_args{.programs = {{whole_mesh, program.run_args}}};
    ttnn::experimental::generic_op({input, output}, mesh_spec, mesh_run_args);
    EXPECT_TRUE(ttnn::allclose<bfloat16>(ttnn::exp(input).cpu(), output.cpu()));

    const ttnn::experimental::MeshProgramSpec overlapping{
        .programs = {{whole_mesh, program.spec}, {whole_mesh, program.spec}}};
    const ttnn::experimental::MeshProgramRunArgs overlapping_run_args{
        .programs = {{whole_mesh, program.run_args}, {whole_mesh, program.run_args}}};
    EXPECT_THAT(
        [&] { ttnn::experimental::generic_op({input, output}, overlapping, overlapping_run_args); },
        ThrowsMessage<std::exception>(HasSubstr("overlap")));
}

TEST_F(TTNNFixtureWithDevice, TestGenericOpProgramSpecPreparationFillsTheProgramCache) {
    auto [input, output] = make_tensors(this->device_, 4);
    const auto program = make_exp_program(input, output);
    const auto entries = this->device_->num_program_cache_entries();
    ttnn::experimental::prepare_generic_op({input, output}, program.spec, program.run_args);
    EXPECT_EQ(this->device_->num_program_cache_entries(), entries + 1);
    ttnn::experimental::generic_op({input, output}, program.spec, program.run_args);
    EXPECT_EQ(this->device_->num_program_cache_entries(), entries + 1);
    EXPECT_TRUE(ttnn::allclose<bfloat16>(ttnn::exp(input).cpu(), output.cpu()));
}

TEST_F(TTNNFixtureWithDevice, TestGenericOpProgramSpecLaterCallRejectsAMissingArgument) {
    auto [input, output] = make_tensors(this->device_, 4);
    auto program = make_exp_program(input, output);
    ttnn::experimental::generic_op({input, output}, program.spec, program.run_args);

    program.run_args.kernel_run_args[0].runtime_arg_values.erase("start_id");
    EXPECT_THAT(
        [&] { ttnn::experimental::generic_op({input, output}, program.spec, program.run_args); },
        ThrowsMessage<std::exception>(
            HasSubstr("kernel 'reader' declares the runtime arguments [num_pages, start_id], but its run args give "
                      "[num_pages]")));
}

TEST_F(TTNNFixtureWithDevice, TestGenericOpProgramSpecCompilationFailureLeavesTheProgramCacheUnchanged) {
    auto [input, output] = make_tensors(this->device_, 4);
    auto program = make_exp_program(input, output);
    program.spec.kernels[2].source = m2::KernelSpec::SourceCode{.code = "this does not compile"};
    const auto entries = this->device_->num_program_cache_entries();
    EXPECT_ANY_THROW(ttnn::experimental::generic_op({input, output}, program.spec, program.run_args));
    EXPECT_EQ(this->device_->num_program_cache_entries(), entries);
}

}  // namespace ttnn::operations::generic::test
