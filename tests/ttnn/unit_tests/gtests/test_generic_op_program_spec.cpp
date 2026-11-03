// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// ttnn::experimental::generic_op on a Metal 2.0 ProgramSpec and ProgramRunArgs, through the C++ API.
// tests/ttnn/unit_tests/operations/debug/test_generic_op_program_spec.py is the main suite.

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <cstdint>
#include <cstdlib>
#include <exception>
#include <optional>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include <tt-metalium/constants.hpp>
#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_run_args.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_spec.hpp>
#include <tt-metalium/experimental/per_core_allocation/memory_config.hpp>
#include <tt-metalium/experimental/prefetcher_pipe.hpp>
#include <tt-metalium/work_split.hpp>
#include <tt_stl/cleanup.hpp>

#include "impl/context/metal_context.hpp"
#include "tt_metal/tt_metal/common/multi_device_fixture.hpp"
#include "ttnn/config.hpp"
#include "ttnn/distributed/distributed_tensor.hpp"
#include "ttnn/operations/core/data_movement_kernel/datamovement_kernel_config.hpp"
#include "ttnn/operations/creation/creation.hpp"
#include "ttnn/operations/eltwise/binary/binary.hpp"
#include "ttnn/operations/eltwise/unary/unary.hpp"
#include "ttnn/operations/functions.hpp"
#include "ttnn/operations/generic/generic_op.hpp"
#include "ttnn/operations/matmul/matmul.hpp"
#include "ttnn/tensor/tensor.hpp"
#include "ttnn_test_fixtures.hpp"

namespace ttnn::operations::generic::test {
namespace {

namespace m2 = tt::tt_metal::experimental;
using ::testing::HasSubstr;
using ::testing::ThrowsMessage;
using tt::tt_metal::MeshDevice1x2Fixture;

constexpr const char* kUnaryKernels = "ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels";
constexpr const char* kMatmulKernels = "ttnn/cpp/ttnn/operations/matmul/device/kernels";
constexpr const char* kBinaryKernels = "tests/ttnn/unit_tests/operations/debug/kernels/program_spec_mul";
constexpr const char* kSemaphoreKernels = "tests/tt_metal/tt_metal/test_kernels/dataflow";

struct Program {
    m2::ProgramSpec spec;
    m2::ProgramRunArgs run_args;
};

// output = exp(input) with the tiles split over the cores, as in the design doc's example. With aliased_buffer, a
// second DFB, in_view, shares in's memory; the kernels never use it.
Program make_exp_program(const Tensor& input, const Tensor& output, bool aliased_buffer = false) {
    const uint32_t num_tiles = input.physical_volume() / tt::constants::TILE_HW;
    const auto [num_cores, cores, group_1, group_2, tiles_1, tiles_2] =
        tt::tt_metal::split_work_to_cores(input.device()->compute_with_storage_grid_size(), num_tiles);
    const uint32_t tile_bytes = tt::tile_size(tt::DataFormat::Float16_b);

    m2::KernelSpec reader{
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
    m2::KernelSpec compute{
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
    std::vector<m2::DataflowBufferSpec> dfbs = {dfb("in"), dfb("out")};
    if (aliased_buffer) {
        reader.dfb_bindings.push_back(m2::ProducerOf(m2::DFBSpecName{"in_view"}, "in_view"));
        compute.dfb_bindings.push_back(m2::ConsumerOf(m2::DFBSpecName{"in_view"}, "in_view"));
        dfbs[0].advanced_options.alias_with = {m2::DFBSpecName{"in_view"}};
        dfbs.push_back(dfb("in_view"));
        dfbs.back().advanced_options.alias_with = {m2::DFBSpecName{"in"}};
    }
    Program program{
        .spec =
            {.name = "exp",
             .kernels = {reader, writer, compute},
             .dataflow_buffers = dfbs,
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

// a, b and out for out = a + b, with num_tiles tiles each.
std::tuple<Tensor, Tensor, Tensor> make_add_tensors(tt::tt_metal::distributed::MeshDevice* device, uint32_t num_tiles) {
    const ttnn::Shape shape{1, num_tiles, tt::constants::TILE_HEIGHT, tt::constants::TILE_WIDTH};
    Tensor a = ttnn::random::random(shape, DataType::BFLOAT16).to_layout(Layout::TILE).to_device(device);
    Tensor b = ttnn::random::random(shape, DataType::BFLOAT16).to_layout(Layout::TILE).to_device(device);
    Tensor out = ttnn::create_device_tensor(a.tensor_spec(), device);
    return {a, b, out};
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

// Sets validate_program_args until the returned object goes out of scope.
[[nodiscard]] auto set_validate_program_args(bool value) {
    const bool previous = ttnn::CONFIG.get<"validate_program_args">();
    ttnn::CONFIG.set<"validate_program_args">(value);
    return ttsl::make_cleanup([previous] { ttnn::CONFIG.set<"validate_program_args">(previous); });
}

// Drops the first core from the compute kernel's num_tiles.
void leave_out_a_core(m2::ProgramRunArgs& run_args) {
    auto& num_tiles = run_args.kernel_run_args[2].runtime_arg_values["num_tiles"];
    const auto node = num_tiles.begin()->first;
    num_tiles.erase(node);
}

// out = a + b with the tiles split over the cores, with the kernels that test_generic_op_program_spec.py runs for
// a * b. With varargs, which only C++ can set, the reader and writer take each core's tile count and first tile as
// runtime varargs 0 and 1 instead of named arguments.
Program make_add_program(const Tensor& a, const Tensor& b, const Tensor& out, bool varargs = false) {
    const uint32_t num_tiles = a.physical_volume() / tt::constants::TILE_HW;
    const auto [num_cores, cores, group_1, group_2, tiles_1, tiles_2] =
        tt::tt_metal::split_work_to_cores(a.device()->compute_with_storage_grid_size(), num_tiles);
    const uint32_t tile_bytes = tt::tile_size(tt::DataFormat::Float16_b);
    const m2::DFBSpecName in_a{"in_a"};
    const m2::DFBSpecName in_b{"in_b"};
    const m2::DFBSpecName dfb_out{"out"};

    m2::KernelSpec reader{
        .unique_id = m2::KernelSpecName{"reader"},
        .source = std::string(kBinaryKernels) + "/reader.cpp",
        .dfb_bindings = {m2::ProducerOf(in_a, "in_a"), m2::ProducerOf(in_b, "in_b")},
        .tensor_bindings =
            {{.tensor_parameter_name = m2::TensorParamName{"a"}, .accessor_name = "a"},
             {.tensor_parameter_name = m2::TensorParamName{"b"}, .accessor_name = "b"}},
        .runtime_arg_schema = {.runtime_arg_names = {"num_tiles", "start_id"}},
        .hw_config = ttnn::create_reader_datamovement_config(),
    };
    const m2::KernelSpec compute{
        .unique_id = m2::KernelSpecName{"compute"},
        .source = std::string(kBinaryKernels) + "/compute.cpp",
        .compiler_options = {.defines = {{"BINARY_ADD", "1"}}},
        .dfb_bindings = {m2::ConsumerOf(in_a, "in_a"), m2::ConsumerOf(in_b, "in_b"), m2::ProducerOf(dfb_out, "out")},
        .runtime_arg_schema = {.runtime_arg_names = {"num_tiles"}},
        .hw_config = m2::ComputeHardwareConfig{},
    };
    m2::KernelSpec writer{
        .unique_id = m2::KernelSpecName{"writer"},
        .source = std::string(kBinaryKernels) + "/writer.cpp",
        .dfb_bindings = {m2::ConsumerOf(dfb_out, "out")},
        .tensor_bindings = {{.tensor_parameter_name = m2::TensorParamName{"out"}, .accessor_name = "out"}},
        .runtime_arg_schema = {.runtime_arg_names = {"num_tiles", "start_id"}},
        .hw_config = ttnn::create_writer_datamovement_config(),
    };
    if (varargs) {
        for (auto* kernel : {&reader, &writer}) {
            kernel->compiler_options.defines["USE_VARARGS"] = "1";
            kernel->runtime_arg_schema = {};
            kernel->advanced_options.num_runtime_varargs = 2;
        }
    }
    const auto dfb = [&](const m2::DFBSpecName& name) {
        return m2::DataflowBufferSpec{
            .unique_id = name,
            .entry_size = tile_bytes,
            .num_entries = 2,
            .data_format_metadata = tt::DataFormat::Float16_b};
    };
    Program program{
        .spec =
            {.name = "add",
             .kernels = {reader, compute, writer},
             .dataflow_buffers = {dfb(in_a), dfb(in_b), dfb(dfb_out)},
             .tensor_parameters =
                 {{.unique_id = m2::TensorParamName{"a"}, .spec = a.tensor_spec()},
                  {.unique_id = m2::TensorParamName{"b"}, .spec = b.tensor_spec()},
                  {.unique_id = m2::TensorParamName{"out"}, .spec = out.tensor_spec()}},
             .work_units =
                 {{.name = "add",
                   .kernels = {reader.unique_id, compute.unique_id, writer.unique_id},
                   .target_nodes = cores}}},
    };

    m2::KernelRunArgs reader_args{.kernel = reader.unique_id};
    m2::KernelRunArgs compute_args{.kernel = compute.unique_id};
    m2::KernelRunArgs writer_args{.kernel = writer.unique_id};
    uint32_t start_id = 0;
    for (const auto& [group, tiles] : {std::pair{group_1, tiles_1}, std::pair{group_2, tiles_2}}) {
        for (const auto& core : corerange_to_cores(group)) {
            if (varargs) {
                reader_args.advanced_options.runtime_varargs[core] = {tiles, start_id};
                writer_args.advanced_options.runtime_varargs[core] = {tiles, start_id};
            } else {
                m2::AddRuntimeArgsForNode(
                    reader_args.runtime_arg_values, core, {{"num_tiles", tiles}, {"start_id", start_id}});
                m2::AddRuntimeArgsForNode(
                    writer_args.runtime_arg_values, core, {{"num_tiles", tiles}, {"start_id", start_id}});
            }
            m2::AddRuntimeArgsForNode(compute_args.runtime_arg_values, core, {{"num_tiles", tiles}});
            start_id += tiles;
        }
    }
    program.run_args = {
        .kernel_run_args = {reader_args, compute_args, writer_args},
        .tensor_args =
            {{m2::TensorParamName{"a"}, a.mesh_tensor()},
             {m2::TensorParamName{"b"}, b.mesh_tensor()},
             {m2::TensorParamName{"out"}, out.mesh_tensor()}},
    };
    return program;
}

// Per-core allocation needs the HYBRID allocator, which Metal reads from the environment once per process, before the
// fixture runs. So these tests run only when the binary runs with TT_METAL_ALLOCATOR_MODE_HYBRID=1.
class GenericOpProgramSpecHybridAllocatorTest : public TTNNFixtureWithDevice {
protected:
    void SetUp() override {
        if (!tt::tt_metal::MetalContext::instance().rtoptions().get_allocator_mode_hybrid()) {
            GTEST_SKIP() << "Needs the HYBRID allocator: run with TT_METAL_ALLOCATOR_MODE_HYBRID=1";
        }
        TTNNFixtureWithDevice::SetUp();
    }
};

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

// Varargs are C++ only, so the Python hash tests do not cover their counts.
TEST(GenericOpProgramSpecHash, VarargCountsChangeHash) {
    const auto base = ttnn::experimental::compute_program_spec_hash(make_hash_spec());

    auto spec = make_hash_spec();
    spec.kernels[0].advanced_options.num_runtime_varargs = 1;
    EXPECT_NE(ttnn::experimental::compute_program_spec_hash(spec), base);

    spec = make_hash_spec();
    spec.kernels[0].advanced_options.num_common_runtime_varargs = 1;
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

    // The ProgramSpec form runs as the mesh form does on the tensors' devices, so it reuses the mesh form's program.
    const auto entries = this->device_->num_program_cache_entries();
    ttnn::experimental::generic_op({input, output}, program.spec, program.run_args);
    EXPECT_EQ(this->device_->num_program_cache_entries(), entries);

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

TEST_F(TTNNFixtureWithDevice, TestGenericOpMeshProgramRunArgsNeedTheSpecsRanges) {
    auto [input, output] = make_tensors(this->device_, 4);
    const auto program = make_exp_program(input, output);
    const ttnn::MeshCoordinateRange whole_mesh(this->device_->shape());
    const ttnn::MeshCoordinateRange other_range(ttnn::MeshCoordinate(5, 5), ttnn::MeshCoordinate(5, 5));
    const ttnn::experimental::MeshProgramSpec mesh_spec{.programs = {{whole_mesh, program.spec}}};

    const ttnn::experimental::MeshProgramRunArgs other_range_run_args{.programs = {{other_range, program.run_args}}};
    EXPECT_THAT(
        [&] { ttnn::experimental::generic_op({input, output}, mesh_spec, other_range_run_args); },
        ThrowsMessage<std::exception>(HasSubstr("the MeshProgramRunArgs has no run args for mesh range")));

    const ttnn::experimental::MeshProgramRunArgs two_ranges_run_args{
        .programs = {{whole_mesh, program.run_args}, {other_range, program.run_args}}};
    EXPECT_THAT(
        [&] { ttnn::experimental::generic_op({input, output}, mesh_spec, two_ranges_run_args); },
        ThrowsMessage<std::exception>(
            HasSubstr("the MeshProgramSpec has 1 programs, but the MeshProgramRunArgs has run args for 2")));
}

// Every call checks its run args and tensors against the spec, so these later calls run without
// validate_program_args, which Metal needs to check them itself.
TEST_F(TTNNFixtureWithDevice, TestGenericOpProgramSpecLaterCallRejectsMismatchedArgumentsAndTensors) {
    auto [input, output] = make_tensors(this->device_, 4);
    const auto program = make_exp_program(input, output);
    ttnn::experimental::generic_op({input, output}, program.spec, program.run_args);
    const auto restore = set_validate_program_args(false);

    auto renamed_argument = program.run_args;
    auto& reader_values = renamed_argument.kernel_run_args[0].runtime_arg_values;
    const auto start_id = reader_values["start_id"];
    reader_values.erase("start_id");
    reader_values["first_page"] = start_id;
    EXPECT_THAT(
        [&] { ttnn::experimental::generic_op({input, output}, program.spec, renamed_argument); },
        ThrowsMessage<std::exception>(HasSubstr(
            "kernel 'reader' declares the runtime arguments [num_pages, start_id], but its run args give [num_pages, "
            "first_page]")));

    auto missing_tensor = program.run_args;
    missing_tensor.tensor_args.erase(m2::TensorParamName{"output"});
    EXPECT_THAT(
        [&] { ttnn::experimental::generic_op({input, output}, program.spec, missing_tensor); },
        ThrowsMessage<std::exception>(HasSubstr("tensor parameter 'output' has no tensor in run_args.tensor_args")));

    auto [larger_input, larger_output] = make_tensors(this->device_, 8);
    auto larger_tensors = program.run_args;
    larger_tensors.tensor_args = {
        {m2::TensorParamName{"input"}, larger_input.mesh_tensor()},
        {m2::TensorParamName{"output"}, larger_output.mesh_tensor()}};
    EXPECT_THAT(
        [&] { ttnn::experimental::generic_op({larger_input, larger_output}, program.spec, larger_tensors); },
        ThrowsMessage<std::exception>(HasSubstr("which does not match the parameter's spec")));
}

// A later call that leaves out a core is rejected with validate_program_args on.
TEST_F(TTNNFixtureWithDevice, TestGenericOpProgramSpecLeftOutCoreWithValidationOn) {
    auto [input, output] = make_tensors(this->device_, 8);
    auto program = make_exp_program(input, output);
    ttnn::experimental::generic_op({input, output}, program.spec, program.run_args);

    const auto restore = set_validate_program_args(true);
    leave_out_a_core(program.run_args);
    EXPECT_THAT(
        [&] { ttnn::experimental::generic_op({input, output}, program.spec, program.run_args); },
        ThrowsMessage<std::exception>(HasSubstr("runtime argument 'num_tiles' does not have exactly one value")));
}

// With validate_program_args off, the left-out core keeps the values of the call before.
TEST_F(TTNNFixtureWithDevice, TestGenericOpProgramSpecLeftOutCoreWithValidationOff) {
    auto [input, output] = make_tensors(this->device_, 8);
    auto program = make_exp_program(input, output);
    ttnn::experimental::generic_op({input, output}, program.spec, program.run_args);

    const auto restore = set_validate_program_args(false);
    leave_out_a_core(program.run_args);
    ttnn::experimental::generic_op({input, output}, program.spec, program.run_args);
    EXPECT_TRUE(ttnn::allclose<bfloat16>(ttnn::exp(input).cpu(), output.cpu()));
}

// A DFB size override stays in effect for later calls until a call changes it. DFBs that share memory keep one total
// size, so Metal rejects a call that resizes only one of them, unless it keeps the size the other already has.
TEST_F(TTNNFixtureWithDevice, TestGenericOpProgramSpecSizeOverrideCarriesOverToLaterCalls) {
    const auto restore = set_validate_program_args(true);
    auto [input, output] = make_tensors(this->device_, 4);
    auto program = make_exp_program(input, output, /*aliased_buffer=*/true);
    const auto resize = [](const char* dfb, uint32_t num_entries) {
        return m2::ProgramRunArgs::DFBRunOverrides{.dfb = m2::DFBSpecName{dfb}, .num_entries = num_entries};
    };
    program.run_args.dfb_run_overrides = {resize("in", 4), resize("in_view", 4)};
    ttnn::experimental::generic_op({input, output}, program.spec, program.run_args);
    program.run_args.dfb_run_overrides = {};
    ttnn::experimental::generic_op({input, output}, program.spec, program.run_args);
    EXPECT_TRUE(ttnn::allclose<bfloat16>(ttnn::exp(input).cpu(), output.cpu()));

    // The sizes stayed: in alone can take the size it already has, but not its original size, which in_view no longer
    // has.
    program.run_args.dfb_run_overrides = {resize("in", 4)};
    ttnn::experimental::generic_op({input, output}, program.spec, program.run_args);
    program.run_args.dfb_run_overrides = {resize("in", 2)};
    EXPECT_THAT(
        [&] { ttnn::experimental::generic_op({input, output}, program.spec, program.run_args); },
        ThrowsMessage<std::exception>(HasSubstr("was not overridden in alias group")));
}

// The producer raises the semaphore by 1 and the consumer waits until it equals 1. If a call started from the value
// the previous one left, the third call would start at 2 and never finish.
TEST_F(TTNNFixtureWithDevice, TestGenericOpProgramSpecSemaphoresStartAtZeroOnEveryCall) {
    const Tensor output = ttnn::create_device_tensor(
        tt::tt_metal::TensorSpec(
            ttnn::Shape({tt::constants::TILE_HEIGHT, tt::constants::TILE_WIDTH}),
            TensorLayout(DataType::BFLOAT16, PageConfig(Layout::TILE), MemoryConfig{})),
        this->device_);
    const CoreCoord node{0, 0};
    const m2::SemaphoreSpecName semaphore{"sem"};
    const m2::ProgramSpec spec{
        .name = "semaphore_loopback",
        .kernels =
            {m2::KernelSpec{
                 .unique_id = m2::KernelSpecName{"producer"},
                 .source = std::string(kSemaphoreKernels) + "/semaphore_accessor_loopback_producer.cpp",
                 .semaphore_bindings = {{.semaphore_spec_name = semaphore, .accessor_name = "signal"}},
                 .hw_config = ttnn::create_writer_datamovement_config(),
             },
             m2::KernelSpec{
                 .unique_id = m2::KernelSpecName{"consumer"},
                 .source = std::string(kSemaphoreKernels) + "/semaphore_accessor_loopback_consumer.cpp",
                 .semaphore_bindings = {{.semaphore_spec_name = semaphore, .accessor_name = "waiter"}},
                 .hw_config = ttnn::create_reader_datamovement_config(),
             }},
        .semaphores = {{.unique_id = semaphore, .target_nodes = node}},
        .work_units =
            {{.name = "loopback",
              .kernels = {m2::KernelSpecName{"producer"}, m2::KernelSpecName{"consumer"}},
              .target_nodes = node}},
    };
    const auto entries = this->device_->num_program_cache_entries();
    for (int call = 0; call < 3; ++call) {
        ttnn::experimental::generic_op({output}, spec, m2::ProgramRunArgs{});
    }
    // The calls only enqueue the program: wait for them, so that one that never finishes fails this test.
    tt::tt_metal::distributed::Synchronize(this->device_, std::nullopt);
    EXPECT_EQ(this->device_->num_program_cache_entries(), entries + 1);
}

// Both DFBs sit on the tensors' own L1 shards, as in ttnn's sharded typecast. A later call with new tensors, made
// while the first ones still exist so that their shards are elsewhere, moves the DFBs to the new tensors' shards.
TEST_F(TTNNFixtureWithDevice, TestGenericOpProgramSpecShardedTypecastOnBorrowedBuffers) {
    if (this->device_->arch() == tt::ARCH::QUASAR && std::getenv("TT_METAL_SIMULATOR") != nullptr) {
        GTEST_SKIP() << "ttsim's Quasar model cannot pack bfloat16 into float32 yet";
    }
    const ttnn::Shape shape{1, 1, 2 * tt::constants::TILE_HEIGHT, 2 * tt::constants::TILE_WIDTH};
    const CoreRangeSet cores(CoreRange(CoreCoord{0, 0}, CoreCoord{1, 0}));
    const uint32_t tiles_per_core = 2;
    const MemoryConfig memory_config{
        TensorMemoryLayout::HEIGHT_SHARDED,
        BufferType::L1,
        tt::tt_metal::ShardSpec(
            cores, {tt::constants::TILE_HEIGHT, 2 * tt::constants::TILE_WIDTH}, ShardOrientation::ROW_MAJOR)};
    const auto make_sharded_tensors = [&] {
        Tensor input = ttnn::random::random(shape, DataType::BFLOAT16)
                           .to_layout(Layout::TILE)
                           .to_device(this->device_, memory_config);
        // The output starts as a known value, so a program that wrote to another tensor's shard would leave it.
        Tensor output = ttnn::full(shape, 7.0f, DataType::FLOAT32, Layout::TILE, *this->device_, memory_config);
        return std::pair{input, output};
    };
    const auto expect_typecast = [](const Tensor& input, const Tensor& output) {
        const auto inputs = input.cpu().to_vector<bfloat16>();
        const auto outputs = output.cpu().to_vector<float>();
        ASSERT_EQ(inputs.size(), outputs.size());
        for (std::size_t i = 0; i < inputs.size(); ++i) {
            ASSERT_EQ(static_cast<float>(inputs[i]), outputs[i]) << "element " << i;
        }
    };

    auto [input, output] = make_sharded_tensors();
    const m2::TensorParamName input_name{"input"};
    const m2::TensorParamName output_name{"output"};
    const m2::ProgramSpec spec{
        .name = "typecast_sharded",
        .kernels =
            {m2::KernelSpec{
                 .unique_id = m2::KernelSpecName{"reader"},
                 .source = std::string(kUnaryKernels) + "/dataflow/reader_unary_sharded_metal2.cpp",
                 .dfb_bindings = {m2::ProducerOf(m2::DFBSpecName{"in"}, "in")},
                 .runtime_arg_schema = {.runtime_arg_names = {"num_tiles_per_core"}},
                 .hw_config = ttnn::create_reader_datamovement_config(),
             },
             m2::KernelSpec{
                 .unique_id = m2::KernelSpecName{"compute"},
                 .source = "ttnn/cpp/ttnn/operations/copy/typecast/device/kernels/compute/eltwise_typecast.cpp",
                 // No writer drains the output DFB, so the compute kernel both produces and consumes it.
                 .dfb_bindings =
                     {m2::ConsumerOf(m2::DFBSpecName{"in"}, "in"),
                      m2::ProducerOf(m2::DFBSpecName{"out"}, "out"),
                      m2::ConsumerOf(m2::DFBSpecName{"out"}, "out")},
                 .compile_time_args =
                     {{"per_core_block_cnt", 1},
                      {"per_core_block_dim", tiles_per_core},
                      {"in_data_format", static_cast<uint32_t>(tt::DataFormat::Float16_b)},
                      {"out_data_format", static_cast<uint32_t>(tt::DataFormat::Float32)}},
                 .hw_config = m2::ComputeHardwareConfig{},
             }},
        .dataflow_buffers =
            {{.unique_id = m2::DFBSpecName{"in"},
              .entry_size = tt::tile_size(tt::DataFormat::Float16_b),
              .num_entries = tiles_per_core,
              .data_format_metadata = tt::DataFormat::Float16_b,
              .borrowed_from = input_name},
             {.unique_id = m2::DFBSpecName{"out"},
              .entry_size = tt::tile_size(tt::DataFormat::Float32),
              .num_entries = tiles_per_core,
              .data_format_metadata = tt::DataFormat::Float32,
              .borrowed_from = output_name}},
        .tensor_parameters =
            {{.unique_id = input_name, .spec = input.tensor_spec()},
             {.unique_id = output_name, .spec = output.tensor_spec()}},
        .work_units =
            {{.name = "typecast",
              .kernels = {m2::KernelSpecName{"reader"}, m2::KernelSpecName{"compute"}},
              .target_nodes = cores}},
    };
    m2::KernelRunArgs reader_args{.kernel = m2::KernelSpecName{"reader"}};
    for (const auto& core : corerange_to_cores(cores)) {
        m2::AddRuntimeArgsForNode(reader_args.runtime_arg_values, core, {{"num_tiles_per_core", tiles_per_core}});
    }
    m2::ProgramRunArgs run_args{
        .kernel_run_args = {reader_args},
        .tensor_args = {{input_name, input.mesh_tensor()}, {output_name, output.mesh_tensor()}},
    };
    ttnn::experimental::generic_op({input, output}, spec, run_args);
    expect_typecast(input, output);

    auto [input_2, output_2] = make_sharded_tensors();
    run_args.tensor_args = {{input_name, input_2.mesh_tensor()}, {output_name, output_2.mesh_tensor()}};
    const auto entries = this->device_->num_program_cache_entries();
    ttnn::experimental::generic_op({input_2, output_2}, spec, run_args);
    EXPECT_EQ(this->device_->num_program_cache_entries(), entries);
    expect_typecast(input_2, output_2);
}

// The binary add case of test_generic_op.cpp: out = a + b on [11, 9, 32, 32] over the full grid.
TEST_F(TTNNFixtureWithDevice, TestGenericOpProgramSpecBinaryAdd) {
    const ttnn::Shape shape{11, 9, tt::constants::TILE_HEIGHT, tt::constants::TILE_WIDTH};
    const auto make_input = [&] {
        return ttnn::random::random(shape).to_layout(Layout::TILE).to_device(this->device_);
    };
    const Tensor a = make_input();
    const Tensor b = make_input();
    const Tensor out = ttnn::create_device_tensor(a.tensor_spec(), this->device_);
    const auto program = make_add_program(a, b, out);
    ttnn::experimental::generic_op({a, b, out}, program.spec, program.run_args);
    EXPECT_TRUE(ttnn::allclose<bfloat16>(ttnn::add(a, b).cpu(), out.cpu()));
}

// Varargs are C++ only. A later call writes its vararg values into the program the first call compiled.
TEST_F(TTNNFixtureWithDevice, TestGenericOpProgramSpecVarargs) {
    const auto [a, b, out] = make_add_tensors(this->device_, 4);
    const auto program = make_add_program(a, b, out, /*varargs=*/true);
    ttnn::experimental::generic_op({a, b, out}, program.spec, program.run_args);
    EXPECT_TRUE(ttnn::allclose<bfloat16>(ttnn::add(a, b).cpu(), out.cpu()));

    // Zero inputs and no tiles for any core: the output keeps what the first call wrote only if the new tile counts
    // reached the kernels.
    const Tensor zeros_a = ttnn::zeros(a.logical_shape(), DataType::BFLOAT16, Layout::TILE, *this->device_);
    const Tensor zeros_b = ttnn::zeros(a.logical_shape(), DataType::BFLOAT16, Layout::TILE, *this->device_);
    auto no_tiles = make_add_program(zeros_a, zeros_b, out, /*varargs=*/true).run_args;
    auto& kernel_run_args = no_tiles.kernel_run_args;  // reader, compute, writer
    for (auto* varargs :
         {&kernel_run_args[0].advanced_options.runtime_varargs, &kernel_run_args[2].advanced_options.runtime_varargs}) {
        for (auto& [node, values] : *varargs) {
            values[0] = 0;
        }
    }
    for (auto& [node, num_tiles] : kernel_run_args[1].runtime_arg_values["num_tiles"]) {
        num_tiles = 0;
    }
    const auto entries = this->device_->num_program_cache_entries();
    ttnn::experimental::generic_op({zeros_a, zeros_b, out}, program.spec, no_tiles);
    EXPECT_EQ(this->device_->num_program_cache_entries(), entries);
    EXPECT_TRUE(ttnn::allclose<bfloat16>(ttnn::add(a, b).cpu(), out.cpu()));
}

// Metal writes a later call's varargs by position without checking their count, so generic_op checks the counts on
// every call, and on a first call also that every core gets its varargs.
TEST_F(TTNNFixtureWithDevice, TestGenericOpProgramSpecChecksVarargCountsOnEveryCall) {
    const auto [a, b, out] = make_add_tensors(this->device_, 4);
    const auto program = make_add_program(a, b, out, /*varargs=*/true);

    auto left_out_core = program.run_args;
    auto& reader_varargs = left_out_core.kernel_run_args[0].advanced_options.runtime_varargs;
    const auto node = reader_varargs.begin()->first;
    reader_varargs.erase(node);
    EXPECT_THAT(
        [&] { ttnn::experimental::generic_op({a, b, out}, program.spec, left_out_core); },
        ThrowsMessage<std::exception>(
            HasSubstr("kernel 'reader' declares runtime varargs, but its run args give none for node")));

    ttnn::experimental::generic_op({a, b, out}, program.spec, program.run_args);
    const auto restore = set_validate_program_args(false);

    auto more_varargs = program.run_args;
    more_varargs.kernel_run_args[0].advanced_options.runtime_varargs.begin()->second.push_back(0);
    EXPECT_THAT(
        [&] { ttnn::experimental::generic_op({a, b, out}, program.spec, more_varargs); },
        ThrowsMessage<std::exception>(HasSubstr("kernel 'reader' declares 2 runtime varargs on node")));

    auto common_varargs = program.run_args;
    common_varargs.kernel_run_args[0].advanced_options.common_runtime_varargs = {1};
    EXPECT_THAT(
        [&] { ttnn::experimental::generic_op({a, b, out}, program.spec, common_varargs); },
        ThrowsMessage<std::exception>(
            HasSubstr("kernel 'reader' declares 0 common runtime varargs, but its run args give 1")));
}

// Metal writes a later call's varargs by position into argument memory sized for the counts it compiled, so a program
// compiled for two varargs per core must not run with three: under the same custom hash, the second call compiles a
// program of its own.
TEST_F(TTNNFixtureWithDevice, TestGenericOpProgramSpecCustomHashStillKeysOnVarargCounts) {
    const auto [a, b, out] = make_add_tensors(this->device_, 4);
    const auto program = make_add_program(a, b, out, /*varargs=*/true);
    const auto entries = this->device_->num_program_cache_entries();
    ttnn::experimental::generic_op({a, b, out}, program.spec, program.run_args, /*custom_program_hash=*/777);

    auto three_varargs = program;
    for (auto& kernel : three_varargs.spec.kernels) {
        if (kernel.advanced_options.num_runtime_varargs == 2) {
            kernel.advanced_options.num_runtime_varargs = 3;
        }
    }
    for (auto& kernel_args : three_varargs.run_args.kernel_run_args) {
        for (auto& [node, values] : kernel_args.advanced_options.runtime_varargs) {
            values.push_back(0);
        }
    }
    ttnn::experimental::generic_op(
        {a, b, out}, three_varargs.spec, three_varargs.run_args, /*custom_program_hash=*/777);
    EXPECT_EQ(this->device_->num_program_cache_entries(), entries + 2);
    EXPECT_TRUE(ttnn::allclose<bfloat16>(ttnn::add(a, b).cpu(), out.cpu()));
}

// The matmul case of test_generic_op.cpp, with the Metal 2.0 kernels of ttnn's multi-core matmul: [3, 1, 320, 64] times
// [3, 1, 64, 128], one output tile after another, split over the full grid in core groups that differ in tile count.
TEST_F(TTNNFixtureWithDevice, TestGenericOpProgramSpecMatmul) {
    constexpr uint32_t batch = 3;
    constexpr uint32_t Mt = 10;
    constexpr uint32_t Kt = 2;
    constexpr uint32_t Nt = 4;
    const auto make_input = [&](uint32_t height_tiles, uint32_t width_tiles) {
        const ttnn::Shape shape{
            batch, 1, height_tiles * tt::constants::TILE_HEIGHT, width_tiles * tt::constants::TILE_WIDTH};
        return ttnn::random::random(shape).to_layout(Layout::TILE).to_device(this->device_);
    };
    const Tensor a = make_input(Mt, Kt);
    const Tensor b = make_input(Kt, Nt);
    const Tensor output = ttnn::create_device_tensor(
        tt::tt_metal::TensorSpec(
            ttnn::Shape{batch, 1, Mt * tt::constants::TILE_HEIGHT, Nt * tt::constants::TILE_WIDTH},
            TensorLayout(DataType::BFLOAT16, PageConfig(Layout::TILE), MemoryConfig{})),
        this->device_);

    const auto [num_cores, all_cores, group_1, group_2, tiles_1, tiles_2] =
        tt::tt_metal::split_work_to_cores(this->device_->compute_with_storage_grid_size(), batch * Mt * Nt);
    const uint32_t tile_bytes = tt::tile_size(tt::DataFormat::Float16_b);
    const m2::DFBSpecName in0{"in0"};
    const m2::DFBSpecName in1{"in1"};
    const m2::DFBSpecName dfb_out{"out"};
    const auto dfb = [&](const m2::DFBSpecName& name) {
        return m2::DataflowBufferSpec{
            .unique_id = name,
            .entry_size = tile_bytes,
            .num_entries = 2,
            .data_format_metadata = tt::DataFormat::Float16_b};
    };
    const m2::KernelSpec reader{
        .unique_id = m2::KernelSpecName{"reader"},
        .source = std::string(kMatmulKernels) + "/dataflow/reader_bmm_8bank_output_tiles_partitioned_metal2.cpp",
        .dfb_bindings = {m2::ProducerOf(in0, "in0"), m2::ProducerOf(in1, "in1")},
        .tensor_bindings =
            {{.tensor_parameter_name = m2::TensorParamName{"in0"}, .accessor_name = "in0"},
             {.tensor_parameter_name = m2::TensorParamName{"in1"}, .accessor_name = "in1"}},
        .compile_time_args = {{"in0_last_ktile_w", 0}, {"in0_last_ktile_h", 0}},
        .runtime_arg_schema =
            {.runtime_arg_names = {"output_tile_start_id", "num_output_tiles"},
             .common_runtime_arg_names = {"Mt", "Kt", "Nt", "MtKt", "KtNt", "batch", "bcast_B", "MtNt"}},
        .hw_config = ttnn::create_reader_datamovement_config(/*disable_dfb_implicit_sync_for_all=*/true),
    };
    const m2::KernelSpec writer{
        .unique_id = m2::KernelSpecName{"writer"},
        .source = std::string(kMatmulKernels) + "/dataflow/writer_unary_interleaved_start_id.cpp",
        .dfb_bindings = {m2::ConsumerOf(dfb_out, "out")},
        .tensor_bindings = {{.tensor_parameter_name = m2::TensorParamName{"output"}, .accessor_name = "output"}},
        .runtime_arg_schema = {.runtime_arg_names = {"num_pages", "start_id"}},
        .hw_config = ttnn::create_writer_datamovement_config(/*disable_dfb_implicit_sync_for_all=*/true),
    };
    // bmm_metal2.cpp loops over batch, Mt and Nt as one loop, so each core group's compute kernel sets only Nt, to the
    // group's tile count.
    const auto compute = [&](const char* name, uint32_t tiles_per_core) {
        m2::ComputeHardwareConfig hw_config;
        hw_config.unpack_modes = {
            {in0, tt::tt_metal::UnpackMode::UnpackToSrc}, {in1, tt::tt_metal::UnpackMode::UnpackToSrc}};
        return m2::KernelSpec{
            .unique_id = m2::KernelSpecName{name},
            .source = std::string(kMatmulKernels) + "/compute/bmm_metal2.cpp",
            .dfb_bindings = {m2::ConsumerOf(in0, "in0"), m2::ConsumerOf(in1, "in1"), m2::ProducerOf(dfb_out, "out")},
            .compile_time_args = {{"batch", 1}, {"Mt", 1}, {"Kt", Kt}, {"Nt", tiles_per_core}},
            .hw_config = hw_config,
        };
    };
    m2::ProgramSpec spec{
        .name = "matmul",
        .kernels = {reader, writer, compute("compute_1", tiles_1)},
        .dataflow_buffers = {dfb(in0), dfb(in1), dfb(dfb_out)},
        .tensor_parameters =
            {{.unique_id = m2::TensorParamName{"in0"}, .spec = a.tensor_spec()},
             {.unique_id = m2::TensorParamName{"in1"}, .spec = b.tensor_spec()},
             {.unique_id = m2::TensorParamName{"output"}, .spec = output.tensor_spec()}},
        .work_units =
            {{.name = "group_1",
              .kernels = {reader.unique_id, writer.unique_id, m2::KernelSpecName{"compute_1"}},
              .target_nodes = group_1}},
    };
    if (!group_2.ranges().empty()) {
        spec.kernels.push_back(compute("compute_2", tiles_2));
        spec.work_units.push_back(
            {.name = "group_2",
             .kernels = {reader.unique_id, writer.unique_id, m2::KernelSpecName{"compute_2"}},
             .target_nodes = group_2});
    }

    m2::KernelRunArgs reader_args{.kernel = reader.unique_id};
    reader_args.common_runtime_arg_values = {
        {"Mt", Mt},
        {"Kt", Kt},
        {"Nt", Nt},
        {"MtKt", Mt * Kt},
        {"KtNt", Kt * Nt},
        {"batch", batch},
        {"bcast_B", 0},
        {"MtNt", Mt * Nt}};
    m2::KernelRunArgs writer_args{.kernel = writer.unique_id};
    uint32_t start = 0;
    for (const auto& core : corerange_to_cores(all_cores, num_cores, /*row_wise=*/false)) {
        const uint32_t tiles = group_1.contains(core) ? tiles_1 : tiles_2;
        m2::AddRuntimeArgsForNode(
            reader_args.runtime_arg_values, core, {{"output_tile_start_id", start}, {"num_output_tiles", tiles}});
        m2::AddRuntimeArgsForNode(writer_args.runtime_arg_values, core, {{"num_pages", tiles}, {"start_id", start}});
        start += tiles;
    }
    const m2::ProgramRunArgs run_args{
        .kernel_run_args = {reader_args, writer_args},
        .tensor_args =
            {{m2::TensorParamName{"in0"}, a.mesh_tensor()},
             {m2::TensorParamName{"in1"}, b.mesh_tensor()},
             {m2::TensorParamName{"output"}, output.mesh_tensor()}},
    };
    ttnn::experimental::generic_op({a, b, output}, spec, run_args);

    const Tensor golden = ttnn::matmul(
        a,
        b,
        /*transpose_a=*/false,
        /*transpose_b=*/false,
        /*memory_config=*/std::nullopt,
        /*dtype=*/std::nullopt,
        matmul::MatmulMultiCoreProgramConfig{});
    EXPECT_TRUE(ttnn::allclose<bfloat16>(golden.cpu(), output.cpu(), 1e-1f, 1e-5f));
}

// A program binds a PrefetcherPipe parameter to one pipe for its lifetime, and the cache key does not hold the pipe, so
// a later call that passes another pipe of the same shape is rejected, with validate_program_args on or off. A caller
// that needs another pipe gives the parameter another name, or the program its own custom_program_hash.
TEST_F(TTNNFixtureWithDevice, TestGenericOpProgramSpecLaterCallRejectsADifferentPrefetcherPipe) {
    const CoreCoord sender{0, 0};
    const CoreRangeSet receivers(CoreRange(CoreCoord{1, 0}));
    constexpr uint32_t ring_size = 1024;
    const m2::PrefetcherPipeSpaceConfig space_config{
        .sender_cores = CoreRangeSet(CoreRange(sender)),
        .receiver_domain = receivers,
        .ring_size = ring_size,
        .max_receivers_per_pipe = 1,
    };
    // A space must outlive its pipes, and a pipe every program bound to it: the cleanup empties the program cache
    // before the pipes go.
    auto space = m2::CreatePrefetcherPipeSpace(*this->device_, space_config);
    auto pipe = space.create_pipe(sender, receivers);
    auto other_space = m2::CreatePrefetcherPipeSpace(*this->device_, space_config);
    auto other_pipe = other_space.create_pipe(sender, receivers);
    const auto clear_program_cache = ttsl::make_cleanup([this] { this->device_->clear_program_cache(); });

    const m2::PrefetcherPipeParamName pipe_name{"pipe"};
    // The sender kernel binds the pipe and sends nothing.
    m2::KernelSpec sender_kernel{
        .unique_id = m2::KernelSpecName{"sender"},
        .source = m2::KernelSpec::SourceCode{.code = "void kernel_main() {}"},
        .hw_config = ttnn::create_writer_datamovement_config(),
    };
    sender_kernel.advanced_options.prefetcher_pipe_bindings = {
        {.pipe_parameter_names = {pipe_name}, .accessor_name = "out"}};
    const m2::ProgramSpec spec{
        .name = "pipe_sender",
        .kernels = {sender_kernel},
        .work_units = {{.name = "sender", .kernels = {sender_kernel.unique_id}, .target_nodes = sender}},
        .advanced_options =
            {.prefetcher_pipe_parameters =
                 {{.unique_id = pipe_name, .receivers = receivers, .ring_size = ring_size, .entry_size = 256}}},
    };
    const auto run_args_with = [&](m2::PrefetcherPipe& prefetcher_pipe) {
        m2::ProgramRunArgs run_args;
        run_args.advanced_options.prefetcher_pipe_args.emplace(pipe_name, m2::PrefetcherPipeArgument{prefetcher_pipe});
        return run_args;
    };
    const Tensor output = ttnn::create_device_tensor(
        tt::tt_metal::TensorSpec(
            ttnn::Shape({tt::constants::TILE_HEIGHT, tt::constants::TILE_WIDTH}),
            TensorLayout(DataType::BFLOAT16, PageConfig(Layout::TILE), MemoryConfig{})),
        this->device_);

    // Preparing the program binds it to the first pipe without running it.
    ttnn::experimental::prepare_generic_op({output}, spec, run_args_with(pipe));
    for (const bool validate : {true, false}) {
        const auto restore = set_validate_program_args(validate);
        EXPECT_THAT(
            [&] { ttnn::experimental::generic_op({output}, spec, run_args_with(other_pipe)); },
            ThrowsMessage<std::exception>(HasSubstr("different PrefetcherPipe object")));
    }
}

// The op does not opt into per-core allocation, so ttnn rejects a per-core allocated tensor before the op sees it.
TEST_F(GenericOpProgramSpecHybridAllocatorTest, TestGenericOpProgramSpecRejectsAPerCoreAllocatedTensor) {
    const CoreRangeSet cores(CoreRange(CoreCoord{0, 0}, CoreCoord{1, 0}));
    MemoryConfig per_core_config{
        TensorMemoryLayout::HEIGHT_SHARDED,
        BufferType::L1,
        tt::tt_metal::ShardSpec(
            cores, {tt::constants::TILE_HEIGHT, tt::constants::TILE_WIDTH}, ShardOrientation::ROW_MAJOR)};
    tt::tt_metal::experimental::per_core_allocation::set_per_core_allocation(per_core_config, true);
    const ttnn::Shape shape{2 * tt::constants::TILE_HEIGHT, tt::constants::TILE_WIDTH};
    const Tensor input = ttnn::create_device_tensor(
        tt::tt_metal::TensorSpec(shape, TensorLayout(DataType::BFLOAT16, PageConfig(Layout::TILE), per_core_config)),
        this->device_);
    const Tensor output = ttnn::create_device_tensor(
        tt::tt_metal::TensorSpec(shape, TensorLayout(DataType::BFLOAT16, PageConfig(Layout::TILE), MemoryConfig{})),
        this->device_);
    const auto program = make_exp_program(input, output);
    EXPECT_THAT(
        [&] { ttnn::experimental::generic_op({input, output}, program.spec, program.run_args); },
        ThrowsMessage<std::exception>(HasSubstr("is per-core allocated, but this operation has not opted in")));
    EXPECT_THAT(
        [&] { ttnn::experimental::prepare_generic_op({input, output}, program.spec, program.run_args); },
        ThrowsMessage<std::exception>(HasSubstr("is per-core allocated, but this operation has not opted in")));
}

// A program runs on every device of its mesh range, so each of them must hold every tensor. Like every
// MeshDevice1x2Fixture test, this needs two devices and fast dispatch.
TEST_F(MeshDevice1x2Fixture, TestGenericOpProgramSpecRejectsARangeOverADeviceWithoutATensor) {
    auto& mesh_device = *get_mesh_device();
    // One tile sharded over the mesh on its batch dimension is on the first device only.
    const Tensor host_input =
        ttnn::random::random(ttnn::Shape{1, 1, tt::constants::TILE_HEIGHT, tt::constants::TILE_WIDTH})
            .to_layout(Layout::TILE);
    const Tensor input = ttnn::distributed::distribute_tensor(
        host_input, *ttnn::distributed::shard_tensor_to_mesh_mapper(mesh_device, /*dim=*/0), mesh_device);
    ASSERT_EQ(input.device_storage().get_coords().size(), 1u);
    const Tensor output = ttnn::create_device_tensor(input.tensor_spec(), &mesh_device);
    const auto program = make_exp_program(input, output);
    const ttnn::MeshCoordinateRange whole_mesh(mesh_device.shape());
    const ttnn::experimental::MeshProgramSpec mesh_spec{.programs = {{whole_mesh, program.spec}}};
    const ttnn::experimental::MeshProgramRunArgs mesh_run_args{.programs = {{whole_mesh, program.run_args}}};
    EXPECT_THAT(
        [&] { ttnn::experimental::generic_op({input, output}, mesh_spec, mesh_run_args); },
        ThrowsMessage<std::exception>(HasSubstr("which does not hold io_tensors[0]")));
}

}  // namespace ttnn::operations::generic::test
