// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "add_integers_hang_op.hpp"

#include <tt-metalium/experimental/metal2_host_api/program_run_args.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_spec.hpp>

#include "ttnn/operations/core/data_movement_kernel/datamovement_kernel_config.hpp"

namespace triage_hang_apps {

using namespace tt;
using namespace tt::tt_metal;
using namespace tt::tt_metal::experimental;

namespace {

const DFBSpecName IN0_DFB{"in0_dfb"};
const DFBSpecName IN1_DFB{"in1_dfb"};
const DFBSpecName OUT_DFB{"out_dfb"};
const TensorParamName IN0_T{"in0_tensor"};
const TensorParamName IN1_T{"in1_tensor"};
const TensorParamName OUT_T{"out_tensor"};
const KernelSpecName READER{"reader"};
const KernelSpecName WRITER{"writer"};
const KernelSpecName COMPUTE{"compute"};

constexpr auto KERNEL_DIR = "tools/tests/triage/hang_apps/add_2_integers_hang/kernels/";

}  // namespace

ttnn::device_operation::ProgramArtifacts AddIntegersHangOperation::SingleCore::create_program_artifacts(
    const operation_attributes_t&, const tensor_args_t& tensor_args, tensor_return_value_t& tensor_return_value) {
    // Bind the MeshTensors reachable from tensor_args / tensor_return_value: the adapter matches
    // TensorArguments to those by pointer identity, so a copy would be rejected.
    const auto& src0 = tensor_args.input_tensor_a.mesh_tensor();
    const auto& src1 = tensor_args.input_tensor_b.mesh_tensor();
    const auto& dst = tensor_return_value.mesh_tensor();

    const tt::DataFormat data_format = datatype_to_dataformat_converter(tensor_args.input_tensor_a.dtype());
    const tt::DataFormat data_format_output = datatype_to_dataformat_converter(tensor_return_value.dtype());

    constexpr uint32_t num_tiles = 1;
    auto make_dfb_spec = [](const DFBSpecName& name, tt::DataFormat format) {
        return DataflowBufferSpec{
            .unique_id = name,
            .entry_size = tile_size(format),
            .num_entries = num_tiles,
            .data_format_metadata = format,
        };
    };

    // Reuse the sibling add_2_integers_hang kernels. They carry the architecture-portable Metal 2.0
    // dataflow-buffer device API, so the same sources run on Wormhole / Blackhole and on Quasar; the
    // hardware configs below select the generation matching the active architecture.
    const tt::ARCH arch = tensor_args.input_tensor_a.device()->arch();
    ComputeHardwareConfig compute_hw_config;
    if (arch == tt::ARCH::QUASAR) {
        compute_hw_config = ComputeGen2Config{.fpu_math_fidelity = MathFidelity::HiFi4};
    } else {
        compute_hw_config = ComputeGen1Config{.fpu_math_fidelity = MathFidelity::HiFi4};
    }

    KernelSpec reader_spec{
        .unique_id = READER,
        .source = std::string(KERNEL_DIR) + "dataflow/reader_binary_1_tile.cpp",
        .dfb_bindings = {ProducerOf(IN0_DFB, "in0"), ProducerOf(IN1_DFB, "in1")},
        .tensor_bindings =
            {{.tensor_parameter_name = IN0_T, .accessor_name = "in0"},
             {.tensor_parameter_name = IN1_T, .accessor_name = "in1"}},
        .hw_config = ttnn::create_reader_datamovement_config(arch),
    };

    KernelSpec writer_spec{
        .unique_id = WRITER,
        .source = std::string(KERNEL_DIR) + "dataflow/writer_1_tile.cpp",
        .dfb_bindings = {ConsumerOf(OUT_DFB, "out")},
        .tensor_bindings = {{.tensor_parameter_name = OUT_T, .accessor_name = "out"}},
        .hw_config = ttnn::create_writer_datamovement_config(arch),
    };

    KernelSpec compute_spec{
        .unique_id = COMPUTE,
        .source = std::string(KERNEL_DIR) + "compute/add_2_tiles_hang.cpp",
        // Metal 2.0's type-agnostic default opt level is O2; a compute kernel used to get O3, so
        // state it explicitly to keep the generated code the same as before.
        .compiler_options = {.opt_level = KernelBuildOptLevel::O3},
        .dfb_bindings = {ConsumerOf(IN0_DFB, "in0"), ConsumerOf(IN1_DFB, "in1"), ProducerOf(OUT_DFB, "out")},
        .hw_config = compute_hw_config,
    };

    ProgramSpec spec{
        .name = "add_integers_hang",
        .kernels = {reader_spec, writer_spec, compute_spec},
        .dataflow_buffers =
            {make_dfb_spec(IN0_DFB, data_format),
             make_dfb_spec(IN1_DFB, data_format),
             make_dfb_spec(OUT_DFB, data_format_output)},
        .tensor_parameters =
            {{.unique_id = IN0_T, .spec = src0.tensor_spec()},
             {.unique_id = IN1_T, .spec = src1.tensor_spec()},
             {.unique_id = OUT_T, .spec = dst.tensor_spec()}},
        .work_units = {{.name = "main", .kernels = {READER, WRITER, COMPUTE}, .target_nodes = NodeCoord{0, 0}}},
    };

    // None of the three kernels declares runtime args of its own: the reader and writer get their
    // addresses from the tensor arguments, and the compute kernel needs none. The framework applies
    // these run args for us; the factory must not call SetProgramRunArgs itself.
    ProgramRunArgs run_args;
    run_args.tensor_args = {
        {IN0_T, ProgramRunArgs::TensorArgument{src0}},
        {IN1_T, ProgramRunArgs::TensorArgument{src1}},
        {OUT_T, ProgramRunArgs::TensorArgument{dst}},
    };

    return ttnn::device_operation::ProgramArtifacts{.spec = std::move(spec), .run_params = std::move(run_args)};
}

}  // namespace triage_hang_apps
