// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "qkv_causal_conv1d_silu_bw_program_factory.hpp"

#include <string>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/experimental/metal2_host_api/compute_hardware_config.hpp>
#include <tt-metalium/experimental/metal2_host_api/dataflow_buffer_spec.hpp>
#include <tt-metalium/experimental/metal2_host_api/kernel_spec.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_run_args.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_spec.hpp>
#include <tt-metalium/experimental/metal2_host_api/tensor_parameter.hpp>
#include <tt-metalium/work_split.hpp>
#include <vector>

#include "metal/common/program_utils.hpp"
#include "ttnn/operations/core/data_movement_kernel/datamovement_kernel_config.hpp"

namespace m2 = tt::tt_metal::experimental;

// The shared layout header spells the tile geometry out so the kernel toolchain can compile it.
static_assert(qkv_conv_bw::tile_height == tt::constants::TILE_HEIGHT);
static_assert(qkv_conv_bw::tile_width == tt::constants::TILE_WIDTH);
static_assert(qkv_conv_bw::face_height == tt::constants::FACE_HEIGHT);
static_assert(qkv_conv_bw::face_width == tt::constants::FACE_WIDTH);

namespace ttml::metal::ops::qkv_causal_conv1d_silu_bw::device {

namespace {

constexpr std::string_view kernel_dir = "tt-train/sources/ttml/metal/ops/qkv_causal_conv1d_silu_bw/device/kernels/";

const m2::KernelSpecName READER{"reader"};
const m2::KernelSpecName WRITER{"writer"};
const m2::KernelSpecName COMPUTE{"compute"};
const m2::ScratchpadSpecName X_WINDOW{"x_window"};
const m2::ScratchpadSpecName GRAD_PREACT_STRIPS{"grad_preact_strips"};

m2::DataflowBufferSpec make_dataflow_buffer(const m2::DFBSpecName& name, uint32_t tiles, tt::DataFormat format) {
    return m2::DataflowBufferSpec{
        .unique_id = name,
        .entry_size = tt::tile_size(format),
        .num_entries = tiles,
        .data_format_metadata = format,
    };
}

m2::DFBBinding producer(const m2::DFBSpecName& name) {
    return {name, name.get(), m2::DFBEndpointType::PRODUCER};
}
m2::DFBBinding consumer(const m2::DFBSpecName& name) {
    return {name, name.get(), m2::DFBEndpointType::CONSUMER};
}
m2::TensorBinding tensor(const m2::TensorParamName& name) {
    return {name, name.get()};
}

}  // namespace

ttnn::device_operation::ProgramArtifacts QkvCausalConv1dSiluBackwardProgramFactory::create_program_artifacts(
    const QkvCausalConv1dSiluBackwardParams& attrs,
    const QkvCausalConv1dSiluBackwardInputs& in,
    std::vector<Tensor>& outputs) {
    const auto& input = in.input.mesh_tensor();
    const auto& grad_input = outputs[0].mesh_tensor();
    const bool tap_grads = attrs.compute_tap_grads;

    const uint32_t tile_rows = attrs.row_count / tt::constants::TILE_HEIGHT;
    const uint32_t q_tiles = attrs.q_width / tt::constants::TILE_WIDTH;
    const uint32_t k_tiles = attrs.k_width / tt::constants::TILE_WIDTH;
    const uint32_t v_tiles = attrs.v_width / tt::constants::TILE_WIDTH;
    const uint32_t channel_tiles = q_tiles + k_tiles + v_tiles;
    const uint32_t channel_tiles_per_block = attrs.channel_chunk_size / tt::constants::TILE_WIDTH;
    const uint32_t channel_blocks = channel_tiles / channel_tiles_per_block;
    const uint32_t row_splits = attrs.row_splits;
    const auto grid = input.device().compute_with_storage_grid_size();
    const auto [num_cores, all_cores, core_group_1, core_group_2, items_per_core_1, items_per_core_2] =
        tt::tt_metal::split_work_to_cores(grid, channel_blocks * row_splits);
    const auto run_args_for = [&](const m2::KernelSpecName& kernel) {
        m2::KernelRunArgs args{.kernel = kernel};
        for_each_core_with_work(
            num_cores, grid.y, core_group_1, core_group_2, items_per_core_1, items_per_core_2, [&](const CoreWork& w) {
                m2::AddRuntimeArgsForNode(
                    args.runtime_arg_values, w.core, {{"wi_start", w.start}, {"wi_count", w.num_units}});
            });
        return args;
    };

    const m2::DFBSpecName X_SHIFTED_TILE{"x_shifted_tile"}, WEIGHTS{"weights"}, GRAD_OUT{"grad_out"}, PREACT{"preact"},
        SIGMOID{"sigmoid"}, GRAD_PREACT_OUT{"grad_preact_out"}, GRAD_PREACT_TILE{"grad_preact_tile"},
        PARTIAL{"partial"}, GRAD_PREACT_SHIFTED_TILE{"grad_preact_shifted_tile"}, OUTPUT{"output"}, ONES{"ones"};
    const m2::TensorParamName INPUT{"input"}, HISTORY{"history"}, GRAD_Q{"grad_q"}, GRAD_K{"grad_k"}, GRAD_V{"grad_v"},
        TAP0{"tap0"}, TAP1{"tap1"}, TAP2{"tap2"}, TAP3{"tap3"}, GRAD_INPUT{"grad_input"},
        GRAD_TAP_PARTIAL{"grad_tap_partial"};

    const auto bf16 = tt::tt_metal::datatype_to_dataformat_converter(input.dtype());
    m2::Group<m2::DataflowBufferSpec> dataflow_buffers;
    for (const auto& buffer : qkv_conv_bw_dataflow_buffers(channel_tiles_per_block, bf16, tap_grads)) {
        dataflow_buffers.push_back(
            make_dataflow_buffer(m2::DFBSpecName{std::string(buffer.name)}, buffer.tiles, buffer.format));
    }
    m2::KernelSpec reader{
        .unique_id = READER,
        .source = std::string(kernel_dir) + "dataflow/reader_qkv_causal_conv1d_silu_bw.cpp",
        .dfb_bindings = {producer(X_SHIFTED_TILE), producer(WEIGHTS), producer(GRAD_OUT), producer(ONES)},
        .scratchpad_bindings = {{X_WINDOW, "x_window"}},
        .tensor_bindings =
            {tensor(INPUT),
             tensor(HISTORY),
             tensor(GRAD_Q),
             tensor(GRAD_K),
             tensor(GRAD_V),
             tensor(TAP0),
             tensor(TAP1),
             tensor(TAP2),
             tensor(TAP3)},
        .compile_time_args =
            {{"channel_tiles_per_block", channel_tiles_per_block},
             {"channel_tiles", channel_tiles},
             {"tile_rows", tile_rows},
             {"row_splits", row_splits},
             {"q_tiles", q_tiles},
             {"k_tiles", k_tiles},
             {"v_tiles", v_tiles}},
        .runtime_arg_schema = {.runtime_arg_names = {"wi_start", "wi_count"}},
        .hw_config = ttnn::create_reader_datamovement_config(),
    };
    m2::KernelSpec writer{
        .unique_id = WRITER,
        .source = std::string(kernel_dir) + "dataflow/writer_qkv_causal_conv1d_silu_bw.cpp",
        .dfb_bindings = {consumer(GRAD_PREACT_OUT), producer(GRAD_PREACT_SHIFTED_TILE), consumer(OUTPUT)},
        .scratchpad_bindings = {{GRAD_PREACT_STRIPS, "grad_preact_strips"}},
        .tensor_bindings = {tensor(GRAD_INPUT)},
        .compile_time_args =
            {{"channel_tiles_per_block", channel_tiles_per_block},
             {"tile_rows", tile_rows},
             {"row_splits", row_splits},
             {"channel_tiles", channel_tiles}},
        .runtime_arg_schema = {.runtime_arg_names = {"wi_start", "wi_count"}},
        .hw_config = ttnn::create_writer_datamovement_config(),
    };
    // The conv accumulates its taps in fp32 DST (the kernel static_asserts DST_ACCUM_MODE) and the partial is
    // accumulated by the packer explicitly, so the config is fixed rather than a hashed attribute.
    m2::ComputeHardwareConfig compute_hw{
        .fpu_math_fidelity = tt::tt_metal::MathFidelity::HiFi4,
        .sfpu_precision_mode = tt::tt_metal::Precision::Precise,
        .enable_32_bit_dest = true,
        .double_buffer_dest = true,
    };
    // The fp32 sigmoid is an FPU operand (mul / reuse-dest), so it unpacks to SrcA/B: 19-bit, still far better
    // than bf16 for 1 - s near s = 1. An fp32 consumer must name its mode when the dest is 32-bit.
    compute_hw.unpack_modes.emplace(SIGMOID, tt::tt_metal::UnpackMode::UnpackToSrc);
    m2::KernelSpec compute{
        .unique_id = COMPUTE,
        .source = std::string(kernel_dir) + "compute/qkv_causal_conv1d_silu_bw.cpp",
        .compiler_options = {.opt_level = tt::tt_metal::KernelBuildOptLevel::O3},
        .dfb_bindings =
            {consumer(X_SHIFTED_TILE),
             consumer(WEIGHTS),
             consumer(GRAD_OUT),
             producer(PREACT),
             consumer(PREACT),
             producer(SIGMOID),
             consumer(SIGMOID),
             producer(GRAD_PREACT_OUT),
             consumer(GRAD_PREACT_SHIFTED_TILE),
             producer(OUTPUT),
             consumer(ONES)},
        .compile_time_args =
            {{"channel_tiles_per_block", channel_tiles_per_block},
             {"tile_rows", tile_rows},
             {"row_splits", row_splits}},
        .runtime_arg_schema = {.runtime_arg_names = {"wi_start", "wi_count"}},
        .hw_config = compute_hw,
    };

    for (auto* kernel : {&writer, &compute}) {
        kernel->compiler_options.defines.insert({"COMPUTE_TAP_GRADS", tap_grads ? "1" : "0"});
    }
    if (tap_grads) {
        writer.dfb_bindings.push_back(consumer(PARTIAL));
        writer.tensor_bindings.push_back(tensor(GRAD_TAP_PARTIAL));
        compute.dfb_bindings.push_back(producer(GRAD_PREACT_TILE));
        compute.dfb_bindings.push_back(consumer(GRAD_PREACT_TILE));
        compute.dfb_bindings.push_back(producer(PARTIAL));
    }
    m2::ProgramSpec spec{
        .name = "qkv_causal_conv1d_silu_bw",
        .kernels = {std::move(reader), std::move(writer), std::move(compute)},
        .dataflow_buffers = std::move(dataflow_buffers),
        .scratchpads =
            {{.unique_id = X_WINDOW, .size_per_node = qkv_conv_bw::ReaderScratch{channel_tiles_per_block}.bytes()},
             {.unique_id = GRAD_PREACT_STRIPS,
              .size_per_node = qkv_conv_bw::WriterScratch{channel_tiles_per_block}.bytes()}},
        .tensor_parameters =
            {
                {.unique_id = INPUT, .spec = input.tensor_spec()},
                {.unique_id = HISTORY, .spec = in.history.mesh_tensor().tensor_spec()},
                {.unique_id = GRAD_Q, .spec = in.grad_q.mesh_tensor().tensor_spec()},
                {.unique_id = GRAD_K, .spec = in.grad_k.mesh_tensor().tensor_spec()},
                {.unique_id = GRAD_V, .spec = in.grad_v.mesh_tensor().tensor_spec()},
                {.unique_id = TAP0, .spec = in.tap0.mesh_tensor().tensor_spec()},
                {.unique_id = TAP1, .spec = in.tap1.mesh_tensor().tensor_spec()},
                {.unique_id = TAP2, .spec = in.tap2.mesh_tensor().tensor_spec()},
                {.unique_id = TAP3, .spec = in.tap3.mesh_tensor().tensor_spec()},
                {.unique_id = GRAD_INPUT, .spec = grad_input.tensor_spec()},
            },
        .work_units = {{.name = "main", .kernels = {READER, WRITER, COMPUTE}, .target_nodes = all_cores}},
    };

    m2::ProgramRunArgs run_args;
    run_args.kernel_run_args = {run_args_for(READER), run_args_for(WRITER), run_args_for(COMPUTE)};
    run_args.tensor_args = {
        {INPUT, input},
        {HISTORY, in.history.mesh_tensor()},
        {GRAD_Q, in.grad_q.mesh_tensor()},
        {GRAD_K, in.grad_k.mesh_tensor()},
        {GRAD_V, in.grad_v.mesh_tensor()},
        {TAP0, in.tap0.mesh_tensor()},
        {TAP1, in.tap1.mesh_tensor()},
        {TAP2, in.tap2.mesh_tensor()},
        {TAP3, in.tap3.mesh_tensor()},
        {GRAD_INPUT, grad_input},
    };
    if (tap_grads) {
        const auto& grad_tap_partial = outputs[1].mesh_tensor();
        spec.tensor_parameters.push_back({.unique_id = GRAD_TAP_PARTIAL, .spec = grad_tap_partial.tensor_spec()});
        run_args.tensor_args.insert({GRAD_TAP_PARTIAL, grad_tap_partial});
    }
    return {std::move(spec), std::move(run_args)};
}

}  // namespace ttml::metal::ops::qkv_causal_conv1d_silu_bw::device
