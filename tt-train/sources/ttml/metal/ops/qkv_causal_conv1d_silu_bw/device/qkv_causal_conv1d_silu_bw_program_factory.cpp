// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "qkv_causal_conv1d_silu_bw_program_factory.hpp"

#include <string>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/experimental/metal2_host_api/dataflow_buffer_spec.hpp>
#include <tt-metalium/experimental/metal2_host_api/kernel_spec.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_run_args.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_spec.hpp>
#include <tt-metalium/experimental/metal2_host_api/tensor_parameter.hpp>
#include <tt-metalium/work_split.hpp>
#include <vector>

#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"
#include "ttnn/operations/core/data_movement_kernel/datamovement_kernel_config.hpp"

namespace m2 = tt::tt_metal::experimental;

namespace ttml::metal::ops::qkv_causal_conv1d_silu_bw::device {

namespace {

constexpr std::string_view kernel_dir = "tt-train/sources/ttml/metal/ops/qkv_causal_conv1d_silu_bw/device/kernels/";

const m2::KernelSpecName READER{"reader"};
const m2::KernelSpecName WRITER{"writer"};
const m2::KernelSpecName COMPUTE{"compute"};
const m2::ScratchpadSpecName ACT_WINDOW{"act_window"};
const m2::ScratchpadSpecName DC_WINDOW{"dc_window"};

m2::DataflowBufferSpec make_dfb(const m2::DFBSpecName& name, uint32_t tiles, tt::DataFormat format) {
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

struct WorkDist {
    std::vector<tt::tt_metal::CoreCoord> cores;
    std::vector<uint32_t> wi_start;
    std::vector<uint32_t> wi_count;
    tt::tt_metal::CoreRangeSet core_set;
};

WorkDist distribute(tt::tt_metal::CoreCoord grid, uint32_t total) {
    const uint32_t count = std::min<uint32_t>(total, grid.x * grid.y);
    WorkDist dist;
    uint32_t offset = 0;
    for (uint32_t index = 0; index < count; ++index) {
        const uint32_t item_count = total / count + (index < total % count ? 1 : 0);
        dist.cores.push_back({index % grid.x, index / grid.x});
        dist.wi_start.push_back(offset);
        dist.wi_count.push_back(item_count);
        offset += item_count;
    }
    dist.core_set = tt::tt_metal::num_cores_to_corerangeset(count, grid, true);
    return dist;
}

m2::KernelRunArgs run_args_for(const m2::KernelSpecName& kernel, const WorkDist& dist) {
    m2::KernelRunArgs args{.kernel = kernel};
    for (uint32_t i = 0; i < dist.cores.size(); ++i) {
        m2::AddRuntimeArgsForNode(
            args.runtime_arg_values, dist.cores[i], {{"wi_start", dist.wi_start[i]}, {"wi_count", dist.wi_count[i]}});
    }
    return args;
}

}  // namespace

ttnn::device_operation::ProgramArtifacts QkvCausalConv1dSiluBackwardProgramFactory::create_program_artifacts(
    const QkvCausalConv1dSiluBackwardParams& attrs,
    const QkvCausalConv1dSiluBackwardInputs& in,
    std::vector<Tensor>& outputs) {
    const auto& input = in.input.mesh_tensor();
    const auto& grad_input = outputs[0].mesh_tensor();
    const auto& grad_tap_partial = outputs[1].mesh_tensor();

    const uint32_t Mt = attrs.sequence / tt::constants::TILE_HEIGHT;
    const uint32_t Qt = attrs.q_width / tt::constants::TILE_WIDTH;
    const uint32_t Kt = attrs.k_width / tt::constants::TILE_WIDTH;
    const uint32_t Vt = attrs.v_width / tt::constants::TILE_WIDTH;
    const uint32_t Ct = Qt + Kt + Vt;
    const uint32_t block_ct = attrs.channel_chunk_size / tt::constants::TILE_WIDTH;
    const uint32_t num_blocks = Ct / block_ct;
    const uint32_t splits = attrs.sequence_splits;
    const auto dist = distribute(input.device().compute_with_storage_grid_size(), num_blocks * splits);

    const m2::DFBSpecName ACT_RM{"act_rm"}, ACT_TILE{"act_tile"}, WEIGHTS{"weights"}, GRAD_OUT{"grad_out"},
        PREACT{"preact"}, SIGMOID{"sigmoid"}, DC_TILE{"dc_tile"}, DC_RM{"dc_rm"}, ACC{"acc"}, GRAD_TAP{"grad_tap"},
        DCV_RM{"dcv_rm"}, DCV_TILE{"dcv_tile"}, OUTPUT{"output"}, ONES{"ones"};
    const m2::TensorParamName INPUT{"input"}, HISTORY{"history"}, GRAD_Q{"grad_q"}, GRAD_K{"grad_k"}, GRAD_V{"grad_v"},
        TAP0{"tap0"}, TAP1{"tap1"}, TAP2{"tap2"}, TAP3{"tap3"}, GRAD_INPUT{"grad_input"},
        GRAD_TAP_PARTIAL{"grad_tap_partial"};

    const auto bf16 = tt::tt_metal::datatype_to_dataformat_converter(input.dtype());
    m2::Group<m2::DataflowBufferSpec> dfbs;
    for (const auto& dfb : qkv_conv_bw_dfbs(block_ct, bf16)) {
        dfbs.push_back(make_dfb(m2::DFBSpecName{std::string(dfb.name)}, dfb.tiles, dfb.format));
    }
    const auto window_bytes = static_cast<uint32_t>(qkv_conv_bw_window_bytes(block_ct, input.element_size()));

    m2::KernelSpec reader{
        .unique_id = READER,
        .source = std::string(kernel_dir) + "dataflow/reader.cpp",
        .dfb_bindings = {producer(ACT_RM), producer(WEIGHTS), producer(GRAD_OUT), producer(ONES)},
        .scratchpad_bindings = {{ACT_WINDOW, "act_window"}},
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
            {{"block_ct", block_ct}, {"Mt", Mt}, {"splits", splits}, {"Qt", Qt}, {"Kt", Kt}, {"Vt", Vt}},
        .runtime_arg_schema = {.runtime_arg_names = {"wi_start", "wi_count"}},
        .hw_config = ttnn::create_reader_datamovement_config(),
    };
    m2::KernelSpec writer{
        .unique_id = WRITER,
        .source = std::string(kernel_dir) + "dataflow/writer.cpp",
        .dfb_bindings = {consumer(DC_RM), producer(DCV_RM), consumer(OUTPUT), consumer(GRAD_TAP)},
        .scratchpad_bindings = {{DC_WINDOW, "dc_window"}},
        .tensor_bindings = {tensor(GRAD_INPUT), tensor(GRAD_TAP_PARTIAL)},
        .compile_time_args = {{"block_ct", block_ct}, {"Mt", Mt}, {"splits", splits}, {"Ct", Ct}},
        .runtime_arg_schema = {.runtime_arg_names = {"wi_start", "wi_count"}},
        .hw_config = ttnn::create_writer_datamovement_config(),
    };
    auto compute_hw = ttnn::to_compute_hardware_config(attrs.compute_kernel_config);
    compute_hw.unpack_modes.emplace(ACC, tt::tt_metal::UnpackMode::UnpackToSrc);
    m2::KernelSpec compute{
        .unique_id = COMPUTE,
        .source = std::string(kernel_dir) + "compute/qkv_causal_conv1d_silu_bw.cpp",
        .dfb_bindings = {consumer(ACT_RM),   producer(ACT_TILE), consumer(ACT_TILE), consumer(WEIGHTS),
                         consumer(GRAD_OUT), producer(PREACT),   consumer(PREACT),   producer(SIGMOID),
                         consumer(SIGMOID),  producer(DC_TILE),  consumer(DC_TILE),  producer(DC_RM),
                         producer(ACC),      consumer(ACC),      producer(GRAD_TAP), consumer(DCV_RM),
                         producer(DCV_TILE), consumer(DCV_TILE), producer(OUTPUT),   consumer(ONES)},
        .compile_time_args = {{"block_ct", block_ct}, {"Mt", Mt}, {"splits", splits}},
        .runtime_arg_schema = {.runtime_arg_names = {"wi_start", "wi_count"}},
        .hw_config = compute_hw,
    };

    m2::ProgramSpec spec{
        .name = "qkv_causal_conv1d_silu_bw",
        .kernels = {std::move(reader), std::move(writer), std::move(compute)},
        .dataflow_buffers = std::move(dfbs),
        .scratchpads =
            {{.unique_id = ACT_WINDOW, .size_per_node = window_bytes},
             {.unique_id = DC_WINDOW, .size_per_node = window_bytes}},
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
                {.unique_id = GRAD_TAP_PARTIAL, .spec = grad_tap_partial.tensor_spec()},
            },
        .work_units = {{.name = "main", .kernels = {READER, WRITER, COMPUTE}, .target_nodes = dist.core_set}},
    };

    m2::ProgramRunArgs run_args;
    run_args.kernel_run_args = {run_args_for(READER, dist), run_args_for(WRITER, dist), run_args_for(COMPUTE, dist)};
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
        {GRAD_TAP_PARTIAL, grad_tap_partial},
    };
    return {std::move(spec), std::move(run_args)};
}

}  // namespace ttml::metal::ops::qkv_causal_conv1d_silu_bw::device
