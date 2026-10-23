// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "polynorm_fw_program_factory.hpp"

#include <bit>
#include <enchantum/enchantum.hpp>
#include <optional>
#include <tt-metalium/program_descriptors.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>

#include "metal/common/program_utils.hpp"
#include "polynorm_fw_device_operation.hpp"

namespace {

constexpr auto kReaderKernelPath =
    "tt-train/sources/ttml/metal/ops/polynorm_fw/device/kernels/dataflow/"
    "reader_polynorm_fw_interleaved_start_id.cpp";
constexpr auto kWriterKernelPath =
    "tt-train/sources/ttml/metal/ops/polynorm_fw/device/kernels/dataflow/"
    "writer_polynorm_fw_interleaved_start_id.cpp";
constexpr auto kComputeKernelPath =
    "tt-train/sources/ttml/metal/ops/polynorm_fw/device/kernels/compute/polynorm_fw_kernel.cpp";

// Kernel push order: reader, writer, then the compute kernel of each work group (group 2 only when it has cores)
constexpr uint32_t kReaderKernelIdx = 0U;
constexpr uint32_t kWriterKernelIdx = 1U;
constexpr uint32_t kComputeGroup1KernelIdx = 2U;
constexpr uint32_t kComputeGroup2KernelIdx = 3U;

// reader runtime args
constexpr uint32_t kInputAddrIdx = 0U;
constexpr uint32_t kWeightAddrIdx = 1U;
constexpr uint32_t kBiasAddrIdx = 2U;
// writer runtime args
constexpr uint32_t kOutputAddrIdx = 0U;

// CBs with input data / scalar parameters
constexpr auto kInputPass1CbIndex = tt::CBIndex::c_0;
constexpr auto kInputPass2CbIndex = tt::CBIndex::c_1;
constexpr auto kScalerCbIndex = tt::CBIndex::c_2;
constexpr auto kW0CbIndex = tt::CBIndex::c_3;
constexpr auto kW1CbIndex = tt::CBIndex::c_4;
constexpr auto kW2CbIndex = tt::CBIndex::c_5;
constexpr auto kBiasCbIndex = tt::CBIndex::c_6;
// CBs with intermediate computations
constexpr auto kSumPowsCbIndex = tt::CBIndex::c_7;
constexpr auto kInvRmsCbIndex = tt::CBIndex::c_8;
// Preweighted inv_rms coefficients: [w2*inv_rms(x), w1*inv_rms(x^2), w0*inv_rms(x^3)]
constexpr auto kWeightedCoeffCbIndex = tt::CBIndex::c_9;
// CBs with output data
constexpr auto kOutputCbIndex = tt::CBIndex::c_10;

constexpr uint32_t kNumOneTile = 1U;

}  // namespace

namespace ttml::metal::ops::polynorm3_fw::device {

namespace {

// The input's tile rows over the device's compute grid. create_descriptor() pins it against split_work_to_cores, and
// override_runtime_arguments() walks the same cores from it.
WorkSplit make_work_split(const ttnn::Tensor& input) {
    const auto& shape = input.padded_shape();
    return compute_work_split(
        input.device()->compute_with_storage_grid_size(),
        shape[0] * shape[1] * (shape[-2] / tt::constants::TILE_HEIGHT));
}

}  // namespace

// Describe the full PolyNorm forward program (reader/compute/writer kernels + CB layout).
tt::tt_metal::ProgramDescriptor PolyNorm3ForwardProgramFactory::create_descriptor(
    const PolyNorm3FWAttributes& args, const PolyNorm3FWTensorArgs& tensor_args, PolyNorm3FWTensorReturn& output) {
    const auto& input = tensor_args.input;
    const auto& weight = tensor_args.weight;
    const auto& bias = tensor_args.bias;
    auto* device = input.device();

    tt::tt_metal::ProgramDescriptor desc;
    const tt::DataFormat data_format = datatype_to_dataformat_converter(input.dtype());
    TT_FATAL(data_format == tt::DataFormat::Float16_b, "PolyNorm3Forward currently supports BF16 input only");
    const uint32_t bfloat16_tile_size = tt::tile_size(tt::DataFormat::Float16_b);
    const uint32_t float32_tile_size = tt::tile_size(tt::DataFormat::Float32);

    const auto padded_tensor_shape = input.padded_shape();
    TT_FATAL(padded_tensor_shape.rank() == 4U, "Input tensor must be 4D");
    const uint32_t Wt = padded_tensor_shape[-1] / tt::constants::TILE_WIDTH;
    const uint32_t Ht = padded_tensor_shape[-2] / tt::constants::TILE_HEIGHT;
    const uint32_t NC = padded_tensor_shape[0] * padded_tensor_shape[1];
    const uint32_t total_rows_to_process = NC * Ht;

    const auto compute_with_storage_grid_size = device->compute_with_storage_grid_size();

    constexpr uint32_t block_size = 4U;
    auto [num_cores, all_cores, core_group_1, core_group_2, num_rows_per_core_group_1, num_rows_per_core_group_2] =
        tt::tt_metal::split_work_to_cores(compute_with_storage_grid_size, total_rows_to_process);
    const WorkSplit split = make_work_split(input);
    check_work_split(
        split, num_cores, core_group_1, core_group_2, num_rows_per_core_group_1, num_rows_per_core_group_2);

    desc.cbs.push_back(make_cb_descriptor(all_cores, kInputPass1CbIndex, data_format, bfloat16_tile_size, block_size));
    desc.cbs.push_back(make_cb_descriptor(all_cores, kInputPass2CbIndex, data_format, bfloat16_tile_size, block_size));
    desc.cbs.push_back(
        make_cb_descriptor(all_cores, kScalerCbIndex, tt::DataFormat::Float32, float32_tile_size, kNumOneTile));
    desc.cbs.push_back(make_cb_descriptor(all_cores, kW0CbIndex, data_format, bfloat16_tile_size, kNumOneTile));
    desc.cbs.push_back(make_cb_descriptor(all_cores, kW1CbIndex, data_format, bfloat16_tile_size, kNumOneTile));
    desc.cbs.push_back(make_cb_descriptor(all_cores, kW2CbIndex, data_format, bfloat16_tile_size, kNumOneTile));
    desc.cbs.push_back(make_cb_descriptor(all_cores, kBiasCbIndex, data_format, bfloat16_tile_size, kNumOneTile));
    desc.cbs.push_back(
        make_cb_descriptor(all_cores, kSumPowsCbIndex, tt::DataFormat::Float32, float32_tile_size, /*num_tiles=*/3U));
    desc.cbs.push_back(
        make_cb_descriptor(all_cores, kInvRmsCbIndex, data_format, bfloat16_tile_size, /*num_tiles=*/3U));
    desc.cbs.push_back(
        make_cb_descriptor(all_cores, kWeightedCoeffCbIndex, data_format, bfloat16_tile_size, /*num_tiles=*/3U));
    desc.cbs.push_back(make_cb_descriptor(all_cores, kOutputCbIndex, data_format, bfloat16_tile_size, block_size));
    auto* input_buffer = input.buffer();
    auto* weight_buffer = weight.buffer();
    auto* bias_buffer = bias.buffer();
    TT_FATAL(
        input_buffer->buffer_type() == ttnn::BufferType::DRAM,
        "Input buffer must be in DRAM. Input buffer of type {}",
        enchantum::to_string(input_buffer->buffer_type()));
    TT_FATAL(
        weight_buffer->buffer_type() == ttnn::BufferType::DRAM,
        "weight buffer must be in DRAM. weight buffer of type {}",
        enchantum::to_string(weight_buffer->buffer_type()));
    TT_FATAL(
        bias_buffer->buffer_type() == ttnn::BufferType::DRAM,
        "bias buffer must be in DRAM. bias buffer of type {}",
        enchantum::to_string(bias_buffer->buffer_type()));

    auto* output_buffer = output.buffer();
    TT_FATAL(
        output_buffer->buffer_type() == ttnn::BufferType::DRAM,
        "Output buffer must be in DRAM. Output buffer of type {}",
        enchantum::to_string(output_buffer->buffer_type()));

    std::vector<uint32_t> reader_compile_time_args{block_size, Wt};
    tt::tt_metal::TensorAccessorArgs(input_buffer).append_to(reader_compile_time_args);
    tt::tt_metal::TensorAccessorArgs(weight_buffer).append_to(reader_compile_time_args);
    tt::tt_metal::TensorAccessorArgs(bias_buffer).append_to(reader_compile_time_args);
    std::map<std::string, std::string> defines;
    defines["REDUCE_OP"] = "PoolType::SUM";
    defines["REDUCE_DIM"] = "ReduceDim::REDUCE_ROW";

    desc.kernels.push_back(
        make_reader_kernel_descriptor(all_cores, reader_compile_time_args, defines, kReaderKernelPath));

    std::vector<uint32_t> writer_compile_time_args{block_size, Wt};
    tt::tt_metal::TensorAccessorArgs(output_buffer).append_to(writer_compile_time_args);
    desc.kernels.push_back(
        make_writer_kernel_descriptor(all_cores, writer_compile_time_args, defines, kWriterKernelPath));

    std::vector<uint32_t> compute_group_1_args = {num_rows_per_core_group_1, block_size, Wt};
    desc.kernels.push_back(make_compute_kernel_descriptor(
        core_group_1, compute_group_1_args, defines, kComputeKernelPath, /*fp32_dest_acc_en=*/true));
    if (!core_group_2.ranges().empty()) {
        std::vector<uint32_t> compute_group_2_args = {num_rows_per_core_group_2, block_size, Wt};
        desc.kernels.push_back(make_compute_kernel_descriptor(
            core_group_2, compute_group_2_args, defines, kComputeKernelPath, /*fp32_dest_acc_en=*/true));
    }

    const uint32_t scaler_fp32_bits = std::bit_cast<uint32_t>(1.0F / static_cast<float>(input.logical_shape()[-1]));
    const uint32_t eps_fp32_bits = std::bit_cast<uint32_t>(args.epsilon);

    // Slots as named at the top of the file; override_runtime_arguments() rewrites the addresses on every cache hit.
    // The row split, the scaler (from the input width) and eps all follow from the hashed input spec and epsilon. A
    // group-2 core only exists when the group-2 compute kernel was pushed.
    auto& reader = desc.kernels[kReaderKernelIdx];
    auto& writer = desc.kernels[kWriterKernelIdx];
    for_each_core_with_work(split, [&](const CoreWork& work) {
        const auto& [core, core_index, num_rows, start_row, in_group_1] = work;
        reader.emplace_runtime_args(
            core, {input_buffer, weight_buffer, bias_buffer, num_rows, start_row, scaler_fp32_bits});
        writer.emplace_runtime_args(core, {output_buffer, num_rows, start_row});
        desc.kernels[in_group_1 ? kComputeGroup1KernelIdx : kComputeGroup2KernelIdx].emplace_runtime_args(
            core, {eps_fp32_bits});
    });

    return desc;
}

void PolyNorm3ForwardProgramFactory::override_runtime_arguments(
    tt::tt_metal::Program& program,
    const PolyNorm3FWAttributes& /*operation_attributes*/,
    const PolyNorm3FWTensorArgs& tensor_args,
    PolyNorm3FWTensorReturn& output,
    const std::optional<ttnn::MeshCoordinate>& /*mesh_dispatch_coordinate*/) {
    const WorkSplit split = make_work_split(tensor_args.input);

    const uint32_t input_addr = tensor_args.input.buffer()->address();
    const uint32_t weight_addr = tensor_args.weight.buffer()->address();
    const uint32_t bias_addr = tensor_args.bias.buffer()->address();
    const uint32_t output_addr = output.buffer()->address();

    auto& reader_runtime_args = tt::tt_metal::GetRuntimeArgs(program, kReaderKernelIdx);
    auto& writer_runtime_args = tt::tt_metal::GetRuntimeArgs(program, kWriterKernelIdx);
    for_each_core_with_work(split, [&](const CoreWork& work) {
        const auto& core = work.core;

        auto& reader_args = reader_runtime_args[core.x][core.y];
        reader_args[kInputAddrIdx] = input_addr;
        reader_args[kWeightAddrIdx] = weight_addr;
        reader_args[kBiasAddrIdx] = bias_addr;

        auto& writer_args = writer_runtime_args[core.x][core.y];
        writer_args[kOutputAddrIdx] = output_addr;
    });
}

TTML_ASSERT_DESCRIPTOR_OVERRIDE(PolyNorm3ForwardDeviceOperation, PolyNorm3ForwardProgramFactory);

}  // namespace ttml::metal::ops::polynorm3_fw::device
