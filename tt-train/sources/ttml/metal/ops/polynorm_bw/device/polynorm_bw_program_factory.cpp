// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "polynorm_bw_program_factory.hpp"

#include <bit>
#include <enchantum/enchantum.hpp>
#include <optional>
#include <tt-metalium/tensor_accessor_args.hpp>

#include "metal/common/program_utils.hpp"
#include "polynorm_bw_device_operation.hpp"

namespace {

constexpr auto kReaderKernelPath =
    "tt-train/sources/ttml/metal/ops/polynorm_bw/device/kernels/dataflow/reader_polynorm_bw_interleaved_start_id.cpp";
constexpr auto kDLdxWriterKernelPath =
    "tt-train/sources/ttml/metal/ops/polynorm_bw/device/kernels/dataflow/writer_polynorm_bw_interleaved_start_id.cpp";
constexpr auto kComputeKernelPath =
    "tt-train/sources/ttml/metal/ops/polynorm_bw/device/kernels/compute/polynorm_bw_kernel.cpp";

// Kernel handles are the push order into ProgramDescriptor::kernels: reader, writer, then one compute kernel per core
// group (group 2 only when it has cores).
constexpr uint32_t kReaderKernelIdx = 0U;
constexpr uint32_t kDLdxWriterKernelIdx = 1U;
constexpr uint32_t kComputeGroup1KernelIdx = 2U;
constexpr uint32_t kComputeGroup2KernelIdx = 3U;

// reader runtime args
constexpr uint32_t kInputAddrIdx = 0U;
constexpr uint32_t kDLdoutAddrIdx = 1U;
constexpr uint32_t kWeightAddrIdx = 2U;
// writer runtime args
constexpr uint32_t kDLdxOutputAddrIdx = 0U;
constexpr uint32_t kPackedPartialsOutputAddrIdx = 1U;

// Data input CBs (shared across all sequential passes)
constexpr auto kXCbIndex = tt::CBIndex::c_0;
constexpr auto kDoutCbIndex = tt::CBIndex::c_1;
constexpr auto kDbAccCbIndex = tt::CBIndex::c_2;

// Scalar/constant CBs
constexpr auto kScalerCbIndex = tt::CBIndex::c_3;
constexpr auto kMatmulReduceCbIndex = tt::CBIndex::c_4;
constexpr auto kOneCbIndex = tt::CBIndex::c_5;
constexpr auto kW0CbIndex = tt::CBIndex::c_6;
constexpr auto kW1CbIndex = tt::CBIndex::c_7;
constexpr auto kW2CbIndex = tt::CBIndex::c_8;
// Reader-only scratch CB used to read weight scalars before publishing constants.
constexpr auto kWeightScalarScratchCbIndex = tt::CBIndex::c_23;

// Intermediate per-row sum tiles (Float32)
constexpr auto kSumX2CbIndex = tt::CBIndex::c_9;
constexpr auto kSumX4CbIndex = tt::CBIndex::c_10;
constexpr auto kSumX6CbIndex = tt::CBIndex::c_11;
constexpr auto kSumXdoutCbIndex = tt::CBIndex::c_12;
constexpr auto kSumX2doutCbIndex = tt::CBIndex::c_13;
constexpr auto kSumX3doutCbIndex = tt::CBIndex::c_14;

// Scalar results (bfloat16)
constexpr auto kInvRmsXCbIndex = tt::CBIndex::c_15;
constexpr auto kInvRmsX2CbIndex = tt::CBIndex::c_16;
constexpr auto kInvRmsX3CbIndex = tt::CBIndex::c_17;

// Per-row correction coefficients γ_k = inv_rms_k² · g_k · (1/N), bfloat16 storage.
constexpr auto kGamma1CbIndex = tt::CBIndex::c_18;
constexpr auto kGamma2CbIndex = tt::CBIndex::c_19;
constexpr auto kGamma3CbIndex = tt::CBIndex::c_20;

// Preweighted inv_rms (three 1-tile Float32 CBs) — one per polynomial branch:
//   c_24 = w2 * inv_rms_x    (linear)
//   c_25 = w1 * inv_rms_x2   (quadratic)
//   c_26 = w0 * inv_rms_x3   (cubic)
constexpr auto kWeightedInvRmsXCbIndex = tt::CBIndex::c_24;
constexpr auto kWeightedInvRmsX2CbIndex = tt::CBIndex::c_25;
constexpr auto kWeightedInvRmsX3CbIndex = tt::CBIndex::c_26;

// Output CBs
constexpr auto kOutputCbIndex = tt::CBIndex::c_21;
constexpr auto kPackedPartialsOutputCbIndex = tt::CBIndex::c_22;

constexpr uint32_t kNumOneTile = 1U;

}  // namespace

namespace ttml::metal::ops::polynorm3_bw::device {

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

tt::tt_metal::ProgramDescriptor PolyNorm3BackwardProgramFactory::create_descriptor(
    const PolyNorm3BWAttributes& args, const PolyNorm3BWTensorArgs& tensor_args, PolyNorm3BWTensorReturn& output) {
    const auto& input = tensor_args.input;
    const auto& dL_dout = tensor_args.dL_dout;
    auto* device = input.device();

    tt::tt_metal::ProgramDescriptor desc;
    const tt::DataFormat data_format = datatype_to_dataformat_converter(input.dtype());
    TT_FATAL(data_format == tt::DataFormat::Float16_b, "PolyNormBackward currently supports BF16 input only");
    const uint32_t bfloat16_tile_size = tt::tile_size(tt::DataFormat::Float16_b);
    const uint32_t float32_tile_size = tt::tile_size(tt::DataFormat::Float32);

    const auto padded_tensor_shape = input.padded_shape();
    TT_FATAL(padded_tensor_shape.rank() == 4U, "Input tensor must be 4D");
    const uint32_t Wt = padded_tensor_shape[-1] / tt::constants::TILE_WIDTH;
    const uint32_t Ht = padded_tensor_shape[-2] / tt::constants::TILE_HEIGHT;
    const uint32_t NC = padded_tensor_shape[0] * padded_tensor_shape[1];
    const uint32_t total_rows_to_process = NC * Ht;

    const auto compute_with_storage_grid_size = device->compute_with_storage_grid_size();

    constexpr uint32_t block_size = 4U;  // Fixed block size; tail tiles handled via current_block_size in kernels.
    constexpr uint32_t packed_partials_tiles_per_row = 4U;
    auto [num_cores, all_cores, core_group_1, core_group_2, num_rows_per_core_group_1, num_rows_per_core_group_2] =
        tt::tt_metal::split_work_to_cores(compute_with_storage_grid_size, total_rows_to_process);
    const WorkSplit split = make_work_split(input);
    check_work_split(
        split, num_cores, core_group_1, core_group_2, num_rows_per_core_group_1, num_rows_per_core_group_2);

    // Data input CBs (shared across all passes)
    desc.cbs.push_back(make_cb_descriptor(all_cores, kXCbIndex, data_format, bfloat16_tile_size, block_size));
    desc.cbs.push_back(make_cb_descriptor(all_cores, kDoutCbIndex, data_format, bfloat16_tile_size, block_size));
    desc.cbs.push_back(
        make_cb_descriptor(all_cores, kDbAccCbIndex, tt::DataFormat::Float32, float32_tile_size, kNumOneTile));

    // Scalar/constant CBs
    desc.cbs.push_back(
        make_cb_descriptor(all_cores, kScalerCbIndex, tt::DataFormat::Float32, float32_tile_size, kNumOneTile));
    desc.cbs.push_back(
        make_cb_descriptor(all_cores, kMatmulReduceCbIndex, tt::DataFormat::Float32, float32_tile_size, kNumOneTile));
    desc.cbs.push_back(
        make_cb_descriptor(all_cores, kOneCbIndex, tt::DataFormat::Float32, float32_tile_size, kNumOneTile));
    desc.cbs.push_back(make_cb_descriptor(all_cores, kW0CbIndex, data_format, bfloat16_tile_size, kNumOneTile));
    desc.cbs.push_back(make_cb_descriptor(all_cores, kW1CbIndex, data_format, bfloat16_tile_size, kNumOneTile));
    desc.cbs.push_back(make_cb_descriptor(all_cores, kW2CbIndex, data_format, bfloat16_tile_size, kNumOneTile));
    desc.cbs.push_back(
        make_cb_descriptor(all_cores, kWeightScalarScratchCbIndex, data_format, bfloat16_tile_size, kNumOneTile));

    // Intermediate per-row sum tiles (Float32)
    desc.cbs.push_back(
        make_cb_descriptor(all_cores, kSumX2CbIndex, tt::DataFormat::Float32, float32_tile_size, kNumOneTile));
    desc.cbs.push_back(
        make_cb_descriptor(all_cores, kSumX4CbIndex, tt::DataFormat::Float32, float32_tile_size, kNumOneTile));
    desc.cbs.push_back(
        make_cb_descriptor(all_cores, kSumX6CbIndex, tt::DataFormat::Float32, float32_tile_size, kNumOneTile));
    desc.cbs.push_back(
        make_cb_descriptor(all_cores, kSumXdoutCbIndex, tt::DataFormat::Float32, float32_tile_size, kNumOneTile));
    desc.cbs.push_back(
        make_cb_descriptor(all_cores, kSumX2doutCbIndex, tt::DataFormat::Float32, float32_tile_size, kNumOneTile));
    desc.cbs.push_back(
        make_cb_descriptor(all_cores, kSumX3doutCbIndex, tt::DataFormat::Float32, float32_tile_size, kNumOneTile));

    // Scalar results (bfloat16)
    desc.cbs.push_back(make_cb_descriptor(all_cores, kInvRmsXCbIndex, data_format, bfloat16_tile_size, kNumOneTile));
    desc.cbs.push_back(make_cb_descriptor(all_cores, kInvRmsX2CbIndex, data_format, bfloat16_tile_size, kNumOneTile));
    desc.cbs.push_back(make_cb_descriptor(all_cores, kInvRmsX3CbIndex, data_format, bfloat16_tile_size, kNumOneTile));

    // Per-row correction γ_k tiles (bfloat16, one per branch).
    desc.cbs.push_back(make_cb_descriptor(all_cores, kGamma1CbIndex, data_format, bfloat16_tile_size, kNumOneTile));
    desc.cbs.push_back(make_cb_descriptor(all_cores, kGamma2CbIndex, data_format, bfloat16_tile_size, kNumOneTile));
    desc.cbs.push_back(make_cb_descriptor(all_cores, kGamma3CbIndex, data_format, bfloat16_tile_size, kNumOneTile));

    // Preweighted inv_rms tiles (Float32, one per branch).
    desc.cbs.push_back(make_cb_descriptor(
        all_cores, kWeightedInvRmsXCbIndex, tt::DataFormat::Float32, float32_tile_size, kNumOneTile));
    desc.cbs.push_back(make_cb_descriptor(
        all_cores, kWeightedInvRmsX2CbIndex, tt::DataFormat::Float32, float32_tile_size, kNumOneTile));
    desc.cbs.push_back(make_cb_descriptor(
        all_cores, kWeightedInvRmsX3CbIndex, tt::DataFormat::Float32, float32_tile_size, kNumOneTile));

    // Output CBs
    desc.cbs.push_back(make_cb_descriptor(all_cores, kOutputCbIndex, data_format, bfloat16_tile_size, block_size));
    desc.cbs.push_back(make_cb_descriptor(
        all_cores,
        kPackedPartialsOutputCbIndex,
        tt::DataFormat::Float32,
        float32_tile_size,
        packed_partials_tiles_per_row));

    auto* input_buffer = input.buffer();
    TT_FATAL(
        input_buffer->buffer_type() == ttnn::BufferType::DRAM,
        "Input buffer must be in DRAM. Input buffer of type {}",
        enchantum::to_string(input_buffer->buffer_type()));

    auto* dL_dout_buffer = dL_dout.buffer();
    TT_FATAL(
        dL_dout_buffer->buffer_type() == ttnn::BufferType::DRAM,
        "dL_dout buffer must be in DRAM. dL_dout buffer of type {}",
        enchantum::to_string(dL_dout_buffer->buffer_type()));

    const auto& weight = tensor_args.weight;
    auto* weight_buffer = weight.buffer();
    TT_FATAL(
        weight_buffer->buffer_type() == ttnn::BufferType::DRAM,
        "Weight buffer must be in DRAM. Weight buffer of type {}",
        enchantum::to_string(weight_buffer->buffer_type()));

    auto* dL_dx_output_buffer = output[0].buffer();
    TT_FATAL(
        dL_dx_output_buffer->buffer_type() == ttnn::BufferType::DRAM,
        "dL_dx output buffer must be in DRAM. Output buffer of type {}",
        enchantum::to_string(dL_dx_output_buffer->buffer_type()));
    auto* packed_partials_output_buffer = output[1].buffer();
    TT_FATAL(
        packed_partials_output_buffer->buffer_type() == ttnn::BufferType::DRAM,
        "Packed partials output buffer must be in DRAM. Output buffer of type {}",
        enchantum::to_string(packed_partials_output_buffer->buffer_type()));

    std::vector<uint32_t> reader_compile_time_args{block_size, Wt};
    tt::tt_metal::TensorAccessorArgs(input_buffer).append_to(reader_compile_time_args);
    tt::tt_metal::TensorAccessorArgs(dL_dout_buffer).append_to(reader_compile_time_args);
    tt::tt_metal::TensorAccessorArgs(weight_buffer).append_to(reader_compile_time_args);
    std::map<std::string, std::string> defines;
    defines["REDUCE_OP"] = "PoolType::SUM";
    defines["REDUCE_DIM"] = "ReduceDim::REDUCE_ROW";

    desc.kernels.push_back(
        make_reader_kernel_descriptor(all_cores, reader_compile_time_args, defines, kReaderKernelPath));

    std::vector<uint32_t> dL_dx_writer_compile_time_args{block_size, Wt, packed_partials_tiles_per_row};
    tt::tt_metal::TensorAccessorArgs(dL_dx_output_buffer).append_to(dL_dx_writer_compile_time_args);
    tt::tt_metal::TensorAccessorArgs(packed_partials_output_buffer).append_to(dL_dx_writer_compile_time_args);
    desc.kernels.push_back(
        make_writer_kernel_descriptor(all_cores, dL_dx_writer_compile_time_args, defines, kDLdxWriterKernelPath));

    // HiFi4, fp32 dest accumulation, no math approximation; every other compute config field at its default.
    std::vector<uint32_t> compute_group_1_args = {num_rows_per_core_group_1, block_size};
    desc.kernels.push_back(make_compute_kernel_descriptor(
        core_group_1, compute_group_1_args, defines, kComputeKernelPath, /*fp32_dest_acc_en=*/true));
    if (!core_group_2.ranges().empty()) {
        std::vector<uint32_t> compute_group_2_args = {num_rows_per_core_group_2, block_size};
        desc.kernels.push_back(make_compute_kernel_descriptor(
            core_group_2, compute_group_2_args, defines, kComputeKernelPath, /*fp32_dest_acc_en=*/true));
    }

    const uint32_t scaler_fp32_bits = std::bit_cast<uint32_t>(1.0F / static_cast<float>(input.logical_shape()[-1]));
    const uint32_t eps_fp32_bits = std::bit_cast<uint32_t>(args.epsilon);

    // Per-core runtime args. Slots as named at the top of the file; override_runtime_arguments() rewrites the
    // addresses on every cache hit. The other args (row split, Wt, scaler, eps) depend only on hashed values.
    // References taken after the last kernel push (a later push could reallocate desc.kernels); a group-2 core only
    // exists when the group-2 compute kernel was pushed.
    auto& reader = desc.kernels[kReaderKernelIdx];
    auto& dL_dx_writer = desc.kernels[kDLdxWriterKernelIdx];
    for_each_core_with_work(split, [&](const CoreWork& work) {
        const auto& [core, core_index, num_rows, start_row, in_group_1] = work;
        // Slot 6 (eps) is read and dropped by the reader (compute takes eps); kept so the arg layout is unchanged.
        reader.emplace_runtime_args(
            core, {input_buffer, dL_dout_buffer, weight_buffer, num_rows, start_row, scaler_fp32_bits, eps_fp32_bits});
        dL_dx_writer.emplace_runtime_args(
            core, {dL_dx_output_buffer, packed_partials_output_buffer, num_rows, start_row});
        desc.kernels[in_group_1 ? kComputeGroup1KernelIdx : kComputeGroup2KernelIdx].emplace_runtime_args(
            core, {Wt, eps_fp32_bits});
    });

    return desc;
}

void PolyNorm3BackwardProgramFactory::override_runtime_arguments(
    tt::tt_metal::Program& program,
    const PolyNorm3BWAttributes& /*operation_attributes*/,
    const PolyNorm3BWTensorArgs& tensor_args,
    PolyNorm3BWTensorReturn& output,
    const std::optional<ttnn::MeshCoordinate>& /*mesh_dispatch_coordinate*/) {
    const WorkSplit split = make_work_split(tensor_args.input);

    const uint32_t input_addr = tensor_args.input.buffer()->address();
    const uint32_t dL_dout_addr = tensor_args.dL_dout.buffer()->address();
    const uint32_t weight_addr = tensor_args.weight.buffer()->address();
    const uint32_t dL_dx_output_addr = output[0].buffer()->address();
    const uint32_t packed_partials_output_addr = output[1].buffer()->address();

    auto& reader_runtime_args = tt::tt_metal::GetRuntimeArgs(program, kReaderKernelIdx);
    auto& dL_dx_writer_runtime_args = tt::tt_metal::GetRuntimeArgs(program, kDLdxWriterKernelIdx);
    for_each_core_with_work(split, [&](const CoreWork& work) {
        const auto& core = work.core;

        auto& reader_args = reader_runtime_args[core.x][core.y];
        reader_args[kInputAddrIdx] = input_addr;
        reader_args[kDLdoutAddrIdx] = dL_dout_addr;
        reader_args[kWeightAddrIdx] = weight_addr;

        auto& dL_dx_writer_args = dL_dx_writer_runtime_args[core.x][core.y];
        dL_dx_writer_args[kDLdxOutputAddrIdx] = dL_dx_output_addr;
        dL_dx_writer_args[kPackedPartialsOutputAddrIdx] = packed_partials_output_addr;
    });
}

TTML_ASSERT_DESCRIPTOR_OVERRIDE(PolyNorm3BackwardDeviceOperation, PolyNorm3BackwardProgramFactory);

}  // namespace ttml::metal::ops::polynorm3_bw::device
