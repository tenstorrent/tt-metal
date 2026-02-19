// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "softmax_backward_program_factory.hpp"

#include <algorithm>
#include <cstdint>
#include <optional>
#include <tt-logger/tt-logger.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/kernel_types.hpp>
#include <tt-metalium/math.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include <tt-metalium/tt_backend_api_types.hpp>
#include <utility>
#include <vector>

#include "hostdevcommon/kernel_structs.h"

using namespace tt;
using namespace tt::tt_metal;

namespace {

constexpr uint32_t src0_cb_index = tt::CBIndex::c_0;         // softmax_output
constexpr uint32_t src1_cb_index = tt::CBIndex::c_1;         // upstream_grad
constexpr uint32_t ones_cb_index = tt::CBIndex::c_2;         // ones vector for matmul reduction
constexpr uint32_t out_cb_index = tt::CBIndex::c_7;          // output
constexpr uint32_t ygrad_cb_index = tt::CBIndex::c_13;       // y * grad
constexpr uint32_t sum_reduce_cb_index = tt::CBIndex::c_14;  // sum(y * grad) - accumulated
constexpr uint32_t block_sum_cb_index = tt::CBIndex::c_15;   // block sum temporary

constexpr const char* const kReaderPath =
    "tt-train/sources/ttml/metal/ops/softmax_backward/device/kernels/dataflow/reader_softmax_backward.cpp";
constexpr const char* const kWriterPath =
    "tt-train/sources/ttml/metal/ops/softmax_backward/device/kernels/dataflow/writer_softmax_backward.cpp";
constexpr const char* const kComputePath =
    "tt-train/sources/ttml/metal/ops/softmax_backward/device/kernels/compute/softmax_backward_kernel.cpp";

struct CoreRowAssignment {
    CoreCoord core;
    uint32_t start_row;
    uint32_t num_rows;
};

static std::vector<CoreCoord> get_worker_cores_in_order(
    const std::optional<CoreRangeSet>& sub_core_grids, const tt::tt_metal::IDevice* device) {
    if (sub_core_grids.has_value()) {
        const CoreRangeSet& sub = *sub_core_grids;
        std::vector<CoreCoord> cores;
        cores.reserve(sub.num_cores());
        for (const CoreRange& range : sub.ranges()) {
            for (CoreCoord core : range) {
                cores.push_back(core);
            }
        }
        return cores;
    }
    const CoreCoord grid_size = device->compute_with_storage_grid_size();
    std::vector<CoreCoord> cores;
    cores.reserve(grid_size.x * grid_size.y);
    for (uint32_t x = 0; x < grid_size.x; ++x) {
        for (uint32_t y = 0; y < grid_size.y; ++y) {
            cores.emplace_back(x, y);
        }
    }
    return cores;
}

static void assign_rows_to_cores(
    const std::vector<CoreCoord>& cores_in_order,
    uint32_t num_rows,
    std::vector<CoreRange>& worker_core_ranges,
    std::vector<CoreRowAssignment>& core_row_assignments) {
    const uint32_t num_cores = static_cast<uint32_t>(cores_in_order.size());
    if (num_cores == 0)
        return;
    const uint32_t rows_per_core = tt::div_up(num_rows, num_cores);
    for (uint32_t core_idx = 0; core_idx < num_cores; ++core_idx) {
        const uint32_t start_row = core_idx * rows_per_core;
        if (start_row >= num_rows)
            continue;
        const uint32_t end_row = std::min(start_row + rows_per_core, num_rows);
        const uint32_t num_rows_this_core = end_row - start_row;
        if (num_rows_this_core == 0)
            continue;
        const CoreCoord& core = cores_in_order[core_idx];
        worker_core_ranges.push_back(CoreRange(core, core));
        core_row_assignments.push_back({core, start_row, num_rows_this_core});
    }
}

}  // namespace

namespace ttml::metal::ops::softmax_backward::device {

struct KernelMode {
    uint32_t buffering_multiplier;
    uint32_t required_memory_bytes;
    uint32_t tiles_per_block;
};

static KernelMode get_kernel_mode(uint32_t width_tiles, uint32_t tile_size, const tt::tt_metal::IDevice* device) {
    const uint32_t available_L1_in_bytes =
        device->l1_size_per_core() - device->allocator()->get_base_allocator_addr(tt::tt_metal::HalMemType::L1);
    const uint32_t memory_needed =
        (width_tiles * tile_size) * 4U + (1U * tile_size) * 2U;  // src0, src1, out, ygrad; sum_reduce, ones
    const uint32_t tiles_per_block = (memory_needed < available_L1_in_bytes) ? width_tiles : 1U;
    const uint32_t buffering_multiplier = (memory_needed * 2U > available_L1_in_bytes) ? 1U : 2U;
    return {buffering_multiplier, memory_needed, tiles_per_block};
}

static void get_tensor_properties(
    const ttnn::Tensor& softmax_output,
    const SoftmaxBackwardParams& operation_attributes,
    uint32_t& num_rows,
    uint32_t& width_tiles,
    uint32_t& mask_w,
    tt::DataFormat& input_data_format,
    tt::DataFormat& output_data_format,
    tt::DataFormat& intermed_data_format,
    uint32_t& input_tile_size,
    uint32_t& output_tile_size,
    uint32_t& intermed_tile_size,
    const ttnn::Tensor& tensor_return_value) {
    const uint32_t dim = operation_attributes.dim;
    TT_FATAL(
        dim == softmax_output.logical_shape().rank() - 1 || dim == static_cast<uint32_t>(-1),
        "Currently only supporting softmax_backward on last dimension");
    const uint32_t height = softmax_output.logical_shape()[-2];
    const uint32_t width = softmax_output.logical_shape()[-1];
    const auto tile = softmax_output.tensor_spec().tile();
    const uint32_t tile_height = tile.get_height();
    const uint32_t tile_width = tile.get_width();
    const uint32_t height_tiles = height / tile_height;
    width_tiles = tt::div_up(width, tile_width);
    const uint32_t logical_width = softmax_output.logical_shape()[-1];
    mask_w = logical_width % tile_width;
    const uint64_t num_outer_dims = softmax_output.physical_volume() / height / width;
    num_rows = num_outer_dims * height_tiles;
    input_data_format = datatype_to_dataformat_converter(softmax_output.dtype());
    output_data_format = datatype_to_dataformat_converter(tensor_return_value.dtype());
    intermed_data_format = tt::DataFormat::Float16_b;
    input_tile_size = tile_size(input_data_format);
    output_tile_size = tile_size(output_data_format);
    intermed_tile_size = tile_size(intermed_data_format);
}

static tt::tt_metal::ComputeConfig precise(
    std::vector<uint32_t> compile_time_args, std::map<std::string, std::string> defines) {
    tt::tt_metal::ComputeConfig config;
    config.fp32_dest_acc_en = true;
    config.math_approx_mode = false;
    config.math_fidelity = MathFidelity::HiFi4;
    config.compile_args = std::move(compile_time_args);
    config.defines = std::move(defines);
    return config;
}

SoftmaxBackwardFactory::cached_program_t SoftmaxBackwardFactory::create(
    const SoftmaxBackwardParams& operation_attributes,
    const SoftmaxBackwardInputs& tensor_args,
    ttnn::Tensor& tensor_return_value) {
    const ttnn::Tensor& softmax_output = tensor_args.softmax_output;
    const ttnn::Tensor& upstream_grad = tensor_args.upstream_grad;
    auto* device = softmax_output.device();
    Program program = CreateProgram();

    uint32_t num_rows, width_tiles, mask_w;
    DataFormat input_data_format, output_data_format, intermed_data_format;
    uint32_t input_tile_size, output_tile_size, intermed_tile_size;
    get_tensor_properties(
        softmax_output,
        operation_attributes,
        num_rows,
        width_tiles,
        mask_w,
        input_data_format,
        output_data_format,
        intermed_data_format,
        input_tile_size,
        output_tile_size,
        intermed_tile_size,
        tensor_return_value);

    const auto [buffering_multiplier, required_memory_bytes, tiles_per_block] =
        get_kernel_mode(width_tiles, intermed_tile_size, device);

    log_debug(
        tt::LogOp,
        "SoftmaxBackward: Using {} kernel | Shape: {}x{} tiles | Estimated L1: {} KB",
        tiles_per_block == width_tiles ? "NON-STREAMING" : "STREAMING",
        num_rows,
        width_tiles,
        required_memory_bytes / 1024);

    // Collect worker cores in deterministic order and assign rows to each.
    std::vector<CoreRange> worker_core_ranges;
    std::vector<CoreRowAssignment> core_row_assignments;
    const std::vector<CoreCoord> cores_in_order =
        get_worker_cores_in_order(operation_attributes.sub_core_grids, device);
    assign_rows_to_cores(cores_in_order, num_rows, worker_core_ranges, core_row_assignments);
    TT_FATAL(!worker_core_ranges.empty(), "SoftmaxBackward: no cores have work");
    const CoreRangeSet worker_cores(worker_core_ranges);

    const uint32_t block_cb_size_in0 = buffering_multiplier * tiles_per_block * input_tile_size;
    const uint32_t block_cb_size_out = buffering_multiplier * tiles_per_block * output_tile_size;
    const uint32_t block_cb_size_intermed0 = buffering_multiplier * tiles_per_block * intermed_tile_size;

    auto c_in0_config = CircularBufferConfig(block_cb_size_in0, {{src0_cb_index, input_data_format}})
                            .set_page_size(src0_cb_index, input_tile_size);
    CreateCircularBuffer(program, worker_cores, c_in0_config);
    auto c_in1_config = CircularBufferConfig(block_cb_size_in0, {{src1_cb_index, input_data_format}})
                            .set_page_size(src1_cb_index, input_tile_size);
    CreateCircularBuffer(program, worker_cores, c_in1_config);
    auto c_scaler_config = CircularBufferConfig(intermed_tile_size, {{ones_cb_index, intermed_data_format}})
                               .set_page_size(ones_cb_index, intermed_tile_size);
    CreateCircularBuffer(program, worker_cores, c_scaler_config);
    auto c_out_config = CircularBufferConfig(block_cb_size_out, {{out_cb_index, output_data_format}})
                            .set_page_size(out_cb_index, output_tile_size);
    CreateCircularBuffer(program, worker_cores, c_out_config);
    auto c_ygrad_config = CircularBufferConfig(block_cb_size_intermed0, {{ygrad_cb_index, intermed_data_format}})
                              .set_page_size(ygrad_cb_index, intermed_tile_size);
    CreateCircularBuffer(program, worker_cores, c_ygrad_config);
    auto c_sum_reduce_config = CircularBufferConfig(intermed_tile_size, {{sum_reduce_cb_index, intermed_data_format}})
                                   .set_page_size(sum_reduce_cb_index, intermed_tile_size);
    CreateCircularBuffer(program, worker_cores, c_sum_reduce_config);
    auto c_block_sum_config = CircularBufferConfig(intermed_tile_size, {{block_sum_cb_index, intermed_data_format}})
                                  .set_page_size(block_sum_cb_index, intermed_tile_size);
    CreateCircularBuffer(program, worker_cores, c_block_sum_config);

    std::vector<uint32_t> reader_compile_time_args = {
        src0_cb_index, src1_cb_index, ones_cb_index, width_tiles, tiles_per_block, mask_w};
    TensorAccessorArgs(softmax_output.buffer()).append_to(reader_compile_time_args);
    TensorAccessorArgs(upstream_grad.buffer()).append_to(reader_compile_time_args);

    std::vector<uint32_t> writer_compile_time_args = {out_cb_index, width_tiles, tiles_per_block};
    TensorAccessorArgs(tensor_return_value.buffer()).append_to(writer_compile_time_args);

    const std::vector<uint32_t> compute_compile_time_args = {
        src0_cb_index,
        src1_cb_index,
        out_cb_index,
        ygrad_cb_index,
        sum_reduce_cb_index,
        ones_cb_index,
        block_sum_cb_index,
        width_tiles,
        tiles_per_block};

    std::map<std::string, std::string> compute_defines = {{"BROADCAST_TYPE", "BroadcastType::COL"}};
    const ComputeConfig wconf = precise(compute_compile_time_args, compute_defines);

    auto reader_kernel_id =
        CreateKernel(program, kReaderPath, worker_cores, ReaderDataMovementConfig(reader_compile_time_args));
    auto writer_kernel_id =
        CreateKernel(program, kWriterPath, worker_cores, WriterDataMovementConfig(writer_compile_time_args));
    auto compute_kernel_id = CreateKernel(program, kComputePath, worker_cores, wconf);

    SetCommonRuntimeArgs(
        program, reader_kernel_id, {softmax_output.buffer()->address(), upstream_grad.buffer()->address()});
    SetCommonRuntimeArgs(program, writer_kernel_id, {tensor_return_value.buffer()->address()});
    for (const CoreRowAssignment& a : core_row_assignments) {
        SetRuntimeArgs(program, reader_kernel_id, a.core, {a.start_row, a.num_rows});
        SetRuntimeArgs(program, writer_kernel_id, a.core, {a.start_row, a.num_rows});
        SetRuntimeArgs(program, compute_kernel_id, a.core, {a.num_rows});
    }

    return {
        std::move(program), {.unary_reader_kernel_id = reader_kernel_id, .unary_writer_kernel_id = writer_kernel_id}};
}

void SoftmaxBackwardFactory::override_runtime_arguments(
    cached_program_t& cached_program,
    const SoftmaxBackwardParams& /*operation_attributes*/,
    const SoftmaxBackwardInputs& tensor_args,
    ttnn::Tensor& tensor_return_value) {
    const Program& program = cached_program.program;
    const KernelHandle& reader_kernel_id = cached_program.shared_variables.unary_reader_kernel_id;
    const KernelHandle& writer_kernel_id = cached_program.shared_variables.unary_writer_kernel_id;
    const ttnn::Tensor& softmax_output = tensor_args.softmax_output;
    const ttnn::Tensor& upstream_grad = tensor_args.upstream_grad;
    RuntimeArgsData& reader_common_args = GetCommonRuntimeArgs(program, reader_kernel_id);
    reader_common_args[0] = softmax_output.buffer()->address();
    reader_common_args[1] = upstream_grad.buffer()->address();
    RuntimeArgsData& writer_common_args = GetCommonRuntimeArgs(program, writer_kernel_id);
    writer_common_args[0] = tensor_return_value.buffer()->address();
}

}  // namespace ttml::metal::ops::softmax_backward::device
