// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "softmax_backward_program_factory.hpp"

#include <algorithm>
#include <cstdint>
#include <map>
#include <optional>
#include <string>
#include <tt-logger/tt-logger.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/math.hpp>
#include <tt-metalium/program_descriptors.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include <tt-metalium/tt_backend_api_types.hpp>
#include <utility>
#include <vector>

#include "hostdevcommon/kernel_structs.h"
#include "metal/common/program_utils.hpp"
#include "softmax_backward_device_operation.hpp"

using namespace tt;
using namespace tt::tt_metal;

namespace {

constexpr uint32_t src0_cb_index = tt::CBIndex::c_0;         // softmax_output
constexpr uint32_t src1_cb_index = tt::CBIndex::c_1;         // upstream_grad
constexpr uint32_t ones_cb_index = tt::CBIndex::c_2;         // ones vector for matmul reduction
constexpr uint32_t out_cb_index = tt::CBIndex::c_7;          // output
constexpr uint32_t sum_reduce_cb_index = tt::CBIndex::c_14;  // sum(y * grad)
constexpr uint32_t partial_cb_index = tt::CBIndex::c_15;     // 32 column partials before final reduction

constexpr const char* const kReaderPath =
    "tt-train/sources/ttml/metal/ops/softmax_backward/device/kernels/dataflow/reader_softmax_backward.cpp";
constexpr const char* const kWriterPath =
    "tt-train/sources/ttml/metal/ops/softmax_backward/device/kernels/dataflow/writer_softmax_backward.cpp";
constexpr const char* const kComputePath =
    "tt-train/sources/ttml/metal/ops/softmax_backward/device/kernels/compute/softmax_backward_kernel.cpp";

// Kernel handles are the push order into ProgramDescriptor::kernels: reader, writer, compute.
constexpr uint32_t kReaderKernelIdx = 0U;
constexpr uint32_t kWriterKernelIdx = 1U;

// reader common runtime args
constexpr uint32_t kSoftmaxOutputAddrIdx = 0U;
constexpr uint32_t kUpstreamGradAddrIdx = 1U;
// writer common runtime args
constexpr uint32_t kOutputAddrIdx = 0U;

struct CoreRowAssignment {
    tt::tt_metal::CoreCoord core;
    uint32_t start_row;
    uint32_t num_rows;
};

static std::vector<tt::tt_metal::CoreCoord> get_worker_cores_in_order(
    const std::optional<tt::tt_metal::CoreRangeSet>& sub_core_grids, const tt::tt_metal::IDevice* device) {
    if (sub_core_grids.has_value()) {
        const tt::tt_metal::CoreRangeSet& sub = *sub_core_grids;
        std::vector<tt::tt_metal::CoreCoord> cores;
        cores.reserve(sub.num_cores());
        for (const tt::tt_metal::CoreRange& range : sub.ranges()) {
            for (tt::tt_metal::CoreCoord core : range) {
                cores.push_back(core);
            }
        }
        return cores;
    }
    const tt::tt_metal::CoreCoord grid_size = device->compute_with_storage_grid_size();
    std::vector<tt::tt_metal::CoreCoord> cores;
    cores.reserve(grid_size.x * grid_size.y);
    for (uint32_t x = 0; x < grid_size.x; ++x) {
        for (uint32_t y = 0; y < grid_size.y; ++y) {
            cores.emplace_back(x, y);
        }
    }
    return cores;
}

static void assign_rows_to_cores(
    const std::vector<tt::tt_metal::CoreCoord>& cores_in_order,
    uint32_t num_rows,
    std::vector<tt::tt_metal::CoreRange>& worker_core_ranges,
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
        const tt::tt_metal::CoreCoord& core = cores_in_order[core_idx];
        worker_core_ranges.push_back(tt::tt_metal::CoreRange(core, core));
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

static constexpr uint32_t memory_estimator(uint32_t width_tiles, uint32_t tile_size) {
    return (width_tiles * tile_size) * 3U + (1U * tile_size) * 3U;  // src0, src1, out; sum_reduce, ones, partial
}

static KernelMode get_kernel_mode(uint32_t width_tiles, uint32_t tile_size, const tt::tt_metal::IDevice* device) {
    const uint32_t available_L1_in_bytes =
        device->l1_size_per_core() - device->allocator()->get_base_allocator_addr(tt::tt_metal::HalMemType::L1);
    const uint32_t full_row_memory_needed = memory_estimator(width_tiles, tile_size);
    const uint32_t tiles_per_block = (full_row_memory_needed < available_L1_in_bytes) ? width_tiles : 4U;
    const uint32_t new_estimation_memory_needed = memory_estimator(tiles_per_block, tile_size);
    // Double buffering when total L1 with 2x fits: for streaming (small blocks) and for short rows (e.g. 6000x5).
    // Long full rows (e.g. 127 tiles) get 1x so we stay under L1.
    const uint32_t buffering_multiplier = (new_estimation_memory_needed * 2U <= available_L1_in_bytes) ? 2U : 1U;
    return {buffering_multiplier, new_estimation_memory_needed, tiles_per_block};
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
        dim == softmax_output.logical_shape().rank() - 1,
        "Currently only supporting softmax_backward on last dimension");
    // Tile counts must come from the padded shape: physical_volume includes tile padding,
    // so dividing it by logical H/W overcounts rows and the kernels index past both input
    // tensors and the output buffer. Padded-H tile-rows are processed like any other row;
    // every compute step is per-lane (elementwise mul, matmul-with-ones reduction, COL
    // broadcast), so padding lanes never contaminate valid lanes.
    const auto& padded_shape = softmax_output.padded_shape();
    const uint32_t padded_height = padded_shape[-2];
    const uint32_t padded_width = padded_shape[-1];
    const auto tile = softmax_output.tensor_spec().tile();
    const uint32_t tile_height = tile.get_height();
    const uint32_t tile_width = tile.get_width();
    const uint32_t height_tiles = padded_height / tile_height;
    width_tiles = padded_width / tile_width;
    const uint32_t logical_width = softmax_output.logical_shape()[-1];
    mask_w = logical_width % tile_width;
    const uint64_t num_outer_dims = softmax_output.physical_volume() / padded_height / padded_width;
    num_rows = num_outer_dims * height_tiles;
    input_data_format = datatype_to_dataformat_converter(softmax_output.dtype());
    output_data_format = datatype_to_dataformat_converter(tensor_return_value.dtype());
    // y*grad and reduction tiles use the same format as activations (BFLOAT16 or FLOAT32 today).
    intermed_data_format = input_data_format;
    input_tile_size = tile_size(input_data_format);
    output_tile_size = tile_size(output_data_format);
    intermed_tile_size = tile_size(intermed_data_format);
}

tt::tt_metal::ProgramDescriptor SoftmaxBackwardFactory::create_descriptor(
    const SoftmaxBackwardParams& operation_attributes,
    const SoftmaxBackwardInputs& tensor_args,
    ttnn::Tensor& tensor_return_value) {
    const ttnn::Tensor& softmax_output = tensor_args.softmax_output;
    const ttnn::Tensor& upstream_grad = tensor_args.upstream_grad;
    auto* device = softmax_output.device();
    tt::tt_metal::ProgramDescriptor desc;

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
        "SoftmaxBackward: Using {} approach | Shape: {}x{} tiles | {}x buffering | Estimated L1: {} KB",
        tiles_per_block == width_tiles ? "NON-STREAMING" : "STREAMING",
        num_rows,
        width_tiles,
        buffering_multiplier,
        required_memory_bytes / 1024);

    // Collect worker cores in deterministic order and assign rows to each.
    std::vector<tt::tt_metal::CoreRange> worker_core_ranges;
    std::vector<CoreRowAssignment> core_row_assignments;
    const std::vector<tt::tt_metal::CoreCoord> cores_in_order =
        get_worker_cores_in_order(operation_attributes.sub_core_grids, device);
    assign_rows_to_cores(cores_in_order, num_rows, worker_core_ranges, core_row_assignments);
    TT_FATAL(!worker_core_ranges.empty(), "SoftmaxBackward: no cores have work");
    const tt::tt_metal::CoreRangeSet worker_cores(worker_core_ranges);
    // CreateCircularBuffer merged CB core ranges (kernels keep the unmerged set); merge here to match.
    const tt::tt_metal::CoreRangeSet cb_cores = worker_cores.merge_ranges();

    const uint32_t block_cb_num_tiles = buffering_multiplier * tiles_per_block;

    desc.cbs.push_back(
        make_cb_descriptor(cb_cores, src0_cb_index, input_data_format, input_tile_size, block_cb_num_tiles));
    desc.cbs.push_back(
        make_cb_descriptor(cb_cores, src1_cb_index, input_data_format, input_tile_size, block_cb_num_tiles));
    desc.cbs.push_back(make_cb_descriptor(cb_cores, ones_cb_index, intermed_data_format, intermed_tile_size, 1U));
    desc.cbs.push_back(
        make_cb_descriptor(cb_cores, out_cb_index, output_data_format, output_tile_size, block_cb_num_tiles));
    desc.cbs.push_back(make_cb_descriptor(cb_cores, sum_reduce_cb_index, intermed_data_format, intermed_tile_size, 1U));
    desc.cbs.push_back(make_cb_descriptor(cb_cores, partial_cb_index, intermed_data_format, intermed_tile_size, 1U));

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
        sum_reduce_cb_index,
        ones_cb_index,
        partial_cb_index,
        width_tiles,
        tiles_per_block};

    std::map<std::string, std::string> compute_defines = {
        {"BROADCAST_TYPE", "BroadcastType::COL"}, {"FP32_DEST_ACC_EN", "1"}};

    KernelDescriptor reader =
        make_reader_kernel_descriptor(worker_cores, reader_compile_time_args, /*defines=*/{}, kReaderPath);
    KernelDescriptor writer =
        make_writer_kernel_descriptor(worker_cores, writer_compile_time_args, /*defines=*/{}, kWriterPath);
    // Precise math: fp32 dest accumulation, HiFi4, no approximation.
    KernelDescriptor compute = make_compute_kernel_descriptor(
        worker_cores, compute_compile_time_args, compute_defines, kComputePath, /*fp32_dest_acc_en=*/true);

    // Buffer addresses travel in common runtime args (slots named at the top of the file); override_runtime_arguments()
    // rewrites them on every cache hit. start_row/num_rows follow from the hashed tensor specs and sub_core_grids.
    reader.emplace_common_runtime_args({softmax_output.buffer(), upstream_grad.buffer()});
    writer.emplace_common_runtime_args({tensor_return_value.buffer()});
    for (const CoreRowAssignment& a : core_row_assignments) {
        reader.emplace_runtime_args(a.core, {a.start_row, a.num_rows});
        writer.emplace_runtime_args(a.core, {a.start_row, a.num_rows});
        compute.emplace_runtime_args(a.core, {a.num_rows});
    }

    desc.kernels.push_back(std::move(reader));  // kReaderKernelIdx
    desc.kernels.push_back(std::move(writer));  // kWriterKernelIdx
    desc.kernels.push_back(std::move(compute));
    return desc;
}

void SoftmaxBackwardFactory::override_runtime_arguments(
    tt::tt_metal::Program& program,
    const SoftmaxBackwardParams& /*operation_attributes*/,
    const SoftmaxBackwardInputs& tensor_args,
    ttnn::Tensor& tensor_return_value,
    const std::optional<ttnn::MeshCoordinate>& /*mesh_dispatch_coordinate*/) {
    auto& reader_common_args = tt::tt_metal::GetCommonRuntimeArgs(program, kReaderKernelIdx);
    reader_common_args[kSoftmaxOutputAddrIdx] = tensor_args.softmax_output.buffer()->address();
    reader_common_args[kUpstreamGradAddrIdx] = tensor_args.upstream_grad.buffer()->address();

    auto& writer_common_args = tt::tt_metal::GetCommonRuntimeArgs(program, kWriterKernelIdx);
    writer_common_args[kOutputAddrIdx] = tensor_return_value.buffer()->address();
}

TTML_ASSERT_DESCRIPTOR_OVERRIDE(SoftmaxBackwardDeviceOperation, SoftmaxBackwardFactory);

}  // namespace ttml::metal::ops::softmax_backward::device
