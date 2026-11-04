// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "swiglu_elemwise_bw_program_factory.hpp"

#include <cstdint>
#include <enchantum/enchantum.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>

#include "metal/common/program_utils.hpp"
#include "swiglu_elemwise_bw_device_operation.hpp"

namespace {

constexpr auto kReaderKernelPath =
    "tt-train/sources/ttml/metal/ops/swiglu_elemwise_bw/device/kernels/dataflow/"
    "reader_swiglu_elemwise_bw_interleaved.cpp";
constexpr auto kWriterKernelPath =
    "tt-train/sources/ttml/metal/ops/swiglu_elemwise_bw/device/kernels/dataflow/"
    "writer_swiglu_elemwise_bw_interleaved.cpp";
constexpr auto kComputeKernelPath =
    "tt-train/sources/ttml/metal/ops/swiglu_elemwise_bw/device/kernels/compute/swiglu_elemwise_bw_kernel.cpp";

// Kernel handles are the push order into ProgramDescriptor::kernels: reader, writer, compute group 1, then compute
// group 2 when it has cores.
constexpr uint32_t kReaderKernelIdx = 0U;
constexpr uint32_t kWriterKernelIdx = 1U;

// Runtime-arg slots of the buffer addresses, set by create_descriptor() and re-applied by the override.
constexpr uint32_t kLinear1BufferIdx = 0;
constexpr uint32_t kGateBufferIdx = 1U;
constexpr uint32_t kDLProdBufferIdx = 2U;
constexpr uint32_t kDLLinear1BufferIdx = 0;
constexpr uint32_t kDLGateBufferIdx = 1U;

constexpr auto kLinear1CbIndex = tt::CBIndex::c_0;
constexpr auto kGateCbIndex = tt::CBIndex::c_1;
constexpr auto kDLProdCbIndex = tt::CBIndex::c_2;
constexpr auto kDLLinear1CbIndex = tt::CBIndex::c_3;
constexpr auto kDLGateCbIndex = tt::CBIndex::c_4;
constexpr auto kSigmoidCbIndex = tt::CBIndex::c_5;
constexpr auto kScratchCbIndex = tt::CBIndex::c_6;
constexpr auto kSiluGradCbIndex = tt::CBIndex::c_7;

// Units of the work split: rows of tiles (NC * Ht) of the padded shape. create_descriptor() and the override both
// split these over the device grid, so the override patches exactly the cores that got runtime args.
uint32_t num_tile_rows(const ttnn::Shape& padded_shape) {
    const uint32_t Ht = padded_shape[-2] / tt::constants::TILE_HEIGHT;
    const uint32_t NC = padded_shape[0] * padded_shape[1];
    return NC * Ht;
}

}  // namespace

namespace ttml::metal::ops::swiglu_elemwise_bw::device {

tt::tt_metal::ProgramDescriptor SwigluElemwiseBwProgramFactory::create_descriptor(
    const operation_attributes_t& args, const tensor_args_t& tensor_args, tensor_return_value_t& output) {
    const auto& linear1 = tensor_args.linear1;
    const auto& gate = tensor_args.gate;
    const auto& dL_dprod = tensor_args.dL_dprod;

    auto* const device = linear1.device();
    tt::tt_metal::ProgramDescriptor program;

    const tt::DataFormat data_format = datatype_to_dataformat_converter(linear1.dtype());
    const uint32_t tile_size_bytes = tt::tile_size(data_format);

    const auto padded_shape = linear1.padded_shape();
    TT_FATAL(padded_shape.rank() == 4U, "Input tensors must be 4D");
    const uint32_t Wt = padded_shape[-1] / tt::constants::TILE_WIDTH;
    const uint32_t total_rows = num_tile_rows(padded_shape);

    const auto grid_size = device->compute_with_storage_grid_size();
    constexpr uint32_t block_size = 4U;

    const auto [num_cores, all_cores, core_group_1, core_group_2, num_rows_g1, num_rows_g2] =
        tt::tt_metal::split_work_to_cores(grid_size, total_rows);
    const auto split = compute_work_split(grid_size, total_rows);
    check_work_split(split, num_cores, core_group_1, core_group_2, num_rows_g1, num_rows_g2);

    const uint32_t twice_block = 2U * block_size;
    program.cbs.push_back(make_cb_descriptor(all_cores, kLinear1CbIndex, data_format, tile_size_bytes, twice_block));
    program.cbs.push_back(make_cb_descriptor(all_cores, kGateCbIndex, data_format, tile_size_bytes, twice_block));
    program.cbs.push_back(make_cb_descriptor(all_cores, kDLProdCbIndex, data_format, tile_size_bytes, twice_block));
    program.cbs.push_back(make_cb_descriptor(all_cores, kDLLinear1CbIndex, data_format, tile_size_bytes, twice_block));
    program.cbs.push_back(make_cb_descriptor(all_cores, kDLGateCbIndex, data_format, tile_size_bytes, twice_block));
    program.cbs.push_back(make_cb_descriptor(all_cores, kSigmoidCbIndex, data_format, tile_size_bytes, twice_block));
    program.cbs.push_back(make_cb_descriptor(all_cores, kScratchCbIndex, data_format, tile_size_bytes, twice_block));
    program.cbs.push_back(make_cb_descriptor(all_cores, kSiluGradCbIndex, data_format, tile_size_bytes, twice_block));

    auto* const linear1_buf = linear1.buffer();
    auto* const gate_buf = gate.buffer();
    auto* const dL_dprod_buf = dL_dprod.buffer();
    auto* const dL_dlinear1_buf = output.dL_dlinear1.buffer();
    auto* const dL_dgate_buf = output.dL_dgate.buffer();

    std::vector<uint32_t> reader_ct_args{block_size, Wt};
    tt::tt_metal::TensorAccessorArgs(linear1_buf).append_to(reader_ct_args);
    tt::tt_metal::TensorAccessorArgs(gate_buf).append_to(reader_ct_args);
    tt::tt_metal::TensorAccessorArgs(dL_dprod_buf).append_to(reader_ct_args);
    program.kernels.push_back(  // kReaderKernelIdx
        make_reader_kernel_descriptor(all_cores, reader_ct_args, {}, kReaderKernelPath));

    std::vector<uint32_t> writer_ct_args{block_size, Wt};
    tt::tt_metal::TensorAccessorArgs(dL_dlinear1_buf).append_to(writer_ct_args);
    tt::tt_metal::TensorAccessorArgs(dL_dgate_buf).append_to(writer_ct_args);
    program.kernels.push_back(  // kWriterKernelIdx
        make_writer_kernel_descriptor(all_cores, writer_ct_args, {}, kWriterKernelPath));

    std::vector<uint32_t> compute_g1_args = {num_rows_g1, block_size, Wt};
    program.kernels.push_back(
        make_compute_kernel_descriptor(core_group_1, compute_g1_args, {}, kComputeKernelPath, true));

    if (!core_group_2.ranges().empty()) {
        std::vector<uint32_t> compute_g2_args = {num_rows_g2, block_size, Wt};
        program.kernels.push_back(
            make_compute_kernel_descriptor(core_group_2, compute_g2_args, {}, kComputeKernelPath, true));
    }

    // Buffers are passed as Buffer*, never as raw addresses; on a cache hit override_runtime_arguments() rewrites
    // these address slots. Row counts and start rows follow from the hashed shape.
    auto& reader = program.kernels[kReaderKernelIdx];
    auto& writer = program.kernels[kWriterKernelIdx];
    for_each_core_with_work(split, [&](const CoreWork& work) {
        const auto& [core, core_index, num_rows, start_row, in_group_1] = work;
        reader.emplace_runtime_args(core, {linear1_buf, gate_buf, dL_dprod_buf, num_rows, start_row});
        writer.emplace_runtime_args(core, {dL_dlinear1_buf, dL_dgate_buf, num_rows, start_row});
    });

    return program;
}

void SwigluElemwiseBwProgramFactory::override_runtime_arguments(
    tt::tt_metal::Program& program,
    const operation_attributes_t&,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& output,
    const std::optional<ttnn::MeshCoordinate>&) {
    // Only the buffer addresses change between hits. The outputs come from the return value, which is right for an
    // in-place call (dL_dlinear1 is linear1), a distinct preallocated output and a newly allocated one alike.
    const uint32_t linear1_addr = tensor_args.linear1.buffer()->address();
    const uint32_t gate_addr = tensor_args.gate.buffer()->address();
    const uint32_t dL_dprod_addr = tensor_args.dL_dprod.buffer()->address();
    const uint32_t dL_dlinear1_addr = output.dL_dlinear1.buffer()->address();
    const uint32_t dL_dgate_addr = output.dL_dgate.buffer()->address();

    const auto& linear1 = tensor_args.linear1;
    const auto split =
        compute_work_split(linear1.device()->compute_with_storage_grid_size(), num_tile_rows(linear1.padded_shape()));

    auto& reader_rt = tt::tt_metal::GetRuntimeArgs(program, kReaderKernelIdx);
    auto& writer_rt = tt::tt_metal::GetRuntimeArgs(program, kWriterKernelIdx);
    for_each_core_with_work(split, [&](const CoreWork& work) {
        auto& reader_args = reader_rt[work.core.x][work.core.y];
        reader_args[kLinear1BufferIdx] = linear1_addr;
        reader_args[kGateBufferIdx] = gate_addr;
        reader_args[kDLProdBufferIdx] = dL_dprod_addr;

        auto& writer_args = writer_rt[work.core.x][work.core.y];
        writer_args[kDLLinear1BufferIdx] = dL_dlinear1_addr;
        writer_args[kDLGateBufferIdx] = dL_dgate_addr;
    });
}

TTML_ASSERT_DESCRIPTOR_OVERRIDE(SwigluElemwiseBwDeviceOperation, SwigluElemwiseBwProgramFactory);

}  // namespace ttml::metal::ops::swiglu_elemwise_bw::device
