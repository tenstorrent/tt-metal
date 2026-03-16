// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "swiglu_elemwise_bw_program_factory.hpp"

#include <cstdint>
#include <enchantum/enchantum.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>

#include "metal/common/program_utils.hpp"

namespace {

constexpr auto kReaderKernelPath =
    "tt-train/sources/ttml/metal/ops/swiglu_elemwise_bw/device/kernels/dataflow/"
    "reader_swiglu_elemwise_bw_interleaved.cpp";
constexpr auto kWriterKernelPath =
    "tt-train/sources/ttml/metal/ops/swiglu_elemwise_bw/device/kernels/dataflow/"
    "writer_swiglu_elemwise_bw_interleaved.cpp";
constexpr auto kComputeKernelPath =
    "tt-train/sources/ttml/metal/ops/swiglu_elemwise_bw/device/kernels/compute/swiglu_elemwise_bw_kernel.cpp";

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

}  // namespace

namespace ttml::metal::ops::swiglu_elemwise_bw::device {

struct SwigluElemwiseBwKernels {
    tt::tt_metal::KernelHandle reader{};
    tt::tt_metal::KernelHandle writer{};
    tt::tt_metal::KernelHandle compute_group_1{};
    tt::tt_metal::KernelHandle compute_group_2{};
};

void assign_per_core_runtime_args(
    tt::tt_metal::Program& program,
    const SwigluElemwiseBwKernels& kernels,
    const tt::tt_metal::Buffer* linear1_buffer,
    const tt::tt_metal::Buffer* gate_buffer,
    const tt::tt_metal::Buffer* dL_dprod_buffer,
    const tt::tt_metal::Buffer* dL_dlinear1_buffer,
    const tt::tt_metal::Buffer* dL_dgate_buffer,
    uint32_t num_cores,
    uint32_t num_cores_y,
    uint32_t num_rows_per_core_group_1,
    uint32_t num_rows_per_core_group_2,
    const tt::tt_metal::CoreRangeSet& core_group_1,
    const tt::tt_metal::CoreRangeSet& core_group_2) {
    for (uint32_t i = 0, num_rows_written = 0; i < num_cores; i++) {
        tt::tt_metal::CoreCoord core = {i / num_cores_y, i % num_cores_y};

        uint32_t num_rows_per_core = 0;
        if (core_group_1.contains(core)) {
            num_rows_per_core = num_rows_per_core_group_1;
        } else if (core_group_2.contains(core)) {
            num_rows_per_core = num_rows_per_core_group_2;
        } else {
            TT_FATAL(false, "Core not in specified core ranges");
        }

        SetRuntimeArgs(
            program,
            kernels.reader,
            core,
            {linear1_buffer->address(),
             gate_buffer->address(),
             dL_dprod_buffer->address(),
             num_rows_per_core,
             num_rows_written});

        SetRuntimeArgs(
            program,
            kernels.writer,
            core,
            {dL_dlinear1_buffer->address(), dL_dgate_buffer->address(), num_rows_per_core, num_rows_written});

        num_rows_written += num_rows_per_core;
    }
}

SwigluElemwiseBwProgramFactory::cached_program_t SwigluElemwiseBwProgramFactory::create(
    const operation_attributes_t& args, const tensor_args_t& tensor_args, tensor_return_value_t& output) {
    const auto& linear1 = tensor_args.linear1;
    const auto& gate = tensor_args.gate;
    const auto& dL_dprod = tensor_args.dL_dprod;

    auto* device = linear1.device();
    tt::tt_metal::Program program{};

    tt::DataFormat data_format = datatype_to_dataformat_converter(linear1.dtype());
    uint32_t tile_size_bytes = tt::tile_size(data_format);

    auto padded_shape = linear1.padded_shape();
    TT_FATAL(padded_shape.rank() == 4U, "Input tensors must be 4D");
    uint32_t Wt = padded_shape[-1] / tt::constants::TILE_WIDTH;
    uint32_t Ht = padded_shape[-2] / tt::constants::TILE_HEIGHT;
    uint32_t NC = padded_shape[0] * padded_shape[1];
    uint32_t total_rows = NC * Ht;

    auto grid_size = device->compute_with_storage_grid_size();
    uint32_t num_cores_y = grid_size.y;
    uint32_t block_size = 4U;

    auto [num_cores, all_cores, core_group_1, core_group_2, num_rows_g1, num_rows_g2] =
        tt::tt_metal::split_work_to_cores(grid_size, total_rows);

    const uint32_t twice_block = 2U * block_size;
    create_circular_buffer(program, all_cores, kLinear1CbIndex, data_format, tile_size_bytes, twice_block);
    create_circular_buffer(program, all_cores, kGateCbIndex, data_format, tile_size_bytes, twice_block);
    create_circular_buffer(program, all_cores, kDLProdCbIndex, data_format, tile_size_bytes, twice_block);
    create_circular_buffer(program, all_cores, kDLLinear1CbIndex, data_format, tile_size_bytes, twice_block);
    create_circular_buffer(program, all_cores, kDLGateCbIndex, data_format, tile_size_bytes, twice_block);
    create_circular_buffer(program, all_cores, kSigmoidCbIndex, data_format, tile_size_bytes, twice_block);
    create_circular_buffer(program, all_cores, kScratchCbIndex, data_format, tile_size_bytes, twice_block);
    create_circular_buffer(program, all_cores, kSiluGradCbIndex, data_format, tile_size_bytes, twice_block);

    auto* linear1_buf = linear1.buffer();
    auto* gate_buf = gate.buffer();
    auto* dL_dprod_buf = dL_dprod.buffer();
    auto* dL_dlinear1_buf = output.dL_dlinear1.buffer();
    auto* dL_dgate_buf = output.dL_dgate.buffer();

    SwigluElemwiseBwKernels kernels;

    std::vector<uint32_t> reader_ct_args{block_size, Wt};
    tt::tt_metal::TensorAccessorArgs(linear1_buf).append_to(reader_ct_args);
    tt::tt_metal::TensorAccessorArgs(gate_buf).append_to(reader_ct_args);
    tt::tt_metal::TensorAccessorArgs(dL_dprod_buf).append_to(reader_ct_args);
    kernels.reader = create_reader_kernel(program, all_cores, reader_ct_args, {}, kReaderKernelPath);

    std::vector<uint32_t> writer_ct_args{block_size, Wt};
    tt::tt_metal::TensorAccessorArgs(dL_dlinear1_buf).append_to(writer_ct_args);
    tt::tt_metal::TensorAccessorArgs(dL_dgate_buf).append_to(writer_ct_args);
    kernels.writer = create_writer_kernel(program, all_cores, writer_ct_args, {}, kWriterKernelPath);

    std::vector<uint32_t> compute_g1_args = {num_rows_g1, block_size, Wt};
    kernels.compute_group_1 =
        create_compute_kernel(program, core_group_1, compute_g1_args, {}, kComputeKernelPath, true);

    if (!core_group_2.ranges().empty()) {
        std::vector<uint32_t> compute_g2_args = {num_rows_g2, block_size, Wt};
        kernels.compute_group_2 =
            create_compute_kernel(program, core_group_2, compute_g2_args, {}, kComputeKernelPath, true);
    }

    assign_per_core_runtime_args(
        program,
        kernels,
        linear1_buf,
        gate_buf,
        dL_dprod_buf,
        dL_dlinear1_buf,
        dL_dgate_buf,
        num_cores,
        num_cores_y,
        num_rows_g1,
        num_rows_g2,
        core_group_1,
        core_group_2);

    return cached_program_t{
        std::move(program),
        {kernels.reader,
         kernels.writer,
         kernels.compute_group_1,
         kernels.compute_group_2,
         core_group_1,
         core_group_2,
         num_cores,
         num_cores_y}};
}

void SwigluElemwiseBwProgramFactory::override_runtime_arguments(
    cached_program_t& cached_program,
    const operation_attributes_t&,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& output) {
    auto& program = cached_program.program;
    auto& sv = cached_program.shared_variables;

    auto& reader_rt = GetRuntimeArgs(program, sv.reader_kernel_id);
    auto& writer_rt = GetRuntimeArgs(program, sv.writer_kernel_id);

    for (uint32_t i = 0; i < sv.num_cores; i++) {
        tt::tt_metal::CoreCoord core = {i / sv.num_cores_y, i % sv.num_cores_y};

        {
            auto& args = reader_rt[core.x][core.y];
            args[kLinear1BufferIdx] = tensor_args.linear1.buffer()->address();
            args[kGateBufferIdx] = tensor_args.gate.buffer()->address();
            args[kDLProdBufferIdx] = tensor_args.dL_dprod.buffer()->address();
        }
        {
            auto& args = writer_rt[core.x][core.y];
            args[kDLLinear1BufferIdx] = output.dL_dlinear1.buffer()->address();
            args[kDLGateBufferIdx] = output.dL_dgate.buffer()->address();
        }
    }
}

}  // namespace ttml::metal::ops::swiglu_elemwise_bw::device
