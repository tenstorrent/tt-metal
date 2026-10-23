// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "swiglu_packed_bw_program_factory.hpp"

#include <cstdint>
#include <optional>
#include <tt-metalium/tensor_accessor_args.hpp>

#include "metal/common/program_utils.hpp"
#include "metal/ops/common/swiglu_packed_common.hpp"
#include "swiglu_packed_bw_device_operation.hpp"

namespace {

constexpr auto kReaderKernelPath =
    "tt-train/sources/ttml/metal/ops/swiglu_packed_bw/device/kernels/dataflow/reader_swiglu_packed_bw_interleaved.cpp";
constexpr auto kWriterKernelPath =
    "tt-train/sources/ttml/metal/ops/swiglu_packed_bw/device/kernels/dataflow/writer_swiglu_packed_bw_interleaved.cpp";
constexpr auto kComputeKernelPath =
    "tt-train/sources/ttml/metal/ops/swiglu_packed_bw/device/kernels/compute/swiglu_packed_bw_kernel.cpp";

// Kernel handles are the push order into ProgramDescriptor::kernels: reader, writer, compute group 1, then compute
// group 2 when it has cores.
constexpr uint32_t kReaderKernelIdx = 0U;
constexpr uint32_t kWriterKernelIdx = 1U;

// reader runtime args
constexpr uint32_t kPackedAddrIdx = 0U;
constexpr uint32_t kDhAddrIdx = 1U;
// writer runtime args
constexpr uint32_t kDPackedAddrIdx = 0U;

constexpr auto kGateCbIndex = tt::CBIndex::c_0;   // gate branch (silu'd)
constexpr auto kUpCbIndex = tt::CBIndex::c_1;     // up branch
constexpr auto kDhCbIndex = tt::CBIndex::c_2;     // upstream grad dL/dh
constexpr auto kDGateCbIndex = tt::CBIndex::c_3;  // output: grad wrt gate branch
constexpr auto kDUpCbIndex = tt::CBIndex::c_4;    // output: grad wrt up branch
constexpr auto kSigmoidCbIndex = tt::CBIndex::c_5;
constexpr auto kScratchCbIndex = tt::CBIndex::c_6;
constexpr auto kSiluGradCbIndex = tt::CBIndex::c_7;

}  // namespace

namespace ttml::metal::ops::swiglu_packed_bw::device {

namespace {

// The packed tensor's blocks over the device's compute grid. create_descriptor() pins it against split_work_to_cores,
// and override_runtime_arguments() walks the same cores from it.
WorkSplit make_work_split(const ttnn::Tensor& packed) {
    return compute_work_split(
        packed.device()->compute_with_storage_grid_size(),
        swiglu_packed::block_geometry(packed.padded_shape()).total_blocks);
}

}  // namespace

tt::tt_metal::ProgramDescriptor SwigluPackedBwProgramFactory::create_descriptor(
    const operation_attributes_t& args, const tensor_args_t& tensor_args, tensor_return_value_t& output) {
    const auto& packed = tensor_args.packed;
    const auto& dh = tensor_args.dL_dh;

    auto* const device = packed.device();
    tt::tt_metal::ProgramDescriptor program;

    const tt::DataFormat data_format = datatype_to_dataformat_converter(packed.dtype());
    const uint32_t tile_size_bytes = tt::tile_size(data_format);

    const auto geometry = swiglu_packed::block_geometry(packed.padded_shape());
    const uint32_t Wt = geometry.Wt;
    const uint32_t block_size = geometry.block_size;
    const auto grid_size = device->compute_with_storage_grid_size();

    const auto [num_cores, all_cores, core_group_1, core_group_2, num_blocks_g1, num_blocks_g2] =
        tt::tt_metal::split_work_to_cores(grid_size, geometry.total_blocks);
    const WorkSplit split = make_work_split(packed);
    check_work_split(split, num_cores, core_group_1, core_group_2, num_blocks_g1, num_blocks_g2);

    const uint32_t twice_block = 2U * block_size;
    program.cbs.push_back(make_cb_descriptor(all_cores, kGateCbIndex, data_format, tile_size_bytes, twice_block));
    program.cbs.push_back(make_cb_descriptor(all_cores, kUpCbIndex, data_format, tile_size_bytes, twice_block));
    program.cbs.push_back(make_cb_descriptor(all_cores, kDhCbIndex, data_format, tile_size_bytes, twice_block));
    program.cbs.push_back(make_cb_descriptor(all_cores, kDGateCbIndex, data_format, tile_size_bytes, twice_block));
    program.cbs.push_back(make_cb_descriptor(all_cores, kDUpCbIndex, data_format, tile_size_bytes, twice_block));
    program.cbs.push_back(make_cb_descriptor(all_cores, kSigmoidCbIndex, data_format, tile_size_bytes, twice_block));
    program.cbs.push_back(make_cb_descriptor(all_cores, kScratchCbIndex, data_format, tile_size_bytes, twice_block));
    program.cbs.push_back(make_cb_descriptor(all_cores, kSiluGradCbIndex, data_format, tile_size_bytes, twice_block));

    auto* const packed_buf = packed.buffer();
    auto* const dh_buf = dh.buffer();
    auto* const dpacked_buf = output.buffer();

    std::vector<uint32_t> reader_ct_args{block_size, Wt};
    tt::tt_metal::TensorAccessorArgs(packed_buf).append_to(reader_ct_args);
    tt::tt_metal::TensorAccessorArgs(dh_buf).append_to(reader_ct_args);
    program.kernels.push_back(  // kReaderKernelIdx
        make_reader_kernel_descriptor(all_cores, reader_ct_args, {}, kReaderKernelPath));

    std::vector<uint32_t> writer_ct_args{block_size, Wt};
    tt::tt_metal::TensorAccessorArgs(dpacked_buf).append_to(writer_ct_args);
    program.kernels.push_back(  // kWriterKernelIdx
        make_writer_kernel_descriptor(all_cores, writer_ct_args, {}, kWriterKernelPath));

    std::vector<uint32_t> compute_g1_args = {num_blocks_g1, block_size};
    program.kernels.push_back(
        make_compute_kernel_descriptor(core_group_1, compute_g1_args, {}, kComputeKernelPath, true));

    if (!core_group_2.ranges().empty()) {
        std::vector<uint32_t> compute_g2_args = {num_blocks_g2, block_size};
        program.kernels.push_back(
            make_compute_kernel_descriptor(core_group_2, compute_g2_args, {}, kComputeKernelPath, true));
    }

    // Slots as named at the top of the file; override_runtime_arguments() rewrites the addresses on every cache hit.
    // num_blocks/start_block follow from the hashed packed shape.
    auto& reader = program.kernels[kReaderKernelIdx];
    auto& writer = program.kernels[kWriterKernelIdx];
    for_each_core_with_work(split, [&](const CoreWork& work) {
        const auto& [core, core_index, num_blocks, start_block, in_group_1] = work;
        reader.emplace_runtime_args(core, {packed_buf, dh_buf, num_blocks, start_block});
        writer.emplace_runtime_args(core, {dpacked_buf, num_blocks, start_block});
    });

    return program;
}

void SwigluPackedBwProgramFactory::override_runtime_arguments(
    tt::tt_metal::Program& program,
    const operation_attributes_t& /*operation_attributes*/,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& output,
    const std::optional<ttnn::MeshCoordinate>& /*mesh_dispatch_coordinate*/) {
    const WorkSplit split = make_work_split(tensor_args.packed);

    const uint32_t packed_addr = tensor_args.packed.buffer()->address();
    const uint32_t dh_addr = tensor_args.dL_dh.buffer()->address();
    const uint32_t dpacked_addr = output.buffer()->address();

    auto& reader_runtime_args = tt::tt_metal::GetRuntimeArgs(program, kReaderKernelIdx);
    auto& writer_runtime_args = tt::tt_metal::GetRuntimeArgs(program, kWriterKernelIdx);
    for_each_core_with_work(split, [&](const CoreWork& work) {
        const auto& core = work.core;

        auto& reader_args = reader_runtime_args[core.x][core.y];
        reader_args[kPackedAddrIdx] = packed_addr;
        reader_args[kDhAddrIdx] = dh_addr;

        auto& writer_args = writer_runtime_args[core.x][core.y];
        writer_args[kDPackedAddrIdx] = dpacked_addr;
    });
}

TTML_ASSERT_DESCRIPTOR_OVERRIDE(SwigluPackedBwDeviceOperation, SwigluPackedBwProgramFactory);

}  // namespace ttml::metal::ops::swiglu_packed_bw::device
