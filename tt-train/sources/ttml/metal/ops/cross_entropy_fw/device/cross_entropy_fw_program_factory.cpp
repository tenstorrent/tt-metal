// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "cross_entropy_fw_program_factory.hpp"

#include "metal/ops/log_softmax/device/row_softmax_program.hpp"

namespace ttml::metal::ops::cross_entropy_fw::device {

tt::tt_metal::WorkloadDescriptor CrossEntropyForwardProgramFactory::create_workload_descriptor(
    const operation_attributes_t& operation_attributes,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& output,
    const ttnn::MeshCoordinateRangeSet& tensor_coords) {
    const auto& input = tensor_args.input;
    const auto& target = tensor_args.target;
    constexpr auto kind = row_softmax::Kind::CrossEntropyForward;
    auto scratch = row_softmax::make_scratch(input);
    // One row-major target page per batch-channel slice, one target per physical row of each tile row.
    const row_softmax::Options options{
        .kind = kind,
        .target_page_bytes = target.logical_shape()[-1] * target.element_size(),
        .target_tiled_height = input.padded_shape()[-2] / input.tensor_spec().tile().get_height()};
    const row_softmax::Buffers buffers{
        .input = input.buffer(),
        .target = target.buffer(),
        .output = output.buffer(),
        .scratch = scratch.has_value() ? scratch->buffer : nullptr};
    return row_softmax::replicated_workload(
        row_softmax::make_descriptor(options, input, buffers), tensor_coords, std::move(scratch));
}

}  // namespace ttml::metal::ops::cross_entropy_fw::device
