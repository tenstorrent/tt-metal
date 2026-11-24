// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "cross_entropy_bw_program_factory.hpp"

#include "metal/ops/log_softmax/device/row_softmax_program.hpp"

namespace ttml::metal::ops::cross_entropy_bw::device {

tt::tt_metal::WorkloadDescriptor CrossEntropyBackwardProgramFactory::create_workload_descriptor(
    const operation_attributes_t& operation_attributes,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& output,
    const ttnn::MeshCoordinateRangeSet& tensor_coords) {
    const auto& input = tensor_args.input;
    const auto& target = tensor_args.target;
    const auto& upstream = tensor_args.upstream;
    constexpr auto kind = row_softmax::Kind::CrossEntropyBackward;
    auto scratch = row_softmax::make_scratch(input, kind);
    // One row-major target page per batch-channel slice, one target per physical row of each tile row.
    const row_softmax::Options options{
        .kind = kind,
        .upstream_dtype = upstream.has_value() ? upstream->dtype() : ttnn::DataType::BFLOAT16,
        .scaler = operation_attributes.scaler,
        .upstream_kind = upstream_kind(tensor_args),
        .target_page_bytes = target.logical_shape()[-1] * target.element_size(),
        .target_tiled_height = input.padded_shape()[-2] / input.tensor_spec().tile().get_height()};
    const row_softmax::Buffers buffers{
        .input = input.buffer(),
        .upstream = upstream.has_value() ? upstream->buffer() : nullptr,
        .target = target.buffer(),
        .output = output.buffer(),
        .scratch = scratch.has_value() ? scratch->buffer : nullptr};
    return row_softmax::replicated_workload(
        row_softmax::make_descriptor(options, input, buffers), tensor_coords, std::move(scratch));
}

}  // namespace ttml::metal::ops::cross_entropy_bw::device
