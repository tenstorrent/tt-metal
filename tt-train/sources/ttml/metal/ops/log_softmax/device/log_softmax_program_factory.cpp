// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "log_softmax_program_factory.hpp"

#include "row_softmax_program.hpp"

namespace ttml::metal::ops::log_softmax::device {

tt::tt_metal::WorkloadDescriptor LogSoftmaxProgramFactory::create_workload_descriptor(
    const operation_attributes_t& operation_attributes,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& output,
    const ttnn::MeshCoordinateRangeSet& tensor_coords) {
    const bool backward = operation_attributes.backward;
    const auto& input = tensor_args.input;
    const auto kind = backward ? row_softmax::Kind::LogSoftmaxBackward : row_softmax::Kind::LogSoftmaxForward;
    auto scratch = row_softmax::make_scratch(input);
    const row_softmax::Options options{
        .kind = kind, .upstream_dtype = backward ? tensor_args.upstream->dtype() : input.dtype()};
    const row_softmax::Buffers buffers{
        .input = input.buffer(),
        .upstream = backward ? tensor_args.upstream->buffer() : nullptr,
        .state = backward ? tensor_args.state->buffer() : output[1].buffer(),
        .output = output[0].buffer(),
        .scratch = scratch.has_value() ? scratch->buffer : nullptr};
    return row_softmax::replicated_workload(
        row_softmax::make_descriptor(options, input, buffers), tensor_coords, std::move(scratch));
}

}  // namespace ttml::metal::ops::log_softmax::device
