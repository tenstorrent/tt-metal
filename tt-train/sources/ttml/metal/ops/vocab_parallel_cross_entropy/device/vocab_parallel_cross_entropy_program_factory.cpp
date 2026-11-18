// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "vocab_parallel_cross_entropy_program_factory.hpp"

#include "metal/ops/log_softmax/device/row_softmax_program.hpp"

namespace ttml::metal::ops::vocab_parallel_cross_entropy::device {

namespace {

row_softmax::Kind kind_of(Mode mode) {
    switch (mode) {
        case Mode::Stats: return row_softmax::Kind::VocabParallelStats;
        case Mode::Combine: return row_softmax::Kind::VocabParallelCombine;
        case Mode::Backward: return row_softmax::Kind::VocabParallelBackward;
    }
    TT_THROW("Unknown vocabulary-parallel cross-entropy mode");
}

row_softmax::Buffers buffers_for(
    const operation_attributes_t& attributes,
    const tensor_args_t& tensor_args,
    const tensor_return_value_t& output,
    tt::tt_metal::Buffer* scratch) {
    row_softmax::Buffers buffers{.input = tensor_args.input.buffer(), .output = output[0].buffer()};
    switch (attributes.mode) {
        case Mode::Stats:
            buffers.target = tensor_args.target->buffer();
            buffers.scratch = scratch;
            break;
        case Mode::Combine: buffers.state = output[1].buffer(); break;
        case Mode::Backward:
            buffers.target = tensor_args.target->buffer();
            buffers.state = tensor_args.state->buffer();
            buffers.upstream = tensor_args.upstream.has_value() ? tensor_args.upstream->buffer() : nullptr;
            break;
    }
    return buffers;
}

// The program of the shard whose global columns start at rank * local_V.
tt::tt_metal::ProgramDescriptor make_descriptor(
    const operation_attributes_t& attributes,
    const tensor_args_t& tensor_args,
    const row_softmax::Buffers& buffers,
    uint32_t rank) {
    const auto& input = tensor_args.input;
    row_softmax::Options options{.kind = kind_of(attributes.mode)};
    if (attributes.mode != Mode::Combine) {
        const auto& target = *tensor_args.target;
        options.first_v = rank * input.logical_shape()[-1];
        // One row-major target page per batch-channel slice, one target per physical row of each tile row.
        options.target_page_bytes = target.logical_shape()[-1] * target.element_size();
        options.target_tiled_height = input.padded_shape()[-2] / input.tensor_spec().tile().get_height();
    }
    if (attributes.mode == Mode::Backward) {
        options.scaler = attributes.scaler;
        options.upstream_kind = upstream_kind(tensor_args);
        if (tensor_args.upstream.has_value()) {
            options.upstream_dtype = tensor_args.upstream->dtype();
        }
    }
    return row_softmax::make_descriptor(options, input, buffers);
}

}  // namespace

tt::tt_metal::WorkloadDescriptor VocabParallelCrossEntropyProgramFactory::create_workload_descriptor(
    const operation_attributes_t& attributes,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& output,
    const ttnn::MeshCoordinateRangeSet& tensor_coords) {
    const auto mesh_shape = tensor_args.input.device()->shape();
    tt::tt_metal::WorkloadDescriptor workload;
    auto scratch = row_softmax::make_scratch(tensor_args.input, kind_of(attributes.mode));
    const auto buffers = buffers_for(attributes, tensor_args, output, scratch.has_value() ? scratch->buffer : nullptr);
    if (scratch.has_value()) {
        workload.buffers.push_back(std::move(*scratch));
    }
    const auto add = [&](const ttnn::MeshCoordinateRange& range, uint32_t rank) {
        workload.programs.push_back(
            {.range = range, .descriptor = make_descriptor(attributes, tensor_args, buffers, rank)});
    };
    if (attributes.cluster_axis.has_value()) {
        // One program per TP rank: the hyperplane of the mesh fixing the TP coordinate.
        const uint32_t axis = *attributes.cluster_axis;
        auto slab_start = ttnn::MeshCoordinate::zero_coordinate(mesh_shape.dims());
        auto slab_end = ttnn::MeshCoordinate::zero_coordinate(mesh_shape.dims());
        for (size_t d = 0; d < mesh_shape.dims(); ++d) {
            slab_end[static_cast<int32_t>(d)] = mesh_shape[d] - 1U;
        }
        for (uint32_t rank = 0; rank < mesh_shape[axis]; ++rank) {
            slab_start[static_cast<int32_t>(axis)] = rank;
            slab_end[static_cast<int32_t>(axis)] = rank;
            const ttnn::MeshCoordinateRange slab{slab_start, slab_end};
            for (const auto& range : tensor_coords.ranges()) {
                if (auto intersection = slab.intersection(range); intersection.has_value()) {
                    add(*intersection, rank);
                }
            }
        }
    } else {
        for (const auto& range : tensor_coords.ranges()) {
            for (const auto& coord : range) {
                add(ttnn::MeshCoordinateRange{coord}, static_cast<uint32_t>(coord.to_linear_index(mesh_shape)));
            }
        }
    }
    return workload;
}

}  // namespace ttml::metal::ops::vocab_parallel_cross_entropy::device
