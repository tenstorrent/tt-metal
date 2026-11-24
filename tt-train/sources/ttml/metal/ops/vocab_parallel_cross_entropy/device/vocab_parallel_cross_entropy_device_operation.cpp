// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "vocab_parallel_cross_entropy_device_operation.hpp"

#include <limits>
#include <optional>

#include "metal/common/tensor_validation.hpp"
#include "metal/ops/log_softmax/device/row_softmax_program.hpp"
#include "ttnn/device_operation.hpp"

namespace ttml::metal::ops::vocab_parallel_cross_entropy::device {

namespace {

constexpr uint32_t kStateTiles = 2;  // Two column-vector tiles (M, l) per tile row.

// A shard's record is one FP32 tile column block per tile row, and shards are combined as the lanes (columns) of one
// tile: both are the tile width.
uint32_t tile_width(const ttnn::Tensor& tensor) {
    return tensor.tensor_spec().tile().get_width();
}

tt::tt_metal::TensorLayout fp32_layout(const ttnn::Tensor& input) {
    return tt::tt_metal::TensorLayout(
        ttnn::DataType::FLOAT32,
        tt::tt_metal::PageConfig(ttnn::Layout::TILE, input.tensor_spec().tile()),
        ttnn::DRAM_MEMORY_CONFIG);
}

ttnn::Shape with_width(const ttnn::Tensor& tensor, uint32_t width) {
    auto shape = tensor.logical_shape();
    shape[-1] = width;
    return shape;
}

void check_tiled_dram(const ttnn::Tensor& tensor, const char* name) {
    TT_FATAL(
        tensor.layout() == ttnn::Layout::TILE && tensor.logical_shape().rank() == 4 &&
            tensor.memory_config() == ttnn::DRAM_MEMORY_CONFIG &&
            row_softmax::supports_tile(tensor.tensor_spec().tile()),
        "VocabParallelCrossEntropy: {} must be a rank-4 interleaved DRAM tensor of supported tiles, got {} with tile "
        "{}",
        name,
        tensor.logical_shape(),
        tensor.tensor_spec().tile().get_tile_shape());
    // The kernels take tile counts from the padded shape and size their outputs from the logical one.
    check_tile_padding(tensor, "VocabParallelCrossEntropy", name);
}

// The reader walks one row-major target page per batch-channel slice of the logits; pin the target's page
// width and page count to the logits (see CrossEntropyBackward).
void check_target(const ttnn::Tensor& logits, const ttnn::Tensor& target) {
    check_device_tensor(
        target,
        "VocabParallelCrossEntropy",
        "target",
        {.dtypes = {tt::tt_metal::DataType::UINT32}, .layout = tt::tt_metal::Layout::ROW_MAJOR});
    check_same_device(target, logits, "VocabParallelCrossEntropy", "target", "logits");
    const auto& target_shape = target.logical_shape();
    TT_FATAL(
        target_shape[-1] == logits.logical_shape()[-2],
        "VocabParallelCrossEntropy: target inner dim ({}) must equal the logits sequence dim ({})",
        target_shape[-1],
        logits.logical_shape()[-2]);
    const auto& padded = logits.padded_shape();
    const uint64_t nc_pages = padded.volume() / (static_cast<uint64_t>(padded[-2]) * padded[-1]);
    TT_FATAL(
        target_shape.volume() / target_shape[-1] == nc_pages,
        "VocabParallelCrossEntropy: target must supply one page per logits batch-channel slice");
}

}  // namespace

void VocabParallelCrossEntropyDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& a, const tensor_args_t& t) {
    const auto& input = t.input;
    check_device_tensor(
        input,
        "VocabParallelCrossEntropy",
        "input",
        {.dtypes = {ttnn::DataType::BFLOAT16, ttnn::DataType::FLOAT32}, .buffer_type = ttnn::BufferType::DRAM});
    check_tiled_dram(input, "input");
    auto* device = input.device();
    TT_FATAL(device != nullptr, "VocabParallelCrossEntropy: input must be on a (mesh) device");
    if (a.cluster_axis.has_value()) {
        TT_FATAL(
            *a.cluster_axis < device->shape().dims(),
            "VocabParallelCrossEntropy: cluster_axis {} is out of range for the mesh",
            *a.cluster_axis);
    }
    const uint32_t width = input.logical_shape()[-1];
    if (a.mode == Mode::Combine) {
        const uint32_t record_width = tile_width(input);
        TT_FATAL(
            input.dtype() == ttnn::DataType::FLOAT32 && width % record_width == 0 && width > 0 &&
                width / record_width <= record_width,
            "VocabParallelCrossEntropy: combine expects FP32 records of {} columns per shard, at most {} shards",
            record_width,
            record_width);
        return;
    }
    TT_FATAL(t.target.has_value(), "VocabParallelCrossEntropy: target required");
    check_target(input, *t.target);
    // Global column indices stay exact FP32 integers.
    const uint64_t shards = a.cluster_axis.has_value() ? device->shape()[*a.cluster_axis] : device->num_devices();
    TT_FATAL(
        width > 0 && static_cast<uint64_t>(width) * shards <= (1U << 24),
        "VocabParallelCrossEntropy: the full vocabulary ({} x {}) must be at most 2^24",
        shards,
        width);
    if (a.mode == Mode::Backward) {
        TT_FATAL(t.state.has_value(), "VocabParallelCrossEntropy: backward needs the row state");
        check_device_tensor(*t.state, "VocabParallelCrossEntropy", "state", {.dtypes = {ttnn::DataType::FLOAT32}});
        check_tiled_dram(*t.state, "state");
        check_same_device(*t.state, input, "VocabParallelCrossEntropy", "state", "logits");
        const auto state_shape = with_width(input, kStateTiles * tile_width(input));
        TT_FATAL(
            t.state->logical_shape() == state_shape && t.state->padded_shape()[-2] == input.padded_shape()[-2] &&
                t.state->tensor_spec().tile() == input.tensor_spec().tile(),
            "VocabParallelCrossEntropy: state must be {} in the input's tiles, got {}",
            state_shape,
            t.state->logical_shape());
        if (t.upstream.has_value()) {
            check_device_tensor(
                *t.upstream,
                "VocabParallelCrossEntropy",
                "upstream",
                {.dtypes = {ttnn::DataType::BFLOAT16, ttnn::DataType::FLOAT32},
                 .layout = ttnn::Layout::TILE,
                 .buffer_type = ttnn::BufferType::DRAM});
            check_same_device(*t.upstream, input, "VocabParallelCrossEntropy", "upstream", "logits");
            TT_FATAL(
                t.upstream->tensor_spec().tile() == input.tensor_spec().tile() &&
                    (t.upstream->logical_shape() == ttnn::Shape({1, 1, 1, 1}) ||
                     (t.upstream->logical_shape() == with_width(input, 1) &&
                      t.upstream->padded_shape()[-2] == input.padded_shape()[-2])),
                "VocabParallelCrossEntropy: upstream must be [1, 1, 1, 1] or {} in the input's tiles",
                with_width(input, 1));
        }
    }
}

VocabParallelCrossEntropyDeviceOperation::spec_return_value_t
VocabParallelCrossEntropyDeviceOperation::compute_output_specs(
    const operation_attributes_t& a, const tensor_args_t& t) {
    const auto fp32 = fp32_layout(t.input);
    switch (a.mode) {
        case Mode::Stats: return {tt::tt_metal::TensorSpec(with_width(t.input, tile_width(t.input)), fp32)};
        case Mode::Combine:
            return {
                tt::tt_metal::TensorSpec(with_width(t.input, 1), fp32),
                tt::tt_metal::TensorSpec(with_width(t.input, kStateTiles * tile_width(t.input)), fp32)};
        case Mode::Backward: return {t.input.tensor_spec()};
    }
    TT_THROW("Unknown vocabulary-parallel cross-entropy mode");
}

VocabParallelCrossEntropyDeviceOperation::tensor_return_value_t
VocabParallelCrossEntropyDeviceOperation::create_output_tensors(
    const operation_attributes_t& a, const tensor_args_t& t) {
    tensor_return_value_t outputs;
    for (const auto& spec : compute_output_specs(a, t)) {
        outputs.push_back(ttnn::create_device_tensor(spec, t.input.device()));
    }
    return outputs;
}

ttsl::hash::hash_t VocabParallelCrossEntropyDeviceOperation::compute_program_hash(
    const operation_attributes_t& a, const tensor_args_t& t) {
    // cluster_axis decides the program-to-coordinate mapping; value_or keeps nullopt distinct from axis 0. The mode
    // and upstream_kind key which optional tensors are present, so buffer bindings stay positional on a cache hit.
    // The target may sit in DRAM or L1, which its reader's TensorAccessorArgs compile in: key its full spec. The
    // other tensors are validated to interleaved DRAM.
    return tt::tt_metal::operation::hash_operation<VocabParallelCrossEntropyDeviceOperation>(
        static_cast<uint32_t>(a.mode),
        a.cluster_axis.value_or(std::numeric_limits<uint32_t>::max()),
        a.scaler,
        t.input.dtype(),
        t.input.logical_shape(),
        t.input.padded_shape(),
        upstream_kind(t),
        t.upstream.has_value() ? t.upstream->dtype() : t.input.dtype(),
        t.target.has_value() ? std::optional<tt::tt_metal::TensorSpec>(t.target->tensor_spec()) : std::nullopt);
}

}  // namespace ttml::metal::ops::vocab_parallel_cross_entropy::device

namespace ttnn::prim {

namespace {
using OperationType = ttml::metal::ops::vocab_parallel_cross_entropy::device::VocabParallelCrossEntropyDeviceOperation;
using ttml::metal::ops::vocab_parallel_cross_entropy::device::Mode;
}  // namespace

ttnn::Tensor ttml_vocab_parallel_ce_stats(
    const ttnn::Tensor& logits, const ttnn::Tensor& target, std::optional<uint32_t> cluster_axis) {
    return ttnn::device_operation::launch<OperationType>(
        OperationType::operation_attributes_t{.mode = Mode::Stats, .cluster_axis = cluster_axis},
        OperationType::tensor_args_t{.input = logits, .target = target})[0];
}

std::vector<ttnn::Tensor> ttml_vocab_parallel_ce_combine(const ttnn::Tensor& gathered_stats) {
    return ttnn::device_operation::launch<OperationType>(
        OperationType::operation_attributes_t{.mode = Mode::Combine},
        OperationType::tensor_args_t{.input = gathered_stats});
}

ttnn::Tensor ttml_vocab_parallel_ce_bw(
    const ttnn::Tensor& logits,
    const ttnn::Tensor& target,
    const ttnn::Tensor& state,
    const std::optional<ttnn::Tensor>& upstream,
    float scaler,
    std::optional<uint32_t> cluster_axis) {
    return ttnn::device_operation::launch<OperationType>(
        OperationType::operation_attributes_t{.mode = Mode::Backward, .cluster_axis = cluster_axis, .scaler = scaler},
        OperationType::tensor_args_t{.input = logits, .target = target, .state = state, .upstream = upstream})[0];
}

}  // namespace ttnn::prim
