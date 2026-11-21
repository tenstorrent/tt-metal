// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "cross_entropy_bw_device_operation.hpp"

#include "cross_entropy_bw_program_factory.hpp"
#include "metal/common/tensor_validation.hpp"
#include "metal/ops/log_softmax/device/row_softmax_program.hpp"
#include "ttnn/device_operation.hpp"

namespace ttml::metal::ops::cross_entropy_bw::device {

namespace {
// The gradient in the input's dtype and memory configuration. The program writes its tiles in the input's format,
// so a preallocated output must have exactly this spec.
tt::tt_metal::TensorSpec output_spec(const ttnn::Tensor& input) {
    return tt::tt_metal::TensorSpec(
        input.logical_shape(),
        tt::tt_metal::TensorLayout(
            input.dtype(),
            tt::tt_metal::PageConfig(tt::tt_metal::Layout::TILE, input.tensor_spec().tile()),
            input.memory_config()));
}
}  // namespace

void CrossEntropyBackwardDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    const auto& input_tensor = tensor_args.input;
    const auto& target_tensor = tensor_args.target;
    const auto& preallocated_output_tensor = tensor_args.preallocated_output;
    check_device_tensor(
        input_tensor,
        "CrossEntropyBackward",
        "Input",
        {.dtypes = {tt::tt_metal::DataType::BFLOAT16, tt::tt_metal::DataType::FLOAT32},
         .layout = tt::tt_metal::Layout::TILE,
         .buffer_type = tt::tt_metal::BufferType::DRAM});
    // The fused row-softmax program streams interleaved tiles and keeps column indices as exact FP32.
    TT_FATAL(
        input_tensor.logical_shape().rank() == 4 && input_tensor.memory_config() == ttnn::DRAM_MEMORY_CONFIG &&
            row_softmax::supports_tile(input_tensor.tensor_spec().tile()),
        "CrossEntropyBackward: input must be a rank-4 interleaved DRAM tensor of supported tiles, got {} with tile {}",
        input_tensor.logical_shape(),
        input_tensor.tensor_spec().tile().get_tile_shape());
    TT_FATAL(
        input_tensor.logical_shape()[-1] > 0 && input_tensor.logical_shape()[-1] <= (1U << 24),
        "CrossEntropyBackward: vocabulary width must be in [1, 2^24], got {}",
        input_tensor.logical_shape()[-1]);
    // The kernels take tile counts from the padded shape and mask only the last tile's columns past the logical
    // width; the output has one tile row per logical tile row.
    check_tile_padding(input_tensor, "CrossEntropyBackward", "Input");
    check_device_tensor(
        target_tensor,
        "CrossEntropyBackward",
        "Target",
        {.dtypes = {tt::tt_metal::DataType::UINT32}, .layout = tt::tt_metal::Layout::ROW_MAJOR});
    check_same_device(target_tensor, input_tensor, "CrossEntropyBackward", "Target", "Input");

    // The reader walks one row-major target page per batch-channel slice of the input
    // (page = tile_row / Ht over NC * Ht rows) and sizes each page read from the target's
    // inner dim. Pinning both the target's page width and its page count to the input keeps
    // every page index the reader can form inside the target allocation.
    const auto& target_shape = target_tensor.logical_shape();
    TT_FATAL(
        target_shape[-1] == input_tensor.logical_shape()[-2],
        "CrossEntropyBackward: target inner dim ({}) must equal input sequence dim ({})",
        target_shape[-1],
        input_tensor.logical_shape()[-2]);
    const auto& input_padded_shape = input_tensor.padded_shape();
    const uint64_t input_nc_pages =
        input_padded_shape.volume() / (static_cast<uint64_t>(input_padded_shape[-2]) * input_padded_shape[-1]);
    const uint64_t target_pages = target_shape.volume() / target_shape[-1];
    TT_FATAL(
        target_pages == input_nc_pages,
        "CrossEntropyBackward: target must supply one page per input batch-channel slice, got {} page(s) for {} "
        "slice(s)",
        target_pages,
        input_nc_pages);

    if (preallocated_output_tensor.has_value()) {
        const auto& output = preallocated_output_tensor.value();
        check_device_tensor(
            output,
            "CrossEntropyBackward",
            "Preallocated Output",
            {.dtypes = {tt::tt_metal::DataType::BFLOAT16, tt::tt_metal::DataType::FLOAT32},
             .buffer_type = tt::tt_metal::BufferType::DRAM});
        check_same_device(output, input_tensor, "CrossEntropyBackward", "Preallocated Output", "Input");
        // Another dtype or shape would be written past the end of the buffer or read back in the wrong format.
        const auto expected = output_spec(input_tensor);
        TT_FATAL(
            output.tensor_spec() == expected,
            "CrossEntropyBackward: preallocated output must be {} {} in interleaved DRAM like the input, got {} {}",
            expected.logical_shape(),
            enchantum::to_string(expected.data_type()),
            output.logical_shape(),
            enchantum::to_string(output.dtype()));
    }

    if (tensor_args.upstream.has_value()) {
        const auto& upstream = tensor_args.upstream.value();
        check_device_tensor(
            upstream,
            "CrossEntropyBackward",
            "Upstream",
            {.dtypes = {tt::tt_metal::DataType::BFLOAT16, tt::tt_metal::DataType::FLOAT32},
             .layout = tt::tt_metal::Layout::TILE,
             .buffer_type = tt::tt_metal::BufferType::DRAM});
        check_same_device(upstream, input_tensor, "CrossEntropyBackward", "Upstream", "Input");
        auto per_row_shape = input_tensor.logical_shape();
        per_row_shape[-1] = 1U;
        TT_FATAL(
            upstream.tensor_spec().tile() == input_tensor.tensor_spec().tile() &&
                (upstream.logical_shape() == ttnn::Shape({1, 1, 1, 1}) ||
                 (upstream.logical_shape() == per_row_shape &&
                  upstream.padded_shape()[-2] == input_tensor.padded_shape()[-2])),
            "CrossEntropyBackward: upstream must be [1, 1, 1, 1] or {} in the input's tiles, got {}",
            per_row_shape,
            upstream.logical_shape());
    }
}

CrossEntropyBackwardDeviceOperation::spec_return_value_t CrossEntropyBackwardDeviceOperation::compute_output_specs(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    if (tensor_args.preallocated_output.has_value()) {
        return tensor_args.preallocated_output->tensor_spec();
    }
    return output_spec(tensor_args.input);
}

CrossEntropyBackwardDeviceOperation::tensor_return_value_t CrossEntropyBackwardDeviceOperation::create_output_tensors(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    tensor_return_value_t output_tensor;

    spec_return_value_t output_specs = compute_output_specs(args, tensor_args);

    if (tensor_args.preallocated_output.has_value()) {
        output_tensor = tensor_args.preallocated_output.value();
    } else {
        output_tensor = ttnn::create_device_tensor(output_specs, tensor_args.input.device());
    }

    return output_tensor;
}

// Hashes what the default key hashes (every attribute, the full spec of every tensor, which optional tensors are
// passed). It is custom only to skip the canonical key string the default key also builds on every dispatch.
ttsl::hash::hash_t CrossEntropyBackwardDeviceOperation::compute_program_hash(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    return tt::tt_metal::operation::hash_operation<CrossEntropyBackwardDeviceOperation>(args, tensor_args);
}

}  // namespace ttml::metal::ops::cross_entropy_bw::device

namespace ttnn::prim {

ttml::metal::ops::cross_entropy_bw::device::CrossEntropyBackwardDeviceOperation::tensor_return_value_t
ttml_cross_entropy_bw(
    const ttnn::Tensor& input_tensor,
    const ttnn::Tensor& target_tensor,
    float scaler,
    const std::optional<ttnn::Tensor>& preallocated_output,
    const std::optional<ttnn::Tensor>& upstream) {
    using OperationType = ttml::metal::ops::cross_entropy_bw::device::CrossEntropyBackwardDeviceOperation;

    auto operation_attributes = OperationType::operation_attributes_t{.scaler = scaler};
    auto tensor_args = OperationType::tensor_args_t{
        .input = input_tensor,
        .target = target_tensor,
        .preallocated_output = preallocated_output,
        .upstream = upstream,
    };

    return ttnn::device_operation::launch<OperationType>(operation_attributes, tensor_args);
}

}  // namespace ttnn::prim
