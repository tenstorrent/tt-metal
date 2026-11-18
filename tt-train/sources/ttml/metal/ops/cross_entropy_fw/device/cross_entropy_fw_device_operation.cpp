// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "cross_entropy_fw_device_operation.hpp"

#include "cross_entropy_fw_program_factory.hpp"
#include "metal/common/tensor_validation.hpp"
#include "metal/ops/log_softmax/device/row_softmax_program.hpp"
#include "ttnn/device_operation.hpp"

namespace ttml::metal::ops::cross_entropy_fw::device {

namespace {
// The per-row loss in the input's dtype, tiles and memory configuration. The program writes its tiles in the input's
// format, so a preallocated output must have exactly this spec.
tt::tt_metal::TensorSpec output_spec(const ttnn::Tensor& input) {
    auto shape = input.logical_shape();
    shape[-1] = 1U;
    return tt::tt_metal::TensorSpec(
        shape,
        tt::tt_metal::TensorLayout(
            input.dtype(),
            tt::tt_metal::PageConfig(tt::tt_metal::Layout::TILE, input.tensor_spec().tile()),
            input.memory_config()));
}
}  // namespace

void CrossEntropyForwardDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    const auto& input_tensor = tensor_args.input;
    const auto& target_tensor = tensor_args.target;
    const auto& preallocated_output_tensor = tensor_args.preallocated_output;
    check_device_tensor(
        input_tensor,
        "CrossEntropyForward",
        "Input",
        {.dtypes = {tt::tt_metal::DataType::BFLOAT16, tt::tt_metal::DataType::FLOAT32},
         .layout = tt::tt_metal::Layout::TILE,
         .buffer_type = tt::tt_metal::BufferType::DRAM});
    // The fused row-softmax program streams interleaved tiles and keeps column indices as exact FP32.
    TT_FATAL(
        input_tensor.logical_shape().rank() == 4 && input_tensor.memory_config() == ttnn::DRAM_MEMORY_CONFIG &&
            row_softmax::supports_tile(input_tensor.tensor_spec().tile()),
        "CrossEntropyForward: input must be a rank-4 interleaved DRAM tensor of supported tiles, got {} with tile {}",
        input_tensor.logical_shape(),
        input_tensor.tensor_spec().tile().get_tile_shape());
    TT_FATAL(
        input_tensor.logical_shape()[-1] > 0 && input_tensor.logical_shape()[-1] <= (1U << 24),
        "CrossEntropyForward: vocabulary width must be in [1, 2^24], got {}",
        input_tensor.logical_shape()[-1]);
    // The kernels take tile counts from the padded shape and mask only the last tile's columns past the logical
    // width; the output has one tile row per logical tile row.
    check_tile_padding(input_tensor, "CrossEntropyForward", "Input");
    check_device_tensor(
        target_tensor,
        "CrossEntropyForward",
        "Target",
        {.dtypes = {tt::tt_metal::DataType::UINT32}, .layout = tt::tt_metal::Layout::ROW_MAJOR});
    check_same_device(target_tensor, input_tensor, "CrossEntropyForward", "Target", "Input");

    // The reader walks one row-major target page per batch-channel slice of the input
    // (page = tile_row / Ht over NC * Ht rows) and sizes each page read from the target's
    // inner dim. Pinning both the target's page width and its page count to the input keeps
    // every page index the reader can form inside the target allocation.
    const auto& target_shape = target_tensor.logical_shape();
    TT_FATAL(
        target_shape[-1] == input_tensor.logical_shape()[-2],
        "CrossEntropyForward: target inner dim ({}) must equal input sequence dim ({})",
        target_shape[-1],
        input_tensor.logical_shape()[-2]);
    const auto& input_padded_shape = input_tensor.padded_shape();
    const uint64_t input_nc_pages =
        input_padded_shape.volume() / (static_cast<uint64_t>(input_padded_shape[-2]) * input_padded_shape[-1]);
    const uint64_t target_pages = target_shape.volume() / target_shape[-1];
    TT_FATAL(
        target_pages == input_nc_pages,
        "CrossEntropyForward: target must supply one page per input batch-channel slice, got {} page(s) for {} "
        "slice(s)",
        target_pages,
        input_nc_pages);

    if (preallocated_output_tensor.has_value()) {
        const auto& output = preallocated_output_tensor.value();
        check_device_tensor(
            output,
            "CrossEntropyForward",
            "Preallocated Output",
            {.dtypes = {tt::tt_metal::DataType::BFLOAT16, tt::tt_metal::DataType::FLOAT32},
             .buffer_type = tt::tt_metal::BufferType::DRAM});
        check_same_device(output, input_tensor, "CrossEntropyForward", "Preallocated Output", "Input");
        // Another dtype or shape would be written past the end of the buffer or read back in the wrong format.
        const auto expected = output_spec(input_tensor);
        TT_FATAL(
            output.tensor_spec() == expected,
            "CrossEntropyForward: preallocated output must be {} {} in interleaved DRAM like the input, got {} {}",
            expected.logical_shape(),
            enchantum::to_string(expected.data_type()),
            output.logical_shape(),
            enchantum::to_string(output.dtype()));
    }
}

CrossEntropyForwardDeviceOperation::spec_return_value_t CrossEntropyForwardDeviceOperation::compute_output_specs(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    if (tensor_args.preallocated_output.has_value()) {
        return tensor_args.preallocated_output->tensor_spec();
    }
    return output_spec(tensor_args.input);
}

CrossEntropyForwardDeviceOperation::tensor_return_value_t CrossEntropyForwardDeviceOperation::create_output_tensors(
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
ttsl::hash::hash_t CrossEntropyForwardDeviceOperation::compute_program_hash(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    return tt::tt_metal::operation::hash_operation<CrossEntropyForwardDeviceOperation>(args, tensor_args);
}

}  // namespace ttml::metal::ops::cross_entropy_fw::device

namespace ttnn::prim {

ttml::metal::ops::cross_entropy_fw::device::CrossEntropyForwardDeviceOperation::tensor_return_value_t
ttml_cross_entropy_fw(
    const ttnn::Tensor& input_tensor,
    const ttnn::Tensor& target_tensor,
    const std::optional<ttnn::Tensor>& preallocated_output) {
    using OperationType = ttml::metal::ops::cross_entropy_fw::device::CrossEntropyForwardDeviceOperation;

    auto operation_attributes = OperationType::operation_attributes_t{};
    auto tensor_args = OperationType::tensor_args_t{
        .input = input_tensor,
        .target = target_tensor,
        .preallocated_output = preallocated_output,
    };

    return ttnn::device_operation::launch<OperationType>(operation_attributes, tensor_args);
}

}  // namespace ttnn::prim
