// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "gumbel_sample_device_operation.hpp"

#include <cmath>
#include <enchantum/enchantum.hpp>

#include "gumbel_sample_program_factory.hpp"
#include "ttnn/device_operation.hpp"

namespace ttml::metal::ops::gumbel_sample::device {

void GumbelSampleDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    auto check_tensor = [](const ttnn::Tensor& tensor, const std::string& name) {
        TT_FATAL(
            tensor.storage_type() == ttnn::StorageType::DEVICE,
            "GumbelSample requires '{}' to be on DEVICE, got storage type '{}'",
            name,
            enchantum::to_string(tensor.storage_type()));
        TT_FATAL(tensor.buffer() != nullptr, "GumbelSample: tensor '{}' has a null buffer", name);
        TT_FATAL(
            tensor.layout() == tt::tt_metal::Layout::TILE,
            "GumbelSample: tensor '{}' must be TILE layout, got '{}'",
            name,
            enchantum::to_string(tensor.layout()));
        TT_FATAL(
            tensor.memory_config().memory_layout() == tt::tt_metal::TensorMemoryLayout::INTERLEAVED,
            "GumbelSample: tensor '{}' must be INTERLEAVED, got '{}'",
            name,
            enchantum::to_string(tensor.memory_config().memory_layout()));
    };

    const auto& logits = tensor_args.logits;
    check_tensor(logits, "logits");

    TT_FATAL(
        logits.dtype() == tt::tt_metal::DataType::BFLOAT16 || logits.dtype() == tt::tt_metal::DataType::FLOAT32,
        "GumbelSample: logits must be BFLOAT16 or FLOAT32, got '{}'",
        enchantum::to_string(logits.dtype()));

    TT_FATAL(logits.padded_shape().rank() == 4U, "GumbelSample: logits must be 4D");

    TT_FATAL(
        args.temperature >= 0.0F && std::isfinite(args.temperature),
        "GumbelSample: temperature must be finite and >= 0, got {}. Zero selects greedy argmax; anything positive "
        "selects Gumbel-max sampling.",
        args.temperature);

    // NOTE: seed == 0 is deliberately NOT rejected. This op drives the SFPU generator directly via
    // rand_tile_init rather than going through ttnn::rand, so it inherits none of that op's
    // seed == 0 => host-entropy contract. Zero is just a state: the hardware generator is an XNOR
    // LFSR whose only lock-up state is all-ones, and ckernel_sfpu_rand.h already rewrites
    // 0xFFFFFFFF to 0xFFFFFFFE. A zero seed therefore yields an ordinary reproducible stream.

    if (tensor_args.logits_padding_mask.has_value()) {
        const auto& mask = tensor_args.logits_padding_mask.value();
        check_tensor(mask, "logits_padding_mask");
        TT_FATAL(
            mask.dtype() == logits.dtype(),
            "GumbelSample: mask dtype '{}' must match logits dtype '{}'",
            enchantum::to_string(mask.dtype()),
            enchantum::to_string(logits.dtype()));
        // The mask spans the vocabulary and is broadcast down the token rows: which columns are
        // padding is a property of the vocabulary, not of sequence position, so a single row covers
        // every token. That is what every caller builds (_sample_logits_mask in generate.py,
        // _build_logits_mask in llama_completer.py). A per-token mask is rejected rather than
        // silently mis-applied -- supporting it would mean a second kernel variant with no user.
        TT_FATAL(
            mask.logical_shape()[-1] == logits.logical_shape()[-1],
            "GumbelSample: mask width {} must match logits width {}",
            mask.logical_shape()[-1],
            logits.logical_shape()[-1]);
        TT_FATAL(
            mask.logical_shape()[-2] == 1U,
            "GumbelSample: mask token dim must be 1 (it is broadcast across all token rows), got {}. Build the "
            "mask as [B, 1, 1, V].",
            mask.logical_shape()[-2]);
    }

    if (tensor_args.preallocated_output.has_value()) {
        const auto& out = tensor_args.preallocated_output.value();
        TT_FATAL(
            out.dtype() == tt::tt_metal::DataType::UINT32,
            "GumbelSample: preallocated output must be UINT32, got '{}'",
            enchantum::to_string(out.dtype()));
        TT_FATAL(
            out.layout() == tt::tt_metal::Layout::ROW_MAJOR,
            "GumbelSample: preallocated output must be ROW_MAJOR, got '{}'",
            enchantum::to_string(out.layout()));
    }
}

GumbelSampleDeviceOperation::spec_return_value_t GumbelSampleDeviceOperation::compute_output_specs(
    const operation_attributes_t& /*args*/, const tensor_args_t& tensor_args) {
    if (tensor_args.preallocated_output.has_value()) {
        return tensor_args.preallocated_output->tensor_spec();
    }

    // Match ttnn::argmax(dim=3, keepdim=true): [B, 1, tokens, 1], UINT32, ROW_MAJOR.
    auto output_shape = tensor_args.logits.logical_shape();
    output_shape[-1] = 1U;
    return tt::tt_metal::TensorSpec(
        ttnn::Shape(output_shape),
        tt::tt_metal::TensorLayout(
            tt::tt_metal::DataType::UINT32,
            tt::tt_metal::PageConfig(tt::tt_metal::Layout::ROW_MAJOR),
            tensor_args.logits.memory_config()));
}

GumbelSampleDeviceOperation::tensor_return_value_t GumbelSampleDeviceOperation::create_output_tensors(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    if (tensor_args.preallocated_output.has_value()) {
        return tensor_args.preallocated_output.value();
    }
    return ttnn::create_device_tensor(compute_output_specs(args, tensor_args), tensor_args.logits.device());
}

ttsl::hash::hash_t GumbelSampleDeviceOperation::compute_program_hash(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    const auto& logits = tensor_args.logits;

    // `seed` and the temperature VALUE are intentionally NOT part of the key: both are pure runtime
    // args that override_runtime_arguments re-applies, so every step of a training loop (new seed
    // each step) reuses one cached program instead of thrashing the cache.
    //
    // Whether the temperature is zero IS part of the key, though: it selects the DO_GUMBEL_NOISE
    // define, so greedy and sampled runs compile to different kernels. Omitting it would let a
    // cached noisy program be reused for a greedy call -- silently sampling when the caller asked
    // for argmax. `seed_axes` is in the key because it changes which mesh coordinates get distinct
    // programs.
    return tt::tt_metal::operation::hash_operation<GumbelSampleDeviceOperation>(
        args.temperature > 0.0F,
        args.seed_axes,
        logits.dtype(),
        logits.logical_shape(),
        logits.padded_shape(),
        tensor_args.logits_padding_mask.has_value());
}

}  // namespace ttml::metal::ops::gumbel_sample::device

namespace ttnn::prim {

ttml::metal::ops::gumbel_sample::device::GumbelSampleDeviceOperation::tensor_return_value_t ttml_gumbel_sample(
    const ttnn::Tensor& logits,
    float temperature,
    uint32_t seed,
    const std::vector<uint32_t>& seed_axes,
    const std::optional<ttnn::Tensor>& logits_padding_mask,
    const std::optional<ttnn::Tensor>& preallocated_output) {
    using OperationType = ttml::metal::ops::gumbel_sample::device::GumbelSampleDeviceOperation;

    auto operation_attributes = OperationType::operation_attributes_t{
        .temperature = temperature,
        .seed = seed,
        .seed_axes = seed_axes,
    };
    auto tensor_args = OperationType::tensor_args_t{
        .logits = logits,
        .logits_padding_mask = logits_padding_mask,
        .preallocated_output = preallocated_output,
    };

    return ttnn::device_operation::launch<OperationType>(operation_attributes, tensor_args);
}

}  // namespace ttnn::prim
