// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "gumbel_sample_device_operation.hpp"

#include <cmath>
#include <enchantum/enchantum.hpp>
#include <optional>

#include "gumbel_sample_program_factory.hpp"
#include "ttnn/device_operation.hpp"

namespace ttml::metal::ops::gumbel_sample::device {

namespace {

// The shape this op WILL write, derived from the logits alone. Single-sourced because the writer
// derives its output page indices from the logits geometry (its logical_tokens / Ht compile-time
// args), never from the output tensor -- so validation and the output spec drifting apart would not
// be caught anywhere downstream.
tt::tt_metal::Shape expected_output_shape(bool position_aware, const ttnn::Tensor& logits) {
    // Matches ttnn::argmax(dim=3, keepdim=true): [B, 1, tokens, 1] -- or [B, 1, 1, 1] when a
    // position per batch entry was given, since then only one row per entry is sampled.
    auto shape = logits.logical_shape();
    shape[-1] = 1U;
    if (position_aware) {
        shape[-2] = 1U;
    }
    return shape;
}

}  // namespace

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

    // The logits pick the device: the program is built against logits.device() and dispatched there.
    // Every OTHER tensor contributes only a raw buffer ADDRESS to that program's runtime args, and
    // addresses are local to a device's address space -- so a tensor belonging to a different mesh
    // would have its address reinterpreted against unrelated memory on this one. That is not a
    // fault, it is a silent read of whatever happens to live at that offset.
    auto* device = logits.device();
    TT_FATAL(device != nullptr, "GumbelSample: logits are not associated with a device");

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
            mask.device() == device,
            "GumbelSample: the mask must be on the same device as the logits (only its buffer address "
            "reaches the kernel, and addresses are not portable across devices)");
        TT_FATAL(
            mask.dtype() == logits.dtype(),
            "GumbelSample: mask dtype '{}' must match logits dtype '{}'",
            enchantum::to_string(mask.dtype()),
            enchantum::to_string(logits.dtype()));
        // The mask spans the vocabulary and is broadcast down the token rows: which columns are
        // padding is a property of the vocabulary, not of sequence position, so a single row covers
        // every token. That is what every caller builds (_sample_logits_mask in generate.py,
        // _build_logits_mask in llama_completer.py). A per-token mask is rejected rather than
        // silently mis-applied.
        TT_FATAL(
            mask.logical_shape()[-1] == logits.logical_shape()[-1],
            "GumbelSample: mask width {} must match logits width {}",
            mask.logical_shape()[-1],
            logits.logical_shape()[-1]);
        TT_FATAL(
            mask.logical_shape()[-2] == 1U,
            "GumbelSample: mask token dim must be 1 (it is broadcast across all token rows), got {}. Build the "
            "mask as [1, 1, 1, V].",
            mask.logical_shape()[-2]);
        // ... and independent of the batch for the same reason, one level up: every sequence in the
        // batch is decoded by the SAME lm_head, so the same columns are padding for all of them. The
        // reader relies on this -- it addresses mask tiles by column index alone, with no batch or
        // row stride -- so a mask carrying real per-batch data would have only its batch-0 slice
        // applied to every entry. Reject that rather than silently using a slice of it.
        TT_FATAL(
            mask.logical_shape()[-3] == 1U && mask.logical_shape()[-4] == 1U,
            "GumbelSample: mask batch and channel dims must be 1 (one mask covers every batch entry), got [{}, {}, "
            "{}, {}]. Build the mask as [1, 1, 1, V].",
            mask.logical_shape()[-4],
            mask.logical_shape()[-3],
            mask.logical_shape()[-2],
            mask.logical_shape()[-1]);
    }

    // [B, 1, 1, 1] UINT32 ROW_MAJOR INTERLEAVED on THIS device -- byte-for-byte this op's own
    // position-mode output spec, so page e of positions IS batch entry e IS output page e: one
    // indexing convention end to end, and a previous sample's output can be fed straight back in.
    //
    // ROW_MAJOR, not TILE: a tiled [B, 1, 1, 1] pads to a 32x32 tile, making each page read 4 KB
    // instead of one aligned word. INTERLEAVED because only the buffer TYPE is in the program hash,
    // not the memory layout, so a sharded tensor would collide with an interleaved one and reuse an
    // accessor compiled for the other encoding. Same device because only a raw ADDRESS reaches the
    // kernel, and addresses are not portable across devices.
    auto check_index_tensor = [&](const ttnn::Tensor& t, const std::string& name, bool position_aware) {
        TT_FATAL(
            t.storage_type() == ttnn::StorageType::DEVICE,
            "GumbelSample requires '{}' to be on DEVICE, got storage type '{}'",
            name,
            enchantum::to_string(t.storage_type()));
        TT_FATAL(t.buffer() != nullptr, "GumbelSample: '{}' has a null buffer", name);
        TT_FATAL(t.device() == device, "GumbelSample: '{}' must be on the same device as the logits", name);
        TT_FATAL(
            t.dtype() == tt::tt_metal::DataType::UINT32,
            "GumbelSample: '{}' must be UINT32, got '{}'",
            name,
            enchantum::to_string(t.dtype()));
        TT_FATAL(
            t.layout() == tt::tt_metal::Layout::ROW_MAJOR,
            "GumbelSample: '{}' must be ROW_MAJOR, got '{}'",
            name,
            enchantum::to_string(t.layout()));
        TT_FATAL(
            t.memory_config().memory_layout() == tt::tt_metal::TensorMemoryLayout::INTERLEAVED,
            "GumbelSample: '{}' must be INTERLEAVED, got '{}'",
            name,
            enchantum::to_string(t.memory_config().memory_layout()));
        // The exact shape is what guarantees page e is in bounds for every entry the kernels index.
        // A device tensor's shape IS its local shard.
        const auto expected = expected_output_shape(position_aware, logits);
        TT_FATAL(
            t.logical_shape() == expected,
            "GumbelSample: '{}' shape must be {}, got {}",
            name,
            expected,
            t.logical_shape());
    };

    if (tensor_args.positions.has_value()) {
        check_index_tensor(*tensor_args.positions, "positions", /*position_aware=*/true);
        TT_FATAL(
            tensor_args.positions->tensor_topology() == logits.tensor_topology(),
            "GumbelSample: 'positions' must be distributed across the mesh exactly as the logits are "
            "-- shard it with the SAME mapper the batch was sharded with");
    }

    if (tensor_args.preallocated_output.has_value()) {
        check_index_tensor(*tensor_args.preallocated_output, "preallocated_output", tensor_args.positions.has_value());
    }
}

GumbelSampleDeviceOperation::spec_return_value_t GumbelSampleDeviceOperation::compute_output_specs(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    if (tensor_args.preallocated_output.has_value()) {
        return tensor_args.preallocated_output->tensor_spec();
    }

    return tt::tt_metal::TensorSpec(
        ttnn::Shape(expected_output_shape(tensor_args.positions.has_value(), tensor_args.logits)),
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
    // Buffer PLACEMENT is part of the key, not just presence. TensorAccessorArgs bakes
    // ArgConfig::IsDram and the buffer's aligned_page_size into the reader/writer COMPILE-TIME args
    // (tensor_accessor_args.cpp), and both differ between DRAM and L1. Two calls that differ only in
    // placement would otherwise collide here, and the cache hit patches addresses only -- leaving
    // accessors compiled for one memory space aimed at the other's addresses.
    auto placement_of = [](const std::optional<ttnn::Tensor>& tensor) -> int {
        return tensor.has_value() ? static_cast<int>(tensor->memory_config().buffer_type()) : -1;
    };

    // Whether positions were supplied changes the output spec and the kernel defines; the position
    // VALUES are runtime args, re-applied per dispatch, so they stay out of the key.
    const bool position_aware = tensor_args.positions.has_value();

    // In position mode the program does not depend on the token dimension AT ALL: the work split is
    // NC * Wt (see the program factory), the reader's Ht is a runtime arg, and the writer's
    // token-dim compile-time args are pinned. Normalizing dim -2 out of the key is therefore what
    // lets one program serve every prompt length. Without it, a rollout whose prompts round to a new
    // Np missed the cache on every prefill and paid a fresh JIT build of all three kernels -- ~6 s
    // against ~3 ms for the dispatch itself, measured across 17 generates.
    //
    // The dim is NORMALIZED rather than omitted because hash_operation forwards a variadic pack with
    // no arity tag, and both shapes must be treated: the host tile-rounds Np, so padded == logical
    // and normalizing one alone would leave the other as a live source of misses.
    //
    // Two invariants keep the relaxation sound, and both are load-bearing: the reader's Ht MUST be
    // re-applied in override_runtime_arguments (a cached program is otherwise replayed with the Ht
    // it was built at, reading a real but WRONG token row -- in bounds, no fault, silently wrong),
    // and total_tiles MUST stay NC * Wt in position mode. It also means a captured trace would
    // freeze Ht into its recorded runtime args; nothing in tt-train captures traces today, but a
    // trace taken at one Np and replayed at another would read the wrong pages.
    auto token_normalized = [position_aware](tt::tt_metal::Shape shape) {
        if (position_aware) {
            shape[-2] = 1U;
        }
        return shape;
    };

    return tt::tt_metal::operation::hash_operation<GumbelSampleDeviceOperation>(
        position_aware,
        args.temperature > 0.0F,
        args.seed_axes,
        logits.dtype(),
        token_normalized(logits.logical_shape()),
        token_normalized(logits.padded_shape()),
        static_cast<int>(logits.memory_config().buffer_type()),
        tensor_args.logits_padding_mask.has_value(),
        placement_of(tensor_args.logits_padding_mask),
        // The positions SHAPE is not hashed separately: its entry count is NC = padded_shape[0] *
        // padded_shape[1], already in the key via dims 0 and 1, which token_normalized leaves alone.
        // Only its placement matters, for the accessor reason above.
        placement_of(tensor_args.positions),
        placement_of(tensor_args.preallocated_output));
}

}  // namespace ttml::metal::ops::gumbel_sample::device

namespace ttnn::prim {

ttml::metal::ops::gumbel_sample::device::GumbelSampleDeviceOperation::tensor_return_value_t ttml_gumbel_sample(
    const ttnn::Tensor& logits,
    float temperature,
    uint32_t seed,
    const std::vector<uint32_t>& seed_axes,
    const std::optional<ttnn::Tensor>& logits_padding_mask,
    const std::optional<ttnn::Tensor>& positions,
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
        .positions = positions,
        .preallocated_output = preallocated_output,
    };

    return ttnn::device_operation::launch<OperationType>(operation_attributes, tensor_args);
}

}  // namespace ttnn::prim
