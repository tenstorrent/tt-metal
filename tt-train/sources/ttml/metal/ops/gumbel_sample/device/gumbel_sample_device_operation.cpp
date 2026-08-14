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
tt::tt_metal::Shape expected_output_shape(const operation_attributes_t& args, const ttnn::Tensor& logits) {
    // Matches ttnn::argmax(dim=3, keepdim=true): [B, 1, tokens, 1] -- or [B, 1, 1, 1] when a
    // position per batch entry was given, since then only one row per entry is sampled.
    auto shape = logits.logical_shape();
    shape[-1] = 1U;
    if (!args.positions.empty()) {
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

    if (!args.positions.empty()) {
        // A device tensor's shape is its LOCAL shard, so `entries` is this device's batch, not the
        // job's. The caller may hand over either: its own rows (already sharded) or the whole global
        // list, which the program factory slices by the same seeded linear index the RNG uses. Any
        // other length is a mismatch the kernel could not detect, so it is rejected here.
        const auto& mesh_shape = device->shape();
        uint32_t batch_shards = 1U;
        for (uint32_t axis : args.seed_axes) {
            if (axis < mesh_shape.dims() && mesh_shape[axis] > 1U) {
                batch_shards *= static_cast<uint32_t>(mesh_shape[axis]);
            }
        }
        const uint32_t entries = logits.padded_shape()[0] * logits.padded_shape()[1];
        TT_FATAL(
            args.positions.size() == entries || args.positions.size() == entries * batch_shards,
            "GumbelSample: positions must hold either this device's {} batch rows or all {} rows "
            "across the {} data-parallel shards, got {}",
            entries,
            entries * batch_shards,
            batch_shards,
            args.positions.size());

        // Both dataflow kernels carry the full local list as runtime args (the origin core has to
        // re-derive the target row of any entry it merges), so the batch is bounded by the per-core
        // runtime-arg budget. The reader binds it: it carries five fixed args to the writer's four.
        //
        // The bound is tt_metal::max_runtime_args, the documented PORTABLE FLOOR, not the enforced
        // ceiling -- for a Tensix kernel that is max_runtime_args_tensix = 4096, with the real L1
        // fit checked at program finalize. Sitting under the floor is deliberate: it holds whatever
        // core type the op is ever placed on, and the practical limit is lower than either number
        // anyway, since every core receives its own copy of the list and the host therefore writes
        // B_local * cores * 2 args per dispatch. Anything approaching this bound should move the
        // positions into a small tensor the cores read once instead.
        constexpr uint32_t kReaderFixedArgs = 5U;
        constexpr uint32_t kMaxEntries = tt::tt_metal::max_runtime_args - kReaderFixedArgs;
        TT_FATAL(
            entries <= kMaxEntries,
            "GumbelSample: position-aware sampling supports up to {} batch rows per device, got {}",
            kMaxEntries,
            entries);

        const uint32_t tokens = logits.logical_shape()[-2];
        for (size_t i = 0; i < args.positions.size(); ++i) {
            TT_FATAL(
                args.positions[i] < tokens,
                "GumbelSample: positions[{}] = {} is outside the {} token positions",
                i,
                args.positions[i],
                tokens);
        }
    }

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
        // silently mis-applied -- supporting it would mean a second kernel variant with no user.
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

    if (tensor_args.preallocated_output.has_value()) {
        const auto& out = tensor_args.preallocated_output.value();
        // Same reasoning as the mask, plus: build_program dereferences output.buffer() directly, so a
        // host-side tensor here would be a null dereference rather than a diagnosable error.
        TT_FATAL(
            out.storage_type() == ttnn::StorageType::DEVICE,
            "GumbelSample requires 'preallocated_output' to be on DEVICE, got storage type '{}'",
            enchantum::to_string(out.storage_type()));
        TT_FATAL(out.buffer() != nullptr, "GumbelSample: preallocated output has a null buffer");
        TT_FATAL(
            out.device() == device, "GumbelSample: the preallocated output must be on the same device as the logits");
        TT_FATAL(
            out.dtype() == tt::tt_metal::DataType::UINT32,
            "GumbelSample: preallocated output must be UINT32, got '{}'",
            enchantum::to_string(out.dtype()));
        TT_FATAL(
            out.layout() == tt::tt_metal::Layout::ROW_MAJOR,
            "GumbelSample: preallocated output must be ROW_MAJOR, got '{}'",
            enchantum::to_string(out.layout()));
        // The writer addresses output pages from the LOGITS geometry -- page_base is built from the
        // logical_tokens / Ht compile-time args, and in position mode the page IS the batch entry --
        // so it emits exactly the page count implied by the logits no matter what it was handed. An
        // undersized output therefore takes NOC writes past the end of its buffer (page addressing
        // is plain arithmetic, with no bounds check), and an oversized or differently-shaped one is
        // returned with only part of it written. Neither is detectable downstream, because
        // compute_output_specs adopts this tensor's spec verbatim.
        //
        // Requiring an exact match also keeps the program cache sound: the output shape is not
        // hashed, but pinning it to a function of the logits shape and positions-presence -- both of
        // which ARE hashed -- means it cannot vary independently of the key.
        const auto expected_shape = expected_output_shape(args, logits);
        TT_FATAL(
            out.logical_shape() == expected_shape,
            "GumbelSample: preallocated output shape must be {}, got {}",
            expected_shape,
            out.logical_shape());
        // Interleaved for the same reason the other tensors are: the writer walks a linear page
        // space, and only the buffer TYPE (DRAM vs L1) is in the program hash, not the memory
        // layout -- so a sharded output would collide with an interleaved one of the same type and
        // reuse an accessor compiled for the other.
        TT_FATAL(
            out.memory_config().memory_layout() == tt::tt_metal::TensorMemoryLayout::INTERLEAVED,
            "GumbelSample: preallocated output must be INTERLEAVED, got '{}'",
            enchantum::to_string(out.memory_config().memory_layout()));
    }
}

GumbelSampleDeviceOperation::spec_return_value_t GumbelSampleDeviceOperation::compute_output_specs(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    if (tensor_args.preallocated_output.has_value()) {
        return tensor_args.preallocated_output->tensor_spec();
    }

    return tt::tt_metal::TensorSpec(
        ttnn::Shape(expected_output_shape(args, tensor_args.logits)),
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
    const bool position_aware = !args.positions.empty();

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
    const std::vector<uint32_t>& positions,
    const std::optional<ttnn::Tensor>& preallocated_output) {
    using OperationType = ttml::metal::ops::gumbel_sample::device::GumbelSampleDeviceOperation;

    auto operation_attributes = OperationType::operation_attributes_t{
        .temperature = temperature,
        .seed = seed,
        .seed_axes = seed_axes,
        .positions = positions,
    };
    auto tensor_args = OperationType::tensor_args_t{
        .logits = logits,
        .logits_padding_mask = logits_padding_mask,
        .preallocated_output = preallocated_output,
    };

    return ttnn::device_operation::launch<OperationType>(operation_attributes, tensor_args);
}

}  // namespace ttnn::prim
