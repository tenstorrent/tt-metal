// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "gumbel_sample_device_operation.hpp"

#include <algorithm>
#include <cmath>
#include <enchantum/enchantum.hpp>
#include <optional>
#include <tt-logger/tt-logger.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/math.hpp>

#include "gumbel_sample_program_factory.hpp"
#include "ttnn/device_operation.hpp"

namespace ttml::metal::ops::gumbel_sample::device {

namespace {

// The shape this op WILL write, derived from the logits alone. Single-sourced because the writer
// derives its output page indices from the logits geometry (its logical_tokens runtime arg and Ht
// compile-time arg), never from the output tensor -- so validation and the output spec drifting
// apart would not be caught anywhere downstream.
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

        const auto tile = tensor.tensor_spec().tile();
        TT_FATAL(
            tile.get_height() == tt::constants::TILE_HEIGHT && tile.get_width() == tt::constants::TILE_WIDTH,
            "GumbelSample: tensor '{}' must use the default {}x{} tile, got {}x{}",
            name,
            tt::constants::TILE_HEIGHT,
            tt::constants::TILE_WIDTH,
            tile.get_height(),
            tile.get_width());
        auto expected_padded = tensor.logical_shape();
        expected_padded[-2] = tt::round_up(expected_padded[-2], tt::constants::TILE_HEIGHT);
        expected_padded[-1] = tt::round_up(expected_padded[-1], tt::constants::TILE_WIDTH);
        TT_FATAL(
            tensor.padded_shape() == expected_padded,
            "GumbelSample: tensor '{}' padded shape {} must be its logical shape {} rounded up to the 32x32 tile; "
            "custom alignments are not supported",
            name,
            tensor.padded_shape(),
            tensor.logical_shape());
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

    // seed_axes name the mesh axes whose devices hold DISTINCT data and must therefore draw
    // DISTINCT noise. An axis outside the mesh is unconditionally a caller bug (a typo, or a
    // config reused across topologies): seeded_linear_index() in the program factory skips axes it
    // cannot find, so every device would fall through to stream id 0 and draw byte-identical
    // noise -- for GRPO that is duplicate completions with zero-variance advantages, and nothing
    // downstream flags it. Extent-1 axes are deliberately NOT rejected: a trivial axis is how a
    // topology-generic caller says "seed this axis if it happens to be sharded here".
    const auto mesh_shape = device->shape();
    for (const uint32_t axis : args.seed_axes) {
        TT_FATAL(
            axis < mesh_shape.dims(),
            "GumbelSample: seed_axes entry {} does not exist on this {}-dimensional mesh {}",
            axis,
            mesh_shape.dims(),
            mesh_shape);
    }
    // A multi-device mesh where no seeded axis has extent > 1 is LEGAL -- a fully replicated
    // (TP-only) mesh deliberately draws identical noise so replicas agree on the sampled token --
    // but it is also exactly what a forgotten seed_axes looks like from a data-parallel caller.
    // Say so at debug level (as the composite sample() this op replaced used to) rather than
    // guessing which of the two the caller meant.
    if (mesh_shape.mesh_size() > 1 && std::none_of(args.seed_axes.begin(), args.seed_axes.end(), [&](uint32_t axis) {
            return mesh_shape[axis] > 1;
        })) {
        log_debug(
            tt::LogOp,
            "GumbelSample: all {} devices share one RNG stream (seed_axes selects no axis with extent > 1) and will "
            "draw identical noise. If these devices hold distinct batch rows, pass the data-parallel mesh axes in "
            "seed_axes.",
            mesh_shape.mesh_size());
    }

    TT_FATAL(
        logits.dtype() == tt::tt_metal::DataType::BFLOAT16 || logits.dtype() == tt::tt_metal::DataType::FLOAT32,
        "GumbelSample: logits must be BFLOAT16 or FLOAT32, got '{}'",
        enchantum::to_string(logits.dtype()));

    TT_FATAL(logits.padded_shape().rank() == 4U, "GumbelSample: logits must be 4D");

    TT_FATAL(
        args.temperature >= 0.0F && std::isfinite(args.temperature),
        "GumbelSample: temperature must be finite and >= 0, got {}. Zero -- or a positive value below ~2.9e-39, whose "
        "reciprocal overflows float32 -- selects greedy argmax; other positive values select Gumbel-max sampling.",
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
            "-- shard it with the SAME mapper the batch was sharded with. (A common cause: a batch "
            "that does not divide across the mesh, which shrinks a 1D-sharded tensor's distribution "
            "shape to the chunk count.)");
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
    // WHICH KERNEL the temperature selects IS part of the key, though: uses_gumbel_noise picks the
    // noise compile-time arg, so greedy and sampled runs compile to different kernels. Omitting it
    // would let a cached noisy program be reused for a greedy call -- silently sampling when the
    // caller asked for argmax. It must be the SAME predicate the factory uses (not a bare
    // `temperature > 0`): a sub-reciprocal-overflow temperature builds the greedy kernel, and
    // hashing it as "noisy" would collide it with real noisy programs. `seed_axes` is in the key
    // because it changes which mesh coordinates get distinct programs.
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
    // NC * Wt (see the program factory), the reader's Ht and both kernels' logical_tokens (the
    // position-clamp bound) are runtime args, and the writer's compile-time Ht is pinned.
    // Normalizing dim -2 out of the key is therefore what lets one program serve every
    // prompt length. Without it, a rollout whose prompts round to a new Np missed the cache on
    // every prefill and paid a fresh JIT build of all three kernels -- ~6 s against ~3 ms for the
    // dispatch itself, measured across 17 generates.
    //
    // The dim is NORMALIZED rather than omitted because hash_operation forwards a variadic pack
    // with no arity tag: conditionally dropping an argument would change the key's structure and
    // let unrelated argument combinations alias.
    //
    // Two invariants keep the relaxation sound, and both are load-bearing: every token-derived
    // runtime arg (the reader's Ht, both kernels' logical_tokens) MUST be re-applied in
    // override_runtime_arguments (a cached program is otherwise replayed with the values it was
    // built at, reading a real but WRONG token row or clamping positions against a stale bound --
    // in bounds, no fault, silently wrong), and total_tiles MUST stay NC * Wt in position mode. It
    // also means a captured trace would freeze those args as recorded; nothing in tt-train
    // captures traces today, but a trace taken at one Np and replayed at another would read the
    // wrong pages.
    auto token_normalized = [position_aware](tt::tt_metal::Shape shape) {
        if (position_aware) {
            shape[-2] = 1U;
        }
        return shape;
    };

    return tt::tt_metal::operation::hash_operation<GumbelSampleDeviceOperation>(
        position_aware,
        uses_gumbel_noise(args.temperature),
        args.seed_axes,
        logits.dtype(),
        // The padded shape is deliberately NOT hashed alongside the logical one: check_tensor pins
        // it to the logical shape's default-tile round-up, so it is fully derived and would only
        // duplicate key material. Everything the program factory takes from padded dims (Wt, Ht,
        // NC) is therefore a function of what IS in the key.
        token_normalized(logits.logical_shape()),
        static_cast<int>(logits.memory_config().buffer_type()),
        tensor_args.logits_padding_mask.has_value(),
        placement_of(tensor_args.logits_padding_mask),
        // The positions SHAPE is not hashed separately: its entry count is NC, which the factory
        // computes from PADDED dims 0 and 1 of the logits. Tile rounding only touches the last two
        // dims (check_tensor enforces exactly that), so padded dims 0 and 1 equal the logical ones
        // already in the key -- and token_normalized leaves them alone. Only the placement of
        // positions matters, for the accessor reason above.
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
