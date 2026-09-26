// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "gumbel_sample_device_operation.hpp"

#include <algorithm>
#include <cmath>
#include <enchantum/enchantum.hpp>
#include <optional>
#include <string_view>
#include <tt-logger/tt-logger.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/math.hpp>
#include <vector>

#include "gumbel_sample_program_factory.hpp"
#include "ttnn/device_operation.hpp"

namespace ttml::metal::ops::gumbel_sample::device {

namespace {

// Op-local stand-ins for the shared validation helpers PR #56523 adds in
// metal/common/tensor_validation.hpp, with the same names and call shapes. Once that PR lands,
// delete this block and include that header instead; no call site changes.
struct DeviceTensorRequirements {
    std::vector<tt::tt_metal::DataType> dtypes = {tt::tt_metal::DataType::BFLOAT16};
    tt::tt_metal::Layout layout = tt::tt_metal::Layout::TILE;
    std::optional<tt::tt_metal::TensorMemoryLayout> memory_layout = tt::tt_metal::TensorMemoryLayout::INTERLEAVED;
};

void check_device_tensor(
    const ttnn::Tensor& tensor, std::string_view op, std::string_view name, const DeviceTensorRequirements& req = {}) {
    TT_FATAL(
        tensor.storage_type() == ttnn::StorageType::DEVICE,
        "{}: {} must be on Device. Storage type: {}",
        op,
        name,
        enchantum::to_string(tensor.storage_type()));
    TT_FATAL(tensor.buffer() != nullptr, "{}: {} buffer is null", op, name);
    TT_FATAL(
        tensor.layout() == req.layout,
        "{}: {} requires {} layout. Got: {}",
        op,
        name,
        enchantum::to_string(req.layout),
        enchantum::to_string(tensor.layout()));
    TT_FATAL(
        std::find(req.dtypes.begin(), req.dtypes.end(), tensor.dtype()) != req.dtypes.end(),
        "{}: {} has unsupported dtype {}",
        op,
        name,
        enchantum::to_string(tensor.dtype()));
    if (req.memory_layout.has_value()) {
        TT_FATAL(
            tensor.memory_config().memory_layout() == *req.memory_layout,
            "{}: {} requires {} memory layout. Got: {}",
            op,
            name,
            enchantum::to_string(*req.memory_layout),
            enchantum::to_string(tensor.memory_config().memory_layout()));
    }
}

void check_same_device(
    const ttnn::Tensor& tensor,
    const ttnn::Tensor& reference,
    std::string_view op,
    std::string_view name,
    std::string_view reference_name) {
    TT_FATAL(
        tensor.device() == reference.device(), "{}: {} must be on the same device as {}", op, name, reference_name);
}

// The shape this op WILL write, derived from the logits alone (the writer derives its output pages
// from the logits geometry, never from the output tensor). Matches ttnn::argmax(dim=3, keepdim):
// [B, 1, tokens, 1], or [B, 1, 1, 1] in position mode.
tt::tt_metal::Shape expected_output_shape(bool position_aware, const ttnn::Tensor& logits) {
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
        // Storage / buffer / layout / dtype / memory layout ride the shared helper; only the
        // op-specific checks (default tile geometry, derived padded shape) stay local.
        check_device_tensor(
            tensor,
            "GumbelSample",
            name,
            {.dtypes = {tt::tt_metal::DataType::BFLOAT16, tt::tt_metal::DataType::FLOAT32}});

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
    };

    const auto& logits = tensor_args.logits;
    check_tensor(logits, "logits");

    // The logits pick the device; every other tensor contributes only a raw buffer ADDRESS, which
    // is meaningless on any other device (a mismatch is a silent wrong-memory read, not a fault).
    auto* device = logits.device();
    TT_FATAL(device != nullptr, "GumbelSample: logits are not associated with a device");

    // An out-of-range seed axis is a caller bug (typo / config reused across topologies) that would
    // otherwise silently degrade to every device drawing identical noise. Extent-1 axes are
    // deliberately allowed: they are how a topology-generic caller says "seed if sharded here".
    const auto mesh_shape = device->shape();
    for (const uint32_t axis : args.seed_axes) {
        TT_FATAL(
            axis < mesh_shape.dims(),
            "GumbelSample: seed_axes entry {} does not exist on this {}-dimensional mesh {}",
            axis,
            mesh_shape.dims(),
            mesh_shape);
    }
    // No seeded axis on a multi-device mesh is legal (TP-only replication wants identical noise)
    // but also looks exactly like a forgotten seed_axes -- say so at debug level.
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

    TT_FATAL(logits.padded_shape().rank() == 4U, "GumbelSample: logits must be 4D");

    TT_FATAL(
        args.temperature >= 0.0F && std::isfinite(args.temperature),
        "GumbelSample: temperature must be finite and >= 0, got {}. Zero -- or a positive value below ~2.9e-39, whose "
        "reciprocal overflows float32 -- selects greedy argmax; other positive values select Gumbel-max sampling.",
        args.temperature);

    // seed == 0 is deliberately accepted: this op drives the SFPU LFSR directly (not ttnn::rand),
    // so zero carries no host-entropy contract and yields an ordinary reproducible stream.

    if (tensor_args.logits_mask.has_value()) {
        const auto& mask = tensor_args.logits_mask.value();
        check_tensor(mask, "logits_mask");
        // Only the mask's buffer ADDRESS reaches the kernel, and addresses are not portable across
        // devices.
        check_same_device(mask, logits, "GumbelSample", "logits_mask", "logits");
        TT_FATAL(
            mask.dtype() == logits.dtype(),
            "GumbelSample: mask dtype '{}' must match logits dtype '{}' (the public ttml::metal::gumbel_sample "
            "wrapper typecasts a mismatched mask, so only direct prim callers can trip this)",
            enchantum::to_string(mask.dtype()),
            enchantum::to_string(logits.dtype()));
        // Two accepted shapes, both broadcast down the token rows: [1, 1, 1, V] (shared vocab
        // padding) and [B, 1, 1, V] (per-request logit bias). A per-token-position mask stays
        // rejected -- it would reintroduce an O(B*T*V) tensor at prefill.
        TT_FATAL(
            mask.logical_shape()[-1] == logits.logical_shape()[-1],
            "GumbelSample: mask width {} must match logits width {}",
            mask.logical_shape()[-1],
            logits.logical_shape()[-1]);
        TT_FATAL(
            mask.logical_shape()[-2] == 1U,
            "GumbelSample: mask token dim must be 1 (it is broadcast across all token rows), got {}. Build the "
            "mask as [1, 1, 1, V] or [B, 1, 1, V].",
            mask.logical_shape()[-2]);
        TT_FATAL(
            mask.logical_shape()[-3] == 1U,
            "GumbelSample: mask channel dim must be 1, got {}",
            mask.logical_shape()[-3]);
        // Batch dim: 1 (shared) or exactly the logits' local batch; anything else leaves the
        // reader's page walk on non-corresponding rows -- in bounds, silently wrong bias.
        const uint32_t mask_batch = mask.logical_shape()[-4];
        TT_FATAL(
            mask_batch == 1U || mask_batch == logits.logical_shape()[-4],
            "GumbelSample: mask batch dim must be 1 (one row shared by every batch entry) or match the logits "
            "batch {} (one row per entry), got {}",
            logits.logical_shape()[-4],
            mask_batch);
        if (mask_batch > 1U) {
            // The reader walks mask pages by entry index over NC = dims[-4] * dims[-3], but a
            // per-row mask expresses only dim -4 -- with C > 1 the walk silently leaves the mask's
            // pages. (A shared mask is stride-0 and safe for any NC.)
            TT_FATAL(
                logits.logical_shape()[-3] == 1U,
                "GumbelSample: a per-row [B, 1, 1, V] mask requires channel-1 logits ([B, 1, T, V]); these logits "
                "have channel dim {}. Use a shared [1, 1, 1, V] mask instead.",
                logits.logical_shape()[-3]);
            // Per-device-row data, exactly like positions: page e must be entry e's bias, which
            // only holds if both tensors shard the batch identically.
            TT_FATAL(
                mask.tensor_topology() == logits.tensor_topology(),
                "GumbelSample: a per-row [B, 1, 1, V] mask must be distributed across the mesh exactly as the "
                "logits are -- shard it with the SAME mapper the batch was sharded with");
        }
    }

    // [B, 1, 1, 1] UINT32 ROW_MAJOR INTERLEAVED on this device -- byte-for-byte the op's own
    // position-mode output spec, so a previous sample's output can be fed straight back in.
    // ROW_MAJOR so each page is one aligned word, not a padded 4 KB tile; INTERLEAVED because the
    // program hash keys buffer type only, so a sharded tensor would reuse an accessor compiled for
    // the interleaved encoding.
    auto check_index_tensor = [&](const ttnn::Tensor& t, const std::string& name, bool position_aware) {
        check_device_tensor(
            t,
            "GumbelSample",
            name,
            {.dtypes = {tt::tt_metal::DataType::UINT32}, .layout = tt::tt_metal::Layout::ROW_MAJOR});
        check_same_device(t, logits, "GumbelSample", name, "logits");
        // The exact shape guarantees page e is in bounds for every entry the kernels index.
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

    // seed and the temperature VALUE are runtime-only (re-applied on cache hits) so a training loop
    // reuses one cached program. WHICH KERNEL the temperature selects IS keyed, via the same
    // uses_gumbel_noise predicate the factory uses -- a bare `temperature > 0` would collide the
    // sub-reciprocal-overflow greedy build with noisy programs. seed_axes changes which mesh
    // coordinates get distinct programs.
    //
    // Buffer PLACEMENT is keyed, not just presence: TensorAccessorArgs bakes DRAM/L1 into the
    // compile-time args, and a placement collision would aim accessors compiled for one memory
    // space at the other's addresses.
    auto placement_of = [](const std::optional<ttnn::Tensor>& tensor) -> int {
        return tensor.has_value() ? static_cast<int>(tensor->memory_config().buffer_type()) : -1;
    };

    const bool position_aware = tensor_args.positions.has_value();

    // In position mode the program does not depend on the token dimension at all (the split is
    // NC * Wt; Ht and logical_tokens are runtime args; the writer's compile-time Ht is pinned), so
    // dim -2 is normalized out of the key and one program serves every prompt length -- otherwise
    // every new prompt length is a fresh ~6 s JIT build. NORMALIZED rather than omitted because
    // hash_operation has no arity tag, so dropping an argument would let unrelated keys alias.
    // Load-bearing consequences: every token-derived runtime arg MUST be re-applied in
    // override_runtime_arguments, and total_tiles must stay NC * Wt in position mode.
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
        // The padded shape is not hashed: check_tensor pins it to the logical shape's round-up, so
        // it is fully derived from what IS in the key. Likewise the positions shape: its entry
        // count is NC, already in the key via (unrounded) dims 0 and 1.
        token_normalized(logits.logical_shape()),
        static_cast<int>(logits.memory_config().buffer_type()),
        tensor_args.logits_mask.has_value(),
        placement_of(tensor_args.logits_mask),
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
    const std::optional<ttnn::Tensor>& logits_mask,
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
        .logits_mask = logits_mask,
        .positions = positions,
        .preallocated_output = preallocated_output,
    };

    return ttnn::device_operation::launch<OperationType>(operation_attributes, tensor_args);
}

}  // namespace ttnn::prim
