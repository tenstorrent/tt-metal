// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "variable_matmul_device_operation.hpp"

#include <tt-metalium/constants.hpp>
#include <tt-metalium/hal.hpp>
#include <tt-metalium/math.hpp>
#include <tt-metalium/tt_metal.hpp>

#include "ttnn/tensor/tensor_ops.hpp"
#include "variable_matmul_program_factory.hpp"

using namespace tt::constants;
using namespace tt::tt_metal;

namespace ttml::metal::ops::variable_matmul::device {

void VariableMatmulDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    const auto& act_tensor = tensor_args.input_tensor;
    const auto& weight_tensor = tensor_args.weight_tensor;
    const auto& config = operation_attributes.config;

    TT_FATAL(act_tensor.storage_type() == StorageType::DEVICE, "variable_matmul activation must be on device");
    auto* device = act_tensor.device();
    auto check_on_device = [device](const ttnn::Tensor& t, const char* name) {
        TT_FATAL(t.storage_type() == StorageType::DEVICE, "variable_matmul {} must be on device", name);
        TT_FATAL(t.buffer() != nullptr, "variable_matmul {} must be allocated in a device buffer", name);
        TT_FATAL(t.device() == device, "variable_matmul {} must reside on the same device as the input", name);
    };
    check_on_device(act_tensor, "activation");
    check_on_device(weight_tensor, "weight");

    TT_FATAL(
        act_tensor.layout() == Layout::TILE && weight_tensor.layout() == Layout::TILE,
        "variable_matmul requires TILE layout for activation and weight");

    auto dtype_supported = [](tt::tt_metal::DataType dt) {
        return dt == DataType::BFLOAT16 || dt == DataType::BFLOAT8_B || dt == DataType::BFLOAT4_B ||
               dt == DataType::FLOAT32;
    };
    TT_FATAL(
        dtype_supported(act_tensor.dtype()) && dtype_supported(weight_tensor.dtype()),
        "variable_matmul supports only BFLOAT16, BFLOAT8_B, BFLOAT4_B, and FLOAT32 for inputs");

    const auto& a_logical = act_tensor.logical_shape();
    const auto& w_logical = weight_tensor.logical_shape();
    TT_FATAL(a_logical.rank() >= 2 && w_logical.rank() >= 2, "variable_matmul expects rank >= 2 tensors");

    // Batched input not supported
    auto leading_dims_volume = [](const ttnn::Shape& s) { return s.volume() / (static_cast<uint64_t>(s[-2]) * s[-1]); };
    TT_FATAL(
        leading_dims_volume(a_logical) == 1,
        "variable_matmul: input leading dims must all be 1 (op is scoped to a 2D matmul over the last two dims)");
    TT_FATAL(
        leading_dims_volume(w_logical) == 1,
        "variable_matmul: weight leading dims must all be 1 (op is scoped to a 2D matmul over the last two dims)");

    // With transpose_a, the input is stored as [K, M], so M is at [-1] and K at [-2].
    const uint32_t M_parent = operation_attributes.transpose_a ? a_logical[-1] : a_logical[-2];
    const uint32_t K_in = operation_attributes.transpose_a ? a_logical[-2] : a_logical[-1];
    // When transpose_b, the weight is stored as [N, K] and K matches w_logical[-1].
    const uint32_t K_w = operation_attributes.transpose_b ? w_logical[-1] : w_logical[-2];

    // Matmul-M: use expected_M_tiles when the caller set it (a hint for offset-read mode),
    // else fall back to the input tensor's full M.
    const uint32_t expected_M_tiles = operation_attributes.expected_M_tiles;
    const uint32_t M_parent_tiles = tt::div_up(M_parent, TILE_HEIGHT);
    const uint32_t K_in_tiles = K_in / TILE_WIDTH;
    const uint32_t K_w_tiles = K_w / TILE_WIDTH;
    const uint32_t M = (expected_M_tiles > 0) ? (expected_M_tiles * TILE_HEIGHT) : M_parent;

    // K-axis: when K_w > K_in the weight is the parent and matmul-K = K_in; otherwise the
    // input is the parent (or both match) and matmul-K = K_w. The InputAndWeightK role
    // overrides the K-offset within the parent at runtime.
    const bool in1_parent_mode = K_w_tiles > K_in_tiles;
    if (in1_parent_mode) {
        TT_FATAL(
            K_in_tiles <= K_w_tiles, "variable_matmul K_in tiles {} exceeds weight K tiles {}", K_in_tiles, K_w_tiles);
    } else {
        TT_FATAL(
            K_w_tiles <= K_in_tiles, "variable_matmul K_w tiles {} exceeds input K tiles {}", K_w_tiles, K_in_tiles);
    }
    TT_FATAL(M > 0 && K_w > 0, "variable_matmul dimensions must be positive");
    // Non-tile-aligned matmul-M is supported (like minimal_matmul / ttnn::matmul): TILE storage
    // rounds M up to TILE_HEIGHT, the kernel works on padded tiles, and the writer's
    // `if (i >= logical_d0) break;` clips at the logical boundary. The last tile's pad rows are
    // zero in → zero out, so they're harmless.
    const uint32_t M_tiles_ceil = tt::div_up(M, TILE_HEIGHT);
    TT_FATAL(
        M_tiles_ceil <= M_parent_tiles,
        "variable_matmul expected_M tiles {} exceeds input M tiles {}",
        M_tiles_ceil,
        M_parent_tiles);

    const auto& a_padded = act_tensor.padded_shape();
    const auto& w_padded = weight_tensor.padded_shape();
    TT_FATAL(
        a_padded[-2] % TILE_HEIGHT == 0 && a_padded[-1] % TILE_WIDTH == 0,
        "variable_matmul activation must be tile-aligned");
    TT_FATAL(
        w_padded[-2] % TILE_HEIGHT == 0 && w_padded[-1] % TILE_WIDTH == 0,
        "variable_matmul weight must be tile-aligned");

    const auto& cfg = config;
    TT_FATAL(cfg.M_block_size > 0 && cfg.K_block_size > 0 && cfg.N_block_size > 0, "Block sizes must be > 0");
    TT_FATAL(cfg.subblock_h > 0 && cfg.subblock_w > 0, "Subblock sizes must be > 0");
    TT_FATAL(
        (cfg.M_block_size % cfg.subblock_h) == 0,
        "M_block_size ({}) must be divisible by subblock_h ({})",
        cfg.M_block_size,
        cfg.subblock_h);
    TT_FATAL(
        (cfg.N_block_size % cfg.subblock_w) == 0,
        "N_block_size ({}) must be divisible by subblock_w ({})",
        cfg.N_block_size,
        cfg.subblock_w);

    TT_FATAL(
        cfg.compute_with_storage_grid_size.x >= 2 && cfg.compute_with_storage_grid_size.y >= 2,
        "compute_with_storage_grid_size must be >= 2x2");

    auto device_grid = act_tensor.device()->compute_with_storage_grid_size();
    TT_FATAL(
        cfg.compute_with_storage_grid_size.x <= device_grid.x && cfg.compute_with_storage_grid_size.y <= device_grid.y,
        "compute_with_storage_grid_size must be <= device grid size");

    const uint32_t max_dest_volume = get_dest_reg_count(operation_attributes.compute_kernel_config);
    TT_FATAL(cfg.subblock_h * cfg.subblock_w <= max_dest_volume, "subblock_h * subblock_w must be <= max_dest_volume");

    if (tensor_args.output_tensor.has_value()) {
        const auto& out = tensor_args.output_tensor.value();
        check_on_device(out, "output tensor");
        const auto& out_logical = out.logical_shape();
        const uint32_t matmul_N = operation_attributes.transpose_b ? w_logical[-2] : w_logical[-1];
        const uint32_t out_N = out_logical[-1];
        TT_FATAL(out.layout() == Layout::TILE, "variable_matmul output tensor must be TILE layout");
        TT_FATAL(out.dtype() == act_tensor.dtype(), "variable_matmul output dtype must match input dtype");
        TT_FATAL(out_N == matmul_N, "variable_matmul output N ({}) must match matmul N ({})", out_N, matmul_N);
        // Write-at-offset writes the same row indices it reads, so the output parent must be the
        // same row space as the input.
        TT_FATAL(
            M_parent == out_logical[-2],
            "variable_matmul output rows ({}) must equal input rows ({})",
            out_logical[-2],
            M_parent);
    }

    const auto role = operation_attributes.offsets_role;
    TT_FATAL(
        role == OffsetsRole::InputAndOutputRow || role == OffsetsRole::InputAndWeightK,
        "variable_matmul: offsets_role must be InputAndOutputRow or InputAndWeightK.");
    if (role == OffsetsRole::InputAndOutputRow) {
        TT_FATAL(
            tensor_args.output_tensor.has_value(),
            "variable_matmul: OffsetsRole::InputAndOutputRow requires a caller-provided output_tensor.");
    } else {  // InputAndWeightK
        TT_FATAL(
            !tensor_args.output_tensor.has_value(),
            "variable_matmul: OffsetsRole::InputAndWeightK allocates its own output; output_tensor must be nullopt.");
    }
    const auto& off = tensor_args.offsets_tensor;
    check_on_device(off, "offsets_tensor");
    TT_FATAL(off.dtype() == ttnn::DataType::UINT32, "variable_matmul: offsets_tensor must be UINT32.");
    TT_FATAL(off.layout() == ttnn::Layout::ROW_MAJOR, "variable_matmul: offsets_tensor must be ROW_MAJOR.");
    // The kernels read offsets[start..start+2] (a {start, end} pair). Bound the start index so
    // the pair stays within the tensor — an out-of-range index would otherwise silently read
    // adjacent L1 garbage rather than failing here.
    const uint32_t offsets_volume = off.logical_shape().volume();
    TT_FATAL(
        operation_attributes.offsets_start_index + 2U <= offsets_volume,
        "variable_matmul: offsets_start_index ({}) + 2 exceeds offsets_tensor volume ({}).",
        operation_attributes.offsets_start_index,
        offsets_volume);
    // The dataflow kernels read exactly ONE page of the offsets tensor and index within it, so
    // the {start, end} pair must fall inside page 0. Enforce the documented limitation here.
    const uint32_t offsets_page_bytes = off.buffer()->page_size();
    TT_FATAL(
        (operation_attributes.offsets_start_index + 2U) * sizeof(uint32_t) <= offsets_page_bytes,
        "variable_matmul: offsets_start_index ({}) + 2 spills past offsets_tensor page 0 ({} bytes); "
        "the on-device offset read only covers page 0.",
        operation_attributes.offsets_start_index,
        offsets_page_bytes);
}

VariableMatmulDeviceOperation::spec_return_value_t VariableMatmulDeviceOperation::compute_output_specs(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    // Write-at-offset: when caller provides an output tensor, its spec is the parent's
    // (not the matmul-sized one) — the kernel writes into a row-range of it.
    if (tensor_args.output_tensor.has_value()) {
        return tensor_args.output_tensor->tensor_spec();
    }
    const auto& in0 = tensor_args.input_tensor;
    const auto& in1 = tensor_args.weight_tensor;
    // With transpose_b, the weight is stored as [N, K] so N is at logical[-2].
    const uint32_t N = operation_attributes.transpose_b ? in1.logical_shape()[-2] : in1.logical_shape()[-1];
    // M dimension: use expected_M_tiles when provided (offset-read mode); otherwise derive
    // from the input tensor's matmul-M dim ([-1] if transpose_a, else [-2]).
    const uint32_t M_from_input = operation_attributes.transpose_a ? in0.logical_shape()[-1] : in0.logical_shape()[-2];
    const uint32_t M = (operation_attributes.expected_M_tiles > 0)
                           ? (operation_attributes.expected_M_tiles * tt::constants::TILE_HEIGHT)
                           : M_from_input;

    const auto& memory_config = in0.memory_config();
    auto dtype = in0.dtype();

    ttnn::Shape output_shape(in0.logical_shape());
    output_shape[-2] = M;
    output_shape[-1] = N;
    return tt::tt_metal::TensorSpec(output_shape, TensorLayout(dtype, PageConfig(Layout::TILE), memory_config));
}

VariableMatmulDeviceOperation::tensor_return_value_t VariableMatmulDeviceOperation::create_output_tensors(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    if (tensor_args.output_tensor.has_value()) {
        return tensor_args.output_tensor.value();
    }
    const auto output_spec = compute_output_specs(operation_attributes, tensor_args);
    auto* device = tensor_args.input_tensor.device();
    return ttnn::create_device_tensor(output_spec, device);
}

ttsl::hash::hash_t VariableMatmulDeviceOperation::compute_program_hash(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    // Program hash. M and matmul-K are runtime args, so they're omitted; the hash keys on N,
    // the transpose flags, the use_offset / use_offset_in1 / has-output-tensor CTAs, grid+block
    // sizing, and transpose_core_grid (so M-vs-N orientation is stable across offset-read calls).
    const auto& w = tensor_args.weight_tensor;
    const auto& a = tensor_args.input_tensor;
    const bool transpose_a = operation_attributes.transpose_a;
    const bool transpose_b = operation_attributes.transpose_b;
    // Derive tile counts from padded_shape() — the program factory builds from padded shapes, so
    // the hash MUST read the same source. Using logical shapes here would floor-divide where the
    // factory ceil-divides (padded): two non-tile-aligned K shapes that floor to the same tile
    // count but pad differently could then collide on the hash yet need different use_offset
    // programs → wrong cache hit. (Tile-aligned shapes are unaffected; floor == ceil there.)
    const auto& a_padded = a.padded_shape();
    const auto& w_padded = w.padded_shape();
    // use_offset is a compile-time knob (offset-enabled vs disabled → two cached programs,
    // keeping the offset=0 hot path at baseline). Parent-K mode (input K extent > weight's)
    // also forces it on, since the kernel then needs the parent stride to read correctly.
    const uint32_t K_in_tiles = (transpose_a ? a_padded[-2] : a_padded[-1]) / tt::constants::TILE_WIDTH;
    const uint32_t K_w_tiles = (transpose_b ? w_padded[-1] : w_padded[-2]) / tt::constants::TILE_WIDTH;
    // in0 side: parent-K is K_in > K_matmul. K_matmul = K_w when in0 is parent, K_in when in1 is.
    const bool in1_parent_k_mode = K_w_tiles > K_in_tiles;
    const bool in0_parent_k_mode = !in1_parent_k_mode && K_in_tiles > K_w_tiles;
    // The offsets role forces use_offset on (InputAndOutputRow → in0 row offset; InputAndWeightK →
    // both in0+in1 K-offsets), so this MUST mirror the program factory's use_offset_in0 /
    // use_offset_in1 exactly — otherwise the cached program's compile-time flags would diverge from
    // what compute_program_hash predicted.
    const bool input_and_weight_k_active = operation_attributes.offsets_role == OffsetsRole::InputAndWeightK;
    const bool input_and_output_row_active = operation_attributes.offsets_role == OffsetsRole::InputAndOutputRow;
    const bool use_offset = input_and_output_row_active || input_and_weight_k_active ||
                            operation_attributes.expected_M_tiles > 0 || in0_parent_k_mode;
    const bool use_offset_in1 = in1_parent_k_mode || input_and_weight_k_active;
    // Mirror the program factory's transpose_core_grid decision exactly: use the caller's
    // expected_M_tiles when set (per-call M upper bound), else fall back to the input's
    // full M dimension. Comparison in tile units.
    const uint32_t parent_M_for_hash = transpose_a ? a_padded[-1] : a_padded[-2];
    const uint32_t parent_M_tiles_for_hash = tt::div_up(parent_M_for_hash, tt::constants::TILE_HEIGHT);
    const uint32_t logical_M_tiles_for_hash =
        (operation_attributes.expected_M_tiles > 0) ? operation_attributes.expected_M_tiles : parent_M_tiles_for_hash;
    const uint32_t N = transpose_b ? w_padded[-2] : w_padded[-1];
    const uint32_t N_tiles_for_hash = N / tt::constants::TILE_WIDTH;
    const bool transpose_core_grid = logical_M_tiles_for_hash > N_tiles_for_hash;

    // Variable-K: only N must be in the hash (matmul-K is RT-driven, fully variable). N is the
    // padded matmul-N (computed above) so the hash groups calls that share a compiled program.
    return ttsl::hash::hash_objects_with_default_seed(
        transpose_core_grid,
        operation_attributes.config.M_block_size,
        operation_attributes.config.K_block_size,
        operation_attributes.config.N_block_size,
        operation_attributes.config.subblock_h,
        operation_attributes.config.subblock_w,
        operation_attributes.config.compute_with_storage_grid_size.x,
        operation_attributes.config.compute_with_storage_grid_size.y,
        operation_attributes.compute_kernel_config,
        a.dtype(),
        w.dtype(),
        N,
        transpose_a,
        transpose_b,
        use_offset,
        use_offset_in1,
        // Write-at-offset toggles a CTA path. Boolean only (offset value is RT-only).
        tensor_args.output_tensor.has_value(),
        // On-device offsets: different role = different cached program (CTA define).
        static_cast<uint32_t>(operation_attributes.offsets_role));
}

}  // namespace ttml::metal::ops::variable_matmul::device

namespace ttnn::prim {

ttnn::Tensor ttml_variable_matmul(
    const ttnn::Tensor& input_tensor,
    const ttnn::Tensor& weight_tensor,
    const ttml::metal::ops::variable_matmul::device::VariableMatmulConfig& config,
    const ttnn::Tensor& offsets_tensor,
    ttml::metal::ops::variable_matmul::device::OffsetsRole offsets_role,
    bool transpose_a,
    bool transpose_b,
    std::optional<ttnn::DeviceComputeKernelConfig> compute_kernel_config,
    std::optional<ttnn::Tensor> output_tensor,
    uint32_t offsets_start_index,
    uint32_t expected_M_tiles) {
    using OperationType = ttml::metal::ops::variable_matmul::device::VariableMatmulDeviceOperation;
    auto kernel_config_val = init_device_compute_kernel_config(
        input_tensor.device()->arch(),
        compute_kernel_config,
        tt::tt_metal::MathFidelity::HiFi4,
        false /*approx_mode*/,
        true /*fp32_acc*/,
        true /*packer_acc*/);

    return ttnn::device_operation::launch<OperationType>(
        OperationType::operation_attributes_t{
            .config = config,
            .compute_kernel_config = kernel_config_val,
            .transpose_a = transpose_a,
            .transpose_b = transpose_b,
            .expected_M_tiles = expected_M_tiles,
            .offsets_role = offsets_role,
            .offsets_start_index = offsets_start_index,
        },
        OperationType::tensor_args_t{
            .input_tensor = input_tensor,
            .weight_tensor = weight_tensor,
            .output_tensor = output_tensor,
            .offsets_tensor = offsets_tensor,
        });
}

}  // namespace ttnn::prim
