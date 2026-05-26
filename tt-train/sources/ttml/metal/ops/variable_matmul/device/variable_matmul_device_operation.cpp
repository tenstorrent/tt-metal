// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
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

    // Basic device/storage checks
    TT_FATAL(
        act_tensor.storage_type() == StorageType::DEVICE && weight_tensor.storage_type() == StorageType::DEVICE,
        "variable_matmul operands must be on device");
    TT_FATAL(act_tensor.device() == weight_tensor.device(), "variable_matmul inputs must reside on the same device");
    TT_FATAL(
        act_tensor.buffer() != nullptr && weight_tensor.buffer() != nullptr,
        "variable_matmul inputs must be allocated in device buffers");

    // Layout requirements: all inputs must be TILE layout
    TT_FATAL(
        act_tensor.layout() == Layout::TILE && weight_tensor.layout() == Layout::TILE,
        "variable_matmul requires TILE layout for activation and weight");

    // DType constraints
    auto dtype_supported = [](tt::tt_metal::DataType dt) {
        return dt == DataType::BFLOAT16 || dt == DataType::BFLOAT8_B || dt == DataType::BFLOAT4_B ||
               dt == DataType::FLOAT32;
    };
    TT_FATAL(
        dtype_supported(act_tensor.dtype()) && dtype_supported(weight_tensor.dtype()),
        "variable_matmul supports only BFLOAT16, BFLOAT8_B, BFLOAT4_B, and FLOAT32 for inputs");

    // Shape constraints
    const auto& a_logical = act_tensor.logical_shape();
    const auto& w_logical = weight_tensor.logical_shape();
    TT_FATAL(a_logical.rank() >= 2 && w_logical.rank() >= 2, "variable_matmul expects rank >= 2 tensors");

    // With transpose_a, the input is stored as [K, M], so M is at [-1] and K at [-2].
    const uint32_t M_parent = operation_attributes.transpose_a ? a_logical[-1] : a_logical[-2];
    const uint32_t K_in = operation_attributes.transpose_a ? a_logical[-2] : a_logical[-1];
    // When transpose_b, the weight is stored as [N, K] and K matches w_logical[-1].
    const uint32_t K_w = operation_attributes.transpose_b ? w_logical[-1] : w_logical[-2];

    // Effective M for the matmul (after applying length); falls back to the input
    // tensor's full M when caller didn't specify a sub-range.
    const uint32_t effective_M_tiles = operation_attributes.effective_M_tiles;
    const uint32_t M_parent_tiles = tt::div_up(M_parent, TILE_HEIGHT);
    const uint32_t K_in_tiles = K_in / TILE_WIDTH;
    const uint32_t K_w_tiles = K_w / TILE_WIDTH;
    const uint32_t M = (effective_M_tiles > 0) ? (effective_M_tiles * TILE_HEIGHT) : M_parent;

    // K-axis: when K_w > K_in the weight is the parent and matmul-K = K_in; otherwise the
    // input is the parent (or both match) and matmul-K = K_w. The EP path's
    // InputK/WeightK/InputAndWeightK roles override the K-offset within the parent at runtime.
    const bool in1_parent_mode = K_w_tiles > K_in_tiles;
    if (in1_parent_mode) {
        TT_FATAL(
            K_in_tiles <= K_w_tiles, "variable_matmul K_in tiles {} exceeds weight K tiles {}", K_in_tiles, K_w_tiles);
    } else {
        TT_FATAL(
            K_w_tiles <= K_in_tiles, "variable_matmul K_w tiles {} exceeds input K tiles {}", K_w_tiles, K_in_tiles);
    }
    TT_FATAL(M > 0 && K_w > 0, "variable_matmul dimensions must be positive");
    // Non-tile-aligned matmul-M is supported (matches minimal_matmul / ttnn::matmul): the
    // tensor's TILE-layout physical storage rounds up to a multiple of TILE_HEIGHT, the
    // kernel operates on padded tiles, and the dataflow writer's `if (i >= logical_d0)
    // break;` clips writes at the logical tile boundary. The last (partial) M tile's
    // padded rows are zero on input → zero on output, so they're harmless.
    const uint32_t M_tiles_ceil = tt::div_up(M, TILE_HEIGHT);
    TT_FATAL(
        M_tiles_ceil <= M_parent_tiles,
        "variable_matmul effective_M tiles {} exceeds input M tiles {}",
        M_tiles_ceil,
        M_parent_tiles);

    // Tile alignment checks
    const auto& a_padded = act_tensor.padded_shape();
    const auto& w_padded = weight_tensor.padded_shape();
    TT_FATAL(
        a_padded[-2] % TILE_HEIGHT == 0 && a_padded[-1] % TILE_WIDTH == 0,
        "variable_matmul activation must be tile-aligned");
    TT_FATAL(
        w_padded[-2] % TILE_HEIGHT == 0 && w_padded[-1] % TILE_WIDTH == 0,
        "variable_matmul weight must be tile-aligned");

    // Config constraints
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

    // Write-at-offset validation
    if (tensor_args.output_tensor.has_value()) {
        const auto& out = tensor_args.output_tensor.value();
        const auto& out_logical = out.logical_shape();
        const uint32_t matmul_N = operation_attributes.transpose_b ? w_logical[-2] : w_logical[-1];
        const uint32_t out_M_tiles = tt::div_up(out_logical[-2], TILE_HEIGHT);
        const uint32_t out_N = out_logical[-1];
        const uint32_t M_tiles_actual = tt::div_up(M, TILE_HEIGHT);
        TT_FATAL(out.layout() == Layout::TILE, "variable_matmul output tensor must be TILE layout");
        TT_FATAL(out.dtype() == act_tensor.dtype(), "variable_matmul output dtype must match input dtype");
        TT_FATAL(out_N == matmul_N, "variable_matmul output N ({}) must match matmul N ({})", out_N, matmul_N);
        TT_FATAL(
            M_tiles_actual <= out_M_tiles,
            "variable_matmul actual_M tiles {} exceeds output M tiles {}",
            M_tiles_actual,
            out_M_tiles);
    }

    // On-device offsets validation.
    const bool has_offsets = tensor_args.offsets_tensor.has_value();
    const bool role_active = operation_attributes.offsets_role != OffsetsRole::None;
    TT_FATAL(
        has_offsets == role_active,
        "variable_matmul: offsets_tensor and offsets_role must both be set or both be unset.");
    if (role_active) {
        const auto role = operation_attributes.offsets_role;
        TT_FATAL(
            role == OffsetsRole::OutputRow || role == OffsetsRole::InputRow || role == OffsetsRole::InputK ||
                role == OffsetsRole::WeightK || role == OffsetsRole::InputAndOutputRow ||
                role == OffsetsRole::InputAndWeightK,
            "variable_matmul: unsupported OffsetsRole value.");
        if (role == OffsetsRole::OutputRow || role == OffsetsRole::InputAndOutputRow) {
            TT_FATAL(
                tensor_args.output_tensor.has_value(),
                "variable_matmul: OffsetsRole::OutputRow / InputAndOutputRow requires a caller-provided "
                "output_tensor.");
        }
        const auto& off = tensor_args.offsets_tensor.value();
        TT_FATAL(off.dtype() == ttnn::DataType::UINT32, "variable_matmul: offsets_tensor must be UINT32.");
        TT_FATAL(off.layout() == ttnn::Layout::ROW_MAJOR, "variable_matmul: offsets_tensor must be ROW_MAJOR.");
    }
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
    // M dimension: use effective_M_tiles when provided (offset-read mode); otherwise derive
    // from the input tensor's matmul-M dim ([-1] if transpose_a, else [-2]).
    const uint32_t M_from_input = operation_attributes.transpose_a ? in0.logical_shape()[-1] : in0.logical_shape()[-2];
    const uint32_t M = (operation_attributes.effective_M_tiles > 0)
                           ? (operation_attributes.effective_M_tiles * tt::constants::TILE_HEIGHT)
                           : M_from_input;

    const auto& memory_config = in0.memory_config();
    auto dtype = in0.dtype();

    ttnn::Shape output_shape(in0.logical_shape());
    output_shape[-2] = M;
    output_shape[-1] = N;
    return TensorSpec(output_shape, TensorLayout(dtype, PageConfig(Layout::TILE), memory_config));
}

VariableMatmulDeviceOperation::tensor_return_value_t VariableMatmulDeviceOperation::create_output_tensors(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    if (tensor_args.output_tensor.has_value()) {
        return tensor_args.output_tensor.value();
    }
    const auto output_spec = compute_output_specs(operation_attributes, tensor_args);
    auto* device = tensor_args.input_tensor.device();
    return create_device_tensor(output_spec, device);
}

ttsl::hash::hash_t VariableMatmulDeviceOperation::compute_program_hash(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    // Program hash. M and matmul-K are runtime args (no recompilation for different
    // M or K within a transpose/offset variant). The hash keys on N, the transpose
    // flags, use_offset / use_offset_in1 / has-output-tensor CTAs, and grid+block
    // sizing. transpose_core_grid (parent_M vs N) is included so the M-vs-N grid
    // orientation is stable across offset-read calls on the same parent.
    const auto& w = tensor_args.weight_tensor;
    const auto& a = tensor_args.input_tensor;
    const bool transpose_a = operation_attributes.transpose_a;
    const bool transpose_b = operation_attributes.transpose_b;
    // use_offset is a compile-time kernel knob: when true, the address formula adds
    // M-offset and K-offset to the respective axes. Splitting on this gives two cached
    // programs (offset-enabled / disabled) but keeps the (offset=0) hot path at baseline.
    // Parent-K mode: input has a larger K extent than the weight's. This sets use_offset
    // even with k_offset=0, since the kernel needs the parent stride to read correctly.
    const uint32_t K_in_tiles =
        (transpose_a ? a.logical_shape()[-2] : a.logical_shape()[-1]) / tt::constants::TILE_WIDTH;
    const uint32_t K_w_tiles =
        (transpose_b ? w.logical_shape()[-1] : w.logical_shape()[-2]) / tt::constants::TILE_WIDTH;
    // in0 side: parent-K is K_in > K_matmul. K_matmul = K_w when in0 is parent, K_in when in1 is.
    const bool in1_parent_k_mode = K_w_tiles > K_in_tiles;
    const bool in0_parent_k_mode = !in1_parent_k_mode && K_in_tiles > K_w_tiles;
    // InputAndWeightK overrides both in0 and in1 K-offsets at runtime; force both use_offset
    // flags true so hash matches the program factory's CTA computation (otherwise the cached
    // program's compile-time flags would diverge from what compute_program_hash predicted).
    const bool input_and_weight_k_active =
        operation_attributes.offsets_role == OffsetsRole::InputAndWeightK && tensor_args.offsets_tensor.has_value();
    const bool use_offset =
        operation_attributes.effective_M_tiles > 0 || in0_parent_k_mode || input_and_weight_k_active;
    const bool use_offset_in1 = in1_parent_k_mode || input_and_weight_k_active;
    // physical_volume / padded[-1] gives the count along the *stored* outer dim. With
    // transpose_a that's K-tiles; without, M-tiles. Either way, actual_M (for the matmul) is
    // along the stored *other* dim: padded_shape[-2] when transpose_a, [-1] otherwise — both
    // equal physical_volume/padded[-1] only when transpose_a is false. Use logical shape to
    // disambiguate for the hash's M-vs-N comparison (purely for transpose_core_grid).
    uint32_t actual_M = transpose_a ? a.logical_shape()[-1] : (a.physical_volume() / a.padded_shape()[-1]);
    uint32_t N = transpose_b ? w.logical_shape()[-2] : w.logical_shape()[-1];
    bool transpose_core_grid = actual_M > N;

    // Variable-K: only N must be in the hash (matmul-K is RT-driven, fully variable).
    const uint32_t N_dim = transpose_b ? w.logical_shape()[-2] : w.logical_shape()[-1];
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
        N_dim,
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
    bool transpose_a,
    bool transpose_b,
    std::optional<ttnn::DeviceComputeKernelConfig> compute_kernel_config,
    std::optional<ttnn::Tensor> output_tensor,
    std::optional<ttnn::Tensor> offsets_tensor,
    ttml::metal::ops::variable_matmul::device::OffsetsRole offsets_role,
    uint32_t offsets_start_index,
    uint32_t effective_M_tiles) {
    using OperationType = ttml::metal::ops::variable_matmul::device::VariableMatmulDeviceOperation;
    auto kernel_config_val = init_device_compute_kernel_config(
        input_tensor.device()->arch(),
        compute_kernel_config,
        MathFidelity::HiFi4,
        false /*approx_mode*/,
        true /*fp32_acc*/,
        true /*packer_acc*/);

    return ttnn::device_operation::launch<OperationType>(
        OperationType::operation_attributes_t{
            .config = config,
            .compute_kernel_config = kernel_config_val,
            .transpose_a = transpose_a,
            .transpose_b = transpose_b,
            .effective_M_tiles = effective_M_tiles,
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
