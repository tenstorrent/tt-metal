// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>

#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"
#include "ttnn/tensor/tensor.hpp"

namespace ttml::metal::ops::variable_matmul::device {

struct VariableMatmulConfig {
    uint32_t M_block_size{};
    uint32_t K_block_size{};
    uint32_t N_block_size{};
    uint32_t subblock_h{};
    uint32_t subblock_w{};

    tt::tt_metal::CoreCoord compute_with_storage_grid_size = {0, 0};

    // When true, the input tensor is interpreted as transposed for matmul purposes:
    // stored shape [..., K, M] but used as [..., M, K]. Reader applies stride swap,
    // compute kernel applies intra-tile transpose via transpose_wh_tile into a dedicated CB.
    bool transpose_a = false;

    // When true, the weight tensor is interpreted as transposed for matmul purposes:
    // stored shape [..., N, K] but used as [..., K, N]. Reader applies stride swap,
    // matmul kernel applies intra-tile transpose via the LLK transpose flag.
    bool transpose_b = false;
};

// EP-friendly on-device offsets: instead of a host-supplied scalar
// out_row_offset_tiles, the kernel reads offsets[start_index] from a device tensor
// at runtime and uses (offsets[start_index] / TILE_HEIGHT) as the write-at-offset
// row. Lets moe_ffn avoid offsets.to_vector() under MeshDevice EP for the
// down_proj / dX_via_* calls.
enum class OffsetsRole : uint32_t {
    None = 0,
    OutputRow = 1,
};

struct VariableMatmulParams {
    VariableMatmulConfig config;
    ttnn::DeviceComputeKernelConfig compute_kernel_config;

    // Read-at-offset support — lets variable_matmul read a sub-range of the input tensor
    // without materializing a slice. The input tensor is treated as a parent buffer.
    //
    // M-axis offset (in0_row_offset_tiles + effective_M_tiles):
    //   only the M-rows [in0_row_offset_tiles, in0_row_offset_tiles + effective_M_tiles)
    //   are processed. For transpose_a, "row" means the M-axis of the matmul (= stored col
    //   axis of the input).
    //
    // Which side is the parent (and which provides matmul-K) is picked at create() time:
    //   - in1_k_offset > 0  OR  K_w > K_in  →  weight is parent, matmul-K = K_in
    //   - otherwise                         →  input is parent (or equal), matmul-K = K_w
    //
    // K-axis offset (in0_k_offset_tiles):
    //   Shifts the start of the in0 K-range read by this many tiles. matmul-K = K_w; in0
    //   is interpreted as a larger parent of which we read [k_offset, k_offset + K_w).
    //   For non-transpose this offsets along in0's stored col axis (matmul-K); for
    //   transpose_a it offsets along in0's stored row axis (matmul-K).
    //
    // in1_k_offset_tiles:
    //   K-axis offset on the weight (in1), analogous to in0_k_offset_tiles. matmul-K = K_in;
    //   the weight is interpreted as a larger parent of which we read
    //   [k_offset, k_offset + K_in) on its K axis (= storage row axis when not transpose_b,
    //   storage col axis when transpose_b). Cannot be combined with in0_k_offset > 0.
    //
    // out_row_offset_tiles:
    //   When an output tensor is provided in tensor_args_t, the matmul writes its
    //   actual_M-row output into rows [out_offset, out_offset + actual_M) of the parent
    //   output tensor. matmul-N must match the parent's N (no N-axis slicing).
    //
    // Defaults preserve "use the whole input" behavior. All offsets are RUNTIME args
    // (excluded from program hash) so different offset values hit the same cached program.
    uint32_t in0_row_offset_tiles = 0;
    uint32_t effective_M_tiles = 0;
    uint32_t in0_k_offset_tiles = 0;
    uint32_t in1_k_offset_tiles = 0;
    uint32_t out_row_offset_tiles = 0;

    // On-device offsets (EP). When role == OutputRow and offsets_tensor is set in
    // tensor_args, in1_sender_out reads offsets[offsets_start_index] and uses
    // (value / TILE_HEIGHT) as out_row_offset_tiles, overriding the scalar above.
    OffsetsRole offsets_role = OffsetsRole::None;
    uint32_t offsets_start_index = 0;
};

struct VariableMatmulInputs {
    ttnn::Tensor input_tensor;   // [actual_M, K]
    ttnn::Tensor weight_tensor;  // [K, N]
    // Optional caller-provided output tensor (write-at-offset mode). When set,
    // out_row_offset_tiles must be a valid sub-range and N must match.
    std::optional<ttnn::Tensor> output_tensor;
    // Optional 1-D UINT32 ROW_MAJOR device tensor. Used with offsets_role.
    std::optional<ttnn::Tensor> offsets_tensor;
};

}  // namespace ttml::metal::ops::variable_matmul::device
