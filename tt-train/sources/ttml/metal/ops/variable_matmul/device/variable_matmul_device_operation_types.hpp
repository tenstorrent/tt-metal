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
    // K-axis offset (in0_k_offset_tiles):
    //   shifts the start of the in0 K-range read by this many tiles. The matmul-K count
    //   still comes from the weight (the K==K_w validation), so the caller specifies only
    //   the offset; in0 is interpreted as a larger parent tensor of which we read the
    //   range [k_offset, k_offset + K) tiles. For non-transpose, this offsets along the
    //   input's stored col axis (matmul-K). For transpose_a, it offsets along the input's
    //   stored row axis (matmul-K).
    //
    // in1_k_offset_tiles:
    //   K-axis offset on the weight (in1), analogous to in0_k_offset_tiles. The weight
    //   is treated as a parent buffer with K extent K_in1 >= matmul-K; we slice
    //   [k_offset, k_offset + K) on the weight's K axis (= storage row axis when not
    //   transpose_b, storage col axis when transpose_b). The matmul-K count must match
    //   the in0 side (either K_in0 == K_in1 == K_matmul, or both come from a common
    //   parent slice). When in1_k_offset > 0, the weight's K_in extent dictates the
    //   matmul-K count (the caller must size things so the in0 K matches the slice).
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
};

struct VariableMatmulInputs {
    ttnn::Tensor input_tensor;   // [actual_M, K]
    ttnn::Tensor weight_tensor;  // [K, N]
    // Optional caller-provided output tensor (write-at-offset mode). When set,
    // out_row_offset_tiles must be a valid sub-range and N must match.
    std::optional<ttnn::Tensor> output_tensor;
};

}  // namespace ttml::metal::ops::variable_matmul::device
