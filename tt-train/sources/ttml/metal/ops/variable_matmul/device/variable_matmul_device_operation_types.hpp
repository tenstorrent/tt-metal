// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
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
};

// On-device offsets (EP-friendly): at runtime the dataflow kernels read offsets[start..start+2]
// (a {start, end} pair) from a UINT32 device tensor and derive the role's row/K ranges.
enum class OffsetsRole : uint32_t {
    // [start, end) drives BOTH the in0 row read and the output row write. Lets a caller (moe_ffn)
    // route every expert's matmul into its slice of one shared [T_cap, N] output, instead of
    // allocating E per-expert, upper-bound-sized intermediates.
    InputAndOutputRow = 1,
    // [start, end) K-slices BOTH in0 and in1. Used by moe_ffn's backward dW matmuls, where both
    // operands are shared [T_cap, *] tensors and only the expert's K-rows participate in the reduce.
    InputAndWeightK = 2,
};

struct VariableMatmulParams {
    VariableMatmulConfig config;
    ttnn::DeviceComputeKernelConfig compute_kernel_config;

    // When true, the input tensor is interpreted as transposed for matmul purposes:
    // stored shape [..., K, M] but used as [..., M, K]. Reader applies stride swap,
    // compute kernel applies intra-tile transpose via transpose_wh_tile into a dedicated CB.
    bool transpose_a = false;
    // When true, the weight tensor is interpreted as transposed for matmul purposes:
    // stored shape [..., N, K] but used as [..., K, N]. Reader applies stride swap,
    // matmul kernel applies intra-tile transpose via the LLK transpose flag.
    bool transpose_b = false;

    // Per-call matmul-M extent in tiles (0 = use the input's full M). Does NOT cap the work:
    // InputAndOutputRow re-derives the actual per-core M from the offsets at runtime. Its one job
    // is picking the grid orientation (transpose_core_grid), which is compile-time and can't be
    // inferred from the tensors (in the shared-buffer path their M is the full T_cap). The offset
    // reads are driven by the role, not by this field. Feeds the program hash via
    // transpose_core_grid, so a stable value reuses one cached program while a value that flips
    // the M-vs-N orientation compiles a distinct one.
    uint32_t expected_M_tiles = 0;

    // How the on-device offsets are interpreted (see OffsetsRole). offsets_start_index is the
    // index of this call's {start, end} pair within the offsets tensor.
    OffsetsRole offsets_role = OffsetsRole::InputAndOutputRow;
    uint32_t offsets_start_index = 0;
};

struct VariableMatmulInputs {
    ttnn::Tensor input_tensor;   // logical [M, K] (stored [K, M] when transpose_a)
    ttnn::Tensor weight_tensor;  // logical [K, N] (stored [N, K] when transpose_b)
    // Caller-provided output (write-at-offset): required for InputAndOutputRow, which writes each
    // call's row range into it (matmul-N must equal its N); must be nullopt for InputAndWeightK,
    // which allocates its own output.
    std::optional<ttnn::Tensor> output_tensor;
    // 1-D UINT32 ROW_MAJOR device tensor of offsets; read on every call per offsets_role.
    ttnn::Tensor offsets_tensor;
};

}  // namespace ttml::metal::ops::variable_matmul::device
