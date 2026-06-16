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
};

// EP-friendly on-device offsets: the kernel reads offsets[start_index..start_index+2]
// from a device tensor at runtime and derives the matching row/K offsets. Lets moe_ffn
// avoid offsets.to_vector() under MeshDevice EP.
enum class OffsetsRole : uint32_t {
    // Reads offsets[start..start+2] and uses the same range for BOTH the in0 read window
    // AND the output write window. Lets moe_ffn use a single shared output tensor of shape
    // [T_cap, N] instead of E per-expert intermediates — each expert's matmul reads
    // grouped[offsets[e]:offsets[e+1]] and writes into the corresponding row range of the
    // shared output. The upper-bound constraint on per-expert size disappears: every
    // expert's actual rows fit naturally into its slice of the shared [T_cap, N] tensor.
    InputAndOutputRow = 1,
    // Reads offsets[start..start+2] and uses the same range for BOTH the in0 K-slice and
    // the in1 K-slice. Used in moe_ffn backward dW matmuls where both operands are shared
    // [T_cap, *] tensors and only the expert's K-row range should participate in the K-reduce.
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

    // expected_M_tiles:
    //   When > 0, the matmul processes only `expected_M_tiles` rows on the M axis (instead
    //   of the input's full M). With an output_tensor, this also bounds the host-side
    //   output-shape validation (the EP path may further override the per-core M split).
    //   Runtime arg — different values hit the same cached program.
    uint32_t expected_M_tiles = 0;

    // On-device offsets (EP). Dataflow kernels read offsets[start..start+2] at runtime and
    // derive the role-appropriate ranges:
    //   InputAndOutputRow: offsets[start..start+2] → input-row range + output write-at-offset
    //                      row + per-core M_start/M_end/M_blocks_per_core (dm_in0_sender
    //                      publishes the latter via cb_ctrl so compute can override RT args).
    //   InputAndWeightK:   offsets[start..start+2] → in0 K-slice + in1 K-slice (same range).
    OffsetsRole offsets_role = OffsetsRole::InputAndOutputRow;
    uint32_t offsets_start_index = 0;
};

struct VariableMatmulInputs {
    ttnn::Tensor input_tensor;   // [actual_M, K]
    ttnn::Tensor weight_tensor;  // [K, N]
    // Optional caller-provided output tensor (write-at-offset mode). When set, the
    // EP path derives the row offset from offsets_tensor and matmul-N must match.
    std::optional<ttnn::Tensor> output_tensor;
    // 1-D UINT32 ROW_MAJOR device tensor of offsets; read on every call per offsets_role.
    ttnn::Tensor offsets_tensor;
};

}  // namespace ttml::metal::ops::variable_matmul::device
