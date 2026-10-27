// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <vector>

#include "chunk_gated_delta_rule_bw_device_operation_types.hpp"
#include "metal/ttnn_all_includes.hpp"

namespace ttml::metal::ops::chunk_gated_delta_rule_bw::device {

// Every shape-derived count of one call, solved once on the host.  The kernels receive these as
// compile-time args and never recompute them.
struct Geometry {
    uint32_t B{}, T{}, HV{}, HK{}, G{}, K{}, V{};
    uint32_t Ct{}, Kt{}, Vt{}, Tt{}, NC{}, BH{}, NI{};
    uint32_t neumann_steps{};
    uint32_t Vb{};   // V tiles per compute pass: the largest divisor of Vt whose L1 footprint fits
    uint32_t NVB{};  // Vt / Vb

    // Tile-index bases of the per-item blocks inside the f32 scratch tensor.
    uint32_t sc_attn{}, sc_kcd{}, sc_p{}, sc_vec{}, sc_vcorr{}, sc_u{}, sc_c{}, sc_s{}, sc_ds{}, sc_vnew{}, sc_dvnew{};
    uint32_t scratch_tiles{};
};

Geometry make_geometry(const operation_attributes_t& args, const tensor_args_t& tensor_args);

struct ChunkGatedDeltaRuleBackwardProgramFactory {
    struct shared_variables_t {
        tt::tt_metal::KernelHandle reader_kernel_id{};
        tt::tt_metal::KernelHandle writer_kernel_id{};
        std::vector<tt::tt_metal::CoreCoord> cores;
    };
    using cached_program_t = ttnn::device_operation::CachedProgram<shared_variables_t>;

    static cached_program_t create(
        const operation_attributes_t& operation_attributes,
        const tensor_args_t& tensor_args,
        tensor_return_value_t& tensor_return_value);

    static void override_runtime_arguments(
        cached_program_t& cached_program,
        const operation_attributes_t& operation_attributes,
        const tensor_args_t& tensor_args,
        tensor_return_value_t& tensor_return_value);
};

}  // namespace ttml::metal::ops::chunk_gated_delta_rule_bw::device
