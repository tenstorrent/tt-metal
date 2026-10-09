// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "gated_rmsnorm_device_operation_types.hpp"
#include "metal/ttnn_all_includes.hpp"

namespace ttml::metal::ops::gated_rmsnorm::device {

// Buffers of one launch. The bw-only slots are null in fw, and dgamma is null when not computed.
struct GatedRmsNormBuffers {
    const tt::tt_metal::Buffer* input = nullptr;
    const tt::tt_metal::Buffer* gate = nullptr;
    const tt::tt_metal::Buffer* gamma = nullptr;
    const tt::tt_metal::Buffer* dL_dout = nullptr;
    const tt::tt_metal::Buffer* out = nullptr;
    const tt::tt_metal::Buffer* dgate = nullptr;
    const tt::tt_metal::Buffer* dgamma = nullptr;
};

struct GatedRmsNormProgramConfig {
    float epsilon = 1e-6F;
    bool backward = false;
    bool compute_dgamma = false;
};

struct GatedRmsNormSharedVariables {
    tt::tt_metal::KernelHandle reader_kernel_id{};
    tt::tt_metal::KernelHandle writer_kernel_id{};
    uint32_t num_cores{};
    uint32_t num_cores_y{};
};

GatedRmsNormSharedVariables build_gated_rmsnorm_program(
    tt::tt_metal::Program& program,
    const tt::tt_metal::CoreCoord& grid_size,
    uint32_t available_l1_bytes,
    const GatedRmsNormGeometry& geometry,
    const GatedRmsNormBuffers& buffers,
    const GatedRmsNormProgramConfig& config);

// Rewrites only the buffer addresses; work ranges depend on shapes, which are in the program hash.
void override_gated_rmsnorm_addresses(
    tt::tt_metal::Program& program, const GatedRmsNormSharedVariables& shared, const GatedRmsNormBuffers& buffers);

}  // namespace ttml::metal::ops::gated_rmsnorm::device
