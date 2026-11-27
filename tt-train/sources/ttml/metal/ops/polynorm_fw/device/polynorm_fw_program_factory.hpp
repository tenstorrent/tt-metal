// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>
#include <tt-metalium/program_descriptors.hpp>

#include "metal/ttnn_all_includes.hpp"
#include "polynorm_fw_device_operation_types.hpp"

namespace ttml::metal::ops::polynorm3_fw::device {

// Program factory for PolyNorm3 forward (reader/compute/writer kernels + runtime args).
struct PolyNorm3ForwardProgramFactory {
    // Describe kernels/circular buffers/runtime args for a given input shape/layout.
    static tt::tt_metal::ProgramDescriptor create_descriptor(
        const PolyNorm3FWAttributes& operation_attributes,
        const PolyNorm3FWTensorArgs& tensor_args,
        PolyNorm3FWTensorReturn& tensor_return_value);

    // The only thing a cache hit runs: rewrites the buffer addresses on every core. Every other runtime arg follows
    // from the program hash.
    static void override_runtime_arguments(
        tt::tt_metal::Program& program,
        const PolyNorm3FWAttributes& operation_attributes,
        const PolyNorm3FWTensorArgs& tensor_args,
        PolyNorm3FWTensorReturn& tensor_return_value,
        const std::optional<ttnn::MeshCoordinate>& mesh_dispatch_coordinate);
};

}  // namespace ttml::metal::ops::polynorm3_fw::device
