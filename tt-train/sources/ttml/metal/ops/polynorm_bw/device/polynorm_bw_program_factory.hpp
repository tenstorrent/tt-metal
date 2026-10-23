// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>
#include <tt-metalium/program_descriptors.hpp>

#include "metal/ttnn_all_includes.hpp"
#include "polynorm_bw_device_operation_types.hpp"

namespace ttml::metal::ops::polynorm3_bw::device {

// Program factory for PolyNorm3 backward (reader/compute/writer kernels + runtime args).
struct PolyNorm3BackwardProgramFactory {
    static tt::tt_metal::ProgramDescriptor create_descriptor(
        const PolyNorm3BWAttributes& operation_attributes,
        const PolyNorm3BWTensorArgs& tensor_args,
        PolyNorm3BWTensorReturn& tensor_return_value);

    // The only thing a cache hit runs: rewrites the buffer addresses on every core. Every other runtime arg follows
    // from the program hash.
    static void override_runtime_arguments(
        tt::tt_metal::Program& program,
        const PolyNorm3BWAttributes& operation_attributes,
        const PolyNorm3BWTensorArgs& tensor_args,
        PolyNorm3BWTensorReturn& tensor_return_value,
        const std::optional<ttnn::MeshCoordinate>& mesh_dispatch_coordinate);
};

}  // namespace ttml::metal::ops::polynorm3_bw::device
