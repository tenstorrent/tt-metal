// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>
#include <tt-metalium/program_descriptors.hpp>

#include "metal/ttnn_all_includes.hpp"
#include "softmax_backward_device_operation_types.hpp"

namespace ttml::metal::ops::softmax_backward::device {

struct SoftmaxBackwardFactory {
    static tt::tt_metal::ProgramDescriptor create_descriptor(
        const SoftmaxBackwardParams& operation_attributes,
        const SoftmaxBackwardInputs& tensor_args,
        ttnn::Tensor& tensor_return_value);

    // The only thing a cache hit runs: rewrites the buffer addresses. Every other runtime arg follows
    // from the program hash.
    static void override_runtime_arguments(
        tt::tt_metal::Program& program,
        const SoftmaxBackwardParams& operation_attributes,
        const SoftmaxBackwardInputs& tensor_args,
        ttnn::Tensor& tensor_return_value,
        const std::optional<ttnn::MeshCoordinate>& mesh_dispatch_coordinate);
};

}  // namespace ttml::metal::ops::softmax_backward::device
