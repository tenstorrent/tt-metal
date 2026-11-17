// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>
#include <tt-metalium/program_descriptors.hpp>

#include "metal/ttnn_all_includes.hpp"
#include "softmax_device_operation_types.hpp"

namespace ttml::metal::ops::softmax::device {

struct SoftmaxProgramFactory {
    static tt::tt_metal::ProgramDescriptor create_descriptor(
        const operation_attributes_t& operation_attributes,
        const tensor_args_t& tensor_args,
        tensor_return_value_t& tensor_return_value);

    // The only thing a cache hit runs: rewrites the buffer addresses on every core. Every other runtime arg follows
    // from the program hash.
    static void override_runtime_arguments(
        tt::tt_metal::Program& program,
        const operation_attributes_t& operation_attributes,
        const tensor_args_t& tensor_args,
        tensor_return_value_t& tensor_return_value,
        const std::optional<ttnn::MeshCoordinate>& mesh_dispatch_coordinate);
};

}  // namespace ttml::metal::ops::softmax::device
