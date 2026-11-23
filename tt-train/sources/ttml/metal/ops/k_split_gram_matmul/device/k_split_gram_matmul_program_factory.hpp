// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>
#include <tt-metalium/program_descriptors.hpp>
#include <ttnn/device_operation.hpp>

#include "k_split_gram_matmul_device_operation_types.hpp"

namespace ttml::metal::ops::k_split_gram_matmul::device {

struct KSplitGramMatmulProgramFactory {
    static tt::tt_metal::ProgramDescriptor create_descriptor(
        const operation_attributes_t& attrs, const tensor_args_t& tensor_args, tensor_return_value_t& output);

    // Runs on every cache hit instead of buffer-binding patching. The input shape is in the program hash, so only the
    // input and output addresses change: it rewrites them on the senders, the upper-triangle writers and the diagonal
    // helpers.
    static void override_runtime_arguments(
        tt::tt_metal::Program& program,
        const operation_attributes_t& attrs,
        const tensor_args_t& tensor_args,
        tensor_return_value_t& output,
        const std::optional<ttnn::MeshCoordinate>& mesh_coordinate);
};

}  // namespace ttml::metal::ops::k_split_gram_matmul::device
