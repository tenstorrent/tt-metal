// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>
#include <tt-metalium/program_descriptors.hpp>

#include "ttnn/device_operation.hpp"
#include "variable_matmul_device_operation_types.hpp"

namespace ttml::metal::ops::variable_matmul::device {

struct VariableMatmulProgramFactory {
    static tt::tt_metal::ProgramDescriptor create_descriptor(
        const VariableMatmulParams& operation_attributes,
        const VariableMatmulInputs& tensor_args,
        ttnn::Tensor& tensor_return_value);

    // Runs on every cache hit instead of buffer-binding patching. M, matmul-K and offsets_start_index are left out of
    // the program hash, so it re-derives the runtime args that depend on them and re-applies every buffer address.
    static void override_runtime_arguments(
        tt::tt_metal::Program& program,
        const VariableMatmulParams& operation_attributes,
        const VariableMatmulInputs& tensor_args,
        ttnn::Tensor& tensor_return_value,
        const std::optional<ttnn::MeshCoordinate>& mesh_coordinate);
};

}  // namespace ttml::metal::ops::variable_matmul::device
