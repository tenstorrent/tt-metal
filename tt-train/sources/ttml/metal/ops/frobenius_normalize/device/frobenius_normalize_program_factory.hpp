// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>
#include <tt-metalium/program_descriptors.hpp>

#include "frobenius_normalize_device_operation_types.hpp"
#include "metal/ttnn_all_includes.hpp"

namespace ttml::metal::ops::frobenius_normalize::device {

struct FrobeniusNormalizeProgramFactory {
    static tt::tt_metal::ProgramDescriptor create_descriptor(
        const FrobeniusNormalizeAttributes& operation_attributes,
        const FrobeniusNormalizeTensorArgs& tensor_args,
        FrobeniusNormalizeTensorReturn& tensor_return_value);

    // Runs on every cache hit instead of buffer-binding patching. epsilon is left out of the program hash, so it
    // re-applies the input and output addresses and epsilon on every core.
    static void override_runtime_arguments(
        tt::tt_metal::Program& program,
        const FrobeniusNormalizeAttributes& operation_attributes,
        const FrobeniusNormalizeTensorArgs& tensor_args,
        FrobeniusNormalizeTensorReturn& tensor_return_value,
        const std::optional<ttnn::MeshCoordinate>& mesh_coordinate);
};

}  // namespace ttml::metal::ops::frobenius_normalize::device
