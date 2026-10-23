// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>
#include <tt-metalium/program_descriptors.hpp>

#include "metal/ttnn_all_includes.hpp"
#include "subtract_at_target_device_operation_types.hpp"

namespace ttml::metal::ops::subtract_at_target::device {

// Descriptor factory with one program per mesh coordinate: the vocab shard window a device handles depends on its TP
// rank. A cache hit runs only override_runtime_arguments(), not binding patching, for two reasons: first_v, local_V
// and subtract_value are left out of the program hash, and production calls the op in place (preallocated_output is
// the input), which binding patching would see as one buffer passed twice and answer with a rebuild on every hit.
struct SubtractAtTargetProgramFactory {
    static tt::tt_metal::ProgramDescriptor create_descriptor(
        const operation_attributes_t& operation_attributes,
        const tensor_args_t& tensor_args,
        tensor_return_value_t& tensor_return_value,
        const std::optional<ttnn::MeshCoordinate>& mesh_coordinate);

    // Re-applies the three buffer addresses, the device's shard window and subtract_value on every core.
    static void override_runtime_arguments(
        tt::tt_metal::Program& program,
        const operation_attributes_t& operation_attributes,
        const tensor_args_t& tensor_args,
        tensor_return_value_t& tensor_return_value,
        const std::optional<ttnn::MeshCoordinate>& mesh_coordinate);
};

}  // namespace ttml::metal::ops::subtract_at_target::device
