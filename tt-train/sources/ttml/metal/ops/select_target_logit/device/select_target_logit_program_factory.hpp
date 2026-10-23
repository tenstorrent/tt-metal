// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>
#include <tt-metalium/program_descriptors.hpp>

#include "metal/ttnn_all_includes.hpp"
#include "select_target_logit_device_operation_types.hpp"

namespace ttml::metal::ops::select_target_logit::device {

// Descriptor factory. Each device reads its own shard window [first_v, last_v), so create_descriptor() takes the
// device's mesh coordinate and the framework builds one program per device. On a cache hit only
// override_runtime_arguments() runs: first_v and local_V are not in the program hash, so it re-applies the device's
// window together with the three buffer addresses, on every core.
struct SelectTargetLogitProgramFactory {
    static tt::tt_metal::ProgramDescriptor create_descriptor(
        const operation_attributes_t& operation_attributes,
        const tensor_args_t& tensor_args,
        tensor_return_value_t& tensor_return_value,
        const std::optional<ttnn::MeshCoordinate>& mesh_coordinate);

    static void override_runtime_arguments(
        tt::tt_metal::Program& program,
        const operation_attributes_t& operation_attributes,
        const tensor_args_t& tensor_args,
        tensor_return_value_t& tensor_return_value,
        const std::optional<ttnn::MeshCoordinate>& mesh_coordinate);
};

}  // namespace ttml::metal::ops::select_target_logit::device
