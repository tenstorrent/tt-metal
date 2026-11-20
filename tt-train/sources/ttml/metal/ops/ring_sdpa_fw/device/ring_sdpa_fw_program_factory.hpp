// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>
#include <tt-metalium/program_descriptors.hpp>

#include "metal/common/const_utils.hpp"
#include "metal/ttnn_all_includes.hpp"
#include "ring_sdpa_fw_device_operation_types.hpp"

namespace ttml::metal::ops::ring_sdpa_fw {

// Builds one sdpa_fw program per device for this ring step, with the mask that device uses at this step. A device that
// sits the step out gets an empty descriptor, which the framework skips.
struct RingSDPAFwProgramFactory {
    static tt::tt_metal::ProgramDescriptor create_descriptor(
        const operation_attributes_t& operation_attributes,
        const tensor_args_t& tensor_args,
        tensor_return_value_t& tensor_return_value,
        const std::optional<ttnn::MeshCoordinate>& mesh_coordinate);

    // Runs on every cache hit: sdpa_fw's override on the device's program.
    static void override_runtime_arguments(
        tt::tt_metal::Program& program,
        const operation_attributes_t& operation_attributes,
        const tensor_args_t& tensor_args,
        tensor_return_value_t& tensor_return_value,
        const std::optional<ttnn::MeshCoordinate>& mesh_coordinate);
};

}  // namespace ttml::metal::ops::ring_sdpa_fw
