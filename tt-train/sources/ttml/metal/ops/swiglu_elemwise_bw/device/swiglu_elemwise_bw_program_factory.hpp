// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>
#include <tt-metalium/program_descriptors.hpp>

#include "metal/ttnn_all_includes.hpp"
#include "swiglu_elemwise_bw_device_operation_types.hpp"

namespace ttml::metal::ops::swiglu_elemwise_bw::device {

// Descriptor factory: the framework builds the Program from the returned descriptor on a cache miss, and on a cache
// hit runs only override_runtime_arguments(), which re-applies the five buffer addresses on every core. Binding
// patching is not used because production calls the op in place (preallocated_dL_dlinear1 == linear1): a buffer that
// appears twice among the inputs would make every hit re-run create_descriptor(). The other runtime args (rows per
// core, start row) follow from the hashed input shape.
struct SwigluElemwiseBwProgramFactory {
    static tt::tt_metal::ProgramDescriptor create_descriptor(
        const operation_attributes_t& operation_attributes,
        const tensor_args_t& tensor_args,
        tensor_return_value_t& tensor_return_value);

    static void override_runtime_arguments(
        tt::tt_metal::Program& program,
        const operation_attributes_t& operation_attributes,
        const tensor_args_t& tensor_args,
        tensor_return_value_t& tensor_return_value,
        const std::optional<ttnn::MeshCoordinate>& mesh_coordinate);
};

}  // namespace ttml::metal::ops::swiglu_elemwise_bw::device
