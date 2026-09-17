// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <tt-metalium/program_descriptors.hpp>

#include "metal/ttnn_all_includes.hpp"
#include "profiler_no_op_device_operation_types.hpp"

namespace ttml::metal::ops::profiler_no_op::device {

// Descriptor factory: the framework builds the Program from the returned descriptor and, on a program-cache hit,
// re-patches the buffers bound through KernelDescriptor::emplace_runtime_args(). Nothing is kept between launches, so
// there is no shared_variables_t and no override_runtime_arguments().
struct ProfilerNoopProgramFactory {
    static tt::tt_metal::ProgramDescriptor create_descriptor(
        const operation_attributes_t& operation_attributes,
        const tensor_args_t& tensor_args,
        tensor_return_value_t& output);
};

}  // namespace ttml::metal::ops::profiler_no_op::device
