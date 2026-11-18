// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <tt-metalium/workload_descriptor.hpp>

#include "log_softmax_device_operation_types.hpp"
#include "metal/ttnn_all_includes.hpp"

namespace ttml::metal::ops::log_softmax::device {

// The collapse-resistant row-softmax program (row_softmax_program.hpp). The workload owns the DRAM scratch the cores
// sharing a row exchange partial statistics through; a program-cache hit only re-patches buffer addresses.
struct LogSoftmaxProgramFactory {
    static tt::tt_metal::WorkloadDescriptor create_workload_descriptor(
        const operation_attributes_t& operation_attributes,
        const tensor_args_t& tensor_args,
        tensor_return_value_t& tensor_return_value,
        const ttnn::MeshCoordinateRangeSet& tensor_coords);
};

}  // namespace ttml::metal::ops::log_softmax::device
