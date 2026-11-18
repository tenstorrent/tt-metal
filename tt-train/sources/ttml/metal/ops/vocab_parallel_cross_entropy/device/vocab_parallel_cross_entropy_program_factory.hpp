// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <tt-metalium/workload_descriptor.hpp>

#include "metal/ttnn_all_includes.hpp"
#include "vocab_parallel_cross_entropy_device_operation_types.hpp"

namespace ttml::metal::ops::vocab_parallel_cross_entropy::device {

// The collapse-resistant row-softmax program (metal/ops/log_softmax/device/row_softmax_program.hpp), one program per
// TP rank: a shard's first global column differs per rank. The workload owns the DRAM scratch the cores sharing a row
// exchange partial statistics through; a program-cache hit only re-patches buffer addresses.
struct VocabParallelCrossEntropyProgramFactory {
    static tt::tt_metal::WorkloadDescriptor create_workload_descriptor(
        const operation_attributes_t& operation_attributes,
        const tensor_args_t& tensor_args,
        tensor_return_value_t& tensor_return_value,
        const ttnn::MeshCoordinateRangeSet& tensor_coords);
};

}  // namespace ttml::metal::ops::vocab_parallel_cross_entropy::device
