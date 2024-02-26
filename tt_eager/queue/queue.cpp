
// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "tt_eager/queue/queue.hpp"
#include "tt_eager/tt_dnn/op_library/operation.hpp"
#include "tt_eager/tt_dnn/op_library/run_operation.hpp"

namespace tt::tt_metal
{
void EnqueueHostToDeviceTransfer(
    CommandQueue& q, Tensor& dst, const void* src, const std::optional<std::size_t> transfer_size) {
    memcpy(q, dst, src, transfer_size);
}

void EnqueueDeviceToHostTransfer(
    CommandQueue& q, Tensor& src, void* dst, const std::optional<std::size_t> transfer_size, size_t src_offset) {
    TT_ASSERT(src_offset == 0, "src_offset is not supported");
    memcpy(q, dst, src, transfer_size);
}

void QueueSynchronize(CommandQueue& q) { Finish(q); }

std::vector<Tensor> EnqueueOperation(
    CommandQueue& queue,
    operation::DeviceOperation& devop,
    const std::vector<Tensor>& input_tensors,
    const std::vector<std::optional<const Tensor>>& optional_input_tensors,
    const std::vector<std::optional<Tensor>>& optional_output_tensors) {
    return operation::run(queue, devop, input_tensors, optional_input_tensors, optional_output_tensors);
}
}
